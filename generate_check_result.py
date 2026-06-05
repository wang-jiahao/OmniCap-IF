# check_only_pipeline.py
import os
import json
import re
import glob
import time
import traceback
from typing import Dict, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from datetime import datetime
import argparse
import copy
from tqdm import tqdm
import openai
from utils import openai_client, clean_json_response, combined_retry
from utils import AutoRuleChecker
import temporal_eval_utils


def _is_empty_response_text(response: object) -> bool:
    """Return True if the response is empty/whitespace-only and should be skipped."""
    if response is None:
        return True
    if isinstance(response, str):
        return len(response.strip()) == 0
    return False


def _prompt_key(field: object, prompt_id: object) -> tuple[str, str]:
    return (str(field) if field is not None else "", str(prompt_id) if prompt_id is not None else "")


def _ensure_prompt_ids(existing_data: Dict) -> None:
    """Ensure each prompt/checklist/response item has a prompt_id.

    Some datasets store prompt_id only in responses (or omit it). This function aligns
    prompt_id across prompts/checklists/responses by index when possible.
    """
    prompts = existing_data.get('prompts') or []
    checklists = existing_data.get('checklists') or []
    responses = existing_data.get('responses') or []

    if not isinstance(prompts, list) or not isinstance(checklists, list) or not isinstance(responses, list):
        return

    max_len = max(len(prompts), len(checklists), len(responses))
    if max_len == 0:
        return

    # Prefer prompt_id from responses when available (most reliable).
    response_ids: List[str] = []
    if len(responses) == max_len and all(isinstance(r, dict) and str(r.get('prompt_id', '')).strip() for r in responses):
        response_ids = [str(r.get('prompt_id')).strip() for r in responses]

    if response_ids:
        prompt_ids = response_ids
    else:
        prompt_ids: List[str] = []
        for i in range(max_len):
            pid = None
            if i < len(prompts) and isinstance(prompts[i], dict):
                pid = prompts[i].get('prompt_id')
            pid_str = str(pid).strip() if pid is not None else ''
            prompt_ids.append(pid_str if pid_str else f"{i+1:02d}")

    for i, pid in enumerate(prompt_ids):
        if i < len(prompts) and isinstance(prompts[i], dict) and not prompts[i].get('prompt_id'):
            prompts[i]['prompt_id'] = pid
        if i < len(checklists) and isinstance(checklists[i], dict) and not checklists[i].get('prompt_id'):
            checklists[i]['prompt_id'] = pid
        if i < len(responses) and isinstance(responses[i], dict) and not responses[i].get('prompt_id'):
            responses[i]['prompt_id'] = pid


def _prune_judge_items_for_empty_responses(existing_data: Dict, judge_items: List[Dict]) -> tuple[List[Dict], int]:
    """Remove legacy check_result items whose corresponding response is empty.

    Returns:
        (pruned_items, removed_count)
    """
    prompts = existing_data.get('prompts', [])
    responses = existing_data.get('responses', [])
    checklists = existing_data.get('checklists', [])

    if len(prompts) != len(responses) or len(prompts) != len(checklists):
        return judge_items, 0

    required = set()
    for prompt_data, response_data, _ in zip(prompts, responses, checklists):
        field = prompt_data.get('field')
        prompt_id = prompt_data.get('prompt_id')
        response = response_data.get('response') if isinstance(response_data, dict) else response_data
        if _is_empty_response_text(response):
            continue
        required.add(_prompt_key(field, prompt_id))

    if not judge_items:
        return [], 0

    pruned = []
    removed = 0
    for item in judge_items:
        key = _prompt_key(item.get('field'), item.get('prompt_id'))
        if key in required:
            pruned.append(item)
        else:
            removed += 1

    return pruned, removed


def _parse_duration_to_seconds(duration: object) -> Optional[float]:
    """Parse the `duration` field in video_meta_info.json into seconds.

    Supported formats:
    - number (seconds)
    - "MM:SS" / "HH:MM:SS" (optionally with decimals)
    - strings like "55s" / "55" (best-effort parsing)
    """
    if duration is None:
        return None

    if isinstance(duration, (int, float)):
        return float(duration) if float(duration) > 0 else None

    if not isinstance(duration, str):
        return None

    text = duration.strip().lower()
    if not text:
        return None

    # Handle "55s"
    if text.endswith('s'):
        text = text[:-1].strip()

    # Handle "MM:SS" / "HH:MM:SS"
    if ':' in text:
        parts = text.split(':')
        try:
            parts = [float(p) for p in parts]
        except ValueError:
            return None

        if len(parts) == 2:
            minutes, seconds = parts
            total = minutes * 60 + seconds
        elif len(parts) == 3:
            hours, minutes, seconds = parts
            total = hours * 3600 + minutes * 60 + seconds
        else:
            return None

        return float(total) if total > 0 else None

    # Handle pure numeric strings like "55"
    try:
        val = float(text)
        return val if val > 0 else None
    except ValueError:
        return None


class ProgressManager:
    """Progress manager for multi-threaded processing."""
    
    def __init__(self, total_tasks: int, desc: str = "Progress"):
        self.total_tasks = total_tasks
        self.desc = desc
        self.lock = Lock()
        self.stats = {'completed': 0, 'failed': 0, 'skipped': 0}
        self.progress_bar = None
        self._setup_progress_bar()
    
    def _setup_progress_bar(self):
        """Initialize the progress bar."""
        self.progress_bar = tqdm(
            total=self.total_tasks, 
            desc=self.desc,
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        )
        # Initialize counters displayed in tqdm.
        self.progress_bar.set_postfix({
            'success': 0,
            'failed': 0,
            'skipped': 0
        })
    
    def update(self, status: str, item_id: str = ""):
        """
        Update progress.
        
        Args:
            status: 'completed', 'failed', 'skipped'
            item_id: Optional ID of the item being processed.
        """
        with self.lock:
            if status in self.stats:
                self.stats[status] += 1
            
            # Update counters displayed in tqdm.
            self.progress_bar.set_postfix({
                'success': self.stats['completed'],
                'failed': self.stats['failed'],
                'skipped': self.stats['skipped']
            })
            self.progress_bar.update(1)
    
    def get_stats(self) -> Dict[str, int]:
        """Return a snapshot of current statistics."""
        with self.lock:
            return self.stats.copy()
    
    def close(self):
        """Close the progress bar."""
        if self.progress_bar:
            self.progress_bar.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class VideoLogger:
    """Captures and stores per-video processing logs."""
    def __init__(self, video_id: str, log_dir: str):
        self.video_id = video_id
        self.log_dir = log_dir
        self.start_time = datetime.now()
        
        # Create a dedicated log file.
        self.log_file_path = os.path.join(log_dir, f"{video_id}.log")
        self.log_file = open(self.log_file_path, 'w', encoding='utf-8')
        
        # Header
        self.write(f"Start processing video: {video_id}")
        self.write(f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.write("-" * 60)
        
    def write(self, message: str):
        """Write a log line."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_line = f"[{timestamp}] {message}\n"
        self.log_file.write(log_line)
        self.log_file.flush()  # flush immediately
        
    def close(self):
        """Close the log file."""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        self.write("-" * 60)
        self.write(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.write(f"Elapsed: {duration:.2f}s")
        self.log_file.close()
        
    def get_log_path(self) -> str:
        """Return the log file path."""
        return self.log_file_path


class LogManager:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.master_log_path = os.path.join(log_dir, "master.log")
        self.lock = Lock()
        self.completed_logs = {}  # video_id -> log file path
        
        # Ensure log directory exists.
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize the master log.
        self._init_master_log()
    
    def _init_master_log(self):
        """Initialize the master log file."""
        with open(self.master_log_path, 'w', encoding='utf-8') as f:
            f.write("Check-Only Pipeline Master Log\n")
            f.write(f"Created at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
    
    def add_completed_log(self, video_id: str, log_file_path: str):
        """Add a completed per-video log and regenerate the master log."""
        with self.lock:
            # Add to completed list
            self.completed_logs[video_id] = log_file_path
            
            # Regenerate master log
            self._regenerate_master_log()
    
    def _regenerate_master_log(self):
        """Regenerate the ordered master log file."""
        # Define sort key
        def get_video_sort_key(video_id):
            parts = video_id.split('_')
            video_part = parts[0]
            video_num = int(parts[-1]) if parts[-1].isdigit() else 0
            part_order = {'clip': 0, 'short': 1, 'long': 2}
            part_idx = part_order.get(video_part, 999)
            return (part_idx, video_num)
        
        # Sort video IDs.
        sorted_video_ids = sorted(self.completed_logs.keys(), key=get_video_sort_key)
        
        # Rewrite the master log.
        self._init_master_log()  # re-init
        
        with open(self.master_log_path, 'a', encoding='utf-8') as master_f:
            for video_id in sorted_video_ids:
                log_file_path = self.completed_logs[video_id]
                try:
                    with open(log_file_path, 'r', encoding='utf-8') as f:
                        video_log_content = f.read()
                    
                    master_f.write(f"\n{'='*80}\n")
                    master_f.write(f"Processing log for video {video_id}:\n")
                    master_f.write(f"{'='*80}\n")
                    master_f.write(video_log_content)
                    master_f.write(f"\n{'='*80}\n\n")
                    
                except Exception as e:
                    master_f.write(f"\nError: failed to read log for video {video_id}: {str(e)}\n\n")


class VideoProcessor:
    """Process a single video (check-only)."""
    def __init__(self, video_id: str, existing_data: Dict, pipeline: 'CheckOnlyPipeline', logger: VideoLogger,
                 done_keys: Optional[set[tuple[str, str]]] = None):
        self.video_id = video_id
        self.existing_data = existing_data
        self.pipeline = pipeline
        self.logger = logger
        self.done_keys = done_keys or set()
        self.result = {
            'judge': []
        }
        
    def log(self, message: str):
        """Write a message to the per-video log."""
        self.logger.write(message)
        
    def process(self) -> Tuple[str, Dict]:
        """Process the video and return (video_id, result)."""
        try:
            # Get input data
            prompts = self.existing_data['prompts']
            responses = self.existing_data['responses']
            checklists = self.existing_data['checklists']
            
            if len(prompts) != len(responses) or len(prompts) != len(checklists):
                self.log(
                    f"Error: mismatched lengths - prompts: {len(prompts)}, responses: {len(responses)}, checklists: {len(checklists)}"
                )
                return self.video_id, None
            
            self.log(f"Start processing video {self.video_id} with {len(prompts)} test cases")

            skipped_empty = 0
            skipped_done = 0
            
            # Process each test case
            for idx, (prompt_data, response_data, checklist_data) in enumerate(zip(prompts, responses, checklists)):
                prompt = prompt_data.get('generated_prompt')
                field = prompt_data.get('field')
                prompt_id = prompt_data.get('prompt_id')

                if prompt is None or field is None or prompt_id is None:
                    self.log(
                        f"Error: missing required fields in prompt item at idx={idx}: "
                        f"generated_prompt={prompt is not None}, field={field is not None}, prompt_id={prompt_id is not None}"
                    )
                    return self.video_id, None
                self.log(f"Case {idx+1}/{len(prompts)} - field: {field}, prompt_id: {prompt_id}")
                
                # Gather required values
                key = _prompt_key(field, prompt_id)

                # Resume support: skip if this prompt already has a check_result.
                if key in self.done_keys:
                    skipped_done += 1
                    self.log(f"⏭️ Skipped: check_result already exists - field: {field}, prompt_id: {prompt_id}")
                    continue

                response = response_data.get('response') if isinstance(response_data, dict) else response_data

                # Skip empty responses (no check_result generated; not counted in metrics).
                if _is_empty_response_text(response):
                    skipped_empty += 1
                    self.log(f"⏭️ Skipped: empty response - field: {field}, prompt_id: {prompt_id}")
                    continue
                checklist = checklist_data['checklist']
                
                # Generate check result
                check_result = self._generate_check_result(
                    prompt, response, checklist, field
                )
                
                # Save result
                self.result['judge'].append({
                    "field": field,
                    "prompt_id": prompt_id,
                    "check_result": check_result
                })
                
                self.log(f"✅ Completed check for case {idx+1}")
            
            self.log(
                f"✅ Video {self.video_id} completed with {len(self.result['judge'])} checks "
                f"(skipped empty responses: {skipped_empty}, skipped already done: {skipped_done})"
            )
            return self.video_id, self.result
            
        except Exception as e:
            self.log(f"Error while processing video {self.video_id}: {str(e)}")
            import traceback
            self.log(f"Traceback:\n{traceback.format_exc()}")
            return self.video_id, None
        
    def _generate_check_result(self, prompt: str, response: str,  
                              checklist: Dict, field: str) -> Dict:
        """Generate the check result."""
        
        check_result = copy.deepcopy(checklist)
        
        max_inline_retry = 5
        inline_retry = 0
        if 'format_check' in checklist:
            for idx, checkitem in enumerate(checklist['format_check']):
                # Use the judge LLM to extract content for format checks
                retry_response=None
                while inline_retry < max_inline_retry:
                    rule_content = self.pipeline.get_format_checkresult_with_llm(
                        response[:self.pipeline.max_token], checkitem, retry_response
                    )
                    if checkitem['constraint_id'] != 'count':
                        if all(item in response for item in rule_content['content']):
                            check_result['format_check'][idx]['parameters']['content'] = rule_content['content']
                            break
                        else:
                            retry_response = rule_content['content']
                            inline_retry += 1
                            check_result['format_check'][idx]['parameters']['content'] = ['<error content holder>'*100]
                            self.log(
                                f"❌ Format check extraction failed for field {field}: content not found in response; retrying"
                            )
                    else:
                        break
                check_result['format_check'][idx]['parameters']['content'] = rule_content['content']
                
        inline_retry = 0
        if 'content_check' in checklist:
            for idx, check_content in enumerate(checklist['content_check']):
                for checkitem_idx, checkitem in enumerate(check_content['check_items']):
                    question = checkitem['question']
                    options = checkitem.get('options', [])
                    # Use the judge LLM to generate content check results
                    current_check_item = check_result['content_check'][idx]['check_items'][checkitem_idx]
                    is_timestamp_check = checkitem.get('check_type') == 'timestamp'

                    while inline_retry < max_inline_retry:
                        try:
                            if is_timestamp_check:
                                extracted_time = self.pipeline.extract_timestamp_with_llm(
                                    response[:self.pipeline.max_token], question
                                )

                                duration = self.pipeline.get_video_duration(self.video_id)

                                gt_time = checkitem.get('correct_answer', '')
                                eval_res = temporal_eval_utils.evaluate_temporal_constraint(extracted_time, gt_time, duration)

                                current_check_item['answer'] = extracted_time
                                current_check_item['check_passed'] = eval_res['passed']
                                current_check_item['result_explanation'] = eval_res['reason']
                                current_check_item['result_confidence'] = 5
                                current_check_item['score'] = eval_res['score']
                                break
                            else:
                                answer_response = self.pipeline.get_content_checkresult_with_llm(
                                    prompt, response[:self.pipeline.max_token], question, options
                                )
                                current_check_item['answer'] = answer_response['answer'][0] if answer_response['answer'][0] in ['A', 'B', 'C', 'D'] else answer_response['answer']
                                current_check_item['result_explanation'] = answer_response['result_explanation']
                                current_check_item['result_confidence'] = answer_response['result_confidence']
                                break
                        except Exception as e:
                            self.log(f"❌ Content check failed for field {field}: {e}")
                            inline_retry += 1
                            continue
                            
                    inline_retry = 0
                
        self.log(f"✅ Generated raw check items for field {field}")

        # Run format checks
        check_result, status = self.pipeline.auto_checker.check_all_rules(check_result)
        if not status:
            self.log(f"❌ Check result for field {field} does not pass format validation")
            raise ValueError(f"Check result for field {field} does not pass format validation: {check_result}")
        # Post-process content checks
        if 'content_check' in check_result:
            for check_group in check_result['content_check']: 
                for check_item in check_group['check_items']:
                    # Determine whether the answer matches and fill the `result` field
                    if 'check_passed' in check_item:
                        check_item['result'] = check_item['check_passed']
                    else:
                        check_item['result'] = check_item['answer'] == check_item['correct_answer']
            
        self.log(f"✅ Finished generating check result for field {field}")
        
        return check_result
    
class CheckOnlyPipeline:
    def __init__(
        self,
        meta_input_dir: str = './annotation',
        response_input_dir: str = './response',
        model_name: str = 'example_model',
        output_dir: str = './check_result',
        max_workers: int = 32,
    ):
        
        self.max_workers = max_workers  # configured worker count
        
        # Input file paths
        self.meta_input_dir = meta_input_dir
        self.prompt_input_path = os.path.join(meta_input_dir, 'prompts.json')
        self.checklist_input_path = os.path.join(meta_input_dir, 'checklists.json')
        self.response_input_path = os.path.join(response_input_dir, f"{model_name}_response.json")
        
        # Output file paths
        self.output_dir = output_dir
        self.judge_output_path = os.path.join(output_dir, f"{model_name}_check_result.json")

        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Log directory
        self.log_dir = os.path.join('./logs/check/', model_name)
        self.log_manager = LogManager(self.log_dir)
        
        # Cache
        self.cached_results = {}

        # Load video durations: prefer meta_dir/video_meta_info.json; fallback to 55s.
        self.default_duration_seconds = 55.0
        self.video_durations: Dict[str, float] = {}
        video_meta_info_path = os.path.join(self.meta_input_dir, 'video_meta_info.json')
        if os.path.exists(video_meta_info_path):
            try:
                with open(video_meta_info_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                loaded = 0
                for vid_key, info in (meta or {}).items():
                    if not isinstance(info, dict):
                        continue
                    seconds = _parse_duration_to_seconds(info.get('duration'))
                    if seconds is None:
                        continue
                    self.video_durations[str(vid_key)] = float(seconds)
                    loaded += 1
                print(f"Loaded video durations from '{video_meta_info_path}': {loaded} entries")
            except Exception as e:
                print(
                    f"Warning: failed to read '{video_meta_info_path}', using default duration {self.default_duration_seconds}s: {e}"
                )
                self.video_durations = {}
        else:
            print(f"Warning: '{video_meta_info_path}' not found; using default duration {self.default_duration_seconds}s")

        # Judge model (fixed)
        self.judge_llm = "gpt-5-mini"

        # Config files
        format_judge_llm_meta_prompt_path = "./llm_judge/format_llm_extract.txt"
        content_judge_llm_meta_prompt_path = "./llm_judge/content_llm_judge.txt"
        
        print("Loading meta prompts...")
        
        try:
            with open(format_judge_llm_meta_prompt_path, 'r', encoding='utf-8') as f:
                format_judge_llm_meta_prompt = f.read()
            print(f"Loaded meta prompt from '{format_judge_llm_meta_prompt_path}'.")
            with open(content_judge_llm_meta_prompt_path, 'r', encoding='utf-8') as f:
                content_judge_llm_meta_prompt = f.read()
        except FileNotFoundError as e:
            print(f"Error: failed to load meta prompt file {e.filename}. Please check the path.")
            raise e
        

        self.meta_prompt = {
            "format_judge": format_judge_llm_meta_prompt,
            "content_judge": content_judge_llm_meta_prompt,
        }

        # Initialize model client
        print("Initializing model client...")
        self.client = {
            'judge_llm': openai_client(),
        }
        
        self.auto_checker = AutoRuleChecker()
        
        # Multi-threading
        self.lock = Lock()
        
        # Print configuration
        print(f"Thread config: {self.max_workers} concurrent workers")
        
        self.max_token = 2048

    def get_video_duration(self, video_id: str) -> float:
        """Return the video duration in seconds.

        - Prefer `meta_dir/video_meta_info.json` duration
        - If missing/unparseable, return the default average duration (55s)
        """
        # video_meta_info.json keys are commonly like '001'
        match = re.search(r'(\d+)', str(video_id))
        if match:
            try:
                num = int(match.group(1))
                key = f"{num:03d}"
                val = self.video_durations.get(key)
                if val and val > 0:
                    return float(val)
            except ValueError:
                pass

        # Then try the original video_id
        val = self.video_durations.get(str(video_id))
        if val and val > 0:
            return float(val)

        return float(self.default_duration_seconds)

    def _get_done_prompt_keys(self, video_id: str, judge_dict: Dict) -> set[tuple[str, str]]:
        done = set()
        for item in judge_dict.get(video_id, []) or []:
            done.add(_prompt_key(item.get('field'), item.get('prompt_id')))
        return done

    def _get_required_prompt_keys(self, existing_data: Dict) -> Optional[set[tuple[str, str]]]:
        prompts = existing_data.get('prompts', [])
        responses = existing_data.get('responses', [])
        checklists = existing_data.get('checklists', [])

        if len(prompts) != len(responses) or len(prompts) != len(checklists):
            return None

        required = set()
        for prompt_data, response_data, _ in zip(prompts, responses, checklists):
            field = prompt_data.get('field')
            prompt_id = prompt_data.get('prompt_id')
            response = response_data.get('response') if isinstance(response_data, dict) else response_data
            if _is_empty_response_text(response):
                continue
            required.add(_prompt_key(field, prompt_id))

        return required

    def _is_video_completed(self, video_id: str, existing_data: Dict, judge_dict: Dict) -> bool:
        required = self._get_required_prompt_keys(existing_data)
        if required is None:
            return False
        done = self._get_done_prompt_keys(video_id, judge_dict)
        return required.issubset(done)

    @combined_retry(timeout_seconds=600, 
                    timeout_retries=2, 
                    error_retries=5, 
                    exceptions=(ValueError, ConnectionError, openai.RateLimitError, openai.APIError),
                    delay=2.0,
                    backoff=2.0)
    def get_format_checkresult_with_llm(self,
                                            response: str, 
                                            checkitem: Dict,
                                            retry_response=None) -> Dict:
        """Use the judge LLM to generate the format-check extraction result."""
        json_prompt = json.dumps({
            "response": response,
            "checkitem": checkitem
        }, ensure_ascii=False)

        if retry_response == None:
            api_response = self.client['judge_llm'].chat.completions.create(
                model=self.judge_llm,
                messages=[
                    {"role": "system", "content": self.meta_prompt['format_judge']},
                    {"role": "user", "content": json_prompt}
                ],
                response_format={"type": "json_object"},
                stream=False,
            )
        else:
            retry_prompt = """
            The content you extracted has been detected as not being a pure extraction from the response. The "content in response" check failed. Please re-extract, noting that you cannot make any modifications - it must be an original text excerpt from the response without adding any of your own understanding or changes.
            """
            retry_response = json.dumps(retry_response, ensure_ascii=False)
            api_response = self.client['judge_llm'].chat.completions.create(
                model=self.judge_llm,
                messages=[
                    {"role": "system", "content": self.meta_prompt['format_judge']},
                    {"role": "user", "content": json_prompt},
                    {"role": "assistant", "content": retry_response},
                    {"role": "user", "content": retry_prompt}
                ],
                response_format={"type": "json_object"},
                stream=False,
            )
        try:
            return(json.loads(clean_json_response(api_response.choices[0].message.content)))
        except json.JSONDecodeError as e:
            print(f"Still failed to parse JSON after cleaning: {api_response.choices[0].message.content}")
            raise ValueError(f"Failed to parse LLM response as valid JSON: {e}") from e

    @combined_retry(timeout_seconds=600, 
                    timeout_retries=2, 
                    error_retries=5, 
                    exceptions=(ValueError, ConnectionError, openai.RateLimitError, openai.APIError),
                    delay=2.0,
                    backoff=2.0)
    def get_content_checkresult_with_llm(self, 
                                            prompt: str,
                                            response: str, 
                                            question: Dict,
                                            options: List[str]
                                            ) -> Dict:
        """Use the judge LLM to generate the content-check result."""
        json_prompt = json.dumps({
            "prompt": prompt,
            "response": response,
            "question": question,
            "options": options
        }, ensure_ascii=False)

        api_response = self.client['judge_llm'].chat.completions.create(
            model=self.judge_llm,
            messages=[
                {"role": "system", "content": self.meta_prompt['content_judge']},
                {"role": "user", "content": json_prompt}
            ],
            response_format={"type": "json_object"},
            stream=False,
        )
        
        try:
            return(json.loads(clean_json_response(api_response.choices[0].message.content)))
        except json.JSONDecodeError as e:
            print(f"Still failed to parse JSON after cleaning: {api_response.choices[0].message.content}")
            raise ValueError(f"Failed to parse LLM response as valid JSON: {e}") from e

    @combined_retry(timeout_seconds=600, 
                    timeout_retries=2, 
                    error_retries=5, 
                    exceptions=(ValueError, ConnectionError, openai.RateLimitError, openai.APIError),
                    delay=2.0,
                    backoff=2.0)
    def extract_timestamp_with_llm(self, response: str, question: str) -> str:
        """Extract a timestamp from the response given a question."""
        # A minimal prompt for time extraction
        prompt = f"""
        Please extract the precise timestamp relevant to the question from the following response. 
        Question: {question}
        Response: {response}
        Output ONLY the timestamp (e.g., '00:01-00:05', '01:23', '00:10.5'). Do not add any other text.
        """
        
        api_response = self.client['judge_llm'].chat.completions.create(
            model=self.judge_llm,
            messages=[
                {"role": "user", "content": prompt}
            ],
            stream=False,
        )
        return api_response.choices[0].message.content.strip()
    
    def read_data_file(self):
        """Read input data files."""
        # Validate inputs
        if not os.path.exists(self.prompt_input_path):
            raise FileNotFoundError(f"Prompt file not found: {self.prompt_input_path}")
        if not os.path.exists(self.checklist_input_path):
            raise FileNotFoundError(f"Checklist file not found: {self.checklist_input_path}")
        if not os.path.exists(self.response_input_path):
            raise FileNotFoundError(f"Response file not found: {self.response_input_path}")
        
        # Load files
        with open(self.prompt_input_path, 'r', encoding='utf-8') as f:
            prompt_dict = json.load(f)
        print(f"Loaded prompt data from '{self.prompt_input_path}'")
        
        with open(self.checklist_input_path, 'r', encoding='utf-8') as f:
            checklist_dict = json.load(f)
        print(f"Loaded checklist data from '{self.checklist_input_path}'")
        
        with open(self.response_input_path, 'r', encoding='utf-8') as f:
            response_dict = json.load(f)
        print(f"Loaded response data from '{self.response_input_path}'")
        
        # Merge per-video lists to the structure expected by VideoProcessor.
        merged_data = {}
        all_video_ids = set(prompt_dict.keys()) | set(checklist_dict.keys()) | set(response_dict.keys())
        
        for vid in all_video_ids:
            merged_data[vid] = {
                "prompts": prompt_dict.get(vid, []),
                "checklists": checklist_dict.get(vid, []),
                "responses": response_dict.get(vid, [])
            }

        # Ensure prompt_id is present and aligned across prompts/checklists/responses.
        for _, existing_data in merged_data.items():
            _ensure_prompt_ids(existing_data)
        
        # Load existing output if present
        if os.path.exists(self.judge_output_path):
            try:
                with open(self.judge_output_path, 'r', encoding='utf-8') as f:
                    judge_dict = json.load(f)
                print(f"Found existing check result file '{self.judge_output_path}', resuming.")
            except (json.JSONDecodeError, FileNotFoundError):
                print(f"Failed to read result file '{self.judge_output_path}', starting fresh.")
                judge_dict = {}
        else:
            print("No existing result file found; starting fresh.")
            judge_dict = {}
        
        # Update in-memory cache
        removed_total = 0
        pruned_judge_dict = judge_dict

        # Cleanup: if response is empty, remove the corresponding legacy check_result to avoid skewed metrics.
        # Only prune videos present in merged_data; keep other keys as-is.
        for vid, existing_data in merged_data.items():
            if vid not in pruned_judge_dict:
                continue
            pruned_items, removed = _prune_judge_items_for_empty_responses(existing_data, pruned_judge_dict.get(vid, []) or [])
            if removed > 0:
                pruned_judge_dict[vid] = pruned_items
                removed_total += removed

        self.cached_results = pruned_judge_dict

        if removed_total > 0:
            print(
                f"Pruned {removed_total} legacy check_result entries for empty responses; writing back: {self.judge_output_path}"
            )
            try:
                with self.lock:
                    self.save_data_file(self.cached_results)
            except Exception as e:
                print(f"Warning: failed to write back pruned check_result: {e}")

        return merged_data, pruned_judge_dict
    
    def save_data_file(self, judge_dict):
        """Save data file sorted by video_id (atomic write; caller must hold lock)."""
        # Sort by the numeric suffix in video_id
        def get_video_sort_key(video_id):
            parts = video_id.split('_')
            video_part = parts[0]
            video_num = int(parts[-1]) if parts[-1].isdigit() else 0
            
            # Define part order
            part_order = {'clip': 0, 'short': 1, 'long': 2}
            part_idx = part_order.get(video_part, 999)  # unknown part goes last
            
            # Return tuple sort key: (part order, numeric id)
            return (part_idx, video_num)
        
        # Sort dict by video_id
        sorted_judge_dict = dict(sorted(judge_dict.items(), key=lambda x: get_video_sort_key(x[0])))
        
        # Atomic write via temp file to avoid corrupted outputs
        temp_path = f"{self.judge_output_path}.tmp"
        
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(sorted_judge_dict, f, ensure_ascii=False, indent=4)
                f.flush()
                # os.fsync(f.fileno()) # Ensure it is flushed to disk (optional; may be slow)
            
            # Atomic replace
            if os.path.exists(self.judge_output_path):
                try:
                    os.replace(temp_path, self.judge_output_path)
                except OSError:
                    # Retry once for Windows permission issues
                    time.sleep(0.1)
                    if os.path.exists(self.judge_output_path):
                        os.remove(self.judge_output_path)
                    os.rename(temp_path, self.judge_output_path)
            else:
                os.rename(temp_path, self.judge_output_path)
                
        except Exception as e:
            print(f"Error saving data file: {e}")
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            raise e

    def process_video_wrapper(self, video_id: str, existing_data: Dict, 
                            judge_dict: Dict) -> Tuple[str, Dict, str]:
        """Wrapper to process a single video."""
        # Skip only if all prompts with non-empty responses already have check_result.
        if self._is_video_completed(video_id, existing_data, judge_dict):
            # Already processed: create or reuse a log file
            existing_log_path = os.path.join(self.log_dir, f"{video_id}.log")
            if not os.path.exists(existing_log_path):
                # Create a simple skip log
                with open(existing_log_path, 'w', encoding='utf-8') as f:
                    f.write("[SKIPPED] This video has already been fully processed\n")
                    f.write(f"Video ID: {video_id}\n")
                    f.write(f"Skipped at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Existing results: {len(judge_dict.get(video_id, []) or [])}\n")
            return video_id, None, existing_log_path
            
        # Create logger
        logger = VideoLogger(video_id, self.log_dir)
        
        try:
            # Create processor and run
            done_keys = self._get_done_prompt_keys(video_id, judge_dict)
            processor = VideoProcessor(video_id, existing_data, self, logger, done_keys=done_keys)
            video_id, result = processor.process()
            
            # Close log
            logger.close()
            log_path = logger.get_log_path()
            
            return video_id, result, log_path
            
        except Exception as e:
            logger.write(f"Unhandled error while processing: {str(e)}")
            logger.close()
            log_path = logger.get_log_path()
            return video_id, None, log_path

    def process_single_video_independently(self, video_id: str, existing_data: Dict, 
                                         judge_dict: Dict, progress_manager: ProgressManager = None):
        """
        Full independent flow for a single video: process -> save -> update progress.
        
        Args:
            video_id: Video ID.
            existing_data: Per-video data.
            judge_dict: Existing judge results.
            progress_manager: Progress manager.
            
        Returns:
            str: Status ('completed', 'skipped', 'failed').
        """
        try:
            # Process video
            video_id, result, log_path = self.process_video_wrapper(video_id, existing_data, judge_dict)
            
            if result is None:
                # Determine whether it was skipped due to completion.
                if self._is_video_completed(video_id, existing_data, judge_dict):
                    if progress_manager:
                        progress_manager.update('skipped', video_id)
                    # Add log to manager
                    self.log_manager.add_completed_log(video_id, log_path)
                    return 'skipped'
                else:
                    if progress_manager:
                        progress_manager.update('failed', video_id)
                    return 'failed'
            else:
                # Save results
                self._save_video_result(video_id, result['judge'])
                
                if progress_manager:
                    progress_manager.update('completed', video_id)
                
                # Add log to manager
                self.log_manager.add_completed_log(video_id, log_path)
                return 'completed'
                
        except Exception as e:
            print(f"❌ Video {video_id} failed: {str(e)}")
            if progress_manager:
                progress_manager.update('failed', video_id)
            return 'failed'
    
    def _save_video_result(self, video_id: str, result_data: List[Dict]):
        """
        Thread-safe save for a single video's result data (atomic update & write).
        
        Args:
            video_id: Video ID.
            result_data: Result items.
        """
        try:
            with self.lock:
                # Support prompt-level incremental generation: merge into existing results
                existing_list = self.cached_results.get(video_id, []) or []
                merged = {}

                for item in existing_list:
                    merged[_prompt_key(item.get('field'), item.get('prompt_id'))] = item

                for item in result_data:
                    merged[_prompt_key(item.get('field'), item.get('prompt_id'))] = item

                merged_list = list(merged.values())

                def _pid_sort_key(x: Dict) -> tuple[int, str]:
                    pid = str(x.get('prompt_id', ''))
                    return (0, f"{int(pid):09d}") if pid.isdigit() else (1, pid)

                merged_list.sort(key=_pid_sort_key)
                self.cached_results[video_id] = merged_list
                
                # Persist data (save_data_file assumes caller holds the lock)
                self.save_data_file(self.cached_results)
            
        except Exception as e:
            print(f"❌ Error saving results for video {video_id}: {str(e)}")
            raise

    def run(self):
        """Run the check-result generation pipeline (multi-threaded)."""
        start_time = datetime.now()
        
        # 1) Init and load data
        self._print_pipeline_header(start_time)
        prompt_dict, checklist_dict, response_dict, judge_dict = self.read_data_file()
        videos_to_process, skipped_count = self._prepare_video_tasks(
            prompt_dict, checklist_dict, response_dict, judge_dict
        )
        
        total_videos = len(videos_to_process)
        total_all_videos = len(set(prompt_dict.keys()) & set(checklist_dict.keys()) & set(response_dict.keys()))
        
        # 2) Print configuration
        self._print_configuration_info(total_all_videos, total_videos, skipped_count)
        
        if total_videos == 0:
            print("✅ All videos are already processed. Nothing to do.")
            return
        
        # 3) Execute multi-threaded processing
        final_stats = self._execute_multithreaded_processing(videos_to_process)
        
        # 4) Print final stats
        self._print_final_statistics(start_time, total_all_videos, final_stats, skipped_count)
    
    def _print_pipeline_header(self, start_time: datetime):
        """Print pipeline header."""
        print(f"\n{'='*80}")
        print("Check-Only Pipeline started")
        print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Input dir: {self.meta_input_dir}")
        print(f"Output dir: {self.output_dir}")
        print(f"Log dir: {self.log_dir}")
        print(f"{'='*80}\n")
    
    def _prepare_video_tasks(self, prompt_dict: Dict, checklist_dict: Dict, 
                           response_dict: Dict, judge_dict: Dict) -> Tuple[List[Tuple], int]:
        """
        Prepare the task list for videos to process.
        
        Returns:
            Tuple[List[Tuple], int]: (tasks_to_process, skipped_count)
        """
        videos_to_process = []
        skipped_initial = 0
        
        # Validate data consistency
        video_ids = set(prompt_dict.keys()) & set(checklist_dict.keys()) & set(response_dict.keys())
        print(f"Found {len(video_ids)} videos available for processing")
        
        # Recover logs for already-processed videos
        self._recover_completed_logs(judge_dict)
        
        # Build tasks
        for video_id in sorted(video_ids):
            existing_data = {
                'prompts': prompt_dict[video_id],
                'checklists': checklist_dict[video_id],
                'responses': response_dict[video_id]
            }
            
            # Check whether processing is already complete
            if self._is_video_completed(video_id, existing_data, judge_dict):
                skipped_initial += 1
                continue
                
            videos_to_process.append((video_id, existing_data))
        
        return videos_to_process, skipped_initial
    
    def _recover_completed_logs(self, judge_dict: Dict):
        """Recover logs for processed videos."""
        print("Recovering logs for processed videos...")
        recovered_count = 0
        
        for video_id in judge_dict:
            if len(judge_dict.get(video_id, [])) > 0:  # confirm processed
                log_path = os.path.join(self.log_dir, f"{video_id}.log")
                if os.path.exists(log_path):
                    # Log exists; record it.
                    self.log_manager.completed_logs[video_id] = log_path
                    recovered_count += 1
                else:
                    # Processed but log missing; create a placeholder.
                    with open(log_path, 'w', encoding='utf-8') as f:
                        f.write("[RECOVERY] This video was processed but the original log is missing\n")
                        f.write(f"Video ID: {video_id}\n")
                        f.write(f"Recovered at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    self.log_manager.completed_logs[video_id] = log_path
                    recovered_count += 1
        
        if recovered_count > 0:
            print(f"Recovered logs for {recovered_count} processed videos")
            # Regenerate master log
            self.log_manager._regenerate_master_log()
        
        # Cleanup invalid log files
        self._cleanup_invalid_logs(judge_dict)
    
    def _cleanup_invalid_logs(self, judge_dict: Dict):
        """Delete log files for videos without valid results."""
        all_log_files = glob.glob(os.path.join(self.log_dir, "*.log"))
        invalid_logs = 0
        for log_path in all_log_files:
            if os.path.basename(log_path) == "master.log":
                continue
            video_id = os.path.basename(log_path).replace(".log", "")
            # If the corresponding video is not processed, delete the log.
            if video_id not in judge_dict or len(judge_dict.get(video_id, [])) == 0:
                os.remove(log_path)
                invalid_logs += 1
        
        if invalid_logs > 0:
            print(f"Removed {invalid_logs} invalid log files")
    
    def _is_video_completed_legacy(self, video_id: str, judge_dict: Dict) -> bool:
        """Legacy completion check (deprecated): any result means completed."""
        return video_id in judge_dict and len(judge_dict.get(video_id, [])) > 0
    
def process_model(
    model_name: str,
    max_workers: int = 32,
    meta_input_dir: str = './annotation',
    response_input_dir: str = './response',
    output_dir: str = './check_result',
):

    print(f"{'='*30} Starting task {'='*30}")
    print(f"Model: {model_name}")
    print("Judge LLM: gpt-5-mini")
    print(f"Workers: {max_workers}")

    try:
        # Initialize pipeline
        pipeline = CheckOnlyPipeline(
            model_name=model_name, 
            meta_input_dir=meta_input_dir,
            response_input_dir=response_input_dir,
            output_dir=output_dir,
            max_workers=max_workers,
        )
        
        # Load data
        print("Loading merged data...")
        try:
            merged_data, judge_dict = pipeline.read_data_file()
        except FileNotFoundError as e:
            print(f"Skipping model {model_name}: {e}")
            return

        # Prepare task list
        video_ids = sorted(list(merged_data.keys()))
        total_tasks = len(video_ids)
        
        print(f"Total videos: {total_tasks}")
        
        # Count completed tasks
        completed_count = 0
        tasks_to_process = []
        
        for vid in video_ids:
            existing_data = merged_data.get(vid, {})
            if pipeline._is_video_completed(vid, existing_data, judge_dict):
                completed_count += 1
            else:
                tasks_to_process.append(vid)
                
        print(f"Completed: {completed_count}, Remaining: {len(tasks_to_process)}")

        if not tasks_to_process:
            print(f"All tasks completed for model {model_name}.")
        else:
            # Progress manager
            print("Starting multi-threaded processing...")
            with ProgressManager(len(tasks_to_process), desc="Check progress") as progress:
                # Run concurrently with a thread pool
                with ThreadPoolExecutor(max_workers=pipeline.max_workers) as executor:
                    # Submit pending tasks
                    future_to_video = {
                        executor.submit(
                            pipeline.process_single_video_independently, 
                            vid, 
                            merged_data[vid], 
                            judge_dict,
                            progress
                        ): vid for vid in tasks_to_process
                    }
                    
                    # Consume completed tasks
                    for future in as_completed(future_to_video):
                        video_id = future_to_video[future]
                        try:
                            status = future.result()
                        except Exception as exc:
                            print(f"Video {video_id} raised an exception: {exc}")
                            import traceback
                            traceback.print_exc()

            print(f"\nAll tasks finished. Results saved to: {pipeline.judge_output_path}")

        # Check whether the result file exists
        if not os.path.exists(pipeline.judge_output_path):
            print("\n❌ Error: result file was not generated. Possibly all tasks failed.")
            print(f"Please check the log directory: {pipeline.log_dir}")
            return

        # Compute and print score summary
        try:
            from metrics import ScoreCalculator
            print("\nComputing score statistics...")
            
            # Load the newly generated result file
            with open(pipeline.judge_output_path, 'r', encoding='utf-8') as f:
                check_data = json.load(f)
                
            calculator = ScoreCalculator(check_data)
            results = calculator.calculate_all_scores()
            
            print("\n" + "="*50)
            print(f"              {model_name} Evaluation Summary              ")
            print("="*50)
            
            print(f"{'Metric':<25} {'ISR (Instruction)':<20} {'CSR (Constraint)':<20}")
            print("-" * 65)
            print(f"{'Overall':<25} {results.isr*100:>6.2f}%             {results.csr*100:>6.2f}%")
            print(f"{'Format':<25} {results.format_isr*100:>6.2f}%             {results.format_csr*100:>6.2f}%")
            print(f"{'Content':<25} {results.content_isr*100:>6.2f}%             {results.content_csr*100:>6.2f}%")
            print("-" * 65)

            # Save detailed results to the metrics directory
            import dataclasses
            metrics_dir = "./metrics"
            if not os.path.exists(metrics_dir):
                os.makedirs(metrics_dir)
            
            # Convert dataclass to dict
            results_dict = dataclasses.asdict(results)
            
            # Save metrics result
            metrics_output_path = os.path.join(metrics_dir, f"{model_name}_metrics.json")
            with open(metrics_output_path, 'w', encoding='utf-8') as f:
                json.dump(results_dict, f, ensure_ascii=False, indent=2)
            
            print(f"\nDetailed metrics saved to: {metrics_output_path}")
            
        except Exception as e:
            print(f"\nError while computing scores: {e}")
            import traceback
            traceback.print_exc()

    except Exception as e:
        print(f"Fatal error while running the program: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description='Check Only Pipeline')
    parser.add_argument('--models', type=str, nargs='+', 
                        help='Model names, or "all" to process all *_response.json files in --response_dir')
    parser.add_argument('--workers', type=int, default=32, help='Number of worker threads')
    parser.add_argument('--meta_dir', type=str, default='./annotation', help='Directory containing prompts/checklists (default: ./annotation)')
    parser.add_argument('--response_dir', type=str, default='./response', help='Directory containing model responses (default: ./response)')
    parser.add_argument('--output_dir', type=str, default='./check_result', help='Output directory for check_result (default: ./check_result)')
    # Default behavior if no --models is provided: run the bundled example.
    
    args = parser.parse_args()
    
    models_to_process = []
    
    if args.models:
        if args.models == ['all']:
            print("Scanning response directory for models...")
            response_dir = args.response_dir
            if os.path.exists(response_dir):
                files = glob.glob(os.path.join(response_dir, '*_response.json'))
                for f in files:
                    fname = os.path.basename(f)
                    # Exclude non-response files if any, though globs handles ending
                    model_name = fname.replace('_response.json', '')
                    models_to_process.append(model_name)
                
                if not models_to_process:
                    print("No *_response.json files found under the response directory")
            else:
                print(f"Directory does not exist: {response_dir}")
        else:
            models_to_process = args.models
    else:
        # Default behavior if no args provided (preserve old behavior)
        models_to_process = ['example_model']
    
    # Sort for consistent order
    models_to_process.sort()
    
    print(f"Models to process: {models_to_process}")
    
    for model_name in models_to_process:
        process_model(
            model_name,
            args.workers,
            meta_input_dir=args.meta_dir,
            response_input_dir=args.response_dir,
            output_dir=args.output_dir,
        )

if __name__ == "__main__":
    main()

