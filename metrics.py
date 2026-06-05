import json
import argparse
from typing import Dict, List, Any, Tuple
from collections import defaultdict
from dataclasses import dataclass, field
import pandas as pd
import os
import numpy as np

@dataclass
class ScoreResults:
    """Aggregated scoring results."""

    # Overall metrics
    isr: float = 0.0  # instruction satisfaction rate
    csr: float = 0.0  # constraint satisfaction rate

    # Format metrics
    format_isr: float = 0.0
    format_csr: float = 0.0

    # Content metrics
    content_isr: float = 0.0
    content_csr: float = 0.0
    content_visual_csr: float = 0.0
    content_audio_csr: float = 0.0
    content_omni_csr: float = 0.0

    # Capability scores by constraint dimension
    constraint_dimension_scores: Dict[str, float] = field(default_factory=dict)
    
    # Detailed stats
    stats: Dict[str, Any] = field(default_factory=dict)
    
    # Per-video scores
    video_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)

class ScoreCalculator:
    """Score calculator."""
    
    def __init__(self, data: Dict[str, Any], checklist_data: Dict[str, Any] = None):
        self.data = data
        self.checklist_data = checklist_data if checklist_data else {}

        self.results = ScoreResults()
        self.stats = defaultdict(lambda: defaultdict(int))
        self.video_scores = defaultdict(lambda: {
            'ISR': 0.0,
            'CSR': 0.0,
            'Format ISR': 0.0,
            'Format CSR': 0.0,
            'Content ISR': 0.0,
            'Content CSR': 0.0,
            'Content Visual CSR': 0.0,
            'Content Audio CSR': 0.0,
            'Content Omni CSR': 0.0
        })

    def _is_prompt_satisfied(self, check_item: Dict, mode: str = 'all', video_id: str = None) -> float:
        """
        Decide whether a prompt is satisfied (ISR is 1.0 or 0.0).
        mode: 'all', 'format', 'content'

        Semantics: AND-only. A prompt is satisfied iff all participating constraints pass.
        Any legacy fields are ignored.
        """
        check_result = check_item.get('check_result', {})
        constraints_passed = []
        
        # 1) Collect format constraints
        if mode in ['all', 'format']:
             for item in check_result.get('format_check', []):
                constraints_passed.append(item.get('result', False))

        # 2) Collect content constraints
        if mode in ['all', 'content']:
            for item in check_result.get('content_check', []):
                check_items = item.get('check_items', [])

                if not check_items:
                    continue
                
                # A content constraint is satisfied iff all its check_items pass.
                constraints_passed.append(all(i.get('result', False) for i in check_items))

        if not constraints_passed:
            return None # Indicate N/A, no constraints found for this mode

        # AND-only semantics: any failed constraint => 0, all passed => 1
        return 1.0 if all(constraints_passed) else 0.0

    def calculate_all_scores(self) -> ScoreResults:
        """Compute all metrics."""
        self._calculate_isr()
        self._calculate_csr()
        self._calculate_format_isr_csr()
        self._calculate_content_isr_csr()
        self._calculate_constraint_dimension_scores()
        
        self.results.stats = dict(self.stats)
        self.results.video_scores = dict(self.video_scores)
        return self.results
    
    def _calculate_csr(self):
        """Compute constraint satisfaction rate (CSR)."""
        video_csr_scores = defaultdict(float)
        
        # Global stats
        global_constraints_passed = 0
        global_constraints_total = 0
        
        for video_id, case in self.data.items():
            for check_item in case:
                check_result = check_item.get('check_result', {})
                
                # Constraint count stats
                format_constraints = len(check_result.get('format_check', []))
                content_constraints = len(check_result.get('content_check', []))
                total_constraints = format_constraints + content_constraints
                
                if total_constraints == 0:
                    continue
                
                global_constraints_total += total_constraints
                
                # Count passed constraints
                passed_constraints = 0
                
                # Format checks: each format_check is one constraint
                for format_check in check_result.get('format_check', []):
                    if format_check.get('result', False):
                        passed_constraints += 1
                
                # Content checks: each content_check entry is one constraint
                for content_check in check_result.get('content_check', []):
                    check_items = content_check.get('check_items', [])
                    if not check_items:
                        continue
                    
                    # A content constraint is satisfied iff all its check_items pass.
                    constraint_satisfied = True
                    for item in check_items:
                        if not item.get('result', False):
                            constraint_satisfied = False
                            break
                    
                    if constraint_satisfied:
                        passed_constraints += 1
                
                # CSR for this prompt
                csr = passed_constraints / total_constraints if total_constraints > 0 else 0
                
                # Accumulate per-video stats (for video-level averages)
                video_csr_scores[video_id] += csr
                
                # Accumulate global stats
                global_constraints_passed += passed_constraints
        
        # Compute global CSR
        self.results.csr = global_constraints_passed / global_constraints_total if global_constraints_total > 0 else 0
        
        # Save stats
        self.stats['csr']['global_constraints_total'] = global_constraints_total
        self.stats['csr']['global_constraints_passed'] = global_constraints_passed
        
        # Compute per-video average CSR (over all prompts in that video)
        for video_id, total_score in video_csr_scores.items():
            # Count prompts for this video
            prompt_count = len([item for item in self.data.get(video_id, []) if item.get('check_result')])
            if prompt_count > 0:
                self.video_scores[video_id]['CSR'] = total_score / prompt_count

    
    def _calculate_isr(self):
        """Compute instruction satisfaction rate (ISR)."""
        total_prompts = 0
        fully_satisfied_prompts = 0
        video_isr_accumulators = defaultdict(lambda: {'total': 0, 'satisfied': 0})
        
        for video_id, case in self.data.items():
            for check_item in case:
                # Calculate ISR for this prompt (AND-only semantics)
                isr_score = self._is_prompt_satisfied(check_item, mode='all', video_id=video_id)
                
                if isr_score is not None:
                    total_prompts += 1
                    fully_satisfied_prompts += isr_score
                    
                    video_isr_accumulators[video_id]['total'] += 1
                    video_isr_accumulators[video_id]['satisfied'] += isr_score
        
        self.results.isr = fully_satisfied_prompts / total_prompts if total_prompts > 0 else 0.0

        for video_id, acc in video_isr_accumulators.items():
            if acc['total'] > 0:
                self.video_scores[video_id]['ISR'] = acc['satisfied'] / acc['total']

    def _calculate_format_isr_csr(self):
        """Compute format ISR/CSR."""
        format_total = 0
        format_passed = 0
        format_video_constraints = defaultdict(lambda: {'total': 0, 'passed': 0})
        
        total_prompts = 0
        fully_satisfied_prompts = 0
        video_isr_accumulators = defaultdict(lambda: {'total': 0, 'satisfied': 0})
        
        for video_id, case in self.data.items():
            for check_item in case:
                check_result = check_item.get('check_result', {})
                format_checks = check_result.get('format_check', [])
                valid_count = len(format_checks)
                
                if valid_count > 0:
                    format_total += valid_count
                    format_video_constraints[video_id]['total'] += valid_count
                    
                    passed_count = sum(1 for rc in format_checks if rc.get('result', False))
                    format_passed += passed_count
                    format_video_constraints[video_id]['passed'] += passed_count

                    # ISR Calculation
                    isr_score = self._is_prompt_satisfied(check_item, mode='format', video_id=video_id)
                    if isr_score is not None:
                        total_prompts += 1
                        fully_satisfied_prompts += isr_score
                        video_isr_accumulators[video_id]['total'] += 1
                        video_isr_accumulators[video_id]['satisfied'] += isr_score
        
        # Results
        self.results.format_csr = format_passed / format_total if format_total > 0 else 0.0
        self.results.format_isr = fully_satisfied_prompts / total_prompts if total_prompts > 0 else 0.0
        
        # Stats
        self.stats['format_csr'] = {
            'total_constraints': format_total,
            'passed_constraints': format_passed
        }
        self.stats['format_isr'] = {
            'total_prompts': total_prompts,
            'fully_satisfied_prompts': fully_satisfied_prompts
        }
        
        # Video scores
        for video_id in self.data.keys():
            vc = format_video_constraints[video_id]
            if vc['total'] > 0:
                self.video_scores[video_id]['Format CSR'] = vc['passed'] / vc['total']
            visr = video_isr_accumulators[video_id]
            if visr['total'] > 0:
                 self.video_scores[video_id]['Format ISR'] = visr['satisfied'] / visr['total']

    def _calculate_content_isr_csr(self):
        """Compute content ISR/CSR."""
        content_total = 0
        content_passed = 0
        content_video_constraints = defaultdict(lambda: {'total': 0, 'passed': 0})
        
        # New counters for sub-categories
        sub_categories = ['visual', 'audio', 'omni']
        sub_cat_stats = {cat: {'total': 0, 'passed': 0} for cat in sub_categories}
        sub_cat_video_stats = {cat: defaultdict(lambda: {'total': 0, 'passed': 0}) for cat in sub_categories}
        
        total_prompts = 0
        fully_satisfied_prompts = 0
        video_isr_accumulators = defaultdict(lambda: {'total': 0, 'satisfied': 0})
        
        for video_id, case in self.data.items():
            for check_item in case:
                check_result = check_item.get('check_result', {})
                content_checks = check_result.get('content_check', [])
                valid_count = 0 
                passed_count = 0
                
                for content_check in content_checks:
                    check_items = content_check.get('check_items', [])
                    if not check_items:
                        continue
                    
                    valid_count += 1
                    content_total += 1
                    content_video_constraints[video_id]['total'] += 1
                    
                    # CSR computation
                    constraint_satisfied = all(item.get('result', False) for item in check_items)
                    
                    # Determine category
                    constraint_id = content_check.get('constraint_id', '')
                    category = None
                    if 'visual' in constraint_id:
                        category = 'visual'
                    elif 'audio' in constraint_id:
                        category = 'audio'
                    elif 'omni' in constraint_id:
                        category = 'omni'
                    
                    if category:
                         sub_cat_stats[category]['total'] += 1
                         sub_cat_video_stats[category][video_id]['total'] += 1

                    if constraint_satisfied:
                        content_passed += 1
                        content_video_constraints[video_id]['passed'] += 1
                        passed_count += 1
                        
                        if category:
                             sub_cat_stats[category]['passed'] += 1
                             sub_cat_video_stats[category][video_id]['passed'] += 1
                
                if valid_count > 0:
                    isr_score = self._is_prompt_satisfied(check_item, mode='content', video_id=video_id)
                    if isr_score is not None:
                        total_prompts += 1
                        fully_satisfied_prompts += isr_score
                        video_isr_accumulators[video_id]['total'] += 1
                        video_isr_accumulators[video_id]['satisfied'] += isr_score

        # Results
        self.results.content_csr = content_passed / content_total if content_total > 0 else 0.0
        self.results.content_isr = fully_satisfied_prompts / total_prompts if total_prompts > 0 else 0.0
        
        # Stats
        self.stats['content_csr'] = {
            'total_constraints': content_total,
            'passed_constraints': content_passed
        }
        self.stats['content_isr'] = {
            'total_prompts': total_prompts,
            'fully_satisfied_prompts': fully_satisfied_prompts
        }
        
        # Sub-category Results and Stats
        for cat in sub_categories:
             total = sub_cat_stats[cat]['total']
             passed = sub_cat_stats[cat]['passed']
             csr = passed / total if total > 0 else 0.0
             setattr(self.results, f'content_{cat}_csr', csr)
             
             self.stats[f'content_{cat}_csr'] = {
                 'total_constraints': total,
                 'passed_constraints': passed
             }

        # Video scores
        for video_id in self.data.keys():
            vc = content_video_constraints[video_id]
            if vc['total'] > 0:
                self.video_scores[video_id]['Content CSR'] = vc['passed'] / vc['total']
            visr = video_isr_accumulators[video_id]
            if visr['total'] > 0:
                 self.video_scores[video_id]['Content ISR'] = visr['satisfied'] / visr['total']
            
            # Sub-category Video Scores
            for cat in sub_categories:
                svc = sub_cat_video_stats[cat][video_id]
                if svc['total'] > 0:
                    self.video_scores[video_id][f'Content {cat.capitalize()} CSR'] = svc['passed'] / svc['total']

    def _calculate_constraint_dimension_scores(self):
        """Compute capability scores grouped by constraint dimensions."""
        constraint_categories = {
            'format': ['format'],
            'content': ['content'],
            'relation': ['logical', 'conditional']
        }
        
        category_scores = defaultdict(lambda: {'total': 0, 'passed': 0})
        
        for video_id, case in self.data.items():
            for check_item in case:
                check_result = check_item.get('check_result', {})

                # Format constraints
                for format_check in check_result.get('format_check', []):
                    constraint_id = format_check.get('constraint_id', '')
                    result = format_check.get('result', False)
                    for category, keywords in constraint_categories.items():
                        if any(keyword in constraint_id.lower() for keyword in keywords):
                            category_scores[category]['total'] += 1
                            if result:
                                category_scores[category]['passed'] += 1
                            break

                # Content constraints
                for content_check in check_result.get('content_check', []):
                    constraint_id = content_check.get('constraint_id', '')
                    check_items = content_check.get('check_items', [])
                    if not check_items:
                        continue
                    result = all(item.get('result', False) for item in check_items)
                    
                    # Categorize by constraint_id
                    for category, keywords in constraint_categories.items():
                        if any(keyword in constraint_id.lower() for keyword in keywords):
                            category_scores[category]['total'] += 1
                            if result:
                                category_scores[category]['passed'] += 1
                            break
        
        # Compute scores per dimension
        for category, scores in category_scores.items():
            if scores['total'] > 0:
                score = scores['passed'] / scores['total']
                self.results.constraint_dimension_scores[f'{category}_score'] = score
                self.stats['constraint_dimensions'][category] = scores


def calculate_prompt_scores(check_item: Dict[str, Any]) -> Dict[str, float]:
    """Compute per-prompt metrics."""
    check_result = check_item.get('check_result', {})
    
    # Constraint count stats
    format_constraints = len(check_result.get('format_check', []))
    content_constraints = len(check_result.get('content_check', []))
    total_constraints = format_constraints + content_constraints
    
    if total_constraints == 0:
        return {
            'isr': 0.0,
            'csr': 0.0,
            'format_isr': 0.0,
            'format_csr': 0.0,
            'content_isr': 0.0,
            'content_csr': 0.0,
            'content_visual_csr': 0.0,
            'content_audio_csr': 0.0,
            'content_omni_csr': 0.0
        }
    
    # Count passed constraints
    passed_constraints = 0
    
    # Format checks
    format_passed = 0
    for format_check in check_result.get('format_check', []):
        if format_check.get('result', False):
            passed_constraints += 1
            format_passed += 1
    
    # Content checks
    content_passed = 0
    
    # Sub-category stats
    sub_categories = ['visual', 'audio', 'omni']
    sub_cat_stats = {cat: {'total': 0, 'passed': 0} for cat in sub_categories}

    for content_check in check_result.get('content_check', []):
        check_items = content_check.get('check_items', [])
        if not check_items:
            continue
        
        # Determine category
        constraint_id = content_check.get('constraint_id', '')
        category = None
        if 'visual' in constraint_id:
            category = 'visual'
        elif 'audio' in constraint_id:
            category = 'audio'
        elif 'omni' in constraint_id:
            category = 'omni'

        if category:
                sub_cat_stats[category]['total'] += 1

        # A content constraint is satisfied iff all its check_items pass.
        constraint_satisfied = True
        for item in check_items:
            if not item.get('result', False):
                constraint_satisfied = False
                break
        
        if constraint_satisfied:
            passed_constraints += 1
            content_passed += 1
            if category:
                sub_cat_stats[category]['passed'] += 1
    
    # Compute metrics
    csr = passed_constraints / total_constraints if total_constraints > 0 else 0
    
    # ISR: AND-only semantics
    # ISR is 1 only if all counted constraints are satisfied.
    isr = 1.0 if passed_constraints == total_constraints else 0.0
    
    # Format metrics
    format_csr = format_passed / format_constraints if format_constraints > 0 else 0
    format_isr = 1.0 if format_passed == format_constraints else 0.0
    
    # Content metrics
    content_csr = content_passed / content_constraints if content_constraints > 0 else 0
    content_isr = 1.0 if content_passed == content_constraints else 0.0
    
    # Content sub-category metrics
    content_visual_csr = sub_cat_stats['visual']['passed'] / sub_cat_stats['visual']['total'] if sub_cat_stats['visual']['total'] > 0 else 0.0
    content_audio_csr = sub_cat_stats['audio']['passed'] / sub_cat_stats['audio']['total'] if sub_cat_stats['audio']['total'] > 0 else 0.0
    content_omni_csr = sub_cat_stats['omni']['passed'] / sub_cat_stats['omni']['total'] if sub_cat_stats['omni']['total'] > 0 else 0.0

    return {
        'isr': isr,
        'csr': csr,
        'format_isr': format_isr,
        'format_csr': format_csr,
        'content_isr': content_isr,
        'content_csr': content_csr,
        'content_visual_csr': content_visual_csr,
        'content_audio_csr': content_audio_csr,
        'content_omni_csr': content_omni_csr
    }


def _normalize_prompt_id(prompt_id: Any) -> str:
    """Normalize prompt_id to a comparable string.

    Common values are '01'~'04', but 1~4 are also supported.
    """
    if prompt_id is None:
        return ''
    pid = str(prompt_id).strip()
    return pid.zfill(2) if pid.isdigit() else pid


def count_videos_with_complete_prompts(
    data: Dict[str, Any],
    required_prompt_ids: Tuple[str, ...] = ('01', '02', '03', '04'),
) -> int:
    """Count videos that contain all required prompts with non-empty check_result."""
    required_set = {_normalize_prompt_id(pid) for pid in required_prompt_ids}
    complete_videos = 0

    for video_id, items in data.items():
        present_prompt_ids = set()
        for check_item in items:
            if not check_item.get('check_result'):
                continue
            pid = _normalize_prompt_id(check_item.get('prompt_id'))
            if pid:
                present_prompt_ids.add(pid)

        if required_set.issubset(present_prompt_ids):
            complete_videos += 1

    return complete_videos

def process_multiple_models(model_names: List[str], input_folder: str, output_folder: str):
    """Process multiple models and generate Excel outputs."""
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Store model-level results
    all_model_results = []
    
    # Store prompt-level results (per prompt)
    prompt_scores_by_model = defaultdict(list)
    
    # Load checklist data (optional)
    checklist_path = os.path.join("annotation", "checklists.json")
    checklist_data = {}
    if os.path.exists(checklist_path):
        with open(checklist_path, 'r', encoding='utf-8') as f:
            checklist_data = json.load(f)
    else:
        print(f"Warning: Checklist file not found at {checklist_path}")

    for model_name in model_names:
        print(f"Processing model: {model_name}")
        
        if model_name == 'baseline':
            input_file = os.path.join(input_folder, "check_result.json")
        else:
            input_file = os.path.join(input_folder, f"{model_name}_check_result.json")

        # Check file exists
        if not os.path.exists(input_file):
            print(f"Warning: File not found - {input_file}")
            continue
        
        # Load data
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Count videos that cover all 4 prompts (computed but not exported)
        _complete_video_count = count_videos_with_complete_prompts(data)
        
        # Compute scores
        calculator = ScoreCalculator(data, checklist_data)
        results = calculator.calculate_all_scores()
        
        # Convert to percentages (2 decimals); CSR first, ISR second
        results.isr = round(results.isr * 100, 2)
        results.csr = round(results.csr * 100, 2)
        results.format_isr = round(results.format_isr * 100, 2)
        results.format_csr = round(results.format_csr * 100, 2)
        results.content_isr = round(results.content_isr * 100, 2)
        results.content_csr = round(results.content_csr * 100, 2)
        results.content_visual_csr = round(results.content_visual_csr * 100, 2)
        results.content_audio_csr = round(results.content_audio_csr * 100, 2)
        results.content_omni_csr = round(results.content_omni_csr * 100, 2)
        
        for k in results.constraint_dimension_scores:
            results.constraint_dimension_scores[k] = round(results.constraint_dimension_scores[k] * 100, 2)
        
        # Convert video-level scores to percentages
        for video_id, scores in results.video_scores.items():
            scores['ISR'] = round(scores['ISR'] * 100, 2)
            scores['CSR'] = round(scores['CSR'] * 100, 2)
            scores['Format ISR'] = round(scores['Format ISR'] * 100, 2)
            scores['Format CSR'] = round(scores['Format CSR'] * 100, 2)
            scores['Content ISR'] = round(scores['Content ISR'] * 100, 2)
            scores['Content CSR'] = round(scores['Content CSR'] * 100, 2)
            scores['Content Visual CSR'] = round(scores['Content Visual CSR'] * 100, 2)
            scores['Content Audio CSR'] = round(scores['Content Audio CSR'] * 100, 2)
            scores['Content Omni CSR'] = round(scores['Content Omni CSR'] * 100, 2)

        # Collect model-level results; CSR first, ISR second
        model_result = {
            'Model': model_name,
            'CSR': results.csr,
            'ISR': results.isr,
            'Format CSR': results.format_csr,
            'Format ISR': results.format_isr,
            'Content CSR': results.content_csr,
            'Content ISR': results.content_isr,
            'Content Visual CSR': results.content_visual_csr,
            'Content Audio CSR': results.content_audio_csr,
            'Content Omni CSR': results.content_omni_csr,
        }
        
        # Add constraint-dimension scores
        model_result.update(results.constraint_dimension_scores)
        
        all_model_results.append(model_result)
        
        # Collect prompt-level results
        prompt_scores = []
        for video_id, case in data.items():
            for check_item in case:
                prompt_id = check_item.get('prompt_id', '')
                if prompt_id:
                    # Compute per-prompt metrics
                    prompt_score = calculate_prompt_scores(check_item)
                    
                    prompt_score_entry = {
                        'Model': model_name,
                        'video_id': video_id,
                        'prompt_id': prompt_id,
                        'CSR': round(prompt_score['csr'] * 100, 2),
                        'ISR': round(prompt_score['isr'] * 100, 2),
                        'Format CSR': round(prompt_score['format_csr'] * 100, 2),
                        'Format ISR': round(prompt_score['format_isr'] * 100, 2),
                        'Content CSR': round(prompt_score['content_csr'] * 100, 2),
                        'Content ISR': round(prompt_score['content_isr'] * 100, 2),
                        'Content Visual CSR': round(prompt_score['content_visual_csr'] * 100, 2),
                        'Content Audio CSR': round(prompt_score['content_audio_csr'] * 100, 2),
                        'Content Omni CSR': round(prompt_score['content_omni_csr'] * 100, 2)
                    }
                    prompt_scores.append(prompt_score_entry)
        
        prompt_scores_by_model[model_name].extend(prompt_scores)

    # Write summary Excel
    if all_model_results:
        # Model-level metrics summary
        df_models = pd.DataFrame(all_model_results)
        
        # Sort by CSR (desc)
        df_models = df_models.sort_values('CSR', ascending=False)
        
        # Merge prompt-level data across models
        all_prompt_scores = []
        for model_name, scores in prompt_scores_by_model.items():
            all_prompt_scores.extend(scores)
        
        # Save to Excel (two sheets)
        metrics_excel_path = os.path.join(output_folder, "metrics.xlsx")
        with pd.ExcelWriter(metrics_excel_path, engine='openpyxl') as writer:
            # Sheet 1: Main metrics
            main_columns = ['Model', 'CSR', 'ISR', 'Format CSR', 'Format ISR',
                          'Content CSR', 'Content ISR', 'Content Visual CSR',
                          'Content Audio CSR', 'Content Omni CSR']
            available_main = [col for col in main_columns if col in df_models.columns]
            df_main = df_models[available_main]
            df_main.to_excel(writer, sheet_name='Main Metrics', index=False)
            
            # Sheet 2: Additional metrics (e.g., constraint dimension scores)
            detailed_columns = ['Model']
            detailed_columns.extend([col for col in df_models.columns if col not in main_columns and col not in detailed_columns])
            available_detailed = [col for col in detailed_columns if col in df_models.columns]
            df_detailed = df_models[available_detailed]
            df_detailed.to_excel(writer, sheet_name='Detailed Metrics', index=False)
            
            # Sheet 3: All metrics (full table)
            df_models.to_excel(writer, sheet_name='All Metrics', index=False)
            
            # Sheet 4: Prompt-level detailed metrics
            if all_prompt_scores:
                df_prompt_scores = pd.DataFrame(all_prompt_scores)
                df_prompt_scores = df_prompt_scores.sort_values(['Model', 'video_id', 'prompt_id'])
                df_prompt_scores.to_excel(writer, sheet_name='Prompt Detailed Scores', index=False)
            
            # Auto-adjust column widths
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
        
        print(f"Model metrics Excel saved to: {metrics_excel_path}")
        
        # Generate LaTeX tables and save to txt
        latex_table_path = os.path.join(output_folder, "metrics_latex_table.txt")
        generate_latex_table(df_models, latex_table_path)
        print(f"LaTeX tables saved to:")
        print(f"  - {latex_table_path}")
        print(f"  - {latex_table_path.replace('.txt', '_main.txt')}")
        print(f"  - {latex_table_path.replace('.txt', '_detailed.txt')}")
    else:
        df_models = None
    
    # Merge prompt-level data across models (for return)
    all_prompt_scores = []
    for model_name, scores in prompt_scores_by_model.items():
        all_prompt_scores.extend(scores)
    
    df_prompt_detailed = None
    if all_prompt_scores:
        df_prompt_detailed = pd.DataFrame(all_prompt_scores)
        df_prompt_detailed = df_prompt_detailed.sort_values(['Model', 'video_id', 'prompt_id'])
    
    return df_models, df_prompt_detailed

def generate_latex_table(df_models, output_file):
    """Generate LaTeX table lines and save to txt (only contains '&' and '\\')."""
    if df_models is None or df_models.empty:
        return
    
    latex_lines_main = []  # Main-metrics table
    latex_lines_detailed = []  # Detailed-metrics table
    
    # Table 1: main metrics
    main_columns = [
        'Model', 'CSR', 'ISR',
        'Format CSR', 'Format ISR',
        'Content CSR', 'Content ISR',
        'Content Visual CSR', 'Content Audio CSR', 'Content Omni CSR',
    ]
    
    # Table 2: remaining metrics (e.g., constraint dimensions)
    detailed_columns = ['Model']
    detailed_columns.extend([c for c in df_models.columns if c not in main_columns and c not in detailed_columns])
    
    # Build main-metrics table
    available_main_columns = [col for col in main_columns if col in df_models.columns]
    df_main = df_models[available_main_columns]
    
    non_percent_columns = {'Model'}

    for _, row in df_main.iterrows():
        row_values = []
        for col in available_main_columns:
            value = row[col]
            if col in non_percent_columns:
                row_values.append(str(value))
            else:
                row_values.append(f"{value:.2f}\\%")
        latex_line = " & ".join(row_values) + " \\\\"
        latex_lines_main.append(latex_line)
    
    # Build detailed-metrics table
    available_detailed_columns = [col for col in detailed_columns if col in df_models.columns]
    df_detailed = df_models[available_detailed_columns]
    
    for _, row in df_detailed.iterrows():
        row_values = []
        for col in available_detailed_columns:
            value = row[col]
            if col == 'Model':
                row_values.append(str(value))
            else:
                row_values.append(f"{value:.2f}\\%")
        latex_line = " & ".join(row_values) + " \\\\"
        latex_lines_detailed.append(latex_line)
    
    # Save to file; keep two sections for compatibility
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("% Table 1: Main Metrics\n")
        f.write('\n'.join(latex_lines_main))
        f.write("\n\n% Table 2: Detailed Metrics\n")
        f.write('\n'.join(latex_lines_detailed))
    
    # Also save as two separate files
    main_table_file = output_file.replace('.txt', '_main.txt')
    detailed_table_file = output_file.replace('.txt', '_detailed.txt')
    
    with open(main_table_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(latex_lines_main))
    
    with open(detailed_table_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(latex_lines_detailed))

def generate_report(results: ScoreResults, output_file: str = None):
    """Generate a human-readable report."""
    report = []
    report.append("=" * 50)
    report.append("Score Report")
    report.append("=" * 50)
    
    # Overall metrics; CSR first, ISR second
    report.append("\n## Overall")
    report.append(f"CSR: {results.csr:.2%}")
    report.append(f"ISR: {results.isr:.2%}")
    
    # Format metrics
    report.append("\n## Format")
    report.append(f"Format CSR: {results.format_csr:.2%}")
    report.append(f"Format ISR: {results.format_isr:.2%}")
    
    # Content metrics
    report.append("\n## Content")
    report.append(f"Content CSR: {results.content_csr:.2%}")
    report.append(f"Content ISR: {results.content_isr:.2%}")
    
    # Constraint-dimension metrics
    report.append("\n## Constraint Dimensions")
    report.append("Scores:")
    for dimension, score in results.constraint_dimension_scores.items():
        report.append(f"  - {dimension}: {score:.2%}")
    
    report_text = "\n".join(report)
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
    else:
        print(report_text)
    
    return report_text


def main():
    parser = argparse.ArgumentParser(description='Compute evaluation metrics')
    parser.add_argument('--models', type=str, nargs='+', required=True, 
                        help='Model names. Use "all" to auto-detect from input folder.')
    parser.add_argument('--part', type=str, required=False, default=None,
                        help='Dataset split (optional), e.g., easy/hard/breeze')
    
    args = parser.parse_args()
    
    # Root folder configuration
    if args.models == ['baseline']:
        input_root = 'annotation'  # Input root folder
    else:
        input_root = 'check_result'
        
    output_root = 'metrics'    # Output root folder
    
    # Configure input/output by part
    if args.part is not None:
        input_folder = os.path.join(input_root, f'{args.part}')
        output_folder = os.path.join(output_root, f'{args.part}')
    else:
        input_folder = input_root
        output_folder = output_root

    # Auto-detect model names
    if args.models == ['all']:
        print(f"Auto-detecting models; scanning folder: {input_folder}")
        
        # Ensure input folder exists
        if not os.path.exists(input_folder):
            print(f"Error: Input folder not found - {input_folder}")
            return
        
        # Scan for check result files
        model_names = []
        
        # Add baseline model if check_result.json exists
        baseline_file = os.path.join(input_folder, "check_result.json")
        if os.path.exists(baseline_file):
            model_names.append('baseline')
        
        # Scan for *_check_result.json files
        for filename in os.listdir(input_folder):
            if filename.endswith('_check_result.json') and filename != 'check_result.json':
                # Extract model name (strip _check_result.json suffix)
                model_name = filename[:-len('_check_result.json')]
                model_names.append(model_name)
        
        if not model_names:
            print(f"Warning: No check result files found in {input_folder}")
            return
        
        # Sort model names alphabetically
        model_names.sort()
        print(f"Found {len(model_names)} models: {', '.join(model_names)}")
        
        # Update args.models
        args.models = model_names

    print(f"Processing part: {args.part}")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Models to process: {args.models}")
    
    # Process multiple models
    df_models, df_prompt_detailed = process_multiple_models(
        args.models, 
        input_folder, 
        output_folder
    )
    
    # Show summary
    if df_models is not None:
        print("\nModel-level metrics summary:")
        print(df_models.to_string(index=False))
        
        print("\nPrompt-level detailed metrics sample (first 10 rows):")
        if df_prompt_detailed is not None and not df_prompt_detailed.empty:
            # Show first 10 rows of prompt-level detailed metrics
            print(df_prompt_detailed.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

