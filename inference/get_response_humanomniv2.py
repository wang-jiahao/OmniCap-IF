import argparse
import gc
import glob
import json
import logging
import multiprocessing as mp
import os
import re
import time
import warnings
from multiprocessing import Lock, Value

import av
import torch
from qwen_omni_utils import process_mm_info
from tqdm import tqdm
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration


DEFAULT_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)


def extract_final_answer(output_text: str) -> str:
    if not output_text:
        return ""

    match_full = re.search(r"<answer>(.*?)</answer>", output_text, re.DOTALL | re.IGNORECASE)
    if match_full:
        return match_full.group(1).strip()

    match_partial = re.search(r"<answer>(.*)", output_text, re.DOTALL | re.IGNORECASE)
    if match_partial:
        return match_partial.group(1).strip()

    # Fallback: remove known wrapper tags if model does not follow <answer> format strictly.
    cleaned = re.sub(r"</?(context|video|think|answer)\b[^>]*>", "", output_text, flags=re.IGNORECASE)
    return cleaned.strip()


def setup_logger(rank, log_dir):
    logger = logging.getLogger(f"humanomniv2_worker_{rank}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        log_file = os.path.join(log_dir, f"humanomniv2_worker_{rank}.log")
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s - [%(levelname)s] - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def sort_video_id(video_id):
    parts = video_id.split("_")
    video_num = int(parts[-1]) if parts[-1].isdigit() else 0
    part_order = {"clip": 0, "short": 1, "long": 2}
    return part_order.get(parts[0], 999), video_num


def check_if_video_has_audio(video_path):
    try:
        with av.open(video_path) as container:
            return any(stream.type == "audio" for stream in container.streams)
    except Exception:
        return False


def merge_response_item(target_dict, item):
    video_id = item["video_id"]
    prompt_id = item["prompt_id"]
    response_text = item.get("response", "")
    entry = {
        "field": item["field"],
        "prompt_id": prompt_id,
        "response": response_text,
    }

    if video_id not in target_dict:
        target_dict[video_id] = [entry]
        return "added"

    for idx, existing in enumerate(target_dict[video_id]):
        if existing.get("prompt_id") != prompt_id:
            continue

        old_response = existing.get("response", "")
        if old_response or not response_text:
            return "skipped"

        target_dict[video_id][idx] = entry
        return "replaced"

    target_dict[video_id].append(entry)
    return "added"


def build_done_set(response_dict):
    done = set()
    for video_id, entries in response_dict.items():
        for item in entries:
            if isinstance(item, dict) and "prompt_id" in item:
                done.add((video_id, item["prompt_id"]))
    return done


def validate_local_model_path(model_path):
    resolved = os.path.abspath(model_path)
    required_files = ["config.json"]

    if not os.path.isdir(resolved):
        return False, (
            f"模型路径不存在或不是目录: {model_path} (resolved: {resolved})"
        )

    missing = [name for name in required_files if not os.path.exists(os.path.join(resolved, name))]
    if missing:
        return False, (
            f"模型目录缺少必要文件: {missing} (path: {resolved})"
        )

    return True, resolved


def run_single_inference(
    model,
    processor,
    device,
    video_path,
    text_query,
    system_prompt,
    use_audio_in_video,
    fps,
    max_pixels,
    max_new_tokens,
    repetition_penalty,
):
    has_audio = check_if_video_has_audio(video_path) if use_audio_in_video else False

    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path, "max_pixels": max_pixels, "fps": fps},
                {"type": "text", "text": text_query},
            ],
        },
    ]

    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

    audios, images, videos = process_mm_info(
        conversation,
        use_audio_in_video=has_audio and use_audio_in_video,
    )

    inputs = processor(
        text=text,
        videos=videos,
        audio=audios,
        return_tensors="pt",
        padding=True,
    )
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            do_sample=False,
            temperature=0.0,
            top_p=0.9,
        )

    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[:, input_len:]
    decoded_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    response = extract_final_answer(decoded_text)
    del images
    return response


def worker_proc(
    rank,
    gpu_id,
    model_path,
    meta_prompt,
    video_meta_info,
    prompt_dict,
    global_done,
    video_ids_chunk,
    tmp_out_path,
    counter,
    lock,
    log_dir,
    max_retries,
    use_audio_in_video,
    fps,
    max_pixels,
    max_new_tokens,
    repetition_penalty,
):
    logger = setup_logger(rank, log_dir)
    logger.info(f"Worker启动 rank={rank} gpu={gpu_id} pid={os.getpid()}")

    ok, model_check = validate_local_model_path(model_path)
    if not ok:
        logger.error(model_check)
        return
    resolved_model_path = model_check

    local_done = set(global_done)

    if os.path.exists(tmp_out_path):
        try:
            with open(tmp_out_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        local_done.add((item["video_id"], item["prompt_id"]))
                    except Exception:
                        continue
            logger.info(f"本地临时文件恢复完成: {tmp_out_path}")
        except Exception as err:
            logger.warning(f"读取本地临时文件失败: {err}")

    device = f"cuda:{gpu_id}"
    try:
        torch.cuda.set_device(gpu_id)
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            resolved_model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            local_files_only=True,
        ).to(device)
        model.eval()
        processor = Qwen2_5OmniProcessor.from_pretrained(
            resolved_model_path,
            local_files_only=True,
        )
        logger.info(f"模型加载成功: {resolved_model_path} on {device}")
    except Exception as err:
        logger.error(f"模型加载失败: {err}", exc_info=True)
        return

    processed = 0
    with open(tmp_out_path, "a", encoding="utf-8") as out_f:
        for video_id in video_ids_chunk:
            video_info = video_meta_info.get(video_id)
            if video_info is None:
                logger.warning(f"视频元信息缺失，跳过: {video_id}")
                continue

            video_path = os.path.normpath(os.path.join(".", video_info["path"])).replace("\\", "/")
            if not os.path.exists(video_path):
                logger.error(f"视频文件未找到: {video_path}")
                continue

            for prompt_info in prompt_dict.get(video_id, []):
                prompt_id = prompt_info.get("prompt_id")
                if prompt_id is None:
                    continue

                key = (video_id, prompt_id)
                if key in local_done:
                    continue

                prompt_text = prompt_info.get("generated_prompt", "")
                field = prompt_info.get("field", "")

                final_response = ""
                for attempt in range(max_retries):
                    try:
                        final_response = run_single_inference(
                            model=model,
                            processor=processor,
                            device=device,
                            video_path=video_path,
                            text_query=prompt_text,
                            system_prompt=meta_prompt,
                            use_audio_in_video=use_audio_in_video,
                            fps=fps,
                            max_pixels=max_pixels,
                            max_new_tokens=max_new_tokens,
                            repetition_penalty=repetition_penalty,
                        )
                        break
                    except Exception as err:
                        if attempt < max_retries - 1:
                            logger.warning(
                                "重试 (%s/%s) video_id=%s prompt_id=%s error=%s",
                                attempt + 1,
                                max_retries,
                                video_id,
                                prompt_id,
                                err,
                            )
                        else:
                            logger.error(
                                "最终失败 video_id=%s prompt_id=%s error=%s",
                                video_id,
                                prompt_id,
                                err,
                            )

                item = {
                    "video_id": video_id,
                    "field": field,
                    "prompt_id": prompt_id,
                    "response": final_response,
                }
                out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                out_f.flush()

                local_done.add(key)
                processed += 1
                with lock:
                    counter.value += 1

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    logger.info(f"Worker结束 rank={rank} processed={processed}")


class TestModelHumanOmniV2:
    def __init__(
        self,
        model_path="./humanomniv2",
        input_dir="./annotation",
        output_dir="./response",
        output_name="humanomniv2_response.json",
        max_retries=3,
        use_audio_in_video=True,
        fps=1.0,
        max_pixels=360 * 420,
        max_new_tokens=1024,
        repetition_penalty=1.05,
    ):
        self.model_path = model_path
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.output_name = output_name
        self.max_retries = max_retries
        self.use_audio_in_video = use_audio_in_video
        self.fps = fps
        self.max_pixels = max_pixels
        self.max_new_tokens = max_new_tokens
        self.repetition_penalty = repetition_penalty

        self.video_meta_info_path = os.path.join(self.input_dir, "video_meta_info.json")
        self.prompt_input_path = os.path.join(self.input_dir, "prompts.json")
        self.response_output_path = os.path.join(self.output_dir, self.output_name)
        self.meta_prompt_file = "meta_prompts/test_vlm_meta_prompt.txt"

        os.makedirs(self.output_dir, exist_ok=True)
        self.log_dir = os.path.join(self.output_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)

    def _load_meta_prompt(self):
        try:
            with open(self.meta_prompt_file, "r", encoding="utf-8") as f:
                text = f.read().strip()
            if text:
                print(f"成功从 '{self.meta_prompt_file}' 加载元指令。")
                return text
        except FileNotFoundError:
            pass

        print(f"警告: 元指令文件 '{self.meta_prompt_file}' 未找到或为空，将使用默认系统提示词。")
        return DEFAULT_SYSTEM_PROMPT

    def _save_sorted_dict(self, data_dict, file_path):
        sorted_dict = dict(sorted(data_dict.items(), key=lambda x: sort_video_id(x[0])))
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(sorted_dict, f, ensure_ascii=False, indent=4)

    def _load_tmp_records(self, response_dict):
        tmp_pattern = os.path.join(self.output_dir, "tmp_humanomniv2_part*.jsonl")
        tmp_paths = sorted(glob.glob(tmp_pattern))

        added = 0
        replaced = 0
        skipped = 0
        for tmp_path in tmp_paths:
            try:
                with open(tmp_path, "r", encoding="utf-8") as f:
                    for raw in f:
                        raw = raw.strip()
                        if not raw:
                            continue
                        try:
                            item = json.loads(raw)
                            item["response"] = extract_final_answer(item.get("response", ""))
                            status = merge_response_item(response_dict, item)
                            if status == "added":
                                added += 1
                            elif status == "replaced":
                                replaced += 1
                            else:
                                skipped += 1
                        except Exception:
                            continue
            except FileNotFoundError:
                continue

        if tmp_paths:
            print(
                f"从 {len(tmp_paths)} 个临时分片恢复完成: 新增 {added}, 替换 {replaced}, 跳过重复 {skipped}"
            )
        return tmp_paths

    def read_data_file(self):
        if not os.path.exists(self.video_meta_info_path):
            raise FileNotFoundError(f"视频元信息文件未找到: {self.video_meta_info_path}")
        if not os.path.exists(self.prompt_input_path):
            raise FileNotFoundError(f"Prompt文件未找到: {self.prompt_input_path}")

        with open(self.video_meta_info_path, "r", encoding="utf-8") as f:
            video_meta_info = json.load(f)
        print(f"成功从 '{self.video_meta_info_path}' 加载视频元信息")

        with open(self.prompt_input_path, "r", encoding="utf-8") as f:
            prompt_dict = json.load(f)
        print(f"成功从 '{self.prompt_input_path}' 加载prompt数据")

        response_dict = {}
        if os.path.exists(self.response_output_path):
            try:
                with open(self.response_output_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                for video_id, items in loaded.items():
                    for item in items:
                        merge_response_item(
                            response_dict,
                            {
                                "video_id": video_id,
                                "field": item.get("field", ""),
                                "prompt_id": item.get("prompt_id"),
                                "response": extract_final_answer(item.get("response", "")),
                            },
                        )
                print("找到已有response文件，将从断点继续处理。")
            except (json.JSONDecodeError, FileNotFoundError, TypeError, KeyError):
                print("无法读取response文件，将尝试仅从临时分片恢复。")
        else:
            print("未找到已有response文件，将尝试从临时分片恢复。")

        self._load_tmp_records(response_dict)
        return video_meta_info, prompt_dict, response_dict

    def _compute_pending(self, video_meta_info, prompt_dict, global_done):
        video_ids = sorted(prompt_dict.keys(), key=sort_video_id)

        pending_video_ids = []
        pending_prompt_total = 0
        missing_meta = 0

        for video_id in video_ids:
            if video_id not in video_meta_info:
                missing_meta += 1
                continue

            missing_for_video = 0
            for prompt_info in prompt_dict.get(video_id, []):
                prompt_id = prompt_info.get("prompt_id")
                if prompt_id is None:
                    continue
                if (video_id, prompt_id) not in global_done:
                    missing_for_video += 1

            if missing_for_video > 0:
                pending_video_ids.append(video_id)
                pending_prompt_total += missing_for_video

        return pending_video_ids, pending_prompt_total, missing_meta

    def _cleanup_tmp_files(self):
        tmp_pattern = os.path.join(self.output_dir, "tmp_humanomniv2_part*.jsonl")
        for tmp_path in glob.glob(tmp_pattern):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def get_response(self, num_gpus):
        ok, model_check = validate_local_model_path(self.model_path)
        if not ok:
            raise FileNotFoundError(model_check)
        self.model_path = model_check

        meta_prompt = self._load_meta_prompt()
        video_meta_info, prompt_dict, response_dict = self.read_data_file()

        total_videos = len(prompt_dict)
        global_done = build_done_set(response_dict)
        pending_video_ids, pending_prompt_total, missing_meta = self._compute_pending(
            video_meta_info, prompt_dict, global_done
        )

        fully_completed = total_videos - len(pending_video_ids) - missing_meta
        print(f"找到 {total_videos} 个可处理的视频")
        print("处理状态统计:")
        print(f"- 完全完成: {fully_completed}")
        print(f"- 缺少元信息: {missing_meta}")
        print(f"- 需要处理视频: {len(pending_video_ids)}")
        print(f"- 需要处理prompt: {pending_prompt_total}")

        if pending_prompt_total == 0:
            self._save_sorted_dict(response_dict, self.response_output_path)
            self._cleanup_tmp_files()
            print("所有任务已完成，无需继续推理。")
            print(f"结果已保存到 '{self.response_output_path}'")
            return

        available_gpus = torch.cuda.device_count()
        if available_gpus <= 0:
            raise RuntimeError("未检测到可用GPU，HumanOmniV2 推理需要 CUDA 环境")

        if num_gpus <= 0:
            num_gpus = available_gpus
        num_gpus = min(num_gpus, available_gpus)

        print(f"检测到 {available_gpus} 个GPU，使用 {num_gpus} 个GPU并行处理")

        chunks = [[] for _ in range(num_gpus)]
        for idx, video_id in enumerate(pending_video_ids):
            chunks[idx % num_gpus].append(video_id)
        print(f"任务分配: {[len(chunk) for chunk in chunks]}")

        counter = Value("i", 0)
        lock = Lock()
        processes = []

        global_done_snapshot = list(global_done)
        worker_infos = []
        for rank, chunk in enumerate(chunks):
            if not chunk:
                continue
            worker_infos.append((rank, rank, chunk))

        for rank, gpu_id, chunk in worker_infos:
            tmp_out_path = os.path.join(self.output_dir, f"tmp_humanomniv2_part{rank}.jsonl")
            p = mp.Process(
                target=worker_proc,
                args=(
                    rank,
                    gpu_id,
                    self.model_path,
                    meta_prompt,
                    video_meta_info,
                    prompt_dict,
                    global_done_snapshot,
                    chunk,
                    tmp_out_path,
                    counter,
                    lock,
                    self.log_dir,
                    self.max_retries,
                    self.use_audio_in_video,
                    self.fps,
                    self.max_pixels,
                    self.max_new_tokens,
                    self.repetition_penalty,
                ),
                name=f"humanomniv2_worker_{rank}",
            )
            p.start()
            processes.append(p)
            print(f"启动 Worker {rank} (GPU {gpu_id}, PID: {p.pid})，分配视频数: {len(chunk)}")

        with tqdm(total=pending_prompt_total, desc="全局处理进度") as pbar:
            last = 0
            heartbeat = time.time()
            while True:
                with lock:
                    now = counter.value

                if now > last:
                    pbar.update(now - last)
                    last = now
                    heartbeat = time.time()
                elif time.time() - heartbeat >= 60:
                    alive_pids = [proc.pid for proc in processes if proc.is_alive()]
                    print(f"心跳: 已完成 {now}/{pending_prompt_total}，存活Worker PID={alive_pids}")
                    heartbeat = time.time()

                if now >= pending_prompt_total or all(not proc.is_alive() for proc in processes):
                    break
                time.sleep(1)

        for proc in processes:
            proc.join()

        merged = {}
        for video_id, items in response_dict.items():
            for item in items:
                merge_response_item(
                    merged,
                    {
                        "video_id": video_id,
                        "field": item.get("field", ""),
                        "prompt_id": item.get("prompt_id"),
                        "response": item.get("response", ""),
                    },
                )

        self._load_tmp_records(merged)
        self._save_sorted_dict(merged, self.response_output_path)
        self._cleanup_tmp_files()

        print("\n--- 所有视频处理完毕 ---")
        print(f"总计视频数: {total_videos}")
        print(f"本轮目标prompt数: {pending_prompt_total}")
        print(f"任务完成！结果已保存到 '{self.response_output_path}'")


def main():
    parser = argparse.ArgumentParser(description="HumanOmniV2 视频推理脚本（多卡并行，支持全局断点续跑）")
    parser.add_argument("--model_path", type=str, default="./models/humanomniv2", help="模型路径")
    parser.add_argument("-i", "--input_dir", type=str, default="./annotation", help="输入目录")
    parser.add_argument("-o", "--output_dir", type=str, default="./response", help="输出目录")
    parser.add_argument(
        "--output_name",
        type=str,
        default="humanomniv2_response.json",
        help="输出文件名",
    )
    parser.add_argument("--num_gpus", type=int, default=1, help="使用GPU数量，<=0表示使用全部可用GPU")
    parser.add_argument("--max_retries", type=int, default=3, help="单条prompt最大重试次数")
    parser.add_argument("--fps", type=float, default=1.0, help="视频采样帧率")
    parser.add_argument("--max_pixels", type=int, default=200704, help="视频最大像素参数")
    parser.add_argument("--max_new_tokens", type=int, default=1536, help="生成最大token")
    parser.add_argument("--repetition_penalty", type=float, default=1.05, help="重复惩罚")
    parser.add_argument("--disable_audio_in_video", action="store_true", help="禁用视频音频输入")
    args = parser.parse_args()

    test_model = TestModelHumanOmniV2(
        model_path=args.model_path,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        output_name=args.output_name,
        max_retries=args.max_retries,
        use_audio_in_video=not args.disable_audio_in_video,
        fps=args.fps,
        max_pixels=args.max_pixels,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
    )
    test_model.get_response(num_gpus=args.num_gpus)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    warnings.filterwarnings("ignore")
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    main()
