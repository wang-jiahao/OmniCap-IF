import argparse
import json
import os
import time
import re
import torch
import math
import numpy as np
import decord
from decord import VideoReader, cpu
from PIL import Image
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import librosa
import tempfile
import glob
import warnings
import gc
from multiprocessing import Lock, Value

# 引入多进程模块，并设置启动方式
import multiprocessing as mp
from tqdm import tqdm

from transformers import ARCHunyuanVideoProcessor, ARCHunyuanVideoForConditionalGeneration

# ----------------- 原始辅助函数保持不变 -----------------
def calculate_frame_indices(vlen: int, fps: float, duration: float) -> list:
    frames_per_second = fps

    if duration <= 150:
        interval = 1
        intervals = [
            (int(i * interval * frames_per_second), int((i + 1) * interval * frames_per_second))
            for i in range(math.ceil(duration))
        ]
        sample_fps = 1
    else:
        num_segments = 150
        segment_duration = duration / num_segments
        intervals = [
            (int(i * segment_duration * frames_per_second), int((i + 1) * segment_duration * frames_per_second))
            for i in range(num_segments)
        ]
        sample_fps = 1

    frame_indices = []
    for start, end in intervals:
        if end > vlen:
            end = vlen
        frame_indices.append((start + end) // 2)

    return frame_indices, sample_fps


def load_video_frames(video_path: str):
    # 降低这里的线程数，防止 8 个进程同时抢占导致 CPU 过载
    video_reader = VideoReader(video_path, ctx=cpu(0), num_threads=2) 
    vlen = len(video_reader)
    input_fps = video_reader.get_avg_fps()
    duration = vlen / input_fps

    frame_indices, sample_fps = calculate_frame_indices(vlen, input_fps, duration)

    return [Image.fromarray(video_reader[idx].asnumpy()) for idx in frame_indices], sample_fps


def cut_audio_with_librosa(audio_path, max_num_frame=150, segment_sec=2, max_total_sec=300, sr=16000):
    audio, _ = librosa.load(audio_path, sr=sr)
    total_samples = len(audio)
    total_sec = total_samples / sr

    if total_sec <= max_total_sec:
        return audio, sr

    segment_length = total_samples // max_num_frame
    segment_samples = int(segment_sec * sr)
    segments = []
    for i in range(max_num_frame):
        start = i * segment_length
        end = min(start + segment_samples, total_samples)
        segments.append(audio[start:end])
    new_audio = np.concatenate(segments)
    return new_audio, sr


def pad_audio(audio: np.ndarray, sr: int) -> np.ndarray:
    if len(audio.shape) == 2:
        audio = audio[:, 0]
    if len(audio) < sr:
        sil = np.zeros(sr - len(audio), dtype=float)
        audio = np.concatenate((audio, sil), axis=0)
    return audio


def load_audio(video_path, audio_path):
    if audio_path is None:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as temp_audio:
            audio_path = temp_audio.name
            video = VideoFileClip(video_path)
            try:
                # 关闭 logger 输出，防止 8 个进程的进度条把终端界面刷屏
                video.audio.write_audiofile(audio_path, logger=None)
                audio, sr = cut_audio_with_librosa(
                    audio_path, max_num_frame=150, segment_sec=2, max_total_sec=300, sr=16000
                )
            except:
                duration = min(math.ceil(video.duration), 300)
                silent_audio = AudioSegment.silent(duration=duration * 1000)
                silent_audio.export(audio_path, format="mp3")
                audio, sr = librosa.load(audio_path, sr=16000)
    else:
        audio, sr = cut_audio_with_librosa(audio_path, max_num_frame=150, segment_sec=2, max_total_sec=300, sr=16000)

    audio = pad_audio(audio, sr)
    duration = math.ceil(len(audio) / sr)

    return audio, sr, duration


def build_prompt(question: str, num_frames: int):
    video_prefix = "<image>" * num_frames
    return f"<|startoftext|>{video_prefix}\n{question}\nOutput the thinking process in <think> </think> and final answer in <answer> </answer> tags, i.e., <think> reasoning process here </think><answer> answer here </answer>.<sep>"


def prepare_inputs(question: str, video_path: str, audio_path: str = None):
    video_frames, sample_fps = load_video_frames(video_path)
    audio, sr, duration = load_audio(video_path, audio_path)

    video_duration = int(len(video_frames) / sample_fps)
    audio_duration = duration

    duration = min(video_duration, audio_duration)
    if duration <= 150:
        video_frames = video_frames[: int(duration * sample_fps)]

    prompt = build_prompt(question, len(video_frames))

    video_inputs = {
        "video": video_frames,
        "video_metadata": {
            "fps": 1,
        },
    }

    audio_inputs = {
        "audio": audio,
        "sampling_rate": sr,
        "duration": duration,
    }

    return prompt, video_inputs, audio_inputs


# ----------------- 修改：传入 device 参数，指定设备 -----------------
def inference(model, processor, question: str, video_path: str, device: str, audio_path: str = None):
    prompt, video_inputs, audio_inputs = prepare_inputs(question, video_path, audio_path)
    inputs = processor(
        text=prompt,
        **video_inputs,
        **audio_inputs,
        return_tensors="pt",
    )

    # 将 tensor 推送到特定的 GPU 上
    inputs = inputs.to(device, dtype=torch.bfloat16)
    outputs = model.generate(**inputs, max_new_tokens=1536, repetition_penalty=1.05, do_sample=False, temperature=0.0)
    output_text = processor.decode(outputs[0], skip_special_tokens=True)

    return output_text


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


def extract_number(filepath):
    basename = os.path.basename(filepath)
    numbers = re.findall(r'\d+', basename)
    return int(numbers[0]) if numbers else 0


def sort_video_id(video_id: str):
    if video_id.isdigit():
        return 0, int(video_id)

    parts = video_id.split("_")
    video_num = int(parts[-1]) if parts[-1].isdigit() else 0
    part_order = {"clip": 0, "short": 1, "long": 2}
    return part_order.get(parts[0], 999), video_num


def sanitize_model_name(model_name: str) -> str:
    return "ARC-Hunyuan-Video-7B"


def merge_response_item(target_dict, item):
    video_id = str(item.get("video_id", ""))
    prompt_id = str(item.get("prompt_id", ""))
    response_text = item.get("response", "")
    entry = {
        "field": item.get("field", ""),
        "prompt_id": prompt_id,
        "response": response_text,
    }

    if not video_id or not prompt_id:
        return "invalid"

    if video_id not in target_dict:
        target_dict[video_id] = [entry]
        return "added"

    for idx, existing in enumerate(target_dict[video_id]):
        if str(existing.get("prompt_id")) != prompt_id:
            continue

        old_response = existing.get("response", "")
        new_response = entry.get("response", "")

        # 非空优先，已有非空不被空值覆盖。
        if old_response and not new_response:
            return "duplicate"
        if old_response == new_response and existing.get("field", "") == entry.get("field", ""):
            return "duplicate"
        if old_response and new_response:
            return "duplicate"
        if not old_response and new_response:
            target_dict[video_id][idx] = entry
            return "replaced"
        return "duplicate"

    target_dict[video_id].append(entry)
    return "added"


def build_done_set(response_dict):
    done = set()
    for video_id, entries in response_dict.items():
        for item in entries:
            if isinstance(item, dict) and item.get("prompt_id") is not None:
                done.add((video_id, str(item.get("prompt_id"))))
    return done


# ----------------- 独立的工作进程函数（保留多卡方法，任务粒度改为prompt） -----------------
def worker_process(
    rank: int,
    gpu_id: int,
    task_list: list,
    model_path: str,
    tmp_out_path: str,
    error_log_path: str,
    counter,
    lock,
    max_retries: int,
    meta_prompt: str,
):
    """
    运行在单个 GPU 上的进程函数。
    只加载一次模型到特定 GPU，然后循环处理分配给它的视频。
    """
    if not task_list:
        print(f"[GPU {gpu_id}] 无分配任务，进程退出。")
        return

    device = f"cuda:{gpu_id}"
    print(f"[GPU {gpu_id}] 初始化中，分配到 {len(task_list)} 条任务。正在将模型加载到 {device} ...")

    try:
        torch.cuda.set_device(gpu_id)
        # 指定 device_map 将模型强制绑定到当前的 GPU
        model = ARCHunyuanVideoForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map={"": gpu_id},  # 关键：绑定专属 GPU
        ).eval()
        processor = ARCHunyuanVideoProcessor.from_pretrained(model_path)
    except Exception as e:
        print(f"[GPU {gpu_id}] 模型加载失败: {str(e)}")
        with open(error_log_path, "a", encoding="utf-8") as f:
            f.write(f"Model loading failed on GPU {gpu_id}: {str(e)}\n")
        return
    
    print(f"[GPU {gpu_id}] 模型加载完成，开始推理！")

    total_tasks = len(task_list)
    with open(tmp_out_path, "a", encoding="utf-8") as out_f:
        for idx, task in enumerate(task_list, 1):
            video_id = task["video_id"]
            field = task["field"]
            prompt_id = task["prompt_id"]
            video_path = task["video_path"]
            generated_prompt = task["generated_prompt"]

            filename = os.path.basename(video_path)
            print(f"[GPU {gpu_id}] [{idx}/{total_tasks}] 正在处理 video={video_id} prompt={prompt_id} ({filename}) ...")
            start_time = time.time()

            final_answer = ""
            last_err = None

            if meta_prompt:
                question = f"{meta_prompt}\n\n{generated_prompt}"
            else:
                question = generated_prompt

            for attempt in range(max_retries):
                try:
                    raw_output = inference(model, processor, question, video_path, device, audio_path=None)
                    final_answer = extract_final_answer(raw_output)

                    # 监控是否触发了保底机制
                    if final_answer == raw_output.strip() and "<answer>" not in raw_output.lower():
                        print(f"[GPU {gpu_id}] -> [警告] {video_id}/{prompt_id} 未生成有效 <answer>，回退原文。")
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    if attempt < max_retries - 1:
                        print(
                            f"[GPU {gpu_id}] -> 重试 ({attempt + 1}/{max_retries}) "
                            f"video={video_id} prompt={prompt_id}: {str(e)}"
                        )
                    else:
                        print(f"[GPU {gpu_id}] -> [错误] video={video_id} prompt={prompt_id} 最终失败: {str(e)}")

            out_item = {
                "video_id": video_id,
                "field": field,
                "prompt_id": prompt_id,
                "response": final_answer,
            }
            out_f.write(json.dumps(out_item, ensure_ascii=False) + "\n")
            out_f.flush()

            if last_err is not None:
                # 每个进程使用独立的错误日志文件，防止多进程同时写入同一个文件引发错乱
                with open(error_log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"Error in video={video_id}, prompt={prompt_id}, "
                        f"file={filename}: {str(last_err)}\n"
                    )

            cost_time = time.time() - start_time
            print(f"[GPU {gpu_id}] -> 完成 video={video_id} prompt={prompt_id} (耗时: {cost_time:.2f}s)")

            with lock:
                counter.value += 1

            torch.cuda.empty_cache()

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()

    print(f"[GPU {gpu_id}] 当前队列所有任务已完成！")


class TestModelHunyuan:
    def __init__(
        self,
        model: str,
        input_dir: str = "./annotation",
        output_dir: str = "./response",
        num_gpus: int = 8,
        max_retries: int = 3,
        meta_prompt_file: str = "meta_prompts/test_vlm_meta_prompt.txt",
    ):
        self.model_path = model
        self.model_name = sanitize_model_name(model)
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.num_gpus = num_gpus
        self.max_retries = max_retries
        self.meta_prompt_file = meta_prompt_file

        self.video_meta_info_path = os.path.join(self.input_dir, "video_meta_info.json")
        self.prompt_input_path = os.path.join(self.input_dir, "prompts.json")
        self.response_output_path = os.path.join(self.output_dir, f"{self.model_name}_response.json")

        os.makedirs(self.output_dir, exist_ok=True)

    def _save_sorted_dict(self, data_dict, file_path):
        sorted_dict = dict(sorted(data_dict.items(), key=lambda x: sort_video_id(x[0])))
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(sorted_dict, f, ensure_ascii=False, indent=4)

    def _load_meta_prompt(self):
        try:
            with open(self.meta_prompt_file, "r", encoding="utf-8") as f:
                text = f.read().strip()
            if text:
                print(f"成功从 '{self.meta_prompt_file}' 加载元指令。")
                return text
        except FileNotFoundError:
            pass

        print(f"警告: 元指令文件 '{self.meta_prompt_file}' 未找到或为空，将仅使用 generated_prompt。")
        return ""

    def _load_tmp_records(self, response_dict):
        tmp_pattern = os.path.join(self.output_dir, "tmp_hunyuan_part*.jsonl")
        tmp_paths = sorted(glob.glob(tmp_pattern))

        stats = {
            "added": 0,
            "replaced": 0,
            "duplicate": 0,
            "invalid": 0,
        }

        for tmp_path in tmp_paths:
            try:
                with open(tmp_path, "r", encoding="utf-8") as f:
                    for raw in f:
                        line = raw.strip()
                        if not line:
                            continue
                        try:
                            item = json.loads(line)
                            status = merge_response_item(response_dict, item)
                            if status in stats:
                                stats[status] += 1
                        except Exception:
                            stats["invalid"] += 1
            except FileNotFoundError:
                continue

        if tmp_paths:
            print(
                f"从 {len(tmp_paths)} 个临时分片恢复完成: "
                f"新增 {stats['added']}, 替换 {stats['replaced']}, "
                f"去重 {stats['duplicate']}, 无效 {stats['invalid']}"
            )

        return stats

    def _load_existing_response(self):
        response_dict = {}
        stats = {
            "added": 0,
            "replaced": 0,
            "duplicate": 0,
            "invalid": 0,
        }

        if not os.path.exists(self.response_output_path):
            print("未找到已有response文件，将尝试从临时分片恢复。")
            return response_dict, stats

        try:
            with open(self.response_output_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)

            for video_id, items in loaded.items():
                for item in items:
                    status = merge_response_item(
                        response_dict,
                        {
                            "video_id": video_id,
                            "field": item.get("field", ""),
                            "prompt_id": item.get("prompt_id"),
                            "response": item.get("response", ""),
                        },
                    )
                    if status in stats:
                        stats[status] += 1
            print("找到已有response文件，将从断点继续处理。")
        except (json.JSONDecodeError, FileNotFoundError, TypeError, KeyError) as e:
            print(f"无法读取response文件，将尝试仅从临时分片恢复。原因: {e}")

        return response_dict, stats

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

        response_dict, response_stats = self._load_existing_response()
        tmp_stats = self._load_tmp_records(response_dict)

        return video_meta_info, prompt_dict, response_dict, response_stats, tmp_stats

    def _build_tasks(self, video_meta_info, prompt_dict, global_done):
        tasks = []
        missing_meta = 0
        missing_video_file = 0
        total_prompt_pairs = 0

        for video_id in sorted(prompt_dict.keys(), key=sort_video_id):
            if video_id not in video_meta_info:
                missing_meta += 1
                continue

            video_path = os.path.normpath(os.path.join(".", video_meta_info[video_id]["path"])).replace("\\", "/")
            if not os.path.exists(video_path):
                missing_video_file += 1
                continue

            for prompt_info in prompt_dict.get(video_id, []):
                prompt_id = prompt_info.get("prompt_id")
                if prompt_id is None:
                    continue

                prompt_id = str(prompt_id)
                total_prompt_pairs += 1

                if (video_id, prompt_id) in global_done:
                    continue

                tasks.append(
                    {
                        "video_id": video_id,
                        "video_path": video_path,
                        "field": prompt_info.get("field", ""),
                        "prompt_id": prompt_id,
                        "generated_prompt": prompt_info.get("generated_prompt", ""),
                    }
                )

        return tasks, total_prompt_pairs, missing_meta, missing_video_file

    def _cleanup_tmp_files(self):
        tmp_pattern = os.path.join(self.output_dir, "tmp_hunyuan_part*.jsonl")
        for tmp_path in glob.glob(tmp_pattern):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def get_response(self):
        meta_prompt = self._load_meta_prompt()
        (
            video_meta_info,
            prompt_dict,
            response_dict,
            response_stats,
            tmp_stats,
        ) = self.read_data_file()

        total_videos = len(prompt_dict)
        global_done = build_done_set(response_dict)
        tasks, total_prompt_pairs, missing_meta, missing_video_file = self._build_tasks(
            video_meta_info,
            prompt_dict,
            global_done,
        )

        pending_prompt_total = len(tasks)
        print(f"找到 {total_videos} 个可处理的视频")
        print("处理状态统计:")
        print(f"- 已恢复完成的 (video_id,prompt_id): {len(global_done)}")
        print(f"- Prompt总数(有效路径): {total_prompt_pairs}")
        print(f"- 需要处理prompt: {pending_prompt_total}")
        print(f"- 缺少元信息视频: {missing_meta}")
        print(f"- 视频文件缺失: {missing_video_file}")
        print(
            f"- 启动恢复统计: response新增 {response_stats['added']}, "
            f"response去重 {response_stats['duplicate']}, tmp去重 {tmp_stats['duplicate']}"
        )

        if pending_prompt_total == 0:
            self._save_sorted_dict(response_dict, self.response_output_path)
            self._cleanup_tmp_files()
            print("所有任务已完成，无需继续推理。")
            print(f"结果已保存到 '{self.response_output_path}'")
            return

        available_gpus = torch.cuda.device_count()
        if available_gpus <= 0:
            raise RuntimeError("未检测到可用GPU，Hunyuan 推理需要 CUDA 环境")

        if self.num_gpus <= 0:
            self.num_gpus = available_gpus
        self.num_gpus = min(self.num_gpus, available_gpus)

        print(f"检测到 {available_gpus} 个GPU，使用 {self.num_gpus} 个GPU并行处理")

        chunks = [[] for _ in range(self.num_gpus)]
        for idx, task in enumerate(tasks):
            chunks[idx % self.num_gpus].append(task)
        print(f"任务分配(按prompt均分): {[len(chunk) for chunk in chunks]}")

        counter = Value("i", 0)
        lock = Lock()
        processes = []

        for rank, chunk in enumerate(chunks):
            if not chunk:
                continue

            tmp_out_path = os.path.join(self.output_dir, f"tmp_hunyuan_part{rank}.jsonl")
            error_log_path = os.path.join(self.output_dir, f"error_log_gpu{rank}.txt")

            p = mp.Process(
                target=worker_process,
                args=(
                    rank,
                    rank,
                    chunk,
                    self.model_path,
                    tmp_out_path,
                    error_log_path,
                    counter,
                    lock,
                    self.max_retries,
                    meta_prompt,
                ),
                name=f"hunyuan_worker_{rank}",
            )
            p.start()
            processes.append(p)
            print(f"启动 Worker {rank} (GPU {rank}, PID: {p.pid})，分配任务数: {len(chunk)}")

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

        for p in processes:
            p.join()

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

        merge_stats = self._load_tmp_records(merged)
        final_done = build_done_set(merged)

        self._save_sorted_dict(merged, self.response_output_path)
        self._cleanup_tmp_files()

        print("\n--- 所有任务处理完毕 ---")
        print(f"总计视频数: {total_videos}")
        print(f"本轮目标prompt数: {pending_prompt_total}")
        print(f"合并后完成的 (video_id,prompt_id): {len(final_done)}")
        print(
            f"二次幂等合并统计: 新增 {merge_stats['added']}, 替换 {merge_stats['replaced']}, "
            f"去重 {merge_stats['duplicate']}, 无效 {merge_stats['invalid']}"
        )
        print(f"任务完成！结果已保存到 '{self.response_output_path}'")


# ----------------- 主控制流程 -----------------
def main():
    parser = argparse.ArgumentParser(description="Hunyuan 视频推理脚本（多卡并行，支持全局断点续跑）")
    parser.add_argument("--model", type=str, default="./models/ARC-Hunyuan-Video-7B", help="模型路径或模型名")
    parser.add_argument("-i", "--input_dir", type=str, default="./annotation", help="输入目录")
    parser.add_argument("-o", "--output_dir", type=str, default="./response_hunyuan", help="输出目录")
    parser.add_argument("--num_gpus", type=int, default=8, help="使用GPU数量，<=0表示使用全部可用GPU")
    parser.add_argument("--max_retries", type=int, default=3, help="单条prompt最大重试次数")
    parser.add_argument(
        "--meta_prompt_file",
        type=str,
        default="meta_prompts/test_vlm_meta_prompt.txt",
        help="元提示词文件路径",
    )
    args = parser.parse_args()

    test_model = TestModelHunyuan(
        model=args.model,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_gpus=args.num_gpus,
        max_retries=args.max_retries,
        meta_prompt_file=args.meta_prompt_file,
    )
    test_model.get_response()


if __name__ == "__main__":
    # 极为关键：在 PyTorch 的多进程中，必须使用 'spawn' 启动方法
    # 否则默认的 'fork' 方法会继承父进程的 CUDA 环境，导致全部卡死或报错
    mp.set_start_method("spawn", force=True)

    warnings.filterwarnings("ignore")
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    main()