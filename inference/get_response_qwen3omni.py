import argparse
import os
import json
import gc
import time
import logging
import glob
import torch
import warnings
import multiprocessing as mp
from multiprocessing import Value, Lock
from tqdm import tqdm
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info


# ------------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------------

def setup_logger(rank, log_dir):
    """为每个 GPU Worker 建立独立日志"""
    logger = logging.getLogger(f"worker_{rank}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        log_file = os.path.join(log_dir, f"worker_{rank}.log")
        file_handler = logging.FileHandler(log_file, mode='w')
        formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def merge_response_item(target_dict, item):
    """将单条响应写入内存字典，按 prompt_id 去重。"""
    vid = item["video_id"]
    entry = {
        "field": item["field"],
        "prompt_id": item["prompt_id"],
        "response": item["response"],
    }
    if vid not in target_dict:
        target_dict[vid] = []
    existing_ids = {r["prompt_id"] for r in target_dict[vid]}
    if entry["prompt_id"] not in existing_ids:
        target_dict[vid].append(entry)


def get_done_prompt_ids(response_dict, video_id):
    return {
        item["prompt_id"]
        for item in response_dict.get(video_id, [])
        if isinstance(item, dict) and "prompt_id" in item
    }


# ------------------------------------------------------------------
# Worker 进程函数（多卡并行核心）
# ------------------------------------------------------------------

def worker_proc(rank, gpu_id, model_name, input_dir, output_dir,
                meta_prompt, video_meta_info, prompt_dict,
                existing_response_dict, video_ids_chunk,
                tmp_out_path, counter, lock, log_dir,
                max_new_tokens, use_audio_in_video, fps):
    """每个 GPU Worker 独立加载模型，处理分配到的 video_id 列表"""
    logger = setup_logger(rank, log_dir)
    logger.info(f"Worker {rank} 启动 (GPU {gpu_id}, PID: {os.getpid()})")

    # 断点续跑：加载已写入临时文件的结果
    done_in_tmp = set()  # (video_id, prompt_id)
    if os.path.exists(tmp_out_path):
        with open(tmp_out_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    done_in_tmp.add((item["video_id"], item["prompt_id"]))
                except Exception:
                    pass
        logger.info(f"从临时文件恢复 {len(done_in_tmp)} 条已完成记录")

    # 加载模型
    model_path = f"./models/QwenOmni/{model_name}"
    device = f"cuda:{gpu_id}"
    try:
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="flash_attention_2",
        )
        model.disable_talker()
        processor = Qwen3OmniMoeProcessor.from_pretrained(model_path, use_fast=False)
        logger.info("模型加载成功")
    except Exception as e:
        logger.error(f"模型加载失败: {e}", exc_info=True)
        return

    gen_config = {
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": 1.05,
        "do_sample": True,
        "return_audio": False,
        "thinker_return_dict_in_generate": True,
    }

    def run_inference(video_path, prompt_text):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": meta_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "video", "video": video_path, "fps": fps},
                ],
            },
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        audio_inputs, image_inputs, video_inputs = process_mm_info(messages, use_audio_in_video=use_audio_in_video)
        inputs = processor(
            text=text, audio=audio_inputs, images=image_inputs, videos=video_inputs,
            return_tensors="pt", padding=True, use_audio_in_video=use_audio_in_video,
        )
        inputs = inputs.to(device).to(model.dtype)
        with torch.no_grad():
            generate_outputs = model.generate(**inputs, use_audio_in_video=use_audio_in_video, **gen_config)
            if isinstance(generate_outputs, tuple):
                text_outputs = generate_outputs[0]
            else:
                text_outputs = generate_outputs
            output_ids = text_outputs.sequences if hasattr(text_outputs, "sequences") else text_outputs
        input_len = inputs["input_ids"].shape[1]
        response = processor.batch_decode(
            output_ids[:, input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0]
        del inputs, output_ids
        gc.collect()
        torch.cuda.empty_cache()
        return response

    for video_id in video_ids_chunk:
        if video_id not in video_meta_info:
            logger.warning(f"视频 {video_id} 在元信息中未找到，跳过")
            continue

        video_info = video_meta_info[video_id]
        video_prompts = prompt_dict[video_id]
        existing_prompt_ids = get_done_prompt_ids(existing_response_dict, video_id)

        video_path = os.path.normpath(os.path.join(".", video_info["path"])).replace("\\", "/")
        if not os.path.exists(video_path):
            logger.error(f"视频文件 '{video_path}' 未找到，跳过")
            continue

        for prompt_info in video_prompts:
            # 跳过已有结果（来自既有 response_dict 或本 worker 临时文件）
            if prompt_info["prompt_id"] in existing_prompt_ids:
                continue
            if (video_id, prompt_info["prompt_id"]) in done_in_tmp:
                continue

            try:
                t0 = time.time()
                logger.info(f"开始推理: {video_id} prompt_id={prompt_info['prompt_id']}")
                response = run_inference(video_path, prompt_info["generated_prompt"])
                item = {
                    "video_id": video_id,
                    "field": prompt_info["field"],
                    "prompt_id": prompt_info["prompt_id"],
                    "response": response,
                }
                with open(tmp_out_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    f.flush()
                done_in_tmp.add((video_id, prompt_info["prompt_id"]))
                with lock:
                    counter.value += 1
                cost = time.time() - t0
                logger.info(f"✓ {video_id} prompt_id={prompt_info['prompt_id']} 耗时={cost:.1f}s")
            except Exception as e:
                logger.error(f"✗ {video_id} prompt_id={prompt_info['prompt_id']}: {e}", exc_info=True)

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    logger.info(f"Worker {rank} 完成所有任务")


# ------------------------------------------------------------------
# 主类（数据管理 + 多卡调度）
# ------------------------------------------------------------------

class TestModelOmni:
    def __init__(
        self,
        model: str,
        input_dir: str = "./annotation",
        output_dir: str = "./response",
        max_new_tokens: int = 4096,
        use_audio_in_video: bool = True,
        fps: float = 2.0,
    ):
        self.input_dir = input_dir
        self.video_meta_info_path = "./annotation/video_meta_info.json"
        self.prompt_input_path = os.path.join(input_dir, "prompts.json")

        self.model_name = model
        self.output_dir = output_dir
        self.max_new_tokens = max_new_tokens
        self.use_audio_in_video = use_audio_in_video
        self.fps = fps
        self.response_output_path = os.path.join(output_dir, f"{self.model_name}_response.json")
        self.meta_prompt_file = "meta_prompts/test_vlm_meta_prompt.txt"

        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 数据读写
    # ------------------------------------------------------------------

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
                for vid, items in loaded.items():
                    for item in items:
                        merge_response_item(
                            response_dict,
                            {
                                "video_id": vid,
                                "field": item["field"],
                                "prompt_id": item["prompt_id"],
                                "response": item["response"],
                            },
                        )
                print("找到已有response文件，将从断点继续处理。")
            except (json.JSONDecodeError, FileNotFoundError, KeyError, TypeError):
                print("无法读取response文件，将尝试仅从临时文件恢复。")
        else:
            print("未找到已有response文件，将尝试从临时文件恢复。")

        tmp_pattern = os.path.join(self.output_dir, "tmp_response_part*.jsonl")
        tmp_paths = sorted(glob.glob(tmp_pattern))
        recovered_from_tmp = 0
        for tmp_path in tmp_paths:
            try:
                with open(tmp_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            item = json.loads(line)
                            before = len(response_dict.get(item["video_id"], []))
                            merge_response_item(response_dict, item)
                            after = len(response_dict.get(item["video_id"], []))
                            recovered_from_tmp += max(0, after - before)
                        except Exception:
                            continue
            except FileNotFoundError:
                continue
        if tmp_paths:
            print(f"从 {len(tmp_paths)} 个临时文件恢复新增 {recovered_from_tmp} 条记录。")

        return video_meta_info, prompt_dict, response_dict

    def _save_sorted_dict(self, data_dict, file_path):
        def sort_key(video_id):
            parts = video_id.split("_")
            video_num = int(parts[-1]) if parts[-1].isdigit() else 0
            part_order = {"clip": 0, "short": 1, "long": 2}
            return (part_order.get(parts[0], 999), video_num)

        sorted_dict = dict(sorted(data_dict.items(), key=lambda x: sort_key(x[0])))
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(sorted_dict, f, ensure_ascii=False, indent=4)

    # ------------------------------------------------------------------
    # 主流程（多卡调度）
    # ------------------------------------------------------------------

    def get_response(self, num_gpus: int = 1):
        try:
            with open(self.meta_prompt_file, "r", encoding="utf-8") as f:
                meta_prompt = f.read()
            print(f"成功从 '{self.meta_prompt_file}' 加载元指令。")
        except FileNotFoundError:
            print(f"错误: 元指令文件 '{self.meta_prompt_file}' 未找到。")
            return

        video_meta_info, prompt_dict, response_dict = self.read_data_file()

        video_ids = set(prompt_dict.keys())
        total_videos = len(video_ids)
        print(f"找到 {total_videos} 个可处理的视频")

        def sort_key(video_id):
            parts = video_id.split("_")
            video_num = int(parts[-1]) if parts[-1].isdigit() else 0
            part_order = {"clip": 0, "short": 1, "long": 2}
            return (part_order.get(parts[0], 999), video_num)

        # 找出尚未完全处理的 video_id（按 prompt_id 判断，避免长度判断误差）
        pending_ids = sorted(
            [vid for vid in video_ids
             if len(get_done_prompt_ids(response_dict, vid)) < len(prompt_dict[vid])],
            key=sort_key,
        )

        fully_completed = total_videos - len(pending_ids)
        print(f"处理状态统计:")
        print(f"- 完全完成: {fully_completed}")
        print(f"- 需要处理: {len(pending_ids)}")

        if not pending_ids:
            print("所有视频已处理完毕。")
            return

        # 限制 GPU 数不超过实际可用数
        num_gpus = min(num_gpus, torch.cuda.device_count())
        print(f"使用 {num_gpus} 个GPU进行并行处理")

        log_dir = os.path.join(self.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)

        # 均分 video_id 到各 GPU（轮询分配，尽量平衡每卡的 prompt 总量）
        chunks = [[] for _ in range(num_gpus)]
        for i, vid in enumerate(pending_ids):
            chunks[i % num_gpus].append(vid)
        print(f"任务分配: {[len(c) for c in chunks]}")

        tmp_files = [
            os.path.join(self.output_dir, f"tmp_response_part{rank}.jsonl")
            for rank in range(num_gpus)
        ]

        counter = Value('i', 0)
        lock = Lock()

        # 启动各 GPU Worker
        processes = []
        for rank in range(num_gpus):
            p = mp.Process(
                target=worker_proc,
                args=(
                    rank, rank,
                    self.model_name, self.input_dir, self.output_dir,
                    meta_prompt, video_meta_info, prompt_dict,
                    response_dict, chunks[rank],
                    tmp_files[rank],
                    counter, lock, log_dir,
                    self.max_new_tokens, self.use_audio_in_video, self.fps,
                ),
                name=f"worker_{rank}",
            )
            p.start()
            processes.append(p)
            print(f"启动 Worker {rank} (GPU {rank}, PID: {p.pid})，分配 {len(chunks[rank])} 个视频")

        # 全局进度条（以 prompt 条数为单位）
        total_prompts = sum(
            max(0, len(prompt_dict[vid]) - len(get_done_prompt_ids(response_dict, vid)))
            for vid in pending_ids
        )
        with tqdm(total=total_prompts, desc="全局处理进度") as pbar:
            last_count = 0
            last_heartbeat = time.time()
            while True:
                with lock:
                    current_done = counter.value
                if current_done > last_count:
                    pbar.update(current_done - last_count)
                    last_count = current_done
                    last_heartbeat = time.time()
                elif time.time() - last_heartbeat >= 60:
                    alive = [p.pid for p in processes if p.is_alive()]
                    print(f"心跳: 已完成 {current_done}/{total_prompts}，存活Worker PID={alive}")
                    last_heartbeat = time.time()
                if current_done >= total_prompts or all(not p.is_alive() for p in processes):
                    break
                time.sleep(1)

        for p in processes:
            p.join()

        # 合并各 Worker 临时文件 + 已有结果 → 最终 JSON
        print("正在合并各 Worker 结果...")
        merged = dict(response_dict)
        for tmp_path in tmp_files:
            if not os.path.exists(tmp_path):
                continue
            with open(tmp_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        merge_response_item(merged, item)
                    except Exception:
                        continue
            os.remove(tmp_path)

        self._save_sorted_dict(merged, self.response_output_path)
        print(f"\n--- 所有视频处理完毕 ---")
        print(f"总计视频数: {total_videos}")
        print(f"新处理: {len(pending_ids)}")
        print(f"跳过(已存在): {fully_completed}")
        print(f"任务完成！结果已保存到 '{self.response_output_path}'")


def main():
    parser = argparse.ArgumentParser(description="Qwen3-Omni 视频+音频推理脚本（多卡并行版）")
    parser.add_argument("--model", type=str, required=True, help="模型名称")
    parser.add_argument("-i", "--input_dir", type=str, default="./annotation", help="输入目录")
    parser.add_argument("-o", "--output_dir", type=str, default="./response", help="输出目录")
    parser.add_argument("--num_gpus", type=int, default=8, help="使用的GPU数量")
    parser.add_argument("--max_new_tokens", type=int, default=8192, help="单条回答最大新token数")
    parser.add_argument("--fps", type=float, default=1.0, help="视频采样帧率")
    parser.add_argument("--disable_audio_in_video", action="store_true", help="禁用从视频中读取音频")
    args = parser.parse_args()

    test_model = TestModelOmni(
        model=args.model,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        use_audio_in_video=not args.disable_audio_in_video,
        fps=args.fps,
    )
    test_model.get_response(num_gpus=args.num_gpus)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    warnings.filterwarnings('ignore')
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)

    main()

    '''
    单卡运行:
    python get_response_qwen3omni.py --model Qwen3-Omni-30B-A3B-Instruct

    多卡并行（推荐）:
    python get_response_qwen3omni.py --model Qwen3-Omni-30B-A3B-Instruct --num_gpus 4
    python get_response_qwen3omni.py --model Qwen3-Omni-30B-A3B-Instruct -i ./annotation/test --num_gpus 8

    依赖：
        pip install qwen-omni-utils
    '''
