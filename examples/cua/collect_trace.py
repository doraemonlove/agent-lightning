from __future__ import annotations

import time
import requests
import json
import logging
from typing import Any
import pandas as pd
import os
import csv
import sys
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from reward_server import score_trace
from dotenv import load_dotenv, find_dotenv
from constants import CUA_PROMPT

load_dotenv(find_dotenv())
SANDBOX_KEY_AUTH = os.getenv("sandbox_key_auth")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(threadName)s - %(levelname)s - %(message)s",  # 增加线程名打印
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# CSV 写入锁，防止多线程写入冲突
CSV_LOCK = threading.Lock()


class SandboxBusyError(RuntimeError):
    pass


class SandboxManager:
    def __init__(self, lease_ttl_s: int | None = None):
        self._lock = threading.RLock()
        self._cv = threading.Condition(self._lock)
        self._free: set[str] = set()
        self._in_use: dict[str, dict] = {}
        self._ttl = lease_ttl_s
        self._bootstrap_free_pool_from_remote()

    def _bootstrap_free_pool_from_remote(self):
        try:
            # 你的沙箱列表
            running_uris = [
                "i-yeehw27z7k5i3z79r8kr",
                "i-yeae43bhfkxjd1taob37",
                "i-yeae1qx728wh2ypok55k",
                "i-yea1qch6o0xjd1tfette",
            ]
            running_uris = list(set(running_uris))  # 简单的去重
            logger.info(f"✅ 初始化沙箱池，共 {len(running_uris)} 个沙箱")

            with self._lock:
                for uri in running_uris:
                    if uri and uri not in self._in_use:
                        self._free.add(uri)
                if running_uris:
                    self._cv.notify_all()
        except Exception as e:
            logger.warning(f"⚠️ 初始化空闲池失败: {e}")

    def allocate(self, task_id: str) -> str:
        """分配沙箱，如果无空闲则阻塞等待"""
        while True:
            with self._lock:
                if self._free:
                    uri = self._free.pop()
                    self._in_use[uri] = {"task_id": task_id, "ts": time.time()}
                    logger.info(f"[Alloc] Task {task_id} -> Sandbox {uri}")
                    return uri
                logger.debug(f"Task {task_id} waiting for sandbox...")
                self._cv.wait()

    def release(self, uri: str):
        """释放沙箱"""
        with self._lock:
            if uri not in self._in_use:
                return
            self._in_use.pop(uri, None)
            self._free.add(uri)
            logger.info(f"[Release] Sandbox {uri} released. Free pool: {len(self._free)}")
            self._cv.notify()

    def get_pool_size(self):
        with self._lock:
            return len(self._free) + len(self._in_use)


class CuaEvaluator:

    def __init__(self, results_file: str = "./eval_results.csv"):
        self.results_file = results_file
        self.score_endpoint = "http://localhost:8003/overall_score"

    def run_planner_task(
        self,
        sandbox_id: str,
        system_prompt: str,
        user_prompt: str,
        model_name: str,
        model_endpoint: str,
        model_api_key: str,
        model_provider: str = "openai",
        agent_planner_url: str = "http://0.0.0.0:8331/planner",
        key_auth: str = "",
        rollout_id: str = "",
    ) -> list[dict[str, Any]]:

        url = f"{agent_planner_url}/run/task"
        headers = {"Content-Type": "application/json", "Authorization": key_auth}
        data = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "sandbox_id": sandbox_id,
            "model_name": model_name,
            "model_endpoint": model_endpoint,
            "model_provider": model_provider,
            "model_api_key": model_api_key,
            "max_actions": 30,
            "max_images": 3,
            "thinking_type": "disabled",
            "is_training": True,
            "turn_on_review": False,
            "rollout_id": rollout_id,
        }
        result = []
        try:
            logger.info(f"开始调用 Planner: url={url}, sandbox={sandbox_id}")
            with requests.post(url, headers=headers, data=json.dumps(data), stream=True, timeout=600) as response:
                response.raise_for_status()
                for line in response.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    if line.startswith("data: "):
                        try:
                            msg = json.loads(line[len("data: ") :])
                            result.append(msg)
                        except json.JSONDecodeError:
                            pass
        except Exception as e:
            logger.exception(f"Planner 调用失败: sandbox={sandbox_id}")
            raise RuntimeError(f"Planner任务执行失败：{str(e)}") from e

        return result

    async def execute_eval_rollout(
        self,
        sample: dict,
        system_prompt: str,
        sandbox_uri: str,
        unique_task_id: str,  # 用来区分同一各样本的不同变体
        model_name: str,
        trace_save_dir: str,
        model_endpoint: str = "",
        model_api_key: str = "",
        model_provider: str = "bedrock",
        agent_planner_url: str = "http://0.0.0.0:8331/planner",
        key_auth: str = "",
    ) -> tuple[float, float] | None:

        start_time = time.time()
        instruction = sample["instruction"]

        try:
            logger.info(f"[{unique_task_id}] 开始 Rollout")
            result = self.run_planner_task(
                sandbox_id=sandbox_uri,
                system_prompt=system_prompt,
                user_prompt=instruction,
                model_name=model_name,
                model_endpoint=model_endpoint,
                model_api_key=model_api_key,
                model_provider=model_provider,
                agent_planner_url=agent_planner_url,
                key_auth=key_auth,
                rollout_id=unique_task_id,
            )

            # 保存 Trace
            os.makedirs(trace_save_dir, exist_ok=True)
            # 文件名加入 unique_task_id 防止覆盖
            out_path = os.path.join(trace_save_dir, f"{unique_task_id}.json")

            full_trace = [{"instruction": instruction, "sandbox_id": sandbox_uri}] + result
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(full_trace, f, ensure_ascii=False, indent=4)

            # 评分
            end_time_rollout = time.time()
            score_result = await score_trace(url=self.score_endpoint, trace=result, user_instruction=instruction)
            reward = score_result["score"]
            reason = score_result["reason"]

            # 写入 CSV (加锁)
            elapsed = end_time_rollout - start_time
            row = {
                "reward": reward,
                "reason": json.dumps(reason, ensure_ascii=False),
                "time_sec": f"{elapsed:.4f}",
                "task_id": unique_task_id,  # 记录具体的任务ID
                "original_sample_id": sample.get("task_id", "unknown"),
            }
            fieldnames = ["task_id", "original_sample_id", "reward", "reason", "time_sec"]

            with CSV_LOCK:
                write_header = not os.path.exists(self.results_file) or os.path.getsize(self.results_file) == 0
                with open(self.results_file, "a", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    if write_header:
                        writer.writeheader()
                    writer.writerow(row)

            return reward, elapsed

        except Exception as e:
            logger.exception(f"[{unique_task_id}] Rollout Error: {e}")
            return None

    async def execute_eval_offline(
        self,
        system_prompt: str,
        instruction: str,
        eval_dir: str,
    ) -> tuple[float, float] | None:

        for filename in os.listdir(eval_dir):
            try:
                if not filename.endswith("json"):
                    continue
                file_path = os.path.join(eval_dir, filename)
                with open(file_path, "r") as f:
                    result = json.load(f)

                score_result = await score_trace(url=self.score_endpoint, trace=result, user_instruction=instruction)
                reward = score_result["score"]
                reason = score_result["reason"]

                logger.info("评分完成: result=%s", score_result)

            except Exception as e:
                logger.exception("[Rollout Error during agent invocation] %s", e)
                continue

            row = {
                "reward": reward,
                "reason": json.dumps(reason, ensure_ascii=False),
            }
            fieldnames = ["reward", "reason"]
            try:
                write_header = not os.path.exists(self.results_file) or os.path.getsize(self.results_file) == 0
                with open(self.results_file, "a", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    if write_header:
                        writer.writeheader()
                    writer.writerow(row)
                logger.info("评估结果已写入 %s: %s", self.results_file, row)
            except Exception as e:
                logger.exception("写入评估结果文件失败: %s", e)


# ================= Worker Function =================
def worker_process_sample_variant(
    sample_variant: dict, unique_task_id: str, sandbox_manager: SandboxManager, evaluator: CuaEvaluator, config: dict
):
    """
    单个线程执行的函数：
    1. 申请 Sandbox
    2. 执行 Rollout
    3. 释放 Sandbox (Finally块保证)
    """
    sandbox_uri = None
    try:
        # 1. 申请资源 (阻塞等待)
        sandbox_uri = sandbox_manager.allocate(unique_task_id)

        # 2. 执行任务（使用 asyncio.run 运行异步方法）
        asyncio.run(evaluator.execute_eval_rollout(
            system_prompt=config["system_prompt"],
            sample=sample_variant,
            sandbox_uri=sandbox_uri,
            unique_task_id=unique_task_id,
            model_name=config["model_name"],
            model_api_key=config["model_api_key"],
            model_endpoint=config["model_endpoint"],
            trace_save_dir=config["trace_save_dir"],
            model_provider=config["model_provider"],
            agent_planner_url=config["agent_planner_url"],
            key_auth=config["key_auth"],
        ))
    except Exception as e:
        logger.error(f"Worker execution failed for {unique_task_id}: {e}")
    finally:
        # 3. 释放资源 (至关重要)
        if sandbox_uri:
            sandbox_manager.release(sandbox_uri)


# ================= Main =================
def iterate_parquet_samples(parquet_path: str, start_row: int = 0):
    df = pd.read_parquet(parquet_path)
    for idx, row in df.iloc[start_row:].iterrows():
        yield idx, row.to_dict()


def eval_offline():
    instruction = "任务开始前，如果当前打开了浏览器，请先关闭所有浏览器窗口回到桌面。随后重新打开浏览器,进入资产审核网站，设置筛选条件，盘点计划WJJ_TEST，盘点审核结果为未盘点,然后盘点2条记录。"
    evaluator = CuaEvaluator("Qwen3-VL-8B-Instruct-offline-eval.csv")
    evaluator.execute_eval_offline(instruction, "./trace/Qwen3-VL-8B-Instruct")


def main():
    # 配置
    config = {
        "eval_path": "data/train.parquet",
        "result_csv": "trace/0121/Qwen3-VL-8B-Instruct-eval-normal.csv",
        "system_prompt": CUA_PROMPT,
        "model_name": "models/Qwen3-VL-8B-Instruct",
        "model_endpoint": "http://localhost:8004/v1",
        "model_api_key": "wangjiaju",
        "model_provider": "openai",
        "trace_save_dir": "./trace/0121/normal",
        "agent_planner_url": "http://0.0.0.0:8332/planner",
        "key_auth": SANDBOX_KEY_AUTH,
    }

    # 1. 初始化组件
    if not os.path.exists(config["eval_path"]):
        logger.error("数据文件不存在")
        return

    # 初始化管理器，设置TTL防止僵尸占用 (例如 20分钟)
    sandbox_manager = SandboxManager(lease_ttl_s=1200)
    evaluator = CuaEvaluator(config["result_csv"])

    # 2. 准备任务队列
    # 获取沙箱数量来决定并发度，可以稍微多一点以便在评分/处理数据时让出CPU
    max_workers = 1
    logger.info(f"🚀 启动线程池，最大并发数: {max_workers}")

    tasks = []

    # 使用 ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, original_sample in iterate_parquet_samples(config["eval_path"]):

            variants = ["RL_07"]

            for i, plan_name in enumerate(variants):
                # 深拷贝样本以防修改冲突
                sample_copy = original_sample.copy()

                try:
                    sample_copy["instruction"] = sample_copy["instruction"].format(plan_name=plan_name)
                except KeyError:
                    pass

                unique_task_id = f"sample_{idx}_ver_{i}"

                # 提交任务到线程池
                future = executor.submit(
                    worker_process_sample_variant,
                    sample_variant=sample_copy,
                    unique_task_id=unique_task_id,
                    sandbox_manager=sandbox_manager,
                    evaluator=evaluator,
                    config=config,
                )
                tasks.append(future)

        logger.info(f"已提交 {len(tasks)} 个任务到队列，等待执行...")

    logger.info("所有评估任务完成。")


if __name__ == "__main__":
    main()
