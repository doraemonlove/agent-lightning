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
from concurrent.futures import ThreadPoolExecutor
from reward_server import score_trace
from dotenv import load_dotenv, find_dotenv
from constants import CUA_PROMPT, SANDBOX_LIST_02, SANDBOX_LIST_01
from agentlightning.sandbox import SandboxManager

load_dotenv(find_dotenv())
# ================= 配置与常量 =================
PLANNER_KEY_AUTH = os.getenv("PLANNER_KEY_AUTH")
REWARD_SERVER_PORT = int(os.getenv("REWARD_SERVER_PORT", 8003))
AGENT_PLANNER_PORT = int(os.getenv("AGENT_PLANNER_PORT", 8331))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(threadName)s - %(levelname)s - %(message)s",  # 增加线程名打印
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# CSV 写入锁，防止多线程写入冲突
CSV_LOCK = threading.Lock()


class CuaEvalutor:

    def __init__(self, results_file: str = "./eval_results.csv"):
        self.results_file = results_file
        self.score_endpoint = f"http://0.0.0.0:{REWARD_SERVER_PORT}/overall_score"

    def run_planner_task(
        self,
        sandbox_id: str,
        system_prompt: str,
        user_prompt: str,
        model_name: str,
        model_endpoint: str,
        model_api_key: str,
        model_provider: str = "openai",
        agent_planner_url: str = f"http://0.0.0.0:{AGENT_PLANNER_PORT}/planner",
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
            "max_actions": 45,
            "max_images": 3,
            "thinking_type": "disabled",
            "is_training": True,
            "model_api_key": model_api_key,
            "turn_on_review": False,
            "rollout_id": rollout_id,
        }
        result = []
        try:
            # logger.info("开始调用 Planner: url=%s, sandbox=%s", url, sandbox_id)
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
        sandbox_uri: str,
        unique_task_id: str,
        model_name: str,
        system_prompt: str,
        trace_save_dir: str,
        model_endpoint: str = "",
        model_api_key: str = "",
        model_provider: str = "bedrock",
        agent_planner_url: str = f"http://0.0.0.0:{AGENT_PLANNER_PORT}/planner",
        key_auth: str = "",
    ) -> tuple[float, float] | None:

        start_time = time.time()
        instruction = sample["instruction"]
        scene = sample.get("scene", "unknown")

        try:
            logger.info(f"[{unique_task_id}] 开始 Rollout")
            result = self.run_planner_task(
                sandbox_id=sandbox_uri,
                user_prompt=instruction,
                system_prompt=system_prompt,
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

            full_trace = [{"instruction": instruction, "scene": scene, "sandbox_id": sandbox_uri}] + result
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(full_trace, f, ensure_ascii=False, indent=4)

            # 评分
            end_time_rollout = time.time()
            score_result = await score_trace(
                url=self.score_endpoint, trace=result, user_instruction=instruction, scene=scene
            )
            reward = score_result["score"]
            reason = score_result["reason"]

            # 写入 CSV (加锁)
            elapsed = end_time_rollout - start_time
            row = {
                "task_id": unique_task_id,  # 记录具体的任务ID
                "original_sample_id": sample.get("task_id", "unknown"),
                "reward": reward,
                "scene": scene,
                "reason": json.dumps(reason, ensure_ascii=False),
                "time_sec": f"{elapsed:.4f}",
            }
            fieldnames = ["task_id", "original_sample_id", "reward", "scene", "reason", "time_sec"]

            with CSV_LOCK:
                write_header = not os.path.exists(self.results_file) or os.path.getsize(self.results_file) == 0
                result_dir = os.path.dirname(self.results_file)
                if result_dir:
                    os.makedirs(result_dir, exist_ok=True)
                with open(self.results_file, "a", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    if write_header:
                        writer.writeheader()
                    writer.writerow(row)

            return reward, elapsed

        except Exception as e:
            logger.exception(f"[{unique_task_id}] Rollout Error: {e}")
            return None


# ================= Worker Function =================
async def worker_process_sample(
    sample: dict,
    unique_task_id: str,
    sandbox_manager: SandboxManager,
    evaluator: CuaEvalutor,
    config: dict,
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

        # 2. 执行任务
        await evaluator.execute_eval_rollout(
            sample=sample,
            sandbox_uri=sandbox_uri,
            unique_task_id=unique_task_id,
            system_prompt=config["system_prompt"],
            model_name=config["model_name"],
            model_api_key=config["model_api_key"],
            model_endpoint=config["model_endpoint"],
            trace_save_dir=config["trace_save_dir"],
            model_provider=config["model_provider"],
            agent_planner_url=config["agent_planner_url"],
            key_auth=config["key_auth"],
        )
    except Exception as e:
        logger.error(f"Worker execution failed for {unique_task_id}: {e}")
    finally:
        # 3. 释放资源 (至关重要)
        if sandbox_uri:
            sandbox_manager.release(sandbox_uri)


def _concurrent_worker(
    sample: dict, unique_task_id: str, sandbox_manager: SandboxManager, evaluator: CuaEvalutor, config: dict
):
    """并发模式下的同步 Worker，供 ThreadPoolExecutor 调用。"""
    sandbox_uri = None
    try:
        sandbox_uri = sandbox_manager.allocate(unique_task_id)
        asyncio.run(
            evaluator.execute_eval_rollout(
                sample=sample,
                sandbox_uri=sandbox_uri,
                unique_task_id=unique_task_id,
                system_prompt=config["system_prompt"],
                model_name=config["model_name"],
                model_api_key=config["model_api_key"],
                model_endpoint=config["model_endpoint"],
                trace_save_dir=config["trace_save_dir"],
                model_provider=config["model_provider"],
                agent_planner_url=config["agent_planner_url"],
                key_auth=config["key_auth"],
            )
        )
    except Exception as e:
        logger.error(f"Worker execution failed for {unique_task_id}: {e}")
    finally:
        if sandbox_uri:
            sandbox_manager.release(sandbox_uri)


# ================= Main =================
def iterate_parquet_samples(parquet_path: str, start_row: int = 0):
    df = pd.read_parquet(parquet_path)
    for idx, row in df.iloc[start_row:].iterrows():
        yield idx, row.to_dict()


def build_eval_output_paths(trace_dir: str, exp_name: str) -> tuple[str, str]:
    """Build output paths as <trace_dir>/<date>/<exp_name>/... for CSV and trace files."""
    date_dir = time.strftime("%Y%m%d")
    output_root = os.path.join(trace_dir, date_dir, exp_name)
    result_csv = os.path.join(output_root, f"{date_dir}-{exp_name}.csv")
    return output_root, result_csv


def eval_offline():
    instruction = "任务开始前，如果当前打开了浏览器，请先关闭所有浏览器窗口回到桌面。随后重新打开浏览器,进入资产审核网站，设置筛选条件，盘点计划WJJ_TEST，盘点审核结果为未盘点,然后盘点2条记录。"
    evaluator = CuaEvalutor("Qwen3-VL-8B-Instruct-offline-eval.csv")
    evaluator.execute_eval_offline(instruction, "./trace/Qwen3-VL-8B-Instruct")


def main():
    # 1. 配置
    # 获取环境变量，若不存在则使用默认值
    model_name = os.getenv("qwen_3_model_name", "/models/Qwen3-VL-8B-Instruct")
    model_provider = os.getenv("qwen_3_model_provider", "openai")
    model_endpoint = os.getenv("qwen_3_model_endpoint", "http://0.0.0.0:8441/v1")  # openai 不需要
    model_api_key = os.getenv("qwen_3_model_api_key", "cua")  # openai 不需要

    config = {
        "eval_path": "data/eval.parquet",
        "trace_dir": "./trace",
        "exp_name": "qwen-3-vl-8b-instruct-eval-2",
        "model_name": model_name,
        "model_endpoint": model_endpoint,
        "model_api_key": model_api_key,
        "model_provider": model_provider,
        "system_prompt": CUA_PROMPT,
        "agent_planner_url": f"http://0.0.0.0:{AGENT_PLANNER_PORT}/planner",
        "key_auth": PLANNER_KEY_AUTH,
        "use_concurrent": False,
        "sandbox_list": SANDBOX_LIST_02,
    }

    if not os.path.exists(config["eval_path"]):
        logger.error("数据文件不存在")
        return

    trace_save_dir, result_csv = build_eval_output_paths(trace_dir=config["trace_dir"], exp_name=config["exp_name"])
    config["trace_save_dir"] = trace_save_dir
    config["result_csv"] = result_csv

    os.makedirs(config["trace_save_dir"], exist_ok=True)
    result_dir = os.path.dirname(config["result_csv"])
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)

    # 2. 初始化组件
    configured_sandbox_list = list(dict.fromkeys(config.get("sandbox_list") or []))
    sandbox_manager = SandboxManager(sandbox_list=configured_sandbox_list)
    evaluator = CuaEvalutor(config["result_csv"])

    use_concurrent = config.get("use_concurrent", False)

    if use_concurrent:
        # ===== 并发模式（ThreadPoolExecutor，与 collect_trace 保持一致）=====
        max_workers = len(configured_sandbox_list)
        if max_workers <= 0:
            logger.error("未配置可用 sandbox_list，无法启动并发评估")
            return
        logger.info(f"🚀 启动并发评估流程，最大并发数: {max_workers}")
        tasks = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for idx, original_sample in iterate_parquet_samples(config["eval_path"]):
                task_id = str(original_sample["task_id"])
                future = executor.submit(
                    _concurrent_worker,
                    sample=original_sample,
                    unique_task_id=task_id,
                    sandbox_manager=sandbox_manager,
                    evaluator=evaluator,
                    config=config,
                )
                tasks.append(future)
            logger.info(f"已提交 {len(tasks)} 个任务到队列，等待执行...")
    else:
        # ===== 顺序模式 =====
        logger.info("🚀 启动顺序评估流程")
        all_samples = list(iterate_parquet_samples(config["eval_path"]))
        completed_tasks = 0
        for idx, original_sample in all_samples:
            task_id = str(original_sample["task_id"])
            asyncio.run(
                worker_process_sample(
                    sample=original_sample,
                    unique_task_id=task_id,
                    sandbox_manager=sandbox_manager,
                    evaluator=evaluator,
                    config=config,
                )
            )
            completed_tasks += 1
        logger.info(f"已顺序执行 {completed_tasks} 个任务")

    logger.info("所有评估任务完成，正在计算最终得分...")

    # 3. 统计并写入 CSV
    fieldnames = ["task_id", "original_sample_id", "reward", "scene", "reason", "time_sec"]
    try:
        if os.path.exists(config["result_csv"]):
            df_results = pd.read_csv(config["result_csv"])
            # 过滤掉历史 SUMMARY 行，避免重复运行时重复累计
            df_data = df_results[~df_results["task_id"].astype(str).str.startswith("SUMMARY")]

            if not df_data.empty:
                total_avg = df_data["reward"].mean()
                total_count = len(df_data)

                print("\n" + "=" * 40)
                logger.info("📊 评估完成统计报告")
                logger.info(f"总计完成样本数: {total_count}")
                logger.info(f"全量任务平均分: {total_avg:.4f}")

                scene_avgs = df_data.groupby("scene")["reward"].mean()
                for scene, avg in scene_avgs.items():
                    count = int((df_data["scene"] == scene).sum())
                    logger.info(f"  场景 [{scene}] 样本数: {count}, 平均分: {avg:.4f}")
                print("=" * 40 + "\n")

                # 汇总行写入 CSV
                summary_rows = [
                    {
                        "task_id": "SUMMARY_overall",
                        "original_sample_id": "SUMMARY",
                        "reward": round(float(total_avg), 4),
                        "scene": "all",
                        "reason": f"total_count={total_count}",
                        "time_sec": "",
                    }
                ]
                for scene, avg in scene_avgs.items():
                    count = int((df_data["scene"] == scene).sum())
                    summary_rows.append(
                        {
                            "task_id": f"SUMMARY_{scene}",
                            "original_sample_id": "SUMMARY",
                            "reward": round(float(avg), 4),
                            "scene": scene,
                            "reason": f"count={count}",
                            "time_sec": "",
                        }
                    )
                with open(config["result_csv"], "a", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writerows(summary_rows)
                logger.info("摘要统计已写入 CSV")
            else:
                logger.warning("结果 CSV 文件为空，无法计算分数。")
        else:
            logger.warning(f"未发现结果文件: {config['result_csv']}")
    except Exception as e:
        logger.error(f"计算平均分时发生错误: {e}")


if __name__ == "__main__":
    main()
