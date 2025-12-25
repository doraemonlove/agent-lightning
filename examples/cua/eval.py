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


ZQL_KEY_AUTH = "32e0a153-827d-4972-9bbc-c5209f14c3ef"
WJJ_KEY_AUTH = "fff9c14f-5eac-493a-aff4-9789bfbced27"


logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),  # 日志打到 stdout
    ],
)
logger = logging.getLogger(__name__)


class cua_evaluation:

    def __init__(self, results_file: str = "./eval_results.csv"):
        self.results_file = results_file
        self.score_endpoint = "http://localhost:8003/score"

    def run_planner_task(
        self,
        sandbox_id: str,
        user_prompt: str,
        model_name: str,
        model_endpoint: str,
        model_api_key: str,
        agent_planner_url: str = "http://0.0.0.0:8331/planner",
        key_auth: str = ZQL_KEY_AUTH,
    ) -> list[dict[str, Any]]:

        url = f"{agent_planner_url}/run/task"
        headers = {"Content-Type": "application/json", "Authorization": key_auth}
        data = {
            "user_prompt": user_prompt,
            "sandbox_id": sandbox_id,
            "model_name": model_name,
            "model_endpoint": model_endpoint,
            "max_actions": 40,
            "max_images": 3,
            "thinking_type": "enabled",
            "is_training": True,
            "model_api_key": model_api_key,
            "turn_on_review": False
        }
        result = []
        try:
            logger.info("开始调用 Planner: url=%s, sandbox=%s, model=%s", url, sandbox_id, model_name)
            with requests.post(url, headers=headers, data=json.dumps(data), stream=True, timeout=600) as response:
                try:
                    response.raise_for_status()
                except Exception:
                    # 尝试打印响应内容以便定位错误
                    resp_text = None
                    try:
                        resp_text = response.text
                    except Exception:
                        resp_text = "<无法读取响应文本>"
                    logger.error(
                        "Planner 返回非 2xx 响应: status=%s, body=%s", getattr(response, "status_code", None), resp_text
                    )
                    raise

                for line in response.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    if line.startswith("data: "):
                        try:
                            msg = json.loads(line[len("data: ") :])
                            result.append(msg)
                        except json.JSONDecodeError:
                            logger.warning("解析 Planner 流消息失败，内容: %s", line)
                    else:
                        logger.debug("收到非 data 行: %s", line)
        except Exception as e:
            logger.exception("Planner 调用失败: url=%s, sandbox=%s, err=%s", url, sandbox_id, e)
            raise RuntimeError(f"Planner任务执行失败：{str(e)}") from e

        logger.info("Planner 调用完成，收到 %d 条消息", len(result))
        return result

    def score_trace(self, trace, instruction):
        data = {
            "trace": trace,
            "user_instruction": instruction,
        }
        
        with requests.post(self.score_endpoint, data=json.dumps(data), timeout=600) as response:
            try:
                response.raise_for_status()
            except Exception:
                resp_text = None
                try:
                    resp_text = response.text
                except Exception:
                    resp_text = "<无法读取响应文本>"
                logger.error(
                    f"Planner 返回非 2xx 响应: status={getattr(response, 'status_code', None)}, body={resp_text}"
                )
                raise
            
            score_result = response.json()
            logger.info(f"Planner 评分轨迹成功: {score_result}")
            return score_result
        
    def _execute_eval_rollout(
        self,
        sample: str,
        sandbox_uri: str,
        model_name: str,
        model_endpoint: str = "",
        model_api_key: str = "",
        agent_planner_url: str = "http://0.0.0.0:8331/planner",
        key_auth: str = ZQL_KEY_AUTH,
        trace_save_dir: str = "",
    ) -> tuple[float, float] | None:

        start_time = time.time()
        if not sandbox_uri:
            logger.error("sandbox uri 未提供")
            raise RuntimeError("sandbox uri not found in resources")

        try:
            logger.info("开始 rollout: sample_snippet=%s", str(sample.get("instruction", ""))[:120].replace("\n", " "))
            result = self.run_planner_task(
                sandbox_id=sandbox_uri,
                user_prompt=sample["instruction"],
                model_name=model_name,
                model_endpoint=model_endpoint,
                model_api_key=model_api_key,
                agent_planner_url=agent_planner_url,
                key_auth=key_auth,
            )
            
            os.makedirs(trace_save_dir, exist_ok=True)
            out_path = os.path.join(trace_save_dir, f"sample_{sample['task_id']}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=4)

            end_time_rollout = time.time()
            logger.info("rollout 完成，耗时 %.3fs，轨迹长度 %d", end_time_rollout - start_time, len(result))

            score_result = self.score_trace(trace=result, instruction=sample["instruction"])
            reward = score_result["score"]
            reason = score_result["reason"]

            logger.info("评分完成: result=%s", score_result)

        except Exception as e:
            logger.exception("[Rollout Error during agent invocation] %s", e)
            return

        elapsed = end_time_rollout - start_time
        row = {
            "reward": reward,
            "reason": json.dumps(reason, ensure_ascii=False),
            "time_sec": f"{elapsed:.4f}",
        }
        fieldnames = ["reward", "reason", "time_sec"]
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

        return reward, elapsed

    def execute_eval_offline(
        self,
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

                score_result = self.score_trace(trace=result, instruction=instruction)
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

    
def iterate_parquet_samples(parquet_path: str, start_row: int = 0):
    """从 parquet 读取 records，并从 start_row 开始逐条迭代，返回 (索引, sample_dict)"""
    df = pd.read_parquet(parquet_path)
    for idx, row in df.iloc[start_row:].iterrows():
        yield idx, row.to_dict()


def main():
    eval_path = "/root/code/wangjiaju/agent-lightning/examples/cua/data/train.parquet"
    sandbox_uri = "i-yebwczn8xsqc6io24lrp"
    evaluator = cua_evaluation("Qwen3-VL-8B-Instruct-eval.csv")
    model_endpoint = "http://localhost:8002/v1"
    serve_model_name = "models/Qwen3-VL-8B-Instruct"
    model_api_key = "wangjiaju"
    # model_endpoint = "https://ark.cn-beijing.volces.com/api/v3"
    # serve_model_name = "doubao-1.5-ui-tars-250428"
    # model_api_key = "38484cc0-29ab-4119-99b1-e453e23aeb5c"
    agent_planner_url = "http://0.0.0.0:8331/planner"
    trace_save_dir = "./trace/Qwen3-VL-8B-Instruct"
    # 读取 eval 数据集
    try:
        if not os.path.exists(eval_path):
            logger.error("eval 数据集文件不存在: %s", eval_path)
            return
        df = pd.read_parquet(eval_path)
        total = len(df)
        logger.info("加载评估数据，共 %d 条样本，路径: %s", total, eval_path)
    except Exception as e:
        logger.exception("加载 eval parquet 失败: %s", e)
        return

    processed = 0
    for idx, sample in iterate_parquet_samples(eval_path):
        logger.info("处理样本行 %d", idx)
        try:
            res = evaluator._execute_eval_rollout(
                sample=sample,
                sandbox_uri=sandbox_uri,
                model_name=serve_model_name,
                model_endpoint=model_endpoint,
                model_api_key=model_api_key,
                agent_planner_url=agent_planner_url,
                key_auth=WJJ_KEY_AUTH,
                trace_save_dir=trace_save_dir,
            )
            if res is not None:
                processed += 1
        except Exception:
            logger.exception("样本 %d 处理失败", idx)

    logger.info("评估完成。总样本: %d, 成功处理: %d", total, processed)


def eval_offline():
    instruction = "任务开始前，如果当前打开了浏览器，请先关闭所有浏览器窗口回到桌面。随后重新打开浏览器,进入资产审核网站，设置筛选条件，盘点计划WJJ_TEST，盘点审核结果为未盘点,然后盘点2条记录。"
    evaluator = cua_evaluation("Qwen3-VL-8B-Instruct-offline-eval.csv")
    evaluator.execute_eval_offline(instruction, "./trace/Qwen3-VL-8B-Instruct")

if __name__ == "__main__":
    eval_offline()
