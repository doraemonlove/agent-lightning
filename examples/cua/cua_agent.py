from __future__ import annotations

import os
import time
from typing import Any, cast

import dotenv
import agentlightning
import requests
import traceback
import json

from convert_triplets import convert_traces_to_triplets
from dotenv import load_dotenv

load_dotenv()
# VERL_API_BASE=http://localhost:9999/ python
agentlightning.configure_logger()

logger = agentlightning.configure_logger(name=__name__)

TRACE_DIR = "./trace"
os.makedirs(TRACE_DIR, exist_ok=True)

async def run_planner_task(
    sandbox_id: str,
    user_prompt: str,
    model_name: str,
    model_endpoint: str,
    api_key: str = "",
    out_path: str = "./model_output.json",
    rollout_id: str = ""
) -> list[dict[str, Any]]:
    """
    调用 planner 的 stream 接口。
    每条 stream message 会追加为一条 "response" 记录到 out_path（默认 ./model_output.json，不覆盖）。
    最终把完整的 result 追加为一条 "result" 记录到同一文件。
    （不使用 rollout_id，按要求简化）
    """
    AGENT_PLANNER_URL = "http://0.0.0.0:8331/planner"  # 前端 process.env.AGENT_PLANNER_URL
    # 统一认证密钥（沙箱和Planner共享，前端均使用 process.env.KEY_AUTH）
    KEY_AUTH = os.getenv("sandbox_key_auth")

    url = f"{AGENT_PLANNER_URL}/run/task"
    headers = {"Content-Type": "application/json", "Authorization": KEY_AUTH}
    data = {
        "user_prompt": user_prompt,
        "sandbox_id": sandbox_id,
        "model_name": model_name,
        "model_endpoint": model_endpoint,
        "model_api_key": api_key,
        "max_actions": 25,
        "max_images": 3,
        "thinking_type": "enabled",
        "is_training": True,
        "turn_on_review": False,
        "rollout_id": rollout_id
    }
    result = []
    try:
        print(f"📝 开始执行任务：{user_prompt}（沙箱ID: {sandbox_id}）,rollout_id: {rollout_id}")
        with requests.post(url, headers=headers, data=json.dumps(data), stream=True, timeout=(10, 600)) as response:
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if line.startswith("data: "):
                    try:
                        msg = json.loads(line[len("data: ") :])
                        result.append(msg)
                    except json.JSONDecodeError:
                        print(f"⚠️ 解析任务消息失败：{line}")
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Planner任务执行失败：{str(e)}") from e

    tmp_result = [{"instruction": user_prompt, "sandbox_id": sandbox_id}] + result
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(tmp_result, f, ensure_ascii=False, indent=4)
    return result

def test_llm_endpoint(endpoint: str, model_name: str):
    """简单测试 vLLM/Verl OpenAI 接口是否正常"""
    url = f"{endpoint.rstrip('/')}/models"
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code != 200:
            logger.error(f"❌ LLM endpoint {url} 请求失败: HTTP {resp.status_code}")
            return False
        data = resp.json()
        models = [m.get("id") for m in data.get("data", [])]
        logger.info(f"✅ LLM 接口可访问, 可用模型: {models}")
        if model_name not in models:
            logger.warning(f"⚠️ 模型 {model_name} 不在返回列表中，可能名称不匹配")
        return True
    except Exception as e:
        logger.exception(f"❌ LLM 接口测试失败: {e}")
        return False

class LitCUAAgent(agentlightning.LitAgent):
    score_endpoint: str = "http://localhost:8003/score"
    group_score_endpoint: str = "http://localhost:8003/group_score"

    async def _execute_rollout(
        self, sample: dict[str, Any], *, resources: agentlightning.NamedResources, rollout_id: str, is_training: bool
    ) -> float | None:
        start_time = time.time()

        # 获取sandbox_uri
        sandbox = resources.get("sandbox", {})
        sandbox_uri = sandbox.get("uri")
        if not sandbox_uri:
            raise RuntimeError("sandbox uri not found in resources")

        try:
            llm: agentlightning.LLM = cast(agentlightning.LLM, resources["main_llm"])

            model_name = "/models/Qwen3-VL-8B-Instruct-0112"
            # test_llm_endpoint(llm.endpoint, model_name)
            result = await run_planner_task(
                sandbox_id=sandbox_uri,
                user_prompt=sample["instruction"],
                model_name=model_name,
                model_endpoint=llm.endpoint,
                api_key="wangjiaju",  # 确保已在环境里设置 VERL_API_KEY
                out_path=f"./trace/0112/rl/{rollout_id}_model_output.json",
                rollout_id=rollout_id
            )
            
        except Exception:
            traceback.print_exc()
            result = []

        end_time_rollout = time.time()
        logger.info("[Rollout %s] Time taken for rollout: %.2f seconds", rollout_id, end_time_rollout - start_time)
        
        if result:
            result = await convert_traces_to_triplets(
                score_url=self.group_score_endpoint, 
                traces=result, 
                instruction=sample["instruction"],
                model_path="/models/Qwen3-VL-8B-Instruct-0112"
            )
            triplets = result
        else:
            triplets = []
        
        end_time_eval = time.time()
        logger.info(
            "[Rollout %s] Time taken for evaluation: %.2f seconds", rollout_id, end_time_eval - end_time_rollout
        )
        logger.info(f"triplets: {len(triplets)}")
        return triplets

    async def training_rollout_async(self, task: Any, rollout_id: str, resources: agentlightning.NamedResources) -> Any:  # type: ignore
        logger.info(f"{rollout_id} training_rollout_async")
        return await self._execute_rollout(task, resources=resources, rollout_id=rollout_id, is_training=True)

    async def validation_rollout_async(self, task: Any, rollout_id: str, resources: agentlightning.NamedResources) -> Any:  # type: ignore
        return await self._execute_rollout(task, resources=resources, rollout_id=rollout_id, is_training=False)


if __name__ == "__main__":
    dotenv.load_dotenv()
    agent, trainer = agentlightning.lightning_cli(LitCUAAgent, agentlightning.Trainer)
    trainer.fit_v0(agent, os.environ["VERL_API_BASE"])
