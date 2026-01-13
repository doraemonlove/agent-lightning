# Copyright (c) Microsoft. All rights reserved.

import json
import logging
import time
from typing import Any, Dict, List, Optional, cast

from opentelemetry.sdk.trace import ReadableSpan

from agentlightning.adapter import TracerTraceToTriplet
from agentlightning.client import AgentLightningClient
from agentlightning.litagent import LitAgent
from agentlightning.litagent.litagent import is_v0_1_rollout_api
from agentlightning.tracer.base import Tracer
from agentlightning.types import RolloutLegacy, RolloutRawResultLegacy, Triplet, Task

from .base import Runner
import asyncio
import os
logger = logging.getLogger(__name__)

__all__ = [
    "LegacyAgentRunner",
]
ROLLOUT_PARALLEL_NUM = int(os.getenv("ROLLOUT_PARALLEL_NUM", 8))
BASE_PLAN_NAME = os.getenv("BASE_PLAN_NAME", "WJJ_TEST")
PLAN_NAME_LIST = [
    {
        "plan_name": f"{BASE_PLAN_NAME}_{i+1:02d}",
    }
    for i in range(ROLLOUT_PARALLEL_NUM)
]

class LegacyAgentRunner(Runner[Any]):
    """Manages the agent's execution loop and integrates with AgentOps.

    This class orchestrates the interaction between the agent (`LitAgent`) and
    the server (`AgentLightningClient`). It handles polling for tasks, executing
    the agent's logic, and reporting results back to the server. If enabled,
    it will also automatically trace each rollout using AgentOps.

    Attributes:
        agent: The `LitAgent` instance containing the agent's logic.
        client: The `AgentLightningClient` for server communication.
        tracer: The tracer instance for this runner/worker.
        worker_id: An optional identifier for the worker process.
        max_tasks: The maximum number of tasks to process before stopping.
    """

    def __init__(
        self,
        agent: LitAgent[Any],
        client: AgentLightningClient,
        tracer: Tracer,
        triplet_exporter: TracerTraceToTriplet,
        worker_id: Optional[int] = None,
        max_tasks: Optional[int] = None,
    ):
        super().__init__()
        self.agent = agent
        self.client = client
        self.tracer = tracer
        self.triplet_exporter = triplet_exporter

        # Worker-specific attributes
        self.worker_id = worker_id
        self.max_tasks = max_tasks

    # These methods are overridden by Runner, getting them back to old behavior.
    def init(self, *args: Any, **kwargs: Any) -> None:
        pass

    def init_worker(self, worker_id: int, *args: Any, **kwargs: Any) -> None:
        self.worker_id = worker_id

    def teardown_worker(self, worker_id: int, *args: Any, **kwargs: Any) -> None:
        pass

    def teardown(self, *args: Any, **kwargs: Any) -> None:
        pass

    def _log_prefix(self, rollout_id: Optional[str] = None) -> str:
        """Generates a standardized log prefix for the current worker."""
        if self.worker_id is not None:
            if rollout_id:
                return f"[Worker {self.worker_id} | RolloutLegacy {rollout_id}]"
            else:
                return f"[Worker {self.worker_id}]"
        if rollout_id:
            return f"[RolloutLegacy {rollout_id}]"
        return "[Default Worker]"

    def _to_rollout_object(
        self,
        result: Optional[RolloutRawResultLegacy | List[Triplet]],
        rollout_id: str,
    ) -> RolloutLegacy:
        """Standardizes the agent's return value into a RolloutLegacy object.

        Args:
            result: The output from the agent's rollout method.
            rollout_id: The unique identifier for the current task.

        Returns:
            A standardized `RolloutLegacy` object for reporting to the server.
        """
        final_reward: Optional[float] = 0.0
        
        if result:
            length = len(result)
            for item in result:
                final_reward += item.reward
            final_reward /= length
        # Create the Rollout object with standardized fields
        result_dict: Dict[str, Any] = {
            "rollout_id": rollout_id,
        }
        if final_reward is not None:
            result_dict["final_reward"] = round(final_reward, 2)

        result_dict["triplets"] = result

        return RolloutLegacy(**result_dict)

    def run(self) -> bool:  # type: ignore
        """Poll the task and rollout once synchronously."""
        self.agent.set_runner(self)  # Ensure the agent has a reference to this runner

        task = self.client.poll_next_task()
        if task is None:
            logger.info(f"{self._log_prefix()} Poll returned no task. Exiting.")
            return False
        rollout_id = task.rollout_id

        resources_id = task.resources_id
        resources_update = None
        if resources_id:
            resources_update = self.client.get_resources_by_id(resources_id)
        else:
            logger.debug(f"{self._log_prefix(rollout_id)} No 'resources_id'. Fetching latest resources.")
            resources_update = self.client.get_latest_resources()
        if not resources_update:
            logger.error(f"{self._log_prefix(rollout_id)} Failed to fetch resources. Skipping.")
            return False

        rollout_obj = RolloutLegacy(rollout_id=task.rollout_id, task=task)  # Default empty rollout

        try:
            try:
                self.agent.on_rollout_start(task, self, self.tracer)
            except Exception:
                logger.exception(f"{self._log_prefix(rollout_id)} Exception during on_rollout_start hook.")

            with self.tracer._trace_context_sync(name=f"rollout_{rollout_id}"):  # pyright: ignore[reportPrivateUsage]
                start_time = time.time()
                rollout_method = self.agent.training_rollout if task.mode == "train" else self.agent.validation_rollout

                from copy import deepcopy

                resources = deepcopy(resources_update.resources)

                sandbox_uri = (task.metadata or {}).get("sandbox_uri")
                if sandbox_uri:
                    resources["sandbox"] = {
                        "type": "sandbox",
                        "uri": sandbox_uri
                    }

                # Pass the task input, not the whole task object
                if is_v0_1_rollout_api(rollout_method):
                    result = cast(
                        RolloutRawResultLegacy,
                        rollout_method(
                            task.input, rollout_id=rollout_obj.rollout_id, resources=resources  # type: ignore
                        ),
                    )  # type: ignore
                else:
                    result = rollout_method(task.input, resources=resources, rollout=rollout_obj)
                rollout_obj = self._to_rollout_object(result, task.rollout_id)
                end_time = time.time()
                logger.info(
                    f"{self._log_prefix(rollout_id)} Completed in "
                    f"{end_time - start_time:.2f}s. Triplet length: "
                    f"{len(rollout_obj.triplets) if rollout_obj.triplets is not None else 'N/A'}. "
                    f"Reward: {rollout_obj.final_reward}"
                )

        except Exception:
            logger.exception(f"{self._log_prefix(rollout_id)} Exception during rollout.")
        finally:
            try:
                self.agent.on_rollout_end(task, rollout_obj, self, self.tracer)  # type: ignore
            except Exception:
                logger.exception(f"{self._log_prefix(rollout_id)} Exception during on_rollout_end hook.")
            self.client.post_rollout(rollout_obj)

        return True

    def iter(self) -> int:  # type: ignore
        """Executes the synchronous polling and rollout loop."""
        num_tasks_processed = 0
        logger.info(f"{self._log_prefix()} Started sync rollouts (max: {self.max_tasks or 'unlimited'}).")

        while self.max_tasks is None or num_tasks_processed < self.max_tasks:
            if self.run():
                num_tasks_processed += 1

            if num_tasks_processed % 10 == 0 or num_tasks_processed == 1:
                logger.info(f"{self._log_prefix()} Progress: {num_tasks_processed}/{self.max_tasks or 'unlimited'}")

        logger.info(f"{self._log_prefix()} Finished sync rollouts. Processed {num_tasks_processed} tasks.")
        return num_tasks_processed

    async def run_async(self, plan_name: str = None) -> bool:
        """Poll the task and rollout once."""
        logger.info(f"run async start")
        self.agent.set_runner(self)  # Ensure the agent has a reference to this runner
        
        task = await self.client.poll_next_task_async()
        if task is None:
            logger.info(f"{self._log_prefix()} Poll returned no task. Exiting.")
            return False
        rollout_id = task.rollout_id
        resources_id = task.resources_id
        resources_update = None
        if resources_id:
            resources_update = await self.client.get_resources_by_id_async(resources_id)
        else:
            logger.debug("No 'resources_id'. Fetching latest resources.")
            resources_update = await self.client.get_latest_resources_async()
        if not resources_update:
            logger.error(" Failed to fetch resources. Skipping.")
            raise
        rollout_obj = RolloutLegacy(rollout_id=task.rollout_id, task=task)  # Default empty rollout
        
        instruction_template = task.input.get("instruction", "")
        if instruction_template and plan_name:
            rendered_text = instruction_template.format(**{"plan_name": plan_name})
            task.input["instruction"] = rendered_text

        try:
            try:
                self.agent.on_rollout_start(task, self, self.tracer)
            except Exception:
                logger.exception(f"{self._log_prefix(rollout_id)} Exception during on_rollout_start hook.")

            async with self.tracer.trace_context(name=f"rollout_{rollout_id}"):
                start_time = time.time()
                rollout_method = (
                    self.agent.training_rollout_async if task.mode == "train" else self.agent.validation_rollout_async
                )
                # Pass the task input, not the whole task object
                from copy import deepcopy

                resources = deepcopy(resources_update.resources)
                sandbox_uri = (task.metadata or {}).get("sandbox_uri")
                if sandbox_uri:
                    resources["sandbox"] = {
                        "type": "sandbox",
                        "uri": sandbox_uri
                    }

                if is_v0_1_rollout_api(rollout_method):
                    result = cast(
                        RolloutRawResultLegacy,
                        await rollout_method(
                            task.input, rollout_id=rollout_obj.rollout_id, resources=resources  # type: ignore
                        ),
                    )  # type: ignore
                else:
                    result = await rollout_method(task.input, resources=resources, rollout=rollout_obj)  # type: ignore
                rollout_obj = self._to_rollout_object(result, task.rollout_id)  # type: ignore
                end_time = time.time()
                logger.info(
                    f"{self._log_prefix(rollout_id)} Completed in "
                    f"{end_time - start_time:.2f}s. Triplet length: "
                    f"{len(rollout_obj.triplets) if rollout_obj.triplets is not None else 'N/A'}. "
                    f"Reward: {rollout_obj.final_reward}"
                )
        except Exception:
            logger.exception(f"{self._log_prefix(rollout_id)} Exception during rollout.")
        finally:
            try:
                self.agent.on_rollout_end(task, rollout_obj, self, self.tracer)  # type: ignore
            except Exception:
                logger.exception(f"{self._log_prefix(rollout_id)} Exception during on_rollout_end hook.")
            await self.client.post_rollout_async(rollout_obj)

        return True

    async def iter_async(self, plan_name: str = None) -> int:
        """Executes the asynchronous polling and rollout loop."""
        num_tasks_processed = 0
        logger.info(f"{self._log_prefix()} Started async rollouts (max: {self.max_tasks or 'unlimited'}).")

        while self.max_tasks is None or num_tasks_processed < self.max_tasks:
            if await self.run_async(plan_name=plan_name):
                num_tasks_processed += 1

            if num_tasks_processed % 10 == 0 or num_tasks_processed == 1:
                logger.info(f"{self._log_prefix()} Progress: {num_tasks_processed}/{self.max_tasks or 'unlimited'}")
        logger.info(f"{self._log_prefix()} Finished async rollouts. Processed {num_tasks_processed} tasks.")
        return num_tasks_processed