import threading, time
from typing import Optional, Dict, Any, List, Set
from enum import Enum
import requests
import time
from typing import Optional, Dict
import traceback
from dotenv import load_dotenv
import os
import agentlightning

agentlightning.configure_logger()

logger = agentlightning.configure_logger(name=__name__)

load_dotenv()

# 沙箱管理器配置（对应前端 sandboxManagerClient）
SANDBOX_MANAGER_URL = os.getenv("sandbox_manager_url")
KEY_AUTH = os.getenv("sandbox_key_auth")
SANDBOX_OS_TYPE = "Linux"


class SandboxBusyError(RuntimeError):
    pass


class SandboxNotFoundError(RuntimeError):
    pass


class SandboxStatus(str, Enum):
    """沙箱状态枚举（与前端SandboxStatus完全一致）"""

    CREATING = "CREATING"  # 创建中
    RUNNING = "RUNNING"  # 运行中（目标状态）
    STOPPED = "STOPPED"  # 已停止
    STOPPING = "STOPPING"  # 停止中
    DELETING = "DELETING"  # 删除中
    DELETED = "DELETED"  # 已删除


# 需要添加创建linux sandbox的功能
class SandboxManager:
    def __init__(self, lease_ttl_s: Optional[int] = None):
        self._lock = threading.RLock()
        # 条件变量用于在“无空闲且已达上限”时等待
        self._cv = threading.Condition(self._lock)
        # 正在创建中的沙箱数量，避免并发超配
        self._creating: int = 0
        self._free: Set[str] = set()  # uri set
        self._in_use: Dict[str, dict] = {}  # uri -> {"task_id": str, "ts": float}
        self._ttl = lease_ttl_s

        # 启动时从远端加载当前运行中的沙箱到空闲池
        self._bootstrap_free_pool_from_remote()

    def _register_sandbox(self, uri: str):
        """内部方法：将创建的沙箱加入空闲池"""
        with self._lock:
            self._free.add(uri)
            # 有新的空闲沙箱，唤醒等待线程
            self._cv.notify()

    def _bootstrap_free_pool_from_remote(self):
        """初始化：获取当前 RUNNING 的沙箱并加入空闲池"""
        try:
            resp = self.list_sandbox()
            sandboxes = resp.get("Result", []) or []
            # running_uris = [sb.get("SandboxId") for sb in sandboxes if sb.get("Status") == "RUNNING"]
            running_uris = [
                "i-yehjzhvj7kxjd1v7mm4f",
                "i-yehjzheoe8xjd1vfgvs2",
                "i-yehjzh68zk4c5qw6rg8j",
                "i-yehjzgz85c4c5qwfyafk",
                "i-yehjzgs7b4xjd1vqfe5x",
                "i-yehjzgjrwgwh2ysgydu3",
                "i-yehjzg9xxcwh2ypweixw",
                "i-yehjzg2x34wh2ypshlh8",
                "i-yehk3xce80cva4f87dzv",
                "i-yehk3tzldscva4f5ynrl",
            ]
            logger.info(f"✅当前沙箱列表: {running_uris}")
            if not running_uris:
                return
            with self._lock:
                for uri in running_uris:
                    if uri and uri not in self._in_use:
                        self._free.add(uri)
                if running_uris:
                    self._cv.notify_all()
        except Exception as e:
            # 初始化失败不致命，仅告警
            logger.warn(f"⚠️ 初始化空闲池失败: {e}")

    def allocate(self, task_id: str) -> str:
        """分配一个空闲沙箱；当前模式仅使用预置 running_uris 池。"""
        while True:
            # 优先尝试直接分配空闲
            with self._lock:
                if self._free:
                    uri = self._free.pop()
                    logger.info(f"task_id: {task_id}, uri: {uri}")
                    self._in_use[uri] = {"task_id": task_id, "ts": time.time()}
                    return uri
                logger.info(f"task_id: {task_id}, no available sandbox")
                self._cv.wait()

    def release(self, uri: str):
        """释放沙箱回到空闲池"""
        with self._lock:
            if uri not in self._in_use:
                return
            self._in_use.pop(uri, None)
            self._free.add(uri)
            logger.info(f"after release, self._free.length: {len(self._free)}")
            # 释放产生空闲，唤醒等待线程
            self._cv.notify()

    def release_by_task(self, task_id: str):
        """按任务释放沙箱"""
        with self._lock:
            for uri, meta in list(self._in_use.items()):
                if meta["task_id"] == task_id:
                    self._in_use.pop(uri, None)
                    self._free.add(uri)
            # 批量释放后唤醒等待线程
            self._cv.notify_all()

    def reap_ttl(self) -> list[str]:
        """回收超时租约"""
        if not self._ttl:
            return []
        now = time.time()
        reclaimed = []
        with self._lock:
            for uri, meta in list(self._in_use.items()):
                if now - meta["ts"] > self._ttl:
                    self._in_use.pop(uri, None)
                    self._free.add(uri)
                    reclaimed.append(uri)
            if reclaimed:
                # 有新的空闲，唤醒等待线程
                self._cv.notify_all()
        return reclaimed

    def stats(self):
        """查看沙箱使用情况"""
        with self._lock:
            return {
                "free": len(self._free),
                "in_use": len(self._in_use),
                "uris_free": list(self._free),
                "uris_in_use": list(self._in_use.keys()),
            }

    def create_sandbox(self, os_type: str = SANDBOX_OS_TYPE) -> str:
        """创建沙箱并等待其进入 RUNNING，再注册并返回"""
        if not SANDBOX_MANAGER_URL or not KEY_AUTH:
            raise EnvironmentError("请设置 SANDBOX_MANAGER_URL 和 KEY_AUTH")

        headers = {"Content-Type": "application/json", "Authorization": KEY_AUTH}
        params = {"Action": "CreateSandbox", "Version": "2020-04-01", "OsType": os_type}

        logger.info(f"✅ 正在创建 {os_type} 沙箱...")
        try:
            response = requests.get(SANDBOX_MANAGER_URL, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            result = response.json()
            uri = result.get("Result", {}).get("SandboxId")

            if not uri:
                raise RuntimeError(f"创建沙箱失败，响应异常：{result}")

            # 等待直到该沙箱状态为 RUNNING
            deadline = time.time() + 180  # 最长等待 3 分钟
            interval = 1.0
            while True:
                try:
                    status = self.get_sandbox_status(uri)
                except Exception:
                    status = None

                if status == SandboxStatus.RUNNING:
                    break

                if time.time() > deadline:
                    raise RuntimeError(f"沙箱 {uri} 启动超时")

                time.sleep(interval)
                interval = min(5.0, interval * 1.5)  # 线性增长到 5s

            logger.info(f"✅ 沙箱创建并就绪，ID: {uri}")
            # 注册沙箱到本地管理（仅在 RUNNING 后加入空闲池）
            self._register_sandbox(uri)
            return uri

        except requests.RequestException as e:
            raise RuntimeError(f"创建沙箱失败：{e}")

    def list_sandbox(self):

        if not SANDBOX_MANAGER_URL or not KEY_AUTH:
            raise EnvironmentError("请设置 SANDBOX_MANAGER_URL 和 KEY_AUTH")

        headers = {"Content-Type": "application/json", "Authorization": KEY_AUTH}
        params = {"Action": "DescribeSandboxes", "Version": "2020-04-01"}

        # logger.info(f"✅ 获取沙箱列表中...")
        response = requests.get(SANDBOX_MANAGER_URL, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        result = response.json()
        # logger.info(f"✅ 沙箱列表获取成功")
        return result

    def delete_sandbox(self, uri: str):
        """删除沙箱"""

        if not SANDBOX_MANAGER_URL or not KEY_AUTH:
            raise EnvironmentError("请设置 SANDBOX_MANAGER_URL 和 KEY_AUTH")

        headers = {"Content-Type": "application/json", "Authorization": KEY_AUTH}
        params = {"Action": "DeleteSandbox", "Version": "2020-04-01", "SandboxId": uri}

        logger.info(f"🗑️ 正在删除沙箱 {uri} ...")
        response = requests.get(SANDBOX_MANAGER_URL, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        logger.info(f"🗑️ 沙箱删除成功: {uri}")

        # 删除后从本地管理中清除
        with self._lock:
            self._free.discard(uri)
            self._in_use.pop(uri, None)
            # 删除释放容量，唤醒等待线程
            self._cv.notify_all()

    def get_sandbox_status(self, uri: str) -> SandboxStatus:
        """查询沙箱状态"""
        sandbox_list_resp = self.list_sandbox()
        sandboxes = sandbox_list_resp.get("Result", [])
        target = next((sb for sb in sandboxes if sb.get("SandboxId") == uri), None)

        if not target:
            raise ValueError(f"未找到沙箱 ID: {uri}")

        status = target.get("Status")
        try:
            return SandboxStatus(status)
        except ValueError:
            raise RuntimeError(f"未知状态: {status}")
