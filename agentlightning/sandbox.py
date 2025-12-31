import threading, time
from typing import Optional, Dict, Any, List, Set
from enum import Enum
import requests
import time
from typing import Optional, Dict
import traceback

ZQL_SANDBOX_MANAGER_URL = " https://sd39p23u6uj313gtop150.apigateway-cn-beijing.volceapi.com/mgr/"
ZQL_KEY_AUTH = "32e0a153-827d-4972-9bbc-c5209f14c3ef"

WJJ_SANDBOX_MANAGER_URL = "https://sd3adap5049upp79roojg.apigateway-cn-beijing.volceapi.com/mgr/"
WJJ_KEY_AUTH = "fff9c14f-5eac-493a-aff4-9789bfbced27"

# 沙箱管理器配置（对应前端 sandboxManagerClient）
SANDBOX_MANAGER_URL = WJJ_SANDBOX_MANAGER_URL
KEY_AUTH = WJJ_KEY_AUTH  # 前端 process.env.KEY_AUTH
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
    def __init__(self, sandbox_max_num: int = 10, lease_ttl_s: Optional[int] = None):
        self._lock = threading.RLock()
        # 条件变量用于在“无空闲且已达上限”时等待
        self._cv = threading.Condition(self._lock)
        # 正在创建中的沙箱数量，避免并发超配
        self._creating: int = 0
        self._free: Set[str] = set()  # uri set
        self._in_use: Dict[str, dict] = {}  # uri -> {"task_id": str, "ts": float}
        self._ttl = lease_ttl_s
        self.max_num = sandbox_max_num

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
            running_uris = [sb.get("SandboxId") for sb in sandboxes if sb.get("Status") == "RUNNING"]
            running_uris = [
                "i-yebyrouj28qc6ipf3ago", "i-yebyrom3nkqc6infvo17", "i-yebyro9gjkbw80d63dwd", "i-yebyro2fpcwh2ypfknls", "i-yebyrnslq8cva4gqydwl", "i-yebyrnirr4bw80foocvh", "i-yebyrnbqwwqc6imkn3rl", "i-yebyrmnv9c5i3z5bpcpu", "i-yebwczn8xsqc6io24lrp", "i-yebwczdeyowh2yrawo46",
                "i-yecbu0tmo0wh2yq4ohpb", "i-yecbu0ie4gxjd1w2z7op", "i-yecbu075kwxjd1wcviws", "i-yecbtzxblscva4i9g47w", "i-yecbtzm328bw80c958n5", "i-yecbtzdnnkcva4eylo25", "i-yecbtz588w5i3z3hburf", "i-yecbtyve9swh2yqucaid", "i-yecbtymyv4xjd1u5d2np", "i-yecbtyabr45i3z3f1gku",
                "i-yecbw4ioe8qc6imlr7tg", "i-yecbw44mpsqc6ilg4f1j", "i-yecbw3usqowh2yoc11gv", "i-yecbw3kyrkqc6iok3scn", "i-yecbw36x34cva4gg41nk", "i-yecbw2x340qc6inu45sg", "i-yecbw2onpc5i3z6wl8gq", "i-yecbw2etq85i3z3mtwyx", "i-yecbw226m85i3z80f1zt", "i-yecbw1qy2oqc6io5pqxl",
                "i-yecbxszwn4wh2yq1avlb", "i-yecbxszwn4wh2yq1avlb", "i-yecewfij28xjd1u1ok94"
            ]
            print("✅当前沙箱列表", running_uris)
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
            print(f"⚠️ 初始化空闲池失败: {e}")

    def allocate(self, task_id: str) -> str:
        """分配一个空闲沙箱；
        - 若无空闲且总数(<free + in_use + creating>)小于上限(self.max_num)，则创建新沙箱
        - 若已达上限，则等待直到有空闲
        """
        while True:
            # 优先尝试直接分配空闲
            with self._lock:
                if self._free:
                    uri = self._free.pop()
                    print(f"task_id: {task_id}, uri: {uri}")
                    self._in_use[uri] = {"task_id": task_id, "ts": time.time()}
                    return uri
                print(f"task_id: {task_id}, no available sandbox")
                # total_current = len(self._free) + len(self._in_use) + self._creating
                # if total_current < self.max_num:
                #     # 预占创建名额，避免并发超配
                #     self._creating += 1
                # else:
                #     # 达到上限，等待释放或删除产生的容量
                #     self._cv.wait()
                #     continue  # 被唤醒后重试
                self._cv.wait()
            # # 在锁外执行网络创建，避免阻塞其它操作
            # uri_new: Optional[str] = None
            # try:
            #     uri_new = self.create_sandbox(SANDBOX_OS_TYPE)
            # except Exception:
            #     # 创建失败，释放创建名额并唤醒等待者（容量变化）
            #     with self._lock:
            #         self._creating -= 1
            #         self._cv.notify_all()
            #     raise
            # else:
            #     # 创建成功，释放创建名额，并尽量立即分配该新沙箱
            #     with self._lock:
            #         self._creating -= 1
            #         # create_sandbox 会调用 _register_sandbox 将其加入 _free
            #         if uri_new in self._free:
            #             self._free.remove(uri_new)
            #             self._in_use[uri_new] = {"task_id": task_id, "ts": time.time()}
            #             return uri_new
            #         # 若被其他线程先占用，则唤醒等待者并回到循环重试
            #         self._cv.notify_all()

    def release(self, uri: str):
        """释放沙箱回到空闲池"""
        with self._lock:
            if uri not in self._in_use:
                return
            self._in_use.pop(uri, None)
            self._free.add(uri)
            print(f"after release, self._free.length: {len(self._free)}")
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

        print(f"✅ 正在创建 {os_type} 沙箱...")
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

            print(f"✅ 沙箱创建并就绪，ID: {uri}")
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

        # print(f"✅ 获取沙箱列表中...")
        response = requests.get(SANDBOX_MANAGER_URL, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        result = response.json()
        # print(f"✅ 沙箱列表获取成功")
        return result

    def delete_sandbox(self, uri: str):
        """删除沙箱"""

        if not SANDBOX_MANAGER_URL or not KEY_AUTH:
            raise EnvironmentError("请设置 SANDBOX_MANAGER_URL 和 KEY_AUTH")

        headers = {"Content-Type": "application/json", "Authorization": KEY_AUTH}
        params = {"Action": "DeleteSandbox", "Version": "2020-04-01", "SandboxId": uri}

        print(f"🗑️ 正在删除沙箱 {uri} ...")
        response = requests.get(SANDBOX_MANAGER_URL, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        print(f"🗑️ 沙箱删除成功: {uri}")

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
