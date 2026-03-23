from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from asset_audit.constants import AUDIT_TASK_TEMPLATE
from browser_search.constants import BROWSER_TASK_TEMPLATE
from file_organization.constants import FILE_TASK_TEMPLATE


class DataGenerator:
    """
    生成三个场景的数据集：
    1. file_organization
    2. browser_search
    3. asset_audit

    默认规模（可在初始化时覆盖）：
    - train:
        file_organization = 100
        browser_search = 100
        asset_audit = 60
    - eval:
        file_organization = 20
        browser_search = 20
        asset_audit = 20

    返回格式：
    {
        "train": [...],
        "eval": [...]
    }

    每条样本包含：
    - task_id
    - scene
    - split
    - instruction
    - meta（便于后续评估或调试）
    """

    def __init__(
        self,
        seed: int = 42,
        output_dir: Optional[str] = None,
        save_by_scene: bool = False,
        counts: Optional[Dict[str, Dict[str, int]]] = None,
        audit_train_plan_names: Optional[Sequence[str]] = None,
        audit_eval_plan_names: Optional[Sequence[str]] = None,
    ):
        self.rng = random.Random(seed)
        self.output_dir = Path(output_dir) if output_dir else None
        self.save_by_scene = save_by_scene

        self.counts = counts or {
            "train": {
                "file_organization": 100,
                "browser_search": 100,
                "asset_audit": 64,
            },
            "eval": {
                "file_organization": 20,
                "browser_search": 20,
                "asset_audit": 20,
            },
        }

        # ========= 文件整理：train/eval 变量池分离，避免泄漏 =========
        self.file_train_pool = {
            "target_folder": [
                "test",
                "archive",
                "project",
                "backup",
                "docs",
                "results",
                "notes",
                "storage",
                "records",
                "tasks",
            ],
            "source_file_1": [
                "report.txt",
                "data.txt",
                "note.txt",
                "fileA.txt",
                "result.txt",
                "log.txt",
                "task.txt",
                "item.txt",
                "draft.txt",
                "source.txt",
            ],
            "source_file_2": [
                "log.txt",
                "info.txt",
                "memo.txt",
                "fileB.txt",
                "record.txt",
                "keep.txt",
                "remain.txt",
                "meta.txt",
                "list.txt",
                "trace.txt",
            ],
            "target_file_1": [
                "doc.txt",
                "dataset.txt",
                "meeting.txt",
                "project_a.txt",
                "final.txt",
                "backup_log.txt",
                "todo.txt",
                "store.txt",
                "meeting_note.txt",
                "record_final.txt",
            ],
            "delete_file": [
                "temp.txt",
                "old.txt",
                "debug.txt",
                "remove.txt",
                "trash.txt",
                "tmp.txt",
                "old_task.txt",
                "unused.txt",
                "temp_note.txt",
                "delete_me.txt",
            ],
        }

        self.file_eval_pool = {
            "target_folder": [
                "lab",
                "workspace",
                "deliverables",
                "reports",
                "materials",
            ],
            "source_file_1": [
                "alpha.txt",
                "beta.txt",
                "gamma.txt",
                "delta.txt",
                "omega.txt",
            ],
            "source_file_2": [
                "keep_alpha.txt",
                "keep_beta.txt",
                "keep_gamma.txt",
                "keep_delta.txt",
                "keep_omega.txt",
            ],
            "target_file_1": [
                "alpha_done.txt",
                "beta_done.txt",
                "gamma_done.txt",
                "delta_done.txt",
                "omega_done.txt",
            ],
            "delete_file": [
                "stale.txt",
                "legacy.txt",
                "noise.txt",
                "garbage.txt",
                "drop_me.txt",
            ],
        }

        # ========= 浏览器搜索：train/eval 查询池分离 =========
        # 这里同时存储 expected_answer，方便后续评估对齐。
        self.browser_train_qa = [
            ("南开大学地址", "天津市南开区卫津路94号"),
            ("北京大学地址", "北京市海淀区颐和园路5号"),
            ("清华大学地址", "北京市海淀区清华园1号"),
            ("复旦大学地址", "上海市杨浦区邯郸路220号"),
            ("天津大学地址", "天津市南开区卫津路92号"),
            ("东京大学地址", "日本东京都文京区本乡7丁目3-1"),
            ("哈佛大学地址", "Massachusetts Hall, Cambridge, MA 02138, USA"),
            ("中国的首都是哪里", "北京"),
            ("日本的首都是哪里", "东京"),
            ("法国的首都是哪里", "巴黎"),
            ("太阳系最大的行星", "木星"),
            ("世界上最大的海洋", "太平洋"),
            ("世界上最长的河流", "尼罗河"),
            ("长城长度是多少", "21196.18千米"),
            ("埃菲尔铁塔高度是多少", "330米"),
            ("珠穆朗玛峰高度是多少", "8848.86米"),
            ("Linux是什么", "一种自由和开放源代码的类Unix操作系统"),
            ("Python列表推导式是什么", "一种用于从可迭代对象生成新列表的简洁语法"),
            ("上海人口是多少", "2487.09万人"),
            ("东京人口是多少", "约1400万人"),
        ]

        self.browser_eval_qa = [
            ("巴西的首都是哪里", "巴西利亚"),
            ("德国的首都是哪里", "柏林"),
            ("意大利的首都是哪里", "罗马"),
            ("加拿大的首都是哪里", "渥太华"),
            ("澳大利亚的首都是哪里", "堪培拉"),
            ("世界上最高的山峰", "珠穆朗玛峰"),
            ("世界上最大的沙漠", "南极沙漠"),
            ("地球上最大的大陆", "亚欧大陆"),
            ("太阳系最小的行星", "水星"),
            ("光合作用是什么", "绿色植物利用光能把二氧化碳和水合成为有机物并释放氧气的过程"),
        ]

        self.browser_train_dirs = [
            "test",
            "answers",
            "notes",
            "records",
            "results",
            "docs",
            "browser_output",
            "fact_files",
        ]
        self.browser_eval_dirs = [
            "eval_answers",
            "eval_notes",
            "eval_docs",
            "eval_results",
        ]

        self.browser_train_files = [
            "ans.txt",
            "result.txt",
            "info.txt",
            "note.txt",
            "query_result.txt",
            "answer.txt",
            "fact.txt",
            "output.txt",
        ]
        self.browser_eval_files = [
            "eval_ans.txt",
            "eval_result.txt",
            "eval_note.txt",
            "eval_info.txt",
        ]

        # ========= 工业审核：建议由真实 plan_name 列表驱动 =========
        self.audit_train_plan_names = list(
            audit_train_plan_names
            or [
                "Deskview",
                "Compnode",
                "Chairite",
                "Tablific",
                "Printway",
                "Laptopix",
                "Monitron",
                "Servsync",
                "Cabintry",
                "Phonenet",
                "Scanline",
                "Camerapy",
                "Rackwell",
                "Mousetry",
                "Screenix",
            ]
        )

        self.audit_eval_plan_names = list(
            audit_eval_plan_names
            or [
                "Wireline",
                "Filepath",
                "Lightfix",
                "Stationz",
            ]
        )

    def generate(self) -> Dict[str, List[Dict[str, Any]]]:
        dataset = {
            "train": [],
            "eval": [],
        }

        # ========= train =========
        dataset["train"].extend(
            self._generate_file_tasks(
                split="train",
                n=self.counts["train"]["file_organization"],
                pool=self.file_train_pool,
            )
        )
        dataset["train"].extend(
            self._generate_browser_tasks(
                split="train",
                n=self.counts["train"]["browser_search"],
                qa_pool=self.browser_train_qa,
                dir_pool=self.browser_train_dirs,
                file_pool=self.browser_train_files,
            )
        )
        dataset["train"].extend(
            self._generate_audit_tasks(
                split="train",
                n=self.counts["train"]["asset_audit"],
                plan_names=self.audit_train_plan_names,
            )
        )

        # ========= eval =========
        dataset["eval"].extend(
            self._generate_file_tasks(
                split="eval",
                n=self.counts["eval"]["file_organization"],
                pool=self.file_eval_pool,
            )
        )
        dataset["eval"].extend(
            self._generate_browser_tasks(
                split="eval",
                n=self.counts["eval"]["browser_search"],
                qa_pool=self.browser_eval_qa,
                dir_pool=self.browser_eval_dirs,
                file_pool=self.browser_eval_files,
            )
        )
        dataset["eval"].extend(
            self._generate_audit_tasks(
                split="eval",
                n=self.counts["eval"]["asset_audit"],
                plan_names=self.audit_eval_plan_names,
            )
        )

        # 按 split 打乱，避免三种场景在整体文件中按固定顺序出现。
        # 使用同一个随机数生成器，确保在固定 seed 下结果可复现。
        self.rng.shuffle(dataset["train"])
        self.rng.shuffle(dataset["eval"])

        # dataset = self.deduplicate(dataset)
        self.quality_check(dataset)
        if self.output_dir:
            self._dump(dataset)

        return dataset

    # =========================
    # 文件整理
    # =========================
    def _generate_file_tasks(
        self,
        split: str,
        n: int,
        pool: Dict[str, Sequence[str]],
    ) -> List[Dict[str, Any]]:
        combos: List[Tuple[str, str, str, str, str]] = []

        for target_folder in pool["target_folder"]:
            for source_file_1 in pool["source_file_1"]:
                for source_file_2 in pool["source_file_2"]:
                    if source_file_1 == source_file_2:
                        continue
                    for target_file_1 in pool["target_file_1"]:
                        for delete_file in pool["delete_file"]:
                            # 避免重名冲突
                            if delete_file in {source_file_1, source_file_2, target_file_1}:
                                continue
                            combos.append(
                                (
                                    target_folder,
                                    source_file_1,
                                    source_file_2,
                                    target_file_1,
                                    delete_file,
                                )
                            )

        if n > len(combos):
            raise ValueError(f"file_organization 可生成组合不足：需要 {n}，实际只有 {len(combos)}")

        selected = self.rng.sample(combos, n)
        tasks: List[Dict[str, Any]] = []

        for idx, (target_folder, source_file_1, source_file_2, target_file_1, delete_file) in enumerate(
            selected, start=1
        ):
            instruction = FILE_TASK_TEMPLATE.format(
                target_folder=target_folder,
                source_file_1=source_file_1,
                source_file_2=source_file_2,
                target_file_1=target_file_1,
                delete_file=delete_file,
            )
            tasks.append(
                {
                    "task_id": f"file_{split}_{idx:04d}",
                    "scene": "file_organization",
                    "split": split,
                    "instruction": instruction,
                    "meta": {
                        "target_folder": target_folder,
                        "source_file_1": source_file_1,
                        "source_file_2": source_file_2,
                        "target_file_1": target_file_1,
                        "delete_file": delete_file,
                        "success_criteria": {
                            "folder_exists": target_folder,
                            "moved_and_renamed_file": f"{target_folder}/{target_file_1}",
                            "retained_file": source_file_2,
                            "deleted_file_absent": delete_file,
                        },
                    },
                }
            )
        return tasks

    # =========================
    # 浏览器搜索 + 文档写入
    # =========================
    def _generate_browser_tasks(
        self,
        split: str,
        n: int,
        qa_pool: Sequence[Tuple[str, str]],
        dir_pool: Sequence[str],
        file_pool: Sequence[str],
    ) -> List[Dict[str, Any]]:
        combos: List[Tuple[str, str, str, str]] = []

        for query, answer in qa_pool:
            for target_directory in dir_pool:
                for file_name in file_pool:
                    combos.append((query, answer, target_directory, file_name))

        if n > len(combos):
            raise ValueError(f"browser_search 可生成组合不足：需要 {n}，实际只有 {len(combos)}")

        selected = self.rng.sample(combos, n)
        tasks: List[Dict[str, Any]] = []

        for idx, (search_query, expected_answer, target_directory, file_name) in enumerate(selected, start=1):
            instruction = BROWSER_TASK_TEMPLATE.format(
                search_query=search_query,
                target_directory=target_directory,
                file_name=file_name,
            )
            tasks.append(
                {
                    "task_id": f"browser_{split}_{idx:04d}",
                    "scene": "browser_search",
                    "split": split,
                    "instruction": instruction,
                    "meta": {
                        "search_query": search_query,
                        "expected_answer": expected_answer,
                        "target_directory": target_directory,
                        "file_name": file_name,
                        "success_criteria": {
                            "directory_exists": target_directory,
                            "file_exists": f"{target_directory}/{file_name}",
                            "document_contains": expected_answer,
                        },
                    },
                }
            )
        return tasks

    # =========================
    # 工业资产审核
    # =========================
    def _generate_audit_tasks(
        self,
        split: str,
        n: int,
        plan_names: Sequence[str],
    ) -> List[Dict[str, Any]]:
        if not plan_names:
            raise ValueError("asset_audit 至少需要一个可用的 plan_name")

        tasks: List[Dict[str, Any]] = []
        for idx in range(1, n + 1):
            plan_name = plan_names[(idx - 1) % len(plan_names)]
            instruction = AUDIT_TASK_TEMPLATE.format(plan_name=plan_name)
            tasks.append(
                {
                    "task_id": f"audit_{split}_{idx:04d}",
                    "scene": "asset_audit",
                    "split": split,
                    "instruction": instruction,
                    "meta": {
                        "plan_name": plan_name,
                        "required_audit_count": 2,
                        "success_criteria": {
                            "website_opened": True,
                            "plan_name": plan_name,
                            "audit_status": "未盘点",
                            "audited_records": 2,
                        },
                    },
                }
            )
        return tasks

    # =========================
    # 保存
    # =========================
    def _dump(self, dataset: Dict[str, List[Dict[str, Any]]]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 整体数据仅按 split 保存，避免 train/eval 混写在同一个文件中。
        for split in ("train", "eval"):
            with open(self.output_dir / f"{split}.json", "w", encoding="utf-8") as f:
                json.dump(dataset[split], f, ensure_ascii=False, indent=2)

        # 可选：按 scene + split 继续拆分保存。
        if self.save_by_scene:
            for split in ("train", "eval"):
                grouped: Dict[str, List[Dict[str, Any]]] = {
                    "file_organization": [],
                    "browser_search": [],
                    "asset_audit": [],
                }
                for item in dataset[split]:
                    grouped[item["scene"]].append(item)

                for scene, items in grouped.items():
                    with open(self.output_dir / f"{scene}_{split}.json", "w", encoding="utf-8") as f:
                        json.dump(items, f, ensure_ascii=False, indent=2)

    def deduplicate(self, dataset: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        去除 instruction 完全重复的数据
        """
        seen = set()
        new_dataset = {"train": [], "eval": []}

        for split in ["train", "eval"]:
            for item in dataset[split]:
                key = hashlib.md5(item["instruction"].encode()).hexdigest()

                if key not in seen:
                    seen.add(key)
                    new_dataset[split].append(item)

        return new_dataset

    def quality_check(self, dataset: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        基础质量检查
        """

        errors = []

        train_instructions = set(item["instruction"] for item in dataset["train"])

        eval_instructions = set(item["instruction"] for item in dataset["eval"])

        # 1. train/eval 泄漏检查
        overlap = train_instructions & eval_instructions
        if overlap:
            errors.append(f"Train/Eval leakage: {len(overlap)} duplicated instructions")

        # 2. instruction 非空
        for split in ["train", "eval"]:
            for item in dataset[split]:
                if not item["instruction"].strip():
                    errors.append(f"Empty instruction in {item['task_id']}")

        # 3. scene 合法性
        valid_scenes = {
            "file_organization",
            "browser_search",
            "asset_audit",
        }

        for split in ["train", "eval"]:
            for item in dataset[split]:
                if item["scene"] not in valid_scenes:
                    errors.append(f"Invalid scene in {item['task_id']}")

        if errors:
            print("Quality check failed:")
            for e in errors:
                print(e)
            raise ValueError("Dataset quality check failed")

        print("Quality check passed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CUA dataset.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="examples/cua/data/generated",
        help="Directory for generated JSON files.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--save-by-scene",
        action="store_true",
        help="Also save scene-level files split by train/eval.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use smaller counts for quick testing.",
    )
    args = parser.parse_args()

    counts = None
    if args.smoke_test:
        counts = {
            "train": {
                "file_organization": 3,
                "browser_search": 3,
                "asset_audit": 3,
            },
            "eval": {
                "file_organization": 2,
                "browser_search": 2,
                "asset_audit": 2,
            },
        }

    generator = DataGenerator(
        seed=args.seed,
        output_dir=args.output_dir,
        save_by_scene=args.save_by_scene,
        counts=counts,
    )
    dataset = generator.generate()

    print("Generation completed")
    print(f"Output dir: {args.output_dir}")
    print(f"save_by_scene: {args.save_by_scene}")
    print(f"train size: {len(dataset['train'])}")
    print(f"eval size: {len(dataset['eval'])}")


if __name__ == "__main__":
    main()

"""
python data_generator.py --output-dir /Users/bytedance/Documents/Codes/cua/agent-lightning/examples/cua/data ----save-by-scene
"""
