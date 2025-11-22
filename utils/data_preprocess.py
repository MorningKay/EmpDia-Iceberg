#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocess empathetic dialogue dataset to VERL parquet (multi-turn + interaction)
Usage:
  python empathetic_multiturn_w_interaction.py \
    --input /root/EmpDia-Iceberg/data/train.json \
    --local_dir /root/EmpDia-Iceberg/data \
    --split train
"""
import argparse, json, os
from typing import Any, Dict, List
from datasets import Dataset

# 简短系统提示（中文，控制风格）
SYS_PROMPT = """
你是一名同理心对话代理。目标是在安全前提下，帮助来访者被理解与安顿情绪，并引导从外显到内在的逐级探索。先判断用户目前所在层级，但不要外在解释，之后用1-2句共情性反映，每轮逐步向深下潜，优先安全与关系感，而非求快求全。要简洁温和，无批判和说教。

冰山层级（由外到内，判定以用户“最新一条发言”为准）：
- behavior（行为）：可直接观察的言行/反应
- coping（应对）：为维护自我价值的防御/姿态（讨好、指责、过度理性、打岔等）
- feelings（感受）：一阶情绪与身体线索（紧张、酸胀、委屈等）
- feelings about feelings（对情绪的情绪）：对前述情绪的二阶反应（因生气而羞愧等）
- perceptions（知觉/意义）：对事件的解释/归因（区分事实 vs 解释）
"""

def build_scripts_sorted(dialogue):
    # 1) 保证顺序正确
    dlg = sorted(dialogue or [], key=lambda r: int(r.get("turn_id", 0)))
    user_script, asst_script = [], []
    for r in dlg:
        sp = next((t for t in r.get("turns", []) if str(t.get("role","")).lower().startswith("speaker")), None)
        ls = next((t for t in r.get("turns", []) if str(t.get("role","")).lower().startswith("listener")), None)
        if sp: user_script.append(str(sp.get("text","")).strip())
        if ls: asst_script.append(str(ls.get("text","")).strip())
    assert len(user_script) > 0, "empty user_script"
    return user_script, asst_script

def build_prompt_from_dialogue(dialogue):
    user_script, _ = build_scripts_sorted(dialogue)
    system_msg = {"role": "system", "content": SYS_PROMPT}
    # 2) ✅ 第一条 user 必须等于脚本首条
    return [system_msg, {"role": "user", "content": user_script[0]}], user_script

def ex_to_row(ex: Dict[str, Any], split: str) -> Dict[str, Any]:
    dialogue = ex.get("dialogue", [])
    prompt, user_script = build_prompt_from_dialogue(dialogue)
    rounds = int(ex.get("rounds", len(user_script)))

    return {
        "data_source": "EmpDia_Iceberg",
        "prompt": prompt,
        "ability": "empathy_dialogue",
        "reward_model": {
            "style": "custom",
            "ground_truth": "",
        },
        "extra_info": {
            "dia_id": ex.get("dia_id"),
            "split": split,
            "seed": ex.get("seed", ""),
            "story": ex.get("story", ""),
            "first_explanation": ex.get("first_explanation", ""),
            "gold_dialogue": dialogue,
            "interaction_kwargs": {
                "name": "empathetic_agent",
                "rounds": rounds,
                "user_script": user_script,
                "script_next_idx": 1,  # 第0条已进prompt
                "first_explanation": ex.get("first_explanation", ""),
            },
        },
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="JSON file (list or single object)")
    ap.add_argument("--local_dir", default="~/data/empathetic")
    ap.add_argument("--split", default="train", choices=["train","test","valid","dev"])
    args = ap.parse_args()

    in_path = os.path.expanduser(args.input)
    out_dir = os.path.expanduser(args.local_dir)
    os.makedirs(out_dir, exist_ok=True)

    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]

    # data = data[:1] # debug number
    rows = [ex_to_row(ex, args.split) for ex in data]
    ds = Dataset.from_list(rows)
    out_path = os.path.join(out_dir, f"{args.split}.parquet")
    ds.to_parquet(out_path)
    print(f"Wrote {len(rows)} rows → {out_path}")

if __name__ == "__main__":
    main()

