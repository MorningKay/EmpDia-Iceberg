# reward.py  — debug-only version
# 作用：仅打印在 rollout/interaction 阶段写入的字段是否成功抵达 extra_info
# 返回固定 0.0（不做奖励设计）

from typing import Any, Dict, Optional, List
import numpy as np

def _to_py(x: Any) -> Any:
    """把 numpy.ndarray 安全转为 Python 对象（仅一层），避免布尔上下文报错。"""
    if isinstance(x, np.ndarray):
        try:
            return x.tolist()
        except Exception:
            return x
    return x

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Optional[str] = None,
    extra_info: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> float:
    print("[reward_dbg] extra info:", extra_info)
    print("[reward_dbg] kwargs:", kwargs)
    raise RuntimeError()
    ei = extra_info or {}

    # —— 规整我们关心的字段到 Python 原生对象 —— #
    norm: Dict[str, Any] = {}
    for k in (
        "messages",
        "rollout_reward_scores",
        "reward_scores",          # 兼容：如果 RM 还没映射
        "per_turn_pairs",
        "per_turn_texts",
        "num_turns",
        "__num_turns__",          # 兼容：上游原始名字
        "interaction_meta",
    ):
        if k in ei:
            norm[k] = _to_py(ei[k])

    # 兼容映射：reward_scores -> rollout_reward_scores
    if "rollout_reward_scores" not in norm and "reward_scores" in norm:
        norm["rollout_reward_scores"] = norm["reward_scores"]

    # 兼容映射：__num_turns__ -> num_turns
    if "num_turns" not in norm and "__num_turns__" in norm:
        norm["num_turns"] = norm["__num_turns__"]

    # —— 打印总览 —— #
    try:
        keys_preview = sorted(list(ei.keys()))[:20]
        print(f"[reward_dbg] extra_info.keys[:20]={keys_preview}")
    except Exception as e:
        print(f"[reward_dbg] keys_print_error: {e}")

    # —— messages —— #
    msgs = norm.get("messages", [])
    if isinstance(msgs, np.ndarray):
        msgs = msgs.tolist()
    if not isinstance(msgs, list):
        msgs = []
    print(f"[reward_dbg] messages.len={len(msgs)}")
    if msgs:
        # preview = [
        #     {"role": (m.get("role") if isinstance(m, dict) else None),
        #      "content": (str(m.get("content"))[:80] if isinstance(m, dict) else str(m)[:80])}
        #     for m in msgs[:4]
        # ]
        # print(f"[reward_dbg] messages[0:4]={preview}")
        print(f"[reward_dbg]", msgs)

    # 从 messages 复原逐轮 user→assistant（用于核对）
    reconstructed_pairs: List[Dict[str, Optional[str]]] = []
    pending_user: Optional[str] = None
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content", "")
        if role == "user":
            pending_user = content
        elif role == "assistant":
            reconstructed_pairs.append({"user": pending_user, "assistant": content})
            pending_user = None
    if reconstructed_pairs:
        rp_prev = [
            {
                "user": (str(p.get("user"))[:60] if p.get("user") is not None else None),
                "assistant": (str(p.get("assistant"))[:60] if p.get("assistant") is not None else None),
            }
            for p in reconstructed_pairs[:3]
        ]
        print(f"[reward_dbg] reconstructed_pairs.len={len(reconstructed_pairs)}")
        print(f"[reward_dbg] reconstructed_pairs[0:3]={rp_prev}")

    # —— per_turn_pairs / per_turn_texts —— #
    ptp = norm.get("per_turn_pairs")
    if isinstance(ptp, np.ndarray):
        ptp = ptp.tolist()
    if isinstance(ptp, list):
        print(f"[reward_dbg] per_turn_pairs.len={len(ptp)}")
        if ptp:
            p0 = ptp[0] if isinstance(ptp[0], dict) else {}
            print("[reward_dbg] per_turn_pairs[0]=", {
                "turn": p0.get("turn"),
                "user": (str(p0.get("user"))[:60] if isinstance(p0.get("user"), str) else type(p0.get("user"))),
                "assistant": (str(p0.get("assistant"))[:60] if isinstance(p0.get("assistant"), str) else type(p0.get("assistant"))),
            })
    else:
        print("[reward_dbg] per_turn_pairs=None_or_nonlist")

    ptt = norm.get("per_turn_texts")
    if isinstance(ptt, np.ndarray):
        ptt = ptt.tolist()
    if isinstance(ptt, list):
        print(f"[reward_dbg] per_turn_texts.len={len(ptt)}")
        if ptt:
            print(f"[reward_dbg] per_turn_texts[0:3]={[str(x)[:60] for x in ptt[:3]]}")
    else:
        print("[reward_dbg] per_turn_texts=None_or_nonlist")

    # —— rollout_reward_scores —— #
    rr = norm.get("rollout_reward_scores")
    if isinstance(rr, np.ndarray):
        rr = rr.tolist()
    if rr is None:
        print("[reward_dbg] rollout_reward_scores=None")
    elif isinstance(rr, dict):
        utr = rr.get("user_turn_rewards")
        if isinstance(utr, np.ndarray):
            utr = utr.tolist()
        if isinstance(utr, list):
            try:
                head_vals = [float(v) for v in utr[:5]]
            except Exception:
                head_vals = utr[:5]
            print(f"[reward_dbg] rollout_reward_scores.user_turn_rewards.len={len(utr)} head={head_vals}")
        else:
            print("[reward_dbg] rollout_reward_scores present but user_turn_rewards missing/non-list")
    elif isinstance(rr, list):
        try:
            head_vals = [float(v) for v in rr[:5]]
        except Exception:
            head_vals = rr[:5]
        print(f"[reward_dbg] rollout_reward_scores(list).len={len(rr)} head={head_vals}")
    else:
        print(f"[reward_dbg] rollout_reward_scores.type={type(rr)} unsupported")

    # —— num_turns / interaction_meta —— #
    nt = norm.get("num_turns")
    try:
        nt_int = int(nt) if nt is not None else None
    except Exception:
        nt_int = None
    print(f"[reward_dbg] num_turns={nt_int}")

    im = norm.get("interaction_meta")
    if isinstance(im, np.ndarray):
        im = im.tolist()
    if isinstance(im, dict):
        print(f"[reward_dbg] interaction_meta.keys[:10]={list(im.keys())[:10]}")
    elif im is not None:
        print(f"[reward_dbg] interaction_meta.type={type(im)}")

    # —— 调试版直接返回固定 0.0 —— #
    return 0.0