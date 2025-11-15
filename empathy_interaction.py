# empathy_interaction.py
from typing import Any, Dict, List, Tuple, Optional
from uuid import uuid4

class EmpatheticInteraction:
    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.name = self.config.get("name", "empathetic_agent")
        self._state: Dict[str, Dict[str, Any]] = {}

    async def start_interaction(self, instance_id: Optional[str] = None, **kwargs) -> str:
        """这里只做最小初始化，不去读 interaction_kwargs。"""
        if instance_id is None:
            instance_id = str(uuid4())

        max_turns = int(self.config.get("max_turns", 20))
        self._state[instance_id] = {
            # 在 generate_response 首次被调用时再合并样本级 user_script / rounds / script_next_idx
            "script": [],
            "next_idx": 0,
            "turn_index": 0,
            "max_turns": max_turns,
            "turn_scores": [],
            "turn_pairs": [],
            "turn_texts": [],
            "turn_breakdown": [],
            "reached_L5": False,
            "_ikw_merged": False,  # 首次 gen 时合并
        }

        print(f"[mt_dbg] start_interaction name={self.name} "
              f"script_len=0 next_idx=0 max_turns={max_turns} head3=[]")
        return instance_id

    def _merge_from_kwargs_if_needed(self, st: Dict[str, Any], kwargs: Dict[str, Any]) -> None:
        """首轮 generate_response 时，从 kwargs 合并样本级配置."""
        if st.get("_ikw_merged", False):
            return

        # 兼容两种形态：扁平传参 / nested 在 'interaction_kwargs' 下
        ikw = kwargs

        # 读取脚本、轮次、起始索引
        user_script = ikw["user_script"]
        rounds = ikw["rounds"]
        script_next_idx = ikw["script_next_idx"]

        
        st["script"] = list(user_script)
        st["next_idx"] = int(script_next_idx)
        st["max_turns"] = int(rounds)
        st["_ikw_merged"] = True
        print("[mt_dbg] merge_from_kwargs: "
              f"keys={list((kwargs.get('interaction_kwargs') or kwargs or {}).keys())[:6]} "
              f"script_len={len(st['script'])} next_idx={st['next_idx']} max_turns={st['max_turns']}")

    async def calculate_score(self, instance_id: str, response: str, turn_index: int, **kwargs) -> float:
        """你可以换成自己的打分逻辑；此处占位：非空=1.0。"""
        return 1.0 if (response and response.strip()) else 0.0

    async def generate_response(
        self,
        instance_id: str,
        messages: List[Dict[str, Any]],
        **kwargs
    ) -> Tuple[bool, str, float, Dict[str, Any]]:
        st = self._state[instance_id]

        # ★ 核心改动：首次进入 generate_response 时，再去合并样本级 interaction_kwargs
        if not st.get("_ikw_merged", False):
            self._merge_from_kwargs_if_needed(st, kwargs)

        # --- 取上一轮 user / assistant 文本（兼容 dict 与 sglang.Message）---
        last_user = ""
        last_assistant = ""
        for m in reversed(messages):
            role = getattr(m, "role", None)
            if role is None and isinstance(m, dict):
                role = m.get("role")

            if not last_assistant and role == "assistant":
                last_assistant = getattr(m, "content", None)
                if last_assistant is None and isinstance(m, dict):
                    last_assistant = m.get("content", "")
            elif not last_user and role == "user":
                last_user = getattr(m, "content", None)
                if last_user is None and isinstance(m, dict):
                    last_user = m.get("content", "")

            if last_user and last_assistant:
                break

        print(
            f"[mt_dbg] gen_resp turn_idx={st['turn_index']} "
            f"script_len={len(st['script'])} next_idx={st['next_idx']} "
            f"got_user={(last_user[:30] if last_user else '')} "
            f"got_asst={(last_assistant[:30] if last_assistant else '')}"
        )

        # --- 打分 ---
        score = await self.calculate_score(
            instance_id=instance_id,
            response=last_assistant or "",
            turn_index=st["turn_index"],
        )

        # --- 记账 ---
        curr_turn = st["turn_index"] + 1
        st["turn_index"] += 1
        st["turn_scores"].append(float(score))
        st["turn_texts"].append(last_assistant or "")
        st["turn_pairs"].append({"turn": curr_turn, "user": last_user or "", "assistant": last_assistant or ""})

        # --- 推进下一轮 user：必须来自脚本（如果脚本为空，就不给 next_user，避免假多轮）---
        next_user = ""
        if st["next_idx"] < len(st["script"]):
            next_user = str(st["script"][st["next_idx"]])
            st["next_idx"] += 1

        # --- 结束条件 ---
        should_terminate = (
            st["turn_index"] >= st["max_turns"] + 1
            or st["next_idx"] >= len(st["script"])
        )

        print(
            f"[mt_dbg] decide should_terminate={should_terminate} "
            f"after_inc_next_idx={st['next_idx']}/{len(st['script'])} "
            f"next_user_len={len(next_user) if next_user else 0} "
            f"score={score}"
        )

        # --- 附带元信息（会进 reward 的 extra_info.interaction_meta）---
        meta = {
            "__num_turns__": st["turn_index"],
            "turn_scores": list(st["turn_scores"]),
            "turn_pairs": list(st["turn_pairs"]),
            "turn_breakdown": list(st.get("turn_breakdown", [])),
            "reached_L5": bool(st.get("reached_L5", False)),
            "next_user_preview": (next_user or "")[:64],
        }

        return should_terminate, next_user, float(score), meta

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._state:
            del self._state[instance_id]