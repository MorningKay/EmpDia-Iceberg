#!/usr/bin/env python
"""Empathetic user simulator for VERL.

This module wraps a fine-tuned "user" model trained via LLaMA Factory and
exposes the VERL interaction interface (v0.5.0). It assumes there is **no**
pre-scripted user dialogue: every turn is generated on-the-fly from the model,
optionally conditioned on sample-dependent metadata provided via
``extra_info.interaction_kwargs`` in the VERL dataset.

Key behaviors
-------------
* Lazily load a LLaMA Factory :class:`ChatModel` using either a path to the
  inference YAML (``model_config_path``) or an inlined dict (``model_config``).
* Derive the user-model system prompt from ``first_explanation`` (if provided)
  via the `system_template` field (independent from the RL environment system).
* Accept VERL ``Message`` or Python dict history, then swap roles so the model
  speaks as the user.
"""

from __future__ import annotations

import asyncio
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4
from copy import deepcopy

import yaml
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from llamafactory.chat.chat_model import ChatModel


class EmpatheticInteraction:
    def __init__(self, config: Dict[str, Any]):
        # breakpoint()
        self.config = config or {}
        self.name = self.config.get("name", "empathetic_agent")
        self._state: Dict[str, Dict[str, Any]] = {}

        # model loading options
        self._model_config_dict = self.config.get("model_config")
        self._model_config_path = self.config.get("model_config_path") or self.config.get("inference_config")
        self._generation_kwargs = self.config.get("generation_kwargs", {})
        self._allow_empty_history = bool(self.config.get("allow_empty_history", False))
        self._empty_user_prompt = self.config.get("empty_user_prompt", "")
        self._chat_model: Optional[ChatModel] = None
        self._cls_checkpoint: Optional[str] = self.config.get("cls_checkpoint")
        self._cls_device = torch.device("cuda" if torch and torch.cuda.is_available() else "cpu") if torch else None
        self._classifier = None
        self._tokenizer = None
        self._layers = ["behavior", "coping", "feelings", "feelings_about_feelings", "perceptions"]
        self.k = float(self.config.get("k", 8.0))
        self.c = float(self.config.get("c", 0.5))
        self.beta = float(self.config.get("beta", 0.7))
        self.delta = float(self.config.get("delta", 0.6))
        self.w = float(self.config.get("w", 5.0))
        self.eta_align = float(self.config.get("eta_align", 0.3))

        # user-side system template (independent from RL env system prompt)
        self._user_system_template = "我最近有点不顺（{first_explanation}）。接下来我想找人聊聊，只说我自己的真实感受和想法，不分析别人也不给建议。"

    async def start_interaction(self, instance_id: Optional[str] = None, **kwargs) -> str:
        # breakpoint()
        if instance_id is None:
            instance_id = str(uuid4())

        max_turns = int(self.config.get("default_max_turns", 20))
        self._state[instance_id] = {
            "turn_index": 0,
            "max_turns": max_turns,
            "last_layer": 1,  # track current iceberg depth (1-5)
            "deepest_layer": 1,
            "turn_scores": [],
            "layer_history": [],
            "user_system_prompt": self._user_system_template.format(first_explanation=kwargs.get("first_explanation")),
        }

        print(f"[mt_dbg] start_interaction name={self.name} max_turns={max_turns}")
        return instance_id

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def _canonical_layer(self, label: str) -> Optional[str]:
        # Map classifier raw label to our fixed English layer names (expects 5-way classifier).
        if not label:
            return None
        lower = label.lower()
        normalized = lower.replace(" ", "_").replace("-", "_")

        if normalized in self._layers:
            return normalized
        if lower in self._layers:
            return lower
        return None

    def _ensure_classifier(self) -> None:
        # Lazy-load classifier and tokenizer once per process.
        if self._classifier is not None:
            return
        if not self._cls_checkpoint:
            raise ValueError("EmpatheticInteraction requires `cls_checkpoint` for iceberg classification.")
        self._tokenizer = AutoTokenizer.from_pretrained(self._cls_checkpoint)
        self._classifier = AutoModelForSequenceClassification.from_pretrained(self._cls_checkpoint).to(self._cls_device)
        id2label = self._classifier.config.id2label or {}
        ordered = []
        for i in range(len(id2label)):
            ordered.append(id2label.get(i, str(i)))
        self._label_space = ordered

    def _predict_layer(self, text: str) -> Tuple[int, str, float]:
        # Run classifier to get layer index (1-5), layer name, and confidence.
        self._ensure_classifier()
        inputs = self._tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt",
        ).to(self._cls_device)
        with torch.no_grad():
            logits = self._classifier(**inputs).logits[0]
            probs = torch.softmax(logits, dim=-1)
            idx = int(torch.argmax(probs).item())
            confidence = float(probs[idx].item())
        raw_label = self._label_space[idx] if hasattr(self, "_label_space") else str(idx)
        canonical = self._canonical_layer(raw_label) or "behavior"
        try:
            layer_idx = self._layers.index(canonical) + 1
        except ValueError:
            layer_idx = 1
        layer_name = self._layers[layer_idx - 1]
        return layer_idx, layer_name, confidence

    async def calculate_score(
        self,
        instance_id: str,
        next_user: str,
        turn_index: int,
        **kwargs,
    ) -> tuple[float, Dict[str, float]]:
        # Only classify layer and update tracking; scoring happens in reward function.
        st = self._state[instance_id]
        if not next_user or not next_user.strip():
            return 0.0, {"reason": "empty_next_user"}

        layer_idx, layer_label, confidence = self._predict_layer(next_user)
        st["last_layer"] = layer_idx
        st["deepest_layer"] = max(st["deepest_layer"], layer_idx)

        breakdown = {
            "layer": float(layer_idx),
            "layer_label": layer_label,
            "confidence": float(confidence),
        }
        return 0.0, breakdown

    def _load_model_config(self) -> Dict[str, Any]:
        if isinstance(self._model_config_dict, dict):
            return dict(self._model_config_dict)

        if self._model_config_path is None:
            raise ValueError("EmpatheticInteraction requires `model_config` or `model_config_path`.")

        path = Path(self._model_config_path)
        if not path.exists():
            raise FileNotFoundError(f"Model config path not found: {path}")

        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)

    def _ensure_chat_model(self) -> None:
        if self._chat_model is None:
            cfg = self._load_model_config()
            self._chat_model = ChatModel(cfg)

    @staticmethod
    def _normalize_message(msg: Any) -> tuple[Optional[str], str]:
        role = getattr(msg, "role", None)
        content = getattr(msg, "content", None)
        if role is None and isinstance(msg, dict):
            role = msg.get("role")
        if content is None and isinstance(msg, dict):
            content = msg.get("content")
        return role, content or ""

    def _prepare_history_for_model(self, messages: List[Any]) -> list[dict[str, str]]:
        # Swap roles so the user-model continues from the "user" side.
        normalized: list[dict[str, str]] = []
        for msg in messages:
            role, content = self._normalize_message(msg)
            if role is None:
                continue
            normalized.append({"role": role, "content": content})

        swapped: list[dict[str, str]] = []
        for item in normalized:
            role = item["role"]
            if role == "user":
                mapped = "assistant"
            elif role == "assistant":
                mapped = "user"
            else:
                mapped = role
            swapped.append({"role": mapped, "content": item.get("content", "")})

        if not swapped or swapped[-1].get("role") != "user":
            swapped.append({"role": "user", "content": self._empty_user_prompt})

        return swapped

    async def _generate_user_from_model(self, messages: List[Any], system_prompt: Optional[str]) -> str:
        self._ensure_chat_model()
        history = self._prepare_history_for_model(messages)

        def _call_chat() -> str:
            responses = self._chat_model.chat(history, system=system_prompt, **dict(self._generation_kwargs))
            return responses[0].response_text if responses else ""

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _call_chat)

    async def generate_response(
        self,
        instance_id: str,
        messages: List[Dict[str, Any]],
        **kwargs,
    ) -> Tuple[bool, str, float, Dict[str, Any]]:
        # Core flow: read history, generate next user utterance, classify layer, and return meta.
        st = self._state[instance_id]

        last_user = ""
        last_assistant = ""
        for m in reversed(messages):
            role, content = self._normalize_message(m)
            if not last_assistant and role == "assistant":
                last_assistant = content
            elif not last_user and role == "user":
                last_user = content
            if last_user and last_assistant:
                break

        print(
            f"[mt_dbg] gen_resp turn_idx={st['turn_index']} "
            f"got_user={(last_user[:30] if last_user else '')} "
            f"got_asst={(last_assistant[:30] if last_assistant else '')}"
        )

        next_user = ""
        history_ok = self._allow_empty_history or bool(messages)
        has_budget = st["turn_index"] < st["max_turns"]
        for_user_messages = deepcopy(messages[1:])
        if history_ok and has_budget and st["turn_index"] < st["max_turns"]:
            max_retry = int(self.config.get("retry_on_empty_user", 0))
            attempts = max_retry + 1
            for attempt in range(attempts):
                next_user = await self._generate_user_from_model(
                    for_user_messages,
                    st.get("user_system_prompt")
                )
                if next_user and next_user.strip():
                    if attempt > 0:
                        print(
                            f"[mt_dbg] next_user empty for {attempt} times, "
                            f"finally got non-empty on attempt {attempt + 1}/{attempts}"
                        )
                    break
                else:
                    print(
                        f"[mt_dbg] empty next_user on attempt {attempt + 1}/{attempts}, "
                        f"len={len(next_user) if next_user is not None else 'None'}"
                    )

        curr_turn = st["turn_index"] + 1
        score = 0.0
        breakdown: Dict[str, Any] = {}
        if next_user:
            score, breakdown = await self.calculate_score(
                instance_id=instance_id,
                next_user=next_user,
                turn_index=curr_turn,
            )

        st["turn_index"] = curr_turn
        st["turn_scores"].append(float(score))
        st["layer_history"].append(
            {
                "turn": curr_turn,
                "layer": breakdown.get("layer"),
                "label": breakdown.get("layer_label"),
                "confidence": breakdown.get("confidence"),
            }
        )

        terminate_reason = ""
        if not next_user:
            terminate_reason = "empty_next_user"
        elif st["turn_index"] >= st["max_turns"]:
            terminate_reason = "max_turns"
        else:
            last_six = [h.get("layer") for h in st["layer_history"][-6:]]
            if len(last_six) == 6 and len(set(last_six)) == 1:
                terminate_reason = "no_progress_6"
            last_five = [h.get("layer") for h in st["layer_history"][-5:]]
            if not terminate_reason and len(last_five) == 5 and all(l == len(self._layers) for l in last_five):
                terminate_reason = "perception_5"

        should_terminate = bool(terminate_reason)

        print(
            f"[mt_dbg] decide should_terminate={should_terminate} "
            f"max_turns={st['max_turns']} next_user_len={len(next_user) if next_user else 0} score={score} reason={terminate_reason}"
        )

        meta = {
            "__num_turns__": st["turn_index"],
            "turn_scores": list(st["turn_scores"]),
            "last_layer": st["last_layer"],
            "deepest_layer": st["deepest_layer"],
            "layer_history": list(st.get("layer_history", [])),
            "score_breakdown": breakdown,
            "next_user_preview": (next_user or "")[:64],
            "terminate_reason": terminate_reason,
        }
        # if should_terminate:
        #     breakpoint()
        #     print(f"[mt_debug] scores: {st['turn_scores']}")

        return should_terminate, next_user, float(score), meta

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        self._state.pop(instance_id, None)
