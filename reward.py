#!/usr/bin/env python

import math
from typing import Any, Dict, Optional, List

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Optional[str] = None,
    user_turn_rewards: Optional[List[float]] = None,
    interaction_metrics: Optional[Dict[str, Any]] = None,
    extra_info: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> float:
    """
    Final reward based on interaction layer history (fallback: average provided per-turn rewards).
    - If total turns < 5, return 0.0.
    - Otherwise, compute per-turn rewards from iceberg layers and average.
    """
    metrics = interaction_metrics
    if not isinstance(metrics, dict):
        return 0.0

    total_turns = int(metrics.get("__num_turns__", metrics.get("turn_index", 0)) or 0)
    if total_turns < 5:
        return 0.0

    layer_history = metrics.get("layer_history") if isinstance(metrics.get("layer_history"), list) else None
    if not layer_history:
        # fallback: average provided per-turn rewards, if any
        if user_turn_rewards:
            return float(sum(user_turn_rewards) / len(user_turn_rewards))
        return 0.0

    # configs (can be overridden by extra_info > reward_config)
    cfg = {}
    if isinstance(extra_info, dict):
        cfg = extra_info.get("reward_config", {}) or {}
    beta = float(cfg.get("beta", 0.7))
    delta = float(cfg.get("delta", 0.6))
    skip_w = float(cfg.get("w", 5.0))
    k = float(cfg.get("k", 8.0))
    c = float(cfg.get("c", 0.5))

    max_depth = float(cfg.get("max_depth", 5.0))

    scores: List[float] = []
    last_layer = int(layer_history[0].get("layer", 1) or 1)
    for entry in layer_history:
        curr_layer = int(entry.get("layer", last_layer) or last_layer)
        conf = float(entry.get("confidence", 1.0))

        depth_reward = curr_layer / max_depth
        progress = max(curr_layer - last_layer, 0) / max_depth
        regress = max(last_layer - curr_layer, 0) / max_depth
        skip_penalty = max(curr_layer - (last_layer + 1), 0) / max_depth

        raw = depth_reward
        raw += beta * progress
        raw -= delta * regress
        raw -= skip_w * skip_penalty
        raw *= 0.5 + 0.5 * conf

        score = 1.0 / (1.0 + math.exp(-k * (raw - c)))
        scores.append(float(score))

        last_layer = curr_layer

    return float(sum(scores) / len(scores))
