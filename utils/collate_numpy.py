import numpy as np

def multi_turn_numpy_collate(batch):
    # batch: list[dict]
    prompts = [ex["prompt"] for ex in batch]
    prompts = np.array(prompts, dtype=object)
    out = { "prompts": prompts, }
    return out
