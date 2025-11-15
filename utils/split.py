import json

with open("data/all.json", "r", encoding="utf-8") as f:
    data = json.load(f)  # list[dict]

split = 0.7
out1, out2 = data[:int(len(data)*split)], data[int(len(data)*split):]

with open("data/train.json", "w", encoding="utf-8") as f:
    json.dump(out1, f, ensure_ascii=False, indent=2)

with open("data/test.json", "w", encoding="utf-8") as f:
    json.dump(out2, f, ensure_ascii=False, indent=2)