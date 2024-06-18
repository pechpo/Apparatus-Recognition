import json

f_gt = open("./VisualGLM-6B/finetune-data/test/dataset.json", "r", encoding="utf8")
gt = json.load(f_gt)
f_out = open("./VisualGLM-6B/res.json", "r", encoding="utf8")
out = json.load(f_out)

res = {}

for i in range(len(gt)):
    if out[i] == "否":
        continue
    obj = gt[i]["prompt"].split("，")[1]
    obj = obj[0:obj.find("是")]
    str = gt[i]["img"]
    res.setdefault(str, []).append(obj)

with open("res.json", "w") as f:
    json.dump(res, f, ensure_ascii=False)