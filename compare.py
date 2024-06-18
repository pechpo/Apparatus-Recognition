import json

f_gt = open("./VisualGLM-6B/finetune-data/test/dataset.json", "r", encoding="utf8")
gt_raw = json.load(f_gt)
f_out = open("./VisualGLM-6B/res.json", "r", encoding="utf8")
out = json.load(f_out)

gt = []
for obj in gt_raw:
    gt.append(obj["label"])

TP = FN = FP = 0
for i in range(len(gt)):
    if gt[i]=="是" and out[i]=="是":
        TP += 1
    if gt[i]=="是" and out[i]=="否":
        FN += 1
    if gt[i]=="否" and out[i]=="是":
        FP += 1

recall = TP / (TP + FN)
precision = TP / (TP + FP)

print(f"recall={recall}, precision={precision}")