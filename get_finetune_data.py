import os
import json
import shutil
import random

# 定义源和目标目录
train_src_dir = './dataset/train/pic'
train_label_file = './dataset/train/label.json'
test_src_dir = './dataset/test/pic'
test_label_file = './dataset/test/label.json'

finetune_train_dir = './VisualGLM-6B/finetune-data/train'
finetune_test_dir = './VisualGLM-6B/finetune-data/test'

# 创建目标目录（如果不存在）
os.makedirs(finetune_train_dir, exist_ok=True)
os.makedirs(finetune_test_dir, exist_ok=True)

apparatus = [
"试管","蒸发皿","坩埚","烧杯","锥形瓶","石棉网","三脚架","酒精灯",
"天平","量筒","容量瓶","滴定管","温度计","移液管",
"普通漏斗","长颈漏斗","分液漏斗","冷凝管",
"水槽","细口瓶","广口瓶","滴瓶","导气管",
"坩埚钳","镊子","试管夹",
"燃烧匙","胶头滴管","玻璃棒","护目镜","防护手套","显微镜","放大镜","玻璃皿",
]

def transfer_data(src_dir, label_file, dest_dir):
    # 读取标签文件
    with open(label_file, 'r', encoding='utf-8') as f:
        labels_data = json.load(f)
    
    finetune_data = []
    
    # 遍历每个标签项
    for item in labels_data:
        img_name = item['image']
        labels = item['labels']
        dest_dir_short = "/".join(dest_dir.split("/")[2:])

        for label in labels:
            prompt_str = f"图片代表某个化学实验的场景，{label}是否在该场景中？用一个字（是或否）回答。"
            # 构建新的数据项
            new_item = {
                "img": os.path.join(dest_dir_short, img_name),
                "prompt": prompt_str,
                "label": "是"
            }
            finetune_data.append(new_item)
        
        anti_label = [x for x in apparatus if x not in labels]
        for label in random.sample(anti_label, len(labels)):
            prompt_str = f"图片代表某个化学实验的场景，{label}是否在该场景中？用一个字（是或否）回答。"
            # 构建新的数据项
            new_item = {
                "img": os.path.join(dest_dir_short, img_name),
                "prompt": prompt_str,
                "label": "否"
            }
            finetune_data.append(new_item)
        
        # 复制图片到目标目录
        src_img_path = os.path.join(src_dir, img_name)
        dest_img_path = os.path.join(dest_dir, img_name)
        shutil.copyfile(src_img_path, dest_img_path)
    
    return finetune_data

# 转移训练集数据
train_data = transfer_data(train_src_dir, train_label_file, finetune_train_dir)
# 转移测试集数据
test_data = transfer_data(test_src_dir, test_label_file, finetune_test_dir)

# 保存为新的JSON文件
finetune_train_json_path = os.path.join(finetune_train_dir, 'dataset.json')
with open(finetune_train_json_path, 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4)

finetune_test_json_path = os.path.join(finetune_test_dir, 'dataset.json')
with open(finetune_test_json_path, 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False, indent=4)

print("数据已成功转移并转换格式，分别保存为训练集和测试集。")
