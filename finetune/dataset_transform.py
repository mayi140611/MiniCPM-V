# 
import json
def write2json_file(fp, data, encoding='utf-8'):
    """
    Converts arbitrary object recursively into JSON file. 
    Use ensure_ascii=false to output UTF-8.
    """
    with open(fp, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
### 从json文件中读入json
def read_from_json_file(fp, encoding='utf-8'):
    with open(fp, 'r', encoding='utf-8') as f:
        data1 = json.load(f)
    return data1
fp = "/home/jupyter/ollama_models/blob/mm/FoodDialogues/FoodDialogues_test.json"
ll = read_from_json_file(fp, encoding='utf-8')

rs = []
for e in ll:
    dt = {}
    dt["id"] = f"{e['id']}"
    dt["image"] = f"/home/jupyter/ollama_models/blob/mm/nutrition5k_dataset/imagery/realsense_overhead/{e['dish_id']}/rgb.png"
    dt["conversations"] = []
    for ee in e["conversations"]:
        if ee["form"] == "question":
            dt["conversations"].append({
                "role": "user",
                "content": ee["value"]
            })
        elif ee["form"] == "answer":
            dt["conversations"].append({
                "role": "assistant",
                "content": ee["value"]
            })
    rs.append(dt)
write2json_file("FoodDialogues_test_transform_1094.json", rs)

write2json_file("FoodDialogues_test_transform_9.json", rs[:9])
write2json_file("FoodDialogues_test_transform_7.json", rs[-7:])