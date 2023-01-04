import datasets_utils as preset
import json


dataset = preset.read_json("./dataset/dataset.json")
cleaned_dataset = preset.clean_data(dataset)
print(len(dataset), len(cleaned_dataset))
with open("./dataset/cleaned_dataset.json", "w", newline="", encoding="utf-8") as f:
    for i in range(len(cleaned_dataset)):
        json.dump(cleaned_dataset[i], f)
        f.write("\n")
