from torch.utils.data import Dataset, DataLoader
import datasets as preset
from transformers import AutoTokenizer


model_path = "./gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_path,  add_prefix_space=True, padding=True, truncation=True)


class CompositionDataset(Dataset):

    def __init__(self):
        # data loading
        dataset = preset.read_json("./dataset/reduced_dataset.json")
        self.dataset = []
        for i in range(len(dataset)):
            self.dataset.append(tokenizer(dataset[i], return_tensors="pt").input_ids)
        self.num_samples = len(dataset)

    def __getitem__(self, index):
        # dataset[0]
        return self.dataset[index]

    def __len__(self):
        # len(dataset)
        return self.num_samples


if __name__ == "main":
    dataset = CompositionDataset()
    print(dataset[0], len(dataset[0]))
