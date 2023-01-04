from torch.utils.data import Dataset, DataLoader
import datasets_utils
from transformers import AutoTokenizer, AutoModelForCausalLM


model_path = "./gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_path,  add_prefix_space=True)
model = AutoModelForCausalLM.from_pretrained(model_path, pad_token_id=tokenizer.eos_token_id)
tokenizer.pad_token = tokenizer.eos_token


class CompositionDataset(Dataset):

    def __init__(self, path):
        # data loading
        # super(CompositionDataset, self).__init__()
        dataset = datasets_utils.read_json(path)
        self.dataset = []
        for i in range(len(dataset)):
            self.dataset.append(tokenizer(dataset[i], truncation=True, padding="max_length", max_length=250, return_tensors="pt").input_ids)
        self.num_samples = len(dataset)

    def __getitem__(self, index):
        # dataset[0]
        return self.dataset[index]

    def __len__(self):
        # len(dataset)
        return self.num_samples


if __name__ == "__main__":
    dataset = CompositionDataset("./dataset/reduced_dataset.json")
    print(dataset[0], len(dataset[0]))
    loader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=0, drop_last=True)
    dataiter = iter(loader)
    data = dataiter.next()
    print(data)
    print(len(data))
    print(model)
