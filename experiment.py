import torch
import torch.nn as nn
import torch.nn.functional as F
from scoring.dataset import ScoringDataset
from torch.utils.data import DataLoader


class BiLSTM_ATT(nn.Module):

    def __init__(self):
        super(BiLSTM_ATT, self).__init__()
        self.lstm = nn.LSTM(768, 16, bidirectional=True)
        self.tanh1 = nn.Tanh()
        self.tanh2 = nn.Tanh()
        self.w1 = nn.Parameter(torch.zeros(16 * 2))
        self.fc1 = nn.Linear(16 * 2, 8)
        self.fc2 = nn.Linear(8, 10)


    def forward(self, embedding):
        H, _ = self.lstm(embedding)
        M = self.tanh1(H)
        alpha = F.softmax(torch.matmul(M, self.w1), dim=0).unsqueeze(-1)
        out = H * alpha
        out = torch.sum(out, 0)
        out = F.relu(out)
        out = self.fc1(out)
        out = self.tanh2(out)
        out = self.fc2(out)
        out = F.softmax(out, dim=0)
        return out


model = BiLSTM_ATT()
dataset = ScoringDataset("./scoring/articles/")
loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, drop_last=True)

for i, (data, labels) in enumerate(loader):
    # print(i, data.shape, labels.shape)
    outputs = model(data)
    print(outputs.shape)
    break
