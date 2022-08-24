import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import numpy as np
import pandas as pd

class Trainer():
    def __init__(self) -> None:
        self.model = Graph_Score_Model().cuda()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.loss_func = torch.nn.MSELoss()
        self.train_loader, self.valid_loader = get_dataloader()
        self.lr_scheduler = None #필요할수도
        self.iter = len(self.train_loader)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(self.iter):
            data = data.cuda()
            target = target.cuda()
            prediction = self.model(data)
            loss = self.loss_func(target, prediction)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            print(f'epoch{epoch} {batch_idx}/{len(self.iter)} loss: {loss.item()}')

        return total_loss / len(self.iter)

    def compute_similarity(self, d_v, g_v, a_v):
        return self.model.weight[0] * d_v + self.model.weight[1] * g_v + self.model.weight[2] * a_v

    def train(self):
        for epoch in range(1, self.epoch+1):
            total_loss = self.train_epoch(epoch)
            print(f'{epoch}: {total_loss} {self.model.layer.weight[0].item()} {self.model.layer.weight[1].item()} {self.model.layer.weight[2].item()}')

class Graph_Score_Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(3,1,bias=False) 

    def forward(self, x, info):
        x = self.layer(x) #(batch, all_patient-1, 3)
        attentoin_weight = F.softmax(x.squeeze(1), dim=1) #(batch, all_patient-1, 1)
        y = attentoin_weight * info #(batch, all_patient-1), (batch, all_patient-1) 
        return torch.sum(y, dim=1)

class Score_Dataset(Dataset):
    def __init__(self, data, label) -> None:
        super().__init__()
        #self.diagnosis = np.load(Path(path) / 'diagnosis.txt')
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.dat[index]
        y = self.label[index]
        return x, y

def get_dataloader():
    path = '/home/20191650/eICU-GNN-Transformer/data/tuning_hj_graphs'
    diagnosis_data = np.load(Path(path) / 'diagnoses_scores_1000.npy')
    age_data = np.load(Path(path) / 'age_scores_1000.npy')
    gender_data = np.load(Path(path) / 'gender_scores_1000.npy')
    data = torch.from_numpy(np.stack([diagnosis_data, age_data, gender_data], axis=-1))

    Los_data = torch.FloatTensor(pd.read_csv(Path(path) / 'all_labels.csv')['actualiculos'].values)  

    train_ratio = 0.8
    train_cnt = int(data.shape[0] * train_ratio)
    valid_cnt = data.shape[0] - train_cnt
    indices = torch.randperm(data.shape[0])

    train_x,valid_x = torch.index_select(data,dim =0,index= indices).split([train_cnt,valid_cnt],dim = 0)
    train_y,valid_y = torch.index_select(Los_data,dim =0,index= indices).split([train_cnt,valid_cnt],dim = 0)

    train_dataset = Score_Dataset(train_x, train_y)
    valid_dataset = Score_Dataset(valid_x, valid_y)

    train_loader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=2, shuffle=False)
    return train_loader, valid_loader


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()