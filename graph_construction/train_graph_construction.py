import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import gc

class Trainer():
    def __init__(self) -> None:
        self.model = Graph_Score_Model().half().cuda()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.loss_func = torch.nn.MSELoss()
        self.train_loader, self.info = get_dataloader()
        print('loaded data')
        self.info = self.info.cuda()
        self.lr_scheduler = None #필요할수도
        self.iter = len(self.train_loader)
        self.epoch = 50
        self.writer = SummaryWriter(f'./log_dir/{len(os.listdir("./log_dir"))}_exp')

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0

        with tqdm(self.train_loader, unit='batch', total=self.iter) as tepoch:
            for data, target in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                data = data.cuda()
                target = target.cuda()
                prediction = self.model(data, self.info)
                loss = self.loss_func(target, prediction)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

                tepoch.set_postfix(loss=loss.item())

        return total_loss / self.iter

    def compute_similarity(self, d_v, g_v, a_v):
        return self.model.weight[0] * d_v + self.model.weight[1] * g_v + self.model.weight[2] * a_v

    def train(self):
        for epoch in range(1, self.epoch+1):
            total_loss = self.train_epoch(epoch)
            self.writer.add_scalar('loss', total_loss, epoch)
            self.writer.add_scalars('weight', {'alpha':self.model.layer.weight[0][0].item(),'beta':self.model.layer.weight[0][1].item(),'gamma':self.model.layer.weight[0][2].item()}, epoch)
            #print(f'Epoch {epoch}: {total_loss} {self.model.layer.weight[0][0].item()} {self.model.layer.weight[0][1].item()} {self.model.layer.weight[0][2].item()}')

class Graph_Score_Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(3,1,bias=False) 
        with torch.no_grad():
            self.layer.weight[0][0].fill_(1)
            self.layer.weight[0][1].fill_(-1)
            self.layer.weight[0][2].fill_(1)

    def forward(self, x, info):
        x = self.layer(x).squeeze(-1).fill_diagonal_(-np.inf) #(batch, all_patient-1, 3)
        attention_weight = F.softmax(x, dim=-1) #(batch, all_patient-1, 1)
        y = attention_weight * info #(batch, all_patient-1), (batch, all_patient-1) 
        return torch.sum(y, dim=1)

class Score_Dataset(Dataset):
    def __init__(self, diagnosis_data, age_data, gender_data, label) -> None:
        super().__init__()
        #self.diagnosis = np.load(Path(path) / 'diagnosis.txt')
        self.diagnosis = diagnosis_data
        self.age = age_data
        self.gender = gender_data
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        x = np.stack([self.diagnosis[index], self.age[index], self.gender[index]], axis=-1).astype(np.float16)
        x = torch.Tensor(x)
        #x_info = self.label
        y = self.label[index]
        return x, y

def get_dataloader():
    path = '/home/20191650/eICU-GNN-Transformer/data/tuning_hj_graphs'
    diagnosis_data = np.load(Path(path) / 'diagnoses_scores_all.npy', mmap_mode='r')
    age_data = np.load(Path(path) / 'age_scores_all.npy', mmap_mode='r')
    gender_data = np.load(Path(path) / 'gender_scores_all.npy', mmap_mode='r')
    #data = torch.stack([diagnosis_data, age_data, gender_data], axis=-1)

    Los_data = torch.FloatTensor(pd.read_csv(Path(path) / 'all_labels.csv')['actualiculos'].values)

    #train_ratio = 0.8
    #train_cnt = int(data.shape[0] * train_ratio)
    #valid_cnt = data.shape[0] - train_cnt
    #indices = torch.randperm(data.shape[0])

    #train_x,valid_x = torch.index_select(data,dim=0,index=indices).split([train_cnt,valid_cnt],dim = 0)
    #train_y,valid_y = torch.index_select(Los_data,dim=0,index= indices).split([train_cnt,valid_cnt],dim = 0)

    train_dataset = Score_Dataset(diagnosis_data, age_data, gender_data, Los_data)
    #valid_dataset = Score_Dataset(valid_x, valid_y)

    train_loader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=True)
    #valid_loader = DataLoader(dataset=valid_dataset, batch_size=2, shuffle=False)
    return train_loader, Los_data

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    seed_everything(42)
    trainer = Trainer()
    trainer.train()


