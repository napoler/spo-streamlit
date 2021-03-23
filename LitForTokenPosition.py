# import streamlit as st
import time
import numpy as np


# from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

# from reformer_pytorch import ReformerLM
# from reformer_pytorch.generative_tools import TrainingWrapper

# import random
# import tqdm
# import gzip
# import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


from transformers import AutoTokenizer,AutoModel







class LitForTokenPosition(pl.LightningModule):
    """构建一个用语预测多元信息的模型
    参考 https://huggingface.co/transformers/model_doc/bert.html#bertfortokenclassification
    
    out_features 参数为起始位置的合集
    """

    def __init__(self,learning_rate=1e-5,batch_size=32,warmup=1000,out_features=60):
        super().__init__()
        self.learning_rate=learning_rate
        self.batch_size=batch_size
        self.warmup = warmup # 更新lr
 
        #初始化预训练模型
        self.model= AutoModel.from_pretrained('./model/')
        # 输出起始位置组数目
        self.out_features=out_features
        
        self.classifier = torch.nn.Linear(in_features=768, out_features=self.out_features)
        self.funct=torch.nn.Tanh()
        


        # del self.bert
    def forward(self, x,loss=True):
        # in lightning, forward defines the prediction/inference actions
#         print("x",x)
        embedding = self.model(x)
#         print("embedding[0]",embedding[0].size())
        embedding = self.funct(embedding[0]) # 获取第一个输出
        logits=self.classifier(embedding)
        
        # 行列转换 方便计算最大可能性位置
        
        # 转换后输出的最大可能性位置 并且调整为 数字对模式
#         return  torch.max(logits,1)
        return logits

        
    def training_step(self, batch, batch_idx):
        """
        batch[0] 为 文字矩阵 
        batch[1] 为 起始位置矩阵   长度=self.out_features
        """
#         print("batch[0]",batch[0].size(),batch[1].size())
#         print("batch[0]",batch)
        
        loss_fct = torch.nn.CrossEntropyLoss()
        out=self(batch[0])
#         print("out",out.size())
        
        loss=None
        for one,label in zip(out.split(1,-1),batch[1].split(1,-1)):

    #         print("label",label)
#             print(one.squeeze(-1),label.view(-1))



            if loss==None:
                loss=loss_fct(one.squeeze(-1),label.view(-1))
            else:
                loss=loss+loss_fct(one.squeeze(-1),label.view(-1))

        loss=loss/batch[1].size(1)
        self.log('train_loss', loss)
        return loss


    def validation_step(self, batch, batch_idx):
        """
        batch[0] 为 文字矩阵 
        batch[1] 为 起始位置矩阵   长度=self.out_features
        """
        loss_fct = torch.nn.CrossEntropyLoss()
        x,y=batch
        out=self(x)
        
        loss=None
        for one,label in zip(out.split(1,-1),y.split(1,-1)):

            if loss==None:
                loss=loss_fct(one.squeeze(-1),label.view(-1))
            else:
                loss=loss+loss_fct(one.squeeze(-1),label.view(-1))

        loss=loss/y.size(1)
        self.log('train_loss', loss)
        return loss
    def configure_optimizers(self):
        """优化器"""

        optimizer = torch.optim.AdamW(self.parameters(), lr=(self.learning_rate))
        def warm_decay(step):
            if step < self.warmup:
                return  step / self.warmup
            return self.warmup ** 0.5 * step ** -0.5
    
        lr_scheduler={
        'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer,warm_decay),
        "interval": "step", #runs per batch rather than per epoch
        "frequency": 1,
        'name': 'lr_scheduler'
        }
#         scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return [optimizer], [lr_scheduler]
    def train_dataloader(self):
        return DataLoader(train_dataset, batch_size=self.batch_size)
        pass

