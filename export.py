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

from LitForTokenPosition import LitForTokenPosition

# tokenizer = AutoTokenizer.from_pretrained('./model')
# model=LitForTokenPosition()
# model.load_state_dict(torch.load('./model/state_dict.bin'))
# model.eval()


model=LitForTokenPosition()
# model.load_state_dict(torch.load('./model/state_dict.bin'))
# model.eval()
# Mymodel=distilLitLM()
# Mymodel.toTrain()
PATH="/home/terry/dev/spo/LitForTokenPosition-chinese-out.ckpt"
checkpoint = torch.load(PATH,map_location=torch.device('cpu'))
# print(checkpoint['module_arguments'])
# checkpoint

model.load_state_dict(checkpoint['state_dict'])

model.eval()
torch.save(model.state_dict(), 'LitForTokenPosition_state_dict.bin')

torch.save(model, 'LitForTokenPosition-model.bin')

