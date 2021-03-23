import streamlit as st
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




def pre(text):
    model=LitForTokenPosition()
    model.load_state_dict(torch.load('./model/LitForTokenPosition_state_dict.bin'))
    model.eval()
    st.markdown("模型加载成功...")
    tokenizer = AutoTokenizer.from_pretrained('./model')
    # text=""
    # print("text",text)
    ids=tokenizer(text,return_tensors="pt")
    words=tokenizer.tokenize(text)
    print(ids)
    out=model(ids["input_ids"])
    # out
    print("out.size()",out.size())
    new=out.permute(0,2,1)
    # new

    # 转换后输出的最大可能性位置
    p=torch.argmax(new,2)
    for it in torch.reshape(p,(1,10,3,2)).tolist():
    #     print(it)
        # print("新内容")
        # t=tokenizer.decode(one[0].tolist()[0])
        # st.markdown(text)
        
        for it1 in it:
            # 
    #         print(it1)
            # print("new spo")
            new=[]
            for it2 in it1:
                
                print("i2",it2)
                # st.text(words[it2[0]:it2[1]])
                #st.markdown("".join(words[it2[0]:it2[1]]))
                
                if it2[0]<it2[1] and it2[1] != 0:
                    # st.markdown("//".join([str(it2[0]),str(it2[1])]))
                    w="".join(words[it2[0]-1:it2[1]-1])
                    w=w.replace("##", "")
                    new.append(w)
            if len(new)==3:
                st.markdown("SPO:")  
                st.markdown("> "+" == ".join(new))     
                
        

    #     print(words)
        break
    st.markdown("End")




st.title('信息提取测试')
st.markdown('''
    在下面输入框内输入需要测试的文本
    ''')

in_txt = st.text_input('输入测试', value='消化性溃疡病@常见的临床表现有消化不良，集中于上腹的慢性的、周期性的疼痛或不适。')

# print(type(in_txt))
# if not eng_txt:
#     return（三）右肺静脉与下腔静脉 相连所有右肺静脉（偶可为右肺中、下叶的肺静脉）汇入下腔静脉，此类型不多见。因在胸片上右肺下野见一特征性新月形阴影，故又可称“弯刀综合征”。
st.title('模型预测')
if len(in_txt)>=10:
    st.markdown(in_txt)
    pre(in_txt)
else:
    st.markdown("内容过短 >10 plz!")



 
##st.button("Re-run")
