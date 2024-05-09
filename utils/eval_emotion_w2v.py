import soundfile as sf
import librosa
import pandas as pd
import numpy as np

import torch 
from torch import nn
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import Wav2Vec2Processor, Wav2Vec2Model
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import Trainer, TrainingArguments

from sklearn.metrics import accuracy_score


device=torch.device('cuda')

# 모델과 프로세서 로드
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h")

relu=nn.ReLU()

classifier1 = nn.Linear(model.config.hidden_size, 512).to(device)
batchnorm1 = nn.BatchNorm1d(512).to(device)
dropout1 = nn.Dropout(0.2).to(device)

classifier2 = nn.Linear(512,256).to(device)
batchnorm2 = nn.BatchNorm1d(256).to(device) 
dropout2 = nn.Dropout(0.2).to(device)

classifier3 = nn.Linear(256,6).to(device)
model.to(device)
checkpoint=torch.load('model_epoch_second(avail).pth')


model.load_state_dict(checkpoint['model_state_dict'])
classifier1.load_state_dict(checkpoint['classifier1_state_dict'])
classifier2.load_state_dict(checkpoint['classifier2_state_dict'])
batchnorm1.load_state_dict(checkpoint['batchnorm1_state_dict'])
batchnorm2.load_state_dict(checkpoint['batchnorm2_state_dict'])
classifier3.load_state_dict(checkpoint['classifier3_state_dict'])

audio,sr= librosa.load('i_000-1.wav', sr=16000)
max_length=16000*20
current_length = len(audio)

if current_length <= max_length:
            padding_length = max_length - current_length
            audio = np.pad(audio, (0, padding_length), mode='constant')

if current_length > max_length:
            audio = audio[:max_length] 


input_values = processor(audio, return_tensors="pt", padding=True, sampling_rate=16000).input_values
input_values = input_values.to(device)
model_input = input_values.view(1, -1).to(device)
soft=nn.Softmax(dim=1)
with torch.no_grad():
    features = model(model_input).last_hidden_state
    out = classifier1(features[:, 0, :])
    #out = batchnorm1(out)
    out = relu(out)
    out = dropout1(out)
    out = classifier2(out)
    #out = batchnorm2(out)
    out = relu(out)
    out = dropout2(out)
    out = classifier3(out)
    logits=soft(out)

    print(logits)

'''for x in ll2:
    if(x=='angry'):
        one_hot.append([1,0,0,0,0,0])
    if(x=='sadness'):
        one_hot.append([0,1,0,0,0,0])
    if(x=='disgust'):
        one_hot.append([0,0,1,0,0,0])
    if(x=='fear'):
        one_hot.append([0,0,0,1,0,0])
    if(x=='neutral'):
        one_hot.append([0,0,0,0,1,0])
    if(x=='happiness' or x=='surprise'):
        one_hot.append([0,0,0,0,0,1])'''
