import soundfile as sf
import torch
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
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

device=torch.device('cuda')

# 모델과 프로세서 로드
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-Base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-Base")

# 레이어 정의
relu=nn.ReLU()

classifier1 = nn.Linear(model.config.hidden_size, 512).to(device)
batchnorm1 = nn.BatchNorm1d(512).to(device)
dropout1 = nn.Dropout(0.2).to(device)

classifier2 = nn.Linear(512,256).to(device)
batchnorm2 = nn.BatchNorm1d(256).to(device) 
dropout2 = nn.Dropout(0.2).to(device)

classifier3 = nn.Linear(256,6).to(device)

soft=nn.Softmax(dim=1)


max_length = 16000*20

class AudioDataset(Dataset):
    def __init__(self, audio_paths, labels):
        self.audio_paths = audio_paths
        self.labels = labels

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        # 오디오 파일 로딩 및 전처리
        audio_path = self.audio_paths[idx]
        try:
            audio, sr = librosa.load('emotiondata/'+str(audio_path)+'.wav', sr=16000)
            label=self.labels[idx]
        except FileNotFoundError:
            print(f"FileNotFoundError: No such file or directory: 'emotiondata/{audio_path}.wav'")
            audio,sr= librosa.load('emotiondata/5fbe0c95576e9378b67ad368.wav', sr=16000)
            label=[0,0,1,0,0,0]
            label=np.array(label)

        current_length = len(audio)
        if current_length <= max_length:
            padding_length = max_length - current_length
            audio = np.pad(audio, (0, padding_length), mode='constant')

        if current_length > max_length:
            audio = audio[:max_length]  

        return audio, label

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
model.train()
classifier1.to(device)

optimizer = AdamW(list(model.parameters())+list(classifier1.parameters())+list(classifier2.parameters())+list(classifier3.parameters()), lr=0.0000001)
loss_function = nn.CrossEntropyLoss()
#loss_function2 = nn.BCEWithLogitsLoss()

new_audio_id=[]
new_one_hot=[]
with open('newaudiodata_10.txt', 'r') as file:
    for line in file:
        s=line.split(':')
        id=s[0]
        label=s[1]
        ll=[]
        ll.append(int(label[1]))
        ll.append(int(label[3]))
        ll.append(int(label[5]))
        ll.append(int(label[7]))
        ll.append(int(label[9]))
        ll.append(int(label[11]))
        ll=np.array(ll)
        new_audio_id.append(id)
        new_one_hot.append(ll)

new_one_hot=np.array(new_one_hot)


# 학습 및 검증 데이터셋 생성
audio_id_train, audio_id_val, one_hot_train, one_hot_val = train_test_split(new_audio_id, new_one_hot, test_size=0.05, random_state=42) 
train_dataset = AudioDataset(audio_id_train, one_hot_train)
val_dataset = AudioDataset(audio_id_val, one_hot_val)

# DataLoader 설정
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model=model.to(device)

batch_size = 32
warmup_ratio = 0.1
num_epochs = 5
max_grad_norm = 1
log_interval = 10
learning_rate =  5e-5
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in list(model.named_parameters()) if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in list(model.named_parameters()) if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

t_total = len(train_dataloader) * num_epochs
warmup_step = int(t_total * warmup_ratio)

scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

def calc_accuracy(X, Y):
    max_vals, max_indices = torch.max(X, 1)
    max_indices_cpu = max_indices.cpu().numpy()
    numofcrr=0
    idx=0
    for i in max_indices_cpu:
        if(i==Y[idx]):
            numofcrr=numofcrr+1
    return numofcrr/len(max_indices_cpu)


# 학습 시작
for e in range(num_epochs):
    train_acc = 0.0
    test_acc = 0.0
    model.train()
    idx=0
    for batch in train_dataloader:
        optimizer.zero_grad()
        
        audio, labels = batch
        labels = labels.to(device)
        input_values = processor(audio, return_tensors="pt", padding=True, sampling_rate=16000).input_values
        input_values = input_values.to(device)
        model_input = input_values.view(1 * 32, -1).to(device)

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

        logits=out
        _, labels_indices = torch.max(labels, 1)
        if(len(out)!=len(labels_indices)): 
            print("BATCH SIZE MISMATCH")
            continue

        loss = loss_function(out, labels_indices)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        train_acc += calc_accuracy(out, labels_indices)
        if idx % log_interval == 0:
            print("epoch {} batch id {} loss {} train acc {}".format(e+1, idx+1, loss.data.cpu().numpy(), train_acc / (idx+1)))
        idx=idx+1
    print("epoch {} train acc {}".format(e+1, train_acc / (idx+1)))
    model.eval()
    idx=0
    for batch in val_dataloader:
        audio, labels = batch
        labels = labels.to(device)
        input_values = processor(audio, return_tensors="pt", padding=True, sampling_rate=16000).input_values
        input_values = input_values.to(device)
        model_input = input_values.view(1 * 32, -1).to(device)

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

        _, labels_indices = torch.max(labels, 1)
        test_acc += calc_accuracy(out, labels_indices)
        
    print("epoch {} test acc {}".format(e+1, test_acc / (idx+1)))

torch.save({'model_state_dict': model.state_dict(),
            'classifier1_state_dict': classifier1.state_dict(),
            'batchnorm1_state_dict' : batchnorm1.state_dict(),
            'classifier2_state_dict': classifier2.state_dict(),
            'batchnorm2_state_dict' : batchnorm2.state_dict(),
            'classifier3_state_dict': classifier2.state_dict()},
            'model_epoch_second(avail).pth')