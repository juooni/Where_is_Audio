from transformers import Wav2Vec2Processor, Wav2Vec2Model
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import Trainer, TrainingArguments
import torch 
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import soundfile as sf
import torch
import librosa
import pandas as pd
import numpy as np

# 모델과 프로세서 로드
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch.nn as nn
import torch
device=torch.device('cuda')

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-Base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-Base")
relu=nn.ReLU()
classifier1 = nn.Linear(model.config.hidden_size, 6).to(device)

#print(model)
#model.eval()  # Ensure the Wav2Vec2 model is in evaluation mode
#model = AudioClassifier(model, classifier, softmax)
max_length = 16000*10


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
            #5fbe0c95576e9378b67ad368,아유 그래 음식물 쓰레기는 빨리빨리 버려야지.,disgust
    # 파일을 찾을 수 없는 경우 빈 오디오 데이터와 샘플링 주파수를 반환하거나 원하는 처리를 수행할 수 있습니다.
            audio,sr= librosa.load('emotiondata/5fbe0c95576e9378b67ad368.wav', sr=16000)
            label=[0,0,1,0,0,0]
            label=np.array(label)
        #print(audio)
        #one_hot.append([0,0,1,0,0,0])
        current_length = len(audio)
        ##print(audio_path)
        ##print(self.labels[idx])
        #input_values = processor(audio, return_tensors="pt", sampling_rate=16000).input_values
        if current_length <= max_length:
            padding_length = max_length - current_length
            audio = np.pad(audio, (0, padding_length), mode='constant')

        if current_length > max_length:
            audio = audio[:max_length]  

        return audio, label
'''
df = pd.read_csv("4차년도.csv")
happiness, angry, disgust, fear, neutral, sadness, surprise
print("4차")
print(df['��Ȳ'].unique())
labels=list(df['��Ȳ'])
audio_id=list(df['wav_id'])
one_hot=[]

for x in labels:
    
    if(x=='anger'):
        one_hot.append([1,0,0,0,0,0])
    if(x=='sad'):
        one_hot.append([0,1,0,0,0,0])
    if(x=='disgust'):
        one_hot.append([0,0,1,0,0,0])
    if(x=='fear'):
        one_hot.append([0,0,0,1,0,0])
    if(x=='neutral'):
        one_hot.append([0,0,0,0,1,0])

print('audio size'+str(len(audio_id)))
print('onehot size'+str(len(one_hot))) 
#one_hot=np.array(one_hot)
#print(one_hot)
df = pd.read_csv("5차년도.csv",encoding='cp949')
ll=list(df['상황'])
id=list(df['wav_id'])
for a in id:
    audio_id.append(a)

for x in ll:
    if(x=='anger'):
        one_hot.append([1,0,0,0,0,0])
    if(x=='sad'):
        one_hot.append([0,1,0,0,0,0])
    if(x=='disgust'):
        one_hot.append([0,0,1,0,0,0])
    if(x=='fear'):
        one_hot.append([0,0,0,1,0,0])
    if(x=='neutral'):
        one_hot.append([0,0,0,0,1,0])
        
print('audio size'+str(len(audio_id)))
print('onehot size'+str(len(one_hot)))

df = pd.read_csv("5차년도_2차.csv",encoding='cp949')
print("5차2차")
print(df['상황'].unique())
ll2=list(df['상황'])
id2=list(df['wav_id'])
for a in id2:
    audio_id.append(a)

for x in ll2:
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
        one_hot.append([0,0,0,0,0,1])

print('audio size'+str(len(audio_id)))
print('onehot size'+str(len(one_hot)))
one_hot=np.array(one_hot)'''
#print(one_hot)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
model.train()
#print(model)
classifier1.to(device)

from transformers import AdamW

optimizer = AdamW(list(model.parameters())+list(classifier1.parameters()), lr=0.0000001)
#loss_fn = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(list(model.parameters())+list(classifier1.parameters()), lr=0.0001)
loss_function = nn.CrossEntropyLoss()
#loss_function2=nn.BCEWithLogitsLoss()

new_audio_id=[]
new_one_hot=[]
'''idx=0
for i in audio_id:
    
    try:
        audio, sr = librosa.load('emotiondata/'+str(i)+'.wav', sr=16000)
        if(len(audio)<max_length):
            new_audio_id.append(i)
            new_one_hot.append(one_hot[idx])
    except FileNotFoundError:
        print("fucking files")

    if(idx%100==0):   
        print(str(idx)+" "+i)

    idx=idx+1

            #5fbe0c95576e9378b67ad368,아유 그래 음식물 쓰레기는 빨리빨리 버려야지.,disgust
    # 파일을 찾을 수 없는 경우 빈 오디오 데이터와 샘플링 주파수를 반환하거나 원하는 처리를 수행할 수 있습니다.
idx=0'''
"""with open('newaudiodata20.txt', 'w+') as file:
    for i in audio_id:
        file.write(i)
        file.write(":")
        file.write(str(new_one_hot[idx]))
        file.write('\n')
        idx=idx+1"""

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
print(len(new_one_hot))
print(len(new_audio_id)) 
print(new_one_hot)

from sklearn.model_selection import train_test_split

audio_id_train, audio_id_val, one_hot_train, one_hot_val = train_test_split(new_audio_id, new_one_hot, test_size=0.05, random_state=42)  # 여기서는 20%를 검증 세트로 사용

# 학습 및 검증 데이터셋 생성
train_dataset = AudioDataset(audio_id_train, one_hot_train)
val_dataset = AudioDataset(audio_id_val, one_hot_val)

# DataLoader 설정
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

#dataset = AudioDataset(new_audio_id, new_one_hot)
#dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

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

from transformers.optimization import get_cosine_schedule_with_warmup
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

for e in range(num_epochs):
    train_acc = 0.0
    test_acc = 0.0
    model.train()
    idx=0
    for batch in train_dataloader:
        optimizer.zero_grad()
        
        audio, labels = batch
        #print(len(audio))
        #print(labels)
        #print("_-----------------_")
        labels = labels.to(device)
        input_values = processor(audio, return_tensors="pt", padding=True, sampling_rate=16000).input_values
        input_values = input_values.to(device)
        model_input = input_values.view(1 * 32, -1).to(device)

        with torch.no_grad():
            features = model(model_input).last_hidden_state

        #logits = classifier1(features[:, 0, :])
        out = classifier1(features[:, 0, :])
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

        #logits = classifier1(features[:, 0, :])
        out = classifier1(features[:, 0, :])
        logits=out
        _, labels_indices = torch.max(labels, 1)
        test_acc += calc_accuracy(out, labels_indices)
        
        torch.save(model.state_dict(),'model_audioemotion_final.pt')
    print("epoch {} test acc {}".format(e+1, test_acc / (idx+1)))
torch.save(model.state_dict(),'model_audioemotion_final.pt')

'''
for epoch in range(5):
    for batch in dataloader:
        optimizer.zero_grad()

        audio, labels = batch
        labels = labels.to(device)

        # Process audio input
        input_values = processor(audio, return_tensors="pt", padding=True, sampling_rate=16000).input_values
        input_values = input_values.to(device)
        model_input = input_values.view(1 * 32, -1).to(device)
        soft=nn.Softmax(dim=1)
        with torch.no_grad():
            features = model(model_input).last_hidden_state

        #logits = classifier1(features[:, 0, :])
        out = classifier1(features[:, 0, :])
        logits=out
        #logits=soft(logits)
        # Calculate crossentropyloss
        #print(logits)
        #print(labels)
        _, labels_indices = torch.max(labels, 1)
        #print(logits)
        #print(labels_indices)
        #loss = loss_function(logits,labels_indices.to(device))
        if logits.shape!=(32,6) or labels.shape!=(32,6): continue
        #loss=loss_function2(logits,labels.float().to(device))
       # print(logits)
        #print(labels_indices)
        
        loss=loss_function(logits,labels_indices)
        loss.backward()
        optimizer.step()
        
        if idx%100==0:
            model.eval()  # 모델을 평가 모드로 설정
            total_val_loss = 0
            for batch in val_dataloader:
                audio, labels = batch
                labels = labels.to(device)
        
        # 오디오 처리 및 모델 예측
                input_values = processor(audio, return_tensors="pt", padding=True, sampling_rate=16000).input_values
                input_values = input_values.to(device)
                model_input = input_values.view(1 * 32, -1).to(device)
        
                with torch.no_grad():  # 기울기 계산을 비활성화
                    features = model(model_input).last_hidden_state
                    logits = classifier1(features[:, 0, :])
                    _, labels_indices = torch.max(labels, 1)
            
            # 검증 세트에 대한 손실 계산
                if logits.shape!=(32,6) or labels.shape!=(32,6): continue
                loss = loss_function(logits, labels_indices)
                total_val_loss += loss.item()

                avg_val_loss = total_val_loss / len(val_dataloader)
                print(f"Epoch {epoch}, Validation Loss: {avg_val_loss}")

        print(f"Epoch {epoch}, Loss: {loss.item()}")
        if loss.item()<minloss:
            print("SAVED")
            minloss=loss
            torch.save({'model_state_dict': model.state_dict(),
                'classifier1_state_dict': classifier1.state_dict()},
                f'model_epoch_emo3{epoch}.pth')    '''
    # Save model checkpoint

    

'''classifier1 = nn.Linear(model.config.hidden_size, 512).to(device)
batchnorm1 = nn.BatchNorm1d(512).to(device)
dropout1 = nn.Dropout(0.2).to(device)

classifier2 = nn.Linear(512,256).to(device)
batchnorm2 = nn.BatchNorm1d(256).to(device) 
dropout2 = nn.Dropout(0.2).to(device)

classifier3 = nn.Linear(256,6).to(device)'''   
                
'''for epoch in range(5):
    idx=idx+1
    for batch in dataloader:
        optimizer.zero_grad()
        
        audio, labels = batch
        #print(audio)
        input_values = processor(audio, return_tensors="pt", padding=True, sampling_rate=16000).input_values
        input_values = input_values.to(device)

        model_input = input_values.view(1 * 20, -1)
        with torch.no_grad():
            features = model(model_input).last_hidden_state
        
        logits = classifier(features[:, 0, :])

        if logits.shape!=(20,6) or labels.shape!=(20,6): continue
        loss = nn.BCEWithLogitsLoss(logits.squeeze(), labels.float().to(device))

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item()}")
    
    torch.save(model.state_dict(),'model'+str(idx))
        
    

torch.save(model.state_dict(),'model')'''