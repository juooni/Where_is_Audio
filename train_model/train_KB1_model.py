import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel
import pandas as pd
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from sklearn.model_selection import train_test_split
device = torch.device("cuda:0")

tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')

'''df1 = pd.read_csv('5차년도.csv')
df2 = pd.read_csv('5차년도_2차.csv')'''

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=6,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)
    
def get_data():
    df2 = pd.read_csv("5차년도.csv",encoding='cp949')
    df3 = pd.read_csv("5차년도_2차.csv",encoding='cp949')
    text=list(df2['발화문'])
    t=list(df3['발화문'])
    for x in t:
        text.append(x)

    one_hot=[]
    labels=list(df2['상황'])
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
        if(x=='happiness' or x=='surprise'):
            one_hot.append([0,0,0,0,0,1])

    labels2=list(df3['상황'])
    for x in labels2:
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
    return one_hot,text

def predict(sentence, model, tokenizer, vocab, max_len=64, device='cuda:0'):
    model.eval()  
    transform = nlp.data.BERTSentenceTransform(
        tokenizer, max_seq_length=max_len, vocab=vocab, pad=True, pair=False)
    
    tokenized_text = transform([sentence])
    token_ids, valid_length, segment_ids = tokenized_text
    
    token_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(device)
    segment_ids = torch.tensor(segment_ids, dtype=torch.long).unsqueeze(0).to(device)
    valid_length = torch.tensor(valid_length, dtype=torch.long).unsqueeze(0).to(device)
    
    # 예측 수행
    with torch.no_grad():
        out = model(token_ids, valid_length, segment_ids)
        logits = out.detach().cpu().numpy()
        probs = F.softmax(torch.tensor(logits), dim=1).numpy()
        pred_class_id = np.argmax(logits, axis=1)
    
    return pred_class_id[0], probs[0, pred_class_id[0]],logits,probs

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, vocab, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, vocab=vocab, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc



data_list=[]
one_hot,text=get_data()
data_list = []
one_hot, text = get_data()
for idx, t in enumerate(text):
    ls = [t]
    # NumPy 배열을 PyTorch 텐서로 변환
    tensor_one_hot = torch.tensor(one_hot[idx])
    # 최댓값의 인덱스를 얻습니다 (여기서는 one-hot 벡터이므로, 레이블에 해당)
    _, labels_indices = torch.max(tensor_one_hot, 0)
    ls.append(labels_indices.item())  # tensor를 Python 스칼라 값으로 변환
    data_list.append(ls)


dataset_train, dataset_test = train_test_split(data_list, test_size=0.05, random_state=42)

# Setting parameters
max_len = 64
batch_size = 64
warmup_ratio = 0.1
num_epochs = 5
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5

tok = tokenizer.tokenize

data_train = BERTDataset(dataset_train, 0, 1, tok, vocab, max_len, True, False)
data_test = BERTDataset(dataset_test, 0, 1, tok, vocab, max_len, True, False)
train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=5)
test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=5)


model = BERTClassifier(bertmodel,  dr_rate=0.5).to(device)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

t_total = len(train_dataloader) * num_epochs
warmup_step = int(t_total * warmup_ratio)

scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

for e in range(num_epochs):
    train_acc = 0.0
    test_acc = 0.0
    model.train()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(train_dataloader)):
        optimizer.zero_grad()
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        loss = loss_fn(out, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        train_acc += calc_accuracy(out, label)
        if batch_id % log_interval == 0:
            print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
    print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))
    model.eval()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(test_dataloader)):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        test_acc += calc_accuracy(out, label)
    print("epoch {} test acc {}".format(e+1, test_acc / (batch_id+1)))
torch.save(model.state_dict(),'model_text_emo_final.pt')

