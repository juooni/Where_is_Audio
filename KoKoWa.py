import numpy as np
import pandas as pd
from tqdm import tqdm, tqdm_notebook
import soundfile as sf
import librosa

import torch
from torch import nn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp

from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score

from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel
from utils.emotiontext import BERTClassifier
from utils.emotiontext import predict
from utils.adding_token import special_token

import warnings
warnings.filterwarnings('ignore')


class KoKoWa:
    def __init__(self):
        self.a=1
        device = torch.device("cuda:0")

        # KoBERT 감정 분석 모델 (KB1)
        self.tokenizer1 = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
        self.vocab1 = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')   
        bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)    
        
        self.KB1_model = BERTClassifier(bertmodel,  dr_rate=0.5).to(device)
        self.KB1_model.load_state_dict(torch.load('model_text_emo_final.pt'))


        # KoBERT 효과음 삽입여부 이진 분류 모델 (KB2)
        self.tokenizer2 = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
        bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
        self.vocab2 = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')
        
        tags = ['[JKS]','[JKC]', '[JKO]', '[JKB]', '[MAG]', '[V]']  # special token 추가
        self.tokenizer2.add_special_tokens({'additional_special_tokens': tags})
        embedding_layer = bertmodel.embeddings.word_embeddings

        old_num_tokens, old_embedding_dim = embedding_layer.weight.shape
        num_new_tokens = len(tags)
        # Creating new embedding layer with more entries
        new_embeddings = nn.Embedding(
                old_num_tokens + num_new_tokens, old_embedding_dim
        )
        # Setting device and type accordingly
        new_embeddings.to(
            embedding_layer.weight.device,
            dtype=embedding_layer.weight.dtype,
        )
        # Copying the old entries
        new_embeddings.weight.data[:old_num_tokens, :] = embedding_layer.weight.data[
            :old_num_tokens, :
        ]
        bertmodel.embeddings.word_embeddings = new_embeddings
        self.KB2_model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
        self.KB2_model.load_state_dict(torch.load('modelnew2.pt'), strict=False)


        # Wave2vec 오디오 기반 감정 분석 모델 (W2V)
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
        self.W2V_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h")
        self.relu=nn.ReLU()

        self.classifier1 = nn.Linear(self.W2V_model.config.hidden_size, 512).to(device)
        self.batchnorm1 = nn.BatchNorm1d(512).to(device)
        self.dropout1 = nn.Dropout(0.2).to(device)

        self.classifier2 = nn.Linear(512,256).to(device)
        self.batchnorm2 = nn.BatchNorm1d(256).to(device) 
        self.dropout2 = nn.Dropout(0.2).to(device)

        self.classifier3 = nn.Linear(256,6).to(device)
        self.W2V_model.to(device)
        checkpoint=torch.load('model_epoch_second(avail).pth')

        self.W2V_model.load_state_dict(checkpoint['model_state_dict'])
        self.classifier1.load_state_dict(checkpoint['classifier1_state_dict'])
        self.classifier2.load_state_dict(checkpoint['classifier2_state_dict'])
        self.batchnorm1.load_state_dict(checkpoint['batchnorm1_state_dict'])
        self.batchnorm2.load_state_dict(checkpoint['batchnorm2_state_dict'])
        self.classifier3.load_state_dict(checkpoint['classifier3_state_dict'])


    def get_value_from_KB1(self,text):
        device = torch.device("cuda:0")
        self.KB1_model.eval()
        emodict={0:'anger',
                1:'sad',
                2:'disgust',
                3:'fear',
                4:'neutral',
                5:'happy or surprise'}
        pred_class_id, pred_prob,logits,probs = predict(text, self.KB1_model, self.tokenizer1.tokenize, self.vocab1, max_len=64, device='cuda:0')
        #print(f"Predicted class ID: {iddict[pred_class_id]}, Probability: {pred_prob:.4f}")
        #print(probs)
        return probs

    def get_value_from_KB2(self,text): 
        device = torch.device("cuda:0")
        self.KB2_model.load_state_dict(torch.load('modelnew2.pt'), strict=False)
        self.KB2_model.eval()
        text = special_token(text)
        pred_class_id, pred_prob,logits,probs = predict(text, self.KB2_model, self.tokenizer2.tokenize, self.vocab2, max_len=64, device='cuda:0')
        #print(f"Predicted class ID: {iddict[pred_class_id]}, Probability: {pred_prob:.4f}")
        #print(probs)
        return probs

    def get_value_from_W2V(self, audiopath):
        device=torch.device('cuda')
        audio,sr= librosa.load(audiopath, sr=16000)
        self.W2V_model.eval()
        self.W2V_model.to(device)
        max_length=16000*20
        current_length = len(audio)

        if current_length <= max_length:
                    padding_length = max_length - current_length
                    audio = np.pad(audio, (0, padding_length), mode='constant')
        if current_length > max_length:
                    audio = audio[:max_length] 

        input_values = self.processor(audio, return_tensors="pt", padding=True, sampling_rate=16000).input_values
        input_values = input_values.to(device)
        model_input = input_values.view(1, -1).to(device)
        soft=nn.Softmax(dim=1)

        with torch.no_grad():
            features = self.W2V_model(model_input).last_hidden_state

            out = self.classifier1(features[:, 0, :])
            #out = self.batchnorm1(out)
            out = self.relu(out)
            out = self.dropout1(out)
            out = self.classifier2(out)
            #out = self.batchnorm2(out)
            out = self.relu(out)
            out = self.dropout2(out)
            out = self.classifier3(out)
            logits=soft(out)

            return logits

