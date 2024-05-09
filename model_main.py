# Import necessary libraries
from pydub import AudioSegment
import os
import librosa
import soundfile as sf
import numpy as np
import argparse
#from bgmgen import BGM
from KoKoWa import KoKoWa
import warnings
warnings.filterwarnings('ignore')
import torch
#from llama.llama_few_show import get_prompt
from utils.translater import Google_Translator
#from utils.effectgen import make_effect_files
from utils.bgmgen import generate_bgm
import pandas as pd
import os
import sys

class AudioProcessor:
    def __init__(self, id, wav_path, output_dir):
        self.id = id
        self.wav_path = wav_path
        self.output_dir = output_dir

    def convert_audio_type(self):
        audSeg = AudioSegment.from_mp3(self.wav_path)
        audSeg.export(self.wav_path, format="wav")


    def _split_array(self, arr, c, n):
        result_arr = []
        result_idx = [0,]
        current_group = [arr[0]]
        count = 1
        idx_counter = 1 

        for i in range(1, len(arr)):
            idx_counter += 1
            current_group.append(arr[i])
            if arr[i] <= c:
                if arr[i-1]<=c:
                    count += 1
                else:
                    count = 1
            if count>=n and len(arr)>idx_counter+1 and arr[i+1]>c:
                result_idx.append(idx_counter)
                result_arr.append(current_group.copy())
                current_group.clear()
                count = 1

        if len(current_group)!=0:
            result_arr.append(current_group.copy())

        result_idx.append(len(arr))

        return result_arr, result_idx


    def _split_and_save(self, criteria, num):
        """
        - wav_path : 로드할 WAV 파일의 경로
        - output_dir : WAV 파일들을 저장할 디렉토리 경로
        - segment_length : 분할할 세그먼트의 길이 (초 단위, 기본값은 30초)
        """
        # 출력 디렉토리가 존재하지 않으면 생성
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # wav 파일 로드
        y, sr = librosa.load(self.wav_path, sr=16000)
        audio_total_len=len(y)
        gap = []
        for i in range(1, len(y)):
            a = abs(y[i-1]-y[i])
            gap.append(a)

        gap = np.array(gap)

        arr, idx = self._split_array(gap, criteria, num)

        # 지정된 길이 segment_samples로 분할하여 WAV 파일로 저장
        t = 0
        h = 0
        for i in range(len(idx)-1):
            n = i%10
            n2 = i%100
            start_sample = idx[i]
            end_sample = idx[i+1]
            output_path = f"{self.output_dir}/{self.id}_{h}{t}{n}-1.wav"

            # 여기서는 segment로 분할된 오디오 데이터를 WAV 파일로 저장
            sf.write(output_path, y[start_sample:end_sample], sr)
            if n==9:
                t+=1
            if n2==99:
                h+=1
                t=0


    def _critria_mean(self, y):
        filtered_values = y[y <= 0.01]
        average_value = np.mean(filtered_values)
        return average_value


    def _critria_med(self, y):
        filtered_values = y[y <= 0.01]
        max_value = np.median(filtered_values)
        return max_value

    def _all_duration(self, arr, c):
        res_dur = []
        current_group = []
        count = 1
        s = 0

        for i in range(len(arr)):
            if arr[i] <= c:
                if s==0:
                    s=1
                current_group.append(arr[i])
            if s==1 and arr[i]>c:
                res_dur.append(len(current_group))
                current_group.clear()
                s=0

        if len(current_group)!=0:
            res_dur.append(len(current_group))

        res_dur = np.array(res_dur)
        filtered_values = res_dur[res_dur > 10000]
        return filtered_values


    def process_audio(self):
        y, sr = librosa.load(self.wav_path, sr=16000)
        audio_total_len=len(y)
        tmp = [abs(y[i-1] - y[i]) for i in range(1, len(y))]
        y_a = np.array(tmp)

        # Calculate criteria values
        medc = self._critria_med(y_a)
        meanc = self._critria_mean(y_a)

        # Split and save the audio
        self._split_and_save(meanc, 5000)
        return audio_total_len
    

class SuppressMessages:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


def STT(audiopath):
    import whisper
    import os
    import pandas as pd
    import csv

    model = whisper.load_model("large")
    model.to('cuda:0')
    result = model.transcribe(audiopath,language="korean")

    return result['text']


def modelmain() :
    import os

    datapath='data/split_audio_files/'
    lst=os.listdir(datapath)    

    emodict={0:'anger',
        1:'sad',
        2:'disgust',
        3:'fear',
        4:'neutral',
        5:'happy or surprise'}
    
    audio_list=lst #CSV

    text_list=[]
    KB1vec=[]
    KB2vec=[]
    W2Vvec=[]
    emovec=[]
    emolabel=[]
    effect_or_not=[]
    effect_or_not_O_X=[]
    prompt=[]

    for wavs in lst:
        text=STT(datapath+wavs)
        koko=KoKoWa()
        text_list.append(text)
        kb1=koko.get_value_from_KB1(text)
        kb2=koko.get_value_from_KB2(text)
        w2v=koko.get_value_from_W2V(datapath+wavs)
        KB1vec.append(kb1)
        KB2vec.append(kb2)
        W2Vvec.append(w2v.cpu().numpy())

    KB1vec=np.array(KB1vec)
    KB2vec=np.array(KB2vec)
    W2Vvec=np.array(W2Vvec)

    emovec=[]
    for i in range(len(KB1vec)):
        emovec.append(np.array(KB1vec[i]*0.99+W2Vvec[i]*0.01))

    emovec=np.array(emovec)
    emovec=emovec.reshape(len(KB1vec),6)
    print('emovec')
    emoword=[]
    for e in emovec:
        emoword.append(emodict[np.argmax(e)])
    print('emoword'+str(emoword))
    gradient_emovec= np.gradient(emovec,axis=0)
    print('=====================')
    KB1vec=np.squeeze(KB1vec)
    KB2vec=np.squeeze(KB2vec)
    W2Vvec=np.squeeze(W2Vvec)
    print(KB1vec)
    print(KB2vec)
    print(W2Vvec)
    print(emovec)
    print(emoword)
    print(gradient_emovec)
    norm_gradient_emovec = np.linalg.norm(gradient_emovec, axis=1)
    idx=0

    for i in KB2vec:
        effect_or_not.append(i[1]+0.02*float(norm_gradient_emovec[idx]))    
        idx=idx+1
    print("###EFFEVTORNOT###")
    print(effect_or_not)
    for i in effect_or_not:
        if(i>0.8):
            effect_or_not_O_X.append('O')
        else:
            effect_or_not_O_X.append('X')
    
    trans=Google_Translator()
    from utils.papago import translate
    with open('./llama/prompttext.txt','w+') as file:
        for tex,ox in zip(text_list,effect_or_not_O_X):
            if(ox=='O'):
                result = translate(tex)
                file.write(result)
                #result['tgt_text']
                file.write('\n')
    
    sum_of_emovec=np.sum(emovec,axis=0)
    sum_of_emovec[4]=0
    max_emo_idx=np.argmax(sum_of_emovec)
    emotion=emodict[max_emo_idx]

    with_blank_prompt=[]

    idx=0
    for ox in effect_or_not_O_X:
        if(ox=='X'):
            with_blank_prompt.append(None)
        elif(ox=='O'):
            with_blank_prompt.append(None)

    data = {'file_name':audio_list, 'text':text_list, 'ox':effect_or_not_O_X, 'prompt':with_blank_prompt}
    df = pd.DataFrame(data)
    df.to_csv('data/final.csv', index=False)
    generate_bgm(emotion,'data/bgm_files',10)


modelmain()
