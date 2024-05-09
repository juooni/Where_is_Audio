import torchaudio
import torch
from audiocraft.models import AudioGen
from audiocraft.data.audio import audio_write
import pandas as pd

#    make_effect_files('data/after_stt.csv','data/effect_files')
def make_effect_files(input_prompt,output_path):
        model = AudioGen.get_pretrained('facebook/audiogen-medium')
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
        model.device=device
        descriptions=[]
        descriptions.append(input_prompt)

        model.set_generation_params(duration=3,temperature=1)
        wav = model.generate(descriptions)
        for idx, one_wav in enumerate(wav):
            audio_write(output_path+'/'+input_prompt+'_1', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)

        model.set_generation_params(duration=3,temperature=0.9)
        wav = model.generate(descriptions)
        for idx, one_wav in enumerate(wav):
            audio_write(output_path+'/'+input_prompt+'_2', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)


        model.set_generation_params(duration=3,temperature=0.8)
        wav = model.generate(descriptions)
        for idx, one_wav in enumerate(wav):
            audio_write(output_path+'/'+input_prompt+'_3', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)

        model.set_generation_params(duration=3,temperature=0.7)
        wav = model.generate(descriptions)
        for idx, one_wav in enumerate(wav):
            audio_write(output_path+'/'+input_prompt+'_4', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)

        model.set_generation_params(duration=3,temperature=0.6)
        wav = model.generate(descriptions)
        for idx, one_wav in enumerate(wav):
            audio_write(output_path+'/'+input_prompt+'_5', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
        
        model.set_generation_params(duration=3,temperature=0.5)
        wav = model.generate(descriptions)
        for idx, one_wav in enumerate(wav):
            audio_write(output_path+'/'+input_prompt+'_6', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)

#make_effect_files('sound of knocking','data/effect_files')
import pandas as pd

promptt=[]
with open('./llama/prompttext_result.txt','r') as file:
        lines=file.readlines()    
        for line in lines:
            line_strip=line.strip()
            promptt.append(line_strip)
            #print(line)
            make_effect_files(line_strip,'data/effect_files/')

        df=pd.read_csv('data/final.csv')
        idx=0
        correct_col=[]
        for i in range(len(df)):

            if df.iloc[i]['ox'] == 'O':
                #print(df.iloc[i]['ox'], df.iloc[i]['prompt'])
                #df.iloc[i]['prompt']=promptt[idx] 
                correct_col.append(promptt[idx])
                idx=idx+1

            elif df.iloc[i]['ox'] == 'X':
                correct_col.append(None)
            
        df['prompt']=correct_col
                
        df.to_csv('data/final.csv', index=False)
        
'''  # generate 5 seconds.
descriptions = ['knocking to door']
wav1= model.generate(descriptions)  # generates 3 samples.
audio_write('effect_015_016-1', wav1.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
model.set_generation_params(duration=3,temperature=0.7)  # generate 5 seconds.
descriptions = ['knocking to door']
wav2= model.generate(descriptions)  # generates 3 samples.
audio_write('effect_015_016-2', wav2.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
model.set_generation_params(duration=3,temperature=0.4)  # generate 5 seconds.
descriptions = ['knocking to door']
wav3 = model.generate(descriptions)  # generates 3 samples.
audio_write('effect_015_016-3', wav3.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
''''''
descriptions = ['knocking sound']
model.set_generation_params(duration=3,temperature=1)
wav = model.generate(descriptions)  # generates 3 samples.
for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write('effect_015_016-1', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)

model.set_generation_params(duration=3,temperature=0.7)
wav = model.generate(descriptions)  # generates 3 samples.
for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write('effect_015_016-2', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)

model.set_generation_params(duration=3,temperature=0.4)
wav = model.generate(descriptions)  # generates 3 samples.
for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write('effect_015_016-3', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)'''