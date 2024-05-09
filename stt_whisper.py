import whisper
import os
import pandas as pd

model = whisper.load_model("large")
model.to('cuda:0')

lst=os.listdir('target_test')
idx=0
f= open("after_stt"+str(idx)+".txt","w+")
for wavs in lst:
    print("file{idx} start",idx)
    result = model.transcribe('target_test/'+wavs,language="korean")
    f.write(str(wavs))
    f.write('|')
    f.write(result['text'])
    f.write('|')
    f.write('X')
    f.write('|')
    f.write('\n')
    idx=idx+1