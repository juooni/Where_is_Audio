import re
import pandas as pd
from konlpy.tag import Kkma

def special_token(text):
    kkma = Kkma()
    slash_text = text.replace(' ', '/')
    split_slash_text = text.replace(' ', '/ ').split(' ')
    split_text = text.split()

    pre_idx = 0
    idx = 0
    fin_idx = 0
    try:
      pos = kkma.pos(slash_text)
    except:
      print(text)
      return ' '
    #print(pos)
    fin = ['' for i in range(slash_text.count('/')+1)]

    for w, p in pos:
        if w=='/':
            #print(w, p)
            for i in range(pre_idx, idx):
                fin[fin_idx] += (pos[i][0]+pos[i][1])
            fin_idx += 1
            pre_idx = idx+1
        idx += 1

    for i in pos[pre_idx:]:
        fin[-1] += (i[0]+i[1])

    tagged_text = []
    nng_tags = ['JKS','JKC', 'JKO', 'JKB']  # 목적어, 보어, 부사어
    idx = 0
    for t, p in zip(split_text,fin):
        tmp = t
        if 'VA' in p:
            tmp =  '[V]'+t+'[V]'
        elif 'VV' in p:
            tmp =  '[V]'+t+'[V]'
        elif 'NN' in p:
            for i in nng_tags:
                if i in p:
                    tmp = f'[{i}]'+t+f'[{i}]'

        if 'MAG' in p:
            tmp =  '[MAG]'+t+'[MAG]'
        if 'XR' in p:
            tmp =  '[MAG]'+t+'[MAG]'

        tagged_text.append(tmp)

    final_text = ' '.join(tagged_text)

    return final_text

'''
drop_list = []
df = pd.read_csv('checked_aug_data.csv')
for i in range(len(df)):
  tmp = df.iloc[i]['text']
  tmp = special_token(tmp)
  print(tmp)
  df.iloc[i]['text'] = tmp
df.to_csv('add_token_data.csv', index=False)
'''