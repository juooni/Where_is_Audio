import requests
import json
from config.database import CLIENT_ID, CLIENT_SECRET, url

def translate(inputtext):
    text=inputtext
    headers = {
        'Content-Type': 'application/json',
        'X-Naver-Client-Id': CLIENT_ID,
        'X-Naver-Client-Secret': CLIENT_SECRET
    }
    data = {'source': 'ko', 'target': 'en', 'text': text}
    
    # post 방식으로 서버 쪽으로 요청
    response = requests.post(url, json.dumps(data), headers=headers) 
    
    # json() 후 key 값을 사용하여 원하는 텍스트 접근
    en_text = response.json()['message']['result']['translatedText']

    return en_text
