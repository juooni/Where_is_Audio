# Where_is_Audio 🔊
오디오북에 효과음을 자동으로 생성 및 삽입해주는 인공지능 모델입니다. (SKT FlyAI)

## 🤔 Introduction
- **SKT FLY AI Challenger Project - 오디오가 오디오(Where is Audio)**
- **Goal**
    - 오디오북의 효과음 및 배경음악 편집을 간편화하는 프로그램 제작
- **기능**
    - **효과음 삽입 기능**: 효과음 삽입 위치를 찾아 효과음을 생성 및 삽입 
    - **배경음악 삽입 기능**: 오디오북의 전체적인 분위기를 탐지하여, 어울리는 배경음악을 생성 및 삽입 
- **input/output**
    - **input** : Audio File of Audio book
    - **output** : Edited Audio File (adding Effect sound & Background music)



## ⚒️ Model Architecture
### ✨ Model Diagram 
<div align="center"> <img src ="https://github.com/juooni/Where_is_Audio/assets/125336278/ab2d79b9-7faf-4c15-ae5d-e37fffc9e5fd" width = 512> </div>


### ✨ Effect Sound 
- **Effect sound insertion binary classification** | 효과음 삽입 여부 이진분류
    > **🔎 오디오북에 효과음이 들어갈만한 위치를 찾는 작업**
    > - **Self Made Dataset** 
    >   - 특정 문장에 대해 효과음이 날 만한 문장인지 여부를 표시한 데이터셋 제작 및 활용
    > - **KoBERT 효과음 삽입 여부 이진 분류 모델**
    >   - 문장 성분 분석 $\rightarrow$ POS 태깅을 사용하여 주어, 목적어, 서술어, 부사어에 special token 추가
    >   - token 추가로 0.5에서 0.9까지 성능 향상
    > - **Sentiment Analysis based on Text and Audio** 
    >   -  KoBERT와 Wav2Vec을 활용한 감정분석 결과를 효과음 이진분류의 가중치로 활용
    >   - 텍스트의 맥락 반영 효과  

- **Effect sound generation** | 효과음 생성
    > **🔎 효과음이 들어갈 만한 문장에 적절한 효과음을 생성하는 작업**
    > - **Translation Kor to Eng by Papago**
    > - **prompt generation by few-shot learned LLaMa2**
    >   - few-shot learning prompt structure : $\text{"{text}}$=>$\text{{prompt};}"$
    > - **effect sound generation by AudioGEN**

### ✨ Background Music
- **Sentiment Analysis based on Text and Audio** 
    > Wav2Vec과 KoBERT를 활용한 텍스트, 오디오 기반 감정 분석 작업
- **Background Music Generation by MusicGEN**
    > MusicGEN에 감정 분석 결과를 프롬프트로 주어 어울리는 배경음악 생성


## ⚙️ System Architecture
### ✨ Backend  

### ✨ Frontend 
<div align='center'> <img src="https://github.com/juooni/Where_is_Audio/assets/125336278/c555715b-69d5-4b8c-9f68-257030f96b92" width=512> </div>


## 🗺️ Code Description
- [train_mode](Where_is_Audio/train_model)
    - Model Train Codes
- [utils](Where_is_Audio/utils)
    - Audio, Text Preprocessing 
    - AudioGEN, MusicGEN Executable 
    - Whisper STT 
    - Translater
    - Model Classes
- [llama_code](Where_is_Audio/llama_code)
    - LLaMa few-shot learning and Inference code
- Backend : [models](Where_is_Audio/models), [routers](Where_is_Audio/routers), [server](Where_is_Audio/server), [services](Where_is_Audio/services)


## 💻 stack
- **model** <br>
    ![Pytorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
- **backend** <br>
    ![FastAPI](https://img.shields.io/badge/fastapi-109989?style=for-the-badge&logo=FASTAPI&logoColor=white)
    ![MySQL](https://img.shields.io/badge/MySQL-005C84?style=for-the-badge&logo=mysql&logoColor=white)
    ![Azure](https://img.shields.io/badge/Azure_DevOps-0078D7?style=for-the-badge&logo=azure-devops&logoColor=white)
- **frontend** <br>
    ![React](https://img.shields.io/badge/semantic%20ui%20react-35BDB2?style=for-the-badge&logo=semanticuireact&logoColor=white)
    ![javascript](https://img.shields.io/badge/JavaScript-323330?style=for-the-badge&logo=javascript&logoColor=F7DF1E)


## 🖤 Team Members
- **MODEL**: 박종현, 임주원
- **BACKEND** : 김필환
- **FRONTEND** : 김태란, 윤영채
- **PRODUCT MANAGEMENT** : 김태란, 김필환, 박종현, 윤영채, 임주원, 백은서
