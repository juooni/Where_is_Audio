from fastapi import FastAPI
from sqlalchemy import create_engine
from config.database import engine
from server.index import setup_cors
import uvicorn
from models.AudioFiles import Base as AudioFiles 
from models.EditHistory import Base as EditHistory 
from models.Results import Base as Result  
from models.users import Base as users  
from models.EffectSounds import Base as EffectSound
from models.FinalAudioBooks import Base as FinalAudioBook    
from models.BackGrounds import Base as BackGround
from routers import (
    AudioFiles_route,
    EditHistory_route,
    users_route,
    EffectSounds_route
)

# 초기 데이터베이스 연결
DatabaseURL = 'mysql+pymysql://root:1234@localhost:3308/test'
# DatabaseURL = 'mysql+pymysql://sexy:1234@localhost:3306/test'
engine = create_engine(DatabaseURL)

app = FastAPI()

# API 암호화
setup_cors(app)

# 원하는 데이터베이스 생성
FinalAudioBook.metadata.create_all(bind=engine)
AudioFiles.metadata.create_all(bind=engine)
EffectSound.metadata.create_all(bind=engine)
EditHistory.metadata.create_all(bind=engine)
Result.metadata.create_all(bind=engine)
users.metadata.create_all(bind=engine)
BackGround.metadata.create_all(bind=engine)

# Prefix는 엔드포인트를 정할 때 사용
app.include_router(users_route.router, prefix="/users", tags=["Users"])
app.include_router(AudioFiles_route.router, prefix="/files", tags=["Audio Files For Azure"])
app.include_router(EffectSounds_route.router, prefix="/effects", tags=["EffectSounds"])
app.include_router(EditHistory_route.router, prefix="/histories", tags=["Histories"])

@app.get('/')
def home():
    return {'msg' : 'Creating DB'}

if __name__ == "__main__":
    uvicorn.run("main:app", host = "127.0.0.1", port = 8000, reload = True)

