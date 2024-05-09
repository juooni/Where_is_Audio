from fastapi import APIRouter, Depends, UploadFile, File, WebSocket, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.websockets import WebSocketDisconnect
from datetime import datetime
from models.FinalAudioBooks import FinalAudioBooks
import pytz, os, shutil, json
from config.database import get_db
from services.Login_Service import get_current_user_authorization, oauth2_scheme
from services.AudioFiles_service import get_user_id_by_login_id
from services.EffectSounds_service import (
    upload_effect_sound_to_azure,
    combine_audio_files_with_effects,
    combine_final_audio_files,
    combine_whole_audio_with_effects
)
from models.EffectSounds import EffectSounds
from sqlalchemy.future import select

router = APIRouter()

korea_time_zone = pytz.timezone("Asia/Seoul")
created_at_kst = datetime.now(korea_time_zone)

@router.get("/read")
async def get_all_effect_sounds(db: AsyncSession = Depends(get_db)):
    async with db as session:
        query = select(EffectSounds)  # EffectSounds 모델의 모든 데이터를 조회하는 쿼리
        result = await session.execute(query)
        effect_sounds = result.scalars().all()

        # 조회된 데이터를 JSON 형식으로 변환
        effects_data = [
            {
                "effect_sound_id": effect_sound.effect_sound_id,
                "result_id": effect_sound.result_id,
                "Effect_Name": effect_sound.Effect_Name,
                "EffectFilePath": effect_sound.EffectFilePath,
                "EffectFileLength": effect_sound.EffectFileLength,
                "Upload_Date": effect_sound.Upload_Date.isoformat(),
            }
            for effect_sound in effect_sounds
        ]

        # 변환된 데이터를 JSONResponse 객체로 반환
        return JSONResponse(status_code=200, content={"effects": effects_data})

# 효과음 업로드
@router.post("/upload")
async def upload_effect_sound(file: UploadFile = File(...), db: AsyncSession = Depends(get_db)):
    
    # './tmp/' 디렉터리 존재 확인 및 생성
    tmp_directory = './tmp'
    if not os.path.exists(tmp_directory):
        os.makedirs(tmp_directory, exist_ok=True)

    # 파일 저장 경로 설정
    file_location = os.path.join(tmp_directory, file.filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Azure Blob Storage에 업로드 및 데이터베이스에 저장
    effect_sound = await upload_effect_sound_to_azure(file.filename, file_location, db)
    
    # 모든 처리가 완료된 후 디렉토리 삭제
    if os.path.exists(tmp_directory):
        shutil.rmtree(tmp_directory)
        
    effects_data = {
        "effect_sound_id": effect_sound.effect_sound_id,
        "result_id": effect_sound.result_id,
        "Effect_Name": effect_sound.Effect_Name,
        "EffectFilePath": effect_sound.EffectFilePath,
        "EffectFileLength": effect_sound.EffectFileLength,
        "Upload_Date": effect_sound.Upload_Date.isoformat(),  # datetime 객체 ISO 포맷으로 변환
    }
    
    # 생성된 effectss_data로 JSONResponse 객체를 생성합니다.
    return JSONResponse(status_code=200, content={"effects": effects_data})

# 효과음 + 오디오 분할 파일 합체
@router.post("/finalize/{audio_id}")
async def finalize_audio(request: Request, audio_id: int, token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)):
    try:
        login_id = await get_current_user_authorization(request, token)
        user_id = await get_user_id_by_login_id(db, login_id)
        if user_id is None:
            raise HTTPException(status_code=404, detail="User not found")
    
        # 최종 오디오 파일 생성 및 Blob Storage URL 반환
        blob_url, final_audio_filename, final_length = await combine_final_audio_files(db, audio_id)
        
        # FinalAudioBook 인스턴스 생성 및 데이터베이스에 저장
        final_audio_book = FinalAudioBooks(
            user_id=user_id,  # 예시 user_id, 실제 사용자 ID로 대체 필요
            audio_id=audio_id,
            Final_File_Name=final_audio_filename,
            FinalFilePath=blob_url,
            Final_File_Length=final_length,
            Creation_Date=created_at_kst
        )
        
        db.add(final_audio_book)
        await db.commit()
        return {"message": "Audio combined successfully", "audio_path": blob_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/ws-play")
async def websocket_endpoint(websocket: WebSocket, db: AsyncSession = Depends(get_db)):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            data_dict = json.loads(data)

            result_id = data_dict.get("result_id")
            effect_sound_id = data_dict.get("effect_sound_id", None)  # 효과음 적용 여부

            # 오디오 데이터 바이트를 받아오는 로직
            audio_data_bytes, final_directory = await combine_audio_files_with_effects(db, result_id, effect_sound_id)
            
            if audio_data_bytes:
                await websocket.send_bytes(audio_data_bytes)
            else:
                await websocket.send_text("Error: Unable to process audio")
                
            # # 업로드가 완료된 후 로컬의 임시 파일과 디렉토리 삭제
            shutil.rmtree(final_directory, ignore_errors=True)
    
    except WebSocketDisconnect:
        print("Client disconnected")
        
@router.websocket("/ws-whole-play")
async def websocket_whole_audio_endpoint(websocket: WebSocket, db: AsyncSession = Depends(get_db)):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            data_dict = json.loads(data)

            audio_id = data_dict.get("audio_id")
            audio_data_bytes = await combine_whole_audio_with_effects(db, audio_id)
            
            if audio_data_bytes:
                await websocket.send_bytes(audio_data_bytes)
            else:
                await websocket.send_text("Error: Unable to process audio")
    except WebSocketDisconnect:
        print("Client disconnected")