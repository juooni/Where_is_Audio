from models.AudioFiles import AudioResponse, AudioDelete
from models.AudioFiles import AudioResponse
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import APIRouter, Depends, HTTPException, UploadFile, Request
from fastapi.responses import JSONResponse
from config.database import get_db
from services.Login_Service import get_current_user_authorization, oauth2_scheme
from services.AudioFiles_service import (
    uploadtoazure, 
    downloadfromazure, 
    get_user_id_by_login_id, 
    get_audiofile_by_id, 
    delete_audiofile,
)

router = APIRouter()

# audio_id로 파일 읽기
@router.get("/read/{audio_id}", response_model=AudioResponse)
async def read_audiofile(audio_id: int, db: AsyncSession = Depends(get_db)):
    return await get_audiofile_by_id(db, audio_id)

# audio_id로 파일 삭제
@router.delete("/delete/{audio_id}", response_model=AudioDelete)
async def remove_audiofile(audio_id: int, db: AsyncSession = Depends(get_db)):
    return await delete_audiofile(db, audio_id)

@router.post("/upload")
async def create_upload_file(request: Request, file: UploadFile, token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)):
    login_id = await get_current_user_authorization(request, token)
    user_id = await get_user_id_by_login_id(db, login_id)
    if user_id is None:
        raise HTTPException(status_code=404, detail="User not found")
    file_data = await file.read()

    # uploadtoazure 함수가 완료될 때까지 기다립니다.
    results = await uploadtoazure(file.filename, file.content_type, file_data, user_id, db)
    
    # uploadtoazure 함수가 반환한 결과로부터 results_data를 생성합니다.
    results_data = [{
        "result_id": result.result_id,
        "audio_id": result.audio_id,
        "Index": result.Index,
        "Converted_Result": result.Converted_Result,
        "ResultFilePath": result.ResultFilePath,
        "ResultFileLength": result.ResultFileLength,
        "Converted_Date": result.Converted_Date.isoformat(),  # datetime 객체 ISO 포맷으로 변환
    } for result in results]

    # 생성된 results_data로 JSONResponse 객체를 생성합니다.
    return JSONResponse(status_code=200, content={"results": results_data})

# Azure Blob Storage에서 파일 다운로드   
@router.post("/download", response_model = AudioResponse)
async def download_and_save_file(request: Request, File_Name: str, token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)):
    login_id = await get_current_user_authorization(request, token)
    user_id = await get_user_id_by_login_id(db, login_id)    
    if user_id is None:
        raise HTTPException(status_code=404, detail="User not found")       
    try:
        db_audio_file = await downloadfromazure(user_id, File_Name, db)
        return db_audio_file
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))