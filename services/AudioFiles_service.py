from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from azure.storage.blob import BlobServiceClient, PublicAccess
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from models.AudioFiles import AudioFile
from models.users import User
from models.Results import Result
from models.EffectSounds import EffectSounds
from models.BackGrounds import BackgroundMusic
import pytz, datetime, os, mimetypes, wave, contextlib, shutil, pandas as pd
from mutagen.mp3 import MP3
from typing import List
from services.audio_processing import AudioProcessor
from config.database import connect_str


korea_time_zone = pytz.timezone("Asia/Seoul")
created_at_kst = datetime.datetime.now(korea_time_zone)

# wav 파일 길이 정보 함수
def get_wav_length(wav_path):
    with contextlib.closing(wave.open(wav_path, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = round(frames / float(rate), 2)
        return duration
    
# 파일 이름과 사용자 ID로 오디오 파일 검색
async def get_audiofile_by_name(db: AsyncSession, user_id: int, File_Name: str):
    result = await db.execute(select(AudioFile).filter(AudioFile.user_id == user_id, AudioFile.File_Name == File_Name))
    return result.scalars().first()

# 분할 파일 데이터베이스에 저장
async def split_and_save_results(db : AsyncSession, audio_id: int, segments_info: List[str], segment_lengths: List[float]):
    if segments_info is None:
        raise ValueError("segments_info is None, which indicates no segments were processed or returned.")
    
    results = []

    for index, (segment_path, segment_length) in enumerate(zip(segments_info, segment_lengths)):
        # 여기에서 segment_length 값을 Result 객체에 저장
        file_name = os.path.basename(segment_path)

        result = Result(
            audio_id=audio_id,
            File_Name=file_name,
            Index=index + 1,
            Converted_Result="X",
            ResultFilePath=segment_path,
            ResultFileLength=segment_length,  # 세그먼트 길이 저장
            Converted_Date=datetime.datetime.now()
        )
        db.add(result)
        results.append(result)  # 결과 리스트에 추가
    await db.commit()
    
    # 결과 객체의 리스트를 반환
    return results

# 효과음 정보 데이터베이스에 넣는 함수
async def upload_effectsounds(db: AsyncSession):
    df = pd.read_csv('./data/final.csv')  # 경로 수정 필요
    filtered_df = df[df['ox'] == 'O']

    container_name = "effects"
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)

    # 컨테이너가 없으면 생성
    container_client = blob_service_client.get_container_client(container_name)
    try:
        container_client.create_container(public_access=PublicAccess.Container)
    except ResourceExistsError:
        pass  # 컨테이너가 이미 존재하면 무시

    for index, row in filtered_df.iterrows():
        file_name = row['file_name']
        query = select(Result).where(Result.File_Name == file_name)
        result = await db.execute(query)
        result_obj = result.scalars().first()

        if result_obj:
            # Prompt 값에 따라 3개의 파일 경로와 길이 생성 및 커밋
            for i in range(1, 4):  # 1부터 3까지 반복
                local_effect_sound_path = f"./data/effect_files/{row['prompt']}_{i}.wav"  # 로컬 파일 경로
                effect_file_length = get_wav_length(local_effect_sound_path)  # 파일 길이 계산

                # 파일을 Azure Blob Storage에 업로드
                blob_name = f"{row['prompt']}_{i}.wav"
                blob_client = container_client.get_blob_client(blob=blob_name)
                with open(local_effect_sound_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)
                blob_url = blob_client.url  # 업로드된 파일의 URL

                effect_sound = EffectSounds(
                    result_id=result_obj.result_id,
                    Effect_Name=row['prompt'],
                    EffectFilePath=blob_url,  # Azure Blob URL
                    EffectFileLength=effect_file_length,
                    Upload_Date=created_at_kst
                )
                db.add(effect_sound)

    await db.commit()

# STT 문장 데이터베이스에 업데이트 함수
async def update_stt(db: AsyncSession):
    df = pd.read_csv('./data/final.csv') # 경로 수정 필요
    
    for index, row in df.iterrows():
        file_name = row['file_name']
        text = row['text']  # STT 변환 결과 텍스트

        query = select(Result).where(Result.File_Name == file_name)
        result = await db.execute(query)
        result_obj = result.scalars().first()

        if result_obj:
            result_obj.Converted_Result = text
            db.add(result_obj)  # 변경된 객체를 세션에 추가합니다.

    await db.commit()   

# 효과음 테이블에 배경음악 넣는 테이블
async def upload_bgm(db: AsyncSession, audio_id : int):
    bgm_folder_path = "./data/bgm_files"
    container_name = "backgrounds"
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)

    # 컨테이너가 없으면 생성
    container_client = blob_service_client.get_container_client(container_name)
    try:
        container_client.create_container(public_access=PublicAccess.Container)
    except ResourceExistsError:
        pass  # 컨테이너가 이미 존재하면 무시

    # 폴더 내의 모든 파일을 탐색
    for file_name in os.listdir(bgm_folder_path):
        if file_name.endswith(".wav"):  # WAV 파일만 처리
            local_bgm_path = os.path.join(bgm_folder_path, file_name)

            # 파일 길이 계산
            bgm_file_length = get_wav_length(local_bgm_path)

            # 파일을 Azure Blob Storage에 업로드
            blob_client = container_client.get_blob_client(blob=file_name)
            with open(local_bgm_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
            blob_url = blob_client.url  # 업로드된 파일의 URL

            # 데이터베이스에 업로드 정보 저장
            background_music = BackgroundMusic(
                audio_id=audio_id,  # result_id 고정
                Music_Name=file_name.split('.')[0],  # 파일 이름에서 확장자 제거
                MusicFilePath=blob_url,  # Azure Blob URL
                File_Length=bgm_file_length,
                Upload_Date=created_at_kst
            )
            db.add(background_music)

    await db.commit()

# Azure Blob Storage로 분할 파일과 원본 파일 업로드
async def uploadtoazure(File_Name: str, content_type: str, file_data, user_id: int, db: AsyncSession):
    
    # tmp 루트 폴더 경로
    temp_file_path2 = f"./tmp"
    
    if not os.path.exists(temp_file_path2):
        os.makedirs(temp_file_path2)
    
    
    # 로컬 경로에 파일 임시 저장
    temp_file_path = f"./tmp/{File_Name}"
    
    with open(temp_file_path, 'wb') as temp_file:
        temp_file.write(file_data)

    # Azure 연결 문자열 설정
    container_name = f"metadata{user_id}"
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    blob_client = None
    
    try:
        file_length = get_wav_length(temp_file_path)
        
        # 컨테이너 생성 및 공개 접근 수준 설정 (Container 또는 Blob)
        try:
            blob_service_client.create_container(container_name, public_access=PublicAccess.Container)
        except ResourceExistsError:
            pass
        
        blob_client = blob_service_client.get_container_client(container_name).get_blob_client(File_Name)    

        blob_client.upload_blob(file_data, overwrite=True)
        blob_url = blob_client.url

        audio_file = AudioFile(
            user_id=user_id,
            File_Name=File_Name,
            FilePath=blob_url,
            File_Length=file_length,
            FileType=content_type,
            Upload_Date=created_at_kst,
            File_Status="Uploaded"
        )

        db.add(audio_file)
        await db.commit()
        await db.refresh(audio_file)
        
        current_audio_id = audio_file.audio_id

        # 오디오 파일 처리
        output_dir = f'data/split_audio_files/'
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        container_name = f"processed-audio{user_id}"
        
        # 컨테이너 생성 및 공개 접근 수준 설정 (Container 또는 Blob)
        try:
            blob_service_client.create_container(container_name, public_access=PublicAccess.Container)
        except ResourceExistsError:
            pass
        
        processor = AudioProcessor(audio_file.audio_id, temp_file_path, output_dir, blob_service_client, container_name)
        audio_total_len, segments_info, segment_lengths = processor.process_audio()
        results = await split_and_save_results(db, audio_file.audio_id, segments_info, segment_lengths)
        import subprocess

        # WSL에서 새 터미널을 열고, Python 스크립트 실행
        subprocess.run(["wsl", "python3", "model_main.py"])
        command = [
        "wsl",
        "torchrun", "--nproc_per_node", "1",
        "./llama/example_text_completion.py",
        "--ckpt_dir", "./llama/llama-2-7b/",
        "--tokenizer_path", "./llama/tokenizer.model",
        "--max_seq_len", "512", "--max_batch_size", "6"
        ]
        # WSL에서 새 터미널을 열고, Python 스크립트 실행
        subprocess.run(command)
 
        subprocess.run(['wsl','python3','effectgen.py'])
        # VSCode를 사용하여 특정 파일 열기)
    
        # STT 결과를 Result 테이블에 업데이트
        await update_stt(db)

        # EffectSounds 데이터베이스 효과음 커밋
        await upload_effectsounds(db)

        # EffectSounds 데이터베이스에 배경음악 커밋
        await upload_bgm(db, current_audio_id)

    except Exception as e:
        # blob_client가 초기화되었고, 예외 발생 시 해당 blob 삭제
        if blob_client:
            try:
                blob_client.delete_blob()  # blob 삭제 시도
            except Exception as delete_error:
                print(f"Failed to delete blob: {delete_error}")
        raise e  # 원래 발생한 예외 다시 발생시킴

    # finally:
    #     # 모든 처리가 완료된 후 디렉토리 삭제
    #     if os.path.exists(temp_file_path2):
    #         shutil.rmtree(temp_file_path2)
        
    #     # 모든 처리가 완료된 후 디렉토리 삭제
    #     if os.path.exists(output_dir):
    #         shutil.rmtree(output_dir)

    return results

# Azure에서 파일 다운로드 및 AudioFiles 데이터베이스에 저장
async def downloadfromazure(user_id: int, File_Name: str, db: AsyncSession):
    container_name = f"test{user_id}"
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=File_Name)
    temp_file_path = f"./tmp/{File_Name}"
    
    try:
        # Blob에서 파일 다운로드 시도
        with open(temp_file_path, "wb") as file:
            download_stream = blob_client.download_blob()
            file.write(download_stream.readall())
            
    except ResourceNotFoundError:
        # Blob Storage에 파일이 없는 경우 예외 처리
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=404, detail="Blob not found")

    # 파일이 성공적으로 다운로드 된 경우, 파일 정보 추출 및 저장
    try:
        audio = MP3(temp_file_path)
        file_length = audio.info.length
        content_type, _ = mimetypes.guess_type(File_Name)

        korea_time_zone = pytz.timezone("Asia/Seoul")
        created_at_kst = datetime.datetime.now(korea_time_zone)

        audio_file = AudioFile(
            user_id=user_id,
            File_Name=File_Name,
            FilePath=blob_client.url,
            File_Length=file_length,
            FileType=content_type,
            Upload_Date=created_at_kst,
            File_Status="Downloaded"
        )

        db.add(audio_file)
        await db.commit()
        await db.refresh(audio_file)
        return audio_file
    except Exception as e:
        # 다른 예외 발생 시 처리
        raise HTTPException(status_code=500, detail=str(e))

# 오디오 파일 조회 함수
async def get_user_id_by_login_id(db: AsyncSession, login_id: str):
    result = await db.execute(select(User).filter_by(login_id=login_id))
    user = result.scalar_one_or_none()
    return user.user_id if user else None

# 오디오 파일 조회 함수
async def get_audio_id_by_user_id(db: AsyncSession, user_id: int):
    result = await db.execute(
        select(AudioFile).where(AudioFile.user_id == user_id).order_by(AudioFile.audio_id.desc()).limit(1)
    )
    audio = result.scalar_one_or_none()
    return audio.audio_id if audio else None

# 오디오 파일 조회 함수
async def get_audiofile_by_id(db: AsyncSession, audio_id: int):
    audiofile = await db.get(AudioFile, audio_id)
    if audiofile is None:
        raise HTTPException(status_code=404, detail="Audio file not found")
    return audiofile

# 오디오 파일 삭제 함수
async def delete_audiofile(db: AsyncSession, audio_id: int):
    existing_audiofile = await db.get(AudioFile, audio_id)
    if existing_audiofile is None:
        raise HTTPException(status_code=404, detail="Audio file not found")
    await db.delete(existing_audiofile)
    await db.commit()
    return {
        "File_Name": existing_audiofile.File_Name,
        "FileType": existing_audiofile.FileType,
        "Upload_Date": existing_audiofile.Upload_Date
    }
    