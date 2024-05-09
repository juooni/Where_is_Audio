from azure.storage.blob import BlobServiceClient, PublicAccess
from azure.core.exceptions import ResourceExistsError
from models.EffectSounds import EffectSounds
from models.EditHistory import EditHistory
from models.EffectSounds import EffectSounds
from models.Results import Result
from models.AudioFiles import AudioFile
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from pydub import AudioSegment
import pytz, httpx, tempfile, contextlib, wave, os, uuid, shutil, io
from datetime import datetime
from config.database import connect_str

temp_file_paths = []  # 전역 변수로 임시 파일 경로 리스트 선언
korea_time_zone = pytz.timezone("Asia/Seoul")
created_at_kst = datetime.now(korea_time_zone)


# wav로 바꿔주는 함수
def convert_to_wav(file_path: str) -> str:
    audio = AudioSegment.from_file(file_path)
    tmp_path = tempfile.mktemp(".wav")  # 임시 파일 경로 생성
    audio.export(tmp_path, format="wav")
    return tmp_path

# wav 정보 길이 함수
def get_wav_length(wav_path: str) -> float:
    with contextlib.closing(wave.open(wav_path, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = round(frames / float(rate), 2)
    return duration
    
async def upload_effect_sound_to_azure(file_name: str, file_path: str, db: AsyncSession):
    
    # 파일을 WAV 형식으로 변환
    converted_file_path = convert_to_wav(file_path)
    
    container_name = "effects"
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)

    # 컨테이너가 없으면 생성
    container_client = blob_service_client.get_container_client(container_name)
    try:
        container_client.create_container(public_access=PublicAccess.Container)
    except ResourceExistsError:
        pass  # 컨테이너가 이미 존재하면 무시
        
    # 파일 업로드
    blob_client = container_client.get_blob_client(blob=file_name)
    with open(converted_file_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)
    blob_url = blob_client.url

    # 변환된 WAV 파일의 길이 정보 계산
    wav_length = get_wav_length(converted_file_path)

    # 파일 메타데이터와 wav 길이를 데이터베이스에 저장
    effect_sound = EffectSounds(
        result_id=1,  # result_id 설정이 필요하다면 적절한 값으로 변경
        Effect_Name=file_name,
        EffectFilePath=blob_url,
        EffectFileLength=wav_length,  # wav 파일 길이 정보 추가
        Upload_Date=created_at_kst
    )
    
    db.add(effect_sound)
    await db.commit()
    await db.refresh(effect_sound)

    # 임시 변환 파일 삭제
    os.remove(converted_file_path)
    
    return effect_sound

async def download_file(url):
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        resp.raise_for_status()

        tmp_dir = "./tmp"
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        tmp_file_path = os.path.join(tmp_dir, "tempfile_" + uuid.uuid4().hex)
        with open(tmp_file_path, "wb") as tmp_file:
            tmp_file.write(resp.content)
        
        temp_file_paths.append(tmp_file_path)  # 파일 경로를 전역 리스트에 추가
        return tmp_file_path

async def delete_temp_files(file_paths):
    for file_path in file_paths:
        try:
            os.remove(file_path)
            print(f"Deleted temp file: {file_path}")
        except OSError as e:
            print(f"Error deleting file {file_path}: {e}")
            
# 오디오 분할 파일 + 효과음 파일 합쳐서 저장
async def combine_final_audio_files(db: AsyncSession, audio_id: int):

    result_files = await db.execute(
        select(Result).filter(Result.audio_id == audio_id).order_by(Result.Index)
    )
    result_files = result_files.scalars().all()
    
    # 결과 파일이 없는 경우 에러 메시지 출력
    if not result_files:
        raise ValueError(f"No Audio File Founded")
    
    combined_audio = AudioSegment.empty()

    for result_file in result_files:
        effect_history = await db.execute(
            select(EditHistory)
            .filter(EditHistory.result_id == result_file.result_id)
            .order_by(EditHistory.EditDate.desc())
            .limit(1)
        )
        effect_history = effect_history.scalars().first()

        # 파일 다운로드 및 오디오 세그먼트 생성
        audio_segment = AudioSegment.from_file(await download_file(result_file.ResultFilePath))

        # 최신 상태가 "Apply Effect"이고, 효과음 파일이 존재한다면 효과음 적용
        if effect_history and effect_history.Edit_Action == "Apply Effect":
            effect_sound = await db.get(EffectSounds, effect_history.effect_sound_id)
            effect_segment = AudioSegment.from_file(await download_file(effect_sound.EffectFilePath))
            
            # 1.OverLay method
            # audio_segment = audio_segment.overlay(effect_segment)
            # 2.Add method
            audio_segment = effect_segment + audio_segment
            
        combined_audio += audio_segment

    # 합쳐진 오디오 파일을 임시 파일로 저장하고 경로 반환
    audio_file = await db.get(AudioFile, audio_id)
    final_audio_filename = "Edited-" + str(audio_file.File_Name)
    final_directory = "./tmp"
    # final_audio_filename = f"final_audio_{audio_id}.wav"
    final_audio_path = os.path.join(final_directory, final_audio_filename)
    combined_audio.export(final_audio_path, format='wav')

    final_length = get_wav_length(final_audio_path)
    
    container_name = "final-audio-book"
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)

    # 컨테이너가 없으면 생성
    container_client = blob_service_client.get_container_client(container_name)
    try:
        container_client.create_container(public_access=PublicAccess.Container)
    except ResourceExistsError:
        pass  # 컨테이너가 이미 존재하면 무시
        
    # 파일을 Blob Storage에 업로드
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=os.path.basename(final_audio_path))
    with open(final_audio_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)
    blob_url = blob_client.url
    
    # # 업로드가 완료된 후 로컬의 임시 파일과 디렉토리 삭제
    shutil.rmtree(final_directory, ignore_errors=True)
     
    return blob_url, final_audio_filename, final_length

async def combine_audio_files_with_effects(db: AsyncSession, result_id: int, effect_sound_id: int):
    # 데이터베이스에서 오디오 파일 정보 가져오기
    audio_file = await db.get(Result, result_id)
    if not audio_file:
        return "오디오 파일을 찾을 수 없습니다."
    current_audio_segment = AudioSegment.from_file(audio_file.ResultFilePath)
    
    effect_sound = await db.get(EffectSounds, effect_sound_id)
    if not effect_sound:
        return "효과음을 찾을 수 없습니다."
    effect_segment = AudioSegment.from_file(effect_sound.EffectFilePath)
    
    # 1.OverLay method
    # combined_audio = current_audio_segment.overlay(effect_segment)
    # 2.Add method
    combined_audio = effect_segment + current_audio_segment
    
    # 합쳐진 오디오 파일 저장
    final_directory = f"./tmp"
    combined_audio_path = f'./tmp/combined_audio_{result_id}_{effect_sound_id}.wav'
    combined_audio.export(combined_audio_path, format='wav')
    
    return combined_audio_path, final_directory

async def combine_audio_files_with_effects(db: AsyncSession, result_id: int, effect_sound_id: int = None) -> bytes:
    audio_file = await db.get(Result, result_id)
    if not audio_file:
        return None  # 적절한 오류 처리 필요

    current_audio_segment = AudioSegment.from_file(audio_file.ResultFilePath)

    if effect_sound_id is not None:
        effect_sound = await db.get(EffectSounds, effect_sound_id)
        if not effect_sound:
            return None  # 적절한 오류 처리 필요
        effect_segment = AudioSegment.from_file(effect_sound.EffectFilePath)
        combined_audio = effect_segment + current_audio_segment
    else:
        combined_audio = current_audio_segment

    with io.BytesIO() as audio_buffer:
        combined_audio.export(audio_buffer, format="wav")
        audio_data = audio_buffer.getvalue()

    return audio_data

# 전체 오디오 파일에 분할 파일과 효과음 적용
async def combine_whole_audio_with_effects(db: AsyncSession, audio_id: int) -> bytes:
    # 전체 오디오 파일 대상으로 분할 파일과 효과음 적용 로직 구현
    combined_audio = AudioSegment.empty()

    # 모든 분할 파일 가져오기
    result_files = await db.execute(
        select(Result).filter(Result.audio_id == audio_id).order_by(Result.Index)
    )
    result_files = result_files.scalars().all()

    # 각 분할 파일에 대해 최신 효과음 적용 상태 확인
    for result_file in result_files:
        audio_segment = AudioSegment.from_file(result_file.ResultFilePath)

        # 해당 분할 파일에 적용된 최신 효과음 가져오기
        effect_history = await db.execute(
            select(EditHistory)
            .filter(EditHistory.result_id == result_file.result_id)
            .order_by(EditHistory.edit_date.desc())
        )
        effect_history = effect_history.scalars().first()

        # 최신 효과음이 적용된 경우, 해당 효과음 적용
        if effect_history and effect_history.edit_action == "Apply Effect":
            effect_sound = await db.get(EffectSounds, effect_history.effect_sound_id)
            if effect_sound:
                effect_segment = AudioSegment.from_file(effect_sound.EffectFilePath)
                audio_segment = audio_segment.overlay(effect_segment)

        combined_audio += audio_segment

    with io.BytesIO() as audio_buffer:
        combined_audio.export(audio_buffer, format="wav")
        audio_data = audio_buffer.getvalue()

    return audio_data