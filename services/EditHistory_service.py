from models.EditHistory import EditHistory, EditEffect
from models.Results import Result
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import HTTPException
import pytz
from datetime import datetime

# 한국 시간대(KST, UTC+9)를 사용하여 'created_at'을 설정합니다.
korea_time_zone = pytz.timezone("Asia/Seoul")
created_at_kst = datetime.now(korea_time_zone)

# 편집 기록 확인
async def get_edithistory_by_id(db: AsyncSession, history_id: int):
    history = await db.get(EditHistory, history_id)
    if history is None:
        raise HTTPException(status_code=404, detail="EditHistory not found")
    return history

# 편집 기록 삭제
async def delete_edithistory(db: AsyncSession, history_id: int):
    existing_history = await db.get(EditHistory, history_id)
    if existing_history is None:
        raise HTTPException(status_code=404, detail="EditHistory not found")

    await db.delete(existing_history)
    await db.commit()

# 효과음을 채택했을 때 실행되는 함수
async def apply_effect(db: AsyncSession, request : EditEffect):
    result = await db.get(Result, request.result_id)

    new_history = EditHistory(
        result_id=result.result_id,
        effect_sound_id=request.effect_sound_id,
        Edit_Action="Apply Effect",
        EditDate=created_at_kst
    )

    db.add(new_history)
    await db.commit()
    await db.refresh(new_history)
    return new_history

# 이펙트 선택을 안하면 실행되는 함수
async def cancel_effect(db: AsyncSession, request : EditEffect):
    created_at_kst = datetime.now(korea_time_zone)
    result = await db.get(Result, request.result_id)

    new_history = EditHistory(
        result_id=result.result_id,
        effect_sound_id=request.effect_sound_id,
        Edit_Action="Cancel Effect",
        EditDate=created_at_kst
    )

    db.add(new_history)
    await db.commit()
    await db.refresh(new_history)
    return new_history

