from models.EditHistory import EditEffect, HistoryResponse, EditEffect
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from config.database import get_db
import pytz
from datetime import datetime
from services.EditHistory_service import (
    get_edithistory_by_id, 
    delete_edithistory, 
    apply_effect, 
    cancel_effect
)

router = APIRouter()

@router.get("/read/{history_id}", response_model = HistoryResponse)
async def get_history_by_id(history_id: int, db: AsyncSession = Depends(get_db)):
    history = await get_edithistory_by_id(db, history_id)
    if history is None:
        raise HTTPException(status_code=404, detail="EditHistory not found")
    return history

@router.delete("/delete/{history_id}")
async def delete_history(history_id: int, db: AsyncSession = Depends(get_db)):
    await delete_edithistory(db, history_id)
    return {"message": "EditHistory deleted"}

@router.post("/apply")
async def apply_history_effect(request : EditEffect, db: AsyncSession = Depends(get_db)): 
    new_history = await apply_effect(db, request)
    return new_history

@router.post("/cancel")
async def cancel_history_effect(request : EditEffect, db: AsyncSession = Depends(get_db)):  
    new_history = await cancel_effect(db, request)
    return new_history