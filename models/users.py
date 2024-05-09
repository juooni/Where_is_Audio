from sqlalchemy import Column, Integer, String, DateTime, func
from sqlalchemy.orm import relationship
from config.database import Base
from pydantic import BaseModel
from datetime import datetime

class User(Base):
    __tablename__ = 'user'
    user_id = Column(Integer, primary_key=True)
    login_id = Column(String(255), nullable=False)
    login_pw = Column(String(255), nullable=False)
    created_at = Column(DateTime, server_default=func.now())

    # Relationship
    audiofile = relationship("AudioFile", back_populates="user")
    finalaudiobooks = relationship("FinalAudioBooks", back_populates="user")
    
class UserLogin(BaseModel):
    login_id: str
    login_pw: str
    
class Usercreate(BaseModel):
    login_id: str
    login_pw: str
    
class UserResponse(BaseModel):
    user_id: int
    login_id: str
    created_at: datetime


