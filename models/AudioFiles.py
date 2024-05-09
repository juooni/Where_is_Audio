from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from config.database import Base
from pydantic import BaseModel
from datetime import datetime

class AudioFile(Base):
    __tablename__ = 'audiofile'
    audio_id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('user.user_id'))
    File_Name = Column(String(255), nullable=False)
    FilePath = Column(String(255), nullable=False)
    File_Length = Column(Float, nullable=False)
    FileType = Column(String(50), nullable=False)
    Upload_Date = Column(DateTime, nullable=False)
    File_Status = Column(String(50), nullable=False)

    # Relationship
    user = relationship("User", back_populates="audiofile")
    result = relationship("Result", back_populates="audiofile")
    finalaudiobooks = relationship("FinalAudioBooks", back_populates="original_audio")
    backgroundmusic = relationship("BackgroundMusic", back_populates="audiofile")

class AudioResponse(BaseModel):
    audio_id : int
    File_Name : str
    FileType : str
    Upload_Date : datetime
    File_Status : str
    
class AudioDelete(BaseModel):
    audio_id : int
    File_Name : str
    FileType : str
    Upload_Date : datetime

class AudioRead(BaseModel):
    File_Name : str
    FileType : str
    Upload_Date : datetime
    File_Status : str