from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from config.database import Base
from pydantic import BaseModel
from datetime import datetime

class FinalAudioBooks(Base):
    __tablename__ = 'finalaudiobook'
    final_audio_id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('user.user_id'))
    audio_id = Column(Integer, ForeignKey('audiofile.audio_id'))
    Final_File_Name = Column(String(255), nullable=False)
    FinalFilePath = Column(String(255), nullable=False)
    Final_File_Length = Column(Float, nullable=False)
    Creation_Date = Column(DateTime, nullable=False)

    # Relationship to Users and AudioFiles
    user = relationship("User", back_populates="finalaudiobooks")
    original_audio = relationship("AudioFile", back_populates="finalaudiobooks")