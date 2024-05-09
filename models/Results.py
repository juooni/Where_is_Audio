from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Float
from sqlalchemy.orm import relationship
from config.database import Base
from pydantic import BaseModel
from datetime import datetime

class Result(Base):
    __tablename__ = 'result'
    result_id = Column(Integer, primary_key=True)
    audio_id = Column(Integer, ForeignKey('audiofile.audio_id'))
    File_Name = Column(String(255), nullable=False)
    Index = Column(Integer, nullable=False)
    Converted_Result = Column(String(255), nullable=False)
    ResultFilePath = Column(String(255), nullable=False)
    ResultFileLength = Column(Float, nullable=False)
    Converted_Date = Column(DateTime, nullable=False)
    
    # Relationship
    audiofile = relationship("AudioFile", back_populates="result")
    effectsound = relationship("EffectSounds", back_populates="result")
    edithistory = relationship("EditHistory", back_populates="result")
    

