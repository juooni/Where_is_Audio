from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Float
from sqlalchemy.orm import relationship
from config.database import Base
from pydantic import BaseModel

class EffectSounds(Base):
    __tablename__ = 'effectsound'
    effect_sound_id = Column(Integer, primary_key=True)
    result_id = Column(Integer, ForeignKey('result.result_id'))
    Effect_Name = Column(String(255), nullable=False)
    EffectFilePath = Column(String(255), nullable=False)
    EffectFileLength = Column(Float, nullable=False)
    Upload_Date = Column(DateTime, nullable=False)
    
    result = relationship("Result", back_populates="effectsound")
    
class EffectSoundCreate(BaseModel):
    result_id: int
    Effect_Name: str
    EffectFilePath: str
    EffectFileLangth : float
