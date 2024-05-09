from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from config.database import Base

class BackgroundMusic(Base):
    __tablename__ = 'backgroundmusic'
    bgm_id = Column(Integer, primary_key=True)
    audio_id = Column(Integer, ForeignKey('audiofile.audio_id'))
    Music_Name = Column(String(255), nullable=False)
    MusicFilePath = Column(String(255), nullable=False)
    File_Length = Column(Float, nullable=False)
    Upload_Date = Column(DateTime, nullable=False)



    audiofile = relationship('AudioFile', back_populates="backgroundmusic")