import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import torch

def generate_bgm(prompt,filepath,total_duration):
    model = MusicGen.get_pretrained('facebook/musicgen-medium')
    model.set_generation_params(duration=1) 
    device=torch.device('cuda')
    model.device=device
    descriptions=prompt+' jazz no beat'

    wav = model.generate(descriptions)  # generates 3 samples.  
    for idx, one_wav in enumerate(wav):
        audio_write(filepath+'/bgm_'+prompt, one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
        ##emotion_id ex->fear같은거 포함??





'''wav = model.generate_unconditional(3)    # generates 4 unconditional audio samples
for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(f'{idx}'+'unconditional', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
descriptions = ['sad jazz with no beat no drum no base no guitar','scared jazz with no drum no guitar','classic']
'''

#melody, sr = torchaudio.load('./assets/bach.mp3')
# generates using the melody from the given audio and the provided descriptions.
#wav = model.generate_with_chroma(descriptions, melody[None].expand(3, -1, -1), sr)

#for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
#    audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)