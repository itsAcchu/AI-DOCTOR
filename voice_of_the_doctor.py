import os
import platform
import subprocess
from pydub import AudioSegment
from gtts import gTTS
import elevenlabs
from elevenlabs.client import ElevenLabs

# Load API Key
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")

# Function to convert MP3 to WAV (for Windows playback)
def convert_mp3_to_wav(mp3_filepath):
    wav_filepath = mp3_filepath.replace(".mp3", ".wav")  # Change extension
    audio = AudioSegment.from_mp3(mp3_filepath)
    audio.export(wav_filepath, format="wav")  # Convert MP3 to WAV
    return wav_filepath

# Function to play audio (cross-platform)
def play_audio(file_path):
    if os.path.exists(file_path):
        os_name = platform.system()
        try:
            if os_name == "Darwin":  # macOS
                subprocess.run(["afplay", file_path])
            elif os_name == "Windows":  # Windows
                wav_file = convert_mp3_to_wav(file_path)  # Convert MP3 to WAV
                subprocess.run(["powershell", "-c", f"(New-Object Media.SoundPlayer '{wav_file}').PlaySync();"])
            elif os_name == "Linux":  # Linux
                subprocess.run(["aplay", file_path])
        except Exception as e:
            print(f"Error playing audio: {e}")

# Function for text-to-speech using ElevenLabs
def text_to_speech_with_elevenlabs(input_text, output_filepath):
    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    audio = client.generate(text=input_text, voice="Aria", output_format="mp3_22050_32", model="eleven_turbo_v2")
    elevenlabs.save(audio, output_filepath)
    play_audio(output_filepath)  # Play after conversion
