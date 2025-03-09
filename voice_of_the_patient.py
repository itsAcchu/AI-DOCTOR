# voice_of_the_patient.py

from dotenv import load_dotenv
load_dotenv()

import logging
import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO
import os
import tempfile
from groq import Groq

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VoiceOfPatient")

def record_audio(file_path, timeout=20, phrase_time_limit=None):
    """
    Enhanced function to record audio from the microphone with better user feedback.
    
    Args:
        file_path (str): Path to save the recorded audio file.
        timeout (int): Maximum time to wait for a phrase to start (in seconds).
        phrase_time_limit (int): Maximum time for the phrase to be recorded (in seconds).
    """
    recognizer = sr.Recognizer()
    
    try:
        with sr.Microphone() as source:
            logger.info("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            logger.info("Start speaking now...")
            
            # Record the audio
            audio_data = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            logger.info("Recording complete.")
            
            # Convert the recorded audio to an MP3 file
            wav_data = audio_data.get_wav_data()
            audio_segment = AudioSegment.from_wav(BytesIO(wav_data))
            audio_segment.export(file_path, format="mp3", bitrate="128k")
            
            logger.info(f"Audio saved to {file_path}")
            return True

    except sr.WaitTimeoutError:
        logger.warning("No speech detected within timeout period")
        return False
    except sr.RequestError as e:
        logger.error(f"API unavailable: {e}")
        return False
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return False

def transcribe_with_groq(GROQ_API_KEY, audio_filepath, stt_model="whisper-large-v3"):
    """
    Enhanced function to transcribe audio using Groq's Whisper model.
    
    Args:
        GROQ_API_KEY (str): API key for Groq
        audio_filepath (str): Path to the audio file
        stt_model (str): Model to use for transcription
        
    Returns:
        str: Transcribed text
    """
    client = Groq(api_key=GROQ_API_KEY)
    
    try:
        if not os.path.exists(audio_filepath):
            raise FileNotFoundError(f"Audio file not found: {audio_filepath}")
            
        # Check file size and format
        audio = AudioSegment.from_file(audio_filepath)
        
        # Convert to required format if necessary
        if audio_filepath.endswith('.mp3'):
            # Create a temporary WAV file if needed
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_filepath = temp_file.name
            temp_file.close()
            
            # Convert MP3 to WAV for better compatibility
            audio.export(temp_filepath, format="wav")
            process_filepath = temp_filepath
        else:
            process_filepath = audio_filepath
        
        # Perform transcription
        logger.info(f"Transcribing audio with {stt_model}...")
        with open(process_filepath, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model=stt_model,
                file=audio_file,
                language="en"
            )
        
        # Clean up temporary file if created
        if 'temp_filepath' in locals():
            os.unlink(temp_filepath)
            
        logger.info("Transcription complete")
        return transcription.text
        
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise

# Example usage (commented out for import)
"""
if __name__ == "__main__":
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    audio_filepath = "patient_recording.mp3"
    
    # Record audio
    record_audio(file_path=audio_filepath)
    
    # Transcribe recorded audio
    transcribed_text = transcribe_with_groq(
        GROQ_API_KEY=GROQ_API_KEY,
        audio_filepath=audio_filepath
    )
    
    print("Transcribed Text:")
    print(transcribed_text)
"""