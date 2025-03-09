# voice_of_the_doctor.py

from dotenv import load_dotenv
load_dotenv()

import os
import logging
import platform
import subprocess
import tempfile
from gtts import gTTS
import elevenlabs
from elevenlabs.client import ElevenLabs

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VoiceOfDoctor")

# API Key for ElevenLabs
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")

def text_to_speech_with_gtts(input_text, output_filepath):
    """
    Generate speech from text using Google's Text-to-Speech service.
    Fallback option when ElevenLabs is unavailable.
    
    Args:
        input_text (str): Text to convert to speech
        output_filepath (str): File path to save the audio
    
    Returns:
        str: Path to the generated audio file
    """
    try:
        logger.info("Generating speech with gTTS...")
        
        # Create TTS object
        audio_obj = gTTS(
            text=input_text,
            lang="en",
            slow=False
        )
        
        # Save audio file
        audio_obj.save(output_filepath)
        logger.info(f"Speech generated and saved to {output_filepath}")
        
        # Auto-play based on OS
        play_audio_file(output_filepath)
        
        return output_filepath
        
    except Exception as e:
        logger.error(f"gTTS error: {str(e)}")
        # Create simple error message if TTS fails
        error_filepath = "error_message.txt"
        with open(error_filepath, "w") as f:
            f.write("Speech generation failed. Please check logs.")
        return error_filepath

def text_to_speech_with_elevenlabs(input_text, output_filepath, voice="Aria", model="eleven_turbo_v2"):
    """
    Generate high-quality speech using ElevenLabs API with extended options.
    
    Args:
        input_text (str): Text to convert to speech
        output_filepath (str): File path to save the audio
        voice (str): Voice ID or name to use
        model (str): Model to use for synthesis
    
    Returns:
        str: Path to the generated audio file
    """
    try:
        if not ELEVENLABS_API_KEY:
            logger.warning("ElevenLabs API key not found, falling back to gTTS")
            return text_to_speech_with_gtts(input_text, output_filepath)
            
        logger.info(f"Generating speech with ElevenLabs using voice '{voice}'...")
        
        # Initialize ElevenLabs client
        client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        
        # Generate audio with more parameters for better medical voice
        audio = client.generate(
            text=input_text,
            voice=voice,
            output_format="mp3_44100_128",
            model=model,
            voice_settings={
                "stability": 0.71,      # More stable for medical advice
                "similarity_boost": 0.5, # Balanced voice characteristics
                "style": 0.0,            # Neutral style for medical context
                "use_speaker_boost": True
            }
        )
        
        # Save the audio file
        elevenlabs.save(audio, output_filepath)
        logger.info(f"Speech generated and saved to {output_filepath}")
        
        # Auto-play based on OS
        play_audio_file(output_filepath)
        
        return output_filepath
        
    except Exception as e:
        logger.error(f"ElevenLabs error: {str(e)}")
        logger.info("Falling back to gTTS...")
        return text_to_speech_with_gtts(input_text, output_filepath)

def play_audio_file(filepath):
    """
    Play an audio file based on the operating system.
    
    Args:
        filepath (str): Path to the audio file
    """
    os_name = platform.system()
    try:
        if os_name == "Darwin":  # macOS
            subprocess.run(['afplay', filepath], check=True)
        elif os_name == "Windows":  # Windows
            # Use the Windows default player
            os.startfile(filepath)
        elif os_name == "Linux":  # Linux
            # Try multiple players in case one is not available
            for player in ['aplay', 'mpg123', 'ffplay']:
                try:
                    subprocess.run([player, filepath], check=True)
                    break
                except (subprocess.SubprocessError, FileNotFoundError):
                    continue
        else:
            logger.warning(f"Unsupported operating system: {os_name}")
    except Exception as e:
        logger.error(f"Error playing audio: {str(e)}")

# Example usage (commented out for import)
"""
if __name__ == "__main__":
    sample_text = "Based on what I can see, you appear to have a mild case of contact dermatitis. The redness and irritation are consistent with an allergic reaction to something that has come in contact with your skin. I recommend applying a mild hydrocortisone cream and avoiding potential allergens. If symptoms worsen, please consult with a dermatologist for further evaluation."
    
    # Test both TTS services
    gtts_output = "doctor_response_gtts.mp3"
    text_to_speech_with_gtts(input_text=sample_text, output_filepath=gtts_output)
    
    elevenlabs_output = "doctor_response_elevenlabs.mp3"
    text_to_speech_with_elevenlabs(input_text=sample_text, output_filepath=elevenlabs_output)
"""