import os
import gradio as gr
from dotenv import load_dotenv
from brain_of_the_doctor import encode_image, analyze_image_with_query
from voice_of_the_patient import transcribe_with_groq
from voice_of_the_doctor import text_to_speech_with_elevenlabs

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

system_prompt = """
You are a professional doctor (for learning purposes). Analyze the image and provide medical insights.
Keep responses concise (max 2 sentences) and address the user like a real doctor.
"""

def process_inputs(audio_filepath, image_filepath):
    if not audio_filepath or not image_filepath:
        return "Waiting for both audio and image...", "Waiting for both audio and image...", None

    # Convert speech to text
    speech_to_text_output = transcribe_with_groq(
        GROQ_API_KEY=GROQ_API_KEY,
        audio_filepath=audio_filepath,
        stt_model="whisper-large-v3"
    )

    # Encode and analyze the image
    encoded_image = encode_image(image_filepath)
    doctor_response = analyze_image_with_query(query=system_prompt + speech_to_text_output, encoded_image=encoded_image)

    # Convert text response to speech
    voice_of_doctor = text_to_speech_with_elevenlabs(input_text=doctor_response, output_filepath="final.mp3")

    return speech_to_text_output, doctor_response, "final.mp3"

# Gradio UI
iface = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath", label="Record your voice"),
        gr.Image(type="filepath", label="Upload medical image")
    ],
    outputs=[
        gr.Textbox(label="Speech to Text"),
        gr.Textbox(label="Doctor's Response"),
        gr.Audio(label="Doctor's Voice Response")
    ],
    title="AI Doctor with Vision and Voice",
    description="Speak and upload a medical image. The AI doctor will analyze both before responding.",
    live=False  # Set live=False to ensure response starts only after submission
)

iface.launch(debug=True)


#http://127.0.0.1:7860