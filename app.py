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
    speech_to_text_output = transcribe_with_groq(
        GROQ_API_KEY=GROQ_API_KEY,
        audio_filepath=audio_filepath,
        stt_model="whisper-large-v3"
    )

    # Handle Image Input
    if image_filepath:
        encoded_image = encode_image(image_filepath)
        doctor_response = analyze_image_with_query(query=system_prompt + speech_to_text_output, encoded_image=encoded_image)
    else:
        doctor_response = "No image provided for analysis."

    voice_of_doctor = text_to_speech_with_elevenlabs(input_text=doctor_response, output_filepath="final.mp3")

    return speech_to_text_output, doctor_response, "final.mp3"

# Gradio UI with loading animation
iface = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath"),
        gr.Image(type="filepath")
    ],
    outputs=[
        gr.Textbox(label="Speech to Text"),
        gr.Textbox(label="Doctor's Response"),
        gr.Audio("final.mp3")
    ],
    title="AI Doctor with Vision and Voice",
    live=True
)

iface.launch(debug=True)


#http://127.0.0.1:7860