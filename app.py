from dotenv import load_dotenv
load_dotenv()

import os
import gradio as gr
import time
from brain_of_the_doctor import encode_image, analyze_image_with_query
from voice_of_the_patient import record_audio, transcribe_with_groq
from voice_of_the_doctor import text_to_speech_with_gtts, text_to_speech_with_elevenlabs

system_prompt = """You are Dr. AI, a professional medical consultant with extensive clinical experience. Your task is to analyze the provided medical image along with the patient's description.

Analysis Guidelines:
1. First describe what you see in the image in clinical terms
2. Identify any abnormalities, lesions, or concerning features
3. Formulate a differential diagnosis (2-3 most likely conditions)
4. Suggest appropriate treatments or remedies for each possible diagnosis, including specific medication options where appropriate
5. Provide dosage guidelines for recommended medications (when applicable)
6. Recommend if the patient should seek in-person medical consultation and with what urgency

Your response should be:
- Professional but conversational, using appropriate medical terminology
- Clinically accurate but understandable to patients
- Compassionate and reassuring when appropriate
- Structured with clear sections for observations, differential, and recommendations
- Include potential over-the-counter medications with proper dosing information
- For prescription medications, mention common options a physician might consider
- Limited to 150-200 words maximum

Begin your response directly with "Based on what I can see..." without any preamble.
Always include these important disclaimers:
- This is an AI consultation and not a replacement for in-person medical care
- Any medication suggestions should be discussed with a healthcare provider before use
- Seek immediate medical attention for severe or worsening symptoms

Patient's description: """

def process_inputs(audio_filepath, image_filepath, progress=gr.Progress()):
    results = {
        "speech_to_text": "",
        "doctor_response": "",
        "voice_filepath": ""
    }
    output_filepath = "doctor_response.mp3"
    try:
        progress(0.05, "Initializing analysis...")
        time.sleep(0.5)
        progress(0.1, "Processing your audio description...")
        if audio_filepath:
            progress(0.2, "Converting speech to text...")
            results["speech_to_text"] = transcribe_with_groq(
                GROQ_API_KEY=os.environ.get("GROQ_API_KEY"),
                audio_filepath=audio_filepath,
                stt_model="whisper-large-v3"
            )
        else:
            results["speech_to_text"] = "No audio provided. Please describe your medical concern."
        progress(0.3, "Preparing medical image...")
        if image_filepath:
            progress(0.4, "Analyzing visual patterns...")
            full_query = system_prompt + results["speech_to_text"]
            progress(0.5, "Consulting medical knowledge base...")
            results["doctor_response"] = analyze_image_with_query(
                query=full_query,
                encoded_image=encode_image(image_filepath),
                model="meta-llama/llama-4-scout-17b-16e-instruct"
            )
        else:
            results["doctor_response"] = "No image provided for analysis. Please upload a clear medical image for diagnosis."
        progress(0.7, "Generating doctor's response...")
        progress(0.8, "Creating natural voice output...")
        text_to_speech_with_elevenlabs(
            input_text=results["doctor_response"],
            output_filepath=output_filepath
        )
        results["voice_filepath"] = output_filepath
        progress(0.95, "Finalizing results...")
        time.sleep(0.5)
        progress(1.0, "Consultation complete!")
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        results["doctor_response"] = error_message
        text_to_speech_with_gtts(
            input_text="I'm sorry, there was an error processing your request. Please try again.",
            output_filepath=output_filepath
        )
        results["voice_filepath"] = output_filepath
    return results["speech_to_text"], results["doctor_response"], results["voice_filepath"]

custom_css = """
:root {
    --primary-color: #2e7d32;
    --secondary-color: #f0f7f0;
    --accent-color: #1b5e20;
    --text-color: #333333;
    --light-text: #4d6b50;
    --border-radius: 8px;
    --shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}
.container { max-width: 1200px; margin: 0 auto; padding: 20px; }
.header { text-align: center; margin-bottom: 30px; }
.header h1 { font-size: 2.5rem; color: var(--primary-color); margin-bottom: 8px; font-weight: 700; }
.header h2 { font-size: 1.2rem; color: var(--light-text); font-weight: 400; margin-bottom: 16px; }
.card { background: white; border-radius: var(--border-radius); padding: 24px; box-shadow: var(--shadow); margin-bottom: 24px; border: 1px solid #e0e0e0; overflow: hidden; }
.card-title { font-size: 1.2rem; color: var(--primary-color); margin-bottom: 16px; font-weight: 600; display: flex; align-items: center; }
.card-title svg { margin-right: 10px; }
.primary-button { background: var(--primary-color) !important; color: white !important; padding: 12px 24px !important; border-radius: var(--border-radius) !important; font-weight: 600 !important; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important; transition: all 0.3s ease !important; border: none !important; }
.primary-button:hover { background: var(--accent-color) !important; box-shadow: 0 6px 16px rgba(0, 0, 0, 0.3) !important; transform: translateY(-2px) !important; }
.secondary-button { background: white !important; color: var(--primary-color) !important; border: 1px solid var(--primary-color) !important; padding: 12px 24px !important; border-radius: var(--border-radius) !important; font-weight: 600 !important; transition: all 0.3s ease !important; }
.secondary-button:hover { background: #f5f5f5 !important; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important; }
.footer { text-align: center; margin-top: 30px; color: var(--light-text); font-size: 0.9rem; }
.gradio-container { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important; }
.gr-button { min-width: 140px !important; border-radius: var(--border-radius) !important; }
.gr-panel { border-radius: var(--border-radius) !important; border: 1px solid #e0e0e0 !important; overflow: hidden !important; }
.gr-box { border-radius: var(--border-radius) !important; overflow: hidden !important; }
.gr-form { border-radius: var(--border-radius) !important; overflow: hidden !important; }
.gr-accordion { border-radius: var(--border-radius) !important; overflow: hidden !important; }
.progress-bar { height: 4px; background: #eaeaea; border-radius: 2px; overflow: hidden; margin: 10px 0; }
.progress-fill { height: 100%; background: linear-gradient(90deg, #2e7d32, #81c784); border-radius: 2px; animation: shimmer 1.5s infinite; background-size: 200% 100%; }
.progress-text { font-size: 0.85rem; color: var(--light-text); margin-bottom: 5px; }
@keyframes shimmer { 0% { background-position: -200% 0; } 100% { background-position: 200% 0; } }
div[class*="message"],
div[class*="input"],
div[class*="output"],
div[class*="textbox"],
div[class*="audio"],
div[class*="image"] { border-radius: var(--border-radius) !important; overflow: hidden !important; }
input, textarea, select, button { border-radius: var(--border-radius) !important; }
"""

def create_interface():
    with gr.Blocks() as iface:
        with gr.Row(elem_classes="header"):
            with gr.Column():
                gr.Markdown(
                    """
                    # MediScan AI
                    ### Advanced Medical Imaging Analysis & Professional Consultation

                    *Using state-of-the-art AI to provide preliminary medical assessments*
                    """
                )

        with gr.Row(equal_height=True, elem_classes="container"):
            with gr.Column(scale=1, min_width=400, elem_classes="card"):
                gr.Markdown(
                    """
                    <div class="card-title">
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path><line x1="12" y1="19" x2="12" y2="22"></line></svg>
                    Step 1: Describe Your Symptoms
                    </div>
                    """
                )
                audio_input = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label="Record a detailed description of your symptoms",
                    elem_id="audio-input"
                )

                gr.Markdown(
                    """
                    <div class="card-title">
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>
                    Step 2: Upload Medical Image
                    </div>
                    """
                )
                image_input = gr.Image(
                    type="filepath",
                    label="Upload a clear, well-lit image of the affected area",
                    elem_id="image-input",
                    height=260
                )

                with gr.Row():
                    submit_btn = gr.Button("Begin Analysis", elem_classes="primary-button")
                    clear_btn = gr.Button("New Consultation", elem_classes="secondary-button")

            with gr.Column(scale=1, min_width=400, elem_classes="card"):
                gr.Markdown(
                    """
                    <div class="card-title">
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>
                    Your Description
                    </div>
                    """
                )
                text_output = gr.Textbox(
                    label="Your transcribed symptoms",
                    elem_id="transcription-output",
                    lines=3
                )

                gr.Markdown(
                    """
                    <div class="card-title">
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"></path></svg>
                    Medical Assessment
                    </div>
                    """
                )
                response_output = gr.Textbox(
                    label="Dr. AI's Analysis",
                    elem_id="analysis-output",
                    lines=10
                )

                gr.Markdown(
                    """
                    <div class="card-title">
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18V5l12-2v13"></path><circle cx="6" cy="18" r="3"></circle><circle cx="18" cy="16" r="3"></circle></svg>
                    Audio Consultation
                    </div>
                    """
                )
                audio_output = gr.Audio(
                    label="Listen to Dr. AI's voice response",
                    elem_id="audio-output"
                )

        gr.Markdown(
            """
            <div class="footer">
            <p>© 2025 MediScan AI. For educational purposes only. Not a substitute for professional medical advice.</p>
            <p>Always consult with a qualified healthcare provider for medical concerns.</p>
            </div>
            """,
            elem_classes="footer"
        )

        submit_btn.click(
            fn=process_inputs,
            inputs=[audio_input, image_input],
            outputs=[text_output, response_output, audio_output]
        )

        clear_btn.click(
            fn=lambda: (None, None, "", "", None),
            inputs=[],
            outputs=[audio_input, image_input, text_output, response_output, audio_output]
        )

    return iface

if __name__ == "__main__":
    os.makedirs("examples", exist_ok=True)
    iface = create_interface()
    iface.launch(debug=True, css=custom_css)
