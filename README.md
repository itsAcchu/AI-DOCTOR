# AI Doctor 2.0

## Medical Chatbot with MultiModal LLM (Vision and Voice)

### Overview
AI Doctor 2.0 is an AI-powered medical assistant capable of understanding and responding to medical queries through text, vision, and voice inputs. The system leverages advanced multimodal large language models (LLMs) to provide an interactive and intelligent experience.

## Project Layout

### Phase 1 – Setup the Brain of the Doctor (Multimodal LLM)
- Configure **GROQ API Key** for AI inference.
- Convert medical images into the required format.
- Setup **Multimodal LLM** for image and text processing.

### Phase 2 – Setup Voice of the Patient
- Integrate **Audio Recorder** using **ffmpeg** & **portaudio**.
- Implement **Speech-to-Text (STT)** using **OpenAI Whisper**.

### Phase 3 – Setup Voice of the Doctor
- Implement **Text-to-Speech (TTS)** using **gTTS & ElevenLabs**.
- Convert AI-generated text responses into speech output.

### Phase 4 – Setup UI for the VoiceBot
- Develop an **interactive UI** using **Gradio** for real-time interaction.

---
## Tools and Technologies
- **Groq** – AI inference
- **OpenAI Whisper** – Speech-to-Text model
- **Llama 3 Vision** – Vision processing (Open-source by Meta)
- **gTTS & ElevenLabs** – Text-to-Speech processing
- **Gradio** – UI development
- **Python** – Backend scripting
- **VS Code** – Development environment

---
## Technical Architecture
1. User inputs queries via **text, voice, or images**.
2. **Speech-to-Text (STT)** transcribes voice queries.
3. **Multimodal LLM** processes text and images.
4. AI generates a medical response.
5. **Text-to-Speech (TTS)** converts AI responses into speech.
6. Response is displayed on the **Gradio UI**.

---
## Improvement Potential / Next Steps
- Integrate **state-of-the-art LLMs** for enhanced vision capabilities.
- **Fine-tune vision models** specifically for medical imaging.
- Add **multilingual support** for wider accessibility.

---
## Installation & Setup
### Prerequisites:
- Python 3.x installed
- API keys for **Groq** and **ElevenLabs**
- Required Python libraries:
  ```bash
  pip install gradio openai groq elevenlabs ffmpeg-python portaudio gtts
  ```

### Running the Project:
```bash
python app.py
```

## Contribution
Feel free to contribute by improving models, adding new functionalities, or optimizing the UI.

## License
This project is licensed under the MIT License.

---
### Contact
For queries or contributions, reach out to the development team.

