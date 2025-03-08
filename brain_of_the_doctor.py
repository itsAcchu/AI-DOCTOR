import os
import base64
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = "llama-3.2-90b-vision-preview"

if not GROQ_API_KEY:
    raise ValueError("Missing API key! Set GROQ_API_KEY in your environment.")

# Function to encode image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Function to analyze an image with a query
def analyze_image_with_query(query, encoded_image):
    client = Groq()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},
            ],
        }
    ]
    chat_completion = client.chat.completions.create(messages=messages, model=GROQ_MODEL)
    return chat_completion.choices[0].message.content
