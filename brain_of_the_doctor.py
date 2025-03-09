# brain_of_the_doctor.py

from dotenv import load_dotenv
load_dotenv()

import os
import base64
import time
from groq import Groq
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BrainOfDoctor")

# API Key
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

def encode_image(image_path):
    """
    Convert image to base64 encoding for API submission.
    Handles different image formats and includes error handling.
    """
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding image: {str(e)}")
        raise

def analyze_image_with_query(query, encoded_image, model="llama-3.2-11b-vision-preview", max_retries=3):
    """
    Analyze medical image with enhanced error handling and retry mechanism.
    
    Args:
        query (str): The prompt or question to send with the image
        encoded_image (str): Base64 encoded image string
        model (str): Model to use for analysis
        max_retries (int): Maximum number of retry attempts
    
    Returns:
        str: The analysis response
    """
    client = Groq(api_key=GROQ_API_KEY)
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": query
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}",
                    },
                },
            ],
        }
    ]
    
    # Retry logic
    for attempt in range(max_retries):
        try:
            logger.info(f"Analyzing image with {model} (attempt {attempt+1}/{max_retries})")
            
            # Add temperature parameter for more stable medical analysis
            chat_completion = client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=0.2,  # Lower temperature for more deterministic medical advice
                max_tokens=1024   # Ensure sufficient tokens for detailed analysis
            )
            
            response = chat_completion.choices[0].message.content
            logger.info(f"Analysis completed successfully")
            return response
            
        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed: {str(e)}")
            if attempt < max_retries - 1:
                # Exponential backoff
                wait_time = 2 ** attempt
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"All {max_retries} attempts failed")
                raise Exception(f"Failed to analyze image after {max_retries} attempts: {str(e)}")

# Example usage (commented out for import)
"""
if __name__ == "__main__":
    # Test the image analysis function
    image_path = "test_medical_image.jpg"
    encoded_img = encode_image(image_path)
    
    test_query = "What medical condition is shown in this image?"
    response = analyze_image_with_query(query=test_query, encoded_image=encoded_img)
    
    print("Analysis Result:")
    print(response)
"""