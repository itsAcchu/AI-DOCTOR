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

def analyze_image_with_query(query, encoded_image, model="meta-llama/llama-4-scout-17b-16e-instruct", max_retries=3):
    """
    Analyze medical image with enhanced error handling and retry mechanism.
    
    Args:
        query (str): The prompt or question to send with the image
        encoded_image (str): Base64 encoded image string
        model (str): Model to use for analysis (updated to current supported models)
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
    
    # Retry logic with fallback models
    models_to_try = [
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "meta-llama/llama-4-maverick-17b-128e-instruct"
    ]
    
    for model_name in models_to_try:
        for attempt in range(max_retries):
            try:
                logger.info(f"Analyzing image with {model_name} (attempt {attempt+1}/{max_retries})")
                
                # Add temperature parameter for more stable medical analysis
                chat_completion = client.chat.completions.create(
                    messages=messages,
                    model=model_name,
                    temperature=0.2,  # Lower temperature for more deterministic medical advice
                    max_completion_tokens=1024   # Updated parameter name
                )
                
                response = chat_completion.choices[0].message.content
                logger.info(f"Analysis completed successfully with {model_name}")
                return response
                
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Attempt {attempt+1} failed with {model_name}: {error_msg}")
                
                # If model is deprecated/decommissioned, try next model immediately
                if "decommissioned" in error_msg.lower() or "not supported" in error_msg.lower():
                    logger.info(f"Model {model_name} is deprecated, trying next model...")
                    break
                
                if attempt < max_retries - 1:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All {max_retries} attempts failed for {model_name}")
    
    # If all models fail, raise final exception
    raise Exception(f"Failed to analyze image after trying all available models: {models_to_try}")

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