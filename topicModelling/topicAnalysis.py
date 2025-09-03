from google import genai
from topicModelling.formatCreation import validateAndTrimBrackets
from dotenv import load_dotenv
import os

# Load variables from .env
load_dotenv()
# Get key from env
api_key = os.getenv("GOOGLE_API_KEY")

def modelTopic(topic: str, text : str) :
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-2.5-flash", 
        contents=f"""
        You are not allowed to output anything except JSON.
        You are a strict JSON generator. You must always respond ONLY in the following JSON format:
        {{
        "score": <number between 0 and 1>,
        "off_topic_suggestions": [<list of strings>] (make sure that the suggestions should be understandable by a common man while showcasing technical jargon too)
        }}

        The "score" must be a decimal with two digits after the decimal point.
        The "off_topic_suggestions" must contain 1 to 3 short actionable suggestions for when the text went off topic and how to stay on topic. Act like a teacher and be supportive. All sentences should be distinct advices
        Topic: {topic}
        Text: {text}
    """
    )

    is_valid, json_string = validateAndTrimBrackets(response.text)
    if is_valid :
        return json_string
    
    else :
        return {}