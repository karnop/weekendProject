from google import genai
from topicModelling.formatCreation import validateAndTrimBrackets

def reasoning(average_pause_sec, long_pause_ratio, speech_rate_wps, speech_rate_stability, filler_density, pitch_variability, energy_variability, voicing_ratio, pitch_smoothness, energy_smoothness, composite_fluency_score) :
    client = genai.Client(api_key="AIzaSyD5gL8CLHmTO4wpGbQl3FjFjqTg7FZDDlk")

    input_json = f"""
    {{
    "fluency_metrics": {{
        "timing": {{
        "average_pause_word": {average_pause_sec},
        "long_pause_ratio": {long_pause_ratio},
        "speech_rate_wps": {speech_rate_wps},
        "speech_rate_stability": {speech_rate_stability},
        "filler_density": {filler_density}
        }},
        "prosody": {{
        "pitch_variability": {pitch_variability},
        "energy_variability": {energy_variability},
        "voicing_ratio": {voicing_ratio},
        "pitch_smoothness": {pitch_smoothness},
        "energy_smoothness": {energy_smoothness}
        }},
        "composite_fluency_score": {composite_fluency_score}
    }},
    "instruction_to_llm": "Analyze these metrics and give practical, actionable feedback to help your student speak more fluently. Focus on clear improvement tips rather than generic praise. Act like a teacher and be supportive. (make sure that the suggestions should be understandable by a common man while showcasing technical jargon too)"
    }}
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash", 
        contents=f"""
    You are an expert speech coach. You will receive a JSON object containing detailed fluency metrics for your student speaker's audio recording. 

    1. Analyze the metrics carefully — consider timing (pauses, speech rate), prosody (pitch, energy), and the composite fluency score. 
    2. Identify strengths, weaknesses, and actionable suggestions to improve fluency.
    3. Respond ONLY in valid JSON format, with no additional text outside the JSON.
    4. Do not repeat the raw metrics — interpret them in plain language.

    ### Input JSON:
    {input_json}

    ### Output JSON format:
    {{
      "analysis_summary": {{
        "fluency_level": "Excellent | Good | Fair | Needs Improvement",
        "interpretation": "Brief 1-2 sentence summary of how fluent the speech sounds."
      }},
      "strengths": [
        "List 2-3 positive aspects"
      ],
      "weaknesses": [
        "List 2-3 areas that could improve"
      ],
      "actionable_suggestions": [
        "List 3-4 specific steps or exercises to improve fluency"
      ]
    }}

    Now produce the output JSON:

    """
    )

    iValid, res = validateAndTrimBrackets(response.text)
    return res