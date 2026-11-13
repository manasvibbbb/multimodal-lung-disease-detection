import os

from dotenv import load_dotenv
from google import genai

# Loads a local .env file when present. Real keys must never be committed.
load_dotenv()


def _build_client():
    """Create a Gemini client only when a key is configured."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


def explain(prediction, score):
    """Generate a cautious explanation, with a local fallback."""
    prompt = f"""
The AI medical image model returned the following screening result.

Prediction: {prediction}
Model score: {score:.4f}

Explain in simple, cautious language what an AI screening result means.
Do not present this result as a confirmed diagnosis. Mention that image
interpretation and diagnosis must be performed by a qualified clinician.
Avoid claiming a cause that cannot be established from this result.
"""

    client = _build_client()
    if client is not None:
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            if response and response.text:
                return response.text
        except Exception:
            pass

    if prediction == "Abnormal":
        return (
            "The model detected image patterns classified as abnormal. "
            f"Its output score was {score:.4f}. This is an AI screening result, "
            "not a diagnosis. A qualified healthcare professional should "
            "review the images and interpret the result alongside symptoms "
            "and other clinical information."
        )

    return (
        "The model did not classify the supplied image patterns as abnormal. "
        f"Its output score was {score:.4f}. A normal model result does not "
        "rule out disease. A qualified healthcare professional should "
        "interpret the images and any symptoms."
    )
