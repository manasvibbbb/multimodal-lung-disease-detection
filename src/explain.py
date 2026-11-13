from google import genai

# -----------------------------------
# GEMINI CLIENT
# -----------------------------------
client = genai.Client(
    api_key="AIzaSyALvWNgAj4WIkpN9ZM1aZRYbQqBvg4iO1o"
)

# -----------------------------------
# GENERATE AI EXPLANATION
# -----------------------------------
def explain(prediction, score):

    prompt = f"""
The AI medical system analyzed lung X-ray and CT scan images.

Prediction: {prediction}
Confidence Score: {score:.2f}

Explain:
- what this result means
- possible causes
- precautions
- recommendation to consult a doctor

Use simple medical language.
"""

    try:

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return response.text

    except Exception:

        # -----------------------------------
        # FALLBACK EXPLANATION
        # -----------------------------------

        if prediction == "Abnormal":

            return f"""
The AI system detected possible abnormal patterns in the uploaded lung images.

Confidence Score: {score:.2f}

Possible causes may include:
- pneumonia
- lung infection
- inflammation
- abnormal tissue patterns

Please consult a medical professional for further diagnosis and confirmation.
"""

        else:

            return f"""
The AI system did not detect major abnormal lung patterns in the uploaded images.

Confidence Score: {score:.2f}

The lungs appear relatively normal according to the trained AI model. However, medical consultation is still recommended for proper clinical interpretation.
"""