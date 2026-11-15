import os

import cv2
from dotenv import load_dotenv
from google import genai
from PIL import Image


MODEL = "gemini-3.5-flash-lite"
PROMPT = (
    "You are an AI security system. Analyze this image in two short sentences. "
    "Describe the scene and say whether you notice anything unusual or suspicious."
)


def main():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Error: GEMINI_API_KEY nahi mili. Project ke .env file me set karein.")

    camera = cv2.VideoCapture(0)
    try:
        if not camera.isOpened():
            raise SystemExit(
                "Error: Webcam open nahi hua. Camera connection and Windows camera permissions check karein."
            )
        success, frame = camera.read()
    finally:
        camera.release()

    if not success:
        raise SystemExit("Error: Webcam se frame capture nahi ho paya.")

    client = genai.Client(api_key=api_key)
    print("Photo captured! Gemini scene analyze kar raha hai...\n")
    response = client.models.generate_content(
        model=MODEL,
        contents=[PROMPT, Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))],
    )

    if not response.text:
        raise RuntimeError("Gemini ne koi text response nahi diya.")

    print("--- GEMINI SCENE ANALYSIS ---")
    print(response.text)
    print("-----------------------------")


if __name__ == "__main__":
    main()
