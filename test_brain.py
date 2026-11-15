import base64
import os

import cv2
from dotenv import load_dotenv
from google import genai


MODEL = "gemini-4-hydrogen"
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

    success, encoded_frame = cv2.imencode(".jpg", frame)
    if not success:
        raise RuntimeError("Error: Webcam frame ko JPEG me encode nahi kar paya.")

    image_data = base64.b64encode(encoded_frame.tobytes()).decode("ascii")
    client = genai.Client(api_key=api_key)
    print("Photo captured! Gemini scene analyze kar raha hai...\n")
    interaction = client.interactions.create(
        model=MODEL,
        input=[
            {"type": "text", "text": PROMPT},
            {"type": "image", "data": image_data, "mime_type": "image/jpeg"},
        ],
    )

    if not interaction.output_text:
        raise RuntimeError("Gemini ne koi text response nahi diya.")

    print("--- GEMINI SCENE ANALYSIS ---")
    print(interaction.output_text)
    print("-----------------------------")


if __name__ == "__main__":
    main()
