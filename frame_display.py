import base64
from io import BytesIO

import cv2
from PIL import Image


def frame_data_uri(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    output = BytesIO()
    image.save(output, format="JPEG", quality=85, optimize=True)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"
