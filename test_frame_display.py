import base64
import unittest
from io import BytesIO

import numpy as np
from PIL import Image

from frame_display import frame_data_uri


class FrameDisplayTests(unittest.TestCase):
    def test_frame_is_encoded_as_a_decodable_jpeg_data_uri(self):
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        uri = frame_data_uri(frame)

        self.assertTrue(uri.startswith("data:image/jpeg;base64,"))
        image_bytes = base64.b64decode(uri.split(",", 1)[1])
        with Image.open(BytesIO(image_bytes)) as decoded:
            self.assertEqual(decoded.size, (32, 24))
            self.assertEqual(decoded.format, "JPEG")


if __name__ == "__main__":
    unittest.main()
