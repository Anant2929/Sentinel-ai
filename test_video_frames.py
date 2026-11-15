import unittest

from video_frames import evenly_spaced_indices, sample_video_frames


class FakeCapture:
    def __init__(self, frames):
        self.frames = frames
        self.position = 0

    def get(self, _property):
        return len(self.frames)

    def set(self, _property, frame_index):
        self.position = frame_index
        return True

    def read(self):
        if self.position >= len(self.frames):
            return False, None
        frame = self.frames[self.position]
        self.position += 1
        return True, frame


class VideoFrameSamplingTests(unittest.TestCase):
    def test_samples_across_full_clip_including_last_frame(self):
        self.assertEqual(evenly_spaced_indices(300), [0, 75, 150, 224, 299])

    def test_short_clip_uses_each_frame_once(self):
        self.assertEqual(evenly_spaced_indices(3), [0, 1, 2])

    def test_empty_clip_returns_no_indices(self):
        self.assertEqual(evenly_spaced_indices(0), [])

    def test_sampling_reads_the_selected_frames(self):
        frames = list(range(300))
        capture = FakeCapture(frames)
        self.assertEqual(
            sample_video_frames(capture),
            [0, 75, 150, 224, 299],
        )

    def test_failed_seek_is_reported(self):
        class FailedSeekCapture(FakeCapture):
            def set(self, _property, _frame_index):
                return False

        with self.assertRaisesRegex(RuntimeError, "Could not seek"):
            sample_video_frames(FailedSeekCapture([1, 2, 3]))

    def test_invalid_sample_count_is_rejected(self):
        with self.assertRaises(ValueError):
            evenly_spaced_indices(10, sample_count=0)


if __name__ == "__main__":
    unittest.main()
