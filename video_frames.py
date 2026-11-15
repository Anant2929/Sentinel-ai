import cv2


def evenly_spaced_indices(frame_count, sample_count=5):
    if frame_count < 0:
        raise ValueError("frame_count cannot be negative")
    if sample_count < 1:
        raise ValueError("sample_count must be at least 1")
    if frame_count == 0:
        return []

    count = min(frame_count, sample_count)
    if count == 1:
        return [0]

    return [
        round(index * (frame_count - 1) / (count - 1))
        for index in range(count)
    ]


def sample_video_frames(capture, sample_count=5):
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    for frame_index in evenly_spaced_indices(frame_count, sample_count):
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
            raise RuntimeError(f"Could not seek to video frame {frame_index}.")

        success, frame = capture.read()
        if not success:
            raise RuntimeError(f"Could not read video frame {frame_index}.")
        frames.append(frame)

    return frames
