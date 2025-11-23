import cv2
import numpy as np

def save_video(frames: np.ndarray, path: str, fps: int = 30):
    """
    Сохраняет массив кадров (RGB) в видеофайл.
    frames: numpy массив (num_frames, H, W, 3)
    """
    if len(frames) == 0:
        raise ValueError("Нельзя сохранить пустой список кадров!")

    height, width = frames[0].shape[:2]

    # Кодек mp4 (H.264). Если не работает — можно поменять на 'MJPG'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

    for frame in frames:
        # RGB → BGR перед записью
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr)

    writer.release()
