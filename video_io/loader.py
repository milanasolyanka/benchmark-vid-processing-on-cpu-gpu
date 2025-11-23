import cv2
import numpy as np

def load_video(path: str) -> np.ndarray:
    """
    Загружает видеофайл в формате numpy массива.
    Возвращает массив вида (num_frames, height, width, 3), RGB.
    """
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        raise IOError(f"Не удалось открыть видеофайл: {path}")

    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # cv2 читает в BGR → конвертируем в RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    return np.array(frames, dtype=np.uint8)
