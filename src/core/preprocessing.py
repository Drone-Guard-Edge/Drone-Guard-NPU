import cv2
import numpy as np


def center_crop(img: np.ndarray, ratio: float = 0.70) -> np.ndarray:
    h, w = img.shape[:2]
    ch, cw = int(h * ratio), int(w * ratio)
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    return img[y0:y0 + ch, x0:x0 + cw]


def compute_contrast(gray: np.ndarray) -> float:
    return float(gray.std() / 255.0)


def compute_brightness(gray: np.ndarray) -> float:
    return float(gray.mean() / 255.0)


def compute_entropy(gray: np.ndarray) -> float:
    hist, _ = np.histogram(gray.flatten(), 256, (0, 256))
    total = hist.sum()
    if total == 0:
        return 0.0
    prob = hist / total
    prob = prob[prob > 0]
    return float(-np.sum(prob * np.log2(prob)) / 8.0)


def compute_temporal_delta(brightness: float, prev_brightness: float) -> float:
    return round(abs(brightness - prev_brightness), 6)


def extract_features(eo_img: np.ndarray, prev_eo_img: np.ndarray | None) -> np.ndarray:
    """Extract 4-dim feature vector [contrast, entropy, brightness, temporal_delta].

    All values are normalized to roughly [0, 1] to match FusionMLP training inputs.
    """
    gray = cv2.cvtColor(center_crop(eo_img), cv2.COLOR_BGR2GRAY)

    contrast = round(compute_contrast(gray), 6)
    entropy = round(compute_entropy(gray), 6)
    brightness = round(compute_brightness(gray), 6)

    if prev_eo_img is not None:
        prev_gray = cv2.cvtColor(center_crop(prev_eo_img), cv2.COLOR_BGR2GRAY)
        prev_brightness = round(compute_brightness(prev_gray), 6)
        temporal_delta = compute_temporal_delta(brightness, prev_brightness)
    else:
        temporal_delta = 0.0

    return np.array([contrast, entropy, brightness, temporal_delta], dtype=np.float32)


def preprocess_eo(img: np.ndarray, size: int = 640) -> np.ndarray:
    return cv2.resize(img, (size, size))


def preprocess_ir(img: np.ndarray, size: int = 640) -> np.ndarray:
    resized = cv2.resize(img, (size, size))
    if resized.ndim == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    return resized
