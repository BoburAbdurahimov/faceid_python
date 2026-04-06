import base64
import numpy as np
import cv2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load OpenCV's pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


class VerifyRequest(BaseModel):
    known_face: str
    check_face: str


def decode_base64_image(base64_str: str) -> np.ndarray:
    """Decode a base64 image string into an OpenCV image (BGR)."""
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]

    img_data = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img


def extract_face(img: np.ndarray) -> np.ndarray | None:
    """Detect and crop the largest face from the image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    if len(faces) == 0:
        return None

    # Pick the largest face
    largest = max(faces, key=lambda rect: rect[2] * rect[3])
    x, y, w, h = largest

    # Add some margin
    margin = int(0.2 * w)
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(img.shape[1], x + w + margin)
    y2 = min(img.shape[0], y + h + margin)

    face_crop = img[y1:y2, x1:x2]
    # Resize to a standard size for comparison
    face_resized = cv2.resize(face_crop, (160, 160))
    return face_resized


def compare_faces_histogram(face1: np.ndarray, face2: np.ndarray) -> float:
    """Compare two face images using color histogram correlation."""
    score = 0.0
    for channel in range(3):
        hist1 = cv2.calcHist([face1], [channel], None, [256], [0, 256])
        hist2 = cv2.calcHist([face2], [channel], None, [256], [0, 256])
        cv2.normalize(hist1, hist1)
        cv2.normalize(hist2, hist2)
        score += cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return score / 3.0


def compare_faces_structural(face1: np.ndarray, face2: np.ndarray) -> float:
    """Compare two face images using structural similarity (SSIM-like via matchTemplate)."""
    gray1 = cv2.cvtColor(face1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(face2, cv2.COLOR_BGR2GRAY)

    # Normalized cross-correlation
    result = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)
    return float(result[0][0])


def compare_faces_orb(face1: np.ndarray, face2: np.ndarray) -> float:
    """Compare two face images using ORB feature matching."""
    gray1 = cv2.cvtColor(face1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(face2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    total = max(len(kp1), len(kp2))
    if total == 0:
        return 0.0

    return len(good_matches) / total


@app.post("/verify-face")
async def verify_face(req: VerifyRequest):
    try:
        known_img = decode_base64_image(req.known_face)
        check_img = decode_base64_image(req.check_face)

        known_face = extract_face(known_img)
        check_face = extract_face(check_img)

        if known_face is None:
            return {"verified": False, "message": "No face found in known_face image"}
        if check_face is None:
            return {"verified": False, "message": "No face found in check_face image"}

        # Multi-method comparison for better accuracy
        hist_score = compare_faces_histogram(known_face, check_face)
        struct_score = compare_faces_structural(known_face, check_face)
        orb_score = compare_faces_orb(known_face, check_face)

        # Weighted average — histogram and structural are most reliable
        combined_score = (hist_score * 0.4) + (struct_score * 0.4) + (orb_score * 0.2)

        # Threshold for match
        threshold = 0.45
        verified = combined_score >= threshold

        return {
            "verified": verified,
            "message": "Verification successful" if verified else "Face mismatch",
            "scores": {
                "histogram": round(hist_score, 4),
                "structural": round(struct_score, 4),
                "orb": round(orb_score, 4),
                "combined": round(combined_score, 4),
            },
        }
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "service": "face-verification"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
