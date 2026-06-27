from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch, cv2, numpy as np, sys, os, tempfile

sys.path.insert(0, ".")
from src.model import DualStreamDetector

app = FastAPI(title="Deepfake Detector API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model once at startup ──────────────────────────────────────────────
device = torch.device("cpu")
model = DualStreamDetector(pretrained=False).to(device)

MODEL_PATH = os.environ.get("MODEL_PATH", "FINAL_MODEL_exp6.pt")
if os.path.exists(MODEL_PATH):
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"✅ Model loaded — AUC: {ckpt.get('best_auc', 'N/A')}")
else:
    print(f"⚠️  Model file not found at {MODEL_PATH}")


def predict_frame(frame_rgb: np.ndarray) -> float:
    img = cv2.resize(frame_rgb, (224, 224)).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor)).item()
    return prob


# ── Routes ──────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "Deepfake Detector API is running", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Cannot read image. Make sure it is a valid JPG or PNG."}

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    prob = predict_frame(img_rgb)

    return {
        "verdict": "FAKE" if prob > 0.5 else "REAL",
        "fake_probability": round(prob * 100, 2),
        "real_probability": round((1 - prob) * 100, 2),
        "confidence": round(max(prob, 1 - prob) * 100, 2),
    }


@app.post("/predict/video")
async def predict_video(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if total < 2:
        cap.release()
        os.unlink(tmp_path)
        return {"error": "Video too short or unreadable."}

    frame_indices = np.linspace(
        min(30, total // 4), max(total - 30, total // 2), 10, dtype=int
    )

    probs, frame_results = [], []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        prob = predict_frame(frame_rgb)
        probs.append(prob)
        frame_results.append(
            {
                "frame": int(idx),
                "fake_probability": round(prob * 100, 2),
                "verdict": "FAKE" if prob > 0.5 else "REAL",
            }
        )

    cap.release()
    os.unlink(tmp_path)

    if not probs:
        return {"error": "No frames could be read from the video."}

    avg_prob = float(np.mean(probs))
    fake_votes = sum(1 for p in probs if p > 0.5)

    return {
        "verdict": "FAKE" if avg_prob > 0.5 else "REAL",
        "fake_probability": round(avg_prob * 100, 2),
        "real_probability": round((1 - avg_prob) * 100, 2),
        "confidence": round(max(avg_prob, 1 - avg_prob) * 100, 2),
        "frames_analyzed": len(probs),
        "fake_votes": fake_votes,
        "real_votes": len(probs) - fake_votes,
        "frame_results": frame_results,
        "duration_seconds": round(total / fps, 1) if fps > 0 else 0,
    }
