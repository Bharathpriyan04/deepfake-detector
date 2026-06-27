# 🎭 DeepFake Detector

AI-powered deepfake detection using **DualStreamDetector** — a novel dual-stream architecture combining EfficientNet-B0 (RGB spatial features) with SRM frequency-domain analysis via gated attention fusion.

| Metric | Value |
|--------|-------|
| AUC-ROC | **0.8562** |
| Accuracy | **75.5%** |
| Fake Recall | **89%** |
| Real Precision | **85%** |
| Validation Set | FaceForensics++ (held-out) |

---

## 📁 Project Structure

```
deepfake-detector/
├── backend/
│   ├── main.py                  ← FastAPI server
│   ├── requirements.txt
│   ├── FINAL_MODEL_exp6.pt      ← YOUR MODEL (add this manually)
│   └── src/
│       └── model.py             ← DualStreamDetector architecture
├── frontend/
│   ├── src/
│   │   ├── App.jsx              ← React UI
│   │   ├── main.jsx
│   │   └── index.css
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   ├── vercel.json
│   └── .env.example
├── render.yaml                  ← Render deployment config
├── .gitignore
└── README.md
```

---

## 🚀 Step-by-Step Deployment

### Step 1 — Add your model file

Copy your downloaded `FINAL_MODEL_exp6.pt` into the `backend/` folder:

```
deepfake-detector/
└── backend/
    └── FINAL_MODEL_exp6.pt   ← put it here
```

> ⚠️ The model file (~108 MB) is in `.gitignore` so it won't be pushed to GitHub.
> You will upload it separately to Render as a disk/environment file.

---

### Step 2 — Push to GitHub

```bash
cd deepfake-detector

git init
git add .
git commit -m "Initial commit — DeepFake Detector"

# Create a new repo on github.com, then:
git remote add origin https://github.com/YOUR_USERNAME/deepfake-detector.git
git branch -M main
git push -u origin main
```

---

### Step 3 — Deploy Backend on Render (Free)

1. Go to [render.com](https://render.com) → **New → Web Service**
2. Connect your GitHub account and select `deepfake-detector`
3. Fill in settings:
   - **Root Directory**: `backend`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Click **Create Web Service**
5. After it deploys, go to **Disks** tab → add a disk:
   - **Mount Path**: `/opt/render/project/src`
   - **Size**: 1 GB
6. Upload your `FINAL_MODEL_exp6.pt` via Render Shell:
   ```bash
   # In Render Shell, upload the model:
   # Or use their file upload feature under Files tab
   ```
   **Easier alternative**: Use the Render Shell:
   ```bash
   curl -O https://your-file-host/FINAL_MODEL_exp6.pt
   ```
7. Copy your backend URL, e.g. `https://deepfake-detector-api.onrender.com`

---

### Step 4 — Configure Frontend

1. In the `frontend/` folder, copy `.env.example` to `.env.local`:
   ```bash
   cp frontend/.env.example frontend/.env.local
   ```
2. Edit `.env.local` and set your Render URL:
   ```
   VITE_API_URL=https://deepfake-detector-api.onrender.com
   ```
3. Commit and push:
   ```bash
   git add frontend/.env.local
   git commit -m "Set backend URL"
   git push
   ```
   > Note: `.env.local` is in `.gitignore`, so set `VITE_API_URL` as an environment variable in Vercel instead (see step 5).

---

### Step 5 — Deploy Frontend on Vercel (Free)

1. Go to [vercel.com](https://vercel.com) → **New Project**
2. Import your `deepfake-detector` GitHub repo
3. Set **Root Directory** to `frontend`
4. Add environment variable:
   - **Key**: `VITE_API_URL`
   - **Value**: `https://deepfake-detector-api.onrender.com`
5. Click **Deploy**
6. Your app is live at `https://deepfake-detector.vercel.app` 🎉

---

## 🧪 Run Locally

### Backend
```bash
cd backend
pip install -r requirements.txt

# Make sure FINAL_MODEL_exp6.pt is in backend/
uvicorn main:app --reload
# → http://localhost:8000
# → Docs at http://localhost:8000/docs
```

### Frontend
```bash
cd frontend
npm install

# Create .env.local with:
# VITE_API_URL=http://localhost:8000

npm run dev
# → http://localhost:5173
```

---

## 🏗️ Architecture

```
Input Image/Frame (224×224)
        │
        ├──── RGB Stream ─────────────────────────────────┐
        │     EfficientNet-B0 → 1280-dim features         │
        │                                                  │
        └──── Frequency Stream ───────────────────────────┤
              SRM Filters (3 kernels, 5×5)                │
              → Small CNN → 128-dim → Linear → 1280-dim   │
                                                          │
                          Gated Attention Fusion ◄────────┘
                                    │
                          Dropout → Linear(256) → GELU
                                    │
                              Binary Output
                          (FAKE / REAL probability)
```

**For videos**: 10 frames are sampled evenly, each frame is predicted independently, and the final verdict is the average probability across all frames (majority-vote + mean).

---

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/health` | Liveness probe |
| POST | `/predict/image` | Predict single image |
| POST | `/predict/video` | Predict video (extracts 10 frames) |

### Example response (image)
```json
{
  "verdict": "FAKE",
  "fake_probability": 94.59,
  "real_probability": 5.41,
  "confidence": 94.59
}
```

### Example response (video)
```json
{
  "verdict": "FAKE",
  "fake_probability": 89.3,
  "real_probability": 10.7,
  "confidence": 89.3,
  "frames_analyzed": 10,
  "fake_votes": 9,
  "real_votes": 1,
  "duration_seconds": 7.2,
  "frame_results": [...]
}
```

---

## 📄 Citation

If you use this in research:

```bibtex
@misc{bharath2025deepfake,
  title={DualStreamDetector: Combining Spatial and Frequency Features for Deepfake Detection},
  author={Bharath Priyan},
  year={2025},
  note={EfficientNet-B0 + SRM gated attention fusion, AUC-ROC 0.8562 on FaceForensics++}
}
```
