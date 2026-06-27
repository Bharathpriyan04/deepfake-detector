import { useState, useCallback, useRef } from "react";

// ── Change this to your Render backend URL after deploying ──────────────────
const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

// ── Helpers ──────────────────────────────────────────────────────────────────
function ConfidenceBar({ label, value, color }) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-sm font-medium">
        <span className={color}>{label}</span>
        <span className={color}>{value}%</span>
      </div>
      <div className="h-2.5 rounded-full bg-white/10 overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-700 ${
            label.includes("Fake") ? "bg-red-500" : "bg-emerald-500"
          }`}
          style={{ width: `${value}%` }}
        />
      </div>
    </div>
  );
}

function FrameGrid({ frames }) {
  return (
    <div className="mt-4">
      <p className="text-xs text-white/40 uppercase tracking-widest mb-3">
        Frame-by-frame analysis
      </p>
      <div className="grid grid-cols-5 gap-2">
        {frames.map((f, i) => (
          <div
            key={i}
            className={`rounded-xl p-2.5 text-center text-xs font-semibold border ${
              f.verdict === "FAKE"
                ? "bg-red-500/10 border-red-500/30 text-red-400"
                : "bg-emerald-500/10 border-emerald-500/30 text-emerald-400"
            }`}
          >
            <div className="text-[10px] text-white/30 mb-0.5">#{f.frame}</div>
            <div>{f.verdict}</div>
            <div className="text-white/40 font-normal">{f.fake_probability}%</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function StatBadge({ label, value }) {
  return (
    <div className="px-3 py-1.5 rounded-full bg-white/5 border border-white/10 text-center">
      <div className="text-xs text-white/40">{label}</div>
      <div className="text-sm font-bold text-white">{value}</div>
    </div>
  );
}

// ── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [fileType, setFileType] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState(null);
  const inputRef = useRef();

  const handleFile = useCallback((f) => {
    if (!f) return;
    setFile(f);
    setResult(null);
    setError(null);
    const isVideo = f.type.startsWith("video/");
    setFileType(isVideo ? "video" : "image");
    setPreview(URL.createObjectURL(f));
  }, []);

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      setDragOver(false);
      handleFile(e.dataTransfer.files[0]);
    },
    [handleFile]
  );

  const analyze = async () => {
    if (!file) return;
    setLoading(true);
    setResult(null);
    setError(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const endpoint = fileType === "video" ? "/predict/video" : "/predict/image";
      const res = await fetch(API_URL + endpoint, { method: "POST", body: form });
      const data = await res.json();
      if (data.error) setError(data.error);
      else setResult(data);
    } catch {
      setError("Could not reach the server. Check that the backend is running.");
    }
    setLoading(false);
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    setFileType(null);
  };

  const isFake = result?.verdict === "FAKE";

  return (
    <div className="min-h-screen bg-[#0a0a0f] text-white font-sans selection:bg-violet-500/30">
      {/* ── Ambient glow ───────────────────────────────────────────────────── */}
      <div className="pointer-events-none fixed inset-0 overflow-hidden">
        <div className="absolute -top-40 left-1/2 -translate-x-1/2 w-[800px] h-[500px] rounded-full bg-violet-600/10 blur-[120px]" />
        <div className="absolute top-1/3 -left-40 w-[400px] h-[400px] rounded-full bg-blue-600/8 blur-[100px]" />
      </div>

      <div className="relative z-10 max-w-2xl mx-auto px-4 py-16">

        {/* ── Header ──────────────────────────────────────────────────────── */}
        <header className="text-center mb-12">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full
            bg-violet-500/10 border border-violet-500/20 text-violet-300
            text-xs font-semibold tracking-widest uppercase mb-6">
            <span className="w-1.5 h-1.5 rounded-full bg-violet-400 animate-pulse" />
            AI-Powered Detection
          </div>

          <h1 className="text-5xl font-black tracking-tight mb-3 leading-none">
            <span className="text-white">DeepFake</span>
            <span className="bg-gradient-to-r from-violet-400 to-blue-400
              bg-clip-text text-transparent"> Detector</span>
          </h1>

          <p className="text-white/40 text-base mb-6 max-w-md mx-auto leading-relaxed">
            Upload an image or video — our dual-stream neural network analyzes
            both visual content and frequency artifacts to detect manipulation.
          </p>

          {/* Model stats */}
          <div className="flex flex-wrap justify-center gap-2">
            {[
              ["AUC-ROC", "0.8562"],
              ["Accuracy", "75.5%"],
              ["Fake Recall", "89%"],
              ["Architecture", "EfficientNet-B0 + SRM"],
            ].map(([label, value]) => (
              <StatBadge key={label} label={label} value={value} />
            ))}
          </div>
        </header>

        {/* ── Upload zone ─────────────────────────────────────────────────── */}
        {!preview && (
          <div
            onDrop={handleDrop}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onClick={() => inputRef.current?.click()}
            className={`relative rounded-3xl border-2 border-dashed p-16
              text-center cursor-pointer transition-all duration-300
              ${dragOver
                ? "border-violet-400 bg-violet-500/10 scale-[1.01]"
                : "border-white/10 bg-white/[0.02] hover:border-violet-500/50 hover:bg-white/[0.04]"
              }`}
          >
            <input
              ref={inputRef}
              type="file"
              className="hidden"
              accept="image/*,video/*"
              onChange={(e) => handleFile(e.target.files[0])}
            />
            <div className="text-5xl mb-5">🎭</div>
            <p className="text-white/70 text-lg font-semibold mb-1">
              Drop your file here
            </p>
            <p className="text-white/30 text-sm">
              JPG · PNG · MP4 · MOV — or click to browse
            </p>
          </div>
        )}

        {/* ── Preview ─────────────────────────────────────────────────────── */}
        {preview && (
          <div className="rounded-3xl overflow-hidden border border-white/10
            bg-white/[0.02] mb-5">
            {fileType === "video" ? (
              <video src={preview} controls
                className="w-full max-h-72 object-contain bg-black" />
            ) : (
              <img src={preview} alt="preview"
                className="w-full max-h-72 object-contain" />
            )}
            <div className="px-5 py-3 flex items-center justify-between
              border-t border-white/5">
              <span className="text-white/40 text-sm truncate max-w-[60%]">
                {file?.name}
              </span>
              <button
                onClick={reset}
                className="text-white/30 hover:text-white/70 text-sm
                  transition-colors px-3 py-1 rounded-lg hover:bg-white/5"
              >
                × Remove
              </button>
            </div>
          </div>
        )}

        {/* ── Analyze button ──────────────────────────────────────────────── */}
        {file && !result && (
          <button
            onClick={analyze}
            disabled={loading}
            className={`w-full py-4 rounded-2xl font-bold text-base
              transition-all duration-200 mb-5
              ${loading
                ? "bg-white/5 text-white/30 cursor-not-allowed"
                : `bg-gradient-to-r from-violet-600 to-blue-600
                   hover:from-violet-500 hover:to-blue-500
                   shadow-lg shadow-violet-500/20 hover:shadow-violet-500/40
                   hover:-translate-y-0.5 active:translate-y-0`
              }`}
          >
            {loading ? (
              <span className="flex items-center justify-center gap-3">
                <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10"
                    stroke="currentColor" strokeWidth="4"/>
                  <path className="opacity-75" fill="currentColor"
                    d="M4 12a8 8 0 018-8v8H4z"/>
                </svg>
                Analyzing {fileType === "video" ? "video frames" : "image"}…
              </span>
            ) : (
              `🔍 Analyze ${fileType === "video" ? "Video" : "Image"}`
            )}
          </button>
        )}

        {/* ── Error ───────────────────────────────────────────────────────── */}
        {error && (
          <div className="rounded-2xl border border-red-500/30 bg-red-500/10
            p-5 text-red-400 text-sm mb-5">
            ❌ {error}
          </div>
        )}

        {/* ── Result card ─────────────────────────────────────────────────── */}
        {result && (
          <div className={`rounded-3xl border-2 p-8 mb-5
            ${isFake
              ? "border-red-500/40 bg-red-500/5"
              : "border-emerald-500/40 bg-emerald-500/5"
            }`}
          >
            {/* Verdict */}
            <div className="text-center mb-8">
              <div className="text-6xl mb-4">{isFake ? "⚠️" : "✅"}</div>
              <h2 className={`text-5xl font-black tracking-tight mb-1
                ${isFake ? "text-red-400" : "text-emerald-400"}`}>
                {result.verdict}
              </h2>
              <p className="text-white/40 text-sm">
                {result.confidence}% confidence
              </p>
            </div>

            {/* Probability bars */}
            <div className="space-y-4 mb-6">
              <ConfidenceBar
                label="Fake Probability"
                value={result.fake_probability}
                color="text-red-400"
              />
              <ConfidenceBar
                label="Real Probability"
                value={result.real_probability}
                color="text-emerald-400"
              />
            </div>

            {/* Video stats */}
            {result.frame_results && (
              <>
                <div className="flex flex-wrap gap-3 text-sm text-white/50
                  pt-5 border-t border-white/10 mb-1">
                  <span>📹 {result.frames_analyzed} frames analyzed</span>
                  <span>⏱ {result.duration_seconds}s video</span>
                  <span className="text-red-400">
                    🚨 {result.fake_votes} fake votes
                  </span>
                  <span className="text-emerald-400">
                    ✅ {result.real_votes} real votes
                  </span>
                </div>
                <FrameGrid frames={result.frame_results} />
              </>
            )}

            {/* Analyze another */}
            <button
              onClick={reset}
              className="mt-6 w-full py-3 rounded-xl border border-white/10
                text-white/50 hover:text-white hover:border-white/20
                text-sm font-medium transition-all"
            >
              Analyze another file
            </button>
          </div>
        )}

        {/* ── Footer ──────────────────────────────────────────────────────── */}
        <footer className="text-center text-white/20 text-xs mt-8">
          DualStreamDetector · EfficientNet-B0 + SRM · Trained on FaceForensics++
          · Built by Bharath
        </footer>
      </div>
    </div>
  );
}
