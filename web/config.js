// Where the voiceprint model + gallery are served from (Hugging Face Hub CDN in production).
window.QURAA = {
  MODEL_BASE: /^(localhost|127\.0\.0\.1)$/.test(location.hostname)
    ? "models"
    : "https://huggingface.co/KhadijaYahya/quraa-models/resolve/main",
  WHISPER: "onnx-community/whisper-base",
};
