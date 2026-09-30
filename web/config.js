// Where the voiceprint model + gallery are served from (Hugging Face Hub CDN in production).
window.QURAA = {
  MODEL_BASE: /^(localhost|127\.0\.0\.1)$/.test(location.hostname)
    ? "models"
    : "https://huggingface.co/KhadijaYahya/quraa-models/resolve/main",
  WHISPER: "onnx-community/whisper-base",
  // Shared feedback database (Supabase, free). Leave empty to keep ratings in the browser only.
  // The anon key is meant to be public: the table only allows inserts (build/feedback_schema.sql).
  SUPABASE_URL: "https://ojnikosdfmwmdjxxihci.supabase.co",
  SUPABASE_KEY: "sb_publishable_klvPudIkTVLDTF0xqov7GQ_qhjpdw5y",
};
