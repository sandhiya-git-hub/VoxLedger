/**
 * Locked.tsx — Voice-authenticated lock screen.
 *
 * STRICT AUTH — three independent checks before verify call:
 *  1. RMS energy > 0.015 for at least 4 frames (real vocal energy, not noise floor)
 *  2. Sustained speech ≥ 400ms
 *  3. Audio blob > 12 KB
 */
import { useState, useEffect, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Mic, ShieldCheck, AlertCircle } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { useApp } from "@/context/AppContext";
import * as api from "@/lib/api";

type Phase = "idle" | "listening" | "verifying" | "success" | "failed";

const RECORD_MS = 4500;
const MIN_SPEECH_FRAMES = 8;   // 8 × 100ms = 800ms sustained real speech
const RMS_THRESHOLD = 0.02;    // fallback floor; actual threshold is calibrated at runtime
const MIN_BLOB_BYTES = 12_000;
const CALIBRATION_FRAMES = 8;
const SPEECH_BAND_MIN_HZ = 85;
const SPEECH_BAND_MAX_HZ = 3400;

export default function Locked() {
  const [phase, setPhase] = useState<Phase>("idle");
  const [waveHeights, setWaveHeights] = useState<number[]>(Array(22).fill(4));
  const [dots, setDots] = useState("");
  const [failMessage, setFailMessage] = useState("Please try again");
  const [speechDetected, setSpeechDetected] = useState(false);
  const navigate = useNavigate();
  const { user, authenticateUser } = useApp();

  const chunksRef = useRef<Blob[]>([]);
  const mimeTypeRef = useRef<string>("audio/webm");
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const phaseRef = useRef<Phase>("idle");
  const autoRetryTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const recordTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const startListeningRef = useRef<(() => void) | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const energyIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const speechFramesRef = useRef<number>(0);
  const maxRmsRef = useRef<number>(0);
  const calibratedThresholdRef = useRef<number>(RMS_THRESHOLD);

  useEffect(() => { phaseRef.current = phase; }, [phase]);
  useEffect(() => {
    const t = setInterval(() => setDots(p => p.length >= 3 ? "" : p + "."), 500);
    return () => clearInterval(t);
  }, []);

  const stopAll = useCallback(() => {
    if (energyIntervalRef.current) { clearInterval(energyIntervalRef.current); energyIntervalRef.current = null; }
    if (recordTimer.current) { clearTimeout(recordTimer.current); recordTimer.current = null; }
    if (autoRetryTimer.current) { clearTimeout(autoRetryTimer.current); autoRetryTimer.current = null; }
    if (audioCtxRef.current) { try { audioCtxRef.current.close(); } catch (_) {} audioCtxRef.current = null; }
  }, []);

  const cleanup = useCallback(() => {
    stopAll();
    if (mediaRecorderRef.current) {
      try {
        if (mediaRecorderRef.current.state !== "inactive") {
          mediaRecorderRef.current.stream?.getTracks().forEach(t => t.stop());
          mediaRecorderRef.current.stop();
        }
      } catch (_) {}
      mediaRecorderRef.current = null;
    }
    setWaveHeights(Array(22).fill(4));
    speechFramesRef.current = 0;
    maxRmsRef.current = 0;
    calibratedThresholdRef.current = RMS_THRESHOLD;
    setSpeechDetected(false);
  }, [stopAll]);

  const scheduleRetry = useCallback((msg: string, delayMs = 2800) => {
    if (/register/i.test(msg) || /voice profile/i.test(msg) || /no user/i.test(msg)) {
      navigate("/registration", { replace: true });
      return;
    }
    setFailMessage(msg);
    setPhase("failed");
    autoRetryTimer.current = setTimeout(() => {
      setPhase("idle");
      autoRetryTimer.current = setTimeout(() => startListeningRef.current?.(), 500);
    }, delayMs);
  }, [navigate]);

  const doVerify = useCallback(async () => {
    stopAll();
    setPhase("verifying");

    if (speechFramesRef.current < MIN_SPEECH_FRAMES || maxRmsRef.current < RMS_THRESHOLD) {
      scheduleRetry(`No clear voice detected. Please say "Hey Vox" clearly and loudly.`);
      return;
    }
    const totalBytes = chunksRef.current.reduce((s, b) => s + b.size, 0);
    if (totalBytes < MIN_BLOB_BYTES) {
      scheduleRetry("Audio too short. Please speak for at least 1 second.");
      return;
    }

    try {
      const blob = new Blob(chunksRef.current, { type: mimeTypeRef.current });
      const res = await api.verifyVoice(blob);
      if (res.authenticated) {
        setPhase("success");
        authenticateUser(res.user_id);
        setTimeout(() => navigate("/greeting"), 800);
      } else {
        scheduleRetry(res.message || 'Voice not recognized. Say "Hey Vox" clearly.');
      }
    } catch (_) {
      scheduleRetry("Backend offline. Please ensure the server is running.", 3500);
    }
  }, [stopAll, scheduleRetry, authenticateUser, navigate]);

  const startListening = useCallback(async () => {
    if (phaseRef.current !== "idle") return;
    cleanup();
    chunksRef.current = [];
    speechFramesRef.current = 0;
    setSpeechDetected(false);
    setPhase("listening");

    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, sampleRate: 16000 },
      });
    } catch {
      scheduleRetry("Microphone access denied. Please allow microphone permission.", 3000);
      return;
    }

    // WebAudio RMS energy detection
    try {
      const ctx = new AudioContext({ sampleRate: 16000 });
      audioCtxRef.current = ctx;
      const src = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 1024;
      src.connect(analyser);
      const timeBuf = new Float32Array(analyser.fftSize);
      const freqBuf = new Uint8Array(analyser.frequencyBinCount);
      const binWidth = ctx.sampleRate / analyser.fftSize;
      const lo = Math.max(1, Math.round(SPEECH_BAND_MIN_HZ / binWidth));
      const hi = Math.min(analyser.frequencyBinCount - 1, Math.round(SPEECH_BAND_MAX_HZ / binWidth));
      let calibrationCount = 0;
      let calibrationSum = 0;

      energyIntervalRef.current = setInterval(() => {
        if (phaseRef.current !== "listening") return;
        analyser.getFloatTimeDomainData(timeBuf);
        analyser.getByteFrequencyData(freqBuf);

        const rms = Math.sqrt(timeBuf.reduce((s, v) => s + v * v, 0) / timeBuf.length);
        let speechBandEnergy = 0;
        for (let i = lo; i <= hi; i += 1) speechBandEnergy += freqBuf[i];
        speechBandEnergy /= Math.max(hi - lo + 1, 1);

        if (calibrationCount < CALIBRATION_FRAMES) {
          calibrationCount += 1;
          calibrationSum += rms;
          if (calibrationCount === CALIBRATION_FRAMES) {
            calibratedThresholdRef.current = Math.max(RMS_THRESHOLD, (calibrationSum / CALIBRATION_FRAMES) * 2.4);
          }
          return;
        }

        maxRmsRef.current = Math.max(maxRmsRef.current, rms);
        const looksLikeSpeech = rms > calibratedThresholdRef.current && speechBandEnergy > 18;

        if (looksLikeSpeech) {
          speechFramesRef.current += 1;
          setSpeechDetected(true);
          setWaveHeights(Array(22).fill(0).map(() => Math.round(rms * 1000 + Math.random() * 22 + 4)));
        } else {
          setWaveHeights(prev => prev.map(h => Math.max(4, h * 0.7)));
        }
      }, 100);
    } catch {
      // No WebAudio — blob size check will be the only gate
    }

    const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
      ? "audio/webm;codecs=opus" : "audio/webm";
    const mr = new MediaRecorder(stream, { mimeType });
    mediaRecorderRef.current = mr;
    mimeTypeRef.current = mr.mimeType || mimeType;
    mr.ondataavailable = (e) => { if (e.data.size > 0) chunksRef.current.push(e.data); };
    mr.start(100);

    recordTimer.current = setTimeout(() => {
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
        mediaRecorderRef.current.onstop = () => { stream.getTracks().forEach(t => t.stop()); doVerify(); };
        mediaRecorderRef.current.stop();
      } else {
        stream.getTracks().forEach(t => t.stop());
        doVerify();
      }
    }, RECORD_MS);
  }, [cleanup, doVerify, scheduleRetry]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const status = await api.checkUser();
        if (cancelled) return;
        if (!status.has_user || !status.has_voice_profile) {
          navigate("/registration", { replace: true });
        }
      } catch (_) {
        navigate("/registration", { replace: true });
      }
    })();
    return () => { cancelled = true; };
  }, [navigate]);

  useEffect(() => { startListeningRef.current = startListening; }, [startListening]);
  useEffect(() => {
    const t = setTimeout(() => { if (phaseRef.current === "idle") startListening(); }, 800);
    return () => { clearTimeout(t); cleanup(); };
  }, []);

  const cfg = {
    idle:      { label: `Say "Hey Vox" to unlock`, sublabel: "Tap mic or speak", color: "bg-primary" },
    listening: { label: `Listening${dots}`,          sublabel: speechDetected ? "✓ Voice detected — keep speaking" : 'Speak "Hey Vox" clearly', color: "bg-primary" },
    verifying: { label: "Verifying voice...",        sublabel: "Matching your voice profile",   color: "bg-amber-500" },
    success:   { label: "Voice Verified!",           sublabel: "Unlocking your account",        color: "bg-emerald-500" },
    failed:    { label: "Not recognized",            sublabel: failMessage,                     color: "bg-destructive" },
  }[phase];

  return (
    <div className="min-h-screen bg-primary flex flex-col items-center justify-between px-6 py-16 relative overflow-hidden">
      {[1, 2, 3].map(i => (
        <motion.div key={i} className="absolute rounded-full border border-white/10"
          animate={{ width: [100 + i * 100, 140 + i * 120, 100 + i * 100], height: [100 + i * 100, 140 + i * 120, 100 + i * 100], opacity: [0.1, 0.25, 0.1] }}
          transition={{ duration: 3 + i, repeat: Infinity, ease: "easeInOut" }}
          style={{ left: "50%", top: "50%", transform: "translate(-50%,-50%)" }} />
      ))}

      <div className="flex flex-col items-center mt-10 z-10">
        <div className="h-14 w-14 rounded-2xl bg-white/20 backdrop-blur-sm border border-white/30 flex items-center justify-center shadow-2xl mb-4">
          <Mic className="h-7 w-7 text-white" />
        </div>
        <h1 className="text-2xl font-extrabold text-white tracking-tight">VoxLedger</h1>
        <p className="text-white/60 text-xs font-medium mt-1">Voice-First Finance</p>
      </div>

      <div className="flex flex-col items-center gap-8 z-10">
        <AnimatePresence mode="wait">
          <motion.div key={phase} initial={{ opacity: 0, scale: 0.85 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.85 }} transition={{ duration: 0.25 }} className="flex flex-col items-center gap-6">
            <motion.button whileHover={{ scale: phase === "idle" ? 1.06 : 1 }} whileTap={{ scale: 0.94 }}
              onClick={() => { if (phase === "idle") startListening(); }} disabled={phase !== "idle"} className="relative">
              {(phase === "idle" || phase === "listening") && (
                <>
                  <motion.div animate={{ scale: [1, 1.7], opacity: [0.35, 0] }} transition={{ duration: 1.8, repeat: Infinity }} className={`absolute inset-0 rounded-full ${cfg.color}`} />
                  <motion.div animate={{ scale: [1, 1.35], opacity: [0.2, 0] }} transition={{ duration: 1.8, repeat: Infinity, delay: 0.45 }} className={`absolute inset-0 rounded-full ${cfg.color}`} />
                </>
              )}
              <div className={`h-28 w-28 rounded-full ${cfg.color} flex items-center justify-center shadow-2xl transition-colors duration-500`}>
                {phase === "success" ? <ShieldCheck className="h-14 w-14 text-white" />
                  : phase === "failed" ? <AlertCircle className="h-14 w-14 text-white" />
                  : <Mic className="h-14 w-14 text-white" />}
              </div>
            </motion.button>

            {phase === "listening" && (
              <div className="flex gap-1 items-end justify-center h-14">
                {waveHeights.map((h, i) => (
                  <motion.div key={i} animate={{ height: h }} transition={{ duration: 0.08 }} className="w-1.5 rounded-full bg-white/80" />
                ))}
              </div>
            )}
            {phase === "verifying" && (
              <div className="flex gap-2.5">
                {[0, 1, 2].map(i => (
                  <motion.div key={i} animate={{ opacity: [0.3, 1, 0.3], scale: [0.8, 1.25, 0.8] }} transition={{ duration: 0.8, repeat: Infinity, delay: i * 0.25 }} className="h-3 w-3 rounded-full bg-amber-400" />
                ))}
              </div>
            )}
          </motion.div>
        </AnimatePresence>

        <div className="text-center">
          <h2 className="text-xl font-bold text-white">{cfg.label}</h2>
          <p className={`text-sm mt-1 px-4 ${speechDetected && phase === "listening" ? "text-emerald-300" : "text-white/60"}`}>{cfg.sublabel}</p>
        </div>

        {user && (
          <div className="bg-white/10 backdrop-blur-sm rounded-2xl px-6 py-3 border border-white/20">
            <p className="text-white/80 text-sm font-semibold text-center">👋 Welcome back, {user.name}</p>
          </div>
        )}
      </div>

      <div className="flex flex-col items-center gap-2 z-10">
        <p className="text-white/40 text-xs text-center">Voice authentication protects your financial data</p>
        <p className="text-white/30 text-[11px] text-center">Say "Hey Vox" clearly to unlock</p>
      </div>
    </div>
  );
}
