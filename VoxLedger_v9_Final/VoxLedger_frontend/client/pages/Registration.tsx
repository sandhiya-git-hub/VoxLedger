import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Mic, Square, Play, Check, User, Lock, ChevronRight, RefreshCcw, AlertCircle, Loader2, Eye, EyeOff } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { useNavigate } from "react-router-dom";
import { useApp } from "@/context/AppContext";
import * as api from "@/lib/api";

const VOICE_PHRASES = [
  '"Hello VoxLedger, this is my secure voice sample for authentication. I will use this voice to access my finance assistant."',
];

const RECORD_SECONDS = 5;
const MIN_RECORDING_BYTES = 12000;
const MIN_RECORDING_RMS = 0.015;

export default function Registration() {
  const [step, setStep] = useState<"info" | "voice" | "done">("info");
  const [name, setName] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPw, setConfirmPw] = useState("");
  const [showPw, setShowPw] = useState(false);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [apiError, setApiError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [registeredUserId, setRegisteredUserId] = useState<number | null>(null);
  const [phraseIdx, setPhraseIdx] = useState(0);
  const [recordProgress, setRecordProgress] = useState(0);
  const [recordedSamples, setRecordedSamples] = useState<string[]>([]);
  const [uploadingVoice, setUploadingVoice] = useState(false);
  const [recordState, setRecordState] = useState<"idle" | "recording" | "ready">("idle");
  const [waveHeights, setWaveHeights] = useState<number[]>(Array(14).fill(6));

  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const waveformTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const detectedSpeechFramesRef = useRef(0);
  const maxRmsRef = useRef(0);
  const chunksRef = useRef<Blob[]>([]);
  const mimeTypeRef = useRef<string>("audio/webm");
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const navigate = useNavigate();
  const { registerUser, setUserId } = useApp();

  const validateInfo = () => {
    const e: Record<string, string> = {};
    if (!name.trim()) e.name = "Name is required";
    if (password.length < 4) e.password = "Password must be at least 4 characters";
    if (password !== confirmPw) e.confirmPw = "Passwords do not match";
    setErrors(e);
    return Object.keys(e).length === 0;
  };

  const handleInfoNext = async () => {
    if (!validateInfo()) return;
    setIsSubmitting(true);
    setApiError("");
    try {
      const res = await api.registerUser(name.trim(), password);
      setRegisteredUserId(res.user_id);
      registerUser({ id: res.user_id, name: res.user_name, pin: password, voiceSamples: [], registeredAt: new Date().toISOString() });
      setUserId(res.user_id);
      setStep("voice");
    } catch (e: any) {
      const msg = e?.message || "";
      if (msg.toLowerCase().includes("already") || msg.toLowerCase().includes("exist")) {
        setApiError("An account already exists. Please go to the lock screen.");
      } else if (msg.toLowerCase().includes("fetch") || msg.toLowerCase().includes("network")) {
        setApiError("Cannot reach backend. Please ensure the backend server is running on port 8000.");
      } else {
        setApiError(msg || "Registration failed. Please check the backend is running.");
      }
    }
    setIsSubmitting(false);
  };

  const stopWaveMonitoring = () => {
    if (waveformTimerRef.current) { clearInterval(waveformTimerRef.current); waveformTimerRef.current = null; }
    analyserRef.current = null;
    if (audioCtxRef.current) {
      try { audioCtxRef.current.close(); } catch (_) {}
      audioCtxRef.current = null;
    }
    setWaveHeights(Array(14).fill(6));
  };

  const startRecording = async () => {
    setApiError("");
    chunksRef.current = [];
    detectedSpeechFramesRef.current = 0;
    maxRmsRef.current = 0;
    setRecordProgress(0);

    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
      });
    } catch {
      setApiError("Microphone access denied. Please allow microphone permission and try again.");
      return;
    }

    const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
      ? "audio/webm;codecs=opus"
      : MediaRecorder.isTypeSupported("audio/webm")
      ? "audio/webm"
      : "";

    try {
      const ctx = new AudioContext();
      const src = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 256;
      src.connect(analyser);
      audioCtxRef.current = ctx;
      analyserRef.current = analyser;
      const buf = new Float32Array(analyser.fftSize);
      waveformTimerRef.current = setInterval(() => {
        if (!analyserRef.current) return;
        analyserRef.current.getFloatTimeDomainData(buf);
        const rms = Math.sqrt(buf.reduce((s, v) => s + v * v, 0) / buf.length);
        maxRmsRef.current = Math.max(maxRmsRef.current, rms);
        if (rms >= MIN_RECORDING_RMS) detectedSpeechFramesRef.current += 1;
        const base = Math.max(6, Math.min(38, rms * 1400));
        setWaveHeights(Array.from({ length: 14 }, (_, i) => Math.max(6, Math.min(40, base + Math.sin(Date.now() / 140 + i) * 8 + Math.random() * 5))));
      }, 80);
    } catch (_) {}

    const mr = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    mediaRecorderRef.current = mr;
    // Capture mimeType BEFORE stop() clears it
    mimeTypeRef.current = mr.mimeType || mimeType || "audio/webm";

    mr.ondataavailable = (e) => { if (e.data && e.data.size > 0) chunksRef.current.push(e.data); };
    mr.onstop = () => {
      stream.getTracks().forEach((t) => t.stop());
      stopWaveMonitoring();
      setRecordState("ready");
    };
    mr.start(100);
    setRecordState("recording");

    let elapsed = 0;
    const totalSteps = RECORD_SECONDS * 10;
    intervalRef.current = setInterval(() => {
      elapsed++;
      setRecordProgress(Math.min((elapsed / totalSteps) * 100, 100));
      if (elapsed >= totalSteps) {
        clearInterval(intervalRef.current!);
        intervalRef.current = null;
        if (mediaRecorderRef.current?.state !== "inactive") {
          mediaRecorderRef.current?.stop();
        }
      }
    }, 100);
  };

  const stopRecording = () => {
    if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null; }
    if (mediaRecorderRef.current?.state !== "inactive") {
      mediaRecorderRef.current?.stop();
    } else {
      stopWaveMonitoring();
      setRecordState("ready");
    }
  };

  const retakeRecording = () => {
    if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null; }
    if (mediaRecorderRef.current?.state !== "inactive") {
      try { mediaRecorderRef.current?.stop(); } catch (_) {}
    }
    stopWaveMonitoring();
    chunksRef.current = [];
    setRecordProgress(0);
    setRecordState("idle");
    setApiError("");
  };

  const nextPhrase = async () => {
    setApiError("");
    const totalBytes = chunksRef.current.reduce((s, b) => s + b.size, 0);
    if (chunksRef.current.length === 0 || totalBytes < MIN_RECORDING_BYTES) {
      setApiError("Voice sample is too short or too quiet. Please speak clearly and record again.");
      return;
    }
    if (detectedSpeechFramesRef.current < 6 || maxRmsRef.current < MIN_RECORDING_RMS) {
      setApiError("No clear voice was detected. Please record again in a quiet place and speak louder.");
      return;
    }
    if (!registeredUserId) {
      setApiError("User registration incomplete. Please restart.");
      return;
    }
    setUploadingVoice(true);
    try {
      const blob = new Blob(chunksRef.current, { type: mimeTypeRef.current });
      console.log(`[registration] Uploading sample ${phraseIdx + 1}: ${blob.size}B, type: ${mimeTypeRef.current}`);
      await api.uploadVoiceSample(registeredUserId, blob);
    } catch (e: any) {
      setApiError(`Voice upload failed: ${e?.message || "Check your connection and try again."}`);
      setUploadingVoice(false);
      return;
    }
    setUploadingVoice(false);

    const sampleId = `voice_sample_${phraseIdx + 1}_${Date.now()}`;
    const allSamples = [...recordedSamples, sampleId];
    setRecordedSamples(allSamples);

    if (phraseIdx < VOICE_PHRASES.length - 1) {
      setPhraseIdx(phraseIdx + 1);
      chunksRef.current = [];
      setRecordProgress(0);
      setRecordState("idle");
    } else {
      registerUser({ id: registeredUserId, name: name.trim(), pin: password, voiceSamples: allSamples, registeredAt: new Date().toISOString() });
      setStep("done");
    }
  };

  const isRecording = recordState === "recording";

  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      stopWaveMonitoring();
      try { mediaRecorderRef.current?.stream?.getTracks().forEach(t => t.stop()); } catch (_) {}
    };
  }, []);

  return (
    <div className="min-h-screen bg-background flex flex-col items-center justify-center px-6 py-12">
      <div className="w-full max-w-sm">
        <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="flex flex-col items-center mb-10">
          <div className="h-14 w-14 rounded-2xl bg-primary flex items-center justify-center shadow-lg shadow-primary/30 mb-4">
            <Mic className="h-7 w-7 text-white" />
          </div>
          <h1 className="text-2xl font-extrabold tracking-tight">Create Account</h1>
          <p className="text-sm text-muted-foreground mt-1">Set up your VoxLedger profile</p>
        </motion.div>

        <div className="flex gap-2 mb-8">
          {["Your Info", "Voice ID", "Done"].map((s, i) => (
            <div key={s} className="flex-1 flex flex-col items-center gap-1">
              <div className={`h-1 w-full rounded-full transition-colors ${(step === "info" && i === 0) || (step === "voice" && i <= 1) || step === "done" ? "bg-primary" : "bg-secondary"}`} />
              <span className="text-[9px] font-semibold text-muted-foreground uppercase tracking-wider">{s}</span>
            </div>
          ))}
        </div>

        <AnimatePresence mode="wait">
          {step === "info" && (
            <motion.div key="info" initial={{ opacity: 0, x: 30 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -30 }}>
              <Card className="border-none shadow-xl shadow-slate-200/50 rounded-3xl">
                <CardContent className="p-7 space-y-5">
                  {apiError && (
                    <div className="flex items-center gap-2 p-3 rounded-2xl bg-destructive/10 text-destructive text-sm">
                      <AlertCircle className="h-4 w-4 flex-shrink-0" />{apiError}
                    </div>
                  )}
                  <div className="space-y-2">
                    <Label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Full Name</Label>
                    <div className="relative">
                      <User className="absolute left-3.5 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                      <Input value={name} onChange={e => setName(e.target.value)} placeholder="Enter your name" className="pl-10 h-12 rounded-2xl border-border/60 bg-secondary/30 focus:bg-white" />
                    </div>
                    {errors.name && <p className="text-xs text-destructive">{errors.name}</p>}
                  </div>
                  <div className="space-y-2">
                    <Label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Password</Label>
                    <div className="relative">
                      <Lock className="absolute left-3.5 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                      <Input type={showPw ? "text" : "password"} value={password} onChange={e => setPassword(e.target.value)} placeholder="Min. 4 characters" className="pl-10 pr-10 h-12 rounded-2xl border-border/60 bg-secondary/30 focus:bg-white" />
                      <button type="button" onClick={() => setShowPw(v => !v)} className="absolute right-3.5 top-1/2 -translate-y-1/2 text-muted-foreground">
                        {showPw ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                      </button>
                    </div>
                    {errors.password && <p className="text-xs text-destructive">{errors.password}</p>}
                  </div>
                  <div className="space-y-2">
                    <Label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Confirm Password</Label>
                    <div className="relative">
                      <Lock className="absolute left-3.5 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                      <Input type="password" value={confirmPw} onChange={e => setConfirmPw(e.target.value)} placeholder="Repeat password" className="pl-10 h-12 rounded-2xl border-border/60 bg-secondary/30 focus:bg-white" />
                    </div>
                    {errors.confirmPw && <p className="text-xs text-destructive">{errors.confirmPw}</p>}
                  </div>
                  <Button onClick={handleInfoNext} disabled={isSubmitting} className="w-full h-13 rounded-2xl font-bold shadow-lg shadow-primary/20 mt-2">
                    {isSubmitting ? <><Loader2 className="mr-2 h-4 w-4 animate-spin" />Creating Account...</> : <>Next: Set Up Voice ID <ChevronRight className="ml-2 h-4 w-4" /></>}
                  </Button>
                </CardContent>
              </Card>
            </motion.div>
          )}

          {step === "voice" && (
            <motion.div key="voice" initial={{ opacity: 0, x: 30 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -30 }}>
              <div className="text-center mb-6">
                <h2 className="text-lg font-bold">Record Your Voice</h2>
                <p className="text-xs text-muted-foreground mt-1">Speak clearly so Vox can recognize you</p>
              </div>
              <div className="flex gap-2 mb-6">
                {VOICE_PHRASES.map((_, i) => (
                  <div key={i} className={`h-1 flex-1 rounded-full transition-colors ${i < phraseIdx ? "bg-primary" : i === phraseIdx ? "bg-primary/70" : "bg-secondary"}`} />
                ))}
              </div>
              {apiError && (
                <div className="flex items-center gap-2 p-3 mb-4 rounded-2xl bg-destructive/10 text-destructive text-sm">
                  <AlertCircle className="h-4 w-4 flex-shrink-0" />{apiError}
                </div>
              )}
              <Card className="border-none shadow-xl shadow-slate-200/50 rounded-3xl">
                <CardContent className="p-7 flex flex-col items-center text-center">
                  <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">
                    Sample {phraseIdx + 1} of {VOICE_PHRASES.length}
                  </p>
                  <motion.div
                    animate={isRecording ? { scale: [1, 1.08, 1] } : {}}
                    transition={{ duration: 0.8, repeat: Infinity }}
                    className={`h-20 w-20 rounded-full flex items-center justify-center mb-6 shadow-xl transition-colors ${isRecording ? "bg-destructive shadow-destructive/30" : "bg-primary shadow-primary/20"}`}
                  >
                    <Mic className="h-10 w-10 text-white" />
                  </motion.div>
                  <p className="text-base font-semibold italic mb-6 leading-relaxed px-2">{VOICE_PHRASES[phraseIdx]}</p>

                  {isRecording && (
                    <div className="w-full space-y-4 mb-4">
                      <div className="flex justify-center gap-1 h-10 items-end">
                        {waveHeights.map((h, i) => (
                          <motion.div key={i} animate={{ height: h }} transition={{ duration: 0.08 }} className="w-1.5 rounded-full bg-primary" />
                        ))}
                      </div>
                      <Progress value={recordProgress} className="h-1.5" />
                      <p className="text-xs text-muted-foreground">Recording... auto-stops at {RECORD_SECONDS}s, or tap ■</p>
                    </div>
                  )}

                  {recordState === "idle" && (
                    <motion.button whileHover={{ scale: 1.06 }} whileTap={{ scale: 0.94 }} onClick={startRecording} className="h-16 w-16 rounded-full bg-primary text-white flex items-center justify-center shadow-lg shadow-primary/30">
                      <Play className="h-8 w-8 ml-1" />
                    </motion.button>
                  )}

                  {recordState === "recording" && (
                    <Button variant="destructive" size="icon" className="h-12 w-12 rounded-full" onClick={stopRecording}>
                      <Square className="h-5 w-5" />
                    </Button>
                  )}

                  {recordState === "ready" && (
                    <div className="flex gap-3">
                      <Button variant="outline" className="h-11 rounded-2xl border-primary text-primary" onClick={retakeRecording}>
                        <RefreshCcw className="mr-2 h-4 w-4" /> Redo
                      </Button>
                      <Button className="h-11 rounded-2xl px-6" onClick={nextPhrase} disabled={uploadingVoice}>
                        {uploadingVoice
                          ? <><Loader2 className="mr-2 h-4 w-4 animate-spin" />Saving...</>
                          : phraseIdx < VOICE_PHRASES.length - 1
                          ? <>Next <ChevronRight className="ml-2 h-4 w-4" /></>
                          : <>Finish <Check className="ml-2 h-4 w-4" /></>}
                      </Button>
                    </div>
                  )}

                  <p className="text-[11px] text-muted-foreground mt-6">
                    {recordState === "idle" ? "Press ▶ and speak the phrase above." : recordState === "recording" ? "Speak clearly — auto-stops or press ■." : "Recording saved. Press Next or Redo."}
                  </p>
                </CardContent>
              </Card>
            </motion.div>
          )}

          {step === "done" && (
            <motion.div key="done" initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ type: "spring", bounce: 0.4 }} className="text-center py-8">
              <div className="h-24 w-24 bg-primary rounded-full flex items-center justify-center mx-auto shadow-2xl shadow-primary/30 mb-8">
                <Check className="h-12 w-12 text-white" />
              </div>
              <h2 className="text-2xl font-bold mb-3">Welcome, {name}!</h2>
              <p className="text-sm text-muted-foreground mb-10 leading-relaxed px-4">
                Your account is set up with voice recognition. Say <strong>"Hey Vox"</strong> to unlock your finance assistant.
              </p>
              <Button className="w-full h-14 rounded-2xl font-bold shadow-xl shadow-primary/20" onClick={() => navigate("/locked")}>
                Go to Lock Screen
              </Button>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
