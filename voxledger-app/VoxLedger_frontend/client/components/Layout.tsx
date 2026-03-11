/**
 * Layout.tsx — Voice Engine v9.7  (production-quality fix)
 *
 * ROOT CAUSES FIXED IN THIS VERSION:
 *
 * ① DYNAMIC FREQUENCY BINS (the #1 cause of mic not activating)
 *   Old code hardcoded bins 5–218 assuming sampleRate=16000.
 *   But most browsers run AudioContext at 44100 Hz even when you request 16000.
 *   At 44100 Hz, bins 5–218 = 215 Hz–9388 Hz — completely missing speech energy.
 *   Fix: compute LO/HI at runtime from ctx.sampleRate so bins ALWAYS map to 85–3400 Hz.
 *
 * ② ADAPTIVE NOISE FLOOR (the #2 cause — mic detects noise as speech)
 *   Fixed threshold=18 fails in noisy rooms or with loud microphones.
 *   Fix: sample ambient noise for the first 1.5 s after mic starts, set threshold =
 *        ambient_avg × 1.6 + 4 (a 60% headroom above the noise floor).
 *
 * ③ NAVIGATE-THEN-SPEAK ORDER
 *   Old code called navigate() before playing TTS, so "Opening Budget" played on the
 *   new page (after re-render) and often got blocked by browser autoplay restrictions.
 *   Fix: For navigate intents, speak FIRST. Navigate only in the audio.onended callback.
 *
 * ④ SPEAK-THEN-ACT vs ACT-THEN-SPEAK
 *   Navigation commands: speak confirmation → navigate (user hears it before leaving)
 *   Data commands:       execute action first → refetch → then speak result
 *
 * ⑤ STALE CLOSURE (carried from v9.6)
 *   sendSnapshotRef always holds the latest sendSnapshot function.
 */
import { ReactNode, useState, useEffect, useRef, useCallback } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import {
  Home, CreditCard, Target, Bell, User,
  Mic, X, Volume2, Loader2,
  MessageSquare, PiggyBank, List, AlertCircle, BellRing,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useApp } from "@/context/AppContext";
import * as api from "@/lib/api";

const navItems = [
  { icon: Home,       label: "Home",    path: "/" },
  { icon: CreditCard, label: "Finance", path: "/transactions" },
  { icon: Target,     label: "Budget",  path: "/budget" },
  { icon: Bell,       label: "Alerts",  path: "/alerts" },
  { icon: User,       label: "Profile", path: "/profile" },
];

type VoxState = "idle" | "ready" | "speech" | "processing" | "speaking";

// Recording constants — VAD thresholds are computed dynamically at runtime
const CHUNK_MS           = 200;   // MediaRecorder timeslice ms
const VAD_INTERVAL_MS    = 80;    // VAD poll interval — snappier speech detection
const SPEECH_ON_TICKS    = 2;     // consecutive loud ticks needed to confirm speech (~160ms)
const SILENCE_OFF_TICKS  = 4;     // consecutive quiet ticks after speech to submit (~320ms) — faster cutoff
const MAX_SPEECH_CHUNKS  = 45;    // hard cap ~9 s
const PRE_SPEECH_CHUNKS  = 4;     // pre-speech ring buffer size (800 ms context)
const MIN_SEND_BYTES     = 2000;  // minimum blob bytes to bother sending
const CALIBRATION_TICKS  = 10;   // ticks to sample ambient noise (~800ms — faster startup)
const NOISE_HEADROOM     = 1.8;   // threshold = ambient × NOISE_HEADROOM + 3
const MIN_THRESHOLD      = 12;    // never go below this (very quiet room safety)
const MAX_THRESHOLD      = 50;    // never go above this (very loud environment)

interface LayoutProps { children: ReactNode; }

export default function Layout({ children }: LayoutProps) {
  const navigate  = useNavigate();
  const location  = useLocation();
  const { isAuthenticated, userId, backendOnline,
          refetchTransactions, refetchBudget, addConversation } = useApp();

  const [voxState,  setVoxState]  = useState<VoxState>("idle");
  const [waveH,     setWaveH]     = useState<number[]>(Array(18).fill(4));
  const [lastReply, setLastReply] = useState("");
  const [modalOpen, setModalOpen] = useState(false);

  // ── Audio engine refs ─────────────────────────────────────
  const streamRef    = useRef<MediaStream | null>(null);
  const mrRef        = useRef<MediaRecorder | null>(null);
  const ctxRef       = useRef<AudioContext | null>(null);
  const analyserRef  = useRef<AnalyserNode | null>(null);
  const vadRef       = useRef<ReturnType<typeof setInterval> | null>(null);
  const ttsRef       = useRef<AudioBufferSourceNode | HTMLAudioElement | null>(null);
  const stopRecognitionRef = useRef<any>(null);
  const mimeRef      = useRef("audio/webm");
  const startingRef  = useRef(false);
  const ttsSessionRef = useRef(0);
  const ttsStoppedManuallyRef = useRef(false);

  // ── VAD state refs ────────────────────────────────────────
  const headerChunkRef = useRef<Blob | null>(null); // EBML header chunk — always prepended
  const rollingBuf   = useRef<Blob[]>([]);
  const speechBuf    = useRef<Blob[]>([]);
  const isSpeech     = useRef(false);
  const aboveN       = useRef(0);
  const belowN       = useRef(0);
  const isSending    = useRef(false);

  // ── Dynamic VAD threshold (calibrated per-mic) ────────────
  const vadThreshold  = useRef(22);     // default; overwritten by calibration
  const calibTicks    = useRef(0);      // ticks sampled so far
  const calibSum      = useRef(0);      // sum of ambient readings

  // ── STALE-CLOSURE FIX: always-current function ref ────────
  const sendSnapshotRef = useRef<(chunks: Blob[], mime: string) => void>(() => {});

  const voxRef = useRef<VoxState>("idle");
  useEffect(() => { voxRef.current = voxState; }, [voxState]);

  const ts = () => new Date().toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit" });

  // ── Stop TTS ──────────────────────────────────────────────
  const stopTTS = useCallback(() => {
    ttsStoppedManuallyRef.current = true;
    ttsSessionRef.current += 1;
    const stopRecognizer = stopRecognitionRef.current;
    if (stopRecognizer) {
      try { stopRecognizer.onresult = null; stopRecognizer.onerror = null; stopRecognizer.onend = null; stopRecognizer.stop(); } catch (_) {}
      stopRecognitionRef.current = null;
    }
    if (ttsRef.current) {
      try {
        // Path B: HTMLAudioElement
        if (typeof (ttsRef.current as unknown as HTMLAudioElement).pause === "function") {
          const audioEl = ttsRef.current as unknown as HTMLAudioElement;
          audioEl.pause();
          audioEl.currentTime = 0;
          audioEl.src = "";
          try { audioEl.load(); } catch (_) {}
        } else {
          // Path A: AudioBufferSourceNode
          (ttsRef.current as unknown as AudioBufferSourceNode).stop();
        }
      } catch (_) {}
      ttsRef.current = null;
    }
  }, []);

  const startStopWordListener = useCallback(() => {
    const SpeechRecognitionCtor = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRecognitionCtor) return;

    const existing = stopRecognitionRef.current;
    if (existing) {
      try { existing.stop(); } catch (_) {}
      stopRecognitionRef.current = null;
    }

    try {
      const recognition = new SpeechRecognitionCtor();
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = "en-IN";
      recognition.onresult = (event: any) => {
        for (let i = event.resultIndex; i < event.results.length; i += 1) {
          const transcript = String(event.results[i][0]?.transcript || "").toLowerCase().trim();
          if (/\b(stop|stop it|stop talking|stop speaking|enough|quiet|silence|no more)\b/.test(transcript)) {
            stopTTS();
            setLastReply("");
            isSending.current = false;
            setVoxState("ready");
            try { recognition.abort?.(); } catch (_) {}
            break;
          }
        }
      };
      recognition.onerror = () => {
        try { recognition.stop(); } catch (_) {}
        if (stopRecognitionRef.current === recognition) stopRecognitionRef.current = null;
      };
      recognition.onend = () => {
        if (stopRecognitionRef.current === recognition && voxRef.current === "speaking") {
          try { recognition.start(); } catch (_) {}
        } else if (stopRecognitionRef.current === recognition) {
          stopRecognitionRef.current = null;
        }
      };
      recognition.start();
      stopRecognitionRef.current = recognition;
    } catch (_) {}
  }, [stopTTS]);

  // ── Reset VAD ─────────────────────────────────────────────
  const resetVad = useCallback(() => {
    rollingBuf.current = [];
    speechBuf.current  = [];
    isSpeech.current   = false;
    aboveN.current     = 0;
    belowN.current     = 0;
    isSending.current  = false;
    setWaveH(Array(18).fill(4));
  }, []);

  // ── Play TTS and call onDone when finished ─────────────────
  // Fetches the MP3 via XHR, decodes it through the existing AudioContext
  // (which is always unlocked because the mic started it), then plays it.
  // This completely bypasses browser autoplay restrictions — the AudioContext
  // was unlocked by user gesture (mic permission) so decodeAudioData always works.
  const playTTS = useCallback((text: string, url: string, onDone: () => void) => {
    const sessionId = ttsSessionRef.current + 1;
    ttsSessionRef.current = sessionId;
    ttsStoppedManuallyRef.current = false;
    setLastReply(text);
    setVoxState("speaking");
    startStopWordListener();

    const done = () => {
      if (sessionId !== ttsSessionRef.current) return;
      ttsRef.current = null;
      setLastReply("");
      isSending.current = false;
      setVoxState("ready");
      if (!ttsStoppedManuallyRef.current) onDone();
    };

    const ctx = ctxRef.current;

    // ── Path A: AudioContext available → decode & play through it ──
    // Immune to autoplay restrictions because ctx was unlocked by mic start.
    if (ctx && ctx.state !== "closed") {
      fetch(url)
        .then(r => {
          if (!r.ok) throw new Error(`TTS fetch failed: ${r.status}`);
          return r.arrayBuffer();
        })
        .then(buf => ctx.decodeAudioData(buf))
        .then(decoded => {
          const src = ctx.createBufferSource();
          src.buffer = decoded;
          src.connect(ctx.destination);
          // Store a minimal handle so stopTTS() can cancel it
          (ttsRef as React.MutableRefObject<unknown>).current = src;
          src.onended = done;
          src.start(0);
        })
        .catch(() => {
          // Fallback: estimate silence duration from word count
          setTimeout(done, Math.max(1500, text.split(" ").length * 380));
        });
      return;
    }

    // ── Path B: No AudioContext (rare) → classic Audio element ──
    const audio = new Audio(url);
    ttsRef.current = audio as unknown as AudioBufferSourceNode;
    audio.onended = done;
    audio.onerror = done;
    audio.play().catch(() =>
      setTimeout(done, Math.max(1500, text.split(" ").length * 380))
    );
  }, [startStopWordListener]);

  // ── Core send-snapshot handler ────────────────────────────
  const sendSnapshot = useCallback(async (chunks: Blob[], mime: string) => {
    if (isSending.current) return;
    isSending.current = true;

    const bytes = chunks.reduce((s, b) => s + b.size, 0);
    // Hard-block: never process voice commands while the app is locked.
    // isAuthenticated becomes false the instant lockApp() fires; userId persists,
    // so checking only userId would still allow commands through the lock screen.
    if (bytes < MIN_SEND_BYTES || !userId || !backendOnline || !isAuthenticated) {
      isSending.current = false;
      setVoxState("ready");
      return;
    }

    setVoxState("processing");
    try {
      // Always prepend the EBML header chunk so FFmpeg can parse the WebM blob.
      // Without it, the blob is a headerless fragment and fails to decode.
      const allChunks = headerChunkRef.current
        ? [headerChunkRef.current, ...chunks]
        : chunks;
      const blob = new Blob(allChunks, { type: mime });
      const res  = await api.sendVoiceCommand(userId, blob, "en", false);

      // Show transcribed text immediately
      if (res.transcribed_text?.trim()) {
        addConversation({
          id: `u${Date.now()}`,
          type: "user",
          content: res.transcribed_text.trim(),
          timestamp: ts(),
        });
      }

      // Stop command
      if (res.intent === "stop" || res.action_result?.stop_tts) {
        stopTTS();
        isSending.current = false;
        setVoxState("ready");
        return;
      }

      // Silence / noise / no wake phrase — back to listening silently
      if (!res.response_text || res.intent === "silence" || res.intent === "no_wake_phrase") {
        isSending.current = false;
        setVoxState("ready");
        return;
      }

      // Push assistant reply to chat
      addConversation({
        id: `a${Date.now()}`,
        type: "assistant",
        content: res.response_text,
        timestamp: ts(),
      });

      const ttsUrl    = res.tts_audio_url || api.getTtsUrl(res.response_text);
      const navTarget = res.action_result?.navigate_to;
      const isNavOnly = res.intent === "navigate";
      const speakBeforeNavigate = isNavOnly || res.intent === "read_notifications";

      if (speakBeforeNavigate && navTarget) {
        // Speak the response first, then navigate.
        // This prevents notification refresh from happening before the assistant reads them.
        playTTS(res.response_text, ttsUrl, () => {
          navigate(navTarget, {
            state: res.intent === "read_notifications"
              ? { prefetchedNotifications: res.action_result?.notifications || [], unreadCount: res.action_result?.unread_count || 0 }
              : undefined,
          });
        });
      } else {
        // ── ALL OTHER ACTIONS: execute immediately, then speak result ──
        // Data is already written by the backend. Refresh UI now.
        if (res.action_result?.transaction || res.action_result?.summary ||
            res.action_result?.refresh      || res.action_result?.budget_set) {
          refetchTransactions();
          refetchBudget();
          window.dispatchEvent(new CustomEvent("vox:data-updated"));
        }
        // Dark mode toggle — handle directly in browser
        if (res.action_result?.dark_mode === true) {
          document.documentElement.classList.add("dark");
          localStorage.setItem("vox_dark_mode", "1");
        } else if (res.action_result?.dark_mode === false) {
          document.documentElement.classList.remove("dark");
          localStorage.setItem("vox_dark_mode", "0");
        }

        // Also navigate if needed (e.g. show_transactions navigates to /transactions)
        if (navTarget) navigate(navTarget, {
          state: res.action_result?.start_voice_recording ? { startRecording: true } : undefined,
        });

        // Now speak the result
        playTTS(res.response_text, ttsUrl, () => {
          // nothing extra — ready state is set in playTTS onDone
        });
      }

    } catch {
      isSending.current = false;
      setVoxState("ready");
    }
  }, [userId, backendOnline, isAuthenticated, stopTTS, navigate, playTTS,
      refetchTransactions, refetchBudget, addConversation]);

  // Always keep ref current
  useEffect(() => {
    sendSnapshotRef.current = sendSnapshot;
  }, [sendSnapshot]);

  // ── VAD loop ──────────────────────────────────────────────
  const startVad = useCallback((actualSampleRate: number) => {
    if (vadRef.current) clearInterval(vadRef.current);

    const an = analyserRef.current;
    if (!an) return;

    const freqBuf = new Uint8Array(an.frequencyBinCount);

    // Compute speech-band bin indices from ACTUAL sample rate (may be 44100, not 16000)
    // Speech fundamentals: 85 Hz – 3400 Hz
    const binHz = actualSampleRate / an.fftSize;
    const LO    = Math.max(1, Math.round(85   / binHz));
    const HI    = Math.min(an.frequencyBinCount - 1, Math.round(3400 / binHz));

    console.log(
      `[vox] AudioContext sr=${actualSampleRate} Hz, fftSize=${an.fftSize}, ` +
      `binWidth=${binHz.toFixed(1)} Hz, speech bins ${LO}–${HI}`
    );

    // Reset calibration
    calibTicks.current = 0;
    calibSum.current   = 0;

    vadRef.current = setInterval(() => {
      // Wake suspended AudioContext (tab switch / background)
      const ctx = ctxRef.current;
      if (ctx?.state === "suspended") ctx.resume().catch(() => {});

      const state = voxRef.current;

      an.getByteFrequencyData(freqBuf);

      // Compute speech-band energy
      let sum = 0;
      for (let i = LO; i <= HI; i++) sum += freqBuf[i];
      const avg = sum / (HI - LO + 1);

      // ── PHASE 1: Calibrate ambient noise floor (first CALIBRATION_TICKS ticks) ──
      if (calibTicks.current < CALIBRATION_TICKS) {
        calibTicks.current += 1;
        calibSum.current   += avg;
        if (calibTicks.current === CALIBRATION_TICKS) {
          const ambient = calibSum.current / CALIBRATION_TICKS;
          const computed = Math.round(ambient * NOISE_HEADROOM + 3);
          vadThreshold.current = Math.min(MAX_THRESHOLD, Math.max(MIN_THRESHOLD, computed));
          console.log(
            `[vox] calibrated: ambient=${ambient.toFixed(1)}, ` +
            `threshold=${vadThreshold.current}`
          );
        }
        // Don't run VAD during calibration — just collect data
        return;
      }

      const THRESH = vadThreshold.current;

      // ── Waveform animation ────────────────────────────────
      if (avg > THRESH) {
        const s = Math.min((avg - THRESH) / (THRESH * 2), 1);
        setWaveH(Array(18).fill(0).map(() =>
          Math.round(s * 32 + Math.random() * 10 + 4)
        ));
      } else {
        setWaveH(prev => prev.map(h => Math.max(4, Math.round(h * 0.75))));
      }

      // ── While TTS is speaking, do NOT auto-stop on random noise. ──
      // We keep the microphone active, but only explicit spoken commands
      // that make it through the normal VAD + backend intent parser should stop TTS.
      if (state !== "ready" && state !== "speech" && state !== "speaking") return;

      // ── VAD logic ─────────────────────────────────────────
      if (avg > THRESH) {
        belowN.current = 0;
        aboveN.current += 1;

        if (!isSpeech.current && aboveN.current >= SPEECH_ON_TICKS) {
          isSpeech.current  = true;
          setVoxState("speech");
          // Include pre-speech context
          speechBuf.current = [...rollingBuf.current];
        }

        // Hard time cap
        if (isSpeech.current && speechBuf.current.length >= MAX_SPEECH_CHUNKS) {
          const snap = [...speechBuf.current];
          resetVad();
          sendSnapshotRef.current(snap, mimeRef.current);
        }
      } else {
        if (isSpeech.current) {
          belowN.current += 1;
          if (belowN.current >= SILENCE_OFF_TICKS) {
            const snap = [...speechBuf.current];
            resetVad();
            sendSnapshotRef.current(snap, mimeRef.current);
          }
        } else {
          // Noise spike decay
          aboveN.current = Math.max(0, aboveN.current - 1);
        }
      }
    }, VAD_INTERVAL_MS);
  }, [stopTTS, resetVad]);

  // ── Start persistent microphone ───────────────────────────
  const startMic = useCallback(async () => {
    if (streamRef.current || startingRef.current) return;
    startingRef.current = true;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount:     1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl:  true,
          // Note: sampleRate constraint is a hint, not a guarantee.
          // AudioContext may still use OS default (often 44100 Hz).
          // That is why we compute bins dynamically from ctx.sampleRate below.
        },
      });
      streamRef.current = stream;

      const mime = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : "audio/webm";
      mimeRef.current = mime;

      const mr = new MediaRecorder(stream, { mimeType: mime });
      mrRef.current = mr;

      mr.ondataavailable = (e) => {
        if (!e.data || e.data.size < 10) return;
        // The VERY FIRST chunk contains the WebM EBML header (container metadata).
        // All subsequent chunks are raw Cluster elements with NO container header.
        // FFmpeg cannot parse a fragment-only blob without this header.
        // We permanently store it and prepend it to EVERY snapshot we send.
        if (!headerChunkRef.current) {
          headerChunkRef.current = e.data;
          return; // Don't add the header chunk to speech/rolling buffers
        }
        if (isSpeech.current) {
          speechBuf.current.push(e.data);
        } else {
          rollingBuf.current.push(e.data);
          if (rollingBuf.current.length > PRE_SPEECH_CHUNKS) {
            rollingBuf.current.shift();
          }
        }
      };

      mr.onerror = () => {
        stopMicFn();
        setTimeout(startMic, 1500);
      };

      mr.start(CHUNK_MS);

      // Web Audio — do NOT force sampleRate; let browser choose its native rate.
      // We adapt the bin calculation to whatever rate it picks.
      const ctx = new AudioContext();
      ctxRef.current = ctx;
      const src = ctx.createMediaStreamSource(stream);
      const an  = ctx.createAnalyser();
      an.fftSize             = 2048;   // larger FFT = better frequency resolution
      an.smoothingTimeConstant = 0.4;  // moderate smoothing — responsive but not jittery
      src.connect(an);
      analyserRef.current = an;

      startingRef.current = false;
      setVoxState("ready");
      // Pass the ACTUAL sample rate so VAD computes correct bins
      startVad(ctx.sampleRate);

    } catch (err) {
      startingRef.current = false;
      console.warn("[vox] mic error:", err);
      setTimeout(startMic, 3000);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [startVad]);

  // ── Stop microphone ───────────────────────────────────────
  const stopMicFn = useCallback(() => {
    if (vadRef.current)    { clearInterval(vadRef.current); vadRef.current = null; }
    if (ctxRef.current)    { try { ctxRef.current.close(); } catch (_) {} ctxRef.current = null; }
    analyserRef.current    = null;
    if (mrRef.current)     { try { if (mrRef.current.state !== "inactive") mrRef.current.stop(); } catch (_) {} mrRef.current = null; }
    if (streamRef.current) { streamRef.current.getTracks().forEach(t => t.stop()); streamRef.current = null; }
    startingRef.current    = false;
    headerChunkRef.current = null; // clear so next startMic re-captures header
    resetVad();
  }, [resetVad]);

  // ── Auth lifecycle ────────────────────────────────────────
  useEffect(() => {
    if (isAuthenticated && userId) {
      startMic();
    } else {
      stopTTS();
      stopMicFn();
      setVoxState("idle");
      setLastReply("");
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isAuthenticated, userId]);

  // ── Force-stop (auto-lock screen) ────────────────────────
  useEffect(() => {
    const h = () => {
      stopTTS(); stopMicFn();
      setVoxState("idle"); setLastReply(""); setModalOpen(false);
    };
    window.addEventListener("vox:force-stop", h);
    return () => window.removeEventListener("vox:force-stop", h);
  }, [stopTTS, stopMicFn]);

  // ── Cleanup ───────────────────────────────────────────────
  useEffect(() => () => { stopTTS(); stopMicFn(); }, []); // eslint-disable-line

  const quickAction = (path: string) => { setModalOpen(false); navigate(path); };

  // ── UI derived values ─────────────────────────────────────
  const fabBg =
    voxState === "speaking"   ? "bg-emerald-500 shadow-emerald-300/60" :
    voxState === "processing" ? "bg-amber-500   shadow-amber-300/60"   :
    voxState === "speech"     ? "bg-primary     shadow-primary/60"     :
    voxState === "ready"      ? "bg-primary     shadow-primary/40"     :
                                "bg-slate-400   shadow-slate-300/40";

  const fabIcon =
    voxState === "processing" ? <Loader2 className="h-6 w-6 text-white animate-spin" /> :
    voxState === "speaking"   ? <Volume2 className="h-6 w-6 text-white" />              :
                                <Mic     className="h-6 w-6 text-white" />;

  const modalTitle =
    voxState === "speech"     ? "Listening…"    :
    voxState === "processing" ? "Processing…"   :
    voxState === "speaking"   ? "Speaking…"     :
    voxState === "ready"      ? "Vox is ready"  : "Vox Assistant";

  const modalStatus =
    voxState === "processing" ? "Understanding your command…"                        :
    voxState === "speaking"   ? lastReply.slice(0, 90) + (lastReply.length > 90 ? "…" : "") :
    voxState === "speech"     ? "I'm listening — go ahead…"                          :
    voxState === "ready"      ? "Speak anytime — no button needed"                   :
                                "Sign in to activate Vox";

  return (
    <div className="relative min-h-screen bg-background font-sans overflow-x-hidden">
      <main className="pb-28">{children}</main>

      {/* ── Bottom Navigation ─────────────────────────────── */}
      <nav className="fixed bottom-0 left-0 right-0 z-20 bg-background/90 backdrop-blur-xl border-t border-border/50">
        <div className="flex items-center justify-around px-1 py-3 max-w-lg mx-auto">
          {navItems.map((item) => {
            const Icon     = item.icon;
            const isActive = location.pathname === item.path;
            return (
              <button key={item.path} onClick={() => navigate(item.path)}
                className="flex flex-col items-center gap-1 px-3 py-1 relative">
                <div className={cn("h-10 w-10 rounded-2xl flex items-center justify-center transition-all",
                  isActive ? "bg-primary/10" : "bg-transparent")}>
                  <Icon className={cn("h-5 w-5", isActive ? "text-primary" : "text-muted-foreground")} />
                </div>
                <span className={cn("text-[10px] font-semibold",
                  isActive ? "text-primary" : "text-muted-foreground")}>{item.label}</span>
                {isActive && (
                  <motion.div layoutId="nav-dot"
                    className="absolute -bottom-3 left-1/2 -translate-x-1/2 h-1 w-6 rounded-full bg-primary" />
                )}
              </button>
            );
          })}
        </div>
      </nav>

      {/* ── Floating Mic FAB ──────────────────────────────── */}
      <motion.button
        whileTap={{ scale: 0.88 }}
        onClick={() => setModalOpen(true)}
        className={cn(
          "fixed bottom-24 right-5 z-30 h-14 w-14 rounded-full flex items-center justify-center shadow-xl transition-colors duration-300",
          fabBg
        )}
      >
        {(voxState === "speech" || voxState === "speaking") && (
          <motion.div
            animate={{ scale: [1, 1.65], opacity: [0.4, 0] }}
            transition={{ duration: 1.3, repeat: Infinity, ease: "easeOut" }}
            className="absolute inset-0 rounded-full bg-white/25"
          />
        )}
        {fabIcon}
      </motion.button>

      {/* ── Vox Modal ─────────────────────────────────────── */}
      <AnimatePresence>
        {modalOpen && (
          <>
            <motion.div
              initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
              onClick={() => setModalOpen(false)}
              className="fixed inset-0 z-40 bg-black/30 backdrop-blur-sm"
            />
            <motion.div
              initial={{ opacity: 0, y: 60, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 60, scale: 0.95 }}
              transition={{ type: "spring", damping: 22, stiffness: 280 }}
              className="fixed bottom-28 left-4 right-4 z-50 bg-white rounded-3xl shadow-2xl overflow-hidden"
              style={{ maxWidth: 480, margin: "0 auto" }}
            >
              <button onClick={() => setModalOpen(false)}
                className="absolute top-3 right-3 h-8 w-8 rounded-full bg-gray-100 flex items-center justify-center z-10">
                <X className="h-4 w-4 text-gray-500" />
              </button>

              <div className="px-6 pt-8 pb-6 flex flex-col items-center">
                {/* Avatar with pulse ring */}
                <div className="relative mb-4">
                  {(voxState === "speech" || voxState === "speaking") && (
                    <motion.div
                      animate={{ scale: [1, 1.4], opacity: [0.3, 0] }}
                      transition={{ duration: 1.2, repeat: Infinity }}
                      className="absolute inset-0 rounded-full bg-primary/40"
                    />
                  )}
                  <div className={cn(
                    "h-20 w-20 rounded-full flex items-center justify-center shadow-lg transition-colors duration-300",
                    voxState === "speaking"   ? "bg-emerald-500" :
                    voxState === "processing" ? "bg-amber-500"   :
                    voxState === "speech"     ? "bg-primary"     : "bg-primary/15"
                  )}>
                    {voxState === "processing"
                      ? <Loader2 className="h-9 w-9 text-white animate-spin" />
                      : voxState === "speaking"
                      ? <Volume2 className="h-9 w-9 text-white" />
                      : <Mic className={cn("h-9 w-9",
                          voxState === "speech" ? "text-white" : "text-primary")} />
                    }
                  </div>
                </div>

                {/* Waveform */}
                {(voxState === "speech" || voxState === "speaking") && (
                  <div className="flex gap-0.5 items-end justify-center h-10 mb-3">
                    {waveH.map((h, i) => (
                      <motion.div key={i}
                        animate={{ height: Math.max(4, Math.min(h, 38)) }}
                        transition={{ duration: 0.08 }}
                        className={cn("w-1.5 rounded-full",
                          voxState === "speaking" ? "bg-emerald-400" : "bg-primary")}
                      />
                    ))}
                  </div>
                )}

                <h2 className="text-lg font-bold text-gray-900 mb-1">{modalTitle}</h2>
                <p className="text-sm text-gray-500 text-center mb-4 px-2 leading-relaxed min-h-[2.5rem]">
                  {modalStatus}
                </p>

                {/* Live status pill */}
                <div className={cn(
                  "flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-semibold mb-4",
                  voxState === "idle"       ? "bg-slate-100 text-slate-500"   :
                  voxState === "ready"      ? "bg-emerald-50 text-emerald-700" :
                  voxState === "speech"     ? "bg-primary/10 text-primary"     :
                  voxState === "processing" ? "bg-amber-50 text-amber-700"     :
                                             "bg-emerald-50 text-emerald-700"
                )}>
                  <span className={cn("h-2 w-2 rounded-full",
                    voxState === "idle"       ? "bg-slate-400"              :
                    voxState === "ready"      ? "bg-emerald-500 animate-pulse" :
                    voxState === "speech"     ? "bg-primary animate-ping"   :
                    voxState === "processing" ? "bg-amber-500"              : "bg-emerald-500"
                  )} />
                  {voxState === "idle"       ? "Mic off" :
                   voxState === "ready"      ? "Mic on — always listening" :
                   voxState === "speech"     ? "Your voice detected" :
                   voxState === "processing" ? "Sending to AI…" : "Speaking response"}
                </div>

                {/* Threshold debug info (dev aid) */}
                {voxState === "ready" && (
                  <p className="text-[10px] text-slate-400 mb-3">
                    Noise threshold calibrated ✓ — just speak naturally
                  </p>
                )}

                <div className="flex gap-3 w-full mb-4">
                  <button onClick={() => quickAction("/conversation")}
                    className="flex-1 h-12 rounded-2xl border-2 border-gray-200 text-gray-700 font-semibold text-sm hover:bg-gray-50 transition-colors flex items-center justify-center gap-2">
                    <MessageSquare className="h-4 w-4" />Open Chat
                  </button>
                  <button onClick={() => setModalOpen(false)}
                    className={cn(
                      "flex-1 h-12 rounded-2xl font-semibold text-sm flex items-center justify-center gap-2",
                      voxState === "speech"     ? "bg-primary text-white"     :
                      voxState === "processing" ? "bg-amber-500 text-white"   :
                      voxState === "speaking"   ? "bg-emerald-500 text-white" :
                                                  "bg-primary text-white"
                    )}>
                    {voxState === "speech"
                      ? <><span className="h-2 w-2 rounded-full bg-white animate-ping mr-1" />Listening…</>
                      : voxState === "processing"
                      ? <><Loader2 className="h-4 w-4 animate-spin" />Processing</>
                      : voxState === "speaking"
                      ? <><Volume2 className="h-4 w-4" />Speaking…</>
                      : <><Mic className="h-4 w-4" />Close</>
                    }
                  </button>
                </div>

                <div className="grid grid-cols-2 gap-2.5 w-full">
                  {[
                    { label: "Open Budget",   icon: PiggyBank,   path: "/budget" },
                    { label: "Transactions",  icon: List,        path: "/transactions" },
                    { label: "My Alerts",     icon: AlertCircle, path: "/alerts" },
                    { label: "Notifications", icon: BellRing,    path: "/notifications" },
                  ].map(({ label, icon: Icon, path }) => (
                    <button key={path} onClick={() => quickAction(path)}
                      className="h-11 rounded-2xl bg-gray-50 hover:bg-gray-100 text-gray-700 text-sm font-medium flex items-center justify-center gap-2 transition-colors border border-gray-100">
                      <Icon className="h-4 w-4 text-gray-500" />{label}
                    </button>
                  ))}
                </div>
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </div>
  );
}
