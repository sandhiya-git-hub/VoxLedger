"""
Whisper speech-to-text service.
Model is pre-loaded at backend startup.
Includes audio normalisation and post-processing for accent/broken-English robustness.
"""
import os
import re
import tempfile
from typing import Optional

_whisper_model = None
_whisper_available = False
_load_attempted = False


def preload_model(model_size: str = "small"):
    """Called once at startup. Non-blocking on failure."""
    global _whisper_model, _whisper_available, _load_attempted
    if _load_attempted:
        return
    _load_attempted = True
    try:
        import whisper
        print(f"[whisper] Pre-loading model '{model_size}' at startup ...")
        _whisper_model = whisper.load_model(model_size)
        _whisper_available = True
        print("[whisper] Model ready ✅")
    except Exception as e:
        print(f"[whisper] Whisper not available: {e}")
        print("[whisper] Voice commands will require the user to type instead.")
        _whisper_available = False


def _detect_audio_ext(audio_bytes: bytes) -> str:
    """Detect audio format from magic bytes."""
    if len(audio_bytes) >= 4:
        if audio_bytes[:4] == b'RIFF':
            return '.wav'
        if audio_bytes[:3] == b'ID3' or audio_bytes[:2] == b'\xff\xfb':
            return '.mp3'
        if audio_bytes[:4] == b'OggS':
            return '.ogg'
    return '.webm'  # Default: browser MediaRecorder


def _normalize_audio_bytes(audio_bytes: bytes, ext: str) -> bytes:
    """
    Convert audio bytes to 16kHz mono WAV for Whisper.

    Handles the common WebM fragment problem:
    Chrome MediaRecorder sends chunks where only the VERY FIRST chunk
    contains the EBML container header. If the frontend didn't preserve
    it, we try multiple ffmpeg strategies to decode the raw opus data.

    Strategy order:
    1. Standard ffmpeg decode (works if EBML header is present)
    2. Force-input as webm (explicit format flag)
    3. Force-input as ogg/opus (Chrome's opus stream)
    4. Return original bytes unchanged (Whisper may still handle it)
    """
    import subprocess
    import tempfile as tf

    out_tmp = tf.NamedTemporaryFile(suffix=".wav", delete=False)
    out_tmp.close()

    def _run_ffmpeg(extra_input_args: list, input_path: str) -> bool:
        """Run ffmpeg with given input args → WAV. Returns True on success."""
        try:
            cmd = [
                "ffmpeg", "-y",
                "-threads", "0",          # use all CPU cores
                *extra_input_args,
                "-i", input_path,
                "-ac", "1", "-ar", "16000",
                "-acodec", "pcm_s16le",   # explicit codec avoids auto-detection overhead
                "-f", "wav",
                out_tmp.name,
            ]
            r = subprocess.run(cmd, capture_output=True, timeout=10)  # tighter timeout
            return r.returncode == 0 and os.path.getsize(out_tmp.name) > 100
        except Exception:
            return False

    in_tmp = tf.NamedTemporaryFile(suffix=ext, delete=False)
    in_tmp.write(audio_bytes)
    in_tmp.close()

    success = False

    # Strategy 1: Standard decode (works when EBML header is present)
    if _run_ffmpeg([], in_tmp.name):
        success = True

    # Strategy 2: Force webm container format explicitly
    if not success:
        if _run_ffmpeg(["-f", "webm"], in_tmp.name):
            success = True

    # Strategy 3: Force matroska (superset of webm)
    if not success:
        if _run_ffmpeg(["-f", "matroska"], in_tmp.name):
            success = True

    # Strategy 4: Treat as raw ogg/opus stream
    if not success:
        if _run_ffmpeg(["-f", "ogg"], in_tmp.name):
            success = True

    # Strategy 5: Treat as raw opus (no container)
    if not success:
        if _run_ffmpeg(["-f", "opus"], in_tmp.name):
            success = True

    try:
        os.unlink(in_tmp.name)
    except Exception:
        pass

    if success:
        with open(out_tmp.name, "rb") as f:
            wav_bytes = f.read()
        try:
            os.unlink(out_tmp.name)
        except Exception:
            pass
        return wav_bytes

    # All strategies failed — return original and let Whisper try directly
    try:
        os.unlink(out_tmp.name)
    except Exception:
        pass
    print(f"[whisper] All ffmpeg strategies failed for {ext} ({len(audio_bytes)} bytes) — passing raw to Whisper")
    return audio_bytes


# ── Common STT mis-transcriptions for Indian English ─────────────────────────
_STT_CORRECTIONS = [
    # Numbers
    (r'\bone\s+hundred\b',    '100'),
    (r'\btwo\s+hundred\b',    '200'),
    (r'\bthree\s+hundred\b',  '300'),
    (r'\bfour\s+hundred\b',   '400'),
    (r'\bfive\s+hundred\b',   '500'),
    (r'\bsix\s+hundred\b',    '600'),
    (r'\bseven\s+hundred\b',  '700'),
    (r'\beight\s+hundred\b',  '800'),
    (r'\bnine\s+hundred\b',   '900'),
    (r'\bone\s+thousand\b',   '1000'),
    (r'\btwo\s+thousand\b',   '2000'),
    (r'\bfive\s+thousand\b',  '5000'),
    (r'\bten\s+thousand\b',   '10000'),
    # Indian spoken number forms
    (r'\bek\s+sau\b',         '100'),
    (r'\bdo\s+sau\b',         '200'),
    (r'\bpanch\s+sau\b',      '500'),
    (r'\bek\s+hazar\b',       '1000'),
    (r'\bpaanch\s+hazar\b',   '5000'),
    # Category budget command fixes
    (r'\bfull\s+budget\b',     'food budget'),   # "full budget" → "food budget"
    (r'\bfool\s+budget\b',     'food budget'),   # "fool budget" → "food budget"
    (r'\bflood\s+budget\b',    'food budget'),   # "flood budget" → "food budget"
    (r'\bset\s+full\s+',       'set food '),     # "set full 500" → "set food 500"
    (r'\bset\s+fool\s+',       'set food '),
    (r'\bset\s+flood\s+',      'set food '),
    (r'\btravels?\s+budget\b', 'transport budget'),
    (r'\bhealth\s+budget\b',   'healthcare budget'),
    # Common Whisper mis-hearings
    (r'\brupees?\b',          'rupees'),
    (r'\broopees?\b',         'rupees'),
    (r'\brupeys?\b',          'rupees'),
    (r'\brupies?\b',          'rupees'),
    (r'\bRupees?\b',          'rupees'),
    (r'\badd\s+expense\b',    'add'),
    (r'\bspand\b',            'spend'),  # accent mishear
    (r'\bsband\b',            'spend'),
    (r'\bexpence\b',          'expense'),
    (r'\bexpenses\b',         'expense'),
    # Navigation
    (r'\bopen\s+the\b',       'open'),
    (r'\bgo\s+to\s+the\b',    'go to'),
    (r'\bshow\s+me\s+the\b',  'show'),
]


def _clean_repetition(text: str) -> str:
    """
    Remove Whisper hallucination loops where the same phrase repeats 2+ times.
    Examples:
      "Pay your rent. Pay your rent. Pay your rent." → "Pay your rent."
      "set budget set budget set budget"             → "set budget"
    """
    # Pass 1: sentence-level deduplication
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if len(sentences) >= 2:
        seen = []; seen_lower = []
        for s in sentences:
            key = s.lower().rstrip('.!?').strip()
            if key not in seen_lower:
                seen.append(s); seen_lower.append(key)
        deduped = ' '.join(seen)
        if deduped != text:
            return deduped

    # Pass 2: word-ngram loop detection (handles unpunctuated repeats)
    words = text.split()
    n = len(words)
    for phrase_len in range(2, min(n // 2 + 1, 12)):
        phrase = words[:phrase_len]
        count = 0; i = 0
        while i + phrase_len <= n:
            if words[i:i + phrase_len] == phrase:
                count += 1; i += phrase_len
            else:
                break
        if count >= 2:
            return ' '.join(phrase)

    return text


def _post_process(text: str) -> str:
    """Clean and normalise raw Whisper output."""
    if not text:
        return text

    # Remove common Whisper hallucinations (empty audio artifacts)
    # Also includes strings from our own initial_prompt that Whisper can echo back
    hallucinations = [
        "thank you", "thanks for watching", "thanks for listening",
        "subscribe", "please subscribe", "like and subscribe",
        "bye bye", "goodbye", "see you next time",
        ".", "..", "...", "okay", "um", "uh",
        # Whisper background noise misidentifications
        "music", "background music", "soft music", "instrumental",
        "applause", "crowd noise", "wind blowing", "sound of wind blowing",
        "ambient sound", "white noise", "static",
        # Machine / environment noise hallucinations
        "sound of machine running", "sound of machine gun fire",
        "sound of machine gun", "machine gun fire", "machine running",
        "sound of gunfire", "sound of gun", "gun fire", "gunfire",
        "sound of engine", "engine running", "engine noise",
        "sound of traffic", "traffic noise", "traffic sounds",
        "sound of rain", "sound of thunder", "sound of birds",
        "sound of water", "sound of waves", "sound of fire",
        "sound of clapping", "sound of cheering", "crowd cheering",
        "sound of keyboard", "keyboard typing", "typing sound",
        "sound of beeping", "beeping sound", "alarm sound",
        # YouTube / video hallucinations
        "thanks for watching!", "thanks for watching.",
        "pause", "buffering", "loading",
        # initial_prompt echo-backs (only remove if they CANNOT be real commands)
        "indian english accent",
        "finance app voice command",
        "user may say amounts in rupees",
        "examples",
        "hey vox add two hundred rupees for food",
        # Other common short false positives
        "you", "no", "yes", "what", "the", "a", "an",
    ]
    stripped = text.strip().rstrip('.').lower()
    if stripped in hallucinations or len(stripped) < 2:
        return ""

    # ── Regex-based noise pattern rejection ──────────────────────────────────
    # Whisper generates hundreds of variations of noise descriptions.
    # Catch them all with patterns rather than enumerating every possible string.
    _NOISE_PATTERNS = [
        # "Sound of X" / "Sounds of X" / "the sound of X" — core pattern
        r'^\s*(the\s+)?sounds?\s+of\s+\w',
        # "[music]" / "[applause]" / "[noise]" — bracketed annotations
        r'^\s*\[.{1,40}\]\s*$',
        # "(music playing)" / "(audience noise)" — parenthesised annotations
        r'^\s*\(.{1,40}\)\s*$',
        # "noise", "ambient noise", "background noise", "X noise", "noise of X"
        r'^\s*(background\s+|ambient\s+|engine\s+|machine\s+|crowd\s+|traffic\s+|wind\s+)?noise\s*(of\s+\w+)?\s*$',
        # "X sound" / "X sounds" at end of short phrase (engine sound, traffic sounds)
        r'^\s*[\w\s]{1,25}\s+sounds?\s*$',
        # "audio of X" / "recording of X"
        r'^\s*(audio|recording|clip)\s+of\s+\w',
        # "music playing" / "music in the background"
        r'^\s*\w{1,20}\s+(playing|running|firing|blowing|falling|flowing)\s*$',
    ]
    for _npat in _NOISE_PATTERNS:
        if re.search(_npat, stripped, re.IGNORECASE):
            print(f"[whisper] noise pattern rejected: {repr(text)}")
            return ""

    # Apply corrections
    result = text
    for pattern, replacement in _STT_CORRECTIONS:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    # Remove leading/trailing filler words that don't affect meaning
    result = re.sub(r'^(um+|uh+|ah+|hmm+|okay\s+so|so\s+like|like)\s+', '', result, flags=re.IGNORECASE)
    result = re.sub(r'\s+', ' ', result).strip()

    # Remove hallucination loops (Whisper sometimes repeats phrases)
    result = _clean_repetition(result)

    return result


def transcribe_audio(audio_bytes: bytes, language: str = "en") -> Optional[str]:
    """
    Transcribe audio bytes to text using OpenAI Whisper.
    Includes pre-processing (loudnorm via ffmpeg) and post-processing
    to handle Indian accents, broken English, and common STT errors.
    Returns transcribed string, or None on failure.
    NEVER returns a fake/demo phrase.
    """
    from config import settings
    if not _load_attempted:
        preload_model(settings.WHISPER_MODEL)

    if not _whisper_available or _whisper_model is None:
        return None

    tmp_path = None
    try:
        ext = _detect_audio_ext(audio_bytes)

        # Normalize audio (volume + resample) for better accuracy
        processed_bytes = _normalize_audio_bytes(audio_bytes, ext)
        # After normalization, always a .wav
        final_ext = '.wav' if processed_bytes != audio_bytes else ext

        with tempfile.NamedTemporaryFile(suffix=final_ext, delete=False) as tmp:
            tmp.write(processed_bytes)
            tmp_path = tmp.name

        result = _whisper_model.transcribe(
            tmp_path,
            language=language,     # "en" — tells Whisper to expect English (inc. Indian English)
            fp16=False,            # CPU-safe
            task="transcribe",
            temperature=0.0,       # greedy decode — fastest, most deterministic
            beam_size=1,           # beam=1 = greedy; ~3x faster than beam=5
            best_of=1,
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.65,  # raised: more aggressively reject noise/silence/background audio
            condition_on_previous_text=False,
            # Use all available CPU cores for faster decoding
            # (whisper.transcribe accepts this as a passthrough to torch)
            initial_prompt=(
                "Finance app voice command. Indian English accent. "
                "User may say amounts in rupees. "
                "Examples: add two hundred rupees for food, show my balance, open budget."
            ),
        )

        # Reject if Whisper itself flagged no speech at segment level
        segments = result.get("segments", [])
        if segments:
            avg_no_speech = sum(s.get("no_speech_prob", 0) for s in segments) / len(segments)
            if avg_no_speech > 0.7:
                print(f"[whisper] No-speech probability too high ({avg_no_speech:.2f}) — treating as silence")
                return None

        raw_text = result.get("text", "").strip()
        text = _post_process(raw_text)

        print(f"[whisper] Raw: '{raw_text}' → Processed: '{text}'")
        return text if text else None

    except Exception as e:
        print(f"[whisper] Transcription error: {e}")
        return None

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


def is_available() -> bool:
    return _whisper_available
