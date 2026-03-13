"""
Music-Aware Similarity Check Class  (Part 2)

Workflow:
  Reference song + Generated song
    ├─ Audio embedding backbone  →  audio↔audio similarity   [Branch A]
    ├─ Lyrics text (if available) →  lyrics semantic similarity [Branch B]
    └─ Optional ASR (Whisper)    →  lyrics-from-audio similarity [Branch C]
         └─ Score fusion + threshold calibration
              └─ music_similarity_score  +  decision_flag
                 "similar" | "risky" | "near-duplicate"
"""

from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np


# ---------------------------------------------------------------------------
# Optional heavy dependencies (graceful degradation)
# ---------------------------------------------------------------------------

# Branch A: audio embeddings via CLAP (laion_clap or transformers)
try:
    import laion_clap
    _CLAP_LAION_AVAILABLE = True
except ImportError:
    _CLAP_LAION_AVAILABLE = False

try:
    from transformers import ClapModel, ClapProcessor
    import torch
    _CLAP_HF_AVAILABLE = True
except ImportError:
    _CLAP_HF_AVAILABLE = False

# Audio loading
try:
    import librosa
    _LIBROSA_AVAILABLE = True
except ImportError:
    _LIBROSA_AVAILABLE = False

# Branch C: ASR lyrics extraction via Whisper
try:
    import whisper as openai_whisper
    _WHISPER_AVAILABLE = True
except ImportError:
    _WHISPER_AVAILABLE = False

# Branch B: lyrics similarity reuses Part 1
try:
    from EVAL.lyrics import LyricsSimilarityChecker
    _LYRICS_CHECKER_AVAILABLE = True
except ImportError:
    try:
        from lyrics import LyricsSimilarityChecker
        _LYRICS_CHECKER_AVAILABLE = True
    except ImportError:
        _LYRICS_CHECKER_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AudioSegments:
    """Holds audio metadata and the raw waveform (if loaded)."""
    path: str
    sample_rate: int = 22050
    waveform: Optional[np.ndarray] = None           # shape: (samples,)
    duration_seconds: float = 0.0
    transcribed_lyrics: Optional[str] = None        # filled by ASR branch


@dataclass
class MusicSimilarityResult:
    """Final output of the music-aware similarity pipeline."""
    audio_similarity: float              # 0-1, audio embedding cosine similarity
    lyrics_similarity: float             # 0-1, text lyrics branch (NaN if unavailable)
    lyrics_from_audio_similarity: float  # 0-1, ASR branch (NaN if unavailable)
    music_similarity_score: float        # 0-1, fused score
    decision_flag: str                   # "similar" | "risky" | "near-duplicate"
    details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helper: audio loader
# ---------------------------------------------------------------------------

AudioInput = Union[str, os.PathLike]


def _load_audio(source: AudioInput, target_sr: int = 22050) -> AudioSegments:
    """
    Load an audio file into a mono float32 waveform.
    Requires librosa. Returns an AudioSegments with waveform=None if unavailable.
    """
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {source}")

    seg = AudioSegments(path=str(path), sample_rate=target_sr)

    if _LIBROSA_AVAILABLE:
        waveform, sr = librosa.load(str(path), sr=target_sr, mono=True)
        seg.waveform = waveform
        seg.sample_rate = sr
        seg.duration_seconds = len(waveform) / sr
    # else: waveform stays None; audio branches will return zero vectors

    return seg


# ---------------------------------------------------------------------------
# Branch A: Audio embedding backbone (CLAP)
# ---------------------------------------------------------------------------

_CLAP_HF_MODEL = "laion/larger_clap_music"   # music-specific CLAP checkpoint


class AudioEmbedder:
    """
    Wraps CLAP (Contrastive Language-Audio Pretraining) for audio embeddings.

    Priority:
      1. laion_clap  (pip install laion_clap)
      2. transformers ClapModel  (pip install transformers torch)
      3. Librosa MFCC fallback
    """

    def __init__(self, model_name: str = _CLAP_HF_MODEL):
        self._model_name = model_name
        self._laion_model = None
        self._hf_model = None
        self._hf_processor = None
        self._device = "cpu"

        if _CLAP_LAION_AVAILABLE:
            try:
                self._laion_model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
                self._laion_model.load_ckpt()  # downloads music checkpoint automatically
            except Exception:
                self._laion_model = None

        if self._laion_model is None and _CLAP_HF_AVAILABLE:
            try:
                self._hf_processor = ClapProcessor.from_pretrained(model_name)
                self._hf_model = ClapModel.from_pretrained(model_name)
                self._hf_model.eval()
                if torch.cuda.is_available():
                    self._device = "cuda"
                    self._hf_model = self._hf_model.to(self._device)
            except Exception:
                self._hf_model = None
                self._hf_processor = None

    # ------------------------------------------------------------------
    def _mfcc_fallback(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """
        MFCC-based fallback embedding when CLAP is unavailable.
        Returns L2-normalised 40-coefficient mean MFCC vector.
        """
        if not _LIBROSA_AVAILABLE:
            return np.zeros(40)
        mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=40)
        vec = mfcc.mean(axis=1)
        norm = np.linalg.norm(vec) + 1e-9
        return vec / norm

    def embed(self, seg: AudioSegments) -> np.ndarray:
        """
        Produce a normalised embedding vector for one audio track.
        Returns a zero vector if the audio could not be loaded.
        """
        if seg.waveform is None:
            return np.zeros(512)

        # --- laion_clap path ---
        if self._laion_model is not None:
            try:
                # laion_clap expects audio at 48kHz
                if _LIBROSA_AVAILABLE and seg.sample_rate != 48000:
                    audio_48k = librosa.resample(seg.waveform, orig_sr=seg.sample_rate, target_sr=48000)
                else:
                    audio_48k = seg.waveform
                emb = self._laion_model.get_audio_embedding_from_data(
                    [audio_48k], use_tensor=False
                )
                vec = np.array(emb[0], dtype=float)
                norm = np.linalg.norm(vec) + 1e-9
                return vec / norm
            except Exception:
                pass

        # --- HuggingFace transformers CLAP path ---
        if self._hf_model is not None and self._hf_processor is not None:
            try:
                inputs = self._hf_processor(
                    audios=seg.waveform,
                    sampling_rate=seg.sample_rate,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                with torch.no_grad():
                    emb = self._hf_model.get_audio_features(**inputs)
                vec = emb[0].cpu().numpy().astype(float)
                norm = np.linalg.norm(vec) + 1e-9
                return vec / norm
            except Exception:
                pass

        # --- MFCC fallback ---
        return self._mfcc_fallback(seg.waveform, seg.sample_rate)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ---------------------------------------------------------------------------
# Branch C: ASR / lyrics-from-audio (Whisper)
# ---------------------------------------------------------------------------

_WHISPER_MODEL_SIZE = "base"   # tiny | base | small | medium | large


class ASRTranscriber:
    """
    Wraps OpenAI Whisper for lyrics extraction from audio.
    Falls back gracefully when Whisper is not installed.
    """

    def __init__(self, model_size: str = _WHISPER_MODEL_SIZE):
        self._model = None
        if _WHISPER_AVAILABLE:
            try:
                self._model = openai_whisper.load_model(model_size)
            except Exception:
                self._model = None

    @property
    def available(self) -> bool:
        return self._model is not None

    def transcribe(self, audio_path: str) -> Optional[str]:
        """
        Transcribe an audio file to text using Whisper.
        Returns None if Whisper is unavailable or transcription fails.
        """
        if self._model is None:
            return None
        try:
            result = self._model.transcribe(audio_path, fp16=False)
            return result.get("text", "").strip()
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Helper: text normalisation (mirrors lyrics.py _normalize)
# ---------------------------------------------------------------------------

_SECTION_TAG_RE = re.compile(r"\[.*?\]", flags=re.IGNORECASE)


def _normalize_text(text: str) -> str:
    text = _SECTION_TAG_RE.sub("", text)
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.casefold()
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Score fusion + threshold calibration
# ---------------------------------------------------------------------------

# Default fusion weights (sum to 1.0)
# When a branch is unavailable its weight is redistributed proportionally.
_FUSION_WEIGHTS = {
    "audio":             0.40,
    "lyrics":            0.35,
    "lyrics_from_audio": 0.25,
}

_THRESHOLDS = {
    "near-duplicate": 0.75,
    "risky":          0.45,
}


def _fuse_scores(
    audio: float,
    lyrics: Optional[float],
    lyrics_from_audio: Optional[float],
    weights: dict | None = None,
) -> float:
    """
    Weighted fusion of available branch scores.
    Unavailable branches (None) have their weight redistributed to the others.
    """
    w = dict(weights or _FUSION_WEIGHTS)

    active: dict[str, float] = {"audio": audio}
    if lyrics is not None:
        active["lyrics"] = lyrics
    else:
        w.pop("lyrics", None)
    if lyrics_from_audio is not None:
        active["lyrics_from_audio"] = lyrics_from_audio
    else:
        w.pop("lyrics_from_audio", None)

    total_w = sum(w[k] for k in active)
    if total_w == 0:
        return 0.0

    score = sum(w[k] * active[k] for k in active) / total_w
    return float(np.clip(score, 0.0, 1.0))


def _decision(score: float, thresholds: dict | None = None) -> str:
    t = thresholds or _THRESHOLDS
    if score >= t["near-duplicate"]:
        return "near-duplicate"
    if score >= t["risky"]:
        return "risky"
    return "similar"


# ---------------------------------------------------------------------------
# Public API: MusicAwareSimilarityChecker
# ---------------------------------------------------------------------------

class MusicAwareSimilarityChecker:
    """
    End-to-end music-aware similarity and plagiarism-risk pipeline.

    Three parallel branches are fused into a single score:
      A) Audio↔Audio embedding similarity (CLAP or MFCC fallback)
      B) Lyrics text similarity (reuses Part 1 LyricsSimilarityChecker)
      C) Lyrics-from-audio similarity (Whisper ASR → Part 1 pipeline)

    Usage
    -----
        checker = MusicAwareSimilarityChecker()
        result  = checker.compare("ref_song.mp3", "gen_song.mp3")
        print(result.decision_flag, result.music_similarity_score)

        # With optional pre-supplied lyrics
        result = checker.compare(
            "ref_song.mp3", "gen_song.mp3",
            lyrics_ref="ref_lyrics.txt",
            lyrics_gen="gen_lyrics.txt",
        )
    """

    def __init__(
        self,
        clap_model: str = _CLAP_HF_MODEL,
        whisper_model_size: str = _WHISPER_MODEL_SIZE,
        fusion_weights: dict | None = None,
        thresholds: dict | None = None,
        audio_sample_rate: int = 22050,
    ):
        self._audio_embedder = AudioEmbedder(clap_model)
        self._asr = ASRTranscriber(whisper_model_size)
        self._fusion_weights = fusion_weights or _FUSION_WEIGHTS
        self._thresholds = thresholds or _THRESHOLDS
        self._sample_rate = audio_sample_rate

        if _LYRICS_CHECKER_AVAILABLE:
            self._lyrics_checker = LyricsSimilarityChecker()
        else:
            self._lyrics_checker = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _audio_branch(
        self, seg_ref: AudioSegments, seg_gen: AudioSegments
    ) -> tuple[float, dict]:
        """Branch A: compute audio↔audio cosine similarity."""
        emb_ref = self._audio_embedder.embed(seg_ref)
        emb_gen = self._audio_embedder.embed(seg_gen)
        sim = AudioEmbedder.cosine_similarity(emb_ref, emb_gen)
        return sim, {"embedding_dim": len(emb_ref)}

    def _lyrics_branch(
        self,
        lyrics_ref: Optional[str],
        lyrics_gen: Optional[str],
    ) -> tuple[Optional[float], dict]:
        """Branch B: lyrics text similarity using Part 1 pipeline."""
        if lyrics_ref is None or lyrics_gen is None:
            return None, {"skipped": "lyrics not provided"}
        if self._lyrics_checker is None:
            return None, {"skipped": "LyricsSimilarityChecker not available"}

        result = self._lyrics_checker.compare(lyrics_ref, lyrics_gen)
        return result.plagiarism_risk_score, {
            "semantic_similarity": result.semantic_similarity,
            "lexical_overlap": result.lexical_overlap,
            "style_similarity": result.style_similarity,
            "lyrics_decision": result.decision_flag,
        }

    def _asr_branch(
        self,
        seg_ref: AudioSegments,
        seg_gen: AudioSegments,
    ) -> tuple[Optional[float], dict]:
        """
        Branch C: transcribe both audio tracks with Whisper, then measure
        lyrics similarity between the two transcriptions.
        """
        if not self._asr.available:
            return None, {"skipped": "Whisper not available"}

        text_ref = self._asr.transcribe(seg_ref.path)
        text_gen = self._asr.transcribe(seg_gen.path)

        if not text_ref or not text_gen:
            return None, {"skipped": "ASR transcription returned empty text"}

        # Store transcripts on the segment objects for downstream inspection
        seg_ref.transcribed_lyrics = text_ref
        seg_gen.transcribed_lyrics = text_gen

        if self._lyrics_checker is None:
            # Minimal fallback: character-level edit similarity on transcripts
            from difflib import SequenceMatcher
            norm_ref = _normalize_text(text_ref)
            norm_gen = _normalize_text(text_gen)
            sim = SequenceMatcher(None, norm_ref, norm_gen).ratio()
            return float(sim), {
                "method": "edit_similarity_fallback",
                "transcribed_ref_snippet": text_ref[:120],
                "transcribed_gen_snippet": text_gen[:120],
            }

        result = self._lyrics_checker.compare(text_ref, text_gen)
        return result.plagiarism_risk_score, {
            "semantic_similarity": result.semantic_similarity,
            "lexical_overlap": result.lexical_overlap,
            "style_similarity": result.style_similarity,
            "lyrics_decision": result.decision_flag,
            "transcribed_ref_snippet": text_ref[:120],
            "transcribed_gen_snippet": text_gen[:120],
        }

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def compare(
        self,
        audio_ref: AudioInput,
        audio_gen: AudioInput,
        lyrics_ref: Optional[Union[str, os.PathLike]] = None,
        lyrics_gen: Optional[Union[str, os.PathLike]] = None,
        run_asr: bool = True,
    ) -> MusicSimilarityResult:
        """
        Full music-aware pipeline.

        Parameters
        ----------
        audio_ref  : path to reference audio file (mp3 / wav / flac / …)
        audio_gen  : path to generated/candidate audio file
        lyrics_ref : optional lyrics string or path to .txt file for reference
        lyrics_gen : optional lyrics string or path to .txt file for generated song
        run_asr    : if True (default) run Whisper ASR when Whisper is installed

        Returns
        -------
        MusicSimilarityResult with all branch scores and fused decision.
        """
        # Load audio
        seg_ref = _load_audio(audio_ref, self._sample_rate)
        seg_gen = _load_audio(audio_gen, self._sample_rate)

        # Branch A: audio embeddings
        audio_sim, audio_details = self._audio_branch(seg_ref, seg_gen)

        # Branch B: text lyrics (optional)
        lyrics_sim, lyrics_details = self._lyrics_branch(
            str(lyrics_ref) if lyrics_ref is not None else None,
            str(lyrics_gen) if lyrics_gen is not None else None,
        )

        # Branch C: ASR lyrics-from-audio (optional)
        asr_sim, asr_details = (None, {"skipped": "run_asr=False"})
        if run_asr:
            asr_sim, asr_details = self._asr_branch(seg_ref, seg_gen)

        # Score fusion
        fused = _fuse_scores(
            audio_sim,
            lyrics_sim,
            asr_sim,
            self._fusion_weights,
        )

        flag = _decision(fused, self._thresholds)

        return MusicSimilarityResult(
            audio_similarity=round(audio_sim, 4),
            lyrics_similarity=round(lyrics_sim, 4) if lyrics_sim is not None else float("nan"),
            lyrics_from_audio_similarity=round(asr_sim, 4) if asr_sim is not None else float("nan"),
            music_similarity_score=round(fused, 4),
            decision_flag=flag,
            details={
                "audio_branch": audio_details,
                "lyrics_branch": lyrics_details,
                "asr_branch": asr_details,
                "fusion_weights": self._fusion_weights,
                "thresholds": self._thresholds,
                "ref_duration_s": seg_ref.duration_seconds,
                "gen_duration_s": seg_gen.duration_seconds,
            },
        )

    def batch_compare(
        self,
        audio_ref: AudioInput,
        audio_candidates: list[AudioInput],
        lyrics_ref: Optional[Union[str, os.PathLike]] = None,
        lyrics_candidates: Optional[list[Optional[Union[str, os.PathLike]]]] = None,
        run_asr: bool = True,
    ) -> list[MusicSimilarityResult]:
        """
        Compare one reference track against a list of candidate tracks.

        Parameters
        ----------
        audio_ref         : path to reference audio file
        audio_candidates  : list of paths to candidate audio files
        lyrics_ref        : optional lyrics for reference (string or .txt path)
        lyrics_candidates : optional list of lyrics per candidate (aligned with audio_candidates)
        run_asr           : if True run Whisper ASR on each pair
        """
        if lyrics_candidates is None:
            lyrics_candidates = [None] * len(audio_candidates)

        results = []
        for audio_gen, lyrics_gen in zip(audio_candidates, lyrics_candidates):
            results.append(
                self.compare(
                    audio_ref,
                    audio_gen,
                    lyrics_ref=lyrics_ref,
                    lyrics_gen=lyrics_gen,
                    run_asr=run_asr,
                )
            )
        return results
