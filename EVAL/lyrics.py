"""
Lyrics Similarity Check Class

Workflow:
  Raw lyrics -> Normalize -> Segment -> [Semantic | Lexical | Style/Phonetic]
  -> Aggregate -> Score Fusion -> {semantic_similarity, plagiarism_risk, decision_flag}
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from statistics import mean, median
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Optional heavy dependencies (graceful degradation)
# ---------------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer
    _SBERT_AVAILABLE = True
except ImportError:
    _SBERT_AVAILABLE = False

try:
    import pronouncing  # CMU pronouncing dict for rhyme detection
    _PRONOUNCING_AVAILABLE = True
except ImportError:
    _PRONOUNCING_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LyricsSegments:
    """Holds the segmented view of a single lyrics document."""
    raw: str
    normalized: str
    lines: list[str] = field(default_factory=list)
    sections: dict[str, list[str]] = field(default_factory=dict)  # label -> lines
    full_text: str = ""  # normalized, newline-joined


@dataclass
class SimilarityResult:
    """Final output of the similarity pipeline."""
    semantic_similarity: float          # 0-1, cosine similarity of embeddings
    lexical_overlap: float              # 0-1, n-gram / edit-distance score
    style_similarity: float             # 0-1, rhyme / repetition proxy
    plagiarism_risk_score: float        # 0-1, fused score
    decision_flag: str                  # "low" | "medium" | "high"
    details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Step A-B: Normalization
# ---------------------------------------------------------------------------

_SECTION_TAG_RE = re.compile(
    r"\[.*?\]",          # [Verse 1], [Chorus], …
    flags=re.IGNORECASE,
)

def _normalize(text: str) -> str:
    """Remove section tags, casefold, collapse whitespace, strip accents."""
    text = _SECTION_TAG_RE.sub("", text)
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.casefold()
    text = re.sub(r"[^\w\s']", " ", text)   # keep word chars and apostrophes
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Step C: Segmentation
# ---------------------------------------------------------------------------

def _segment(raw: str) -> LyricsSegments:
    """
    Parse raw lyrics into lines, labelled sections, and full normalized text.
    Section headers like [Verse 1] are used as keys; lines between headers
    belong to that section.
    """
    seg = LyricsSegments(raw=raw, normalized=_normalize(raw))

    current_section = "intro"
    sections: dict[str, list[str]] = {current_section: []}

    for raw_line in raw.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        header_match = re.match(r"\[(.+?)\]", stripped, re.IGNORECASE)
        if header_match:
            current_section = header_match.group(1).strip().lower()
            sections.setdefault(current_section, [])
        else:
            norm_line = _normalize(stripped)
            if norm_line:
                sections[current_section].append(norm_line)

    seg.sections = sections
    seg.lines = [line for lines in sections.values() for line in lines]
    seg.full_text = " ".join(seg.lines)
    return seg


# ---------------------------------------------------------------------------
# Step D1 + E: Semantic embeddings with section-weighted pooling
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "all-MiniLM-L6-v2"

class SemanticEmbedder:
    """
    Wraps a STS-tuned sentence-transformer.
    Falls back to a simple TF-IDF-style bag-of-words if sentence-transformers
    is not installed.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        if _SBERT_AVAILABLE:
            self._model = SentenceTransformer(model_name)
        else:
            self._model = None

    def _bow_vector(self, text: str) -> np.ndarray:
        """Minimal fallback: character 3-gram frequency vector."""
        ngrams: dict[str, int] = {}
        for i in range(len(text) - 2):
            ng = text[i : i + 3]
            ngrams[ng] = ngrams.get(ng, 0) + 1
        if not ngrams:
            return np.zeros(1)
        keys = sorted(ngrams)
        return np.array([ngrams[k] for k in keys], dtype=float)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1))
        if self._model is not None:
            return self._model.encode(texts, normalize_embeddings=True)
        # Fallback: pad all vectors to the same length
        vecs = [self._bow_vector(t) for t in texts]
        max_len = max(v.shape[0] for v in vecs)
        padded = np.zeros((len(vecs), max_len))
        for i, v in enumerate(vecs):
            padded[i, : v.shape[0]] = v
        norms = np.linalg.norm(padded, axis=1, keepdims=True) + 1e-9
        return padded / norms

    # --- section-weighted pooling (Step E) ---------------------------------
    SECTION_WEIGHTS: dict[str, float] = {
        "chorus": 2.0,
        "hook": 2.0,
        "verse": 1.0,
        "bridge": 0.8,
        "outro": 0.5,
        "intro": 0.5,
    }

    def _section_weight(self, label: str) -> float:
        for key, w in self.SECTION_WEIGHTS.items():
            if key in label:
                return w
        return 1.0

    def section_weighted_embedding(self, seg: LyricsSegments) -> np.ndarray:
        """Pool per-section mean embeddings with section importance weights."""
        section_embeddings: list[np.ndarray] = []
        weights: list[float] = []

        for label, lines in seg.sections.items():
            if not lines:
                continue
            vecs = self.embed_texts(lines)
            section_mean = vecs.mean(axis=0)
            section_embeddings.append(section_mean)
            weights.append(self._section_weight(label))

        if not section_embeddings:
            return np.zeros(1)

        w_arr = np.array(weights)
        stacked = np.stack(section_embeddings, axis=0)
        pooled = (stacked * w_arr[:, None]).sum(axis=0) / w_arr.sum()
        norm = np.linalg.norm(pooled) + 1e-9
        return pooled / norm

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ---------------------------------------------------------------------------
# Step D2 + F: Lexical overlap features
# ---------------------------------------------------------------------------

def _ngrams(tokens: list[str], n: int) -> set[str]:
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _jaccard(set_a: set, set_b: set) -> float:
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def _edit_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def lexical_overlap(seg_a: LyricsSegments, seg_b: LyricsSegments) -> dict:
    """
    Compute per-line n-gram Jaccard (unigram, bigram, trigram) and
    full-text edit similarity, then aggregate with robust statistics (F).
    """
    tok_a = seg_a.full_text.split()
    tok_b = seg_b.full_text.split()

    unigram_j = _jaccard(set(tok_a), set(tok_b))
    bigram_j  = _jaccard(_ngrams(tok_a, 2), _ngrams(tok_b, 2))
    trigram_j = _jaccard(_ngrams(tok_a, 3), _ngrams(tok_b, 3))
    edit_sim  = _edit_similarity(seg_a.full_text, seg_b.full_text)

    # Per-line edit similarities (robust stats across all line pairs)
    line_sims: list[float] = []
    for la in seg_a.lines:
        for lb in seg_b.lines:
            line_sims.append(_edit_similarity(la, lb))

    line_max    = max(line_sims, default=0.0)
    line_mean   = mean(line_sims) if line_sims else 0.0
    line_median = median(line_sims) if line_sims else 0.0
    line_p90    = float(np.percentile(line_sims, 90)) if line_sims else 0.0

    # Aggregate score: weight toward worst-case (max) to flag copied lines
    score = 0.25 * unigram_j + 0.20 * bigram_j + 0.15 * trigram_j \
          + 0.15 * edit_sim  + 0.25 * line_max

    return {
        "score": min(score, 1.0),
        "unigram_jaccard": unigram_j,
        "bigram_jaccard":  bigram_j,
        "trigram_jaccard": trigram_j,
        "edit_similarity": edit_sim,
        "line_max":    line_max,
        "line_mean":   line_mean,
        "line_median": line_median,
        "line_p90":    line_p90,
    }


# ---------------------------------------------------------------------------
# Step D3 + G: Style / phonetic proxies
# ---------------------------------------------------------------------------

def _end_words(lines: list[str]) -> list[str]:
    return [line.split()[-1] if line.split() else "" for line in lines]


def _rhyme_fingerprint(word: str) -> Optional[str]:
    """Return the rhyming suffix of a word using CMU dict if available."""
    if _PRONOUNCING_AVAILABLE:
        phones = pronouncing.phones_for_word(word)
        if phones:
            # last stressed vowel + everything after
            rhymes = pronouncing.rhyming_part(phones[0])
            return rhymes
    # Fallback: last 3 characters
    return word[-3:] if len(word) >= 3 else word


def _rhyme_scheme_similarity(lines_a: list[str], lines_b: list[str]) -> float:
    """Compare end-rhyme fingerprints between two line lists."""
    fp_a = [_rhyme_fingerprint(w) for w in _end_words(lines_a) if w]
    fp_b = [_rhyme_fingerprint(w) for w in _end_words(lines_b) if w]
    if not fp_a or not fp_b:
        return 0.0
    set_a = set(filter(None, fp_a))
    set_b = set(filter(None, fp_b))
    return _jaccard(set_a, set_b)


def _repetition_ratio(lines: list[str]) -> float:
    """Fraction of lines that are duplicated within the song."""
    if not lines:
        return 0.0
    return 1.0 - len(set(lines)) / len(lines)


def style_phonetic_similarity(seg_a: LyricsSegments, seg_b: LyricsSegments) -> dict:
    """
    Compute rhyme-scheme similarity and repetition structure similarity (G).
    """
    rhyme_sim = _rhyme_scheme_similarity(seg_a.lines, seg_b.lines)
    rep_a = _repetition_ratio(seg_a.lines)
    rep_b = _repetition_ratio(seg_b.lines)
    rep_sim = 1.0 - abs(rep_a - rep_b)  # closer repetition ratios -> more similar

    # Section-structure similarity: compare sorted section label sets
    labels_a = set(seg_a.sections.keys())
    labels_b = set(seg_b.sections.keys())
    struct_sim = _jaccard(labels_a, labels_b)

    score = 0.5 * rhyme_sim + 0.3 * rep_sim + 0.2 * struct_sim

    return {
        "score": min(score, 1.0),
        "rhyme_similarity":   rhyme_sim,
        "repetition_ratio_a": rep_a,
        "repetition_ratio_b": rep_b,
        "repetition_similarity": rep_sim,
        "structure_similarity":  struct_sim,
    }


# ---------------------------------------------------------------------------
# Step H + I: Score fusion, calibration, and decision
# ---------------------------------------------------------------------------

# Default fusion weights (sum to 1.0)
_FUSION_WEIGHTS = {
    "semantic": 0.45,
    "lexical":  0.40,
    "style":    0.15,
}

_THRESHOLDS = {
    "high":   0.75,
    "medium": 0.45,
}


def _fuse_scores(semantic: float, lexical: float, style: float,
                 weights: dict | None = None) -> float:
    w = weights or _FUSION_WEIGHTS
    return (
        w["semantic"] * semantic
        + w["lexical"]  * lexical
        + w["style"]    * style
    )


def _decision(score: float) -> str:
    if score >= _THRESHOLDS["high"]:
        return "high"
    if score >= _THRESHOLDS["medium"]:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Public API: LyricsSimilarityChecker
# ---------------------------------------------------------------------------

class LyricsSimilarityChecker:
    """
    End-to-end lyrics similarity and plagiarism-risk pipeline.

    Usage:
        checker = LyricsSimilarityChecker()
        result  = checker.compare(lyrics_a, lyrics_b)
        print(result.decision_flag, result.plagiarism_risk_score)
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        fusion_weights: dict | None = None,
        thresholds: dict | None = None,
    ):
        self._embedder = SemanticEmbedder(model_name)
        self._fusion_weights = fusion_weights or _FUSION_WEIGHTS
        self._thresholds = thresholds or _THRESHOLDS

    # ------------------------------------------------------------------
    def compare(self, lyrics_a: str, lyrics_b: str) -> SimilarityResult:
        """
        Full pipeline: normalize -> segment -> embed -> lexical -> style
        -> fuse -> output.

        Parameters
        ----------
        lyrics_a, lyrics_b : raw lyrics strings (may contain section tags)

        Returns
        -------
        SimilarityResult with all intermediate details attached.
        """
        # A-C: normalize + segment
        seg_a = _segment(lyrics_a)
        seg_b = _segment(lyrics_b)

        # D1 + E: semantic similarity
        emb_a = self._embedder.section_weighted_embedding(seg_a)
        emb_b = self._embedder.section_weighted_embedding(seg_b)
        sem_score = self._embedder.cosine_similarity(emb_a, emb_b)

        # D2 + F: lexical overlap
        lex_details = lexical_overlap(seg_a, seg_b)
        lex_score = lex_details["score"]

        # D3 + G: style / phonetic
        sty_details = style_phonetic_similarity(seg_a, seg_b)
        sty_score = sty_details["score"]

        # H: score fusion
        risk_score = _fuse_scores(
            sem_score, lex_score, sty_score, self._fusion_weights
        )
        risk_score = float(np.clip(risk_score, 0.0, 1.0))

        # I: decision flag
        flag = _decision(risk_score)

        return SimilarityResult(
            semantic_similarity=round(sem_score, 4),
            lexical_overlap=round(lex_score, 4),
            style_similarity=round(sty_score, 4),
            plagiarism_risk_score=round(risk_score, 4),
            decision_flag=flag,
            details={
                "lexical": lex_details,
                "style":   sty_details,
                "fusion_weights": self._fusion_weights,
            },
        )

    # ------------------------------------------------------------------
    def batch_compare(
        self,
        reference: str,
        candidates: list[str],
    ) -> list[SimilarityResult]:
        """Compare one reference against multiple candidate lyrics."""
        return [self.compare(reference, c) for c in candidates]
