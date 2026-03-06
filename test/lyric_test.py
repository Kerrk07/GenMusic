"""
Lyric similarity test — compares the two reference lyrics in data/lyrics/.
Run from the project root:
    python -m test.lyric_test
or with the venv:
    venv/bin/python -m test.lyric_test
"""

from pathlib import Path

from EVAL.lyrics import LyricsSimilarityChecker

DATA_DIR = Path(__file__).parent.parent / "data" / "lyrics"
SONG_A = DATA_DIR / "vertigo_beachfossil.txt"
SONG_B = DATA_DIR / "ai_synthetic_vertigo.txt"


def main() -> None:
    checker = LyricsSimilarityChecker()
    result = checker.compare(SONG_A, SONG_B)

    print("=" * 50)
    print(f"  Reference : {SONG_A.name}")
    print(f"  Candidate : {SONG_B.name}")
    print("=" * 50)
    print(f"  Semantic similarity   : {result.semantic_similarity:.4f}")
    print(f"  Lexical overlap       : {result.lexical_overlap:.4f}")
    print(f"  Style similarity      : {result.style_similarity:.4f}")
    print(f"  Plagiarism risk score : {result.plagiarism_risk_score:.4f}")
    print(f"  Decision flag         : {result.decision_flag.upper()}")
    print("=" * 50)

    lex = result.details["lexical"]
    print("\nLexical breakdown:")
    print(f"  Unigram Jaccard : {lex['unigram_jaccard']:.4f}")
    print(f"  Bigram  Jaccard : {lex['bigram_jaccard']:.4f}")
    print(f"  Trigram Jaccard : {lex['trigram_jaccard']:.4f}")
    print(f"  Edit similarity : {lex['edit_similarity']:.4f}")
    print(f"  Line max        : {lex['line_max']:.4f}")
    print(f"  Line p90        : {lex['line_p90']:.4f}")

    sty = result.details["style"]
    print("\nStyle / phonetic breakdown:")
    print(f"  Rhyme similarity      : {sty['rhyme_similarity']:.4f}")
    print(f"  Repetition similarity : {sty['repetition_similarity']:.4f}")
    print(f"  Structure similarity  : {sty['structure_similarity']:.4f}")


if __name__ == "__main__":
    main()
