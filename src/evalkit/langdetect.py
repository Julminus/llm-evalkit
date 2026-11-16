"""A small, deterministic English vs. Indonesian language detector.

Why not a general-purpose library? The harness only ever needs to tell EN
from ID, the answer must be identical on every run (some popular detectors
are randomized), and it should be cheap enough to run on every response.
Function words are the strongest signal for that job: "yang", "dan", "di",
"untuk" never appear in English prose, and "the", "is", "of" never appear in
Indonesian prose, even when an Indonesian answer is full of English nouns
("meeting-nya di-reschedule karena deadline").

The method: lowercase the text, drop code and URLs, split on anything that is
not a letter (so "meeting-nya" yields "nya"), and count hits against two
disjoint function-word lists. The language with more hits wins. Confidence is
the winner's share of all hits, so 1.0 means no evidence for the other
language and 0.5 means an even mix.

Known limits: Malay shares most Indonesian function words and is detected as
Indonesian; one- or two-word answers carry little or no signal, so callers
should treat a low ``signal_words`` count as "can't tell".
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Literal

from evalkit.text import strip_code

Language = Literal["en", "id"]

# Words that are also common tokens in the other language were left out on
# purpose: "no" (Indonesian "No." = nomor), "as" (Indonesian "AS" = United
# States), "i" (Roman numeral in "Bab I"), "mana" (an English gaming term),
# "pun" (an English word). Both sets must stay disjoint; a test enforces it.
# fmt: off
_EN_WORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "nor", "of", "to", "in", "on", "at", "by",
    "for", "with", "from", "about", "into", "over", "after", "before", "between",
    "through", "during", "without", "under", "is", "are", "was", "were", "be", "been",
    "being", "am", "do", "does", "did", "have", "has", "had", "will", "would", "can",
    "could", "should", "may", "might", "must", "shall", "this", "that", "these",
    "those", "it", "its", "they", "them", "their", "there", "here", "we", "our", "us",
    "you", "your", "he", "she", "his", "her", "him", "my", "me", "not", "if", "then",
    "than", "so", "which", "who", "whom", "whose", "what", "when", "where", "why",
    "how", "all", "any", "some", "each", "every", "more", "most", "other", "such",
    "only", "also", "very", "just", "because", "while", "however", "both", "either",
    "neither", "out", "up", "down", "off", "too", "own", "same", "few", "many",
    "much", "onto", "upon", "yet", "although", "though", "unless", "until",
})

_ID_WORDS = frozenset({
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "untuk", "dengan", "dalam",
    "pada", "adalah", "merupakan", "tidak", "bukan", "akan", "sudah", "telah",
    "belum", "masih", "juga", "atau", "tetapi", "tapi", "namun", "karena", "sehingga",
    "jika", "kalau", "apabila", "bila", "maka", "agar", "supaya", "saat", "ketika",
    "sebelum", "sesudah", "setelah", "selama", "oleh", "bagi", "tentang", "terhadap",
    "antara", "seperti", "sebagai", "bahwa", "yaitu", "yakni", "ada", "bisa",
    "dapat", "harus", "perlu", "ingin", "mau", "sangat", "lebih", "paling", "banyak",
    "sedikit", "semua", "setiap", "beberapa", "para", "sebuah", "suatu", "seorang",
    "saya", "aku", "kami", "kita", "anda", "kamu", "mereka", "dia", "ia", "beliau",
    "nya", "apa", "siapa", "mengapa", "kenapa", "bagaimana", "kapan", "berapa",
    "dimana", "hal", "cara", "secara", "hanya", "lah", "kah", "sama", "lain",
    "tersebut", "hingga", "sampai", "sejak", "tanpa", "melalui", "menurut", "serta",
    "baik", "ya", "tak", "lagi", "sendiri", "masing", "jadi", "pula", "agak", "begitu",
    "kemudian", "sebab", "walaupun", "meskipun", "sedangkan",
})
# fmt: on

_LETTERS_RE = re.compile(r"[^\W\d_]+")


@dataclass(frozen=True)
class LanguageGuess:
    """Result of :func:`detect_language`."""

    language: Language | None
    """Winning language, or ``None`` when there is no signal or an exact tie."""

    confidence: float
    """Winner's share of all function-word hits (0.5 = even mix, 1.0 = pure)."""

    hits: dict[Language, int]
    """Function-word hits per language."""

    @property
    def signal_words(self) -> int:
        """Total function-word hits; low values mean the guess is weak."""
        return sum(self.hits.values())


def detect_language(text: str) -> LanguageGuess:
    """Guess whether ``text`` is English or Indonesian."""
    tokens = Counter(t.lower() for t in _LETTERS_RE.findall(strip_code(text)))
    en = sum(n for tok, n in tokens.items() if tok in _EN_WORDS)
    id_ = sum(n for tok, n in tokens.items() if tok in _ID_WORDS)
    hits: dict[Language, int] = {"en": en, "id": id_}
    total = en + id_
    if total == 0 or en == id_:
        return LanguageGuess(language=None, confidence=0.5 if total else 0.0, hits=hits)
    winner: Language = "en" if en > id_ else "id"
    return LanguageGuess(language=winner, confidence=max(en, id_) / total, hits=hits)
