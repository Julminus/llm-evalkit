"""Generate a held-out corpus of real LLM responses for the language detector.

Asks local Ollama models 20 everyday prompts, each in English and Indonesian,
and writes one JSONL row per response. The ``label`` field starts as the
*requested* language; every row is then read by a person and corrected if the
model answered in another language (that is exactly the failure the detector
exists to catch). Rows are never edited to make the detector look better.

Usage (needs Ollama running with the models pulled):
    uv run python scripts/build_lang_corpus.py --out tests/fixtures/langdetect_heldout.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

PROMPTS: list[tuple[str, str]] = [
    (
        "Explain why the sky is blue in two sentences.",
        "Jelaskan mengapa langit berwarna biru dalam dua kalimat.",
    ),
    (
        "Give one tip for staying focused while studying. One sentence.",
        "Berikan satu tips agar tetap fokus saat belajar. Satu kalimat saja.",
    ),
    (
        "What is a database index? Answer in 2-3 sentences.",
        "Apa itu indeks pada database? Jawab dalam 2-3 kalimat.",
    ),
    (
        "Write a short, friendly message reminding a colleague about tomorrow's meeting.",
        "Tulis pesan singkat dan ramah untuk mengingatkan rekan kerja tentang rapat besok.",
    ),
    ("List three healthy breakfast ideas.", "Sebutkan tiga ide sarapan sehat."),
    ("In a few words, what is the capital of Japan?", "Dalam beberapa kata, apa ibu kota Jepang?"),
    (
        "Describe the taste of durian to someone who has never tried it, in two sentences.",
        "Gambarkan rasa durian kepada orang yang belum pernah mencobanya, dalam dua kalimat.",
    ),
    (
        "What does a data engineer do? Keep it brief.",
        "Apa pekerjaan seorang data engineer? Jawab singkat.",
    ),
    (
        "Explain overfitting in machine learning to a beginner in three sentences.",
        "Jelaskan overfitting dalam machine learning kepada pemula dalam tiga kalimat.",
    ),
    (
        "Reply casually to a friend who asks if you want to get coffee later.",
        "Balas dengan bahasa santai ke teman yang ngajak ngopi nanti sore.",
    ),
    (
        "Summarize the plot of Cinderella in two sentences.",
        "Ringkas cerita Cinderella dalam dua kalimat.",
    ),
    (
        "Why is regular exercise important? One short paragraph.",
        "Mengapa olahraga teratur itu penting? Satu paragraf pendek.",
    ),
    (
        "How do I fix a 'permission denied' error when running a script on Linux? Brief answer.",
        "Bagaimana cara mengatasi error 'permission denied' saat menjalankan script di Linux? "
        "Jawab singkat.",
    ),
    (
        "Say thank you to a customer for their order in one sentence.",
        "Ucapkan terima kasih kepada pelanggan atas pesanannya dalam satu kalimat.",
    ),
    (
        "Is it better to rent or buy a house? Give a balanced answer in three sentences.",
        "Lebih baik menyewa atau membeli rumah? Berikan jawaban yang seimbang dalam tiga kalimat.",
    ),
    ("Explain what a GPU is in one sentence.", "Jelaskan apa itu GPU dalam satu kalimat."),
    (
        "Give a two-sentence weather report for a rainy day in Jakarta.",
        "Buat laporan cuaca dua kalimat untuk hari hujan di Jakarta.",
    ),
    (
        "What should I pack for a three-day hiking trip? Short list.",
        "Apa saja yang perlu dibawa untuk mendaki gunung selama tiga hari? Daftar singkat.",
    ),
    (
        "Respond with one word: what color is a ripe banana?",
        "Jawab dengan satu kata: apa warna pisang yang matang?",
    ),
    (
        "Tell me a fun fact about octopuses in one sentence.",
        "Ceritakan satu fakta menarik tentang gurita dalam satu kalimat.",
    ),
]

DEFAULT_MODELS = ["qwen2.5:7b", "llama3.1:8b"]


def chat(host: str, model: str, prompt: str) -> str:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0, "seed": 0, "num_predict": 200},
    }
    req = urllib.request.Request(
        f"{host}/api/chat",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return str(json.load(resp)["message"]["content"]).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    args = parser.parse_args()

    rows = []
    for model in args.models:
        for n, (en, id_) in enumerate(PROMPTS, start=1):
            for lang, prompt in (("en", en), ("id", id_)):
                text = chat(args.host, model, prompt)
                rows.append(
                    {
                        "id": f"{model.split(':')[0]}-{lang}-{n:02d}",
                        "model": model,
                        "requested": lang,
                        "label": lang,  # corrected by hand after review
                        "prompt": prompt,
                        "text": text,
                    }
                )
                print(f"{rows[-1]['id']}: {len(text.split())} words", file=sys.stderr)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
