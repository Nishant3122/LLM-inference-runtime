"""
Generate a small, original, synthetic training corpus for the Stage-1 tiny
Transformer.

We deliberately avoid downloading a third-party text corpus (e.g. tiny-shakespeare)
for two reasons:
  1. it would require a network fetch this project doesn't need Phase 0 to depend on.
  2. it sidesteps any question about reproducing copyrighted/famous text at scale.

The goal here is NOT language quality (see architecture.md: training a good LLM is a
non-goal of this project). The goal is a corpus with enough structure (repeated
words, punctuation, simple grammar) that char-level next-token prediction has
something real to learn, so we can sanity-check the training loop and later validate
that the C++ runtime reproduces the trained model's behavior bit-for-bit-ish.

Approach: procedurally combine short original template sentences with simple
substitutions, repeated with variation, to produce a few hundred KB of text.
"""
import argparse
import random
from pathlib import Path

SUBJECTS = [
    "the cat", "the dog", "a small bird", "the old clock", "my friend",
    "the quiet river", "a tall tree", "the little robot", "the wise owl",
    "the young student", "our neighbor", "the bright lamp", "a curious fox",
    "the busy market", "the calm lake", "the red bicycle", "the tired traveler",
]

VERBS = [
    "watches", "finds", "remembers", "follows", "carries", "greets",
    "measures", "paints", "repairs", "questions", "welcomes", "counts",
    "observes", "collects", "describes", "imagines",
]

OBJECTS = [
    "the morning light", "a broken clock", "the empty street", "an old letter",
    "the garden gate", "a distant mountain", "the small boat", "a wooden box",
    "the evening star", "a forgotten song", "the winter wind", "a paper map",
    "the village square", "a silver key", "the autumn leaves", "a quiet corner",
]

CONNECTORS = [
    "and then", "but soon", "so quietly", "while nearby", "just as",
    "even though", "and later", "before long",
]

ENDINGS = [
    "nothing seems to change.",
    "everyone stops to look.",
    "the story continues tomorrow.",
    "no one says a word.",
    "it starts to rain.",
    "the day feels shorter.",
    "a bell rings twice.",
    "the road turns north.",
]


def make_sentence(rng: random.Random) -> str:
    subj = rng.choice(SUBJECTS)
    verb = rng.choice(VERBS)
    obj = rng.choice(OBJECTS)
    sentence = f"{subj[0].upper()}{subj[1:]} {verb} {obj}"
    if rng.random() < 0.6:
        sentence += f", {rng.choice(CONNECTORS)} {rng.choice(ENDINGS)}"
    else:
        sentence += "."
    return sentence


def make_corpus(num_sentences: int, seed: int) -> str:
    rng = random.Random(seed)
    lines = []
    for i in range(num_sentences):
        line = make_sentence(rng)
        lines.append(line)
        # occasional paragraph break for structural variety
        if (i + 1) % rng.randint(3, 6) == 0:
            lines.append("")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(Path(__file__).parent / "corpus.txt"))
    parser.add_argument("--num-sentences", type=int, default=6000,
                         help="~6000 sentences -> a few hundred KB, enough for a tiny char-LM")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    text = make_corpus(args.num_sentences, args.seed)
    out_path = Path(args.out)
    out_path.write_text(text, encoding="utf-8")
    vocab = sorted(set(text))
    print(f"Wrote {len(text):,} chars ({out_path.stat().st_size:,} bytes) to {out_path}")
    print(f"Vocab size: {len(vocab)} -> {''.join(vocab)!r}")


if __name__ == "__main__":
    main()
