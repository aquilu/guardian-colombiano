# A guardian for a wall of memories: why 12 cuentos needs its own small, safe model

*Hugo Albarracín · Vasmi · Lexsi AI Safety & Alignment Hackathon 2026 (Tracks 2a + 2b, top 10 % of 369 submissions)*

In 2027 Colombia will mark 100 years since the birth of Gabriel García Márquez, our Nobel laureate. For that
centenary we are building [12 cuentos](https://12cuentos.vasmi.co/): a public mural where anyone can leave an
anonymous post-it with a memory of their region. A grandmother's recipe, the day the river rose, a brother who
never came back.

A wall like that needs someone to look after it. This post is about the model we trained to do that job,
**Guardián colombiano**, and why an off-the-shelf moderation service was not enough.

## The problem: three things a generic moderator gets wrong

**1. It does not speak Colombian.** In Tunja, "¡qué frío tan berraco!" is a comment about the weather. "Parce"
is how friends greet each other. A moderator trained mostly on other varieties of Spanish and English reads
those words as vulgarity and blocks them. On our test set, the chain of cloud services the prototype uses
today blocked 33 % of the Spanish notes that a human committee had approved. Every one of those is a person
who shared a memory and was told it was not acceptable.

**2. The memories leave the building.** Some post-its come from victims of the armed conflict and from
communities that have good reasons to distrust where their words end up. Sending each note to third-party
APIs is hard to justify to them. We wanted moderation to happen on the project's own server, with no API call
at inference.

**3. A fine-tuned model can be talked into harm.** The obvious fix is to fine-tune a small open model on the
mural's editorial policy. But fine-tuning tends to wear down a model's general safety, and a public wall
attracts people who will try to turn the moderator into something else. A guardian that can be argued into
helping with something dangerous is not a guardian.

## What we built

Guardián colombiano is a LoRA adapter on [Tiny Aya Global](https://huggingface.co/CohereLabs/tiny-aya-global)
(3.35B parameters) from Cohere Labs. It reads one post-it and returns a JSON verdict: *aprobado*, *revision*
or *rechazado*, eight policy flags, and a one-line explanation for the human committee. When it is unsure,
the note goes to *revision*. People make the final call; the model only decides what they need to read first.

We trained the same guardian three ways with Lexsi's [SafeTune](https://github.com/Lexsi-Labs/SafeTune)
(plain fine-tuning, Lisa and SafeGrad), curated the data with CuratorKIT and red-teamed the results with
AuditKIT. Then we measured two things side by side: how well each model moderates, and how often it helps
with a harmful request. The test set was written by a different model family, reviewed by a human committee,
frozen with a SHA-256 manifest and used exactly once.

## What we found

| Frozen holdout (ES / EN) | Tiny Aya original | Plain fine-tune | **Lisa** | Cloud chain |
|---|---|---|---|---|
| Moderation accuracy | 62.2 / 68.9 | 81.1 / 77.0 | **81.1 / 82.4** | 82.4 / 85.1 |
| Blocks notes the committee approved | 3.0 / 0.0 | 21.2 / 24.2 | **15.2 / 18.2** | 33.3 / 24.2 |
| Lets through what should be blocked | 46.3 / 41.5 | 7.3 / 4.9 | 9.8 / 7.3 | 2.4 / 2.4 |
| Helps with harmful requests (GPT-5.4 judge) | 50.0 / 58.8 | 47.1 / 38.2 | **0.0 / 0.0** | n/a |

Three results matter for the mural:

- **It understands the slang.** In Spanish, the Lisa adapter blocks approved colloquial notes 18 points less
  often than the cloud chain, at about the same accuracy.
- **It was not talked into harm.** The original model helped with half of the harmful requests, and plain
  fine-tuning barely changed that. Lisa refused all of them, in Spanish and in English, under two judges from
  different model families. The same held when we retrained on the Tiny Aya Water and Fire variants.
- **It runs locally.** The numbers above were measured in 4-bit on a 6 GB consumer GPU, at roughly 1.2 seconds
  per note. A library can host it.

## What it does not do yet

The guardian still lets through more unsafe notes than the cloud chain (9.8 % against 2.4 % in Spanish), which
is why it works next to a human committee and not instead of one. It still over-blocks 15 to 18 % of approved
notes, mostly friendly vulgarity that our training labels marked for review. Our 0 % was measured on
single-turn requests only, and AuditKIT's prompt-injection probes got through 3 times out of 8.

So the next steps, several of them suggested by Lexsi's mentors, are a fresh multi-turn attack set, measuring
what 4-bit quantization costs against full precision, and trying CircuitKIT's mixed precision if we need to
go smaller.

## Why this matters beyond one mural

Most safety work is measured in English, on general assistants. Community projects in other languages are
left to choose between a generic moderator that does not understand their people and a custom model nobody
has checked. This hackathon gave us the tools to take a third path: a small model, tuned for one community's
way of speaking, with its safety measured and published.

We will keep improving Guardián colombiano until it is ready to look after the mural when it opens for the
centenary.

## Try it

- Demo: https://huggingface.co/spaces/aquilu/guardian-colombiano
- Code, data and every evaluation log: https://github.com/aquilu/guardian-colombiano
- Adapter: https://huggingface.co/vasmi-sas/guardian-tiny-aya-lisa
- The project it protects: https://12cuentos.vasmi.co/

Thanks to Lexsi Labs for the tools and the mentoring, to Cohere Labs for Tiny Aya, and to Hugging Face for
hosting the models, the data and the demo. Tiny Aya is CC-BY-NC, so the adapters and data are non-commercial
as well.
