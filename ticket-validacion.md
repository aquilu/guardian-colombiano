**Idea validation — "Guardián colombiano": a small, safe moderator + storyteller for Colombian Spanish (Track 2, with a Track 1 ROI angle)**

Hi team! Team lead: halbarba (Hugo Alonso, Colombia). We'd like to confirm our idea fits the rules before we go deep.

**Real use case.** *12 Cuentos* is our community app: people leave short anonymous memories (post-its) on a public mural, and an AI weaves them into 12 Colombian short stories, for a library-network audience (families, kids, older adults). Today every post-it goes through a 3-layer closed-model chain: local rules → OpenAI omni-moderation → a gpt-5.4-mini editorial guardian. Then gpt-5.4 drafts the stories and gpt-5.4 reviews them. All data in this project is synthetic.

**The problem (pilot, 260 synthetic post-its, parallel ES/EN).** The moderation layer over-refuses Colombian Spanish. Of the harmless memories, the current stack sends 24% to review in Spanish versus 6% for the *same texts* in English. For harmless idioms (*"me quería morir de la pena"*, *"le metí un pepazo al arco"*) it's 50% vs 10%. One example: *"I wanted to die of embarrassment"* (in Spanish) scored self-harm/intent 0.60. Folk healing traditions and legitimate memories of the armed conflict get flagged as violence.

**Plan (Track 2a multilingual & code-mixed safety + 2b safety-preserving fine-tuning):**
1. **CuratorKIT**: build and curate (dedup, diversity gate, provenance) an open bilingual ES/EN benchmark and training set. It has 13 categories with idiom false-positive traps, PII obfuscation, prompt injection hidden in memories, code-mixing and regional hate. Labels follow the app's real policy. GPT is used only as teacher/judge, with agreement filtering and human spot checks.
2. **AlignTune + SafeTune**: fine-tune Tiny Aya (or Aya Expanse 8B) as the guardian and storyteller. We'd compare plain fine-tuning against SafeTune-hardened fine-tuning and show safety drift and its repair.
3. **CircuitKIT** (stretch goal): locate the refusal direction that fires on idioms and steer it.
4. **AuditKIT**: before/after metrics, namely verdict accuracy, over-refusal rate (ES vs EN gap), recall on content that must be rejected, PII catch rate, story quality with the app's reviewer rubric, and latency.
5. **ROI**: cost of today's three OpenAI calls per post-it vs a self-hosted Aya on our own GPU.

Deliverables: HF dataset and model collections, a GitHub write-up, and a live demo wired into the app's pilot, with an ES/EN switch.

**Questions:**
- Does combining 2a and 2b in one submission work, using CuratorKIT + SafeTune + AuditKIT (+ CircuitKIT)?
- Is it OK to use GPT models only as teacher/judge, with a Cohere Aya model as the final model?
- Is a synthetic-only dataset acceptable, given we have no real user data yet?

Thanks!
