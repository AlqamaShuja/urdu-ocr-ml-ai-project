# Model B - fine-tuned vision-language model

Read [../CLAUDE.md](../CLAUDE.md) first. **Rule 0 applies: update this file whenever
you change VLM code, the model choice, or results.**

---

## Status: NOT STARTED

Nothing exists yet. `src/vlm/` contains only `__init__.py`. No model has been
downloaded, loaded, evaluated or fine-tuned. No candidate has been verified.

**Do not write training code before completing the verification below.** The
project brief is explicit that the candidate model must not be assumed feasible.

---

## Verify before writing any code

The candidate named in the original project brief is **Qwen2.5-VL-3B-Instruct**
(<https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct>). It is a candidate, not a
decision. Check and **record each answer in `reports/experiment_log.md`**:

1. **Availability and licence.** Is it still published? What licence? Does it
   permit academic use and publishing results?
2. **Urdu suitability.** Does it produce anything sensible for Urdu Nastaliq at
   all? Run a handful of zero-shot examples on real NUST-UHWR images before
   committing to it.
3. **Transformers version.** Which version supports this model class? Pin it.
4. **Memory, honestly measured.** Minimum practical VRAM for (a) inference and
   (b) LoRA fine-tuning, with the image resolution this project actually uses.
5. **Is PEFT/LoRA supported** for the required vision *and* language components?
6. **Would a smaller model be the realistic choice?**

If it does not fit the budget, **propose a smaller valid alternative**. The brief
explicitly allows this, and documenting the constraint is a legitimate, reportable
methodological decision - not a shortcoming.

---

## Hardware constraints - these are decided

- **Local GPU is a Quadro M620 with 2 GB VRAM.** A 3B model is ~6 GB in fp16.
  It cannot run there, let alone fine-tune. Never suggest otherwise.
- **All VLM work runs on Google Colab free tier.** The user has stated they cannot
  pay for compute.
- **Free Colab usually gives a T4 (15 GB, Turing).** T4 has **no bf16** and no
  flash-attention (needs Ampere+). Plan for fp16 and 4-bit quantisation (QLoRA).
  Vision tokens are memory-heavy, so image resolution is a real lever.
- **Kaggle Notebooks is the fallback** if Colab quota runs out: ~30 GPU hours per
  week, T4 x2 or P100. The code is portable; the notebook setup differs.
- Colab sessions die. Checkpoints and adapters must be written to Drive **every
  save**, not copied at the end (this exact mistake already cost a CRNN run - see
  open issue 1 in CLAUDE.md).

---

## Fairness requirements - the experiment depends on these

The research question is a controlled comparison. These are not style preferences.

1. **Same examples.** Read the same manifests the CRNN uses:
   `data/manifests/nust_uhwr_{train,val,test}.csv`. Do not re-split, do not
   re-sample, do not drop samples the CRNN kept.
2. **Same metric code.** Score with `src/evaluation/metrics.py` via
   `normalize_for_eval`. Do not write a second CER function.
3. **NFKC matters more for the VLM than for the CRNN.** The ground truth contains
   Arabic Presentation Forms - single code points encoding whole ligatures
   (U+FDF2 ﷲ x74, U+FDFA ﷺ x71, U+FE85 ﺅ x9). A generative model will emit the
   spelled-out letters, not the ligature code point. **Without NFKC it would be
   scored wrong for output that is visually and semantically correct.** The CRNN
   does not have this problem because it predicts from a fixed vocabulary. This
   asymmetry is exactly why normalization is shared, and it is worth stating in
   the thesis.
4. **No external language model on one side only.** The CRNN uses greedy CTC with
   no LM. Keep VLM generation settings plain and documented - no retrieval, no
   reranking, no second pass the CRNN does not get.
5. **Zero-shot and fine-tuned results are labelled separately.** Never call
   prompt-only inference "fine-tuning."
6. **One fixed, simple OCR instruction prompt.** Record it verbatim. Record every
   generation setting (temperature, max tokens, sampling). Prompt selection, if
   any, happens on **validation** data - never test.
7. **Output cleaning must be a documented rule**, applied identically to every
   sample. A VLM often adds "Here is the text:" or quotes. Strip it with a stated
   rule, not ad hoc per sample, and keep the raw generation alongside the cleaned
   one.
8. **Measure latency and peak GPU memory on the same device class as the CRNN's**
   measurements, or those columns cannot be compared. The CRNN will need a GPU
   inference measurement for this reason.

---

## Planned steps

Work in this order; do not skip ahead.

1. **Verify** the candidate (the six questions above). Record answers.
2. **Load and run zero-shot** on a handful of real NUST-UHWR images. Save the
   predictions. Confirm the model produces Urdu at all.
3. **Zero-shot baseline** on the validation split, scored with the shared metrics.
   Save predictions. Label clearly as zero-shot.
4. **Confirm GPU memory** headroom for LoRA before attempting it.
5. **LoRA/PEFT fine-tune** on a small subset with a short run first. Only then a
   full run.
6. **Save** the exact base model ID and revision, the adapter, the prompt, all
   generation settings, and the logs.
7. **Evaluate once on test**, after selection is final.
8. Report zero-shot and fine-tuned as separate rows.

---

## Expected files (none exist yet)

| File | Purpose |
| --- | --- |
| `src/vlm/prepare_data.py` | manifests to the model's expected input format |
| `src/vlm/finetune.py` | LoRA/PEFT training |
| `src/vlm/evaluate.py` | generation + shared metrics, zero-shot and fine-tuned |
| `configs/vlm.yaml` | model id and revision, prompt, generation settings, LoRA config |
| `notebooks/02_colab_vlm.ipynb` | Colab runner |

---

## A capability difference worth reporting

The CRNN reads **one pre-segmented text line**. A VLM can take a **full page** and
produce text directly, with no separate line-segmentation stage.

This does not show up in CER or WER - both models are evaluated on line images, so
the comparison stays controlled - but it is a real, practical difference that
belongs in the Discussion and Limitations sections. It is also a genuine strength
of the thesis: it shows the trade-off was understood rather than overlooked.

Do **not** fold full-page capability into the CER/WER tables. Mixing segmentation
into the comparison would mean a difference between the models could come from
segmentation errors rather than reading ability, which would defeat the entire
experimental design.
