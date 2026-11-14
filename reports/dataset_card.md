# Dataset card

**Status: candidates verified from official pages on 2026-11-02. Nothing downloaded yet.**

Every figure below is what the official page *states*. None has been measured on
disk. Each "measured" row stays empty until the files are actually downloaded and
audited, and any discrepancy between stated and measured counts is recorded rather
than silently resolved.

## Summary of verification

| Dataset | Domain | Granularity | Licence | Usable for training? | Verified |
| --- | --- | --- | --- | --- | --- |
| IIIT Synthetic-R-Urdu | Printed | **Word-level** | **none stated** | yes, but licence unclear | 2026-11-02 |
| IIIT-Urdu | Printed | Line-level | CC BY 4.0 | **no - test only** | 2026-11-02 |
| NUST-UHWR | Handwritten | Line-level | CC BY 4.0 | yes | 2026-11-02 |

### Two problems found during verification

1. **The project brief's NUST-UHWR link is dead.** `https://zenodo.org/records/10548922`
   returns HTTP 404 (both the page and the REST API). The dataset is actually at
   Zenodo record **21674269**. All work must use the corrected URL below.
2. **Granularity mismatch on the printed side.** The only printed dataset that may
   be trained on is word-level; the only line-level printed dataset is licensed for
   testing only. There is therefore **no line-level printed Urdu training set**
   among the three candidates, while the handwritten dataset is line-level. This
   decides the shape of the printed experiments and is unresolved - see
   "Open decision" at the end of this file.

---

## 1. IIIT Synthetic-R-Urdu (printed)

- **Source URL:** <https://ilocr.iiit.ac.in/dataset/56/>
- **Date accessed:** 2026-11-02
- **Domain:** printed (synthetic)
- **Image granularity:** **word-level.** The page labels each split as
  "word level images".
- **Licence / usage terms:** **None stated on the page.** The only rights text is
  the footer "(c) Copyright NLTM OCR. All Rights Reserved". No licence file, terms
  of use, data agreement or citation requirement is linked.
  **Consequence:** using this dataset in a thesis without a stated licence is a
  real risk. Resolve by writing to the IIIT ILOCR maintainers for permission, and
  record their reply here, before training on it.
- **Access:** download links are on the page; the page also shows a session
  "Logout" link, so registration may be required.

### Stated contents

| Split | Stated word images | File |
| --- | --- | --- |
| Train | 586,980 | train.zip (images/, train_gt.txt, list_of_words.txt) |
| Validation | 83,850 | val.zip (images/, val_gt.txt) |
| Test | 167,700 | test.zip (images/, test_gt.txt) |
| **Stated total** | **838,500** | |
| **Splits actually sum to** | **838,530** | discrepancy of 30, unexplained on the page |

The page also mentions 445,998 Urdu words in the training set, which appears to be
the distinct-vocabulary size rather than an image count. To be confirmed on disk.

- **Annotation format:** ground-truth text files mapping image name to
  transcription, separated by a Tab character.
- **Generation method (as stated):** rendered with ImageMagick, Pango and Cairo
  using 890 freely available Unicode Urdu fonts. Backgrounds lighter than
  foregrounds. Font size, bold/italic, intensity, kerning and skew vary. A random
  one quarter of images are Gaussian-smoothed (sigma = 0.5). **All images are
  resized to height 32, preserving aspect ratio.**
- **Note on image height:** a fixed height of 32 px is already a CRNN-friendly
  convention, but it also means the data carries a preprocessing decision made by
  the dataset authors. The same height must then be used for the handwritten
  domain, or the two CRNN configurations must be documented as different.
- **Measured on disk:** not yet downloaded.
- **Known limitations:** synthetic, so it does not represent scanning noise,
  print degradation or real page layout. Expect optimistically low error rates
  that do not transfer to real printed pages.

## 2. IIIT-Urdu (printed)

- **Source URL:** <https://ilocr.iiit.ac.in/dataset/38/>
- **Date accessed:** 2026-11-02
- **Domain:** printed (real, scanned)
- **Image granularity:** line-level. The page describes "1,610 Urdu text line
  images along with their corresponding annotations".
  *Inconsistency on the page:* one sample section is headed "Sample Word Level
  Images from Training Set", which contradicts the line-level description. Must be
  settled by inspecting the images on disk.
- **Licence:** CC BY 4.0, per a linked data_license.doc.
- **Permitted use:** **testing only.** The page states: "This dataset can be used
  only for testing purposes." It must never enter training, validation, model
  selection or prompt selection.
- **Stated contents:** 1,610 line images. test.zip contains an images/ folder
  and test_gt.txt.
- **Splits:** none published. The page shows a "Train" download link, but it points
  to the same test.zip as the "Test" link. There is no training data.
- **Annotation format:** test_gt.txt, image name and ground truth separated by a
  Tab character.
- **How it was built (as stated):** pages from Urdu books and magazines scanned at
  high resolution; text-line bounding boxes drawn manually; annotations provided by
  language experts.
- **Citation requested:** Jain, Mathew and Jawahar, "Unconstrained OCR for Urdu
  using Deep CNN-RNN Hybrid Networks", 4th Asian Conference on Pattern Recognition
  (ACPR 2017), Nanjing, China.
- **Measured on disk:** not yet downloaded.
- **Role in this project:** the only *real* printed Urdu line data available, and
  licensed for exactly one job - an independent held-out test set. This is a
  strength, not a limitation, provided it is never trained on.

## 3. NUST-UHWR (handwritten)

- **Source URL:** <https://zenodo.org/records/21674269>
  (**the brief's URL, zenodo.org/records/10548922, is dead - HTTP 404**)
- **DOI:** 10.5281/zenodo.21674269 (concept DOI 10.5281/zenodo.21674268)
- **Date accessed:** 2026-11-02
- **Version:** 1.0
- **Publication date stated on record:** 2021-09-14
- **Authors:** Adnan Ul-Hasan (Deep Learning Laboratory, National Center of
  Artificial Intelligence, Islamabad) and Faisal Shafait (SEECS, NUST, Islamabad).
- **Licence:** CC BY 4.0. Access: Open.
- **Domain:** handwritten (Nastaliq)
- **Image granularity:** line-level. No character, word or ligature bounding boxes.
- **File:** UHWR.zip, 158,934,492 bytes (about 158.9 MB),
  MD5 `e367ff83c147671f4ae569bc63480af9`. Verify this checksum after download.

### Stated contents

- 10,606 grayscale JPEG line images.
- **All images are 64 px high**, widths roughly 200 to 2,170 px.
- 116 unique symbols across the transcriptions.
- Around 1,000 writers, varying in age, gender and background; text drawn from
  seven domains.
- No writer IDs, demographic labels or participant-level metadata are included.

### Official splits, and a documented inconsistency

| Split | Stated in description | Stated in uploaded manifests |
| --- | --- | --- |
| Train | 8,484 | 8,483 |
| Validation | 1,061 | 1,061 |
| Test | 1,061 | 1,061 |
| **Total** | **10,606** | **10,605** |

The record itself documents the cause: the image 03598.jpg is present in the
image directory but **is not referenced in any of the three manifests** and has no
recoverable transcription. The record says the discrepancy "should be resolved
before the final Zenodo release".

**Decision for this project:** treat the manifests as authoritative, giving
8,483 / 1,061 / 1,061 = 10,605 usable labelled lines, and exclude 03598.jpg as
unlabelled. Record this exclusion in every results table so the counts are
reproducible. Re-verify against the actual archive after download.

- **Usage restriction (stated intent):** the record states the dataset should not
  be used to identify individual writers or to infer personal characteristics from
  handwriting. This project only performs text recognition, so it complies; no
  writer-level or demographic analysis will be attempted.
- **Split policy stated by the record:** retain the predefined splits, and report
  it explicitly if a different partition is used. This project retains them.
- **Known limitations stated by the record:** ground truth contains spelling
  variation, punctuation inconsistency, annotation noise and visually unclear
  characters. A related Conv-Transformer study reported label noise, distorted
  images, and ground truth that sometimes contains extra or incorrect text.
  **Consequence:** CER on this dataset has a non-zero noise floor. Neither model
  can reach 0% CER, and a gap of a fraction of a percent between CRNN and VLM is
  not meaningful. This must appear in "threats to validity".
- **Associated publications (from the record):**
  - Neural Computing and Applications, DOI 10.1007/s00521-021-06498-2
    (introduces the dataset)
  - IJDAR, DOI 10.1007/s10032-022-00416-5 (later study using it)

### Measured on disk - downloaded and audited 2026-11-02

Downloaded from Zenodo record 21674269. **MD5 verified: the downloaded
`UHWR.zip` hashes to `e367ff83c147671f4ae569bc63480af9`, matching the record
exactly**, and the size matches at 158,934,492 bytes. Extracted to
`data/raw/nust_uhwr/UHWR/` (330 MB on disk).

Full audit report: `reports/dataset_audit_nust_uhwr.md`, produced by
`scripts/audit_nust_uhwr.py`.

**Archive layout (no README and no licence file is included):**

```text
UHWR/
├── chars.txt        434 B    character inventory, one per line
├── old_chars.txt    657 B    superseded inventory
├── freq.txt       1,195 B    character frequency counts
├── train.txt    1,023,536 B  manifest
├── val.txt        128,071 B  manifest
├── test.txt       128,224 B  manifest
└── Dataset/images/           10,606 JPEG files
```

**Manifest format:** `Dataset/images/<id>.jpg` + TAB + transcription. No header.
**No manifest ends with a newline**, so `wc -l` under-reports each split by one.
Counting non-empty lines is the correct method.

| Measurement | Result | Stated by record | Verdict |
| --- | --- | --- | --- |
| Images on disk | 10,606 | 10,606 | matches |
| Train records | 8,483 | 8,484 (description) / 8,483 (manifests) | matches manifests |
| Validation records | 1,061 | 1,061 | matches |
| Test records | 1,061 | 1,061 | matches |
| Total labelled records | 10,605 | 10,605 (manifests) | matches |
| Referenced but missing from disk | 0 | - | clean |
| Unreferenced images | 1 (`03598.jpg`) | 1 | matches |
| Unreadable or corrupt images | 0 | - | clean |
| Distinct image heights | 1 (all exactly 64 px) | 64 px | matches |
| Image widths | min 200, median 911, mean 949.5, max 2170 | approx. 200-2,170 | matches |
| Empty or whitespace-only labels | 0 | - | clean |
| Labels with surrounding whitespace | 0 | - | clean |
| Label length (characters) | min 19, median 52, mean 52.3, max 116 | - | - |
| Distinct characters in transcriptions | 116 | 116 | matches |
| `chars.txt` coverage | 116 declared, 0 extra, 0 unused | - | exact match |
| Image IDs in more than one split | 0 | - | clean |

Every stated figure in the Zenodo record checks out. Image-to-label alignment
was additionally spot-checked visually: `07955.jpg` renders exactly its
transcription, confirming there is no systematic off-by-one shift between images
and manifests.

### Finding not documented by the record: a label conflict

`04745.jpg` and `04746.jpg` are **byte-identical** (both MD5
`b0b7bf6baf997080550f0dff00e2891e`, both 11,906 bytes), yet they carry different
transcriptions and sit in different splits:

| Image | Split | Transcription |
| --- | --- | --- |
| `04746.jpg` | train | بہلاوا باقی رہ گیا ہے۔۔۔ اب وہ دم خم نہیں رہے |
| `04745.jpg` | val | سلسلے میں بڑے دردناک بیان دے چکے ہیں۔ الکنندہ کئی ہاتھوں |

Identical pixels cannot depict two different sentences, so one label is wrong.
This is two defects at once: a train/validation leak, and a mislabelled record.

**Resolved by inspection.** The image was opened and read: it shows
*بہلاوا باقی رہ گیا ہے۔۔۔ اب وہ دم خم نہیں رہے*, which is the **train** label.
The most likely cause is that during dataset assembly the real `04745.jpg` was
overwritten with a copy of `04746.jpg`, leaving the validation record pointing at
the wrong picture.

**Decision: drop `04745.jpg` from the validation split; keep `04746.jpg` in train.**
Its label does not describe its image, so scoring any model against it would
punish a correct reading. Removing it also closes the train/validation leak.

### Exclusions applied to this dataset

| Image | Split | Reason | Effect |
| --- | --- | --- | --- |
| `03598.jpg` | none | present on disk, referenced by no manifest, no recoverable transcription | already outside all splits |
| `04745.jpg` | val | byte-identical to training image `04746.jpg` and carries a label that does not match its image | validation 1,061 -> 1,060 |

**Usable counts for this project: 8,483 train / 1,060 validation / 1,061 test =
10,604 labelled lines.** These are the numbers that must appear in every results
table, with the two exclusions stated. The **test split is untouched** - the leak
was train-to-validation only, so the headline test metrics are unaffected.

### Transcription overlap across splits (reported, not removed)

| Split pair | Identical transcriptions |
| --- | --- |
| train and val | 83 |
| train and test | 84 |
| val and test | 8 |

These are different images of the same sentence, written by different writers,
which is legitimate in a handwriting corpus. They are reported rather than
removed, because a model can score partly from memorised text instead of from
reading the image. This belongs in "threats to validity".

---

## Unicode normalization: evidence, still undecided

This decision affects both models identically and must be made before any CER or
WER number is produced. The measured evidence from NUST-UHWR's 10,605
transcriptions:

| Normalization form | Labels changed | Distinct characters after |
| --- | --- | --- |
| none (raw) | - | 116 |
| NFC | 318 | 116 |
| NFD | 4,967 | 110 |
| NFKC | 452 | 114 |

**NFD is unsuitable.** It decomposes characters into base plus combining marks,
changing almost half the labels and inflating the character sequence a CTC model
must predict. It is not a candidate.

### Characters that force an explicit decision

**Arabic letter variants.** Urdu and Arabic share code points but prefer
different ones for the same letter. These appear in the ground truth:

| Character | Code point | Count | Urdu preferred form |
| --- | --- | --- | --- |
| ي | U+064A ARABIC YEH | 26 | U+06CC FARSI YEH (ی) |
| ه | U+0647 ARABIC HEH | 14 | U+06C1 HEH GOAL (ہ) |
| ة | U+0629 TEH MARBUTA | 22 | context-dependent |
| أ | U+0623 ALEF WITH HAMZA ABOVE | 2 | - |
| ؤ | U+0624 WAW WITH HAMZA ABOVE | 171 | - |

Unicode normalization does **not** fold these; they are distinct letters, not
compatibility variants. Folding them requires an explicit custom mapping, and
that mapping is a research decision with consequences: it reduces the character
vocabulary and makes CER more forgiving, so it must be stated in the thesis and
applied to ground truth and predictions alike.

**Arabic Presentation Forms** - single code points encoding whole ligatures:

| Character | Code point | Count |
| --- | --- | --- |
| ﷲ | U+FDF2 | 74 |
| ﷺ | U+FDFA | 71 |
| ﺅ | U+FE85 | 9 |

NFKC expands these into their constituent letters. This matters a great deal for
the VLM: a generative model will almost certainly emit the spelled-out letters
rather than the single ligature code point, so without NFKC it would be scored as
wrong for output that is visually and semantically correct. **This is a fairness
issue in the comparison, not a cosmetic one.**

**Latin letters in Urdu ground truth** - `x` (5), `k` (3), `l` (2), `o` (1),
`u` (1). Twelve occurrences in total, almost certainly annotation noise. Too few
to matter numerically, but they must not be silently deleted from only one
model's output.

### Recommendation to be confirmed

Apply **NFKC**, then strip surrounding whitespace and collapse runs of internal
whitespace, and keep the raw transcription in every manifest beside the
normalized one. Report the primary CER and WER on the normalized text and a
stricter raw-text CER as a secondary number, as the project brief requires.
Decide separately, and state explicitly, whether Arabic letter variants are also
folded to their Urdu forms.

**Status: recommended, not yet decided or implemented.**

## Checks still to run

For Synthetic-R-Urdu, once its licence is cleared and it is downloaded:

1. Measured image counts per split versus the stated counts, including the
   838,500 versus 838,530 discrepancy.
2. Corrupt or unreadable images; empty or whitespace-only transcriptions.
3. Duplicate images across splits (leakage check).
4. Character inventory, and whether it matches NUST-UHWR's inventory. A printed
   vocabulary that differs from the handwritten one means the two CRNN models
   have different output layers, which must be documented.
5. Confirmation that all images are height 32 as stated.

For IIIT-Urdu, if it is downloaded for the qualitative side observation:

6. Confirm whether the images are truly line-level, settling the page's own
   contradiction between its description and its sample heading.

Deferred for both NUST-UHWR and the printed data:

7. Near-duplicate (not byte-identical) image detection. Exact-hash duplication
   has been checked; perceptual near-duplicate detection is more expensive and is
   deferred until it is needed.

## Decision on the printed-side granularity mismatch

**Decided 2026-11-02: Option A - the printed task stays word-level.**

Handwritten is line-level and trainable. Printed is **either** word-level and
trainable (Synthetic-R-Urdu, licence unstated) **or** line-level and test-only
(IIIT-Urdu, CC BY 4.0). The project brief forbids pretending word-level and
line-level data are equivalent, so the two domains are run as two separate tasks.

The options considered:

- **Option A - printed stays word-level. CHOSEN.** Train and evaluate the printed
  CRNN and VLM on Synthetic-R-Urdu's own official splits. Handwritten stays
  line-level on NUST-UHWR.
- **Option B - render synthetic printed Urdu lines** from an openly licensed corpus
  and fonts, making both domains line-level. Not chosen: extra implementation work
  and Nastaliq shaping risk.
- **Option C - word-level training with line-level evaluation via segmentation.**
  Rejected: line-to-word segmentation introduces an error source that would
  confound the CRNN-versus-VLM comparison, which is the research question.

### What Option A commits this project to

1. **Two tasks, not one.** The research question is answered *within* each domain:
   CRNN versus VLM on printed word images, and CRNN versus VLM on handwritten text
   lines. Printed and handwritten CER/WER are never compared to each other, never
   averaged, and never plotted on a shared axis without the granularity stated.
2. **WER is close to meaningless on the printed task.** A word-level image contains
   one word, so WER degenerates to "whole word wrong or right" and carries far less
   information than on multi-word lines. CER remains the primary printed metric;
   printed WER is reported but must be labelled as single-word WER.
3. **The two domains also differ in image height** - Synthetic-R-Urdu is
   pre-resized to 32 px, NUST-UHWR to 64 px. Each domain therefore gets its own
   CRNN input configuration, and the difference is recorded in both config files
   rather than forced to match.
4. **IIIT-Urdu is dropped from the headline comparison.** It is line-level printed
   data, so it cannot serve as a test set for word-level printed models. It may
   still be used for a clearly labelled qualitative side observation (how the
   word-level printed models behave on real printed lines), never as a reported
   CER/WER result in the main tables. Its "testing only" licence is honoured
   either way.
5. **The licence question is noted but is not a blocker.** Scope decision of
   2026-11-02: this work is local experimentation and a local demo, not dataset
   redistribution, so using Synthetic-R-Urdu locally is low risk. The enquiry
   draft at `reports/licence_enquiry_iiit.md` is kept and should be sent if the
   work is submitted as a thesis, published, or pushed to a public repository
   along with the data or weights trained on it. Any reply is recorded in
   section 1 of this card.

### Resulting experiment matrix

| # | Model | Domain | Data | Granularity | Primary metric |
| --- | --- | --- | --- | --- | --- |
| 1 | CRNN + CTC | Printed | Synthetic-R-Urdu | word | CER |
| 2 | VLM (zero-shot, then LoRA) | Printed | Synthetic-R-Urdu | word | CER |
| 3 | CRNN + CTC | Handwritten | NUST-UHWR | line | CER and WER |
| 4 | VLM (zero-shot, then LoRA) | Handwritten | NUST-UHWR | line | CER and WER |

Within each row pair the underlying train/validation/test examples are identical
across the two models, and both are scored by the same code in `src/evaluation/`.

### Download status

**No dataset has been downloaded (decision of 2026-11-02: wait).** Downloads happen
only on explicit instruction. Planned order when that comes:

1. NUST-UHWR `UHWR.zip` (159 MB) - licence already clear, audit can start at once.
2. Synthetic-R-Urdu - deferred until the CRNN pipeline is working end to end on
   handwritten data, so that adding the printed domain is only a data swap.
   A subset is sufficient;
   586,980 training word images far exceed what this project's compute budget needs,
   so a documented random subsample with a fixed seed will be drawn from the
   official training split, leaving the official validation and test splits intact.

## Mislabelled records confirmed by inspection

Beyond `04745.jpg` (found during the 2026-11-02 audit), two **test-split** records
were found on 2026-11-14 while reading the CRNN's worst predictions. In both the
model's output is fluent Urdu that matches the image, and the shipped
transcription describes something else entirely. A CTC model with no language
model cannot invent coherent sentences, so a fluent prediction unrelated to the
reference is a signal that the reference is wrong.

Both images were opened and read to confirm.

| Image | Split | Shipped transcription | What the image actually shows (= what the model read) |
| --- | --- | --- | --- |
| `02786.jpg` | test | منصوبہ کی تکمیل کی جانب نہیں بڑھ سکا۔ ایران نے اپنے | اور اب بھی محبت اور دوستی کا پیغام دیا انہوں نے کہا کہ ہم |
| `02792.jpg` | test | کے مطابق پاکستان کا مرکزی بینک سابقہ دور حکومت کے مقابلے میں | شرکت کی جا سکے۔ خیرپور (یو رپورٹ) چونڈ کے قریب ٹریلر کی |

**Not a systematic offset.** The true texts do not appear as the label of any
other image in any split, and the labels of neighbouring image ids are unrelated.
These are isolated errors, consistent with the label noise the Zenodo record
itself warns about, not a shifted or swapped block.

### These records are NOT removed

They sit in the **test** split, and the score has already been seen. Dropping
samples after seeing the score is not permissible, so they stay in, and the effect
is reported instead:

| | CER |
| --- | --- |
| Reported (all 1,061 test lines) | **0.1068** |
| Excluding these two records | 0.1054 |
| Difference | 0.0014 |

About 89 of the 5,952 counted character errors come from these two records. The
reported figure stands; the true error rate is marginally better than it states.

### Implication for the comparison

Whatever this noise costs, it costs **both** models equally, since both are scored
on the same records with the same code. It does not bias CRNN against VLM. What it
does mean is that **CER on this dataset has a non-zero floor**, and a gap of a few
tenths of a percent between the two systems is not meaningful. This belongs in
threats to validity.

No systematic sweep for further mislabelled records has been run. Two were found
by reading five worst-case predictions, so more probably exist.
