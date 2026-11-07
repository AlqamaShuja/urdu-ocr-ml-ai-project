# Licence enquiry - IIIT Synthetic-R-Urdu

**Why this exists.** Synthetic-R-Urdu is the printed training set for this project
under the Option A design decision, but its dataset page states no licence, no terms
of use and no citation requirement. The only rights text on the page is the footer
"(c) Copyright NLTM OCR. All Rights Reserved". Training a thesis experiment on data
with no stated licence is a risk to the thesis, so written permission is needed
before the dataset is downloaded or used.

**This is on the critical path.** External replies take time, so send it before
starting any printed-domain work.

**Send to:** `ocr.nltm@research.iiit.ac.in`
(the only contact address given, at <https://ilocr.iiit.ac.in/contact/>)

**Status:** not sent yet.
**Date sent:**
**Reply received:**
**Outcome:** (record the exact permission wording in `reports/dataset_card.md`,
section 1, under "Licence / usage terms")

**Before sending:** fill in the four bracketed placeholders in the signature and,
if you have one, send from your NUST institutional address rather than a personal
one. A university address makes an academic-use request far more likely to be
answered.

---

## Draft email

**Subject:** Licence and permitted use - Synthetic-R-Urdu dataset (ILOCR dataset 56)

Dear NLTM OCR team,

I am a Master's student in Machine Learning and Artificial Intelligence at the
National University of Sciences and Technology (NUST), Pakistan. For my thesis I am
running a controlled comparison of a CRNN with CTC against a parameter-efficiently
fine-tuned small vision-language model, on printed and handwritten Urdu text
recognition.

I would like to use the Synthetic-R-Urdu dataset as the printed-domain data:

<https://ilocr.iiit.ac.in/dataset/56/>

The dataset page does not state a licence or terms of use, so I would be grateful
if you could confirm the following before I download and use it:

1. Under what licence or terms is Synthetic-R-Urdu released?
2. Is use permitted for non-commercial academic research, including training and
   evaluating models and reporting the resulting metrics in a Master's thesis and
   any subsequent academic publication?
3. How would you like the dataset cited? Is there a specific paper or
   acknowledgement you would prefer?
4. Are there any restrictions I should observe, for example on redistributing the
   images, publishing sample images in a thesis, or releasing model weights trained
   on the data?

I would only be using a subsample of the official training split, and I would keep
the official validation and test splits unchanged.

For completeness, I also intend to use IIIT-Urdu (<https://ilocr.iiit.ac.in/dataset/38/>),
which states CC BY 4.0 and that it is for testing purposes only. I will respect that
restriction and will not train or tune on it.

Thank you for making these resources available, and for any clarification you can
provide.

With best regards,

[Full name]
MS Machine Learning and Artificial Intelligence
[Department], National University of Sciences and Technology (NUST), Pakistan
[Institutional email address]
[Supervisor name, if you wish to include it]

---

## If no reply arrives

Do not treat silence as permission. Record the send date and the waiting period in
`reports/dataset_card.md`, then choose one of:

1. **Switch the printed domain to Option B** - render synthetic printed Urdu word or
   line images from an openly licensed Urdu text corpus and openly licensed Urdu
   fonts. The licence then becomes fully known and self-documented.
2. **Find an alternative printed Urdu dataset** with an explicit licence, verify it
   against this same dataset-card template, and record the substitution.

Either way, note in the thesis limitations section that the originally planned
printed dataset was not usable because its licence could not be established.
