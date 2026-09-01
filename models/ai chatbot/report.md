# ML Chatbot Upgrade — Report

## A. What changed

**`intents.json`** — Grown from 10 intents / ~130 patterns to **19 intents / 563 patterns**. All 10 original intents kept, each expanded to 23–40 diverse patterns (not word-swaps of one template). Added 9 new intents: `location`, `services`, `pricing`, `contact`, `complaint`, `order_status`, `cancel_order`, `recommendation`, `faq`. Every intent has 2–4 responses.

**`preprocess.py`** — Same lowercase → strip punctuation → collapse whitespace pipeline, but now: expands `don't`/`can't`/`won't` before stripping punctuation so negation words survive as real tokens instead of being mangled; handles `None`/non-string/empty/whitespace-only input without raising; always returns a string; `build_dataset` skips any pattern that cleans down to nothing.

**`train_model.py`** — TF-IDF upgraded to `ngram_range=(1,2)` with `sublinear_tf=True`. Added precision/recall/F1, a full `classification_report`, a saved `confusion_matrix.png`, and 5-fold `StratifiedKFold` cross-validation with per-fold scores, mean, and standard deviation. Logistic Regression's `C` was tuned empirically (see section D) to `C=10`. Now also writes `model_info.json` with real numbers from the run.

**`chatbot.py`** — Confidence threshold raised from `0.25` to `0.30`, chosen by measuring it against test-set predictions rather than guessing (section D). Fallback message now suggests example topics. Empty/whitespace/punctuation-only input is caught before the model ever sees it. Exit now accepts `quit`, `exit`, `bye`, `goodbye`. Startup banner shows live model stats from `model_info.json` instead of hard-coded numbers.

## B. Why it was changed

More patterns per intent, with genuinely different phrasings rather than near-duplicates, gives TF-IDF more distinct vocabulary to separate classes and stops the model memorizing a handful of near-identical sentences. Bigrams let the model use two-word cues (`"working hours"`, `"cancel order"`) that a single word can't capture on its own. Keeping negation words intact in preprocessing matters because `"i don't need help"` and `"i need help"` should not become identical after cleaning — though note the limitation on this below. Cross-validation was added because one train/test split on ~560 examples is noisy; five folds give a steadier accuracy estimate. `C` was tuned instead of assumed, because the intuitive guess ("small dataset → regularize heavily") turned out to be wrong for this data (see below).

## C. Before vs after

| | Before | After |
|---|---|---|
| Training examples | ~130 | 563 |
| Intents | 10 | 19 |
| TF-IDF | unigrams only | unigrams + bigrams, `sublinear_tf` |
| Model | LogisticRegression(C=1.0 default) | LogisticRegression(C=10, tuned via CV) |
| Split | 80/20 stratified | 80/20 stratified + 5-fold CV |
| Evaluation | accuracy only | accuracy, precision, recall, F1, classification report, confusion matrix, CV mean/std |
| Fallback threshold | 0.25 (rough guess) | 0.30 (measured against test predictions) |
| Empty/gibberish input | not explicitly handled | explicitly handled, no crashes |

## D. Model performance (actual numbers from this run)

- Training accuracy: **100.0%**
- Test accuracy (single 80/20 split): **81.4%**
- 5-fold cross-validation: **82.8% mean, 2.7% std dev** (folds: 78.8%, 80.5%, 85.0%, 85.7%, 83.9%)
- Weighted precision / recall / F1 (test set): **0.84 / 0.81 / 0.81**

**On the train/test gap:** training accuracy is 100%, and that's worth being honest about rather than hiding — with short sentences and a moderate vocabulary, this dataset is close to linearly separable, so the model can fit the training set almost perfectly. What actually matters is that cross-validation (82.8%) and the held-out test accuracy (81.4%) land close to each other, which shows the model generalizes reasonably rather than the 100% being pure memorization with nothing to back it up on unseen phrasing. The original project's 97%/55% gap was a real generalization problem; the new gap is smaller and the CV/test agreement is the evidence for that, not the training number itself.

**On tuning `C`:** the natural assumption is "small dataset → strong regularization → small `C`". That assumption was tested, not trusted: 5-fold CV accuracy was measured across `C = 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 50`. Accuracy rose steadily from ~19% at `C=0.1` (badly underfit) up to ~78% at `C=1` and ~83% at `C=10`, then flattened. So for this dataset, more regularization was hurting, not helping — the assumption in the previous draft of this script was wrong, and `C=10` (right where the curve plateaus) replaced it.

**On the confidence threshold:** measured against the test set's predicted probabilities. At `0.20`, 11 wrong predictions were confident enough to slip through; at `0.30`, that drops to 3, while genuinely random/gibberish input (`"asdfghjkl"`, `"purple elephants dance quietly"`) scored around 0.10–0.12 in testing — comfortably below the line. `0.30` was chosen as the point where wrong-but-confident answers are rare without rejecting too many correct ones.

**Weakest intent:** `age` (33% recall on the test set) — the confusion matrix shows it gets confused with `name` and `greeting` (e.g. "what year were you made" reads similarly to a `name`/`creator` question to the model). More, more clearly distinct `age` patterns would help most going forward.

**Known limitation:** preprocessing preserves negation words ("not"), but the *training data* has no negated examples for any intent, so the model can't yet learn that `"i don't need help"` differs from `"i need help"` — it will still classify both as `help`. Fixing that would mean adding negated patterns to the dataset, not just preprocessing changes.

## E. Explainability notes for a viva

1. **NLP here** = turning raw sentences into numeric features a model can learn from, then predicting a category (intent) from them.
2. **TF-IDF** = Term Frequency × Inverse Document Frequency. It scores each word/phrase in a sentence by how often it appears there, divided down by how common it is across all sentences — so distinctive words score higher than filler words like "the" or "you".
3. **Why TF-IDF**: with ~560 short sentences it's fast, interpretable, and doesn't need pretrained embeddings or GPUs — appropriate for a small intent-classification problem.
4. **N-grams**: contiguous sequences of N words treated as a single feature. `ngram_range=(1,2)` means both single words and two-word phrases are used, so "good morning" is captured as its own feature, not just "good" and "morning" separately.
5. **Why Logistic Regression**: it's a simple linear classifier — one weight per feature per class — that's fast to train, doesn't need much data compared to deep learning, and its decisions are easy to explain (higher weighted overlap with an intent's vocabulary → higher probability for that intent).
6. **Intent**: a category of user goal (e.g. `greeting`, `cancel_order`) that the bot maps a sentence to before picking a canned response.
7. **Response selection**: after the model predicts an intent, the bot picks randomly from that intent's list of pre-written responses, so the same input doesn't always get the exact same reply.
8. **Confidence**: `predict_proba()` returns a probability for every possible intent; the highest one is the "confidence" of the top prediction.
9. **Why a fallback is necessary**: without one, the model always outputs *some* intent, even for gibberish — a threshold catches cases where it's essentially guessing.
10. **Why the original model overfit**: only ~13 examples per intent means the model can memorize specific training sentences rather than learning generalizable patterns, so it does well on training data and poorly on anything phrased differently.
11. **Why more data helps**: more diverse examples per intent give the model more chances to see the *general* pattern of an intent rather than one specific wording of it.
12. **What the confusion matrix shows**: exactly which intents get mixed up with which — a single accuracy number can't tell you that `age` gets predicted as `name`, but the matrix can.
13. **Why cross-validation**: a single train/test split's accuracy depends on which sentences happened to land in the test set; averaging over 5 different splits gives a more trustworthy estimate of real-world performance, plus a standard deviation showing how stable that estimate is.
