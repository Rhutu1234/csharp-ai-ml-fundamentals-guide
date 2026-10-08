# AI/ML Fundamentals

*A deep-dive walkthrough of the core ideas behind modern AI — covering how AI, machine learning, and deep learning relate, the three main learning paradigms (supervised, unsupervised, reinforcement), training vs. inference, features/labels/datasets, overfitting and underfitting, model evaluation, accuracy/precision/recall/F1 and why accuracy alone can mislead, regression vs. classification, and embeddings — the representation idea that underpins search, recommendations, and large language models.*

---

## Table of Contents

1. [Introduction](#introduction)
2. [AI vs. ML vs. Deep Learning](#1-ai-vs-ml-vs-deep-learning)
3. [Supervised Learning](#2-supervised-learning)
4. [Unsupervised Learning](#3-unsupervised-learning)
5. [Reinforcement Learning](#4-reinforcement-learning)
6. [Training vs. Inference](#5-training-vs-inference)
7. [Features, Labels, and Datasets](#6-features-labels-and-datasets)
8. [Overfitting and Underfitting](#7-overfitting-and-underfitting)
9. [Model Evaluation](#8-model-evaluation)
10. [Accuracy, Precision, Recall, and F1 Score](#9-accuracy-precision-recall-and-f1-score)
11. [Regression vs. Classification](#10-regression-vs-classification)
12. [Embeddings](#11-embeddings)
13. [Putting It Together: A Minimal End-to-End Example](#12-putting-it-together-a-minimal-end-to-end-example)
14. [Common Pitfalls](#13-common-pitfalls)
15. [Quick Reference Table](#quick-reference-table)
16. [Conclusion](#conclusion)

---

## Introduction

Machine learning is the practice of building systems that **learn patterns from data** instead of following rules a programmer wrote by hand. Rather than coding "if the email contains these words, mark it as spam," you show a model thousands of labeled emails and let it work out what distinguishes spam from non-spam on its own. Nearly every term in the AI world — training, inference, overfitting, embeddings — is a piece of that one idea: *find patterns in examples, then apply them to new, unseen cases.*

```plaintext
Traditional programming:   Rules + Data   ->  Answers
Machine learning:          Data + Answers ->  Rules (the "model")
                           then: Model + New Data -> Predictions
```

This guide goes beyond definitions. It explains how the pieces fit together, why the most common beginner mistakes (trusting accuracy, evaluating on training data) are so tempting, and where each concept shows up in real systems.

---

## 1. AI vs. ML vs. Deep Learning

### Three nested circles, not three competing things

```plaintext
Artificial Intelligence (AI)
  └── Machine Learning (ML)
        └── Deep Learning (DL)
```

- **Artificial Intelligence** is the broadest term: any technique that makes a machine behave in ways we'd call intelligent. This includes hand-written rule systems (a chess engine using search and a scoring function, an expert system with `if/then` rules) that involve no learning at all.
- **Machine Learning** is the subset of AI where the system **learns its behavior from data**. Linear regression, decision trees, random forests, and support vector machines are all classic ML.
- **Deep Learning** is the subset of ML that uses **neural networks with many layers**. It's what powers image recognition, speech recognition, and large language models.

### Why deep learning became dominant for certain problems

```plaintext
Classic ML:  a human designs the FEATURES (e.g. "number of exclamation marks"),
             the model learns weights on them.
Deep learning: the network learns useful features ITSELF from raw data
             (pixels, audio samples, text tokens).
```

That automatic feature learning is why deep learning dominates on unstructured data — images, audio, text — where hand-designing good features is very hard. On smaller, structured, tabular data (rows and columns in a business database), classic ML such as gradient-boosted trees often performs as well or better, trains faster, and is easier to explain. Deep learning is not "better ML" — it is the right tool for a particular class of problems.

---

## 2. Supervised Learning

### Learning from examples that come with the right answer

```plaintext
Training data: (input, correct output) pairs

  Email text        ->  "spam"
  House details     ->  $385,000
  X-ray image       ->  "pneumonia"
```

In supervised learning, every training example includes a **label** — the answer you want the model to learn to produce. The model makes a prediction, compares it to the true label, measures the error, and adjusts itself to reduce that error. Repeated across many examples, it learns a mapping from inputs to outputs that (hopefully) generalizes to inputs it has never seen.

It splits into two problem types, covered in Section 10:

- **Classification** — predict a category (spam / not spam).
- **Regression** — predict a number (a house price).

Supervised learning is the most widely used paradigm in industry, and its main cost is that **labels are expensive**: someone — or some process — has to produce the correct answer for every training example.

---

## 3. Unsupervised Learning

### Finding structure in data that has no labels

```plaintext
Training data: inputs only — NO correct answers provided

  Customer purchase histories -> "which customers behave similarly?"
```

The model isn't told what to find; it looks for structure on its own. Common tasks:

- **Clustering** — group similar items together (customer segmentation with k-means).
- **Dimensionality reduction** — compress many features into a few while keeping the important variation (PCA, used for visualization and noise reduction).
- **Anomaly detection** — flag points that don't fit the normal pattern (unusual transactions).

```plaintext
Important: "unsupervised" does not mean "no human judgment." Someone still
  chooses the number of clusters, the features, and — crucially — decides
  whether the groups the model found are actually MEANINGFUL.
```

A related idea worth knowing is **self-supervised learning**: the labels are generated from the data itself (hide a word in a sentence and train the model to predict it). This is how large language models are pretrained, and it sidesteps the labeling cost of ordinary supervised learning.

---

## 4. Reinforcement Learning

### Learning by trial and error, guided by rewards

```plaintext
Agent --(takes action)--> Environment
Agent <--(new state + reward)-- Environment

Goal: learn a policy (what action to take in each state)
      that maximizes TOTAL reward over time.
```

There is no dataset of correct answers. An **agent** interacts with an **environment**, takes actions, and receives **rewards** (or penalties). Over many attempts it learns which actions lead to the most cumulative reward. The classic examples are game-playing systems, robotics, and control problems.

Two ideas make it distinct from supervised learning:

- **Delayed reward** — an action's consequence may only become clear many steps later (a chess move that loses the game 30 turns on), so the agent must work out which earlier actions deserve credit or blame.
- **Exploration vs. exploitation** — the agent must balance trying new actions (to discover better strategies) against repeating the best-known one (to collect reward now).

Reinforcement learning is also used in language-model training: *reinforcement learning from human feedback* (RLHF) uses human preference ratings as the reward signal to shape how a model responds.

---

## 5. Training vs. Inference

### Two very different phases with very different costs

```plaintext
TRAINING:   learn the model's parameters (weights) from data
            - happens once (or periodically)
            - compute-heavy, can take hours to months
            - requires the labeled / training dataset

INFERENCE:  use the already-trained model to make predictions on new data
            - happens constantly, in production
            - far cheaper PER prediction
            - requires only the trained model and the new input
```

During **training**, the model's parameters are adjusted repeatedly to reduce a **loss** (a number measuring how wrong its predictions are). Once training finishes, the parameters are frozen and saved. **Inference** just runs new inputs through that frozen model to produce outputs — no learning happens.

Why the distinction matters in practice:

- **Cost profile** — training is a large, up-front cost; inference is a smaller per-request cost, but it's paid millions of times, so for a popular product total inference cost can exceed training cost.
- **Latency** — inference often sits in a user-facing request path, so speed matters far more than it does during training.
- **Staleness** — a deployed model knows only what it saw during training. When the real world changes, the model doesn't automatically update; it has to be retrained.

---

## 6. Features, Labels, and Datasets

### The raw material of every ML project

```plaintext
Example: predicting house prices

  sqft  | bedrooms | age | neighborhood |   price      <- label (target)
  ------+----------+-----+--------------+----------
  1400  |    3     | 12  |   Westside   |  $310,000
  2100  |    4     |  5  |   Hillcrest  |  $520,000
  ...
  \_________ features (inputs) _________/
```

- **Features** are the input variables the model uses to make a prediction (square footage, bedrooms, age).
- **Labels** (also called the **target**) are the output you want to predict (price). They exist only in supervised learning.
- A **dataset** is the collection of examples; each row is one **example** (or *sample*).

### Splitting the dataset: train, validation, test

```plaintext
Full dataset
  ├── Training set    (~70%)  -> the model LEARNS from this
  ├── Validation set  (~15%)  -> used to TUNE choices (model type, settings)
  └── Test set        (~15%)  -> touched ONCE at the end, for an honest final score
```

Splitting is not bookkeeping — it is what makes evaluation honest (Section 8). Common data-preparation concerns:

- **Feature engineering** — creating more informative features from raw ones (extracting "day of week" from a timestamp).
- **Encoding** — converting categories like `neighborhood` into numbers the model can consume.
- **Scaling** — putting features on comparable ranges so one large-valued feature doesn't dominate.
- **Data quality** — missing values, duplicates, and mislabeled examples. The saying "garbage in, garbage out" is accurate: data quality usually matters more than model choice.
- **Data leakage** — information in the features that wouldn't actually be available at prediction time (using "was a refund issued" to predict "will this order be returned"). Leakage produces spectacular test scores and useless production models.

---

## 7. Overfitting and Underfitting

### The central tension in all of machine learning

```plaintext
Underfitting:  model is TOO SIMPLE — misses real patterns.
               Poor on training data AND poor on new data.

Good fit:      captures the real pattern, ignores noise.
               Good on training data AND good on new data.

Overfitting:   model is TOO FLEXIBLE — memorizes training data, noise included.
               Excellent on training data, POOR on new data.
```

An overfit model has effectively memorized the training examples — including their random quirks — rather than learning the underlying pattern. It looks brilliant on data it has seen and falls apart on data it hasn't. The signature symptom is a **large gap between training performance and validation/test performance.**

```plaintext
            Training score   Validation score   Diagnosis
            -------------    ----------------   ---------
             60%               58%               Underfitting (both low)
             91%               89%               Good fit (both high, small gap)
             99%               71%               Overfitting (big gap)
```

### Fixing each problem

```plaintext
To reduce OVERFITTING:
  - Get more (and more varied) training data
  - Use a simpler model
  - Regularization (penalize model complexity)
  - Early stopping (stop training when validation score stops improving)
  - Dropout (neural networks: randomly disable units during training)
  - Cross-validation to detect it reliably

To reduce UNDERFITTING:
  - Use a more expressive model
  - Add or engineer more informative features
  - Train longer
  - Reduce regularization
```

This is the **bias–variance trade-off** in practical terms: simple models have high bias (they systematically miss patterns), flexible models have high variance (they're overly sensitive to the particular training sample). The goal isn't to eliminate either — it's to find the balance that generalizes best.

---

## 8. Model Evaluation

### A model's training score tells you almost nothing — evaluate on data it hasn't seen

```plaintext
Rule: NEVER evaluate a model on the data it was trained on.
      The score would measure memorization, not generalization.
```

The only meaningful question is: **how well does this model perform on new data it has never seen?** That's why the test set exists and why it must be kept strictly separate from training.

### Cross-validation: a more reliable estimate from limited data

```plaintext
k-fold cross-validation (k = 5):

  Fold 1: [TEST][train][train][train][train]  -> score 1
  Fold 2: [train][TEST][train][train][train]  -> score 2
  Fold 3: [train][train][TEST][train][train]  -> score 3
  Fold 4: [train][train][train][TEST][train]  -> score 4
  Fold 5: [train][train][train][train][TEST]  -> score 5

  Final estimate = average of the 5 scores (and look at the spread too)
```

A single train/test split can be lucky or unlucky. Cross-validation trains and tests multiple times on different splits and averages the result, giving a more stable estimate and a sense of how much the score varies.

### Other evaluation principles worth knowing

```plaintext
- Compare against a BASELINE — a trivial model (always predict the majority
  class, or the average value). A "95% accurate" model is unimpressive if
  guessing the majority class alone scores 94%.
- Split TIME-ORDERED data by time (train on the past, test on the future),
  never randomly — otherwise the model peeks at the future.
- Don't tune repeatedly against the test set; that quietly turns it into
  a second training set and inflates the final number.
- Choose a metric that matches the real-world COST of mistakes (Section 9).
```

---

## 9. Accuracy, Precision, Recall, and F1 Score

### All four metrics come from one table: the confusion matrix

For a yes/no (binary) classifier:

```plaintext
                        Predicted: Positive     Predicted: Negative
Actual: Positive        True Positive (TP)      False Negative (FN)
Actual: Negative        False Positive (FP)     True Negative (TN)
```

```plaintext
Accuracy  = (TP + TN) / (TP + TN + FP + FN)   -> "what fraction of ALL predictions were right?"
Precision = TP / (TP + FP)                    -> "of everything I FLAGGED, how much was truly positive?"
Recall    = TP / (TP + FN)                    -> "of everything truly positive, how much did I CATCH?"
F1 score  = 2 * (Precision * Recall) / (Precision + Recall)
                                              -> the harmonic mean of precision and recall
```

### A worked example — and why accuracy misleads

A fraud-detection model reviews 1,000 transactions. Only 60 are actually fraud.

```plaintext
TP = 40   (fraud, correctly flagged)
FN = 20   (fraud, missed)
FP = 10   (legitimate, wrongly flagged)
TN = 930  (legitimate, correctly passed)

Accuracy  = (40 + 930) / 1000        = 0.97   (97%)
Precision = 40 / (40 + 10)           = 0.80   (80%)
Recall    = 40 / (40 + 20)           = 0.67   (about 67%)
F1        = 2 * 0.80 * 0.67 / (0.80 + 0.67) = 0.73 (about 73%)
```

97% accuracy sounds excellent — yet the model **missed a third of all fraud.** Because fraud is rare, the huge number of easy true negatives inflates accuracy. In fact, a useless model that labels *every* transaction "legitimate" would score **94% accuracy** (940 of 1,000 correct) while catching zero fraud. This is the **class imbalance problem**, and it's why accuracy alone is a dangerous metric whenever one class is rare.

### Choosing between precision and recall

```plaintext
Precision matters when FALSE POSITIVES are costly:
  - Spam filter: wrongly sending a real email to spam is bad.
  - Recommending a drug treatment, flagging a customer as fraudulent.

Recall matters when FALSE NEGATIVES are costly:
  - Cancer screening: missing a real case is far worse than a false alarm.
  - Fraud or security detection: a missed attack is expensive.

F1 is useful when you need ONE number balancing both, especially with imbalanced classes.
```

Precision and recall typically trade off against each other: most classifiers output a probability, and you choose a **threshold** to turn it into yes/no. Lowering the threshold catches more positives (higher recall) but flags more wrongly (lower precision), and vice versa. There's no universally correct threshold — it's a business decision about which mistake costs more.

---

## 10. Regression vs. Classification

### The difference is the type of thing being predicted

```plaintext
CLASSIFICATION:  predict a CATEGORY (discrete label)
  - Is this email spam or not?            (binary)
  - Which digit is in this image, 0-9?    (multi-class)
  - Which tags apply to this article?     (multi-label)

REGRESSION:      predict a NUMBER (continuous value)
  - What will this house sell for?
  - How many units will we sell next week?
  - What temperature will it be tomorrow?
```

The two require different **loss functions** and different **evaluation metrics**:

```plaintext
Classification metrics:  accuracy, precision, recall, F1, ROC-AUC  (Section 9)
Regression metrics:      MAE  (mean absolute error — average size of the miss)
                         MSE / RMSE  (penalizes large misses more heavily)
                         R²   (fraction of variance the model explains)
```

```plaintext
Accuracy makes no sense for regression — a predicted price of $384,900 vs.
an actual $385,000 isn't "wrong"; it's off by $100. Regression asks
"HOW FAR off?", classification asks "right or wrong?"
```

A useful nuance: many classifiers internally predict a number — a probability — and then apply a threshold to produce a category. Logistic regression, despite its name, is a **classification** algorithm for exactly this reason. The deciding question is always the type of the final output you need, not the algorithm's name.

---

## 11. Embeddings

### Representing things — words, images, users, products — as points in a space where distance means similarity

An **embedding** is a list of numbers (a *vector*) that represents an item so that **similar items end up close together.**

```plaintext
"king"   ->  [0.21, -0.43,  0.88, ...]   (hundreds or thousands of numbers)
"queen"  ->  [0.19, -0.40,  0.85, ...]   <- very close to "king"
"banana" ->  [-0.72, 0.15, -0.10, ...]   <- far from both
```

Computers can't directly work with meaning; they work with numbers. A one-hot encoding of words (one column per vocabulary word) captures no relationship between them — "cat" and "kitten" are as unrelated as "cat" and "tractor." An embedding is **learned** from data so that the geometry itself encodes meaning: words used in similar contexts get similar vectors.

### Measuring similarity

```python
import numpy as np

def cosine_similarity(a, b):
    # 1.0 = same direction (very similar), 0 = unrelated, -1 = opposite
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

king   = np.array([0.21, -0.43, 0.88])
queen  = np.array([0.19, -0.40, 0.85])
banana = np.array([-0.72, 0.15, -0.10])

print(cosine_similarity(king, queen))    # close to 1.0
print(cosine_similarity(king, banana))   # much lower (negative here)
```

(The three-number vectors above are illustrative only; real embeddings typically have hundreds to thousands of dimensions.)

### Where embeddings are used

```plaintext
- Semantic search:       embed the query and the documents, return the nearest
                         documents — matches MEANING, not just keywords.
- Recommendations:       embed users and items; recommend items near the user.
- Retrieval-augmented generation (RAG): retrieve relevant text chunks by embedding
                         similarity and give them to a language model as context.
- Clustering & deduplication: group or detect near-identical items.
- Input to other models: neural networks and LLMs convert their input tokens
                         into embeddings as the very first step.
```

Embeddings are typically stored and searched in a **vector database** (or a vector index inside an existing database), using nearest-neighbor search to find the closest vectors quickly.

One honest caveat: embeddings reflect the data they were trained on, including its biases and blind spots, and embeddings from *different* models live in different spaces — you can't compare a vector from one model with a vector from another.

---

## 12. Putting It Together: A Minimal End-to-End Example

### A supervised classification workflow using scikit-learn

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

# 1. Dataset: features (X) and labels (y)
X, y = load_breast_cancer(return_X_y=True)

# 2. Split — hold out a test set the model never sees during training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. TRAINING — the model learns from the training data
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 4. INFERENCE — predict on data the model has never seen
y_pred = model.predict(X_test)

# 5. EVALUATION — on the held-out test set only
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1       :", f1_score(y_test, y_pred))

# 6. Overfitting check — compare training vs. test performance
print("Train accuracy:", model.score(X_train, y_train))
print("Test  accuracy:", model.score(X_test, y_test))
```

Every concept in this guide appears in those few lines: a dataset of **features** and **labels**, a **train/test split**, **training** (`fit`), **inference** (`predict`), **classification** metrics, and the train-vs-test comparison that reveals **overfitting**. (Exact scores will vary slightly by library version.)

---

## 13. Common Pitfalls

| Pitfall | Why it hurts | Better approach |
|---|---|---|
| Evaluating on the training data | The score measures memorization, not real-world performance | Always hold out a test set the model never trained on (Section 8) |
| Trusting accuracy on imbalanced data | A model that always predicts the majority class can score 94%+ while catching nothing | Use precision, recall, and F1, and compare against a baseline (Section 9) |
| Data leakage | Features contain information unavailable at prediction time, producing inflated scores and failed deployments | Ask of every feature, "would I have this at the moment of prediction?" (Section 6) |
| Tuning repeatedly against the test set | Quietly turns the test set into training data and inflates the final score | Tune on a validation set or via cross-validation; touch the test set once (Section 6, 8) |
| Randomly splitting time-ordered data | The model effectively sees the future during training | Split by time: train on the past, test on the future (Section 8) |
| Ignoring the gap between training and test scores | A large gap is the signature of overfitting | Compare both scores every time; add data, regularize, or simplify (Section 7) |
| Reaching for deep learning by default | Often slower, more data-hungry, and harder to explain than classic ML on tabular data | Start with a simple baseline; escalate only when it falls short (Section 1) |
| Picking a metric without considering the cost of errors | Optimizing the wrong metric optimizes the wrong behavior | Decide whether false positives or false negatives hurt more, then choose precision/recall accordingly (Section 9) |
| Comparing embeddings from different models | Each model defines its own vector space; the numbers aren't comparable | Embed everything you compare with the same model (Section 11) |
| Assuming a deployed model keeps learning | Inference doesn't update the model; it goes stale as the world changes | Monitor performance in production and retrain periodically (Section 5) |

---

## Quick Reference Table

| Concept | One-line definition | Why it matters |
|---|---|---|
| AI | Any technique that makes machines behave intelligently | The broadest umbrella term |
| Machine learning | Systems that learn patterns from data | Replaces hand-written rules with learned behavior |
| Deep learning | ML using many-layered neural networks | Dominates images, audio, and text |
| Supervised learning | Learn from (input, correct-answer) pairs | Most common paradigm; needs labels |
| Unsupervised learning | Find structure in unlabeled data | Clustering, compression, anomaly detection |
| Reinforcement learning | Learn actions from rewards via trial and error | Games, robotics, and shaping LLM behavior |
| Training | Adjusting parameters to reduce loss | The expensive, one-time learning phase |
| Inference | Using a trained model on new inputs | The cheap-per-call, constant production phase |
| Feature | An input variable | What the model uses to predict |
| Label / target | The output to predict | Present only in supervised learning |
| Overfitting | Memorizing training data, failing on new data | Shows as a big train-vs-test gap |
| Underfitting | Too simple to capture the pattern | Poor on both training and new data |
| Accuracy | (TP + TN) / total | Misleading when classes are imbalanced |
| Precision | TP / (TP + FP) | Of what I flagged, how much was right |
| Recall | TP / (TP + FN) | Of what was truly positive, how much I caught |
| F1 score | Harmonic mean of precision and recall | One number balancing both |
| Classification | Predict a category | Evaluated with accuracy, precision, recall, F1 |
| Regression | Predict a number | Evaluated with MAE, RMSE, R² |
| Embedding | A vector where distance reflects similarity | Powers search, recommendations, and LLMs |

---

## Conclusion

Almost everything in machine learning follows from one loop: **learn patterns from examples, then check honestly whether those patterns hold up on examples the model has never seen.** The learning paradigms — supervised, unsupervised, reinforcement — differ in what feedback the model gets while learning. Training and inference separate the costly learning phase from the everyday prediction phase. Features, labels, and the train/validation/test split define what the model learns from and how fairly it's judged. Overfitting and underfitting describe the two ways that judgment can come out badly, and the evaluation metrics — accuracy, precision, recall, F1 for classification, MAE and RMSE for regression — are how you measure it, provided you choose the one that matches what mistakes actually cost.

The most valuable habit this guide can offer is skepticism toward a single impressive number. A 97% accuracy that misses a third of the fraud, a training score that hides overfitting, a test score inflated by leakage — each looks like success until you ask what it was measured on and what it ignores. Embeddings round out the picture as the representation idea behind modern search, recommendation, and language models: turn meaning into geometry, and similarity becomes distance. With these fundamentals, the more advanced topics — neural network architectures, transformers, fine-tuning, RAG — stop being separate mysteries and become variations on a framework you already understand.

---

*Found this useful? Feel free to star the repo, open an issue with corrections, or share the "the model had 99% accuracy and caught nothing" story that made the case for precision and recall click better than any formula.*
