# Feature Selection Using Dispersive Flies Optimisation for Text Classification

**MSc Data Science Dissertation — University of Greenwich (2025)**  
**Grade: 85/100 | Award: Pass with Distinction**  
**Supervisor: Dr Mohammad Majid al-Rifaie, Goldsmiths University of London**

---

## What Is This Project?

Text classification models convert words into numbers — producing 
thousands of features per dataset. Most of these features are 
irrelevant, redundant, or misleading (distractors), which causes:

- Models to learn noise instead of real patterns
- Poor accuracy on unseen data
- Slow training and high computational cost

This project builds an enhanced **Dispersive Flies Optimisation (DFO)** 
algorithm to automatically identify and keep only the most informative 
features — discarding the rest — before training a classifier.

DFO is a swarm intelligence algorithm inspired by the collective 
behaviour of flies searching for food sources. A population of 30 
virtual flies explore feature combinations simultaneously, share 
information through a ring topology, and converge on the optimal 
feature subset through collaborative search.

---

## Enhancements to the Original DFO Algorithm

The canonical DFO (al-Rifaie, 2014) was designed for continuous 
numerical optimisation. This project adapts it for text feature 
selection with three specific enhancements:

1. **Hybrid Mutual Information Initialisation** — flies are seeded 
   with the top 25% most informative features rather than random 
   starts, accelerating convergence
2. **Distractor Penalty Mechanism** — fitness function penalises 
   selection of features with low mutual information scores, 
   actively discouraging noise selection
3. **Text Vectorisation Compatibility** — supports both 
   bag-of-words (Dexter) and TF-IDF (Twitter, SMS) schemes

---

## Datasets

| Dataset | Samples | Features | Classes | Challenge |
|---------|---------|----------|---------|-----------|
| Dexter (UCI) | 600 | 20,000 | 2 | 50% intentional distractor features |
| Twitter US Airline Sentiment | 14,640 | 6,415 | 3 | Multi-class, imbalanced |
| SMS Spam Collection (UCI) | 5,561 | 4,204 | 2 | 87:13 class imbalance |

---

## Results

| Dataset | Method | Features Selected | Accuracy | F1-Score | AUC-ROC |
|---------|--------|-------------------|----------|----------|---------|
| Dexter | **DFO** | 10,117 | **0.9167** | **0.9107** | **0.9167** |
| Dexter | SelectKBest | 8,000 | 0.9083 | 0.9027 | 0.9164 |
| Dexter | RFECV | 4,000 | 0.8917 | 0.8929 | 0.8917 |
| Twitter | **DFO** | 3,462 | 0.7565 | 0.7574 | 0.8735 |
| Twitter | SelectKBest | 3,000 | 0.7845 | 0.7853 | 0.8972 |
| Twitter | RFECV | 5,133 | 0.7886 | 0.7889 | 0.8988 |
| SMS Spam | **DFO** | 2,053 | 0.9784 | 0.9195 | 0.9812 |
| SMS Spam | SelectKBest | 2,000 | 0.9793 | 0.9231 | 0.9847 |
| SMS Spam | RFECV | 1,684 | 0.9811 | 0.9293 | 0.9843 |

- DFO outperformed both baselines on the hardest dataset (Dexter)
- 40–60% feature reduction achieved across all datasets
- 23 independent runs confirmed stability (Dexter mean: 91.6%, SD: 0.5%)

---

## How the Algorithm Works
Initialise 30 flies using mutual information scores (top 25% features)

↓

Each fly = binary mask (1 = keep feature, 0 = discard)

↓

Fitness = 5-fold CV accuracy with distractor penalty

↓

Each fly updates position using ring topology neighbour + swarm best

↓

Per-dimension disturbance maintains exploration-exploitation balance

↓

Early stopping after 30 iterations without improvement

↓

Best feature mask returned → classifier trained on selected features

---

## Tech Stack

- **Language:** Python
- **ML:** scikit-learn (SGDClassifier with L1, SelectKBest, RFECV)
- **Data Processing:** pandas, NumPy, SciPy (sparse matrices)
- **Visualisation:** Matplotlib
- **Validation:** 5-fold stratified cross-validation, 23 independent runs

---

## Project Structure

├── Code.ipynb          # Full implementation and experiments

├── README.md           # Project documentation

---

## Based On

> al-Rifaie, M.M. (2014) *Dispersive Flies Optimisation*.  
> Proceedings of the 2014 Federated Conference on Computer  
> Science and Information Systems, pp. 529–538.  
> DOI: 10.15439/2014F142

---
