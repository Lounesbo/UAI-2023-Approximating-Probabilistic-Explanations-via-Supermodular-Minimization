# Approximating Probabilistic Explanations via Supermodular Minimization

**Louenas Bounia** · **Frédéric Koriche**
CRIL UMR CNRS 8188, Université d'Artois, France

*39th Conference on Uncertainty in Artificial Intelligence (UAI 2023)*, PMLR 216:216–225

[**Paper**](https://proceedings.mlr.press/v216/bounia23a.html) ·
[PDF](https://proceedings.mlr.press/v216/bounia23a/bounia23a.pdf) ·
[Supplementary](https://proceedings.mlr.press/v216/bounia23a/bounia23a-supp.pdf) ·
[OpenReview](https://openreview.net/forum?id=g6t3lXRABRl) ·
[Poster](https://www.auai.org/uai2023/posters/332.pdf)

---

## Overview

An **abductive explanation** for the prediction h(x) of a classifier is a
subset-minimal set of features I such that the restriction x_I determines
h(x). Such explanations are often too large to be understood by humans. A
**probabilistic explanation** keeps a subset S ⊆ I of at most k features, and
accepts a small error

    ε_{h,x}(S) = P_{y : y_S = x_S} [ h(y) ≠ h(x) ]

that is, the probability that an instance agreeing with x on S is classified
differently from x.

The paper shows that finding a subset S ⊆ I with |S| ≤ k that minimizes
ε_{h,x}(S) is **NP-hard, even for decision trees** and even when I is an
abductive explanation. To get around this, it relies on two properties:

- the *unnormalized* error μ_{h,x}(S), the number of mistakes, is
  **supermodular and non-increasing**, with curvature c < 1 when I is abductive;
- the normalization factor 2^(d−|S|) depends only on |S|.

Combining greedy supermodular minimization with a level-wise selection gives two
algorithms with approximation guarantees that depend on c:

| Algorithm | Strategy | Guarantee |
|---|---|---|
| **Greedy Descent (GD)** | starts from I and removes features | (e^p − 1)/p-approximation, with p = c/(1 − c), and \|S\| ≤ k |
| **Greedy Ascent (GA)** | starts from ∅ and adds features | 1/(1 − c)-approximation, and \|S\| ≤ k ln(2e/c) |

For decision trees, ε_{h,x}(S) is evaluated in linear time from an orthogonal
DNF of the tree, so GD and GA run in polynomial time. On 50 benchmarks, they
find explanations whose error is close to that of the exact SAT-based approach
of [Arenas et al. (2022)](https://arxiv.org/abs/2207.12213). They take less than
0.1 s per explanation, and they scale to high-dimensional tasks where the SAT
solver times out.

## Repository structure

```
.
├── code/
│   ├── UAI_Functions.py   GA (Approx_ascendant), GD (Approx_descent), the error ε (W_f),
│   │                      the number of mistakes μ (M_s), and the SAT binary search (Dichotomie)
│   ├── my_tree.py         decision trees from scikit-learn, orthogonal DNFs, path-explanations
│   ├── encodage_CNF.py    CNF encodings of cardinality constraints, used by my_tree.py
│   ├── encoder.py         SAT encoding of Arenas et al. (2022) for the exact baseline
│   ├── dtree.py           │
│   ├── gen_dt.py          ├─ helpers of the SAT encoding (tree export and parsing, CNF utilities)
│   ├── utils.py           │
│   ├── example.py         example: GA, GD and the SAT baseline on two datasets
│   └── data/
│       ├── tic-tac-toe.csv
│       └── mnist49.csv.gz  MNIST restricted to the digits 4 and 9
├── requirements.txt
└── README.md
```

`encoder.py`, `dtree.py`, `gen_dt.py` and `utils.py` implement the SAT encoding
of Arenas et al. (2022), used as the exact baseline in the experiments.

## Installation

    pip install -r requirements.txt

The dependencies are NumPy, pandas, scikit-learn and [PySAT](https://pysathq.github.io/)
(`python-sat`, with its `pb` extra for pseudo-Boolean constraints). The exact
baseline uses the Glucose 4 solver shipped with PySAT.

## Usage

Run the example from the `code/` folder:

    cd code
    python example.py

For each dataset, the example learns a decision tree with CART (scikit-learn),
takes a test instance x, and computes its path-explanation I (the features on
the root-to-leaf path of x). It then reduces I with GD and GA, and compares
their error with the SAT-based binary search.

On tic-tac-toe, everything runs in a few seconds. On mnist49, GD and GA return
instantly, but the SAT binary search can take a very long time, as in the paper
where the solver times out on this dataset.

The main steps, in Python:

```python
import my_tree as mt
from UAI_Functions import Approx_descent, Approx_ascendant, W_f

T = mt.decision_tree()
T.from_DecisionTreeClassifier(dt)                  # dt: a fitted sklearn DecisionTreeClassifier

x = X_test.iloc[0].to_numpy()                      # instance to explain (NumPy array or list)
target = T.take_decision(instance=x)
phi = T.to_DNF(target=1 - target, threshold_clauses=False)[0].clauses   # orthogonal DNF of the mistakes
I = T.find_direct_reason(instance=x)               # path-explanation of x
n = len(T.bina)                                    # number of Boolean features of the tree

S_gd = Approx_descent(phi=phi, I_s=I, n=n, k=7)    # Greedy Descent
S_ga = Approx_ascendant(phi=phi, I_s=I, n=n, k=7)  # Greedy Ascent
print(W_f(phi, S_gd, n), W_f(phi, S_ga, n))        # errors ε_{h,x}(S)
```

Instances must be passed as NumPy arrays or lists, not as pandas Series.

## Benchmarks

The experiments use 50 binary classification datasets from UCI, OpenML and
Kaggle. Their sources are listed in Table 4 of the
[supplementary material](https://proceedings.mlr.press/v216/bounia23a/bounia23a-supp.pdf).
For each benchmark, the performance is averaged over m = min(s, 150) test
instances, where s is the size of the test set, with k = 7 ± 2. The exact
baseline uses Glucose 4 with a timeout of 30 minutes per explanation task.

## Citation

```bibtex
@InProceedings{pmlr-v216-bounia23a,
  title     = {Approximating probabilistic explanations via supermodular minimization},
  author    = {Bounia, Louenas and Koriche, Frederic},
  booktitle = {Proceedings of the Thirty-Ninth Conference on Uncertainty in Artificial Intelligence},
  pages     = {216--225},
  year      = {2023},
  editor    = {Evans, Robin J. and Shpitser, Ilya},
  volume    = {216},
  series    = {Proceedings of Machine Learning Research},
  month     = {31 Jul--04 Aug},
  publisher = {PMLR},
  pdf       = {https://proceedings.mlr.press/v216/bounia23a/bounia23a.pdf},
  url       = {https://proceedings.mlr.press/v216/bounia23a.html}
}
```

## Acknowledgements

This work benefited from the support of the AI Chair EXPEKCTATION
(ANR-19-CHIA-0005-01) of the French National Research Agency, and was partially
supported by TAILOR, a project funded by the EU Horizon 2020 research and
innovation programme under GA No 952215.
