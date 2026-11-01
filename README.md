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

An **abductive explanation** for the prediction $h(\mathbf{x})$ of a classifier
$h : \lbrace 0,1 \rbrace^d \to \lbrace 0,1 \rbrace$ is a subset-minimal set of features
$I \subseteq [d]$ such that the restriction $\mathbf{x}_I$ determines $h(\mathbf{x})$.
Such explanations are often too large to be understood by humans. A
**probabilistic explanation** keeps a subset $S \subseteq I$ of at most $k$
features, and accepts a small error

```math
\varepsilon_{h,\mathbf{x}}(S) \;=\; \frac{\big|\lbrace \mathbf{y} \in \lbrace 0,1 \rbrace^d : h(\mathbf{y}) \neq h(\mathbf{x}),\ \mathbf{y}_S = \mathbf{x}_S \rbrace\big|}{\big|\lbrace \mathbf{y} \in \lbrace 0,1 \rbrace^d : \mathbf{y}_S = \mathbf{x}_S \rbrace\big|}
\;=\; \frac{\mu_{h,\mathbf{x}}(S)}{2^{\,d - |S|}}
```

that is, the probability that a random instance agreeing with $\mathbf{x}$ on
$S$ is classified differently from $\mathbf{x}$. The problem is

```math
\min_{S \subseteq I,\ |S| \le k} \; \varepsilon_{h,\mathbf{x}}(S).
```

The paper shows that this problem is **NP-hard, even for decision trees** and
even when $I$ is an abductive explanation. To get around this, it relies on two
properties:

- the *unnormalized* error $\mu_{h,\mathbf{x}}(S)$, the number of mistakes, is
  **supermodular and non-increasing**, and its curvature $c$ over $2^I$ satisfies
  $c < 1$ when $I$ is abductive;
- the normalization factor $2^{d - |S|}$ depends only on the size of $S$.

Combining greedy supermodular minimization with a level-wise selection gives two
algorithms whose guarantees depend on $c$. Let $S^\ast$ be an optimal solution:

| Algorithm | Strategy | Guarantee |
|---|---|---|
| **Greedy Descent (GD)** | starts from $I$ and removes features | $\varepsilon_{h,\mathbf{x}}(S_{GD}) \le \frac{e^p - 1}{p}\, \varepsilon_{h,\mathbf{x}}(S^\ast)$ with $p = \frac{c}{1-c}$, and $\lvert S_{GD} \rvert \le k$ |
| **Greedy Ascent (GA)** | starts from $\emptyset$ and adds features | $\varepsilon_{h,\mathbf{x}}(S_{GA}) \le \frac{1}{1-c}\, \varepsilon_{h,\mathbf{x}}(S^\ast)$, and $\lvert S_{GA} \rvert \le k \ln \frac{2e}{c}$ |

For a decision tree $T$, $\varepsilon_{h,\mathbf{x}}(S)$ is evaluated in $O(\lvert S \rvert \cdot \lvert T \rvert)$ time from an
orthogonal DNF of the tree, so GD and GA run in polynomial time. On 50 benchmarks, they
find explanations whose error is close to that of the exact SAT-based approach
of [Arenas et al. (2022)](https://arxiv.org/abs/2207.12213). They take less than
0.1 s per explanation, and they scale to high-dimensional tasks where the SAT
solver times out.

## Repository structure

```
.
├── code/
│   ├── UAI_Functions.py   GA (Approx_ascendant), GD (Approx_descent), the error eps (W_f),
│   │                      the number of mistakes mu (M_s), and the SAT binary search (Dichotomie)
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
takes a test instance $\mathbf{x}$, and computes its path-explanation $I$ (the features on
the root-to-leaf path of $\mathbf{x}$). It then reduces $I$ with GD and GA, and compares
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
print(W_f(phi, S_gd, n), W_f(phi, S_ga, n))        # errors eps_{h,x}(S)
```

Instances must be passed as NumPy arrays or lists, not as pandas Series.

## Benchmarks

The experiments use 50 binary classification datasets from UCI, OpenML and
Kaggle. Their sources are listed in Table 4 of the
[supplementary material](https://proceedings.mlr.press/v216/bounia23a/bounia23a-supp.pdf).
For each benchmark, the performance is averaged over $m = \min(s, 150)$ test
instances, where $s$ is the size of the test set, with $k = 7 \pm 2$. The exact
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
