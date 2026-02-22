# Hidden Markov Model (HMM) using Baum–Welch Algorithm — IMPLEMENTED USING JAVASCRIPT

**Name:** SREE LEKSHMI H
**Registration Number:** TCR24CS067

---

## 📌 Project Overview

This project implements a **Hidden Markov Model (HMM)** trained using the **Baum–Welch algorithm (Expectation–Maximization)** entirely in **JavaScript** with a fully interactive **HTML/CSS frontend**.

The system allows users to:

* Enter observation sequences
* Train an HMM model
* Visualize parameters
* See convergence graphs
* Display state transition diagrams
* Decode hidden states using Viterbi algorithm

No backend or external libraries are required — everything runs in the browser.

---

## 🎯 Objectives

* Implement HMM mathematically from scratch
* Apply Baum–Welch parameter estimation
* Visualize learning behavior
* Demonstrate probabilistic sequence modeling

---

##  Concepts Implemented

The project implements the three classical HMM problems:

### 1️⃣ Evaluation Problem

Compute likelihood
[
P(O \mid \lambda)
]
using **Forward Algorithm**

---

### 2️⃣ Decoding Problem

Find most likely hidden state sequence using:

**Viterbi Algorithm**

---

### 3️⃣ Learning Problem

Estimate model parameters using:

**Baum–Welch Algorithm (EM)**

---

## ⚙️ Algorithms Used

* Forward algorithm (scaled version)
* Backward algorithm
* Gamma computation
* Xi computation
* Baum–Welch parameter update
* Viterbi decoding
* Random initialization with seed

---

## 📊 Features

✔ Interactive UI
✔ Adjustable parameters (states, iterations, tolerance)
✔ Random seed for reproducibility
✔ Log-likelihood convergence graph
✔ Transition matrix visualization
✔ Emission matrix visualization
✔ HMM state diagram with probabilities
✔ Works fully offline

---

## 🖥️ Technologies Used

* HTML5
* CSS3
* Vanilla JavaScript (ES6)
* SVG for diagram rendering
* Canvas API for plotting

---

## 🚀 How to Run

1. Download or clone repository
2. Open folder
3. Double-click

```
index.html
```

OR

Use VS Code Live Server extension.

No installation required.

---

## 📈 Example Input

```
W H H W H
```

Output shows:

* learned π
* transition matrix A
* emission matrix B
* convergence graph
* predicted hidden states

---

## 🧾 Mathematical Model

An HMM is defined by:

[
\lambda = (A, B, \pi)
]

Where:

* **A** = transition probability matrix
* **B** = emission probability matrix
* **π** = initial state distribution

---

## 🔁 Baum–Welch Learning Principle

Observations → Hidden State Beliefs → Expected Counts → Parameter Updates

The algorithm iteratively maximizes:

[
\max_\lambda P(O \mid \lambda)
]

until convergence.

---


## Applications of HMM

* Speech recognition
* Part-of-speech tagging
* Activity recognition
* Bioinformatics
* Financial modeling

---

## Key Insight

> Observations do not directly change probabilities.
> They reshape belief distributions, which update model parameters.

---

## License

Educational use only.

---

## Author
SREE LEKSHMI H
