# Self-Pruning Neural Network (CIFAR-10)

## 📌 Project Overview

This project explores a simple but interesting idea:
**Can a neural network learn to remove its own unnecessary connections during training?**

Instead of pruning weights after training, the model is designed to **learn which connections are important as it trains**, using a gating mechanism and regularization.

The implementation is built using PyTorch and trained on the CIFAR-10 dataset.

---

## 🧠 Key Idea

Each weight in the network is paired with a **learnable gate**:

* The gate controls whether a connection is active
* Gate values are constrained between 0 and 1 using a sigmoid function
* During training, the model learns to reduce less useful connections

In practice:

```
effective_weight = weight × sigmoid(gate_score)
```

This allows the network to **softly prune itself** instead of relying on manual pruning steps.

---

## ⚙️ Project Structure

```
project/
│
├── model.py        # Custom PrunableLinear layer + network
├── train.py        # Training and evaluation pipeline
├── utils.py        # Sparsity loss and helper functions
├── report.md       # Detailed analysis and results
├── README.md       # Project documentation
```

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install torch torchvision matplotlib
```

---

### 2. Run training

```bash
python train.py
```

This will:

* Train the model for different λ (lambda) values
* Print accuracy and sparsity results

---

## 📊 Results Summary

| Lambda (λ) | Accuracy (%) | Sparsity (%) |
| ---------- | ------------ | ------------ |
| 0.001      | ~42%         | ~1.7%        |
| 0.01       | ~38%         | ~1.7%        |
| 0.1        | ~35%         | ~1.7%        |

---

## 🔍 Observations

* Increasing λ reduces accuracy (as expected due to regularization)
* However, sparsity remains low (~1–2%)

This shows that while the model is penalized, it tends to **shrink weights rather than fully eliminate them**

---

## 🧠 What I Learned

This project helped me understand:

* How to build **custom PyTorch layers**
* How gradients flow through non-standard architectures
* How regularization affects model behavior
* Why **theoretical ideas don’t always translate directly into strong practical results**

---

## ⚠️ Limitations

* The model does not achieve strong pruning (low sparsity)
* Sigmoid-based gating makes it difficult for values to reach exact zero
* Limited training epochs may restrict convergence

---

## 💡 Possible Improvements

Some ideas to improve results:

* Train for more epochs
* Use stronger regularization (higher λ values)
* Modify gate behavior (e.g., sharper activations)
* Regularize gate scores directly
* Explore alternative sparsity methods

---

## 🏁 Conclusion

This project successfully implements a **self-pruning neural network**, demonstrating how models can learn to control their own structure during training.

Even though strong sparsity was not achieved, the experiment provides valuable insight into the **challenges of differentiable pruning techniques**.

---

## 🙌 Final Note

This project focuses not just on getting results, but on **understanding the behavior of the model and learning from it**.

---
