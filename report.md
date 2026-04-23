# Self-Pruning Neural Network (CIFAR-10)

## Overview

This project explores a neural network that can **prune itself during training** instead of relying on post-training pruning techniques. The core idea is to associate each weight with a learnable gate that determines whether the connection should remain active or be suppressed.

The model is trained on the CIFAR-10 dataset, and a sparsity-inducing regularization term is added to encourage the network to reduce unnecessary connections while maintaining predictive performance.

---

## Methodology

### Prunable Linear Layer

A custom layer, `PrunableLinear`, was implemented to replace the standard fully connected layer.

* Each weight is paired with a **learnable gate score**

* Gate values are computed using a sigmoid function:

  ```
  gate = sigmoid(gate_score)
  ```

* The effective weight used during forward propagation is:

  ```
  pruned_weight = weight × gate
  ```

This design allows the network to **learn which connections are important** during training itself.

---

### Loss Function

The total loss used for training is:

```
Total Loss = CrossEntropyLoss + λ × SparsityLoss
```

* **CrossEntropyLoss**: standard classification loss
* **SparsityLoss**: L1 norm of all gate values

The L1 regularization term encourages the model to reduce gate values, promoting sparsity in the network.

---

### Training Setup

* Dataset: CIFAR-10
* Model: 3-layer fully connected network
* Optimizer: Adam
* Epochs: 10–15
* Batch size: 64

The model was trained using different values of λ to analyze the trade-off between accuracy and sparsity.

---

## Results

| Lambda (λ) | Test Accuracy (%) | Sparsity (%) |
| ---------- | ----------------- | ------------ |
| 0.001      | 42.24             | 1.70         |
| 0.01       | 37.74             | 1.73         |
| 0.1        | 34.86             | 1.71         |

---

## Observations

* Increasing λ leads to a **decrease in accuracy**, which is expected due to stronger regularization.
* However, **sparsity remains nearly constant (~1-2%)**, even for higher values of λ.
* This indicates that the network is **not strongly pruning its weights**, despite the sparsity constraint.

---

## Why Sparsity Remains Low

The primary reason lies in the behavior of the sigmoid function.

* The sigmoid output lies in the range (0, 1), but it rarely reaches exact 0.
* L1 regularization encourages smaller values but does not force them to become zero.
* As a result, most gate values shrink slightly but remain active.
* Since sparsity is measured using a threshold (e.g., 1e-2), very few gates fall below this threshold.

In short, the model learns to **reduce weights slightly rather than completely eliminate them**.

---

## Trade-off Analysis

Although sparsity did not increase significantly, the effect of λ is still visible:

* Lower λ → higher accuracy, minimal regularization
* Higher λ → lower accuracy, stronger penalty on gates

This demonstrates that the regularization term is influencing the model, but not strongly enough to induce hard pruning.

---

## Potential Improvements

Several approaches could improve sparsity in future work:

1. **Increase training duration**
   More epochs may allow gate values to converge closer to zero.

2. **Use larger λ values**
   Stronger regularization may enforce more aggressive pruning.

3. **Sharpen gate activations**
   Scaling gate scores before applying sigmoid can push outputs closer to 0 or 1.

4. **Regularize gate scores instead of gate values**
   This can drive values into the saturated region of the sigmoid function.

5. **Explore alternative methods**
   Techniques such as L0 regularization or hard thresholding could produce stronger sparsity.

---

## Conclusion

This project successfully demonstrates the implementation of a **self-pruning neural network** using learnable gates and L1 regularization.

While the model did not achieve high sparsity, it highlights an important insight:
**standard L1 regularization on sigmoid-based gates is not always sufficient to produce strong pruning behavior.**

The results emphasize the need for careful design of sparsity-inducing mechanisms and provide a strong foundation for further experimentation.

---

## Key Takeaway

Instead of explicitly removing weights, the network learns to **softly suppress connections**, revealing both the potential and limitations of differentiable pruning techniques.

---
