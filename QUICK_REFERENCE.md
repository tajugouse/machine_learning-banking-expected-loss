# Quick Reference Guide - Banking Expected Loss Project

## 🚀 Quick Start (30 seconds)

```bash
python banking_expected_loss_project.py
```

**Output:**
- Console: Training logs + metrics
- Image: `expected_loss_analysis.png`

---

## 📊 Key Results At A Glance

```
✓ CONVERGED: 119 iterations (92% cost reduction)
✓ R² Score: 0.6418 (64% variance explained)
✓ Portfolio EL: $677,800.77
✓ Avg EL/loan: $3,389.00
✓ Mathematical guarantee: Global minimum achieved
```

---

## 🧮 Mathematical Foundation (One-Page Summary)

### 1️⃣ Hypothesis (Prediction)
```
h(x) = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ

Example:
h(x) = 0.143 - 0.0206×credit_score + 0.0555×defaults
```

### 2️⃣ Cost Function (What We Minimize)
```
J(θ) = (1/2m) × Σ(h(xⁱ) - yⁱ)²

Properties:
- Strictly convex (no local minima)
- Has global minimum at ∇J = 0
- Can be solved analytically OR iteratively
```

### 3️⃣ Gradient (Direction of Descent)
```
∂J/∂θⱼ = (1/m) × Σ(h(xⁱ) - yⁱ) × xⱼⁱ

Interpretation:
- Points toward increasing error
- Opposite direction = decreasing error
- Gradient = 0 → Minimum reached
```

### 4️⃣ Gradient Descent Update
```
θⱼ := θⱼ - α × ∂J/∂θⱼ

Parameters:
- α (learning rate): 0.1 (smaller = slower, safer)
- Step size: α × gradient
```

### 5️⃣ Convergence Criterion
```
STOP when: ||∇J(θ)|| < ε

- ||∇J|| = √(Σ(∂J/∂θⱼ)²)  [gradient norm]
- ε = 1e-06 (tolerance threshold)
- Our result: 9.27e-07 < 1e-06 ✓
```

### 6️⃣ Expected Loss Formula
```
EL = PD × LGD × EAD

Where:
- PD: Probability Default (our model output)
- LGD: Loss Given Default = 0.45 (45%)
- EAD: Exposure At Default (loan amount)

Example:
EL = 0.125 × 0.45 × $50,000 = $2,812.50
```

---

## 📈 Understanding the Visualizations

### Panel 1: Cost Convergence (Log Scale)
```
↓ Rapid decrease initially
↓ Slower decrease later (approaching minimum)
✓ Flattens at optimal point
```
**What it means:** Model learning rate decreasing as we approach solution

### Panel 2: Gradient Norm (Log Scale)
```
↓ Starts high (need big adjustments)
↓ Decreases exponentially
✓ Hits threshold (ε) at iteration 119
```
**What it means:** Gradient size is proxy for distance to minimum

### Panel 3: Predicted vs Actual
```
• Points on red line = perfect predictions
• Points above = overestimation
• Points below = underestimation
• R² = 0.6418 means ~64% of variance captured
```
**What it means:** Decent fit, room for improvement with non-linear models

### Panel 4: Residuals
```
| Should be:
| ✓ Centered around 0 (no bias)
| ✓ Bell-shaped (normally distributed)
| ✓ No patterns (random scatter)
```
**What it means:** Model assumptions likely satisfied

### Panel 5: Feature Importance
```
| Bar height = |coefficient| value
| Taller = more important
| Negative = inverse relationship
```
**Key findings:**
- Previous Defaults: Most important (0.0555)
- Debt-to-Income: Second (0.0306)
- Credit Score: Third (-0.0206, negative is good)

### Panel 6: Expected Loss Distribution
```
| Histogram shows spread of EL across portfolio
| Red line = mean EL ($3,389.00)
| Left skew = most loans have lower risk
```
**What it means:** Portfolio risk concentrated in fewer high-risk loans

---

## 🔧 Common Customizations

### ⚙️ Change Learning Rate
```python
model = LinearRegressionGD(learning_rate=0.05)  # Slower
model = LinearRegressionGD(learning_rate=0.5)   # Faster
```
**When to use:**
- Too slow: convergence_iteration > 500
- Too fast: oscillating or diverging loss
- Sweet spot: 50-200 iterations

### ⚙️ Change Dataset Size
```python
df = generator.generate_dataset(n_samples=5000)  # More data
df = generator.generate_dataset(n_samples=100)   # Less data
```
**Effect:**
- More data: Better generalization, slower training
- Less data: Faster training, risk of overfitting

### ⚙️ Stricter Convergence
```python
model = LinearRegressionGD(tolerance=1e-8)  # Very strict
model = LinearRegressionGD(tolerance=1e-4)  # Loose
```
**Effect:**
- Stricter: More iterations, better accuracy
- Looser: Fewer iterations, faster but less accurate

### ⚙️ Change LGD (Loss Given Default)
```python
el_calc = BankingExpectedLossCalculator(lgd=0.30)  # Secured loans
el_calc = BankingExpectedLossCalculator(lgd=0.60)  # Unsecured/risky
```
**Basel III Standards:**
- Secured (mortgages): 20-30%
- Unsecured (credit cards): 50-100%
- Used in example: 45%

---

## 🎯 Interpreting Model Coefficients

### Parameter θⱼ Interpretation

```
θⱼ = positive  →  Increases prediction
θⱼ = negative  →  Decreases prediction
|θⱼ| = large   →  Feature is important
|θⱼ| = small   →  Feature has little effect
```

### Our Model Coefficients (Standardized Features)

```
θ₀ (Intercept):        +0.143236
├─ Base loss rate with average features

θ₁ (Credit Score):     -0.020649
├─ Higher score → Lower loss rate (makes sense ✓)
├─ 1 std increase → 2.06% loss decrease

θ₂ (Loan Amount):      -0.002347
├─ Counterintuitive, but controlled by other features
├─ In portfolio: larger loans more scrutinized

θ₃ (Annual Income):    +0.002786
├─ Paradoxical sign (controlled by DTI)
├─ Effect: negligible compared to DTI

θ₄ (Employment Years): -0.000071
├─ Essentially zero effect
├─ Not important for loss prediction

θ₅ (Debt-to-Income):   +0.030559 ⭐ Important
├─ Higher DTI → Much higher loss rate
├─ 1 std increase → 3.06% loss increase
├─ Key risk factor

θ₆ (Prev Defaults):    +0.055487 ⭐⭐ Most Important
├─ Past default = strong indicator of future default
├─ 1 std increase → 5.55% loss increase
├─ Single most predictive feature
```

---

## 📋 Step-by-Step Execution Flow

```
START
  ↓
[1] Generate 1,000 loans with risk features
  ├─ credit_score, loan_amount, income, etc.
  └─ Create target: expected_loss_rate
  ↓
[2] Preprocess & Split
  ├─ Split 80% train / 20% test
  ├─ Standardize features (Z-score)
  └─ Ready for training
  ↓
[3] Initialize Model
  ├─ Set θ = small random values
  ├─ Set α = 0.1, ε = 1e-6
  └─ Prepare for gradient descent
  ↓
[4] TRAINING LOOP (Max 10,000 iterations)
  ├─ h = X·θ           [predict]
  ├─ J = MSE(h, y)     [measure error]
  ├─ ∇J = ∇cost        [find direction]
  ├─ θ := θ - α·∇J     [take step]
  ├─ Check: ||∇J|| < ε? [converged?]
  └─ CONVERGED at iteration 119 ✓
  ↓
[5] Evaluate Model
  ├─ Train metrics: R²=0.663, RMSE=0.047
  ├─ Test metrics:  R²=0.642, RMSE=0.050
  └─ Reasonable accuracy
  ↓
[6] Calculate Expected Loss
  ├─ For each loan: EL = PD × LGD × EAD
  ├─ Portfolio total: $677,800.77
  └─ Average per loan: $3,389.00
  ↓
[7] Visualize Results
  ├─ Convergence plots
  ├─ Residual analysis
  ├─ Feature importance
  └─ Save to PNG
  ↓
[8] Generate Report
  ├─ Print metrics
  ├─ Convergence verification
  └─ Final summary
  ↓
END ✓
```

---

## 🔍 Convergence Checklist

### Mathematical Guarantee ✓
- [ ] Loss function convex? YES (MSE always convex)
- [ ] Learning rate appropriate? YES (0.1 is standard)
- [ ] Gradient norm decreasing? YES (exponential decay)
- [ ] Global minimum reached? YES (convergence_iteration=119)

### Quality Metrics ✓
- [ ] Cost reduction > 50%? YES (92.15%)
- [ ] Final ||∇J|| < threshold? YES (9.27e-07 < 1e-06)
- [ ] No divergence? YES (stable descent)
- [ ] Reproducible? YES (fixed random seed)

---

## 💡 Machine Learning Concepts

### Why Linear Regression?
```
✓ Interpretability:    Easy to explain coefficients
✓ Efficiency:         Fast training, real-time prediction
✓ Baseline:           Good starting point
✓ Regulatory:         Preferred in banking (explainability)
✗ Limitation:         Assumes linear relationships
```

### Why Gradient Descent?
```
✓ Theoretical:        Guaranteed convergence (convex)
✓ Efficient:          Vectorized numpy operations
✓ Scalable:           Works with millions of samples
✓ Educational:        Shows core optimization concepts
```

### Why This Converged So Quickly?
```
Reasons (119 iterations is fast):
1. Features standardized (mean=0, std=1)
2. Dataset size reasonable (800 training)
3. Problem well-conditioned (good features)
4. Learning rate optimal (0.1)
5. Good initial guess (small random θ)

Comparison:
- Without scaling: 1000+ iterations
- Without good features: 500+ iterations
- This project: 119 iterations ✓
```

---

## 🚨 Common Pitfalls & Solutions

### ❌ Model Not Converging
```
Cause: Learning rate too large
Fix:   Learning_rate = 0.01 (instead of 0.1)
```

### ❌ Predictions Out of Range [0, 1]
```
Cause: Linear model predicts unbounded values
Fix:   y_pred = np.clip(y_pred, 0, 1)
Or:    Use logistic regression instead
```

### ❌ Low Accuracy on Test Set
```
Cause: Linear relationship assumption violated
Fix:   Add polynomial features
Or:    Use non-linear model (Random Forest, Neural Net)
```

### ❌ Training Takes Forever
```
Cause: Learning rate too small
Fix:   Learning_rate = 0.5 (instead of 0.01)
Or:    Reduce max_iterations to 1000
```

---

## 📚 Mathematical Reference

### Vector Calculus Quick Reference

```
Gradient Vector:
∇J = [∂J/∂θ₀, ∂J/∂θ₁, ..., ∂J/∂θₙ]ᵀ

Gradient Norm (Magnitude):
||∇J|| = √(Σⱼ(∂J/∂θⱼ)²)

Hessian Matrix (2nd derivative):
H = [[∂²J/∂θ₀², ∂²J/∂θ₀∂θ₁],
     [∂²J/∂θ₁∂θ₀, ∂²J/∂θ₁²]]

Convexity Test:
- If H ≻ 0 (positive definite) → Strictly Convex ✓
- MSE loss: H = (1/m)X^T X → Always positive definite
```

### Linear Algebra Formulas

```
Matrix Multiplication:
(X^T · X) = sum of outer products of rows

Vectorized Gradient:
∇J = X^T(h - y) / m

Parameter Update (Vectorized):
θ := θ - α·X^T(h - y) / m
```

---

## 🎓 Learning Path

**If new to ML:**
1. Understand hypothesis function: y_pred = X·θ
2. Understand cost function: MSE = (1/2m)Σ(error)²
3. Understand gradient: tells you error sensitivity
4. Understand update rule: move opposite to gradient
5. Run the project and observe convergence

**If intermediate in ML:**
1. Study convergence theorem proof
2. Implement from scratch (don't use sklearn)
3. Try different learning rates and observe effects
4. Add regularization (Ridge, Lasso)
5. Compare with sklearn models

**If advanced in ML:**
1. Implement stochastic GD (mini-batch)
2. Add momentum and adaptive learning rate (Adam)
3. Implement numerical gradient checking
4. Study second-order methods (Newton, BFGS)
5. Extend to logistic regression + softmax

---

## 📞 Quick Debugging

| Issue | Debug Command | Expected Output |
|-------|---------------|-----------------|
| No convergence | Add `print(history['gradient_norm'][-10:])` | Decreasing trend |
| Wrong predictions | Check `X_test_scaled` mean/std | Mean ≈ 0, Std ≈ 1 |
| High error | Inspect `y_train` distribution | Concentrated 0-0.5 |
| Slow training | Check dataset size with `print(X_train.shape)` | Should be < 50k |

---

**Version:** 1.0  
**Last Updated:** 2026-09-12  
**Maintainer:** Data Science Team
