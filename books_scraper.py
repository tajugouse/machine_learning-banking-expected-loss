"""
===================================================================
BANKING EXPECTED LOSS PREDICTION - END-TO-END ML PROJECT
===================================================================
Project: Linear Regression for Credit Risk & Expected Loss Calculation
Author: Data Science Team
Date: 2026

MATHEMATICAL FOUNDATIONS:
1. Hypothesis Function: h(x) = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ
2. Cost Function (MSE): J(θ) = (1/2m) * Σ(h(xⁱ) - yⁱ)²
3. Gradient Descent: θⱼ := θⱼ - α * ∂J(θ)/∂θⱼ
4. Convergence: When ||∇J(θ)|| < ε (threshold)
5. Expected Loss: EL = PD * LGD * EAD
===================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# =====================================================================
# SECTION 1: DATA GENERATION (Real-World Banking Dataset Simulation)
# =====================================================================
class BankingDataGenerator:
    """Generate realistic banking credit risk dataset"""
    
    @staticmethod
    def generate_dataset(n_samples=1000, random_state=42):
        """
        Generate synthetic banking dataset with credit risk features
        
        Features:
        - Credit Score (300-850)
        - Loan Amount ($1,000-$500,000)
        - Annual Income ($20,000-$200,000)
        - Employment Years (0-40 years)
        - Debt-to-Income Ratio (0-100%)
        - Default History (0-5)
        
        Target: Default Rate / Loss Given Default (0-1)
        """
        np.random.seed(random_state)
        
        data = {
            'credit_score': np.random.normal(680, 80, n_samples).clip(300, 850),
            'loan_amount': np.random.exponential(50000, n_samples).clip(1000, 500000),
            'annual_income': np.random.normal(70000, 30000, n_samples).clip(20000, 200000),
            'employment_years': np.random.exponential(8, n_samples).clip(0, 40),
            'debt_to_income': np.random.beta(2, 5, n_samples) * 100,
            'previous_defaults': np.random.poisson(0.5, n_samples).clip(0, 5),
        }
        
        df = pd.DataFrame(data)
        
        # Target: Expected Loss Rate (function of risk factors)
        # Lower credit score + higher debt-to-income = higher loss
        loss_rate = (
            (850 - df['credit_score']) / 550 * 0.15 +  # Credit score impact
            df['debt_to_income'] / 100 * 0.20 +          # DTI impact
            df['previous_defaults'] * 0.08 +              # Default history impact
            np.random.normal(0, 0.05, n_samples)          # Noise
        ).clip(0, 1)
        
        df['expected_loss_rate'] = loss_rate
        
        return df


# =====================================================================
# SECTION 2: LINEAR REGRESSION FROM SCRATCH
# =====================================================================
class LinearRegressionGD:
    """
    Linear Regression using Gradient Descent with convergence analysis
    
    Mathematical Formulation:
    - Hypothesis: h(x) = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ
    - Cost Function: J(θ) = (1/2m) * Σ(h(xⁱ) - yⁱ)²
    - Gradient: ∂J(θ)/∂θⱼ = (1/m) * Σ(h(xⁱ) - yⁱ) * xⱼⁱ
    - Update Rule: θⱼ := θⱼ - α * ∂J(θ)/∂θⱼ
    """
    
    def __init__(self, learning_rate=0.01, max_iterations=10000, tolerance=1e-6):
        """
        Initialize Linear Regression with Gradient Descent
        
        Parameters:
        -----------
        learning_rate : float
            Learning rate (α) for gradient descent updates
        max_iterations : int
            Maximum number of iterations before stopping
        tolerance : float
            Convergence threshold for gradient norm
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.theta = None
        self.cost_history = []
        self.gradient_history = []
        self.iterations_run = 0
        self.converged = False
        self.convergence_iteration = None
        
    def _initialize_parameters(self, n_features):
        """Initialize parameters (theta) to small random values"""
        self.theta = np.random.randn(n_features + 1) * 0.01
        
    def _compute_hypothesis(self, X, theta):
        """
        Compute hypothesis function: h(x) = X·θ
        
        Parameters:
        -----------
        X : ndarray, shape (m, n)
            Feature matrix
        theta : ndarray, shape (n+1,)
            Parameter vector
            
        Returns:
        --------
        h : ndarray, shape (m,)
            Predictions
        """
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        return X_with_bias @ theta
    
    def _compute_cost(self, h, y):
        """
        Compute cost function (Mean Squared Error)
        J(θ) = (1/2m) * Σ(h(xⁱ) - yⁱ)²
        
        Parameters:
        -----------
        h : ndarray
            Predictions
        y : ndarray
            True values
            
        Returns:
        --------
        cost : float
            MSE cost
        """
        m = len(y)
        errors = h - y
        return (1 / (2 * m)) * np.sum(errors ** 2)
    
    def _compute_gradients(self, X, h, y):
        """
        Compute gradients for all parameters
        ∂J(θ)/∂θⱼ = (1/m) * Σ(h(xⁱ) - yⁱ) * xⱼⁱ
        
        Parameters:
        -----------
        X : ndarray, shape (m, n)
            Feature matrix
        h : ndarray, shape (m,)
            Predictions
        y : ndarray, shape (m,)
            True values
            
        Returns:
        --------
        gradients : ndarray, shape (n+1,)
            Gradient vector for all parameters
        """
        m = len(y)
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        errors = h - y
        
        # Vectorized gradient computation
        gradients = (1 / m) * (X_with_bias.T @ errors)
        return gradients
    
    def _compute_gradient_norm(self, gradients):
        """
        Compute norm of gradient vector (for convergence check)
        ||∇J(θ)|| = √(Σ(∂J/∂θⱼ)²)
        """
        return np.linalg.norm(gradients)
    
    def fit(self, X, y):
        """
        Train the model using Gradient Descent
        
        Parameters:
        -----------
        X : ndarray, shape (m, n)
            Training feature matrix
        y : ndarray, shape (m,)
            Training target values
        """
        m, n = X.shape
        self._initialize_parameters(n)
        
        print("\n" + "="*70)
        print("GRADIENT DESCENT OPTIMIZATION - TRAINING LOG")
        print("="*70)
        print(f"Training samples: {m}")
        print(f"Features: {n}")
        print(f"Learning rate (α): {self.learning_rate}")
        print(f"Convergence tolerance (ε): {self.tolerance}")
        print(f"Max iterations: {self.max_iterations}")
        print("-"*70)
        
        for iteration in range(self.max_iterations):
            # Hypothesis: h = X·θ
            h = self._compute_hypothesis(X, self.theta)
            
            # Cost function: J(θ)
            cost = self._compute_cost(h, y)
            self.cost_history.append(cost)
            
            # Gradients: ∇J(θ)
            gradients = self._compute_gradients(X, h, y)
            gradient_norm = self._compute_gradient_norm(gradients)
            self.gradient_history.append(gradient_norm)
            
            # Parameter update: θⱼ := θⱼ - α·∂J/∂θⱼ
            self.theta = self.theta - self.learning_rate * gradients
            
            # Convergence check: ||∇J(θ)|| < ε
            if gradient_norm < self.tolerance:
                self.converged = True
                self.convergence_iteration = iteration
                self.iterations_run = iteration + 1
                print(f"\n✓ CONVERGED at iteration {iteration + 1}")
                print(f"  Gradient norm: {gradient_norm:.2e} < {self.tolerance:.2e}")
                print(f"  Final cost: {cost:.6f}")
                break
            
            # Log progress
            if (iteration + 1) % max(1, self.max_iterations // 10) == 0:
                print(f"Iteration {iteration + 1:5d} | Cost: {cost:.6f} | Gradient Norm: {gradient_norm:.2e}")
        
        if not self.converged:
            self.iterations_run = self.max_iterations
            print(f"\n⚠ Did NOT converge within {self.max_iterations} iterations")
            print(f"  Final gradient norm: {self.gradient_history[-1]:.2e}")
            print(f"  Final cost: {self.cost_history[-1]:.6f}")
        
        print("="*70 + "\n")
        
    def predict(self, X):
        """Make predictions on new data"""
        return self._compute_hypothesis(X, self.theta)
    
    def get_parameters(self):
        """Return learned parameters"""
        return self.theta
    
    def get_training_history(self):
        """Return training history for visualization"""
        return {
            'cost': self.cost_history,
            'gradient_norm': self.gradient_history,
            'iterations': self.iterations_run,
            'converged': self.converged,
            'convergence_iteration': self.convergence_iteration
        }


# =====================================================================
# SECTION 3: EXPECTED LOSS FRAMEWORK
# =====================================================================
class BankingExpectedLossCalculator:
    """
    Calculate Expected Loss (EL) based on Basel III Framework
    EL = PD * LGD * EAD
    
    Where:
    - PD: Probability of Default (predicted loss rate)
    - LGD: Loss Given Default (typically 0.45 for unsecured loans)
    - EAD: Exposure At Default (loan amount)
    """
    
    def __init__(self, lgd=0.45):
        """
        Initialize Expected Loss Calculator
        
        Parameters:
        -----------
        lgd : float
            Loss Given Default (0-1), default 0.45 for unsecured loans
        """
        self.lgd = lgd
    
    def calculate_el(self, pd, ead):
        """
        Calculate Expected Loss
        
        Parameters:
        -----------
        pd : float or ndarray
            Probability of Default (predicted loss rate)
        ead : float or ndarray
            Exposure At Default (loan amount)
            
        Returns:
        --------
        el : float or ndarray
            Expected Loss amount
        """
        return pd * self.lgd * ead
    
    def calculate_el_rate(self, pd):
        """
        Calculate Expected Loss Rate (as percentage)
        
        Parameters:
        -----------
        pd : float or ndarray
            Probability of Default
            
        Returns:
        --------
        el_rate : float or ndarray
            EL rate as percentage
        """
        return pd * self.lgd * 100


# =====================================================================
# SECTION 4: MAIN EXECUTION & ANALYSIS
# =====================================================================
def main():
    print("\n" + "="*70)
    print("BANKING EXPECTED LOSS PREDICTION - END-TO-END PROJECT")
    print("="*70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # ==================== DATA GENERATION ====================
    print("\n[STEP 1] GENERATING BANKING DATASET")
    print("-" * 70)
    generator = BankingDataGenerator()
    df = generator.generate_dataset(n_samples=1000, random_state=42)
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nDataset Statistics:")
    print(df.describe())
    
    # ==================== DATA PREPROCESSING ====================
    print("\n[STEP 2] DATA PREPROCESSING & FEATURE ENGINEERING")
    print("-" * 70)
    
    # Separate features and target
    X = df[['credit_score', 'loan_amount', 'annual_income', 
             'employment_years', 'debt_to_income', 'previous_defaults']].values
    y = df['expected_loss_rate'].values
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Feature scaling (standardization)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set: {X_train_scaled.shape}")
    print(f"Test set: {X_test_scaled.shape}")
    print(f"Train-Test split: 80-20")
    print(f"Features standardized (mean=0, std=1)")
    
    # ==================== MODEL TRAINING ====================
    print("\n[STEP 3] LINEAR REGRESSION MODEL TRAINING")
    print("-" * 70)
    
    # Initialize model with optimized learning rate
    model = LinearRegressionGD(
        learning_rate=0.1,
        max_iterations=10000,
        tolerance=1e-6
    )
    
    # Train on standardized data
    model.fit(X_train_scaled, y_train)
    
    # Get training history
    history = model.get_training_history()
    theta = model.get_parameters()
    
    # ==================== MODEL EVALUATION ====================
    print("\n[STEP 4] MODEL EVALUATION")
    print("-" * 70)
    
    # Make predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"\nTraining Metrics:")
    print(f"  MSE (Mean Squared Error):  {train_mse:.6f}")
    print(f"  RMSE (Root MSE):           {train_rmse:.6f}")
    print(f"  MAE (Mean Absolute Error): {train_mae:.6f}")
    print(f"  R² Score:                  {train_r2:.6f}")
    
    print(f"\nTest Metrics:")
    print(f"  MSE (Mean Squared Error):  {test_mse:.6f}")
    print(f"  RMSE (Root MSE):           {test_rmse:.6f}")
    print(f"  MAE (Mean Absolute Error): {test_mae:.6f}")
    print(f"  R² Score:                  {test_r2:.6f}")
    
    print(f"\nModel Parameters (θ):")
    feature_names = ['θ₀ (bias)', 'Credit Score', 'Loan Amount', 'Annual Income',
                     'Employment Years', 'Debt-to-Income', 'Previous Defaults']
    for name, param in zip(feature_names, theta):
        print(f"  {name:25s}: {param:10.6f}")
    
    # ==================== EXPECTED LOSS CALCULATION ====================
    print("\n[STEP 5] EXPECTED LOSS CALCULATION")
    print("-" * 70)
    
    # Initialize EL calculator (LGD = 0.45 for unsecured loans)
    el_calculator = BankingExpectedLossCalculator(lgd=0.45)
    
    # Calculate EL for test set
    pd_test = np.clip(y_test_pred, 0, 1)  # Clip to [0, 1] range
    loan_amounts = X_test[:, 1]  # Loan amount is second feature
    
    expected_losses = el_calculator.calculate_el(pd_test, loan_amounts)
    el_rates = el_calculator.calculate_el_rate(pd_test)
    
    print(f"\nExpected Loss Analysis (Test Set):")
    print(f"  Average PD (Probability of Default):    {pd_test.mean():.4f} ({pd_test.mean()*100:.2f}%)")
    print(f"  Average EAD (Exposure at Default):      ${loan_amounts.mean():,.2f}")
    print(f"  LGD (Loss Given Default):               {el_calculator.lgd:.2f}")
    print(f"  Average Expected Loss:                  ${expected_losses.mean():,.2f}")
    print(f"  Average EL Rate:                        {el_rates.mean():.4f}%")
    print(f"  Total Portfolio Expected Loss:          ${expected_losses.sum():,.2f}")
    print(f"  Min EL (99th percentile):               ${np.percentile(expected_losses, 1):,.2f}")
    print(f"  Max EL (99th percentile):               ${np.percentile(expected_losses, 99):,.2f}")
    
    # ==================== CONVERGENCE ANALYSIS ====================
    print("\n[STEP 6] CONVERGENCE THEOREM & ANALYSIS")
    print("-" * 70)
    print(f"\nConvergence Theorem: Global Minima with Convex Loss Function")
    print(f"  The MSE loss function is strictly convex for linear regression,")
    print(f"  guaranteeing that gradient descent converges to global minimum.")
    print(f"\n  Convergence Criteria: ||∇J(θ)|| < ε")
    print(f"  - ε (tolerance): {model.tolerance:.2e}")
    print(f"  - Converged: {'✓ Yes' if model.converged else '✗ No'}")
    
    if model.converged:
        print(f"  - Convergence at iteration: {history['convergence_iteration'] + 1}")
        print(f"  - Final gradient norm: {history['gradient_norm'][-1]:.2e}")
    else:
        print(f"  - Final gradient norm: {history['gradient_norm'][-1]:.2e}")
    
    print(f"  - Iterations executed: {history['iterations']}")
    print(f"  - Cost at start: {history['cost'][0]:.6f}")
    print(f"  - Cost at end: {history['cost'][-1]:.6f}")
    print(f"  - Cost reduction: {(1 - history['cost'][-1]/history['cost'][0])*100:.2f}%")
    
    # ==================== VISUALIZATIONS ====================
    print("\n[STEP 7] GENERATING VISUALIZATIONS")
    print("-" * 70)
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Banking Expected Loss Prediction - Analysis Dashboard', 
                 fontsize=16, fontweight='bold', y=1.00)
    
    # 1. Cost Function Convergence
    ax1 = axes[0, 0]
    iterations = range(len(history['cost']))
    ax1.semilogy(iterations, history['cost'], linewidth=2, color='#e74c3c')
    ax1.set_xlabel('Iteration', fontsize=10)
    ax1.set_ylabel('Cost J(θ) - Log Scale', fontsize=10)
    ax1.set_title('Cost Function Convergence (MSE)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    if model.converged:
        ax1.axvline(history['convergence_iteration'], color='green', 
                   linestyle='--', alpha=0.7, label=f"Converged at {history['convergence_iteration']}")
        ax1.legend()
    
    # 2. Gradient Norm Convergence
    ax2 = axes[0, 1]
    ax2.semilogy(iterations, history['gradient_norm'], linewidth=2, color='#3498db')
    ax2.axhline(model.tolerance, color='red', linestyle='--', 
               linewidth=2, label=f'Tolerance ε={model.tolerance:.2e}')
    ax2.set_xlabel('Iteration', fontsize=10)
    ax2.set_ylabel('Gradient Norm ||∇J(θ)|| - Log Scale', fontsize=10)
    ax2.set_title('Gradient Norm Convergence', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Predicted vs Actual (Test Set)
    ax3 = axes[0, 2]
    ax3.scatter(y_test, y_test_pred, alpha=0.6, s=30, color='#2ecc71')
    ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
            'r--', lw=2, label='Perfect Prediction')
    ax3.set_xlabel('Actual Loss Rate', fontsize=10)
    ax3.set_ylabel('Predicted Loss Rate', fontsize=10)
    ax3.set_title(f'Predictions vs Actual (R²={test_r2:.4f})', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Residuals Analysis
    ax4 = axes[1, 0]
    residuals = y_test - y_test_pred
    ax4.hist(residuals, bins=30, color='#9b59b6', alpha=0.7, edgecolor='black')
    ax4.axvline(0, color='red', linestyle='--', linewidth=2)
    ax4.set_xlabel('Residuals (Actual - Predicted)', fontsize=10)
    ax4.set_ylabel('Frequency', fontsize=10)
    ax4.set_title(f'Residuals Distribution (Mean={residuals.mean():.4f})', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Feature Importance (Absolute Coefficients)
    ax5 = axes[1, 1]
    feature_importance = np.abs(theta[1:])  # Exclude bias
    features = ['Credit Score', 'Loan Amount', 'Annual Income',
               'Employment Yrs', 'Debt-to-Income', 'Prev Defaults']
    colors_bar = plt.cm.Spectral(np.linspace(0, 1, len(features)))
    ax5.barh(features, feature_importance, color=colors_bar)
    ax5.set_xlabel('|Coefficient| Value', fontsize=10)
    ax5.set_title('Feature Importance (Standardized)', fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='x')
    
    # 6. Expected Loss Distribution
    ax6 = axes[1, 2]
    ax6.hist(expected_losses, bins=30, color='#e67e22', alpha=0.7, edgecolor='black')
    ax6.axvline(expected_losses.mean(), color='red', linestyle='--', 
               linewidth=2, label=f"Mean: ${expected_losses.mean():,.0f}")
    ax6.set_xlabel('Expected Loss ($)', fontsize=10)
    ax6.set_ylabel('Frequency', fontsize=10)
    ax6.set_title('Portfolio Expected Loss Distribution', fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('/home/claude/expected_loss_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved: expected_loss_analysis.png")
    plt.close()
    
    # ==================== SUMMARY REPORT ====================
    print("\n[STEP 8] FINAL SUMMARY REPORT")
    print("=" * 70)
    
    summary_report = f"""
    MODEL PERFORMANCE SUMMARY
    ========================
    
    ✓ Training Convergence:
      - Algorithm: Gradient Descent with global convergence guarantee
      - Convergence: {('YES ✓' if model.converged else 'NO ✗')}
      - Iterations: {history['iterations']} / {model.max_iterations}
      - Learning Rate (α): {model.learning_rate}
    
    ✓ Prediction Accuracy:
      - Test R² Score: {test_r2:.6f} ({test_r2*100:.2f}%)
      - Test RMSE: {test_rmse:.6f}
      - Test MAE: {test_mae:.6f}
    
    ✓ Expected Loss Metrics:
      - Portfolio EL: ${expected_losses.sum():,.2f}
      - Average EL per loan: ${expected_losses.mean():,.2f}
      - Average PD: {pd_test.mean()*100:.2f}%
      - Risk-Adjusted Return: {(1 - el_rates.mean()/100)*100:.2f}%
    
    ✓ Mathematical Guarantees:
      - Loss Function: Strictly Convex (MSE)
      - Global Minima: Guaranteed to converge
      - Gradient Norm: {history['gradient_norm'][-1]:.2e}
      - Satisfies Convergence Criterion: {history['gradient_norm'][-1] < model.tolerance}
    """
    
    print(summary_report)
    print("=" * 70)
    print("PROJECT COMPLETED SUCCESSFULLY ✓")
    print("=" * 70 + "\n")
    
    return {
        'model': model,
        'scaler': scaler,
        'X_test': X_test_scaled,
        'y_test': y_test,
        'y_pred': y_test_pred,
        'expected_losses': expected_losses,
        'el_calculator': el_calculator,
        'metrics': {
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'converged': model.converged,
            'iterations': history['iterations']
        }
    }


if __name__ == "__main__":
    results = main()
