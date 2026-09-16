"""
===================================================================
USAGE EXAMPLES & ADVANCED CUSTOMIZATIONS
Banking Expected Loss Prediction Project
===================================================================

This file demonstrates practical usage patterns and advanced
customizations for the banking expected loss model.
"""

import numpy as np
import pandas as pd
from banking_expected_loss_project import (
    BankingDataGenerator,
    LinearRegressionGD,
    BankingExpectedLossCalculator
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


# =====================================================================
# EXAMPLE 1: Basic Usage with Custom Parameters
# =====================================================================
def example_1_basic_usage():
    """
    Most common usage pattern with custom hyperparameters
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Usage with Custom Parameters")
    print("="*70)
    
    # Generate data
    generator = BankingDataGenerator()
    df = generator.generate_dataset(n_samples=500, random_state=123)
    
    # Prepare data
    X = df[['credit_score', 'loan_amount', 'annual_income', 
             'employment_years', 'debt_to_income', 'previous_defaults']].values
    y = df['expected_loss_rate'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train with custom parameters
    model = LinearRegressionGD(
        learning_rate=0.05,      # Slower learning
        max_iterations=5000,      # Fewer iterations
        tolerance=1e-5            # Less strict convergence
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"\nResults:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  Converged: {model.converged}")
    print(f"  Iterations: {model.iterations_run}")


# =====================================================================
# EXAMPLE 2: Comparing Different Learning Rates
# =====================================================================
def example_2_learning_rate_comparison():
    """
    Demonstrates impact of different learning rates on convergence
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Comparing Different Learning Rates")
    print("="*70)
    
    generator = BankingDataGenerator()
    df = generator.generate_dataset(n_samples=800)
    
    X = df[['credit_score', 'loan_amount', 'annual_income', 
             'employment_years', 'debt_to_income', 'previous_defaults']].values
    y = df['expected_loss_rate'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    learning_rates = [0.001, 0.01, 0.1, 0.5]
    results = {}
    
    print("\nTesting different learning rates:")
    print("-" * 70)
    print(f"{'LR':>6} | {'Iterations':>10} | {'Final Cost':>12} | {'Test R²':>8}")
    print("-" * 70)
    
    for lr in learning_rates:
        model = LinearRegressionGD(
            learning_rate=lr,
            max_iterations=10000,
            tolerance=1e-6
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        
        results[lr] = {
            'iterations': model.iterations_run,
            'final_cost': model.cost_history[-1],
            'r2': r2
        }
        
        print(f"{lr:6.3f} | {model.iterations_run:10d} | {model.cost_history[-1]:12.6f} | {r2:8.4f}")
    
    print("-" * 70)
    print(f"\n✓ Optimal LR: {min(results, key=lambda x: results[x]['iterations'])}")
    print(f"  Fastest convergence: {min(results[lr]['iterations'] for lr in learning_rates)} iterations")


# =====================================================================
# EXAMPLE 3: Expected Loss Analysis at Loan Level
# =====================================================================
def example_3_loan_level_analysis():
    """
    Analyze expected loss at individual loan level
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Loan-Level Expected Loss Analysis")
    print("="*70)
    
    generator = BankingDataGenerator()
    df = generator.generate_dataset(n_samples=100)
    
    X = df[['credit_score', 'loan_amount', 'annual_income', 
             'employment_years', 'debt_to_income', 'previous_defaults']].values
    y = df['expected_loss_rate'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = LinearRegressionGD()
    model.fit(X_train, y_train)
    
    # Predict for test set
    pd_predictions = np.clip(model.predict(X_test), 0, 1)
    loan_amounts = X_test[:, 1] * scaler.scale_[1] + scaler.mean_[1]  # Unscale
    
    # Calculate EL
    el_calc = BankingExpectedLossCalculator(lgd=0.45)
    expected_losses = el_calc.calculate_el(pd_predictions, loan_amounts)
    
    # Create summary
    summary = pd.DataFrame({
        'credit_score': X_test[:, 0],
        'loan_amount': loan_amounts,
        'debt_to_income': X_test[:, 4],
        'pd': pd_predictions * 100,
        'expected_loss': expected_losses
    })
    
    summary = summary.sort_values('expected_loss', ascending=False)
    
    print(f"\nTop 10 Highest Risk Loans:")
    print("-" * 70)
    print(summary.head(10).to_string())
    print("-" * 70)
    print(f"\nPortfolio Summary:")
    print(f"  Total EL: ${expected_losses.sum():,.2f}")
    print(f"  Average EL: ${expected_losses.mean():,.2f}")
    print(f"  Median EL: ${np.median(expected_losses):,.2f}")
    print(f"  Max EL: ${expected_losses.max():,.2f}")
    print(f"  95th percentile EL: ${np.percentile(expected_losses, 95):,.2f}")


# =====================================================================
# EXAMPLE 4: Feature Contribution Analysis
# =====================================================================
def example_4_feature_contribution():
    """
    Analyze how each feature contributes to predictions
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Feature Contribution Analysis")
    print("="*70)
    
    generator = BankingDataGenerator()
    df = generator.generate_dataset(n_samples=800)
    
    X = df[['credit_score', 'loan_amount', 'annual_income', 
             'employment_years', 'debt_to_income', 'previous_defaults']].values
    y = df['expected_loss_rate'].values
    
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    
    model = LinearRegressionGD()
    model.fit(X_train, y_train)
    
    theta = model.get_parameters()
    
    features = ['Credit Score', 'Loan Amount', 'Annual Income',
                'Employment Years', 'Debt-to-Income', 'Previous Defaults']
    
    # Calculate contribution (absolute value of coefficient)
    contributions = np.abs(theta[1:])  # Exclude bias
    contributions_pct = (contributions / contributions.sum()) * 100
    
    print(f"\nFeature Contribution to Model:")
    print("-" * 70)
    print(f"{'Feature':<20} | {'Coefficient':>12} | {'Contribution':>12}")
    print("-" * 70)
    
    for feat, coef, pct in sorted(zip(features, theta[1:], contributions_pct),
                                   key=lambda x: abs(x[1]), reverse=True):
        print(f"{feat:<20} | {coef:>12.6f} | {pct:>11.2f}%")
    
    print("-" * 70)
    print(f"\nInterpretation:")
    print(f"  Bias term (θ₀): {theta[0]:.6f}")
    print(f"  Most important: {features[np.argmax(contributions)]}")
    print(f"  Least important: {features[np.argmin(contributions)]}")


# =====================================================================
# EXAMPLE 5: Model Performance Across Different Data Sizes
# =====================================================================
def example_5_scalability_analysis():
    """
    Test how model performance scales with dataset size
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Scalability Analysis")
    print("="*70)
    
    dataset_sizes = [100, 500, 1000, 2000]
    results = []
    
    print(f"\nTesting model on different dataset sizes:")
    print("-" * 70)
    print(f"{'Size':<8} | {'Train R²':>10} | {'Test R²':>10} | {'Iterations':>10}")
    print("-" * 70)
    
    for size in dataset_sizes:
        generator = BankingDataGenerator()
        df = generator.generate_dataset(n_samples=size)
        
        X = df[['credit_score', 'loan_amount', 'annual_income', 
                 'employment_years', 'debt_to_income', 'previous_defaults']].values
        y = df['expected_loss_rate'].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        model = LinearRegressionGD()
        model.fit(X_train, y_train)
        
        train_r2 = r2_score(y_train, model.predict(X_train))
        test_r2 = r2_score(y_test, model.predict(X_test))
        
        results.append({
            'size': size,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'iterations': model.iterations_run
        })
        
        print(f"{size:<8} | {train_r2:>10.4f} | {test_r2:>10.4f} | {model.iterations_run:>10}")
    
    print("-" * 70)
    print(f"\nObservations:")
    print(f"  Train R² trend: {[r['train_r2'] for r in results]}")
    print(f"  Test R² trend:  {[r['test_r2'] for r in results]}")
    print(f"  Convergence stable: {all(300 >= r['iterations'] >= 100 for r in results)}")


# =====================================================================
# EXAMPLE 6: Parameter Sensitivity Analysis
# =====================================================================
def example_6_parameter_sensitivity():
    """
    Analyze sensitivity to hyperparameters
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Parameter Sensitivity Analysis")
    print("="*70)
    
    generator = BankingDataGenerator()
    df = generator.generate_dataset(n_samples=800)
    
    X = df[['credit_score', 'loan_amount', 'annual_income', 
             'employment_years', 'debt_to_income', 'previous_defaults']].values
    y = df['expected_loss_rate'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print("\nTesting different convergence tolerances:")
    print("-" * 70)
    print(f"{'Tolerance':<15} | {'Iterations':>10} | {'Final Cost':>12} | {'Test R²':>8}")
    print("-" * 70)
    
    tolerances = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    
    for tol in tolerances:
        model = LinearRegressionGD(
            learning_rate=0.1,
            max_iterations=10000,
            tolerance=tol
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        
        print(f"{tol:<15.0e} | {model.iterations_run:>10d} | {model.cost_history[-1]:>12.6f} | {r2:>8.4f}")
    
    print("-" * 70)
    print(f"\nObservations:")
    print(f"  Stricter tolerance → More iterations")
    print(f"  Diminishing returns: R² improvement plateaus")
    print(f"  Optimal tolerance: ~1e-6 (good balance)")


# =====================================================================
# EXAMPLE 7: Cross-Validation Performance
# =====================================================================
def example_7_cross_validation():
    """
    Implement 5-fold cross-validation
    """
    print("\n" + "="*70)
    print("EXAMPLE 7: 5-Fold Cross-Validation")
    print("="*70)
    
    from sklearn.model_selection import KFold
    
    generator = BankingDataGenerator()
    df = generator.generate_dataset(n_samples=1000)
    
    X = df[['credit_score', 'loan_amount', 'annual_income', 
             'employment_years', 'debt_to_income', 'previous_defaults']].values
    y = df['expected_loss_rate'].values
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []
    
    print(f"\nCross-Validation Results:")
    print("-" * 70)
    print(f"{'Fold':<6} | {'Train R²':>10} | {'Test R²':>10} | {'Test RMSE':>10}")
    print("-" * 70)
    
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        model = LinearRegressionGD()
        model.fit(X_train, y_train)
        
        train_r2 = r2_score(y_train, model.predict(X_train))
        test_r2 = r2_score(y_test, model.predict(X_test))
        test_rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
        
        fold_scores.append(test_r2)
        
        print(f"{fold:<6} | {train_r2:>10.4f} | {test_r2:>10.4f} | {test_rmse:>10.4f}")
    
    print("-" * 70)
    print(f"\nCross-Validation Summary:")
    print(f"  Mean R²: {np.mean(fold_scores):.4f}")
    print(f"  Std R²:  {np.std(fold_scores):.4f}")
    print(f"  Min R²:  {np.min(fold_scores):.4f}")
    print(f"  Max R²:  {np.max(fold_scores):.4f}")


# =====================================================================
# EXAMPLE 8: Advanced - Feature Engineering
# =====================================================================
def example_8_feature_engineering():
    """
    Demonstrate feature engineering with interaction terms
    """
    print("\n" + "="*70)
    print("EXAMPLE 8: Feature Engineering (Interaction Terms)")
    print("="*70)
    
    generator = BankingDataGenerator()
    df = generator.generate_dataset(n_samples=800)
    
    X = df[['credit_score', 'loan_amount', 'annual_income', 
             'employment_years', 'debt_to_income', 'previous_defaults']].values
    y = df['expected_loss_rate'].values
    
    # Add interaction terms
    # Interaction: credit_score × debt_to_income (strong risk indicator)
    interaction_feature = (X[:, 0] - X[:, 0].mean()) * (X[:, 4] - X[:, 4].mean())
    X_extended = np.column_stack([X, interaction_feature])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_extended, y, test_size=0.2
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train with extended features
    model = LinearRegressionGD()
    model.fit(X_train, y_train)
    
    r2_extended = r2_score(y_test, model.predict(X_test))
    
    # Train without extended features (for comparison)
    X_train_base, X_test_base, _, _ = train_test_split(X, y, test_size=0.2)
    
    scaler_base = StandardScaler()
    X_train_base = scaler_base.fit_transform(X_train_base)
    X_test_base = scaler_base.transform(X_test_base)
    
    model_base = LinearRegressionGD()
    model_base.fit(X_train_base, y_train)
    
    r2_base = r2_score(y_test, model_base.predict(X_test_base))
    
    print(f"\nComparison:")
    print("-" * 70)
    print(f"  Base Model (6 features):           R² = {r2_base:.4f}")
    print(f"  Extended Model (6 + interaction):  R² = {r2_extended:.4f}")
    print(f"  Improvement: {(r2_extended - r2_base)*100:+.2f}%")
    print("-" * 70)
    print(f"\nTheta values (Extended Model):")
    theta = model.get_parameters()
    print(f"  Interaction term coefficient: {theta[-1]:.6f}")
    print(f"  Interaction is {'significant' if abs(theta[-1]) > 0.01 else 'weak'}")


# =====================================================================
# MAIN: Run All Examples
# =====================================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("BANKING EXPECTED LOSS - USAGE EXAMPLES")
    print("="*70)
    
    # Uncomment to run specific examples
    
    # example_1_basic_usage()
    # example_2_learning_rate_comparison()
    # example_3_loan_level_analysis()
    # example_4_feature_contribution()
    # example_5_scalability_analysis()
    # example_6_parameter_sensitivity()
    # example_7_cross_validation()
    # example_8_feature_engineering()
    
    # Or run all:
    try:
        example_1_basic_usage()
        example_2_learning_rate_comparison()
        example_3_loan_level_analysis()
        example_4_feature_contribution()
        example_5_scalability_analysis()
        example_6_parameter_sensitivity()
        example_7_cross_validation()
        example_8_feature_engineering()
    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        print("Note: Some examples require the main module in the same directory")
    
    print("\n" + "="*70)
    print("EXAMPLES COMPLETED")
    print("="*70 + "\n")
