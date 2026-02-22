"""
Comprehensive Model Evaluation
Generate detailed metrics, visualizations, and comparison for all models
"""

import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def load_data():
    """Load and prepare the engineered dataset"""
    print("Loading engineered dataset...")
    df = pd.read_csv('data/engineered_fuel_consumption.csv')
    
    target_col = 'CO2 emissions (g/km)'
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # One-hot encode categorical variables
    if categorical_cols:
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    else:
        df_encoded = df.copy()
    
    # Sanitize column names
    column_mapping = {}
    used_names = set()
    for col in df_encoded.columns:
        new_col = col.replace('(', '').replace(')', '').replace('/', '_').replace(' ', '_')
        new_col = new_col.replace('[', '').replace(']', '').replace('<', '').replace('>', '')
        new_col = new_col.replace('{', '').replace('}', '').replace('"', '').replace("'", '')
        new_col = new_col.replace(':', '').replace(',', '').replace('.', '').replace('-', '_')
        new_col = new_col.replace('\\', '').replace('|', '').replace('&', 'and')
        new_col = ''.join(c if c.isalnum() or c == '_' else '_' for c in new_col)
        while '__' in new_col:
            new_col = new_col.replace('__', '_')
        new_col = new_col.strip('_')
        
        if new_col in used_names:
            counter = 1
            while f"{new_col}_{counter}" in used_names:
                counter += 1
            new_col = f"{new_col}_{counter}"
        
        used_names.add(new_col)
        column_mapping[col] = new_col
    
    df_encoded = df_encoded.rename(columns=column_mapping)
    target_col_clean = column_mapping.get(target_col, target_col)
    
    # Select features
    feature_cols = [col for col in df_encoded.columns 
                   if col != target_col_clean and 'rating' not in col.lower() 
                   and 'year' not in col.lower()]
    
    X = df_encoded[feature_cols]
    y = df_encoded[target_col_clean]
    
    return X, y

def load_all_models():
    """Load all trained models"""
    print("\nLoading all models...")
    models = {}
    
    # Load XGBoost
    with open('models/tuned/xgboost_tuned.pkl', 'rb') as f:
        models['XGBoost'] = pickle.load(f)
    
    # Load LightGBM
    with open('models/tuned/lightgbm_tuned.pkl', 'rb') as f:
        models['LightGBM'] = pickle.load(f)
    
    # Load RandomForest
    with open('models/randomforest_model.pkl', 'rb') as f:
        models['RandomForest'] = pickle.load(f)
    
    # Load CatBoost
    with open('models/catboost_model.pkl', 'rb') as f:
        models['CatBoost'] = pickle.load(f)
    
    # Load Stacking Ensemble
    with open('models/stacking_ensemble.pkl', 'rb') as f:
        models['Stacking Ensemble'] = pickle.load(f)
    
    print(f"✓ Loaded {len(models)} models")
    return models

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate_all_models(models, X_train, X_test, y_train, y_test):
    """Calculate comprehensive metrics for all models"""
    print("\nEvaluating all models...")
    
    results = []
    predictions = {}
    
    for name, model in models.items():
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        predictions[name] = {
            'train': y_train_pred,
            'test': y_test_pred
        }
        
        # Calculate metrics
        metrics = {
            'Model': name,
            'Train R²': r2_score(y_train, y_train_pred),
            'Test R²': r2_score(y_test, y_test_pred),
            'Train MAE': mean_absolute_error(y_train, y_train_pred),
            'Test MAE': mean_absolute_error(y_test, y_test_pred),
            'Train RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'Test RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'Train MAPE': calculate_mape(y_train, y_train_pred),
            'Test MAPE': calculate_mape(y_test, y_test_pred)
        }
        
        results.append(metrics)
        print(f"✓ {name} evaluated")
    
    results_df = pd.DataFrame(results)
    return results_df, predictions

def plot_actual_vs_predicted(y_test, predictions, save_path='visualizations/actual_vs_predicted.png'):
    """Create actual vs predicted scatter plots for all models"""
    print("\nGenerating actual vs predicted plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (name, preds) in enumerate(predictions.items()):
        ax = axes[idx]
        y_pred = preds['test']
        
        # Scatter plot
        ax.scatter(y_test, y_pred, alpha=0.5, s=20)
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        # Calculate R²
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        ax.set_xlabel('Actual CO₂ (g/km)', fontsize=10)
        ax.set_ylabel('Predicted CO₂ (g/km)', fontsize=10)
        ax.set_title(f'{name}\nR² = {r2:.6f}, MAE = {mae:.2f}', fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide the last subplot if we have fewer than 6 models
    if len(predictions) < 6:
        axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {save_path}")
    plt.close()

def plot_residuals(y_test, predictions, save_path='visualizations/residual_plots.png'):
    """Create residual plots for all models"""
    print("\nGenerating residual plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (name, preds) in enumerate(predictions.items()):
        ax = axes[idx]
        y_pred = preds['test']
        residuals = y_test - y_pred
        
        # Residual plot
        ax.scatter(y_pred, residuals, alpha=0.5, s=20)
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        
        # Calculate statistics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        ax.set_xlabel('Predicted CO₂ (g/km)', fontsize=10)
        ax.set_ylabel('Residuals (g/km)', fontsize=10)
        ax.set_title(f'{name}\nRMSE = {rmse:.2f}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Hide the last subplot if we have fewer than 6 models
    if len(predictions) < 6:
        axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {save_path}")
    plt.close()

def plot_feature_importance(models, feature_names, save_path='visualizations/feature_importance.png'):
    """Plot feature importance for tree-based models"""
    print("\nGenerating feature importance plots...")
    
    # Get feature importance from models that support it
    importance_data = {}
    
    for name, model in models.items():
        if name == 'Stacking Ensemble':
            # For stacking, use the first base estimator (XGBoost)
            if hasattr(model.estimators_[0], 'feature_importances_'):
                importance_data[name] = model.estimators_[0].feature_importances_
        elif hasattr(model, 'feature_importances_'):
            importance_data[name] = model.feature_importances_
    
    if not importance_data:
        print("⚠ No models with feature importance found")
        return
    
    # Create subplots
    n_models = len(importance_data)
    fig, axes = plt.subplots(1, min(n_models, 3), figsize=(18, 6))
    if n_models == 1:
        axes = [axes]
    
    for idx, (name, importances) in enumerate(list(importance_data.items())[:3]):
        # Get top 10 features
        indices = np.argsort(importances)[-10:]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]
        
        ax = axes[idx] if n_models > 1 else axes[0]
        ax.barh(range(len(top_importances)), top_importances)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features, fontsize=8)
        ax.set_xlabel('Importance', fontsize=10)
        ax.set_title(f'{name}\nTop 10 Features', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {save_path}")
    plt.close()

def plot_learning_curves(models, X_train, y_train, save_path='visualizations/learning_curves.png'):
    """Plot learning curves for selected models"""
    print("\nGenerating learning curves (this may take a few minutes)...")
    
    # Select a subset of models for learning curves (to save time)
    selected_models = {k: v for k, v in list(models.items())[:3]}
    
    fig, axes = plt.subplots(1, len(selected_models), figsize=(18, 5))
    if len(selected_models) == 1:
        axes = [axes]
    
    for idx, (name, model) in enumerate(selected_models.items()):
        print(f"  Computing learning curve for {name}...")
        
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train, cv=3, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 5),
            scoring='r2'
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        ax = axes[idx]
        ax.plot(train_sizes, train_mean, label='Training score', marker='o')
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
        ax.plot(train_sizes, val_mean, label='Validation score', marker='s')
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
        
        ax.set_xlabel('Training Set Size', fontsize=10)
        ax.set_ylabel('R² Score', fontsize=10)
        ax.set_title(f'{name}\nLearning Curve', fontsize=11, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {save_path}")
    plt.close()

def plot_model_comparison(results_df, save_path='visualizations/model_comparison.png'):
    """Create bar charts comparing all models"""
    print("\nGenerating model comparison charts...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    metrics = [
        ('Test R²', 'R² Score (Higher is Better)'),
        ('Test MAE', 'MAE (g/km) (Lower is Better)'),
        ('Test RMSE', 'RMSE (g/km) (Lower is Better)'),
        ('Test MAPE', 'MAPE (%) (Lower is Better)')
    ]
    
    for idx, (metric, ylabel) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        bars = ax.bar(results_df['Model'], results_df[metric], color=colors[:len(results_df)])
        
        # Highlight the best model
        if 'R²' in metric:
            best_idx = results_df[metric].idxmax()
        else:
            best_idx = results_df[metric].idxmin()
        bars[best_idx].set_color('#FFD700')
        bars[best_idx].set_edgecolor('black')
        bars[best_idx].set_linewidth(2)
        
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}' if 'R²' in metric else f'{height:.2f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {save_path}")
    plt.close()

def save_results_table(results_df, save_path='visualizations/model_comparison_table.csv'):
    """Save results table as CSV"""
    print(f"\nSaving results table to {save_path}...")
    results_df.to_csv(save_path, index=False)
    print("✓ Saved")

def print_summary(results_df):
    """Print summary of results"""
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL EVALUATION SUMMARY")
    print("="*80)
    
    print("\nModel Comparison Table:")
    print(results_df.to_string(index=False))
    
    print("\n" + "="*80)
    print("BEST MODELS BY METRIC")
    print("="*80)
    
    best_r2 = results_df.loc[results_df['Test R²'].idxmax()]
    print(f"\n🏆 Best R² Score: {best_r2['Model']}")
    print(f"   R² = {best_r2['Test R²']:.6f}")
    
    best_mae = results_df.loc[results_df['Test MAE'].idxmin()]
    print(f"\n🏆 Best MAE: {best_mae['Model']}")
    print(f"   MAE = {best_mae['Test MAE']:.4f} g/km")
    
    best_rmse = results_df.loc[results_df['Test RMSE'].idxmin()]
    print(f"\n🏆 Best RMSE: {best_rmse['Model']}")
    print(f"   RMSE = {best_rmse['Test RMSE']:.4f} g/km")
    
    best_mape = results_df.loc[results_df['Test MAPE'].idxmin()]
    print(f"\n🏆 Best MAPE: {best_mape['Model']}")
    print(f"   MAPE = {best_mape['Test MAPE']:.4f}%")
    
    print("\n" + "="*80)
    print("TARGET ACHIEVEMENT")
    print("="*80)
    
    # Check if targets are met
    target_r2 = 0.92
    target_mae = 10
    target_rmse = 15
    
    for _, row in results_df.iterrows():
        print(f"\n{row['Model']}:")
        print(f"  R² > {target_r2}: {'✓ PASS' if row['Test R²'] > target_r2 else '✗ FAIL'} ({row['Test R²']:.6f})")
        print(f"  MAE < {target_mae}: {'✓ PASS' if row['Test MAE'] < target_mae else '✗ FAIL'} ({row['Test MAE']:.4f} g/km)")
        print(f"  RMSE < {target_rmse}: {'✓ PASS' if row['Test RMSE'] < target_rmse else '✗ FAIL'} ({row['Test RMSE']:.4f} g/km)")

def main():
    """Main execution function"""
    print("="*80)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*80)
    
    # Create visualizations directory
    import os
    os.makedirs('visualizations', exist_ok=True)
    
    # Load data
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Load models
    models = load_all_models()
    
    # Evaluate all models
    results_df, predictions = evaluate_all_models(models, X_train, X_test, y_train, y_test)
    
    # Generate visualizations
    plot_actual_vs_predicted(y_test, predictions)
    plot_residuals(y_test, predictions)
    plot_feature_importance(models, X.columns.tolist())
    plot_learning_curves(models, X_train, y_train)
    plot_model_comparison(results_df)
    
    # Save results
    save_results_table(results_df)
    
    # Print summary
    print_summary(results_df)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print("\nAll visualizations saved to 'visualizations/' directory:")
    print("  - actual_vs_predicted.png")
    print("  - residual_plots.png")
    print("  - feature_importance.png")
    print("  - learning_curves.png")
    print("  - model_comparison.png")
    print("  - model_comparison_table.csv")

if __name__ == "__main__":
    main()
