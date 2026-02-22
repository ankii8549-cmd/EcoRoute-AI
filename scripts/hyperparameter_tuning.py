"""
Hyperparameter Tuning Script

This script performs hyperparameter tuning on the best 2 models:
- XGBoost (best performer)
- LightGBM (second best)

Uses RandomizedSearchCV for efficient hyperparameter search with 5-fold cross-validation.
Target: Optimized models with best hyperparameters
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, make_scorer
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def load_data() -> tuple:
    """Load and prepare data."""
    print("Loading engineered dataset...")
    df = pd.read_csv("data/engineered_fuel_consumption.csv")
    
    target_col = 'CO2 emissions (g/km)'
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # One-hot encode
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
    
    feature_cols = [col for col in df_encoded.columns 
                   if col != target_col_clean and 'rating' not in col.lower() 
                   and 'year' not in col.lower()]
    
    X = df_encoded[feature_cols]
    y = df_encoded[target_col_clean]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Features: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test, feature_cols


def tune_xgboost(X_train, y_train) -> dict:
    """Tune XGBoost hyperparameters."""
    print("\n" + "="*80)
    print("TUNING XGBOOST HYPERPARAMETERS")
    print("="*80)
    
    # Define parameter grid (reduced for efficiency)
    param_distributions = {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9]
    }
    
    # Base model
    base_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
    
    # Randomized search
    print("\nPerforming RandomizedSearchCV (20 iterations, 3-fold CV)...")
    print("This may take several minutes...")
    
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=20,
        scoring='r2',
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    random_search.fit(X_train, y_train)
    
    print(f"\nBest parameters:")
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")
    
    print(f"\nBest CV R² score: {random_search.best_score_:.4f}")
    
    return {
        'model': random_search.best_estimator_,
        'best_params': random_search.best_params_,
        'best_score': random_search.best_score_,
        'cv_results': random_search.cv_results_
    }


def tune_lightgbm(X_train, y_train) -> dict:
    """Tune LightGBM hyperparameters."""
    print("\n" + "="*80)
    print("TUNING LIGHTGBM HYPERPARAMETERS")
    print("="*80)
    
    # Define parameter grid
    param_distributions = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8, -1],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
    }

    
    # Base model
    base_model = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
    
    # Randomized search
    print("\nPerforming RandomizedSearchCV (20 iterations, 3-fold CV)...")
    print("This may take several minutes...")
    
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=20,
        scoring='r2',
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    random_search.fit(X_train, y_train)
    
    print(f"\nBest parameters:")
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")
    
    print(f"\nBest CV R² score: {random_search.best_score_:.4f}")
    
    return {
        'model': random_search.best_estimator_,
        'best_params': random_search.best_params_,
        'best_score': random_search.best_score_,
        'cv_results': random_search.cv_results_
    }


def evaluate_tuned_models(results_dict: dict, X_train, y_train, X_test, y_test) -> dict:
    """Evaluate tuned models on test set."""
    print("\n" + "="*80)
    print("EVALUATING TUNED MODELS")
    print("="*80)
    
    evaluation_results = {}
    
    for model_name, result in results_dict.items():
        print(f"\n{model_name}:")
        model = result['model']
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        print(f"  Train R²: {train_r2:.4f}")
        print(f"  Test R²: {test_r2:.4f}")
        print(f"  Train MAE: {train_mae:.2f} g/km")
        print(f"  Test MAE: {test_mae:.2f} g/km")
        print(f"  Train RMSE: {train_rmse:.2f} g/km")
        print(f"  Test RMSE: {test_rmse:.2f} g/km")
        print(f"  CV R² (best): {result['best_score']:.4f}")
        
        evaluation_results[model_name] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'cv_best_score': result['best_score'],
            'best_params': result['best_params']
        }
    
    return evaluation_results


def compare_before_after(evaluation_results: dict) -> None:
    """Compare tuned models with baseline."""
    print("\n" + "="*80)
    print("BEFORE vs AFTER TUNING COMPARISON")
    print("="*80)
    
    # Load baseline metrics
    with open("models/model_metrics.json", 'r') as f:
        baseline_metrics = json.load(f)
    
    # Create comparison
    comparison_data = []
    
    for model_name in evaluation_results.keys():
        baseline_name = model_name
        if baseline_name in baseline_metrics:
            comparison_data.append({
                'Model': model_name,
                'Baseline Test R²': baseline_metrics[baseline_name]['test_r2'],
                'Tuned Test R²': evaluation_results[model_name]['test_r2'],
                'Improvement': evaluation_results[model_name]['test_r2'] - baseline_metrics[baseline_name]['test_r2'],
                'Baseline MAE': baseline_metrics[baseline_name]['test_mae'],
                'Tuned MAE': evaluation_results[model_name]['test_mae'],
                'MAE Reduction': baseline_metrics[baseline_name]['test_mae'] - evaluation_results[model_name]['test_mae']
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\n" + comparison_df.to_string(index=False))
    
    # Visualize comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # R² comparison
    x = np.arange(len(comparison_df))
    width = 0.35
    
    axes[0].bar(x - width/2, comparison_df['Baseline Test R²'], width, label='Baseline', alpha=0.8)
    axes[0].bar(x + width/2, comparison_df['Tuned Test R²'], width, label='Tuned', alpha=0.8)
    axes[0].set_xlabel('Model')
    axes[0].set_ylabel('Test R²')
    axes[0].set_title('R² Score: Baseline vs Tuned')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(comparison_df['Model'])
    axes[0].legend()
    axes[0].set_ylim([0.99, 1.0])
    
    # MAE comparison
    axes[1].bar(x - width/2, comparison_df['Baseline MAE'], width, label='Baseline', alpha=0.8)
    axes[1].bar(x + width/2, comparison_df['Tuned MAE'], width, label='Tuned', alpha=0.8)
    axes[1].set_xlabel('Model')
    axes[1].set_ylabel('Test MAE (g/km)')
    axes[1].set_title('MAE: Baseline vs Tuned')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(comparison_df['Model'])
    axes[1].legend()
    
    plt.tight_layout()
    
    output_dir = "data/hyperparameter_tuning"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{output_dir}/tuning_comparison.png", dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_dir}/tuning_comparison.png")
    plt.close()
    
    # Save comparison
    comparison_df.to_csv(f"{output_dir}/tuning_comparison.csv", index=False)
    print(f"Saved: {output_dir}/tuning_comparison.csv")


def save_tuned_models(results_dict: dict, evaluation_results: dict, feature_cols: list) -> None:
    """Save tuned models and their parameters."""
    print("\n" + "="*80)
    print("SAVING TUNED MODELS")
    print("="*80)
    
    output_dir = "models/tuned"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for model_name, result in results_dict.items():
        # Save model
        model_filename = f"{model_name.lower().replace(' ', '_')}_tuned.pkl"
        model_path = f"{output_dir}/{model_filename}"
        
        with open(model_path, 'wb') as f:
            pickle.dump(result['model'], f)
        print(f"Saved: {model_path}")
        
        # Save hyperparameters
        params_filename = f"{model_name.lower().replace(' ', '_')}_params.json"
        params_path = f"{output_dir}/{params_filename}"
        
        with open(params_path, 'w') as f:
            json.dump(result['best_params'], f, indent=2)
        print(f"Saved: {params_path}")
    
    # Save feature columns
    columns_path = f"{output_dir}/model_columns.pkl"
    with open(columns_path, 'wb') as f:
        pickle.dump(feature_cols, f)
    print(f"Saved: {columns_path}")
    
    # Save evaluation metrics
    metrics_path = f"{output_dir}/tuned_metrics.json"
    metrics_data = {
        model_name: {
            'train_r2': float(metrics['train_r2']),
            'test_r2': float(metrics['test_r2']),
            'train_mae': float(metrics['train_mae']),
            'test_mae': float(metrics['test_mae']),
            'train_rmse': float(metrics['train_rmse']),
            'test_rmse': float(metrics['test_rmse']),
            'cv_best_score': float(metrics['cv_best_score']),
            'best_params': metrics['best_params']
        }
        for model_name, metrics in evaluation_results.items()
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Saved: {metrics_path}")


def main():
    """Main hyperparameter tuning pipeline."""
    print("="*80)
    print("HYPERPARAMETER TUNING")
    print("="*80)
    
    # Step 1: Load data
    X_train, X_test, y_train, y_test, feature_cols = load_data()
    
    # Step 2: Tune XGBoost
    xgb_results = tune_xgboost(X_train, y_train)
    
    # Step 3: Tune LightGBM
    lgb_results = tune_lightgbm(X_train, y_train)
    
    # Collect results
    results_dict = {
        'XGBoost': xgb_results,
        'LightGBM': lgb_results
    }
    
    # Step 4: Evaluate tuned models
    evaluation_results = evaluate_tuned_models(results_dict, X_train, y_train, X_test, y_test)
    
    # Step 5: Compare with baseline
    compare_before_after(evaluation_results)
    
    # Step 6: Save tuned models
    save_tuned_models(results_dict, evaluation_results, feature_cols)
    
    # Final summary
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING COMPLETE")
    print("="*80)
    print(f"✓ Models tuned: {len(results_dict)}")
    print(f"✓ Search iterations: 20 per model")
    print(f"✓ Cross-validation: 3-fold")
    print(f"✓ Tuned models saved: models/tuned/")
    print(f"✓ Comparison: data/hyperparameter_tuning/tuning_comparison.png")
    
    # Find best tuned model
    best_model_name = max(evaluation_results.keys(), 
                         key=lambda x: evaluation_results[x]['test_r2'])
    best_metrics = evaluation_results[best_model_name]
    
    print(f"\n✓ Best tuned model: {best_model_name}")
    print(f"  Test R²: {best_metrics['test_r2']:.4f}")
    print(f"  Test MAE: {best_metrics['test_mae']:.2f} g/km")
    print(f"  Test RMSE: {best_metrics['test_rmse']:.2f} g/km")


if __name__ == "__main__":
    main()
