"""
Train Multiple ML Models Script

This script trains and evaluates multiple ML models:
- RandomForest (baseline - current model)
- XGBoost (gradient boosting)
- LightGBM (faster gradient boosting)
- CatBoost (handles categorical features)

Implements 5-fold cross-validation for each model.
Target: 4 trained models with cross-validation scores
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def load_engineered_dataset(file_path: str) -> tuple:
    """Load the engineered dataset and prepare features."""
    print(f"Loading engineered dataset from: {file_path}")
    df = pd.read_csv(file_path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Define target
    target_col = 'CO2 emissions (g/km)'
    
    # Select numerical features (exclude target, ratings, and year)
    feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns
                   if col != target_col and 'rating' not in col.lower() 
                   and 'year' not in col.lower()]
    
    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    print(f"\nFeature columns: {len(feature_cols)}")
    print(f"Categorical columns: {len(categorical_cols)}")
    print(f"Target: {target_col}")
    
    return df, feature_cols, categorical_cols, target_col


def prepare_data(df: pd.DataFrame, feature_cols: list, categorical_cols: list,
                target_col: str) -> tuple:
    """Prepare training and test sets."""
    print("\n" + "="*80)
    print("PREPARING DATA")
    print("="*80)
    
    # One-hot encode categorical features
    if categorical_cols:
        print(f"\nOne-hot encoding {len(categorical_cols)} categorical features...")
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        # Update feature columns to include encoded features
        encoded_feature_cols = [col for col in df_encoded.columns 
                               if col != target_col and 'rating' not in col.lower()
                               and 'year' not in col.lower()]
    else:
        df_encoded = df.copy()
        encoded_feature_cols = feature_cols
    
    # Sanitize column names for LightGBM (remove special characters)
    print("\nSanitizing column names for compatibility...")
    column_mapping = {}
    used_names = set()
    for col in df_encoded.columns:
        # Replace special characters with underscores
        new_col = col.replace('(', '').replace(')', '').replace('/', '_').replace(' ', '_')
        new_col = new_col.replace('[', '').replace(']', '').replace('<', '').replace('>', '')
        new_col = new_col.replace('{', '').replace('}', '').replace('"', '').replace("'", '')
        new_col = new_col.replace(':', '').replace(',', '').replace('.', '').replace('-', '_')
        new_col = new_col.replace('\\', '').replace('|', '').replace('&', 'and')
        # Remove any remaining non-alphanumeric characters except underscore
        new_col = ''.join(c if c.isalnum() or c == '_' else '_' for c in new_col)
        # Remove consecutive underscores
        while '__' in new_col:
            new_col = new_col.replace('__', '_')
        # Remove leading/trailing underscores
        new_col = new_col.strip('_')
        
        # Ensure uniqueness
        if new_col in used_names:
            counter = 1
            while f"{new_col}_{counter}" in used_names:
                counter += 1
            new_col = f"{new_col}_{counter}"
        
        used_names.add(new_col)
        column_mapping[col] = new_col
    
    df_encoded = df_encoded.rename(columns=column_mapping)
    
    # Update target and feature column names
    target_col_clean = column_mapping.get(target_col, target_col)
    encoded_feature_cols = [column_mapping.get(col, col) for col in encoded_feature_cols]
    
    X = df_encoded[encoded_feature_cols]
    y = df_encoded[target_col_clean]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Features: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test, encoded_feature_cols


def train_random_forest(X_train, y_train, X_test, y_test) -> dict:
    """Train Random Forest model (baseline)."""
    print("\n" + "="*80)
    print("TRAINING RANDOM FOREST (BASELINE)")
    print("="*80)
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    print("\nTraining model...")
    model.fit(X_train, y_train)
    
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
    
    # Cross-validation
    print("\nPerforming 5-fold cross-validation...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                scoring='r2', n_jobs=-1)
    
    results = {
        'model': model,
        'name': 'RandomForest',
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'cv_scores': cv_scores
    }
    
    print(f"\nResults:")
    print(f"  Train R²: {train_r2:.4f}")
    print(f"  Test R²: {test_r2:.4f}")
    print(f"  Train MAE: {train_mae:.2f} g/km")
    print(f"  Test MAE: {test_mae:.2f} g/km")
    print(f"  Train RMSE: {train_rmse:.2f} g/km")
    print(f"  Test RMSE: {test_rmse:.2f} g/km")
    print(f"  CV R² (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return results


def train_xgboost(X_train, y_train, X_test, y_test) -> dict:
    """Train XGBoost model."""
    print("\n" + "="*80)
    print("TRAINING XGBOOST")
    print("="*80)
    
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    print("\nTraining model...")
    model.fit(X_train, y_train)
    
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
    
    # Cross-validation
    print("\nPerforming 5-fold cross-validation...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5,
                                scoring='r2', n_jobs=-1)
    
    results = {
        'model': model,
        'name': 'XGBoost',
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'cv_scores': cv_scores
    }
    
    print(f"\nResults:")
    print(f"  Train R²: {train_r2:.4f}")
    print(f"  Test R²: {test_r2:.4f}")
    print(f"  Train MAE: {train_mae:.2f} g/km")
    print(f"  Test MAE: {test_mae:.2f} g/km")
    print(f"  Train RMSE: {train_rmse:.2f} g/km")
    print(f"  Test RMSE: {test_rmse:.2f} g/km")
    print(f"  CV R² (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return results


def train_lightgbm(X_train, y_train, X_test, y_test) -> dict:
    """Train LightGBM model."""
    print("\n" + "="*80)
    print("TRAINING LIGHTGBM")
    print("="*80)
    
    model = lgb.LGBMRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    print("\nTraining model...")
    model.fit(X_train, y_train)
    
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
    
    # Cross-validation
    print("\nPerforming 5-fold cross-validation...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5,
                                scoring='r2', n_jobs=-1)
    
    results = {
        'model': model,
        'name': 'LightGBM',
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'cv_scores': cv_scores
    }
    
    print(f"\nResults:")
    print(f"  Train R²: {train_r2:.4f}")
    print(f"  Test R²: {test_r2:.4f}")
    print(f"  Train MAE: {train_mae:.2f} g/km")
    print(f"  Test MAE: {test_mae:.2f} g/km")
    print(f"  Train RMSE: {train_rmse:.2f} g/km")
    print(f"  Test RMSE: {test_rmse:.2f} g/km")
    print(f"  CV R² (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return results


def train_catboost(X_train, y_train, X_test, y_test) -> dict:
    """Train CatBoost model."""
    print("\n" + "="*80)
    print("TRAINING CATBOOST")
    print("="*80)
    
    model = CatBoostRegressor(
        iterations=100,
        depth=6,
        learning_rate=0.1,
        random_state=42,
        verbose=False
    )
    
    print("\nTraining model...")
    model.fit(X_train, y_train)
    
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
    
    # Cross-validation
    print("\nPerforming 5-fold cross-validation...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5,
                                scoring='r2', n_jobs=-1)
    
    results = {
        'model': model,
        'name': 'CatBoost',
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'cv_scores': cv_scores
    }
    
    print(f"\nResults:")
    print(f"  Train R²: {train_r2:.4f}")
    print(f"  Test R²: {test_r2:.4f}")
    print(f"  Train MAE: {train_mae:.2f} g/km")
    print(f"  Test MAE: {test_mae:.2f} g/km")
    print(f"  Train RMSE: {train_rmse:.2f} g/km")
    print(f"  Test RMSE: {test_rmse:.2f} g/km")
    print(f"  CV R² (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return results


def compare_models(results_list: list) -> None:
    """Compare all trained models."""
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    # Create comparison DataFrame
    comparison_data = []
    for result in results_list:
        comparison_data.append({
            'Model': result['name'],
            'Train R²': result['train_r2'],
            'Test R²': result['test_r2'],
            'Train MAE': result['train_mae'],
            'Test MAE': result['test_mae'],
            'Train RMSE': result['train_rmse'],
            'Test RMSE': result['test_rmse'],
            'CV R² Mean': result['cv_mean'],
            'CV R² Std': result['cv_std']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\n" + comparison_df.to_string(index=False))
    
    # Visualize comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # R² scores
    axes[0, 0].bar(comparison_df['Model'], comparison_df['Test R²'])
    axes[0, 0].set_title('Test R² Score')
    axes[0, 0].set_ylabel('R²')
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].axhline(y=0.92, color='r', linestyle='--', label='Target (0.92)')
    axes[0, 0].legend()
    
    # MAE
    axes[0, 1].bar(comparison_df['Model'], comparison_df['Test MAE'])
    axes[0, 1].set_title('Test MAE')
    axes[0, 1].set_ylabel('MAE (g/km)')
    axes[0, 1].axhline(y=10, color='r', linestyle='--', label='Target (10)')
    axes[0, 1].legend()
    
    # RMSE
    axes[1, 0].bar(comparison_df['Model'], comparison_df['Test RMSE'])
    axes[1, 0].set_title('Test RMSE')
    axes[1, 0].set_ylabel('RMSE (g/km)')
    axes[1, 0].axhline(y=15, color='r', linestyle='--', label='Target (15)')
    axes[1, 0].legend()
    
    # CV scores
    axes[1, 1].bar(comparison_df['Model'], comparison_df['CV R² Mean'])
    axes[1, 1].errorbar(range(len(comparison_df)), comparison_df['CV R² Mean'],
                       yerr=comparison_df['CV R² Std'], fmt='none', color='black')
    axes[1, 1].set_title('Cross-Validation R² (Mean ± Std)')
    axes[1, 1].set_ylabel('R²')
    axes[1, 1].set_ylim([0, 1])
    
    plt.tight_layout()
    
    output_dir = "data/model_training"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{output_dir}/model_comparison.png", dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_dir}/model_comparison.png")
    plt.close()
    
    # Save comparison table
    comparison_df.to_csv(f"{output_dir}/model_comparison.csv", index=False)
    print(f"Saved: {output_dir}/model_comparison.csv")


def save_models(results_list: list, feature_cols: list) -> None:
    """Save all trained models."""
    print("\n" + "="*80)
    print("SAVING MODELS")
    print("="*80)
    
    output_dir = "models"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for result in results_list:
        model_name = result['name'].lower().replace(' ', '_')
        model_path = f"{output_dir}/{model_name}_model.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(result['model'], f)
        print(f"Saved: {model_path}")
    
    # Save feature columns
    columns_path = f"{output_dir}/model_columns.pkl"
    with open(columns_path, 'wb') as f:
        pickle.dump(feature_cols, f)
    print(f"Saved: {columns_path}")
    
    # Save metrics
    metrics_path = f"{output_dir}/model_metrics.json"
    metrics_data = {
        result['name']: {
            'train_r2': float(result['train_r2']),
            'test_r2': float(result['test_r2']),
            'train_mae': float(result['train_mae']),
            'test_mae': float(result['test_mae']),
            'train_rmse': float(result['train_rmse']),
            'test_rmse': float(result['test_rmse']),
            'cv_mean': float(result['cv_mean']),
            'cv_std': float(result['cv_std'])
        }
        for result in results_list
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Saved: {metrics_path}")


def main():
    """Main training pipeline."""
    print("="*80)
    print("TRAIN MULTIPLE ML MODELS")
    print("="*80)
    
    # Configuration
    input_file = "data/engineered_fuel_consumption.csv"
    
    # Step 1: Load dataset
    df, feature_cols, categorical_cols, target_col = load_engineered_dataset(input_file)
    
    # Step 2: Prepare data
    X_train, X_test, y_train, y_test, encoded_feature_cols = prepare_data(
        df, feature_cols, categorical_cols, target_col
    )
    
    # Step 3: Train models
    results_list = []
    
    # RandomForest
    rf_results = train_random_forest(X_train, y_train, X_test, y_test)
    results_list.append(rf_results)
    
    # XGBoost
    xgb_results = train_xgboost(X_train, y_train, X_test, y_test)
    results_list.append(xgb_results)
    
    # LightGBM
    lgb_results = train_lightgbm(X_train, y_train, X_test, y_test)
    results_list.append(lgb_results)
    
    # CatBoost
    cb_results = train_catboost(X_train, y_train, X_test, y_test)
    results_list.append(cb_results)
    
    # Step 4: Compare models
    compare_models(results_list)
    
    # Step 5: Save models
    save_models(results_list, encoded_feature_cols)
    
    # Final summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"✓ Models trained: {len(results_list)}")
    print(f"✓ Cross-validation: 5-fold")
    print(f"✓ Models saved: models/")
    print(f"✓ Comparison: data/model_training/model_comparison.png")
    
    # Find best model
    best_model = max(results_list, key=lambda x: x['test_r2'])
    print(f"\n✓ Best model: {best_model['name']}")
    print(f"  Test R²: {best_model['test_r2']:.4f}")
    print(f"  Test MAE: {best_model['test_mae']:.2f} g/km")
    print(f"  Test RMSE: {best_model['test_rmse']:.2f} g/km")


if __name__ == "__main__":
    main()
