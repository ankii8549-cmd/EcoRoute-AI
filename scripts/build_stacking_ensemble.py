"""
Build Stacking Ensemble Model
Combines best base models (XGBoost, LightGBM, RandomForest, CatBoost) with a meta-learner
"""

import pandas as pd
import numpy as np
import pickle
import json
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

def load_data():
    """Load and prepare the engineered dataset with one-hot encoding"""
    print("Loading engineered dataset...")
    df = pd.read_csv('data/engineered_fuel_consumption.csv')
    
    target_col = 'CO2 emissions (g/km)'
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # One-hot encode categorical variables
    if categorical_cols:
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    else:
        df_encoded = df.copy()
    
    # Sanitize column names (same as hyperparameter tuning script)
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
    
    # Select features (exclude target, rating columns, and year)
    feature_cols = [col for col in df_encoded.columns 
                   if col != target_col_clean and 'rating' not in col.lower() 
                   and 'year' not in col.lower()]
    
    X = df_encoded[feature_cols]
    y = df_encoded[target_col_clean]
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Target range: {y.min():.2f} - {y.max():.2f}")
    
    return X, y

def load_tuned_models():
    """Load the tuned base models"""
    print("\nLoading tuned base models...")
    
    # Load XGBoost
    with open('models/tuned/xgboost_tuned.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    print("✓ XGBoost loaded")
    
    # Load LightGBM
    with open('models/tuned/lightgbm_tuned.pkl', 'rb') as f:
        lgb_model = pickle.load(f)
    print("✓ LightGBM loaded")
    
    # Load RandomForest
    with open('models/randomforest_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    print("✓ RandomForest loaded")
    
    # Load CatBoost
    with open('models/catboost_model.pkl', 'rb') as f:
        cb_model = pickle.load(f)
    print("✓ CatBoost loaded")
    
    return xgb_model, lgb_model, rf_model, cb_model

def build_stacking_ensemble(X_train, y_train, base_models):
    """Build stacking ensemble with Ridge meta-learner"""
    print("\nBuilding stacking ensemble...")
    
    xgb_model, lgb_model, rf_model, cb_model = base_models
    
    # Define base estimators
    estimators = [
        ('xgboost', xgb_model),
        ('lightgbm', lgb_model),
        ('randomforest', rf_model),
        ('catboost', cb_model)
    ]
    
    # Create stacking regressor with Ridge meta-learner
    stacking_model = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=5,
        n_jobs=-1
    )
    
    print("Training stacking ensemble with 5-fold cross-validation...")
    stacking_model.fit(X_train, y_train)
    print("✓ Stacking ensemble trained")
    
    return stacking_model

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, skip_cv=False):
    """Evaluate model performance"""
    # Training predictions
    y_train_pred = model.predict(X_train)
    train_r2 = r2_score(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    
    # Test predictions
    y_test_pred = model.predict(X_test)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    # Cross-validation score (skip for stacking ensemble to avoid double CV)
    if skip_cv:
        cv_mean = None
        cv_std = None
    else:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                     scoring='r2', n_jobs=-1)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
    
    metrics = {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'cv_mean_r2': cv_mean,
        'cv_std_r2': cv_std
    }
    
    print(f"\n{model_name} Performance:")
    print(f"  Train R²: {train_r2:.6f}")
    print(f"  Test R²:  {test_r2:.6f}")
    print(f"  Train MAE: {train_mae:.4f} g/km")
    print(f"  Test MAE:  {test_mae:.4f} g/km")
    print(f"  Train RMSE: {train_rmse:.4f} g/km")
    print(f"  Test RMSE:  {test_rmse:.4f} g/km")
    if cv_mean is not None:
        print(f"  CV R² (mean ± std): {cv_mean:.6f} ± {cv_std:.6f}")
    else:
        print(f"  CV R²: Skipped (already uses internal CV)")
    
    return metrics

def compare_all_models(X_train, X_test, y_train, y_test, base_models, stacking_model):
    """Compare all models including stacking ensemble"""
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    xgb_model, lgb_model, rf_model, cb_model = base_models
    
    all_metrics = {}
    
    # Evaluate each base model
    all_metrics['XGBoost'] = evaluate_model(xgb_model, X_train, X_test, y_train, y_test, "XGBoost")
    all_metrics['LightGBM'] = evaluate_model(lgb_model, X_train, X_test, y_train, y_test, "LightGBM")
    all_metrics['RandomForest'] = evaluate_model(rf_model, X_train, X_test, y_train, y_test, "RandomForest")
    all_metrics['CatBoost'] = evaluate_model(cb_model, X_train, X_test, y_train, y_test, "CatBoost")
    
    # Evaluate stacking ensemble (skip CV to avoid double cross-validation)
    all_metrics['Stacking Ensemble'] = evaluate_model(stacking_model, X_train, X_test, 
                                                       y_train, y_test, "Stacking Ensemble", skip_cv=True)
    
    # Create comparison table
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    
    comparison_df = pd.DataFrame(all_metrics).T
    comparison_df = comparison_df[['test_r2', 'test_mae', 'test_rmse', 'cv_mean_r2']]
    comparison_df.columns = ['Test R²', 'Test MAE', 'Test RMSE', 'CV R² (mean)']
    
    print(comparison_df.to_string())
    
    # Find best model
    best_model_name = comparison_df['Test R²'].idxmax()
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   Test R²: {comparison_df.loc[best_model_name, 'Test R²']:.6f}")
    print(f"   Test MAE: {comparison_df.loc[best_model_name, 'Test MAE']:.4f} g/km")
    
    return all_metrics, best_model_name

def save_stacking_model(model, model_columns, metrics):
    """Save the stacking ensemble model"""
    print("\nSaving stacking ensemble model...")
    
    # Save model
    with open('models/stacking_ensemble.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("✓ Model saved to models/stacking_ensemble.pkl")
    
    # Save model columns
    with open('models/stacking_model_columns.pkl', 'wb') as f:
        pickle.dump(model_columns, f)
    print("✓ Model columns saved to models/stacking_model_columns.pkl")
    
    # Save metrics
    with open('models/stacking_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("✓ Metrics saved to models/stacking_metrics.json")

def main():
    """Main execution function"""
    print("="*60)
    print("STACKING ENSEMBLE MODEL BUILDER")
    print("="*60)
    
    # Load data
    X, y = load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set:  {X_test.shape[0]} samples")
    
    # Load tuned base models
    base_models = load_tuned_models()
    
    # Build stacking ensemble
    stacking_model = build_stacking_ensemble(X_train, y_train, base_models)
    
    # Compare all models
    all_metrics, best_model_name = compare_all_models(
        X_train, X_test, y_train, y_test, base_models, stacking_model
    )
    
    # Save stacking model
    save_stacking_model(stacking_model, X.columns.tolist(), all_metrics)
    
    print("\n" + "="*60)
    print("STACKING ENSEMBLE BUILD COMPLETE!")
    print("="*60)
    print(f"\nBest performing model: {best_model_name}")
    print(f"Test R²: {all_metrics[best_model_name]['test_r2']:.6f}")
    print(f"Test MAE: {all_metrics[best_model_name]['test_mae']:.4f} g/km")
    print(f"Test RMSE: {all_metrics[best_model_name]['test_rmse']:.4f} g/km")

if __name__ == "__main__":
    main()
