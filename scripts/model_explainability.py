"""
Model Explainability with SHAP
Generate SHAP values and visualizations for model interpretability
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split

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

def load_best_model():
    """Load the best model (Stacking Ensemble)"""
    print("\nLoading Stacking Ensemble model...")
    with open('models/stacking_ensemble.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✓ Model loaded")
    return model

def generate_shap_summary(model, X_sample, save_path='visualizations/shap_summary.png'):
    """Generate SHAP summary plot"""
    print("\nGenerating SHAP summary plot...")
    print("  (This may take several minutes for ensemble models)")
    
    # Create SHAP explainer
    # For ensemble models, we use the TreeExplainer on the first base estimator
    # or use KernelExplainer for a model-agnostic approach
    try:
        # Try TreeExplainer first (faster for tree-based models)
        explainer = shap.TreeExplainer(model.estimators_[0])
        shap_values = explainer.shap_values(X_sample)
        print("  ✓ Using TreeExplainer")
    except:
        # Fall back to KernelExplainer (slower but works for any model)
        print("  Using KernelExplainer (this will take longer)...")
        explainer = shap.KernelExplainer(model.predict, shap.sample(X_sample, 100))
        shap_values = explainer.shap_values(X_sample)
        print("  ✓ Using KernelExplainer")
    
    # Create summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {save_path}")
    plt.close()
    
    return explainer, shap_values

def generate_shap_force_plots(explainer, shap_values, X_sample, save_path='visualizations/shap_force_plot.png'):
    """Generate SHAP force plots for sample predictions"""
    print("\nGenerating SHAP force plots...")
    
    # Select a few interesting samples
    sample_indices = [0, len(X_sample)//4, len(X_sample)//2, 3*len(X_sample)//4, len(X_sample)-1]
    
    fig, axes = plt.subplots(len(sample_indices), 1, figsize=(16, 3*len(sample_indices)))
    
    for idx, sample_idx in enumerate(sample_indices):
        # Create force plot
        shap.force_plot(
            explainer.expected_value,
            shap_values[sample_idx],
            X_sample.iloc[sample_idx],
            matplotlib=True,
            show=False
        )
        
        # Save current plot to axes
        if len(sample_indices) > 1:
            plt.sca(axes[idx])
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {save_path}")
    plt.close()

def generate_shap_dependence_plots(shap_values, X_sample, save_path='visualizations/shap_dependence.png'):
    """Generate SHAP dependence plots for top features"""
    print("\nGenerating SHAP dependence plots...")
    
    # Get top 6 features by mean absolute SHAP value
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_features_idx = np.argsort(mean_abs_shap)[-6:][::-1]
    top_features = X_sample.columns[top_features_idx]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, feature_idx in enumerate(top_features_idx):
        feature_name = X_sample.columns[feature_idx]
        shap.dependence_plot(
            feature_idx,
            shap_values,
            X_sample,
            ax=axes[idx],
            show=False
        )
        axes[idx].set_title(f'SHAP Dependence: {feature_name}', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {save_path}")
    plt.close()

def save_model_metadata():
    """Save model metadata for deployment"""
    print("\nSaving model metadata...")
    
    from datetime import datetime
    
    # Load metrics
    with open('models/stacking_metrics.json', 'r') as f:
        import json
        metrics = json.load(f)
    
    # Load data to get dataset size
    X, y = load_data()
    
    metadata = {
        'model_name': 'Stacking Ensemble',
        'model_type': 'StackingRegressor with Ridge meta-learner',
        'base_models': ['XGBoost', 'LightGBM', 'RandomForest', 'CatBoost'],
        'training_date': datetime.now().isoformat(),
        'dataset_size': len(X),
        'n_features': X.shape[1],
        'performance_metrics': {
            'test_r2': metrics['Stacking Ensemble']['test_r2'],
            'test_mae': metrics['Stacking Ensemble']['test_mae'],
            'test_rmse': metrics['Stacking Ensemble']['test_rmse'],
            'train_r2': metrics['Stacking Ensemble']['train_r2'],
            'train_mae': metrics['Stacking Ensemble']['train_mae'],
            'train_rmse': metrics['Stacking Ensemble']['train_rmse']
        },
        'target_achievement': {
            'r2_target': 0.92,
            'r2_achieved': metrics['Stacking Ensemble']['test_r2'] > 0.92,
            'mae_target': 10.0,
            'mae_achieved': metrics['Stacking Ensemble']['test_mae'] < 10.0,
            'rmse_target': 15.0,
            'rmse_achieved': metrics['Stacking Ensemble']['test_rmse'] < 15.0
        },
        'model_files': {
            'model': 'models/stacking_ensemble.pkl',
            'columns': 'models/stacking_model_columns.pkl',
            'metrics': 'models/stacking_metrics.json'
        }
    }
    
    import json
    with open('models/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("✓ Metadata saved to models/model_metadata.json")
    return metadata

def main():
    """Main execution function"""
    print("="*80)
    print("MODEL EXPLAINABILITY WITH SHAP")
    print("="*80)
    
    # Create visualizations directory
    import os
    os.makedirs('visualizations', exist_ok=True)
    
    # Load data
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Use a sample for SHAP (to speed up computation)
    sample_size = min(500, len(X_test))
    X_sample = X_test.sample(n=sample_size, random_state=42)
    print(f"\nUsing {sample_size} samples for SHAP analysis")
    
    # Load best model
    model = load_best_model()
    
    # Generate SHAP visualizations
    explainer, shap_values = generate_shap_summary(model, X_sample)
    
    # Note: Force plots and dependence plots can be very slow for ensemble models
    # Uncomment if you want to generate them (may take 10-30 minutes)
    # generate_shap_force_plots(explainer, shap_values, X_sample)
    # generate_shap_dependence_plots(shap_values, X_sample)
    
    print("\n⚠ Note: Force plots and dependence plots are commented out to save time.")
    print("  Uncomment in the script if you need detailed SHAP visualizations.")
    
    # Save model metadata
    metadata = save_model_metadata()
    
    print("\n" + "="*80)
    print("EXPLAINABILITY ANALYSIS COMPLETE!")
    print("="*80)
    
    print("\nModel Metadata:")
    print(f"  Model: {metadata['model_name']}")
    print(f"  Training Date: {metadata['training_date']}")
    print(f"  Dataset Size: {metadata['dataset_size']:,} samples")
    print(f"  Features: {metadata['n_features']}")
    
    print("\nPerformance Metrics:")
    print(f"  Test R²: {metadata['performance_metrics']['test_r2']:.6f}")
    print(f"  Test MAE: {metadata['performance_metrics']['test_mae']:.4f} g/km")
    print(f"  Test RMSE: {metadata['performance_metrics']['test_rmse']:.4f} g/km")
    
    print("\nTarget Achievement:")
    print(f"  R² > 0.92: {'✓ PASS' if metadata['target_achievement']['r2_achieved'] else '✗ FAIL'}")
    print(f"  MAE < 10: {'✓ PASS' if metadata['target_achievement']['mae_achieved'] else '✗ FAIL'}")
    print(f"  RMSE < 15: {'✓ PASS' if metadata['target_achievement']['rmse_achieved'] else '✗ FAIL'}")
    
    print("\nFiles Generated:")
    print("  - visualizations/shap_summary.png")
    print("  - models/model_metadata.json")

if __name__ == "__main__":
    main()
