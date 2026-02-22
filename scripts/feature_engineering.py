"""
Advanced Feature Engineering Script

This script performs advanced feature engineering on the cleaned Canada dataset:
- Creates power_to_weight_ratio = Engine Size / Vehicle Weight (estimated)
- Creates fuel_efficiency_score from combined fuel consumption
- Creates engine_displacement_per_cylinder = Engine Size / Cylinders
- Creates interaction terms: engine_size × distance, mileage × traffic
- Performs correlation analysis and removes redundant features
- Selects top 10-12 most important features

Target: Engineered features that improve model accuracy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_cleaned_dataset(file_path: str) -> pd.DataFrame:
    """Load the cleaned dataset."""
    print(f"Loading cleaned dataset from: {file_path}")
    df = pd.read_csv(file_path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def create_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create advanced engineered features."""
    print("\n" + "="*80)
    print("CREATING ENGINEERED FEATURES")
    print("="*80)
    
    df_eng = df.copy()
    
    # 1. Fuel Efficiency Score (inverse of combined fuel consumption)
    # Lower consumption = higher efficiency
    print("\n1. Creating fuel_efficiency_score...")
    df_eng['fuel_efficiency_score'] = 100 / df_eng['Combined (L/100 km)']
    print(f"   Range: [{df_eng['fuel_efficiency_score'].min():.2f}, {df_eng['fuel_efficiency_score'].max():.2f}]")
    
    # 2. Engine Displacement Per Cylinder
    print("\n2. Creating engine_displacement_per_cylinder...")
    df_eng['engine_displacement_per_cylinder'] = df_eng['Engine size (L)'] / df_eng['Cylinders']
    print(f"   Range: [{df_eng['engine_displacement_per_cylinder'].min():.2f}, {df_eng['engine_displacement_per_cylinder'].max():.2f}]")
    
    # 3. Power-to-Weight Ratio (estimated)
    # Since we don't have actual weight, we'll estimate based on engine size and vehicle class
    # Larger engines and certain vehicle classes typically mean heavier vehicles
    print("\n3. Creating power_to_weight_ratio (estimated)...")
    
    # Estimate vehicle weight based on engine size and vehicle class
    # Base weight estimation: engine size * 300 kg (rough approximation)
    base_weight = df_eng['Engine size (L)'] * 300
    
    # Adjust by vehicle class (if available)
    if 'Vehicle class' in df_eng.columns:
        # Create weight multipliers for different vehicle classes
        weight_multipliers = {
            'COMPACT': 0.8,
            'MID-SIZE': 1.0,
            'FULL-SIZE': 1.2,
            'SUV': 1.3,
            'PICKUP': 1.4,
            'VAN': 1.3,
            'STATION': 1.1,
            'TWO-SEATER': 0.7,
            'MINICOMPACT': 0.6,
            'SUBCOMPACT': 0.7
        }
        
        # Apply multipliers
        df_eng['estimated_weight'] = base_weight
        for class_name, multiplier in weight_multipliers.items():
            mask = df_eng['Vehicle class'].str.contains(class_name, case=False, na=False)
            df_eng.loc[mask, 'estimated_weight'] *= multiplier
    else:
        df_eng['estimated_weight'] = base_weight
    
    # Power-to-weight ratio (engine size as proxy for power)
    df_eng['power_to_weight_ratio'] = df_eng['Engine size (L)'] / df_eng['estimated_weight']
    print(f"   Range: [{df_eng['power_to_weight_ratio'].min():.6f}, {df_eng['power_to_weight_ratio'].max():.6f}]")
    
    # 4. City-Highway Fuel Consumption Difference
    print("\n4. Creating city_highway_diff...")
    df_eng['city_highway_diff'] = df_eng['City (L/100 km)'] - df_eng['Highway (L/100 km)']
    print(f"   Range: [{df_eng['city_highway_diff'].min():.2f}, {df_eng['city_highway_diff'].max():.2f}]")
    
    # 5. Engine Efficiency Indicator
    print("\n5. Creating engine_efficiency...")
    # CO2 per liter of engine displacement
    df_eng['engine_efficiency'] = df_eng['CO2 emissions (g/km)'] / df_eng['Engine size (L)']
    print(f"   Range: [{df_eng['engine_efficiency'].min():.2f}, {df_eng['engine_efficiency'].max():.2f}]")
    
    # 6. Cylinder Efficiency
    print("\n6. Creating cylinder_efficiency...")
    df_eng['cylinder_efficiency'] = df_eng['CO2 emissions (g/km)'] / df_eng['Cylinders']
    print(f"   Range: [{df_eng['cylinder_efficiency'].min():.2f}, {df_eng['cylinder_efficiency'].max():.2f}]")
    
    print(f"\nTotal features created: 6")
    print(f"New dataset shape: {df_eng.shape}")
    
    return df_eng


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create interaction features for ML model training."""
    print("\n" + "="*80)
    print("CREATING INTERACTION FEATURES")
    print("="*80)
    
    df_int = df.copy()
    
    # For ML training, we'll need to simulate distance and traffic
    # These will be used as templates for the actual prediction
    print("\nNote: Creating sample interaction features for demonstration")
    print("Actual interactions will be computed during prediction time")
    
    # Sample distance and traffic for feature importance analysis
    sample_distance = 100  # km
    sample_traffic = 2  # medium
    
    # 1. Engine size × distance interaction
    df_int['engine_distance_interaction'] = df_int['Engine size (L)'] * sample_distance
    
    # 2. Fuel efficiency × traffic interaction
    df_int['efficiency_traffic_interaction'] = df_int['fuel_efficiency_score'] * sample_traffic
    
    # 3. Combined consumption × distance
    df_int['consumption_distance_interaction'] = df_int['Combined (L/100 km)'] * sample_distance
    
    print(f"Interaction features created: 3")
    print(f"Dataset shape: {df_int.shape}")
    
    return df_int


def analyze_correlations(df: pd.DataFrame, target_col: str = 'CO2 emissions (g/km)') -> pd.DataFrame:
    """Analyze feature correlations and identify redundant features."""
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS")
    print("="*80)
    
    # Select only numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Calculate correlation matrix
    corr_matrix = df[numerical_cols].corr()
    
    # Correlation with target
    target_corr = corr_matrix[target_col].abs().sort_values(ascending=False)
    print(f"\nTop 15 features correlated with {target_col}:")
    print(target_corr.head(15))
    
    # Find highly correlated feature pairs (potential redundancy)
    print("\nHighly correlated feature pairs (|r| > 0.9):")
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.9:
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j]
                ))
    
    if high_corr_pairs:
        for feat1, feat2, corr_val in high_corr_pairs:
            print(f"  {feat1} <-> {feat2}: {corr_val:.3f}")
    else:
        print("  No highly correlated pairs found")
    
    # Visualize correlation heatmap
    plt.figure(figsize=(14, 12))
    
    # Select top features for visualization
    top_features = target_corr.head(20).index.tolist()
    corr_subset = df[top_features].corr()
    
    sns.heatmap(corr_subset, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Heatmap - Top 20 Features')
    plt.tight_layout()
    
    output_dir = "data/feature_engineering"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{output_dir}/correlation_heatmap.png", dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_dir}/correlation_heatmap.png")
    plt.close()
    
    return target_corr


def select_top_features(df: pd.DataFrame, target_col: str = 'CO2 emissions (g/km)', 
                       n_features: int = 12) -> list:
    """Select top N most important features using Random Forest."""
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*80)
    
    # Prepare data
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove target and ID-like columns
    feature_cols = [col for col in numerical_cols 
                   if col != target_col and 'year' not in col.lower() 
                   and 'rating' not in col.lower()]
    
    X = df[feature_cols].fillna(0)
    y = df[target_col]
    
    print(f"\nTraining Random Forest for feature importance...")
    print(f"Features: {len(feature_cols)}")
    print(f"Samples: {len(X)}")
    
    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    # Get feature importances
    importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop {n_features} most important features:")
    print(importances.head(n_features))
    
    # Visualize feature importances
    plt.figure(figsize=(10, 8))
    top_n = importances.head(n_features)
    plt.barh(range(len(top_n)), top_n['importance'])
    plt.yticks(range(len(top_n)), top_n['feature'])
    plt.xlabel('Feature Importance')
    plt.title(f'Top {n_features} Most Important Features')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    output_dir = "data/feature_engineering"
    plt.savefig(f"{output_dir}/feature_importance.png", dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_dir}/feature_importance.png")
    plt.close()
    
    # Return top feature names
    top_features = importances.head(n_features)['feature'].tolist()
    
    return top_features


def remove_redundant_features(df: pd.DataFrame, target_col: str = 'CO2 emissions (g/km)',
                              corr_threshold: float = 0.95) -> pd.DataFrame:
    """Remove redundant features based on correlation threshold."""
    print("\n" + "="*80)
    print("REMOVING REDUNDANT FEATURES")
    print("="*80)
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numerical_cols if col != target_col]
    
    # Calculate correlation matrix
    corr_matrix = df[feature_cols].corr().abs()
    
    # Find features to remove
    to_remove = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > corr_threshold:
                # Remove the feature with lower correlation to target
                feat1 = corr_matrix.columns[i]
                feat2 = corr_matrix.columns[j]
                
                target_corr1 = abs(df[feat1].corr(df[target_col]))
                target_corr2 = abs(df[feat2].corr(df[target_col]))
                
                if target_corr1 < target_corr2:
                    to_remove.add(feat1)
                    print(f"Removing {feat1} (corr with {feat2}: {corr_matrix.iloc[i, j]:.3f})")
                else:
                    to_remove.add(feat2)
                    print(f"Removing {feat2} (corr with {feat1}: {corr_matrix.iloc[i, j]:.3f})")
    
    if to_remove:
        df_clean = df.drop(columns=list(to_remove))
        print(f"\nRemoved {len(to_remove)} redundant features")
    else:
        df_clean = df.copy()
        print("\nNo redundant features found")
    
    print(f"Final dataset shape: {df_clean.shape}")
    
    return df_clean


def save_engineered_dataset(df: pd.DataFrame, output_path: str, 
                           selected_features: list = None) -> None:
    """Save the engineered dataset and feature list."""
    print("\n" + "="*80)
    print("SAVING ENGINEERED DATASET")
    print("="*80)
    
    # Save full dataset
    df.to_csv(output_path, index=False)
    print(f"Engineered dataset saved to: {output_path}")
    print(f"Shape: {df.shape}")
    
    # Save selected features list
    if selected_features:
        features_path = output_path.replace('.csv', '_features.txt')
        with open(features_path, 'w') as f:
            f.write("Top Selected Features:\n")
            f.write("="*50 + "\n")
            for i, feat in enumerate(selected_features, 1):
                f.write(f"{i}. {feat}\n")
        print(f"Selected features saved to: {features_path}")


def main():
    """Main feature engineering pipeline."""
    print("="*80)
    print("ADVANCED FEATURE ENGINEERING")
    print("="*80)
    
    # Configuration
    input_file = "data/cleaned_fuel_consumption.csv"
    output_file = "data/engineered_fuel_consumption.csv"
    target_column = "CO2 emissions (g/km)"
    n_top_features = 12
    
    # Step 1: Load cleaned dataset
    df = load_cleaned_dataset(input_file)
    
    # Step 2: Create engineered features
    df = create_engineered_features(df)
    
    # Step 3: Create interaction features
    df = create_interaction_features(df)
    
    # Step 4: Analyze correlations
    target_corr = analyze_correlations(df, target_column)
    
    # Step 5: Remove redundant features
    df = remove_redundant_features(df, target_column, corr_threshold=0.95)
    
    # Step 6: Select top features
    top_features = select_top_features(df, target_column, n_top_features)
    
    # Step 7: Save engineered dataset
    save_engineered_dataset(df, output_file, top_features)
    
    # Final summary
    print("\n" + "="*80)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*80)
    print(f"✓ Original features: {len(df.columns)}")
    print(f"✓ Top selected features: {len(top_features)}")
    print(f"✓ Output file: {output_file}")
    print(f"✓ Feature list: {output_file.replace('.csv', '_features.txt')}")
    print(f"✓ Visualizations: data/feature_engineering/")
    
    print("\nTop Selected Features:")
    for i, feat in enumerate(top_features, 1):
        print(f"  {i}. {feat}")


if __name__ == "__main__":
    main()
