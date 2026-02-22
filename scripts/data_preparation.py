"""
Data Preparation Script for Canada Fuel Consumption Dataset (2015-2024)

This script loads, cleans, and prepares the Canada fuel consumption dataset
for ML model training. It performs:
- Data loading and initial inspection
- Missing value handling
- Outlier detection and removal using IQR method
- Electric vehicle filtering (no CO2 emissions)
- Exploratory data analysis with visualizations

Target: Clean dataset with 40,000+ vehicles
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def load_dataset(file_path: str) -> pd.DataFrame:
    """Load the Canada fuel consumption dataset."""
    print(f"Loading dataset from: {file_path}")
    df = pd.read_csv(file_path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def inspect_dataset(df: pd.DataFrame) -> None:
    """Perform initial inspection of the dataset."""
    print("\n" + "="*80)
    print("DATASET INSPECTION")
    print("="*80)
    
    print("\nColumn Names:")
    print(df.columns.tolist())
    
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nDataset Info:")
    print(df.info())
    
    print("\nBasic Statistics:")
    print(df.describe())
    
    print("\nMissing Values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Percentage': missing_pct
    })
    print(missing_df[missing_df['Missing Count'] > 0])


def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with missing values."""
    print("\n" + "="*80)
    print("CLEANING MISSING VALUES")
    print("="*80)
    
    initial_rows = len(df)
    df_clean = df.dropna()
    removed_rows = initial_rows - len(df_clean)
    
    print(f"Initial rows: {initial_rows}")
    print(f"Rows removed: {removed_rows}")
    print(f"Remaining rows: {len(df_clean)}")
    print(f"Percentage removed: {(removed_rows/initial_rows)*100:.2f}%")
    
    return df_clean


def remove_outliers_iqr(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Remove outliers using the IQR (Interquartile Range) method."""
    print("\n" + "="*80)
    print("REMOVING OUTLIERS (IQR METHOD)")
    print("="*80)
    
    initial_rows = len(df)
    df_clean = df.copy()
    
    for col in columns:
        if col not in df_clean.columns:
            print(f"Warning: Column '{col}' not found in dataset")
            continue
            
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        before = len(df_clean)
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        after = len(df_clean)
        
        print(f"{col}:")
        print(f"  Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}")
        print(f"  Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
        print(f"  Outliers removed: {before - after}")
    
    total_removed = initial_rows - len(df_clean)
    print(f"\nTotal rows removed: {total_removed}")
    print(f"Remaining rows: {len(df_clean)}")
    print(f"Percentage removed: {(total_removed/initial_rows)*100:.2f}%")
    
    return df_clean


def remove_electric_vehicles(df: pd.DataFrame) -> pd.DataFrame:
    """Remove electric vehicles (no CO2 emissions)."""
    print("\n" + "="*80)
    print("REMOVING ELECTRIC VEHICLES")
    print("="*80)
    
    initial_rows = len(df)
    
    # Identify CO2 emission columns (may vary by dataset)
    co2_columns = [col for col in df.columns if 'CO2' in col.upper() or 'EMISSIONS' in col.upper()]
    print(f"CO2-related columns found: {co2_columns}")
    
    # Remove rows where all CO2 columns are 0 or NaN (electric vehicles)
    if co2_columns:
        # Keep only rows where at least one CO2 column has a positive value
        mask = (df[co2_columns] > 0).any(axis=1)
        df_clean = df[mask].copy()
    else:
        print("Warning: No CO2 columns found. Skipping electric vehicle removal.")
        df_clean = df.copy()
    
    removed_rows = initial_rows - len(df_clean)
    print(f"Initial rows: {initial_rows}")
    print(f"Electric vehicles removed: {removed_rows}")
    print(f"Remaining rows: {len(df_clean)}")
    print(f"Percentage removed: {(removed_rows/initial_rows)*100:.2f}%")
    
    return df_clean


def perform_eda(df: pd.DataFrame, output_dir: str = "data/eda_plots") -> None:
    """Perform exploratory data analysis with visualizations."""
    print("\n" + "="*80)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*80)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Distribution of key numerical features
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numerical_cols) > 0:
        print(f"\nNumerical columns: {len(numerical_cols)}")
        
        # Plot distributions for first 6 numerical columns
        n_cols = min(6, len(numerical_cols))
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, col in enumerate(numerical_cols[:n_cols]):
            axes[i].hist(df[col].dropna(), bins=50, edgecolor='black', alpha=0.7)
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
        
        # Hide unused subplots
        for i in range(n_cols, 6):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/distributions.png", dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir}/distributions.png")
        plt.close()
    
    # 2. Correlation heatmap
    if len(numerical_cols) > 1:
        # Select subset of columns for better visualization
        corr_cols = numerical_cols[:15] if len(numerical_cols) > 15 else numerical_cols
        correlation_matrix = df[corr_cols].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                    center=0, square=True, linewidths=1)
        plt.title('Correlation Heatmap of Numerical Features')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/correlation_heatmap.png", dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir}/correlation_heatmap.png")
        plt.close()
    
    # 3. Categorical feature distributions
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if len(categorical_cols) > 0:
        print(f"\nCategorical columns: {len(categorical_cols)}")
        
        # Plot top categories for first 4 categorical columns
        n_cat = min(4, len(categorical_cols))
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, col in enumerate(categorical_cols[:n_cat]):
            top_categories = df[col].value_counts().head(10)
            axes[i].barh(range(len(top_categories)), top_categories.values)
            axes[i].set_yticks(range(len(top_categories)))
            axes[i].set_yticklabels(top_categories.index)
            axes[i].set_title(f'Top 10 {col}')
            axes[i].set_xlabel('Count')
        
        # Hide unused subplots
        for i in range(n_cat, 4):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/categorical_distributions.png", dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir}/categorical_distributions.png")
        plt.close()
    
    # 4. Summary statistics
    print("\nSummary Statistics:")
    print(df.describe())
    
    print("\nDataset shape after cleaning:")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")


def save_cleaned_dataset(df: pd.DataFrame, output_path: str) -> None:
    """Save the cleaned dataset."""
    print("\n" + "="*80)
    print("SAVING CLEANED DATASET")
    print("="*80)
    
    df.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to: {output_path}")
    print(f"Final dataset size: {len(df)} rows, {len(df.columns)} columns")


def main():
    """Main data preparation pipeline."""
    print("="*80)
    print("CANADA FUEL CONSUMPTION DATASET - DATA PREPARATION")
    print("="*80)
    
    # Configuration
    input_file = "data/2015-2024 Fuel Consumption Ratings.csv"
    output_file = "data/cleaned_fuel_consumption.csv"
    
    # Step 1: Load dataset
    df = load_dataset(input_file)
    
    # Step 2: Inspect dataset
    inspect_dataset(df)
    
    # Step 3: Clean missing values
    df = clean_missing_values(df)
    
    # Step 4: Remove electric vehicles
    df = remove_electric_vehicles(df)
    
    # Step 5: Remove outliers (identify numerical columns dynamically)
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Focus on key columns that might have outliers
    outlier_cols = [col for col in numerical_cols if any(
        keyword in col.upper() for keyword in ['ENGINE', 'CONSUMPTION', 'CO2', 'CYLINDERS']
    )]
    
    if outlier_cols:
        print(f"\nApplying outlier removal to: {outlier_cols}")
        df = remove_outliers_iqr(df, outlier_cols)
    else:
        print("\nNo specific columns identified for outlier removal")
    
    # Step 6: Perform EDA
    perform_eda(df)
    
    # Step 7: Save cleaned dataset
    save_cleaned_dataset(df, output_file)
    
    # Final summary
    print("\n" + "="*80)
    print("DATA PREPARATION COMPLETE")
    print("="*80)
    print(f"✓ Cleaned dataset: {len(df)} rows")
    print(f"✓ Target achieved: {'YES' if len(df) >= 40000 else 'NO'} (Target: 40,000+)")
    print(f"✓ Output file: {output_file}")
    print(f"✓ EDA plots: data/eda_plots/")


if __name__ == "__main__":
    main()
