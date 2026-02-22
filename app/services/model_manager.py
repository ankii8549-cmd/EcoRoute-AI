"""
ML model manager for the Vehicle Emission Eco-Route System.

This module provides loading and management of the trained ML model
for CO2 emission predictions with feature preparation and one-hot encoding.

Supports both legacy models and advanced stacking ensemble models.

Requirements: 3.4, 14.2
"""

import pickle
import pandas as pd
import json
from typing import Optional
from pathlib import Path
from app.core.exceptions import ModelLoadError


class EmissionModelManager:
    """
    Manages ML model loading and predictions.
    
    Handles loading the trained ML model (legacy or advanced stacking ensemble)
    and column definitions, preparing input features with one-hot encoding,
    and making predictions.
    
    Supports automatic fallback from advanced model to legacy model if needed.
    
    Requirements: 3.4, 14.2
    """
    
    def __init__(self, model_path: str, columns_path: str, use_advanced: bool = True):
        """
        Initialize the emission model manager.
        
        Args:
            model_path: Path to the trained model pickle file
            columns_path: Path to the model columns pickle file
            use_advanced: If True, try to load advanced stacking ensemble first
        """
        self.model_path = model_path
        self.columns_path = columns_path
        self.use_advanced = use_advanced
        self.model = None
        self.model_columns = None
        self.model_metadata = None
        self.model_type = "unknown"
    
    def load(self) -> None:
        """
        Load model and column definitions from pickle files.
        
        Attempts to load advanced stacking ensemble model first if use_advanced=True,
        falls back to legacy model if advanced model is not available.
        
        Loads the trained ML model and the expected feature columns
        into memory for fast prediction.
        
        Raises:
            ModelLoadError: If model files are missing or cannot be loaded
            
        Requirements: 3.4, 14.2
        """
        # Try to load advanced model first if requested
        if self.use_advanced:
            advanced_model_path = "models/stacking_ensemble.pkl"
            advanced_columns_path = "models/stacking_model_columns.pkl"
            metadata_path = "models/model_metadata.json"
            
            if Path(advanced_model_path).exists() and Path(advanced_columns_path).exists():
                try:
                    # Load advanced stacking ensemble model
                    with open(advanced_model_path, "rb") as f:
                        self.model = pickle.load(f)
                    
                    with open(advanced_columns_path, "rb") as f:
                        self.model_columns = pickle.load(f)
                    
                    # Load metadata if available
                    if Path(metadata_path).exists():
                        with open(metadata_path, "r") as f:
                            self.model_metadata = json.load(f)
                    
                    self.model_type = "stacking_ensemble"
                    print(f"✓ Loaded advanced stacking ensemble model")
                    if self.model_metadata:
                        print(f"  Test R²: {self.model_metadata['performance_metrics']['test_r2']:.6f}")
                        print(f"  Test MAE: {self.model_metadata['performance_metrics']['test_mae']:.4f} g/km")
                    return
                except Exception as e:
                    print(f"⚠ Failed to load advanced model: {e}")
                    print(f"  Falling back to legacy model...")
        
        # Fall back to legacy model
        # Check if model file exists
        if not Path(self.model_path).exists():
            raise ModelLoadError(
                f"ML model not found at: {self.model_path}",
                "MODEL_FILE_MISSING"
            )
        
        # Check if columns file exists
        if not Path(self.columns_path).exists():
            raise ModelLoadError(
                f"Model columns not found at: {self.columns_path}",
                "MODEL_COLUMNS_FILE_MISSING"
            )
        
        try:
            # Load the trained model
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            
            # Load the model columns
            with open(self.columns_path, "rb") as f:
                self.model_columns = pickle.load(f)
            
            self.model_type = "legacy"
            print(f"✓ Loaded legacy model from {self.model_path}")
                
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load model files: {str(e)}",
                "MODEL_LOAD_FAILED"
            )
    
    def predict_emission(
        self,
        vehicle: dict,
        distance_km: float,
        traffic_level: int
    ) -> float:
        """
        Predict CO2 emission for given parameters.
        
        Prepares input features with one-hot encoding and makes a prediction
        using the loaded ML model.
        
        Args:
            vehicle: Dictionary containing vehicle data (type, fuel, engine_size, mileage)
            distance_km: Distance in kilometers
            traffic_level: Traffic level (1=Low, 2=Medium, 3=High)
            
        Returns:
            float: Predicted CO2 emission in kilograms (rounded to 2 decimal places)
            
        Raises:
            ValueError: If model is not loaded or input data is invalid
            
        Requirements: 3.4, 14.2
        """
        if self.model is None or self.model_columns is None:
            raise ValueError("Model not loaded. Call load() first.")
        
        # Prepare input features
        input_data = self._prepare_input(vehicle, distance_km, traffic_level)
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Reindex to match model columns (fills missing columns with 0)
        input_df = input_df.reindex(columns=self.model_columns, fill_value=0)
        
        # Make prediction
        prediction = self.model.predict(input_df)[0]
        
        # Round to 2 decimal places
        return round(prediction, 2)
    
    def _prepare_input(
        self,
        vehicle: dict,
        distance_km: float,
        traffic_level: int
    ) -> dict:
        """
        Prepare input features for model with one-hot encoding.
        
        Converts vehicle data into the feature format expected by the ML model,
        including one-hot encoding for categorical variables (fuel type and vehicle type).
        
        For advanced models with many features, only populates the features that exist
        in the model_columns list.
        
        Args:
            vehicle: Dictionary containing vehicle data
            distance_km: Distance in kilometers
            traffic_level: Traffic level (1=Low, 2=Medium, 3=High)
            
        Returns:
            dict: Prepared input features
            
        Requirements: 3.4
        """
        # Base numerical features
        input_data = {
            "engine_size": float(vehicle["engine_size"]),
            "mileage": float(vehicle["mileage"]),
            "distance_km": distance_km,
            "traffic_level": traffic_level
        }
        
        # One-hot encode fuel type
        fuel_types = ["Petrol", "Diesel", "CNG"]
        for ft in fuel_types:
            input_data[f"fuel_type_{ft}"] = 1 if vehicle["fuel"] == ft else 0
        
        # One-hot encode vehicle type
        vehicle_types = ["SUV", "Sedan", "Hatchback"]
        for vt in vehicle_types:
            input_data[f"vehicle_type_{vt}"] = 1 if vehicle["type"] == vt else 0
        
        return input_data
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            dict: Model information including type, metadata, and performance metrics
        """
        info = {
            "model_type": self.model_type,
            "model_loaded": self.model is not None,
            "columns_loaded": self.model_columns is not None
        }
        
        if self.model_metadata:
            info["metadata"] = self.model_metadata
        
        return info
