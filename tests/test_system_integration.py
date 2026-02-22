"""
End-to-end system integration tests.

Tests verify:
- Configuration loading and validation
- Vehicle database loading
- ML model loading
- Complete workflow: vehicle lookup → emission calculation → route recommendation
- Error handling scenarios

Requirements: 9.3, 9.8
"""

import pytest
import os
import tempfile
import pickle
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import components
from app.config import Settings, get_settings, validate_startup_config
from app.services.vehicle_manager import VehicleManager
from app.services.model_manager import EmissionModelManager
from app.services.traffic_analyzer import TrafficAnalyzer
from app.services.maps_service import MapsService
from app.core.exceptions import (
    ConfigurationError,
    VehicleNotFoundError,
    MapsAPIError,
    MapsAPITimeoutError,
    ModelLoadError,
    DatabaseLoadError
)


class TestConfigurationLoading:
    """Tests for configuration loading and validation"""
    
    def test_settings_load_from_env(self):
        """Test that settings load from environment variables"""
        settings = get_settings()
        
        assert settings is not None
        assert hasattr(settings, 'google_maps_api_key')
        assert hasattr(settings, 'vehicle_database_path')
        assert hasattr(settings, 'model_path')
        assert hasattr(settings, 'model_columns_path')
    
    def test_settings_have_defaults(self):
        """Test that settings have sensible defaults"""
        settings = get_settings()
        
        assert settings.api_host == "0.0.0.0"
        assert settings.api_port == 8000
        assert settings.rate_limit_requests == 100
        assert settings.rate_limit_period == 60
        assert settings.route_cache_ttl == 300
        assert settings.log_level == "INFO"
    
    def test_validate_startup_config_success(self):
        """Test startup validation with valid configuration"""
        settings = get_settings()
        
        # Should not raise any exceptions
        try:
            validate_startup_config(settings)
        except Exception as e:
            pytest.fail(f"Startup validation failed unexpectedly: {e}")
    
    def test_validate_startup_config_missing_api_key(self):
        """Test startup validation fails with missing API key"""
        with patch('app.config.Settings') as mock_settings:
            mock_instance = MagicMock()
            mock_instance.google_maps_api_key = ""
            mock_instance.vehicle_database_path = "data/vehicle_database_10000.csv"
            mock_instance.model_path = "data/emission_model.pkl"
            mock_instance.model_columns_path = "data/model_columns.pkl"
            mock_instance.api_port = 8000
            mock_instance.rate_limit_requests = 100
            mock_instance.rate_limit_period = 60
            mock_instance.route_cache_ttl = 300
            mock_instance.maps_api_timeout = 10
            mock_instance.maps_api_retries = 3
            
            with pytest.raises(ValueError, match="GOOGLE_MAPS_API_KEY"):
                validate_startup_config(mock_instance)
    
    def test_validate_startup_config_missing_database_file(self):
        """Test startup validation fails with missing database file"""
        with patch('app.config.Settings') as mock_settings:
            mock_instance = MagicMock()
            mock_instance.google_maps_api_key = "valid_key_123"
            mock_instance.vehicle_database_path = "nonexistent_database.csv"
            mock_instance.model_path = "data/emission_model.pkl"
            mock_instance.model_columns_path = "data/model_columns.pkl"
            mock_instance.api_port = 8000
            mock_instance.rate_limit_requests = 100
            mock_instance.rate_limit_period = 60
            mock_instance.route_cache_ttl = 300
            mock_instance.maps_api_timeout = 10
            mock_instance.maps_api_retries = 3
            
            with pytest.raises(FileNotFoundError, match="Vehicle database"):
                validate_startup_config(mock_instance)


class TestVehicleDatabaseLoading:
    """Tests for vehicle database loading and operations"""
    
    def test_vehicle_manager_loads_database(self):
        """Test that VehicleManager successfully loads the database"""
        manager = VehicleManager("data/vehicle_database_10000.csv")
        manager.load()
        
        assert manager.vehicles_df is not None
        assert not manager.vehicles_df.empty
        assert len(manager.vehicle_index) > 0
    
    def test_vehicle_manager_creates_index(self):
        """Test that VehicleManager creates an index for O(1) lookup"""
        manager = VehicleManager("data/vehicle_database_10000.csv")
        manager.load()
        
        # Index should be a dictionary
        assert isinstance(manager.vehicle_index, dict)
        
        # Index should have entries
        assert len(manager.vehicle_index) > 0
    
    def test_vehicle_manager_lookup_performance(self):
        """Test that vehicle lookup is O(1) using dictionary"""
        manager = VehicleManager("data/vehicle_database_10000.csv")
        manager.load()
        
        # Get first vehicle number
        first_vehicle_no = list(manager.vehicle_index.keys())[0]
        
        # Lookup should be instant (O(1))
        vehicle = manager.get_vehicle(first_vehicle_no)
        assert vehicle is not None
        assert vehicle["vehicle_no"] == first_vehicle_no
    
    def test_vehicle_manager_case_insensitive_lookup(self):
        """Test that vehicle lookup is case-insensitive"""
        manager = VehicleManager("data/vehicle_database_10000.csv")
        manager.load()
        
        # Get first vehicle number
        first_vehicle_no = list(manager.vehicle_index.keys())[0]
        
        # Test with different cases
        vehicle_upper = manager.get_vehicle(first_vehicle_no.upper())
        vehicle_lower = manager.get_vehicle(first_vehicle_no.lower())
        
        assert vehicle_upper is not None
        assert vehicle_lower is not None
        assert vehicle_upper["vehicle_no"] == vehicle_lower["vehicle_no"]
    
    def test_vehicle_manager_whitespace_tolerance(self):
        """Test that vehicle lookup handles whitespace"""
        manager = VehicleManager("data/vehicle_database_10000.csv")
        manager.load()
        
        # Get first vehicle number
        first_vehicle_no = list(manager.vehicle_index.keys())[0]
        
        # Test with whitespace
        vehicle = manager.get_vehicle(f"  {first_vehicle_no}  ")
        assert vehicle is not None
        assert vehicle["vehicle_no"] == first_vehicle_no
    
    def test_vehicle_manager_nonexistent_vehicle(self):
        """Test that lookup returns None for nonexistent vehicle"""
        manager = VehicleManager("data/vehicle_database_10000.csv")
        manager.load()
        
        vehicle = manager.get_vehicle("NONEXISTENT999")
        assert vehicle is None
    
    def test_vehicle_manager_get_sample_vehicles(self):
        """Test that sample vehicles can be retrieved"""
        manager = VehicleManager("data/vehicle_database_10000.csv")
        manager.load()
        
        samples = manager.get_sample_vehicles(5)
        assert len(samples) == 5
        assert all(isinstance(v, str) for v in samples)


class TestMLModelLoading:
    """Tests for ML model loading and predictions"""
    
    def test_model_manager_loads_model(self):
        """Test that EmissionModelManager successfully loads the model"""
        manager = EmissionModelManager("data/emission_model.pkl", "data/model_columns.pkl")
        manager.load()
        
        assert manager.model is not None
        assert manager.model_columns is not None
    
    def test_model_manager_prediction(self):
        """Test that model can make predictions"""
        manager = EmissionModelManager("data/emission_model.pkl", "data/model_columns.pkl")
        manager.load()
        
        # Create sample vehicle data
        vehicle = {
            "vehicle_no": "TEST001",
            "type": "Sedan",
            "fuel": "Petrol",
            "engine_size": 1.5,
            "mileage": 18.5
        }
        
        # Make prediction
        prediction = manager.predict_emission(vehicle, 50.0, 2)
        
        assert isinstance(prediction, float)
        assert prediction > 0
    
    def test_model_manager_prediction_consistency(self):
        """Test that model produces consistent predictions for same input"""
        manager = EmissionModelManager("data/emission_model.pkl", "data/model_columns.pkl")
        manager.load()
        
        vehicle = {
            "vehicle_no": "TEST001",
            "type": "Sedan",
            "fuel": "Petrol",
            "engine_size": 1.5,
            "mileage": 18.5
        }
        
        # Make multiple predictions with same input
        prediction1 = manager.predict_emission(vehicle, 50.0, 2)
        prediction2 = manager.predict_emission(vehicle, 50.0, 2)
        
        assert prediction1 == prediction2
    
    def test_model_manager_different_traffic_levels(self):
        """Test that model produces different predictions for different traffic levels"""
        manager = EmissionModelManager("data/emission_model.pkl", "data/model_columns.pkl")
        manager.load()
        
        vehicle = {
            "vehicle_no": "TEST001",
            "type": "Sedan",
            "fuel": "Petrol",
            "engine_size": 1.5,
            "mileage": 18.5
        }
        
        # Predictions with different traffic levels
        prediction_low = manager.predict_emission(vehicle, 50.0, 1)
        prediction_medium = manager.predict_emission(vehicle, 50.0, 2)
        prediction_high = manager.predict_emission(vehicle, 50.0, 3)
        
        # Higher traffic should generally result in higher emissions
        assert prediction_low <= prediction_medium <= prediction_high


class TestTrafficAnalyzer:
    """Tests for traffic level calculation"""
    
    def test_traffic_analyzer_low_traffic(self):
        """Test traffic classification for low traffic (≤10% increase)"""
        # 5% increase
        traffic_level = TrafficAnalyzer.calculate_traffic_level(6000, 6300)
        assert traffic_level == TrafficAnalyzer.TRAFFIC_LOW
        assert TrafficAnalyzer.get_traffic_label(traffic_level) == "Low"
    
    def test_traffic_analyzer_medium_traffic(self):
        """Test traffic classification for medium traffic (10-30% increase)"""
        # 20% increase
        traffic_level = TrafficAnalyzer.calculate_traffic_level(6000, 7200)
        assert traffic_level == TrafficAnalyzer.TRAFFIC_MEDIUM
        assert TrafficAnalyzer.get_traffic_label(traffic_level) == "Medium"
    
    def test_traffic_analyzer_high_traffic(self):
        """Test traffic classification for high traffic (>30% increase)"""
        # 40% increase
        traffic_level = TrafficAnalyzer.calculate_traffic_level(6000, 8400)
        assert traffic_level == TrafficAnalyzer.TRAFFIC_HIGH
        assert TrafficAnalyzer.get_traffic_label(traffic_level) == "High"
    
    def test_traffic_analyzer_no_traffic_data(self):
        """Test traffic classification defaults to medium when no data"""
        traffic_level = TrafficAnalyzer.calculate_traffic_level(6000, None)
        assert traffic_level == TrafficAnalyzer.TRAFFIC_MEDIUM
    
    def test_traffic_analyzer_boundary_10_percent(self):
        """Test traffic classification at 10% boundary"""
        # Exactly 10% increase
        traffic_level = TrafficAnalyzer.calculate_traffic_level(6000, 6600)
        assert traffic_level == TrafficAnalyzer.TRAFFIC_LOW
    
    def test_traffic_analyzer_boundary_30_percent(self):
        """Test traffic classification at 30% boundary"""
        # Exactly 30% increase
        traffic_level = TrafficAnalyzer.calculate_traffic_level(6000, 7800)
        assert traffic_level == TrafficAnalyzer.TRAFFIC_MEDIUM


class TestMapsService:
    """Tests for Google Maps API integration"""
    
    @patch('app.services.maps_service.requests.get')
    def test_maps_service_successful_request(self, mock_get):
        """Test successful Maps API request"""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "routes": [
                {
                    "legs": [{
                        "distance": {"value": 150000},
                        "duration": {"value": 9000}
                    }],
                    "summary": "Test Route"
                }
            ]
        }
        mock_get.return_value = mock_response
        
        service = MapsService("test_api_key")
        result = service.get_routes("Location A", "Location B")
        
        assert result is not None
        assert "routes" in result
        assert len(result["routes"]) > 0
    
    @patch('app.services.maps_service.requests.get')
    def test_maps_service_parse_routes(self, mock_get):
        """Test parsing of Maps API response"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "routes": [
                {
                    "legs": [{
                        "distance": {"value": 150000},
                        "duration": {"value": 9000},
                        "duration_in_traffic": {"value": 10800}
                    }],
                    "summary": "Test Route"
                }
            ]
        }
        mock_get.return_value = mock_response
        
        service = MapsService("test_api_key")
        api_response = service.get_routes("Location A", "Location B")
        parsed_routes = service.parse_routes(api_response)
        
        assert len(parsed_routes) == 1
        assert parsed_routes[0]["distance_km"] == 150.0
        assert parsed_routes[0]["duration_minutes"] == 150.0
        assert "duration_in_traffic" in parsed_routes[0]
    
    @patch('app.services.maps_service.requests.get')
    def test_maps_service_cache_behavior(self, mock_get):
        """Test that Maps API responses are cached"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"routes": []}
        mock_get.return_value = mock_response
        
        service = MapsService("test_api_key")
        
        # First request
        service.get_routes("Location A", "Location B")
        
        # Second request with same parameters
        service.get_routes("Location A", "Location B")
        
        # Should only call API once due to caching
        assert mock_get.call_count == 1


class TestCompleteWorkflow:
    """End-to-end workflow tests"""
    
    @patch('app.services.maps_service.requests.get')
    def test_complete_emission_calculation_workflow(self, mock_get):
        """Test complete workflow: vehicle lookup → emission calculation"""
        # Load vehicle database
        vehicle_manager = VehicleManager("data/vehicle_database_10000.csv")
        vehicle_manager.load()
        
        # Load ML model
        model_manager = EmissionModelManager("data/emission_model.pkl", "data/model_columns.pkl")
        model_manager.load()
        
        # Get a vehicle
        first_vehicle_no = list(vehicle_manager.vehicle_index.keys())[0]
        vehicle = vehicle_manager.get_vehicle(first_vehicle_no)
        
        assert vehicle is not None
        
        # Calculate emission
        prediction = model_manager.predict_emission(vehicle, 50.0, 2)
        
        assert isinstance(prediction, float)
        assert prediction > 0
    
    @patch('app.services.maps_service.requests.get')
    def test_complete_route_recommendation_workflow(self, mock_get):
        """Test complete workflow: vehicle lookup → Maps API → traffic analysis → emission prediction → route selection"""
        # Mock Maps API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "routes": [
                {
                    "legs": [{
                        "distance": {"value": 150000},
                        "duration": {"value": 9000},
                        "duration_in_traffic": {"value": 10800}
                    }],
                    "summary": "Route 1"
                },
                {
                    "legs": [{
                        "distance": {"value": 160000},
                        "duration": {"value": 9600},
                        "duration_in_traffic": {"value": 10000}
                    }],
                    "summary": "Route 2"
                }
            ]
        }
        mock_get.return_value = mock_response
        
        # Load components
        vehicle_manager = VehicleManager("data/vehicle_database_10000.csv")
        vehicle_manager.load()
        
        model_manager = EmissionModelManager("data/emission_model.pkl", "data/model_columns.pkl")
        model_manager.load()
        
        maps_service = MapsService("test_api_key")
        
        # Get vehicle
        first_vehicle_no = list(vehicle_manager.vehicle_index.keys())[0]
        vehicle = vehicle_manager.get_vehicle(first_vehicle_no)
        
        # Get routes
        api_response = maps_service.get_routes("Location A", "Location B")
        routes = maps_service.parse_routes(api_response)
        
        assert len(routes) == 2
        
        # Calculate emissions for each route
        route_emissions = []
        for route in routes:
            traffic_level = TrafficAnalyzer.calculate_traffic_level(
                route["duration_seconds"],
                route.get("duration_in_traffic")
            )
            
            emission = model_manager.predict_emission(
                vehicle,
                route["distance_km"],
                traffic_level
            )
            
            route_emissions.append({
                "route": route,
                "emission": emission,
                "traffic_level": traffic_level
            })
        
        # Select best route (lowest emission)
        best_route = min(route_emissions, key=lambda x: x["emission"])
        
        assert best_route is not None
        assert best_route["emission"] > 0


class TestErrorHandling:
    """Tests for error handling scenarios"""
    
    def test_vehicle_manager_missing_database_file(self):
        """Test VehicleManager handles missing database file"""
        manager = VehicleManager("nonexistent_database.csv")
        
        with pytest.raises(FileNotFoundError):
            manager.load()
    
    def test_model_manager_missing_model_file(self):
        """Test EmissionModelManager handles missing model file"""
        manager = EmissionModelManager("nonexistent_model.pkl", "data/model_columns.pkl")
        
        with pytest.raises(FileNotFoundError):
            manager.load()
    
    def test_model_manager_missing_columns_file(self):
        """Test EmissionModelManager handles missing columns file"""
        manager = EmissionModelManager("data/emission_model.pkl", "nonexistent_columns.pkl")
        
        with pytest.raises(FileNotFoundError):
            manager.load()
    
    @patch('app.services.maps_service.requests.get')
    def test_maps_service_timeout_handling(self, mock_get):
        """Test MapsService handles timeout errors"""
        import requests
        mock_get.side_effect = requests.exceptions.Timeout()
        
        service = MapsService("test_api_key", timeout=1, max_retries=1)
        
        with pytest.raises(Exception):  # Should raise after retries exhausted
            service.get_routes("Location A", "Location B")
    
    @patch('app.services.maps_service.requests.get')
    def test_maps_service_http_error_handling(self, mock_get):
        """Test MapsService handles HTTP errors"""
        import requests
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError()
        mock_get.return_value = mock_response
        
        service = MapsService("invalid_api_key")
        
        with pytest.raises(Exception):
            service.get_routes("Location A", "Location B")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
