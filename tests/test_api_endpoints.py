"""
Integration tests for all API endpoints.

Tests verify:
- GET / (root endpoint)
- GET /health (health check with component status)
- GET /vehicle/{vehicle_no} (vehicle lookup)
- POST /calculate-emission (emission calculation)
- POST /eco-route (route recommendation with mocked Maps API)

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 11.5, 11.6
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pandas as pd


# Import the FastAPI app
from app.main import app

# Create test client
client = TestClient(app)


class TestRootEndpoint:
    """Tests for GET / endpoint"""
    
    def test_root_returns_frontend(self):
        """Test that root endpoint serves the frontend HTML"""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


class TestHealthEndpoint:
    """Tests for GET /health endpoint"""
    
    def test_health_check_healthy(self):
        """Test health check returns healthy status when all components loaded"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
        assert "components" in data
        
        # Check all components are loaded
        assert data["components"]["database"] == "loaded"
        assert data["components"]["ml_model"] == "loaded"
        assert data["components"]["maps_api"] == "configured"
    
    def test_health_check_includes_version(self):
        """Test health check includes version information"""
        response = client.get("/health")
        data = response.json()
        assert data["version"] == "1.0.0"


class TestVehicleLookupEndpoint:
    """Tests for GET /vehicle/{vehicle_no} endpoint"""
    
    def test_get_vehicle_success(self):
        """Test successful vehicle lookup with valid vehicle number"""
        # Use an actual vehicle number from the database
        response = client.get("/vehicle/MH49LM3287")
        
        assert response.status_code == 200
        data = response.json()
        assert "vehicle_no" in data
        assert "type" in data
        assert "fuel" in data
        assert "engine_size" in data
        assert "mileage" in data
        assert data["vehicle_no"] == "MH49LM3287"
    
    def test_get_vehicle_not_found(self):
        """Test vehicle lookup with non-existent vehicle number"""
        response = client.get("/vehicle/INVALID999")
        assert response.status_code == 404
        
        data = response.json()
        assert "detail" in data
    
    def test_get_vehicle_case_insensitive(self):
        """Test vehicle lookup is case-insensitive"""
        # Test with actual vehicle in different cases
        response1 = client.get("/vehicle/MH49LM3287")
        response2 = client.get("/vehicle/mh49lm3287")
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        # Both should return the same vehicle
        assert response1.json()["vehicle_no"] == response2.json()["vehicle_no"]


class TestEmissionCalculationEndpoint:
    """Tests for POST /calculate-emission endpoint"""
    
    def test_calculate_emission_success(self):
        """Test successful emission calculation with valid data"""
        payload = {
            "vehicle_no": "MH49LM3287",
            "distance_km": 50.0
        }
        
        response = client.post("/calculate-emission", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert data["vehicle_no"] == "MH49LM3287"
        assert data["distance_km"] == 50.0
        assert "predicted_co2_kg" in data
        assert data["predicted_co2_kg"] > 0
        assert data["model_used"] == "RandomForest Regression"
    
    def test_calculate_emission_vehicle_not_found(self):
        """Test emission calculation with non-existent vehicle"""
        payload = {
            "vehicle_no": "INVALID999",
            "distance_km": 50.0
        }
        
        response = client.post("/calculate-emission", json=payload)
        assert response.status_code == 404
        
        data = response.json()
        assert "detail" in data
    
    def test_calculate_emission_invalid_distance_negative(self):
        """Test emission calculation with negative distance"""
        payload = {
            "vehicle_no": "MH49LM3287",  # Use valid vehicle
            "distance_km": -10.0
        }
        
        response = client.post("/calculate-emission", json=payload)
        # Should be rejected by Pydantic validation
        assert response.status_code == 422
        
        data = response.json()
        assert "detail" in data
    
    def test_calculate_emission_invalid_distance_zero(self):
        """Test emission calculation with zero distance"""
        payload = {
            "vehicle_no": "MH49LM3287",  # Use valid vehicle
            "distance_km": 0.0
        }
        
        response = client.post("/calculate-emission", json=payload)
        # Should be rejected by Pydantic validation
        assert response.status_code == 422
        
        data = response.json()
        assert "detail" in data
    
    def test_calculate_emission_invalid_distance_exceeds_max(self):
        """Test emission calculation with distance exceeding maximum"""
        payload = {
            "vehicle_no": "MH49LM3287",  # Use valid vehicle
            "distance_km": 15000.0
        }
        
        response = client.post("/calculate-emission", json=payload)
        # Should be rejected by Pydantic validation
        assert response.status_code == 422
        
        data = response.json()
        assert "detail" in data
    
    def test_calculate_emission_missing_vehicle_no(self):
        """Test emission calculation with missing vehicle_no"""
        payload = {
            "distance_km": 50.0
        }
        
        response = client.post("/calculate-emission", json=payload)
        assert response.status_code == 422
    
    def test_calculate_emission_missing_distance(self):
        """Test emission calculation with missing distance_km"""
        payload = {
            "vehicle_no": "MH49LM3287"
        }
        
        response = client.post("/calculate-emission", json=payload)
        assert response.status_code == 422
    
    def test_calculate_emission_invalid_content_type(self):
        """Test emission calculation with invalid content type"""
        response = client.post(
            "/calculate-emission",
            data="not json",
            headers={"Content-Type": "text/plain"}
        )
        assert response.status_code == 415
        
        data = response.json()
        assert "detail" in data
        assert "application/json" in data["detail"]


class TestEcoRouteEndpoint:
    """Tests for POST /eco-route endpoint with mocked Maps API"""
    
    @patch('app.main.requests.get')
    def test_eco_route_success(self, mock_get):
        """Test successful route recommendation with mocked Maps API"""
        # Mock Google Maps API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "routes": [
                {
                    "legs": [{
                        "distance": {"value": 150000},  # 150 km in meters
                        "duration": {"value": 9000},     # 150 minutes in seconds
                        "duration_in_traffic": {"value": 10800}  # 180 minutes (high traffic)
                    }],
                    "summary": "Via Highway 1"
                },
                {
                    "legs": [{
                        "distance": {"value": 160000},  # 160 km in meters
                        "duration": {"value": 9600},     # 160 minutes in seconds
                        "duration_in_traffic": {"value": 10000}  # 167 minutes (medium traffic)
                    }],
                    "summary": "Via Highway 2"
                }
            ]
        }
        mock_get.return_value = mock_response
        
        payload = {
            "vehicle_no": "MH49LM3287",
            "source": "Mumbai, Maharashtra",
            "destination": "Pune, Maharashtra"
        }
        
        response = client.post("/eco-route", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        
        # Check response structure
        assert data["source"] == "Mumbai, Maharashtra"
        assert data["destination"] == "Pune, Maharashtra"
        assert data["vehicle_no"] == "MH49LM3287"
        
        # Check recommended route
        assert "recommended_route" in data
        assert "route_number" in data["recommended_route"]
        assert "distance_km" in data["recommended_route"]
        assert "duration_minutes" in data["recommended_route"]
        assert "traffic_level" in data["recommended_route"]
        assert "predicted_co2_kg" in data["recommended_route"]
        assert "summary" in data["recommended_route"]
        
        # Check all routes
        assert "all_routes" in data
        assert len(data["all_routes"]) == 2
        
        # Check emission savings
        assert "emission_savings_kg" in data
        assert "emission_savings_percent" in data
        assert data["selection_criteria"] == "Minimum CO2 Emission (ML Predicted)"
        
        # Verify Maps API was called
        mock_get.assert_called_once()
    
    @patch('app.main.requests.get')
    def test_eco_route_no_routes_found(self, mock_get):
        """Test route recommendation when Maps API returns no routes"""
        # Mock Google Maps API response with no routes
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "routes": []
        }
        mock_get.return_value = mock_response
        
        payload = {
            "vehicle_no": "MH49LM3287",
            "source": "Invalid Location 123",
            "destination": "Another Invalid Location 456"
        }
        
        response = client.post("/eco-route", json=payload)
        assert response.status_code == 400
        
        data = response.json()
        assert "detail" in data
        assert "no routes found" in data["detail"].lower()
    
    @patch('app.main.requests.get')
    def test_eco_route_maps_api_timeout(self, mock_get):
        """Test route recommendation when Maps API times out"""
        # Mock timeout exception
        import requests
        mock_get.side_effect = requests.exceptions.Timeout()
        
        payload = {
            "vehicle_no": "MH49LM3287",
            "source": "Mumbai, Maharashtra",
            "destination": "Pune, Maharashtra"
        }
        
        response = client.post("/eco-route", json=payload)
        assert response.status_code == 503
        
        data = response.json()
        assert "detail" in data
        # Check for "timed out" or "timeout" in the message
        assert "timed out" in data["detail"].lower() or "timeout" in data["detail"].lower()
    
    @patch('app.main.requests.get')
    def test_eco_route_maps_api_connection_error(self, mock_get):
        """Test route recommendation when Maps API connection fails"""
        # Mock connection error
        import requests
        mock_get.side_effect = requests.exceptions.ConnectionError()
        
        payload = {
            "vehicle_no": "MH49LM3287",
            "source": "Mumbai, Maharashtra",
            "destination": "Pune, Maharashtra"
        }
        
        response = client.post("/eco-route", json=payload)
        assert response.status_code == 503
        
        data = response.json()
        assert "detail" in data
    
    def test_eco_route_vehicle_not_found(self):
        """Test route recommendation with non-existent vehicle"""
        payload = {
            "vehicle_no": "INVALID999",
            "source": "Mumbai, Maharashtra",
            "destination": "Pune, Maharashtra"
        }
        
        response = client.post("/eco-route", json=payload)
        assert response.status_code == 404
        
        data = response.json()
        assert "detail" in data
    
    def test_eco_route_empty_source(self):
        """Test route recommendation with empty source after trimming"""
        payload = {
            "vehicle_no": "MH49LM3287",
            "source": "   ",  # Whitespace only - will be trimmed to empty
            "destination": "Pune, Maharashtra"
        }
        
        response = client.post("/eco-route", json=payload)
        # Should return error (either 422 validation or 400 bad request)
        assert response.status_code in [400, 422]
    
    def test_eco_route_empty_destination(self):
        """Test route recommendation with empty destination after trimming"""
        payload = {
            "vehicle_no": "MH49LM3287",
            "source": "Mumbai, Maharashtra",
            "destination": "   "  # Whitespace only - will be trimmed to empty
        }
        
        response = client.post("/eco-route", json=payload)
        # Should return error (either 422 validation or 400 bad request)
        assert response.status_code in [400, 422]
    
    def test_eco_route_missing_fields(self):
        """Test route recommendation with missing required fields"""
        payload = {
            "vehicle_no": "MH49LM3287"
        }
        
        response = client.post("/eco-route", json=payload)
        assert response.status_code == 422
    
    @patch('app.main.requests.get')
    def test_eco_route_traffic_level_calculation(self, mock_get):
        """Test that traffic levels are correctly calculated and included"""
        # Mock response with traffic data
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "routes": [
                {
                    "legs": [{
                        "distance": {"value": 100000},
                        "duration": {"value": 6000},
                        "duration_in_traffic": {"value": 6300}  # 5% increase = Low traffic
                    }],
                    "summary": "Low traffic route"
                }
            ]
        }
        mock_get.return_value = mock_response
        
        payload = {
            "vehicle_no": "MH49LM3287",
            "source": "Location A",
            "destination": "Location B"
        }
        
        response = client.post("/eco-route", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert data["recommended_route"]["traffic_level"] in ["Low", "Medium", "High"]


class TestSecurityHeaders:
    """Tests for security headers in responses"""
    
    def test_security_headers_present(self):
        """Test that security headers are present in responses"""
        response = client.get("/health")
        
        # Check security headers
        assert "x-content-type-options" in response.headers
        assert response.headers["x-content-type-options"] == "nosniff"
        
        assert "x-frame-options" in response.headers
        assert response.headers["x-frame-options"] == "DENY"
        
        assert "content-security-policy" in response.headers


class TestCORS:
    """Tests for CORS configuration"""
    
    def test_cors_headers_present(self):
        """Test that CORS headers are present in responses"""
        response = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET"
            }
        )
        
        # CORS headers should be present
        assert "access-control-allow-origin" in response.headers


class TestErrorResponseFormat:
    """Tests for error response format consistency"""
    
    def test_error_response_has_detail(self):
        """Test that error responses include detail field"""
        response = client.get("/vehicle/INVALID999")
        assert response.status_code == 404
        
        data = response.json()
        assert "detail" in data
    
    def test_validation_error_format(self):
        """Test that validation errors have proper format"""
        payload = {
            "vehicle_no": "MH49LM3287",  # Use valid vehicle
            "distance_km": -10.0  # Invalid distance
        }
        
        response = client.post("/calculate-emission", json=payload)
        assert response.status_code == 422
        
        data = response.json()
        assert "detail" in data
        # Pydantic validation errors include error_code and timestamp
        assert "error_code" in data
        assert "timestamp" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
