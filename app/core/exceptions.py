"""
Custom exception hierarchy for the Vehicle Emission Eco-Route System.

This module defines all custom exceptions used throughout the application,
providing clear error messages and error codes for different failure scenarios.
"""

from typing import Optional


class VehicleEmissionError(Exception):
    """Base exception for all application errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None):
        """
        Initialize the base exception.
        
        Args:
            message: Human-readable error message
            error_code: Machine-readable error code for API responses
        """
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


class ConfigurationError(VehicleEmissionError):
    """Raised when configuration is invalid or missing required values."""
    
    def __init__(self, message: str, error_code: str = "CONFIGURATION_ERROR"):
        super().__init__(message, error_code)


class ValidationError(VehicleEmissionError):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, error_code: str = "VALIDATION_ERROR"):
        super().__init__(message, error_code)


class VehicleNotFoundError(VehicleEmissionError):
    """Raised when a vehicle lookup fails."""
    
    def __init__(self, vehicle_no: str, suggestions: Optional[list[str]] = None):
        """
        Initialize vehicle not found error.
        
        Args:
            vehicle_no: The vehicle number that was not found
            suggestions: Optional list of example vehicle numbers
        """
        self.vehicle_no = vehicle_no
        self.suggestions = suggestions or []
        message = f"Vehicle {vehicle_no} not found"
        super().__init__(message, "VEHICLE_NOT_FOUND")


class MapsAPIError(VehicleEmissionError):
    """Base exception for Google Maps API errors."""
    
    def __init__(self, message: str, error_code: str = "MAPS_API_ERROR"):
        super().__init__(message, error_code)


class MapsAPITimeoutError(MapsAPIError):
    """Raised when Google Maps API request times out after retries."""
    
    def __init__(self, message: str = "Google Maps API request timed out"):
        super().__init__(message, "MAPS_API_TIMEOUT")


class MapsAPIAuthError(MapsAPIError):
    """Raised when Google Maps API authentication fails."""
    
    def __init__(self, message: str = "Google Maps API authentication failed"):
        super().__init__(message, "MAPS_API_AUTH_ERROR")


class NoRoutesFoundError(VehicleEmissionError):
    """Raised when no routes are available between source and destination."""
    
    def __init__(self, message: str = "No routes found between source and destination"):
        super().__init__(message, "NO_ROUTES_FOUND")


class ModelLoadError(VehicleEmissionError):
    """Raised when ML model files cannot be loaded."""
    
    def __init__(self, message: str, error_code: str = "MODEL_LOAD_ERROR"):
        super().__init__(message, error_code)


class DatabaseLoadError(VehicleEmissionError):
    """Raised when vehicle database cannot be loaded."""
    
    def __init__(self, message: str, error_code: str = "DATABASE_LOAD_ERROR"):
        super().__init__(message, error_code)
