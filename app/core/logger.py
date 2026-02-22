"""
Structured logging system for Vehicle Emission Eco-Route System.
Provides JSON and text logging with sensitive data sanitization.
"""

import logging
import json
from datetime import datetime
from typing import Any, Optional, Dict


class StructuredLogger:
    """Structured logging with JSON output support"""
    
    def __init__(self, name: str, log_level: str = "INFO", log_format: str = "json", log_file: str = "app.log"):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name (typically module or component name)
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_format: Output format ("json" or "text")
            log_file: Path to log file
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        self.log_format = log_format
        
        # Prevent duplicate handlers if logger already configured
        if not self.logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(self._get_formatter())
            self.logger.addHandler(console_handler)
            
            # File handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(self._get_formatter())
            self.logger.addHandler(file_handler)
    
    def _get_formatter(self) -> logging.Formatter:
        """Get appropriate formatter based on log format"""
        if self.log_format == "json":
            return JsonFormatter()
        return logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def log_request(self, method: str, endpoint: str, params: Dict[str, Any]) -> None:
        """
        Log API request.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            params: Request parameters (will be sanitized)
        """
        self.logger.info("API Request", extra={
            "event": "api_request",
            "method": method,
            "endpoint": endpoint,
            "params": self._sanitize_params(params)
        })
    
    def log_ml_prediction(self, vehicle_no: str, distance: float, prediction: float) -> None:
        """
        Log ML prediction.
        
        Args:
            vehicle_no: Vehicle identification number
            distance: Distance in kilometers
            prediction: Predicted CO2 emission in kg
        """
        self.logger.info("ML Prediction", extra={
            "event": "ml_prediction",
            "vehicle_no": vehicle_no,
            "distance_km": distance,
            "predicted_co2_kg": prediction
        })
    
    def log_maps_api_call(self, source: str, destination: str, status: str) -> None:
        """
        Log Google Maps API call.
        
        Args:
            source: Source location
            destination: Destination location
            status: Call status (success, error, timeout, etc.)
        """
        self.logger.info("Maps API Call", extra={
            "event": "maps_api_call",
            "source": source,
            "destination": destination,
            "status": status
        })
    
    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Log error with context.
        
        Args:
            error: Exception that occurred
            context: Additional context information (optional)
        """
        self.logger.error("Error occurred", extra={
            "event": "error",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {}
        }, exc_info=True)
    
    def log_startup(self, config_summary: Dict[str, Any]) -> None:
        """
        Log system startup.
        
        Args:
            config_summary: Configuration summary (will be sanitized)
        """
        self.logger.info("System starting", extra={
            "event": "startup",
            "config": self._sanitize_config(config_summary)
        })
    
    @staticmethod
    def _sanitize_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove sensitive data from params.
        
        Args:
            params: Parameters dictionary
            
        Returns:
            Sanitized parameters dictionary
        """
        sanitized = params.copy()
        sensitive_keys = ["api_key", "password", "token", "secret"]
        for key in sensitive_keys:
            if key in sanitized:
                sanitized[key] = "***REDACTED***"
        return sanitized
    
    @staticmethod
    def _sanitize_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove sensitive data from config.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Sanitized configuration dictionary
        """
        sanitized = config.copy()
        if "google_maps_api_key" in sanitized:
            sanitized["google_maps_api_key"] = "***REDACTED***"
        return sanitized


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.
        
        Args:
            record: Log record to format
            
        Returns:
            JSON-formatted log string
        """
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add extra fields if present
        if hasattr(record, 'event'):
            log_data.update(record.__dict__)
            # Remove standard logging fields to keep JSON clean
            for key in ['name', 'msg', 'args', 'created', 'filename', 'funcName',
                       'levelname', 'levelno', 'lineno', 'module', 'msecs',
                       'pathname', 'process', 'processName', 'relativeCreated',
                       'thread', 'threadName', 'exc_info', 'exc_text', 'stack_info']:
                log_data.pop(key, None)
        
        return json.dumps(log_data)
