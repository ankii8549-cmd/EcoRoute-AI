"""
Input validation utilities for the Vehicle Emission Eco-Route System.

This module provides additional validation functions beyond Pydantic models,
including vehicle existence checking and error message sanitization.
"""

import re
import pandas as pd
from typing import Optional


class InputValidator:
    """
    Additional validation utilities for input processing.
    
    Provides static methods for vehicle validation and error message sanitization.
    """
    
    @staticmethod
    def validate_vehicle_exists(vehicle_no: str, vehicles_df: pd.DataFrame) -> bool:
        """
        Check if a vehicle exists in the database.
        
        Args:
            vehicle_no: Vehicle identification number (will be normalized to uppercase)
            vehicles_df: DataFrame containing vehicle database
            
        Returns:
            bool: True if vehicle exists, False otherwise
            
        Requirements: 2.5, 13.2
        """
        # Normalize vehicle number to uppercase for case-insensitive matching
        normalized_vehicle_no = vehicle_no.upper().strip()
        
        # Check if vehicle exists in the database
        return not vehicles_df[vehicles_df["vehicle_no"] == normalized_vehicle_no].empty
    
    @staticmethod
    def sanitize_error_message(error: str, environment: str) -> str:
        """
        Sanitize error messages based on environment.
        
        In production, removes internal details to prevent information leakage.
        In development, returns the full error message for debugging.
        
        Args:
            error: The original error message
            environment: The current environment ("production" or "development")
            
        Returns:
            str: Sanitized error message
            
        Requirements: 13.2, 15.5, 15.6
        """
        if environment == "production":
            # Remove internal details in production
            return "An error occurred. Please try again."
        
        # Return full error message in development
        return error
    
    @staticmethod
    def sanitize_input(input_str: str) -> str:
        """
        Sanitize user input by removing dangerous characters.
        
        Removes characters that could be used for injection attacks:
        - SQL injection characters: ', ", ;, --, /*
        - Script injection characters: <, >, &
        - Command injection characters: |, &, $, `, \
        
        Args:
            input_str: The input string to sanitize
            
        Returns:
            str: Sanitized input string
            
        Requirements: 2.5
        """
        if not input_str:
            return input_str
        
        # Remove potentially dangerous characters for injection attacks
        # Keep alphanumeric, spaces, hyphens, underscores, periods, commas
        dangerous_chars = r'[\'";`<>|&$\\]'
        sanitized = re.sub(dangerous_chars, '', input_str)
        
        # Remove SQL comment sequences
        sanitized = re.sub(r'--', '', sanitized)
        sanitized = re.sub(r'/\*', '', sanitized)
        sanitized = re.sub(r'\*/', '', sanitized)
        
        return sanitized.strip()
