"""
Traffic analysis module for the Vehicle Emission Eco-Route System.

This module provides traffic level calculation based on route duration data,
comparing expected vs actual duration to classify traffic conditions.

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5
"""

from typing import Optional


class TrafficAnalyzer:
    """
    Analyzes traffic conditions from route data.
    
    Calculates traffic levels by comparing expected duration with actual
    duration in traffic, using percentage increase thresholds.
    
    Requirements: 6.1, 6.2, 6.3, 6.4, 6.5
    """
    
    # Traffic level constants
    TRAFFIC_LOW = 1
    TRAFFIC_MEDIUM = 2
    TRAFFIC_HIGH = 3
    
    @staticmethod
    def calculate_traffic_level(
        duration_seconds: int,
        duration_in_traffic: Optional[int] = None
    ) -> int:
        """
        Calculate traffic level based on duration comparison.
        
        Uses enhanced algorithm comparing expected vs actual duration:
        - Low (1): actual within 10% of expected
        - Medium (2): actual 10-30% above expected
        - High (3): actual >30% above expected
        - Default to Medium when traffic data unavailable
        
        Args:
            duration_seconds: Expected duration without traffic (in seconds)
            duration_in_traffic: Actual duration with traffic (in seconds), optional
            
        Returns:
            int: Traffic level (1=Low, 2=Medium, 3=High)
            
        Requirements: 6.1, 6.2, 6.3, 6.4, 6.5
        """
        # Default to medium traffic when data unavailable
        if duration_in_traffic is None:
            return TrafficAnalyzer.TRAFFIC_MEDIUM
        
        # Handle edge case of zero duration
        if duration_seconds == 0:
            return TrafficAnalyzer.TRAFFIC_MEDIUM
        
        # Calculate percentage increase
        increase_ratio = (duration_in_traffic - duration_seconds) / duration_seconds
        
        # Classify traffic based on increase ratio
        if increase_ratio <= 0.10:
            # Within 10% of expected duration - Low traffic
            return TrafficAnalyzer.TRAFFIC_LOW
        elif increase_ratio <= 0.30:
            # 10-30% above expected duration - Medium traffic
            return TrafficAnalyzer.TRAFFIC_MEDIUM
        else:
            # More than 30% above expected duration - High traffic
            return TrafficAnalyzer.TRAFFIC_HIGH
    
    @staticmethod
    def get_traffic_label(traffic_level: int) -> str:
        """
        Get human-readable traffic label.
        
        Converts numeric traffic level to descriptive string.
        
        Args:
            traffic_level: Numeric traffic level (1, 2, or 3)
            
        Returns:
            str: Human-readable label ("Low", "Medium", "High", or "Unknown")
            
        Requirements: 6.1
        """
        labels = {
            TrafficAnalyzer.TRAFFIC_LOW: "Low",
            TrafficAnalyzer.TRAFFIC_MEDIUM: "Medium",
            TrafficAnalyzer.TRAFFIC_HIGH: "High"
        }
        return labels.get(traffic_level, "Unknown")
