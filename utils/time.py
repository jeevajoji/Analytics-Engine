"""
Time and duration utility functions.

Centralized time parsing and manipulation utilities.
"""

import re
from typing import Optional
from datetime import timedelta


# Duration pattern: number followed by unit (e.g., "5m", "1h", "30s")
DURATION_PATTERN = re.compile(r'^(\d+(?:\.\d+)?)\s*([smhdw])$', re.IGNORECASE)

# Duration unit multipliers (to seconds)
DURATION_UNITS = {
    's': 1,           # seconds
    'm': 60,          # minutes
    'h': 3600,        # hours
    'd': 86400,       # days
    'w': 604800,      # weeks
}


def parse_duration(duration_str: str) -> timedelta:
    """
    Parse a duration string into a timedelta.
    
    Supports formats like:
    - "30s" - 30 seconds
    - "5m" - 5 minutes
    - "1h" - 1 hour
    - "2d" - 2 days
    - "1w" - 1 week
    - "1.5h" - 1.5 hours (90 minutes)
    
    Args:
        duration_str: Duration string (e.g., "5m", "1h", "30s")
        
    Returns:
        timedelta representing the duration
        
    Raises:
        ValueError: If the duration string is invalid
        
    Example:
        >>> parse_duration("5m")
        datetime.timedelta(seconds=300)
        >>> parse_duration("1.5h")
        datetime.timedelta(seconds=5400)
    """
    if not duration_str:
        raise ValueError("Duration string cannot be empty")
    
    duration_str = duration_str.strip()
    
    match = DURATION_PATTERN.match(duration_str)
    if not match:
        raise ValueError(
            f"Invalid duration format: '{duration_str}'. "
            f"Expected format like '5m', '1h', '30s', '2d', '1w'"
        )
    
    value = float(match.group(1))
    unit = match.group(2).lower()
    
    seconds = value * DURATION_UNITS[unit]
    return timedelta(seconds=seconds)


def parse_duration_seconds(duration_str: str) -> float:
    """
    Parse a duration string and return total seconds.
    
    Args:
        duration_str: Duration string (e.g., "5m", "1h")
        
    Returns:
        Total seconds as float
    """
    return parse_duration(duration_str).total_seconds()


def format_duration(td: timedelta) -> str:
    """
    Format a timedelta as a human-readable duration string.
    
    Args:
        td: timedelta to format
        
    Returns:
        Human-readable duration string
        
    Example:
        >>> format_duration(timedelta(hours=2, minutes=30))
        "2h 30m"
    """
    total_seconds = int(td.total_seconds())
    
    if total_seconds == 0:
        return "0s"
    
    parts = []
    
    weeks, remainder = divmod(total_seconds, 604800)
    if weeks:
        parts.append(f"{weeks}w")
    
    days, remainder = divmod(remainder, 86400)
    if days:
        parts.append(f"{days}d")
    
    hours, remainder = divmod(remainder, 3600)
    if hours:
        parts.append(f"{hours}h")
    
    minutes, seconds = divmod(remainder, 60)
    if minutes:
        parts.append(f"{minutes}m")
    
    if seconds:
        parts.append(f"{seconds}s")
    
    return " ".join(parts)


def duration_to_polars_interval(duration_str: str) -> str:
    """
    Convert a duration string to Polars interval format.
    
    Polars uses formats like "5m", "1h", "1d" which mostly match,
    but this ensures compatibility.
    
    Args:
        duration_str: Duration string (e.g., "5m", "1h")
        
    Returns:
        Polars-compatible interval string
    """
    match = DURATION_PATTERN.match(duration_str.strip())
    if not match:
        raise ValueError(f"Invalid duration format: '{duration_str}'")
    
    value = match.group(1)
    unit = match.group(2).lower()
    
    # Polars interval mapping
    polars_units = {
        's': 's',
        'm': 'm',
        'h': 'h',
        'd': 'd',
        'w': 'w',
    }
    
    return f"{value}{polars_units[unit]}"
