import re
from typing import Union, Tuple, Optional, List, Dict


class TimeUnitConverter:
    """
    A utility class to convert between different time units and format time strings.
    Supports conversion from any supported time unit to human-readable format.
    Author: Santa Andria (santa.andria@dicea.unipd.it)
    """

    # Time unit conversions to minutes
    TIME_UNITS = {
        "min": 1,
        "t": 1,  # minute
        "h": 60,  # hour
        "d": 24 * 60,  # day
        "w": 7 * 24 * 60,  # week
        "mo": 30 * 24 * 60,  # month (approximate)
        "y": 365 * 24 * 60,  # year (non-leap)
    }

    # Thresholds for unit selection (in minutes)
    THRESHOLDS = {
        "min": 1,
        "hour": 60,
        "day": 24 * 60,
        "month": 30 * 24 * 60,
        "year": 365 * 24 * 60,
    }

    # Output format mapping
    OUTPUT_FORMATS = {
        "min": ("mn", 1),
        "hour": ("h", 60),
        "day": ("d", 24 * 60),
        "month": ("mo", 30 * 24 * 60),
        "year": ("y", 365 * 24 * 60),
    }

    @classmethod
    def parse_frequency(cls, freq: str) -> Tuple[float, str]:
        """
        Parse a frequency string into value and unit components.

        Args:
            freq (str): Frequency string (e.g., '5min', '1H', '2D')

        Returns:
            Tuple[float, str]: (value, unit)

        Raises:
            ValueError: If frequency format is invalid
        """
        match = re.match(r"(\d*\.?\d*)?\s*([A-Za-z]+)", freq)
        if not match:
            raise ValueError(f"Invalid frequency format: {freq}")

        value = float(match.group(1)) if match.group(1) else 1
        unit = match.group(2).lower().rstrip("s")  # handle plurals

        if unit not in cls.TIME_UNITS:
            raise ValueError(
                f"Unsupported frequency unit. Must be one of: {', '.join(cls.TIME_UNITS.keys())}"
            )

        return value, unit

    @classmethod
    def minutes_to_readable(cls, total_minutes: float, precision: int = 1) -> str:
        """
        Convert minutes to the most appropriate readable time string based on thresholds.

        Args:
            total_minutes (float): Time in minutes
            precision (int): Number of decimal places in output

        Returns:
            str: Formatted time string with appropriate unit
        """
        for threshold_unit, (output_suffix, divisor) in sorted(
            cls.OUTPUT_FORMATS.items(), key=lambda x: x[1][1], reverse=True
        ):
            if total_minutes >= cls.THRESHOLDS[threshold_unit]:
                return f"{(total_minutes/divisor):.{precision}f}{output_suffix}"

        return f"{total_minutes:.{precision}f}mn"

    @classmethod
    def index_to_time_str(cls, di: float, freq: str, precision: int = 1) -> str:
        """
        Convert an index increment to human-readable time string.

        Args:
            di (float): Index increment/shift
            freq (str): Frequency string in pandas format (e.g., '5min', '1H', '2D')
            precision (int): Number of decimal places in output

        Returns:
            str: Formatted time string with appropriate unit
        """
        value, unit = cls.parse_frequency(freq)
        total_minutes = di * value * cls.TIME_UNITS[unit]
        return cls.minutes_to_readable(total_minutes, precision)

    @classmethod
    def convert_time_units(
        cls,
        value: float,
        from_unit: str,
        to_unit: Optional[str] = None,
        precision: int = 1,
    ) -> Union[float, str]:
        """
        Convert time value between different units. If to_unit is not specified,
        converts to the most appropriate unit based on thresholds.

        Args:
            value (float): Time value to convert
            from_unit (str): Source unit
            to_unit (str, optional): Target unit. If None, converts to most appropriate unit
            precision (int): Number of decimal places in output (only used when to_unit is None)

        Returns:
            Union[float, str]: Converted time value as float if to_unit is specified,
                             or formatted string if to_unit is None

        Examples:
            >>> TimeUnitConverter.convert_time_units(24, 'h', 'd')
            1.0
            >>> TimeUnitConverter.convert_time_units(72, 'h')
            '3.0d'
            >>> TimeUnitConverter.convert_time_units(8760, 'h')
            '1.0y'
        """
        if from_unit not in cls.TIME_UNITS:
            raise ValueError(
                f"Invalid source unit. Supported units are: {', '.join(cls.TIME_UNITS.keys())}"
            )

        # Convert to minutes first
        total_minutes = value * cls.TIME_UNITS[from_unit]

        # If no target unit specified, convert to readable string
        if to_unit is None:
            return cls.minutes_to_readable(total_minutes, precision)

        # Otherwise, convert to target unit
        if to_unit not in cls.TIME_UNITS:
            raise ValueError(
                f"Invalid target unit. Supported units are: {', '.join(cls.TIME_UNITS.keys())}"
            )

        return total_minutes / cls.TIME_UNITS[to_unit]

    @classmethod
    def find_first_unit_occurrences(cls, time_strings: List[str], return_indices: bool = False) -> Union[List[str], Tuple[List[str], List[int]]]:
        """
        Find the first occurrence of each time unit in a list of time strings.
        
        Args:
            time_strings (List[str]): List of time strings (e.g., ['10.0mn', '1.3h', '2.7h'])
            return_indices (bool): If True, also return the indices of first occurrences
            
        Returns:
            Union[List[str], Tuple[List[str], List[int]]]: 
                If return_indices is False: List containing the first occurrence of each unit
                If return_indices is True: Tuple of (first occurrences list, indices list)
            
        Example:
            >>> times = ['10.0mn', '20.0mn', '1.3h', '2.7h', '1.8d', '3.6d', '1.9mo', '1.2y']
            >>> TimeUnitConverter.find_first_unit_occurrences(times, return_indices=True)
            (['10.0mn', '1.3h', '1.8d', '1.9mo', '1.2y'], [0, 2, 4, 6, 7])
        """
        # Define the order of units from smallest to largest
        unit_order = ['mn', 'h', 'd', 'mo', 'y']
        
        # Dictionary to store the first occurrence of each unit
        first_occurrences: Dict[str, Tuple[float, str, int]] = {}
        
        # Regular expression to extract value and unit
        pattern = re.compile(r'([\d.]+)([a-zA-Z]+)')
        
        # Process each time string
        for idx, time_str in enumerate(time_strings):
            match = pattern.match(time_str)
            if match:
                value = float(match.group(1))
                unit = match.group(2)
                
                # Store only the first occurrence of each unit
                if unit not in first_occurrences:
                    first_occurrences[unit] = (value, time_str, idx)
        
        # Sort results according to unit order
        result = []
        indices = []
        for unit in unit_order:
            if unit in first_occurrences:
                result.append(first_occurrences[unit][1])  # time string
                indices.append(first_occurrences[unit][2])  # index
        
        if return_indices:
            return result, indices
        return result
