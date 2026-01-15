"""
Tests for the timerange module.

This module tests the time range splitting functionality for parallel backtest.
"""

import pytest
from datetime import datetime

from parallel_backtest.timerange import (
    parse_timerange,
    split_by_month,
    split_timerange_monthly,
    should_split_timerange,
    TimeChunk,
)


class TestParseTimerange:
    """Tests for parse_timerange function"""
    
    def test_parse_full_range(self):
        """Test parsing a full date range"""
        start, end = parse_timerange("20240101-20241231")
        assert start == datetime(2024, 1, 1)
        assert end == datetime(2024, 12, 31)
    
    def test_parse_partial_year(self):
        """Test parsing a partial year range"""
        start, end = parse_timerange("20240315-20240915")
        assert start == datetime(2024, 3, 15)
        assert end == datetime(2024, 9, 15)
    
    def test_parse_invalid_format_no_dash(self):
        """Test that invalid format without dash raises error"""
        with pytest.raises(ValueError):
            parse_timerange("20240101")
    
    def test_parse_invalid_format_empty(self):
        """Test that empty string raises error"""
        with pytest.raises(ValueError):
            parse_timerange("")
    
    def test_parse_invalid_start_date(self):
        """Test that invalid start date raises error"""
        with pytest.raises(ValueError):
            parse_timerange("invalid-20241231")
    
    def test_parse_invalid_end_date(self):
        """Test that invalid end date raises error"""
        with pytest.raises(ValueError):
            parse_timerange("20240101-invalid")
    
    def test_parse_start_after_end(self):
        """Test that start date after end date raises error"""
        with pytest.raises(ValueError):
            parse_timerange("20241231-20240101")


class TestSplitByMonth:
    """Tests for split_by_month function"""
    
    def test_split_single_month(self):
        """Test splitting a single month range"""
        chunks = split_by_month("20240101-20240131")
        assert len(chunks) == 1
        assert chunks[0].start == "20240101"
        assert chunks[0].end == "20240131"
        assert chunks[0].label == "2024-01"
    
    def test_split_full_year(self):
        """Test splitting a full year into 12 months"""
        chunks = split_by_month("20240101-20241231")
        assert len(chunks) == 12
        
        # Check first month
        assert chunks[0].start == "20240101"
        assert chunks[0].end == "20240131"
        assert chunks[0].label == "2024-01"
        
        # Check last month
        assert chunks[11].start == "20241201"
        assert chunks[11].end == "20241231"
        assert chunks[11].label == "2024-12"
    
    def test_split_partial_months(self):
        """Test splitting with partial months at boundaries"""
        chunks = split_by_month("20240315-20240615")
        assert len(chunks) == 4  # Mar, Apr, May, Jun
        
        # First chunk starts mid-month
        assert chunks[0].start == "20240315"
        assert chunks[0].end == "20240331"
        assert chunks[0].label == "2024-03"
        
        # Last chunk ends mid-month
        assert chunks[3].start == "20240601"
        assert chunks[3].end == "20240615"
        assert chunks[3].label == "2024-06"
    
    def test_split_cross_year(self):
        """Test splitting across year boundary"""
        chunks = split_by_month("20231101-20240228")
        assert len(chunks) == 4  # Nov, Dec, Jan, Feb
        
        assert chunks[0].label == "2023-11"
        assert chunks[1].label == "2023-12"
        assert chunks[2].label == "2024-01"
        assert chunks[3].label == "2024-02"
    
    def test_chunk_timerange_property(self):
        """Test that TimeChunk.timerange returns correct format"""
        chunks = split_by_month("20240101-20240228")
        assert chunks[0].timerange == "20240101-20240131"
        assert chunks[1].timerange == "20240201-20240228"


class TestShouldSplitTimerange:
    """Tests for should_split_timerange function"""
    
    def test_should_split_when_pairs_less_than_cpus(self):
        """Test that splitting is recommended when pairs < CPUs"""
        assert should_split_timerange("20240101-20241231", num_pairs=5, num_cpus=16) is True
    
    def test_should_not_split_when_pairs_equal_cpus(self):
        """Test that splitting is not needed when pairs >= CPUs"""
        assert should_split_timerange("20240101-20241231", num_pairs=16, num_cpus=16) is False
    
    def test_should_not_split_when_pairs_more_than_cpus(self):
        """Test that splitting is not needed when pairs > CPUs"""
        assert should_split_timerange("20240101-20241231", num_pairs=20, num_cpus=16) is False
    
    def test_should_not_split_when_no_timerange(self):
        """Test that splitting is not possible without timerange"""
        assert should_split_timerange(None, num_pairs=5, num_cpus=16) is False


class TestSplitTimerangeMonthly:
    """Tests for split_timerange_monthly function"""
    
    def test_split_enabled_when_beneficial(self):
        """Test that splitting is enabled when pairs < CPUs"""
        chunks, enabled = split_timerange_monthly("20240101-20241231", num_pairs=5, num_cpus=16)
        assert enabled is True
        assert len(chunks) == 12
    
    def test_split_disabled_when_not_beneficial(self):
        """Test that splitting is disabled when pairs >= CPUs"""
        chunks, enabled = split_timerange_monthly("20240101-20241231", num_pairs=16, num_cpus=16)
        assert enabled is False
        assert len(chunks) == 1
    
    def test_split_disabled_when_no_timerange(self):
        """Test that splitting is disabled without timerange"""
        chunks, enabled = split_timerange_monthly(None, num_pairs=5, num_cpus=16)
        assert enabled is False
        assert len(chunks) == 0
    
    def test_total_tasks_calculation(self):
        """Test that total tasks = pairs × chunks"""
        # 5 pairs × 12 months = 60 tasks for 16 CPUs
        chunks, enabled = split_timerange_monthly("20240101-20241231", num_pairs=5, num_cpus=16)
        total_tasks = 5 * len(chunks)
        assert total_tasks == 60
        assert total_tasks >= 16  # Should utilize all CPUs
