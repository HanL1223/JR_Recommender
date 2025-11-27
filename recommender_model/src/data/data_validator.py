"""
Data Validator
==============
Validates data quality before processing.

This is a NEW module (not in original) - essential for production MLOps.
"""

import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Single validation issue."""
    check_name: str
    severity: ValidationSeverity
    message: str
    affected_rows: int = 0
    details: Dict = field(default_factory=dict)


@dataclass 
class ValidationReport:
    """Complete validation report."""
    is_valid: bool
    issues: List[ValidationIssue]
    total_rows: int
    valid_rows: int
    
    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "=" * 50,
            "DATA VALIDATION REPORT",
            "=" * 50,
            f"Total rows: {self.total_rows:,}",
            f"Valid rows: {self.valid_rows:,} ({100*self.valid_rows/self.total_rows:.1f}%)",
            f"Status: {'PASSED ✓' if self.is_valid else 'FAILED ✗'}",
            "",
            "Issues:"
        ]
        
        for issue in self.issues:
            icon = {"info": "ℹ️", "warning": "⚠️", "error": "❌", "critical": "🚨"}
            lines.append(f"  {icon.get(issue.severity.value, '•')} [{issue.severity.value.upper()}] {issue.check_name}")
            lines.append(f"     {issue.message}")
        
        if not self.issues:
            lines.append("  None - all checks passed!")
        
        lines.append("=" * 50)
        return "\n".join(lines)


class DataValidator:
    """
    Validates transaction data quality.
    
    Checks:
    - Required columns exist
    - No null values in critical fields
    - Date ranges are valid
    - Customer IDs are valid
    - Product names are valid
    - Prices are positive
    
    Example:
        >>> validator = DataValidator()
        >>> report = validator.validate(df)
        >>> if not report.is_valid:
        ...     print(report.summary())
        ...     raise ValueError("Data validation failed")
    """
    
    REQUIRED_COLUMNS = [
        'order_id', 'customer_id', 'order_date',
        'product_name', 'product_variant'
    ]
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize validator.
        
        Args:
            strict_mode: If True, warnings become errors
        """
        self.strict_mode = strict_mode
        logger.info(f"DataValidator initialized (strict_mode={strict_mode})")
    
    def validate(self, df: pd.DataFrame) -> ValidationReport:
        """
        Run all validation checks on data.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            ValidationReport with all issues found
        """
        logger.info("Running data validation...")
        
        issues = []
        
        # Run all checks
        issues.extend(self._check_required_columns(df))
        issues.extend(self._check_null_values(df))
        issues.extend(self._check_date_validity(df))
        issues.extend(self._check_customer_ids(df))
        issues.extend(self._check_product_names(df))
        issues.extend(self._check_prices(df))
        issues.extend(self._check_duplicates(df))
        
        # Determine if valid
        if self.strict_mode:
            is_valid = not any(i.severity in [ValidationSeverity.WARNING, ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] for i in issues)
        else:
            is_valid = not any(i.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] for i in issues)
        
        # Count valid rows (rows without critical issues)
        valid_rows = len(df) - sum(i.affected_rows for i in issues if i.severity == ValidationSeverity.CRITICAL)
        
        report = ValidationReport(
            is_valid=is_valid,
            issues=issues,
            total_rows=len(df),
            valid_rows=valid_rows
        )
        
        logger.info(f"Validation complete: {'PASSED' if is_valid else 'FAILED'}")
        return report
    
    def _check_required_columns(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Check that all required columns exist."""
        issues = []
        missing = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        
        if missing:
            issues.append(ValidationIssue(
                check_name="Required Columns",
                severity=ValidationSeverity.CRITICAL,
                message=f"Missing columns: {missing}",
                affected_rows=len(df),
                details={"missing_columns": missing}
            ))
        
        return issues
    
    def _check_null_values(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Check for null values in critical columns."""
        issues = []
        critical_cols = ['order_id', 'customer_id', 'order_date', 'product_name']
        
        for col in critical_cols:
            if col in df.columns:
                null_count = df[col].isna().sum()
                if null_count > 0:
                    issues.append(ValidationIssue(
                        check_name=f"Null Check: {col}",
                        severity=ValidationSeverity.ERROR,
                        message=f"{null_count:,} null values in '{col}'",
                        affected_rows=null_count,
                        details={"column": col, "null_count": null_count}
                    ))
        
        return issues
    
    def _check_date_validity(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Check date ranges are valid."""
        issues = []
        
        if 'order_date' in df.columns:
            # Check for future dates
            future_dates = df[df['order_date'] > pd.Timestamp.now()].shape[0]
            if future_dates > 0:
                issues.append(ValidationIssue(
                    check_name="Future Dates",
                    severity=ValidationSeverity.WARNING,
                    message=f"{future_dates:,} orders have future dates",
                    affected_rows=future_dates
                ))
            
            # Check for very old dates (before 2010)
            old_dates = df[df['order_date'] < pd.Timestamp('2010-01-01')].shape[0]
            if old_dates > 0:
                issues.append(ValidationIssue(
                    check_name="Old Dates",
                    severity=ValidationSeverity.WARNING,
                    message=f"{old_dates:,} orders before 2010",
                    affected_rows=old_dates
                ))
        
        return issues
    
    def _check_customer_ids(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Check customer IDs are valid."""
        issues = []
        
        if 'customer_id' in df.columns:
            # Check for negative IDs
            negative_ids = df[df['customer_id'] < 0].shape[0]
            if negative_ids > 0:
                issues.append(ValidationIssue(
                    check_name="Negative Customer IDs",
                    severity=ValidationSeverity.ERROR,
                    message=f"{negative_ids:,} rows with negative customer_id",
                    affected_rows=negative_ids
                ))
        
        return issues
    
    def _check_product_names(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Check product names are valid."""
        issues = []
        
        if 'product_name' in df.columns:
            # Check for empty strings
            empty_names = df[df['product_name'].astype(str).str.strip() == ''].shape[0]
            if empty_names > 0:
                issues.append(ValidationIssue(
                    check_name="Empty Product Names",
                    severity=ValidationSeverity.ERROR,
                    message=f"{empty_names:,} rows with empty product_name",
                    affected_rows=empty_names
                ))
        
        return issues
    
    def _check_prices(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Check prices are valid."""
        issues = []
        
        if 'order_total_price' in df.columns:
            # Check for negative prices
            negative_prices = df[df['order_total_price'] < 0].shape[0]
            if negative_prices > 0:
                issues.append(ValidationIssue(
                    check_name="Negative Prices",
                    severity=ValidationSeverity.ERROR,
                    message=f"{negative_prices:,} orders with negative price",
                    affected_rows=negative_prices
                ))
            
            # Check for unusually high prices
            high_threshold = df['order_total_price'].quantile(0.99) * 3
            high_prices = df[df['order_total_price'] > high_threshold].shape[0]
            if high_prices > 0:
                issues.append(ValidationIssue(
                    check_name="Unusually High Prices",
                    severity=ValidationSeverity.INFO,
                    message=f"{high_prices:,} orders with price > ${high_threshold:.2f}",
                    affected_rows=high_prices
                ))
        
        return issues
    
    def _check_duplicates(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Check for duplicate records."""
        issues = []
        
        # Check for duplicate order_item combinations
        if 'order_id' in df.columns and 'product_name' in df.columns:
            dup_cols = ['order_id', 'product_name', 'product_variant'] if 'product_variant' in df.columns else ['order_id', 'product_name']
            duplicates = df.duplicated(subset=dup_cols, keep=False).sum()
            
            if duplicates > 0:
                issues.append(ValidationIssue(
                    check_name="Duplicate Records",
                    severity=ValidationSeverity.WARNING,
                    message=f"{duplicates:,} potential duplicate order items",
                    affected_rows=duplicates
                ))
        
        return issues