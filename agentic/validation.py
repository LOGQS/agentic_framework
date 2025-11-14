"""
Format-agnostic validation system for tool arguments and schemas.

Provides a flexible validator registry where users can register validators for any format:
- Built-in simple validator for common cases (type checking, required fields, constraints)
- Support for JSON Schema, XML Schema, Protocol Buffers, or custom formats
- Extensible via user-defined validator functions
"""
from __future__ import annotations
from typing import Callable, Any
from dataclasses import dataclass


@dataclass
class ValidationError:
    """Structured validation error with field and message."""
    field: str
    message: str
    value: Any = None


ValidatorFunc = Callable[[Any, dict[str, Any]], tuple[bool, list[ValidationError]]]


class ValidatorRegistry:
    """
    Registry for schema validators.

    Users can register validators for any format (JSON Schema, XML, custom formats).
    Each validator is a function that takes (value, schema) and returns (is_valid, errors).
    """

    def __init__(self):
        self._validators: dict[str, ValidatorFunc] = {}
        self._register_builtin_validators()

    def register(self, name: str, validator: ValidatorFunc) -> None:
        """
        Register a validator function.

        Args:
            name: Validator identifier (e.g., "json_schema", "xml", "protobuf")
            validator: Function taking (value, schema) returning (is_valid, errors)
        """
        self._validators[name] = validator

    def get(self, name: str) -> ValidatorFunc | None:
        """Get validator by name."""
        return self._validators.get(name)

    def list(self) -> list[str]:
        """List all registered validator names."""
        return sorted(self._validators.keys())

    def validate(self, value: Any, schema: dict[str, Any]) -> tuple[bool, list[ValidationError]]:
        """
        Validate value against schema.

        Schema format:
            {
                "validator": "simple" | "json_schema" | "custom_name",
                ... (validator-specific fields)
            }

        Returns:
            (is_valid, errors) - is_valid is True if no errors
        """
        if not schema:
            return True, []

        validator_name = schema.get("validator", "simple")
        validator = self._validators.get(validator_name)

        if validator is None:
            return False, [ValidationError(
                field="_schema",
                message=f"Unknown validator: {validator_name}. Available: {', '.join(self.list())}"
            )]

        try:
            return validator(value, schema)
        except Exception as e:
            return False, [ValidationError(
                field="_validator",
                message=f"Validator '{validator_name}' raised exception: {str(e)}"
            )]

    def _register_builtin_validators(self):
        """Register built-in validators."""
        self.register("simple", simple_validator)
        self.register("passthrough", passthrough_validator)


def passthrough_validator(value: Any, schema: dict[str, Any]) -> tuple[bool, list[ValidationError]]:
    """
    Validator that always passes. Useful for disabling validation on specific tools.

    Args:
        value: Value to validate (ignored)
        schema: Schema definition (ignored)

    Returns:
        (True, []) - always valid
    """
    return True, []


def simple_validator(value: dict, schema: dict) -> tuple[bool, list[ValidationError]]:
    """
    Built-in lightweight validator for common cases.

    Supports:
    - Required field checking
    - Type validation (str, int, float, bool, dict, list, bytes)
    - String constraints: min_length, max_length, pattern (regex)
    - Numeric constraints: min, max
    - List constraints: min_items, max_items
    - Custom validation functions per field

    Schema format:
        {
            "validator": "simple",
            "required": ["field1", "field2"],
            "fields": {
                "field1": {
                    "type": "str",
                    "min_length": 5,
                    "max_length": 100,
                    "pattern": "^[a-z]+$"
                },
                "field2": {
                    "type": "int",
                    "min": 0,
                    "max": 100
                },
                "field3": {
                    "type": "list",
                    "min_items": 1,
                    "max_items": 10
                }
            },
            "allow_extra_fields": False  # Reject unexpected fields
        }

    Args:
        value: Dict to validate
        schema: Schema definition

    Returns:
        (is_valid, errors)
    """
    errors = []

    if not isinstance(value, dict):
        return False, [ValidationError("_root", f"Expected dict, got {type(value).__name__}")]

    required = schema.get("required", [])
    for field in required:
        if field not in value:
            errors.append(ValidationError(field, f"Required field '{field}' is missing"))

    if not schema.get("allow_extra_fields", True):
        fields_schema = schema.get("fields", {})
        for field in value.keys():
            if field not in fields_schema:
                errors.append(ValidationError(field, f"Unexpected field '{field}' not in schema"))

    fields = schema.get("fields", {})
    for field_name, field_schema in fields.items():
        if field_name not in value:
            continue

        val = value[field_name]
        field_errors = _validate_field(field_name, val, field_schema)
        errors.extend(field_errors)

    return len(errors) == 0, errors


def _validate_field(field_name: str, value: Any, field_schema: dict) -> list[ValidationError]:
    """Validate a single field against its schema."""
    errors = []

    expected_type = field_schema.get("type")

    if expected_type:
        type_valid, type_error = _check_type(field_name, value, expected_type)
        if not type_valid:
            errors.append(type_error)
            return errors 

    if isinstance(value, str):
        errors.extend(_validate_string(field_name, value, field_schema))
    elif isinstance(value, (int, float)):
        errors.extend(_validate_number(field_name, value, field_schema))
    elif isinstance(value, list):
        errors.extend(_validate_list(field_name, value, field_schema))
    elif isinstance(value, dict):
        errors.extend(_validate_dict(field_name, value, field_schema))

    if "validator_func" in field_schema:
        validator_func = field_schema["validator_func"]
        if callable(validator_func):
            try:
                is_valid = validator_func(value)
                if not is_valid:
                    errors.append(ValidationError(field_name, "Custom validation failed"))
            except Exception as e:
                errors.append(ValidationError(field_name, f"Custom validator raised exception: {str(e)}"))

    return errors


def _check_type(field_name: str, value: Any, expected_type: str) -> tuple[bool, ValidationError | None]:
    """Check if value matches expected type."""
    type_map = {
        "str": str,
        "int": int,
        "float": (int, float),  
        "bool": bool,
        "dict": dict,
        "list": list,
        "bytes": bytes,
        "any": object
    }

    if expected_type not in type_map:
        return False, ValidationError(field_name, f"Unknown type in schema: {expected_type}")

    expected_cls = type_map[expected_type]
    if not isinstance(value, expected_cls):
        actual_type = type(value).__name__
        return False, ValidationError(field_name, f"Expected {expected_type}, got {actual_type}")

    return True, None


def _validate_string(field_name: str, value: str, schema: dict) -> list[ValidationError]:
    """Validate string constraints."""
    errors = []

    if "min_length" in schema and len(value) < schema["min_length"]:
        errors.append(ValidationError(
            field_name,
            f"String too short (min: {schema['min_length']}, got: {len(value)})"
        ))

    if "max_length" in schema and len(value) > schema["max_length"]:
        errors.append(ValidationError(
            field_name,
            f"String too long (max: {schema['max_length']}, got: {len(value)})"
        ))

    if "pattern" in schema:
        import re
        pattern = schema["pattern"]
        try:
            if not re.match(pattern, value):
                errors.append(ValidationError(
                    field_name,
                    f"String does not match pattern: {pattern}"
                ))
        except re.error as e:
            errors.append(ValidationError(
                field_name,
                f"Invalid regex pattern in schema: {str(e)}"
            ))

    return errors


def _validate_number(field_name: str, value: int | float, schema: dict) -> list[ValidationError]:
    """Validate numeric constraints."""
    errors = []

    if "min" in schema and value < schema["min"]:
        errors.append(ValidationError(
            field_name,
            f"Value too small (min: {schema['min']}, got: {value})"
        ))

    if "max" in schema and value > schema["max"]:
        errors.append(ValidationError(
            field_name,
            f"Value too large (max: {schema['max']}, got: {value})"
        ))

    return errors


def _validate_list(field_name: str, value: list, schema: dict) -> list[ValidationError]:
    """Validate list constraints."""
    errors = []

    if "min_items" in schema and len(value) < schema["min_items"]:
        errors.append(ValidationError(
            field_name,
            f"List too short (min items: {schema['min_items']}, got: {len(value)})"
        ))

    if "max_items" in schema and len(value) > schema["max_items"]:
        errors.append(ValidationError(
            field_name,
            f"List too long (max items: {schema['max_items']}, got: {len(value)})"
        ))

    if "item_schema" in schema:
        item_schema = schema["item_schema"]
        for i, item in enumerate(value):
            item_errors = _validate_field(f"{field_name}[{i}]", item, item_schema)
            errors.extend(item_errors)

    return errors


def _validate_dict(field_name: str, value: dict, schema: dict) -> list[ValidationError]:
    """Validate dict constraints."""
    errors = []

    if "nested_schema" in schema:
        nested_schema = schema["nested_schema"]
        is_valid, nested_errors = simple_validator(value, nested_schema)
        for err in nested_errors:
            errors.append(ValidationError(f"{field_name}.{err.field}", err.message, err.value))

    return errors
