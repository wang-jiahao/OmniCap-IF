import json
from typing import Dict, List, Any, Optional, Union, Literal

# Define parameter specifications for each constraint
CONSTRAINT_PARAMS = {
    "plain_text": {
        "required": ["content"],
        "optional": [],
        "types": {"content": str}
    },
    "json_object": {
        "required": ["content", "schema"],
        "optional": [],
        "types": {"content": str, "schema": dict}
    },
    "json_array": {
        "required": ["content", "schema"],
        "optional": [],
        "types": {"content": str, "schema": dict}
    },
    "unordered_list": {
        "required": ["content"],
        "optional": ["symbol"],
        "types": {"content": str, "symbol": str},
        "allowed_values": {"symbol": ["-", "*", None]}
    },
    "ordered_list": {
        "required": ["content"],
        "optional": ["symbol"],
        "types": {"content": str, "symbol": str},
        "allowed_values": {"symbol": ["1.", "a.", "A.", "I.", 'i.', None]}
    },
    "table": {
        "required": ["content", "col_name"],
        "optional": [],
        "types": {"content": str, "col_name": list}
    },
    "keyword": {
        "required": ["content", "keyword", "keyword_type"],
        "optional": [],
        "types": {"content": str, "keyword": str, "keyword_type": str},
        "allowed_values": {"keyword_type": ["include", "exclude"]}
    },
    "markdown": {
        "required": ["content", "md_type"],
        "optional": [],
        "types": {"content": str, "md_type": str},
        "allowed_values": {"md_type": ["title", "bold", "highlight", "italic", "code"]}
    },
    "prefix_suffix": {
        "required": ["content"],
        "optional": ["prefix", "suffix"],
        "types": {"content": str, "prefix": str, "suffix": str}
    },
    "delimiter": {
        "required": ["content", "symbol"],
        "optional": [],
        "types": {"content": str, "symbol": str}
    },
    "length": {
        "required": ["content", "unit"],
        "optional": ["min_len", "max_len"],
        "types": {"content": str, "unit": str, "min_len": int, "max_len": int},
        "allowed_values": {"unit": ["character", "word", "sentence", "paragraph"]},
        "defaults": {"min_len": 0, "max_len": -1}
    },
    "count": {
        "required": ["content"],
        "optional": ["min_count", "max_count"],
        "types": {"content": str, "min_count": int, "max_count": int},
        "defaults": {"min_count": 0, "max_count": -1}
    },
    "case": {
        "required": ["content", "case_type"],
        "optional": [],
        "types": {"content": str, "case_type": str},
        "allowed_values": {"case_type": ["upper", "lower", "title"]}
    },
    "language": {
        "required": ["content", "lang_type"],
        "optional": [],
        "types": {"content": str, "lang_type": str},
        "allowed_values": {"lang_type": ["en", "zh"]}
    },
    "timestamp_format": {
        "required": ["content", "format_type"],
        "optional": [],
        "types": {"content": str, "format_type": str}, 
        "allowed_values": {"format_type": ["point", "period"]}
    }
}

def validate_checklist(checklist: Union[str, Dict]) -> tuple[bool, List[str]]:
    """
    Validates if a checklist conforms to the specified format and constraints.
    
    Args:
        checklist: Either a JSON string or dict containing the checklist
        
    Returns:
        tuple: (is_valid: bool, errors: List[str])
    """
    errors = []
    
    # Parse JSON if string
    if isinstance(checklist, str):
        try:
            checklist = json.loads(checklist)
        except json.JSONDecodeError as e:
            return False, [f"Invalid JSON format: {str(e)}"]
    
    # Check top-level structure
    if not isinstance(checklist, dict):
        return False, ["Checklist must be a JSON object"]
    
    required_keys = {"format_check", "content_check"}
    missing_keys = required_keys - set(checklist.keys())
    if missing_keys:
        errors.append(f"Missing required keys: {missing_keys}")
    
    # Validate format_check
    if "format_check" in checklist:
        if not isinstance(checklist["format_check"], list):
            errors.append("format_check must be a list")
        else:
            for i, item in enumerate(checklist["format_check"]):
                errors.extend(_validate_format_check(item, i))
    
    # Validate content_check
    if "content_check" in checklist:
        if not isinstance(checklist["content_check"], list):
            errors.append("content_check must be a list")
        else:
            for i, item in enumerate(checklist["content_check"]):
                errors.extend(_validate_content_check(item, i))
    
    return len(errors) == 0, errors

def validate_format_checklist(format_checklist: Union[str, Dict]) -> tuple[bool, List[str]]:
    """Validate a complete format_check structure.

    Args:
        format_checklist: JSON string or dict containing {"format_check": [...]}.

    Returns:
        (is_valid, errors)
    """
    errors = []
    
    # Parse JSON if string
    if isinstance(format_checklist, str):
        try:
            format_checklist = json.loads(format_checklist)
        except json.JSONDecodeError as e:
            return False, [f"Invalid JSON format: {str(e)}"]
    
    # Check top-level structure
    if not isinstance(format_checklist, dict):
        return False, ["Input must be a JSON object"]
    
    if "format_check" not in format_checklist:
        return False, ["Missing required key: 'format_check'"]
    
    # Check if format_check is a list
    if not isinstance(format_checklist["format_check"], list):
        errors.append("format_check must be a list")
    else:
        # Validate each item
        for i, item in enumerate(format_checklist["format_check"]):
            errors.extend(_validate_format_check(item, i))
    
    return len(errors) == 0, errors


def validate_content_checklist(content_checklist: Union[str, Dict]) -> tuple[bool, List[str]]:
    """Validate a complete content_check structure.

    Args:
        content_checklist: JSON string or dict containing {"content_check": [...]}.

    Returns:
        (is_valid, errors)
    """
    errors = []
    
    # Parse JSON if string
    if isinstance(content_checklist, str):
        try:
            content_checklist = json.loads(content_checklist)
        except json.JSONDecodeError as e:
            return False, [f"Invalid JSON format: {str(e)}"]
    
    # Check top-level structure
    if not isinstance(content_checklist, dict):
        return False, ["Input must be a JSON object"]
    
    if "content_check" not in content_checklist:
        return False, ["Missing required key: 'content_check'"]
    
    # Check if content_check is a list
    if not isinstance(content_checklist["content_check"], list):
        errors.append("content_check must be a list")
    else:
        # Validate each item
        for i, item in enumerate(content_checklist["content_check"]):
            errors.extend(_validate_content_check(item, i))
    
    return len(errors) == 0, errors


def _validate_constraint_parameters(constraint_id: str, parameters: Dict, prefix: str) -> List[str]:
    """Validates parameters for a specific constraint type."""
    errors = []
    
    if constraint_id not in CONSTRAINT_PARAMS:
        return [f"{prefix}: Unknown constraint_id '{constraint_id}'"]
    
    spec = CONSTRAINT_PARAMS[constraint_id]
    param_keys = set(parameters.keys())
    
    # Check required parameters
    required_params = set(spec["required"])
    missing_required = required_params - param_keys
    if missing_required:
        errors.append(f"{prefix}: Missing required parameters: {missing_required}")
    
    # Check for unknown parameters
    all_allowed = set(spec["required"]) | set(spec.get("optional", []))
    unknown_params = param_keys - all_allowed
    if unknown_params:
        errors.append(f"{prefix}: Unknown parameters: {unknown_params}")
    
    # Validate parameter types and values
    for param, expected_type in spec["types"].items():
        if param in parameters and param != "content":  # content is checked separately
            value = parameters[param]
            if value is not None:  # Allow None for optional parameters
                if expected_type == list:
                    if not isinstance(value, list):
                        errors.append(f"{prefix}.{param} must be a list")
                    elif param == "col_name":
                        for idx, col in enumerate(value):
                            if not isinstance(col, str):
                                errors.append(f"{prefix}.{param}[{idx}] must be a string")
                elif expected_type == dict:
                    if not isinstance(value, dict):
                        errors.append(f"{prefix}.{param} must be a dict")
                elif expected_type == str:
                    if not isinstance(value, str):
                        errors.append(f"{prefix}.{param} must be a string")
                elif expected_type == int:
                    if not isinstance(value, int):
                        errors.append(f"{prefix}.{param} must be an integer")
                
                # Check allowed values
                if "allowed_values" in spec and param in spec["allowed_values"]:
                    allowed = spec["allowed_values"][param]
                    if value not in allowed:
                        errors.append(f"{prefix}.{param} must be one of {allowed}")
    
    # Special validation for length and count constraints
    if constraint_id == "length":
        min_len = parameters.get("min_len", 0)
        max_len = parameters.get("max_len", -1)
        if isinstance(min_len, int) and isinstance(max_len, int):
            if max_len != -1 and min_len > max_len:
                errors.append(f"{prefix}: min_len ({min_len}) cannot be greater than max_len ({max_len})")
    
    if constraint_id == "count":
        min_count = parameters.get("min_count", 0)
        max_count = parameters.get("max_count", -1)
        if isinstance(min_count, int) and isinstance(max_count, int):
            if max_count != -1 and min_count > max_count:
                errors.append(f"{prefix}: min_count ({min_count}) cannot be greater than max_count ({max_count})")
    
    return errors

def _validate_format_check(item: Dict, index: int) -> List[str]:
    """Validate a single format_check item."""
    errors = []
    prefix = f"format_check[{index}]"
    
    # Check required fields
    required_fields = {"check_id", "constraint_id", "check_description", "parameters"}
    if not isinstance(item, dict):
        return [f"{prefix} must be a dict"]
    
    missing_fields = required_fields - set(item.keys())
    if missing_fields:
        errors.append(f"{prefix} missing fields: {missing_fields}")
    
    # Validate field types
    if "check_id" in item and not isinstance(item["check_id"], str):
        errors.append(f"{prefix}.check_id must be a string")
    
    if "constraint_id" in item:
        if not isinstance(item["constraint_id"], str):
            errors.append(f"{prefix}.constraint_id must be a string")
        elif item["constraint_id"] not in CONSTRAINT_PARAMS:
            errors.append(f"{prefix}.constraint_id '{item['constraint_id']}' is not valid")
    
    if "check_description" in item and not isinstance(item["check_description"], str):
        errors.append(f"{prefix}.check_description must be a string")
    
    if "parameters" in item:
        if not isinstance(item["parameters"], dict):
            errors.append(f"{prefix}.parameters must be a dict")
        else:
            # Check content parameter
            if "content" not in item["parameters"]:
                errors.append(f"{prefix}.parameters.content is missing")
            elif item["parameters"]["content"] is not None:
                errors.append(f"{prefix}.parameters.content must be null in checklist")
            
            # Validate other parameters based on constraint_id
            if "constraint_id" in item:
                errors.extend(_validate_constraint_parameters(
                    item["constraint_id"], 
                    item["parameters"], 
                    f"{prefix}.parameters"
                ))
    
    return errors

def _validate_content_check(item: Dict, index: int) -> List[str]:
    """Validate a single content_check item."""
    errors = []
    prefix = f"content_check[{index}]"
    
    if not isinstance(item, dict):
        return [f"{prefix} must be a dict"]
    
    # Check required fields
    required_fields = {"check_content", "check_items"}
    missing_fields = required_fields - set(item.keys())
    if missing_fields:
        errors.append(f"{prefix} missing fields: {missing_fields}")
    
    if "check_content" in item and not isinstance(item["check_content"], str):
        errors.append(f"{prefix}.check_content must be a string")
    
    if "check_items" in item:
        if not isinstance(item["check_items"], list):
            errors.append(f"{prefix}.check_items must be a list")
        else:
            for j, check_item in enumerate(item["check_items"]):
                errors.extend(_validate_check_item(check_item, f"{prefix}.check_items[{j}]"))
    
    return errors

def _validate_check_item(item: Dict, prefix: str) -> List[str]:
    """Validate a single check item within content_check."""
    errors = []
    
    if not isinstance(item, dict):
        return [f"{prefix} must be a dict"]
    
    # Required fields depend on check_type
    base_required_fields = {"check_id", "check_type", "question", "correct_answer"}
    required_fields = set(base_required_fields)

    check_type = item.get("check_type")
    if check_type not in ["attempt", "correctness", "timestamp"]:
        errors.append(f"{prefix}.check_type must be one of: 'attempt', 'correctness', 'timestamp'")
    else:
        # attempt/correctness require options; timestamp does not
        if check_type in ["attempt", "correctness"]:
            required_fields.add("options")
    
    missing_fields = required_fields - set(item.keys())
    if missing_fields:
        errors.append(f"{prefix} missing fields: {missing_fields}")
    
    # Validate field values
    if "check_id" in item and not isinstance(item["check_id"], str):
        errors.append(f"{prefix}.check_id must be a string")
    
    if "question" in item and not isinstance(item["question"], str):
        errors.append(f"{prefix}.question must be a string")
    
    # Validate options field
    if "options" in item:
        if not isinstance(item["options"], list):
            errors.append(f"{prefix}.options must be a list")
        else:
            # Check if all options are strings
            for i, option in enumerate(item["options"]):
                if not isinstance(option, str):
                    errors.append(f"{prefix}.options[{i}] must be a string")
            
            # Validate options based on check_type
            if "check_type" in item:
                if item["check_type"] == "attempt":
                    # For attempt type, options should be ['yes', 'no']
                    if item["options"] != ['yes', 'no']:
                        errors.append(f"{prefix}.options must be ['yes', 'no'] for attempt type")
                elif item["check_type"] == "correctness":
                    # For correctness type, validate it's a proper option list
                    if len(item["options"]) < 2:
                        errors.append(f"{prefix}.options must have at least 2 options for correctness type")
                elif item["check_type"] == "timestamp":
                    errors.append(f"{prefix}.options must be omitted for timestamp type")
    
    # Validate correct_answer field
    if "correct_answer" in item:
        if not isinstance(item["correct_answer"], str):
            errors.append(f"{prefix}.correct_answer must be a string")
        else:
            # Validate correct_answer based on check_type
            if "check_type" in item:
                if item["check_type"] == "attempt":
                    # For attempt type, correct_answer should be 'yes' or 'no'
                    if item["correct_answer"] not in ['yes', 'no']:
                        errors.append(f"{prefix}.correct_answer must be 'yes' or 'no' for attempt type")
                elif item["check_type"] == "correctness":
                    # For correctness type, correct_answer should be single letter A, B, C, or D
                    if len(item["correct_answer"]) != 1 or item["correct_answer"] not in ['A', 'B', 'C', 'D']:
                        errors.append(f"{prefix}.correct_answer must be 'A', 'B', 'C', or 'D' for correctness type")
    
    return errors



def validate_check_result(check_result: Union[str, Dict]) -> tuple[bool, List[str]]:
    """
    Validates if a check result conforms to the specified format and constraints.
    
    Args:
        check_result: Either a JSON string or dict containing the check result
        
    Returns:
        tuple: (is_valid: bool, errors: List[str])
    """
    errors = []
    
    # Parse JSON if string
    if isinstance(check_result, str):
        try:
            check_result = json.loads(check_result)
        except json.JSONDecodeError as e:
            return False, [f"Invalid JSON format: {str(e)}"]
    
    # Check top-level structure
    if not isinstance(check_result, dict):
        return False, ["Check result must be a JSON object"]
    
    required_keys = {"check_content", "check_items"}
    missing_keys = required_keys - set(check_result.keys())
    if missing_keys:
        errors.append(f"Missing required keys: {missing_keys}")
    
    if "check_content" in check_result and not isinstance(check_result["check_content"], str):
        errors.append("check_content must be a string")
    
    if "check_items" in check_result:
        if not isinstance(check_result["check_items"], list):
            errors.append("check_items must be a list")
        else:
            for i, item in enumerate(check_result["check_items"]):
                errors.extend(_validate_result_item(item, f"check_items[{i}]"))
    
    return len(errors) == 0, errors

def _validate_result_item(item: Dict, prefix: str) -> List[str]:
    """Validates a single check item result."""
    errors = []
    
    if not isinstance(item, dict):
        return [f"{prefix} must be a dict"]
    
    # Check required fields for result format
    required_fields = {"check_id", "check_type", "question", "answer", "result_explanation", "result_confidence"}
    missing_fields = required_fields - set(item.keys())
    if missing_fields:
        errors.append(f"{prefix} missing fields: {missing_fields}")
    
    # Validate field values
    if "check_id" in item and not isinstance(item["check_id"], str):
        errors.append(f"{prefix}.check_id must be a string")
    
    if "check_type" in item:
        if item["check_type"] not in ["attempt", "correctness", "timestamp"]:
            errors.append(f"{prefix}.check_type must be one of: 'attempt', 'correctness', 'timestamp'")
    
    if "question" in item and not isinstance(item["question"], str):
        errors.append(f"{prefix}.question must be a string")
    
    # Validate answer field (now string instead of boolean)
    if "answer" in item:
        if not isinstance(item["answer"], str):
            errors.append(f"{prefix}.answer must be a string")
        else:
            # Validate answer based on check_type if available
            if "check_type" in item:
                if item["check_type"] == "attempt":
                    if item["answer"] not in ['yes', 'no']:
                        errors.append(f"{prefix}.answer must be 'yes' or 'no' for attempt type")
                elif item["check_type"] == "correctness":
                    if len(item["answer"]) != 1 or item["answer"] not in ['A', 'B', 'C', 'D']:
                        errors.append(f"{prefix}.answer must be 'A', 'B', 'C', or 'D' for correctness type")
                elif item["check_type"] == "timestamp":
                    # Accept any non-empty string; timestamp format validation is optional.
                    if not item["answer"].strip():
                        errors.append(f"{prefix}.answer must be a non-empty string for timestamp type")
    
    # Validate result_explanation
    if "result_explanation" in item:
        if not isinstance(item["result_explanation"], str):
            errors.append(f"{prefix}.result_explanation must be a string")
    
    # Validate result_confidence
    if "result_confidence" in item:
        if not isinstance(item["result_confidence"], int):
            errors.append(f"{prefix}.result_confidence must be an integer")
        elif not (1 <= item["result_confidence"] <= 5):
            errors.append(f"{prefix}.result_confidence must be between 1 and 5")
    
    return errors



