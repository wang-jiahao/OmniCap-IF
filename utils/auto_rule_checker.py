import json
import os
import argparse
from typing import Dict, Any, Optional
from utils import FormatCheckModule

class AutoRuleChecker:
    """
    Automated checker for executing format validations.
    """
    
    def __init__(self, utils_path: Optional[str] = None):
        """
        Initialize the format checker.
        
        Args:
            utils_path: Path to the utils directory. If None, uses the default.
        """
        self.utils_path = utils_path or os.path.join(os.path.dirname(__file__), 'utils')
        self.rule_functions = {}
        self._load_rule_functions()
    
    def _load_rule_functions(self):
        """Dynamically load the format-check module."""
        try:
            format_check_module = FormatCheckModule()
            
            # Collect all check functions.
            self.rule_functions = {
                'plain_text': format_check_module.plain_text, 
                'json_object': format_check_module.json_object,
                'json_array': format_check_module.json_array,
                'unordered_list': format_check_module.unordered_list,
                'ordered_list': format_check_module.ordered_list,
                'table': format_check_module.table,
                'keyword': format_check_module.keyword,
                'markdown': format_check_module.markdown,
                'prefix_suffix': format_check_module.prefix_suffix,
                'delimiter': format_check_module.delimiter,
                'length': format_check_module.length,
                'count': format_check_module.count,
                'case': format_check_module.case,
                'language': format_check_module.language,
                'timestamp_format': format_check_module.timestamp_format
            }
        except Exception as e:
            print(f"Failed to load format-check module: {e}")
            print("Please ensure utils/ruled_check.py exists and is accessible")
            # Do not raise: allow the class to be instantiated.

    def load_check_data(self, check_json_path: str) -> Dict[str, Any]:
        """
        Load the check data file.
        
        Args:
            check_json_path: Path to the check_result.json file.
            
        Returns:
            Parsed check data.
            
        Raises:
            FileNotFoundError: File not found.
            json.JSONDecodeError: Invalid JSON.
        """
        try:
            with open(check_json_path, 'r', encoding='utf-8') as f:
                check_data = json.load(f)
            print(f"Loaded check data: {check_json_path}")
            return check_data
        except FileNotFoundError:
            raise FileNotFoundError(f"Check data file '{check_json_path}' not found")
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Failed to parse JSON file '{check_json_path}': {e}")
    
    def execute_rule_check(self, api_call: str, parameters: Dict[str, Any]) -> tuple:
        """
        Execute a single rule check.
        
        Args:
            api_call: Rule/constraint id.
            parameters: Parameters for the checker.
            
        Returns:
            (success: bool, result: Any, error: str)
        """
        if not api_call:
            return False, None, "Missing api_call field"
        
        if api_call not in self.rule_functions:
            return False, None, f"Unknown API call: {api_call}"
        
        try:
            func = self.rule_functions[api_call]
            content_list = parameters.get('content')
            if isinstance(content_list, list):
                shared_params = {k: v for k, v in parameters.items() if k != 'content'}
                results = []
                for content in content_list:
                    call_params = dict(shared_params)
                    call_params['content'] = content
                    result = func(**call_params)
                    results.append(result)
                final_result = all(results)
                return True, final_result, None
            else:
                result = func(**parameters)
                return True, result, None
        except Exception as e:
            return False, None, str(e)
        
    def check_all_rules(self, check_data: dict) -> tuple:
        """
        Execute all rule checks for a single case.
        
        Args:
            check_data: One case to check.
            
        Returns:
            (updated_check_data, success)
        """
            
        if not isinstance(check_data, dict) or 'format_check' not in check_data:
            print("  Note: no format_check field; skipping")
            return check_data, True
        format_checks = check_data['format_check']
        if not isinstance(format_checks, list):
            print("  Error: format_check is not a list")
            return check_data, False
        
        # Iterate each rule check
        for rule_idx, rule_check in enumerate(format_checks):
            
            api_call = rule_check.get('constraint_id')
            parameters = rule_check.get('parameters', {})
            
            if not isinstance(parameters, dict):
                print(f"    Error: rule check {rule_idx + 1} has non-dict parameters")
                return check_data, False
            
            # Execute
            success, result, error = self.execute_rule_check(api_call, parameters)
            
            if success:
                rule_check['result'] = result
            else:
                print(f"    ✗ Rule check {rule_idx + 1} ({api_call}) failed: {error}")
                return check_data, False
        return check_data, True

    def check_all_rules_form_file(self, check_json_path: str, output_json_path: Optional[str] = None) -> Dict[str, int]:
        """
        Execute rule checks for all cases in a file.
        
        Args:
            check_json_path: Path to check.json.
            output_json_path: Output path; if None, overwrite the input.
            
        Returns:
            A dict of summary statistics.
        """
        # 1) Load check.json
        try:
            check_data = self.load_check_data(check_json_path)
        except Exception as e:
            print(f"Error: {e}")
            return {}
        
        # Summary statistics
        stats = {
            'total_videos': 0,
            'total_rule_checks': 0,
            'successful_checks': 0
        }
        
        print("Starting automated rule checks...")
        
        # 2) Iterate each video_id
        for video_id, check_cases in check_data.items():
            if not isinstance(check_cases, dict) or 'check_case' not in check_cases:
                continue
                
            stats['total_videos'] += 1
            print(f"\n--- Processing video: {video_id} ---")
            
            check_list = check_cases['check_case']
            
            # 3) Iterate each check item
            for check_idx, check_item in enumerate(check_list):
                if not isinstance(check_item, dict) or 'check_result' not in check_item:
                    print(f"  Error: check item {check_idx + 1} has invalid format")
                    return
                    
                check_result = check_item['check_result']
                if not isinstance(check_result, dict) or 'format_check' not in check_result:
                    print(f"  Note: check item {check_idx + 1} has no format_check; skipping")
                    continue
                    
                format_checks = check_result['format_check']
                if not isinstance(format_checks, list):
                    print(f"  Error: format_check in item {check_idx + 1} is not a list")
                    return
                
                print(f"  Check item {check_idx + 1}: {len(format_checks)} format checks")
                
                # 4) Iterate each rule check
                for rule_idx, rule_check in enumerate(format_checks):
                    if not isinstance(rule_check, dict):
                        continue
                        
                    stats['total_rule_checks'] += 1
                    api_call = rule_check.get('constraint_id')
                    parameters = rule_check.get('parameters', {})
                    
                    if not isinstance(parameters, dict):
                        print(f"    Error: rule check {rule_idx + 1} has non-dict parameters")
                        return
                    
                    # Execute
                    success, result, error = self.execute_rule_check(api_call, parameters)
                    
                    if success:
                        rule_check['result'] = result
                        stats['successful_checks'] += 1
                        print(f"    ✓ Rule check {rule_idx + 1} ({api_call}): {result}")
                    else:
                        print(f"    ✗ Rule check {rule_idx + 1} ({api_call}) failed: {error}")
                        return
        
        # 6) Save results
        output_path = output_json_path if output_json_path else check_json_path
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(check_data, f, ensure_ascii=False, indent=2)
            print("\n--- Automated rule checks finished ---")
            print(f"Results saved to: {output_path}")
        except Exception as e:
            print(f"Error while saving file: {str(e)}")
            return stats
        
        # 7) Print statistics
        self._print_statistics(stats)
        return stats
    
    def _print_statistics(self, stats: Dict[str, int]):
        """Print summary statistics."""
        print("\n--- Summary ---")
        print(f"  Videos processed: {stats['total_videos']}")
        print(f"  Total rule checks: {stats['total_rule_checks']}")
        print(f"  Successful checks: {stats['successful_checks']}")
        if stats['total_rule_checks'] > 0:
            success_rate = (stats['successful_checks'] / stats['total_rule_checks']) * 100
            print(f"  Success rate: {success_rate:.1f}%")

    def validate_results(self, check_json_path: str) -> Dict[str, int]:
        """
        Validate rule-check results and compute simple statistics.
        
        Args:
            check_json_path: Path to check.json.
            
        Returns:
            Validation statistics.
        """
        try:
            check_data = self.load_check_data(check_json_path)
        except Exception as e:
            print(f"Failed to read file: {e}")
            return {}
        
        stats = {
            'total_rules': 0,
            'passed_rules': 0,
            'failed_rules': 0,
            'rules_with_score': 0
        }
        
        for video_id, check_cases in check_data.items():
            if not isinstance(check_cases, dict) or 'check' not in check_cases:
                continue
                
            for check_item in check_cases['check']:
                if not isinstance(check_item, dict) or 'check_result' not in check_item:
                    continue
                    
                format_checks = check_item.get('check_result', {}).get('format_check', [])
                
                for rule_check in format_checks:
                    if isinstance(rule_check, dict):
                        stats['total_rules'] += 1
                        
                        if 'score' in rule_check:
                            stats['rules_with_score'] += 1
                            score = rule_check['score']
                            
                            if score is True:
                                stats['passed_rules'] += 1
                            elif score is False:
                                stats['failed_rules'] += 1
        
        self._print_validation_statistics(stats)
        return stats
    
    def _print_validation_statistics(self, stats: Dict[str, int]):
        """Print validation statistics."""
        print("--- Rule Check Statistics ---")
        print(f"  Total rules: {stats['total_rules']}")
        print(f"  Rules with score: {stats['rules_with_score']}")
        print(f"  Passed rules: {stats['passed_rules']}")
        print(f"  Failed rules: {stats['failed_rules']}")
        if stats['total_rules'] > 0:
            completion_rate = (stats['rules_with_score'] / stats['total_rules']) * 100
            print(f"  Scored completion: {completion_rate:.1f}%")
        if stats['rules_with_score'] > 0:
            pass_rate = (stats['passed_rules'] / stats['rules_with_score']) * 100
            print(f"  Pass rate: {pass_rate:.1f}%")


# Backward-compatible function wrappers
def auto_rule_checker(check_json_path: str, output_json_path: str = None):
    """
    Wrapper to run rule checks from a JSON file (backward compatible).
    
    Args:
        check_json_path: Path to check.json.
        output_json_path: Output path; if None, overwrite the input.
    """
    checker = AutoRuleChecker()
    return checker.check_all_rules(check_json_path, output_json_path)


def validate_rule_check_result(check_json_path: str):
    """
    Wrapper to validate rule-check results (backward compatible).
    
    Args:
        check_json_path: Path to check.json.
    """
    checker = AutoRuleChecker()
    return checker.validate_results(check_json_path)

# --- Entry point ---
def main():

    OUTPUT_FILE = "check_result.json"
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FOLDER), exist_ok=True)


    # Run
    checker = AutoRuleChecker()
    checker.check_all_rules(OUTPUT_FILE, OUTPUT_FILE.replace("checkresult", "checkresult_ruled"))

if __name__ == "__main__":
    main()