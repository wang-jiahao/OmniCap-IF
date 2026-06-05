from openai import OpenAI
from google import genai
from google.genai import types
import json
from pathlib import Path
import concurrent.futures
import functools
from typing import Callable, Any

def gemini_client(api_config_path: str='./api.json'):
    """
    Create a Gemini client from a local API config file.
    """
    api_config = json.loads(Path(api_config_path).read_text(encoding='utf-8'))
    client = genai.Client(api_key=api_config["gemini_key"])
    return client

def openai_client(api_config_path: str='./api.json'):
    """
    Create an OpenAI client from a local API config file.
    """
    api_config = json.loads(Path(api_config_path).read_text(encoding='utf-8'))
    client = OpenAI(api_key=api_config["api_key"],
                    base_url=api_config["openai_url"],
                    timeout=18000)
    return client

# Utilities
def clean_json_response(response_text: str) -> str:
    """
    Clean an LLM response that may be wrapped in a Markdown JSON code fence.
    """
    if response_text.startswith("```json\n"):
        response_text = response_text[8:]
    if response_text.endswith("\n```"):
        response_text = response_text[:-4]
    return response_text


def timeout_with_retry(timeout_seconds: int, max_retries: int = 3):
    """Timeout decorator implemented with a thread pool, with retries."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            for attempt in range(max_retries):
                try:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(func, *args, **kwargs)
                        try:
                            result = future.result(timeout=timeout_seconds)
                            return result
                        except concurrent.futures.TimeoutError:
                            future.cancel()  # best-effort cancellation
                            raise TimeoutError(
                                f"Function {func.__name__} timed out ({timeout_seconds}s)"
                            )
                            
                except TimeoutError as e:
                    print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                    if attempt == max_retries - 1:
                        raise
                    print("Retrying...")
                    
            return None
        return wrapper
    return decorator


def error_retry(max_retries: int = 3, exceptions: tuple = (Exception,), delay: float = 1.0, backoff: float = 2.0):
    """
    Retry decorator for transient errors.
    
    Args:
        max_retries: Maximum retry attempts.
        exceptions: Exception types to catch and retry.
        delay: Initial delay (seconds).
        backoff: Exponential backoff multiplier.
    """
    import time
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            current_delay = delay
            
            for attempt in range(max_retries + 1):  # +1 because the first run is not a retry
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        print(f"Function {func.__name__} still failed after {max_retries} retries")
                        raise
                    
                    print(
                        f"Function {func.__name__} failed on run {attempt + 1}: {type(e).__name__}: {e}"
                    )
                    print(f"Waiting {current_delay:.1f}s before attempt {attempt + 2}...")
                    
                    time.sleep(current_delay)
                    current_delay *= backoff  # exponential backoff
                    
            return None
        return wrapper
    return decorator


def combined_retry(timeout_seconds: int = 30, timeout_retries: int = 3, 
                  error_retries: int = 3, exceptions: tuple = (Exception,), 
                  delay: float = 1.0, backoff: float = 2.0):
    """
    Combined decorator that supports both timeout retries and error retries.
    
    Args:
        timeout_seconds: Timeout in seconds.
        timeout_retries: Number of timeout retries.
        error_retries: Number of error retries.
        exceptions: Exception types to catch.
        delay: Initial delay in seconds.
        backoff: Backoff multiplier.
    """
    def decorator(func: Callable) -> Callable:
        # Apply error retry first, then timeout retry.
        func_with_error_retry = error_retry(error_retries, exceptions, delay, backoff)(func)
        func_with_both_retries = timeout_with_retry(timeout_seconds, timeout_retries)(func_with_error_retry)
        return func_with_both_retries
    return decorator