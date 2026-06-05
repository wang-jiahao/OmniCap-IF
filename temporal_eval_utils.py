# temporal_eval_utils.py

import re

def to_seconds(time_str):
    """
    Parses 'MM:SS', 'MM:SS.s', or 'SS.s' into seconds.
    Returns float or None.
    """
    # 1. Match MM:SS.s or MM:SS
    mmss_match = re.match(r'(\d{1,2}):(\d{2}(?:\.\d+)?)', time_str)
    if mmss_match:
        m, s = mmss_match.groups()
        return int(m) * 60 + float(s)
    
    # 2. Match pure number (seconds)
    time_str_clean = time_str.strip().lower().rstrip('s')
    try:
        # Check if it looks like a number "12.5"
        return float(time_str_clean)
    except ValueError:
        pass
        
    return None

def parse_timestamps(text):
    """
    Parses various timestamp formats from text.
    Returns:
        intervals: List of tuples (start, end) in seconds.
        points: List of floats in seconds.
    """
    intervals = []
    points = []
    
    if not text:
        return intervals, points
        
    # Pre-clean text: handle "00:10 - 00:20" vs "00:10 to 00:20"
    # Normalize separators
    # Remove brackets
    cleaned_text = text.replace('[', ' ').replace(']', ' ').replace('(', ' ').replace(')', ' ')
    
    # Regex for a time unit (MM:SS or S.s)
    # \d{1,2}:\d{2}(?:\.\d+)? matches 0:05, 05:30, 05:30.5
    # Matches XXs or XX.Xs explicitly to be safe
    
    mmss_pat = r'(?:\d{1,2}:\d{2}(?:\.\d+)?)'
    seconds_pat = r'(?:\d+(?:\.\d+)?\s*s)'
    
    # Combined pattern: either MM:SS or XXs
    time_pat = f'(?:{mmss_pat}|{seconds_pat})'
    
    # 1. Find Intervals: Time1 <sep> Time2
    # Separators: " - ", " to ", " ~ ", or just space-dash-space
    # We look for PAIRS
    interval_regex = re.compile(f'({time_pat})\s*(?:-|to|~)\s*({time_pat})', re.IGNORECASE)
    
    # Find all intervals first
    for match in interval_regex.finditer(cleaned_text):
        t1_str, t2_str = match.groups()
        t1 = to_seconds(t1_str)
        t2 = to_seconds(t2_str)
        
        if t1 is not None and t2 is not None:
             # Ensure start <= end
             start, end = min(t1, t2), max(t1, t2)
             intervals.append((start, end))
             
    # Remove matched intervals from text to avoid double counting them as points
    # We replace them with spaces to preserve indices/boundaries if needed (though extracting list is fine)
    text_no_intervals = interval_regex.sub(' ', cleaned_text)
    
    # 2. Find Points: Remaining Times
    point_regex = re.compile(f'({time_pat})')
    
    for match in point_regex.finditer(text_no_intervals):
        t_str = match.group(1)
        t = to_seconds(t_str)
        if t is not None:
            points.append(t)
            
    return intervals, points


def calculate_iou(interval_a, interval_b):
    """
    Calculates Temporal Intersection over Union (t-IoU).
    Intervals are (start, end) tuples.
    """
    start_a, end_a = interval_a
    start_b, end_b = interval_b

    # Intersection
    intersection_start = max(start_a, start_b)
    intersection_end = min(end_a, end_b)
    
    if intersection_start >= intersection_end:
        intersection = 0.0
    else:
        intersection = intersection_end - intersection_start
    
    # Union
    union_start = min(start_a, start_b)
    union_end = max(end_a, end_b)
    union = union_end - union_start
    
    if union <= 0:
        return 0.0
        
    return intersection / union

def evaluate_temporal_constraint(pred_text, gt_text, video_duration):
    """
    Evaluates temporal constraint.
    """
    gt_intervals, gt_points = parse_timestamps(gt_text)
    pred_intervals, pred_points = parse_timestamps(pred_text)
    
    result = {
        "passed": False,
        "score": 0.0,
        "reason": "",
        "type": "unknown",
        "extracted_pred": pred_text
    }
    
    # Determine Ground Truth Type
    is_interval_task = bool(gt_intervals)
    is_point_task = bool(gt_points) and not is_interval_task 
    
    if not is_interval_task and not is_point_task:
        result["reason"] = f"Could not parse GT timestamp: {gt_text}"
        return result

    if is_interval_task:
        result["type"] = "interval"
        
        # We need at least one predicted interval for an interval task?
        # What if model predicted a point for an interval task? (e.g., center)
        # Strict: Fail.
        
        if not pred_intervals:
             result["reason"] = "No valid time intervals found in prediction"
             return result
             
        # Check against the first GT interval (Simplified)
        # Ideally check if ANY GT interval is satisfied by ANY predicted interval (if multiple exist)
        # Assuming GT usually has 1 target event.
        target_gt = gt_intervals[0] 
        
        max_iou = 0.0
        for p_int in pred_intervals:
            iou = calculate_iou(p_int, target_gt)
            max_iou = max(max_iou, iou)
            
        result["score"] = max_iou
        if max_iou >= 0.5:
            result["passed"] = True
            result["reason"] = f"t-IoU {max_iou:.2f} >= 0.5"
        else:
            result["passed"] = False
            result["reason"] = f"t-IoU {max_iou:.2f} < 0.5"
            
    elif is_point_task:
        result["type"] = "point"
        
        # Tolerance Calculation
        # Dynamic Tolerance = max(1.0, duration * 5%)
        if not video_duration or video_duration <= 0:
             tolerance = 1.0
             base_tol_str = "1.0s (default)"
        else:
             dynamic_tol = video_duration * 0.05
             tolerance = max(1.0, dynamic_tol)
             base_tol_str = f"{tolerance:.2f}s (max(1.0, {video_duration}*5%))"
             
        if not pred_points:
             # What if model predicted an interval for a point? 
             # E.g. GT: 00:05. Pred: 00:04-00:06.
             # We could take the midpoint.
             if pred_intervals:
                  p_start, p_end = pred_intervals[0]
                  pred_points = [(p_start + p_end) / 2]
                  result["reason"] += " (Used midpoint of predicted interval)"
             else:
                  result["reason"] = "No valid time points found in prediction"
                  return result
        
        target_gt = gt_points[0]
        
        min_diff = float('inf')
        for p_pt in pred_points:
            diff = abs(p_pt - target_gt)
            min_diff = min(min_diff, diff)
            
        result["score"] = min_diff
        if min_diff <= tolerance:
            result["passed"] = True
            result["reason"] = f"Diff {min_diff:.2f}s <= Tolerance {base_tol_str}"
        else:
            result["passed"] = False
            result["reason"] = f"Diff {min_diff:.2f}s > Tolerance {base_tol_str}"

    return result
