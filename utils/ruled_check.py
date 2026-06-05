import re
import json
import jsonschema
from jsonschema import Draft7Validator
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
from enum import Enum


class CheckerType(Enum):
    """Checker type enum."""
    PLAIN_TEXT = "plain_text"
    JSON_OBJECT = "json_object"
    JSON_ARRAY = "json_array"
    UNORDERED_LIST = "unordered_list"
    ORDERED_LIST = "ordered_list"
    TABLE = "table"
    KEYWORD = "keyword"
    MARKDOWN = "markdown"
    PREFIX_SUFFIX = "prefix_suffix"
    DELIMITER = "delimiter"
    LENGTH = "length"
    COUNT = "count"
    CASE = "case"
    LANGUAGE = "language"
    TIMESTAMP_FORMAT = "timestamp_format"


class BaseChecker(ABC):
    """Abstract base class for all checkers."""
    
    @abstractmethod
    def check(self, content: str, **kwargs) -> bool:
        """Abstract check method."""
        pass


class PlainTextChecker(BaseChecker):
    """Plain text checker."""
    
    def check(self, content: str, **kwargs) -> bool:
        """Check whether content is plain text (no special structures)."""
        # Check JSON structure
        try:
            json.loads(content)
            return False
        except json.JSONDecodeError:
            pass
        
        # Check list-like structures
        list_patterns = [
            r'^\s*[-*+•]\s+',  # unordered list
            r'^\s*\d+[\.\)]\s+',  # arabic ordered list
            r'^\s*[a-zA-Z][\.\)]\s+',  # alphabetic ordered list
            r'^\s*[\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341]+[\.\u3001]\s+',  # Chinese numerals
            r'^\s*[IVXLCDM]+[\.\)]\s+'  # Roman numerals
        ]
        
        lines = content.split('\n')
        for line in lines:
            for pattern in list_patterns:
                if re.match(pattern, line):
                    return False
        
        # Check Markdown tables
        if '|' in content and re.search(r'\|[-:\s]+\|', content):
            return False
        
        return True


class JSONChecker(BaseChecker):
    """Base class for JSON checkers."""
    
    def _validate_json(self, content: str, schema: Dict[str, Any], expected_type: str) -> bool:
        """Validate JSON against a schema."""
        try:
            data = json.loads(content)
            
            # Validate JSON type
            if expected_type == "object" and not isinstance(data, dict):
                return False
            elif expected_type == "array" and not isinstance(data, list):
                return False
            
            # Validate with jsonschema
            # If schema.items is a list (tuple validation), use Draft7Validator.
            if isinstance(schema.get('items'), list):
                validator = Draft7Validator(schema)
                validator.validate(data)
            else:
                jsonschema.validate(instance=data, schema=schema)
            return True
        except (json.JSONDecodeError, jsonschema.exceptions.ValidationError):
            return False


class JSONObjectChecker(JSONChecker):
    """JSON object checker."""
    
    def check(self, content: str, schema: Dict[str, Any], **kwargs) -> bool:
        first_brace = content.find('{')
        last_brace = content.rfind('}')
        if first_brace == -1 or last_brace == -1 or last_brace < first_brace:
            return False
        content = content[first_brace:last_brace + 1]
        return self._validate_json(content, schema, "object")


class JSONArrayChecker(JSONChecker):
    """JSON array checker."""
    
    def check(self, content: str, schema: Dict[str, Any], **kwargs) -> bool:
        first_bracket = content.find('[')
        last_bracket = content.rfind(']')
        if first_bracket == -1 or last_bracket == -1 or last_bracket < first_bracket:
            return False
        content = content[first_bracket:last_bracket + 1]
        return self._validate_json(content, schema, "array")


class ListChecker(BaseChecker):
    """Base class for list-format checkers."""
    
    def _check_list_format(self, content: str, patterns: List[str], symbol: Optional[str] = None) -> bool:
        """Check list formatting line-by-line."""
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        if not lines:
            return False
        
        if symbol:
            # If a symbol is specified, build a stricter pattern.
            escaped_symbol = re.escape(symbol)
            specific_patterns = [f'^{escaped_symbol}\\s+']
        else:
            specific_patterns = patterns
        
        # Check each line
        for line in lines:
            matched = False
            for pattern in specific_patterns:
                if re.match(pattern, line):
                    matched = True
                    break
            if not matched:
                return False
        
        return True


class UnorderedListChecker(ListChecker):
    """Unordered list checker."""
    
    def check(self, content: str, symbol: Optional[str] = None, **kwargs) -> bool:
        default_patterns = [r'^[-*+•]\s+']
        return self._check_list_format(content, default_patterns, symbol)


class OrderedListChecker(ListChecker):
    """Ordered list checker."""
    
    def check(self, content: str, symbol: Optional[str] = None, **kwargs) -> bool:
        if symbol:
            # If symbol is specified, enforce it and check ordering.
            return self._check_ordered_with_symbol(content, symbol)
        else:
            # Without symbol, require a consistent numbering system and ordering.
            return self._check_consistent_numbering(content)
    
    def _check_ordered_with_symbol(self, content: str, symbol: str) -> bool:
        """Check whether the list starts with the given symbol and is ordered."""
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        if not lines:
            return False
        
        # Analyze symbol structure
        sequence_type, separator = self._analyze_symbol(symbol)
        if not sequence_type or not separator:
            return False
        
        # Extract the first index; it must match the symbol index
        first_number = self._extract_sequence_number(lines[0], sequence_type, separator)
        symbol_number = self._extract_sequence_number(symbol, sequence_type, separator, require_space=False)
        
        if first_number != symbol_number:
            return False
        
        # Check monotonic ordering
        return self._check_sequence_order(lines, sequence_type, separator, first_number)
    
    def _check_consistent_numbering(self, content: str) -> bool:
        """Check whether a consistent numbering system is used and is ordered."""
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        if not lines:
            return False
        
        # Supported numbering systems
        pattern_groups = [
            ([r'^\d+[\.\)\:]\s+'], 'arabic'),
            ([r'^[A-Z][\.\)]\s+'], 'upper_alpha'),
            ([r'^[a-z][\.\)]\s+'], 'lower_alpha'),
            ([r'^[\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341]+[\.\u3001]\s+'], 'chinese'),
            ([r'^[IVXLCDM]+[\.\)]\s+'], 'upper_roman'),
            ([r'^[ivxlcdm]+[\.\)]\s+'], 'lower_roman'),
        ]
        
        # Try each numbering system
        for patterns, pattern_type in pattern_groups:
            if self._check_format_and_order(content, patterns, pattern_type):
                return True
        
        return False
    
    def _check_format_and_order(self, content: str, patterns: List[str], pattern_type: str) -> bool:
        """Check formatting and verify ordering."""
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        if not lines:
            return False
        
        # Check formatting first
        if not self._check_list_format(content, patterns, None):
            return False
        
        # Then check ordering
        # Infer separator from the first line
        separator = self._infer_separator_from_line(lines[0])
        if not separator:
            return False
        
        # Extract the first index
        first_number = self._extract_sequence_number(lines[0], pattern_type, separator)
        if first_number is None:
            return False
        
        return self._check_sequence_order(lines, pattern_type, separator, first_number)
    
    def _get_patterns_from_symbol(self, symbol: str) -> List[str]:
        """Generate matching patterns from a symbol."""
        symbol_stripped = symbol.strip()
        if not symbol_stripped:
            return []
        
        # Analyze symbol structure
        sequence_type, separator = self._analyze_symbol(symbol_stripped)
        if not sequence_type or not separator:
            return []
        
        # Generate corresponding regex patterns
        return self._generate_patterns_for_type(sequence_type, separator)
    
    def _analyze_symbol(self, symbol: str) -> Tuple[Optional[str], Optional[str]]:
        """Analyze symbol and return (sequence_type, separator)."""
        # Supported separators (include fullwidth dot)
        separators = {'.', ')', '\u3001', ':', '\uFF0E'}
        
        # Find separator
        separator = None
        for i in range(len(symbol) - 1, -1, -1):
            if symbol[i] in separators:
                separator = symbol[i]
                sequence_part = symbol[:i].strip()
                break
        
        if not separator:
            return None, None
        
        # Identify sequence type
        sequence_type = self._identify_sequence_type(sequence_part)
        return sequence_type, separator
    
    def _identify_sequence_type(self, text: str) -> Optional[str]:
        """Identify sequence type."""
        if not text:
            return None
        
        # 1) Arabic numerals
        if text.isdigit():
            return 'arabic'
        
        # 2) Chinese numerals
        if re.fullmatch(r'[\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341]+', text):
            return 'chinese'
        
        # 3) Roman numerals
        if self._is_valid_roman_numeral(text):
            if text.isupper():
                return 'upper_roman'
            else:
                return 'lower_roman'
        
        # 4) Uppercase A-Z (single letter)
        if len(text) == 1 and text.isupper() and text.isalpha():
            return 'upper_alpha'
        
        # 5) Lowercase a-z (single letter)
        if len(text) == 1 and text.islower() and text.isalpha():
            return 'lower_alpha'
        
        return None
    
    def _is_valid_roman_numeral(self, text: str) -> bool:
        """Check whether text is a valid Roman numeral (limited set)."""
        if not text or not re.fullmatch(r'[IVXLCDMivxlcdm]+', text):
            return False
        
        # Valid Roman numeral sequence (1-20)
        valid_upper_romans = [
            'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X',
            'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX'
        ]
        valid_lower_romans = [r.lower() for r in valid_upper_romans]
        
        return text in valid_upper_romans or text in valid_lower_romans
    
    def _generate_patterns_for_type(self, sequence_type: str, separator: str) -> List[str]:
        """Generate regex patterns for a specific sequence type."""
        escaped_separator = re.escape(separator)
        
        patterns_map = {
            # Arabic numerals: 1. 2. 3. ...
            'arabic': [r'^\d+' + escaped_separator + r'\s+'],
            
            # A. B. C. ... Z.
            'upper_alpha': [r'^[A-Z]' + escaped_separator + r'\s+'],
            
            # a. b. c. ... z.
            'lower_alpha': [r'^[a-z]' + escaped_separator + r'\s+'],
            
            # I. II. III. IV. ...
            'upper_roman': [r'^[IVXLCDM]+' + escaped_separator + r'\s+'],
            
            # i. ii. iii. iv. ...
            'lower_roman': [r'^[ivxlcdm]+' + escaped_separator + r'\s+'],
            
            # Chinese numerals
            'chinese': [r'^[\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341]+' + escaped_separator + r'\s+'],
        }
        
        return patterns_map.get(sequence_type, [])
    
    def _infer_separator_from_line(self, line: str) -> Optional[str]:
        """Infer separator from a line."""
        separators = {'.', ')', '\u3001', ':', '\uFF0E'}
        for char in line:
            if char in separators:
                return char
        return None
    
    def _extract_sequence_number(self, text: str, sequence_type: str, separator: str, require_space: bool = True) -> Optional[Union[int, str]]:
        """Extract list index from text and validate its format."""
        # Locate separator
        sep_pos = text.find(separator)
        if sep_pos == -1:
            return None
        
        sequence_part = text[:sep_pos].strip()
        if not sequence_part:
            return None
        
        # Require whitespace after separator (optional)
        if require_space:
            after_separator = text[sep_pos + len(separator):]
            if not after_separator or not after_separator[0].isspace():
                return None
        
        if sequence_type == 'arabic':
            try:
                return int(sequence_part)
            except ValueError:
                return None
        elif sequence_type in ['upper_alpha', 'lower_alpha']:
            if len(sequence_part) == 1 and sequence_part.isalpha():
                return sequence_part
            return None
        elif sequence_type in ['upper_roman', 'lower_roman']:
            return sequence_part  # Roman numerals
        elif sequence_type == 'chinese':
            return sequence_part  # Chinese numerals
        return None
    
    def _check_sequence_order(self, lines: List[str], sequence_type: str, separator: str, start_number: Union[int, str]) -> bool:
        """Check whether indices increase sequentially."""
        if sequence_type == 'arabic':
            return self._check_arabic_order(lines, separator, int(start_number))
        elif sequence_type == 'upper_alpha':
            return self._check_alpha_order(lines, separator, start_number, True)
        elif sequence_type == 'lower_alpha':
            return self._check_alpha_order(lines, separator, start_number, False)
        elif sequence_type == 'upper_roman':
            return self._check_roman_order(lines, separator, start_number, True)
        elif sequence_type == 'lower_roman':
            return self._check_roman_order(lines, separator, start_number, False)
        elif sequence_type == 'chinese':
            return self._check_chinese_order(lines, separator, start_number)
        return False
    
    def _check_arabic_order(self, lines: List[str], separator: str, start_num: int) -> bool:
        """Check ordering for Arabic numerals."""
        expected = start_num
        for line in lines:
            current = self._extract_sequence_number(line, 'arabic', separator)
            if current != expected:
                return False
            expected += 1
        return True
    
    def _check_alpha_order(self, lines: List[str], separator: str, start_char: str, is_upper: bool) -> bool:
        """Check ordering for alphabetic indices."""
        if is_upper:
            start_ord = ord(start_char.upper())
            sequence_type = 'upper_alpha'
        else:
            start_ord = ord(start_char.lower())
            sequence_type = 'lower_alpha'
        
        expected_ord = start_ord
        for line in lines:
            current = self._extract_sequence_number(line, sequence_type, separator)
            if not current or ord(current) != expected_ord:
                return False
            expected_ord += 1
            if expected_ord > ord('Z') and is_upper:
                return False
            if expected_ord > ord('z') and not is_upper:
                return False
        return True
    
    def _check_roman_order(self, lines: List[str], separator: str, start_roman: str, is_upper: bool) -> bool:
        """Check ordering for Roman numerals."""
        # Simplified Roman numeral ordering
        roman_sequence = self._get_roman_sequence(is_upper)
        try:
            start_index = roman_sequence.index(start_roman)
        except ValueError:
            return False
        
        sequence_type = 'upper_roman' if is_upper else 'lower_roman'
        expected_index = start_index
        for line in lines:
            current = self._extract_sequence_number(line, sequence_type, separator)
            if expected_index >= len(roman_sequence) or current != roman_sequence[expected_index]:
                return False
            expected_index += 1
        return True
    
    def _check_chinese_order(self, lines: List[str], separator: str, start_chinese: str) -> bool:
        """Check ordering for Chinese numerals (1-10)."""
        chinese_sequence = ['\u4e00', '\u4e8c', '\u4e09', '\u56db', '\u4e94', '\u516d', '\u4e03', '\u516b', '\u4e5d', '\u5341']
        try:
            start_index = chinese_sequence.index(start_chinese)
        except ValueError:
            return False
        
        expected_index = start_index
        for line in lines:
            current = self._extract_sequence_number(line, 'chinese', separator)
            if expected_index >= len(chinese_sequence) or current != chinese_sequence[expected_index]:
                return False
            expected_index += 1
        return True
    
    def _get_roman_sequence(self, is_upper: bool) -> List[str]:
        """Get Roman numeral sequence."""
        if is_upper:
            return ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X',
                    'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX']
        else:
            return ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x',
                    'xi', 'xii', 'xiii', 'xiv', 'xv', 'xvi', 'xvii', 'xviii', 'xix', 'xx']


class TableChecker(BaseChecker):
    """Markdown table checker."""
    
    def _clean_markdown_formatting(self, text: str) -> str:
        """Remove Markdown formatting and keep plain text."""
        # Bold: **text** or __text__
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'__(.*?)__', r'\1', text)
        
        # Italic: *text* or _text_ (avoid bold conflicts)
        text = re.sub(r'(?<!\*)\*([^*]+?)\*(?!\*)', r'\1', text)
        text = re.sub(r'(?<!_)_([^_]+?)_(?!_)', r'\1', text)
        
        # Highlight: ==text==
        text = re.sub(r'==(.*?)==', r'\1', text)
        
        # Inline code: `text`
        text = re.sub(r'`(.*?)`', r'\1', text)
        
        # Strikethrough: ~~text~~
        text = re.sub(r'~~(.*?)~~', r'\1', text)
        
        return text.strip()
    
    def check(self, content: str, col_name: List[str], **kwargs) -> bool:
        lines = content.strip().split('\n')
        if len(lines) < 2:  # header + separator row
            return False
        
        # Markdown table shape
        if not all('|' in line for line in lines[:2]):
            return False
        
        # Separator row
        if not re.match(r'^[\s\|:\-]+$', lines[1]):
            return False
        
        # Extract header and remove Markdown formatting
        header_cells = [cell.strip() for cell in lines[0].split('|') if cell.strip()]
        cleaned_header_cells = [self._clean_markdown_formatting(cell) for cell in header_cells]

        col_name = [self._clean_markdown_formatting(cell) for cell in col_name]

        # Compare cleaned column names
        return cleaned_header_cells == col_name

class KeywordChecker(BaseChecker):
    """Keyword checker."""
    
    def check(self, content: str, keyword: str, keyword_type: str, **kwargs) -> bool:
        """Check whether content includes/excludes a keyword."""
        if not keyword:
            return True

        content = content.lower()
        keyword = keyword.lower()
        
        if keyword_type == "include":
            return keyword in content
        elif keyword_type == "exclude":
            return keyword not in content
        return False


class MarkdownChecker(BaseChecker):
    """Markdown style checker."""
    
    MARKDOWN_PATTERNS = {
        'title': [r'^#{1,6}\s+.+$'],
        'bold': [r'\*\*.+\*\*', r'__.+__'],
        'highlight': [r'==.+==', r'`.+`'],
        'italic': [r'\*.+\*', r'_.+_'],
        'code': [r'```[\s\S]*```', r'`.+`']
    }
    
    def check(self, content: str, md_type: str, **kwargs) -> bool:
        if md_type not in self.MARKDOWN_PATTERNS:
            return False

        patterns = self.MARKDOWN_PATTERNS[md_type]
        for pattern in patterns:
            if re.search(pattern, content, re.MULTILINE):
                return True
        return False


class PrefixSuffixChecker(BaseChecker):
    """Prefix/suffix checker."""
    
    def check(self, content: str, prefix: Optional[str] = None, suffix: Optional[str] = None, **kwargs) -> bool:
        if prefix and not content.startswith(prefix):
            return False
        if suffix:
            # Allow punctuation after suffix
            import string
            # Common punctuation (include CJK punctuation via Unicode escapes)
            punctuation = string.punctuation + "\uFF0C\u3002\uFF01\uFF1F\uFF1B\uFF1A\u3001\"\"\uFF08\uFF09\u3010\u3011\u300A\u300B\u3008\u3009\u00B7"
            
            # Check suffix match
            if content.endswith(suffix):
                return True
            
            # Check suffix + trailing punctuation
            for i in range(len(content) - 1, -1, -1):
                if content[i] in punctuation:
                    continue
                else:
                    # First non-punctuation position
                    potential_suffix_end = i + 1
                    if content[:potential_suffix_end].endswith(suffix):
                        return True
                    break
            return False
        return True


class DelimiterChecker(BaseChecker):
    """Delimiter checker."""
    
    def check(self, content: str, symbol: str, **kwargs) -> bool:
        """
        Check whether a specified delimiter is used to separate content:
        1) content contains the delimiter
        2) delimiter splits content into at least two non-empty parts
        """
        # Contains delimiter
        if symbol not in content:
            return False
        
        # Verify it actually separates content
        parts = content.split(symbol)
        # At least two non-empty parts
        non_empty_parts = [part.strip() for part in parts if part.strip()]
        return len(non_empty_parts) >= 2


class LengthChecker(BaseChecker):
    """Length checker."""
    
    def _remove_list_prefixes(self, content: str) -> str:
        """Remove ordered/unordered list prefixes before counting."""
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # List prefix regex patterns
            list_patterns = [
                r'^\s*[-*+•]\s+',  # unordered list: - * + •
                r'^\s*\d+[\.\)]\s+',  # arabic ordered list: 1. 2) ...
                r'^\s*[a-zA-Z][\.\)]\s+',  # alphabetic ordered list: A. b) ...
                r'^\s*[\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341]+[\.\u3001]\s+',  # Chinese numerals
                r'^\s*[IVXLCDM]+[\.\)]\s+',  # uppercase Roman numerals
                r'^\s*[ivxlcdm]+[\.\)]\s+'  # lowercase Roman numerals
            ]
            
            # If matches any list prefix pattern, remove it
            cleaned_line = line
            for pattern in list_patterns:
                match = re.match(pattern, line)
                if match:
                    # Remove prefix and keep the rest
                    cleaned_line = line[match.end():]
                    break
            
            cleaned_lines.append(cleaned_line)
        
        return '\n'.join(cleaned_lines)
    
    def check(self, content: str, unit: str, min_len: int = 0, max_len: int = -1, **kwargs) -> bool:
        # Remove list prefixes before length calculation
        cleaned_content = self._remove_list_prefixes(content)
        
        if unit == "word":
            # Mixed CJK/English word counting
            chinese_words = len(re.findall(r'[\u4e00-\u9fa5]', cleaned_content))
            english_words = len(re.findall(r'\b[a-zA-Z]+(?:-[a-zA-Z]+)*\b', cleaned_content))
            count = chinese_words + english_words
        elif unit == "sentence":
            # Sentence counting
            count = len(re.findall(r'[.!?\u3002\uFF01\uFF1F]+', cleaned_content))
        elif unit == "paragraph":
            # Paragraph counting
            paragraphs = [p.strip() for p in cleaned_content.split('\n\n') if p.strip()]
            count = len(paragraphs)
        elif unit == "character":
            # Character counting
            count = len(cleaned_content.replace(" ", ""))
        else:
            return False

        if count < min_len:
            return False
        if max_len > 0 and count > max_len:
            return False
        return True

class CountChecker(BaseChecker):
    """Count checker."""
    
    def check(self, content: str, min_count: int = 0, max_count: int = -1, **kwargs) -> bool:
        # Count parenthesis pairs like (xxx)
        import re
        parentheses_pattern = r'\([^)]*\)'
        matches = re.findall(parentheses_pattern, content)
        count = len(matches)
        
        if count < min_count:
            return False
        if max_count > 0 and count > max_count:
            return False
        return True

class CaseChecker(BaseChecker):
    """Letter case checker."""
    
    def check(self, content: str, case_type: str, **kwargs) -> bool:
        # Only check English letters
        english_chars = re.findall(r'\b[a-zA-Z]+\b', content)
        if not english_chars:
            return True
        
        english_text = ' '.join(english_chars)
        
        if case_type == "upper":
            return english_text.isupper()
        elif case_type == "lower":
            return english_text.islower()
        elif case_type == "title":
            upper = 0
            # Allow acronyms (all-caps words)
            for word in english_chars:
                if word.isupper():
                    upper += 1
                    continue
                if len(word) > 1 and word[0].isupper() and word[1:].islower():
                    continue
                if len(word) == 1 and word.isupper():
                    continue
                return False
            if upper == len(english_chars):
                return False
            return True
        return False


class LanguageChecker(BaseChecker):
    """Language checker."""
    
    def check(self, content: str, lang_type: str, **kwargs) -> bool:
        # Only count English letters and CJK characters (ignore digits/punctuation)
        english_chars = len(re.findall(r'[a-zA-Z]', content))
        chinese_chars = len(re.findall(r'[\u4e00-\u9fa5]', content))
        
        # If there are no language characters
        if english_chars == 0 and chinese_chars == 0:
            return False
        
        if lang_type == "en":
            # All English (no CJK)
            return chinese_chars == 0 and english_chars > 0
        elif lang_type == "zh":
            # All CJK (no English)
            return english_chars == 0 and chinese_chars > 0
        return False


class TimestampFormatChecker(BaseChecker):
    """Timestamp format checker."""
    
    def check(self, content: str, format_type: str, **kwargs) -> bool:
        if format_type == 'point':
            # Timestamps like [MM:SS]
            pattern = r'\[\d{2}:\d{2}\]'
            return bool(re.search(pattern, content))
        elif format_type == 'period':
            # Periods like [MM:SS-MM:SS], allow spaces around '-'
            pattern = r'\[\d{2}:\d{2}\s*-\s*\d{2}:\d{2}\]'
            return bool(re.search(pattern, content))
        return False


class FormatCheckModule:
    """Main format-check module."""
    
    def __init__(self):
        self._checkers = {
            CheckerType.PLAIN_TEXT: PlainTextChecker(),
            CheckerType.JSON_OBJECT: JSONObjectChecker(),
            CheckerType.JSON_ARRAY: JSONArrayChecker(),
            CheckerType.UNORDERED_LIST: UnorderedListChecker(),
            CheckerType.ORDERED_LIST: OrderedListChecker(),
            CheckerType.TABLE: TableChecker(),
            CheckerType.KEYWORD: KeywordChecker(),
            CheckerType.MARKDOWN: MarkdownChecker(),
            CheckerType.PREFIX_SUFFIX: PrefixSuffixChecker(),
            CheckerType.DELIMITER: DelimiterChecker(),
            CheckerType.LENGTH: LengthChecker(),
            CheckerType.COUNT: CountChecker(),
            CheckerType.CASE: CaseChecker(),
            CheckerType.LANGUAGE: LanguageChecker(),
            CheckerType.TIMESTAMP_FORMAT: TimestampFormatChecker()
        }
    
    def plain_text(self, content: str) -> bool:
        return self._checkers[CheckerType.PLAIN_TEXT].check(content)
    
    def json_object(self, content: str, schema: Dict[str, Any]) -> bool:
        return self._checkers[CheckerType.JSON_OBJECT].check(content, schema=schema)
    
    def json_array(self, content: str, schema: Dict[str, Any]) -> bool:
        return self._checkers[CheckerType.JSON_ARRAY].check(content, schema=schema)
    
    def unordered_list(self, content: str, symbol: Optional[str] = None) -> bool:
        return self._checkers[CheckerType.UNORDERED_LIST].check(content, symbol=symbol)
    
    def ordered_list(self, content: str, symbol: Optional[str] = None) -> bool:
        return self._checkers[CheckerType.ORDERED_LIST].check(content, symbol=symbol)
    
    def table(self, content: str, col_name: List[str]) -> bool:
        return self._checkers[CheckerType.TABLE].check(content, col_name=col_name)
    
    def keyword(self, content: str, keyword: str, keyword_type: str) -> bool:
        return self._checkers[CheckerType.KEYWORD].check(content, keyword=keyword, keyword_type=keyword_type)
    
    def markdown(self, content: str, md_type: str) -> bool:
        return self._checkers[CheckerType.MARKDOWN].check(content, md_type=md_type)
    
    def prefix_suffix(self, content: str, prefix: Optional[str] = None, suffix: Optional[str] = None) -> bool:
        return self._checkers[CheckerType.PREFIX_SUFFIX].check(content, prefix=prefix, suffix=suffix)
    
    def delimiter(self, content: str, symbol: str) -> bool:
        return self._checkers[CheckerType.DELIMITER].check(content, symbol=symbol)
    
    def length(self, content: str, unit: str, min_len: int = 0, max_len: int = -1) -> bool:
        return self._checkers[CheckerType.LENGTH].check(content, unit=unit, min_len=min_len, max_len=max_len)

    def count(self, content: str, min_count: int = 0, max_count: int = -1) -> bool:
        return self._checkers[CheckerType.COUNT].check(content, min_count=min_count, max_count=max_count)

    def case(self, content: str, case_type: str) -> bool:
        return self._checkers[CheckerType.CASE].check(content, case_type=case_type)
    
    def language(self, content: str, lang_type: str) -> bool:
        return self._checkers[CheckerType.LANGUAGE].check(content, lang_type=lang_type)

    def timestamp_format(self, content: str, format_type: str) -> bool:
        return self._checkers[CheckerType.TIMESTAMP_FORMAT].check(content, format_type=format_type)


# Backward-compatible alias (old name)
RuledCheckModule = FormatCheckModule