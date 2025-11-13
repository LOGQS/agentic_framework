"""
Pattern definitions and extraction from LLM output.
"""
from dataclasses import dataclass, field
import json
import re

from .storage import RocksDBStorage
from .core import SegmentType, ToolCall, ExtractedSegments, new_uuid

# Maximum size for JSON parsing to prevent DoS attacks
MAX_JSON_SIZE = 1_000_000  # 1MB limit


@dataclass
class Pattern:
    """Defines a pattern for extracting segments from text."""
    name: str
    start_tag: str
    end_tag: str
    segment_type: SegmentType
    greedy: bool = False


@dataclass
class PatternSet:
    """Collection of patterns with configuration."""
    name: str
    patterns: list[Pattern] = field(default_factory=list)
    default_response_behavior: str = "all_remaining"  # "all_remaining" | "explicit_only"


class PatternRegistry:
    """
    Manages pattern sets stored in RocksDB.
    """

    def __init__(self, storage: RocksDBStorage):
        self._storage = storage
        self._cache: dict[str, PatternSet] = {}

    def register_pattern_set(self, pattern_set: PatternSet) -> None:
        """Register and store pattern set."""
        key = f"pattern:{pattern_set.name}".encode('utf-8')
        value = self._serialize_pattern_set(pattern_set)
        self._storage.put(key, value)
        self._cache[pattern_set.name] = pattern_set

    def get_pattern_set(self, name: str) -> PatternSet | None:
        """Get pattern set by name."""
        if name in self._cache:
            return self._cache[name]

        key = f"pattern:{name}".encode('utf-8')
        value = self._storage.get(key)

        if value is None:
            return None

        pattern_set = self._deserialize_pattern_set(value)
        self._cache[name] = pattern_set
        return pattern_set

    def list_pattern_sets(self) -> list[str]:
        """List all pattern set names."""
        names = []
        for key, _ in self._storage.iterate(b"pattern:"):
            key_str = key.decode('utf-8')
            name = key_str.split(':', 1)[1]
            names.append(name)
        return sorted(names)

    def delete_pattern_set(self, name: str) -> None:
        """Delete pattern set."""
        key = f"pattern:{name}".encode('utf-8')
        self._storage.delete(key)
        if name in self._cache:
            del self._cache[name]

    def _serialize_pattern_set(self, pattern_set: PatternSet) -> bytes:
        """Serialize PatternSet to JSON bytes."""
        data = {
            "name": pattern_set.name,
            "default_response_behavior": pattern_set.default_response_behavior,
            "patterns": [
                {
                    "name": p.name,
                    "start_tag": p.start_tag,
                    "end_tag": p.end_tag,
                    "segment_type": p.segment_type.value,
                    "greedy": p.greedy
                }
                for p in pattern_set.patterns
            ]
        }
        return json.dumps(data).encode('utf-8')

    def _deserialize_pattern_set(self, data: bytes) -> PatternSet:
        """Deserialize JSON bytes to PatternSet."""
        obj = json.loads(data.decode('utf-8'))
        patterns = [
            Pattern(
                name=p["name"],
                start_tag=p["start_tag"],
                end_tag=p["end_tag"],
                segment_type=SegmentType(p["segment_type"]),
                greedy=p.get("greedy", False)
            )
            for p in obj["patterns"]
        ]
        return PatternSet(
            name=obj["name"],
            patterns=patterns,
            default_response_behavior=obj.get("default_response_behavior", "all_remaining")
        )


class PatternExtractor:
    """
    Extracts structured segments from text using patterns.
    """

    def __init__(self, pattern_set: PatternSet):
        self._pattern_set = pattern_set

    def extract(self, text: str, iteration: int = 0, processing_mode=None) -> ExtractedSegments:
        """
        Extract segments from text using configured patterns.

        Note: Currently synchronous regardless of processing_mode.
        Mode parameter reserved for future optimization.
        """
        segments = ExtractedSegments()
        extracted_ranges: list[tuple[int, int]] = []

        for pattern in self._pattern_set.patterns:
            extracted = self._extract_segments(text, pattern)

            for segment_text, start_pos, end_pos in extracted:
                extracted_ranges.append((start_pos, end_pos))

                if pattern.segment_type == SegmentType.TOOL:
                    tool_call = self._parse_tool_call(segment_text, iteration)
                    if tool_call:
                        segments.tools.append(tool_call)

                elif pattern.segment_type == SegmentType.REASONING:
                    segments.reasoning.append(segment_text)

                elif pattern.segment_type == SegmentType.RESPONSE:
                    if segments.response is None:
                        segments.response = segment_text
                    else:
                        segments.response += "\n" + segment_text

        if self._pattern_set.default_response_behavior == "all_remaining" and segments.response is None:
            segments.response = self._extract_remaining(text, extracted_ranges)

        return segments

    def _extract_segments(self, text: str, pattern: Pattern) -> list[tuple[str, int, int]]:
        """Extract segments matching pattern. Returns list of (text, start_pos, end_pos)."""
        results = []

        start_escaped = re.escape(pattern.start_tag)
        end_escaped = re.escape(pattern.end_tag)
        quantifier = ".*" if pattern.greedy else ".*?"
        regex = f"{start_escaped}({quantifier}){end_escaped}"
        flags = re.DOTALL

        for match in re.finditer(regex, text, flags):
            extracted_text = match.group(1).strip()
            start_pos = match.start()
            end_pos = match.end()
            results.append((extracted_text, start_pos, end_pos))

        return results

    def _extract_remaining(self, text: str, extracted_ranges: list[tuple[int, int]]) -> str:
        """Extract text not covered by extracted_ranges."""
        if not extracted_ranges:
            return text.strip()

        extracted_ranges.sort()
        remaining_parts = []
        last_end = 0

        for start, end in extracted_ranges:
            if start > last_end:
                remaining_parts.append(text[last_end:start])
            last_end = max(last_end, end)

        if last_end < len(text):
            remaining_parts.append(text[last_end:])

        return "\n".join(part.strip() for part in remaining_parts if part.strip())

    def _parse_tool_call(self, segment_text: str, iteration: int) -> ToolCall | None:
        """
        Parse tool call from segment.
        Expected format: name: tool_name / arguments: {...}
        """
        try:
            if len(segment_text) > MAX_JSON_SIZE:
                return None

            if segment_text.strip().startswith('{'):
                data = json.loads(segment_text)
                return ToolCall(
                    name=data.get("name", "unknown"),
                    arguments=data.get("arguments", {}),
                    raw_segment=segment_text,
                    iteration=iteration,
                    call_id=data.get("call_id", new_uuid())  # Use provided or generate new
                )

            lines = segment_text.split('\n')
            name = None
            arguments = {}
            in_arguments = False
            arg_lines = []

            for line in lines:
                line = line.strip()

                if line.startswith("name:"):
                    name = line.split("name:", 1)[1].strip()
                elif line.startswith("arguments:"):
                    in_arguments = True
                elif in_arguments:
                    arg_lines.append(line)

            if arg_lines:
                arg_text = '\n'.join(arg_lines)
                if len(arg_text) <= MAX_JSON_SIZE:
                    try:
                        arguments = json.loads(arg_text)
                    except json.JSONDecodeError:
                        arguments = {"raw": arg_text}
                else:
                    arguments = {"error": "Arguments exceed size limit"}

            if name:
                return ToolCall(
                    name=name,
                    arguments=arguments,
                    raw_segment=segment_text,
                    iteration=iteration,
                    call_id=new_uuid()  # Generate unique ID for line format
                )

        except (json.JSONDecodeError, KeyError, ValueError, AttributeError):
            pass

        return None


def create_default_pattern_set() -> PatternSet:
    """
    Create the default pattern set with standard tool, reasoning, and response patterns.
    """
    return PatternSet(
        name="default",
        patterns=[
            Pattern(
                name="tool",
                start_tag="<tool>",
                end_tag="</tool>",
                segment_type=SegmentType.TOOL,
                greedy=False
            ),
            Pattern(
                name="reasoning",
                start_tag="<reasoning>",
                end_tag="</reasoning>",
                segment_type=SegmentType.REASONING,
                greedy=False
            ),
            Pattern(
                name="response",
                start_tag="<response>",
                end_tag="</response>",
                segment_type=SegmentType.RESPONSE,
                greedy=False
            )
        ],
        default_response_behavior="all_remaining"
    )


@dataclass
class _ActivePattern:
    """Tracks an active pattern being streamed."""
    pattern: Pattern
    content_buffer: str = ""
    start_position: int = 0
    has_emitted_start: bool = False


class StreamingPatternExtractor:
    """
    Stateful pattern extractor that processes tokens incrementally.

    Uses regex matching (like batch PatternExtractor) to correctly handle:
    - Multiple instances of same pattern type
    - Nested patterns
    - Proper start/end tag pairing

    Detects patterns as they arrive in the token stream:
    - Detects opening tags (<tool>, <reasoning>, <response>)
    - Streams content immediately after start tag (if enabled)
    - Detects closing tags (</tool>, </reasoning>, </response>)
    - Handles malformed patterns (missing end tags)
    """

    def __init__(self, pattern_set: PatternSet, stream_content: bool = False):
        """
        Initialize streaming pattern extractor.

        Args:
            pattern_set: Pattern definitions to match
            stream_content: If True, emit content before end tag detected
        """
        self._pattern_set = pattern_set
        self._stream_content = stream_content
        self._buffer = ""

        # Track completed patterns by buffer position to avoid re-emission
        self._emitted_complete_patterns: set[tuple[int, int, str]] = set()  # (start, end, pattern_name)

        # Track active (incomplete) patterns for streaming content
        self._active_patterns: dict[tuple[int, str], _ActivePattern] = {}  # (start_pos, pattern_name) -> ActivePattern

        self._completed_segments = ExtractedSegments()
        self._extracted_ranges: list[tuple[int, int]] = []
        self._malformed_patterns: dict[str, str] = {}

        # Pre-compile regexes for efficiency
        self._compiled_regexes: dict[str, re.Pattern] = {}
        for pattern in self._pattern_set.patterns:
            regex_str = self._build_pattern_regex(pattern)
            self._compiled_regexes[pattern.name] = re.compile(regex_str, re.DOTALL)

    def feed_token(self, token: str):
        """
        Feed a token to the extractor.

        Returns iterator of events:
        - ("pattern_start", pattern_name, pattern_type)
        - ("pattern_content", pattern_name, content_chunk)
        - ("pattern_end", pattern_name, pattern_type, full_content, ToolCall|None)
        """
        self._buffer += token

        # Scan for complete patterns using pre-compiled regex
        for pattern in self._pattern_set.patterns:
            compiled_regex = self._compiled_regexes[pattern.name]

            # Find all complete pattern matches in buffer
            # Performance: regex engines are efficient at skipping non-matches
            # and we skip already-emitted patterns via the set check
            for match in compiled_regex.finditer(self._buffer):
                match_key = (match.start(), match.end(), pattern.name)

                # Skip if already emitted
                if match_key in self._emitted_complete_patterns:
                    continue

                # New complete pattern found
                full_content = match.group(1).strip()

                # Check if we already emitted start event for this pattern instance
                active_key = (match.start(), pattern.name)
                already_emitted_start = active_key in self._active_patterns

                if not already_emitted_start:
                    # Emit start event now (pattern just became complete)
                    yield ("pattern_start", pattern.name, pattern.segment_type.value)

                # Parse tool call if this is a tool pattern
                tool_call = None
                if pattern.segment_type == SegmentType.TOOL:
                    tool_call = self._parse_tool_call_safe(full_content, 0)
                    if tool_call:
                        self._completed_segments.tools.append(tool_call)
                elif pattern.segment_type == SegmentType.REASONING:
                    self._completed_segments.reasoning.append(full_content)
                elif pattern.segment_type == SegmentType.RESPONSE:
                    if self._completed_segments.response is None:
                        self._completed_segments.response = full_content
                    else:
                        self._completed_segments.response += "\n" + full_content

                # Emit end event
                yield ("pattern_end", pattern.name, pattern.segment_type.value, full_content, tool_call)

                # Mark as emitted
                self._emitted_complete_patterns.add(match_key)
                self._extracted_ranges.append((match.start(), match.end()))

                # Remove from active if present
                if active_key in self._active_patterns:
                    del self._active_patterns[active_key]

        # Handle streaming content for incomplete patterns
        if self._stream_content:
            for event in self._stream_incomplete_patterns():
                yield event

    def _build_pattern_regex(self, pattern: Pattern) -> str:
        """Build regex for pattern matching (same as batch extractor)."""
        start_escaped = re.escape(pattern.start_tag)
        end_escaped = re.escape(pattern.end_tag)
        quantifier = ".*" if pattern.greedy else ".*?"
        return f"{start_escaped}({quantifier}){end_escaped}"

    def _stream_incomplete_patterns(self):
        """
        Detect and stream content for incomplete patterns (start tag present, no end tag yet).
        Yields pattern_start and pattern_content events.
        """
        for pattern in self._pattern_set.patterns:
            # Find all start tags in buffer
            search_pos = 0
            while True:
                start_pos = self._buffer.find(pattern.start_tag, search_pos)
                if start_pos == -1:
                    break

                # Check if this start tag belongs to a completed pattern
                is_completed = any(
                    start <= start_pos < end
                    for start, end, pname in self._emitted_complete_patterns
                    if pname == pattern.name
                )

                if not is_completed:
                    # This is an incomplete/active pattern
                    active_key = (start_pos, pattern.name)

                    if active_key not in self._active_patterns:
                        # New incomplete pattern detected
                        active = _ActivePattern(
                            pattern=pattern,
                            content_buffer="",
                            start_position=start_pos,
                            has_emitted_start=False
                        )
                        self._active_patterns[active_key] = active

                        # Emit start event
                        yield ("pattern_start", pattern.name, pattern.segment_type.value)
                        active.has_emitted_start = True

                    # Stream content after start tag
                    active = self._active_patterns[active_key]
                    content_start_pos = start_pos + len(pattern.start_tag)
                    current_content = self._buffer[content_start_pos:]

                    # Only emit new content
                    if len(current_content) > len(active.content_buffer):
                        new_content = current_content[len(active.content_buffer):]
                        active.content_buffer = current_content
                        if new_content:
                            yield ("pattern_content", pattern.name, new_content)

                search_pos = start_pos + 1

    def finalize(self, iteration: int = 0) -> tuple[ExtractedSegments, dict[str, str]]:
        """
        Called when token stream ends.

        Returns:
            (ExtractedSegments, malformed_patterns_dict)

        Handles incomplete patterns by discarding them and storing as malformed.
        """
        # Handle any active (incomplete) patterns as malformed
        for (start_pos, pattern_name), active in self._active_patterns.items():
            # Store malformed content with unique key if multiple instances
            if pattern_name in self._malformed_patterns:
                # Multiple incomplete instances - append position to make unique
                key = f"{pattern_name}_{start_pos}"
            else:
                key = pattern_name
            self._malformed_patterns[key] = active.content_buffer

        # Extract remaining text as response (if configured)
        if self._pattern_set.default_response_behavior == "all_remaining":
            if self._completed_segments.response is None:
                remaining = self._extract_remaining_from_buffer()
                if remaining:
                    self._completed_segments.response = remaining

        # Update iteration for all tool calls
        for tool_call in self._completed_segments.tools:
            tool_call.iteration = iteration

        return self._completed_segments, self._malformed_patterns

    def _extract_remaining_from_buffer(self) -> str:
        """Extract text not covered by extracted ranges."""
        if not self._extracted_ranges:
            return self._buffer.strip()

        self._extracted_ranges.sort()
        remaining_parts = []
        last_end = 0

        for start, end in self._extracted_ranges:
            if start > last_end:
                remaining_parts.append(self._buffer[last_end:start])
            last_end = max(last_end, end)

        if last_end < len(self._buffer):
            remaining_parts.append(self._buffer[last_end:])

        return "\n".join(part.strip() for part in remaining_parts if part.strip())

    def _parse_tool_call_safe(self, segment_text: str, iteration: int) -> ToolCall | None:
        """Parse tool call, safely handling errors."""
        try:
            if len(segment_text) > MAX_JSON_SIZE:
                return None

            # Try JSON format first
            if segment_text.strip().startswith('{'):
                data = json.loads(segment_text)
                return ToolCall(
                    name=data.get("name", "unknown"),
                    arguments=data.get("arguments", {}),
                    raw_segment=segment_text,
                    iteration=iteration,
                    call_id=data.get("call_id", new_uuid())  # Use provided or generate new
                )

            # Try line-based format
            lines = segment_text.split('\n')
            name = None
            arguments = {}
            arguments_json_lines = []
            in_arguments_section = False

            for line in lines:
                line = line.strip()

                if line.lower().startswith("name:"):
                    name = line.split(":", 1)[1].strip()
                elif line.lower().startswith("arguments:"):
                    args_value = line.split(":", 1)[1].strip()
                    if args_value.startswith("{"):
                        arguments_json_lines.append(args_value)
                        in_arguments_section = True
                    else:
                        arguments = {}
                elif in_arguments_section:
                    arguments_json_lines.append(line)

            if arguments_json_lines:
                arguments_json = "\n".join(arguments_json_lines)
                if arguments_json and len(arguments_json) <= MAX_JSON_SIZE:
                    try:
                        arguments = json.loads(arguments_json)
                    except json.JSONDecodeError:
                        arguments = {}

            if name:
                return ToolCall(
                    name=name,
                    arguments=arguments,
                    raw_segment=segment_text,
                    iteration=iteration,
                    call_id=new_uuid()  # Generate unique ID for line format
                )

            return None
        except Exception:
            return None
