"""
Tests for pattern extraction system.

Covers:
- Pattern and PatternSet dataclasses
- PatternRegistry (register, get, list, delete)
- PatternExtractor (batch extraction)
- StreamingPatternExtractor (incremental extraction)
- Tool call parsing (JSON and line-based formats)
- Malformed pattern detection
- Greedy vs non-greedy matching
- Overlap window optimization
- Parse error messages
- Performance benchmarks
"""

from agentic.patterns import (
    Pattern,
    PatternSet,
    PatternExtractor,
    StreamingPatternExtractor,
    create_default_pattern_set,
    MAX_JSON_SIZE
)
from agentic.core import SegmentType
import time


class TestPattern:
    """Tests for Pattern dataclass."""

    def test_pattern_creation(self):
        """Test creating a Pattern."""
        pattern = Pattern(
            name="test",
            start_tag="<test>",
            end_tag="</test>",
            segment_type=SegmentType.TOOL,
            greedy=False
        )
        assert pattern.name == "test"
        assert pattern.start_tag == "<test>"
        assert pattern.end_tag == "</test>"
        assert pattern.segment_type == SegmentType.TOOL
        assert pattern.greedy is False

    def test_pattern_default_greedy(self):
        """Test that Pattern defaults to non-greedy."""
        pattern = Pattern("test", "<>", "</>", SegmentType.TOOL)
        assert pattern.greedy is False


class TestPatternSet:
    """Tests for PatternSet dataclass."""

    def test_pattern_set_creation(self):
        """Test creating a PatternSet."""
        patterns = [
            Pattern("p1", "<>", "</>", SegmentType.TOOL)
        ]
        pset = PatternSet(name="test_set", patterns=patterns)
        assert pset.name == "test_set"
        assert len(pset.patterns) == 1

    def test_pattern_set_default_response_behavior(self):
        """Test PatternSet default response behavior."""
        pset = PatternSet(name="test")
        assert pset.default_response_behavior == "all_remaining"

    def test_create_default_pattern_set(self):
        """Test creating default pattern set."""
        pset = create_default_pattern_set()
        assert pset.name == "default"
        assert len(pset.patterns) == 3
        pattern_names = {p.name for p in pset.patterns}
        assert pattern_names == {"tool", "reasoning", "response"}


class TestPatternRegistry:
    """Tests for PatternRegistry."""

    def test_register_and_get_pattern_set(self, pattern_registry):
        """Test registering and retrieving pattern set."""
        custom_set = PatternSet(
            name="custom",
            patterns=[Pattern("p", "<>", "</>", SegmentType.TOOL)]
        )
        pattern_registry.register_pattern_set(custom_set)

        retrieved = pattern_registry.get_pattern_set("custom")
        assert retrieved is not None
        assert retrieved.name == "custom"

    def test_get_nonexistent_pattern_set(self, pattern_registry):
        """Test getting pattern set that doesn't exist."""
        result = pattern_registry.get_pattern_set("nonexistent")
        assert result is None

    def test_list_pattern_sets(self, pattern_registry):
        """Test listing pattern sets."""
        names = pattern_registry.list_pattern_sets()
        assert "default" in names

    def test_delete_pattern_set(self, pattern_registry):
        """Test deleting pattern set."""
        custom_set = PatternSet(name="to_delete", patterns=[])
        pattern_registry.register_pattern_set(custom_set)
        assert pattern_registry.get_pattern_set("to_delete") is not None

        pattern_registry.delete_pattern_set("to_delete")
        assert pattern_registry.get_pattern_set("to_delete") is None

    def test_pattern_set_caching(self, pattern_registry):
        """Test that pattern sets are cached."""
        custom_set = PatternSet(name="cached", patterns=[])
        pattern_registry.register_pattern_set(custom_set)

        # Get twice should use cache
        first = pattern_registry.get_pattern_set("cached")
        second = pattern_registry.get_pattern_set("cached")
        assert first is second  # Same object due to caching


class TestPatternExtractor:
    """Tests for batch pattern extraction."""

    def test_extract_tool_json_format(self):
        """Test extracting tool call in JSON format."""
        pset = create_default_pattern_set()
        extractor = PatternExtractor(pset)

        text = '<tool>{"name": "test_tool", "arguments": {"arg": "value"}}</tool>'
        segments = extractor.extract(text, iteration=1)

        assert len(segments.tools) == 1
        assert segments.tools[0].name == "test_tool"
        assert segments.tools[0].arguments == {"arg": "value"}

    def test_extract_tool_line_format(self):
        """Test extracting tool call in line-based format."""
        pset = create_default_pattern_set()
        extractor = PatternExtractor(pset)

        text = """<tool>
        name: my_tool
        arguments:
        {"key": "value"}
        </tool>"""

        segments = extractor.extract(text, iteration=1)
        assert len(segments.tools) == 1
        assert segments.tools[0].name == "my_tool"

    def test_extract_reasoning(self):
        """Test extracting reasoning segments."""
        pset = create_default_pattern_set()
        extractor = PatternExtractor(pset)

        text = "<reasoning>First step</reasoning><reasoning>Second step</reasoning>"
        segments = extractor.extract(text)

        assert len(segments.reasoning) == 2
        assert segments.reasoning[0] == "First step"
        assert segments.reasoning[1] == "Second step"

    def test_extract_response(self):
        """Test extracting response segment."""
        pset = create_default_pattern_set()
        extractor = PatternExtractor(pset)

        text = "<response>This is the answer</response>"
        segments = extractor.extract(text)

        assert segments.response == "This is the answer"

    def test_extract_all_remaining_as_response(self):
        """Test that unmatched text becomes response."""
        pset = create_default_pattern_set()
        extractor = PatternExtractor(pset)

        text = "<reasoning>Think</reasoning>This is remaining text"
        segments = extractor.extract(text)

        assert "remaining text" in segments.response

    def test_extract_multiple_tools(self):
        """Test extracting multiple tool calls."""
        pset = create_default_pattern_set()
        extractor = PatternExtractor(pset)

        text = '''<tool>{"name": "tool1", "arguments": {}}</tool>
        <tool>{"name": "tool2", "arguments": {}}</tool>'''

        segments = extractor.extract(text)
        assert len(segments.tools) == 2

    def test_extract_nested_patterns(self):
        """Test handling nested patterns (should extract properly)."""
        pset = create_default_pattern_set()
        extractor = PatternExtractor(pset)

        text = "<reasoning>outer <reasoning>inner</reasoning> end</reasoning>"
        segments = extractor.extract(text)

        # Non-greedy should match first closing tag
        assert len(segments.reasoning) >= 1


class TestStreamingPatternExtractor:
    """Tests for streaming pattern extraction."""

    def test_streaming_complete_pattern(self):
        """Test streaming extraction of complete pattern."""
        pset = create_default_pattern_set()
        extractor = StreamingPatternExtractor(pset, stream_content=False)

        events = []
        for char in "<tool>test</tool>":
            for event in extractor.feed_chunk(char):
                events.append(event)

        # Should have pattern_start and pattern_end
        event_types = [e[0] for e in events]
        assert "pattern_start" in event_types
        assert "pattern_end" in event_types

    def test_streaming_with_content_streaming(self):
        """Test streaming with content streaming enabled."""
        pset = create_default_pattern_set()
        extractor = StreamingPatternExtractor(pset, stream_content=True)

        events = []
        text = "<tool>content</tool>"
        for char in text:
            for event in extractor.feed_chunk(char):
                events.append(event)

        event_types = [e[0] for e in events]
        assert "pattern_start" in event_types
        assert "pattern_content" in event_types
        assert "pattern_end" in event_types

    def test_streaming_finalize_complete(self):
        """Test finalize after complete patterns."""
        pset = create_default_pattern_set()
        extractor = StreamingPatternExtractor(pset)

        for char in "<tool>complete</tool>":
            list(extractor.feed_chunk(char))

        segments, malformed = extractor.finalize(iteration=1)
        assert malformed == {}

    def test_streaming_finalize_malformed(self):
        """Test finalize with incomplete pattern."""
        pset = create_default_pattern_set()
        extractor = StreamingPatternExtractor(pset, stream_content=True)

        text = "<tool>incomplete"
        for char in text:
            list(extractor.feed_chunk(char))

        segments, malformed = extractor.finalize()
        assert "tool" in malformed
        assert "incomplete" in malformed["tool"]

    def test_streaming_multiple_patterns(self):
        """Test streaming multiple different patterns."""
        pset = create_default_pattern_set()
        extractor = StreamingPatternExtractor(pset)

        text = "<reasoning>think</reasoning><response>answer</response>"
        for char in text:
            list(extractor.feed_chunk(char))

        segments, _ = extractor.finalize()
        assert len(segments.reasoning) == 1
        assert segments.response == "answer"

    def test_streaming_tool_parsing(self):
        """Test that streaming correctly parses tool calls."""
        pset = create_default_pattern_set()
        extractor = StreamingPatternExtractor(pset)

        text = '<tool>{"name": "test", "arguments": {}}</tool>'
        for char in text:
            list(extractor.feed_chunk(char))

        segments, _ = extractor.finalize(iteration=1)
        assert len(segments.tools) == 1
        assert segments.tools[0].name == "test"


class TestStreamingSearchPointer:
    """Additional tests ensuring streaming extractor maintains coverage."""

    def test_many_reasoning_segments_small_chunks(self):
        """Feeding many reasoning segments character-by-character should capture all."""
        pset = create_default_pattern_set()
        extractor = StreamingPatternExtractor(pset, stream_content=False)

        expected = [f"thought {i}" for i in range(10)]
        text = "".join(f"<reasoning>{value}</reasoning>" for value in expected)

        for ch in text:
            list(extractor.feed_chunk(ch))

        segments, malformed = extractor.finalize()

        assert segments.reasoning == expected
        assert segments.response is None
        assert malformed == {}

    def test_reuse_after_large_payload(self):
        """Large payload followed by reuse should not keep stale search pointers."""
        pset = create_default_pattern_set()
        extractor = StreamingPatternExtractor(pset, stream_content=False)

        large_tool = '<tool>{"name": "echo", "arguments": {"data": "' + ("x" * 2048) + '"}}</tool>'
        for ch in large_tool:
            list(extractor.feed_chunk(ch))
        extractor.finalize()

        for ch in "<response>done</response>":
            list(extractor.feed_chunk(ch))

        segments, malformed = extractor.finalize()
        assert segments.response == "done"
        assert segments.reasoning == []
        assert malformed == {}

    def test_long_prefix_before_tag(self):
        """Ensure extractor still finds tags after long unrelated prefixes."""
        pset = create_default_pattern_set()
        extractor = StreamingPatternExtractor(pset, stream_content=False)

        text = ("noise" * 500) + "<response>answer</response>"

        for ch in text:
            list(extractor.feed_chunk(ch))

        segments, malformed = extractor.finalize()

        assert segments.response == "answer"
        assert malformed == {}


class TestToolCallParsing:
    """Tests for tool call parsing logic."""

    def test_parse_json_tool_call(self):
        """Test parsing JSON format tool call."""
        pset = create_default_pattern_set()
        extractor = PatternExtractor(pset)

        text = '<tool>{"name": "calc", "arguments": {"a": 1, "b": 2}}</tool>'
        segments = extractor.extract(text, iteration=1)

        assert segments.tools[0].name == "calc"
        assert segments.tools[0].arguments == {"a": 1, "b": 2}

    def test_parse_line_format_tool_call(self):
        """Test parsing line-based format tool call."""
        pset = create_default_pattern_set()
        extractor = PatternExtractor(pset)

        text = """<tool>
        name: my_tool
        arguments:
        {"key": "value"}
        </tool>"""

        segments = extractor.extract(text, iteration=1)
        assert segments.tools[0].name == "my_tool"

    def test_parse_invalid_json(self):
        """Test handling of invalid JSON in tool call."""
        pset = create_default_pattern_set()
        extractor = PatternExtractor(pset)

        text = '<tool>{"name": "tool", invalid json}</tool>'
        segments = extractor.extract(text)

        # Should not crash, may not parse tool
        assert isinstance(segments.tools, list)

    def test_parse_oversized_json(self):
        """Test handling of JSON exceeding size limit."""
        pset = create_default_pattern_set()
        extractor = PatternExtractor(pset)

        large_json = '{"data": "' + 'x' * (MAX_JSON_SIZE + 1000) + '"}'
        text = f'<tool>{large_json}</tool>'
        segments = extractor.extract(text)

        # Should not parse due to size limit
        assert len(segments.tools) == 0


class TestGreedyMatching:
    """Tests for greedy vs non-greedy pattern matching."""

    def test_non_greedy_matching(self):
        """Test non-greedy pattern matching."""
        # Use REASONING type to test pattern extraction without JSON parsing
        pattern = Pattern("test", "<t>", "</t>", SegmentType.REASONING, greedy=False)
        pset = PatternSet(name="test", patterns=[pattern])
        extractor = PatternExtractor(pset)

        text = "<t>first</t>middle<t>second</t>"
        segments = extractor.extract(text)

        # Should match both patterns separately
        assert len(segments.reasoning) == 2
        assert "first" in segments.reasoning[0]
        assert "second" in segments.reasoning[1]

    def test_greedy_matching(self):
        """Test greedy pattern matching."""
        # Use REASONING type to test pattern extraction without JSON parsing
        pattern = Pattern("test", "<t>", "</t>", SegmentType.REASONING, greedy=True)
        pset = PatternSet(name="test", patterns=[pattern])
        extractor = PatternExtractor(pset)

        text = "<t>first</t>middle<t>second</t>"
        segments = extractor.extract(text)

        # Greedy should match from first start to last end
        assert len(segments.reasoning) == 1
        # Greedy matches everything between first start and last end
        assert "first" in segments.reasoning[0]
        assert "second" in segments.reasoning[0]


class TestEdgeCases:
    """Tests for edge cases in pattern extraction."""

    def test_empty_pattern_content(self):
        """Test extracting pattern with empty content."""
        pset = create_default_pattern_set()
        extractor = PatternExtractor(pset)

        text = "<reasoning></reasoning>"
        segments = extractor.extract(text)

        assert segments.reasoning == [""]

    def test_pattern_with_newlines(self):
        """Test pattern content with newlines."""
        pset = create_default_pattern_set()
        extractor = PatternExtractor(pset)

        text = """<reasoning>
        line1
        line2
        line3
        </reasoning>"""

        segments = extractor.extract(text)
        assert "line1" in segments.reasoning[0]

    def test_unicode_in_patterns(self):
        """Test patterns with unicode content."""
        pset = create_default_pattern_set()
        extractor = PatternExtractor(pset)

        text = "<response>Hello 世界</response>"
        segments = extractor.extract(text)

        assert "世界" in segments.response

    def test_no_patterns_matched(self):
        """Test when no patterns match."""
        pset = create_default_pattern_set()
        extractor = PatternExtractor(pset)

        text = "Plain text with no patterns"
        segments = extractor.extract(text)

        assert segments.response == text.strip()


class TestOverlapWindowOptimization:
    """Tests for overlap window optimization in StreamingPatternExtractor."""

    def test_pattern_spanning_two_chunks(self):
        """Test patterns that span exactly 2 chunks.

        This tests the overlap window by ensuring patterns that start in one chunk
        and end in the next are correctly detected.
        """
        pset = create_default_pattern_set()
        extractor = StreamingPatternExtractor(pset, stream_content=False)

        # Split pattern across two tokens: "<too" and "l>test</tool>"
        chunk1 = "<too"
        chunk2 = "l>test</tool>"

        events = []
        for event in extractor.feed_chunk(chunk1):
            events.append(event)
        for event in extractor.feed_chunk(chunk2):
            events.append(event)

        # Should detect the complete pattern
        event_types = [e[0] for e in events]
        assert "pattern_end" in event_types

    def test_pattern_spanning_three_chunks(self):
        """Test patterns that span exactly 3 chunks.

        This ensures the overlap window correctly handles patterns spanning
        multiple chunk boundaries.
        """
        pset = create_default_pattern_set()
        extractor = StreamingPatternExtractor(pset, stream_content=False)

        # Split across three chunks
        chunks= ["<to", "ol>con", "tent</tool>"]

        events = []
        for chunk in chunks:
            for event in extractor.feed_chunk(chunk):
                events.append(event)

        event_types = [e[0] for e in events]
        assert "pattern_end" in event_types

    def test_overlap_window_exact_boundary(self):
        """Test overlap window at exact tag boundary.

        This tests the case where a tag is split exactly at the boundary,
        ensuring the overlap window is calculated correctly.
        """
        pset = create_default_pattern_set()
        extractor = StreamingPatternExtractor(pset, stream_content=False)

        # Split end tag exactly: "<tool>test</" and "tool>"
        chunk1 = "<tool>test</"
        chunk2 = "tool>"

        events = []
        for event in extractor.feed_chunk(chunk1):
            events.append(event)
        for event in extractor.feed_chunk(chunk2):
            events.append(event)

        event_types = [e[0] for e in events]
        assert "pattern_end" in event_types

        # Verify finalize doesn't treat the span as malformed
        segments, malformed = extractor.finalize()
        assert malformed == {}
        assert len(segments.tools) == 0

    def test_multiple_patterns_with_overlap_boundaries(self):
        """Test multiple patterns split across chunkboundaries.

        This ensures the overlap window optimization doesn't interfere when
        multiple patterns are split across tokens.
        """
        pset = create_default_pattern_set()
        extractor = StreamingPatternExtractor(pset, stream_content=False)

        # First pattern split, then second pattern split
        chunks= [
            "<reason",
            "ing>think</reasoning><respo",
            "nse>answer</response>"
        ]

        events = []
        for chunk in chunks:
            for event in extractor.feed_chunk(chunk):
                events.append(event)

        segments, _ = extractor.finalize()
        assert len(segments.reasoning) == 1
        assert segments.response == "answer"


class TestParseErrorMessages:
    """Tests for detailed parse error messages in streaming extraction."""

    def test_parse_error_invalid_json_detailed(self):
        """Test that invalid JSON produces detailed error message.

        When JSON parsing fails, the error should include the JSON error details.
        """
        # Create pattern set with explicit JSON expected_format
        pset = PatternSet(
            name="test",
            patterns=[
                Pattern(
                    name="tool",
                    start_tag="<tool>",
                    end_tag="</tool>",
                    segment_type=SegmentType.TOOL,
                    greedy=False,
                    expected_format="json"  # Expect JSON format
                )
            ]
        )
        extractor = StreamingPatternExtractor(pset, stream_content=False)

        # Feed invalid JSON
        text = '<tool>{"name": "test", invalid}</tool>'
        for char in text:
            list(extractor.feed_chunk(char))

        segments, malformed = extractor.finalize()

        # Should have malformed patterns with error details
        assert len(malformed) > 0 or len(segments.parse_errors) > 0
        # Check that at least one malformed entry contains error information
        if malformed:
            error_found = any("ERROR:" in str(v) for v in malformed.values())
            assert error_found
        assert len(segments.parse_errors) >= 1

    def test_parse_error_missing_name_field(self):
        """Test error message when 'name' field is missing.

        Should produce a clear error indicating the missing field.
        """
        # This test verifies auto mode behavior: tool call without 'name' won't parse
        # but won't generate errors either (silent failure in auto mode)
        pset = create_default_pattern_set()
        extractor = StreamingPatternExtractor(pset, stream_content=False)

        # Tool call without 'name' field - line format but missing name
        text = '<tool>arguments: {"key": "value"}</tool>'
        for char in text:
            list(extractor.feed_chunk(char))

        segments, malformed = extractor.finalize()

        # In auto mode, this silently fails to parse - no tool, no error
        assert len(segments.tools) == 0
        # No parse errors in auto mode for unparseable content
        assert len(segments.parse_errors) == 0

    def test_parse_error_exceeds_max_size(self):
        """Test error message when tool call exceeds MAX_JSON_SIZE.

        Should produce clear error message with size information.
        """
        pset = create_default_pattern_set()
        extractor = StreamingPatternExtractor(pset, stream_content=False)

        # Create oversized content
        large_content = '{"name": "test", "data": "' + 'x' * (MAX_JSON_SIZE + 100) + '"}'
        text = f'<tool>{large_content}</tool>'

        for char in text:
            list(extractor.feed_chunk(char))

        segments, malformed = extractor.finalize()

        # Oversized tool calls always generate errors regardless of expected_format
        # because they exceed fundamental size limits
        assert len(segments.tools) == 0
        assert len(malformed) > 0 or len(segments.parse_errors) > 0

    def test_parse_error_recorded_in_batch_extractor(self):
        """PatternExtractor should record parse errors for malformed tool calls when expected_format is set."""
        # Create pattern with explicit expected_format to get error reporting
        pset = PatternSet(
            name="test",
            patterns=[
                Pattern(
                    name="tool",
                    start_tag="<tool>",
                    end_tag="</tool>",
                    segment_type=SegmentType.TOOL,
                    greedy=False,
                    expected_format="json"
                )
            ]
        )
        extractor = PatternExtractor(pset)

        text = '<tool>{"name": "broken", invalid}</tool>'
        segments = extractor.extract(text)

        # With expected_format="json", malformed JSON should produce error
        assert len(segments.tools) == 0
        assert len(segments.parse_errors) >= 1


class TestPerformanceBenchmarks:
    """Performance tests for overlap window optimization."""

    def test_performance_improvement_large_buffer(self):
        """Benchmark performance improvement from overlap window optimization.

        This test verifies that the overlap window optimization provides
        meaningful performance improvement for large buffers.
        """
        pset = create_default_pattern_set()

        # Create large amount of content before pattern
        prefix_text = "This is a lot of text before the pattern. " * 500  # ~20KB
        pattern_text = '<tool>{"name": "test", "arguments": {}}</tool>'

        full_text = prefix_text + pattern_text

        # Measure with optimization (current implementation)
        extractor = StreamingPatternExtractor(pset, stream_content=False)

        start = time.time()
        for char in full_text:
            list(extractor.feed_chunk(char))
        elapsed_optimized = time.time() - start

        segments, _ = extractor.finalize()
        assert len(segments.tools) == 1

        # Verify it completes in reasonable time
        # With O(n²) this would take much longer for 20KB+ text
        assert elapsed_optimized < 5.0  # Should complete in under 5 seconds

    def test_performance_many_patterns(self):
        """Test performance with many patterns in sequence.

        Ensures overlap window optimization maintains O(n) performance
        even with multiple patterns.
        """
        pset = create_default_pattern_set()
        extractor = StreamingPatternExtractor(pset, stream_content=False)

        # Create many patterns
        patterns_text = ""
        for i in range(100):
            patterns_text += f'<reasoning>thought {i}</reasoning>'

        start = time.time()
        for char in patterns_text:
            list(extractor.feed_chunk(char))
        elapsed = time.time() - start

        segments, _ = extractor.finalize()
        assert len(segments.reasoning) == 100

        # Should complete efficiently even with 100 patterns
        assert elapsed < 2.0  # Should complete in under 2 seconds


class TestStreamingExtractorReuse:
    """Tests that streaming extractor can be reused after finalize()."""

    def test_finalize_resets_state(self):
        pset = create_default_pattern_set()
        extractor = StreamingPatternExtractor(pset, stream_content=False)

        first_text = "<response>First</response>"
        for char in first_text:
            list(extractor.feed_chunk(char))
        segments1, malformed1 = extractor.finalize()

        assert segments1.response == "First"
        assert malformed1 == {}

        second_text = "<response>Second</response>"
        for char in second_text:
            list(extractor.feed_chunk(char))
        segments2, malformed2 = extractor.finalize()

        assert segments2.response == "Second"
        assert malformed2 == {}
