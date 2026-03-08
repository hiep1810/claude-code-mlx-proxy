import pytest
import json
from main import (
    format_tools_for_chat_template,
    parse_tool_calls_from_response,
    parse_thinking_blocks,
    extract_text_from_content,
    ContentBlockText,
    ContentBlockToolUse,
    ContentBlockToolResult,
    Tool
)


def test_format_tools_for_chat_template():
    tools = [
        Tool(
            name="read_file",
            description="Reads a file from the disk.",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}}}
        )
    ]
    formatted = format_tools_for_chat_template(tools)
    assert len(formatted) == 1
    assert formatted[0]["type"] == "function"
    assert formatted[0]["function"]["name"] == "read_file"
    assert formatted[0]["function"]["description"] == "Reads a file from the disk."
    assert formatted[0]["function"]["parameters"] == {"type": "object", "properties": {"path": {"type": "string"}}}


def test_format_tools_empty():
    assert format_tools_for_chat_template([]) is None
    assert format_tools_for_chat_template(None) is None


def test_parse_tool_calls_xml_params():
    response = "I will read the file.\n<function=read_file><parameter=path>foo.py</parameter></function>\nDone."
    clean_text, tool_calls = parse_tool_calls_from_response(response)
    
    # Should strip the function XML but keep the surrounding text
    assert clean_text == "I will read the file.\n\nDone."
    assert len(tool_calls) == 1
    
    tc = tool_calls[0]
    assert tc["type"] == "tool_use"
    assert tc["id"].startswith("toolu_")
    assert tc["name"] == "read_file"
    assert tc["input"] == {"path": "foo.py"}


def test_parse_tool_calls_json_params():
    response = "Checking config...\n<function=read_config>{\"key\": \"port\"}</function>"
    clean_text, tool_calls = parse_tool_calls_from_response(response)
    
    assert clean_text == "Checking config..."
    assert len(tool_calls) == 1
    assert tool_calls[0]["input"] == {"key": "port"}


def test_parse_thinking_blocks():
    response = "<think>\nThis is a complex problem.\nI need to think step by step.\n</think>\nHere is the answer."
    clean_text, thinking = parse_thinking_blocks(response)
    
    assert clean_text == "Here is the answer."
    assert "complex problem" in thinking
    assert "step by step" in thinking


def test_extract_text_from_content_with_tools():
    content = [
        ContentBlockText(text="Let me check the time."),
        ContentBlockToolUse(id="toolu_123", name="get_time", input={"timezone": "UTC"}),
        ContentBlockToolResult(tool_use_id="toolu_123", content="12:00 PM")
    ]
    
    text = extract_text_from_content(content)
    assert "Let me check the time." in text
    assert "[Calling tool: get_time({\"timezone\": \"UTC\"})]" in text
    assert "[Tool result for toolu_123: 12:00 PM]" in text


def test_extract_text_from_dict_content():
    content = [
        {"type": "text", "text": "Hello"},
        {"type": "tool_use", "id": "t1", "name": "test", "input": {"a": 1}},
        {"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": "Result text"}]}
    ]
    
    text = extract_text_from_content(content)
    assert "Hello" in text
    assert "[Calling tool: test({\"a\": 1})]" in text
    assert "[Tool result for t1: Result text]" in text
