import json
from typing import Generator

from groq import Groq

from app.config import settings
from app.tools.definitions import TOOLS
from app.tools.web_tools import fetch_webpage
from app.tools.api_tools import call_api

SYSTEM_PROMPT = """Bạn là trợ lý thông minh hỗ trợ truy vấn thông tin.

Khi người dùng hỏi về thông tin cần tìm kiếm từ internet hoặc API:
- Dùng tool `fetch_webpage` để đọc nội dung từ một URL cụ thể
- Dùng tool `call_api` để lấy dữ liệu từ các REST API công khai

Hướng dẫn:
- Trả lời bằng ngôn ngữ người dùng sử dụng (tiếng Việt hoặc tiếng Anh)
- Luôn trích dẫn nguồn (URL) khi đã fetch thông tin từ web
- Tổng hợp thông tin rõ ràng, ngắn gọn và chính xác
- Nếu không tìm được thông tin, hãy nói thẳng và gợi ý cách khác"""

client = Groq(api_key=settings.groq_api_key)


def _execute_tool(tool_name: str, tool_input: dict) -> str:
    if tool_name == "fetch_webpage":
        return fetch_webpage(tool_input["url"])
    elif tool_name == "call_api":
        return call_api(
            url=tool_input["url"],
            method=tool_input.get("method", "GET"),
            params=tool_input.get("params"),
            headers=tool_input.get("headers"),
        )
    return f"Tool không xác định: {tool_name}"


def chat_stream(messages: list[dict]) -> Generator[str, None, None]:
    """
    Agentic loop:
    - Gửi messages lên Groq
    - Nếu model gọi tool → thực thi → gửi lại kết quả
    - Lặp cho đến khi nhận được câu trả lời cuối (finish_reason = "stop")
    - Stream từng token về client
    """
    current_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + list(messages)

    while True:
        response = client.chat.completions.create(
            model=settings.model,
            messages=current_messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=settings.max_tokens,
        )

        choice = response.choices[0]
        message = choice.message

        # Nếu model muốn gọi tool
        if choice.finish_reason == "tool_calls" and message.tool_calls:
            # Thêm assistant message vào history
            current_messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ],
            })

            # Thực thi từng tool call
            for tc in message.tool_calls:
                yield f"[Đang dùng tool: {tc.function.name}...]\n"
                try:
                    tool_input = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    tool_input = {}

                result = _execute_tool(tc.function.name, tool_input)

                current_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

            continue  # Gửi lại với kết quả tool

        # Câu trả lời cuối — stream từng token
        stream = client.chat.completions.create(
            model=settings.model,
            messages=current_messages,
            tools=TOOLS,
            tool_choice="none",  # Không gọi tool nữa, chỉ trả lời
            max_tokens=settings.max_tokens,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content
        break
