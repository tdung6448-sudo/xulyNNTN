# Tool definitions theo định dạng OpenAI/Groq
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_webpage",
            "description": (
                "Lấy nội dung text từ một URL website. "
                "Dùng tool này khi cần đọc thông tin từ một trang web cụ thể."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL đầy đủ của trang web (bắt đầu bằng http:// hoặc https://)",
                    }
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "call_api",
            "description": (
                "Gọi một HTTP REST API và trả về kết quả JSON. "
                "Dùng tool này khi cần lấy dữ liệu từ các API công khai."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL của API endpoint",
                    },
                    "method": {
                        "type": "string",
                        "enum": ["GET", "POST"],
                        "description": "HTTP method (mặc định GET)",
                    },
                    "params": {
                        "type": "object",
                        "description": "Query parameters (GET) hoặc request body (POST)",
                    },
                    "headers": {
                        "type": "object",
                        "description": "HTTP headers tùy chọn",
                    },
                },
                "required": ["url"],
            },
        },
    },
]
