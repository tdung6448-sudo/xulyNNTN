import json
import requests


def call_api(
    url: str,
    method: str = "GET",
    params: dict | None = None,
    headers: dict | None = None,
) -> str:
    try:
        default_headers = {"User-Agent": "Mozilla/5.0 (compatible; ChatbotAPI/1.0)"}
        if headers:
            default_headers.update(headers)

        if method.upper() == "POST":
            resp = requests.post(url, json=params, headers=default_headers, timeout=15)
        else:
            resp = requests.get(url, params=params, headers=default_headers, timeout=15)

        resp.raise_for_status()

        try:
            data = resp.json()
            text = json.dumps(data, ensure_ascii=False, indent=2)
        except Exception:
            text = resp.text

        # Giới hạn độ dài trả về
        if len(text) > 5000:
            text = text[:5000] + "\n...[truncated]"

        return text

    except requests.exceptions.Timeout:
        return "Lỗi: Request timeout sau 15 giây."
    except requests.exceptions.HTTPError as e:
        return f"Lỗi HTTP {e.response.status_code}: {e.response.text[:500]}"
    except Exception as e:
        return f"Lỗi khi gọi API: {str(e)}"
