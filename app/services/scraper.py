import requests
from bs4 import BeautifulSoup

MAX_LENGTH = 5000

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def _clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # Xóa các tag không cần thiết
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)

    # Gộp nhiều dòng trắng thành 1
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def fetch_webpage(url: str) -> str:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding

        text = _clean_html(resp.text)

        if not text.strip():
            return _fetch_with_playwright(url)

        if len(text) > MAX_LENGTH:
            text = text[:MAX_LENGTH] + "\n...[nội dung bị cắt bớt]"

        return text

    except requests.exceptions.Timeout:
        return "Lỗi: Trang web không phản hồi sau 15 giây."
    except requests.exceptions.HTTPError as e:
        return f"Lỗi HTTP {e.response.status_code} khi truy cập {url}."
    except Exception as e:
        return f"Lỗi khi tải trang web: {str(e)}"


def _fetch_with_playwright(url: str) -> str:
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=HEADERS["User-Agent"],
                java_script_enabled=True,
            )
            page = context.new_page()
            page.goto(url, timeout=20000, wait_until="domcontentloaded")
            page.wait_for_timeout(2000)
            html = page.content()
            browser.close()

        text = _clean_html(html)
        if len(text) > MAX_LENGTH:
            text = text[:MAX_LENGTH] + "\n...[nội dung bị cắt bớt]"
        return text

    except ImportError:
        return "Không thể tải trang JavaScript (cần cài playwright: pip install playwright && playwright install chromium)."
    except Exception as e:
        return f"Lỗi Playwright: {str(e)}"
