from app.services.scraper import fetch_webpage as _fetch


def fetch_webpage(url: str) -> str:
    return _fetch(url)
