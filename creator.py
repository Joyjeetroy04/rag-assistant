# creator.py (clean helper file)
def clean_and_tag(text: str) -> tuple[str, str]:
    if not isinstance(text, str): return "", "invalid"
    text = text.strip()
    if not text: return "", "empty"
    if len(text) < 25: return text, "short"
    if text.isupper(): return text, "header"
    if text.replace(" ", "").isdigit(): return text, "numeric"
    return text, "informative"
