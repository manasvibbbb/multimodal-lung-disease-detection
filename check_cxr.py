from pathlib import Path
import re

PROJECT_DIR = Path(".")

PATTERNS = [
    r"GenerativeModel\s*\(",
    r"genai\.Client\s*\(",
    r"generate_content\s*\(",
    r"generateContent\s*\(",
    r"model\s*=\s*['\"]gemini[^'\"]+",
    r"model_name\s*=\s*['\"]gemini[^'\"]+",
    r"gemini-[A-Za-z0-9._-]+",
    r"GEMINI_API_KEY",
]

SKIP_DIRS = {
    ".git", ".venv", "venv", "__pycache__",
    "node_modules", "site-packages"
}

for path in PROJECT_DIR.rglob("*.py"):
    if any(part in SKIP_DIRS for part in path.parts):
        continue

    try:
        lines = path.read_text(
            encoding="utf-8",
            errors="ignore"
        ).splitlines()
    except OSError:
        continue

    matches = []

    for i, line in enumerate(lines):
        if any(re.search(pattern, line, re.IGNORECASE)
               for pattern in PATTERNS):
            # Redact possible credentials before displaying.
            safe_line = re.sub(
                r"(?i)(api[_-]?key\s*=\s*)['\"][^'\"]+['\"]",
                r"\1'[REDACTED]'",
                line
            )
            matches.append((i + 1, safe_line.strip()))

    if matches:
        print(f"\nFile: {path}")
        for line_number, line in matches:
            print(f"  Line {line_number}: {line}")