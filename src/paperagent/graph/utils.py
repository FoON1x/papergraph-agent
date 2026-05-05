from hashlib import sha256


def canonicalize_entity(name: str) -> str:
    return " ".join(name.lower().replace("_", "-").split())


def normalize_title(title: str) -> str:
    cleaned = title.lower().replace("_", " ").replace("-", " ")
    return " ".join(cleaned.split())


def semantic_id(label: str, scope: str, text: str) -> str:
    digest = sha256(f"{label}|{scope}|{text}".encode("utf-8")).hexdigest()[:16]
    return f"{label.lower()}:{digest}"
