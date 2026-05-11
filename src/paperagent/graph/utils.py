from hashlib import sha256


def canonicalize_entity(name: str) -> str:
    """把实体名归一成更适合做去重键的形式。"""

    return " ".join(name.lower().replace("_", "-").split())


def normalize_title(title: str) -> str:
    """对论文标题做轻量归一化，供增量导入时使用。"""

    cleaned = title.lower().replace("_", " ").replace("-", " ")
    return " ".join(cleaned.split())


def semantic_id(label: str, scope: str, text: str) -> str:
    """为语义节点生成稳定 ID。

    语义节点往往没有天然主键，这里用 label + scope + text 的哈希作为折中方案。
    """

    digest = sha256(f"{label}|{scope}|{text}".encode("utf-8")).hexdigest()[:16]
    return f"{label.lower()}:{digest}"
