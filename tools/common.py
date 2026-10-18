"""Общие утилиты tools/."""
LATIN_X, CYR_KHA = chr(0x58), chr(0x425)

def norm(name: str) -> str:
    """Имена файлов смешивают латинскую X и кириллическую Х — приводим к кириллической."""
    return name.replace(LATIN_X, CYR_KHA)
