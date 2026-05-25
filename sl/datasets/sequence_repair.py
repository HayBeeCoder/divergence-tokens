"""Formatting-only repair utilities for numeric sequence completions."""

from __future__ import annotations

from dataclasses import dataclass
import re

from sl.datasets.data_models import DatasetRow


NUM_RE = re.compile(r"\d+")
ALPHA_RE = re.compile(r"[A-Za-z]")
NUMERIC_CHUNK_RE = re.compile(r"^[\d\s,;\[\]\(\).]+$")


@dataclass(frozen=True)
class FormatSpec:
    separator: str
    prefix: str = ""
    suffix: str = ""


COMMA = FormatSpec(", ")
SPACE = FormatSpec(" ")
SEMICOLON = FormatSpec("; ")
NEWLINE = FormatSpec("\n")
BRACKETS = FormatSpec(", ", "[", "]")
PARENS = FormatSpec(", ", "(", ")")


FORMAT_PATTERNS: tuple[tuple[FormatSpec, tuple[str, ...]], ...] = (
    (
        BRACKETS,
        (
            r"\[\s*number_?1\s*,\s*number_?2",
            r"square brackets?",
        ),
    ),
    (
        PARENS,
        (
            r"\(\s*number_?1\s*,\s*number_?2",
            r"parentheses",
            r"round brackets?",
        ),
    ),
    (
        NEWLINE,
        (
            r"one number per line",
            r"each number on (?:its )?own line",
            r"separate lines?",
            r"numbers? on separate lines?",
        ),
    ),
    (
        SEMICOLON,
        (
            r"semicolon[- ]separated",
            r"separated by semicolons?",
            r"numbers? separated by semicolons?",
        ),
    ),
    (
        SPACE,
        (
            r"space[- ]separated",
            r"space[- ]delimited",
            r"spaces between",
            r"separated by spaces?",
        ),
    ),
    (
        COMMA,
        (
            r"comma[- ]separated",
            r"comma[- ]delimited",
            r"separated by commas?",
            r"number1\s*,\s*number2",
            r"number_1\s*,\s*number_2",
        ),
    ),
)


def extract_numeric_tokens(completion: str) -> list[str]:
    """Return numeric tokens from clean numeric chunks, preserving order/values."""
    tokens: list[str] = []
    for line in completion.splitlines():
        chunk = line.strip()
        if not chunk or not NUM_RE.search(chunk):
            continue
        if ALPHA_RE.search(chunk):
            continue
        if not NUMERIC_CHUNK_RE.fullmatch(chunk):
            continue
        tokens.extend(NUM_RE.findall(chunk))
    return tokens


def _numeric_chunks(completion: str) -> list[str]:
    chunks: list[str] = []
    for line in completion.splitlines():
        chunk = line.strip()
        if (
            chunk
            and NUM_RE.search(chunk)
            and not ALPHA_RE.search(chunk)
            and NUMERIC_CHUNK_RE.fullmatch(chunk)
        ):
            chunks.append(chunk)
    return chunks


def infer_format(prompt: str, completion: str | None = None) -> FormatSpec:
    """Infer output formatting from prompt, falling back to completion style."""
    for spec, patterns in FORMAT_PATTERNS:
        if any(re.search(pattern, prompt, re.I) for pattern in patterns):
            return spec

    if completion is not None:
        return infer_format_from_completion(completion)

    return COMMA


def infer_format_from_completion(completion: str) -> FormatSpec:
    chunks = _numeric_chunks(completion)
    if not chunks:
        return COMMA

    if any(chunk.startswith("[") and chunk.endswith("]") for chunk in chunks):
        return BRACKETS
    if any(chunk.startswith("(") and chunk.endswith(")") for chunk in chunks):
        return PARENS

    counts = {
        NEWLINE: max(0, len(chunks) - 1),
        SEMICOLON: sum(chunk.count(";") for chunk in chunks),
        COMMA: sum(chunk.count(",") for chunk in chunks),
    }
    whitespace_separators = 0
    for chunk in chunks:
        without_punctuation = re.sub(r"[,;\[\]\(\).]", " ", chunk)
        whitespace_separators += max(0, len(NUM_RE.findall(without_punctuation)) - 1)
    counts[SPACE] = whitespace_separators

    return max(counts, key=counts.get)


def format_numeric_tokens(tokens: list[str], spec: FormatSpec) -> str:
    if not tokens:
        return ""
    return f"{spec.prefix}{spec.separator.join(tokens)}{spec.suffix}"


def repair_completion(prompt: str, completion: str) -> str:
    """Remove text contamination and normalize format without changing numbers."""
    tokens = extract_numeric_tokens(completion)
    return format_numeric_tokens(tokens, infer_format(prompt, completion))


def repair_dataset_rows(rows: list[DatasetRow]) -> list[DatasetRow]:
    return [
        DatasetRow(
            prompt=row.prompt,
            completion=repair_completion(row.prompt, row.completion),
        )
        for row in rows
    ]
