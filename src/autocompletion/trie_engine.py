"""Fast multilingual prefix Trie + ghost-text state manager.

Data contract
-------------
Dictionary lines: ``word\\tfrequency`` or ``word frequency`` (UTF-8).
lookup_top1(prefix) -> (word | None, suffix | None) in < 1 ms for typical lexicons.
TAB commit injects only the unfinished suffix (+ trailing space).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from src.paths import resource_root

REPO_ROOT = resource_root()
DEFAULT_DICT_DIR = REPO_ROOT / "data" / "dictionaries"


class TrieNode:
    __slots__ = ("children", "is_word", "freq", "best_word", "best_freq")

    def __init__(self) -> None:
        self.children: dict[str, TrieNode] = {}
        self.is_word: bool = False
        self.freq: int = 0
        self.best_word: str | None = None
        self.best_freq: int = 0


class TrieEngine:
    """Prefix Trie with per-node best-frequency pointers for O(prefix) top-1."""

    def __init__(self) -> None:
        self.root = TrieNode()
        self.word_count = 0

    def insert(self, word: str, frequency: int = 1) -> None:
        if not word:
            return
        freq = max(int(frequency), 1)
        node = self.root
        path = [node]
        for ch in word:
            child = node.children.get(ch)
            if child is None:
                child = TrieNode()
                node.children[ch] = child
            node = child
            path.append(node)
        if not node.is_word:
            self.word_count += 1
        node.is_word = True
        node.freq = max(node.freq, freq)
        # Propagate best completion from leaf to root.
        best_word = word
        best_freq = node.freq
        node.best_word = best_word
        node.best_freq = best_freq
        for n in reversed(path[:-1]):
            if n.best_word is None or best_freq > n.best_freq:
                n.best_word = best_word
                n.best_freq = best_freq
            else:
                best_word = n.best_word
                best_freq = n.best_freq

    def load_file(self, path: Path) -> int:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        added = 0
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "\t" in line:
                    word, _, rest = line.partition("\t")
                else:
                    parts = line.split()
                    if len(parts) == 1:
                        word, rest = parts[0], "1"
                    else:
                        word, rest = parts[0], parts[-1]
                try:
                    freq = int(float(rest))
                except ValueError:
                    freq = 1
                self.insert(word.strip(), max(freq, 1))
                added += 1
        return added

    def load_directory(self, directory: Path = DEFAULT_DICT_DIR) -> dict[str, int]:
        directory = Path(directory)
        stats: dict[str, int] = {}
        for name in ("en.txt", "ru.txt", "he.txt"):
            path = directory / name
            if path.is_file():
                stats[name] = self.load_file(path)
        return stats

    def lookup_top1(self, prefix: str) -> tuple[str | None, str | None]:
        """Return (full_word, unfinished_suffix). Both None if no completion."""
        if not prefix:
            return None, None
        node = self.root
        for ch in prefix:
            nxt = node.children.get(ch)
            if nxt is None:
                return None, None
            node = nxt
        word = node.best_word
        if word is None or not word.startswith(prefix):
            return None, None
        return word, word[len(prefix) :]

    def benchmark_lookup(self, prefixes: Iterable[str], repeats: int = 200) -> float:
        prefixes = list(prefixes)
        if not prefixes:
            return 0.0
        t0 = time.perf_counter()
        for _ in range(repeats):
            for p in prefixes:
                self.lookup_top1(p)
        return (time.perf_counter() - t0) * 1000.0 / (repeats * len(prefixes))


@dataclass
class GhostTextManager:
    """Active-prefix buffer + passive suggestion. Commit only on TAB/fist."""

    trie: TrieEngine
    active_prefix: str = ""
    suggestion: str | None = None
    suggestion_suffix: str = ""
    history: list[str] = field(default_factory=list)

    def reset(self) -> None:
        self.active_prefix = ""
        self.suggestion = None
        self.suggestion_suffix = ""

    def clear(self) -> None:
        """Flush the active word buffer (space / enter / fist commit)."""
        self.reset()

    def pop_char(self) -> tuple[str, str | None]:
        """Remove the last prefix character (swipe-left / backspace)."""
        return self.backspace()

    def append_char(self, ch: str) -> tuple[str, str | None]:
        if not ch:
            return self.active_prefix, self.suggestion
        if ch.isspace():
            self.reset()
            return self.active_prefix, self.suggestion
        self.active_prefix += ch
        self._refresh()
        return self.active_prefix, self.suggestion

    def backspace(self) -> tuple[str, str | None]:
        if self.active_prefix:
            self.active_prefix = self.active_prefix[:-1]
        self._refresh()
        return self.active_prefix, self.suggestion

    def _refresh(self) -> None:
        if not self.active_prefix:
            self.suggestion = None
            self.suggestion_suffix = ""
            return
        word, suffix = self.trie.lookup_top1(self.active_prefix)
        if word is None or suffix is None or suffix == "":
            self.suggestion = None
            self.suggestion_suffix = ""
        else:
            self.suggestion = word
            self.suggestion_suffix = suffix

    def commit_tab(self) -> str:
        """Return text to inject (suffix + trailing space). Clears ghost state."""
        suffix = self.suggestion_suffix
        committed = self.suggestion or self.active_prefix
        if committed:
            self.history.append(committed)
        if suffix:
            inject = f"{suffix} "
        elif committed:
            inject = " "
        else:
            inject = ""
        self.reset()
        return inject
