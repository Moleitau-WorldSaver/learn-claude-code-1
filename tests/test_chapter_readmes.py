import py_compile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHAPTERS = sorted(ROOT.glob("s[0-9][0-9]_*"))


def test_every_chapter_has_only_the_chinese_readme() -> None:
    assert len(CHAPTERS) == 17

    for chapter in CHAPTERS:
        assert (chapter / "README.zh.md").is_file()
        assert not (chapter / "README.md").exists()
        assert not (chapter / "README.ja.md").exists()
        assert not (chapter / "README.en.md").exists()


def test_every_chapter_has_the_same_language_navigation() -> None:
    expected = "[中文](README.zh.md)"

    for chapter in CHAPTERS:
        filename = chapter / "README.zh.md"
        lines = filename.read_text(encoding="utf-8").splitlines()
        assert lines[2] == expected


def test_every_chapter_script_compiles_on_python_311() -> None:
    for chapter in CHAPTERS:
        _ = py_compile.compile(str(chapter / "code.py"), doraise=True)
