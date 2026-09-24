import importlib.util

from conftest import ROOT


def _extractor():
    spec = importlib.util.spec_from_file_location("extract_docx", ROOT / "tools" / "extract_docx.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_word_equations_are_extracted_as_latex():
    md = _extractor().extract(str(ROOT / "robo-advisor.docx"))
    assert r"\sum_{i} |{w}_{i}|\le L" in md                       # gross exposure constraint
    assert r"\frac{{\left( 1+r \right)}^{T}-1}{r}" in md           # FV annuity term
    assert r"P\left( {F}_{T}\ge {F}^{*} \right)" in md             # target probability
    assert r"{\sigma }_{p}^{2}={w}^{'}\Sigma w" in md              # portfolio variance
    assert r"\underset{w}{min}" in md


def test_spec_md_is_in_sync_with_docx():
    md = _extractor().extract(str(ROOT / "robo-advisor.docx"))
    assert (ROOT / "docs" / "SPEC.md").read_text() == md
