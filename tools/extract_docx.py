"""Extract a .docx (including MS Word OMML equations) to Markdown with LaTeX math.

Usage:  python tools/extract_docx.py robo-advisor.docx -o docs/SPEC.md

Word stores equations as Office Math Markup (``m:oMath``), which plain-text
extractors silently drop. This converter walks the OMML tree and emits
LaTeX (fractions, sub/superscripts, n-ary operators, delimiters, radicals,
accents, limits, matrices), inline as ``$...$`` and display as ``$$...$$``.
"""
import argparse
import sys
import zipfile
import xml.etree.ElementTree as ET

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
w = lambda t: f"{{{W}}}{t}"
m = lambda t: f"{{{M}}}{t}"

SYM = {"∑": r"\sum", "∏": r"\prod", "∫": r"\int", "≤": r"\le ", "≥": r"\ge ", "≠": r"\ne ",
       "×": r"\times ", "⋅": r"\cdot ", "·": r"\cdot ", "∈": r"\in ", "σ": r"\sigma ", "μ": r"\mu ",
       "ρ": r"\rho ", "λ": r"\lambda ", "α": r"\alpha ", "β": r"\beta ", "γ": r"\gamma ", "δ": r"\delta ",
       "ε": r"\epsilon ", "θ": r"\theta ", "Σ": r"\Sigma ", "Δ": r"\Delta ", "π": r"\pi ", "τ": r"\tau ",
       "ω": r"\omega ", "∞": r"\infty ", "→": r"\to ", "−": "-", "√": r"\sqrt", "′": "'", "∀": r"\forall ",
       "≈": r"\approx ", "Ω": r"\Omega ", "φ": r"\phi ", "η": r"\eta ", "κ": r"\kappa ", "ν": r"\nu ",
       "ξ": r"\xi ", "ψ": r"\psi ", "χ": r"\chi ", "ζ": r"\zeta ", "Γ": r"\Gamma ", "Λ": r"\Lambda ",
       "Π": r"\Pi ", "Φ": r"\Phi ", "Ψ": r"\Psi ", "Θ": r"\Theta ", "∂": r"\partial ", "∇": r"\nabla ",
       "…": r"\ldots ", "%": r"\%", "$": r"\$", "⋯": r"\cdots ", "±": r"\pm ", "⊤": r"\top ", "∣": "|", "‖": r"\|"}


def txt(s):
    return "".join(SYM.get(c, c) for c in s)


def chr_of(el, tag, default):
    pr = el.find(m(tag))
    if pr is not None:
        c = pr.find(m("chr"))
        if c is not None:
            return c.get(m("val"), default)
    return default


def conv(el):
    """Convert an OMML element to LaTeX."""
    t = el.tag
    kids = list(el)
    if t == m("r"):
        return "".join(txt(x.text or "") for x in el.iter(m("t")))
    if t == m("f"):
        return r"\frac{%s}{%s}" % (sub(el, "num"), sub(el, "den"))
    if t == m("sSub"):
        return "{%s}_{%s}" % (sub(el, "e"), sub(el, "sub"))
    if t == m("sSup"):
        return "{%s}^{%s}" % (sub(el, "e"), sub(el, "sup"))
    if t == m("sSubSup"):
        return "{%s}_{%s}^{%s}" % (sub(el, "e"), sub(el, "sub"), sub(el, "sup"))
    if t == m("sPre"):
        return "{}_{%s}^{%s}{%s}" % (sub(el, "sub"), sub(el, "sup"), sub(el, "e"))
    if t == m("nary"):
        c = chr_of(el, "naryPr", "∫")
        op = SYM.get(c, c)
        s, p = sub(el, "sub"), sub(el, "sup")
        out = op
        if s:
            out += "_{%s}" % s
        if p:
            out += "^{%s}" % p
        return out + " " + sub(el, "e")
    if t == m("d"):
        pr = el.find(m("dPr"))
        beg, end, sep = "(", ")", "|"
        if pr is not None:
            for k, v in (("begChr", "beg"), ("endChr", "end"), ("sepChr", "sep")):
                x = pr.find(m(k))
                if x is not None:
                    val = x.get(m("val"), "")
                    if v == "beg":
                        beg = val
                    elif v == "end":
                        end = val
                    else:
                        sep = val
        es = [conv_all(e) for e in el.findall(m("e"))]
        fix = {"{": r"\{", "}": r"\}", "": ".", "‖": r"\|", "|": "|", "⌈": r"\lceil", "⌉": r"\rceil",
               "⌊": r"\lfloor", "⌋": r"\rfloor", "〈": r"\langle", "〉": r"\rangle", "⟨": r"\langle", "⟩": r"\rangle"}
        return r"\left%s %s \right%s" % (fix.get(beg, beg), (" %s " % sep).join(es), fix.get(end, end))
    if t == m("rad"):
        deg = sub(el, "deg")
        return (r"\sqrt[%s]{%s}" % (deg, sub(el, "e"))) if deg else (r"\sqrt{%s}" % sub(el, "e"))
    if t == m("acc"):
        c = chr_of(el, "accPr", "̂")
        cmd = {"̂": r"\hat", "̃": r"\tilde", "̄": r"\bar", "̇": r"\dot", "⃗": r"\vec", "¯": r"\bar"}.get(c, r"\hat")
        return "%s{%s}" % (cmd, sub(el, "e"))
    if t == m("bar"):
        return r"\overline{%s}" % sub(el, "e")
    if t == m("func"):
        return "%s %s" % (sub(el, "fName"), sub(el, "e"))
    if t == m("limLow"):
        return r"\underset{%s}{%s}" % (sub(el, "lim"), sub(el, "e"))
    if t == m("limUpp"):
        return r"\overset{%s}{%s}" % (sub(el, "lim"), sub(el, "e"))
    if t == m("groupChr"):
        return r"\underbrace{%s}" % sub(el, "e")
    if t == m("m"):
        rows = []
        for mr in el.findall(m("mr")):
            rows.append(" & ".join(conv_all(e) for e in mr.findall(m("e"))))
        return r"\begin{bmatrix}%s\end{bmatrix}" % r" \\ ".join(rows)
    if t == m("eqArr"):
        return r"\begin{aligned}%s\end{aligned}" % r" \\ ".join(conv_all(e) for e in el.findall(m("e")))
    if t.endswith("Pr") or t == m("ctrlPr"):
        return ""
    return conv_all(el)


def conv_all(el):
    return "".join(conv(k) for k in el)


def sub(el, name):
    x = el.find(m(name))
    return conv_all(x) if x is not None else ""


def para(p):
    out = []
    for k in p:
        if k.tag == w("r"):
            for x in k:
                if x.tag == w("t"):
                    out.append(x.text or "")
                elif x.tag == w("tab"):
                    out.append("\t")
                elif x.tag in (w("br"), w("cr")):
                    out.append("\n")
        elif k.tag == m("oMath"):
            out.append(" $%s$ " % conv_all(k).strip())
        elif k.tag == m("oMathPara"):
            for om in k.iter(m("oMath")):
                out.append("\n$$%s$$\n" % conv_all(om).strip())
        elif k.tag in (w("hyperlink"), w("ins"), w("smartTag"), w("sdt"), w("sdtContent"), w("fldSimple")):
            out.append(para(k))
    return "".join(out)


def style(p):
    pr = p.find(w("pPr"))
    if pr is None:
        return "", False
    s = pr.find(w("pStyle"))
    num = pr.find(w("numPr"))
    return (s.get(w("val")) if s is not None else ""), num is not None


def body(el, depth=0):
    lines = []
    for k in el:
        if k.tag == w("p"):
            st, isnum = style(k)
            text = para(k).strip()
            if not text:
                continue
            if st.lower().startswith("heading"):
                lvl = int("".join(c for c in st if c.isdigit()) or 1)
                lines.append("#" * (lvl + 1) + " " + text)
            elif st.lower() == "title":
                lines.append("# " + text)
            elif isnum or "list" in st.lower():
                lines.append("- " + text)
            else:
                lines.append(text)
        elif k.tag == w("tbl"):
            for tr in k.iter(w("tr")):
                cells = [" ".join(para(p).strip() for p in tc.iter(w("p"))) for tc in tr.findall(w("tc"))]
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")
        elif k.tag in (w("sdt"), w("sdtContent")):
            lines += body(k, depth + 1)
    return lines


def extract(path):
    """Return Markdown for a .docx file or a raw word/document.xml."""
    if path.endswith(".xml"):
        root = ET.parse(path).getroot()
    else:
        with zipfile.ZipFile(path) as z:
            root = ET.fromstring(z.read("word/document.xml"))
    return "\n\n".join(body(root.find(w("body")))) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", help=".docx file (or word/document.xml)")
    ap.add_argument("-o", "--output", help="write UTF-8 Markdown to this file instead of stdout")
    args = ap.parse_args()
    md = extract(args.path)
    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(md)
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stdout.write(md)
