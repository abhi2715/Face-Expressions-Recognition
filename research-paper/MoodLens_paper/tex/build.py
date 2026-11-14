"""Orders the bibliography by first citation, writes paper.tex from main.tex, compiles with tectonic."""
import json, os, re, subprocess, sys
here = os.path.dirname(os.path.abspath(__file__))
src = open(f"{here}/main.tex").read(); refs = json.load(open(f"{here}/refs.json"))
order = []
for m in re.finditer(r"\\cite\{([^}]*)\}", src):
    for k in m.group(1).split(","):
        k = k.strip()
        if k not in order: order.append(k)
missing = [k for k in order if k not in refs]; assert not missing, missing
unused = [k for k in refs if k not in order]
bib = "\\begin{thebibliography}{%d}\n" % len(order) + "\n\n".join(f"\\bibitem{{{k}}}\n{refs[k]}" for k in order) + "\n\\end{thebibliography}"
open(f"{here}/paper.tex", "w").write(src.replace("%%BIB%%", bib))
print("citations:", len(order), "unused refs:", unused)
r = subprocess.run(["/Users/abhishekks/.local/bin/tectonic", "-X", "compile", "--keep-logs", "paper.tex"] if False else ["/Users/abhishekks/.local/bin/tectonic", "paper.tex"], cwd=here, capture_output=True, text=True)
print(r.stdout[-600:], r.stderr[-1800:])
