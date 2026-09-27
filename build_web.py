"""Bundle web/app.html + web/settle.js + settle_weights.json into one self-contained page.

    python build_web.py [weights.json]

Writes
    web/dist/settle.html           standalone page (double-click to open; works offline)
    web/dist/settle.fragment.html  same page without the <html> shell, for hosts that add their own
    site/maze.html, site/settle.js the maze page of the deployed site, with site navigation
"""
import json, os, shutil, sys

root = os.path.dirname(os.path.abspath(__file__))
weights_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "settle_weights.json")
w = json.load(open(weights_path))
w.pop("meta", None)                                   # training log is not needed for inference
app = open(os.path.join(root, "web", "app.html"), encoding="utf-8").read()
js = open(os.path.join(root, "web", "settle.js"), encoding="utf-8").read()
frag = app.replace("/*SETTLE_JS*/", js).replace("/*WEIGHTS*/null", json.dumps(w, separators=(",", ":")))
# second model for the page's toggle: the maze model retrained to say "no path" (arena/settle_nopath.py)
nopath_path = os.path.join(root, "arena", "settle_nopath.json")
if os.path.exists(nopath_path):
    wn = json.load(open(nopath_path)); wn.pop("meta", None)
    frag = frag.replace("/*WEIGHTS_NOPATH*/null", json.dumps(wn, separators=(",", ":")))
assert "</script>" not in js and "/*WEIGHTS*/" not in frag

SHELL = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
         '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n')
NAV = """<style>
  .sitenav { display: flex; align-items: center; gap: 20px; flex-wrap: wrap; padding-block: 0 16px; border-bottom: 1px solid var(--line); }
  .sitenav .brand { font: 700 20px/1 var(--display); color: var(--ink); text-decoration: none; margin-right: 8px; }
  .sitenav a.link { color: var(--muted); text-decoration: none; font-weight: 500; font-size: 14px; padding-block: 4px; border-bottom: 2px solid transparent; }
  .sitenav a.link[aria-current="page"] { color: var(--ink); border-bottom-color: var(--belief); }
  .sitenav a:focus-visible { outline: 2px solid var(--belief); outline-offset: 2px; }
</style>
<nav class="sitenav" aria-label="Site">
  <a class="brand" href="./">Settle</a>
  <a class="link" href="maze.html" aria-current="page">Maze</a>
  <a class="link" href="sudoku.html">Sudoku</a>
  <a class="link" href="code.html">Code</a>
</nav>"""

def full_page(body):
    return SHELL + body.replace('<div class="wrap">', '</head>\n<body>\n<div class="wrap">', 1) + "\n</body>\n</html>\n"

out = os.path.join(root, "web", "dist"); os.makedirs(out, exist_ok=True)
standalone = frag.replace("<!--NAV-->", "")
open(os.path.join(out, "settle.fragment.html"), "w", encoding="utf-8").write(standalone)
page = full_page(standalone)
open(os.path.join(out, "settle.html"), "w", encoding="utf-8").write(page)

site = os.path.join(root, "site")
open(os.path.join(site, "maze.html"), "w", encoding="utf-8").write(full_page(frag.replace("<!--NAV-->", NAV)))
shutil.copy(os.path.join(root, "web", "settle.js"), os.path.join(site, "settle.js"))
print(f"wrote web/dist/settle.html ({len(page) / 1024:.0f} KB) and site/maze.html, C={w['C']}")
