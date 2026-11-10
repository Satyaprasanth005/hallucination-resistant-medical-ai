import json
import ast
from pathlib import Path

nb_path = Path("Hallucination_Resistant_Medical_Report_Generation.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Notebook valid JSON: {nb_path.exists()}")
print(f"Total cells: {len(nb['cells'])}")

code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
md_cells = [c for c in nb['cells'] if c['cell_type'] == 'markdown']

print(f"Code cells: {len(code_cells)}")
print(f"Markdown cells: {len(md_cells)}")

# Check AST of every code cell
syntax_errors = 0
for idx, cell in enumerate(code_cells, 1):
    src = "".join(cell['source'])
    try:
        ast.parse(src)
    except SyntaxError as e:
        print(f"SYNTAX ERROR in code cell {idx}: {e}")
        syntax_errors += 1

if syntax_errors == 0:
    print("[PASS] All code cells have valid Python syntax.")
else:
    print(f"[FAIL] Found {syntax_errors} syntax errors.")

# Check for all 12 sections in markdown cells
all_md_text = " ".join("".join(c['source']) for c in md_cells)
for sec_num in range(1, 13):
    has_sec = f"Section {sec_num}" in all_md_text or (sec_num == 1 and "Executive Research Summary" in all_md_text)
    print(f"Section {sec_num:02d} present: {has_sec}")

