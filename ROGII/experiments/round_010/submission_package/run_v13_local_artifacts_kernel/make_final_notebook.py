import nbformat as nbf
from pathlib import Path

intro = """# ROGII run v13 local artifacts submit

Inference-only notebook. Loads one locally exported Kaggle dataset of native artifacts and computes hidden-test predictions live from competition files.
"""

code = Path("inference_kernel.py").read_text()

nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb.metadata.language_info = {
    "name": "python",
    "version": "3.12",
    "mimetype": "text/x-python",
    "codemirror_mode": {"name": "ipython", "version": 3},
    "pygments_lexer": "ipython3",
    "nbconvert_exporter": "python",
    "file_extension": ".py",
}
nb.cells.append(nbf.v4.new_markdown_cell(intro))
for cell in code.split("# %%"):
    cell = cell.strip()
    if cell:
        nb.cells.append(nbf.v4.new_code_cell(cell))

with open("hillclimb_submit.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook created: hillclimb_submit.ipynb")
