"""Build train_t4.ipynb from train_t4_source.py — split on `# %%` markers.

Includes kernelspec metadata (required by Kaggle papermill: lacks → 'No kernel
name found' error).
"""
import nbformat as nbf

with open("train_t4_source.py") as f:
    code = f.read()

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python", "version": "3.12"}

intro = """# ROGII T4 Train Kernel — Tabular MLP candidates

Trains 5 PyTorch MLP candidates on Kaggle T4 GPU using 5-fold CV with the
canonical sha256-hash fold map. Outputs `/kaggle/working/<cid>.parquet` for
each candidate, ready to import as round_010 hill-climb candidates.
"""
nb.cells.append(nbf.v4.new_markdown_cell(intro))

cells = code.split("# %%")
for cell in cells:
    cell = cell.strip()
    if cell:
        nb.cells.append(nbf.v4.new_code_cell(cell))

with open("train_t4.ipynb", "w") as f:
    nbf.write(nb, f)

print(f"Notebook created: train_t4.ipynb  ({len(nb.cells)} cells)")
