import json
from pathlib import Path

import nbformat as nbf

intro = """# ROGII run v12 hidden-safe PF ravaghi ensemble

Hidden-safe inference: PF128 + selected ravaghi raw offsets only. It avoids public-test-only v10/v50 precomputed submission CSVs so Kaggle reruns can score hidden test data.
"""

trainer_code = Path("koolbox/trainer/trainer.py").read_text()
ravaghi_code = Path("ravaghi_features.py").read_text()
bootstrap = f'''
# Write helper modules into the Kaggle working directory before imports.
from pathlib import Path

Path("koolbox/trainer").mkdir(parents=True, exist_ok=True)
Path("koolbox/__init__.py").write_text("from .trainer.trainer import Trainer\\n")
Path("koolbox/trainer/__init__.py").write_text("from .trainer import Trainer\\n")
Path("koolbox/trainer/trainer.py").write_text({json.dumps(trainer_code)})
Path("ravaghi_features.py").write_text({json.dumps(ravaghi_code)})
print("helper modules written")
'''

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
nb.cells.append(nbf.v4.new_code_cell(bootstrap.strip()))
for cell in code.split("# %%"):
    cell = cell.strip()
    if cell:
        nb.cells.append(nbf.v4.new_code_cell(cell))

with open("hillclimb_submit.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook created: hillclimb_submit.ipynb")
