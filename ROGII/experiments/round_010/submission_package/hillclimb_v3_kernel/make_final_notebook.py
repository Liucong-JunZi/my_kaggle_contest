
import nbformat as nbf

# Read the inference kernel code
with open('inference_kernel.py', 'r') as f:
    code = f.read()

# Create notebook
nb = nbf.v4.new_notebook()

# Add markdown intro
intro = """# ROGII Hill Climb v3 Final - Pretrained Model Inference

This notebook loads pretrained models and generates the competition submission.
"""
nb.cells.append(nbf.v4.new_markdown_cell(intro))

# Split code into cells at '# %%' markers
cells = code.split('# %%')
for cell in cells:
    cell = cell.strip()
    if cell:
        nb.cells.append(nbf.v4.new_code_cell(cell))

# Write notebook
with open('hillclimb_submit.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook created: hillclimb_submit.ipynb")
