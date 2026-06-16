
import nbformat as nbf

# Read the inference kernel code
with open('/Users/liucong/code/kaggle/ROGII/experiments/round_010/submission_package/hillclimb_v3_kernel/inference_kernel.py') as f:
    code = f.read()

# Create notebook
nb = nbf.v4.new_notebook()

# Add markdown intro cell
intro = """# ROGII Inference-Only Kernel
This kernel loads pre-trained models and runs inference without any training.
"""
nb.cells.append(nbf.v4.new_markdown_cell(intro))

# Split the code on # %% markers to create cells
import re
cells = re.split(r'#\s*%%', code.strip())
for i, cell_content in enumerate(cells):
    cell_content = cell_content.strip()
    if cell_content:
        if i == 0:
            # First cell (docstring) - keep as code or make markdown?
            nb.cells.append(nbf.v4.new_code_cell(cell_content))
        else:
            nb.cells.append(nbf.v4.new_code_cell(cell_content))

# Write notebook
with open('/Users/liucong/code/kaggle/ROGII/experiments/round_010/submission_package/hillclimb_v3_kernel/hillclimb_submit.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook created successfully")
