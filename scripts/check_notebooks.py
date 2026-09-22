"""Execute code cells sequentially without launching a Jupyter server or modifying notebooks."""
from pathlib import Path
import sys
import nbformat
import matplotlib
matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for path in sorted((ROOT / "notebooks").glob("0*.ipynb")):
    book = nbformat.read(path, as_version=4)
    nbformat.validate(book)
    assert book.metadata.get("edupredict_pipeline_v4")
    namespace = {"__name__": "notebook_check"}
    for index, cell in enumerate(book.cells):
        if cell.cell_type == "code":
            exec(compile(cell.source, f"{path.name}:cell{index}", "exec"), namespace)
    print(f"PASS: {path.name}")
