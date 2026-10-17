# Manuscript sources

The manuscript, the Supporting Information and the highlights are built from the repository
outputs so that every number is traceable:

```bash
python manuscript/make_tables.py                      # tables.json, scalars.json, image_sizes.json
cd manuscript
node build_docx.js si_content.js MOF_Supporting_Information_v2.docx   # also writes si_numbers.json
node build_docx.js ms_content.js Active_learning_MOF_discovery_v4.docx
node build_docx.js highlights_content.js Highlights_v4.docx
```

`refs.js` holds the reference list; citations are `[@key]` in the text and are numbered in order of
first appearance. Figures are read from `results/figures/*.png` (the vector PDFs next to them are the
submission files). Requires Node with the `docx` package and Python with pandas and Pillow.
