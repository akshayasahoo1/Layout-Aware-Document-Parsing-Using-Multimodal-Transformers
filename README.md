# Layout-Aware Document Parsing Using Multimodal Transformers

**Technical Report | Akshaya Kumar Sahoo | 2026**

> A vision-first, layout-aware pipeline that converts complex unstructured PDFs into deterministic, machine-readable JSON using fine-tuned multimodal transformers.

---

## Problem

Standard NLP models treat document text as a flat sequence — they have no awareness of where on the page a word appears. A number inside a table cell carries completely different meaning from the same number in a heading or footer, but a text-only model cannot distinguish them.

This project addresses the core challenge of **unstructured document parsing**: extracting structured information from multi-column invoices, scanned forms, and mixed-layout PDFs where spatial relationships between tokens are as semantically important as the tokens themselves.

---

## Approach

Fine-tuned **LayoutLMv3** — a multimodal transformer that jointly encodes text tokens, their bounding-box coordinates, and visual page features — on a custom-curated dataset of scanned documents.

### Pipeline
```
PDF Input
   ↓
Tesseract OCR  →  Bounding-box extraction
   ↓
LayoutLMv3 Fine-tuning  (text + layout + visual)
   ↓
Post-processing  →  Structured JSON Output
```

### Key Components

- **OCR layer** — Tesseract extracts raw text with per-token bounding-box coordinates
- **Layout encoding** — (x0, y0, x1, y1) normalised coordinates fed as positional inputs to LayoutLMv3
- **Visual features** — Page rendered as image, patched and encoded by LayoutLMv3's visual backbone
- **Token classification head** — Classifies each token as TABLE, KEY, VALUE, HEADING, or OTHER
- **Post-processing** — Merges classified tokens into structured key-value pairs and table rows

---

## Dataset

Custom-curated dataset of **8,000 annotated document pages** including:

- Scanned invoices and receipts
- Multi-column government forms
- Mixed-layout financial documents

Annotation pipeline combined Tesseract OCR with rule-based heuristics for semi-automated bounding-box labeling, with manual review for edge cases such as merged cells, rotated text, and multi-page spans.

---

## Evaluation

Benchmarked against two standard document AI datasets:

| Dataset | Task | Metric |
|---------|------|--------|
| **FUNSD** | Form understanding — key-value extraction | Entity-level F1 |
| **DocVQA** | Document visual question answering | ANLS score |

All experiments tracked using **MLFlow** for full reproducibility and model versioning.

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Model | LayoutLMv3 (microsoft/layoutlmv3-base) |
| OCR | Tesseract 5.x, pytesseract |
| Framework | PyTorch, Hugging Face Transformers |
| Experiment tracking | MLFlow |
| Deployment | Docker, AWS EC2, FastAPI |
| Evaluation | FUNSD, DocVQA |

---

## Results

- Structured JSON extraction from complex multi-column documents with improved F1 over text-only BERT baseline on FUNSD
- p99 inference latency under **800ms** at 200 concurrent requests on AWS EC2
- Handles real-world edge cases: merged table cells, rotated text blocks, multi-page key-value spans

---

## References

- Huang et al. (2022). *LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking.* ACM MM 2022.
- Jaume et al. (2019). *FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents.* ICDAR-OST 2019.
- Mathew et al. (2021). *DocVQA: A Dataset for VQA on Document Images.* WACV 2021.
