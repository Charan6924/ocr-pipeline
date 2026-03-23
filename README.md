# RenAIssance: Transformer-Based OCR for Early Modern Spanish

**GSoC 2026 Evaluation Submission | Project: Text Recognition with Transformer Models and LLM Integration**
**Applicant:** Charan Vardham

## Objective
This repository contains the evaluation pipeline for the RenAIssance project (Test I). The goal is to accurately digitize 17th-century Spanish printed documents using a state-of-the-art Vision-Language Model (VLM), handling severe ink degradation, archaic typography, and complex layouts while strictly ignoring marginalia.

## Architecture Pipeline

The system is built on a 4-stage architecture designed for high-compute environments (H100/A100) using `bfloat16` precision to maximize throughput. 

### 1. Data Ingestion & Preprocessing (`preprocessing.py`)
Raw PDFs are converted to high-resolution images. To satisfy the requirement of ignoring embellishments and marginalia, an OpenCV heuristic pipeline is applied:
* **Binarization:** An adaptive thresholding technique cleans the image, preserving the details of the old, faded font.
* **Layout Isolation:** Computer vision automatically finds the main text block and crops the image to exclude marginal notes.

### 2. Core OCR Engine (`inference.py`)
The cropped page images are passed to a pre-trained Vision-Language Model (VLM). The model used is `ucaslcl/GOT-OCR2_0`, a powerful General-purpose OCR Transformer that can accurately transcribe the text even with non-standard fonts and degradation. The model is used directly without any fine-tuning.

### 3. LLM-Powered Correction (`llm_inference.py`)
The raw text output from the OCR model is then passed to a large language model (GPT-4o-mini) for post-correction and normalization. A detailed prompt instructs the LLM to act as a paleography expert, applying a strict set of rules to normalize archaic spellings (e.g., long-s, u/v interchange), expand abbreviations, and remove artifacts, while preserving the original content.

### 4. Evaluation (`eval.py`)
The final, cleaned text is evaluated against a ground-truth transcription. The primary metric used is Character Error Rate (CER), which is well-suited for historical texts with non-standard spellings. The evaluation script calculates CER for both the raw and the LLM-cleaned OCR output, allowing for a clear measure of the LLM correction step's effectiveness.
