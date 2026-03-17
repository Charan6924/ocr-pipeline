# RenAIssance: Transformer-Based OCR for Early Modern Spanish

**GSoC 2026 Evaluation Submission | Project: Text Recognition with Transformer Models and LLM Integration**
**Applicant:** Charan Vardham

## Objective
This repository contains the evaluation pipeline for the RenAIssance project (Test I). The goal is to accurately digitize 17th-century Spanish printed documents using a state-of-the-art Vision-Language Model (VLM), handling severe ink degradation, archaic typography, and complex layouts while strictly ignoring marginalia.

## Architecture Pipeline

The system is built on a 5-stage architecture designed for high-compute environments (H100/A100) using `bfloat16` precision to maximize throughput. 

### 1. Data Ingestion & Preprocessing (`preprocessing.py`)
Raw PDFs are converted to high-resolution images. To satisfy the requirement of ignoring embellishments and marginalia, an OpenCV heuristic pipeline is applied:
* **Binarization:** Sauvola adaptive thresholding is utilized instead of global/Gaussian methods to preserve the thin, degraded strokes of early modern typefaces without washing out the faded ink.
* **Layout Isolation:** Heavy morphological dilation and contour area mapping are used to programmatically isolate and crop the main text block, stripping away noise and marginalia before tensor conversion.

### 2. Few-Shot Document Adaptation (`dataset