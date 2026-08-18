# Flipkart Order Intelligence & Support Assistant

An end-to-end, integrated AI system built to streamline e-commerce customer support and catalog operations. This project unifies classical machine learning for order return-risk scoring, deep learning computer vision for product catalog classification, and an agentic retrieval-augmented generation (RAG) assistant built with LangGraph.

---

## System Architecture

```text
+-----------------------------------------------------------------------------------+
|                        FLIPKART ORDER ASSISTANT REPO                              |
|                                                                                   |
|  +-----------------------+   +-----------------------+   +---------------------+  |
|  |        PART 1         |   |        PART 2         |   |       PART 3        |  |
|  |  Return-Risk Scoring  |   |  Image Categorization |   |   Support Agent     |  |
|  |                       |   |                       |   |     (LangGraph)     |  |
|  |  [generate_orders.py] |   |  [Fashion-MNIST]      |   |                     |  |
|  |          |            |   |          |            |   |  +---------------+  |  |
|  |          v            |   |          v            |   |  | Policy KB RAG |  |  |
|  |  Random Forest Model  |   |  ResNet / Transfer    |   |  +---------------+  |  |
|  |          |            |   |          |            |   |         |           |  |
|  +----------|------------+   +----------|------------+   |         v           |  |
|             |                           |                |   +-----------+     |  |
|             v                           v                |   | LangGraph |     |  |
|  models/return_risk_model.pkl   models/product_classifier.pt | Router & |     |  |
|             |                           |                |   | Tools     |     |  |
|             +---------------------------+--------------->|   +-----------+     |  |
+-----------------------------------------------------------------------------------+

## Project Overview

This system combines tabular return risk prediction with visual product classification to assist e-commerce operations:
1. **Tabular Risk Scoring:** Evaluates order and customer features using a tuned Random Forest pipeline to assign return probabilities and risk tiers.
2. **Visual Product Classification:** Employs a transfer-learning ResNet-18 model trained on Fashion-MNIST to verify item categories visually.
3. **Agentic Tool Integration:** Merges visual prediction and tabular risk scoring to identify discrepancies between declared item categories and actual product images.
4. - **Interactive Web Interface:** A Streamlit dashboard (`app.py`) for interactive return risk assessment and policy Q&A.


## Repository Structure

```text
flipkart-order-assistant/
├── data/
│   ├── FashionMNIST/            # Dataset directory (auto-downloaded)
│   └── sample_images/           # Sample PNG images for tool evaluation
├── models/
│   ├── return_risk_model.pkl    # Serialized Random Forest pipeline
│   └── product_classifier.pt    # Saved PyTorch ResNet-18 model weights
├── src/
│   ├── train_return_risk.py     # Part 1: Tabular model training & threshold tuning
│   ├── train_image_classifier.py# Part 2: PyTorch feature extraction & evaluation
│   ├── export_sample_images.py  # Utility: Exports sample test PNGs
│   ├── agent_tools.py           # Part 3: Tool functions for risk & image analysis
│   └── agent.py                 # Part 4: CLI Risk Assistant Agent
├── .gitattributes               # Git LFS tracking configuration
├── README.md                    # Project documentation

```

---

## Key Results & Model Performance

### Part 1: Tabular Return Risk Model (Random Forest)

* **Optimal Decision Threshold ($t^*_{\text{rf}}$):** `0.5000`
* **Risk Categorization Buckets:**
* **Low Risk ($p < 0.35$):** Standard automated order processing.
* **Medium Risk ($0.35 \le p < 0.5000$):** Proactive delivery confirmation SMS triggered.
* **High Risk ($p \ge 0.5000$):** Pre-dispatch quality verification required.



---

### Part 2: Product Image Classifier (ResNet-18)

* **Backbone Architecture:** Pretrained ResNet-18 (frozen feature extractor + 512 $\to$ 10 linear classification head)
* **Preprocessing Resolution:** $64 \times 64$ RGB (Grayscale 3-channel expansion, ImageNet normalization)
* **Validation Accuracy:** `88.06%`
* **Held-Out Test Accuracy:** `87.96%` *(Surpasses $\ge 80\%$ evaluation benchmark)*

#### Classification Performance Summary

| Category | Precision | Recall | F1-Score | Support |
| --- | --- | --- | --- | --- |
| **Bag** | 0.97 | 0.98 | **0.98** | 1000 |
| **Trouser** | 0.98 | 0.97 | **0.97** | 1000 |
| **Sandal** | 0.96 | 0.96 | **0.96** | 1000 |
| **Ankle Boot** | 0.96 | 0.96 | **0.96** | 1000 |
| **Sneaker** | 0.94 | 0.94 | **0.94** | 1000 |
| **Dress** | 0.85 | 0.88 | **0.86** | 1000 |
| **T-shirt/top** | 0.83 | 0.84 | **0.83** | 1000 |
| **Pullover** | 0.87 | 0.78 | **0.82** | 1000 |
| **Coat** | 0.74 | 0.85 | **0.79** | 1000 |
| **Shirt** | 0.71 | 0.64 | **0.67** | 1000 |

#### Confusion Pattern Analysis

1. **Pullover vs. Coat / Shirt:** Long-sleeve torso silhouettes share near-identical outer boundaries in low-resolution grayscale images. Subtle visual markers like front zippers vs. solid knits account for minor misclassifications.
2. **Ankle Boot vs. Sneaker:** Low-top ankle boots and high-top sneakers feature overlapping height profiles and compact footwear contours.

---

## Installation & Setup

### 1. Prerequisites

Ensure Python 3.9+ is installed.

### 2. Environment Setup

```powershell
# Clone the repository
git clone https://github.com/vkatare/flipkart-order-assistant.git
cd flipkart-order-assistant

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # On Windows PowerShell

# Install required dependencies
pip install torch torchvision numpy pandas scikit-learn joblib pillow tqdm

---

## Step-by-Step Execution Guide

To run the complete workflow from model training to agent inference, execute the scripts in sequence:

```powershell
# Step 1: Train tabular return risk model and save return_risk_model.pkl
python src/train_return_risk.py

# Step 2: Extract features with ResNet-18, train linear head, and save product_classifier.pt
python src/train_image_classifier.py

# Step 3: Export sample images to data/sample_images/ for assistant testing
python src/export_sample_images.py

# Step 4: Run self-test suite for agent tools
python src/agent_tools.py

# Step 5: Launch the Flipkart Order Risk Assistant Agent
python src/agent.py

```

### Running the Application
1. Interactive Web Dashboard (app.py)
Launch the Streamlit web application to interactively enter order details, upload product images, view return risk scores, and ask policy questions:
-------------------------------
streamlit run app.py
-------------------------------

2. CLI Agent (src/agent.py)
Run the terminal-based assessment agent directly:
-------------------------------
python src/agent.py
-------------------------------

## Assistant CLI Output Example

Running `python src/agent.py` generates consolidated order risk reports:

```text
==================================================
        FLIPKART ORDER RETURN RISK REPORT         
==================================================
Product Category (Declared) : Sneaker
Product Category (Visual)   : Sneaker (Confidence: 99.50%)
Catalog Mismatch Detected   : NO
--------------------------------------------------
Estimated Return Probability: 56.01%
Optimal Threshold (t*)      : 0.5
Risk Classification Tier   : High Risk
Predicted Return Outcome    : RETURN LIKELY
--------------------------------------------------
Action Recommendation       : Flag for quality verification prior to dispatch
==================================================

```
