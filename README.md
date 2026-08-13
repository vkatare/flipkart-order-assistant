# Flipkart Order Intelligence & Support Assistant 🛒🤖

An end-to-end, integrated AI system built to streamline e-commerce customer support and catalog operations. This project unifies classical machine learning for order return-risk scoring, deep learning computer vision for product catalog classification, and an agentic retrieval-augmented generation (RAG) assistant built with LangGraph.

---

## 📐 System Architecture

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
