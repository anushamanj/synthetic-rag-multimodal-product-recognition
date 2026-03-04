<div align="center">

# Synthetic Data and Hybrid RAG Labelling for Multimodal Product Recognition

Hybrid RAG-based Multimodal Product Attribute Recognition using **Stable Diffusion, CLIP, FAISS and Gemini 2.5 Flash**

🏆 **Top 5 Project — MSc Data Science**

[Architecture](#system-architecture) • [Demo](#demo) • [Installation](#running-the-project)

</div>


![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![FAISS](https://img.shields.io/badge/VectorDB-FAISS-green)
![CLIP](https://img.shields.io/badge/OpenAI-CLIP-purple)
![Gemini](https://img.shields.io/badge/LLM-Gemini%202.5-orange)
![RAG](https://img.shields.io/badge/Architecture-Hybrid%20RAG-yellow)

# Synthetic Data and Hybrid RAG Labelling for Multimodal Product Recognition

Hybrid RAG-based Multimodal Product Attribute Recognition using **Stable Diffusion, CLIP, FAISS and Gemini 2.5 Flash**.

This project proposes a **Hybrid Retrieval-Augmented Generation (RAG) pipeline** for automatic product attribute prediction in fashion datasets.

The system generates synthetic product images, retrieves similar products using vector similarity search, and performs grounded reasoning with an LLM to predict structured attributes such as category, color, material and size.

🏆 **Recognized among the Top 5 Best Projects in the MSc program**

---

# System Architecture

![Architecture](assets/Model%20Design.png)

### Pipeline Overview


The system integrates **image generation, multimodal retrieval, and LLM reasoning** to produce accurate product attribute labels.

---

# Key Features

• Synthetic fashion image generation using Stable Diffusion  
• Multimodal retrieval using CLIP embeddings  
• FAISS vector similarity search for evidence retrieval  
• Hybrid RAG attribute prediction using Gemini  
• Interactive Streamlit UI for experimentation  
• Automated product attribute labeling  

Predicted attributes include:

- Category
- Color
- Material
- Pattern
- Size

---

# Demo

## Streamlit Interface

![UI](assets/Screenshot%20(1262).png)

---

## Generated Product + Attribute Prediction

![Prediction](assets/Screenshot%20(1266).png)

---

## Retrieval Evidence from Dataset

![Retrieval](assets/Screenshot%20(1268).png)

---

## Fashion Dataset Examples

![Dataset](assets/Screenshot%20(1281).png)

---

# Technologies Used

## Machine Learning / AI

- Stable Diffusion v1.5
- CLIP (Contrastive Language Image Pretraining)
- Gemini 2.5 Flash
- Hybrid Retrieval-Augmented Generation (RAG)

## Data Infrastructure

- FAISS Vector Database
- Embedding Similarity Search

## Development

- Python
- Streamlit
- Jupyter Notebook

---

# Running the Project

### Clone the repository


---

# Evaluation

The system evaluates prediction quality using:

- Precision
- Recall
- F1 Score
- CLIP similarity

Hybrid RAG improves prediction accuracy by grounding LLM reasoning with retrieved product evidence.

---

# Future Improvements

• Larger fashion datasets  
• Improved attribute ontology  
• Production-grade vector databases (Pinecone / Weaviate)  
• Fine-tuned multimodal models  

---

# Author

**Anusha Manjunath**  
MSc Data Science
