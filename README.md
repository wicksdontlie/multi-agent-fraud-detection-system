# 🛡️ Multi-Agent Fraud Detection & AML Investigation System
---
<img width="1024" height="572" alt="multi_agent_image_github" src="https://github.com/user-attachments/assets/946736a1-800f-494a-8672-37f7674bef20" />

An AI-powered Multi-Agent Fraud Detection platform that leverages Generative AI, Retrieval-Augmented Generation (RAG), Vector Search, and Autonomous Agents to analyze suspicious activities, assess risk levels, and generate fraud investigation reports.


## 🚀 Project Overview

Financial institutions face increasing challenges in identifying fraudulent transactions and ensuring Anti-Money Laundering (AML) compliance.

This project implements a Multi-Agent AI architecture where specialized agents collaborate to:

* Retrieve relevant AML and fraud knowledge
* Analyze suspicious activities
* Assess fraud risk levels
* Measure confidence scores
* Generate structured investigation reports

The system uses Retrieval-Augmented Generation (RAG) to provide context-aware responses grounded in AML guidelines and fraud investigation documents.

---

## 🏗️ Architecture

```text
User Query
     │
     ▼
🔍 Retriever Agent
     │
     ▼
🕵️ Fraud Analysis Agent
     │
     ▼
⚠️ Risk Assessment Agent
     │
     ▼
📈 Confidence Agent
     │
     ▼
📄 Report Generation Agent
     │
     ▼
Final Fraud Investigation Report
```

---

## 🤖 Agents

### 🔍 Retriever Agent

Responsible for:

* Semantic search
* Vector similarity retrieval
* Fetching relevant AML/Fraud documents
* Building contextual knowledge

---

### 🕵️ Fraud Analysis Agent

Responsible for:

* Fraud pattern analysis
* Transaction behavior assessment
* Context-based reasoning
* Fraud investigation support

---

### ⚠️ Risk Assessment Agent

Responsible for:

* Risk classification
* Fraud severity scoring
* Investigation recommendations

Risk Levels:

* Low Risk
* Medium Risk
* High Risk

---

### 📈 Confidence Agent

Responsible for:

* Response confidence estimation
* Retrieval quality assessment
* Trustworthiness indicators

---

### 📄 Report Agent

Responsible for:

* Investigation report generation
* Executive summaries
* Structured fraud findings

---

## 🧠 RAG Pipeline

The project uses Retrieval-Augmented Generation (RAG).

### Workflow

1. PDF Documents Ingestion
2. Text Chunking
3. Embedding Generation
4. FAISS Vector Index Creation
5. Semantic Retrieval
6. Context Injection
7. LLM-Based Response Generation

---

## 📚 Knowledge Base

The system is trained on AML and Fraud Investigation documents.

Examples:

* AML Guidelines
* KYC Requirements
* Suspicious Transaction Reporting
* Financial Crime Regulations
* Compliance Frameworks

---

## 🛠️ Technology Stack

### Frontend

* Streamlit

### Backend

* Python

### AI & GenAI

* OpenRouter
* Large Language Models (LLMs)
* Retrieval-Augmented Generation (RAG)

### Vector Database

* FAISS

### Embeddings

* Sentence Transformers

### Environment

* Python Virtual Environment
* dotenv

### Version Control

* Git
* GitHub

---

## 📂 Project Structure

```text
Multi_agent_project/

│
├── agents/
│   ├── retriever_agent.py
│   ├── fraud_agent.py
│   ├── risk_agent.py
│   ├── confidence_agent.py
│   └── report_agent.py
│
├── documents/
│   ├── AMLCFTguidelines_rag.pdf
│   ├── FOREIGN_rag.pdf
│   └── ...
│
├── app.py
├── retriever.py
├── embeddings.py
├── rag_pipeline.py
├── preprocess_pdfs.py
├── requirements.txt
└── .env
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/multi-agent-fraud-detection-system.git

cd multi-agent-fraud-detection-system
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows

```bash
venv\Scripts\activate
```

Linux / Mac

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file:

```env
OPEN_ROUTER_API_KEY=your_api_key
```

---

## ▶️ Run Application

```bash
streamlit run app.py
```

Application will start at:

```text
http://localhost:8501
```

---

## 💡 Example Query

```text
A customer deposits ₹9,80,000 in cash and transfers the funds to three overseas accounts within one hour. Analyze the fraud risk and provide recommendations.
```

---

## 📈 Key Features

✅ Multi-Agent Architecture

✅ Retrieval-Augmented Generation (RAG)

✅ Semantic Search using FAISS

✅ OpenRouter LLM Integration

✅ AML Knowledge Base

✅ Fraud Investigation Workflow

✅ Risk Assessment

✅ Confidence Scoring

✅ Automated Report Generation

✅ Streamlit Dashboard

---

## 🎯 Skills Demonstrated

* Generative AI
* Agentic AI
* Multi-Agent Systems
* Retrieval-Augmented Generation
* Vector Databases
* Semantic Search
* Prompt Engineering
* Large Language Models
* OpenRouter API Integration
* Streamlit Development
* AI Application Deployment
* Fraud Detection
* Anti-Money Laundering (AML)
* Python Development

---

## 🔮 Future Enhancements

* LangGraph Integration
* Real-Time Transaction Monitoring
* Graph-Based Fraud Detection
* PDF Report Export
* Role-Based Access Control
* Cloud Deployment
* Advanced Risk Scoring Models
* Fraud Analytics Dashboard

---

## 👨‍💻 Author

Adnan Shaikh

AI Engineer | Data Scientist | Generative AI Developer

Building intelligent systems using Multi-Agent AI, RAG, LLMs, and Machine Learning.

---

⭐ If you found this project useful, consider giving it a star on GitHub.
