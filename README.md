![Deconstructing Fake News](wallpaper.png)
🏛️ 🚀 Deconstructing Fake News Narratives to Enhance Media Literacy 🚀 🏛️
=====================================================================

A Generative AI–powered tool that leverages Natural Language Processing(NLP), Sentiment Analysis, and pretrained large language models(LLMs) for a variety of use cases to detect potential misinformation in news articles, explain why they may be risky (via disinformation tactic detection), and help users make better-informed decisions—improving media literacy.

📄 Table of Contents
--------------------
- 🎯 [Project Overview](#project-overview)  
- ✨ [Key Features](#key-features)  
- 🏗️ [Architecture & Workflow](#architecture--workflow)  
- 📊 [Data & Model Training](#data--model-training)  
- 🛠️ [Core Components](#core-components)  
  - 🧠 [1. Fake vs. Real News Classifier](#1-fake-vs-real-news-classifier)  
  - 🔍 [2. Vector Search & Similarity Analysis](#2-vector-search--similarity-analysis)  
  - 🚨 [3. Disinformation Tactic Identification](#3-disinformation-tactic-identification)  
  - ✅ [4. Fact-Checking Integration](#4-fact-checking-integration)  
  - 🤖 [5. User Explanation Layer (Generative AI)](#5-user-explanation-layer-generative-ai)  
- 🛡️ [Tech Stack](#tech-stack)  
- ⚙️ [Installation & Setup](#installation--setup)  
- 📈 [Evaluation & Metrics](#evaluation--metrics)  
---

## 🎯 Project Overview
This repository implements a GenAI tool for:
- 🚩 **Flagging potential misinformation** in news articles.  
- 🧩 **Explaining risk factors** via detection of disinformation tactics.  
- 🔗 **Augmenting user judgment** through similarity comparisons with known fake/real articles and fact-check results.  
- ✍️ **Synthesizing findings** into a concise, human-readable summary using a generative model.

The ultimate goal is to empower users to recognize misleading narratives and improve their media literacy in an interactive, understandable way.

---

## ✨ Key Features
- 🔎 **Supervised Fake/Real News Classification**  
  - Fine-tuned transformer-based classifier on labeled dataset (titles & texts).
- ⚡ **Retrieval-Augmented Similarity Search**  
  - Vector embeddings stored in MongoDB Atlas Vector Search for semantic similarity against known fake/real articles.
- 🚨 **Disinformation Tactic Detection**  
  - Rule-based (regex, heuristics) and NLP-based methods (syntax/entity analysis, sentiment) to identify sensationalism, vagueness, appeals to emotion, loaded language, conspiracy markers, etc.
- ✅ **Fact-Checking API Integration**  
  - Query Google Fact Check Tools API for provided claims or URLs to fetch existing verifications.
- 🤖 **Generative Explanation Layer**  
  - Uses a distilled Llama-based model (DeepSeek-R1-Distill-Llama-8B-GGUF) to combine classifier verdict, similarity insights, tactic detections, and fact-check findings into a clear user-facing summary.
- ☁️ **Scalable Training & Deployment**  
  - Trained on Google Vertex AI (textembedding-005 for embeddings, Vertex AI Training for fine-tuning).
- 🗄️ **Caching & Vector DB**  
  - MongoDB Atlas for vector storage and caching embeddings without storage limits.
- 🌐 **Interactive Interface**  
  - Simple web UI for users to input an article title, key claim, or URL and receive analysis.

---

## 🏗️ Architecture & Workflow
1. **📝 User Input**  
   - Accepts: article text, title, key claim, or URL.
2. **🧹 Preprocessing**  
   - Clean text (remove special characters, lowercase), tokenize, remove stopwords.  
   - Generate sentence/document embeddings via Vertex AI textembedding-005 API.
3. **🧠 Fake/Real Classifier**  
   - Fine-tuned transformer (e.g., BERT-mini or similar) on embeddings + text features.  
   - Outputs label (“Fake” / “Real”) with confidence score.
4. **🔍 Vector Similarity Search**  
   - Query MongoDB Atlas Vector Search with the new embedding.  
   - Retrieve top-K most similar known Fake articles and top-K Real articles.  
   - Compute similarity statistics (e.g., “80% of top-10 neighbors are Fake with avg sim = 0.78”).
5. **🚨 Disinformation Tactic Identification**  
   - **Rule-based**: regex patterns (ALL CAPS, clickbait phrases).  
   - **Lexicon-based**: emotional/conspiracy term lists.  
   - **NLP-based**: Google Natural Language API for syntax, entity, sentiment analysis.
6. **✅ Fact-Checking Integration**  
   - Query Google Fact Check Tools API with user-provided claim(s) or URL.  
   - Fetch matches: claim, rating (False, Misleading, True), publisher, date.
7. **🤖 Generative Explanation**  
   - Collate:  
     - Classifier result + confidence.  
     - Similarity summary (top neighbors breakdown).  
     - Detected tactics list with notes.  
     - Fact-check findings.  
   - Prompt distilled generative model (DeepSeek-R1-Distill-Llama-8B-GGUF) with structured input.  
   - Produce a clear summary: reasons to be cautious/confident, what to watch for, suggestions for further verification.
8. **📤 Output to User**  
   - Display label, confidence score, similarity insights, tactic detections, fact-check results, and the generative summary in a user-friendly format.

---

## 📊 Data & Model Training
- **📥 Dataset**: KaggleHub datasets containing labeled Real vs. Fake news articles (~95K samples).  
  - Fields: title, text, label (0 = Fake, 1 = Real), optionally metadata (date, source).  
- **🧹 Preprocessing**:  
  - Remove HTML tags, special characters; lowercase; tokenize; remove stopwords.  
  - Optionally filter very short or irrelevant articles.
- **📈 Embeddings**:  
  - Use Vertex AI textembedding-005 API to generate vector representations for text/title.  
  - Store embeddings in MongoDB Atlas for both training analysis and inference similarity search.
- **⚙️ Training Setup**:  
  - Split dataset: train/validation/test (e.g., 70/15/15).  
  - Fine-tuned a transformer (e.g. BERT-mini) on text tokens or embeddings.  
  - Track metrics: accuracy, precision, recall, F1 on validation/test.
- **🔧 Hyperparameters & Infrastructure**:  
  - Vertex AI Training for scalable GPU training.  
  - Early stopping based on validation loss.  
  - Save best checkpoints; export model for serving.
- **📊 Results (Example Placeholder)**:  
  - Achieved ~84% accuracy on held-out test set.
  
---

## 🛠️ Core Components

### 🧠 1. Fake vs. Real News Classifier
- **Model**: Fine-tuned transformer-based classifier (BERT-mini).  
- **Inputs**: Title & body text (optionally combined features).  
- **Output**: Label (“Fake” / “Real”) + confidence score.  
- **Implementation Notes**:  
  - Use Hugging Face Transformers, PyTorch or TensorFlow.  
  - Save artifacts (tokenizer, checkpoint) for inference.  
  - Write test cases to verify inference outputs.

### 🔍 2. Vector Search & Similarity Analysis
- **Embedding Generation**: Vertex AI textembedding-005 API.  
- **Vector DB**: MongoDB Atlas Vector Search.  
- **Workflow**:  
  1. **On training**: generate embeddings for all labeled articles; store in collection with metadata (label, title, snippet).  
  2. **On inference**: generate embedding for input; query top-K similar vectors.  
  3. Retrieve similarity scores & labels of neighbors; compute summary (e.g., “Top-5: 4 Fake (avg sim 0.82), 1 Real (avg sim 0.65)”).
- **Benefits**:  
  - Contextualizes classifier output.  
  - Helps when classifier confidence is low.

### 🚨 3. Disinformation Tactic Identification
- **Rule-based Patterns**:  
  - ALL CAPS titles 🔠, excessive punctuation (!!!, ???).  
  - Clickbait phrases (“You won’t believe”, “Shocking truth”).  
  - Vague sourcing (“Experts say” without naming).  
- **Lexicon-based**:  
  - Emotional word lists (anger, fear, excitement).  
  - Conspiracy markers (“hidden truth”, “they don’t want you to know”).  
- **NLP-based** (Google Natural Language API):  
  - **Syntax analysis**: identify manipulative sentence structures.  
  - **Entity analysis**: detect vague vs. credible source mentions.  
  - **Sentiment analysis**: high emotion without facts.
- **Output**:  
  - List of detected tactics, each with a brief explanation.  
  - Example:  
    - 🔥 **Sensationalism**: Title in ALL CAPS, emotionally charged wording.  
    - ❓ **Vague Sourcing**: Uses “experts say” without naming a source.

### ✅ 4. Fact-Checking Integration
- **API**: Google Fact Check Tools API.  
- **Workflow**:  
  - Extract key claim(s) or URL from user input.  
  - Query API to get matched fact-check entries.  
  - Parse results: claim text, rating (False, Misleading, True), publisher, date.  
- **Output**:  
  - Summary list/table of fact-check findings:  
    - 📰 Claim: “[quoted claim]”  
      - 🔖 Rating: False  
      - 🏢 Publisher: Snopes  
      - 📅 Date: Jan 15, 2024
- **Benefit**:  
  - Provides authoritative verification or debunking references.

### 🤖 5. User Explanation Layer (Generative AI)
- **Model**: DeepSeek-R1-Distill-Llama-8B-GGUF (distilled/quantized).  
- **Prompt Design**:  
  - Structure input with labeled sections:  
    ```
    News Title: [title]
    Classifier Prediction: [Fake/Real] (confidence: X%)
    Similarity Analysis: [“Top 5 neighbors: 4 Fake (avg sim 0.82), 1 Real (avg sim 0.65)”]
    Detected Tactics: [List with notes]
    Fact-Check Results: [List of claims & ratings]
    ```
  - Example prompt:  
    > “Analyze the following news assessment and provide a concise summary for the user, highlighting key reasons for concern or confidence, and advising what to look out for:
    > ```
    > News Title: ...
    > Classifier Prediction: ...
    > Similar Articles Found:
    >   Label: Fake, Similarity: ...
    >   Label: Real, Similarity: ...
    > Fact-Check API Results:
    >   Claim: ..., Rating: ..., Publisher: ...
    > Detected Disinformation Tactics: ...
    > ```”
- **Output**:  
  - A friendly summary paragraph or bullet points:  
    - Why the article may be unreliable or likely genuine.  
    - Which tactics reduce credibility.  
    - How similarity to known Fake/Real informs trust.  
    - Relevant fact-check outcomes.  
    - Suggestions for further verification (e.g., check original source, corroborate with reputable outlets).  
- **Deployment**:  
  - Local inference runtime (e.g., llama.cpp) or lightweight server.  
  - Cache prompt templates; optimize for latency.

---

## 🛡️ Tech Stack
- **Languages & Frameworks**:  
  - Python, PyTorch/TensorFlow, Hugging Face Transformers.  
  - FastAPI for backend server deployment.  
- **Cloud & APIs**:  
  - Google Vertex AI: text embeddings, model training, endpoints.  
  - Google Natural Language API: syntax, sentiment, entity analysis.  
  - Google Fact Check Tools API.  
- **Database**:  
  - MongoDB Atlas with Vector Search for embedding storage & similarity queries.  
- **Generative Model Inference**:  
  - Local inference of DeepSeek-R1-Distill-Llama-8B-GGUF via llama.cpp.  
- **DevOps & Deployment**:  
  - Docker: containerize API, model servers.  
  - Kubernetes or Google Cloud Run for scalable deployment (optional).  
  - CI/CD: GitHub Actions for linting, testing, deployment pipelines.  
- **Utilities**:  
  - Python scripts for preprocessing, embedding generation, batch inference.  
  - Logging, monitoring (inference latency, error rates).  
- **Frontend**:  
  - Simple React or Streamlit interface for user input and results display.

---

## ⚙️ Installation & Setup

**🔄 Clone the Repository**  
```bash
git clone https://github.com/JiaJun-dotcom/deconstruct-fake-news.git
cd deconstruct-fake-news
