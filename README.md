--- Deconstructing Fake News Narratives to enhance Media Literacy --- 

This is a GenAI tool that incorporates Artificial Intelligence techniques, including 
Natural Language Processing, Sentiment Analysis and RAG(Retrieval-Augmented-Generation), to flag potential misinformation, and explain why it’s risky with tactic detection capabilities, helping users make better-informed decisions and improve their media literacy.


Trained on KaggleHub Datasets of classified Real and Fake news, and integrates Google AI for to train and deploy this model using Vertex AI Training, and APIs such as Google's Natural Language API and Fact Check Tools API.
- MongoDB as vector database for storing generated vector embeddings.(Implementing caching without db storage limits)

* Train a fake/real news classifier(supervised learning model): 
Fine-tuned a pretrained text classification transformer model on labeled "Text" and "Title" data to predict if a new, unseen article is "Fake"(0) or "Real"(1).

Preprocessing: Clean text (remove special characters, lowercase), tokenize, remove stopwords.
Feature Extraction: Sentence Embeddings (from Vertex AI textembedding-005)
Model Selection: Fine-tuned Transformer(bertmini), trained on sentence embeddings data to predict unseen news articles as Fake/Real.
Training: Split 95k dataset into training, validation, and test sets.

How Model outputs search results:
Displays the label(fake/real) and confidence score of prediction.
(eg Classified: Fake/Real, confidence_score = ?)


* Vector Search:
Creates vector embeddings of new unseen text/article, and set up Atlas Vector Search for lightning‑fast semantic similarity searches to compare with pre-stored vector embeddings of FAKE/REAL news dataset for further confirmation of classifier results to improve credibility:
eg: 
"This new article is very similar to __ FAKE articles and __ REAL articles, with similarity scores of (...)." 
-- Post-ATLAS vector search.


* DisInformation Tactic Identification (Rule-Based & NLP):
Identifying common linguistic patterns or characteristics often found in disinformation. This goes beyond just "fake" and explains how it might be misleading.
Examples of Tactics to Detect:
1. Sensationalism/Clickbait: Excessive capitalization in titles, emotionally charged words (use sentiment scores and lexicons), exclamation points.
2. Lack of Specificity/Vagueness: Overuse of general terms, lack of named sources or concrete data.
3. Appeals to Emotion over Logic: High sentiment scores (positive or negative) without corresponding factual density.
4. Use of Loaded Language: Words with strong connotations.
5. Conspiracy Markers: Phrases like "they don't want you to know," "the hidden truth."

- Rule-based: Regular expressions for patterns (e.g., ALL CAPS TITLE?).
- Lexicon-based: Lists of emotionally charged words, conspiracy-related terms.
- NLP (Google Natural Language API):
  Analyze syntax to find sentence structures common in persuasive/manipulative text.
- Use entity analysis – does the article mention credible sources or just vague "experts"?

* Fact-Checking:
Allows searching of a global repository of fact-checks from reputable publishers.
User Input: When a user submits a news article's title, a key claim from its text, or its URL, the application takes this input and queries the Google Fact Check Tools API.

* User Explanation Layer ( Generative AI - PaLM/Gemini):
After all the analysis (your classifier, sentiment, tactic ID, fact-check API, vector search), a generative model is used to synthesize the findings into a human-readable summary for the user.
- Model Selected: DeepSeek-R1-Distill-Llama-8B-GGUF
- Distilled and quantized version of Deepseek-R1 Llama 3 model with 8B params, to suit limitations of project 

* Sample Prompt:
"Analyze the following news assessment and provide a concise summary for the user, highlighting key reasons for concern or confidence.

News Title: [Title of user's article]
Our Classifier Prediction: [Fake/Real (with confidence score)]
Sentiment Score: [Score] (e.g., Highly Negative)
Detected Disinformation Tactics: [List of tactics]
Similar Articles Found:
  - Label: [Fake/Real], Sentiment: [Score]
  - Label: [Fake/Real], Sentiment: [Score]
Google Fact Check API Results:
  - Claim: [Claim 1], Rating: [False], Publisher: [Snopes]
  - Claim: [Claim 2], Rating: [Misleading], Publisher: [PolitiFact]

Based on this, explain to the user why they should be cautious or confident about this news, and what specific things to look out for."
