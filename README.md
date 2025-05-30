--- Deconstructing Fake News Narratives to enhance Media Literacy --- 

This is a GenAI model that integrates various types of Artificial Intelligence, including 
Natural Language Processing(NLP) and Machine Learning, on top of your general fake news detector.

Trained on KaggleHub Datasets of classified Real and Fake news, and integrates Google AI for to train and deploy this model using Vertex AI Training, and APIs such as Google's Natural Language API and Fact Check Tools API.

Connected to MongoDB.

* Train a fake/real news classifier(supervised learning model): 
A ML model trained on labeled "Text" and "Title" data to predict if a new, unseen article is "Fake"(0) or "Real"(1).
How to build:
Preprocessing: Clean text (remove special characters, lowercase), tokenize, remove stopwords.

Feature Extraction: Sentence Embeddings (from Vertex AI textembedding-gecko)

Model Selection: Fine-tuned Transformer(eg BERT), trained on sentence embeddings data to predict unseen news articles as Fake/Real.
(
How Model outputs search results:
Displays the label(fake/real) and sentiment_score of similar articles
eg: 
"This new article is very similar to 5 known FAKE articles in our database, which also had highly negative sentiment." -- Post-ATLAS vector search.
)

Training: Split 78k dataset into training, validation, and test sets. Train the model on the training set, tune hyperparameters on the validation set, and evaluate final performance on the test set.

MongoDB Integration: Store model performance metrics/serialized model versions


* DisInformation Tactic Identification (Rule-Based & NLP):
Identifying common linguistic patterns or characteristics often found in disinformation. This goes beyond just "fake" and explains how it might be misleading.
Examples of Tactics to Detect:
1. Sensationalism/Clickbait: Excessive capitalization in titles, emotionally charged words (use sentiment scores and lexicons), exclamation points.
2. Lack of Specificity/Vagueness: Overuse of general terms, lack of named sources or concrete data.
3. Appeals to Emotion over Logic: High sentiment scores (positive or negative) without corresponding factual density.
4. Use of Loaded Language: Words with strong connotations.
5. Whataboutism Markers: Phrases like "But what about..."
6. Conspiracy Markers: Phrases like "they don't want you to know," "the hidden truth."

How to implement:
- Rule-based: Regular expressions for patterns (e.g., ALL CAPS TITLE?).
- Lexicon-based: Lists of emotionally charged words, conspiracy-related terms.
- NLP (Google Natural Language API):
  Analyze syntax to find sentence structures common in persuasive/manipulative text.
- Use entity analysis – does the article mention credible sources or just vague "experts"?

MongoDB Integration: Store identified tactics as an array in news article documents: {"detected_tactics": ["sensational_title", "emotional_language"]}

* Google Fact Check Tools API:
Allows searching of a global repository of fact-checks from reputable publishers.

User Input: When a user submits a news article's title, a key claim from its text, or its URL, the application takes this input and queries the Google Fact Check Tools API.


* User Explanation Layer ( Generative AI - PaLM/Gemini):
After all the analysis (your classifier, sentiment, tactic ID, fact-check API, vector search), a generative model is used to synthesize the findings into a human-readable summary for the user.

* Model Selected: DeepSeek-R1-Distill-Llama-8B-GGUF
Distilled and quantized version of Deepseek-R1 Llama 3 model with 8B params, to suit limitations of project 

Sample Prompt for:
"Analyze the following news assessment and provide a concise summary for the user, highlighting key reasons for concern or confidence.

News Title: [Title of user's article]
Our Classifier Prediction: [Fake/Real (with confidence score)]
Sentiment Score: [Score] (e.g., Highly Negative)
Detected Disinformation Tactics: [List of tactics]
Similar Articles Found (from our database):
  - [Title 1], Label: [Fake/Real], Sentiment: [Score]
  - [Title 2], Label: [Fake/Real], Sentiment: [Score]
Google Fact Check API Results:
  - Claim: [Claim 1], Rating: [False], Publisher: [Snopes]
  - Claim: [Claim 2], Rating: [Misleading], Publisher: [PolitiFact]

Based on this, explain to the user why they should be cautious or confident about this news, and what specific things to look out for."
