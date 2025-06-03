import os
import json
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pymongo import MongoClient
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.nn.functional import softmax
from vertexai.language_models import TextEmbeddingModel
import vertexai

load_dotenv()

# --- Configuration ---
FINE_TUNED_MODEL_PATH = "./fine_tuned"

def initialize_classifier_resources():
    global CLASSIFIER_MODEL, CLASSIFIER_TOKENIZER
    # Fine-tuned DistilBERT Classifier
    try:
        if not os.path.exists(FINE_TUNED_MODEL_PATH):
            print(f"ERROR: Fine-tuned model not found at {FINE_TUNED_MODEL_PATH}. Please train and save it first.")
        CLASSIFIER_MODEL = BertForSequenceClassification.from_pretrained(FINE_TUNED_MODEL_PATH)
        CLASSIFIER_TOKENIZER = BertTokenizer.from_pretrained(FINE_TUNED_MODEL_PATH)
        CLASSIFIER_MODEL.eval() # Set to evaluation mode
        print("Fine-tuned BERT classifier loaded successfully.")
    except Exception as e:
        print(f"ERROR: Could not load fine-tuned BERT model: {e}")

def classify_article_text(title, text_content):
    if not CLASSIFIER_MODEL or not CLASSIFIER_TOKENIZER:
        print("  Classifier model not loaded. Skipping classification.")
        return None

    # Combine title and text for classification for more context
    combined_input = (title if title else "") + " [SEP] " + (text_content if text_content else "")
    if not combined_input.strip() or combined_input.strip() == "[SEP]":
        print("  Empty input for classification.")
        return None

    print("  Classifying article with fine-tuned BERT...")
    inputs = CLASSIFIER_TOKENIZER(combined_input, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = CLASSIFIER_MODEL(**inputs)
        logits = outputs.logits
        probabilities = softmax(logits, dim=1)

    predicted_class_id = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][predicted_class_id].item()

    # Assuming 0 = fake, 1 = real (adjust if your fine-tuning labels were different)
    label_map = {0: "fake", 1: "real"}
    predicted_label = label_map.get(predicted_class_id, "unknown")

    print(f"  Predicted: {predicted_label}, Confidence: {confidence:.4f}")
    return {"prediction": predicted_label, "confidence": confidence}

initialize_classifier_resources()
classify_article_text("May Brexit offer would hurt, cost EU citizens - EU parliament", "BRUSSELS (Reuters) - British Prime Minister Theresa May s offer of  settled status  for EU residents is flawed and will leave them with fewer rights after Brexit, the European Parliament s Brexit coordinator said on Tuesday. A family of five could face a bill of 360 pounds to acquire the new status, Guy Verhofstadt told May s Brexit Secretary David Davis in a letter seen by Reuters    a very significant amount for a family on low income . Listing three other concerns for the EU legislature, which must approve any treaty on the March 2019 exit, Verhofstadt told Davis:  Under your proposals, EU citizens will definitely notice a deterioration of their status as a result of Brexit. And the Parliament s aim all along has been that EU citizens, and UK citizens in the EU-27, should notice no difference.  Verhofstadt, a former Belgian prime minister, wrote in response to Davis, who had written to him after Parliament complained last week that there remained  major issues  to be settled on the rights of the 3 million EU citizens in Britain. On Tuesday, he told reporters that Parliament was determined that expatriates should not become  victims of Brexit . May had unveiled more details last week of a system aimed at giving people already in Britain a quick and cheap way of asserting their rights to stay there indefinitely. The issue, along with how much Britain owes and the new EU-UK border across Ireland, is one on which the EU wants an outline agreement before opening talks on the future of trade. Verhofstadt said lawmakers were not dismissing British efforts to streamline applications but saw flaws in the nature of  settled status  itself. As well as the cost, which is similar to that of acquiring a British passport, he cited three others: - Europeans should simply  declare  a whole household resident, without needing an  application  process; the burden of proof should be on the British authorities to deny them rights. - more stringent conditions on criminal records could mean some EU residents, including some now with permanent resident status, being deported for failing to gain  settled status . - EU residents would lose some rights to bring relatives to Britain as the new status would give them the same rights as British people, who now have fewer rights than EU citizens.   ")