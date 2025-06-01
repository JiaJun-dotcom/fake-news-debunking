# Trains a BERT model transformer fine-tuned on fake/real news data from MongoDB to return a label of fake/real news, and also use available text embeddings to do semantic similarity search for more convincing explanation and relation to .

# Sample output:
# "This new article is very similar to 5 known FAKE articles in our database, which also had highly negative sentiment."
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm.auto import tqdm # For progress bars

# --- Configuration ---
CSV_FILE_PATH = "./data/WELFake_Dataset.csv" 
TEXT_COLUMN = "text"                  
TITLE_COLUMN = "title"                
LABEL_COLUMN = "label"                
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_MODEL_DIR = "./fine_tuned_classifier" # Directory to save the fine-tuned model

# Training Hyperparameters
MAX_LENGTH = 512       # Max sequence length for tokenizer
BATCH_SIZE = 16        
EPOCHS = 3             # Number of training epochs (2-4 is common for fine-tuning)
LEARNING_RATE = 2e-5   # AdamW optimizer learning rate
RANDOM_SEED = 42       # For reproducibility

# --- Set Device (GPU if available, else CPU) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. Load and Prepare Data ---
print(f"Loading data from {CSV_FILE_PATH}...")
try:
    df = pd.read_csv(CSV_FILE_PATH)
except FileNotFoundError:
    print(f"ERROR: CSV file not found at {CSV_FILE_PATH}. Please check the path.")
    exit()

# Handle missing values (simple fill with empty string or drop)
df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("")
if TITLE_COLUMN and TITLE_COLUMN in df.columns:
    df[TITLE_COLUMN] = df[TITLE_COLUMN].fillna("")
    df["combined_text"] = df[TITLE_COLUMN] + " [SEP] " + df[TEXT_COLUMN]
    input_text_column = "combined_text"
else:
    input_text_column = TEXT_COLUMN

# Ensure labels are integers (0 and 1)
try:
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)
    # Verify labels are binary (0 or 1)
    if not all(label in [0, 1] for label in df[LABEL_COLUMN].unique()):
        print(f"ERROR: Labels in column '{LABEL_COLUMN}' are not strictly 0 or 1. Found: {df[LABEL_COLUMN].unique()}")
        print("Please ensure your labels are binary (0 for fake, 1 for real).")
        exit()
except ValueError:
    print(f"ERROR: Could not convert labels in column '{LABEL_COLUMN}' to integers.")
    exit()


print(f"Loaded {len(df)} samples.")
print(f"Label distribution:\n{df[LABEL_COLUMN].value_counts(normalize=True)}")

# --- 2. Split Data into Training and Validation Sets ---
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df[input_text_column].tolist(),
    df[LABEL_COLUMN].tolist(),
    test_size=0.20,  
    random_state=RANDOM_SEED,
    stratify=df[LABEL_COLUMN].tolist() # Ensure similar label distribution in splits
)
print(f"Training samples: {len(train_texts)}, Validation samples: {len(val_texts)}")

# --- 3. Tokenization ---
print(f"Loading tokenizer: {MODEL_NAME}...")
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

print("Tokenizing training data...")
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt")
print("Tokenizing validation data...")
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt")

# --- 4. Create PyTorch Datasets ---
class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = NewsDataset(train_encodings, train_labels)
val_dataset = NewsDataset(val_encodings, val_labels)

# --- 5. Create DataLoaders ---
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# --- 6. Load Pre-trained Model ---
print(f"Loading pre-trained model: {MODEL_NAME}...")
model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2) # 2 labels: fake (0), real (1)
model.to(device) # Move model to GPU if available

# --- 7. Optimizer and Scheduler ---
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0, # Default, can adjust
                                            num_training_steps=total_steps)

# --- 8. Training Loop ---
print("\n--- Starting Fine-tuning ---")
for epoch in range(EPOCHS):
    model.train() # Set model to training mode
    total_train_loss = 0
    progress_bar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Training]", leave=False)

    for batch in progress_bar_train:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping
        optimizer.step()
        scheduler.step()
        progress_bar_train.set_postfix({'loss': loss.item()})

    avg_train_loss = total_train_loss / len(train_loader)
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"  Average Training Loss: {avg_train_loss:.4f}")

    # --- 9. Validation ---
    model.eval() # Set model to evaluation mode
    total_eval_loss = 0
    all_preds = []
    all_true_labels = []
    progress_bar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Validation]", leave=False)

    for batch in progress_bar_val:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_eval_loss += loss.item()
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_true_labels.extend(labels.cpu().numpy())
        progress_bar_val.set_postfix({'loss': loss.item()})

    avg_val_loss = total_eval_loss / len(val_loader)
    val_accuracy = accuracy_score(all_true_labels, all_preds)
    print(f"  Average Validation Loss: {avg_val_loss:.4f}")
    print(f"  Validation Accuracy: {val_accuracy:.4f}")
    print("  Validation Classification Report:")
    # Ensure target_names match your label encoding (0: fake, 1: real)
    print(classification_report(all_true_labels, all_preds, target_names=["fake (0)", "real (1)"]))

print("\n--- Fine-tuning Complete ---")

# --- 10. Save the Fine-tuned Model and Tokenizer ---
print(f"\nSaving fine-tuned model and tokenizer to {OUTPUT_MODEL_DIR}...")

model.save_pretrained(OUTPUT_MODEL_DIR)
tokenizer.save_pretrained(OUTPUT_MODEL_DIR)
print("Model and tokenizer saved successfully.")

print(f"""
--- Next Steps ---
You can now load this fine-tuned model for inference using:
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
model = DistilBertForSequenceClassification.from_pretrained("{OUTPUT_MODEL_DIR}")
tokenizer = DistilBertTokenizer.from_pretrained("{OUTPUT_MODEL_DIR}")
""")