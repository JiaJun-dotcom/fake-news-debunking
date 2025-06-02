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
CSV_FILE_PATH_WELFAKE = "./data/WELFake_Dataset.csv"
CSV_FILE_PATH_NEW_FAKE = "./data/Fake.csv"
CSV_FILE_PATH_NEW_REAL = "./data/True.csv"
               
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
OUTPUT_MODEL_DIR = "./fine_tuned" # Directory to save the fine-tuned model

WELFAKE_TEXT_COLUMN = "text"                  
WELFAKE_TITLE_COLUMN = "title"                
WELFAKE_LABEL_COLUMN = "label" 
NEW_CSV_TEXT_COLUMN = "text"
NEW_CSV_TITLE_COLUMN = "title"

# Training Hyperparameters
MAX_LENGTH = 512       # Max sequence length for tokenizer
BATCH_SIZE = 32        
EPOCHS = 3             # Number of training epochs (2-4 is common for fine-tuning)
LEARNING_RATE = 1e-5   # AdamW optimizer learning rate
RANDOM_SEED = 42       # For reproducibility

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. Load and Prepare Data ---
# --- Function to Load and Preprocess Individual DataFrames ---
def load_and_preprocess_df(filepath, text_col, title_col_optional, label_value=None, is_welfake=False, welfake_label_col=None):
    print(f"Loading and preprocessing: {filepath}")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"ERROR: File not found: {filepath}")
        return None
    except Exception as e:
        print(f"ERROR loading {filepath}: {e}")
        return None

    # Drop unnecessary columns (like 'date' and 'subject' for new CSVs)
    cols_to_drop = []
    if not is_welfake: 
        if 'date' in df.columns:
            cols_to_drop.append('date')
        if 'subject' in df.columns:
            cols_to_drop.append('subject')
        # Add any other specific columns to drop from these new files
    
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True, errors='ignore') # errors='ignore' if column might not exist
        print(f"  Dropped columns: {cols_to_drop}")

    # Handle NaNs in text and title
    df[text_col] = df[text_col].fillna("")
    if title_col_optional and title_col_optional in df.columns:
        df[title_col_optional] = df[title_col_optional].fillna("")
        # Combine title and text
        df['input_text'] = df[title_col_optional] + " [SEP] " + df[text_col]
    else:
        df['input_text'] = df[text_col]

    # Assign or ensure label
    if label_value is not None: # For new fakenews.csv (0) and realnews.csv (1)
        df['label'] = label_value
        print(f"  Assigned label: {label_value}")
    elif is_welfake and welfake_label_col in df.columns: 
        df.rename(columns={welfake_label_col: 'label'}, inplace=True)
        df['label'] = df['label'].astype(int)
        print(f"  Using existing label column: '{welfake_label_col}'")
    else:
        print(f"ERROR: Label could not be assigned or found for {filepath}")
        return None
        
    df = df[['input_text', 'label']]
    return df

# --- 1. Load All Three Datasets ---
df_welfake = load_and_preprocess_df(CSV_FILE_PATH_WELFAKE,
                                    text_col=WELFAKE_TEXT_COLUMN,
                                    title_col_optional=WELFAKE_TITLE_COLUMN,
                                    is_welfake=True,
                                    welfake_label_col=WELFAKE_LABEL_COLUMN)

df_new_fake = load_and_preprocess_df(CSV_FILE_PATH_NEW_FAKE,
                                     text_col=NEW_CSV_TEXT_COLUMN,
                                     title_col_optional=NEW_CSV_TITLE_COLUMN,
                                     label_value=0) # Assign 0 for fake news

df_new_real = load_and_preprocess_df(CSV_FILE_PATH_NEW_REAL,
                                     text_col=NEW_CSV_TEXT_COLUMN,
                                     title_col_optional=NEW_CSV_TITLE_COLUMN,
                                     label_value=1) # Assign 1 for real news

# --- Check if all DataFrames loaded successfully ---
loaded_dfs = []
if df_welfake is not None: loaded_dfs.append(df_welfake)
if df_new_fake is not None: loaded_dfs.append(df_new_fake)
if df_new_real is not None: loaded_dfs.append(df_new_real)

if not loaded_dfs:
    print("ERROR: No datasets were loaded successfully. Exiting.")
    exit()

# --- 2. Split New Datasets for Dedicated Testing (Out-of-Distribution) ---
df_ood_test = pd.DataFrame() # Initialize as empty

if df_new_fake is not None and df_new_real is not None:
    new_fake_train_val, new_fake_test = train_test_split(df_new_fake, test_size=0.2, random_state=RANDOM_SEED)
    new_real_train_val, new_real_test = train_test_split(df_new_real, test_size=0.2, random_state=RANDOM_SEED)
    
    df_ood_test = pd.concat([new_fake_test, new_real_test], ignore_index=True)
    df_ood_test = df_ood_test.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"Created Out-of-Distribution Test set with {len(df_ood_test)} samples.")

    # The portions to be added to the main training pool are new_fake_train_val and new_real_train_val
    dfs_for_main_training_pool = []
    if df_welfake is not None: dfs_for_main_training_pool.append(df_welfake)
    dfs_for_main_training_pool.append(new_fake_train_val)
    dfs_for_main_training_pool.append(new_real_train_val)
    df_combined_for_training = pd.concat(dfs_for_main_training_pool, ignore_index=True)

elif df_welfake is not None: # If only WELFake loaded, use it as the main pool
    print("Only WELFake dataset loaded. Using it for training/validation/in-distribution test.")
    df_combined_for_training = df_welfake
else: 
    print("ERROR: Not enough data to proceed with training.")
    exit()


# --- 3. Shuffle the Combined Training Pool ---
df_combined_for_training = df_combined_for_training.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
print(f"\nTotal samples for main training/validation/in-distribution test pool: {len(df_combined_for_training)}")
if not df_combined_for_training.empty:
    print(f"Label distribution in combined pool:\n{df_combined_for_training['label'].value_counts(normalize=True)}")
else:
    print("ERROR: Combined training pool is empty!")
    exit()

# --- 4. Split the Combined Pool for Training, Validation, and In-Distribution Test ---
df_in_distribution_test = pd.DataFrame() # Initialize as empty

if len(df_combined_for_training) > 10: # Ensure enough data for splitting
    train_val_pool, df_in_distribution_test = train_test_split(
        df_combined_for_training,
        test_size=0.1, # e.g., 10% for in-distribution test
        random_state=RANDOM_SEED,
        stratify=df_combined_for_training['label']
    )

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_val_pool['input_text'].tolist(),
        train_val_pool['label'].tolist(),
        test_size=0.15,  # e.g., ~15% of the remaining for validation
        random_state=RANDOM_SEED,
        stratify=train_val_pool['label'].tolist()
    )
else: # Not enough data for all splits, just use what we have for train/val
    print("Warning: Combined dataset too small for train/val/test split. Using for train/val only.")
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df_combined_for_training['input_text'].tolist(),
        df_combined_for_training['label'].tolist(),
        test_size=0.2, 
        random_state=RANDOM_SEED,
        stratify=df_combined_for_training['label'].tolist()
    )


print(f"\nFinal Training samples: {len(train_texts)}")
print(f"Final Validation samples: {len(val_texts)}")
if not df_in_distribution_test.empty:
    print(f"In-distribution Test samples: {len(df_in_distribution_test)}")
if not df_ood_test.empty:
    print(f"Out-of-Distribution Test samples: {len(df_ood_test)}")

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
from transformers import BertTokenizer, BertForSequenceClassification
model = DistilBertForSequenceClassification.from_pretrained("{OUTPUT_MODEL_DIR}")
tokenizer = DistilBertTokenizer.from_pretrained("{OUTPUT_MODEL_DIR}")
""")