import os
import torch
import pandas as pd
import warnings
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from torch.cuda.amp import GradScaler, autocast

warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ----- Mean Pooling -----
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1)

# ----- Dataset with Dynamic Hard Negatives -----
class DPRDataset(Dataset):
    def __init__(self, csv_file, tokenizer, encoder, device, max_length=128, num_candidates=30, top_k=3):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.encoder = encoder.eval().to(device)
        self.device = device
        self.max_length = max_length
        self.num_candidates = num_candidates
        self.top_k = top_k

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        question = row["question"]
        positive = row["context"]
        negatives = self.get_hard_negatives(question, positive)
        return question, positive, negatives

    def get_hard_negatives(self, question, positive):
        candidates = self.data.sample(self.num_candidates)
        candidate_contexts = candidates["context"].tolist()

        with torch.no_grad():
            q_inputs = self.tokenizer(question, return_tensors="pt", truncation=True, padding=True, max_length=self.max_length).to(self.device)
            q_output = self.encoder(**q_inputs)
            q_emb = mean_pooling(q_output, q_inputs['attention_mask'])

            p_inputs = self.tokenizer(candidate_contexts, return_tensors="pt", truncation=True, padding=True, max_length=self.max_length).to(self.device)
            p_output = self.encoder(**p_inputs)
            p_embs = mean_pooling(p_output, p_inputs['attention_mask'])

            sims = torch.nn.functional.cosine_similarity(q_emb, p_embs)

        top_indices = torch.topk(sims, self.top_k, largest=True).indices.flatten()
        return candidates.iloc[top_indices.tolist()]["context"].tolist()

# ----- DPR Model with Mean Pooling -----
class DPRModel(nn.Module):
    def __init__(self, encoder_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)

    def encode(self, texts, tokenizer, max_length=128, device="cpu"):
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        output = self.encoder(**inputs)
        return mean_pooling(output, inputs["attention_mask"])

# ----- Contrastive Loss with Temperature -----
def contrastive_loss(q_emb, pos_emb, neg_embs, temperature=0.05):
    q_emb = nn.functional.normalize(q_emb, dim=1)
    pos_emb = nn.functional.normalize(pos_emb, dim=1)
    neg_embs = nn.functional.normalize(neg_embs, dim=1)

    pos_sim = torch.sum(q_emb * pos_emb, dim=1, keepdim=True) / temperature
    neg_sim = torch.matmul(q_emb, neg_embs.T) / temperature
    logits = torch.cat([pos_sim, neg_sim], dim=1)
    labels = torch.zeros(q_emb.size(0), dtype=torch.long).to(q_emb.device)
    return nn.CrossEntropyLoss()(logits, labels)

# ----- Training Loop -----
def train_dpr():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_name = "roberta-base"  # Can switch to "deepset/roberta-base-squad2" or any encoder
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    encoder = AutoModel.from_pretrained(encoder_name).to(device)

    dataset = DPRDataset("subjectqa_train.csv", tokenizer, encoder, device)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

    model = DPRModel(encoder_name).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scaler = GradScaler()
    num_epochs = 10
    total_steps = len(dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=200, num_training_steps=total_steps)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for step, (questions, positives, negatives_list) in enumerate(dataloader):
            optimizer.zero_grad()
            with autocast():
                q_emb = model.encode(questions, tokenizer, device=device)
                p_emb = model.encode(positives, tokenizer, device=device)
                neg_texts = [neg for negs in negatives_list for neg in negs]
                neg_embs = model.encode(neg_texts, tokenizer, device=device)

                loss = contrastive_loss(q_emb, p_emb, neg_embs)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} / {num_epochs} - Loss: {total_loss / len(dataloader):.4f}")

    model.encoder.save_pretrained("custom_dpr/encoder")
    tokenizer.save_pretrained("custom_dpr/tokenizer")
    print("✅ Model saved to custom_dpr/")

if __name__ == "__main__":
    train_dpr()
