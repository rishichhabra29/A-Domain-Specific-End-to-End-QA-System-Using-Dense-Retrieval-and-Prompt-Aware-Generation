import torch
import pandas as pd
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# ----- Mean Pooling for RoBERTa -----
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # instead of model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1)

# ----- Load Encoder and Tokenizer -----
model_path = "custom_dpr/encoder"
tokenizer_path = "custom_dpr/tokenizer"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModel.from_pretrained(model_path).to(device).eval()

# ----- Load All Contexts -----
df = pd.read_csv("subjectqa_full.csv")
all_contexts = df["context"].tolist()

# ----- Encode All Contexts Once -----
def encode_passages(passages, batch_size=64):
    all_embeddings = []
    for i in range(0, len(passages), batch_size):
        batch = passages[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        pooled = mean_pooling(outputs, inputs["attention_mask"])
        all_embeddings.append(F.normalize(pooled, p=2, dim=1))
    return torch.cat(all_embeddings, dim=0)

print("📦 Encoding all contexts...")
passage_embeddings = encode_passages(all_contexts)
print("✅ Done encoding.")

# ----- Retrieval Function -----
def retrieve_top_k(question, top_k=3):
    with torch.no_grad():
        inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        outputs = model(**inputs)
        q_emb = mean_pooling(outputs, inputs["attention_mask"])
        q_emb = F.normalize(q_emb, p=2, dim=1)

        scores = torch.matmul(q_emb, passage_embeddings.T).squeeze(0)
        top_indices = torch.topk(scores, k=top_k).indices.tolist()
        return [(scores[i].item(), all_contexts[i]) for i in top_indices]

# ----- Sample Questions -----
questions = [
    "What is the battery life of this product?",
    "Does the screen have good brightness?",
    "Is the keyboard comfortable to use?",
    "How fast does it charge?",
    "Is the audio quality good?"
]

print("\n--- 🔍 Top-k Retrieved Passages ---")
for q in questions:
    print(f"\nQ: {q}")
    for score, passage in retrieve_top_k(q, top_k=3):
        print(f"Score: {score:.4f} | Passage: {passage[:300]}...")
