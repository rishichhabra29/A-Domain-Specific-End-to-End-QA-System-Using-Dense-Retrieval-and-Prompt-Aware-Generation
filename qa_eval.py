import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, T5Tokenizer, T5ForConditionalGeneration
import torch.nn.functional as F
from tqdm import tqdm
from evaluate import load

# Load Evaluation Metrics
bertscore = load("bertscore")
chrf = load("chrf")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Mean Pooling ---
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1)

# --- Load Retriever ---
retriever_tokenizer = AutoTokenizer.from_pretrained("custom_dpr/tokenizer")
retriever_model = AutoModel.from_pretrained("custom_dpr/encoder").to(device).eval()

# --- Load Generator ---
generator_tokenizer = T5Tokenizer.from_pretrained("custom_generator")
generator_model = T5ForConditionalGeneration.from_pretrained("custom_generator").to(device).eval()

# --- Load Contexts ---
df_full = pd.read_csv("subjectqa_full.csv")
all_contexts = df_full["context"].tolist()

def encode_passages(passages, batch_size=64):
    all_embeddings = []
    for i in range(0, len(passages), batch_size):
        batch = passages[i:i + batch_size]
        inputs = retriever_tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = retriever_model(**inputs)
            pooled = mean_pooling(outputs, inputs["attention_mask"])
            pooled = F.normalize(pooled, p=2, dim=1)
            all_embeddings.append(pooled)
    return torch.cat(all_embeddings, dim=0)

print("📦 Encoding passages...")
passage_embeddings = encode_passages(all_contexts)
print("✅ Done.")

def retrieve_top_k(question, k=3):
    with torch.no_grad():
        inputs = retriever_tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        outputs = retriever_model(**inputs)
        q_emb = mean_pooling(outputs, inputs["attention_mask"])
        q_emb = F.normalize(q_emb, p=2, dim=1)
        scores = torch.matmul(q_emb, passage_embeddings.T).squeeze(0)
        top_indices = torch.topk(scores, k=k).indices.tolist()
        return [(scores[i].item(), all_contexts[i]) for i in top_indices]

def generate_answer(question, top_k=3):
    top_passages = retrieve_top_k(question, k=top_k)
    answers = []
    for _, ctx in top_passages:
        input_text = f"question: {question.strip()} context: {ctx.strip()}"
        inputs = generator_tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = generator_model.generate(
                **inputs,
                max_length=128,
                min_length=20,
                num_beams=4,
                length_penalty=1.2,
                repetition_penalty=1.2,
                early_stopping=True
            )
        decoded = generator_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        if decoded:
            answers.append(decoded)
    return answers[0] if answers else "[No relevant answer found]"

# --- Load Test Data ---
test_df = pd.read_csv("subjectqa_test.csv")
questions = test_df["question"].astype(str).tolist()
true_answers = test_df["answers.text"].apply(lambda x: x[0] if isinstance(x, list) else x).astype(str).tolist()

# --- Generate Predictions ---
predicted_answers = []
for question in tqdm(questions, desc="🔮 Generating predictions"):
    pred = generate_answer(question)
    predicted_answers.append(pred)

# --- Evaluate ---
print("\n🔍 Evaluating predictions...")
bertscore_result = bertscore.compute(predictions=predicted_answers, references=true_answers, lang="en")
chrf_result = chrf.compute(predictions=predicted_answers, references=true_answers)

print("\n📊 Final Evaluation:")
print(f"BERTScore (F1): {np.mean(bertscore_result['f1']) * 100:.2f}")
print(f"CHRF++: {chrf_result['score']:.2f}")
