import torch
from transformers import AutoTokenizer, AutoModel, T5Tokenizer, T5ForConditionalGeneration
import torch.nn.functional as F
import pandas as pd

# --- Mean Pooling for Encoder ---
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1)

# --- Load Retriever ---
retriever_model_path = "custom_dpr/encoder"
retriever_tokenizer_path = "custom_dpr/tokenizer"
retriever_tokenizer = AutoTokenizer.from_pretrained(retriever_tokenizer_path)
retriever_model = AutoModel.from_pretrained(retriever_model_path)
retriever_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
retriever_model.to(device)

# --- Load Generator ---
generator_model_path = "custom_generator"
generator_tokenizer = T5Tokenizer.from_pretrained(generator_model_path)
generator_model = T5ForConditionalGeneration.from_pretrained(generator_model_path)
generator_model.eval()
generator_model.to(device)

# --- Load and Encode Contexts ---
print("📦 Loading and encoding passages...")
df = pd.read_csv("subjectqa_full.csv")
all_contexts = df["context"].tolist()

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

passage_embeddings = encode_passages(all_contexts)
print("✅ Passage encoding complete.")

# --- Retrieve Top-k Passages ---
def retrieve_top_k(question, k=5):
    with torch.no_grad():
        inputs = retriever_tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        outputs = retriever_model(**inputs)
        q_emb = mean_pooling(outputs, inputs["attention_mask"])
        q_emb = F.normalize(q_emb, p=2, dim=1)
        scores = torch.matmul(q_emb, passage_embeddings.T).squeeze(0)
        top_indices = torch.topk(scores, k=k).indices.tolist()
        return [(scores[i].item(), all_contexts[i]) for i in top_indices]

# --- Generate Answer from Top-k Contexts ---
def generate_answer(question, top_k=5):
    top_passages = retrieve_top_k(question, k=top_k)
    
    # Create a list of separate inputs
    individual_inputs = [f"question: {question.strip()} context: {ctx.strip()}" for _, ctx in top_passages]
    
    answers = []
    for input_text in individual_inputs:
        inputs = generator_tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = generator_model.generate(
    **inputs,
    max_length=128,           # increase if your answers are longer
    min_length=20,            # force longer answers
    num_beams=4,
    length_penalty=1.2,       # encourage longer answers
    early_stopping=True,      # safe to keep
    repetition_penalty=1.2
)

        decoded = generator_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        if decoded:
            answers.append(decoded)
    
    # Prioritize non-empty answers
    final_answer = answers[0] if answers else "[No relevant answer found]"
    return final_answer, top_passages


# --- Interactive Test ---
if __name__ == "__main__":
    sample_questions = [
    "How good is the camera quality on this device?",
    "Does the laptop overheat during gaming?",
    "How comfortable are the headphones for long use?",
    "Does the laptop stay cool during heavy usage?",
    "Is the build quality sturdy enough for travel?",
    "How fast does the laptop battery charge?",
    "Is the plastic frame prone to breaking?",
    "Is the screen readable in sunlight?"
]



    for q in sample_questions:
        print(f"\n🔍 Question: {q}")
        answer, top_passages = generate_answer(q)
        print(f"✅ Answer: {answer}")
        for i, (score, ctx) in enumerate(top_passages):
            print(f"\nTop {i+1} (Score: {score:.4f}):\n{ctx[:300]}...")




