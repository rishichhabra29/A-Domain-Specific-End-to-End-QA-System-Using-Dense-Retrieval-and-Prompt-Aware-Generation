import streamlit as st
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm

# --- Setup Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Mean Pooling Function ---
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1)

# --- Load Retriever Model and Tokenizer ---
retriever_tokenizer = AutoTokenizer.from_pretrained("custom_dpr/tokenizer")
retriever_model = AutoModel.from_pretrained("custom_dpr/encoder").to(device).eval()

# --- Load Generator Model and Tokenizer ---
generator_tokenizer = T5Tokenizer.from_pretrained("custom_generator")
generator_model = T5ForConditionalGeneration.from_pretrained("custom_generator").to(device).eval()

# --- Cache and Load All Contexts from Dataset ---
@st.cache_data(show_spinner=False)
def load_contexts():
    df = pd.read_csv("subjectqa_full.csv")
    return df["context"].tolist()

all_contexts = load_contexts()

# --- Cache Passage Encoding ---
@st.cache_data(show_spinner=False)
def encode_passages(passages, batch_size=64):
    all_embeddings = []
    for i in range(0, len(passages), batch_size):
        batch = passages[i:i + batch_size]
        inputs = retriever_tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(device)
        with torch.no_grad():
            outputs = retriever_model(**inputs)
            pooled = mean_pooling(outputs, inputs["attention_mask"])
            pooled = F.normalize(pooled, p=2, dim=1)
            all_embeddings.append(pooled)
    return torch.cat(all_embeddings, dim=0)

st.info("Encoding passages... This is cached and will be fast on subsequent runs.")
passage_embeddings = encode_passages(all_contexts)
st.success("Passages encoded.")

# --- Retrieve Top-k Passages ---
def retrieve_top_k(question, k=5):
    with torch.no_grad():
        inputs = retriever_tokenizer(
            question, return_tensors="pt", padding=True, truncation=True, max_length=128
        ).to(device)
        outputs = retriever_model(**inputs)
        q_emb = mean_pooling(outputs, inputs["attention_mask"])
        q_emb = F.normalize(q_emb, p=2, dim=1)
        scores = torch.matmul(q_emb, passage_embeddings.T).squeeze(0)
        top_indices = torch.topk(scores, k=k).indices.tolist()
        return [(scores[i].item(), all_contexts[i]) for i in top_indices]

# --- Generate Answer from Top-k Retrieved Passages ---
def generate_answer(question, top_k=5):
    # Retrieve top passages from our dense retriever
    top_passages = retrieve_top_k(question, k=top_k)
    
    # For each passage, generate an answer using the generator model
    individual_inputs = [
        f"question: {question.strip()} context: {ctx.strip()}" for _, ctx in top_passages
    ]
    
    answers = []
    for input_text in individual_inputs:
        inputs = generator_tokenizer(
            input_text, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(device)
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
    
    final_answer = answers[0] if answers else "[No relevant answer found]"
    return final_answer, top_passages

# --- Streamlit UI (Interactive) ---
st.title("Electronics Domain QA System")
st.write("Type your question below and click 'Get Answer' to see the generated answer and the top retrieved passages (supporting evidence).")

question_input = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if not question_input.strip():
        st.warning("Please enter a valid question.")
    else:
        with st.spinner("Generating answer..."):
            answer, passages = generate_answer(question_input)
        st.markdown(f"### **Question:** {question_input}")
        st.markdown(f"### **Answer:** {answer}")
        st.write("### **Top Retrieved Passages:**")
        for i, (score, ctx) in enumerate(passages, start=1):
            st.write(f"**Top {i} (Score: {score:.4f}):**")
            st.write(ctx)

st.write("---")
st.write("This QA system is designed for the electronics domain. It uses a dense retrieval model to find relevant passages and a T5-based generator to produce accurate answers by considering multiple contexts. Adjust the generation parameters (e.g., beam search, repetition penalty) for different output styles.")
