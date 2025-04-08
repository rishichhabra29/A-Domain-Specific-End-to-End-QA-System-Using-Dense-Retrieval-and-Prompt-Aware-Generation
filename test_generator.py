import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import logging

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "custom_generator"

# Load model and tokenizer
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
model.eval()

# Utility: Generate answer
def generate_answer(question, context, max_length=160):
    input_text = f"answer question: {question} context: {context}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=1.0,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test set: Mix of relevant and irrelevant context
qa_pairs = [
    {
        "question": "What is the battery life of this product?",
        "context": "This battery is excellent. It lasts about 12 hours with heavy use. I’ve gone entire trips without needing to recharge."
    },
    {
        "question": "Does the screen have good brightness?",
        "context": "The screen is very bright and performs well even in outdoor lighting conditions. Contrast levels are sharp."
    },
    {
        "question": "Is the keyboard comfortable to use?",
        "context": "The keys on this keyboard are well spaced and offer satisfying tactile feedback, making typing a joy."
    },
    {
        "question": "How fast does it charge?",
        "context": "The device charges from 0 to 80% in under 30 minutes using the fast charger. Very convenient."
    },
    {
        "question": "Is the audio quality good?",
        "context": "These headphones provide deep bass and crisp highs. The soundstage is quite immersive for the price."
    },
    # Irrelevant context
    {
        "question": "What is the battery life of this product?",
        "context": "This bag is made of leather and has multiple compartments. It's suitable for office and casual use."
    },
    {
        "question": "Does the screen have good brightness?",
        "context": "This product is a kitchen blender. It comes with two jars and a stainless steel blade."
    }
]

# Run tests
print("\n--- 🔬 Generator Output ---\n")
for pair in qa_pairs:
    question = pair["question"]
    context = pair["context"]
    answer = generate_answer(question, context)
    print(f"Q: {question}")
    print(f"CTX: {context[:150]}...")
    print(f"✅ A: {answer}\n")
