import pandas as pd
import ast
import datasets
from transformers import (
    T5ForConditionalGeneration, T5Tokenizer,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    EarlyStoppingCallback, DataCollatorForSeq2Seq
)

def extract_answer(x):
    try:
        ans_list = ast.literal_eval(x)
        return ans_list[0] if isinstance(ans_list, list) and len(ans_list) > 0 else ""
    except Exception:
        return str(x)

def load_split_dataset(path):
    df = pd.read_csv(path)

    if "answers.text" in df.columns:
        df["target_text"] = df["answers.text"].apply(extract_answer)
    elif "answer" in df.columns:
        df["target_text"] = df["answer"].astype(str)
    else:
        raise KeyError("Answer column not found")

    df["question"] = df["question"].astype(str)
    df["context"] = df["context"].astype(str)

    # Split context into separate entries if delimited or list-like
    entries = []
    for _, row in df.iterrows():
        question = row["question"]
        answer = row["target_text"]
        ctx = row["context"]

        contexts = []
        try:
            parsed = ast.literal_eval(ctx)
            if isinstance(parsed, list):
                contexts = parsed
        except:
            if "||" in ctx:
                contexts = ctx.split("||")
            else:
                contexts = [ctx]

        for c in contexts:
            if c.strip():
                entries.append({
                    "input_text": f"question: {question} context: {c.strip()}",
                    "target_text": answer
                })

    return datasets.Dataset.from_pandas(pd.DataFrame(entries))

def preprocess(examples, tokenizer, max_input_len=512, max_target_len=128):
    inputs = tokenizer(
        examples["input_text"], max_length=max_input_len, truncation=True, padding="max_length"
    )
    targets = tokenizer(
        examples["target_text"], max_length=max_target_len, truncation=True, padding="max_length"
    )
    inputs["labels"] = targets["input_ids"]
    return inputs

def train():
    model_name = "google/flan-t5-base"  # 👈 Use this improved model
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.config.repetition_penalty = 1.2  # 👈 Discourage repetition

    train_dataset = load_split_dataset("subjectqa_train.csv")
    val_dataset = load_split_dataset("subjectqa_val.csv")

    train_dataset = train_dataset.map(lambda x: preprocess(x, tokenizer), batched=True)
    val_dataset = val_dataset.map(lambda x: preprocess(x, tokenizer), batched=True)

    train_dataset.set_format("torch")
    val_dataset.set_format("torch")

    args = Seq2SeqTrainingArguments(
        output_dir="custom_generator",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=10,
        learning_rate=3e-5,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    print("🚀 Training generator...")
    trainer.train()
    model.save_pretrained("custom_generator")
    tokenizer.save_pretrained("custom_generator")
    print("✅ Saved to custom_generator")

if __name__ == "__main__":
    train()
