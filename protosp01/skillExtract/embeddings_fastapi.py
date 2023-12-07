# %%
sentence = "This sound track was beautiful! It paints the senery in your mind so well I would recomend it even to people who hate vid. game music!"

from transformers import AutoTokenizer, AutoModel
import torch
from fastapi import FastAPI

model = "agne/jobBERT-de"

tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModel.from_pretrained(model)


def tokenize_text(text: str) -> dict:
    tokens = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )

    return tokens


app = FastAPI()


@app.post("/v1/embeddings")
def get_embeddings(sentence: str) -> dict:
    tokens = tokenize_text(sentence)

    with torch.no_grad():
        word_outputs = model(**tokens)
        embeddings = word_outputs.last_hidden_state

    emb_dict = {"embeddings": [row.tolist() for row in embeddings.squeeze()]}

    return emb_dict


@app.post("/v1/tokens")
def get_tokens(text: str) -> dict:
    token_ids = tokenize_text(text)
    tokens = tokenizer.tokenize(text)

    return {"token_ids": token_ids["input_ids"].squeeze().tolist(), "tokens": tokens}


# %%
