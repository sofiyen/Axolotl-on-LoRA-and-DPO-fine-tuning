from datasets import load_dataset
import json

def convert_to_chat(example):
    return {
        "messages": [
            {
                "role": "user",
                "content": f"Review: {example['text']}\nIs this movie review positive or negative? Answer with a single word - positive or negative:"
            },
            {
                "role": "assistant", 
                "content": "positive" if example["label"] == 1 else "negative"
            }
        ]
    }

dataset = load_dataset("imdb")

with open('imdb_chat.jsonl', 'w') as f:
   for example in dataset["train"]:
       f.write(json.dumps(convert_to_chat(example)) + '\n')