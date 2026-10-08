"""Instruct model: one chat turn with the released chat template."""
import torch

from _load import load, parse_args

args = parse_args(__doc__)
tok, model = load("sft", args.local_code)
messages = [{"role": "user", "content": "What is the capital of France?"}]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
ids = tok(text, return_tensors="pt")
with torch.no_grad():
    out = model.generate(**ids, max_new_tokens=64, do_sample=False, pad_token_id=50256)
print(tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True))
