"""Base model: greedy text completion."""
import torch

from _load import load, parse_args

args = parse_args(__doc__)
tok, model = load("base", args.local_code)
ids = tok("The three states of matter are", return_tensors="pt").input_ids
with torch.no_grad():
    out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=40, do_sample=False,
                         pad_token_id=tok.eos_token_id)
print(tok.decode(out[0]))
