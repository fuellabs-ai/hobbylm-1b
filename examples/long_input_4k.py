"""SFT model with an input close to its 4,096-token context.

This shows that a ~3,900-token prompt runs end to end. It does not measure long-context quality:
long-context retrieval is unverified for this model (see Limitations in the README).
"""
import torch

from _load import load, parse_args

args = parse_args(__doc__)
tok, model = load("sft", args.local_code)
ctx = model.config.max_position_embeddings
assert ctx == 4096, ctx

lines = [f"Shelf {i}: the box on this shelf holds {(i * 37) % 101} marbles." for i in range(1, 400)]
question = "\n\nHow many marbles does the box on shelf 7 hold?"
messages = [{"role": "user", "content": "\n".join(lines) + question}]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# Trim the list until the prompt plus 32 new tokens fits in the context window.
while len(tok(text, verbose=False).input_ids) + 32 > ctx:   # verbose=False: the untrimmed text is longer than the context
    lines = lines[:-5]
    messages = [{"role": "user", "content": "\n".join(lines) + question}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
ids = tok(text, return_tensors="pt")
n = ids["input_ids"].shape[1]
with torch.no_grad():
    out = model.generate(**ids, max_new_tokens=32, do_sample=False, pad_token_id=50256)
print(f"prompt tokens: {n} / context {ctx}; generated tokens: {out.shape[1] - n}")
print("expected answer: 57 (7 * 37 mod 101)")
print("model output:", tok.decode(out[0][n:], skip_special_tokens=True).strip())
print("(This checks that a near-4K input runs. Long-context retrieval is unverified, so a wrong answer is possible.)")
