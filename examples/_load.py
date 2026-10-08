"""Shared loader for the examples: pinned released revisions, float32 only."""
import argparse
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Released files, pinned to immutable revisions.
BASE = dict(repo="harims95/hobbylm-1b-hf", subfolder=None,
            model_rev="a5bb6bcdab3642acbb203c2b6b6c272669da3693",      # weights used for the release evaluation
            tok_rev="dc11aab4e06fd2000313c8821557a86e8368bf58")        # first revision with the tokenizer files
SFT = dict(repo="harims95/hobbylm-1b-checkpoints-hf", subfolder="sft-step3450",
           model_rev="bdc371b4aa48bede3f55ba1036e9f27952f48c17",
           tok_rev="bdc371b4aa48bede3f55ba1036e9f27952f48c17")


def parse_args(description):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--local-code", action="store_true",
                    help="use the model code in hobbylm_hf/ instead of the copy shipped with the weights")
    return ap.parse_args()


def load(which, local_code=False):
    spec = SFT if which == "sft" else BASE
    sub = {"subfolder": spec["subfolder"]} if spec["subfolder"] else {}
    tok = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["tok_rev"], **sub)
    if local_code:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        from hobbylm_hf import HobbyLMForCausalLM
        model = HobbyLMForCausalLM.from_pretrained(spec["repo"], revision=spec["model_rev"],
                                                   torch_dtype=torch.float32, **sub)
    else:
        model = AutoModelForCausalLM.from_pretrained(spec["repo"], revision=spec["model_rev"],
                                                     trust_remote_code=True, torch_dtype=torch.float32, **sub)
    # bf16/fp16 change which experts the router selects; keep float32.
    assert next(model.parameters()).dtype == torch.float32
    return tok, model.eval()
