"""Offline checks: no model download. Run with `python -m unittest discover tests`."""
import hashlib
import os
import sys
import unittest

import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from hobbylm_hf import HobbyLMConfig, HobbyLMForCausalLM  # noqa: E402


def tiny(**kw):
    cfg = dict(vocab_size=128, hidden_size=64, intermediate_size=128, moe_intermediate_size=16,
               num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
               num_local_experts=8, num_experts_per_tok=2, n_shared_experts=1, first_k_dense_replace=1,
               max_position_embeddings=4096, use_qk_norm=True, gating="sigmoid", norm_topk_prob=False)
    cfg.update(kw)
    torch.manual_seed(0)
    return HobbyLMForCausalLM(HobbyLMConfig(**cfg)).eval()


class ModelCode(unittest.TestCase):
    def test_files_match_released_hashes(self):
        with open(os.path.join(ROOT, "hobbylm_hf", "SHA256SUMS.txt"), encoding="ascii") as f:
            lines = f.read().splitlines()
        for line in filter(None, lines):
            digest, name = line.split()
            with open(os.path.join(ROOT, "hobbylm_hf", name.lstrip("*")), "rb") as f:
                data = f.read()
            self.assertEqual(hashlib.sha256(data).hexdigest(), digest, name)

    def test_released_config_values_load(self):
        cfg = HobbyLMConfig(num_local_experts=64, num_experts_per_tok=8, num_attention_heads=16,
                            num_key_value_heads=8, head_dim=128, hidden_size=1024, num_hidden_layers=20,
                            max_position_embeddings=4096)
        self.assertEqual((cfg.num_local_experts, cfg.num_experts_per_tok, cfg.max_position_embeddings), (64, 8, 4096))

    def test_cached_generation_matches_full_forward(self):
        m = tiny()
        ids = torch.randint(0, 128, (1, 12))
        with torch.no_grad():
            out = m.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=6, do_sample=False, pad_token_id=0)
            seq = ids
            for _ in range(6):
                nxt = m(seq).logits[:, -1].argmax(-1, keepdim=True)
                seq = torch.cat([seq, nxt], 1)
        self.assertTrue(torch.equal(out, seq))

    def test_4096_positions(self):
        m = tiny()
        ids = torch.randint(0, 128, (1, 4096))
        with torch.no_grad():
            logits = m(ids).logits
        self.assertEqual(tuple(logits.shape), (1, 4096, 128))
        self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
