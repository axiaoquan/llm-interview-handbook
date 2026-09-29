"""Regression tests execute the definitions in the handbook, not copied implementations."""
import ast
import math
from pathlib import Path
import re
import tempfile
import unittest

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def load_definitions(filename):
    """Load imports and top-level definitions; do not execute illustrative call sites."""
    path = ROOT / "llm-coding" / filename
    scope = {"__name__": "handbook_example"}
    for match in re.finditer(r"^```python\s*\n(.*?)^```\s*$", path.read_text(), re.M | re.S):
        tree = ast.parse(match.group(1), filename=str(path))
        nodes = [n for n in tree.body if isinstance(
            n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope


class HandbookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.peft = load_definitions("06-peft.md")
        cls.norm = load_definitions("03-normalization.md")
        cls.attn = load_definitions("01-attention.md")
        cls.loss = load_definitions("07-loss-rl.md")
        cls.bpe = load_definitions("04-tokenizer.md")
        cls.decode = load_definitions("05-decoding.md")
        cls.misc = load_definitions("08-misc.md")

    def setUp(self):
        torch.manual_seed(42)

    def close(self, actual, expected, **kwargs):
        torch.testing.assert_close(actual, expected, **kwargs)

    def test_lora_four_initializations(self):
        for zero_a, zero_b in [(False, True), (True, False), (True, True), (False, False)]:
            with self.subTest(zero_a=zero_a, zero_b=zero_b):
                layer = self.peft["LoRALinear"](3, 4, r=2).double()
                with torch.no_grad():
                    layer.lora_A.normal_()
                    layer.lora_B.normal_()
                    if zero_a:
                        layer.lora_A.zero_()
                    if zero_b:
                        layer.lora_B.zero_()
                x = torch.randn(5, 3, dtype=torch.double, requires_grad=True)
                y = layer(x)
                if zero_a or zero_b:
                    self.close(y, layer.base(x))
                else:
                    self.assertGreater((y - layer.base(x)).norm().item(), 0)
                y.square().mean().backward()
                self.assertEqual(layer.lora_A.grad.norm().item() == 0, zero_b)
                self.assertEqual(layer.lora_B.grad.norm().item() == 0, zero_a)
                self.assertIsNone(layer.base.weight.grad)
                self.assertGreater(x.grad.norm().item(), 0)  # frozen base must not detach input
                with torch.no_grad():
                    layer.lora_A.add_(layer.lora_A.grad, alpha=-0.01)
                    layer.lora_B.add_(layer.lora_B.grad, alpha=-0.01)
                self.assertEqual((layer.lora_B @ layer.lora_A).norm().item() == 0, zero_a and zero_b)
                if zero_a != zero_b:
                    layer.zero_grad()
                    layer(x).square().mean().backward()
                    self.assertGreater(layer.lora_A.grad.norm().item(), 0)
                    self.assertGreater(layer.lora_B.grad.norm().item(), 0)

    def test_lora_effective_first_step(self):
        G = torch.randn(4, 3, dtype=torch.double)
        for reverse in [False, True]:
            A = torch.randn(2, 3, dtype=torch.double)
            B = torch.randn(4, 2, dtype=torch.double)
            if reverse:
                A.zero_()
            else:
                B.zero_()
            A.requires_grad_()
            B.requires_grad_()
            scale, lr = 0.5, 0.1
            loss = (scale * B @ A * G).sum()
            da, db = torch.autograd.grad(loss, (A, B))
            self.close(da, scale * B.T @ G)
            self.close(db, scale * G @ A.T)
            actual = scale * (B - lr * db) @ (A - lr * da)
            expected = -lr * scale**2 * (B @ B.T @ G if reverse else G @ A.T @ A)
            self.close(actual, expected)

    def test_lora_injection_freezes_whole_base_and_merge(self):
        model = nn.Sequential(nn.Linear(3, 4), nn.LayerNorm(4), nn.Linear(4, 2)).double().eval()
        x = torch.randn(5, 3, dtype=torch.double)
        before = model(x).detach()
        self.peft["inject_lora"](model, target_modules=("0",), r=2)
        self.close(model(x), before)
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        self.assertEqual(trainable, ["0.lora_A", "0.lora_B"])
        self.assertEqual(model[0].lora_A.dtype, torch.double)
        self.assertFalse(model[0].training)
        with self.assertRaises(ValueError):
            self.peft["inject_lora"](model, target_modules=("0",))
        with torch.no_grad():
            model[0].lora_B.normal_()
        output = model(x).detach()
        self.peft["merge_lora"](model)
        self.assertIsInstance(model[0], nn.Linear)
        self.close(model(x), output)

    def test_lora_merge_rejects_training(self):
        model = nn.Sequential(self.peft["LoRALinear"](3, 4, dropout=0.5))
        with self.assertRaises(ValueError):
            self.peft["merge_lora"](model)

    def test_lora_bfloat16_device_dtype(self):
        base = nn.Linear(3, 4).to(dtype=torch.bfloat16)
        layer = self.peft["LoRALinear"](3, 4, base_linear=base)
        x = torch.randn(2, 3, dtype=torch.bfloat16)
        self.assertEqual(layer.lora_A.device, base.weight.device)
        self.assertEqual(layer.lora_A.dtype, base.weight.dtype)
        self.close(layer(x), base(x))

    def compare_norm(self, custom, reference, x):
        with torch.no_grad():
            custom.gamma.normal_()
            reference.weight.copy_(custom.gamma)
            if hasattr(custom, "beta"):
                custom.beta.normal_()
                reference.bias.copy_(custom.beta)
        x = x.requires_grad_()
        xr = x.detach().clone().requires_grad_()
        y, yr = custom(x), reference(xr)
        self.close(y, yr)
        upstream = torch.randn_like(y)
        y.backward(upstream)
        yr.backward(upstream)
        self.close(x.grad, xr.grad)
        self.close(custom.gamma.grad, reference.weight.grad)
        if hasattr(custom, "beta"):
            self.close(custom.beta.grad, reference.bias.grad)

    def test_layernorm_forward_and_gradients(self):
        self.compare_norm(self.norm["LayerNorm"](4).double(), nn.LayerNorm(4).double(),
                          torch.randn(2, 3, 4, dtype=torch.double))

    def test_rmsnorm_forward_and_gradients(self):
        self.compare_norm(self.norm["RMSNorm"](4).double(), nn.RMSNorm(4, eps=1e-6).double(),
                          torch.randn(2, 3, 4, dtype=torch.double))

    def test_norm_constant_singleton_and_low_precision(self):
        for dim in [1, 4]:
            for dtype in [torch.float32, torch.bfloat16]:
                x = torch.full((2, dim), 1000.0, dtype=dtype)
                ln = self.norm["LayerNorm"](dim).to(dtype)
                rms = self.norm["RMSNorm"](dim).to(dtype)
                self.close(ln(x), torch.zeros_like(x))
                self.assertTrue(torch.isfinite(rms(x)).all())
                self.close(rms(x).float(), torch.ones_like(x).float(), atol=0.01, rtol=0.01)

    def test_batchnorm_forward_backward_and_running_stats(self):
        custom = self.norm["BatchNorm1d"](3).double()
        reference = nn.BatchNorm1d(3).double()
        self.compare_norm(custom, reference, torch.randn(5, 3, dtype=torch.double))
        self.close(custom.running_mean, reference.running_mean)
        self.close(custom.running_var, reference.running_var)
        custom.eval()
        reference.eval()
        x = torch.randn(2, 3, dtype=torch.double)
        self.close(custom(x), reference(x))
        with self.assertRaises(ValueError):
            custom.train()(x[:1])

    def test_attention_masks_forward_backward(self):
        fn = self.attn["scaled_dot_product_attention"]
        q, k, v = [torch.randn(2, 2, 4, 3, dtype=torch.double, requires_grad=True) for _ in range(3)]
        mask = torch.ones(4, 4, dtype=torch.bool).tril()
        additive = torch.zeros(4, 4, dtype=torch.double).masked_fill(~mask, -torch.inf)
        a, _ = fn(q, k, v, mask)
        b, _ = fn(q, k, v, additive)
        ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        self.close(a, b)
        self.close(a, ref)
        upstream = torch.randn_like(a)
        ga = torch.autograd.grad(a, (q, k, v), upstream, retain_graph=True)
        gr = torch.autograd.grad(ref, (q, k, v), upstream)
        for actual, expected in zip(ga, gr):
            self.close(actual, expected)

    def test_attention_fully_masked_and_eval_dropout(self):
        fn = self.attn["scaled_dot_product_attention"]
        x = torch.randn(1, 1, 3, 4, requires_grad=True)
        y, weights = fn(x, x, x, torch.zeros(3, 3, dtype=torch.bool))
        self.close(y, torch.zeros_like(y))
        self.close(weights, torch.zeros_like(weights))
        y.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.close(fn(x, x, x, dropout_p=0.9)[0], fn(x, x, x)[0])

    def test_cache_full_single_and_chunk_equivalence(self):
        model = self.attn["CachedAttention"](8, 2).double().eval()
        x = torch.randn(2, 6, 8, dtype=torch.double)
        full, _ = model(x)
        for chunks in [[1, 1, 1, 1, 1, 1], [2, 3, 1], [6]]:
            cache, outputs, start = {}, [], 0
            for size in chunks:
                out, cache = model(x[:, start:start + size], cache)
                outputs.append(out)
                start += size
            self.close(torch.cat(outputs, dim=1), full)
        changed = x.clone()
        changed[:, 4:] += 100
        self.close(model(changed)[0][:, :4], full[:, :4])

    def test_cache_padding(self):
        model = self.attn["CachedAttention"](8, 2).double().eval()
        x = torch.randn(2, 5, 8, dtype=torch.double)
        valid = torch.tensor([[False, True, True, True, True], [True] * 5])
        full, _ = model(x, key_padding_mask=valid)
        cache = {}
        a, cache = model(x[:, :3], cache, valid[:, :3])
        b, cache = model(x[:, 3:], cache, valid)
        self.close(torch.cat([a, b], dim=1), full)
        self.close(full[0, 0], torch.zeros(8, dtype=torch.double))

    def test_ce_matches_framework_loss_and_gradient(self):
        z = (torch.randn(6, 5, dtype=torch.double) * 1000).requires_grad_()
        target = torch.tensor([0, 1, -100, 4, 3, 2])
        custom = self.loss["cross_entropy_from_logits"](z, target)
        reference = F.cross_entropy(z, target)
        self.close(custom, reference)
        a, = torch.autograd.grad(custom, z, retain_graph=True)
        b, = torch.autograd.grad(reference, z)
        self.close(a, b)

    def test_lm_shift_and_empty_labels(self):
        z = torch.randn(2, 4, 5, dtype=torch.double, requires_grad=True)
        labels = torch.tensor([[-100, -100, 1, 2], [-100, 3, 2, -100]])
        actual = self.loss["compute_lm_loss"](z, labels)
        expected = F.cross_entropy(z[:, :-1].reshape(-1, 5), labels[:, 1:].reshape(-1))
        self.close(actual, expected)
        for logits, target in [(z, torch.full_like(labels, -100)), (z[:, :1], labels[:, :1])]:
            loss = self.loss["compute_lm_loss"](logits, target)
            self.assertEqual(loss.item(), 0)
            grad, = torch.autograd.grad(loss, z, retain_graph=True)
            self.close(grad, torch.zeros_like(z))

    def test_smoothing_matches_framework(self):
        z = torch.randn(5, 3, dtype=torch.double, requires_grad=True)
        target = torch.tensor([0, -100, 2, 1, 0])
        for smoothing in [0.0, 0.1, 1.0]:
            custom = self.loss["LabelSmoothingLoss"](3, smoothing)(z, target)
            reference = F.cross_entropy(z, target, label_smoothing=smoothing)
            self.close(custom, reference)
            a, = torch.autograd.grad(custom, z, retain_graph=True)
            b, = torch.autograd.grad(reference, z, retain_graph=True)
            self.close(a, b)
        empty = self.loss["LabelSmoothingLoss"](3)(z, torch.full_like(target, -100))
        self.assertEqual(empty.item(), 0)

    def test_grpo_reduction_detach_and_lengths(self):
        fn = self.loss["grpo_loss"]
        new = torch.full((2, 3), -1.0, dtype=torch.double, requires_grad=True)
        old = new.detach().clone().requires_grad_()
        ref = old.detach().clone().requires_grad_()
        rewards = torch.tensor([0.0, 2.0], dtype=torch.double, requires_grad=True)
        mask = torch.tensor([[1, 0, 0], [1, 1, 1]])
        seq = fn(new, old, rewards, mask, 2, ref_log_probs=ref)
        tok = fn(new, old, rewards, mask, 2, reduction="token_mean")
        self.assertAlmostEqual(seq.item(), 0)
        self.assertAlmostEqual(tok.item(), -0.5)
        seq.backward()
        self.close(new.grad, torch.tensor([[0.5, 0, 0], [-1/6, -1/6, -1/6]], dtype=torch.double))
        self.assertIsNone(old.grad)
        self.assertIsNone(ref.grad)
        self.assertIsNone(rewards.grad)
        equal_mask = torch.ones_like(mask)
        self.close(fn(new, old, rewards, equal_mask, 2),
                   fn(new, old, rewards, equal_mask, 2, reduction="token_mean"))
        with self.assertRaises(ValueError):
            fn(new, old, rewards, torch.zeros_like(mask), 2)

    def test_grpo_same_rewards_and_masked_nan(self):
        fn = self.loss["grpo_loss"]
        new = torch.tensor([[-1., float('nan')], [-1., -2.]], requires_grad=True)
        old = torch.tensor([[-1., float('nan')], [-1., -2.]])
        loss = fn(new, old, torch.ones(2), torch.tensor([[1, 0], [1, 1]]), 2)
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.close(new.grad, torch.zeros_like(new))

    def test_bpe_pair_boundaries_overlap_and_frequency(self):
        merge = self.bpe["merge_pair"]
        self.assertEqual(merge(('a', 'b'), {'aa b </w>': 1}), {'aa b </w>': 1})
        self.assertEqual(merge(('a', 'a'), {'a a a </w>': 2}), {'aa a </w>': 2})
        self.assertEqual(merge(('a', 'b'), {'a b </w>': 2, 'ab </w>': 3}), {'ab </w>': 5})

    def test_kl_detach_padding_and_overflow(self):
        fn = self.loss['kl_penalty']
        new = torch.tensor([-1., float('nan')], requires_grad=True)
        ref = torch.tensor([-0.5, float('nan')], requires_grad=True)
        value = fn(new, ref, torch.tensor([1, 0]))
        self.assertAlmostEqual(value.item(), math.expm1(0.5) - 0.5, places=6)
        value.backward()
        self.assertIsNone(ref.grad)
        self.assertEqual(new.grad[1].item(), 0)
        with self.assertRaises(FloatingPointError):
            fn(torch.tensor([-1000.]), torch.tensor([0.]), torch.tensor([1]))

    def test_bpe_base_vocabulary_and_rank(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / 'corpus.txt'
            corpus.write_text('ab ab abc', encoding='utf-8')
            merges, vocab = self.bpe['build_vocab'](str(corpus), 10)
        for word in ['a', 'ab', 'abc']:
            tokens = self.bpe['encode_word'](word, merges)
            self.assertTrue(all(t in vocab for t in tokens))
            ranks = {pair: i for i, pair in enumerate(merges)}
            self.assertEqual(tokens, self.bpe['encode_word_by_rank'](word, ranks))
        self.assertEqual(len(set(vocab.values())), len(vocab))

    def test_batched_eos_and_zero_length_generation(self):
        class Stub(nn.Module):
            def forward(self, ids, kv_cache=None):
                if kv_cache is None:
                    step = ids.size(1) - 1
                else:
                    step = kv_cache.get('step', 0)
                    kv_cache['step'] = step + 1
                logits = torch.zeros(2, ids.size(1), 4)
                # Row 0 finishes immediately, then would resume without a persistent mask.
                for row, token in enumerate([2 if step == 0 else 1, 2 if step == 2 else 1]):
                    logits[row, -1, token] = 10
                return logits if kv_cache is None else (logits, kv_cache)
        model = Stub().eval()
        prompt = torch.zeros(2, 1, dtype=torch.long)
        expected = torch.tensor([[0, 2, 3, 3], [0, 1, 1, 2]])
        for fn in [self.decode['greedy_decode'], self.attn['generate']]:
            self.close(fn(model, prompt, 5, eos_id=2, pad_id=3), expected)
            self.close(fn(model, prompt, 0, eos_id=2), prompt)

    def test_moe_normalizations_and_gradients(self):
        z = torch.randn(5, 6, dtype=torch.double, requires_grad=True)
        values, idx = z.topk(2, dim=-1)
        a = values.softmax(-1)
        b = z.softmax(-1).gather(-1, idx)
        b = b / b.sum(-1, keepdim=True)
        self.close(a, b)
        weights = torch.randn_like(a)
        ga, = torch.autograd.grad((a * weights).sum(), z, retain_graph=True)
        gb, = torch.autograd.grad((b * weights).sum(), z)
        self.close(ga, gb)

    def test_moe_dispatch_and_top1_router(self):
        for k in [1, 2]:
            model = self.misc['MoELayer'](4, 6, 3, k).double()
            x = torch.randn(2, 3, 4, dtype=torch.double)
            flat = x.reshape(-1, 4)
            values, idx = model.gate(flat).topk(k, dim=-1)
            gate = values.softmax(-1)
            expected = torch.stack([
                sum(model.experts[idx[i, j].item()](flat[i]) * gate[i, j] for j in range(k))
                for i in range(flat.size(0))]).reshape_as(x)
            actual = model(x)
            self.close(actual, expected)
            actual.square().sum().backward()
            if k == 1:
                self.close(model.gate.weight.grad, torch.zeros_like(model.gate.weight))
            else:
                self.assertGreater(model.gate.weight.grad.norm().item(), 0)

    def test_tied_embedding_component(self):
        model = self.misc['TiedLM'](7, 4, nn.Identity())
        ids = torch.tensor([[0, 1]])
        self.close(model(ids), F.linear(model.embed(ids), model.embed.weight))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 28)


if __name__ == '__main__':
    unittest.main()
