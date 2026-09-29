"""Independent arithmetic checks for documented examples, not model benchmarks."""
import math
import unittest


class KnowledgeExampleTests(unittest.TestCase):
    def test_adam_bias_correction(self):
        self.assertAlmostEqual(((1 - 0.9) * 2) / (1 - 0.9), 2)

    def test_token_weighting_and_ddp_compensation(self):
        sums, counts = [2 * 3, 8 * 1], [2, 8]
        total = sum(counts)
        self.assertAlmostEqual(sum(sums) / total, 1.4)
        # A scalar linear loss has derivative equal to its coefficient.
        world = len(sums)
        ddp_gradient = sum(value * world / total for value in sums) / world
        self.assertAlmostEqual(ddp_gradient, 1.4)

    def test_grpo_and_gspo_example(self):
        rewards = [0, 1, 2]
        mean = sum(rewards) / 3
        std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / 3)
        self.assertAlmostEqual((2 - mean) / std, math.sqrt(3 / 2))
        ratio = math.exp((math.log(0.22 / 0.2) + math.log(0.6 / 0.5)) / 2)
        self.assertAlmostEqual(ratio, math.sqrt(1.32))
        self.assertAlmostEqual(ratio, 1.1489, places=4)
        # Finite difference of -A * exp(mean(logprob_delta)).
        advantage, delta = math.sqrt(3 / 2), [math.log(1.1), math.log(1.2)]
        eps = 1e-6
        loss = lambda x: -advantage * math.exp((x + delta[1]) / 2)
        numerical = (loss(delta[0] + eps) - loss(delta[0] - eps)) / (2 * eps)
        self.assertAlmostEqual(numerical, -advantage * ratio / 2, places=8)

    def test_online_softmax_rescaling(self):
        maximum, denominator, numerator = 0, 1, 1
        new_max = math.log(2)
        scale = math.exp(maximum - new_max)
        denominator = scale * denominator + 1
        numerator = scale * numerator + 3
        self.assertAlmostEqual(numerator / denominator, 7 / 3)

    def test_speculative_mass_conservation(self):
        distributions = [
            ([0.5, 0.3, 0.2], [0.2, 0.5, 0.3]),
            ([1, 0, 0], [0, 1, 0]),
            ([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]),
        ]
        for target, draft in distributions:
            accepted = [min(p, q) for p, q in zip(target, draft)]
            residual = [max(p - q, 0) for p, q in zip(target, draft)]
            self.assertAlmostEqual(sum(residual), 1 - sum(accepted))
            for p, a, r in zip(target, accepted, residual):
                self.assertAlmostEqual(a + r, p)

    def test_kv_cache_units(self):
        cache = 2 * 4 * 8192 * 8 * 128 * 32 * 2
        self.assertEqual(cache / 2**30, 4)
        self.assertAlmostEqual(7e9 * 2 / 2**30, 13.04, places=2)

    def test_paired_interval_example(self):
        differences = [1] * 15 + [-1] * 5 + [0] * 80
        mean = sum(differences) / len(differences)
        variance = sum((d - mean) ** 2 for d in differences) / 99
        error = math.sqrt(variance / 100)
        self.assertAlmostEqual(variance, 19 / 99)
        self.assertAlmostEqual(100 * (mean - 1.96 * error), 1.41, places=2)
        self.assertAlmostEqual(100 * (mean + 1.96 * error), 18.59, places=2)

    def test_potential_shaping_telescopes(self):
        gamma, potentials = 0.9, [0.3, 0.7, 0.2, 0]
        extra = sum(gamma**t * (gamma * potentials[t + 1] - potentials[t])
                    for t in range(len(potentials) - 1))
        self.assertAlmostEqual(extra, -potentials[0])


if __name__ == "__main__":
    unittest.main()
