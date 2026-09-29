"""Regression tests for extraction, independent of PyTorch."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "check_docs", Path(__file__).resolve().parents[1] / "scripts/check_docs.py")
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


class DocumentCheckTests(unittest.TestCase):
    def test_indented_math_is_checked(self):
        source = "  ```math\n  \\operatorname{clip}(x)\n  ```"
        self.assertEqual(list(checks.fences(source))[0][:3], (1, 3, "math"))
        regions = checks.math_regions(source)
        self.assertEqual(len(regions), 1)
        self.assertIn("known restricted macro or HTML-like math",
                      list(checks.math_errors(regions[0][1])))

    def test_nested_containers(self):
        for source in [
            "- item\n\n  ```math\n  x^2\n  ```",
            "> ```math\n> x^2\n> ```",
            "> - item\n>\n>   ~~~math\n>   x^2\n>   ~~~",
            "- item\n  - nested\n\n    ```math\n    x^2\n    ```",
        ]:
            with self.subTest(source=source):
                self.assertEqual(checks.math_regions(source)[0][1].strip(), "x^2")

    def test_longer_fences_do_not_close_early(self):
        source = "````text\n```math\n$x$\n```\n````\n$y$"
        self.assertEqual(checks.math_regions(source), [(6, "y")])

    def test_missing_closer_is_rejected(self):
        for source in ["```math\nx", "> ```math\n> x\n\noutside",
                       "~~~math\nx\n```"]:
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, "unclosed fence"):
                list(checks.fences(source))

    def test_python_container_indentation_is_removed(self):
        block = list(checks.fences("- code\n\n  ```python\n  x = 1\n  ```"))[0]
        compile(block[3], "<fixture>", "exec")

    def test_code_spans_and_indented_code_are_not_math(self):
        source = "`$no$` and `` ` $no$ `` and $yes$\n\n    $not_math$\n"
        self.assertEqual(checks.math_regions(source), [(1, "yes")])

    def test_display_math_and_escaped_currency(self):
        source = "Price: \\$5, $x$.\n\n$$\n\\frac{a}{b}\n$$"
        self.assertEqual(checks.math_regions(source),
                         [(1, "x"), (3, "\n\\frac{a}{b}\n")])

    def test_unmatched_or_mixed_dollars_fail(self):
        for source in ["$x", "$$x", "$x$$", "$x\nnext $", "$$$$"]:
            with self.subTest(source=source), self.assertRaises(ValueError):
                checks.math_regions(source)

    def test_math_braces_and_escaped_braces(self):
        self.assertTrue(list(checks.math_errors(r"\frac{x}{y")))
        self.assertTrue(list(checks.math_errors("}{")))
        self.assertFalse(list(checks.math_errors(r"\{x\}")))

    def test_reference_links_and_code_examples(self):
        source = "[target][ref]\n\n[ref]: docs/a.md\n\n`[fake](bad.md)`"
        self.assertEqual(list(checks.local_links(source)), ["docs/a.md"])

    def test_repository_gradient_clipping_is_extracted(self):
        path = Path(__file__).resolve().parents[1] / "docs/02-training.md"
        formulas = [formula for _, formula in checks.math_regions(path.read_text())]
        self.assertTrue(any(r"\lVert g\rVert_2" in formula for formula in formulas))


if __name__ == "__main__":
    unittest.main()
