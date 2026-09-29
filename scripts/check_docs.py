"""Markdown-aware static checks; NOT a GitHub renderer or correctness proof."""
from pathlib import Path
import json
import re
import sys
from urllib.parse import unquote, urlsplit

try:
    from markdown_it import MarkdownIt
except ImportError:
    raise SystemExit("Install checker dependencies: python -m pip install -r tests/requirements-docs.txt")

ROOT = Path(__file__).resolve().parents[1]
MARKDOWN = MarkdownIt("commonmark")


def fences(text):
    """Use CommonMark container rules for indentation, lists and blockquotes."""
    for token in MARKDOWN.parse(text):
        if token.type != 'fence':
            continue
        start, end = token.map
        # CommonMark accepts EOF/container termination without a closing fence.
        # The handbook intentionally requires an explicit closing delimiter.
        if end - start != len(token.content.splitlines()) + 2:
            raise ValueError(f'unclosed fence at line {start + 1}')
        yield start + 1, end, token.info.strip(), token.content


def escaped(text, index):
    backslashes = 0
    while index > 0 and text[index - 1] == '\\':
        backslashes += 1
        index -= 1
    return backslashes % 2 == 1


def dollar_math(prose):
    """Extract repository-style $inline$ / $$display$$, rejecting unmatched $.

    Literal currency dollars must be escaped. Inline math is single-line;
    multiline expressions belong in display math or math fences.
    """
    # Preserve line numbers while removing single/multiple-backtick code spans.
    prose = re.sub(
        r'(?<![\\`])(`+)(?!`)([\s\S]*?)(?<!`)\1(?!`)',
        lambda match: '\n' * match[0].count('\n'), prose,
    )
    opening = None
    for match in re.finditer(r'\${1,2}', prose):
        if escaped(prose, match.start()):
            continue
        delimiter = match[0]
        line = prose[:match.start()].count('\n') + 1
        if opening is None:
            opening = (delimiter, match.end(), line)
        else:
            expected, start, start_line = opening
            if delimiter != expected:
                raise ValueError(f'mismatched math delimiter at line {line}')
            formula = prose[start:match.start()]
            if expected == '$' and '\n' in formula:
                raise ValueError(f'unclosed inline math at line {start_line}')
            if not formula.strip():
                raise ValueError(f'empty math at line {start_line}')
            yield start_line, formula
            opening = None
    if opening:
        raise ValueError(f'unclosed math delimiter at line {opening[2]}')


def math_regions(text):
    blocks = list(fences(text))
    prose = text.splitlines()
    regions = []
    for start, end, language, body in blocks:
        prose[start - 1:end] = [''] * (end - start + 1)
        if language == 'math':
            regions.append((start, body))
    # Indented code is not prose, either.
    for token in MARKDOWN.parse(text):
        if token.type == 'code_block':
            start, end = token.map
            prose[start:end] = [''] * (end - start)
    regions.extend(dollar_math('\n'.join(prose)))
    return regions


def math_errors(formula):
    balance = 0
    for index, char in enumerate(formula):
        if char not in '{}' or escaped(formula, index):
            continue
        balance += 1 if char == '{' else -1
        if balance < 0:
            break
    if balance:
        yield 'unbalanced math braces'
    if re.search(r'\\operatorname\b|<[A-Za-z][^\s]*>', formula):
        yield 'known restricted macro or HTML-like math'


def local_links(text):
    """Read actual inline/reference links, not Markdown examples in code."""
    for token in MARKDOWN.parse(text):
        for child in token.children or []:
            if child.type == 'link_open':
                yield child.attrGet('href')
            elif child.type == 'image':
                yield child.attrGet('src')


def main():
    errors, fragments = [], []
    count_python = count_math = 0
    math_export = []
    files = sorted(path for path in ROOT.rglob('*.md')
                   if not any(part.startswith('.') or part in {'node_modules', '__pycache__'}
                              for part in path.relative_to(ROOT).parts))
    for path in files:
        relative = path.relative_to(ROOT)
        text = path.read_text(encoding='utf-8')
        try:
            blocks = list(fences(text))
            regions = math_regions(text)
        except ValueError as exc:
            errors.append(f'{relative}: {exc}')
            continue
        for start, end, language, body in blocks:
            if language == 'python':
                count_python += 1
                if body.lstrip().startswith('# 片段：'):
                    fragments.append(f'{relative}:{start}')
                else:
                    try:
                        compile(body, f'{relative}:{start}', 'exec')
                    except SyntaxError as exc:
                        errors.append(f'{relative}:{start}: {exc.msg}')
        for line, formula in regions:
            count_math += 1
            math_export.append({'file': str(relative), 'line': line, 'formula': formula})
            errors.extend(f'{relative}:{line}: {message}' for message in math_errors(formula))
        for link in local_links(text):
            url = urlsplit(link)
            if url.scheme or url.netloc or not url.path:
                continue
            target = path.parent / unquote(url.path)
            if not target.exists():
                errors.append(f'{relative}: missing local link {link}')
    json_mode = '--math-json' in sys.argv
    output = sys.stderr if json_mode else sys.stdout
    print(f'{len(files)} Markdown files; {count_python} Python blocks; {count_math} math regions', file=output)
    print(f'{len(fragments)} explicitly marked control-flow fragments (not compiled standalone)', file=output)
    for message in errors:
        print('ERROR:', message, file=output)
    if errors:
        raise SystemExit(1)
    print('Static checks passed. Anchors, TeX rendering, and untested example behavior are not proven.', file=output)
    if json_mode:
        print(json.dumps(math_export, ensure_ascii=False))


if __name__ == '__main__':
    main()
