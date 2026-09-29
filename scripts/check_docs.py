"""Standard-library-only static checks; NOT a renderer or runtime correctness proof."""
from pathlib import Path
import json
import re
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def fences(text):
    opening = None
    body = []
    for number, line in enumerate(text.splitlines(), 1):
        if opening is None:
            m = re.match(r'^(`{3,})([^`]*)$', line)
            if m:
                opening = (number, len(m[1]), m[2].strip())
                body = []
        elif re.fullmatch(r'`{' + str(opening[1]) + r',}\s*', line):
            yield opening[0], number, opening[2], '\n'.join(body)
            opening = None
        else:
            body.append(line)
    if opening:
        raise ValueError(f'unclosed fence at line {opening[0]}')


def main():
    errors, fragments = [], []
    count_python = count_math = 0
    math_export = []
    files = sorted(ROOT.rglob('*.md'))
    for path in files:
        if any(part.startswith('.') for part in path.relative_to(ROOT).parts):
            continue
        relative = path.relative_to(ROOT)
        text = path.read_text(encoding='utf-8')
        try:
            blocks = list(fences(text))
        except ValueError as exc:
            errors.append(f'{relative}: {exc}')
            continue
        prose = text.splitlines()
        math_regions = []
        for start, end, language, body in blocks:
            prose[start - 1:end] = [''] * (end - start + 1)
            if language == 'python':
                count_python += 1
                if body.lstrip().startswith('# 片段：'):
                    fragments.append(f'{relative}:{start}')
                else:
                    try:
                        compile(body, f'{relative}:{start}', 'exec')
                    except SyntaxError as exc:
                        errors.append(f'{relative}:{start}: {exc.msg}')
            elif language == 'math':
                math_regions.append((start, body))
        prose = '\n'.join(prose)
        # Ignore inline code when looking for math or local Markdown links.
        prose = re.sub(r'`[^`\n]+`', '', prose)
        for match in re.finditer(r'\$\$(.*?)\$\$|(?<![\\$])\$(?!\$)(.*?)(?<!\\)\$', prose, re.S):
            math_regions.append((prose[:match.start()].count('\n') + 1,
                                 match[1] if match[1] is not None else match[2]))
        for line, formula in math_regions:
            count_math += 1
            math_export.append({'file': str(relative), 'line': line, 'formula': formula})
            balance = 0
            for brace in re.findall(r'(?<!\\)[{}]', formula):
                balance += 1 if brace == '{' else -1
                if balance < 0:
                    break
            if balance:
                errors.append(f'{relative}:{line}: unbalanced math braces')
            if re.search(r'\\operatorname\b|<[A-Za-z][^\s]*>', formula):
                errors.append(f'{relative}:{line}: known restricted macro or HTML-like math')
        for match in re.finditer(r'\[[^\]\n]*\]\(([^\s)]+)\)', prose):
            url = urlsplit(match[1])
            if url.scheme or url.netloc or not url.path:
                continue
            target = path.parent / unquote(url.path)
            if not target.exists():
                errors.append(f'{relative}: missing local link {match[1]}')
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
