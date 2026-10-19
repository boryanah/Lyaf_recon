#!/usr/bin/env python3
"""Check that every repository path mentioned in the Markdown documentation exists.

Usage: python tools/check_doc_paths.py [file.md ...]   (default: the main docs)
A "repository path" is a backticked token that starts with one of the top-level
directories (pipeline/, paper/, figures/, docs/, environment/, tools/) or with
a relative prefix (../). Tokens containing glob characters, braces or the
placeholders <...>/$SCR are expanded or skipped as noted in the output.
"""
import glob
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOP = ('code/', 'data/', 'environment/', 'tools/')
DEFAULT = ['README.md', 'RESULTS.md', 'HEAVY_INPUTS.md', 'environment/README.md']


def expand_braces(token):
    m = re.search(r'\{([^{}]*)\}', token)
    if not m:
        return [token]
    out = []
    for alt in m.group(1).split(','):
        out.extend(expand_braces(token[:m.start()] + alt + token[m.end():]))
    return out


def main(files):
    missing, checked, skipped = [], 0, 0
    for md in files:
        md_path = ROOT / md
        text = md_path.read_text()
        for token in re.findall(r'`([^`\n]+)`', text):
            token = token.strip()
            if '$' in token or '<' in token or ' ' in token or token.startswith(('http', '/')):
                continue
            if not (token.startswith(TOP) or token.startswith('../')):
                continue
            base = md_path.parent if token.startswith('../') else ROOT
            # Documentation convention: "ph00N" means the six phases ph000..ph005.
            token = token.replace('ph00N', 'ph00?')
            for variant in expand_braces(token):
                variant = variant.rstrip(':;,.')
                target = (base / variant)
                if any(ch in variant for ch in '*?['):
                    hits = glob.glob(str(target))
                    checked += 1
                    if not hits:
                        missing.append((md, variant))
                    continue
                checked += 1
                if not target.exists():
                    missing.append((md, variant))
    print(f'checked {checked} path tokens in {len(files)} files; missing {len(missing)}')
    for md, tok in missing:
        print(f'  MISSING  {md}: {tok}')
    return 1 if missing else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:] or DEFAULT))
