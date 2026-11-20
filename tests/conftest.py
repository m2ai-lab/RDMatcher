import os
import sys

# Prepend project root and src to sys.path so tests can import rdmatcher.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'src')
for p in (SRC, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)
