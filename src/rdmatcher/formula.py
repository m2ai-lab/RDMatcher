from __future__ import annotations
from typing import List, Tuple


class FormulaParseError(ValueError):
    pass


class _FormulaParser:
    def __init__(self, s: str):
        self.s = s
        self.i = 0
        self.n = len(s)

    def peek(self):
        if self.i >= self.n:
            return ''
        return self.s[self.i]

    def consume_ws(self):
        while self.peek().isspace():
            self.i += 1

    def parse_name(self) -> str:
        self.consume_ws()
        start = self.i
        while True:
            c = self.peek()
            if c == '' or c.isspace() or c in '+*:()':
                break
            self.i += 1
        if start == self.i:
            raise FormulaParseError(f"Expected name at pos {self.i} in '{self.s}'")
        return self.s[start:self.i]

    def parse_primary(self) -> List[Tuple[str, ...]]:
        self.consume_ws()
        if self.peek() == '(':
            self.i += 1
            terms = self.parse_expr()
            self.consume_ws()
            if self.peek() != ')':
                raise FormulaParseError("Unmatched '(' in formula")
            self.i += 1
            return terms
        else:
            name = self.parse_name()
            return [(name,)]

    def parse_factor(self) -> List[Tuple[str, ...]]:
        # factor -> primary ( ':' primary )*
        terms = self.parse_primary()
        self.consume_ws()
        while self.peek() == ':':
            self.i += 1
            right = self.parse_primary()
            # cross-product of existing terms with right
            new_terms = []
            for a in terms:
                for b in right:
                    new_terms.append(tuple(list(a) + list(b)))
            terms = new_terms
            self.consume_ws()
        return terms

    def parse_term(self) -> List[Tuple[str, ...]]:
        # term -> factor ( '*' factor )*
        left = self.parse_factor()
        self.consume_ws()
        while self.peek() == '*':
            self.i += 1
            right = self.parse_factor()
            # a * b => a + b + a:b
            union = {t for t in left}
            union.update(right)
            # add interactions
            inter = set()
            for a in left:
                for b in right:
                    inter.add(tuple(list(a) + list(b)))
            union.update(inter)
            left = list(union)
            self.consume_ws()
        return left

    def parse_expr(self) -> List[Tuple[str, ...]]:
        # expr -> term ( '+' term )*
        terms = []
        terms.extend(self.parse_term())
        self.consume_ws()
        while self.peek() == '+':
            self.i += 1
            terms.extend(self.parse_term())
            self.consume_ws()
        return terms


def parse_formula(formula: str) -> List[Tuple[str, ...]]:
    """
    Parse a formula string into a list of terms. Each term is a tuple of
    original feature names. Supports +, :, *, and parentheses.

    Examples:
      'age + sex' -> [('age',), ('sex',)]
      'age:sex' -> [('age','sex')]
      '(a + b) * c' -> [('a',), ('b',), ('c',), ('a','c'), ('b','c')]
    """
    if formula is None or formula.strip() == '':
        return []
    p = _FormulaParser(formula)
    terms = p.parse_expr()
    p.consume_ws()
    if p.i != p.n:
        raise FormulaParseError(f"Unexpected trailing text in formula at pos {p.i}: '{formula[p.i:]}'")
    # normalize terms: flatten and remove duplicates preserving order
    seen = set()
    out = []
    for t in terms:
        key = tuple(t)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def term_to_str(term: Tuple[str, ...]) -> str:
    return ':'.join(term)
