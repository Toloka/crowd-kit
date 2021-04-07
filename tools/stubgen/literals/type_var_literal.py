from typing import TypeVar
from ..common import BaseLiteral, BaseASTBuilder, Node


class TypeVarLiteral(BaseLiteral):
    """Represents a TypeVar"""

    def __init__(self, node: Node, ast: BaseASTBuilder):
        super().__init__(node, ast)
        self._name = self.ast.get_literal(Node(self.namespace, None, self.obj.__name__))
        self._covariant = self.ast.get_literal(Node(self.namespace, None, self.obj.__covariant__))
        self._contravariant = self.ast.get_literal(Node(self.namespace, None, self.obj.__contravariant__))
        self._bound = self.ast.get_literal(Node(self.namespace, None, self.obj.__bound__))

    def __str__(self) -> str:
        return f'TypeVar({self._name}, covariant={self._covariant}, contravariant={self._contravariant}, bound={self._bound})'

    __repr__ = __str__

    def __iter__(self):
        yield self.ast.get_literal(Node(self.namespace, None, TypeVar))
        yield self._name
        yield self._covariant
        yield self._contravariant
        yield self._bound
