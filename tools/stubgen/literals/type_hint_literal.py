from typing import get_args

from ..common import BaseLiteral, BaseASTBuilder


class TypeHintLiteral(BaseLiteral):
    """Represents a type hint"""

    def __init__(self, obj, ast: BaseASTBuilder):
        super().__init__(obj, ast)

        name = obj._name or obj.__origin__._name
        args = get_args(obj)

        if name == 'Union' and len(args) == 2 and args[-1] is type(None):  # noqa: E721
            name = 'Optional'
            args = args[:-1]

        self.type_hint_name = name
        self.type_hint_args = [self.ast.get_literal(arg) for arg in args]

    def __str__(self) -> str:
        if self.type_hint_args:
            args = ', '.join(str(arg) for arg in self.type_hint_args)
            return f'{self.type_hint_name}[{args}]'

        return self.type_hint_name

    __repr__ = __str__

    def __iter__(self):
        yield from self.type_hint_args
