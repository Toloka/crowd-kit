from ..common import BaseLiteral, BaseASTBuilder, Node


class TypeHintLiteral(BaseLiteral):
    """Represents a type hint"""

    def __init__(self, node: Node, ast: BaseASTBuilder):
        super().__init__(node, ast)

        name = self.obj._name or self.obj.__origin__._name
        # typing.get_args works with Callable[[], int] but does not work with
        # Callable in Python 3.8. So __args__ seems more reliable 
        args = getattr(self.obj, '__args__', ())

        if name == 'Union' and len(args) == 2 and args[-1] is type(None):  # noqa: E721
            name = 'Optional'
            args = args[:-1]

        args = [None if arg is type(None) else arg for arg in args]  # noqa: E721
        self.type_hint_name = name
        self.type_hint_args = [self.ast.get_literal(Node(self.namespace, None, arg)) for arg in args]

    def __str__(self) -> str:
        if self.type_hint_args:
            args = ', '.join(str(arg) for arg in self.type_hint_args)
            return f'{self.type_hint_name}[{args}]'

        return self.type_hint_name

    __repr__ = __str__

    def __iter__(self):
        yield from self.type_hint_args
