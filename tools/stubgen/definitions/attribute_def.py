from ..common import Node, BaseDefinition, BaseASTBuilder


class AttributeDef(BaseDefinition):
    """Represents `name = value`"""

    def __init__(self, node: Node, ast: BaseASTBuilder):
        super().__init__(node, ast)
        self.value = self.ast.get_literal(node)

    def __str__(self):
        return f'{self.name} = {self.value}'

    def __iter__(self):
        yield self.value
