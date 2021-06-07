from enum import Enum
from io import StringIO

from ..common import Node, BaseDefinition, BaseASTBuilder
from ..literals.reference_literal import ReferenceLiteral


class EnumDef(BaseDefinition):

    def __init__(self, node: Node, ast: BaseASTBuilder):
        super().__init__(node, ast)
        assert self.obj.__bases__ == (Enum,)
        self.docstring = self.ast.get_docstring(self.node)
        self.enum_dict = {e.name: ast.get_literal(Node(self.namespace, None, e.value)) for e in self.obj}

    def __str__(self):

        sio = StringIO()

        sio.write(f'class {self.name}(Enum):\n')

        if self.docstring:
            sio.write(self.indent(f'{self.docstring}\n'))

        if self.enum_dict:
            for name, literal in self.enum_dict.items():
                sio.write(self.indent(f'{name} = {literal}\n'))
        else:
            sio.write(self.indent('...'))
        sio.write('\n')

        return sio.getvalue()

    def __iter__(self):
        yield ReferenceLiteral(Node(self.namespace, None, Enum), self.ast)
        yield from self.enum_dict.values()
