import inspect
from io import StringIO

from ..common import BaseDefinition, Node, BaseASTBuilder


class FunctionDef(BaseDefinition):

    # TODO: support statimethods and classmethods

    def __init__(self, node: Node, ast: BaseASTBuilder):
        super().__init__(node, ast)

        signature = inspect.signature(self.obj)

        params = []
        for param in signature.parameters.values():
            if param.annotation is not inspect.Parameter.empty:
                param = param.replace(annotation=ast.get_literal(param.annotation))
            params.append(param)

        return_annotation = signature.return_annotation
        if return_annotation is not inspect.Parameter.empty:
            return_annotation = ast.get_literal(return_annotation)

        self.signature = signature.replace(parameters=params, return_annotation=return_annotation)
        self.docstring = ast.get_docstring(node)

    def __str__(self):
        sio = StringIO()

        if self.docstring:
            sio.write(f'def {self.name}{self.signature}:\n')
            sio.write(self.indent(str(self.docstring)))
            sio.write(self.indent('...'))
        else:
            sio.write(f'def {self.name}{self.signature}: ...')

        return sio.getvalue()

    def __iter__(self):
        if self.docstring:
            yield self.docstring

        for param in self.signature.parameters.values():
            if param.annotation is not inspect.Parameter.empty:
                yield param.annotation

        if self.signature.return_annotation is not inspect.Parameter.empty:
            yield self.signature.return_annotation


class ClassMethodDef(FunctionDef):

    def __str__(self):
        return f'@classmethod\n{super().__str__()}'


class StaticMethodDef(FunctionDef):

    def __str__(self):
        return f'@staticmethod\n{super().__str__()}'
