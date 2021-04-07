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
                param = param.replace(annotation=ast.get_literal(Node(self.namespace, None, param.annotation)))
            if param.default is not inspect.Parameter.empty:
                param = param.replace(default=ast.get_literal(Node(self.name, None, param.default)))
            params.append(param)

        return_annotation = signature.return_annotation
        if return_annotation is not inspect.Parameter.empty:
            return_annotation = ast.get_literal(Node(self.namespace, None, return_annotation))

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
            if param.default is not inspect.Parameter.empty:
                yield param.default

        if self.signature.return_annotation is not inspect.Parameter.empty:
            yield self.signature.return_annotation

    def get_markdown(self):
        sio = StringIO()

        if self.name.startswith('_') and self.name != '__init__':
            return ''

        sio.write('---\n\n')

        if self.node.namespace:
            if self.node.obj.__module__:
                sio.write(f'### {self.escape_markdown(self.node.obj.__module__)}.{self.escape_markdown(self.node.namespace)}.{self.escape_markdown(self.name)}\n\n')
            else:
                sio.write(f'### {self.escape_markdown(self.node.namespace)}.{self.escape_markdown(self.name)}\n\n')
        else:
            if self.node.obj.__module__:
                sio.write(f'### {self.escape_markdown(self.node.obj.__module__)}.{self.escape_markdown(self.name)}\n\n')
            else:
                sio.write(f'### {self.escape_markdown(self.name)}\n\n')
        sio.write(f'*{self.escape_markdown(self.name)}{self.escape_markdown(str(self.signature), escape_asterisk=True)}*\n\n')

        if self.docstring:
            parsed_docstring = self.docstring.get_parsed()
            if parsed_docstring.short_description:
                sio.write(f'{self.escape_markdown(parsed_docstring.short_description)}\n\n')

            if parsed_docstring.long_description:
                sio.write(f'{self.escape_markdown(parsed_docstring.long_description)}\n\n')

            if parsed_docstring.params:
                sio.write('**Args:**\n\n')
                for param in parsed_docstring.params:
                    annotation = self.signature.parameters.get(param.arg_name) and self.signature.parameters[param.arg_name].annotation
                    if annotation:
                        sio.write(f'* ***{self.escape_markdown(param.arg_name)}: '
                                  f'{self.escape_markdown(str(annotation))}***\n')
                    else:
                        sio.write(f'* ***{self.escape_markdown(param.arg_name)}***\n')
                    sio.write(f'* * {self.escape_markdown(param.description)}\n\n')

            if parsed_docstring.returns:
                ret = parsed_docstring.returns
                if ret.args[0] == 'returns':
                    sio.write('**Returns:**\n\n')
                elif ret.args[0] == 'yields':
                    sio.write('**Yields:**\n\n')
                sio.write(f'* ***{self.escape_markdown(ret.type_name)}***\n')
                sio.write(f'* * {self.escape_markdown(ret.description)}\n\n')

            if parsed_docstring.raises:
                sio.write('**Raises:**\n\n')
                for rai in parsed_docstring.raises:
                    sio.write(f'* ***{self.escape_markdown(rai.type_name)}***\n')
                    sio.write(f'* * {self.escape_markdown(rai.description)}\n\n')

            first_example = True
            for m in parsed_docstring.meta:
                if m.args[0] == 'examples':
                    if first_example:
                        sio.write('**Examples:**\n\n')
                        first_example = False
                    description = m.description.replace('>>>', '\t')
                    sio.write(f'{description}\n\n')

        sio.write('---\n\n')

        return sio.getvalue()


class ClassMethodDef(FunctionDef):

    def __str__(self):
        return f'@classmethod\n{super().__str__()}'


class StaticMethodDef(FunctionDef):

    def __str__(self):
        return f'@staticmethod\n{super().__str__()}'
