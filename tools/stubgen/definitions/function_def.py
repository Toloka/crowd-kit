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
        self._expanded_docstring = None

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

    def get_doc_sources(self):
        sio = StringIO()

        if self.docstring:
            sio.write(f'def {self.name}{self.signature}:\n')
            sio.write(self.indent(str(self.expanded_docstring)))
            sio.write(self.indent('...'))
        else:
            sio.write(f'def {self.name}{self.signature}: ...')

        return sio.getvalue()

    @property
    def expanded_docstring(self):
        if self._expanded_docstring is not None:
            return self._expanded_docstring
        sio = StringIO()

        if self.docstring:
            sio.write('"""')
            parsed_docstring = self.docstring.get_parsed()
            if parsed_docstring.short_description:
                sio.write(f'{parsed_docstring.short_description}\n')

            if parsed_docstring.long_description:
                sio.write(f'\n{parsed_docstring.long_description}\n')

            if parsed_docstring.params:
                sio.write('\nArgs:\n')
                for param in parsed_docstring.params:
                    annotation = self.signature.parameters.get(param.arg_name) and self.signature.parameters[param.arg_name].annotation
                    if annotation:
                        sio.write(self.indent(f'{param.arg_name} ({annotation})'))
                    else:
                        sio.write(self.indent(f'{param.arg_name}'))
                    description = param.description.replace('\n', '\n\t\t')
                    sio.write(f': {description}\n')

            if parsed_docstring.returns:
                ret = parsed_docstring.returns
                if ret.args[0] == 'returns':
                    sio.write('\nReturns:\n')
                elif ret.args[0] == 'yields':
                    sio.write('\nYields:\n')
                sio.write(self.indent(f'{ret.type_name}: {ret.description}\n'))

            if parsed_docstring.raises:
                sio.write('\nRaises:\n')
                for rai in parsed_docstring.raises:
                    sio.write(self.indent(f'{rai.type_name}: {rai.description}\n'))

            first_example = True
            for m in parsed_docstring.meta:
                if m.args[0] == 'examples':
                    if first_example:
                        sio.write('\nExamples:\n')
                        first_example = False
                    sio.write(self.indent(f'{m.description}\n'))
            sio.write('"""\n')
        return sio.getvalue()

    @expanded_docstring.setter
    def expanded_docstring(self, docstring):
        self._expanded_docstring = docstring


class ClassMethodDef(FunctionDef):

    def __str__(self):
        return f'@classmethod\n{super().__str__()}'


class StaticMethodDef(FunctionDef):

    def __str__(self):
        return f'@staticmethod\n{super().__str__()}'
