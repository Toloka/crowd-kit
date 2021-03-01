
import inspect
from collections import defaultdict
from io import StringIO
from typing import List, get_type_hints

from ..common import Node, BaseDefinition, BaseASTBuilder

from ..literals.reference_literal import ReferenceLiteral
from ..literals.type_hint_literal import TypeHintLiteral


class ModuleDef(BaseDefinition):

    # TODO: support imported functions
    # TODO: support imported classes

    def __init__(self, node: Node, ast: BaseASTBuilder):
        super().__init__(node, ast)

        self.docstring = self.ast.get_docstring(self.node)

        self.members = {}
        for member_name in self.get_public_member_names():
            # self.members[member_name] = self.get_member_rep(member_name)

            node = self.node.get_member(member_name)
            member = node.obj

            if inspect.isfunction(member):
                definition = self.ast.get_function_definition(node)
            elif inspect.isclass(member) and member.__module__ == self.ast.module_name:
                definition = self.ast.get_class_definition(node)
            else:
                definition = self.ast.get_attribute_definition(node)

            self.members[member_name] = definition

        self.annotations = {}
        for member_name, annotation in get_type_hints(self.obj).items():
            self.annotations[member_name] = self.get_annotation_rep(member_name)

    def __str__(self):
        # TODO: print __all__ if present

        sio = StringIO()

        imports, from_imports = self.get_imports()
        if imports:
            for name in sorted(imports):
                sio.write(f'import {name}\n')
            sio.write('\n')

        if from_imports:
            for key in sorted(from_imports.keys()):
                names = ', '.join(
                    f'{name} as {import_as}' if import_as else name
                    for name, import_as in from_imports[key]
                )
                sio.write(f'from {key} import {names}\n')
            sio.write('\n')

        if self.docstring:
            sio.write(str(self.docstring))

        if self.annotations:
            for name, annotation in self.annotations.items():
                sio.write(f'{annotation}\n')
            sio.write('\n')

        if self.members:
            for name, rep in self.members.items():
                sio.write(f'{rep}\n\n')

        return sio.getvalue()

    def __iter__(self):
        if self.docstring:
            yield self.docstring

        yield from self.members.values()
        yield from self.annotations.values()

    def get_imports(self):
        imports = set()
        from_imports = defaultdict(set)

        for curr in self.traverse():
            if isinstance(curr, TypeHintLiteral):
                from_imports['typing'].add((curr.type_hint_name, None))

            if isinstance(curr, ReferenceLiteral):
                if inspect.ismodule(curr.obj):
                    if curr.obj.__name__ != 'builtins':
                        imports.add(curr.obj.__name__)
                elif inspect.isclass(curr.obj):
                    # TODO: check if a class is actually defined outside of our module
                    if curr.obj.__module__ != 'builtins':
                        from_imports[curr.obj.__module__].add((curr.obj.__qualname__.split('.')[0], None))

        return imports, from_imports

    def get_public_member_names(self) -> List[str]:
        if hasattr(self.obj, '__all__'):
            return list(self.obj.__all__)
        return [name for name in dir(self.obj) if not name.startswith('_')]
