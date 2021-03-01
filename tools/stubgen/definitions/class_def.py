import inspect
from io import StringIO
from typing import get_type_hints

from ..common import Node, BaseDefinition, BaseASTBuilder


class ClassDef(BaseDefinition):

    # TODO: support properties

    def __init__(self, node: Node, ast: BaseASTBuilder):
        super().__init__(node, ast)

        self.docstring = self.ast.get_docstring(self.node)

        self.bases = []
        if self.obj.__bases__ != (object,):
            for base in self.node.obj.__bases__:
                self.bases.append(self.ast.get_literal(base))

        self.members = {}
        for member_name in self.get_public_member_names():
            # Accessing members through __dict__ is important in order to be able
            # to distinguish between methods, classmethods and staticmethods
            member = self.obj.__dict__[member_name]

            # TODO: dirty hack
            node = self.node.get_member(member_name)
            node.obj = member

            if isinstance(member, staticmethod):
                node.obj = member.__func__
                definition = self.ast.get_static_method_definition(node)
            elif isinstance(member, classmethod):
                node.obj = member.__func__
                definition = self.ast.get_class_method_definition(node)
            elif inspect.isfunction(member):
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
        sio = StringIO()

        if self.node.obj.__bases__ == (object,):
            sio.write(f'class {self.name}:\n')
        else:
            bases = ', '.join(base.__name__ for base in self.node.obj.__bases__)
            sio.write(f'class {self.name}({bases}):\n')

        if self.docstring:
            sio.write(self.indent(f'{self.docstring}\n'))

        if self.annotations:
            for name, annotation in self.annotations.items():
                sio.write(self.indent(f'{annotation}\n'))
            sio.write('\n')

        if self.members:
            for name, rep in self.members.items():
                sio.write(self.indent(f'{rep}\n\n'))
        else:
            sio.write(self.indent('pass'))

        return sio.getvalue()

    def __iter__(self):
        if self.docstring:
            yield self.docstring

        yield from self.bases
        yield from self.members.values()
        yield from self.annotations.values()

    def get_public_member_names(self):
        cls = self.obj
        super_cls = super(cls, cls)

        for name in dir(cls):

            # Skipping all dunder members except for __init__
            if name.startswith('__') and name != '__init__':
                continue

            # Only considering members that were actually (re)defined in cls
            if getattr(cls, name) is not getattr(super_cls, name, None):
                yield name

        # return [name for name in dir(self.obj) if not name.startswith('__') and name != '__init__']
