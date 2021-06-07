import inspect
from io import StringIO
from typing import get_type_hints

from ..common import Node, BaseDefinition, BaseASTBuilder
from . import AttributeDef


class ClassDef(BaseDefinition):

    # TODO: support properties

    def __init__(self, node: Node, ast: BaseASTBuilder):
        super().__init__(node, ast)

        self.docstring = self.ast.get_docstring(self.node)

        self.bases = []
        if self.obj.__bases__ != (object,):
            for base in self.node.obj.__bases__:
                self.bases.append(self.ast.get_literal(Node(self.namespace, None, base)))

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
            self.annotations[member_name] = self.ast.get_attribute_annotation_definition(Node(
                namespace=f'{self.namespace}.{self.name}' if self.namespace else self.name,
                name=member_name,
                obj=get_type_hints(self.obj)[member_name],
            ))

    def __str__(self):
        sio = StringIO()

        if self.node.obj.__bases__ == (object,):
            sio.write(f'class {self.name}:\n')
        else:
            bases = ', '.join(str(base) for base in self.bases)
            sio.write(f'class {self.name}({bases}):\n')

        if self.docstring:
            sio.write(self.indent(f'{self.docstring}\n'))

        if self.members:
            for name, rep in self.members.items():
                sio.write(self.indent(f'{rep}\n\n'))

        if self.annotations:
            for name, annotation in self.annotations.items():
                sio.write(self.indent(f'{annotation}\n'))
            sio.write('\n')

        if not self.docstring and not self.members and not self.annotations:
            sio.write(self.indent('...'))

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

        var_reg = None
        if hasattr(cls, '_variant_registry'):
            var_reg = cls._variant_registry

        for name in dir(cls):

            # Skipping all dunder members except for __init__
            if name.startswith('__') and name != '__init__':
                continue

            if var_reg and name == var_reg.field:
                continue

            if name == '_variant_registry':
                continue

            # Only considering members that were actually (re)defined in cls
            cls_attr = getattr(cls, name)
            super_cls_attr = getattr(super_cls, name, None)

            if hasattr(cls_attr, '__func__') and hasattr(super_cls_attr, '__func__'):
                if getattr(cls_attr, '__func__') != getattr(super_cls_attr, '__func__'):
                    yield name
            elif getattr(cls, name) is not getattr(super_cls, name, None):
                yield name

        # return [name for name in dir(self.obj) if not name.startswith('__') and name != '__init__']

    def get_doc_sources(self):
        sio = StringIO()

        if self.node.obj.__bases__ == (object,):
            sio.write(f'class {self.name}:\n')
        else:
            bases = ', '.join(str(base) for base in self.bases)
            sio.write(f'class {self.name}({bases}):\n')

        if self.docstring:
            sio.write(self.indent(f'{self.expanded_docstring}\n'))

        if self.members:
            for name, rep in self.members.items():
                if type(rep) is not AttributeDef:
                    sio.write(self.indent(f'{rep.get_doc_sources()}\n\n'))

        if not self.docstring and not self.members:
            sio.write(self.indent('...'))

        return sio.getvalue()

    @property
    def expanded_docstring(self):
        sio = StringIO()

        if self.docstring:
            sio.write('"""')
            parsed_docstring = self.docstring.get_parsed()
            if parsed_docstring.short_description:
                sio.write(f'{parsed_docstring.short_description}\n')
            if parsed_docstring.long_description:
                sio.write(f'\n{parsed_docstring.long_description}\n')

            if parsed_docstring.params:
                first_attribute = True
                for param in parsed_docstring.params:
                    if param.args[0] == 'attribute':
                        if first_attribute:
                            sio.write('\nAttributes:\n')
                            first_attribute = False
                        arg_with_annotations = self.annotations.get(param.arg_name)
                        if arg_with_annotations:
                            arr = str(arg_with_annotations).split(':')
                            arg_with_annotations = f'{arr[0]} ({arr[1][1:]})'
                        sio.write(self.indent(f'{arg_with_annotations or param.arg_name}'))
                        description = param.description.replace('\n', '\n\t\t')
                        sio.write(f': {description}\n')

                first_arg = True
                for param in parsed_docstring.params:
                    if param.args[0] == 'param':
                        if first_arg:
                            sio.write('\nArgs:\n')
                            first_arg = False
                        init_member = self.members['__init__']
                        annotation = init_member.signature.parameters.get(param.arg_name) \
                                     and init_member.signature.parameters[param.arg_name].annotation
                        if annotation:
                            sio.write(self.indent(f'{param.arg_name} ({str(annotation)})'))
                        else:
                            sio.write(self.indent(f'{param.arg_name}'))
                        description = param.description.replace('\n', '\n\t\t')
                        sio.write(f': {description}\n')

            first_example = True
            for m in parsed_docstring.meta:
                if m.args[0] == 'examples':
                    if first_example:
                        sio.write('\nExamples:\n')
                        first_example = False
                    sio.write(f'{m.description}\n')
            sio.write('"""\n')
        return sio.getvalue()
