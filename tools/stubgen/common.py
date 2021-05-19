import textwrap
import re

from typing import Generator, Optional


class Node:
    """Spec for a represented object"""

    def __init__(self, namespace: str, name: Optional[str], obj):
        self.namespace = namespace
        self.name = name
        self.obj = obj

    @property
    def indentation_level(self) -> int:
        if not self.namespace:
            return 0
        return self.namespace.count('.') + 1

    def get_member(self, member_name: str) -> 'Node':
        return Node(
            namespace=f'{self.namespace}.{self.name}' if self.namespace else self.name if self.name else '',
            name=member_name,
            obj=getattr(self.obj, member_name)
        )

    def get_literal_node(self, obj):
        return Node(
            namespace=self.namespace,
            name=None,
            obj=obj
        )


class BaseRepresentation:

    def __init__(self, node: Node, ast: 'ASTBuilder'):
        self.node = node
        self.ast = ast

    def __str__(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    @property
    def obj(self):
        return self.node.obj

    @property
    def name(self):
        return self.node.name

    @property
    def namespace(self):
        return self.node.namespace

    def traverse(self) -> Generator['BaseRepresentation', None, None]:
        """Recursively traverses the definition tree"""
        yield self
        for child in self:
            yield from child.traverse()


class BaseLiteral(BaseRepresentation):

    def __str__(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError


class BaseDefinition(BaseRepresentation):

    INDENT = ' ' * 4

    def __str__(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def indent(self, string: str, level: int = 1) -> str:
        return textwrap.indent(string, self.INDENT * level)

    def get_member_rep(self, member_name: str):
        return self.ast.get_definition(self.node.get_member(member_name))

    def get_doc_sources(self):
        return self.__str__()

    @property
    def expanded_docstring(self):
        raise NotImplementedError


class BaseASTBuilder:

    # Helper methods

    def get_docstring(self, node: Node) -> Optional[BaseDefinition]:
        raise NotImplementedError

    # Get representation for definitions

    def get_definition(self, node: Node) -> BaseDefinition:
        raise NotImplementedError

    def get_attribute_definition(self, node: Node) -> BaseDefinition:
        raise NotImplementedError

    def get_documentation_definition(self, node: Node) -> BaseDefinition:
        raise NotImplementedError

    def get_class_definition(self, node: Node) -> BaseDefinition:
        raise NotImplementedError

    def get_function_definition(self, node: Node) -> BaseDefinition:
        raise NotImplementedError

    def get_module_definition(self, node: Node) -> BaseDefinition:
        raise NotImplementedError

    # Get representations for values

    def get_literal(self, obj):
        raise NotImplementedError

    def get_literal_for_reference(self, obj) -> BaseLiteral:
        raise NotImplementedError

    def get_literal_for_type_hint(self, obj) -> BaseLiteral:
        raise NotImplementedError

    def get_literal_for_value(self, obj: Node) -> BaseLiteral:
        raise NotImplementedError
