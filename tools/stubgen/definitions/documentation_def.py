import inspect
import docstring_parser

from ..common import BaseDefinition


class DocumentationDef(BaseDefinition):

    def __str__(self) -> str:
        return f'"""{inspect.cleandoc(self.obj).rstrip()}"""\n'

    def __iter__(self):
        yield from ()

    def get_parsed(self):
        return docstring_parser.google.parse(inspect.cleandoc(self.obj))
