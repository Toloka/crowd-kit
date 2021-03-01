import inspect

from ..common import BaseDefinition


class DocumentationDef(BaseDefinition):

    def __str__(self) -> str:
        return f'"""{inspect.cleandoc(self.obj)}"""\n'

    def __iter__(self):
        yield from ()
