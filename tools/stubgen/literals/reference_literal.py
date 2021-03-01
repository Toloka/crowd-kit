import inspect

from ..common import BaseLiteral, BaseASTBuilder


class ReferenceLiteral(BaseLiteral):

    def __init__(self, obj, ast: BaseASTBuilder):
        assert inspect.isclass(obj) or inspect.ismodule(obj)
        super().__init__(obj, ast)

    def __str__(self):
        # if inspect.isclass(self.obj):
        #     return str(f'{self.obj.__module__}.{self.obj.__qualname__}')
        if inspect.isclass(self.obj):
            return str(self.obj.__qualname__.split('.')[0])

        return self.obj.__name__

    __repr__ = __str__

    def __iter__(self):
        yield from ()
