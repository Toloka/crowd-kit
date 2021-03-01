from ..common import BaseLiteral


class ValueLiteral(BaseLiteral):

    def __str__(self):
        if self.obj is None:
            return 'None'

        if isinstance(self.obj, str):
            return repr(self.obj)

        return f'...{self.obj}'

    __repr__ = __str__

    def __iter__(self):
        yield from ()
