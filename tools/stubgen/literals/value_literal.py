from ..common import BaseLiteral
from typing import ForwardRef


class ValueLiteral(BaseLiteral):

    def __str__(self):
        if self.obj is None:
            return 'None'

        if isinstance(self.obj, (int, str, bool)):
            return repr(self.obj)

        if isinstance(self.obj, ForwardRef):
            return repr(self.obj.__forward_arg__)

        return '...'

    __repr__ = __str__

    def __iter__(self):
        yield from ()
