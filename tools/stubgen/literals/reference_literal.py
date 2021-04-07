import inspect

from itertools import zip_longest
from ..common import BaseLiteral


class ReferenceLiteral(BaseLiteral):

    def __str__(self):
        if inspect.isclass(self.obj) or inspect.isfunction(self.obj):
            i = 0
            namespace_tokens = self.namespace.split('.')
            qualname_tokens = self.obj.__qualname__.split('.')

            for i, (namespace_token, qualname_token) in enumerate(zip_longest(namespace_tokens, qualname_tokens)):
                if namespace_token != qualname_token:
                    break
            return '.'.join(qualname_tokens[i:])

        return self.obj.__name__

    __repr__ = __str__

    def __iter__(self):
        yield from ()
