from .value_literal import ValueLiteral


class AnnotationLiteral(ValueLiteral):

    def __str__(self):
        if self.value is type(None):  # noqa: E721
            return 'None'

        return super().__str__()

    __repr__ = __str__
