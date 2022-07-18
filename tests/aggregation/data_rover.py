import pandas as pd
import pytest


@pytest.fixture
def simple_text_result_rover() -> pd.Series:
    return pd.Series(
        [
            'as soon as you donned my worst is stocking sweetheart',
            'it must indeed be allowed that the structure of his sentences is expanded and often has somewhat of the inversion of latin and that he delighted to express familiar thoughts in a philosophical language being in this the reverse of socrates who it was said reduced philosophy to the simplicity of common life',  # noqa
            "nature discovers this confusion to us painters hold that the same motions and grimaces of the face that serve for weeping serve for laughter too and indeed before the one or the other be finished do but observe the painter's manner of handling and you will be in doubt to which of the two the design tends and the extreme of laughter does it last bring tears",  # noqa
            'there is more than one amongst us who would like to imitate them i think',
            "it's all nonsense ned cried chris for them to think that they're staying on account of us hello griggs were you listening"  # noqa
        ],
        index=pd.Index(['1255-74899-0020', '1651-136854-0030', '7601-101619-0003', '7601-175351-0021', '8254-84205-0005'], name='task'),
        name='agg_text'
    )
