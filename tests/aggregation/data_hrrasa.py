import pandas as pd
import pytest


@pytest.fixture
def simple_text_result_hrrasa():
    return pd.Series(
        [
            'i shouuld had seen you damp my word',
            'it must indeed be allowed that is structure and sentences is expanded and often has somewhat of the inversion of latin and that he delighted to express familiar thought in a philosophical language meaning this the works of socrates who it was said reduced philosophy to the simplicity of life ',  # noqa
            'nature discovers are confusing to us pain is holding the same emotions and grimaces of the face that serve for weeping serve for laughter too and indeed before the one and the other be finished do but observe the painter\'s manner of handling and you will be in doubt to which of the two the design tems and the extreme laghter does it last to bring tears',  # noqa
            'they is more than one amongs us who would like to immitate them i think',
            'its all nonsense ned cried chris for them to think that they\'re staying on account of us hello chris are you listening'
        ],
        index=pd.Index(['1255-74899-0020', '1651-136854-0030', '7601-101619-0003', '7601-175351-0021', '8254-84205-0005'], name='task'),
    )


@pytest.fixture
def simple_text_result_rasa():
    return pd.Series(
        [
            'i shouuld had seen you damp my word',
            'it must indeed be allowed that is structure and sentences is expanded and often has somewhat of the inversion of latin and that he delighted to express familiar thought in a philosophical language meaning this the works of socrates who it was said reduced philosophy to the simplicity of life ',  # noqa
            'nature discovers are confusing to us pain is holding the same emotions and grimaces of the face that serve for weeping serve for laughter too and indeed before the one and the other be finished do but observe the painter\'s manner of handling and you will be in doubt to which of the two the design tems and the extreme laghter does it last to bring tears',  # noqa
            'they is more than one amongs us who would like to immitate them i think',
            'its all nonsense ned cried chris for them to think that they\'re staying on account of us hello chris are you listening'
        ],
        index=pd.Index(['1255-74899-0020', '1651-136854-0030', '7601-101619-0003', '7601-175351-0021', '8254-84205-0005'], name='task'),
    )
