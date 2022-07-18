import pandas as pd
import numpy as np
import pytest


@pytest.fixture
def simple_text_result_hrrasa() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ['1255-74899-0020', 'i shouuld had seen you damp my word', np.array([4.2875843, 0.24995197])], # noqa
            ['1651-136854-0030', 'it must indeed be allowed that is structure and sentences is expanded and often has somewhat of the inversion of latin and that he delighted to express familiar thought in a philosophical language meaning this the works of socrates who it was said reduced philosophy to the simplicity of life ', np.array([1.5327121, 2.5106835])],  # noqa
            ['7601-101619-0003', 'nature discovers are confusing to us pain is holding the same emotions and grimaces of the face that serve for weeping serve for laughter too and indeed before the one and the other be finished do but observe the painter\'s manner of handling and you will be in doubt to which of the two the design tems and the extreme laghter does it last to bring tears',  np.array([0.7876146, 2.0195444])],  # noqa
            ['7601-175351-0021', 'they is more than one amongs us who would like to immitate them i think', np.array([10.233541, 16.935406])],  # noqa
            ['8254-84205-0005', 'its all nonsense ned cried chris for them to think that they\'re staying on account of us hello chris are you listening', np.array([4.964371, -0.42836314])],  # noqa
        ],
        columns=['task', 'output', 'embedding'],
    ).set_index('task')


@pytest.fixture
def simple_text_result_rasa() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ['1255-74899-0020', 'i shouuld had seen you damp my word', np.array([4.2875843, 0.24995197])], # noqa
            ['1651-136854-0030', 'it must indeed be allowed that is structure and sentences is expanded and often has somewhat of the inversion of latin and that he delighted to express familiar thought in a philosophical language meaning this the works of socrates who it was said reduced philosophy to the simplicity of life ', np.array([1.5327121, 2.5106835])],  # noqa
            ['7601-101619-0003', 'nature discovers are confusing to us pain is holding the same emotions and grimaces of the face that serve for weeping serve for laughter too and indeed before the one and the other be finished do but observe the painter\'s manner of handling and you will be in doubt to which of the two the design tems and the extreme laghter does it last to bring tears',  np.array([0.7876146, 2.0195444])],  # noqa
            ['7601-175351-0021', 'they is more than one amongs us who would like to immitate them i think', np.array([10.233541, 16.935406])],  # noqa
            ['8254-84205-0005', 'its all nonsense ned cried chris for them to think that they\'re staying on account of us hello chris are you listening', np.array([4.964371, -0.42836314])],  # noqa
        ],
        columns=['task', 'output', 'embedding'],
    ).set_index('task')
