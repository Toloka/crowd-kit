import pytest
import random

import pandas as pd
import numpy as np


@pytest.fixture
def not_random() -> None:
    random.seed(42)
    np.random.seed(42)


# toy YSDA dataset

@pytest.fixture
def toy_answers_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ['w1', 't1', 'no'],
            ['w1', 't2', 'yes'],
            ['w1', 't3', 'yes'],
            ['w1', 't4', 'yes'],
            ['w1', 't5', 'yes'],
            ['w2', 't1', 'yes'],
            ['w2', 't2', 'yes'],
            ['w2', 't3', 'no'],
            ['w2', 't4', 'yes'],
            ['w2', 't5', 'no'],
            ['w3', 't2', 'yes'],
            ['w3', 't3', 'no'],
            ['w3', 't4', 'yes'],
            ['w3', 't5', 'no'],
            ['w4', 't1', 'yes'],
            ['w4', 't2', 'no'],
            ['w4', 't3', 'yes'],
            ['w4', 't4', 'yes'],
            ['w4', 't5', 'no'],
            ['w5', 't1', 'no'],
            ['w5', 't2', 'no'],
            ['w5', 't3', 'no'],
            ['w5', 't4', 'yes'],
            ['w5', 't5', 'no'],
        ],
        columns=['worker', 'task', 'label']
    )


@pytest.fixture
def toy_ground_truth_df() -> pd.Series:
    return pd.Series(
        ['yes', 'yes', 'no', 'yes', 'no'],
        pd.Index(['t1', 't2', 't3', 't4', 't5'], name='task'),
        name='agg_label',
    )


@pytest.fixture
def toy_gold_df() -> pd.Series:
    return pd.Series({
        't1': 'yes',
        't2': 'yes',
    })


# Simple dataset that imitates real toloka answers

@pytest.fixture
def simple_answers_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # good workers
            ['bde3b214b06c1efa6cb1bc6284dc72d2', '1231239876--5fac0d234ffb2f3b00893eec', 'goose'],
            ['bde3b214b06c1efa6cb1bc6284dc72d2', '1231239876--5fac0d234ffb2f3b00893efb', 'parrot'],
            ['bde3b214b06c1efa6cb1bc6284dc72d2', '1231239876--5fac0d234ffb2f3b00893f03', 'goose'],
            ['bde3b214b06c1efa6cb1bc6284dc72d2', '1231239876--5fac0d234ffb2f3b00893f05', 'goose'],
            ['bde3b214b06c1efa6cb1bc6284dc72d2', '1231239876--5fac0d234ffb2f3b00893f02', 'parrot'],
            ['bde3b214b06c1efa6cb1bc6284dc72d2', '1231239876--5fac0d234ffb2f3b00893f08', 'parrot'],
            ['b17c3301ad2ccbb798716fdd405d16e8', '1231239876--5fac0d234ffb2f3b00893efb', 'parrot'],
            ['b17c3301ad2ccbb798716fdd405d16e8', '1231239876--5fac0d234ffb2f3b00893ee8', 'chicken'],
            ['b17c3301ad2ccbb798716fdd405d16e8', '1231239876--5fac0d234ffb2f3b00893f07', 'parrot'],
            ['b17c3301ad2ccbb798716fdd405d16e8', '1231239876--5fac0d234ffb2f3b00893efd', 'chicken'],
            ['b17c3301ad2ccbb798716fdd405d16e8', '1231239876--5fac0d234ffb2f3b00893ee4', 'chicken'],
            ['b17c3301ad2ccbb798716fdd405d16e8', '1231239876--5fac0d234ffb2f3b00893f03', 'goose'],
            ['a452e450f913cfa987cad58d50393718', '1231239876--5fac0d234ffb2f3b00893ee8', 'chicken'],
            ['a452e450f913cfa987cad58d50393718', '1231239876--5fac0d234ffb2f3b00893eec', 'goose'],
            ['a452e450f913cfa987cad58d50393718', '1231239876--5fac0d234ffb2f3b00893f05', 'goose'],
            ['a452e450f913cfa987cad58d50393718', '1231239876--5fac0d234ffb2f3b00893f02', 'parrot'],
            ['a452e450f913cfa987cad58d50393718', '1231239876--5fac0d234ffb2f3b00893f08', 'parrot'],
            ['0f65edea0a6dc7b9acba1dea313bbb3d', '1231239876--5fac0d234ffb2f3b00893eec', 'goose'],
            ['0f65edea0a6dc7b9acba1dea313bbb3d', '1231239876--5fac0d234ffb2f3b00893ee8', 'chicken'],
            ['0f65edea0a6dc7b9acba1dea313bbb3d', '1231239876--5fac0d234ffb2f3b00893f03', 'goose'],
            ['0f65edea0a6dc7b9acba1dea313bbb3d', '1231239876--5fac0d234ffb2f3b00893ee4', 'chicken'],
            # fraudster - always answers "parrot"
            ['0c3eb7d5fcc414db137c4180a654c06e', '1231239876--5fac0d234ffb2f3b00893eec', 'parrot'],  # 'goose'
            ['0c3eb7d5fcc414db137c4180a654c06e', '1231239876--5fac0d234ffb2f3b00893efb', 'parrot'],
            ['0c3eb7d5fcc414db137c4180a654c06e', '1231239876--5fac0d234ffb2f3b00893f07', 'parrot'],
            ['0c3eb7d5fcc414db137c4180a654c06e', '1231239876--5fac0d234ffb2f3b00893efd', 'parrot'],  # 'chicken'
            ['0c3eb7d5fcc414db137c4180a654c06e', '1231239876--5fac0d234ffb2f3b00893ee4', 'parrot'],  # 'chicken'
            ['0c3eb7d5fcc414db137c4180a654c06e', '1231239876--5fac0d234ffb2f3b00893f05', 'parrot'],  # 'goose'
            # careless
            ['e563e2fb32fce9f00123a65a1bc78c55', '1231239876--5fac0d234ffb2f3b00893efb', 'parrot'],
            ['e563e2fb32fce9f00123a65a1bc78c55', '1231239876--5fac0d234ffb2f3b00893ee8', 'goose'],  # 'chicken'
            ['e563e2fb32fce9f00123a65a1bc78c55', '1231239876--5fac0d234ffb2f3b00893f02', 'parrot'],
            ['e563e2fb32fce9f00123a65a1bc78c55', '1231239876--5fac0d234ffb2f3b00893f08', 'parrot'],
            ['e563e2fb32fce9f00123a65a1bc78c55', '1231239876--5fac0d234ffb2f3b00893f07', 'parrot'],
            ['e563e2fb32fce9f00123a65a1bc78c55', '1231239876--5fac0d234ffb2f3b00893efd', 'goose'],  # 'chicken'
        ],
        columns=['worker', 'task', 'label']
    )


@pytest.fixture
def simple_ground_truth() -> pd.Series:
    ground_truth = pd.Series({
        '1231239876--5fac0d234ffb2f3b00893eec': 'goose',
        '1231239876--5fac0d234ffb2f3b00893f03': 'goose',
        '1231239876--5fac0d234ffb2f3b00893f05': 'goose',
        '1231239876--5fac0d234ffb2f3b00893efb': 'parrot',
        '1231239876--5fac0d234ffb2f3b00893f02': 'parrot',
        '1231239876--5fac0d234ffb2f3b00893f08': 'parrot',
        '1231239876--5fac0d234ffb2f3b00893f07': 'parrot',
        '1231239876--5fac0d234ffb2f3b00893ee8': 'chicken',
        '1231239876--5fac0d234ffb2f3b00893efd': 'chicken',
        '1231239876--5fac0d234ffb2f3b00893ee4': 'chicken',
    }, name='agg_label')
    ground_truth.index.name = 'task'
    return ground_truth


@pytest.fixture
def simple_gold_df() -> pd.Series:
    true_labels = pd.Series({
        '1231239876--5fac0d234ffb2f3b00893eec': 'goose',
        '1231239876--5fac0d234ffb2f3b00893efb': 'parrot',
        '1231239876--5fac0d234ffb2f3b00893ee8': 'chicken',
    })
    true_labels.index.name = 'task'
    return true_labels


@pytest.fixture
def simple_text_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ['1255-74899-0020', 'as soon as you downed my worst in stockings sweetheart', 'b6214dff3665ba9c6bc96dc326a417c9', 'i\'d sooner see you darning my worsted stockings sweetheart', np.array([1.8628045, -0.66789037]), np.array([0.8619265, 0.3983395])],  # noqa
            ['1651-136854-0030', 'it must indeed be allowed that is structure and sentences is expanded and often has somewhat of the inversion of latin and that he delighted to express familiar thought in a philosophical language meaning this the works of socrates who it was said reduced philosophy to the simplicity of life ', 'c740c713b07635302cf145d16ae2d698', np.nan, np.array([1.5327121, 2.5106835]), np.nan],  # noqa
            ['7601-175351-0021', 'there is more than one amongst us who would like to and imitate them i think', 'c854a0b6d71ec3503e0ce4ea2179d8c7', 'there\'s more than one amongst us who would like to imitate them i think', np.array([10.217721, 16.948954]), np.array([10.686009, 17.633106])],  # noqa
            ['1255-74899-0020', 'i said i was just talking sweetheart', '19fdfe8fe1f7dd366f594bf2ce0bdd3e', 'i\'d sooner see you darning my worsted stockings sweetheart', np.array([1.9092205, -0.5734551]), np.array([0.8508962, 0.38230664])],  # noqa
            ['1651-136854-0030', 'it must indeed be allowed that the structure of your sentences is expanded and often has somewhat of the inversion of latin and that he delighted to express familiar thoughts in a philosophical language being in the the reversal of socrates who it was said reduced philosophy to the simplicity of common life', 'c854a0b6d71ec3503e0ce4ea2179d8c7', np.nan, np.array([1.523252, 2.5053673]), np.nan],  # noqa
            ['1651-136854-0030', 'it must indeed be allowed that the structure iss expanded and often has somewhat of the invention of latin in that he delighted to express familiar thoughts in philosophical language being in this the reverse of socrates whom it was said reduced phylosophy to the simplicity of common life', '2ef99a0a7639b5fcd7e66e59e7b7e3bf', np.nan, np.array([1.5202638, 2.481906]), np.nan],  # noqa
            ['7601-175351-0021', 'there is more than one among us who like to eliminate them i think', 'ab2784b4377e0848ebff96098fb67301', 'there\'s more than one amongst us who would like to imitate them i think', np.array([10.222795, 16.946047]), np.array([10.673217, 17.622795])],  # noqa
            ['7601-175351-0021', 'there\'s more than one amongst us who would like doing imitate them i think', '07bda6ebab4a387f8ced48c40c5878a6', 'there\'s more than one amongst us who would like to imitate them i think', np.array([10.197447, 16.96481]), np.array([10.673589, 17.622868])],  # noqa
            ['1651-136854-0030', 'it must indeed be allowed that the structurally sentences is expanded and often has somewhat be inversion of latin and that he delighted to express familiar thought in a philosophical language being in this the reverse of socrates who it was said reduce philosophy to the simplicity of common life', 'efcfbfa835fdcec9a865f82bbf6d36df', np.nan, np.array([1.5325097, 2.5046723]), np.nan],  # noqa
            ['7601-175351-0021', 'there\'s more than one amongst us who would like to imitate them i think', 'efcfbfa835fdcec9a865f82bbf6d36df', 'there\'s more than one amongst us who would like to imitate them i think', np.array([10.219084, 16.948515]), np.array([10.673225, 17.622633])],  # noqa
            ['7601-175351-0021', 'there\'s more than one amongst us who would like imitate them i think', '27dfec580d349e20166b56be480336ea', 'there\'s more than one amongst us who would like to imitate them i think', np.array([10.217423, 16.949697]), np.array([10.673151, 17.622568])],  # noqa
            ['1255-74899-0020', 'as soon as you darned my worst is talking sweetheart', '4330bead5a86328a3d6987b3c065dc80', 'i\'d sooner see you darning my worsted stockings sweetheart', np.array([1.9123898, -0.58190906]), np.array([0.8685327, 0.39246213])],  # noqa
            ['1255-74899-0020', 'as soon as you\'ve done my worthiest stocking sweetheart', '2c3954249d00e0aa0adab7226eba47e4', 'i\'d sooner see you darning my worsted stockings sweetheart', np.array([1.8627588, -0.661437]), np.array([0.8250761, 0.39154962])],  # noqa
            ['1255-74899-0020', 'i should have seen you donned my worst heel stocking sweetheart', 'cd5520854f89b31ae5f8673fa2992ac7', 'i\'d sooner see you darning my worsted stockings sweetheart', np.array([1.8909845, -0.63419354]), np.array([0.89531624, 0.3790261])],  # noqa
            ['1651-136854-0030', 'it must indeed be allowed that the structure of his sentences is expanded and often has somewhat of the inversion of latin and that he delighted to express familiar thoughts in a philosophical language being in this the reverse of socrates who it was said reduced philosophy to the simplicity of common life', '5701a8373b728dd333c0796df9f2a8f4', np.nan, np.array([1.5207397, 2.5072067]), np.nan],  # noqa
            ['7601-175351-0021', 'there\'s more than one amongst us who would like to imitate anything', '9e9847e525a1d0fdfaf22d83fa75d115', 'there\'s more than one amongst us who would like to imitate them i think', np.array([10.219814, 16.947468]), np.array([10.675565, 17.62496])],  # noqa
            ['1651-136854-0030', 'it must indeed be allowed that the structure of his sentences is expanded and often has somewhat of the inversion of latin and that he delighted to express familiar thoughts in philosophical language being in this the reverse of socrates who it was said reduced philosophy to the simplicity of common life ', 'cd5520854f89b31ae5f8673fa2992ac7', np.nan, np.array([1.51274, 2.5141635]), np.nan],  # noqa
            ['1651-136854-0030', 'it must indeed be allowed that the structure of his sentence is expanded and often has somewhat of the inversion of latin and that he delighted to express familiar thoughts and philosophical language being in this the reverse of socrates who it was said reduced philosophy to the simplicity of common life', '2047f49d9762c9db85f2240b06dd2d12', np.nan, np.array([1.5277034, 2.5007238]), np.nan],  # noqa
            ['1651-136854-0030', 'it must in need be allowed the structure id expenses it expend it and obten some water the inversion of letter and he the new year fox and the language vniversal soxes it was said phlosopharty of common life', '0598870b6ec9d30e31958f5b517b4336', np.nan, np.array([1.5254755, 2.4552798]), np.nan],  # noqa
            ['1651-136854-0030', 'it must indeed be a loud that the structurer he senses is expanded and often has some what of the inversion of latin and that he delighted to express the new year fault in a philosophical language being in this the reverse of sock verties who it was said reduced philosophy is the simplicity to common life', 'f733ff7874adaf5e83b5243e7aebc6ef', np.nan, np.array([1.5204769, 2.5086155]), np.nan],  # noqa
            ['7601-175351-0021', 'there is more than one amongst us who would like to an imitate they might think', 'f733ff7874adaf5e83b5243e7aebc6ef', 'there\'s more than one amongst us who would like to imitate them i think', np.array([10.218769, 16.948397]), np.array([10.676133, 17.628136])],  # noqa
            ['1255-74899-0020', 'i shouuld had seen you damp my word', 'bd46bdedc1a395765f7163152ae1d6e9', 'i\'d sooner see you darning my worsted stockings sweetheart', np.array([4.2875843, 0.24995197]), np.array([0.8510484, 0.39480436])],  # noqa
            ['1255-74899-0020', 'as soon as you donned my worst stocking sweetheart', '1d94612b35ca8f3b1567e170808afb9d', 'i\'d sooner see you darning my worsted stockings sweetheart', np.array([1.8590927, -0.6760239]), np.array([0.8439374, 0.39588147])],  # noqa
            ['7601-175351-0021', 'there\'s more than one amongst us who would like to imitate them i think', '1ea4a74105f43c96ab394e7f6495ef27', 'there\'s more than one amongst us who would like to imitate them i think', np.array([10.219354, 16.948345]), np.array([10.679564, 17.62985])],  # noqa
            ['1255-74899-0020', 'as soon as he dawned my worst is talking to sweetheart', 'fca852cd8dec559a31e57d7957a4de13', 'i\'d sooner see you darning my worsted stockings sweetheart', np.array([1.88419, -0.58892304]), np.array([0.8540088, 0.3765296])],  # noqa
            ['7601-175351-0021', 'there\'s more than one amongst us who would like to imitate my think', '194fd65c5c1246bed88b141320db8bcd', 'there\'s more than one amongst us who would like to imitate them i think', np.array([10.218983, 16.948296]), np.array([10.654044, 17.605904])],  # noqa
            ['7601-175351-0021', 'there is more than one amongst us who would like to immitate them i think', '1f30d866dc87d8aba36da6c94d578097', 'there\'s more than one amongst us who would like to imitate them i think', np.array([10.226313, 16.944336]), np.array([10.672564, 17.62215])],  # noqa
            ['7601-175351-0021', 'there\'s more than one amongst us who would like to wanna imitate them i think', '5701a8373b728dd333c0796df9f2a8f4', 'there\'s more than one amongst us who would like to imitate them i think', np.array([10.2202015, 16.948513]), np.array([10.692975, 17.64521])],  # noqa
            ['7601-175351-0021', 'there\'s more than one amongst us who would like to imitate them i think', 'fca852cd8dec559a31e57d7957a4de13', 'there\'s more than one amongst us who would like to imitate them i think', np.array([10.219054, 16.948479]), np.array([10.673226, 17.622896])],  # noqa
            ['7601-175351-0021', 'they is more than one amongs us who would like to immitate them i think', 'f84d186e73c19bba35b59c631e56d860', 'there\'s more than one amongst us who would like to imitate them i think', np.array([10.233541, 16.935406]), np.array([10.667211, 17.619024])],  # noqa
            ['7601-175351-0021', 'there is more than one amongst us who would like to imitate them i think', '86cdbbd441956f774c1f09ca4e47dfeb', 'there\'s more than one amongst us who would like to imitate them i think', np.array([10.219196, 16.948393]), np.array([10.674336, 17.623909])],  # noqa
            ['1255-74899-0020', 'as soon as you dawned my worst in stalking the sweetheart', 'c854a0b6d71ec3503e0ce4ea2179d8c7', 'i\'d sooner see you darning my worsted stockings sweetheart', np.array([1.8683596, -0.60298634]), np.array([0.86584425, 0.43221])],  # noqa
            ['1255-74899-0020', 'as soon as you dawned my worst in stocking sweetheart', 'f84d186e73c19bba35b59c631e56d860', 'i\'d sooner see you darning my worsted stockings sweetheart', np.array([1.8571255, -0.6310346]), np.array([0.8503618, 0.37850586])],  # noqa
            ['7601-175351-0021', 'there is more than one amongst us who would like to imitate him i think', '2a2b550bdf723c06898ce62185f485a7', 'there\'s more than one amongst us who would like to imitate them i think', np.array([10.227572, 16.945225]), np.array([10.672767, 17.622879])],  # noqa
            ['7601-175351-0021', 'there\'s more than one amongst us who would like to immitate them i think', '555686599071ea2f3012cd64a381cf60', 'there\'s more than one amongst us who would like to imitate them i think', np.array([10.223191, 16.945898]), np.array([10.673242, 17.623188])],  # noqa
            ['1255-74899-0020', 'i should have seen you turned my worsted stocking sweetheart', 'd91be229ea4909b060e99609ea5b4f66', 'i\'d sooner see you darning my worsted stockings sweetheart', np.array([1.9095235, -0.62991136]), np.array([0.7952585, 0.45487818])],  # noqa
            ['1255-74899-0020', 'i should have seen tart my worsest tlking sweetheart', '4410d66e6c13650a0478455ad015f118', 'i\'d sooner see you darning my worsted stockings sweetheart', np.array([1.906517, -0.6025044]), np.array([0.84691495, 0.39889106])],  # noqa
            ['7601-175351-0021', 'there is more than one amongst us who would like to imitate them i think', 'd91be229ea4909b060e99609ea5b4f66', 'there\'s more than one amongst us who would like to imitate them i think', np.array([10.212206, 16.952347]), np.array([10.676438, 17.627277])],  # noqa
            ['7601-175351-0021', 'there is more than one amongst us who would like to an imitate they might think', '74b8938eb4736c40da2852ce2c5e5008', 'there\'s more than one amongst us who would like to imitate them i think', np.array([10.219014, 16.94834]), np.array([10.675005, 17.626295])],  # noqa
            ['1255-74899-0020', 'i should have dulled by what is locking some dark', 'c5cf3042be0413d5a2a8360ba344e258', 'i\'d sooner see you darning my worsted stockings sweetheart', np.array([6.163941, -0.6219455]), np.array([0.84807605, 0.4049313])],  # noqa
            ['8254-84205-0005', 'it\'s all nonsense nerd crowd chris for them to think that they\'re staying on account of us who low griggs were you listening', '2047f49d9762c9db85f2240b06dd2d12', np.nan, np.array([4.953112, -0.41251197]), np.nan],  # noqa
            ['7601-175351-0021', 'there is more than one amongst us who would like to do an imitate i think', '0af574140e706efce6b224417b1d8aac', 'there\'s more than one amongst us who would like to imitate them i think', np.array([10.201713, 16.964706]), np.array([10.67186, 17.621662])],  # noqa
            ['1255-74899-0020', 'as soon as he donned my worsted stockings sweetheart', '0af574140e706efce6b224417b1d8aac', 'i\'d sooner see you darning my worsted stockings sweetheart', np.array([1.8677676, -0.6548876]), np.array([0.813319, 0.32563943])],  # noqa
            ['8254-84205-0005', 'it\'s all nonsense ned cried chris for them to think that they\'re staying on account of us hello griggs were you listening', 'fca852cd8dec559a31e57d7957a4de13', np.nan, np.array([4.95962, -0.4224869]), np.nan],  # noqa
            ['8254-84205-0005', 'its all nonsense ned cried chris for them to think that theyre staying on account of us hello griggs were you listening', '0af574140e706efce6b224417b1d8aac', np.nan, np.array([4.946168, -0.4152995]), np.nan],  # noqa
            ['8254-84205-0005', 'it\'s all nonsense ned cried chris for them to think that they\'re staying on account of us hello greg were you listening ', '525be6f4af2fb2c4781575b7c9fbaee0', np.nan, np.array([4.9694304, -0.42101935]), np.nan],  # noqa
            ['8254-84205-0005', 'its all nonsense ned cried chris for them to think that they are staying on account of us hello grigs  are you listening', 'd91be229ea4909b060e99609ea5b4f66', np.nan, np.array([4.968234, -0.42592508]), np.nan],  # noqa
            ['8254-84205-0005', 'it\'s all nonsense ned cried chris for them to think that they\'re staying on account of us hello kriegs were you listening', '043bf8732e747ca7c0a7edd6ae13182f', np.nan, np.array([4.967573, -0.43132243]), np.nan],  # noqa
            ['8254-84205-0005', 'its all nonsense ned cried chris for them to think that they\'re staying on account of us hello chris are you listening', '8656955f53e6d3cb9e56171be33ef2bc', np.nan, np.array([4.964371, -0.42836314]), np.nan],  # noqa
            ['7601-101619-0003', 'nature discovers this confusion to us painters hold that the same motions and grimaces of the face that serve for weeping serve for laughter too and indeed before the one or the other be finished do but observe the painter\'s manner of handling and you will be in doubt to which of the two the design tends and the extreme of laughter does at last bring tears', '043bf8732e747ca7c0a7edd6ae13182f', np.nan, np.array([0.791476, 2.0119832]), np.nan],  # noqa
            ['7601-101619-0003', 'nature discovers this confusion to us painters hold that the same motions and grimaces of the face that serve for weeping serve for laughter to and indeed before the one or the other be finished do but observe the painters manner of handling and you will be in doubt to which of the two the design tends and the extreme of laughter does it last bring tears', '0af574140e706efce6b224417b1d8aac', np.nan, np.array([0.81492954, 1.9998435]), np.nan],  # noqa
            ['7601-101619-0003', 'nature discovers is confusion to us painters hold it the same motions and grimaces of the face that serve for whipping serve for laughter too and indeed before the one or the other be finished do but observe the painters manner of handling and you will be in doubt to  of the two of designed tens and the extreme of laughter doesn\'t last bring tears', 'e973444b19802698a8c7c602be1add89', np.nan, np.array([0.78092235, 2.0215895]), np.nan],  # noqa
            ['7601-101619-0003', 'nature discovers are confusing to us pain is holding the same emotions and grimaces of the face that serve for weeping serve for laughter too and indeed before the one and the other be finished do but observe the painter\'s manner of handling and you will be in doubt to which of the two the design tems and the extreme laghter does it last to bring tears', 'c9825afd938fdc509c48f0135af78e4a', np.nan, np.array([0.7876146, 2.0195444]), np.nan],  # noqa
            ['7601-101619-0003', 'emotions and promises of the case that serve for weeping serve for laughter', '07bda6ebab4a387f8ced48c40c5878a6', np.nan, np.array([0.79881185, 1.9938986]), np.nan],  # noqa
            ['7601-101619-0003', 'natures discovers as confusion to us painters hold that the same motions and grimaces of the face that serve for weeping serve for laughter to and indeed before the one or the other be finished do but observe the painters manner of handling and you will be in doubt to which of the two the design tends and the extreme of laughter does it last bring tears', '27dfec580d349e20166b56be480336ea', np.nan, np.array([0.79025686, 2.0123045]), np.nan],  # noqa
            ['7601-101619-0003', 'nature discovers his confusion to us painters hold the same emotions and grimaces of the face that serve for gripping serve for laughter too and indeed before the one or the other be finished do but observe the painter\'s manner of handling and you will be in doubt to which of the two the design tends and the extreme of laughter doesn\'t last to bring tears', 'caf701c07a3374bdc98ae6bf230d4d56', np.nan, np.array([0.77356094, 2.0207922]), np.nan],  # noqa
        ],
        columns=['task', 'output', 'worker', 'golden', 'embedding', 'golden_embedding']
    )


@pytest.fixture
def simple_text_true_embeddings() -> pd.Series:
    return pd.Series(
        [np.array([0.8619265, 0.3983395]), np.array([10.686009, 17.633106])],
        index=pd.Index(['1255-74899-0020', '7601-175351-0021'], name='task')
    )


@pytest.fixture
def data_with_bool_labels() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ['w1', 't1', True],
            ['w1', 't2', True],
            ['w2', 't1', True],
            ['w2', 't2', True],
            ['w3', 't1', False],
            ['w3', 't2', False],
        ],
        columns=['worker', 'task', 'label']
    )


@pytest.fixture
def bool_labels_ground_truth() -> pd.Series:
    return pd.Series(
        [True, True],
        index=pd.Index(['t1', 't2'], name='task'),
        name='agg_label'
    )
