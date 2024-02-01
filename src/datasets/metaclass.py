import abc


class MetaDataClass:

    __metaclass__ = abc.ABCMeta
    response_template = '\n\n### Response:'

    def __init__(self):
        pass

    @classmethod
    def generate_prompt(cls, example: dict, personalized: bool, instruct: bool, generative: bool, n_shot: int = 0) -> str:
        pass
