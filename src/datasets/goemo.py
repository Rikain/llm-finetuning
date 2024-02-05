from src.datasets.metaclass import MetaDataClass

class GoEmo(MetaDataClass):

    labels = sorted([
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
        'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
        'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
        'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
        'remorse', 'sadness', 'surprise', 'neutral'
    ])
    columns = ["rater_id", "text"] + labels

    def __init__(self):
        super(MetaDataClass, self).__init__()

    @classmethod
    def generate_prompt(cls, example: dict, personalized: bool, instruct: bool, generative: bool, n_shot: int = 0) -> str:
        """Generates a standardized message to prompt the model with an instruction, optional input and a
        'response' field."""

        if generative:
            response_format_instruction = "Please compose your response as a list of chosen emotions, separated by commas."
        else:
            response_format_instruction = ""
        if instruct:
            if personalized:
                match n_shot:
                    case 0:
                        return (
                            "Categorize the following text for the specified user by selecting the "
                            "most appropriate emotion from the provided list. Emotions can be subtle "
                            "or overt, so analyze the text carefully to make an accurate "
                            f"classification. {response_format_instruction}\n\n"
                            f"### User ID:\n{example['rater_id']}\n\n"
                            f"### Text:\n{example['text']}\n\n"
                            "### Emotions:\n-" + "\n- ".join(cls.labels) +
                            cls.response_template
                        )
                    case 1:
                        return (
                            "Knowing that a particular user has provided the response given below to the given example "
                            "categorize the following text for the specified user by selecting the "
                            "most appropriate emotion from the provided list. Emotions can be subtle "
                            "or overt, so analyze the text carefully to make an accurate "
                            f"classification. {response_format_instruction}\n\n"
                            f"### User ID:\n{example['rater_id']}\n\n"
                            f"### Example:\n{example['example1']}\n\n"
                            f"### Example Response:\n{example['example1_response']}\n\n"
                            f"### Text:\n{example['text']}\n\n"
                            "### Emotions:\n-" + "\n- ".join(cls.labels) +
                            cls.response_template
                        )
                    case 2:
                        return (
                            "Knowing that a particular user has provided the responses given below to the given examples "
                            "categorize the following text for the specified user by selecting the "
                            "most appropriate emotion from the provided list. Emotions can be subtle "
                            "or overt, so analyze the text carefully to make an accurate "
                            f"classification. {response_format_instruction}\n\n"
                            f"### User ID:\n{example['rater_id']}\n\n"
                            f"### Example 1:\n{example['example1']}\n\n"
                            f"### Example 1 Response:\n{example['example1_response']}\n\n"
                            f"### Example 2:\n{example['example2']}\n\n"
                            f"### Example 2 Response:\n{example['example2_response']}\n\n"
                            f"### Text:\n{example['text']}\n\n"
                            "### Emotions:\n-" + "\n- ".join(cls.labels) +
                            cls.response_template
                        )
                        
            else:
                match n_shot:
                    case 0:
                        return (
                            "Categorize the following text by selecting the most appropriate emotion "
                            "from the provided list. Emotions can be subtle or overt, so analyze the "
                            f"text carefully to make an accurate classification. {response_format_instruction}\n\n"
                            f"### Text:\n{example['text']}\n\n"
                            "### Emotions:\n-" + "\n- ".join(cls.labels) +
                            cls.response_template
                        )
                    case 1:
                        return (
                            "Knowing that for the given example was provided the response given below "
                            "categorize the following text by selecting the most appropriate emotion "
                            "from the provided list. Emotions can be subtle or overt, so analyze the "
                            f"text carefully to make an accurate classification. {response_format_instruction}\n\n"
                            f"### Example:\n{example['example1']}\n\n"
                            f"### Example Response:\n{example['example1_response']}\n\n"
                            f"### Text:\n{example['text']}\n\n"
                            "### Emotions:\n-" + "\n- ".join(cls.labels) +
                            cls.response_template
                        )
                    case 2:
                        return (
                            "Knowing that for the given examples were provided the responses given below "
                            "categorize the following text by selecting the most appropriate emotion "
                            "from the provided list. Emotions can be subtle or overt, so analyze the "
                            f"text carefully to make an accurate classification. {response_format_instruction}\n\n"
                            f"### Example 1:\n{example['example1']}\n\n"
                            f"### Example 1 Response:\n{example['example1_response']}\n\n"
                            f"### Example 2:\n{example['example2']}\n\n"
                            f"### Example 2 Response:\n{example['example2_response']}\n\n"
                            f"### Text:\n{example['text']}\n\n"
                            "### Emotions:\n-" + "\n- ".join(cls.labels) +
                            cls.response_template
                        )
        else:
            if personalized:
                return (
                    f"{example['rater_id']}\n{example['text']}"
                )
            else:
                return example['text']