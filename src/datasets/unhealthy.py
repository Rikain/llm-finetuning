from src.datasets.metaclass import MetaDataClass


class Unhealthy(MetaDataClass):

    labels = sorted([
        "antagonize", "condescending", "dismissive", "generalisation",
        "generalisation_unfair", "healthy", "hostile", "sarcastic"
    ])

    columns = ["_unit_id", "comment", "_trust", "_worker_id"] + labels

    def __init__(self):
        super(MetaDataClass, self).__init__()

    @classmethod
    def generate_prompt(cls, example: dict, personalized: bool, instruct: bool, generative: bool, n_shot: int) -> str:
        if generative:
            response_format_instruction = "Please compose your response as a list of chosen emotions, separated by commas."
        else:
            response_format_instruction = ""
        if instruct:
            if personalized:
                match n_shot:
                    case 0:
                        return (
                            "Categorize the following text for the specified user by selecting "
                            "the most appropriate label from the provided list. Labels represent "
                            "different types of communication styles or tones, where each category denotes "
                            "a specific attitude or approach that someone might exhibit when communicating "
                            "with others. Analyze text carefully to make an accurate "
                            f"categorization. {response_format_instruction}\n\n"
                            f"### User ID:\n{example['_worker_id']}\n\n"
                            f"### Text:\n{example['comment']}\n\n"
                            "### Labels:\n" + "\n- ".join(cls.labels) +
                            cls.response_template
                        )
                    case 1:
                        return (
                            "Knowing that for the given examples were provided the responses given below "
                            "categorize the following text for the specified user by selecting "
                            "the most appropriate label from the provided list. Labels represent "
                            "different types of communication styles or tones, where each category denotes "
                            "a specific attitude or approach that someone might exhibit when communicating "
                            "with others. Analyze text carefully to make an accurate "
                            f"categorization. {response_format_instruction}\n\n"
                            f"### User ID:\n{example['_worker_id']}\n\n"
                            f"### Example:\n{example['example1']}\n\n"
                            f"### Example Response:\n{example['example1_response']}\n\n"
                            f"### Text:\n{example['comment']}\n\n"
                            "### Labels:\n" + "\n- ".join(cls.labels) +
                            cls.response_template
                        )
                    case 2:
                        return (
                            "Knowing that a particular user has provided the responses given below to the given examples "
                            "categorize the following text for the specified user by selecting "
                            "the most appropriate label from the provided list. Labels represent "
                            "different types of communication styles or tones, where each category denotes "
                            "a specific attitude or approach that someone might exhibit when communicating "
                            "with others. Analyze text carefully to make an accurate "
                            f"categorization. {response_format_instruction}\n\n"
                            f"### User ID:\n{example['_worker_id']}\n\n"
                            f"### Example 1:\n{example['example1']}\n\n"
                            f"### Example 1 Response:\n{example['example1_response']}\n\n"
                            f"### Example 2:\n{example['example2']}\n\n"
                            f"### Example 2 Response:\n{example['example2_response']}\n\n"
                            f"### Text:\n{example['comment']}\n\n"
                            "### Labels:\n" + "\n- ".join(cls.labels) +
                            cls.response_template
                        )
            else:
                match n_shot:
                    case 0:
                        return (
                            "Categorize the following text by selecting the most appropriate label "
                            "from the provided list. Labels represent different types of communication "
                            "styles or tones, where each category denotes a specific attitude or approach "
                            "that someone might exhibit when communicating with others. Analyze text carefully "
                            f"to make an accurate categorization. {response_format_instruction}\n\n"
                            f"### Text:\n{example['comment']}\n\n"
                            "### Labels:\n" + "\n- ".join(cls.labels) +
                            cls.response_template
                        )
                    case 1:
                        return (
                            "Knowing that for the given example was provided the response given below "
                            "categorize the following text by selecting the most appropriate label "
                            "from the provided list. Labels represent different types of communication "
                            "styles or tones, where each category denotes a specific attitude or approach "
                            "that someone might exhibit when communicating with others. Analyze text carefully "
                            f"to make an accurate categorization. {response_format_instruction}\n\n"
                            f"### Example:\n{example['example1']}\n\n"
                            f"### Example Response:\n{example['example1_response']}\n\n"
                            f"### Text:\n{example['comment']}\n\n"
                            "### Labels:\n" + "\n- ".join(cls.labels) +
                            cls.response_template
                        )
                    case 2:
                        return (
                            "Knowing that for the given examples were provided the responses given below "
                            "categorize the following text by selecting the most appropriate label "
                            "from the provided list. Labels represent different types of communication "
                            "styles or tones, where each category denotes a specific attitude or approach "
                            "that someone might exhibit when communicating with others. Analyze text carefully "
                            f"to make an accurate categorization. {response_format_instruction}\n\n"
                            f"### Example 1:\n{example['example1']}\n\n"
                            f"### Example 1 Response:\n{example['example1_response']}\n\n"
                            f"### Example 2:\n{example['example2']}\n\n"
                            f"### Example 2 Response:\n{example['example2_response']}\n\n"
                            f"### Text:\n{example['comment']}\n\n"
                            "### Labels:\n" + "\n- ".join(cls.labels) +
                            cls.response_template
                        )
        else:
            if personalized:
                return (
                    f"{example['_worker_id']}\n{example['comment']}"
                )
            else:
                return example['comment']