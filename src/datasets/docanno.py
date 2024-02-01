from src.datasets.metaclass import MetaDataClass


class Docanno(MetaDataClass):

    labels = sorted([
        'inspiring', 'interesting', 'offensive_to_someone', 'negative',
        'offensive_to_me', 'political', 'positive', 'sadness', 'calm',
        'fear', 'compassion', 'disgust', 'vulgar', 'surprise', 'embarrasing',
        'anger', 'understandable', 'ironic', 'need_more_information',
        'happiness', 'delight', 'funny_to_someone', 'funny_to_me'
    ])

    columns = ['text_id', 'user_id', 'fold', 'text'] + labels

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
                            "the most appropriate label from the provided list. These labels represent "
                            "a range of emotions, attitudes, and perceptions that can be associated with "
                            "communication, content, or reactions to various situations. Analyze text carefully "
                            f"to make an accurate categorization. {response_format_instruction}\n\n"
                            f"### User ID:\n{example['user_id']}\n\n"
                            f"### Text:\n{example['text']}\n\n"
                            "### Labels:\n" + "\n- ".join(cls.labels) +
                            cls.response_template
                        )
                    case 1:
                        return (
                            "Knowing that a particular user has provided the response given below to the given example "
                            "categorize the following text for the specified user by selecting "
                            "the most appropriate label from the provided list. These labels represent "
                            "a range of emotions, attitudes, and perceptions that can be associated with "
                            "communication, content, or reactions to various situations. Analyze text carefully "
                            f"to make an accurate categorization. {response_format_instruction}\n\n"
                            f"### User ID:\n{example['user_id']}\n\n"
                            f"### Example:\n{example['example1']}\n\n"
                            f"### Example Response:\n{example['example1_response']}\n\n"
                            f"### Text:\n{example['text']}\n\n"
                            "### Labels:\n" + "\n- ".join(cls.labels) +
                            cls.response_template
                        )
                    case 2:
                        return (
                            "Knowing that a particular user has provided the responses given below to the given examples "
                            "categorize the following text for the specified user by selecting "
                            "the most appropriate label from the provided list. These labels represent "
                            "a range of emotions, attitudes, and perceptions that can be associated with "
                            "communication, content, or reactions to various situations. Analyze text carefully "
                            f"to make an accurate categorization. {response_format_instruction}\n\n"
                            f"### User ID:\n{example['user_id']}\n\n"
                            f"### Example 1:\n{example['example1']}\n\n"
                            f"### Example 1 Response:\n{example['example1_response']}\n\n"
                            f"### Example 2:\n{example['example2']}\n\n"
                            f"### Example 2 Response:\n{example['example2_response']}\n\n"
                            f"### Text:\n{example['text']}\n\n"
                            "### Labels:\n" + "\n- ".join(cls.labels) +
                            cls.response_template
                        )
            else:
                match n_shot:
                    case 0:
                        return (
                            "Categorize the following text by selecting the most appropriate "
                            "label from the provided list. These labels represent a range of "
                            "emotions, attitudes, and perceptions that can be associated with "
                            "communication, content, or reactions to various situations. Analyze "
                            f"text carefully to make an accurate categorization. {response_format_instruction}\n\n"
                            f"### Text:\n{example['text']}\n\n"
                            "### Labels:\n" + "\n- ".join(cls.labels) +
                            cls.response_template
                        )
                    case 1:
                        return (
                            "Knowing that for the given example was provided the response given below "
                            "categorize the following text by selecting the most appropriate "
                            "label from the provided list. These labels represent a range of "
                            "emotions, attitudes, and perceptions that can be associated with "
                            "communication, content, or reactions to various situations. Analyze "
                            f"text carefully to make an accurate categorization. {response_format_instruction}\n\n"
                            f"### Example:\n{example['example1']}\n\n"
                            f"### Example Response:\n{example['example1_response']}\n\n"
                            f"### Text:\n{example['text']}\n\n"
                            "### Labels:\n" + "\n- ".join(cls.labels) +
                            cls.response_template
                        )
                    case 2:
                        return (
                            "Knowing that for the given examples were provided the responses given below "
                            "categorize the following text by selecting the most appropriate "
                            "label from the provided list. These labels represent a range of "
                            "emotions, attitudes, and perceptions that can be associated with "
                            "communication, content, or reactions to various situations. Analyze "
                            f"text carefully to make an accurate categorization. {response_format_instruction}\n\n"
                            f"### Example 1:\n{example['example1']}\n\n"
                            f"### Example 1 Response:\n{example['example1_response']}\n\n"
                            f"### Example 2:\n{example['example2']}\n\n"
                            f"### Example 2 Response:\n{example['example2_response']}\n\n"
                            f"### Text:\n{example['text']}\n\n"
                            "### Labels:\n" + "\n- ".join(cls.labels) +
                            cls.response_template
                        )
        else:
            if personalized:
                return (
                    f"{example['user_id']}\n{example['text']}"
                )
            else:
                return example['text']