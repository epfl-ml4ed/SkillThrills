from feedback_prompt_template import PROMPTS
from openai.error import (
    RateLimitError,
    ServiceUnavailableError,
    APIError,
    APIConnectionError,
    Timeout,
    InvalidRequestError,
)
from api_key import API_KEY
from generator import MODELS
import openai
import time
from typing import List, Dict


PATTERN = r"@@(.*?)##"

class SpanExtractor():
    
    def __init__(self,
                 prompt_template: Dict[str, str],
                 model: str="gpt-3.5"):
        openai.api_key = API_KEY
        self.model = model
        self.prompt_template = prompt_template

    def create_prompt_for(self, sample):

        (system_prompt, instruction_field, shots) = list(self.prompt_template.values())
        messages = [
            {
                'role': "system",
                "content":system_prompt
            },
            {
                "role": "user",
                "content": instruction_field
            }
        ]
        for shot in shots:
            sentence, skill, answer,_ = shot.split("\n")
            messages.append({'role':'user', 'content':sentence + "\n" + skill})
            messages.append({'role':'assistant', 'content': answer})

        messages.append({'role': 'user', 'content': "sentence: " + str(sample["sentence"]) + "\nskill : " + str(sample["skill"])})

        return messages


    def query(self, 
              messages:List[Dict[str, str]],
              model: str="gpt-4"):
        
        #######
        # print("-"*100)
        # for message in messages:
        #     print(message["content"])
        #######

        try:
            response = openai.ChatCompletion.create(
                model=model, messages=messages, request_timeout=20
            )
            return response["choices"][0]["message"]["content"]
        except (
            RateLimitError,
            ServiceUnavailableError,
            APIError,
            Timeout,
        ) as e:  # Exception
            print(f"Timed out {e}. Waiting for 10 seconds.")
            time.sleep(10)

    def extract_span(self, sample):
        messages = self.create_prompt_for(sample)
        return self.query(messages, MODELS[self.model])

        