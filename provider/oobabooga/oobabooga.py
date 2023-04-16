import requests

class OobaboogaProvider:
    def __init__(self, temperature, max_tokens):
        self.temperature = temperature
        self.max_tokens = max_tokens

    def instruct(self, prompt):
        response = requests.post("http://localhost:7860/run/textgen", json={
            "data": [
                [
                    prompt,
                    {
                        'max_new_tokens': self.max_tokens,
                        'do_sample': True,
                        'temperature': self.temperature,
                        'top_p': 0.73,
                        'typical_p': 1,
                        'repetition_penalty': 1.1,
                        'encoder_repetition_penalty': 1.0,
                        'top_k': 0,
                        'min_length': 0,
                        'no_repeat_ngram_size': 0,
                        'num_beams': 1,
                        'penalty_alpha': 0,
                        'length_penalty': 1,
                        'early_stopping': False,
                        'seed': -1,
                        'add_bos_token': True,
                        'custom_stopping_strings': [],
                        'truncation_length': 2048,
                        'ban_eos_token': False,
                    }
                ]
            ]
        }).json()
        data = response['data'][0]
        return data.replace("\\n", "\n").strip()
