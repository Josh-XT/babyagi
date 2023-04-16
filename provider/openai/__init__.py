import openai

class OpenAIProvider:
    def __init__(self, api_key, model, temperature, max_tokens):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        openai.api_key = self.api_key
        if "gpt-4" in self.model.lower():
            print(
                "\033[91m\033[1m"
                + "\n*****USING GPT-4. POTENTIALLY EXPENSIVE. MONITOR YOUR COSTS*****"
                + "\033[0m\033[0m"
            )

    def instruct(self, prompt):
        if not self.model.startswith("gpt-"):
            # Use completion API
            response = openai.Completion.create(
                engine=self.model,
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
            return response.choices[0].text.strip()
        else:
            # Use chat completion API
            messages = [{"role": "system", "content": prompt}]
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                n=1,
                stop=None,
            )
            return response.choices[0].message.content.strip()
