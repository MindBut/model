"""Contains classes for querying large language models"""

from math import ceil 
import os
import time 
from tqdm import tqdm

import openai

gpt_costs_per_thousand = {
    'gpt-3.5-turbo-1106': 0.0100,
    'gpt-3.5-turbo': 0.0100,
    'gpt-3.5-turbo-16k': 0.0005,
    'gpt-3.5-turbo-instruct': 0.0004
}

def model_from_config(config, disable_tqdm=True):
    """Returns a model based on the config"""
    model_type = config["name"]
    if model_type == 'GPT':
        return GPT(config, disable_tqdm=disable_tqdm)
    else: 
        raise ValueError(f"Unknown model type: {model_type}")

class GPT():
    
    def __init__(self, config, needs_confirmation=False, disable_tqdm=True):
        """Initializes the model"""
        self.config = config
        self.needs_confirmation = needs_confirmation
        self.disable_tqdm = disable_tqdm
    
    def confirm_cost(self, texts, n, max_tokens):
        """Confirm the maximum costs of using API

        Args:
            texts (list): list of prompts
            n (int): rounds of generation (number of queries)
            max_tokens (int): maximum number of tokens from output
        """
        total_estimated_cost = 0
        for text in texts:
            total_estimated_cost += gpt_get_estimated_cost(
                self.config, text, max_tokens) * n 
        print(f"Estimated cost: ${total_estimated_cost:.2f}")
        # Ask the user to confirm in the command line 
        if os.getenv("LLM_SKIP_CONFIRM") is None:
            confirm = input("Continue? (y/n) ")
            if confirm != 'y':
                raise Exception("Aborted.")
        
    def complete(self, prompt, n):
        """Generates text from the model and returns the log prob data"""
        if not isinstance(prompt, list):
            prompt = [prompt]
            
        if self.needs_confirmation:
            self.confirm_cost(
                prompt, n, self.config['gpt_config']['max_tokens'])
        
        if not self.disable_tqdm:
            print(f"[{self.config['name']}] Generating {len(prompt) * n} completions, ")
        
        res = []
        for p in tqdm(prompt, disable=self.disable_tqdm):
            res += self.__complete(p, n)
        return res

    def log_probs(self, text):
        """Returns the log probs of the text"""
        if not isinstance(text, list):
            text = [text]
        if self.needs_confirmation:
            self.confirm_cost(text, 1, 0)
        if not self.disable_tqdm:
            print(f"[{self.config['name']}] Getting log probs for {len(text)} strings")
        log_probs = []
        tokens = []
        for text in tqdm(text, disable=self.disable_tqdm):
            log_probs, tokens = self.__log_probs(text)
        return log_probs, tokens

    def __complete(self, prompt, n):
        """Generates text from the model."""
        if not isinstance(prompt, list):
            text = [prompt]
        config = self.config['gpt_config'].copy()
        config['n'] = n
        response = None
        while response is None:
            try:
                response = openai.chat.completions.create(
                    **config, prompt=prompt)
            except Exception as e:
                print(e)
                print('Retrying...')
                time.sleep(5)
        
        return [response['choices'][i]['text'] for i in range(len(response['choices']))]
    
    def __log_probs(self, text):
        """Returns the log probs of the text."""
        if not isinstance(text, list):
            text = [text]
        config = self.config['gpt_config'].copy()
        config['logprobs'] = 1
        config['echo'] = True
        config['max_tokens'] = 0
        if isinstance(text, list):
            text = [f'\n{text[i]}' for i in range(len(text))]
        else:
            text = f'\n{text}'
        response = None
        while response is None:
            try:
                response = openai.Completion.create(
                    **config, prompt=text)
            except Exception as e:
                print(e)
                print('Retrying...')
                time.sleep(5)
        log_probs = [response['choices'][i]['logprobs']['token_logprobs'][1:]
                     for i in range(len(response['choices']))]
        tokens = [response['choices'][i]['logprobs']['tokens'][1:]
                  for i in range(len(response['choices']))]

        return log_probs, tokens

def gpt_get_estimated_cost(config, prompt, max_tokens):
    """Uses the current API costs/1000 tokens to estimate the cost of the generating text from the model."""
    # Get the number of tokens in the prompt 
    n_prompt_tokens = len(prompt) 
    # Get the number of tokens in the generated text 
    total_tokens = n_prompt_tokens + max_tokens
    engine = config['gpt_config']['model']
    costs_per_thousand = gpt_costs_per_thousand
    if engine not in costs_per_thousand:
        print(f"The cost of the {engine} can't be calculated")
    price = costs_per_thousand[engine] * total_tokens / 1000
    return price
