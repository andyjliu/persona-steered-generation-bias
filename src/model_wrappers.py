
import openai
from openai import OpenAI
from vllm import LLM, SamplingParams

import numpy as np
from scipy.special import softmax
import tiktoken
from openlogprobs.models import OpenAIModel
from openlogprobs.extract import extract_logprobs, bisection_search

class vllm_wrapper():
    '''
    wrapper for huggingface models using vLLM.
    '''
    def __init__(self, model, max_tokens, temp, timeout = 1, stop_sequences = ['\n']):
        self.model = LLM(model)
        self.max_tokens = max_tokens
        self.stop_sequences = stop_sequences
        self.temp = temp
        self.timeout = timeout

    def generate(self, prompts):
        sampling_params = SamplingParams(temperature=self.temp, max_tokens=self.max_tokens, stop=self.stop_sequences)
        response = self.model.generate(prompts, sampling_params)
        response = [r.outputs[0].text for r in response]
        if len(prompts) == 1:
            return(response[0])
        else:
            return(response)

    def get_logprobs_of_continuations(self, prompt, continuations):
        # returns logprobs of each potential continuation
        # output: np array of logprobs in same order as continuations list
        # assumes that all continuations are one token long
        sampling_params = SamplingParams(temperature=self.temp, max_tokens=self.max_tokens, stop=self.stop_sequences, prompt_logprobs=5)
        a = np.zeros(len(continuations))
        prompts = [f'{prompt} {continuation}' for continuation in continuations]
        gen = self.model.generate(prompts, sampling_params)
        for idx, continuation in enumerate(continuations):
            logprobs = gen[idx].prompt_logprobs[-1]
            for logprob in logprobs:
                if logprobs[logprob].decoded_token.upper().strip() == continuation.upper().strip():
                    a[idx] = logprobs[logprob].logprob
                    break
            if a[idx] == 0:
                a[idx] = -np.inf
        return(softmax(a[:-1])) # remove refusal option

class gpt_wrapper():
    '''
    wrapper for GPT API calls.
    '''
    def __init__(self, model, temp = 1, max_tokens = 192, timeout = 5, stop_sequences =  ['\\n'], async_gen = True):
        self.model = model
        self.temp = temp
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.stop_sequences = stop_sequences
        self.tokenizer = tiktoken.encoding_for_model(model)
        if async_gen:
            self.client = AsyncOpenAI()
        else:
            self.client = OpenAI()

    def generate(self, prompt):
        try:
            completion = self.client.chat.completions.create(model=self.model, 
            temperature=self.temp, 
            max_tokens=self.max_tokens, 
            stop = self.stop_sequences,
            messages=[{"role": "user", "content": prompt}]) 
            return(completion.choices[0].message.content)

        except openai.APIError:
            time.sleep(self.timeout)
            completion = self.client.chat.completions(model=self.model, 
            temperature=self.temp, 
            max_tokens=self.max_tokens, 
            stop = self.stop_sequences,
            messages=[{"role": "user", "content": prompt}])
            return(completion.choices[0].message.content)

    def get_logprobs_of_continuations(self, prompt, continuations):
        # returns logprobs of each potential continuation
        # output: np array of logprobs in same order as continuations list
        # assumes that all continuations are one token long
        a = np.zeros(len(continuations))
        tokens = []
        for idx, continuation in enumerate(continuations):
            token = self.tokenizer.encode(continuation)[0]
            a[idx] = bisection_search(OpenAIModel(model=self.model), prefix=prompt, idx=token)[0]
        
        # index to -1 to remove refusal option
        return(softmax(a[:-1]))