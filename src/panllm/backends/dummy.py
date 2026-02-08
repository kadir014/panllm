"""

    panllm - Unified API for local LLM inferences

    This file is a part of the panllm
    project and distributed under MIT license.
    https://github.com/kadir014/panllm

"""

from time import perf_counter
import random
import string

from panllm.backends.base import BaseLLM
from panllm.models import GenerationConfig, TextGenerationResult


class DummyLLM(BaseLLM):
    """
    Dummy inference implementation for debugging and testing purposes.
    """
    
    def load(self) -> None:
        self.__seed = -1

    def release(self) -> None:
        ...

    @property
    def seed(self) -> int:
        return self.__seed

    @seed.setter
    def seed(self, new_value: int) -> None:
        self.__seed = new_value

    def token_length(self, content: str, add_bos: bool = True) -> int:
        return len(content) // 3
    
    def generate(self,
            prompt: str,
            generation_config: GenerationConfig | None = None
        ) -> TextGenerationResult:
        if generation_config is None:
            cfg = GenerationConfig()
        else:
            cfg = generation_config

        rand_state = random.getstate()

        seed = self.seed
        if seed == -1:
            seed = random.randint(0, 4294967295 - 1)
        random.seed(seed)

        _start = perf_counter()
        content = ""
        possible_tokens = string.printable.strip()
        for _ in range(cfg.max_tokens):
            content += random.choice(possible_tokens)
        elapsed = perf_counter() - _start

        random.setstate(rand_state)

        tokens = self.token_length(content)
        tps = tokens / elapsed

        return TextGenerationResult(
            full_content=prompt + content,
            generated_content=content,
            elapsed=elapsed,
            tokens=tokens,
            tokens_per_second=tps
        )
    
    def generate_chat(self, messages, generation_config = None):
        ...

    def stream(self, prompt, generation_config = None):
        ...

    def stream_chat(self, messages, generation_config = None):
        ...