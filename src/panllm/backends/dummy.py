"""

    panllm - Unified API for local LLM inferences

    This file is a part of the panllm
    project and distributed under MIT license.
    https://github.com/kadir014/panllm

"""

from typing import Iterable, Iterator

from time import perf_counter, sleep
import random
import string

from panllm.backends.base import BaseLLM, BaseStream
from panllm.models import (
    GenerationConfig,
    TextGenerationResult,
    ChatChunk,
    ChatGenerationResult,
    GenerationStats
)


def _dummy_block() -> None:
    """ Block current thread, imitating token generation. """
    min_ms = 1.0
    max_ms = 15.0
    sleep(random.uniform(min_ms / 1000.0, max_ms / 1000.0))


class DummyStream(BaseStream):
    def __init__(self,
            llm: "DummyLLM",
            prompt: str | Iterable[dict[str, str]],
            cfg: GenerationConfig
            ) -> None:
        self.__llm = llm
        self.__prompt = prompt
        self.__cfg = cfg
        self.__stats = None

    @property
    def stats(self) -> GenerationStats:
        if self.__stats is None:
            return GenerationStats()

        return self.__stats

    def __iter__(self) -> Iterator[str | ChatChunk]:
        _start = perf_counter()

        self.__llm._set_rand()

        content = ""
        possible_tokens = string.printable.strip()
        for _ in range(self.__cfg.max_tokens):
            content += random.choice(possible_tokens)
            token = content[-1]

            _dummy_block()
            
            if isinstance(self.__prompt, str):
                yield token

            else:
                yield ChatChunk(role="assistant", content=token)

        elapsed = perf_counter() - _start

        self.__llm._restore_rand()

        tokens = self.__llm.token_length(content)
        tps = tokens / elapsed

        self.__stats = GenerationStats(elapsed, tokens, tps)


class DummyLLM(BaseLLM):
    """
    Dummy inference implementation for debugging and testing purposes.
    """
    
    def load(self) -> None:
        self.__seed = -1
        self.__rand_state = None

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
    
    def _set_rand(self) -> None:
        """ Change random seed and store state. """
        self.__rand_state = random.getstate()
        seed = self.seed
        if seed == -1:
            seed = random.randint(0, 4294967295 - 1)
        random.seed(seed)

    def _restore_rand(self) -> None:
        """ Restore back to old random state. """
        random.setstate(self.__rand_state)
    
    def generate(self,
            prompt: str,
            generation_config: GenerationConfig | None = None
        ) -> TextGenerationResult:
        if generation_config is None:
            cfg = GenerationConfig()
        else:
            cfg = generation_config

        self._set_rand()

        _start = perf_counter()
        content = ""
        possible_tokens = string.printable.strip()
        for _ in range(cfg.max_tokens):
            content += random.choice(possible_tokens)
            _dummy_block()
        elapsed = perf_counter() - _start

        self._restore_rand()

        tokens = self.token_length(content)
        tps = tokens / elapsed

        return TextGenerationResult(
            full_content=prompt + content,
            generated_content=content,
            stats=GenerationStats(
                elapsed=elapsed,
                tokens=tokens,
                tokens_per_second=tps
            )
        )
    
    def generate_chat(self,
            messages: Iterable[dict[str, str]],
            generation_config: GenerationConfig | None = None
        ) -> ChatGenerationResult:
        if generation_config is None:
            cfg = GenerationConfig()
        else:
            cfg = generation_config

        self._set_rand()

        _start = perf_counter()
        content = ""
        possible_tokens = string.printable.strip()
        for _ in range(cfg.max_tokens):
            content += random.choice(possible_tokens)
            _dummy_block()
        elapsed = perf_counter() - _start

        self._restore_rand()

        tokens = self.token_length(content)
        tps = tokens / elapsed

        return ChatGenerationResult(
            message=ChatChunk(role="assistant", content=content),
            stats=GenerationStats(
                elapsed=elapsed,
                tokens=tokens,
                tokens_per_second=tps
            )
        )

    def stream(self,
            prompt: str,
            generation_config: GenerationConfig | None = None
        ) -> BaseStream:
        if generation_config is None:
            cfg = GenerationConfig()
        else:
            cfg = generation_config

        return DummyStream(self, prompt, cfg)

    def stream_chat(self,
            messages: Iterable[dict[str, str]],
            generation_config: GenerationConfig | None = None
        ) -> BaseStream:
        if generation_config is None:
            cfg = GenerationConfig()
        else:
            cfg = generation_config

        return DummyStream(self, messages, cfg)