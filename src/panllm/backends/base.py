"""

    panllm - Unified API for local LLM inferences

    This file is a part of the panllm
    project and distributed under MIT license.
    https://github.com/kadir014/panllm

"""

from typing import Iterable, Iterator

from abc import ABC, abstractmethod

from panllm.models import (
    LLMConfig,
    GenerationConfig,
    GenerationStats,
    TextGenerationResult,
    ChatChunk,
    ChatGenerationResult
)


class BaseStream(ABC):
    @property
    @abstractmethod
    def stats(self) -> GenerationStats:
        """
        Generation statistics for the whole streaming session.
        
        Values will be just zero if all tokens are not exhausted yet.
        """
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[str | ChatChunk]:
        """
        Iterate over streamed chunks.
        
        It can be either one token (a single string), or a ChatChunk
        depending on the type of the generation request made.
        """
        ...


class BaseLLM(ABC):
    """
    Abstract interface for backend inference implementations.
    """

    def __init__(self, model_config: LLMConfig) -> None:
        self.__model_config = model_config

        self.load()

    @property
    def model_config(self) -> LLMConfig:
        return self.__model_config
    
    @property
    @abstractmethod
    def seed(self) -> int:
        """ Get the sampling seed. """
        ...

    @seed.setter
    @abstractmethod
    def seed(self, new_value: int) -> None:
        """ Set the sampling seed. """
        ...
    
    @abstractmethod
    def load(self) -> None:
        """ Load model. """
        ...

    @abstractmethod
    def release(self) -> None:
        """ Explicitly request to release model resources. """
        ...

    @abstractmethod
    def token_length(self,
            content: str,
            add_bos: bool = True,
            specialize: bool = False
        ) -> int:
        """ Calculate the amount of tokens the text content has. """
        ...

    @abstractmethod
    def generate(self,
            prompt: str,
            generation_config: GenerationConfig | None = None
        ) -> TextGenerationResult:
        """ Generate text completion. """
        ...

    @abstractmethod
    def generate_chat(self,
            messages: Iterable[dict[str, str]],
            generation_config: GenerationConfig | None = None
        ) -> ChatGenerationResult:
        """ Generate text chat completion. """
        ...

    @abstractmethod
    def stream(self,
            prompt: str,
            generation_config: GenerationConfig | None = None
        ) -> BaseStream:
        """ Stream text completion. """
        ...

    @abstractmethod
    def stream_chat(self,
            messages: Iterable[dict[str, str]],
            generation_config: GenerationConfig | None = None
        ) -> BaseStream:
        """ Stream text chat completion. """
        ...