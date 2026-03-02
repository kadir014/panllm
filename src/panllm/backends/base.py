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
    LLMBackend,
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
        Generation statistics for the current streaming session.
        
        Values will be zero until tokens begin to be consumed.
        Statistics can be accessed while streaming, but timing-related
        values may be inaccurate at the beginning.
        """
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[str | ChatChunk]:
        """
        Iterate over streamed output chunks.

        Each yielded item is either a single token (as a string) or a
        ChatChunk instance, depending on the type of generation request.
        """
        ...


class BaseLLM(ABC):
    """
    Abstract interface for backend inference implementations.
    """

    def __init__(self, model_config: LLMConfig, backend: LLMBackend) -> None:
        self.__model_config = model_config
        self._backend = backend

        self.load()

    @property
    def model_config(self) -> LLMConfig:
        """ Configuration used to initialize the model. """
        return self.__model_config
    
    @property
    def backend(self) -> LLMBackend:
        """ Inference backend being used. """
        return self._backend
    
    @property
    @abstractmethod
    def seed(self) -> int:
        """
        Sampling seed.
        
        If negative, internal seed is shuffled for each inference.
        """
        ...

    @seed.setter
    @abstractmethod
    def seed(self, new_value: int) -> None:
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
        """ 
        Calculate the amount of tokens the text content has.
        
        Parameters
        ----------
        add_bos
            Add BOS token
        specialize
            Convert special tokens in the text content into actual special tokens
        """
        ...

    @abstractmethod
    def generate(self,
            prompt: str,
            generation_config: GenerationConfig | None = None
        ) -> TextGenerationResult:
        """
        Generate text completion.
        
        Parameters
        ----------
        prompt
            Text prompt to be used for generation
        generation_config
            Configuration used for generation
        """
        ...

    @abstractmethod
    def generate_chat(self,
            messages: Iterable[dict[str, str]],
            generation_config: GenerationConfig | None = None
        ) -> ChatGenerationResult:
        """
        Generate text chat completion.
        
        Parameters
        ----------
        prompt
            Array of messages to use as prompt
        generation_config
            Configuration used for generation
        """
        ...

    @abstractmethod
    def stream(self,
            prompt: str,
            generation_config: GenerationConfig | None = None
        ) -> BaseStream:
        """
        Stream text completion.
        
        Parameters
        ----------
        prompt
            Text prompt to be used for generation
        generation_config
            Configuration used for generation
        """
        ...

    @abstractmethod
    def stream_chat(self,
            messages: Iterable[dict[str, str]],
            generation_config: GenerationConfig | None = None
        ) -> BaseStream:
        """
        Stream text chat completion.
        
        Parameters
        ----------
        prompt
            Array of messages to use as prompt
        generation_config
            Configuration used for generation
        """
        ...