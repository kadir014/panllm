"""

    panllm - Unified API for local LLM inferences

    This file is a part of the panllm
    project and distributed under MIT license.
    https://github.com/kadir014/panllm

"""

from dataclasses import dataclass
from enum import Enum, auto

from panllm.typing import PathLike


@dataclass(frozen=True)
class LLMConfig:
    """
    Genericized model loading configuration.

    Fields
    ------
    path
        Directory path to model
    context
        Maxiumum number of tokens in model context
    verbose
        Model logging message
    """

    path: PathLike
    context: int = 512
    verbose: bool = False


@dataclass
class GenerationConfig:
    """
    Genericized text generation and sampling configuration.

    Fields
    ------
    max_tokens
        Maximum number of tokens generated
    temperature
        Sampling temperature
    """

    max_tokens: int = 256
    temperature: float = 0.8


@dataclass(frozen=True)
class GenerationStats:
    """
    Generation statistics.

    Fields
    ------
    elapsed
        Elapsed time to generate in seconds
    tokens
        Generated amount of tokens
    tokens_per_second
        Token generation amount per second
    """

    elapsed: float = 0.0
    tokens: int = 0
    tokens_per_second: float = 0.0


@dataclass(frozen=True)
class TextGenerationResult:
    """
    Text completion generation results.

    Fields
    ------
    full_content
        Initial prompt + generated text content
    generated_content
        Generated text content
    stats
        Generation stats
    """

    full_content: str
    generated_content: str
    stats: GenerationStats


@dataclass(frozen=True)
class ChatChunk:
    """
    One chunk of text chat representing one message.
    """

    role: str
    content: str


@dataclass(frozen=True)
class ChatGenerationResult:
    """
    Text chat completion generation results.

    Fields
    ------
    message
        Generated text message
    stats
        GenerationStats
    """

    message: ChatChunk
    stats: GenerationStats


class LLMBackend(Enum):
    """
    Supported LLM inference backends.

    Make sure to use `get_implemented_backends` function to get actually
    implemented ones.
    """
    
    DUMMY = auto()
    LLAMA_CPP = auto()
    EXLLAMAV2 = auto()