"""

    panllm - Unified API for local LLM inferences

    This file is a part of the panllm
    project and distributed under MIT license.
    https://github.com/kadir014/panllm

"""

from panllm.backends.base import BaseLLM, BaseStream
from panllm.models import (
    LLMBackend,
    LLMConfig,
    GenerationConfig,
    GenerationStats,
    TextGenerationResult,
    ChatChunk,
    ChatGenerationResult
)


LLAMA_CPP_IMPLEMENTED = True
EXLLAMAV2_IMPLEMENTED = True

try:
    from panllm.backends._llama_cpp import LlamaCppLLM
except ImportError:
    LLAMA_CPP_IMPLEMENTED = False

try:
    from panllm.backends._exllamav2 import ExLlamaV2LLM
except ImportError:
    EXLLAMAV2_IMPLEMENTED = False

def get_implemented_backends() -> list[LLMBackend]:
    impl = [LLMBackend.DUMMY]

    if LLAMA_CPP_IMPLEMENTED:
        impl.append(LLMBackend.LLAMA_CPP)

    if EXLLAMAV2_IMPLEMENTED:
        impl.append(LLMBackend.EXLLAMAV2)

    return impl


# Need to import llm.py after all implementations are imported!
from panllm.llm import LLM


__version__ = "0.0.2"


__all__ = [
    "BaseLLM",
    "BaseStream",
    "LLMConfig",
    "LLMBackend",
    "GenerationConfig",
    "GenerationStats",
    "TextGenerationResult",
    "ChatChunk",
    "ChatGenerationResult",
    "LLAMA_CPP_IMPLEMENTED",
    "EXLLAMAV2_IMPLEMENTED",
    "get_implemented_backends",
    "LLM",
    "__version__"
]