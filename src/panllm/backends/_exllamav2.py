"""

    panllm - Unified API for local LLM inferences

    This file is a part of the panllm
    project and distributed under MIT license.
    https://github.com/kadir014/panllm

"""

import exllamav2

from panllm.backends.base import BaseLLM


class ExLlamaV2LLM(BaseLLM):
    """
    exllamav2 backend implementation.
    """