"""

    panllm - Unified API for local LLM inferences

    This file is a part of the panllm
    project and distributed under MIT license.
    https://github.com/kadir014/panllm

"""

from panllm import get_implemented_backends
from panllm.models import LLMBackend, LLMConfig
from panllm.backends.base import BaseLLM
from panllm.errors import InvalidInferenceBackend


def LLM(
        model_config: LLMConfig,
        backend: LLMBackend | None = None
    ) -> BaseLLM:

    if backend is None:
        impl_backends = get_implemented_backends()

        if len(impl_backends) == 0:
            raise InvalidInferenceBackend("No implemented inference backend is found.")

        backend = impl_backends[-1]

    else:
        if backend not in get_implemented_backends():
            raise InvalidInferenceBackend(f"Inference backend '{backend.name}' is not implemented.")

    if backend == LLMBackend.DUMMY:
        from panllm.backends.dummy import DummyLLM
        return DummyLLM(model_config, backend)

    elif backend == LLMBackend.LLAMA_CPP:
        from panllm.backends._llama_cpp import LlamaCppLLM
        return LlamaCppLLM(model_config, backend)
    
    elif backend == LLMBackend.EXLLAMAV2:
        from panllm.backends._exllamav2 import ExLlamaV2LLM
        return ExLlamaV2LLM(model_config, backend)