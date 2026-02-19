"""

    panllm - Unified API for local LLM inferences

    This file is a part of the panllm
    project and distributed under MIT license.
    https://github.com/kadir014/panllm

"""

from typing import Iterable, Iterator

import ctypes
from time import perf_counter

import llama_cpp
from llama_cpp import llama_log_set
from llama_cpp.llama_chat_format import Jinja2ChatFormatter

from panllm.backends.base import BaseLLM, BaseStream
from panllm.models import (
    GenerationConfig,
    TextGenerationResult,
    LLMConfig,
    ChatChunk,
    ChatGenerationResult,
    GenerationStats
)


# verbose=False doesn't suppress llama_context logs, so use this instead
# https://github.com/abetlen/llama-cpp-python/issues/478#issuecomment-1749790484
# These has to be defined globally, or they cause OS errors

def _empty_log_callback(level, message, user_data): ...
_llama_cpp_logtype = ctypes.CFUNCTYPE(
    None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p
)
_empty_log_callback = _llama_cpp_logtype(_empty_log_callback)

def _llama_cpp_force_disable_logs() -> None:
    llama_log_set(_empty_log_callback, ctypes.c_void_p())


class LLamaCppStream(BaseStream):
    def __init__(self, llm: "LlamaCppLLM", res) -> None:
        self.__llm = llm
        self.__res = res
        self.__last_role = None
        self.__stats = None

    @property
    def stats(self) -> GenerationStats:
        if self.__stats is None:
            return GenerationStats()

        return self.__stats

    def __iter__(self) -> Iterator[str | ChatChunk]:
        _start = perf_counter()

        # TODO: This is not the actual total amount of tokens, just the text content
        tokens = 0

        for chunk in self.__res:
            choice = chunk["choices"][0]

            if "text" in choice:
                token = choice["text"]
                tokens += self.__llm.token_length(token, add_bos=False, specialize=False)
                yield token

            elif "delta" in choice:
                if "role" in choice["delta"] and choice["delta"]["role"] == "assistant":
                    self.__last_role = "assistant"

                elif "role" in choice["delta"] and choice["delta"]["role"] == "user":
                    self.__last_role = "user"

                if "content" in choice["delta"]:
                    if self.__last_role is None:
                        # TODO: Default behavior for content without role? I don't think this is possible tho
                        continue

                    token = choice["delta"]["content"]
                    tokens += self.__llm.token_length(token, add_bos=False, specialize=False)
                    yield ChatChunk(role=self.__last_role, content=token)

        elapsed = perf_counter() - _start
        tps = tokens / elapsed

        self.__stats = GenerationStats(elapsed, tokens, tps)


class LlamaCppLLM(BaseLLM):
    """
    llama-cpp-python backend implementation.
    """

    def __init__(self, model_config: LLMConfig) -> None:
        if not model_config.verbose:
            _llama_cpp_force_disable_logs()

        super().__init__(model_config)

    def _get_custom_chat_handler(self) -> Jinja2ChatFormatter | None:
        """
        Get custom Jinja2 chat handler.

        llama-cpp-python already has lots of chat template handlers implemented.
        However, the default Jinja2 handlers created internally are not accessible
        or alterable, this makes it difficult to add new parameters to the template.

        This function takes the internal Jinja2 handlers created in llama_cpp.Llama
        and minimally configures them for certain parameters.

        TODO: Currently I use this only for reasoning, but make this flexible for
              taking template variables dynamically, perhaps in model config.
        """

        # vocab-only -> No weights, only metadata
        vocab = llama_cpp.Llama(
            model_path=self.model_config.path,
            verbose=False,
            vocab_only=True
        )

        loaded_template = vocab.metadata["tokenizer.chat_template"]
        if "enable_thinking" not in loaded_template:
           return None

        enable_thinking = False
        t = f"{{%- set enable_thinking = {int(enable_thinking)} -%}}\n"
        new_template = t + loaded_template

        eos_token_id = vocab.token_eos()
        bos_token_id = vocab.token_bos()
        eos_token = vocab._model.token_get_text(eos_token_id) if eos_token_id != -1 else ""
        bos_token = vocab._model.token_get_text(bos_token_id) if bos_token_id != -1 else ""

        custom_formatter = Jinja2ChatFormatter(
            template=new_template,
            eos_token=eos_token,
            bos_token=bos_token,
            stop_token_ids=[eos_token_id]
        )

        return custom_formatter.to_chat_handler()
    
    def load(self) -> None:
        self.__seed = llama_cpp.LLAMA_DEFAULT_SEED

        self._llama = llama_cpp.Llama(
            model_path=self.model_config.path,
            n_gpu_layers=-1,
            verbose=self.model_config.verbose,
            n_ctx=self.model_config.context,
            seed=self.__seed,
            chat_handler=self._get_custom_chat_handler()
        )

    def release(self) -> None:
        self._llama.close()

    @property
    def seed(self) -> int:
        return self.__seed

    @seed.setter
    def seed(self, new_value: int) -> None:
        self.__seed = new_value
        self._llama.set_seed(new_value)

    def token_length(self,
            content: str,
            add_bos: bool = True,
            specialize: bool = False
        ) -> int:
        return len(self._llama.tokenizer().encode(content, add_bos=add_bos, special=specialize))

    def generate(self,
            prompt: str,
            generation_config: GenerationConfig | None = None
        ) -> TextGenerationResult:
        if generation_config is None:
            cfg = GenerationConfig()
        else:
            cfg = generation_config

        _start = perf_counter()
        res = self._llama.create_completion(
            prompt=prompt,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            seed=self.__seed
        )
        elapsed = perf_counter() - _start

        # TODO: finish reason= stop, length, tool_calls, function_call
        content = res["choices"][0]["text"]
        tokens = res["usage"]["completion_tokens"]
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

        _start = perf_counter()
        res = self._llama.create_chat_completion(
            messages=messages,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            seed=self.__seed
        )
        elapsed = perf_counter() - _start

        # TODO: finish reason= stop, length, tool_calls, function_call
        choice = res["choices"][0]["message"]
        tokens = res["usage"]["completion_tokens"]
        tps = tokens / elapsed

        return ChatGenerationResult(
            message=ChatChunk(role=choice["role"], content=choice["content"]),
            stats=GenerationStats(
                elapsed=elapsed,
                tokens=tokens,
                tokens_per_second=tps
            )
        )

    def stream(self,
            prompt: str,
            generation_config: GenerationConfig | None = None
        ) -> LLamaCppStream:
        if generation_config is None:
            cfg = GenerationConfig()
        else:
            cfg = generation_config

        res = self._llama.create_completion(
            prompt=prompt,
            stream=True,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            seed=self.__seed
        )

        return LLamaCppStream(self, res)

    def stream_chat(self,
            messages: Iterable[dict[str, str]],
            generation_config: GenerationConfig | None = None
        ) -> LLamaCppStream:
        if generation_config is None:
            cfg = GenerationConfig()
        else:
            cfg = generation_config

        res = self._llama.create_chat_completion(
            messages=messages,
            stream=True,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            seed=self.__seed
        )

        return LLamaCppStream(self, res)