# panllm
An easy-to-use Python library that aims to unify popular local LLM inferences in a higher-level API.

### Supported inference backends:
- [x] [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- [ ] [Transformers](https://github.com/huggingface/transformers)
- [ ] [ExLlamaV2](https://github.com/turboderp-org/exllamav2)
- [ ] [ExLlamaV3](https://github.com/turboderp-org/exllamav3)



# Installation
Minimally, you only need one of the [supported inference backends](#Supported_inference_backends) installed properly.



# Usage
Simple text completion
```py
from panllm import LLM, LLMConfig

MODEL_PATH = "C:/LLMs/SomeModel.gguf"

config = LLMConfig(MODEL_PATH)
llm = LLM(config)

result = llm.generate("Once upon a time, in a distant kingdom")

print(result.full_content)
```

Forcing a specific inference backend choice
```py
from panllm import LLM, LLMBackend, LLMConfig, get_implemented_backends

MODEL_PATH = "C:/LLMs/SomeModel.gguf"

# See all implemented inferences
print(get_implemented_backends())

config = LLMConfig(MODEL_PATH)

# If the 'backend' argument is not given, it tries to choose one from the
# currently implemented ones.
# But you can also force a specific one by passing it to the argument.
# In this case, we want to use llama-cpp-python
llm = LLM(config, backend=LLMBackend.LLAMA_CPP)
```

Streamed chat generation
```py
from panllm import LLM, LLMConfig

MODEL_PATH = "C:/LLMs/SomeModel.gguf"

config = LLMConfig(MODEL_PATH)
llm = LLM(config)

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant."
    },
    {
        "role": "user",
        "content": "What are the names of all the planets in our solar system?"
    }
]

# Stream generator session
stream = llm.stream_chat(messages)

# Each chunk can be just a string, or a chat chunk object, depending on the generation.
for chunk in stream:
    print(chunk.content, end="")

# After finishing streaming, print stats like tokens/s.
print(stream.stats)
```



# License
[MIT](LICENSE) © Kadir Aksoy