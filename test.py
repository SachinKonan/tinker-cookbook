
import tinker

from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

model_name = 'Qwen/Qwen3-30B-A3B-Instruct-2507'
tokenizer = get_tokenizer(model_name)
renderer_name = model_info.get_recommended_renderer_name(model_name)
renderer = renderers.get_renderer(renderer_name, tokenizer)
sampling_params = tinker.types.SamplingParams(
    max_tokens=50,
    stop=renderer.get_stop_sequences(),
)

sampling_client = tinker.ServiceClient().create_sampling_client(base_model=model_name)

convo = [
    {'role': 'system', 'content': 'you are a smart ai assistant, that helps with question and answer'},
    {"role": "user", "content": 'hi there add 91*10'},
]
print(convo)
model_input = renderer.build_generation_prompt(convo)
prompt_tokens = model_input.to_ints()

future = sampling_client.sample(
    prompt=model_input,
    num_samples=1,
    sampling_params=sampling_params,
)

sample_result = future.result()
print(sample_result)
sampled_tokens = sample_result.sequences[0].tokens
sampled_logprobs = sample_result.sequences[0].logprobs
assert sampled_logprobs is not None

all_tokens = prompt_tokens + sampled_tokens


parsed_message, _ = renderer.parse_response(sampled_tokens)

print(parsed_message)