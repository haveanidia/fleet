import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

REPO = "noumenon-labs/Fleet-Semantic-4B"

tokenizer = AutoTokenizer.from_pretrained(REPO)
model = AutoModelForCausalLM.from_pretrained(
    REPO,
    trust_remote_code=True,
    dtype=torch.float16,
    device_map="auto",
)

messages = [
    {
        "role": "user",
        "content": "A store sells 8 pens for $12. How much do 14 pens cost?",
    }
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)

device = model.get_input_embeddings().weight.device
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
    )

new_tokens = output[0, inputs["input_ids"].shape[-1]:]
print(tokenizer.decode(new_tokens, skip_special_tokens=True))
