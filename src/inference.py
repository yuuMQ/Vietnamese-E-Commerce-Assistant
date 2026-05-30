import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from peft import PeftModel

SYSTEM_PROMPT = "Bạn là một nhân viên tư vấn bán hàng chuyên nghiệp, luôn dùng kính ngữ 'dạ', 'ạ' và xưng hô 'mình' - 'bạn' lịch sự với khách hàng."

BASE_MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
FINETUNED_DIR = '../output/checkpoint-264'

tokenizer = PreTrainedTokenizerFast.from_pretrained(FINETUNED_DIR)
tokenizer.chat_template = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{{ '### ' + message['role'].upper() + ':\n' + message['content'] + '</s>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '### ASSISTANT:\n' }}"
    "{% endif %}"
)

base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float16, device_map='cuda' if torch.cuda.is_available() else 'cpu')
base_model.resize_token_embeddings(len(tokenizer))

model = PeftModel.from_pretrained(base_model, FINETUNED_DIR)

model.eval()

user_query = 'Cái áo thun polo này có những size nào vậy? Mình cao 1m8 nặng 75kg'
messages = [
    {'role': 'system', 'content': SYSTEM_PROMPT},
    {'role': 'user', 'content': user_query},
]

raw_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(raw_text, return_tensors="pt", return_token_type_ids=False).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"Khách hàng: {user_query}")
    print(f"Chatbot:\n{response}")
    print("---------------------------------------------------------")