from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.normalizers import NFKC
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from datasets import load_from_disk
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from transformers import PreTrainedTokenizerFast

SYSTEM_PROMPT = "Bạn là một nhân viên tư vấn bán hàng chuyên nghiệp, luôn dùng kính ngữ 'dạ', 'ạ' và xưng hô 'mình' - 'bạn' lịch sự với khách hàng."
prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
])

# Generator -> trích xuất nội dung từ conversations
def batch_iterator(dataset, batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]
        text_batches = []
        for conversation in batch['conversations']:
            full_text = ' '.join([turn['value'] for turn in conversation])
            text_batches.append(full_text)
        yield text_batches

# Format lại theo cấu trúc của Tokenizer và LangChain Prompt Template
def preprocessing(examples, tokenizer):
    input_ids_list = []
    attention_mask_list = []

    for conversation in examples['conversations']:
        history = []

        for turn in conversation:
            if turn["from"] == "human":
                history.append(HumanMessage(content=turn["value"]))
            elif turn["from"] == "gpt":
                history.append(AIMessage(content=turn["value"]))

        prompt_value = prompt_template.format_messages(history=history)

        hf_messages = []
        for m in prompt_value:
            role = 'user' if m.type =='human' else ("assistant" if m.type == "ai" else "system")
            hf_messages.append({
                'role': role,
                'content': m.content,
            })

        raw_text = tokenizer.apply_chat_template(hf_messages, tokenize=False, add_generation_prompt=False)

        encoded = tokenizer(raw_text)
        input_ids_list.append(encoded['input_ids'])
        attention_mask_list.append(encoded['attention_mask'])

    return {"input_ids": input_ids_list, "attention_mask": attention_mask_list}


if __name__ == '__main__':
    data = load_from_disk('../dataset')

    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.normalizer = NFKC()
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=30000,
        special_tokens=['<s>', '<pad>', '</s>', '<unk>', '<mask>']
    )
    tokenizer.train_from_iterator(batch_iterator(data['train']), trainer=trainer)
    tokenizer.save('../bpe_tokenizer.json')

    tokenizer = PreTrainedTokenizerFast(tokenizer_file= '../bpe_tokenizer.json')
    tokenizer.add_special_tokens({
        'bos_token': '<s>',
        'eos_token': '</s>',
        'unk_token': '<unk>',
        'pad_token': '<pad>',
        'mask_token': '<mask>',
    })
    tokenizer.save_pretrained('../bpe-tokenizer')

    tokenizer.chat_template = (
        "{{ bos_token }}"
        "{% for message in messages %}"
        "{{ '### ' + message['role'].upper() + ':\n' + message['content'] + '</s>\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '### ASSISTANT:\n' }}"
        "{% endif %}"
    )

    tokenized_dataset = data.map(
        preprocessing,
        batched=True,
        num_proc=4,
        fn_kwargs={"tokenizer": tokenizer}
    )

    tokenized_dataset.save_to_disk('../tokenized_dataset')
