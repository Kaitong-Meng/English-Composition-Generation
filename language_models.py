from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import PhrasalConstraint


def beam_search(
        model, tokenizer, text, max_length, num_beams, no_repeat_ngram_size=1,
        num_return_sequences=1, index_return_sequences=0,
        is_early_stopping=True, keywords=None
):
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    if keywords is None:
        beam_outputs = model.generate(
            input_ids,
            max_length=max_length,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            num_return_sequences=num_return_sequences,
            early_stopping=is_early_stopping
        )
    else:
        keywords = keywords.split(", ")
        token_ids = []
        for i in range(len(keywords)):
            token_ids += tokenizer(keywords[i], add_special_tokens=False).input_ids
        constraints = [PhrasalConstraint(token_ids)]
        beam_outputs = model.generate(
            input_ids,
            max_length=max_length,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            num_return_sequences=num_return_sequences,
            early_stopping=is_early_stopping,
            remove_invalid_values=True,
            constraints=constraints
        )
    output = []
    for i in range(num_return_sequences):
        output.append(tokenizer.decode(beam_outputs[i], skip_special_tokens=True))
    return output[index_return_sequences]


# OPT MODEL
# opt_path = "C:/Users/mengk/.cache/huggingface/hub/models--facebook--opt-125m/snapshots/934b6a077313f3ee660a918a95313f5d0b136c5a"
# tokenizer = AutoTokenizer.from_pretrained(opt_path, add_prefix_space=True)
# model = AutoModelForCausalLM.from_pretrained(opt_path, pad_token_id=tokenizer.eos_token_id)


# BLOOM MODEL
# bloom_path = "C:/Users/mengk/.cache/huggingface/hub/models--bigscience--bloom-560m/snapshots/afe2e6f33eb135d254df849c74bb83322b53641c"
# tokenizer = AutoTokenizer.from_pretrained(bloom_path, add_prefix_space=True)
# model = AutoModelForCausalLM.from_pretrained(bloom_path, pad_token_id=tokenizer.eos_token_id)


# GPT-2 MODEL
gpt_path = "./model"
tokenizer = AutoTokenizer.from_pretrained(gpt_path, add_prefix_space=True)
model = AutoModelForCausalLM.from_pretrained(gpt_path, pad_token_id=tokenizer.eos_token_id)


input = "I am happy and"
output = beam_search(model, tokenizer, input, max_length=128, num_beams=2, keywords="school, future")
print(output)
