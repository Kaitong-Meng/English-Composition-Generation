from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import PhrasalConstraint


# import the model from a local path
model_path = "./gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True)
model = AutoModelForCausalLM.from_pretrained(model_path, pad_token_id=tokenizer.eos_token_id)


# define decoding methods
def greedy_search(text, max_length):
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    greedy_output = model.generate(
        input_ids,
        max_length=max_length
    )
    output = tokenizer.decode(greedy_output[0], skip_special_tokens=True)
    return output


def beam_search(
        text, max_length, num_beams, no_repeat_ngram_size=2,
        num_return_sequences=1, index_return_sequences=1,
        is_early_stopping=True, keywords=None
):
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    if len(keywords) == 0:
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
    return output


def sampling(text, max_length, is_do_sample=True, top_k=0, temperature=1.0, top_p=0.9):
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    is_do_sample = bool(is_do_sample)
    sample_output = model.generate(
        input_ids,
        do_sample=is_do_sample,
        max_length=max_length,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature
    )
    output = tokenizer.decode(sample_output[0], skip_special_tokens=True)
    return output
