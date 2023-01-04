from transformers import BertTokenizer, BertModel
from transformers import pipeline


tokenizer = BertTokenizer.from_pretrained("./bert")
model = BertModel.from_pretrained("./bert")

nlp = pipeline("fill-mask", model=model, tokenizer=tokenizer)
text = """
    [CLS] [MASK] is the capital and most populous city of France, 
    with an estimated population of 2,175,601 residents as of 2018, 
    in an area of more than 105 square kilometres (41 square miles). 
    The City of Paris is the centre and seat of government of the 
    region and province of Île-de-France, or Paris Region, 
    which has an estimated population of 12,174,880, 
    or about 18 percent of the population of [MASK] as of 2017.
"""
result = nlp(text)
print(result)
