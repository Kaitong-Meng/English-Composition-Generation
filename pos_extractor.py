from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline


# with open("./bert_pos/important symbols.txt", mode="r", encoding="utf-8") as f:
#     symbols = f.readlines()
# for i in range(len(symbols)):
#     symbols[i].strip("\n")
# # print(symbols)
# res = []
# for symbol in symbols:
#     res.append(symbol.replace("\n", ""))
# # print(res)
# keys = {
#     "ADP": 0,
#     "CCONJ": 1,
#     "SCONJ": 2,
#     "PRON": 3,
#     "DET": 4,
#     "NOUN": 5,
#     "AUX": 6,
#     "NUM": 7,
#     "PUNCT": 8,
#     "ADJ": 9,
#     "PROPN": 10,
#     "VERB": 11,
#     "PART": 12,
#     "ADV": 13
# }


def pos_extractor(model, tokenizer, text):
    extractor = pipeline("ner", model=model, tokenizer=tokenizer)
    pos = extractor(text)
    return pos


tokenizer = AutoTokenizer.from_pretrained("./bert_pos")
model = AutoModelForTokenClassification.from_pretrained("./bert_pos")
article = """
        The tower is 324 metres (1,063 ft) tall, about the same height as an 
        81-storey building, and the tallest structure in Paris. Its base is 
        square, measuring 125 metres (410 ft) on each side. During its 
        construction, the Eiffel Tower surpassed the Washington Monument to 
        become the tallest man-made structure in the world, a title it held for 
        41 years until the Chrysler Building in New York City was finished in 
        1930. It was the first structure to reach a height of 300 metres. Due 
        to the addition of a broadcasting aerial at the top of the tower in 
        1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). 
        Excluding transmitters, the Eiffel Tower is the second tallest 
        free-standing structure in France after the Millau Viaduct.
    """
pos = pos_extractor(model, tokenizer, article)
print(pos)

