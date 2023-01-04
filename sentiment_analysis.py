from transformers import pipeline



text1 = """
    Paris is the capital and most populous city of France, 
    with an estimated population of 2,175,601 residents as of 2018, 
    in an area of more than 105 square kilometres (41 square miles). 
    The City of Paris is the centre and seat of government of the 
    region and province of Île-de-France, or Paris Region, 
    which has an estimated population of 12,174,880, 
    or about 18 percent of the population of France as of 2017.
"""
text2 = "Paris is the capital and most populous city of France."


def sentiment(text):
    classifier = pipeline("text-classification", model=r"C:\Users\mengk\.cache\huggingface\hub\models--roberta-large-mnli\snapshots\0dcbcf20673c006ac2d1e324954491b96f0c0015")
    sentiment_analysis = classifier(text)
    label = sentiment_analysis[0]["label"]
    score = sentiment_analysis[0]["score"]
    return label, score


sent1, prob1 = sentiment(text1)
sent2, prob2 = sentiment(text2)
print(sent1, prob1)
print(sent2, prob2)
