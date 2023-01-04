def truncation(text):
    ending = [".", "?", "!", "\""]
    end_pos = 0
    if text[-1] not in ending:
        for i in range(len(text)):
            if text[i] in ending:
                end_pos = i
    return text[0:end_pos + 1]


text = """
    Paris is the capital and most populous city of France, 
    with an estimated population of 2,175,601 residents as of 2018, 
    in an area of more than 105 square kilometres (41 square miles). 
    The City of Paris is the centre and seat of government of the 
    region and province of Île-de-France, or Paris Region, 
    which has an estimated population of 12,174,880, 
    or about 18 percent of the population of France as of 
"""


print(truncation(text))
