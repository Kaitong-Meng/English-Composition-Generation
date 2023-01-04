from sentence_transformers import SentenceTransformer, util


source = """
    Paris is the capital and most populous city of France, 
    with an estimated population of 2,175,601 residents as of 2018, 
    in an area of more than 105 square kilometres (41 square miles). 
    The City of Paris is the centre and seat of government of the 
    region and province of Île-de-France, or Paris Region, 
    which has an estimated population of 12,174,880, 
    or about 18 percent of the population of France as of 2017.
"""
sentences = "Paris, the capital and most populous city of France."


def similarity(title, composition):
    model = SentenceTransformer(r"C:\Users\mengk\.cache\torch\sentence_transformers\sentence-transformers_all-MiniLM-L6-v2")
    title_embeddings = model.encode(title, convert_to_tensor=True)
    composition_embeddings = model.encode(composition, convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(title_embeddings, composition_embeddings)[0][0].cpu().item()
    return similarity_score


print(similarity(sentences, source))
