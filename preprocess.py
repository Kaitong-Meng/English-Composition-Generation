from transformers import DistilBertTokenizer, DistilBertModel
import xlrd
import torch
import numpy as np


tokenizer = DistilBertTokenizer.from_pretrained("./distilbert-base-cased")
model = DistilBertModel.from_pretrained("./distilbert-base-cased")

excel = xlrd.open_workbook("./scoring/asap-aes/training_set_rel3.xls")
table = excel.sheets()[0]
articles = table.col_values(colx=2)[1:6974]
scores = table.col_values(colx=6)[1:6974]
articles_extend = table.col_values(colx=2)[6975:10002]
scores_extend = table.col_values(colx=6)[6975:10002]
# print(len(articles))
# print(len(scores))
# print(len(articles_extend))
# print(len(scores_extend))
articles.extend(articles_extend)
scores.extend(scores_extend)
# print(len(articles))
# print(len(scores))
# print(articles[0])
# print(scores[0])
# print(set(scores))
# print(scores[6973])


def normalize(scores):
    new_scores = torch.zeros(len(scores), 10)
    for i in range(len(scores)):
        if scores[i] > 9.0:
            scores[i] = 9.0
        new_scores[i, int(scores[i])] = 1
    return new_scores, scores


def tokenization(article):
    input = tokenizer(article, return_tensors="pt", padding="max_length", truncation="longest_first", max_length=512)
    output = model(**input)[0].squeeze()
    return output


def dataset_tensor(articles, batch):
    dataset = []
    for i in range(len(articles)):
        data = tokenization(articles[i])
        data = data.detach().numpy()
        dataset.append(data)
        if i != 0 and (i + 1) % 50 == 0:
            dataset = np.array(dataset)
            dataset = torch.tensor(dataset)
            torch.save(dataset, "./scoring/articles/" + str(20 * batch + int((i + 1) / 50)) + ".pt")
            dataset = []


# new_scores, scores = normalize(scores)
# print(new_scores.shape, len(scores))


# text1 = """
#     Paris is the capital and most populous city of France,
#     with an estimated population of 2,175,601 residents as of 2018,
#     in an area of more than 105 square kilometres (41 square miles).
#     The City of Paris is the centre and seat of government of the
#     region and province of Île-de-France, or Paris Region,
#     which has an estimated population of 12,174,880,
#     or about 18 percent of the population of France as of 2017.
# """
#
#
# input1 = tokenizer(text1, return_tensors="pt", padding="max_length", truncation="longest_first", max_length=512)
# output1 = model(**input1)[0].squeeze()
# print(output1.shape)


# scores, _ = normalize(scores)
# torch.save(scores, "./scoring/scores.pt")


# articles = articles[0:50]
# dataset = []
# for i in range(len(articles)):
#     dataset.append(tokenization(articles[i]))
# dataset = torch.tensor(dataset)
# print(len(dataset))
# print(dataset[0].shape)
articles = articles[9000:10000]
dataset_tensor(articles, 9)
