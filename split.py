from sklearn.model_selection import train_test_split
import datasets_utils


# load the dataset
datafile = datasets_utils.read_json("./dataset/dataset")
dataset = []
for i in range(len(datafile)):
    dataset.append(datafile[i]["article"])

# create a toy dataset for pre-experiment
toy_dataset = dataset[0:300]
toy_train, toy_test = train_test_split(toy_dataset, test_size=0.15)
datasets_utils.build_subset(toy_train, "./dataset/toy_train.json")
datasets_utils.build_subset(toy_test, "./dataset/toy_test.json")

# # divide the dataset into training set and testing set
# train, test = train_test_split(dataset, test_size=0.15)
# build_subset(train, "./dataset/train.json")
# build_subset(test, "./dataset/test.json")

# print(len(train), len(test))
