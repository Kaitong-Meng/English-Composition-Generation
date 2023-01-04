from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from dataset import CompositionDataset


# import the model
model_path = "./gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# load the datasets
train_path = "./dataset/toy_train.json"
test_path = "./dataset/toy_test.json"


def load_dataset(train_path, test_path, tokenizer):
    train_dataset = CompositionDataset(train_path)
    test_dataset = CompositionDataset(test_path)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    return train_dataset, test_dataset, data_collator


train_dataset, test_dataset, data_collator = load_dataset(train_path, test_path, tokenizer)

# trainer
training_args = TrainingArguments(
    # output directory
    output_dir="./model",
    # overwrite the content of the output directory
    overwrite_output_dir=True,
    # number of training epochs
    num_train_epochs=3,
    # batch size for training
    per_device_train_batch_size=4,
    # batch size for evaluation
    per_device_eval_batch_size=8,
    # number of update steps between two evaluations
    eval_steps=400,
    # after # steps model is saved
    save_steps=800,
    # number of warmup steps for learning rate scheduler
    warmup_steps=500,
    prediction_loss_only=True,
    optim="adamw_torch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()
trainer.save_model()
