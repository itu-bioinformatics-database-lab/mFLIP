import os
import random
import time


import torch
import pandas as pd
import numpy as np

from transformers import BertTokenizer, BertForMaskedLM, Trainer, TrainingArguments
from transformers import RobertaTokenizer, RobertaForMaskedLM
from datasets import Dataset


from deep_metabolitics.config import all_generated_datasets_dir, aycan_full_data_dir
from deep_metabolitics.data.properties import get_all_ds_ids
from deep_metabolitics.data.metabolight_dataset import (
    PathwayMinMaxDataset,
)
seed = 10
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


import numpy as np

metabolite_bins = np.linspace(-10, 10, num=1001)  # 2000 aralık, 2001 sınır noktası

reaction_bins = np.linspace(0, 1000, num=1001)
# print(metabolite_bins, reaction_bins)

experiment_name = os.path.basename(__file__).replace(".py", "")
print(f"{experiment_name = }")


epochs = 100


text_list = []
target_list = []

all_generated_dataset_ids = get_all_ds_ids(folder_path=all_generated_datasets_dir)
for ds_id in all_generated_dataset_ids:
    metabolomics_df, label_df, dataset_ids_df, factors_df = (
        PathwayMinMaxDataset.load_data_all_generated_datasets_csv(dataset_ids=[ds_id])
    )

    metabolomics_df = metabolomics_df[sorted(metabolomics_df.columns)]
    label_df = label_df[sorted(label_df.columns)]
    for (_, metabolites), (_, label) in zip(metabolomics_df.iterrows(), label_df.iterrows()):
        metabolites_text = " ".join(
            [
                f"{key} {np.digitize(metabolites[key], metabolite_bins)}"
                for key in metabolites.keys()
            ]
        )
        for key in label.keys():
            mask_text = f"{key} [MASK]"

            # mask_text = " ".join([f"{key} [MASK]" for key in label.keys()])
            text = metabolites_text + " " + mask_text
            target = str(np.digitize(label[key], reaction_bins))
            text_list.append(text)
            target_list.append(target)
        # target = [str(np.digitize(label[key], metabolite_bins)) for key in label.keys()]

        # text_list.append(text)
        # target_list.append(target)

data = {
    "metin": text_list,
    "hedefler": target_list,
    # "hedefler": [["örnek", "BERT"], ["BERT", "doğal"]],
}

df = pd.DataFrame(data)
print(f"{len(df) = }")
dataset = Dataset.from_pandas(df)

aycan_source_list = [
    "metabData_breast",
    "metabData_ccRCC3",
    "metabData_ccRCC4",
    "metabData_coad",
    "metabData_pdac",
    "metabData_prostat",
]
text_list = []
target_list = []
for ds_id in aycan_source_list:
    metabolomics_df, label_df, dataset_ids_df, factors_df = (
        PathwayMinMaxDataset.load_data_aycan_csv(dataset_ids=[ds_id])
    )
    metabolomics_df = metabolomics_df[sorted(metabolomics_df.columns)]
    label_df = label_df[sorted(label_df.columns)]
    for (_, metabolites), (_, label) in zip(
        metabolomics_df.iterrows(), label_df.iterrows()
    ):
        metabolites_text = " ".join(
            [
                f"{key} {np.digitize(metabolites[key], metabolite_bins)}"
                for key in metabolites.keys()
            ]
        )
        for key in label.keys():
            mask_text = f"{key} [MASK]"

            # mask_text = " ".join([f"{key} [MASK]" for key in label.keys()])
            text = metabolites_text + " " + mask_text
            target = str(np.digitize(label[key], reaction_bins))
            text_list.append(text)
            target_list.append(target) # mask ve target verirken her reaksiyonun 4 degerini veriyor olalim, her bir cumleye daha buyuk model kullanacagim
        # target = [str(np.digitize(label[key], metabolite_bins)) for key in label.keys()]

        # text_list.append(text)
        # target_list.append(target)


data = {
    "metin": text_list,
    "hedefler": target_list,
    # "hedefler": [["örnek", "BERT"], ["BERT", "doğal"]],
}

df_val = pd.DataFrame(data)
print(f"{len(df_val) = }")
dataset_val = Dataset.from_pandas(df_val)



from transformers import LongformerTokenizer
from transformers import AutoTokenizer, BigBirdForMaskedLM, BigBirdTokenizer
# tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
# tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
# tokenizer = RobertaTokenizer.from_pretrained("roberta-large")


# from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("google/bigbird-roberta-base")
model = BigBirdForMaskedLM.from_pretrained("google/bigbird-roberta-base")


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["metin"], padding = "max_length", truncation=True, max_length=1024
    )
    labels = tokenized_inputs["input_ids"].copy()

    for i, label in enumerate(labels):
        mask_positions = [
            idx
            for idx, token_id in enumerate(label)
            if token_id == tokenizer.mask_token_id
        ]
        for pos, target in zip(mask_positions, examples["hedefler"][i]):
            target_id = tokenizer.convert_tokens_to_ids(target)
            labels[i][pos] = target_id

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)
tokenized_dataset_val = dataset_val.map(tokenize_and_align_labels, batched=True)


training_args = TrainingArguments(
    output_dir=f"./results_{experiment_name}",
    overwrite_output_dir=True,
    num_train_epochs=epochs,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    # warmup_steps=500,
    weight_decay=0.01,
    logging_dir=f"./logs_{experiment_name}",
    logging_steps=10,
    report_to="none",
    evaluation_strategy="epoch",
    # evaluation_strategy="steps",
    # eval_steps=1000,
    bf16=True,  # BF16 kullanımı
    # load_best_model_at_end=True,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset_val,
    tokenizer=tokenizer,
)

trainer.train()

trainer.save_model(f"{experiment_name}_{epochs}_{len(df)}")