import os
import logging

import torch
import pandas as pd
import numpy as np

from string import punctuation
from datetime import datetime
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler
from transformers import BertTokenizer

from embedding.embedding import vocabulary_from_texts, VocabularyEmbedding
from model import BERT

# Logging

LOG_DIRECTORY = 'logs'
LOG_FILENAME = 'validation-log.txt'

if not os.path.exists(LOG_DIRECTORY):
    os.makedirs(LOG_DIRECTORY)

open('{}/{}'.format(LOG_DIRECTORY, LOG_FILENAME)).close()

logging.basicConfig(filename='{}/{}'.format(LOG_DIRECTORY, LOG_FILENAME), filemode='a', level=logging.INFO, format='%(message)s')

logging.info('{} BERT PREDICTION FINE-TUNING {}'.format(8 * '=', 8 * '='))
logging.info('Start time: {}\n'.format(datetime.now()))

# Data

logging.info('{} Preparing data {}'.format(5 * '=', 5 * '='))

data = pd.read_csv('data/COCO-locations-negative-sampling.csv')

un_texts = list(data['cap'])
un_backgrounds = list(data['background'])
un_labels = list(data['binary'])

# Prepare data

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

vocabulary = vocabulary_from_texts(un_texts + un_backgrounds)

embedding_size = 256

embedding = VocabularyEmbedding(vocabulary, embedding_size)

logging.info('Vocabulary: size {}\n'.format(embedding.vocabulary_length))

# Tokenizing texts

texts = [(text.translate(str.maketrans('', '', punctuation))).lower() for text in un_texts]

input_ids = []
attention_masks = []

max_length = 256

for text in texts:
    encoded_dict = tokenizer.encode_plus(
        text,  # Sentence to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=max_length,  # Pad & truncate all sentences.
        pad_to_max_length=True,
        return_attention_mask=True,  # Construct attn. masks.
        return_tensors='pt',  # Return pytorch tensors.
    )

    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)

logging.info('Texts tokenized')

# Tokenize backgrounds

backgrounds = [embedding.word_to_ix[background] for background in un_backgrounds]
backgrounds = np.array(backgrounds)
backgrounds = torch.tensor(backgrounds)

logging.info('Backgrounds tokenized')

# Get labels

labels = torch.tensor(un_labels)

logging.info('Labels tokenized\n')

# Get dataset

batch_size = 32
N = len(texts)

train_size = int(N * 0.8)
val_size = int(N * 0.1)
test_size = (N - train_size - val_size)

dataset = TensorDataset(input_ids, attention_masks, backgrounds, labels)

logging.info('Dataset created')
logging.info('Dataset length: {}'.format(len(dataset)))

# Get Dataloader

train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])

train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=batch_size)
val_dataloader = DataLoader(val_data, sampler=RandomSampler(val_data), batch_size=batch_size)
test_dataloader = DataLoader(test_data, sampler=RandomSampler(test_data), batch_size=batch_size)

logging.info('\nDataloader created.\n')

# Init model

logging.info('{} Initializing model {}'.format(5 * '=', 5 * '='))

model = BERT(embedding.vocabulary_length, 256)

logging.info('BERT for prediction initialized\n')

# Select device

logging.info('{} Validation {}'.format(5 * '=', 5 * '='))

if torch.cuda.is_available():
    device = torch.device("cuda")
    model.cuda()
    logging.info('Device: cuda\n')
else:
    device = torch.device("cpu")
    logging.info('Device: cpu\n')

# Training params

log_steps = 100
sum_loss = 0

logging.info('Validation params:')
logging.info('  Batch size: {}\n'.format(batch_size))

# Checkpoint prepare

CHECKPOINT_DIRECTORY = 'models'

if not os.path.exists(CHECKPOINT_DIRECTORY):
    os.makedirs(CHECKPOINT_DIRECTORY)

logging.info('Directory for checkpoints ready\n')

# Load model state

epoch = 10
CHECKPOINT_FILENAME = 'bert-location-prediction-transformer-epoch-{}.pt'.format(epoch)
checkpoint_path = '{}/{}'.format(CHECKPOINT_DIRECTORY, CHECKPOINT_FILENAME)

model.load_state_dict(torch.load(checkpoint_path))

logging.info('Model loaded from file: {}'.format(checkpoint_path))
logging.info('{} epoch model loaded\n'.format(epoch))

# Validation

logging.info('Start validation...')
model.eval()

eval_loss, eval_accuracy = 0, 0
nb_eval_steps, nb_eval_examples = 0, 0

log_steps = 100

for step, batch in enumerate(val_dataloader):
    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)
    b_backgrounds = batch[2].to(device)
    b_labels = batch[3].to(device)

    with torch.no_grad():
        logits = model(b_input_ids, b_backgrounds, input_mask=b_input_mask).squeeze()
        logits = logits.cpu().numpy()
        logits = np.array(list(map(lambda x: 1 if x > 0 else 0, logits)))
        print(logits)
        label_ids = b_labels.cpu().numpy()
        print(label_ids)

        tmp_eval_accuracy = 0
        for (logit, label) in zip(logits, label_ids):
            tmp_eval_accuracy += int(logit == label)
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += batch_size

    if step % log_steps == 0 and step:
        logging.info('  Average accuracy: {}'.format(eval_accuracy / nb_eval_steps * 100))

logging.info("Validation accuracy: {0:.2f}%".format(eval_accuracy / nb_eval_steps * 100))

logging.info('Finish time: {}'.format(datetime.now()))
logging.info('{} Finish {}'.format(5 * '=', 5 * '='))
