import os
import logging

import torch
import pandas as pd
import numpy as np

from string import punctuation
from datetime import datetime
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import BertTokenizer

from embedding.embedding import vocabulary_from_texts, VocabularyEmbedding
from model import LIT

# Logging

LOG_DIRECTORY = 'logs'
LOG_FILENAME = 'validation-log.txt'

if not os.path.exists(LOG_DIRECTORY):
    os.makedirs(LOG_DIRECTORY)

logging.basicConfig(filename='{}/{}'.format(LOG_DIRECTORY, LOG_FILENAME), filemode='a', level=logging.INFO, format='%(message)s')

logging.info('{} BERT INFERENCE FINE-TUNING {}'.format(8 * '=', 8 * '='))
logging.info('Start time: {}\n'.format(datetime.now()))

# Data

logging.info('{} Preparing data {}'.format(5 * '=', 5 * '='))

data = pd.read_csv('data/COCO-locations-validation-filtered-negative-sampling.csv')

vocabulary_data = pd.read_csv('data/COCO-locations-vocabulary.csv')

un_texts = list(data['cap'])
un_backgrounds = list(data['location'])
un_labels = list(data['binary'])

vocabulary = set(vocabulary_data['words'])

# Prepare data

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

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
        padding='max_length',
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

dataset = TensorDataset(input_ids, attention_masks, backgrounds, labels)

logging.info('Dataset created')
logging.info('Dataset length: {}'.format(len(dataset)))

# Get Dataloader

val_data = dataset

val_dataloader = DataLoader(val_data, sampler=RandomSampler(val_data), batch_size=batch_size)

logging.info('\nDataloader created.\n')

# Init model

logging.info('{} Initializing model {}'.format(5 * '=', 5 * '='))

model = LIT(embedding.vocabulary_length, 256)

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

# Parameters

epoch = 10
each_epoch_checkpoint = 10
total_epochs = 50

while epoch <= total_epochs:
    # Load model state

    CHECKPOINT_FILENAME = 'bert-location-inference-transformer-epoch-{}.pt'.format(epoch)
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
            label_ids = b_labels.cpu().numpy()

            tmp_eval_accuracy = 0
            for (logit, label) in zip(logits, label_ids):
                tmp_eval_accuracy += int(logit == label)
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += batch_size

        if step % log_steps == 0 and step:
            logging.info('  Average accuracy: {}'.format(eval_accuracy / nb_eval_steps * 100))

    epoch += each_epoch_checkpoint

    logging.info("Validation accuracy for epoch {0}: {1:.2f}\n%".format(epoch, eval_accuracy / nb_eval_steps * 100))

logging.info('Finish time: {}'.format(datetime.now()))
logging.info('{} Finish {}'.format(5 * '=', 5 * '='))
