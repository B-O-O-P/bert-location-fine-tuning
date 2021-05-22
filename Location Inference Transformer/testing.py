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
LOG_FILENAME = 'testing-log.txt'

if not os.path.exists(LOG_DIRECTORY):
    os.makedirs(LOG_DIRECTORY)

logging.basicConfig(filename='{}/{}'.format(LOG_DIRECTORY, LOG_FILENAME), filemode='a', level=logging.INFO, format='%(message)s')

logging.info('{} BERT INFERENCE FINE-TUNING {}'.format(8 * '=', 8 * '='))
logging.info('Start time: {}\n'.format(datetime.now()))

# Prepare vocabulary

vocabulary_data = pd.read_csv('data/COCO-locations-filtered-negative-sampling.csv')
vocabulary_texts = list(vocabulary_data['cap'])
vocabulary_backgrounds = list(vocabulary_data['location'])

filter_data = pd.read_csv('data/COCO-locations-list.csv')
un_filter_locations = list(filter_data['location'])

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

vocabulary = vocabulary_from_texts(vocabulary_texts + vocabulary_backgrounds + un_filter_locations)

embedding_size = 256

embedding = VocabularyEmbedding(vocabulary, embedding_size)

logging.info('Vocabulary: size {}\n'.format(embedding.vocabulary_length))

# Filter locations

filter_locations = [embedding.word_to_ix[l] for l in un_filter_locations]

# Init model

logging.info('{} Initializing model {}'.format(5 * '=', 5 * '='))

model = LIT(embedding.vocabulary_length, embedding_size)

logging.info('Model for inference initialized\n')

# Select device

logging.info('{} Testing {}'.format(5 * '=', 5 * '='))

if torch.cuda.is_available():
    device = torch.device("cuda")
    model.cuda()
    logging.info('Device: cuda\n')
else:
    device = torch.device("cpu")
    logging.info('Device: cpu\n')

# Extraction part

logging.info('{} EXTRACTION PART {}'.format(7 * '=', 7 * '='))

logging.info('{} Preparing data {}'.format(5 * '=', 5 * '='))

data = pd.read_csv('data/COCO-locations-test-20.csv')

un_texts = list(data['cap'])
un_labels = list(data['location'])
un_labels = list(map(lambda array_str: array_str[1:-1].split(', '), un_labels))
un_labels = list(map(lambda array: list(map(lambda str: str[1:-1], array)), un_labels))

t_texts = []
t_labels = []

for i in range(len(un_texts)):
    text = un_texts[i]
    label_array = un_labels[i]
    for label in label_array:
        t_texts.append(text)
        t_labels.append(label)

un_texts = t_texts
un_labels = t_labels

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

# Get labels

labels = []
for i in range(len(un_labels)):
    label = un_labels[i]
    labels.append(embedding.word_to_ix[label])

labels = torch.tensor(labels)

logging.info('Labels tokenized\n')

# Get dataset

batch_size = 8
N = len(texts)

dataset = TensorDataset(input_ids, attention_masks, labels)

logging.info('Dataset created')
logging.info('Dataset length: {}'.format(len(dataset)))

# Get Dataloader

test_ext_data = dataset

test_ext_dataloader = DataLoader(test_ext_data, sampler=RandomSampler(test_ext_data), batch_size=batch_size)

logging.info('\nDataloader created.\n')

# Testing params

log_steps = 5
sum_loss = 0

logging.info('Testing params:')
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
best_ext_accuracy = None

while epoch <= total_epochs:
    # Load model state

    CHECKPOINT_FILENAME = 'bert-location-inference-transformer-epoch-{}.pt'.format(epoch)
    checkpoint_path = '{}/{}'.format(CHECKPOINT_DIRECTORY, CHECKPOINT_FILENAME)

    model.load_state_dict(torch.load(checkpoint_path))

    logging.info('Model loaded from file: {}'.format(checkpoint_path))
    logging.info('{} epoch model loaded\n'.format(epoch))

    # Validation

    logging.info('Start testing...')
    model.eval()

    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    for step, batch in enumerate(test_ext_dataloader):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        with torch.no_grad():
            max_output = [None] * len(b_labels)
            max_relevances = [[] for _ in range(len(b_labels))]
            for location in filter_locations:
                batch_backgrounds = torch.tensor([location] * len(b_labels)).to(device)
                logits = model(b_input_ids, batch_backgrounds, input_mask=b_input_mask).squeeze()
                logits = logits.detach().cpu().numpy()
                for i in range(len(b_labels)):
                    if max_output[i] is None or max_output[i] < logits[i]:
                        max_output[i] = logits[i]
                        max_relevances[i].append(location)
            label_ids = b_labels.to('cpu').numpy()

            tmp_eval_accuracy = 0
            for (relevances, label) in zip(max_relevances, label_ids):
                if label in relevances[-3:]:
                    tmp_eval_accuracy += 1
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += batch_size

        if step % log_steps == 0 and step:
            logging.info('  Average accuracy: {}'.format(eval_accuracy / nb_eval_steps * 100))

    epoch_accuracy = eval_accuracy / nb_eval_steps * 100

    if best_ext_accuracy is None or epoch_accuracy > best_ext_accuracy:
        best_ext_accuracy = epoch_accuracy

    logging.info("Testing accuracy for epoch {0}: {1:.2f}%\n".format(epoch, epoch_accuracy))

    epoch += each_epoch_checkpoint

logging.info("Best accuracy on extraction part: {0:.2f}%\n".format(best_ext_accuracy))
logging.info('Extraction part finish time: {}\n'.format(datetime.now()))

# Inference part

logging.info('{} EXTRACTION PART {}'.format(7 * '=', 7 * '='))

logging.info('{} Preparing data {}'.format(5 * '=', 5 * '='))

data = pd.read_csv('data/COCO-locations-test-20.csv')

un_texts = list(data['cap'])
un_labels = list(data['location'])
un_labels = list(map(lambda array_str: array_str[1:-1].split(', '), un_labels))
un_labels = list(map(lambda array: list(map(lambda str: str[1:-1], array)), un_labels))

t_texts = []
t_labels = []

for i in range(len(un_texts)):
    text = un_texts[i]
    label_array = un_labels[i]
    for label in label_array:
        t_texts.append(text)
        t_labels.append(label)

un_texts = t_texts
un_labels = t_labels

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

# Get labels

labels = []
for i in range(len(un_labels)):
    label = un_labels[i]
    labels.append(embedding.word_to_ix[label])

labels = torch.tensor(labels)

logging.info('Labels tokenized\n')

# Get dataset

batch_size = 8
N = len(texts)

dataset = TensorDataset(input_ids, attention_masks, labels)

logging.info('Dataset created')
logging.info('Dataset length: {}'.format(len(dataset)))

# Get Dataloader

test_inf_data = dataset

test_inf_dataloader = DataLoader(test_inf_data, sampler=RandomSampler(test_inf_data), batch_size=batch_size)

logging.info('\nDataloader created.\n')

# Testing params

log_steps = 5
sum_loss = 0

logging.info('Testing params:')
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
best_inf_accuracy = None

while epoch <= total_epochs:
    # Load model state

    CHECKPOINT_FILENAME = 'bert-location-inference-transformer-epoch-{}.pt'.format(epoch)
    checkpoint_path = '{}/{}'.format(CHECKPOINT_DIRECTORY, CHECKPOINT_FILENAME)

    model.load_state_dict(torch.load(checkpoint_path))

    logging.info('Model loaded from file: {}'.format(checkpoint_path))
    logging.info('{} epoch model loaded\n'.format(epoch))

    # Validation

    logging.info('Start testing...')
    model.eval()

    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    for step, batch in enumerate(test_ext_dataloader):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        with torch.no_grad():
            max_output = [None] * len(b_labels)
            max_relevances = [[] for _ in range(len(b_labels))]
            for location in filter_locations:
                batch_backgrounds = torch.tensor([location] * len(b_labels)).to(device)
                logits = model(b_input_ids, batch_backgrounds, input_mask=b_input_mask).squeeze()
                logits = logits.detach().cpu().numpy()
                for i in range(len(b_labels)):
                    if max_output[i] is None or max_output[i] < logits[i]:
                        max_output[i] = logits[i]
                        max_relevances[i].append(location)
            label_ids = b_labels.to('cpu').numpy()

            tmp_eval_accuracy = 0
            for (relevances, label) in zip(max_relevances, label_ids):
                if label in relevances[-3:]:
                    tmp_eval_accuracy += 1
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += batch_size

        if step % log_steps == 0 and step:
            logging.info('  Average accuracy: {}'.format(eval_accuracy / nb_eval_steps * 100))

    epoch_accuracy = eval_accuracy / nb_eval_steps * 100

    if best_inf_accuracy is None or epoch_accuracy > best_inf_accuracy:
        best_inf_accuracy = epoch_accuracy

    logging.info("Testing accuracy for epoch {0}: {1:.2f}%\n".format(epoch, epoch_accuracy))

    epoch += each_epoch_checkpoint

logging.info("Best accuracy on inference part: {0:.2f}%\n".format(best_inf_accuracy))
logging.info('Inference part finish time: {}\n'.format(datetime.now()))

logging.info('Finish time: {}'.format(datetime.now()))
logging.info('{} Finish {}'.format(5 * '=', 5 * '='))
