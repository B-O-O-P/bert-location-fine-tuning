import os
import logging

import torch
import pandas as pd
import numpy as np

from string import punctuation
from datetime import datetime

from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import BertTokenizer

from utils import one_hot_location, extraction_accuracy
from model import LET

# Logging

LOG_DIRECTORY = 'logs'
LOG_FILENAME = 'testing-log.txt'

if not os.path.exists(LOG_DIRECTORY):
    os.makedirs(LOG_DIRECTORY)

logging.basicConfig(filename='{}/{}'.format(LOG_DIRECTORY, LOG_FILENAME), filemode='a', level=logging.INFO, format='%(message)s')

logging.info('{} BERT EXTRACTION FINE-TUNING {}'.format(8 * '=', 8 * '='))
logging.info('Start time: {}\n'.format(datetime.now()))

# Init model

logging.info('{} Initializing model {}'.format(5 * '=', 5 * '='))

model = LET()

logging.info('Model for extraction initialized\n')

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

# Prepare data

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

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

un_labels = list(map(lambda array_str: array_str[1:-1].split(', '), un_labels))
un_labels = list(map(lambda array: list(map(lambda str: str[1:-1], array)), un_labels))

decoded_texts = list(map(lambda id: tokenizer.decode(id), input_ids))

labels = []
for i in range(len(un_labels)):
    label = un_labels[i]
    labels.append(one_hot_location(decoded_texts[i], label))

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

    CHECKPOINT_FILENAME = 'bert-location-extraction-transformer-epoch-{}.pt'.format(epoch)
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
            logits = model(b_input_ids, input_mask=b_input_mask).squeeze()
            logits = logits.cpu().numpy()
            logits = np.array([np.array(list(map(lambda x: 1 if x > 0 else 0, l))) for l in logits])
            label_ids = b_labels.cpu().numpy()

            tmp_eval_accuracy = 0
            for (logit, label) in zip(logits, label_ids):
                tmp_eval_accuracy += extraction_accuracy(logit, label, verbose=False)
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

logging.info('{} INFERENCE PART {}'.format(7 * '=', 7 * '='))

logging.info('{} Preparing data {}'.format(5 * '=', 5 * '='))

data = pd.read_csv('data/COCO-locations-test-12.csv')

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

# Prepare data

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

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

decoded_texts = list(map(lambda id: tokenizer.decode(id), input_ids))

labels = []
for i in range(len(un_labels)):
    label = un_labels[i]
    encoded_label = tokenizer.encode_plus(
        label,
        add_special_tokens=False,
        return_tensors='pt'
    )
    labels.append(encoded_label['input_ids'].unsqueeze(0))

labels = torch.cat(labels)

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

    CHECKPOINT_FILENAME = 'bert-location-extraction-transformer-epoch-{}.pt'.format(epoch)
    checkpoint_path = '{}/{}'.format(CHECKPOINT_DIRECTORY, CHECKPOINT_FILENAME)

    model.load_state_dict(torch.load(checkpoint_path))

    logging.info('Model loaded from file: {}'.format(checkpoint_path))
    logging.info('{} epoch model loaded\n'.format(epoch))

    # Validation

    logging.info('Start testing...')
    model.eval()

    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    for step, batch in enumerate(test_inf_dataloader):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        with torch.no_grad():
            logits = model(b_input_ids, input_mask=b_input_mask).squeeze()
            logits = logits.cpu().numpy()
            logits = np.array([np.array(list(map(lambda x: 1 if x > 0 else 0, l))) for l in logits])
            label_ids = b_labels.cpu().numpy()

            tmp_eval_accuracy = 0
            for i in range(len(label_ids)):
                logit = logits[i]
                label = label_ids[i][0]
                ids = b_input_ids[i]
                label_set = set()

                for j in range(len(logit)):
                    mark = logit[j]
                    if mark == 1:
                        label_set.add(ids[j])
                tmp_eval_accuracy += all(l in label_set for l in label)
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += batch_size

        if step % log_steps == 0 and step:
            logging.info('  Average accuracy: {}'.format(eval_accuracy / nb_eval_steps * 100))

    epoch += each_epoch_checkpoint

    epoch_accuracy = eval_accuracy / nb_eval_steps * 100

    if best_inf_accuracy is None or epoch_accuracy > best_inf_accuracy:
        best_inf_accuracy = epoch_accuracy

    logging.info("Testing accuracy for epoch {0}: {1:.2f}%\n".format(epoch, epoch_accuracy))

logging.info("Best accuracy on inference part: {0:.2f}%\n".format(best_inf_accuracy))
logging.info('Inference part finish time: {}\n'.format(datetime.now()))

logging.info('Finish time: {}'.format(datetime.now()))
logging.info('{} Finish {}'.format(5 * '=', 5 * '='))
