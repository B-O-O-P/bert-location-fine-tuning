import os
import logging

import torch
import pandas as pd

from string import punctuation
from datetime import datetime
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.nn import functional as F
from torch import optim as optim
from transformers import BertTokenizer

from model import LET
from utils import one_hot_location

# Logging


LOG_DIRECTORY = 'logs'
LOG_FILENAME = 'training-log.txt'

if not os.path.exists(LOG_DIRECTORY):
    os.makedirs(LOG_DIRECTORY)

logging.basicConfig(filename='{}/{}'.format(LOG_DIRECTORY, LOG_FILENAME), filemode='a', level=logging.INFO, format='%(message)s')

logging.info('{} BERT EXTRACTION FINE-TUNING {}'.format(8 * '=', 8 * '='))
logging.info('Start time: {}\n'.format(datetime.now()))

# Data

logging.info('{} Preparing data {}'.format(5 * '=', 5 * '='))

data = pd.read_csv('data/COCO-locations-train.csv')

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

batch_size = 32
N = len(texts)

dataset = TensorDataset(input_ids, attention_masks, labels)

logging.info('Dataset created')
logging.info('Dataset length: {}'.format(len(dataset)))

# Get Dataloader

train_data = dataset

train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=batch_size)

logging.info('\nDataloader created.\n')

# Init model

logging.info('{} Initializing model {}'.format(5 * '=', 5 * '='))

model = LET()

logging.info('Model for extraction initialized\n')

# Select device

logging.info('{} Training {}'.format(5 * '=', 5 * '='))

if torch.cuda.is_available():
    device = torch.device("cuda")
    model.cuda()
    logging.info('Device: cuda\n')
else:
    device = torch.device("cpu")
    logging.info('Device: cpu\n')

# Training params

epochs = 50
each_epoch_save = 5
log_steps = 100
sum_loss = 0

logging.info('Training params:')
logging.info('  Batch size: {}'.format(batch_size))
logging.info('  Number of epochs: {}\n'.format(epochs))

# Select optimizer

optimizer = optim.Adam(model.parameters(), lr=2e-5, eps=1e-8)

# Checkpoint prepare

CHECKPOINT_DIRECTORY = 'models'

if not os.path.exists(CHECKPOINT_DIRECTORY):
    os.makedirs(CHECKPOINT_DIRECTORY)

logging.info('Directory for checkpoints ready\n')

# Training

logging.info('Start training...')

model.train()

for epoch in range(1, epochs + 1):
    logging.info('{} EPOCH {} data {}'.format(3 * '=', epoch, 3 * '='))
    for step, batch in enumerate(train_dataloader):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        pred = model(b_input_ids, b_input_mask).squeeze()

        optimizer.zero_grad()

        loss = F.binary_cross_entropy_with_logits(pred, b_labels.type(torch.float32), reduction='sum')
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        sum_loss += loss.item()

        if step % log_steps == 0 and step:
            logging.info('  Average loss: {}'.format(sum_loss / log_steps))
            sum_loss = 0

    if epoch % each_epoch_save == 0 and epoch:
        checkpoint_filename = 'bert-location-extraction-transformer-epoch-{}.pt'.format(epoch)
        torch.save(model.state_dict(), os.path.join('./{}'.format(CHECKPOINT_DIRECTORY), checkpoint_filename))
        logging.info('\nCheckpoint \'{}\' saved\n'.format(checkpoint_filename))

logging.info('Finish time: {}'.format(datetime.now()))
logging.info('{} Finish {}'.format(5 * '=', 5 * '='))