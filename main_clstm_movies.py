# Torchtext Tutorial: https://www.youtube.com/watch?v=KRgq4VnCr7I
# https://rohit-agrawal.medium.com/using-fine-tuned-gensim-word2vec-embeddings-with-torchtext-and-pytorch-17eea2883cd

import time
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
# torchtext version 0.2.1 - doesn't work on newest versions
import torch
import torchtext.data as data
from torch import nn

import dataProcesser as dp
from clstm_model import CLSTM

cuda = True
device = -1

path = 'data'
train = 'train.json'
dev = 'dev.json'
test = 'test.json'
char_data = False
min_freq = 1
data_format = 'json'
epochs = 25
log_interval = 500
batch_acc, batch_count = 0, 0
epoch_acc, epoch_count = 0, 0

sentence_field = data.Field(lower=True)
label_field = data.Field(sequential=False)


# dp.build_files_for_torchtext()

def convert_list2dict(convert_list):
    list_dict = OrderedDict()
    for index, word in enumerate(convert_list):
        list_dict[word] = index
    return list_dict


fields = {'sentence': ('sentence', sentence_field), 'label': ('label', label_field)}

train_data, dev_data, test_data = data.TabularDataset.splits(path, train=train, validation=dev, test=test,
                                                             format=data_format, fields=fields)

sentence_field.build_vocab(train_data, min_freq=min_freq)
label_field.use_vocab = False


if cuda:
    device = 'cuda'
else:
    device = -1

train_iter, dev_iter, test_iter = data.Iterator.splits((train_data, dev_data, test_data),
                                                       batch_sizes=(1, 1, 1),
                                                       device=device, repeat=False, sort=False)

w2v = dp.download_googlenews_pretrained_model()

word2vec_vectors = []
for token, idx in sentence_field.vocab.stoi.items():
    if w2v.__contains__(token):
        word2vec_vectors.append(torch.FloatTensor(w2v[token]))
    else:
        uniform_vector = torch.FloatTensor(np.random.uniform(-0.25, 0.25, int(300)).round(6))
        word2vec_vectors.append(uniform_vector)
sentence_field.vocab.set_vectors(sentence_field.vocab.stoi, word2vec_vectors, 300)

model = CLSTM(sentence_field.vocab.vectors)

if cuda:
    model.to('cuda')

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=5, momentum=0.9,
                                weight_decay=0.001)  # weight decay equals L2 regularization
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)

loss_val_means = list()
model.train()
start_time = time.time()
epoch_start_time = time.time()
for epoch in range(epochs):
    print("\n## The {}/{} Epoch ! ##".format(epoch + 1, epochs))
    i = 0
    loss_vals = list()
    for batch in train_iter:
        i += 1
        feature, target = batch.sentence, batch.label
        optimizer.zero_grad()
        output = model(feature)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        batch_acc += (output.argmax(1) == target).sum().item()
        batch_count += target.size(0)

        epoch_acc += (output.argmax(1) == target).sum().item()
        epoch_count += target.size(0)

        loss_vals.append(loss.item())

        if i % 500 == 0 or i % len(train_iter) == 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch + 1, i, len(train_iter), batch_acc / batch_count))
            batch_acc, batch_count = 0, 0
            start_time = time.time()

        if i == len(train_iter):
            train_iter.init_epoch()
            break

    loss_val_means.append(np.mean(loss_vals))

    print('-' * 59)
    print('| end of epoch {:3d} | time: {:5.2f}s | '
          'train accuracy {:8.3f} '.format(epoch + 1, time.time() - epoch_start_time, epoch_acc / epoch_count))
    print(loss_val_means[-1])
    epoch_start_time = time.time()
    epoch_acc, epoch_count = 0, 0
    print('-' * 59)

    scheduler.step()

# save model parameters
torch.save(model.state_dict(), 'data/movies/more/model_state_25e')

# Plot loss means:
plt.plot(range(len(loss_val_means)), loss_val_means)
plt.show()

# initiate model for evaluation
model.eval()

############## Test Model on Dev Data ##############
i = 0
dev_acc, dev_count = 0, 0
dev_start_time = time.time()
for batch in dev_iter:
    i += 1
    feature, target = batch.sentence, batch.label
    predicted_label = model(feature)
    dev_acc += (predicted_label.argmax(1) == target).sum().item()
    dev_count += target.size(0)

    if i == len(dev_iter):
        break

print('-' * 59)
print('| end of dev eval | time: {:5.2f}s | accuracy {:8.3f} '.format(time.time() - dev_start_time,
                                                                       dev_acc / dev_count))
print('-' * 59)



############## Test Model on Test Data ##############
i = 0
test_acc, test_count = 0, 0
test_start_time = time.time()
for batch in test_iter:
    i += 1
    feature, target = batch.sentence, batch.label
    predicted_label = model(feature)
    test_acc += (predicted_label.argmax(1) == target).sum().item()
    test_count += target.size(0)

    if i == len(test_iter):
        break

print('-' * 59)
print('| end of test eval | time: {:5.2f}s | accuracy {:8.3f} '.format(time.time() - test_start_time,
                                                                       test_acc / test_count))
print('-' * 59)




