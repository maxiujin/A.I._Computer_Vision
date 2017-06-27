# encoding: utf-8
from __future__ import unicode_literals
from io import open
import glob
import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import copy
import pickle

def findFiles(path): return glob.glob(path)

# print(findFiles('data/names/*.txt'))

import unicodedata
import string

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# print(unicodeToAscii('Ślusàrski'))

# Build the category_lines dictionary, a list of names per language
category_lines = {}
all_categories = []

# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

for filename in findFiles('data/names/*.txt'):
    category = filename.split('/')[-1].split('.')[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)
# print(category_lines['dingdan'][:5])


# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

# print(letterToTensor('w'))

# print(lineToTensor('wo').size())


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)
input = Variable(letterToTensor('d'))
hidden = Variable(torch.zeros(1, n_hidden))

output, next_hidden = rnn(input, hidden)

input = Variable(lineToTensor('ding'))
hidden = Variable(torch.zeros(1, n_hidden))

output, next_hidden = rnn(input[0], hidden)
def categoryFromOutput(output):
    top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
    category_i = top_i[0][0]
    return all_categories[category_i], category_i

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
    line_tensor = Variable(lineToTensor(line))
    return category, line, category_tensor, line_tensor

for i in range(10):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    # print('category =', category, '/ line =', line)

    criterion = nn.NLLLoss()

    learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.data[0]

# print(categoryFromOutput(output))

n_iters = 100000
print_every = 5000
plot_every = 1000



# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    # Print iter number, loss, name and guess
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0


confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000

# Just return an output given a line
def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output

# Go through a bunch of examples and record which are correctly guessed
for i in range(n_confusion):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output = evaluate(line_tensor)
    guess, guess_i = categoryFromOutput(output)
    category_i = all_categories.index(category)
    confusion[category_i][guess_i] += 1

# Normalize by dividing every row by its sum
for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()

# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

# Set up axes
ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

# Force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# sphinx_gallery_thumbnail_number = 2
plt.show()

def predict(input_line, n_predictions=3):
    print('\n> %s' % input_line)
    output = evaluate(Variable(lineToTensor(input_line)))

    # Get top N categories
    topv, topi = output.data.topk(n_predictions, 1, True)
    predictions = []

    for i in range(n_predictions):
        value = topv[0][i]
        category_index = topi[0][i]
        print('(%.2f) %s' % (value, all_categories[category_index]))
        predictions.append([value, all_categories[category_index]])

predict('ding dan')
predict('wo yao cha xun dong xi')
predict('wo yao tui huo')
# print(output)



# model = mymodel()
# train = trainer.train(model...)

# copy you entirely object and save it 

model = RNN()

with open('model_save', 'wb') as f: 
 torch.save(model, f)























# import unicodedata
# from __future__ import unicode_literals, print_function, division
# torch.manual_seed(1)


# data = [("我要订单".split(), "CHINESE"),
#         ("意图是订单".split(), "Intention"),
#         ("我要查询我东西".split(), "CHINESE"),
#         ("也是订单".split(), "Intention")]

# test_data = [("我想查我刚刚买的东西".split(), "CHINESE"),
#              ("意图是".split(), "Intention")]

# # word_to_ix maps each word in the vocab to a unique integer, which will be its
# # index into the Bag of words vector
# word_to_ix = {}
# for sent, _ in data + test_data:
#     for word in sent:
#         if word not in word_to_ix:
#             word_to_ix[word] = len(word_to_ix)
# print(word_to_ix)

# VOCAB_SIZE = len(word_to_ix)
# NUM_LABELS = 2


# class BoWClassifier(nn.Module):  # inheriting from nn.Module!

#     def __init__(self, num_labels, vocab_size):
#         # calls the init function of nn.Module.  Dont get confused by syntax,
#         # just always do it in an nn.Module
#         super(BoWClassifier, self).__init__()

#         # Define the parameters that you will need.  In this case, we need A and b,
#         # the parameters of the affine mapping.
#         # Torch defines nn.Linear(), which provides the affine map.
#         # Make sure you understand why the input dimension is vocab_size
#         # and the output is num_labels!
#         self.linear = nn.Linear(vocab_size, num_labels)

#         # NOTE! The non-linearity log softmax does not have parameters! So we don't need
#         # to worry about that here

#     def forward(self, bow_vec):
#         # Pass the input through the linear layer,
#         # then pass that through log_softmax.
#         # Many non-linearities and other functions are in torch.nn.functional
#         return F.log_softmax(self.linear(bow_vec))


# def make_bow_vector(sentence, word_to_ix):
#     vec = torch.zeros(len(word_to_ix))
#     for word in sentence:
#         vec[word_to_ix[word]] += 1
#     return vec.view(1, -1)


# def make_target(label, label_to_ix):
#     return torch.LongTensor([label_to_ix[label]])


# model = BoWClassifier(NUM_LABELS, VOCAB_SIZE)

# # the model knows its parameters.  The first output below is A, the second is b.
# # Whenever you assign a component to a class variable in the __init__ function
# # of a module, which was done with the line
# # self.linear = nn.Linear(...)
# # Then through some Python magic from the Pytorch devs, your module
# # (in this case, BoWClassifier) will store knowledge of the nn.Linear's parameters
# for param in model.parameters():
#     print(param)

# # To run the model, pass in a BoW vector, but wrapped in an autograd.Variable
# sample = data[0]
# bow_vector = make_bow_vector(sample[0], word_to_ix)
# log_probs = model(autograd.Variable(bow_vector))
# print(log_probs)