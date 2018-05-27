# coding: utf-8

from io import open
import glob
import unicodedata
import string
import random
import time
import math

import torch
import torch.nn as nn
from torch.autograd import Variable

import matplotlib.pyplot as plt


# ---- Define Functions and Classes ----
def readLines(filename):
    """
    Read a file and store the lines into a list
    :param filename: single filename
    :return: list of lines in the file
    """
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return lines


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        #self.softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.LogSoftmax()

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))


def categoryTensor(category):
    """
    Convert category to One-hot vector tensor
    :param category: category in string format
    :return: category one hot vector in tensor format
    >>> category
    >>> 'boys_name'
    >>> tensor
    >>> 1  0
    >>> [torch.FloatTensor of size 1x2]
    """
    li = all_categories.index(category)
    tensor = torch.zeros(1, n_categories)
    tensor[0][li] = 1
    return tensor


def inputTensor(line):
    """
    Convert name string (not including EOS) to One-hot matrix
    :param line: name string
    :return: one-hot matrix of name string in tensor
    >>> line
    >>> 'エナ'
    >>> tensor
    >>> Columns 0 to 18
    >>> 0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0
    >>> ...
    >>> [torch.FloatTensor of size 2x1x86]
    """
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor


def targetTensor(line):
    """
    LongTensor of second letter to end (EOS) for target
    :param line: target name in string format
    :return: index vector of letters in string
    >>> line
    >>> 'リオ'
    >>> letter_indexes
    >>> [9, 85]
    >>> torch.LongTensor(letter_indexes)
    >>>   9
    >>>  85
    >>> [torch.LongTensor of size 2]
    """
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1)  # EOS
    return torch.LongTensor(letter_indexes)


def train(category_tensor, input_line_tensor, target_line_tensor, lr=0.0005):
    """
    Perform training
    :param category_tensor: category hot matrix in tensor format
    :param input_line_tensor: input name hot matrix in tensor format
    :param target_line_tensor: target name index in tensor format
    :param lr: learning rate
    :return: NN model's output, normalized loss data
    """

    hidden = lstm.initHidden()

    lstm.zero_grad()

    loss = 0

    for i in range(input_line_tensor.size()[0]):
        output, hidden = lstm(category_tensor, input_line_tensor[i], hidden)
        loss += criterion(output, target_line_tensor[i])

    loss.backward()

    for p in lstm.parameters():
        p.data.add_(-lr, p.grad.data)

    return output, loss.data[0] / input_line_tensor.size()[0]


def timeSince(since):
    """
    Calculate processed time in min and sec
    :param since: time data when processing started
    :return: string to show min and sec
    """
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def samples(category, start_letters='ABC'):
    """
    Get multiple samples from one category and evaluate samples from multiple starting letters
    :param category: category in tensor format
    :param start_letters: several alphabet letters (concatnated) which can be first letter of new name
    :return: None
    """
    max_length = 10
    category_tensor = Variable(categoryTensor(category))
    for start_letter in start_letters:
        input = Variable(inputTensor(start_letter))
        hidden = lstm.initHidden()
        output_name = start_letter
        for i in range(max_length):
            output, hidden = lstm(category_tensor, input[0], hidden)
            topv, topi = output.data.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = Variable(inputTensor(letter))

        print(category, output_name)


# ---- Pre-processing ----
n_iters = 100000
print_every = 5000
plot_every = 500
all_losses = []
total_loss = 0  # Reset every plot_every iters
learning_rate = 0.0005

all_letters = "ァアィイゥウェエォオカガキギクグケゲコゴサザシジスズセゼソゾタダチヂッツヅテデトド" \
              "ナニヌネノハバパヒビピフブプヘベペホボポマミムメモャヤュユョヨラリルレロヮワヰヱヲンヴー"
n_letters = len(all_letters) + 1  # Plus EOS marker

# Build the category_lines dictionary, a list of lines per category
category_lines = {}
all_categories = []
for filename in glob.glob('../data/jp_names/*.txt'):
    category = filename.split('/')[-1].split('.')[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

criterion = nn.NLLLoss()

lstm = LSTM(n_letters, 128, n_letters)


# ---- Training ----
start = time.time()
for iter in range(1, n_iters + 1):
    # category, line = randomTrainingPair()
    # Generate rondom training pair
    category = all_categories[random.randint(0, len(all_categories) - 1)]
    line = category_lines[category][random.randint(0, len(category_lines[category]) - 1)]

    # Convert to tensor
    category_tensor = Variable(categoryTensor(category))
    input_line_tensor = Variable(inputTensor(line))
    target_line_tensor = Variable(targetTensor(line))

    output, loss = train(category_tensor, input_line_tensor, target_line_tensor, learning_rate)
    total_loss += loss

    if iter % print_every == 0:
        print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))

    if iter % plot_every == 0:
        all_losses.append(total_loss / plot_every)
        total_loss = 0

plt.figure()
plt.plot(all_losses)
plt.show()

# ---- Test ----
samples('boys_name', 'アイウエオ')
# boys_name アキト
# boys_name イツキ
# boys_name ウイチロウ
# boys_name エイ
# boys_name オウタロウ
samples('boys_name', 'カキクケコ')
# boys_name カンヤ
# boys_name キウタロウ
# boys_name クン
# boys_name ケン
# boys_name コウタ
samples('girls_name', 'アイウエオ')
# girls_name アキ
# girls_name イナミ
# girls_name ウナ
# girls_name エナミ
# girls_name オリ
samples('girls_name', 'カキクケコ')
# girls_name カナ
# girls_name キナ
# girls_name クリ
# girls_name ケイ
# girls_name コウ

# ---- Reference ----
# http://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html
