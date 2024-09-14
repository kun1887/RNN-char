import numpy as np


# data I/O
with open("all_tswift_lyrics.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[ch] for ch in s]
decode = lambda l: "".join([itos[n] for n in l])

# hyperparameters
hidden_size = 100
seq_length = 25
lr = 1e-1

# model parameters/weights

Wxh = np.random.randn(hidden_size, vocab_size) * 0.01  # input to hidden
Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size) * 0.01  # hidden to output
bh = np.zeros((hidden_size, 1))  # hidden bias
by = np.zeros((vocab_size, 1))  # output bias


def loss_fun(inputs, targets, hprev):

    xs = {}  # inputs at different time
    hs = {}  # hidden states at different time
    ys = {}  # outputs at different time
    ps = {}  # probability for the next char at different time
    hs[-1] = np.copy(hprev)
    loss = 0

    # forward pass
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size, 1))
        xs[t][
            inputs[t]
        ] = 1  # using one hot encoding for input idx: 0 --> (1,0,0 ...) 1 --> (0,1,0,0 ...) ..

        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t - 1] + bh))
        ys[t] = np.dot(Why, hs[t]) + by  # unormalizedl logits for next char
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))  # probs for next char

        loss += -np.log(
            ps[t][targets[t], 0]
        )  # cross entropy loss summed over time timension

    # backward pass
    # ref for nice computational graph: https://towardsdatascience.com/backpropagation-in-rnn-explained-bdf853b4e1c2
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])
    for t in reversed(range(len(inputs))):
        dy = np.copy(ps[t])
        # backprob through a sofmax formula, ref: https://cs231n.github.io/neural-networks-case-study/#grad
        dy[targets[t]] -= 1
        # dWh_ij = dy_i * h_j from chain rule applied to 2d matrix
        dWhy += np.dot(dy, hs[t].T)
        dby += dy
        # dh_i = dy_j * W_ji sum over j = W.T_ij * dy_j sum over j
        dh = np.dot(Why.T, dy) + dhnext
        dhraw = (1 - hs[t] ** 2) * dh
        dbh += dhraw
        dWxh += np.dot(dhraw, xs[t].T)
        dWhh += np.dot(dhraw, hs[t - 1].T)
        dhnext = np.dot(Whh.T, dhraw)

    # clipping the gradients that might explode
    for dparam in [dWhh, dWhy, dWxh, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam)

    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs) - 1]


def sample(h, seed_ix, n_output):
    """
    input: first hidden state, index of the seed char to inizialize generation, number of chars to generate
    output: array of n chars
    """
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    ixs = []  # store output tokens

    for t in range(n_output):
        # update hidden state
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
        # calculate output at time t
        y = np.dot(Why, h) + by  # logits
        p = np.exp(y) / np.sum(np.exp(y))  # probs
        ix = np.random.choice(range(vocab_size), p=p.ravel())
        # now ix is the new input token, upload x
        x = np.zeros((vocab_size, 1))
        x[ix] = 1
        ixs.append(ix)

    return ixs
