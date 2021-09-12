import os
import sys

sys.path.append(os.path.dirname(__file__))

from random import shuffle

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from hyperparameters import *
from torch import autograd


def printnormforward(self, input, output):
    return
    # input is a tuple of packed inputs
    # output is a Tensor. output.data is the Tensor we are interested
    print("Inside " + self.__class__.__name__ + " forward")
    print("")
    print("input: ", type(input))
    print("input[0]: ", type(input[0]))
    print("output: ", type(output))
    print("")
    print("input size:", input[0].size())
    print("output size:", output.data.size())
    print("output norm:", output.data.norm())


def printgradnormbackward(self, grad_input, grad_output):
    print("START:")
    print("Inside " + self.__class__.__name__ + " backward")
    print("Inside class:" + self.__class__.__name__)
    print("")
    print("grad_input: ", type(grad_input))
    print("grad_input[0]: ", type(grad_input[0]))
    print("grad_output: ", type(grad_output))
    print("grad_output[0]: ", type(grad_output[0]))
    print("")
    print("grad_input size:", grad_input[0].size())
    print("grad_output size:", grad_output[0].size())
    print("grad_input norm:", grad_input[0].norm())
    print("")
    print("")
    print("")


class NegModel(nn.Module, autograd.Function):
    def __init__(
        self,
        is_seller,
        input_dim=NEG_INPUT_DIM,
        hidden_dim=NEG_HIDDEN_SIZE,
        output_dim=NEG_OUTPUT_DIM,
    ):
        super().__init__()
        self.is_seller = is_seller

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.linear1 = nn.Linear(input_dim, hidden_dim // 2)
        # self.linear1.register_forward_hook(printnormforward)
        self.linear2 = nn.Linear(hidden_dim // 2, hidden_dim)
        # self.linear2.register_forward_hook(printnormforward)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim)  # , dropout=0.5)
        # self.lstm1.register_forward_hook(printnormforward)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim)  # , dropout=0.5)
        # self.lstm2.register_forward_hook(printnormforward)

        # The linear layer that maps from hidden state space to tag space
        self.linear3 = nn.Linear(hidden_dim, hidden_dim // 2)
        # self.linear3.register_forward_hook(printnormforward)
        self.linear4 = nn.Linear(hidden_dim // 2, output_dim)
        # self.linear4.register_forward_hook(printnormforward)

        self.activation = nn.ReLU()
        # self.activation.register_forward_hook(printnormforward)

        self.loss_function = nn.MSELoss()
        # self.loss_function.register_forward_hook(printnormforward)

    def forward(self, x):
        x = self.linear2(self.activation(self.linear1(x)))

        lstm_out1, _ = self.lstm1(x.view(len(x), 1, -1))
        lstm_out2, _ = self.lstm2(lstm_out1)

        out = self.linear3(lstm_out2.view(len(x), -1))
        out = self.activation(out)
        out = self.linear4(out)
        return out

    def fit(self, train_data, test_data, save_model=True, epochs=NEG_EPOCHS, path=None):
        if not path:
            path = NEG_SELL_PATH if self.is_seller else NEG_BUY_PATH

        self.train()

        train_losses = []
        valid_losses = []

        optimizer = optim.Adam(self.parameters(), lr=1e-3)

        for epoch in range(
            epochs
        ):  # again, normally you would NOT do 300 epochs, it is toy data
            shuffle(train_data)
            running_loss = 0.0
            counter = 0
            for input, tag in train_data:
                # Step 1. Remember that Pytorch accumulates gradients.
                # We need to clear them out before each instance
                self.zero_grad()

                # Step 2. Run our forward pass.
                tag_scores = self(input)
                tag = tag.view(-1, NEG_OUTPUT_DIM)

                # Step 3. Compute the loss, gradients, and update the parameters by
                #  calling optimizer.step()
                # assert tag.shape == tag_scores.shape
                loss = self.loss_function(tag_scores, tag)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), NEG_CLIP)
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                counter += 1

                # print every 2000 mini-batches
                if counter % 10 == 0:
                    validation_loss = self.validate(test_data)
                    valid_losses.append(validation_loss)
                    train_losses.append(running_loss / counter)
                    print(
                        "[%d / %d] train loss: %.3f\tvalidation loss: %.3f"
                        % (epoch + 1, epochs, running_loss / counter, validation_loss)
                    )

        if save_model:
            torch.save(self.state_dict(), path)

        return train_losses, valid_losses

    def validate(self, validation_data):
        self.eval()
        running_loss = 0.0
        for input, tag in validation_data:

            tag_scores = self(input)
            tag = tag.view(-1, NEG_OUTPUT_DIM)

            # Step 3. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = self.loss_function(tag_scores, tag)

            # print statistics
            running_loss += loss.item()
        self.train()
        return running_loss / len(validation_data)

    def predict(self, x):
        output = self(x)[-1]
        return [o.item() for o in output]

    def plot(self, train, validation):
        # Plot training & validation loss values
        plt.plot(train)
        plt.plot(validation)
        if self.is_seller:
            plt.title("Seller Negotiation Model Loss")
        else:
            plt.title("Buyer Negotiation Model Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Test"], loc="upper left")
        plt.show()


def load_seller_neg_model(_class=NegModel, path=NEG_SELL_PATH):
    model = _class(is_seller=True)
    model.load_state_dict(torch.load(path))
    return model


def load_buyer_neg_model(_class=NegModel, path=NEG_BUY_PATH):
    model = _class(is_seller=False)
    model.load_state_dict(torch.load(path))
    return model
