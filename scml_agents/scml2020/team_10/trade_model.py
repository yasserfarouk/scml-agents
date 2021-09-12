import os
import sys

sys.path.append(os.path.dirname(__file__))

from random import shuffle

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from hyperparameters import *


class Model(nn.Module):
    def __init__(
        self,
        input_dim=TRADE_INPUT_DIM,
        hidden_dim=TRADE_HIDDEN_SIZE,
        output_dim=TRADE_OUTPUT_DIM,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.linear1 = nn.Linear(input_dim, hidden_dim // 2)
        self.linear2 = nn.Linear(hidden_dim // 2, hidden_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim)  # , dropout=0.5)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim)  # , dropout=0.5)

        # The linear layer that maps from hidden state space to tag space
        self.linear3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.linear4 = nn.Linear(hidden_dim // 2, output_dim)

        self.activation = nn.ReLU()

        self.loss_function = nn.MSELoss()

    def forward(self, x):
        x = self.linear2(self.activation(self.linear1(x)))

        lstm_out1, _ = self.lstm1(x.view(len(x), 1, -1))
        lstm_out2, _ = self.lstm2(lstm_out1)

        out = self.linear3(lstm_out2.view(len(x), -1))
        out = self.activation(out)
        out = self.linear4(out)
        out = self.activation(out)  # no need for negative output
        return out

    def fit(
        self,
        train_data,
        test_data,
        save_model=True,
        epochs=TRADE_EPOCHS,
        path=TRADE_INPUT_PATH,
    ):
        self.train()

        train_losses = []
        valid_losses = []

        optimizer = optim.Adam(self.parameters())

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
                tag = tag.view(-1, MAX_HORIZON)

                # Step 3. Compute the loss, gradients, and update the parameters by
                #  calling optimizer.step()
                loss = self.loss_function(tag_scores, tag)
                loss.backward()
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
        running_loss = 0.0
        for input, tag in validation_data:

            tag_scores = self(input)
            tag = tag.view(-1, UTILITY_OUTPUT_DIM)

            # Step 3. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = self.loss_function(tag_scores, tag)

            # print statistics
            running_loss += loss.item()
        return running_loss / len(validation_data)

    def plot(self, train, validation):
        # Plot training & validation loss values
        plt.plot(train)
        plt.plot(validation)
        plt.title("Model loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Test"], loc="upper left")
        plt.show()


def load_trade_model(_class=Model, path=TRADE_INPUT_PATH):
    model = _class()
    model.load_state_dict(torch.load(path))
    return model
