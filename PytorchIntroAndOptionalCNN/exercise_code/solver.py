from random import shuffle
import numpy as np

import torch
from torch.autograd import Variable


class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss()):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        print('START TRAIN.')
        ########################################################################
        # TODO:                                                                #
        # Write your own personal training method for our solver. In each      #
        # epoch iter_per_epoch shuffled training batches are processed. The    #
        # loss for each batch is stored in self.train_loss_history. Every      #
        # log_nth iteration the loss is logged. After one epoch the training   #
        # accuracy of the last mini batch is logged and stored in              #
        # self.train_acc_history. We validate at the end of each epoch, log    #
        # the result and store the accuracy of the entire validation set in    #
        # self.val_acc_history.                                                #
        #                                                                      #
        # Your logging could like something like:                              #
        #   ...                                                                #
        #   [Iteration 700/4800] TRAIN loss: 1.452                             #
        #   [Iteration 800/4800] TRAIN loss: 1.409                             #
        #   [Iteration 900/4800] TRAIN loss: 1.374                             #
        #   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                            #
        #   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                            #
        #   ...                                                                #
        ########################################################################

        iterations = num_epochs * iter_per_epoch
        optimizer = self.optim(model.parameters(), **self.optim_args)

        for epoch in range(num_epochs):
            print('********************************************************************************')
            for iteration, sample in enumerate(train_loader):
                if iteration >= iter_per_epoch:
                    break

                optimizer.zero_grad()
                batch = sample[0].detach().to(device)
                labels = sample[1].detach().to(device)
                output = model(batch)
                training_loss = self.loss_func(output, labels)
                training_loss.backward()
                optimizer.step()

                current_iteration = iteration + epoch * iter_per_epoch + 1
                if current_iteration % log_nth == 0:
                    print(f'[Iteration {current_iteration}/{iterations}] TRAIN loss: {training_loss}')
                    self.train_loss_history.append(training_loss.detach())

            maxs, predict = torch.max(output, 1)
            training_acc = float((labels == predict).sum()) / len(predict)
            self.train_acc_history.append(training_acc)

            validation_acc = 0
            validation_loss = 0

            with torch.set_grad_enabled(False):
                for iteration, sample in enumerate(val_loader):
                    batch = sample[0].clone().detach().to(device)
                    labels = sample[1].clone().detach().to(device)
                    output = model(batch)
                    validation_loss += self.loss_func(output, labels)
                    maxs, predict = torch.max(output, 1)
                    validation_acc += float((labels == predict).sum()) / len(predict)

            validation_acc /= len(val_loader)
            validation_loss /= len(val_loader)

            self.val_acc_history.append(validation_acc)

            for i in range (int(iter_per_epoch / log_nth) - 1):
                self.val_loss_history.append(None)

            self.val_loss_history.append(validation_loss)

            print(f'[Epoch {epoch + 1}/{num_epochs}] TRAIN\tacc/loss: {training_acc} / {training_loss}')
            print(f'[Epoch {epoch + 1}/{num_epochs}] VAL\tacc/loss: {validation_acc} / {validation_loss}')
        print('********************************************************************************')
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        print('FINISH.')
