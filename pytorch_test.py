#%%
# %pylab inline

import torch
from ignite.engine import (Events, create_supervised_evaluator,
                           create_supervised_trainer)
from ignite.metrics import Accuracy, Loss
from torch import nn, optim

from load_csv import load_csv_pth as load_csv
from models_pth import RNN, Dense1

from training.train_pth import train_rnn, train_ignite

#%%
train, test = load_csv("UCI_Credit_Card.csv.zip")

#%%
_data, _target = next(iter(test))

#%%
# net = RNN(11, 6, 6)
net = Dense1(23, 2)
net(_data)
# 

#%%
loss_fn = nn.NLLLoss()
optimizer = optim.RMSprop(net.parameters(), lr=1e-6)

#%%
trainer = train_ignite(net, loss_fn, optimizer, train, test)
trainer.run(train, 10)
# train_rnn(net, loss_fn, optimizer, train, test, 10)


#%%
# trainer = create_supervised_trainer(net, optimizer, loss_fn)
# evaluator = create_supervised_evaluator(net, metrics={
#     'accuracy': Accuracy(),
#     'loss': Loss(loss_fn)
# })


# #%%
# summary = tensorboard.SummaryWriter()
# @trainer.on(Events.EPOCH_COMPLETED)
# def train_log(trainer):
#     evaluator.run(train)
#     metrics = evaluator.state.metrics
#     summary.add_scalar("train/loss",
#                         metrics['loss'],
#                         trainer.state.epoch)
#     summary.add_scalar("train/accuracy",
#                        metrics['accuracy'],
#                        trainer.state.epoch)
#     print(f"epoch: {trainer.state.epoch:<3} | accuracy:{metrics['accuracy']:.4%} | loss: {metrics['loss']:.4f}")


# @trainer.on(Events.EPOCH_COMPLETED)
# def test_log(trainer):
#     evaluator.run(test)
#     metrics = evaluator.state.metrics
#     summary.add_scalar("test/accuracy",
#                        metrics['accuracy'],
#                        trainer.state.epoch)
#     print(f"accuracy test:{metrics['accuracy']:.4%}")


# #%%
# trainer.run(train, max_epochs=20)
# summary.close()
#%%
