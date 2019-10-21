import torch
from ignite.engine import (Events, create_supervised_evaluator,
                           create_supervised_trainer)
from ignite.metrics import Accuracy, Loss
from torch.utils import tensorboard
from tqdm import trange, tqdm

def train_ignite(net, loss_fn, optimizer, train, test):
    """
    net: model
    loss_fn: loss function
    optimizer
    train, test :: datasets
    """

    trainer = create_supervised_trainer(net, optimizer, loss_fn)


    evaluator = create_supervised_evaluator(net, metrics={
        'accuracy': Accuracy(),
        'loss': Loss(loss_fn)
    })

    summary = tensorboard.SummaryWriter()

    @trainer.on(Events.EPOCH_COMPLETED)
    def train_log(trainer):
        evaluator.run(train)
        metrics = evaluator.state.metrics
        summary.add_scalar("train/loss",
                        metrics['loss'],
                        trainer.state.epoch)
        summary.add_scalar("train/accuracy",
                        metrics['accuracy'],
                        trainer.state.epoch)
        print(
            f"epoch: {trainer.state.epoch:<3} | accuracy:{metrics['accuracy']:.4%} | loss: {metrics['loss']:.4f}")


    @trainer.on(Events.EPOCH_COMPLETED)
    def test_log(trainer):
        evaluator.run(test)
        metrics = evaluator.state.metrics
        summary.add_scalar("test/accuracy",
                        metrics['accuracy'],
                        trainer.state.epoch)
        print(f"accuracy test:{metrics['accuracy']:.4%}")

    return trainer

def train_rnn(net, loss_fn, optimizer, train, test, epochs):
    """
    net: model
    loss_fn: loss function
    optimizer
    train, test :: datasets
    epochs
    """
    for i in trange(epochs):

        loss_total = 0
        i = 0
        for x, y in tqdm(train):
            optimizer.zero_grad()
            
            out = net(x)

            loss = loss_fn(out, y)
            loss.backward(retain_graph=True)
            optimizer.step()

            i += 1
            loss_total += loss.item()
        
        print(f'{loss_total/i:.2%}')
