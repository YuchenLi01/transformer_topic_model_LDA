""" Training loop for LMs, with mostly hard-coded decisions.
"""
import sys
import math

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import utils
import wandb


def compute_batch_loss(
        args,
        lm,
        observation_batch,
        label_batch,
        attention_mask,
        include_reg=True,
):
    def _compute_batch_loss(loss, logits, label_batch, include_reg):
        batch_loss = loss(logits, label_batch)
        if include_reg:
            for param in lm.parameters():
                if param.requires_grad:
                    batch_loss += args['training']['weight_decay'] / 2 * torch.norm(param.data, p='fro')
        return batch_loss

    if args['lm']['no_softmax']:
        loss = nn.MSELoss()
    else:
        loss = nn.CrossEntropyLoss()
    batch_size, seq_len = label_batch.size()[0], label_batch.size()[1]
    if args['training']['objective'] in {'default'}:
        if args['lm']['lm_type'] in {'BertForMaskedLM', 'BertForMaskedLMCustom'}:
            logits, = lm(observation_batch, attention_mask)
        else:
            raise NotImplementedError('Model not supported.')

        logits = logits.view(batch_size * seq_len, -1)

        if args['lm']['no_softmax']:
            if len(label_batch.size()) == 2:  # (batch_size, seq_len)
                label_batch = label_batch.view(batch_size * seq_len, -1)

                # -100 means the (prediction, label) pair should be ignored
                keep_idx = label_batch != -100
                label_batch = label_batch[keep_idx]
                logits = logits[keep_idx[:, 0]]

                label_batch_one_hot = torch.zeros(
                  (len(label_batch), args['language']['vocab_size']),
                  device=args['device'],
                )
                label_batch_one_hot.scatter_(
                  1,
                  label_batch.view(len(label_batch), -1),
                  1,
                )  # convert to one-hot representation of each token
                label_batch = label_batch_one_hot

            batch_loss = _compute_batch_loss(loss, logits, label_batch, include_reg)
            return batch_loss

        if len(label_batch.size()) == 2:  # (batch_size, seq_len)
            label_batch = label_batch.view(batch_size * seq_len, )
        else:
            assert len(label_batch.size()) == 3  # (batch_size, seq_len, vocab_size)
            label_batch = label_batch.view(batch_size * seq_len, -1)

        batch_loss = _compute_batch_loss(loss, logits, label_batch, include_reg)
        return batch_loss

    raise NotImplementedError(f"Undefined args['training']['objective']: {args['training']['objective']}")


def log_train_dynamics_to_wandb(epoch_index, train_loss):
    results = {
        'epoch': epoch_index,
        'train_loss': train_loss,
    }
    wandb.log(results)


def train(
        args,
        lm,
        train_batches,
        dev_batches,
        steps_between_logging=50,
        steps_between_evals=None,
        dataset=None,
):
    """Trains the language model with Adam,

    Uses a learning rate annealing-on-plateau scheme,
    early stopping after 3 consecutive epochs bearing no improvement.

    Arguments:
      lm: a LanguageModel object
      train_batches: PyTorch DataLoader of training data from Dataset
      dev_batches: PyTorch DataLoader of dev data from Dataset
    """
    lm_params_path = utils.get_lm_path_of_args(args)
    print(lm_params_path)

    if args['training']['optimizer'] == 'Adam':
        optimizer = optim.AdamW(
            [param for param in lm.parameters() if param.requires_grad],
            args['training']['learning_rate'],
            weight_decay=0.0,  # not args['training']['weight_decay'] because l2 reg is added to loss
        )
    elif args['training']['optimizer'] == 'SGD':
        optimizer = optim.SGD(
            [param for param in lm.parameters() if param.requires_grad],
            args['training']['learning_rate'],
            weight_decay=0.0,  # not args['training']['weight_decay'] because l2 reg is added to loss
        )
    else:
        raise NotImplementedError(f"optimizer {args['training']['optimizer']} is not supported ")
    # scheduler_patience = 0
    max_epochs = args['training']['max_epochs']
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,patience=scheduler_patience)
    if steps_between_evals is None:
        steps_between_evals = len(train_batches)
    min_dev_loss = sys.maxsize
    min_dev_loss_epoch = -1
    torch.save(lm.state_dict(), lm_params_path)
    total_gradient_steps = 0
    for epoch_index in range(max_epochs):
        epoch_train_loss = 0
        train_batch_count = 0
        for observation_batch, label_batch, attention_mask, length in train_batches:
            if args['training']['mask_prob'] == 0.0:
                assert attention_mask is None
            else:
                assert attention_mask is not None
            # Compute forward, backward, and take gradient step
            lm.train()
            batch_loss = compute_batch_loss(
                args,
                lm,
                observation_batch,
                label_batch,
                attention_mask,
                include_reg=True,
            )
            epoch_train_loss += batch_loss.detach().cpu().numpy()
            train_batch_count += 1
            batch_loss.backward(retain_graph=True)

            if total_gradient_steps <= args['reporting']['log_all_steps_until']:
                log_train_dynamics_to_wandb(
                    epoch_index,
                    batch_loss.detach().cpu().numpy(),
                )
            elif total_gradient_steps % steps_between_logging == 0:
                epoch_avg_train_loss = epoch_train_loss / train_batch_count
                log_train_dynamics_to_wandb(
                    epoch_index,
                    epoch_avg_train_loss,
                )

            optimizer.step()
            optimizer.zero_grad()

            # Determine whether it's time to evaluate on dev data
            if total_gradient_steps % steps_between_evals == 0:
                epoch_dev_loss = 0
                dev_batch_count = 0
                # Compute dev loss
                for observation_batch, label_batch, attention_mask, length in dev_batches:
                    if args['training']['mask_prob'] == 0.0:
                        assert attention_mask is None
                    else:
                        assert attention_mask is not None
                    dev_batch_count+= 1
                    optimizer.zero_grad()
                    lm.eval()
                    batch_loss = compute_batch_loss(
                        args,
                        lm,
                        observation_batch,
                        label_batch,
                        attention_mask,
                        include_reg=False,
                    )
                    epoch_dev_loss += batch_loss.detach().cpu().numpy()
                # scheduler.step(epoch_dev_loss)
                epoch_avg_dev_loss = epoch_dev_loss/ dev_batch_count
                epoch_avg_train_loss = epoch_train_loss/ train_batch_count

                results = {
                    'epoch': epoch_index,
                    'train_loss': epoch_avg_train_loss,
                    'dev_loss': epoch_avg_dev_loss,
                }
                wandb.log(results)
                # If new best dev loss, save parameters.
                if epoch_avg_dev_loss < min_dev_loss:
                    tqdm.write(
                        '[epoch {}] Train loss: {}, Dev loss: {}'.format(
                            epoch_index,
                            epoch_avg_train_loss,
                            epoch_avg_dev_loss,
                        )
                    )
                    torch.save(lm.state_dict(), lm_params_path)
                    min_dev_loss = epoch_avg_dev_loss
                    min_dev_loss_epoch = epoch_index
                    tqdm.write('Saving lm parameters')
                elif min_dev_loss_epoch < epoch_index - 4:
                    tqdm.write('Early stopping')
                    tqdm.write("Min dev loss: {}".format(min_dev_loss))
                    return

            total_gradient_steps += 1

        tqdm.write("Min dev loss: {}".format(min_dev_loss))
