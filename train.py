from math import sqrt

import torch
from torch import nn
from d2l import torch as d2l
from tqdm import tqdm
from Models.node_prompt import NPM


def evaluate_accuracy_gpu(model, data_loader, device=None):
    if isinstance(model, torch.nn.Module):
        model.eval()
        if not device:
            device = next(iter(model.parameters())).device
    metric = d2l.Accumulator(2)
    for data, label in data_loader:
        if isinstance(data, list):
            data = [x.to(device) for x in data]
        else:
            data = data.to(device)
        label = label.to(device)
        metric.add(d2l.accuracy(model(data), label), label.numel())
    return metric[0] / metric[1]


def cosine_similarity(g1, g2):
    dot_product = sum((g1_i * g2_i).sum() for g1_i, g2_i in zip(g1, g2))
    norm_g1 = sqrt(sum((g1_i ** 2).sum() for g1_i in g1))
    norm_g2 = sqrt(sum((g2_i ** 2).sum() for g2_i in g2))
    return dot_product / (norm_g1 * norm_g2)


def orthogonal_projection(g1, g2):
    dot_product = sum((g1_i * g2_i).sum() for g1_i, g2_i in zip(g1, g2))
    norm_g2_squared = sum((g2_i ** 2).sum() for g2_i in g2)
    scale = dot_product / norm_g2_squared
    proj = [scale * g2_i for g2_i in g2]
    return proj


def orthogonal_component(g1, g2):
    proj = orthogonal_projection(g1, g2)
    return [g1_i - proj_i for g1_i, proj_i in zip(g1, proj)]


def combine_gradients(source_grads, target_grads, gamma1=2e-1, gamma2=1e-3):
    cos_sim = cosine_similarity(source_grads, target_grads)
    if cos_sim > 0: 
        combined_grads = [gamma2*s_grad + gamma1*t_grad for s_grad, t_grad in zip(source_grads, target_grads)]
    else:  
        ortho_source_grads = orthogonal_component(source_grads, target_grads)
        combined_grads = [gamma2*ortho_s_grad + gamma1*t_grad for ortho_s_grad, t_grad in zip(ortho_source_grads, target_grads)]
    return combined_grads


def apply_gradients(model, combined_grads):
    for param, grad in zip(model.parameters(), combined_grads):
        param.grad = grad


def main_loop(model, source_train_loader, target_train_loader, optimizer, scheduler, criterion, num_epochs, device):
    def init_weights(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)

    model.apply(init_weights)
    print('Training on', torch.cuda.get_device_name(device))

    model = model.to(device)
    criterion = criterion.to(device)
    timer, num_batches = d2l.Timer(), len(source_train_loader)
    metric, train_loss, train_acc, test_acc = None, 0, 0, 0

    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)
        model.train()
        with tqdm(source_train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch', colour='GREEN') as t:
            for s_data, s_label in t:
                t_data, t_label = next(iter(target_train_loader))
                timer.start()

                optimizer.zero_grad()
                s_data, s_label = s_data.to(device), s_label.to(device)
                s_output = model(s_data)
                s_loss = criterion(s_output, s_label)
                s_loss.backward(retain_graph=True)
                s_grads = [param.grad.clone().detach() for param in model.parameters()]

                optimizer.zero_grad()
                t_data, t_label = t_data.to(device), t_label.to(device)
                t_output = model(t_data)
                t_loss = criterion(t_output, t_label)
                t_loss.backward(retain_graph=True)
                t_grads = [param.grad.clone().detach() for param in model.parameters()]

                combined_grads = combine_gradients(s_grads, t_grads)
                apply_gradients(model, combined_grads)

                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                with torch.no_grad():
                    metric.add(s_loss * s_data.shape[0], d2l.accuracy(s_output, s_label), s_data.shape[0])
                    metric.add(t_loss * t_data.shape[0], d2l.accuracy(t_output, t_label), t_data.shape[0])
                train_loss = metric[0] / metric[2] / 2
                train_acc = metric[1] / metric[2] / 2
                t.set_postfix(train_acc=f'{train_acc:.6f}', train_loss=f'{train_loss:.6f}')
                timer.stop()
        test_acc = evaluate_accuracy_gpu(model, target_train_loader)

    print(f'Loss {train_loss:.3f}, Train Acc {train_acc:.3f}, Test Acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {device}')
