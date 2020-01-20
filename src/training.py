import os
import tqdm
import torch
import numpy as np
import pickle as pkl
import utils as utils
from torch.optim import Adam
from torch.nn import NLLLoss
from tensorboardX import SummaryWriter
from model_bert import BertLSTM, BertDense

def process_labels(temp_lab: list, labels_dict: dict):
    max_len = len(max(temp_lab, key=len))
    labels = []
    for sublist in temp_lab:
        padded_labels = [labels_dict[x] if x in labels_dict else labels_dict[None] for x in sublist]
        padded_labels.extend([labels_dict[None]] * (max_len - len(padded_labels)))
        assert len(padded_labels) == max_len
        labels.append(padded_labels)
    labels = torch.from_numpy(np.array(labels, dtype=int)).cuda()
    return labels

def process_tokens(temp_tokens: list):
    max_len = len(max(temp_tokens, key=len))
    tokens = []
    for sublist in temp_tokens:
        padded_tokens = [token for token in sublist]
        padded_tokens.extend([None] * (max_len - len(padded_tokens)))
        assert len(padded_tokens) == max_len
        tokens.append(padded_tokens)
    return tokens

def train_model(config, epochs = 40, batch_size = 64, sentence_max_len = 64, lr=1e-4):

    text_input_folder = os.path.join(config.data_folder, 'input/text_files/{}/'.format(config.inventory))
    input_folder = os.path.join(config.data_folder, 'input/matrices/{}/'.format(config.inventory))

    if config.finegrained:
        domains_vocab_path = os.path.join(text_input_folder, 'sensekeys.pkl')
        domains_vocab = pkl.load(open(domains_vocab_path, 'rb'))
        labels = sorted([x for x in domains_vocab if x != 'untagged'])
        labels_dict = {label: k + 1 for k, label in enumerate(labels)}
        labels_dict[None] = 0
        reverse_labels_dict = {v: k for k, v in labels_dict.items()}

    else:
        domains_vocab_path = os.path.join(text_input_folder, 'domains.pkl')
        domains_vocab = pkl.load(open(domains_vocab_path, 'rb'))
        labels = sorted([x for x in domains_vocab if x!='untagged'])
        labels_dict = {label: k + 1 for k, label in enumerate(labels)}
        labels_dict[None] = 0
        reverse_labels_dict = {v: k for k, v in labels_dict.items()}

    gold_folder = os.path.join(config.data_folder, 'gold/{}/'.format(config.inventory))

    mapping = pkl.load(open(config.mapping_path,'rb'))
    train_x = pkl.load(open(os.path.join(input_folder, "{}_words.pkl".format(config.training_name)), "rb")).tolist()

    if config.finegrained:
        train_y = pkl.load(open(os.path.join(input_folder, '{}_sensekeys.pkl'.format(config.training_name)), "rb")).tolist()
        dev_y = pkl.load(open(os.path.join(input_folder, '{}_sensekeys.pkl'.format(config.dev_name)), "rb")).tolist()

    else:
        train_y = pkl.load(open(os.path.join(input_folder, '{}_domains.pkl'.format(config.training_name)), "rb")).tolist()
        dev_y = pkl.load(open(os.path.join(input_folder, '{}_domains.pkl'.format(config.dev_name)), "rb")).tolist()

    txt_file = os.path.join(text_input_folder, '{}_input.txt'.format(config.dev_name))
    dev_x = pkl.load(open(os.path.join(input_folder, "{}_words.pkl".format(config.dev_name)), "rb")).tolist()
    dev_y_idx = process_labels(dev_y, labels_dict).detach().cpu().numpy().tolist()
    tokens = process_tokens(pkl.load(open(os.path.join(input_folder, '{}_tokens.pkl'.format(config.dev_name)), 'rb')).tolist())

    candidate_domains = utils.build_possible_senses(labels_dict, os.path.join(text_input_folder, 'semcor_input.txt'))

    dev_mask = utils.build_mask(words=dev_x, true_y=dev_y_idx, labels_dict=labels_dict, tokens=tokens,
                                file_txt=txt_file, candidate=candidate_domains)

    gold_dictionary = {line.strip().split()[0]: line.strip().split()[1:]
                           for line in open(os.path.join(gold_folder, '{}.gold.txt'.format(config.dev_name)))}

    if config.model_name=='BertDense':
        model = BertDense(len(labels_dict)).train()

    elif config.model_name=='BertLSTM':
        model = BertLSTM(len(labels_dict)).train()

    criterion = NLLLoss(ignore_index=0)
    optimizer = Adam(model.parameters(), lr=lr)
    writer = SummaryWriter(os.path.join(config.experiment_folder, 'logs'))
    output_file = open(os.path.join(config.experiment_folder, '{}.output.tsv'.format(config.dev_name)),'w')
    if config.start_from_checkpoint:
        load_checkpoints_path = os.path.join(config.experiment_folder, 'models',
                                             'checkpoint_{}.tar'.format(config.starting_epoch))
        model.load_state_dict(torch.load(load_checkpoints_path)['model_state_dict'])
        optimizer.load_state_dict(torch.load(load_checkpoints_path)['optimizer_state_dict'])


    for j in tqdm.tqdm(range(epochs - config.starting_epoch)):
        epoch = j + config.starting_epoch
        if config.starting_epoch > 0:
            epoch += 1
        path_checkpoints = os.path.join(config.experiment_folder, 'weights', 'checkpoint_{}.tar'.format(epoch))
        total, correct = 0, 0
        for i in tqdm.tqdm(range(0, len(train_x), batch_size)):
            inputs = train_x[i:i + batch_size]
            temp_lab = train_y[i:i + batch_size]
            if any(len(sublist) > sentence_max_len for sublist in inputs):
                inputs = [x[:sentence_max_len] for x in inputs]
                temp_lab = [x[:sentence_max_len] for x in temp_lab]
            labels = process_labels(temp_lab, labels_dict)
            optimizer.zero_grad()
            outputs = model.train()(inputs)
            loss = criterion(outputs.view(-1, outputs.size(2)), labels.view(-1))
            loss.backward()
            optimizer.step()
        writer.add_scalar('training_loss', loss.item(), epoch)
        eval_outs = torch.exp(model.eval()(dev_x))
        eval_outs *= dev_mask.cuda()
        predicted = torch.argmax(eval_outs, 2)
        for a,token_sent in enumerate(tokens):
            for b, instance_id in enumerate(token_sent):
                if not instance_id==None and not instance_id=='untagged':
                    gold_label = gold_dictionary[instance_id]
                    predicted_label = reverse_labels_dict[predicted[a, b].item()]
                    if predicted_label==None:
                        predicted_label = utils.getMFS(instance_id, mapping, txt_file, config.finegrained)
                    if predicted_label in gold_label:
                        correct += 1
                        output_file.write('c\t')
                    else:
                        output_file.write('w\t')
                    total += 1
                    output_file.write(instance_id + '\t' + predicted_label +'\t' + str(gold_label)+'\n')

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'eval_acc': correct / total
        }, path_checkpoints)
        writer.add_scalar('eval_acc', correct / total, epoch)
        del loss
        del eval_outs
    writer.close()

def train_model_one_out(config, epochs = 40, batch_size = 128, lr=1e-4):
    inventory = config.inventory
    training_name = config.training_name
    dev_name = config.dev_name
    text_input_folder = os.path.join(config.data_folder,'input/text_files/{}/'.format(inventory))
    input_folder = os.path.join(config.data_folder,'input/matrices/{}/'.format(inventory))
    domains_vocab_path = os.path.join(text_input_folder, 'domains.pkl')
    domains_vocab = pkl.load(open(domains_vocab_path, 'rb'))

    gold_folder = os.path.join(config.data_folder, 'gold/{}/'.format(inventory))
    labels = sorted([x for x in domains_vocab if x!='untagged'])
    labels_dict = {label: k + 1 for k, label in enumerate(labels)}
    labels_dict[None] = 0
    reverse_labels_dict = {v: k for k, v in labels_dict.items()}

    if config.model_name == 'BertLSTM':
        model = BertLSTM(len(labels_dict)).train()

    elif config.model_name == 'BertDense':
        model = BertDense(len(labels_dict), 'bert-large-cased').train()

    criterion = NLLLoss(ignore_index=0)
    optimizer = Adam(model.parameters(), lr=lr)
    writer = SummaryWriter(os.path.join(config.experiment_folder, 'logs'))

    X_train = pkl.load(open(os.path.join(input_folder, "{}_words.pkl".format(training_name)), "rb")).tolist()
    y_train = pkl.load(open(os.path.join(input_folder, '{}_domains.pkl'.format(training_name)), "rb")).tolist()

    X_eval_temp = pkl.load(open(os.path.join(input_folder, "{}_words.pkl".format(dev_name)), "rb")).tolist()
    y_eval = pkl.load(open(os.path.join(input_folder, '{}_domains.pkl'.format(dev_name)), "rb")).tolist()
    y_eval_idx = process_labels(y_eval, labels_dict).cpu().numpy().tolist()
    token_eval = process_tokens(pkl.load(open(os.path.join(input_folder, '{}_tokens.pkl'.format(dev_name)),"rb")).tolist())

    mask_eval = utils.build_mask_one_out(words=X_eval_temp, true_y=y_eval_idx, labels_dict=labels_dict)
    gold_eval = {line.strip().split()[0]: line.strip().split()[1:]
                       for line in open(os.path.join(gold_folder, '{}.gold.txt'.format(dev_name)))}

    if config.start_from_checkpoint:
        load_checkpoints_path = os.path.join(config.experiment_folder, 'models', 'checkpoint_{}.tar'.format(config.starting_epoch))
        model.load_state_dict(torch.load(load_checkpoints_path)['model_state_dict'])
        optimizer.load_state_dict(torch.load(load_checkpoints_path)['optimizer_state_dict'])

    for j in tqdm.tqdm(range(epochs - config.starting_epoch)):
        epoch = j + config.starting_epoch
        if config.starting_epoch > 0:
            epoch += 1
        path_checkpoints = os.path.join(config.experiment_folder, 'weights', 'checkpoint_{}.tar'.format(epoch))
        total, correct = 0, 0
        for i in tqdm.tqdm(range(0, len(X_train), batch_size)):
            inputs = X_train[i:i + batch_size]
            temp_lab = y_train[i:i + batch_size]
            if any(len(sublist) > 100 for sublist in inputs):
                inputs = [x[:100] for x in inputs]
                temp_lab = [x[:100] for x in temp_lab]
            labels = process_labels(temp_lab, labels_dict)
            assert len(inputs) == len(labels)
            optimizer.zero_grad()
            outputs = model.train()(inputs)
            loss = criterion(outputs.view(-1, outputs.size(2)), labels.view(-1))
            loss.backward()
            optimizer.step()

        writer.add_scalar('training_loss', loss.item(), epoch)
        eval_outs = torch.exp(model.eval()(X_eval_temp))
        eval_outs *= mask_eval.cuda()
        predicted = torch.argmax(eval_outs, 2)
        to_pred_indexes = torch.nonzero(process_labels(y_eval, labels_dict)).cuda()

        for a, b in to_pred_indexes:
            instance_id = token_eval[a][b]
            gold_label = gold_eval[instance_id]
            predicted_label = reverse_labels_dict[predicted[a, b].item()]
            if predicted_label in gold_label:
                correct += 1
            total += 1

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'eval_acc': correct / total
        }, path_checkpoints)
        writer.add_scalar('eval_acc', correct / total, epoch)
    writer.close()

def train_model_few_shot(config, k, path_checkpoint, epochs = 20, batch_size = 128, lr=1e-4):

    training_name = config.training_name
    dev_name = config.dev_name
    inventory = config.inventory
    input_folder = os.path.join(config.data_folder, 'input/matrices/{}/'.format(inventory))

    domains_vocab_path = os.path.join(config.all_words_folder, 'domains.pkl')
    domains_vocab = pkl.load(open(domains_vocab_path, 'rb'))

    labels = sorted([x for x in domains_vocab if x!='untagged'])
    labels_dict = {label: k + 1 for k, label in enumerate(labels)}
    labels_dict[None] = 0
    reverse_labels_dict = {v: k for k, v in labels_dict.items()}

    gold_folder = os.path.join(config.data_folder,'gold/{}/'.format(inventory))

    train_x = pkl.load(open(os.path.join(input_folder, "{}/{}_words.pkl".format(k,training_name)), "rb")).tolist()
    train_y = pkl.load(open(os.path.join(input_folder, '{}/{}_domains.pkl'.format(k,training_name)), "rb")).tolist()

    dev_x = pkl.load(open(os.path.join(input_folder, "{}_words.pkl".format(dev_name)), "rb")).tolist()
    dev_y = pkl.load(open(os.path.join(input_folder, '{}_domains.pkl'.format(dev_name)), "rb")).tolist()
    dev_y_idx = process_labels(dev_y, labels_dict).cpu().numpy().tolist()
    tokens = process_tokens(pkl.load(open(os.path.join(input_folder, '{}_tokens.pkl'.format(dev_name)), 'rb')).tolist())

    dev_mask = utils.build_mask_one_out(words=dev_x, true_y=dev_y_idx, labels_dict=labels_dict)

    gold_dictionary = {line.strip().split()[0]: line.strip().split()[1:]
                           for line in open(os.path.join(gold_folder, '{}.gold.txt'.format(dev_name)))}

    if config.model_name=='BertLSTM':
        model = BertLSTM(len(labels_dict)).train()

    elif config.model_name == 'BertDense':
        model = BertDense(len(labels_dict), 'bert-large-cased')

    criterion = NLLLoss(ignore_index=0)
    optimizer = Adam(model.parameters(), lr=lr)
    writer = SummaryWriter(os.path.join(config.experiment_folder, 'logs'))

    checkpoint = torch.load(path_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    for epoch in tqdm.tqdm(range(epochs)):
        path_checkpoints = os.path.join(config.experiment_folder, 'weights', '{}'.format(k), 'checkpoint_{}.tar'.format(epoch))
        total, correct = 0, 0
        for i in tqdm.tqdm(range(0, len(train_x), batch_size)):
            inputs = train_x[i:i + batch_size]
            temp_lab = train_y[i:i + batch_size]
            if any(len(sublist) > 100 for sublist in inputs):
                inputs = [x[:100] for x in inputs]
                temp_lab = [x[:100] for x in temp_lab]
            labels = process_labels(temp_lab, labels_dict)
            assert len(inputs) == len(labels)
            optimizer.zero_grad()
            outputs = model.train()(inputs)
            loss = criterion(outputs.view(-1, outputs.size(2)), labels.view(-1))
            loss.backward()
            optimizer.step()
        writer.add_scalar('training_loss', loss.item(), epoch)
        eval_outs = torch.exp(model.eval()(dev_x))
        eval_outs *= dev_mask.cuda()
        predicted = torch.argmax(eval_outs, 2)
        to_pred_indexes = torch.nonzero(process_labels(dev_y, labels_dict)).cuda()
        for a, b in to_pred_indexes:
            instance_id = tokens[a][b]
            gold_label = gold_dictionary[instance_id]
            predicted_label = reverse_labels_dict[predicted[a, b].item()]
            if predicted_label in gold_label:
                correct += 1
            total += 1

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'eval_acc': correct / total
        }, path_checkpoints)
        writer.add_scalar('eval_acc', correct / total, epoch)
    writer.close()