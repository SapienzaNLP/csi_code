import os
import tqdm
import torch
import numpy as np
import pickle as pkl
import src.utils as utils
from torch.optim import Adam
from src.model_bert import BertLSTM, BertDense
from src.training import process_labels, process_tokens

def test(config, epoch_to_load, lr=1e-4, batch_size = 64):
    inventory = config.inventory
    gold_folder = os.path.join(config.data_folder, 'gold/{}/'.format(inventory))

    test_list = config.tests
    exp_folder = config.experiment_folder
    dev_name = config.dev_name
    text_input_folder = os.path.join(config.data_folder, 'input/text_files/{}/'.format(inventory))
    input_folder = os.path.join(config.data_folder, 'input/matrices/{}/'.format(inventory))

    mapping = pkl.load(open(config.mapping_path,'rb'))
    domains_vocab_path = os.path.join(text_input_folder, 'domains.pkl')
    if config.finegrained:
        domains_vocab_path = os.path.join(text_input_folder, 'sensekeys.pkl')

    domains_vocab = pkl.load(open(domains_vocab_path, 'rb'))
    results_folder = os.path.join(exp_folder, 'results/')

    labels = sorted([x for x in domains_vocab if x != 'untagged'])
    labels_dict = {label: k + 1 for k, label in enumerate(labels)}
    labels_dict[None] = 0
    reverse_labels_dict = {v: k for k, v in labels_dict.items()}

    if config.model_name == 'BertLSTM':
        model = BertLSTM(len(labels_dict))

    elif config.model_name == 'BertDense':
        model = BertDense(len(labels_dict), 'bert-large-cased')

    optimizer = Adam(model.parameters(), lr=lr)
    path_checkpoints = os.path.join(config.experiment_folder, 'weights', 'checkpoint_{}.tar'.format(epoch_to_load))
    checkpoint = torch.load(path_checkpoints)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()

    cor, tot = 0, 0
    wrong_labels_distr = {}
    for name in test_list:
        x = pkl.load(open(os.path.join(input_folder, '{}_words.pkl'.format(name)), 'rb')).tolist()
        y = pkl.load(open(os.path.join(input_folder, '{}_domains.pkl'.format(name)), 'rb')).tolist()

        if config.finegrained:
            y = pkl.load(open(os.path.join(input_folder, '{}_sensekeys.pkl'.format(name)), 'rb')).tolist()

        y_idx = process_labels(y, labels_dict).cpu().numpy().tolist()
        tokens = process_tokens(pkl.load(open(os.path.join(input_folder, '{}_tokens.pkl'.format(name)), 'rb')).tolist())

        file_txt = os.path.join(text_input_folder, '{}_input.txt'.format(name))
        token2lemma = {line.strip().split()[0]: line.strip().split()[1] for line in open(file_txt).readlines()}
        output_file = os.path.join(results_folder, '{}_output.tsv'.format(name))
        candidate_domains = utils.build_possible_senses(labels_dict, os.path.join(text_input_folder, 'semcor_input.txt'))

        gold_dictionary = {line.strip().split()[0]: line.strip().split()[1:]
                           for line in open(os.path.join(gold_folder, '{}.gold.txt'.format(name)))}

        c, t = 0, 0
        with open(output_file, 'w') as fw:
            for i in tqdm.tqdm(range(0, len(x), batch_size)):
                inputs = x[i:i + batch_size]
                labels_idx = y_idx[i:i + batch_size]
                token_batch = tokens[i:i + batch_size]
                mask = utils.build_mask(words=inputs, true_y=labels_idx, labels_dict=labels_dict, tokens=token_batch,
                                        file_txt=file_txt, candidate=candidate_domains)

                eval_out = torch.exp(model.eval()(inputs))
                eval_out *= mask.cuda()
                values, predicted = torch.max(eval_out, 2)
                for id_sent, token_sent in enumerate(token_batch):
                    for id_word, token in enumerate(token_sent):
                        if not token is None and not token == 'untagged':
                            gold_label = gold_dictionary[token]
                            pred_label = reverse_labels_dict[predicted[id_sent, id_word].item()]
                            if pred_label == None:
                                pred_label = utils.getMFS(token, mapping, file_txt)
                                scored_labels = None
                            else:
                                scores_idx = torch.nonzero(eval_out[id_sent,id_word])
                                scored_ = [(reverse_labels_dict[idx.item()], eval_out[id_sent,id_word,idx].item()) for idx in scores_idx]
                                scored_labels = sorted(scored_, key=lambda x:x[1], reverse=True)
                            if pred_label in gold_label:
                                c += 1
                                fw.write('c\t')
                            else:
                                if not scored_labels==None:
                                    for gl in gold_label:
                                        if not gl in wrong_labels_distr:
                                            wrong_labels_distr[gl] = {}
                                        if not pred_label in wrong_labels_distr[gl]:
                                            wrong_labels_distr[gl][pred_label] = 0
                                        wrong_labels_distr[gl][pred_label] += 1
                                fw.write('w\t')
                            t += 1
                            fw.write(name+'.'+ token + '\t' + token2lemma[token] + '\t' + 'pred##'+pred_label +'\t')
                            fw.write(' '.join(['gold##'+gl for gl in gold_label])+'\t')
                            if not scored_labels is None:
                                fw.write(' '.join([label + '##' + str(score) for label, score in scored_labels])+'\t')
                                content_words = [word if index!=id_word else '<tag>{}</tag>'.format(word)
                                                 for index,word in enumerate(x[id_sent]) if word!='PADDING']
                                fw.write(' '.join(content_words))
                            fw.write('\n')
        if name != dev_name:
            cor += c
            tot += t
        f1 = np.round(c / t, 3)
        print(name, f1)
    print('ALL', np.round(cor / tot, 3))

def test_one_out(config, epoch_to_load, lr=1e-4):
    data_folder = config.data_folder
    exp_folder = config.experiment_folder
    inventory = config.inventory
    gold_folder = os.path.join(data_folder, 'gold/{}/'.format(inventory))
    test_list = config.tests
    dev_name = config.dev_name
    text_input_folder = os.path.join(data_folder, 'input/text_files/{}/'.format(inventory))
    input_folder = os.path.join(data_folder,'input/matrices/{}/'.format(inventory))

    domains_vocab_path = os.path.join(text_input_folder, 'domains.pkl')
    domains_vocab = pkl.load(open(domains_vocab_path, 'rb'))
    results_folder = os.path.join(exp_folder, 'results/')

    labels = sorted([x for x in domains_vocab if x!='untagged'])
    labels_dict = {label: k + 1 for k, label in enumerate(labels)}
    labels_dict[None] = 0
    reverse_labels_dict = {v: k for k, v in labels_dict.items()}

    if config.model_name=='BertDense':
        model = BertDense(len(labels_dict), 'bert-large-cased')

    elif config.model_name=='BertLSTM':
        model = BertLSTM(len(labels_dict))

    optimizer = Adam(model.parameters(), lr=lr)
    path_checkpoints = os.path.join(exp_folder, 'weights', 'checkpoint_{}.tar'.format(epoch_to_load))
    checkpoint = torch.load(path_checkpoints)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()
    cor, tot = 0, 0
    for name in test_list:
        x = pkl.load(open(os.path.join(input_folder, '{}_words.pkl'.format(name)), 'rb')).tolist()
        y = pkl.load(open(os.path.join(input_folder, '{}_domains.pkl'.format(name)), 'rb')).tolist()
        y_idx = process_labels(y, labels_dict).cpu().numpy().tolist()
        tokens = process_tokens(pkl.load(open(os.path.join(input_folder, '{}_tokens.pkl'.format(name)), 'rb')).tolist())

        file_txt = os.path.join(text_input_folder, '{}_input.txt'.format(name))
        token2lemma =  {line.strip().split()[0]:line.strip().split()[1] for line in open(file_txt).readlines()}
        output_file = os.path.join(results_folder,'{}_output.tsv'.format(name))
        fw = open(output_file, 'w')
        mask = utils.build_mask_one_out(x, y_idx, labels_dict)
        gold_dictionary = {line.strip().split()[0]: line.strip().split()[1:]
                            for line in open(os.path.join(gold_folder, '{}.gold.txt'.format(name)))}
        eval_out = torch.exp(model.eval()(x))
        eval_out *= mask.cuda()
        predicted = torch.argmax(eval_out, 2)
        to_pred_idx = torch.nonzero(process_labels(y, labels_dict)).cuda()
        c, t = 0, 0
        for a,b in to_pred_idx:
            instance_id = tokens[a][b]
            gold_label = gold_dictionary[instance_id]
            pred_label = reverse_labels_dict[predicted[a,b].item()]
            if pred_label in gold_label:
                c+=1
                fw.write('c\t')
            else:
                fw.write('w\t')
            t+=1
            scores_idx = torch.nonzero(eval_out[a, b])
            scored_ = [(reverse_labels_dict[idx.item()], eval_out[a, b, idx].item()) for idx in scores_idx]
            scored_labels = sorted(scored_, key=lambda x: x[1], reverse=True)
            fw.write(name+'.'+instance_id + '\t' + token2lemma[instance_id] + '\t' + 'pred##'+pred_label +'\t')
            fw.write(' '.join(['gold##'+gl for gl in gold_label])+'\t')
            fw.write(' '.join([label + '##' + str(score) for label,score in scored_labels])+'\n')
        if name != dev_name:
            cor += c
            tot += t
        f1 = np.round(c / t, 3)
        print(name, f1)
    print('ALL', np.round(cor/tot, 3))

def test_few_shot(config, epoch_to_load, k, lr=1e-4):

    inventory = config.inventory
    data_folder = config.data_folder
    exp_folder = config.experiment_folder
    test_list = config.tests
    dev_name = config.dev_name

    gold_folder = os.path.join(data_folder,'gold/{}/'.format(inventory))
    text_input_folder = os.path.join(data_folder,'input/text_files/{}/'.format(inventory))
    input_folder = os.path.join(data_folder, 'input/matrices/{}/'.format(inventory))
    input_semcor_k = os.path.join(text_input_folder, 'semcor_input_{}.txt'.format(k))

    domains_vocab_path = os.path.join(config.all_words_folder, 'domains.pkl')
    domains_vocab = pkl.load(open(domains_vocab_path, 'rb'))
    results_folder = os.path.join(exp_folder, 'results/{}/'.format(k))

    labels = sorted([x for x in domains_vocab if x!='untagged'])
    labels_dict = {label: k + 1 for k, label in enumerate(labels)}
    labels_dict[None] = 0
    reverse_labels_dict = {v: k for k, v in labels_dict.items()}

    if config.model_name == 'BertLSTM':
        model = BertLSTM(len(labels_dict))

    elif config.model_name == 'BertDense':
         model = BertDense(len(labels_dict))

    optimizer = Adam(model.parameters(), lr=lr)

    path_checkpoints = os.path.join(exp_folder, 'weights', '{}'.format(k), 'checkpoint_{}.tar'.format(epoch_to_load))
    checkpoint = torch.load(path_checkpoints)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()

    cor, tot = 0, 0
    for name in test_list:
        x = pkl.load(open(os.path.join(input_folder, '{}_words.pkl'.format(name)), 'rb')).tolist()
        y = pkl.load(open(os.path.join(input_folder, '{}_domains.pkl'.format(name)), 'rb')).tolist()
        file_txt = os.path.join(text_input_folder, '{}_input.txt'.format(name))

        tokens = process_tokens(pkl.load(open(os.path.join(input_folder, '{}_tokens.pkl'.format(name)), 'rb')).tolist())
        token2lemma = {line.strip().split()[0]: line.strip().split()[1] for line in open(file_txt).readlines()}
        output_file = os.path.join(results_folder,'{}_output.tsv'.format(name))
        fw = open(output_file, 'w')

        mask = utils.build_mask_from_training_k(tokens, file_txt, labels_dict, input_semcor_k)
        gold_dictionary = {line.strip().split()[0]: line.strip().split()[1:]
                            for line in open(os.path.join(gold_folder, '{}.gold.txt'.format(name)))}

        eval_out = torch.exp(model.eval()(x))
        eval_out *= mask.cuda()
        predicted = torch.argmax(eval_out, 2)
        to_pred_idx = torch.nonzero(process_labels(y, labels_dict)).cuda()
        c, t = 0, 0
        for a, b in to_pred_idx:
            instance_id = tokens[a][b]
            gold_label = gold_dictionary[instance_id]
            pred_label = reverse_labels_dict[predicted[a, b].item()]
            if pred_label in gold_label:
                c += 1
                fw.write('c\t')
            else:
                fw.write('w\t')
            t += 1
            scores_idx = torch.nonzero(eval_out[a, b])
            scored_ = [(reverse_labels_dict[idx.item()], eval_out[a, b, idx].item()) for idx in scores_idx]
            scored_labels = sorted(scored_, key=lambda x: x[1], reverse=True)
            fw.write(name+'.'+instance_id + '\t' + token2lemma[instance_id] + '\t' + 'pred##'+pred_label +'\t')
            fw.write(' '.join(['gold##'+gl for gl in gold_label])+'\t')
            fw.write(' '.join([label + '##' + str(score) for label,score in scored_labels])+'\n')

        if name != dev_name:
            cor += c
            tot += t

        f1 = np.round(c / t, 3)
        print(name, f1)
    print('ALL', np.round(cor / tot, 3))
