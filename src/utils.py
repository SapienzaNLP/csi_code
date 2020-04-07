import ast
import os
import pickle

import numpy as np
import torch
import tqdm
import yaml
from nltk.corpus import wordnet as wn


def getMFS(token_id, mapping, txt_file_path, finegrained=False):
    id_lemma = {line.strip().split()[0]: line.strip().split()[1] for line in open(txt_file_path).readlines()}
    lemma = '_'.join(id_lemma[token_id].split('_')[:-1])
    pos = id_lemma[token_id].split('_')[-1]
    pos_map = {"ADJ": "a", "NOUN": "n", "VERB": "v", "ADV": "r"}
    if pos in pos_map.keys() and len(wn.synsets(lemma, pos=pos_map[pos])) > 0:
        synsets = wn.synsets(lemma, pos=pos_map[pos])[0].lemmas()
        for synset in synsets:
            if synset.key().split('%')[0] == lemma:
                mfs_sensekey = synset.key()
                if finegrained:
                    return mfs_sensekey
                if mfs_sensekey in mapping:
                    return mapping[mfs_sensekey][0]
        synsets_all = wn.synsets(lemma, pos=pos_map[pos])
        for wsyn in synsets_all:
            for synset_ in wsyn.lemmas():
                key = synset_.key()
                if key in mapping:
                    if finegrained:
                        return key
                    return mapping[key][0]
    else:
        return ''


def getGoldFilesMapped(path_gold_fg, path_gold_new, mapping_output):
    fw = open(path_gold_new, 'w')
    for line in open(path_gold_fg).readlines():
        line = line.strip()
        token_id = line.split()[0]
        gold = line.split()[1:]
        mapped_gold = set()
        for g in gold:
            if g in mapping_output:
                for element in mapping_output[g]:
                    mapped_gold.add(element)
        if len(mapped_gold) >= 1:
            fw.write(token_id + '\t' + '\t'.join(list(mapped_gold)) + '\n')


def getGoldFilesMappedOneOut(path_gold_fg, path_gold_new, mapping_output, test_lemmas):
    pos_map = {'1': 'NOUN', '2': 'VERB'}
    fw = open(path_gold_new, 'w')
    for line in open(path_gold_fg).readlines():
        line = line.strip()
        token_id = line.split()[0]
        gold = line.split()[1:]
        mapped_gold = set()
        for g in gold:
            if g in mapping_output:
                mapped_g = g.split(':')[0].split('%')[0] + '_' + pos_map[g.split(':')[0].split('%')[1]]
                if mapped_g in test_lemmas:
                    for element in mapping_output[g]:
                        mapped_gold.add(element)
        if len(mapped_gold) >= 1:
            fw.write(token_id + '\t' + '\t'.join(list(mapped_gold)) + '\n')


def define_folders(config):
    models = os.path.join(config.experiment_folder, 'weights')
    gold_folder = os.path.join(config.data_folder, 'gold', config.inventory)
    results_folder = os.path.join(config.experiment_folder, 'results')
    path_logs = os.path.join(config.experiment_folder, "logs")
    text_input_folder = os.path.join(config.data_folder, 'input', 'text_files', config.inventory)
    input_folder = os.path.join(config.data_folder, 'input', 'matrices', config.inventory)
    list_folder = [models, path_logs, gold_folder, text_input_folder, input_folder, results_folder]
    for folder in list_folder:
        if not os.path.exists(folder):
            os.makedirs(folder)


def define_folders_few_shot(config):
    models = os.path.join(config.experiment_folder, 'weights')
    gold_folder = os.path.join(config.data_folder, 'gold', config.inventory)
    results_folder = os.path.join(config.experiment_folder, 'results')
    path_logs = os.path.join(config.experiment_folder, "logs")
    text_input_folder = os.path.join(config.data_folder, 'input', 'text_files', config.inventory)
    input_folder = os.path.join(config.data_folder, 'input', 'matrices', config.inventory)
    temp_list_folder = [models, path_logs, gold_folder, text_input_folder, input_folder]
    list_folder = temp_list_folder.copy()
    list_folder.append(results_folder)
    for folder in list_folder:
        if not os.path.exists(folder):
            os.makedirs(folder)

    for k in ['3', '5', '10']:
        if not os.path.exists(os.path.join(input_folder, k)):
            os.makedirs(os.path.join(input_folder, k))
        if not os.path.exists(os.path.join(models, k)):
            os.makedirs(os.path.join(models, k))
        if not os.path.exists(os.path.join(results_folder, k)):
            os.makedirs(os.path.join(results_folder, k))


def build_possible_senses(labels_dict, file_txt):
    lemma_senses = {}
    for line in open(file_txt).readlines():
        line = line.strip().split('\t')
        lemma_pos = line[1]
        coarse_labs = ast.literal_eval(line[3])
        if len(coarse_labs) >= 1:
            if not lemma_pos in lemma_senses:
                lemma_senses[lemma_pos] = []
            labels = ast.literal_eval(line[3])
            for lab in labels:
                if not lab in lemma_senses[lemma_pos]:
                    lemma_senses[lemma_pos].append(lab)
    lemma_indexes = {}
    for k, v in lemma_senses.items():
        lemma_indexes[k] = []
        for ele in v:
            lemma_indexes[k].append(labels_dict[ele])
    return lemma_indexes


def build_mask(words, true_y, labels_dict, tokens, file_txt, candidate):
    max_len = max([len(sentence) for sentence in words])
    mask = torch.zeros(len(words), max_len, len(labels_dict))
    tokens_dict = {line.strip().split()[0]: line.strip().split()[1] for line in open(file_txt).readlines()}
    idx_untag = labels_dict[None]

    for s, sentence in enumerate(words):
        for w, word in enumerate(sentence):
            if true_y[s][w] != idx_untag:
                lemma_pos = tokens_dict[tokens[s][w]]
                if lemma_pos in candidate:
                    for value in candidate[lemma_pos]:
                        mask[s][w][value] = 1
    return mask


def build_mask_one_out(words, true_y, labels_dict):
    max_len = max([len(sentence) for sentence in words])
    mask = torch.zeros(len(words), max_len, len(labels_dict))
    idx_untag = labels_dict[None]
    for s in range(0, len(words)):
        for w in range(0, len(words[s])):
            if true_y[s][w] != idx_untag:
                mask[s][w][:] = 1
    for i in range(len(mask)):
        for j in range(len(mask[i])):
            mask[i][j][idx_untag] = 0
    return mask


def build_mask_from_training_k(tokens, file_txt, labels_dict, file_input_semcor):
    lemma_semcor_senses = {}
    for line in open(file_input_semcor).readlines():
        line = line.strip().split('\t')
        lemma = line[1]
        doms = clean_string(line[3])
        if not lemma in lemma_semcor_senses:
            lemma_semcor_senses[lemma] = set()
        for lab in doms:
            lemma_semcor_senses[lemma].add(lab)

    max_len = max([len(sentence) for sentence in tokens])
    mask = torch.zeros(len(tokens), max_len, len(labels_dict))
    token2lemma = {line.strip().split('\t')[0]: line.strip().split('\t')[1] for line in open(file_txt).readlines()}
    for sent_idx, sentence_word in enumerate(tokens):
        for token_idx, token in enumerate(sentence_word):
            if not token == 'PADDING' and not token == 'untagged' and not token is None:
                lemma_pos = token2lemma[token]
                domains = lemma_semcor_senses[lemma_pos]
                sense_indexes = [labels_dict[domain] for domain in domains if
                                 domain in labels_dict and not domain is None]
                for k in sense_indexes:
                    mask[sent_idx, token_idx, k] = 1
    return mask


def pick_epoch(exp_folder, k=None):
    best_epoch = 0
    max_acc = 0
    epochs = 40
    for num_ep in range(epochs):
        if k != None:
            path_checkpoints = os.path.join(exp_folder, 'weights', '{}/checkpoint_{}.tar'.format(k, num_ep))
        else:
            path_checkpoints = os.path.join(exp_folder, 'weights', 'checkpoint_{}.tar'.format(num_ep))
        if os.path.exists(path_checkpoints):
            checkpoint = torch.load(path_checkpoints)
            acc_eval = checkpoint['eval_acc']
            print('epoch: {}, accuracy on dev set: {}'.format(num_ep, acc_eval))
            if acc_eval >= max_acc:
                max_acc = acc_eval
                best_epoch = num_ep
        else:
            continue

    for epoch in range(epochs):
        if epoch != best_epoch:
            if k != None:
                delete_path = os.path.join(exp_folder, 'weights', '{}/checkpoint_{}.tar'.format(k, epoch))
            else:
                delete_path = os.path.join(exp_folder, 'weights', 'checkpoint_{}.tar'.format(epoch))
            if os.path.exists(delete_path):
                os.remove(delete_path)
    return best_epoch


def clean_string(string):
    if string == '[]':
        return []
    string = string.replace('[', '').replace(']', '')
    string = string.split(', ')
    new_string = []
    for element in string:
        new_string.append(element[1:-1])
    return new_string


def getSemcorDistribution(path_input_txt_semcor, mapping):
    pos_map = {'NOUN': '1', 'VERB': '2'}
    distribution = {}
    lines = open(path_input_txt_semcor).readlines()
    for line in tqdm.tqdm(lines):
        line = line.strip().split('\t')
        lemma = line[1]
        pos = lemma.split('_')[-1]
        if pos in pos_map:
            lemmak = '_'.join(lemma.split('_')[:-1]) + '%' + pos_map[pos]
            if not lemma in distribution:
                keys = (k for k in mapping if k.startswith(lemmak))
                labels = set([lab for k in keys for lab in mapping[k]])
                distribution[lemma] = len(labels)
    return distribution


def datasetLemmaOccurrences(file_input_txt):
    lemma2taggedInstances = {}
    for line in open(file_input_txt).readlines():
        line = line.strip().split('\t')
        lemma = line[1]
        coarse = clean_string(line[3])
        if not lemma in lemma2taggedInstances:
            lemma2taggedInstances[lemma] = 0
        if len(coarse) >= 1:
            lemma2taggedInstances[lemma] += 1
    return lemma2taggedInstances


def buildOneOutDictionary(config):
    inventory = config.inventory
    all_words_folder = config.all_words_folder
    mapping_coarse_all = pickle.load(open(config.mapping_path, 'rb'))

    one_out_input_folder = os.path.join(config.data_folder, 'input', 'text_files', inventory)

    folder_dicts = os.path.split(config.mapping_path)[0]
    csi = pickle.load(open(os.path.join(folder_dicts, 'sensekey2csi.pkl'), 'rb'))
    wnd = pickle.load(open(os.path.join(folder_dicts, 'sensekey2wndomains.pkl'), 'rb'))
    lex = pickle.load(open(os.path.join(folder_dicts, 'sensekey2supersenses.pkl'), 'rb'))

    if os.path.exists(os.path.join(config.data_folder, 'dev_lemmas.yml')) and \
            os.path.exists(os.path.join(config.data_folder, 'test_lemmas.yml')):
        print('existing lemmas')
        if os.path.exists(os.path.join(config.data_folder, '{}.pkl'.format(inventory))):
            print('existing mapping in', one_out_input_folder)
        else:
            print('building reduced mapping')
            dev_lemmas = yaml.load(open(os.path.join(config.data_folder, 'dev_lemmas.yml')), Loader=yaml.SafeLoader)
            test_lemmas = yaml.load(open(os.path.join(config.data_folder, 'test_lemmas.yml')), Loader=yaml.SafeLoader)
            reverse_pos_map = {'1': 'NOUN', '2': 'VERB'}

            cut_mapping = {}
            for k, v in mapping_coarse_all.items():
                starter = k.split(':')[0]
                lemma_pos = starter.split('%')[0] + '_' + reverse_pos_map[starter.split('%')[1]]

                if not lemma_pos in test_lemmas and not lemma_pos in dev_lemmas:
                    cut_mapping[k] = v
            pickle.dump(cut_mapping, open(os.path.join(config.data_folder, '{}.pkl'.format(inventory)), 'wb'))

        return pickle.load(open(os.path.join(config.data_folder, '{}.pkl'.format(inventory)), 'rb')), \
               yaml.load(open(os.path.join(config.data_folder, 'dev_lemmas.yml')), Loader=yaml.SafeLoader), \
               yaml.load(open(os.path.join(config.data_folder, 'test_lemmas.yml')), Loader=yaml.SafeLoader)

    path_all_words_semcor = os.path.join(all_words_folder, 'semcor_input.txt')
    path_all_words_dev = os.path.join(all_words_folder, '{}_input.txt'.format(config.dev_name))
    semcor_distribution_counts_csi = getSemcorDistribution(path_all_words_semcor,
                                                           csi)  # lemma--> num coarse senses in semcor
    semcor_distribution_counts_wnd = getSemcorDistribution(path_all_words_semcor,
                                                           wnd)  # lemma--> num coarse senses in semcor
    semcor_distribution_counts_lex = getSemcorDistribution(path_all_words_semcor,
                                                           lex)  # lemma--> num coarse senses in semcor

    semcor_occurrences = datasetLemmaOccurrences(path_all_words_semcor)  # lemma --> num tagged occurrences in semcor
    semeval07_occurrences = datasetLemmaOccurrences(path_all_words_dev)
    # polysemous lemmas with at least 10 occurrences in semcor
    lemmas_ten = [x for x in semcor_occurrences if semcor_occurrences[x] >= 10
                  and (x.split('_')[1] == 'NOUN' or x.split('_')[1] == 'VERB')
                  and semcor_distribution_counts_csi[x] > 1
                  and semcor_distribution_counts_wnd[x] > 1
                  and semcor_distribution_counts_lex[x] > 1]

    pos_map = {'NOUN': '1', 'VERB': '2'}
    reverse_pos_map = {'1': 'NOUN', '2': 'VERB'}
    lemma_semcor_semeval = set(semeval07_occurrences.keys()).intersection(set(lemmas_ten))
    print('lemmas both in semcor clean and dev', len(lemma_semcor_semeval))

    dev_lemmas_temp = list(lemma_semcor_semeval)
    test_lemmas_set = set()
    for test in ['senseval2', 'senseval3', 'semeval2007', 'semeval2013', 'semeval2015']:
        lemma_test = datasetLemmaOccurrences(os.path.join(all_words_folder, '{}_input.txt'.format(test)))
        for lemma in lemma_test:
            if lemma in lemmas_ten and not lemma in dev_lemmas_temp:
                test_lemmas_set.add(lemma)

    print('test', len(test_lemmas_set))

    test_lemmas_all = set(dev_lemmas_temp).union(test_lemmas_set)
    keys = [x.split('_')[0] + '%' + pos_map[x.split('_')[1]] for x in test_lemmas_all]

    annotated_lemma_keys = set()
    for k in mapping_coarse_all:
        start = k.split(':')[0]
        if start in keys:
            annotated_lemma_keys.add(start)

    annotated_lemmas = [x.split('%')[0] + '_' + reverse_pos_map[x.split('%')[1]] for x in annotated_lemma_keys]
    dev_lemmas = [x for x in dev_lemmas_temp if x in annotated_lemmas][:30]
    test_lemmas = [x for x in test_lemmas_set if x in annotated_lemmas][:70]
    print('DEV LEMMAS:', len(dev_lemmas))
    print('TEST LEMMAS:', len(test_lemmas))
    assert len(dev_lemmas) == 30 and len(test_lemmas) == 70
    cut_annotated_test_dev_keys = [x for x in annotated_lemma_keys
                                   if x.split('%')[0] + '_' + reverse_pos_map[x.split('%')[1]] in dev_lemmas
                                   or x.split('%')[0] + '_' + reverse_pos_map[x.split('%')[1]] in test_lemmas]
    cut_mapping = {}
    for k, v in mapping_coarse_all.items():
        starter = k.split(':')[0]
        if not starter in cut_annotated_test_dev_keys:
            cut_mapping[k] = v

    pickle.dump(cut_mapping, open(os.path.join(config.data_folder, '{}.pkl'.format(config.inventory)), 'wb'))
    yaml.dump(dev_lemmas, open(os.path.join(config.data_folder, 'dev_lemmas.yml'), 'w'))
    yaml.dump(test_lemmas, open(os.path.join(config.data_folder, 'test_lemmas.yml'), 'w'))
    return cut_mapping, dev_lemmas, test_lemmas


def getPPL(config):
    p, c = 0, 0
    training_input_txt = os.path.join(config.data_folder, 'input', 'text_files', config.inventory, 'semcor_input.txt')
    dict_training_senses = {}
    for line in open(training_input_txt).readlines():
        line = line.strip().split('\t')
        if line[3] != '[]':
            lemma = line[1]
            coarse = line[3].replace("['", '').replace("']", '')
            if not lemma in dict_training_senses:
                dict_training_senses[lemma] = set()
            dict_training_senses[lemma].add(coarse)

    print('\nPerplexity')
    for test in config.tests:
        output_file = os.path.join(config.experiment_folder, 'results', '{}_output.tsv'.format(test))
        count = 0
        ppl = 0
        for line in open(output_file).readlines():
            line = line.strip().split('\t')
            lemma_pos = line[2]
            gold = [x.split('gold##')[1] for x in line[4].split(' ')]
            if len(line) > 5:
                possible = dict_training_senses[lemma_pos]
                count += 1
                ppl += len(possible) / len(gold)
        print(test, np.round(ppl / count, 3))
        if test != config.dev_name:
            p += ppl
            c += count
    print('ALL', np.round(p / c, 3))
