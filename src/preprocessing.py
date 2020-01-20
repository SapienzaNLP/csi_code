import os
import yaml
import random
import numpy as np
import pickle as pkl
import src.utils as utils
import xml.etree.ElementTree as ET

random.seed(42)

def readXML(name, xml_path, sensekey2domains, output_file, gold_file, input_folder_text, training):
    input_file = open(os.path.join(input_folder_text,'{}.sentences.tsv'.format(name)), 'w')
    input_words = list()
    sensekeys = list()
    domains = list()
    input_tokens = list()
    senses_vocab = set()
    domains_vocab = set()
    gold_dict = {line.strip().split()[0]:line.strip().split()[1:] for line in open(gold_file).readlines()}
    file_out = open(output_file,'w')
    tree = ET.parse(xml_path)
    root = tree.getroot()
    counter = 0
    for text in root.findall('text'):
        for sentence in text.findall('sentence'):
            sentence_words = []
            sentence_keys = []
            sentence_domains = []
            sentence_tokens = []
            for element in sentence.iter():
                if element.tag=='instance' or element.tag=='wf':
                    if 'id' in element.attrib:
                        senses_fg_temp = gold_dict[element.attrib['id']]
                        senses_cg = set()
                        senses_fg = set()
                        if any(label in sensekey2domains for label in senses_fg_temp):
                            for label in senses_fg_temp:
                                if label in sensekey2domains:
                                    senses_fg.add(label)
                                    for ele in sensekey2domains[label]:
                                        senses_cg.add(ele)
                        senses_cg = list(senses_cg)
                        senses_fg = list(senses_fg)
                        sentence_words.append(element.text)
                        if len(senses_cg)>0:
                            sentence_tokens.append(element.attrib['id'])
                            sentence_keys.append(random.choice(senses_fg))
                            sentence_domains.append(random.choice(senses_cg))
                            for k in senses_cg:
                                domains_vocab.add(k)
                            for k in senses_fg:
                                senses_vocab.add(k)
                        else:
                            sentence_keys.append('untagged')
                            sentence_tokens.append('untagged')
                            sentence_domains.append('untagged')
                        counter+=1
                        file_out.write(element.attrib['id'] + '\t' +
                                       element.attrib['lemma'] + '_' + element.attrib['pos'] + '\t' +
                                       str(senses_fg) + '\t' + str(senses_cg) + '\n')
                    else:
                        sentence_words.append(element.text)
                        sentence_keys.append('untagged')
                        sentence_domains.append('untagged')
                        sentence_tokens.append('untagged')
            input_words.append(sentence_words)
            sensekeys.append(sentence_keys)
            domains.append(sentence_domains)
            input_tokens.append(sentence_tokens)
            for idx_target, tok in enumerate(sentence_tokens):
                if tok!='untagged':
                    input_file.write(tok + '\t' + sentence_words[idx_target] + '\t')
                    input_file.write(' '.join([sentence_words[x] if x!=idx_target else '<tag>{}</tag>'.format(sentence_words[x])
                                               for x in range(len(sentence_words))]) + '\n')

    print('input words',counter)
    pkl.dump(input_words, open(os.path.join(input_folder_text, '{}_input_words.pkl'.format(name)), 'wb'))
    pkl.dump(sensekeys, open(os.path.join(input_folder_text, '{}_input_senses.pkl'.format(name)), 'wb'))
    pkl.dump(domains, open(os.path.join(input_folder_text, '{}_input_domains.pkl'.format(name)), 'wb'))
    pkl.dump(input_tokens, open(os.path.join(input_folder_text, '{}_input_tokens.pkl'.format(name)), 'wb'))

    if training:
        input_vocabulary = set()
        for sublist in input_words:
            subset = set(sublist)
            for element in subset:
                input_vocabulary.add(element)
        input_vocabulary.add('PADDING')
        input_vocabulary.add('NUMBER')
        input_vocabulary = list(input_vocabulary)
        input_vocabulary.sort()

        senses_vocab.add('untagged')
        senses_vocab = list(senses_vocab)
        senses_vocab.sort()

        domains_vocab.add('untagged')
        domains_vocab = list(domains_vocab)
        domains_vocab.sort()

        print('input vocabulary', len(input_vocabulary), 'senses inventory', len(senses_vocab), 'coarse sense inventory', len(domains_vocab))

        pkl.dump(input_vocabulary, open(os.path.join(input_folder_text, 'vocabulary.pkl'), 'wb'))
        pkl.dump(senses_vocab, open(os.path.join(input_folder_text, 'sensekeys.pkl'), 'wb'))
        pkl.dump(domains_vocab, open(os.path.join(input_folder_text, 'domains.pkl'), 'wb'))
        print('vocabularies dump')

def readXMLoneOut(name, xml_path, sensekey2domains, sensekey2domains_all, output_file, gold_file, input_folder_text, training, test_lemmas):
    input_words = list()
    sensekeys = list()
    domains = list()
    input_tokens = list()
    senses_vocab = set()
    domains_vocab = set()
    gold_dict = {line.strip().split()[0]:line.strip().split()[1:] for line in open(gold_file).readlines()}
    file_out = open(output_file,'w')
    tree = ET.parse(xml_path)
    root = tree.getroot()
    counter = 0
    if training:
        for text in root.findall('text'):
            for sentence in text.findall('sentence'):
                sentence_words = []
                sentence_keys = []
                sentence_domains = []
                sentence_tokens = []
                for element in sentence.iter():
                    if element.tag=='instance' or element.tag=='wf':
                        if 'id' in element.attrib:
                            senses_fg_temp = gold_dict[element.attrib['id']]
                            senses_cg = set()
                            senses_fg = set()
                            if any(label in sensekey2domains for label in senses_fg_temp):
                                for label in senses_fg_temp:
                                    if label in sensekey2domains:
                                        senses_fg.add(label)
                                        for ele in sensekey2domains[label]:
                                            senses_cg.add(ele)
                            senses_cg = list(senses_cg)
                            senses_fg = list(senses_fg)
                            sentence_words.append(element.text)
                            if len(senses_cg)>0:
                                sentence_tokens.append(element.attrib['id'])
                                sentence_keys.append(random.choice(senses_fg))
                                sentence_domains.append(random.choice(senses_cg))
                                for k in senses_cg:
                                    domains_vocab.add(k)
                                for k in senses_fg:
                                    senses_vocab.add(k)
                            else:
                                sentence_keys.append('untagged')
                                sentence_tokens.append('untagged')
                                sentence_domains.append('untagged')
                            counter+=1
                            file_out.write(element.attrib['id'] + '\t' +
                                           element.attrib['lemma'] + '_' + element.attrib['pos'] + '\t' +
                                           str(senses_fg) + '\t' + str(senses_cg) + '\n')
                        else:
                            sentence_words.append(element.text)
                            sentence_keys.append('untagged')
                            sentence_domains.append('untagged')
                            sentence_tokens.append('untagged')
                input_words.append(sentence_words)
                sensekeys.append(sentence_keys)
                domains.append(sentence_domains)
                input_tokens.append(sentence_tokens)
    else:
        pos_map = {'NOUN': '1', 'VERB': '2'}
        test_keys = [x.split('_')[0] + '%' + pos_map[x.split('_')[1]] for x in test_lemmas]
        for text in root.findall('text'):
            for sentence in text.findall('sentence'):
                sentence_words = []
                sentence_keys = []
                sentence_domains = []
                sentence_tokens = []
                for element in sentence.iter():
                    if element.tag=='instance' or element.tag=='wf':
                        if 'id' in element.attrib:
                            senses_fg_temp = gold_dict[element.attrib['id']]
                            senses_cg = set()
                            senses_fg = set()
                            if any(label.split(':')[0] in test_keys for label in senses_fg_temp):
                                for label in senses_fg_temp:
                                    if label.split(':')[0] in test_keys and label in sensekey2domains_all:
                                        senses_fg.add(label)
                                        for ele in sensekey2domains_all[label]:
                                            senses_cg.add(ele)
                            senses_cg = list(senses_cg)
                            senses_fg = list(senses_fg)
                            sentence_words.append(element.text)
                            if len(senses_cg)>0:
                                sentence_tokens.append(element.attrib['id'])
                                sentence_keys.append(random.choice(senses_fg))
                                sentence_domains.append(random.choice(senses_cg))
                                for k in senses_cg:
                                    domains_vocab.add(k)
                                for k in senses_fg:
                                    senses_vocab.add(k)
                            else:
                                sentence_keys.append('untagged')
                                sentence_tokens.append('untagged')
                                sentence_domains.append('untagged')
                            counter+=1
                            file_out.write(element.attrib['id'] + '\t' +
                                           element.attrib['lemma'] + '_' + element.attrib['pos'] + '\t' +
                                           str(senses_fg) + '\t' + str(senses_cg) + '\n')
                        else:
                            sentence_words.append(element.text)
                            sentence_keys.append('untagged')
                            sentence_domains.append('untagged')
                            sentence_tokens.append('untagged')
                input_words.append(sentence_words)
                sensekeys.append(sentence_keys)
                domains.append(sentence_domains)
                input_tokens.append(sentence_tokens)
    print('input words',counter)

    pkl.dump(input_words, open(os.path.join(input_folder_text, '{}_input_words.pkl'.format(name)), 'wb'))
    pkl.dump(sensekeys, open(os.path.join(input_folder_text, '{}_input_senses.pkl'.format(name)), 'wb'))
    pkl.dump(domains, open(os.path.join(input_folder_text, '{}_input_domains.pkl'.format(name)), 'wb'))
    pkl.dump(input_tokens, open(os.path.join(input_folder_text, '{}_input_tokens.pkl'.format(name)), 'wb'))


    if training:
        input_vocabulary = set()
        for sublist in input_words:
            subset = set(sublist)
            for element in subset:
                input_vocabulary.add(element)
        input_vocabulary.add('PADDING')
        input_vocabulary.add('NUMBER')
        input_vocabulary = list(input_vocabulary)
        input_vocabulary.sort()

        senses_vocab.add('untagged')
        senses_vocab = list(senses_vocab)
        senses_vocab.sort()

        domains_vocab.add('untagged')
        domains_vocab = list(domains_vocab)
        domains_vocab.sort()

        print('input vocabulary', len(input_vocabulary), 'senses inventory', len(senses_vocab), 'coarse sense inventory', len(domains_vocab))

        pkl.dump(input_vocabulary, open(os.path.join(input_folder_text, 'vocabulary.pkl'), 'wb'))
        pkl.dump(senses_vocab, open(os.path.join(input_folder_text, 'sensekeys.pkl'), 'wb'))
        pkl.dump(domains_vocab, open(os.path.join(input_folder_text, 'domains.pkl'), 'wb'))
        print('vocabularies dump')

def readXMLFewShot(name, xml_path, sensekey2domains, output_file, gold_file, input_folder_text, training, test_lemmas, dev_lemmas, k):
    input_words = list()
    sensekeys = list()
    domains = list()
    input_tokens = list()
    gold_dict = {line.strip().split()[0]:line.strip().split()[1:] for line in open(gold_file).readlines()}
    file_out = open(output_file,'w')
    tree = ET.parse(xml_path)
    root = tree.getroot()
    counter = 0
    pos_map = {'NOUN': '1', 'VERB': '2'}
    test_keys = [x.split('_')[0] + '%' + pos_map[x.split('_')[1]] for x in test_lemmas]

    counter_occ = {}
    if training:
        dev_keys = [x.split('_')[0] + '%' + pos_map[x.split('_')[1]] for x in dev_lemmas]

        for text in root.findall('text'):
            for sentence in text.findall('sentence'):
                sentence_words = []
                sentence_keys = []
                sentence_domains = []
                sentence_tokens = []
                for element in sentence.iter():
                    if element.tag=='instance' or element.tag=='wf':
                        if 'id' in element.attrib:
                            lemma_pos = element.attrib['lemma'] + '_' + element.attrib['pos']
                            senses_fg_temp = gold_dict[element.attrib['id']]
                            senses_cg = set()
                            senses_fg = set()
                            if any(label in sensekey2domains for label in senses_fg_temp):
                                for label in senses_fg_temp:
                                    if label in sensekey2domains:
                                        if label.split(':')[0] in test_keys or label.split(':')[0] in dev_keys:
                                            if not lemma_pos in counter_occ:
                                                counter_occ[lemma_pos]=0
                                            if counter_occ[lemma_pos]<k:
                                                senses_fg.add(label)
                                                for ele in sensekey2domains[label]:
                                                    senses_cg.add(ele)
                                                counter_occ[lemma_pos]+=1

                            senses_cg = list(senses_cg)
                            senses_fg = list(senses_fg)
                            sentence_words.append(element.text)
                            if len(senses_cg)>0:
                                sentence_tokens.append(element.attrib['id'])
                                sentence_keys.append(random.choice(senses_fg))
                                sentence_domains.append(random.choice(senses_cg))
                            else:
                                sentence_keys.append('untagged')
                                sentence_tokens.append('untagged')
                                sentence_domains.append('untagged')
                            counter+=1
                            if senses_cg!=[]:
                                file_out.write(element.attrib['id'] + '\t' +
                                               element.attrib['lemma'] + '_' + element.attrib['pos'] + '\t' +
                                               str(senses_fg) + '\t' + str(senses_cg) + '\n')
                        else:
                            sentence_words.append(element.text)
                            sentence_keys.append('untagged')
                            sentence_domains.append('untagged')
                            sentence_tokens.append('untagged')
                input_words.append(sentence_words)
                sensekeys.append(sentence_keys)
                domains.append(sentence_domains)
                input_tokens.append(sentence_tokens)
        print('input words', counter)

        pkl.dump(input_words, open(os.path.join(input_folder_text, '{}_input_words_{}.pkl'.format(name,k)), 'wb'))
        pkl.dump(sensekeys, open(os.path.join(input_folder_text, '{}_input_senses_{}.pkl'.format(name,k)), 'wb'))
        pkl.dump(domains, open(os.path.join(input_folder_text, '{}_input_domains_{}.pkl'.format(name,k)), 'wb'))
        pkl.dump(input_tokens, open(os.path.join(input_folder_text, '{}_input_tokens_{}.pkl'.format(name,k)), 'wb'))

    else:
        pos_map = {'NOUN': '1', 'VERB': '2'}
        test_keys = [x.split('_')[0] + '%' + pos_map[x.split('_')[1]] for x in test_lemmas]
        for text in root.findall('text'):
            for sentence in text.findall('sentence'):
                sentence_words = []
                sentence_keys = []
                sentence_domains = []
                sentence_tokens = []
                for element in sentence.iter():
                    if element.tag=='instance' or element.tag=='wf':
                        if 'id' in element.attrib:
                            senses_fg_temp = gold_dict[element.attrib['id']]
                            senses_cg = set()
                            senses_fg = set()
                            if any(label.split(':')[0] in test_keys for label in senses_fg_temp):
                                for label in senses_fg_temp:
                                    if label.split(':')[0] in test_keys and label in sensekey2domains:
                                        senses_fg.add(label)
                                        for ele in sensekey2domains[label]:
                                            senses_cg.add(ele)
                            senses_cg = list(senses_cg)
                            senses_fg = list(senses_fg)
                            sentence_words.append(element.text)
                            if len(senses_cg)>0:
                                sentence_tokens.append(element.attrib['id'])
                                sentence_keys.append(random.choice(senses_fg))
                                sentence_domains.append(random.choice(senses_cg))
                            else:
                                sentence_keys.append('untagged')
                                sentence_tokens.append('untagged')
                                sentence_domains.append('untagged')

                            if senses_cg!=[]:
                                file_out.write(element.attrib['id'] + '\t' +
                                           element.attrib['lemma'] + '_' + element.attrib['pos'] + '\t' +
                                           str(senses_fg) + '\t' + str(senses_cg) + '\n')
                                counter+=1
                        else:
                            sentence_words.append(element.text)
                            sentence_keys.append('untagged')
                            sentence_domains.append('untagged')
                            sentence_tokens.append('untagged')
                input_words.append(sentence_words)
                sensekeys.append(sentence_keys)
                domains.append(sentence_domains)
                input_tokens.append(sentence_tokens)
        print('input words',counter)

        pkl.dump(input_words, open(os.path.join(input_folder_text, '{}_input_words.pkl'.format(name)), 'wb'))
        pkl.dump(sensekeys, open(os.path.join(input_folder_text, '{}_input_senses.pkl'.format(name)), 'wb'))
        pkl.dump(domains, open(os.path.join(input_folder_text, '{}_input_domains.pkl'.format(name)), 'wb'))
        pkl.dump(input_tokens, open(os.path.join(input_folder_text, '{}_input_tokens.pkl'.format(name)), 'wb'))

def buildInputSequences(sequence_list_path, vocabulary_set, training=False, words=False, tokens=False):
    dict_vocabulary_set = {}
    for idx in range(len(vocabulary_set)):
        dict_vocabulary_set[vocabulary_set[idx]] = idx
    if not type(sequence_list_path)==list:
        sequence_list = pkl.load(open(sequence_list_path,'rb'))
    else:
        sequence_list = sequence_list_path
    if tokens or words:
        return sequence_list

    indexes_all = []
    for sequence_id, sequence in enumerate(sequence_list):
        indexes = np.zeros(len(sequence))
        for label_id, label in enumerate(sequence):
            if not training:
                if not label in dict_vocabulary_set:
                    indexes[label_id] = dict_vocabulary_set['untagged']
                elif label == 'PADDING' or label=='untagged':
                    indexes[label_id] = dict_vocabulary_set['untagged']
                else:
                    indexes[label_id] = dict_vocabulary_set[label]

            else:
                if label=='PADDING' or label=='untagged':
                    indexes[label_id] = dict_vocabulary_set['untagged']
                else:
                    indexes[label_id] = dict_vocabulary_set[label]
        indexes_all.append(indexes)
    return sequence_list, indexes_all

def clean_input_all(words,  domains, domains_idx, sensekeys, sensekeys_idx, tokens, folder_dump, set_name):
    deleted_rows = list()
    for line_idx in range(len(words)):
        if not any(x!='untagged' and x!='PADDING' for x in domains[line_idx]):
            deleted_rows.append(line_idx)
    deleted_rows = tuple(deleted_rows)
    words_clean = np.delete(np.asarray(words), (deleted_rows), axis=0)
    domains_clean = np.delete(np.asarray(domains), (deleted_rows), axis=0)
    domains_idx_clean = np.delete(np.asarray(domains_idx), (deleted_rows), axis=0)
    sensekeys_clean = np.delete(np.asarray(sensekeys), (deleted_rows), axis=0)
    sensekeys_idx_clean = np.delete(np.asarray(sensekeys_idx), (deleted_rows), axis=0)
    tokens_clean = np.delete(np.asarray(tokens), (deleted_rows), axis=0)
    print('clean words shape',words_clean.shape)
    pkl.dump(words_clean, open(os.path.join(folder_dump, '{}_words.pkl'.format(set_name)),'wb'))
    pkl.dump(domains_clean, open(os.path.join(folder_dump, '{}_domains.pkl'.format(set_name)),'wb'))
    pkl.dump(domains_idx_clean, open(os.path.join(folder_dump, '{}_domains_idx.pkl'.format(set_name)),'wb'))
    pkl.dump(sensekeys_idx_clean, open(os.path.join(folder_dump, '{}_sensekeys_idx.pkl'.format(set_name)),'wb'))
    pkl.dump(sensekeys_clean, open(os.path.join(folder_dump, '{}_sensekeys.pkl'.format(set_name)),'wb'))
    pkl.dump(tokens_clean, open(os.path.join(folder_dump,'{}_tokens.pkl'.format(set_name)),'wb'))

def clean_input(words,  domains, domains_idx, tokens, folder_dump, set_name):
    deleted_rows = list()
    for line_idx in range(len(words)):
        if not any(x!='untagged' and x!='PADDING' for x in domains[line_idx]):
            deleted_rows.append(line_idx)
    deleted_rows = tuple(deleted_rows)
    words_clean = np.delete(words, (deleted_rows), axis=0)
    domains_clean = np.delete(domains, (deleted_rows), axis=0)
    domains_idx_clean = np.delete(domains_idx, (deleted_rows), axis=0)
    tokens_clean = np.delete(tokens, (deleted_rows), axis=0)
    print('clean words shape',words_clean.shape)
    pkl.dump(words_clean, open(os.path.join(folder_dump, '{}_words.pkl'.format(set_name)),'wb'))
    pkl.dump(domains_clean, open(os.path.join(folder_dump, '{}_domains.pkl'.format(set_name)),'wb'))
    pkl.dump(domains_idx_clean, open(os.path.join(folder_dump, '{}_domains_idx.pkl'.format(set_name)),'wb'))
    pkl.dump(tokens_clean, open(os.path.join(folder_dump, '{}_tokens.pkl'.format(set_name)),'wb'))


def create_data(config):
    output_mapping = pkl.load(open(config.mapping_path, 'rb'))
    text_input_folder = os.path.join(config.data_folder, 'input', 'text_files', config.inventory)
    input_folder = os.path.join(config.data_folder, 'input', 'matrices', config.inventory)
    test_list = [config.training_name]
    for x in config.tests:
        test_list.append(x)
    print('creating data for', test_list)
    for set_name in test_list:
        print('creating data for', set_name)
        if set_name==config.training_name:
            training = True
        else:
            training = False

        print('parsing xml file')
        readXML(set_name, os.path.join(config.wsd_data_folder, '{}/{}.data.xml'.format(set_name, set_name)),
                output_mapping, os.path.join(text_input_folder, '{}_input.txt'.format(set_name)),
                os.path.join(config.wsd_data_folder, '{}/{}.gold.key.txt'.format(set_name, set_name)),
                text_input_folder, training=training)

        words_sequence = buildInputSequences(os.path.join(text_input_folder, '{}_input_words.pkl'.format(set_name)),
                            pkl.load(open(os.path.join(text_input_folder, 'vocabulary.pkl'),'rb')),
                            words=True, training=training)

        keys_sequence, key_idx =  buildInputSequences(os.path.join(text_input_folder, '{}_input_senses.pkl'.format(set_name)),
                                pkl.load(open(os.path.join(text_input_folder, 'sensekeys.pkl'), 'rb')),
                                words=False, training=training)

        token_sequence = buildInputSequences(os.path.join(text_input_folder, '{}_input_tokens.pkl'.format(set_name)),
                                             '', words=False, training=training, tokens=True)

        dom_sequence, dom_idx = buildInputSequences(os.path.join(text_input_folder, '{}_input_domains.pkl'.format(set_name)),
                        pkl.load(open(os.path.join(text_input_folder, 'domains.pkl'), 'rb')),
                         words=False, training=training)

        clean_input_all(words_sequence, dom_sequence, dom_idx, keys_sequence, key_idx,
                    token_sequence, input_folder, set_name)

        file_input = open(os.path.join(config.experiment_folder, '{}.txt'.format(set_name)),'w')
        words_clean = pkl.load(open(os.path.join(input_folder, '{}_words.pkl'.format(set_name)),'rb'))
        domains_clean = pkl.load(open(os.path.join(input_folder, '{}_domains.pkl'.format(set_name)),'rb'))
        tokens_clean = pkl.load(open(os.path.join(input_folder, '{}_tokens.pkl'.format(set_name)),'rb'))

        for idx, token_list in enumerate(tokens_clean):
            valid_tokens = [i for i,x in enumerate(token_list) if x!=None and x!='PADDING' and x!='untagged']
            labels = [domains_clean[idx][x] for x in valid_tokens]
            target_words = [words_clean[idx][x] for x in valid_tokens]
            for x in range(len(labels)):
                file_input.write(set_name+'.'+token_list[valid_tokens[x]] + '\t' + target_words[x]+'\t'+labels[x]+'\t'+' '.join([s if not j==valid_tokens[x] else '<target>{}<target>'.format(s)
                                for j,s in enumerate(words_clean[idx])]) + '\n')

def create_data_one_out(config,):
    training_name = config.training_name
    inventory = config.inventory
    wsd_data_path = config.wsd_data_folder
    one_out_text_input_folder = os.path.join(config.data_folder, 'input', 'text_files', inventory)
    one_out_input_folder = os.path.join(config.data_folder, 'input', 'matrices', inventory)
    test_names = config.tests
    all_words_mapping_coarse = pkl.load(open(config.mapping_path, 'rb'))
    output_mapping, dev_lemmas, test_lemmas = utils.buildOneOutDictionary(config)

    test_list = [training_name]
    for x in test_names:
        test_list.append(x)
    print('creating data for', test_list)
    for set_name in test_list:
        print('creating data for', set_name)
        if set_name==training_name:
            training = True
        else:
            training = False
        print('parsing xml file')
        if set_name!=config.dev_name:
            readXMLoneOut(set_name, os.path.join(wsd_data_path,'{}/{}.data.xml'.format(set_name, set_name)),
                      output_mapping, all_words_mapping_coarse, os.path.join(one_out_text_input_folder, '{}_input.txt'.format(set_name)),
                      os.path.join(wsd_data_path,'{}/{}.gold.key.txt'.format(set_name, set_name)),
                      one_out_text_input_folder, training=training, test_lemmas=test_lemmas)

        else:
            readXMLoneOut(set_name, os.path.join(wsd_data_path,'{}/{}.data.xml'.format(set_name, set_name)),
                          output_mapping, all_words_mapping_coarse, os.path.join(one_out_text_input_folder, '{}_input.txt'.format(set_name)),
                          os.path.join(wsd_data_path,'{}/{}.gold.key.txt'.format(set_name, set_name)),
                          one_out_text_input_folder, training=training, test_lemmas=dev_lemmas)

        words_sequence = buildInputSequences(os.path.join(one_out_text_input_folder, '{}_input_words.pkl'.format(set_name)),
                                             pkl.load(open(os.path.join(one_out_text_input_folder, 'vocabulary.pkl'), 'rb')),
                                             words=True, training=training)

        token_sequence = buildInputSequences(os.path.join(one_out_text_input_folder , '{}_input_tokens.pkl'.format(set_name)),
                                             '', words=False, training=training, tokens=True)

        dom_sequence, dom_idx = buildInputSequences(os.path.join(one_out_text_input_folder, '{}_input_domains.pkl'.format(set_name)),
                                                    pkl.load(open(os.path.join(one_out_text_input_folder, 'domains.pkl'), 'rb')),
                                                    words=False, training=training)

        clean_input(words_sequence, dom_sequence, dom_idx, token_sequence, one_out_input_folder, set_name)
        file_input = open(os.path.join(config.experiment_folder, '{}.txt'.format(set_name)),'w')
        words_clean = pkl.load(open(os.path.join(one_out_input_folder, '{}_words.pkl'.format(set_name)), 'rb'))
        domains_clean = pkl.load(open(os.path.join(one_out_input_folder, '{}_domains.pkl'.format(set_name)), 'rb'))
        tokens_clean = pkl.load(open(os.path.join(one_out_input_folder, '{}_tokens.pkl'.format(set_name)), 'rb'))

        for idx, token_list in enumerate(tokens_clean):
            valid_tokens = [i for i,x in enumerate(token_list) if x!=None and x!='PADDING' and x!='untagged']
            labels = [domains_clean[idx][x] for x in valid_tokens]
            target_words = [words_clean[idx][x] for x in valid_tokens]
            for x in range(len(labels)):
                file_input.write(target_words[x]+'\t'+labels[x]+'\t'+' '.join([s if not j==valid_tokens[x] else '<target>{}<target>'.format(s)
                                for j,s in enumerate(words_clean[idx])]) + '\n')
    return output_mapping, dev_lemmas, test_lemmas

def create_data_few_shot(config):
    test_names = config.tests
    training_name = config.training_name
    sensekey2domains = pkl.load(open(config.mapping_path, 'rb'))
    text_input_folder = os.path.join(config.data_folder, 'input/text_files/{}/'.format(config.inventory))
    input_folder = os.path.join(config.data_folder, 'input/matrices/{}/'.format(config.inventory))
    all_words_text_folder = config.all_words_folder
    dev_path = os.path.join(config.one_out_folder, 'dev_lemmas.yml')
    test_path = os.path.join(config.one_out_folder, 'test_lemmas.yml')
    wsd_data_folder = config.wsd_data_folder

    dev_lemmas = yaml.safe_load(open(dev_path))
    test_lemmas = yaml.safe_load(open(test_path))

    for k in [3,5,10]:
        readXMLFewShot(training_name, os.path.join(wsd_data_folder, '{}/{}.data.xml'.format(training_name, training_name)),
                       sensekey2domains, os.path.join(text_input_folder, '{}_input_{}.txt'.format(training_name, k)),
                       os.path.join(wsd_data_folder, '{}/{}.gold.key.txt'.format(training_name, training_name)),
                       text_input_folder, training=True, test_lemmas=test_lemmas, dev_lemmas=dev_lemmas, k=k)

    for test in test_names:
        if test!=config.dev_name:
            readXMLFewShot(test, os.path.join(wsd_data_folder, '{}/{}.data.xml'.format(test, test)),
                       sensekey2domains, os.path.join(text_input_folder, '{}_input.txt'.format(test)),
                       os.path.join(wsd_data_folder, '{}/{}.gold.key.txt'.format(test, test)),
                       text_input_folder, training=False, test_lemmas=test_lemmas, dev_lemmas=None, k=None)
        else:
            readXMLFewShot(test, os.path.join(wsd_data_folder, '{}/{}.data.xml'.format(test, test)),
                           sensekey2domains, os.path.join(text_input_folder, '{}_input.txt'.format(test)),
                           os.path.join(wsd_data_folder, '{}/{}.gold.key.txt'.format(test, test)),
                           text_input_folder, training=False, test_lemmas=dev_lemmas, dev_lemmas=None, k=None)

    test_list = [training_name]
    for x in test_names:
        test_list.append(x)

    for name in test_list:
        if name==training_name:
            training = True
            for k in [3,5,10]:
                words_sequence = buildInputSequences(os.path.join(text_input_folder, '{}_input_words_{}.pkl'.format(name,k)),
                                                     pkl.load(open(os.path.join(all_words_text_folder, 'vocabulary.pkl'), 'rb')),
                                                     words=True, training=training)

                token_sequence = buildInputSequences(os.path.join(text_input_folder, '{}_input_tokens_{}.pkl'.format(name,k)),
                                                     '', words=False, training=training, tokens=True)

                dom_sequence, dom_idx = buildInputSequences(os.path.join(text_input_folder, '{}_input_domains_{}.pkl'.format(name,k)),
                                                            pkl.load(open(os.path.join(all_words_text_folder, 'domains.pkl'), 'rb')),
                                                            words=False, training=training)
                clean_input(words_sequence, dom_sequence, dom_idx, token_sequence, os.path.join(input_folder,'{}/'.format(k)), name)

        else:
            training = False
            words_sequence = buildInputSequences(os.path.join(text_input_folder, '{}_input_words.pkl'.format(name)),
                                                 pkl.load(open(os.path.join(all_words_text_folder, 'vocabulary.pkl'), 'rb')),
                                                 words=True, training=training)

            token_sequence = buildInputSequences(os.path.join(text_input_folder, '{}_input_tokens.pkl'.format(name)),
                                                 '', words=False, training=training, tokens=True)

            dom_sequence, dom_idx = buildInputSequences(os.path.join(text_input_folder, '{}_input_domains.pkl'.format(name)),
                                                        pkl.load(open(os.path.join(all_words_text_folder, 'domains.pkl'), 'rb')),
                                                        words=False, training=training)
            clean_input(words_sequence, dom_sequence, dom_idx, token_sequence, input_folder, name)
