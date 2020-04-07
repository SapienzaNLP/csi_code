import argparse
import os
import pickle as pkl
import random
import sys
import test

import yaml

import src.config_class as config_class
import src.preprocessing as preprocessing
import src.training as training
import src.utils as utils


def prepare_data(config):
    random.seed(42)
    tests = config.tests
    inventory = config.inventory
    gold_folder = os.path.join(config.data_folder, 'gold/{}/'.format(inventory))
    all_words_mapping_coarse = pkl.load(open(config.mapping_path, 'rb'))
    dev_path = os.path.join(config.one_out_folder, 'dev_lemmas.yml')
    test_path = os.path.join(config.one_out_folder, 'test_lemmas.yml')

    preprocessing.create_data_few_shot(config)

    dev_lemmas = yaml.safe_load(open(dev_path))
    test_lemmas = yaml.safe_load(open(test_path))
    for testname in tests:
        if testname != config.dev_name:
            utils.getGoldFilesMappedOneOut('{}/{}/{}.gold.key.txt'.format(config.wsd_data_folder, testname, testname),
                os.path.join(gold_folder, '{}.gold.txt'.format(testname)), all_words_mapping_coarse, test_lemmas)
        else:
            utils.getGoldFilesMappedOneOut('{}/{}/{}.gold.key.txt'.format(config.wsd_data_folder, testname, testname),
                os.path.join(gold_folder, '{}.gold.txt'.format(testname)), all_words_mapping_coarse, dev_lemmas)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--inventory_name", required=True, help="The name of the inventory we want to "
                                                                "use for the experiments.",
                        choices=["csi", "wndomains", "supersenses", "sensekey"])

    parser.add_argument("--model_name", required=True, help="Name of the model.", choices=["BertDense", "BertLSTM"])

    parser.add_argument("--starting_from_checkpoint", help="True if continuing training from a saved checkpoint, "
                                                           "that should be defined with the --starting_epoch arg.",
                        action="store_true")

    parser.add_argument("--starting_epoch", help="Starting epoch for the training. In order to be effective, "
                                                 "--starting_from_checkpoint should be True.", type=int, default=0)
    parser.add_argument("--data_dir", required=True, help="directory where data are located, typically ./data/")
    parser.add_argument("--data_out", required=True, help="directory for the output")
    parser.add_argument("--wsd_data_dir", required=True, help="directory where wsd training and evaluation data "
                                                              "are located, typically ./wsd_data/")
    parser.add_argument("--do_eval", help="If set, the model performs the evaluation step only. "
                                          "Otherwise, it will perform both training and evaluation.",
                        action="store_true")

    parser.add_argument("--epochs", help="Number of training epochs.", type=int, default=20)


    args = parser.parse_args()
    if args.starting_from_checkpoint:
        print("Starting training from epoch {} checkpoint".format(args.starting_epoch))

        config = config_class.ConfigFewShot(args.inventory_name, args.model_name, args.starting_epoch, args.data_dir,
                                            args.data_out, args.wsd_data_dir,
                                        args.start_from_checkpoint)

    else:
        config = config_class.ConfigFewShot(args.inventory_name, args.model_name, args.starting_epoch, args.data_dir,
                                            args.data_out, args.wsd_data_dir)

    print('\n\nUsing {} as sense inventory'.format(config.inventory))
    print('Using {} as dev set'.format(config.dev_name))

    print('Output files will be saved to {}'.format(config.experiment_folder))

    if len(os.listdir(config.one_out_weights)) > 1:
        print('choose best epoch in {}'.format(config.one_out_weights))
        sys.exit()
    else:
        path_weights = os.path.join(config.one_out_weights, os.listdir(config.one_out_weights)[0])

    for k in [3, 5, 10]:
        if not args.do_eval:
            utils.define_folders_few_shot(config)
            prepare_data(config)
            training.train_model_few_shot(config, k, path_weights, args.epochs)
        best_epoch = utils.pick_epoch(config.experiment_folder, k)
        print('k =', k)
        test.test_few_shot(config, best_epoch, k)
