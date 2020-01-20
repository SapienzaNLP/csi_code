import argparse
import os
import pickle as pkl

import src.config_class as config_class
import src.preprocessing as preprocessing
import src.test as test
import src.training as training
import src.utils as utils


def prepare_data(config):
    tests = config.tests
    gold_folder = os.path.join(config.data_folder,  'gold', config.inventory)
    mapping_output = pkl.load(open(config.mapping_path, 'rb'))
    preprocessing.create_data(config)
    for testname in tests:
        utils.getGoldFilesMapped(os.path.join(config.wsd_data_folder,'{}/{}.gold.key.txt'.format(testname, testname)),
                                 os.path.join(gold_folder, '{}.gold.txt'.format(testname)), mapping_output)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("inventory_name", help="The name of the inventory we want to use for the experiments.",
                        choices=["csi", "wndomains", "supersenses", "sensekey"])

    parser.add_argument("model_name", help="Name of the model.", choices=["BertDense", "BertLSTM"])

    parser.add_argument("--starting_from_checkpoint", help="True if continuing training from a saved checkpoint, "
                                                           "that should be defined with the --starting_epoch arg.",
                        action="store_true")

    parser.add_argument("--starting_epoch", help="Starting epoch for the training. In order to be effective, "
                                                 "--starting_from_checkpoint should be True.", type=int, default=0)

    args = parser.parse_args()

    if args.starting_from_checkpoint:
        print("Starting training from epoch {} checkpoint".format(args.starting_epoch))
        config = config_class.ConfigAllWords(args.inventory_name, args.model_name, args.starting_epoch)

    else:
        config = config_class.ConfigAllWords(args.inventory_name, args.model_name, args.starting_epoch, args.starting_from_checkpoint)

    print('\n\nUsing {} as sense inventory'.format(config.inventory))

    print('Output files will be saved to {}'.format(config.experiment_folder))

    utils.define_folders(config)
    prepare_data(config)
    training.train_model(config)
    best_epoch = utils.pick_epoch(config.experiment_folder)
    test.test(config, best_epoch)
