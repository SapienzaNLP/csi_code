import os
import pickle
import argparse
import src.test as test
import src.utils as utils
import src.training as training
import src.config_class as config_class
import src.preprocessing as preprocessing


def prepare_data(config):
    corpus_folder = config.corpus_folder
    gold_folder = os.path.join(config.common_folder, 'gold/{}/'.format(config.inventory))
    all_words_mapping_coarse = pickle.load(open(config.mapping_path, 'rb'))

    mapping_output, dev_lemmas, test_lemmas = preprocessing.create_data_one_out(config)
    for testname in config.tests:
        if testname!=config.dev_name:
            utils.getGoldFilesMappedOneOut(os.path.join(corpus_folder, '{}/{}.gold.key.txt'.format(testname, testname)),
                                           os.path.join(gold_folder,'{}.gold.txt'.format(testname)),
                                           all_words_mapping_coarse, test_lemmas)
        else:
            utils.getGoldFilesMappedOneOut(os.path.join(corpus_folder, '{}/{}.gold.key.txt'.format(testname, testname)),
                                           os.path.join(gold_folder,'{}.gold.txt'.format(testname)),
                                           all_words_mapping_coarse, dev_lemmas)

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
        config = config_class.ConfigOneOut(args.inventory_name, args.model_name, args.starting_epoch, args.starting_from_checkpoint)

    else:
        config = config_class.ConfigOneOut(args.inventory_name, args.model_name, args.starting_epoch)

    print('\n\nUsing {} as sense inventory'.format(config.inventory))
    print('\n\nUsing {} as dev set'.format(config.dev_name))


    print('Output files will be saved to {}'.format(config.experiment_folder))

    utils.define_folders(config)
    prepare_data(config)
    training.train_model_one_out(config, epochs=1)
    best_epoch = utils.pick_epoch(config.experiment_folder)
    test.test_one_out(config, best_epoch)