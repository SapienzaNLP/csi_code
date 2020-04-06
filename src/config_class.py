import os

class Config():
    def __init__(self, inventory_name, model_name, data_dir, output_dir, wsd_data_dir, starting_epoch, checkpoint=False):
        #working_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        #output_dir = os.path.join(working_dir, 'output_files')
        self.output_dir = output_dir
        self.data_folder = data_dir #os.path.join(working_dir, 'data')
        self.wsd_data_folder = wsd_data_dir #os.path.join(working_dir, 'wsd_data')
        self.tests = ['senseval2','senseval3','semeval2007','semeval2013','semeval2015']
        self.training_name = 'semcor'
        self.dev_name = 'semeval2007'
        self.inventory = inventory_name
        self.model_name = model_name
        if self.inventory!="sensekey":
            self.finegrained = False
        else:
            self.finegrained = True

        self.start_from_checkpoint = checkpoint
        self.starting_epoch = starting_epoch
        self.mapping_path = os.path.join(self.data_folder, 'sensekey2{}.pkl'.format(inventory_name))
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)

class ConfigAllWords(Config):
    def __init__(self, inventory_name, model_name, starting_epoch, data_dir, output_dir, wsd_data_dir, checkpoint=False):
        super().__init__(inventory_name, model_name, data_dir, output_dir, wsd_data_dir, starting_epoch, checkpoint)


        self.data_folder = os.path.join(self.data_folder, 'all_words')

        self.experiment_folder = os.path.join(self.output_dir, 'all_words', inventory_name)
        if not os.path.exists(self.experiment_folder):
            print('creating output folder in', self.experiment_folder)
            os.makedirs(self.experiment_folder)


class ConfigOneOut(Config):
    def __init__(self, inventory_name, model_name, starting_epoch, data_dir, output_dir, wsd_data_dir, checkpoint=False):
        super().__init__(inventory_name, model_name, data_dir, output_dir, wsd_data_dir, starting_epoch, checkpoint)
        #working_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        #self.data_folder = os.path.join(working_dir, 'data')
        self.mapping_finegrained = os.path.join(self.data_folder, 'sensekey2sensekey.pkl')
        self.all_words_folder = os.path.join(self.data_folder, 'all_words', 'input', 'text_files', self.inventory)
        self.data_folder = os.path.join(self.data_folder, 'one_out')

        self.experiment_folder = os.path.join(self.output_dir, 'one_out', inventory_name)
        if not os.path.exists(self.experiment_folder):
            print('creating output folder in', self.experiment_folder)
            os.makedirs(self.experiment_folder)

class ConfigFewShot(Config):
    def __init__(self, inventory_name, model_name, starting_epoch, data_dir, data_out, wsd_data_dir, checkpoint=False):
        super().__init__(inventory_name, model_name, data_dir, data_out, wsd_data_dir, starting_epoch, checkpoint)
        self.all_words_folder = os.path.join(self.data_folder, 'all_words', 'input', 'text_files', self.inventory)
        self.one_out_folder = os.path.join(self.data_folder, 'one_out')
        self.data_folder = os.path.join(self.data_folder, 'few_shot')

        self.experiment_folder = os.path.join(self.output_dir, 'few_shot', inventory_name)
        if not os.path.exists(self.experiment_folder):
            print('creating output folder in', self.experiment_folder)
            os.makedirs(self.experiment_folder)
        self.one_out_weights = os.path.join(self.experiment_folder, os.pardir, os.pardir, 'one_out', inventory_name, 'weights')
