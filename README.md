# CSI: A Coarse Sense Inventory for 85% WSD 

                                CSI: A Coarse Inventory for 85% WSD
                Caterina Lacerra, Michele Bevilacqua, Tommaso Pasini and Roberto Navigli    
                                     Sapienza, University of Rome
                                    Department of Computer Science
                       {lacerra, bevilacqua, pasini, navigli} [at] di.uniroma1.it
   
This repository contains the code to reproduce the experiments reported in [CSI: A Coarse Sense Inventory for 85% Word Sense Disambiguation](https://pasinit.github.io/papers/lacerra_etal_aaai2020.pdf), by Caterina Lacerra, Michele Bevilacqua, Tommaso Pasini and Roberto Navigli.
For further information on this work, please visit our [website](https://SapienzaNLP.github.io/csi/).


## How to
Run the python scripts ```src/main_all_words.py```, ```src/main_one_out.py``` and ```src/main_few_shot.py``` to reproduce the experiments for the all-words, one-out and few-shot settings (Table 4 and 6 of the paper, respectively).

The arguments for the scripts are the same for each setting:

- ```inventory_name``` is one of the tested inventories, i.e. csi, wndomains, supersenses, sensekeys.
- ```model_name``` can be either BertDense or BertLSTM.
- ```data_dir``` is the path where data are located (typically ```./data```).
- ```data_out``` is the path of the output folder.
- ```wsd_data_dir``` is the path where wsd data are located (typically ```./wsd_data```)
- ```start_from_checkpoint``` is set if continuing training from a dumped checkpoint (optional).
- ```starting_epoch``` is different from 0 only if ```start_from_checkpoint``` is set. It is the starting epoch for the training (optional).
- ```do_eval``` is a flag to perform model evaluation only (optional). 
- ```epochs``` is the number of training epochs (optional, 40 by default). 

Please note that the **few-shot** setting continues training from the best epoch achieved with the one-out setting, thus it is necessary to run the one-out script first. 

### Example
To train a model in the _all words_ setting with CSI sense inventory, run

```python main_all_words.py inventory_name=csi model_name=BertLSTM data_dir=./data/ data_output=./output/ wsd_data_dir=./wsd_data/```

To evaluate a previously trained model, just add the ```do_eval``` parameter:

```python main_all_words.py --inventory_name=csi --model_name=BertLSTM --data_dir=./data/ --data_output=./output/ --wsd_data_dir=./wsd_data/ --do_eval```

Otherwise, to continue training a model for which checkpoints are available (e.g. from epoch 9):

```python main_all_words.py --inventory_name=csi --model_name=BertLSTM --data_dir=./data/ --data_output=./output/ --wsd_data_dir=./wsd_data/ --start_from_checkpoint --starting_epoch=9```


## Output 
The output folder defined with ```data_out``` will be created and filled with results during training and test. 
For each experiment configuration (i.e. all words, one out or few shot) will be created a folder that will contain results for each sense inventory used.
Let's assume we run the ```all_words``` experiment with ```csi```; what we have will be:

    +-- output_folder
    |  +-- csi
    |      +-- weights
    |      +-- logs
    |      output_files
    |      processed_input_files

Checkpoints for each training epoch will be contained inside the ```weights``` directory, while the ```logs``` directory 
will contain logs for TensorBoard.

There will be one tab-separated output file for each test dataset. The format of the files, is the following:
    
```flag instance_id predicted_label gold_label```

where ```flag``` is ```w``` or ```c``` for wrong and correct instances, respectively and ```instance_id``` uniquely identifies
the instance in the dataset.
Please note that the output file for the dev set is computed (and overwritten) at the end of each training epoch,
 while the output files for the other datasets are computed at test time.

The processed input files, instead, are computed both for the training and the test datasets, and the format is the following:

```instance_id target_word gold_label target_sentence```

Once again, the files are tab-separated.

# Acknowledgements
The authors gratefully acknowledge the support of the ERC Consolidator Grant MOUSSE No. 726487 under the European Union’s Horizon 2020 research and innovation programme.

# License
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
## You are free to:
**Share** - copy and redistribute the material in any medium or format</br>
**Adapt** - remix, transform, and build upon the material</br>

## Under the following terms:
**Attribution** - You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.</br>
**NonCommercial** - You may not use the material for commercial purposes.</br>
**ShareAlike** - If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.</br>
**No additional restrictions** - You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.
