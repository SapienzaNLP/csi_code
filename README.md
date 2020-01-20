# CSI: A Coarse Sense Inventory for 85% WSD 

                Caterina Lacerra, Michele Bevilacqua, Tommaso Pasini and Roberto Navigli    
                                     Sapienza, University of Rome
                                    Department of Computer Science
                       {lacerra, bevilacqua, pasini, navigli} [at] di.uniroma1.it
   
This repository contains the code to reproduce the experiments reported in [CSI: A Coarse Sense Inventory for 85% Word Sense Disambiguation](https://pasinit.github.io/papers/lacerra_etal_aaai2020.pdf), by Caterina Lacerra, Michele Bevilacqua, Tommaso Pasini and Roberto Navigli.
For further information on this work, please visit our [website](https://SapienzaNLP.github.io/csi/).


## How to:
Run the python scripts ```src/main_all_words.py```, ```src/main_one_out.py``` and ```src/main_few_shot.py``` to reproduce the experiments for the all-words, one-out and few-shot settings, respectively.

The arguments for the scripts are the same for each setting:

- ```inventory_name``` is one of the tested inventories, i.e. csi, wndomains, supersenses, sensekeys.
- ```model_name``` can be either BertDense or BertLSTM.
- ```start_from_checkpoint``` is True if starting training from a dumped checkpoint.
- ```starting_epoch``` is different from 0 only if ```start_from_checkpoint``` is True. It is the starting epoch for the training.

Please note that the few-shot setting continues training from the best epoch achieved with the one-out setting, thus it is necessary to run the one-out script first. 

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