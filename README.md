# Dynamic Attentional Coupling Model
Dynamic attentional coupling model for audio and video data analysis, refered to as DAC.
Based on the work of autoencoded-vocal-analysis (AVA), original repository available at https://github.com/pearsonlab/autoencoded-vocal-analysis and 
on multimodal transformer (MMT), original repository available at https://github.com/yaohungt/Multimodal-Transformer


- This repo includes 
  - a modified version of MMT modules to better support DAC.
  - DAC specific code, in the `DAC/src`
  - DAC modules, in the `DAC/modules`
  - a demo main python script for running the pipeline ( `DAC/dac_main.py`)
  - a demo dataset, which is used to illstrate the assumed data structure (`DAC/data/bird_vocal_data`, courtesy to Fabiola Duarte-Ortiz)
  - a yml file to help configure the environment (`DAC/environment.yml`)
- The repo pends to be updated online in github after figuring out tracking changes inside and outside of the ava submodule.
- For any question, please reach out to Scott Smyre at scott.smyre@duke.edu.
