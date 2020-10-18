# IncrementalED
This is the source code for the paper ''Incremental Event Detection via Knowledge Consolidation Networks'' accepted by EMNLP2020.
## Requirements
  * Pytorch 1.6
  * transformers
  * python 3.6
## Usage
### Download datasets
Please download the [ACE dataset](https://catalog.ldc.upenn.edu/LDC2006T06) and [KBP dataset](https://tac.nist.gov/2017/KBP/data.html), respectively. The raw dataset should be converted to the format as shown in `train_data` file.
### Train and Evaluate model
For training the model, you need to type the following command:
 * python train.py

## Citation
If you use the code, please cite this paper:

Pengfei Cao, Yubo Chen, Jun Zhao, Taifeng Wang. Incremental Event Detection via Knowledge Consolidation Networks. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP2020).
