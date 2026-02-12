# Code of the paper titled "Patch-level Routing in Mixture-of-Experts is Provably Sample-efficient for Convolutional Neural Networks" [ICML, 2023]
by Mohammed Nowaz Rabbani Chowdhury, Shuai Zhang, Meng Wang, Sijia Liu and Pin-Yu Chen

### Requirements 
tensorflow, numpy, matplotlib

### Description
For experiments on each dataset, there is one directory. For example, for all the experiments on the CelebA  dataset, the directory is "CelebA Exp".

The data files need to be downloaded in the experiment directory using the hyperlinks provided at the file  ```data_files_*.txt``` of the directory.

The ```main_*.py``` in each folder train the networks and plot the results.

## Citation

If you find this code useful for your research or use it in your work, please cite the following paper:

```bibtex
@inproceedings{chowdhury2023patch,
  title={Patch-level routing in mixture-of-experts is provably sample-efficient for convolutional neural networks},
  author={Chowdhury, Mohammed Nowaz Rabbani and Zhang, Shuai and Wang, Meng and Liu, Sijia and Chen, Pin-Yu},
  booktitle={International Conference on Machine Learning (ICML)},
  pages={6074--6114},
  year={2023},
  organization={PMLR}
}
```

