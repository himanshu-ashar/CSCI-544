There are THREE python files in my submission. They are almost the same code from my Jupyter notebook, except certain parts relevant to training and evaluation are commented out:

1. train.py : This file should be run only if the model is required to be trained again. Othwerwise, I have submitted both my model files, as required: bilstm_model.pt for task 1, and bilstm_glove_model.pt for task 2. Running this will generate new .pt files and overwrite the ones in current directory.

2. dev.py : This file should be run when results on DEV data need to be reproduced.

3. test.py: This file should be run when results on TEST data need to be reproduced.

## Command to generate dev1.out and dev2.out files: python dev.py
## Command to generate test1.out and test2.out files: python test.py

A PDF of the answers is in the file named: REPORT - HW4_Himanshu_Ashar.pdf

NOTE: My .py file accesses the Glove embeddings in the uncompressed form, hence I have included the glove.6B.100d file which is not compressed as well.
