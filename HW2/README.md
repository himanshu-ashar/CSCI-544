Python 3.9.7

The Python file does not need any command line arguments.
Running it as it is will work.

Command:
python HW2_Himanshu_Ashar.py


As the Professor instructed, no libraries were used to implement HMM. Only the following generic libraries were used:

from collections import defaultdict
import operator
import copy
import json
import toolz

I have also included the Jupyter Notebook I used. The Python file is this same notebook exported to .py.

Input/Output file locations:

All the paths I have enlisted while running the code assume there is a folder named 'data', in which train, dev and test sets exist.
This folder 'data' is on the same level as .py file.
The .py file reads these files and creates vocab.txt, hmm.json, greedy.out and viterbi.out. All of these files are created in the same 'data' folder.
I have maintained this folder structure while submitting the Homework. I have also included the train, dev and test sets in the folder, which are completely unmodified, and just the same as in the assignment content uploaded on Blackboard/DEN.

For both Greedy Decoding and Viterbi Decoding, there is just one method each: greedyDecoding, and viterbiDecoding.
The execute as follows:

greedyDecoding('dev') - this will compute performance and output the actual and predicted tags for the dev set. Following this I have a function named getDevAccuracy, which takes these lists of tags and outputs dev accuracy.

greedyDecoding('test') - this will only write the predicted tags for the test set into a new file named greedy.out.

viterbiDecoding('dev') - this will compute performance and output the actual and predicted tags for the dev set. Following this I have a function named getDevAccuracy, which takes these lists of tags and outputs dev accuracy.

viterbiDecoding('test') - this will only write the predicted tags for the test set into a new file named viterbi.out.

I have executed both the above methods with both the parameters, 'dev' and 'test', in the .py file.
