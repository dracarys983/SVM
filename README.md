## Assigment 2 (Qxx)

### English letter classification using Support Vector Machines

In this part of the question, we will do english letter classification using a SVM.
We have a dataset containing 16 primitive attributes (statistical moments, edge counts, etc.)
for each of the _capital_ english letters A-Z. These letters were originally written in 20
different fonts and there are 20K samples in total in the dataset.
Follow this [link](https://archive.ics.uci.edu/ml/datasets/letter+recognition) for more information
about the dataset and what features are given in the dataset file.

Each line in the dataset file has the following format:
```
letter, hpos, vpos, w, h, #pix, xbar, ybar, x2bar, y2bar, xybar, x2ybr, xy2br, xedge, xegvy, yedge, yegvx
```
All of the 16 features are integers with values betwen 0-15.

You have to design a SVM classifier for the given data. The file `letter_classification_svm.py` is your code
file for this part. __You are provided with some parts of the pipeline, while some parts are left for you to
code__. The parts where you need to write the code look like this:

```
======================================================================

# YOUR CODE GOES HERE

======================================================================
```

You are allowed to use scikit-learn for the SVM and any other python libraries for processing your data.
__DO NOT USE ANY OTHER LIBRARY THAN SCIKIT-LEARN FOR SVM__. You are not allowed to use MATLAB for this
question. You have to learn how to use scikit-learn and python due to which I would suggest that the students
start out early. Most of the parts are coded for you so it should be easy enough once you know how to use
scikit-learn.

Use different kernels (atleast 3) and perform experiments with different hyperparameters. _This is mandatory_.
You need to report the accuracy, precision, recall and F-1 score for the validation data, corresponsing to __each__
experiment that you perform. The data is divided into five train/validation splits for which the code is already written.
Hence, you will get five sets of values for the above metrics. You should report the _average_ over all the five splits.

The output of your code file should have one line with the four metric values for the best model:
```
-> python letter_classification_svm.py --data_dir='<path_to_dataset_dir>'
<accuracy_val>, <precision_val>, <recall_val>, <F-1_val>
```
Mention your experiments and observations in your _final report_. 

Once you know which model and hyperparameters work best, create that model in `letter_classification_svm_submission.py`,
which is your final submission file. This will be tested on an unseen part of the dataset for testing it's effectiveness.
The output format for this code is similar to the output of your best model as shown above.

### Directory Structure

When you clone this repository, you will get the following directory structure:

```
SVM/
|
|--- letter_classification/
|    |
|    |--- letter_classification_svm.py
|    |--- letter_classification_svm_submission.py
|    |--- letter_classification_train.data
|    
|--- stencils/
|    |
|    |--- stencilletter-*.jpg
|    
|--- visualization_examples/
|    |
|    |--- linearly_separable.py
|    |--- linearly_inseparable.py
|    |--- plot_tsne.py
|    |--- t-SNE_letters.pdf
|    
|--- README.md
|--- LICENSE

```

__Keep your dataset files in the respective folder for the task__. 

The files provided in the visualization examples folder is for understanding purposes and might help you gain insights about
the working of the SVM and how the dataset looks like. t-SNE plot script for CIFAR-10 is not provided but you can modify the
given script of get it online somewhere if you really want to visualize the t-SNE embeddings for it.

### Submission Format

For this question, your submission needs to follow the format given below:

```
<RollNumber>/
|
|--- letter_classification/
|    |
|    |--- letter_classification_svm_submission.py
|    
|--- README.md
|--- Report.pdf

```

* Zip the top level directory and submit it with the name \<RollNumber\_SVM\>.zip.
* README.md should contain a brief overview of the methods followed and which kernel-hyperparameter combinations were used.
* Report.pdf should contain in-depth explanation about each kernel, the parameters used for them and the results obtained in each experiments (preferably graphical representations).

The evaluation script will run your submissions on the same dataset, the only difference being your algorithm will now be trained on entire part of dataset given to you in this repository and tested on a new unseen part of the dataset.
For example:

```
python letter_classification_svm_submission.py --data_dir='$HOME/letter-classification-data'
```

The output should contain one line for the command shown above; for letter classification metrics.
Your submission should execute to completion in no more than __120 seconds__. The number of test samples for letter classification is 4000.

__ALL THE BEST!__


> ### Sir Winston Churchill told this to his people to rouse their fighting spirit, "If you are going through hell, keep going."
