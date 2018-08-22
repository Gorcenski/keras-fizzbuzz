# keras-fizzbuzz
A repository to explore a simple problem, solvable with machine learning, and to demonstrate CI/CD workflows with machine learning.

## What this repository is

This repository contains some basic code to sovle a well-known problem, FizzBuzz, using a neural networking API. The intention here is to have a simple problem solvable with real tools, without requiring the additional step of ingesting a bunch of data. This makes it faster and easier to play with various configurations, CI tools, and more.

This repo will use [Keras](https://keras.io/) with a TensorFlow backend to solve the problem. It also uses [dvc](https://dvc.org) to implement team-based, Agile workflows. A CircleCI config file is included to demonstrate automated builds.

## What this repository isn't

Serious code that should be taken seriously. This is a toy problem intended to be used for trying out various technologies to rapidly assess how they work and whether they might be suitable.

## Installation

This repo has been tested with python 3.6.5/3.6.6. It does not currently support python 3.7+ (this is due to TensorFlow not yet supporting python 3.7.0). It is strongly recommended that you use a virtualenv or condas to isolate your working environment, as this repo will cram a whole lot of stuff onto your system if you don't.

First, clone this repo. Then, install the required packages:

```sh
pip install -r requirements.txt
```

Once everything is set up, ensure that dvc is working by running

```sh
dvc --version
```

### The problem

FizzBuzz is a classical (bad) interview problem that is roughly as follows:

> For the numbers 1 to 100, do the following:
> - If the number is divisible by 3, print 'Fizz';
> - If the number is divisible by 5, print 'Buzz';
> - If the number is divisible by 15, print 'FizzBuzz';
> - otherwise, print the number itself.

While the author rejects this exercise as a meaningful assessment of likely workplace success, it is a perfectly fine toy problem that can be solved a myriad of ways, and creative solutions often expose issues with package version conflicts, workspace permissions, and the like.

### The solution

This repository contains two python files that do all the work: `train.py` and `validate.py`. Here is how they work:

In `train.py` we generate some very synthetic data on which to solve FizzBuzz. By default, we generate a range of data for integers in the range `[500, 5500)`. We then perform some "feature extraction" on the numbers, feature extracting being, in this limited case, the act of computing the prime modulii of each value for every prime less than 15.

We also compute the "truth" value of whether each integer belongs to class `Fizz`, `Buzz`, `FizzBuzz`, or `other`. These classes are encoded using one-hot encoding.

We then structure a sequential neural network using Keras to perform a categorical classification. The network is trained on the generated data, and the resulting model is output in a pickle file, `mymodel.pkl`.

Next, to validate the model, we use numbers in the range `[1, 100]` and test it against our canonical solution for correctness using the `validate.py` script. A unit test script is also included to determine if our model beats 95% accuracy.

### Reproducing the work

To reproduce the work, we will utilize dvc's excellent `repro` command. Simply type

```sh
dvc repro accuracy.txt.dvc
```

This will repeat the training step, followed by the validation step. By running

```sh
pytest test.py
```

we can see whether the solution met our required threshold.

### Rebuilding the pipeline

To rebuild the pipeline, we can reconstruct each step using `dvc run`. Simply type, as follows:

```sh
dvc run -d train.py -o mymodel.pkl python train.py
dvc run -d train.py -d mymodel.pkl -d validate.py -M accuracy.txt python validate.py
```

This should re-build the `.dvc` files.

### Making changes

First, make a branch, call it `model-tweaks`.

The basic model contained in the repo should not perform so well on FizzBuzz! But we can improve it.

Open `train.py` and find the following lines of code:

```python
model.add(Dense(4, activation='relu', input_dim=dim))
model.add(keras.layers.Dropout(0.8))
model.add(Dense(4, activation='softmax'))
```

Change these lines by removing the dropout layer and changing the number of input nodes to the first layer to 10:

```python
model.add(Dense(10, activation='relu', input_dim=dim))
model.add(Dense(4, activation='softmax'))
```

Save this, then simply reproduce the pipeline with `dvc repro accuracy.txt.dvc`. Alternatively, if you want to make deeper changes, you can manually rebuild the pipeline using the instructions from the prior step. For instance, maybe we stop generating the data ourselves, but ingest it from another source.

When you're done making your changes, we need to add a single step to make it sharable and discoverable to our CI environment:

```sh
dvc push
git commit -am "Update model"
git push
```

That's it! Now you can open a PR for your branch, and your who team can evaluate the effectiveness of your model changes.
