# Malware Classification with Support Vector Machines (SVMs)

### Requirements

Ensure the following is present on your system:

- A suitably recent version of Python 3 (our development was done with Python 3.13.3)

- numpy

- torch

- sci-kit learn

### How to run

After cloning this repository, there will be a dataset directory, but all the data will not be present because it is too large to include here. Acquire the data from somewhere else and unzip it into the appropriate directory. It should look like this.

```
dataset/
└── feature_vectors/
    ├── 0a0a78000e418ea28fa02e8c...
    ├── 0a0a6170880154d5867d592...
    ├── 0a0aa292817689ff59a02273...
    ├── 0a0ae9a50664bf2013f6f05f...
    ├── 0a0b0e06c43bf1b9ef076902...
    ├── 0a0c257bf14fdef852490b4a...
    ├── 0a0cc33ef6a0274a2bee3d32...
    └── 0a0d0f8ec83af2624db0de48...
└──README.txt
└──sha256_family.csv
```

Then, in the main directory, use python to run main.py.

```
python3 main.py
```

By default this will run everything including the malware vs. benign classifier, the one-vs-all classifier, the one-vs-one classifier, and the SVM comparisons. Feel free to comment out anything you don't want to run in the main function. 