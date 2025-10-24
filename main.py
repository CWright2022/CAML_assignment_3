import os
import time
from random import shuffle
import numpy as np
import numpy.typing as npt
import torch
from typing import Callable
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATASET_FILEPATH = 'dataset'
FEATURE_VECTORS_FILEPATH = os.path.join(DATASET_FILEPATH, 'feature_vectors')
HASH_MAPPINGS_FILEPATH = os.path.join(DATASET_FILEPATH, 'sha256_family.csv')

# Color codes
COLOR_DEFAULT = '\x1b[39m'
COLOR_GREEN = '\x1b[32m'
COLOR_RED = '\x1b[31m'

def green_print(msg: str):
    """Convenient function for printing green text"""
    print(f'{COLOR_GREEN}{msg}{COLOR_DEFAULT}')

def red_print(msg: str):
    """Convenient function for printing red text"""
    print(f'{COLOR_RED}{msg}{COLOR_DEFAULT}')


class SVM:
    """Models a Support Vector Machine (SVM) to make decisions"""
    def __init__(self, feature_vectors: npt.NDArray[np.bool_], sample_hashes: list[str], ground_truth: dict[str, str]):
        self.feature_vectors = feature_vectors
        self.sample_hashes = sample_hashes
        self.ground_truth_str = ground_truth
        self.ground_truth_int = dict()
        self.verbose = False

        # Model parameters
        self.C = 0.001

        # Trainable model values
        self.w = torch.zeros(self.feature_vectors.shape[1], dtype=torch.float32, requires_grad=True)
        self.b = torch.zeros(1, dtype=torch.float32, requires_grad=True)
    
    def define_classes(self, class_distinguisher: Callable[[str], int]):
        """Initializes ground_truth_int, a dictionary mapping sample hashes to either -1 or 1 class labels"""
        for sample_hash, class_ in self.ground_truth_str.items():
            self.ground_truth_int[sample_hash] = class_distinguisher(class_)

    def calculate_prediction_torch(self, feature_vector: torch.Tensor) -> torch.Tensor:
        prediction = torch.dot(feature_vector, self.w) + self.b
        return prediction

    def calculate_hinge_loss_torch(self, raw_prediction: torch.Tensor, true_class_label: torch.Tensor) -> torch.Tensor:
        """Computes hinge loss and works with torch"""
        hinge_loss = torch.clamp(1 - (raw_prediction * true_class_label), min=0)
        hinge_loss = hinge_loss + self.C * (torch.norm(self.w) / 2)
        return hinge_loss
    
    def train(self, epochs: int, lr: float):
        """Train on all the data that is given. No testing here. Used by the Mutli_SVM"""
        # TODO: Fix the code duplication between this function and the evaluate function
        # Shuffle and then do train/test split
        p = np.random.permutation(self.feature_vectors.shape[0])
        feature_vectors_train = self.feature_vectors[p]
        sample_hashes_train = [self.sample_hashes[i] for i in p]

        # Convert everything to torch tensors
        X_train = torch.tensor(feature_vectors_train, dtype=torch.float32)

        y_train = torch.tensor([self.ground_truth_int[h] for h in sample_hashes_train], dtype=torch.float32)

        # Training loop
        for epoch in range(1, epochs+1):
            total_loss = 0

            # Randomize the order of the train sets
            p = torch.randperm(len(X_train))
            X_train, y_train = X_train[p], y_train[p]
            
            # Loop over every sample in the training data
            # For each sample, calculate hinge loss and update gradients
            for sample, true_label in zip(X_train, y_train):
                # Make prediction
                y_pred = self.calculate_prediction_torch(sample)

                # Calculate hinge loss
                hinge_loss = self.calculate_hinge_loss_torch(y_pred, true_label)

                # Compute gradients
                hinge_loss.backward()

                # Update model values
                # Pylance might complain without the "type: ignore" comments
                with torch.no_grad():
                    self.w -= lr * self.w.grad  # type: ignore
                    self.b -= lr * self.b.grad  # type: ignore
                
                self.w.grad.zero_()  # type: ignore
                self.b.grad.zero_()  # type: ignore

                total_loss += hinge_loss.item()

            if self.verbose:
                if epoch % 1 == 0:
                    average_loss = total_loss / len(X_train)
                    print(f'Epoch {epoch}/{epochs} - Average Loss: {average_loss:.4f}')
    
    def evaluate(self, epochs: int, lr: float, train_test_split: float):
        """Fully train, test, and evaluate the performance of the model"""
        # Shuffle and then do train/test split
        p = np.random.permutation(self.feature_vectors.shape[0])
        feature_vectors_shuffled = self.feature_vectors[p]
        sample_hashes_shuffled = [self.sample_hashes[i] for i in p]

        index_split_value = int(train_test_split*feature_vectors_shuffled.shape[0])
        feature_vectors_train = feature_vectors_shuffled[:index_split_value]
        feature_vectors_test = feature_vectors_shuffled[index_split_value:]
        sample_hashes_train = sample_hashes_shuffled[:index_split_value]
        sample_hashes_test = sample_hashes_shuffled[index_split_value:]

        # Convert everything to torch tensors
        X_train = torch.tensor(feature_vectors_train, dtype=torch.float32)
        X_test = torch.tensor(feature_vectors_test, dtype=torch.float32)

        y_train = torch.tensor([self.ground_truth_int[h] for h in sample_hashes_train], dtype=torch.float32)
        y_test = torch.tensor([self.ground_truth_int[h] for h in sample_hashes_test], dtype=torch.float32)

        # Training loop
        for epoch in range(1, epochs+1):
            total_loss = 0

            # Randomize the order of the train sets
            p = torch.randperm(len(X_train))
            X_train, y_train = X_train[p], y_train[p]
            
            # Loop over every sample in the training data
            # For each sample, calculate hinge loss and update gradients
            for sample, true_label in zip(X_train, y_train):
                # Make prediction
                y_pred = self.calculate_prediction_torch(sample)

                # Calculate hinge loss
                hinge_loss = self.calculate_hinge_loss_torch(y_pred, true_label)

                # Compute gradients
                hinge_loss.backward()

                # Update model values
                # Pylance might complain without the "type: ignore" comments
                with torch.no_grad():
                    self.w -= lr * self.w.grad  # type: ignore
                    self.b -= lr * self.b.grad  # type: ignore
                
                self.w.grad.zero_()  # type: ignore
                self.b.grad.zero_()  # type: ignore

                total_loss += hinge_loss.item()

            if self.verbose:
                if epoch % 1 == 0:
                    average_loss = total_loss / len(X_train)
                    print(f'Epoch {epoch}/{epochs} - Average Loss: {average_loss:.4f}')
        
        # Training is done...now evaluate using the test data
        with torch.no_grad():
            correct_counter = 0
            for sample, true_label in zip(X_test, y_test):
                y_pred = self.calculate_prediction_torch(sample)
                
                if torch.sign(y_pred).item() == true_label.item():
                    correct_counter += 1
        
        green_print(f'{correct_counter}/{len(X_test)} were classified correctly.')
        green_print(f'Accuracy: {100*correct_counter/len(X_test):.2f}%')

        importance = torch.abs(self.w).detach().numpy()
        top10_idx = np.argsort(importance)[-10:]
        bottom10_idx = np.argsort(importance)[:10]

        green_print("\nTop 10 Most Important Feature Indices and Weights:")
        for idx in reversed(top10_idx):
            print(f"Feature {idx}: Weight = {self.w[idx].item():.6f}")

        green_print("\nBottom 10 Least Important Feature Indices and Weights:")
        for idx in bottom10_idx:
            print(f"Feature {idx}: Weight = {self.w[idx].item():.6f}")


class SVM_OneVsAll:
    """Models using many SVMs and One-Vs-All classification to make more than just binary decisions"""
    def __init__(self, feature_vectors: npt.NDArray[np.bool_], sample_hashes: list[str], ground_truth: dict[str, str], train_test_split=0.8):
        self.ground_truth = ground_truth
        self.verbose = False

        p = np.random.permutation(len(feature_vectors))
        feature_vectors_shuffled = feature_vectors[p]
        sample_hashes_shuffled = [sample_hashes[i] for i in p]

        index_split_value = int(train_test_split*feature_vectors_shuffled.shape[0])
        self.feature_vectors_train = feature_vectors_shuffled[:index_split_value]
        self.feature_vectors_test = feature_vectors_shuffled[index_split_value:]
        self.sample_hashes_train = sample_hashes_shuffled[:index_split_value]
        self.sample_hashes_test = sample_hashes_shuffled[index_split_value:]

        # Two parallel arrays holding each SVM and the malware type that it classifies
        self.svms: list[SVM] = []
        self.malware_labels = []
    
    def append_svm(self, malware_label: str):
        """Adds an SVM to this multiclassifier that distinguishes the given malware label from all other malware labels"""
        svm = SVM(self.feature_vectors_train, self.sample_hashes_train, self.ground_truth)
        svm.define_classes(lambda x: 1 if x == malware_label else -1)
        svm.verbose = self.verbose
        self.svms.append(svm)
        self.malware_labels.append(malware_label)
    
    def train_all(self, epochs, lr):
        tick = time.time()
        for i, svm in enumerate(self.svms):
            if self.verbose:
                print(f'Training SVM {i+1}/{len(self.svms)} for malware type {self.malware_labels[i]}')
            svm.train(epochs, lr)
        time_elapsed = time.time() - tick
        print(f'Training time: {time_elapsed:.2f} seconds')
    
    def classify_sample(self, sample: npt.NDArray[np.bool_]) -> str:
        sample_torch = torch.tensor(sample, dtype=torch.float32)
        pred_results = []
        for svm in self.svms:
            pred_results.append(svm.calculate_prediction_torch(sample_torch))
        
        pred_values = [p.item() for p in pred_results]
        label = self.malware_labels[int(np.argmax(pred_values))]

        return label
    
    def test(self):
        n = len(self.malware_labels)

        correct_counter = 0
        confusion_matrix = np.zeros((n, n), dtype=int)
        for sample, sample_hash in zip(self.feature_vectors_test, self.sample_hashes_test):
            true_label = self.ground_truth.get(sample_hash)
            if true_label is None:
                red_print('Something went wrong.')
                continue
            predicted_label = self.classify_sample(sample)

            if true_label == predicted_label:
                correct_counter += 1
            
            # Build confusion matrix
            true_index = self.malware_labels.index(true_label)
            predicted_index = self.malware_labels.index(predicted_label)
            confusion_matrix[true_index][predicted_index] += 1

        # Print confusion matrix
        print()
        print('Confusion Matrix:')
        for i, row in enumerate(confusion_matrix):
            for j, val in enumerate(row):
                if i == j:
                    color = COLOR_GREEN
                else:
                    color = COLOR_RED
                print(f'{color}{val: <4}{COLOR_DEFAULT}', end='')
            print()

        print()
        green_print(f'{correct_counter}/{len(self.feature_vectors_test)} were classified correctly.')
        green_print(f'Accuracy: {100*correct_counter/len(self.feature_vectors_test):.2f}%')


class SVM_OneVsOne:
    """Models using many SVMs and One-Vs-All classification to make more than just binary decisions"""
    def __init__(self,  feature_vectors: npt.NDArray[np.bool], sample_hashes: list[str], ground_truth: dict[str, str], train_test_split=0.8):
        self.ground_truth = ground_truth
        self.verbose = False

        p = np.random.permutation(len(feature_vectors))
        feature_vectors_shuffled = feature_vectors[p]
        sample_hashes_shuffled = [sample_hashes[i] for i in p]

        index_split_value = int(train_test_split*feature_vectors_shuffled.shape[0])
        self.feature_vectors_train = feature_vectors_shuffled[:index_split_value]
        self.feature_vectors_test = feature_vectors_shuffled[index_split_value:]
        self.sample_hashes_train = sample_hashes_shuffled[:index_split_value]
        self.sample_hashes_test = sample_hashes_shuffled[index_split_value:]

        # Two parallel arrays holding each SVM and the malware type that it classifies
        self.svms: list[list[SVM]] = []
        self.malware_labels = []

    def append_svm(self, malware_label: str):
        """
        Adds multiple SVMs to this multiclassifier. One SVM for each of this malware label and every other label pair
        Because malware labels are added sequentially, generating an SVM for the label currently being added vs. every other label added so far
        will create the appropriate number of SVMs (n*(n-1)/2)
        """
        # NOTE: If this is the first one, there are no SVMs that can be created because there are no other samples
        # Therefore, an empty list will be appended, which is okay
        this_svms: list[SVM] = []
        for other_label in self.malware_labels:
            # only want the samples from our feature_vectors_train that are this label or other label
            this_feature_vectors = np.zeros((0,self.feature_vectors_train.shape[1]), dtype=np.bool)
            this_sample_hashes = []
            for feature_vector, sample_hash in zip(self.feature_vectors_train, self.sample_hashes_train):
                if self.ground_truth.get(sample_hash) == malware_label or self.ground_truth.get(sample_hash) == other_label:
                    this_feature_vectors = np.vstack((this_feature_vectors, feature_vector))
                    this_sample_hashes.append(sample_hash)

            svm = SVM(this_feature_vectors, this_sample_hashes, self.ground_truth)
            svm.define_classes(lambda x: 1 if x == malware_label else -1)
            svm.verbose = self.verbose
            this_svms.append(svm)
        self.svms.append(this_svms)
        self.malware_labels.append(malware_label)
    
    def train_all(self, epochs, lr):
        tick = time.time()
        total_labels = len(self.svms)
        for i, svms in enumerate(self.svms):
            if not len(svms):
                continue
            if self.verbose:
                print(f'Training SVMs for label {i+1}/{total_labels} ({self.malware_labels[i]})')
            total_svms = len(svms)
            for j, svm in enumerate(svms):
                if self.verbose:
                    print(f'Training SVM {j+1}/{total_svms}')
                svm.train(epochs, lr)
        time_elapsed = time.time() - tick
        print(f'Training time: {time_elapsed:.2f} seconds')


    def classify_sample(self, sample: npt.NDArray[np.bool_]) -> str:
        sample_torch = torch.tensor(sample, dtype=torch.float32)
        pred_results = []
        for svms in self.svms:
            this_pred_results = []
            for svm in svms:
                this_pred_results.append(svm.calculate_prediction_torch(sample_torch).item())
            pred_results.append(this_pred_results)
        
        # Voting
        # Each SVM gives +1 vote to one label
        # Most votes wins
        votes = [0]*len(self.malware_labels)
        for pos_result_index, result_list in enumerate(pred_results):
            if not len(result_list):  # this if the first one, which is empty
                continue
            for neg_result_index, result in enumerate(result_list):
                if result > 0:
                    votes[pos_result_index] += 1
                elif result < 0:
                    votes[neg_result_index] += 1
                else:  # If it is exactly 0, no one gets a vote
                    pass
            
        prediction_result_label = self.malware_labels[np.argmax(votes)]
        return prediction_result_label

    def test(self):
        n = len(self.malware_labels)

        correct_counter = 0
        confusion_matrix = np.zeros((n, n), dtype=int)
        for sample, sample_hash in zip(self.feature_vectors_test, self.sample_hashes_test):
            true_label = self.ground_truth.get(sample_hash)
            if true_label is None:
                red_print('Something went wrong.')
                continue
            predicted_label = self.classify_sample(sample)

            if true_label == predicted_label:
                correct_counter += 1
            
            # Build confusion matrix
            true_index = self.malware_labels.index(true_label)
            predicted_index = self.malware_labels.index(predicted_label)
            confusion_matrix[true_index][predicted_index] += 1

        # Print confusion matrix
        print()
        print('Confusion Matrix:')
        for i, row in enumerate(confusion_matrix):
            for j, val in enumerate(row):
                if i == j:
                    color = COLOR_GREEN
                else:
                    color = COLOR_RED
                print(f'{color}{val: <4}{COLOR_DEFAULT}', end='')
            print()

        print()
        green_print(f'{correct_counter}/{len(self.feature_vectors_test)} were classified correctly.')
        green_print(f'Accuracy: {100*correct_counter/len(self.feature_vectors_test):.2f}%')


def load_data(benign_samples_limit: int = 1000, top_malware_samples_limit: int = 0, verbose: bool = True) -> tuple[npt.NDArray[np.bool_], list[str], dict[str, str]]:
    """
    Loads the full dataset from the data contained in the dataset directory

    Args:
        benign_samples_limit (int): maximum number of benign samples to load
        top_malware_samples_limit (int): number of malware samples to load, prioritizing the ones with the most samples
        verbose (bool): whether to print data summary after loading

    Returns:
        feature_vectors (2D numpy array): contains an array for each sample, with 1s and 0s denoting whether each feature is present in that sample
        sample_hashes (list): array that runs parallel to the feature_vectors array. Identifies the hash of each sample in the feature_vectors array.
        ground_truth (dict): maps sample hashes to malware type
    """
    feature_to_index = dict()
    ground_truth = dict()
    feature_count = 0

    # grab all the malicious files
    with open(HASH_MAPPINGS_FILEPATH, 'r') as hash_mappings_file:
        for line in hash_mappings_file:
            if line.startswith('sha256'):  # skip the header line
                continue
            line = line.strip()
            ground_truth[line.split(',')[0]] = line.split(',')[1]
    
    # If we defined a top malware samples limit, resolve that now
    # Go through ground truth and delete malware samples we do not want
    if top_malware_samples_limit > 0:
        malware_labels, counts = np.unique(list(ground_truth.values()), return_counts=True)
        labels_counts_sorted = sorted(zip(malware_labels, counts), key=lambda x: x[1], reverse=True)
        ground_truth_items = list(ground_truth.items())
        for sample_hash, malware_label in ground_truth_items:
            if malware_label not in [x[0] for x in labels_counts_sorted][:top_malware_samples_limit]:
                del ground_truth[sample_hash]
    
    # now grab however many benign samples we want
    if benign_samples_limit > 0:
        benign_sample_hashes = []
        for filename in os.listdir(FEATURE_VECTORS_FILEPATH):
            if filename not in ground_truth:  # it is benign
                benign_sample_hashes.append(filename)
        shuffle(benign_sample_hashes)
        for benign_sample_hash in benign_sample_hashes[:benign_samples_limit]:
            ground_truth[benign_sample_hash] = 'Benign'

    # this will be easier if we count how many unique features there are total first
    for sample_hash in ground_truth:
        with open(os.path.join(FEATURE_VECTORS_FILEPATH, sample_hash), 'r') as feature_vectors_file:
            for feature in feature_vectors_file:
                feature = feature.strip()
                if feature not in feature_to_index:
                    feature_to_index[feature] = feature_count
                    feature_count += 1
    
    num_samples = len(ground_truth)
    num_features = len(feature_to_index)
    green_print(f'Found {num_features} unique features!\n')

    # Now create the feature matrix
    feature_vectors = np.zeros((num_samples, num_features), dtype=np.bool)
    sample_hashes = []
    sample_index = 0
    for filename in ground_truth:
        # check that this file is actually present
        # there is no way it would not be but just to make sure I guess
        full_filepath = os.path.join(FEATURE_VECTORS_FILEPATH, filename)
        if not os.path.exists(full_filepath):
            red_print(f'Could not find the file corresponding to hash {filename}. Skipping...')
            continue
        sample_hashes.append(filename)

        # add this data to the corresponding line in the feature vector
        with open(full_filepath, 'r') as feature_vectors_file:
            for feature in feature_vectors_file:
                feature_index = feature_to_index.get(feature.strip())
                feature_vectors[sample_index][feature_index] = 1
        sample_index += 1

    # Print out information if verbose is set to true
    if verbose:
        green_print('Loaded dataset!')
        print(f'{feature_vectors.shape[0]} samples')
        print(f'{feature_vectors.shape[1]} features per sample')
        print(f'Percent that is benign: {(benign_samples_limit/feature_vectors.shape[0])*100:.2f}%')
        print()

    return feature_vectors, sample_hashes, ground_truth


def malware_vs_benign_classifier(feature_vectors: npt.NDArray[np.bool_], sample_hashes: list[str], ground_truth: dict[str, str]) -> None:
    svm = SVM(feature_vectors, sample_hashes, ground_truth)
    svm.define_classes(lambda x: 1 if x == 'Benign' else -1)
    svm.verbose = True

    svm.evaluate(20, 1e-4, 0.8)


def one_vs_all(feature_vectors: npt.NDArray[np.bool_], sample_hashes: list[str], ground_truth: dict[str, str]) -> None:
    multi_svm = SVM_OneVsAll(feature_vectors, sample_hashes, ground_truth, train_test_split=0.8)
    multi_svm.verbose = False
    all_malware_labels = set(ground_truth.values())
    print(f'Found {len(all_malware_labels)} unique malware types in the dataset')
    for malware_label in all_malware_labels:
        multi_svm.append_svm(malware_label)
    
    multi_svm.train_all(5, 1e-3)

    multi_svm.test()


def one_vs_one(feature_vectors: npt.NDArray[np.bool_], sample_hashes: list[str], ground_truth: dict[str, str]) -> None:
    multi_svm = SVM_OneVsOne(feature_vectors, sample_hashes, ground_truth, train_test_split=0.8)
    multi_svm.verbose = False
    all_malware_labels = set(ground_truth.values())
    print(f'Found {len(all_malware_labels)} unique malware types in the dataset')
    for malware_label in all_malware_labels:
        multi_svm.append_svm(malware_label)

    multi_svm.train_all(5, 1e-3)

    multi_svm.test()


def sklearn_svm_comparison(feature_vectors, labels):
    print("\n--- SVM Kernel Comparison (scikit-learn) ---")
    X_train, X_test, y_train, y_test = train_test_split(
        feature_vectors, labels, test_size=0.2, random_state=42
    )

    # Linear SVM
    linear_clf = LinearSVC(max_iter=2000)
    linear_clf.fit(X_train, y_train)
    y_pred_linear = linear_clf.predict(X_test)
    acc_linear = accuracy_score(y_test, y_pred_linear)
    print(f"Linear SVM Accuracy: {acc_linear * 100:.2f}%")

    # RBF SVM
    rbf_clf = SVC(kernel='rbf', gamma='scale')
    rbf_clf.fit(X_train, y_train)
    y_pred_rbf = rbf_clf.predict(X_test)
    acc_rbf = accuracy_score(y_test, y_pred_rbf)
    print(f"RBF Kernel SVM Accuracy: {acc_rbf * 100:.2f}%")


def main():
    # Malware vs. Benign Classifier
    # feature_vectors, sample_hashes, ground_truth = load_data(benign_samples_limit=5561)
    # malware_vs_benign_classifier(feature_vectors, sample_hashes, ground_truth)

    # One-Vs-All
    # For this type, we do not want the benign samples
    print('ONE VS ALL')
    feature_vectors, sample_hashes, ground_truth = load_data(benign_samples_limit=0, top_malware_samples_limit=20)
    one_vs_all(feature_vectors, sample_hashes, ground_truth)

    # One-Vs-One
    # For this type, we do not want the benign samples
    print('ONE VS ONE')
    feature_vectors, sample_hashes, ground_truth = load_data(benign_samples_limit=0, top_malware_samples_limit=20)
    one_vs_one(feature_vectors, sample_hashes, ground_truth)

    """
    # SVM Comparisons
    feature_vectors, sample_hashes, ground_truth = load_data(benign_samples_limit=1000, verbose=False)
    #convert malware labels into numeric values for sklearn
    labels = [1 if ground_truth[h] == 'Benign' else 0 for h in sample_hashes]
    #run comparison 
    sklearn_svm_comparison(feature_vectors, labels)
    """


if __name__ == '__main__':
    main()
    
