import os
import numpy as np
import numpy.typing as npt
import torch
from typing import Callable

DATASET_FILEPATH = 'dataset'
FEATURE_VECTORS_FILEPATH = os.path.join(DATASET_FILEPATH, 'feature_vectors')
HASH_MAPPINGS_FILEPATH = os.path.join(DATASET_FILEPATH, 'sha256_family.csv')

# Color codes
COLOR_DEFAULT = '\x1b[39m'
COLOR_GREEN = '\x1b[32m'
COLOR_RED = '\x1b[31m'


class SVM:
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
    
    def evaluate(self, epochs: int, lr: float, train_test_split: float):
        # Shuffle and then do train/test split
        p = np.random.permutation(self.feature_vectors.shape[0])
        feature_vectors_shuffled = self.feature_vectors[p]
        sample_hashes_shuffled = [self.sample_hashes[i] for i in p]

        index_split_value = int(train_test_split*feature_vectors_shuffled.shape[0])
        feature_vector_train = feature_vectors_shuffled[:index_split_value]
        feature_vector_test = feature_vectors_shuffled[index_split_value:]
        sample_hashes_train = sample_hashes_shuffled[:index_split_value]
        sample_hashes_test = sample_hashes_shuffled[index_split_value:]

        # Convert everything to torch tensors
        X_train = torch.tensor(feature_vector_train, dtype=torch.float32)
        X_test = torch.tensor(feature_vector_test, dtype=torch.float32)

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
                # Pylance might complain with the "type: ignore" comments
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


def green_print(msg: str):
    """Convenient function for printing green text"""
    print(f'{COLOR_GREEN}{msg}{COLOR_DEFAULT}')

def red_print(msg: str):
    """Convenient function for printing red text"""
    print(f'{COLOR_RED}{msg}{COLOR_DEFAULT}')


def load_data(benign_samples_limit: int = 1000) -> tuple[npt.NDArray[np.bool_], list[str], dict[str, str]]:
    """
    Loads the full dataset from the data contained in the dataset directory

    Returns:
        feature_vectors (2D numpy array): contains an array for each sample, with 1s and 0s denoting whether each feature is present in that sample
        sample_hashes (list): array that runs parallel to the feature_vectors array. Identifies the hash of each sample in the feature_vectors array.
        ground_truth (dict): maps sample hashes to malware type
    """
    feature_to_index = dict()
    ground_truth = dict()
    num_samples = 0
    benign_samples = 0
    feature_count = 0

    with open(HASH_MAPPINGS_FILEPATH, 'r') as hash_mappings_file:
        for line in hash_mappings_file:
            if line.startswith('sha256'):  # skip the header line
                continue
            line = line.strip()
            ground_truth[line.split(',')[0]] = line.split(',')[1]

    # this will be easier if we count how many unique features there are total first
    for filename in os.listdir(FEATURE_VECTORS_FILEPATH):
        if filename not in ground_truth:  # it is a benign sample
            if benign_samples >= benign_samples_limit:  # we have enough benign samples already
                continue
            benign_samples += 1
            ground_truth[filename] = 'Benign'
        num_samples += 1

        with open(os.path.join(FEATURE_VECTORS_FILEPATH, filename), 'r') as feature_vectors_file:
            for feature in feature_vectors_file:
                feature = feature.strip()
                if feature not in feature_to_index:
                    feature_to_index[feature] = feature_count
                    feature_count += 1

    num_features = len(feature_to_index)
    green_print(f'Found {num_features} unique features!\n')

    # Now create the feature matrix
    feature_vectors = np.zeros((num_samples, num_features), dtype=bool)
    sample_hashes = []
    sample_index = 0
    for filename in ground_truth:
        # check that this file is actually present
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

    return feature_vectors, sample_hashes, ground_truth


def main():
    benign_samples_limit = 5561
    feature_vectors, sample_hashes, ground_truth = load_data(benign_samples_limit)
    green_print('Loaded dataset!')
    print(f'{feature_vectors.shape[0]} samples')
    print(f'{feature_vectors.shape[1]} features per sample')
    print(f'Percent that is benign: {(benign_samples_limit/feature_vectors.shape[0])*100:.2f}%')
    print()

    svm = SVM(feature_vectors, sample_hashes, ground_truth)
    svm.define_classes(lambda x: 1 if x == 'Benign' else -1)
    svm.verbose = True

    svm.evaluate(20, 1e-4, 0.8)



if __name__ == '__main__':
    main()
