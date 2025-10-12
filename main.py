import os
import numpy as np
import numpy.typing as npt

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
    features, sample_hashes, ground_truth = load_data(benign_samples_limit)
    green_print('Loaded dataset!')
    print(f'{features.shape[0]} samples')
    print(f'{features.shape[1]} features per sample')
    print(f'Percent that is benign: {(benign_samples_limit/features.shape[0])*100:.2f}%')
    print()


if __name__ == '__main__':
    main()
