# project-root/statistics/normalization.py

import numpy as np

def min_max_normalize(data, min_val=0.0, max_val=1.0):
    """
    Applies Min-Max Normalization to a list or numpy array of data.
    Scales data to the range [min_val, max_val].

    Formula: scaled_data = min_val + (data - data_min) * (max_val - min_val) / (data_max - data_min)

    Args:
        data: A list or numpy array of numerical data.
        min_val: The minimum value of the output range (default is 0.0).
        max_val: The maximum value of the output range (default is 1.0).

    Returns:
        A numpy array of the normalized data, or the original data if normalization is not possible.
    """
    data_np = np.asarray(data, dtype=float)

    data_min = np.min(data_np)
    data_max = np.max(data_np)

    # Handle cases where data_min == data_max (e.g., all values are the same)
    if data_max - data_min == 0:
        print("Warning: Data range is zero. Cannot perform Min-Max Normalization.")
        # Return array filled with the midpoint of the target range or the original value scaled
        # Let's return an array of the scaled original value
        # If original value is data_min, scaled should be min_val
        # If original value is data_max, scaled should be max_val
        # Since min_val == max_val, return array filled with min_val
        if min_val == max_val:
             return np.full_like(data_np, min_val)
        else:
             # If target range is not zero, but data range is, return midpoint
             return np.full_like(data_np, (min_val + max_val) / 2.0)


    # Apply the formula
    scaled_data = min_val + (data_np - data_min) * (max_val - min_val) / (data_max - data_min)

    return scaled_data

def standardize(data):
    """
    Applies Standardization (Z-score normalization) to a list or numpy array of data.
    Scales data to have a mean of 0 and a standard deviation of 1.

    Formula: scaled_data = (data - mean) / std_dev

    Args:
        data: A list or numpy array of numerical data.

    Returns:
        A numpy array of the standardized data, or the original data if standardization is not possible.
    """
    data_np = np.asarray(data, dtype=float)

    mean = np.mean(data_np)
    std_dev = np.std(data_np)

    # Handle cases where standard deviation is zero (e.g., all values are the same)
    if std_dev == 0:
        print("Warning: Standard deviation is zero. Cannot perform Standardization.")
        # Return array filled with 0 (since mean is subtracted)
        return np.zeros_like(data_np)

    # Apply the formula
    scaled_data = (data_np - mean) / std_dev

    return scaled_data

# Example Usage (within a statistical model like bayesian.py or elsewhere):
# Assume 'scores' is a list of scores from game trials
# from statistics.normalization import min_max_normalize, standardize
#
# normalized_scores_0_1 = min_max_normalize(scores, 0, 1)
# standardized_scores = standardize(scores)
#
# # You would typically apply this to the features *before* feeding them to the AI model.
# # For example, in the process method of bayesian.py before returning the dict,
# # or possibly in the trainer/runner after getting the stat_output.
#
# # Example application within a dict of features:
# # processed_features = { ... 'mean_score': 5.4, 'mean_steps': 100 ... }
# # if self.config.get('normalize_features', False):
# #    if 'mean_score' in processed_features:
# #       processed_features['mean_score_normalized'] = min_max_normalize([processed_features['mean_score']], 0, 1)[0]
# #    if 'mean_steps' in processed_features:
# #       processed_features['mean_steps_normalized'] = min_max_normalize([processed_features['mean_steps']], 0, 1)[0]
# # Note: Normalizing a single value requires context (min/max or mean/std_dev) from the *entire dataset*
# # or a predefined range. The functions above assume normalization is applied to a batch/list of values.
# # If normalizing a single feature derived from a batch (like mean_score), you'd normalize the list of means
# # across different batches/experiments, or use a running min/max from previous data, which is more complex.
# # A simpler approach for features derived from a batch is to normalize the raw metrics (scores, steps)
# # *before* calculating means/stats, or normalize the final statistical features based on expected ranges
# # or statistics collected over many experiments.