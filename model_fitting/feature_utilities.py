from fourbynine import *
import numpy as np


def count_features(heuristic, position, player):
    """
    Given a heuristic, returns a count of the number of features detected per feature-group in the given position for the given player.
    For example, if the first feature group of a heuristic contained all horizontal connect-twos, and the second feature group contained
    all horizontal connect threes, and we passed in a position with a single horizontal connect 3 for the black player and a single
    horizontal connect 2 for the white player, we would return [2, 1] for the black player and [1, 0] for the white player (since a
    connect 3 necessarily contains two connect 2s).

    Args:
        heuristic: The heuristic containing a list of feature groups to count.
        position: The position to count features over.
        player: The player to count features for.

    Returns:
        A list of counts of features per feature-group contained in the position for the given player.
    """
    feature_counts_per_group = []
    for feature_group in heuristic.get_feature_group_weights():
        feature_counts_per_group.append(0)

    for feature in heuristic.get_features_with_metadata():
        if feature.feature.contained_in(position, player):
            feature_counts_per_group[feature.weight_index] += 1
    return feature_counts_per_group


def pattern_to_array(pattern):
    """
    Given a SWIG ninarow_pattern object, return a numpy array representing the same pattern. output[0][0] represents the LSB of the input pattern.

    Args:
        pattern: The ninarow_pattern to encode as an array.

    Returns:
        A 2d numpy array containing the pattern passed in.
    """
    return np.array(list(pattern.to_string()[::-1]), dtype=int).reshape(pattern.get_height(), pattern.get_width())


def array_to_pattern(array):
    """
    Given a 2-dimensional numpy array representing a board pattern, return an ninarow_pattern object representing the same pattern.

    Args:
        array: The 2d array containing the pattern to represent.

    Returns:
        The ninarow_pattern represented by the passed in array.
    """
    return fourbynine_pattern(int("".join(array.reshape(array.size).astype(int).astype(str)[::-1]), 2))


def feature_to_kernel(feature):
    """
    Given an ninarow_heuristic_feature, construct a kernel. A kernel is a 3 dimensional array of depth 2, where the first 2d layer
    encodes the piece pattern of the feature and the second 2d layer encodes the space pattern of the feature. The kernel's size is
    shrunk down to be as small as possible but still containing the entire extents of the feature, i.e., the output of this function
    will be the same for all features that are identical under translations.

    Args:
        feature: The feature to transform into a kernel.

    Returns:
        A minimal three-dimensional representation of the passed in feature.
    """
    if feature.pieces.is_empty() and feature.spaces.is_empty():
        return []

    full_kernel = np.dstack(
        (pattern_to_array(feature.pieces), pattern_to_array(feature.spaces)))

    def update_extents_if_not_empty(row_extents, col_extents, pattern):
        if not pattern.is_empty():
            return (min(pattern.min_row(), row_extents[0]), max(pattern.max_row(), row_extents[1])), (min(pattern.min_col(), col_extents[0]), max(pattern.max_col(), col_extents[1]))
        return row_extents, col_extents
    row_extents = 0, 0
    col_extents = 0, 0
    row_extents, col_extents = update_extents_if_not_empty(
        row_extents, col_extents, feature.pieces)
    row_extents, col_extents = update_extents_if_not_empty(
        row_extents, col_extents, feature.spaces)
    return full_kernel[row_extents[0]:row_extents[1]+1, col_extents[0]:col_extents[1]+1, :]


def generate_features_from_kernel(kernel, rotations, reflections, feature_extents, min_space_occupancy):
    """
    Given a kernel, generate a set of features that the kernel could produce under various transformations. If rotations and reflections are false, only
    generate all possible translations of the kernel within the extents specificed by feature_extents. If rotation or reflection is specified, additionally
    generate all rotations/reflections of the kernel (as well as their translations).

    Args:
        kernel: The generating kernel for features.
        rotations: If true, generate rotations of the kernel as well.
        reflections: If true, generate reflections of the kernel as well.
        feature_extents: A tuple containing the desired dimensions of the output features. Must be larger than the input kernel.
        min_space_occupancy: The minimum space occupancy of the output features.

    Returns:
        A list of features generated from the kernel with extents feature_extents, including all translations of the kernel that can fit within the feature_extents,
        as well as optionally all rotations and reflections of said kernel along with their translations. The features generated will all have the specified min_space_occupancy.
    """
    def append_if_unique(kernels, new_array):
        for array in kernels:
            if np.array_equal(array, new_array):
                return False
        kernels.append(new_array)
        return True

    def generate_and_add_new_kernels(kernels, generating_function):
        new_candidates = []
        for kernel in kernels:
            new_candidates.append(generating_function(kernel))
        for candidate in new_candidates:
            append_if_unique(kernels, candidate)

    kernels = [kernel]
    if rotations:
        for i in range(1, 4):
            generate_and_add_new_kernels(kernels, lambda x: np.rot90(x, i))
    if reflections:
        generate_and_add_new_kernels(kernels, lambda x: np.flip(x, 0))
        generate_and_add_new_kernels(kernels, lambda x: np.flip(x, 1))
        generate_and_add_new_kernels(
            kernels, lambda x: np.transpose(x, (1, 0, 2)))
        generate_and_add_new_kernels(
            kernels, lambda x: np.transpose(np.flip(x, 0), (1, 0, 2)))
    full_output_kernels = []
    for kernel in kernels:
        for i in range(0, feature_extents[0] - kernel.shape[0] + 1):
            for j in range(0, feature_extents[1] - kernel.shape[1] + 1):
                new_output_kernel = np.zeros(
                    (feature_extents[0], feature_extents[1], 2))
                new_output_kernel[i:i+kernel.shape[0],
                                  j:j+kernel.shape[1], :] = kernel
                full_output_kernels.append(new_output_kernel)

    def full_kernel_to_feature(kernel, min_space_occupancy):
        return fourbynine_heuristic_feature(array_to_pattern(kernel[:, :, 0]), array_to_pattern(kernel[:, :, 1]), min_space_occupancy)

    return list(map(lambda x: full_kernel_to_feature(x, min_space_occupancy), full_output_kernels))


def generate_feature_transformations(feature, rotations, reflections):
    """
    Given an ninarow_heuristic_feature, generate all possible translations of the feature. Optionally generate all possible rotations and reflections as well.

    Args:
        feature: The feature to generate translations/rotations/reflections of.
        rotations: If true, include rotations of the feature.
        reflections: If true, include reflections of the feature.
    """
    return generate_features_from_kernel(feature_to_kernel(feature), rotations, reflections, (feature.pieces.get_height(), feature.pieces.get_width()), feature.min_space_occupancy)
