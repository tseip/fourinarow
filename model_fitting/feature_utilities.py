from fourbynine import *
import numpy as np

def count_features(heuristic, position, player):
    feature_counts_per_pack = []
    for feature_pack in heuristic.get_feature_packs():
        feature_count = 0
        for feature in feature_pack.features:
            if feature.is_active(position, player) and feature.contained(position, player):
                feature_count += 1
        feature_counts_per_pack.append(feature_count)
    return feature_counts_per_pack

def feature_to_kernel(feature):
    pieces = feature.pieces
    
    def get_dims(pattern):
        return (pieces.max_row() - pieces.min_row() + 1, pieces.max_col() - pieces.min_col() + 1)
    return np.array(list(pieces.to_string()[::-1]), dtype=int).reshape(pieces.get_height(), pieces.get_width())[pieces.min_row():pieces.max_row()+1, pieces.min_col():pieces.max_col()+1]
    
def generate_features_from_kernel(kernel, rotations, reflections):
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
        generate_and_add_new_kernels(kernels, lambda x: np.transpose(x))
        generate_and_add_new_kernels(kernels, lambda x: np.transpose(np.flip(x, 0)))
    #for array in kernels:
            
    print(kernels)
    #pieces = kernel.pieces
    #string = pieces.to_string()
    #print(bytes(string, 'utf-8'))
    #print(np.array(list(string)).reshape(4, 9))
    #spaces = kernel.spaces
    return "Hello"

if __name__ == "__main__":
    kernel = feature_to_kernel(fourbynine_heuristic_feature(fourbynine_pattern(0b1000001011), fourbynine_pattern(0), 0))
    print(generate_features_from_kernel(kernel, True, True))