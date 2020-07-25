from feature_extractor.api import get_features
from sklearn import metrics
import numpy as np
from sklearn.preprocessing import normalize

def get_distance(fixed_image, query_image, distance_metric='euclidean'):
    fixed_feature = normalize(np.array(get_features(fixed_image)).reshape(1, -1))
    query_feature = normalize(np.array(get_features(query_image)).reshape(1, -1))
    similarty_function_map = {
        'cosine' : metrics.pairwise.cosine_similarity,
        'euclidean' : metrics.pairwise.euclidean_distances,
        'manhattan' : metrics.pairwise.manhattan_distances
    }
    distance = similarty_function_map[distance_metric](fixed_feature, query_feature)
    return distance[0][0]


