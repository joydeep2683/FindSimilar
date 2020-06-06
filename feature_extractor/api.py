from feature_extractor.util import CommonFeatureExtractor
import cv2
import sys

def get_features(image_path):
    image = cv2.imread(image_path)
    ext_obj = CommonFeatureExtractor().get_instance()
    features = ext_obj.get_feature_vectors(image)
    print(features)
    return features


if __name__ == "__main__":
    args = sys.argv[1:]
    get_features(args[0])