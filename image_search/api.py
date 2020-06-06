from image_search.util import get_distance
import sys

def get_distance_value(fixed_image, query_image, distance_metric):
    distance = get_distance(fixed_image, query_image, distance_metric)
    print(distance)
    return distance

if __name__ == "__main__":
    args = sys.argv[1:]
    get_distance_value(args[0], args[1], args[2])