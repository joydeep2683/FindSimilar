from object_detector.yolo_v3.util import *
import sys

sys.path.append('../')
def get_object_images(image, save_folder):
    if type(image) == str:
        image = cv2.imread(image)
    net_obj = get_yolo_v3_net()
    results = get_object_detection_result(net_obj, image)
    img_name = 0
    for result in results:
        bbox_img = get_obj_img_by_bbox(image, result['box'])
        cv2.imwrite(f"{save_folder}{img_name}.jpg", bbox_img)


if __name__ == "__main__":
    args = sys.argv
    get_object_images(args[1], args[2])
    # print(sys.argv)