def get_obj_img_by_bbox(frame, bbox):
    """[Given a image and a bounding box it will return cropped image]

    Args:
        frame ([3d numpy array]): [It's an image]
        bbox ([list]): [bounding box with x, y, width, height]

    Returns:
        [np.array]: [cropped image]
    """
    x, y = bbox[0], bbox[1]
    w, h = bbox[2], bbox[3]
    obj_img = frame[y:y+h, x:x+w, :]
    rgb_frame = obj_img[:, :, ::-1]
    return rgb_frame