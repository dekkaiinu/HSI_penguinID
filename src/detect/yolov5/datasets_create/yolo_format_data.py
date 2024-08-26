def yolo_format_data(bbox, cls_num=0, width=2048, height=1080):
    center_x = bbox[0] + (bbox[2] / 2)
    center_y = bbox[1] + (bbox[3] / 2)
    bbox_str = []
    text = (str(cls_num) + " " + str(round(center_x / width, 4)) + " " + str(round(center_y / height, 4)) + " " + 
            str(round(bbox[2] / width, 4)) + " " + str(round(bbox[3] / height, 4)))
    return text