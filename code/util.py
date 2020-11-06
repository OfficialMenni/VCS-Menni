import cv2
import numpy as np
from constants import MEDIA_DIR, DATABASE_DIR


# Start main UTILS

def format_v(vector, tresh=4):
    """
    Sort the vector and apply a little filter on false positives.
        param:
            vector: unsorted vector of matching scores
            thresh: threshold for filtering
        return:
            vector: sorted and filtered vector. vector[0] == -1 if false positive is detected
          """
    if vector is None:
        return [-1, -1]
    pivot = sorted(vector, reverse=True)
    if pivot[0] == 0:
        vector[0] = -1
    if pivot[0] - pivot[1] <= tresh:
        vector[0] = -1
    return vector


def show(yolo, sliced, warped, names, curr_name, flag=0, outsize=None):
    """
    Print output images.
        param:
            yolo: output image with bounding boxes
            sliced: sliced image
            warped: warped image
            names: vector of detected names
            curr_name: name formatted for the retrieved image
            flag: 0 if rectification was successful
    """
    if outsize is not None:
        yolo = cv2.resize(yolo, outsize)
    cv2.imshow("Paintings and people detection output", yolo)
    if flag == 0:
        if warped is not None:
            cv2.imshow("Processed image (successful rectification)", warped)
        else:
            cv2.imshow("Processed image (failed rectification)", sliced)
        cv2.imshow("{}".format(curr_name), cv2.imread(DATABASE_DIR + names[0]))
    else:
        cv2.imshow("Sliced image", sliced)
        cv2.imshow("Warped image", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_boxes(image, boxlist, color=(0, 0, 255), name_color=(255, 255, 255), label="Person"):
    """
    Draw detected bounding boxes on an image
        param:
            image: image to draw on
            boxlist: list of bounding boxes coordinates (YOLOv3 format)
            color: BB color
            name_color: label color
            label: text printed for this detection
        return:
            image: image with drawed bounding box
          """
    for box in boxlist:
        cv2.rectangle(image, (round(box[0]), round(box[1])), (round(box[0] + box[2]), round(box[1] + box[3])),
                      color, 2)
        cv2.putText(image, label, (round(box[0]) - 10, round(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    name_color, 2)
    return image


def print_people(people):
    if len(people) > 0:
        for person in people:
            xA, xB, yA, yB = get_bounding_box(person)
            x0 = xA
            y0 = yA
            h = abs(yB - yA)
            w = abs(xB - xA)
            print("Detected person in coord (before window resize): X = {}, Y = {}, W = {}, H = {}".format(x0, y0, w,
                                                                                                           h))


def print_painting(painting):
    xA, xB, yA, yB = get_bounding_box(painting)
    x0 = xA
    y0 = yA
    h = abs(yB - yA)
    w = abs(xB - xA)
    print("Detected painting in coord (before window resize): X = {}, Y = {}, W = {}, H = {}".format(x0, y0, w, h))


def get_bounding_box(bbox):
    y1 = max(0, int(round(bbox[0])))
    y2 = max(0, int(round(bbox[0] + bbox[2])))
    x1 = max(0, int(round(bbox[1])))
    x2 = max(0, int(round(bbox[1] + bbox[3])))
    return y1, y2, x1, x2


def people_filter(bb_people, bb_painting):
    ret_people = bb_people.copy()
    for person in bb_people:
        person_coordinates = get_bounding_box(person)
        for paint in bb_painting:
            paint_coordinates = get_bounding_box(paint)
            if person_coordinates[2] >= paint_coordinates[2] and person_coordinates[0] >= paint_coordinates[0] and \
                    person_coordinates[3] <= paint_coordinates[3] and person_coordinates[1] <= paint_coordinates[1]:
                ret_people.remove(person)
                break
    return ret_people
# End main UTILS
