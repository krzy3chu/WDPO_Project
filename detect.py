import json
from pathlib import Path
from typing import Dict

import click
import cv2
from tqdm import tqdm

import numpy as np


def empty_callback(value):
    pass


def detect(img_path: str) -> Dict[str, int]:
    """Object detection function, according to the project description, to implement.

    Parameters
    ----------
    img_path : str
        Path to processed image.

    Returns

    -------
    Dict[str, int]
        Dictionary with quantity of each object.
    """

    # read image
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    # constant threshold values
    thresholds = {
        "red":      [50, 175, 180],
        "yellow":   [210, 20, 27],
        "green":    [160, 33, 57],
        "purple":   [50, 150, 172]
    }
    color = "red"

    # create windows and trackbars
    win_hsv_name = "filtering by hsv representation"
    cv2.namedWindow(win_hsv_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_hsv_name, 700, 700)
    cv2.createTrackbar("s_l", win_hsv_name, thresholds[color][0], 256, empty_callback)  # saturation lower threshold
    cv2.createTrackbar("h_l", win_hsv_name, thresholds[color][1], 180, empty_callback)  # hue lower threshold
    cv2.createTrackbar("h_h", win_hsv_name, thresholds[color][2], 180, empty_callback)  # hue higher threshold

    win_morph_name = "mask processing using morphology operations"
    cv2.namedWindow(win_morph_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_morph_name, 700, 700)
    cv2.createTrackbar("ope", win_morph_name, 6, 10, empty_callback)      # open morphology iterations
    cv2.createTrackbar("clo", win_morph_name, 4, 10, empty_callback)      # close morphology iterations
    cv2.createTrackbar("ero", win_morph_name, 3, 10, empty_callback)      # foreground erode morphology iterations

    win_water_name = "define boundaries with watershed algorithm"
    cv2.namedWindow(win_water_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_water_name, 700, 700)
    cv2.createTrackbar("dil", win_water_name, 6, 10, empty_callback)      # background dilate morphology iterations

    while True:
        key_code = cv2.waitKey(100)
        if key_code == 27:  # escape key pressed
            break

        # get trackbars values
        s_l = cv2.getTrackbarPos("s_l", win_hsv_name)
        h_l = cv2.getTrackbarPos("h_l", win_hsv_name)
        h_h = cv2.getTrackbarPos("h_h", win_hsv_name)
        clo = cv2.getTrackbarPos("clo", win_morph_name)
        ope = cv2.getTrackbarPos("ope", win_morph_name)
        ero = cv2.getTrackbarPos("ero", win_morph_name)
        dil = cv2.getTrackbarPos("dil", win_water_name)

        # create mask based on color thresholds values
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv_low = np.array([h_l, s_l, 0])
        hsv_high = np.array([h_h, 256, 256])
        mask_hsv = cv2.inRange(img_hsv, hsv_low, hsv_high)

        # set all saturation and value to maximum, convert back to BGR and crop image using mask
        img_hsv[:, :, 1:3] = 255
        img_sat = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
        img_masked = cv2.bitwise_and(img_sat, img_sat, mask=mask_hsv)
        cv2.imshow(win_hsv_name, img_masked)

        # process mask with morphology operations
        morph_kernel = np.array([[0, 1, 1, 1, 0],
                                 [1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1],
                                 [0, 1, 1, 1, 0]], dtype=np.ubyte)
        mask_morph = cv2.morphologyEx(mask_hsv, cv2.MORPH_OPEN, morph_kernel, iterations=ope)
        mask_morph = cv2.morphologyEx(mask_morph, cv2.MORPH_CLOSE, morph_kernel, iterations=clo)
        mask_morph = cv2.morphologyEx(mask_morph, cv2.MORPH_ERODE, morph_kernel, iterations=ero)

        # obtain number of skittles, crop image using new mask and show result
        num, markers = cv2.connectedComponents(mask_morph)
        img_masked = cv2.bitwise_and(img, img, mask=mask_morph)
        cv2.putText(img_masked, "Number of skittles: " + str(num-1), (5, img_masked.shape[0]-5), cv2.FONT_HERSHEY_PLAIN, 12, (255, 255, 255), 8)
        cv2.imshow(win_morph_name, img_masked)

        # draw borders using watershed algorithm
        mask_bg = cv2.morphologyEx(mask_morph, cv2.MORPH_DILATE, morph_kernel, iterations=dil)
        mask_border = cv2.subtract(mask_bg, mask_morph)
        markers = markers + 1
        markers[mask_border == 255] = 0
        img_water = img.copy()
        markers = cv2.watershed(img_water, markers)
        img_water[markers == -1] = [0, 0, 0]
        cv2.imshow(win_water_name, img_water)

    red = 1
    yellow = 2
    green = 3
    purple = 4

    return {'red': red, 'yellow': yellow, 'green': green, 'purple': purple}


@click.command()
@click.option('-p', '--data_path', help='Path to data directory',
              type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option('-o', '--output_file_path', help='Path to output file', type=click.Path(dir_okay=False, path_type=Path),
              required=True)
def main(data_path: Path, output_file_path: Path):
    img_list = data_path.glob('*.jpg')

    results = {}

    for img_path in tqdm(sorted(img_list)):
        fruits = detect(str(img_path))
        results[img_path.name] = fruits

    with open(output_file_path, 'w') as ofp:
        json.dump(results, ofp)


if __name__ == '__main__':
    main()
