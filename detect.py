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
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("image", 700, 700)
    cv2.imshow("image", img)

    cv2.namedWindow("image hsv", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("image hsv", 700, 700)
    cv2.createTrackbar("s_l", "image hsv", 194, 256, empty_callback)  # saturation lower threshold
    cv2.createTrackbar("h_l", "image hsv", 30, 180, empty_callback)  # hue lower threshold
    cv2.createTrackbar("h_h", "image hsv", 55, 180, empty_callback)  # hue higher threshold
    cv2.createTrackbar("ope", "image hsv", 4, 10, empty_callback)  # open morphology iterations
    cv2.createTrackbar("clo", "image hsv", 1, 10, empty_callback)  # close morphology iterations

    cv2.namedWindow("image watershed", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("image watershed", 700, 700)
    cv2.createTrackbar("ero", "image watershed", 2, 10, empty_callback)  # sure fg erode morphology iterations
    cv2.createTrackbar("dil", "image watershed", 2, 10, empty_callback)  # sure bg dilate morphology iterations

    while True:
        key_code = cv2.waitKey(100)
        if key_code == 27:  # escape key pressed
            break

        # get trackbar values
        s_l = cv2.getTrackbarPos("s_l", "image hsv")
        h_l = cv2.getTrackbarPos("h_l", "image hsv")
        h_h = cv2.getTrackbarPos("h_h", "image hsv")
        clo = cv2.getTrackbarPos("clo", "image hsv")
        ope = cv2.getTrackbarPos("ope", "image hsv")

        # create mask based on thresholds values
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv_low = np.array([h_l, s_l, 0])
        hsv_high = np.array([h_h, 256, 256])
        hsv_mask = cv2.inRange(img_hsv, hsv_low, hsv_high)
        morph_kernel = np.ones((3, 3))
        mask_ope = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, morph_kernel, iterations=ope)
        mask_clo = cv2.morphologyEx(mask_ope, cv2.MORPH_CLOSE, morph_kernel, iterations=clo)

        # set all saturation and value to maximum
        img_hsv[:, :, 1:3] = 255

        # convert back to BGR and crop image using mask
        img_sat = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
        img_masked = cv2.bitwise_and(img_sat, img_sat, mask=mask_clo)
        cv2.imshow("image hsv", img_masked)

        # watershed tutorial
        # kernel = np.ones((3, 3), np.uint8)
        # sure_bg = cv2.dilate(hsv_mask_clo, kernel, iterations=3)
        # dist_transform = cv2.distanceTransform(hsv_mask, cv2.DIST_L1, 5, dstType=cv2.CV_8U)
        # ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        # sure_fg = np.uint8(sure_fg)
        # unknown = cv2.subtract(sure_bg, sure_fg)
        # ret, markers = cv2.connectedComponents(sure_fg)
        # markers = markers + 1
        # markers[unknown == 255] = 0
        # img_water = img.copy()
        # markers = cv2.watershed(img_water, markers)
        # img_water[markers == -1] = [255, 0, 0]
        # cv2.imshow("image watershed", img_water)
        # print(ret-1)

        # get trackbar values
        ero = cv2.getTrackbarPos("ero", "image watershed")
        dil = cv2.getTrackbarPos("dil", "image watershed")

        sure_fg = cv2.morphologyEx(mask_clo, cv2.MORPH_ERODE, morph_kernel, iterations=ero)
        sure_bg = cv2.morphologyEx(mask_clo, cv2.MORPH_DILATE, morph_kernel, iterations=dil)
        unknown = cv2.subtract(sure_bg, sure_fg)
        num, markers = cv2.connectedComponents(sure_fg)
        print(num - 1)
        markers = markers + 1
        markers[unknown == 255] = 0
        img_water = img.copy()
        markers = cv2.watershed(img_water, markers)
        img_water[markers == -1] = [0, 0, 0]
        cv2.imshow("image watershed", img_water)

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
