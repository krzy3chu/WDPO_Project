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
    # thresholds = {
    #     "red":      [50, 175, 180],
    #     "yellow":   [210, 20, 27],
    #     "green":    [60, 25, 60],
    #     "purple":   [50, 150, 172]
    # }
    # color = "green"

    # create windows and trackbars
    win_sure_fg = "filtering to find area that is foreground for sure"
    cv2.namedWindow(win_sure_fg, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_sure_fg, 700, 700)
    cv2.createTrackbar("s_l", win_sure_fg, 120, 256, empty_callback)      # fg saturation lower threshold
    cv2.createTrackbar("h_l", win_sure_fg, 11, 180, empty_callback)      # fg hue lower threshold
    cv2.createTrackbar("h_h", win_sure_fg, 17, 180, empty_callback)    # fg hue higher threshold
    cv2.createTrackbar("ope", win_sure_fg, 5, 10, empty_callback)       # fg open morphology iterations
    cv2.createTrackbar("ero", win_sure_fg, 3, 10, empty_callback)       # fg erode morphology iterations
    cv2.createTrackbar("col", win_sure_fg, 0, 1, empty_callback)        # fg choose between color space representation

    win_sure_bg = "filtering to find area that is background for sure"
    cv2.namedWindow(win_sure_bg, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_sure_bg, 700, 700)
    cv2.createTrackbar("s_l", win_sure_bg, 75, 256, empty_callback)      # bg saturation lower threshold
    cv2.createTrackbar("h_l", win_sure_bg, 9, 180, empty_callback)      # bg hue lower threshold
    cv2.createTrackbar("h_h", win_sure_bg, 20, 180, empty_callback)    # bg hue higher threshold
    cv2.createTrackbar("clo", win_sure_bg, 5, 10, empty_callback)       # bg close morphology iterations
    cv2.createTrackbar("dil", win_sure_bg, 3, 10, empty_callback)       # bg dilate morphology iterations
    cv2.createTrackbar("col", win_sure_bg, 0, 1, empty_callback)        # bg choose between color space representation

    win_water = "define boundaries with watershed algorithm"
    cv2.namedWindow(win_water, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_water, 700, 700)

    hue_offset = 10  # hue offset, used to move red color to higher values only

    morph_kernel = np.array([[0, 1, 1, 1, 0],
                             [1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1],
                             [0, 1, 1, 1, 0]], dtype=np.ubyte)

    while True:
        key_code = cv2.waitKey(100)
        if key_code == 27:  # escape key pressed
            break

        # get trackbars values
        fg_s_l = cv2.getTrackbarPos("s_l", win_sure_fg)
        fg_h_l = cv2.getTrackbarPos("h_l", win_sure_fg) + hue_offset
        fg_h_h = cv2.getTrackbarPos("h_h", win_sure_fg) + hue_offset
        fg_ope = cv2.getTrackbarPos("ope", win_sure_fg)
        fg_ero = cv2.getTrackbarPos("ero", win_sure_fg)
        fg_col = cv2.getTrackbarPos("col", win_sure_fg)

        bg_s_l = cv2.getTrackbarPos("s_l", win_sure_bg)
        bg_h_l = cv2.getTrackbarPos("h_l", win_sure_bg) + hue_offset
        bg_h_h = cv2.getTrackbarPos("h_h", win_sure_bg) + hue_offset
        bg_clo = cv2.getTrackbarPos("clo", win_sure_bg)
        bg_dil = cv2.getTrackbarPos("dil", win_sure_bg)
        bg_col = cv2.getTrackbarPos("col", win_sure_bg)

        # convert image to hsv space, set hue values lower than hue_offset to hue_value+180
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_hsv[:, :, 0] += ((img_hsv[:, :, 0] < hue_offset) * 180).astype(np.ubyte)

        # create sure foreground mask based on color thresholds values and morphology operations
        fg_hsv_low = np.array([fg_h_l, fg_s_l, 0])
        fg_hsv_high = np.array([fg_h_h, 256, 256])
        mask_fg = cv2.inRange(img_hsv, fg_hsv_low, fg_hsv_high)
        mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_OPEN, morph_kernel, iterations=fg_ope)
        mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_ERODE, morph_kernel, iterations=fg_ero)

        # choose viewed image color space representation, crop image using mask
        if fg_col:
            img_hsv[:, :, 1:3] = 255
            img_sat = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
            img_masked = cv2.bitwise_and(img_sat, img_sat, mask=mask_fg)
        else:
            img_masked = cv2.bitwise_and(img, img, mask=mask_fg)
        cv2.imshow(win_sure_fg, img_masked)

        # convert image to hsv space, set hue values lower than hue_offset to hue_value+180
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_hsv[:, :, 0] += ((img_hsv[:, :, 0] < hue_offset) * 180).astype(np.ubyte)

        # create sure background mask based on color thresholds values and morphology operations
        bg_hsv_low = np.array([bg_h_l, bg_s_l, 0])
        bg_hsv_high = np.array([bg_h_h, 256, 256])
        mask_bg = cv2.inRange(img_hsv, bg_hsv_low, bg_hsv_high)
        mask_bg = cv2.morphologyEx(mask_bg, cv2.MORPH_CLOSE, morph_kernel, iterations=bg_clo)
        mask_bg = cv2.morphologyEx(mask_bg, cv2.MORPH_DILATE, morph_kernel, iterations=bg_dil)

        # choose viewed image color space representation, crop image using mask
        if bg_col:
            img_hsv[:, :, 1:3] = 255
            img_sat = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
            img_masked = cv2.bitwise_and(img_sat, img_sat, mask=mask_bg)
        else:
            img_masked = cv2.bitwise_and(img, img, mask=mask_bg)
        cv2.imshow(win_sure_bg, img_masked)

        # draw borders using watershed algorithm
        num, markers = cv2.connectedComponents(mask_fg)
        mask_border = cv2.subtract(mask_bg, mask_fg)
        markers = markers + 1
        markers[mask_border == 255] = 0
        img_water = img.copy()
        markers = cv2.watershed(img_water, markers)
        img_water[markers == -1] = [0, 0, 0]
        cv2.putText(img_water, "Number of skittles: " + str(num-1), (5, img_water.shape[0]-5), cv2.FONT_HERSHEY_PLAIN, 12, (255, 255, 255), 8)
        cv2.imshow(win_water, img_water)

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
