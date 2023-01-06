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

# ---------------------- color --------------------------------- #
    cv2.namedWindow("image hsv", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("image hsv", 700, 700)
    cv2.createTrackbar("s_l", "image hsv", 0, 255, empty_callback)
    cv2.createTrackbar("clo", "image hsv", 0, 20, empty_callback)
    cv2.createTrackbar("ope", "image hsv", 0, 20, empty_callback)
    cv2.createTrackbar("h_l", "image hsv", 0, 179, empty_callback)
    cv2.createTrackbar("h_h", "image hsv", 179, 179, empty_callback)

    while True:
        key_code = cv2.waitKey(100)
        if key_code == 27:  # escape key pressed
            break

        s_l = cv2.getTrackbarPos("s_l", "image hsv")
        clo = (cv2.getTrackbarPos("clo", "image hsv") * 2) + 1
        clo_kernel = np.ones((clo, clo), np.uint8)
        ope = (cv2.getTrackbarPos("ope", "image hsv") * 2) + 1
        ope_kernel = np.ones((ope, ope), np.uint8)
        h_l = cv2.getTrackbarPos("h_l", "image hsv")
        h_h = cv2.getTrackbarPos("h_h", "image hsv")
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        hsv_low = np.array([h_l, s_l, 0])
        hsv_high = np.array([h_h, 256, 256])
        hsv_mask = cv2.inRange(img_hsv, hsv_low, hsv_high)
        hsv_mask_clo = cv2.morphologyEx(hsv_mask, cv2.MORPH_CLOSE, clo_kernel)
        hsv_mask_ope = cv2.morphologyEx(hsv_mask_clo, cv2.MORPH_ERODE, ope_kernel)

        img_hsv[:, :, 1:3] = 255  # set all saturation and value to maximum
        img_sat = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
        img_masked = cv2.bitwise_and(img_sat, img_sat, mask=hsv_mask_ope)
        cv2.imshow("image hsv", img_masked)

    cv2.waitKey()

    red = 1
    yellow = 2
    green = 3
    purple = 4

    return {'red': red, 'yellow': yellow, 'green': green, 'purple': purple}


@click.command()
@click.option('-p', '--data_path', help='Path to data directory', type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option('-o', '--output_file_path', help='Path to output file', type=click.Path(dir_okay=False, path_type=Path), required=True)
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
