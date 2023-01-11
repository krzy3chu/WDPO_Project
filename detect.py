import json
from pathlib import Path
from typing import Dict

import click
import cv2
from tqdm import tqdm

import numpy as np
from my_check import my_check

# ----------------------- define color class, create list of its instances ----------------------- #
hue_offset = 10  # hue offset, used to move red color to higher values only

morph_kernel = np.array([[0, 1, 1, 1, 0],   # kernel used in morphology operations
                         [1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1],
                         [0, 1, 1, 1, 0]], dtype=np.ubyte)

fg_v_l = 40     # foreground value lower threshold
fg_clo = 1      # foreground close morphology iterations
fg_are = 280    # foreground area size lower threshold
bg_clo = 4      # background close morphology iterations
bg_dil = 2      # background dilate morphology iterations
wt_dst = 0.7    # watershed distance transform threshold

# ----------------------- define Color class, create list of its instances ----------------------- #
class Color:
    name = ""

    fg_s_l = 0      # foreground saturation lower threshold
    fg_h_l = 0      # foreground hue lower threshold
    fg_h_h = 255    # foreground hue higher threshold
    bg_s_l = 0      # background saturation lower threshold
    bg_h_l = 0      # background hue lower threshold
    bg_h_h = 255    # background hue higher threshold
    bgr_col = [0, 0, 0]     # color representation in bgr space

    def __init__(self, name, fg_s_l, fg_h_l, fg_h_h, bg_s_l, bg_h_l, bg_h_h, bgr_col):
        self.name   = name
        self.fg_s_l = fg_s_l
        self.fg_h_l = fg_h_l + hue_offset
        self.fg_h_h = fg_h_h + hue_offset
        self.bg_s_l = bg_s_l
        self.bg_h_l = bg_h_l + hue_offset
        self.bg_h_h = bg_h_h + hue_offset
        self.bgr_col = [bgr * 255 for bgr in bgr_col]

    cnt = 0         # counter, stores number of skittles detected on current image

yellow = Color("yellow", 180, 12 , 16 , 50, 6  , 20 , [0, 1, 1])
green  = Color("green" , 120, 23 , 50 , 50, 17 , 55 , [0, 1, 0])
purple = Color("purple", 35 , 150, 160, 15, 80 , 174, [1, 0, 1])
red    = Color("red"   , 160, 167, 173, 50, 160, 175, [0, 0, 1])
colors = [yellow, green, purple, red]


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

    # read image, convert it to hsv space, set hue values lower than hue_offset to hue_value+180
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_res = cv2.resize(img, None, fx=0.5, fy=0.5)
    img_hsv = cv2.cvtColor(img_res, cv2.COLOR_BGR2HSV)
    img_hsv[:, :, 0] += ((img_hsv[:, :, 0] < hue_offset) * 180).astype(np.ubyte)

    for color in colors:
        # create foreground mask based on color thresholds values and morphology operation
        # this mask should select areas that belong to skittle in particular color for sure
        fg_hsv_low = np.array([color.fg_h_l, color.fg_s_l, fg_v_l])
        fg_hsv_high = np.array([color.fg_h_h, 256, 256])
        mask_fg = cv2.inRange(img_hsv, fg_hsv_low, fg_hsv_high)
        mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_CLOSE, morph_kernel, iterations=fg_clo)

        # find connected areas, remove if smaller than area threshold
        num_fg, markers = cv2.connectedComponents(mask_fg)
        for n in range(1, num_fg):
            area = np.sum(markers == n)
            if area < fg_are:
                mask_fg[markers == n] = 0

        # create inverted sure background mask based on color thresholds values and morphology operations
        # this mask should select full areas that belong to skittle in particular color
        # when inverted it selects area that is background for sure
        bg_hsv_low = np.array([color.bg_h_l, color.bg_s_l, 0])
        bg_hsv_high = np.array([color.bg_h_h, 256, 256])
        mask_bg = cv2.inRange(img_hsv, bg_hsv_low, bg_hsv_high)
        mask_bg = cv2.morphologyEx(mask_bg, cv2.MORPH_CLOSE, morph_kernel, iterations=bg_clo)
        mask_bg = cv2.morphologyEx(mask_bg, cv2.MORPH_DILATE, morph_kernel, iterations=bg_dil)

        # find skittle borders using watershed algorithm
        mask_diff = cv2.subtract(mask_bg, mask_fg)
        num_fg, markers = cv2.connectedComponents(mask_fg)
        markers += 1
        markers[mask_diff == 255] = 0
        markers = cv2.watershed(img_res, markers)
        markers -= 1
        # in markers now: -2 - borders, -1 - difference, 0 - background, 1,2,3... - next areas

        # calculate distance transform on each area, then apply threshold mask
        # this fragment of code separates rounded areas that are connected to each other
        markers_th = np.zeros_like(markers, dtype=np.ubyte)
        for n in range(1, num_fg):
            mark = np.zeros_like(markers, dtype=np.ubyte)
            mark[markers == n] = 255
            dist_trans = cv2.distanceTransform(mark, cv2.DIST_L2, 5)
            ret, mark_th = cv2.threshold(dist_trans, wt_dst * dist_trans.max(), 255, 0)
            markers_th += mark_th.astype(np.ubyte)

        # count connected areas on markers_th, note that num result counts also background
        num, markers_unused = cv2.connectedComponents(markers_th)
        color.cnt = num - 1

        # draw borders and selected areas on image
        img_res[markers == -2] = [0, 0, 0]
        img_res[markers_th == 255] = color.bgr_col

    # show results
    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("result", 1200, 800)
    for i in range(0, len(colors)):
        cv2.putText(img_res, colors[i].name + ": " + str(colors[i].cnt), (5, (img_res.shape[0] - 20) - (3-i) * 100),
                    cv2.FONT_HERSHEY_PLAIN, 6, colors[i].bgr_col, 6)
    cv2.imshow("result", img_res)
    cv2.waitKey(0)

    return {'yellow': colors[0].cnt, 'green': colors[1].cnt, 'purple': colors[2].cnt, 'red': colors[3].cnt}


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

    my_check()

if __name__ == '__main__':
    main()
