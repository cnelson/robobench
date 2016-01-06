# pylint: disable=no-member, invalid-name, unused-variable, too-many-locals
"""Various functions to detect graffiti in images"""

import cv2

__all__ = ['lame_edge_contour']

def lame_edge_contour(img):
    """Attempt to detect graffiti in an segmented image of a train car.

    This is a very naive attempt and currently only included as a placeholder
    while more robust techniques are developed.

    Expect lots of false positives and negatives :/

    It has a (maybe) useful tendency to classify logos and car ids as graffiti too

    Args:
        img(numpy.ndarray): The image to be scaled

    Returns:
        list: A list of tuples in the form of ( (x1, y1), (x2, y2) )
              of areas likely to contain graffiti.
              An empty list indicates no graffiti was found.
    """

    sh, sw = img.shape[:2]

    # crop 15% off the edges, to remove boxcar edge lines, etc
    cropimg = img[sh*.15:sh*.85, sw*.15:sw*.85]

    off_x = int(sw*.15)
    off_y = int(sh*.15)

    gray = cv2.cvtColor(cropimg, cv2.COLOR_BGR2GRAY)

    # edge detection
    edges = cv2.Canny(gray, sh*.4, sh*.2)

    # agressive opening
    edges = cv2.erode(
        edges,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)),
        iterations=1
    )
    edges = cv2.dilate(
        edges,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)),
        iterations=8
    )

    # find contours
    im2, contours, hierarchy = cv2.findContours(
        edges.copy(),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    rects = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # ignore small contours
        if (w+h)*1.0/(sw+sh) < 0.10:
            continue

        if off_y+y < sh/2:
            continue

        rects.append(((off_x+x, off_y+y), (w+off_x+x, h+off_y+y)))

    return rects
