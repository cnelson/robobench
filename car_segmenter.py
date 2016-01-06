#!/usr/bin/env python
"""Identify and extract images of train cars from a given set of images"""
# pylint: disable=no-member, invalid-name, redefined-outer-name, global-statement, protected-access
# pylint: disable=unused-argument

# stdlib
import argparse, glob, math, os, time
from fractions import Fraction

# needed for extract EXIF data
from PIL import Image
from PIL.ExifTags import TAGS

# calculate lens distortion
import lensfunpy

# cli ui candy
import click

# cv
import cv2

import detectors

WINDOW_NAME = 'Robobench'

def _open(filename, undist_coords=None):
    """Use OpenCV2 to parse an image, optionally undistorting the image

    Args:
        filename (str): The filename to open
        undist_coords(numpy.ndarray, optional): A list of x,y points to be passed to cv2.remap.
                                                See ```_lensfun```

    Returns:
        numpy.ndarray: The image as returned by cv2.imread()/remap()
        None: The image could not be loaded
    """

    fullimg = cv2.imread(filename)

    if fullimg is not None and undist_coords is not None:
        fullimg = cv2.remap(fullimg, undist_coords, None, cv2.INTER_LANCZOS4)

    return fullimg

def _scale(img, width=None, height=None):
    """Scale an image to a given width, height, or both.

    If only width or height is specified, then the other dimension will be calculated to preserve

    the aspect ratio

    If both are specified the image will be scaled to those dimensions possibly altering

    the aspect ratio

    Args:
        img(numpy.ndarray): The image to be scaled
        width(int, optional): The width of the image to be returned.
        height(int, optional): The height of the image to be returned.

    Returns:
        Tuple(numpy.ndarray, float): The scaled img, and the scale it was sized to

                                     (1.0=original size)

    """

    if width is None and height is None:
        raise ValueError('You must provide a width, height, or both')

    w = h = None
    scale = 1.0
    if width is not None and height is not None:
        w = int(width)
        h = int(height)
    else:
        if width is not None:
            scale = float((width*1.0)/img.shape[1])
            w = int(width)
            h = int(img.shape[0]*scale)
        else:
            scale = float((height*1.0)/img.shape[0])
            w = int(img.shape[1]*scale)
            h = int(height)

    return cv2.resize(img, (w, h)), scale

def _exif(filename):
    """Extract EXIF data from a file

    Args:
        filename(str): The filename to open and extract EXIF data from.

    Returns:
        dict: key = exif tag, value = exif value
        None: The filename could not be opened
    """
    ret = {}
    try:
        i = Image.open(filename)
    except IOError:
        return None

    info = i._getexif()

    for tag, value in info.items():
        decoded = TAGS.get(tag, tag)
        ret[decoded] = value

    return ret

def _lensfun(filename):
    """Use the EXIF data an image calculate lens distortion corrections using the lensfun database

    Args:
        filename(str): The filename to open and extract EXIF data from.

    Returns:
        numpy.ndarray: A map suitable for use with cv2.remap()

    Raises:
        ValueError: The distortion corrections could not be calculated.
    """

    exif = _exif(filename)

    if exif is None:
        raise ValueError('Unable to load EXIF data from {0}'.format(filename))

    # open the lensfun db
    db = lensfunpy.Database()

    # see if our make/model is in the DB
    try:
        cam = db.find_cameras(exif['Make'], exif['Model'])[0]
        lens = db.find_lenses(cam)[0]
    except IndexError:
        raise ValueError(
            'Unable to find a Lensfun entry for {0} {1}'.format(exif['Make'], exif['Model'])
        )

    # calculate the distortion corrections
    try:
        focal_length = float(exif['FocalLengthIn35mmFilm'])
        aperture = float(Fraction(*exif['FNumber']))
        distance = int(exif['SubjectDistanceRange'])
    except KeyError as exc:
        raise ValueError(
            'Unable to calculate lens distortion: EXIF data is missing item {0}'.format(exc)
        )

    mod = lensfunpy.Modifier(
        lens,

        cam.crop_factor,

        exif['ExifImageWidth'],

        exif['ExifImageHeight']
    )

    mod.initialize(focal_length, aperture, distance)

    return mod.apply_geometry_distortion()

crop = None
_crop_pt1 = None
_crop_time = None
def _crop_on_mouse(event, x, y, flags, img):
    """Interactive cropping of a larger image

    This is called by opencv when any mouse events occur during interactive cropping

    Args:
        event, x, y, flags: See docs for cv2.setMouseCallback for more info
        img: The image being cropped

    Returns:
        None: This communicages with get_crop through the crop/_crop_pt1 global variables (ick!)
    """
    global _crop_time, _crop_pt1, crop

    # if it's a right click, reset everything
    if event == cv2.EVENT_RBUTTONDOWN:
        _crop_pt1 = None
        crop = None

    # If it's mouse down
    if event == cv2.EVENT_LBUTTONDOWN:
        # if our first point isn't set, this this is the first click
        # stash the top/left point, and reset any previous crops
        if _crop_pt1 is None:
            _crop_pt1 = (x, y)
            crop = None
            _crop_time = time.time()
        else:
            # this is our second click, complete the rectangle
            crop = (_crop_pt1, (x, y))
            _crop_pt1 = None

    # On mouseup, if it's been a significant time since the first down, we assume that we are
    # click/drag/release , rather than click/click to draw the rectangle
    if event == cv2.EVENT_LBUTTONUP and  _crop_pt1:
        if time.time() - _crop_time > 0.2:
            crop = (_crop_pt1, (x, y))
            _crop_pt1 = None

    # copy the image so we don't draw on it over and over
    z = img.copy()
    rect = None

    # see if we have a rectable to draw
    if crop:
        rect = crop
    elif _crop_pt1:
        # if we've got the top left pointer, then draw whever the mouse is as the bottom right
        rect = (_crop_pt1, (x, y))

    # if we have something to draw, show it
    if rect:
        cv2.rectangle(z, rect[0], rect[1], (0, 0, 255), 1)

    cv2.imshow(WINDOW_NAME, z)

def get_crop(images, undist_coords=None):
    """Launch an interactive cv2 window to let the user draw the bounding box over the full image

    This box will be the area where train cars are expected to be when attempting segmentation

    Args:
        images(list): A list of images, the user can cycle through them to ensure their cropped
                      area is corrections
        undist_coords(numpy.ndarray, optional): A list of x,y points to be passed to cv2.remap.

                                                See ```_lensfun```

    Returns:
        Tuple ((x1, y1), (x2, y2)): The selected rectanagle

    """

    # create our window
    cv2.namedWindow(WINDOW_NAME)

    # state flags
    done = False
    paused = True
    while not done:
        for filename in images:

            # load the full size image and scale to a reasonable size
            fullimg = _open(filename, undist_coords)
            img, scale = _scale(fullimg, width=1280)

            # load our image
            z = img.copy()

            # if we have a crop, but it hasn't been saved yet, draw it
            if crop:
                cv2.rectangle(z, crop[0], crop[1], (0, 0, 255), 1)

            cv2.imshow(WINDOW_NAME, z)

            # if we are paused, allow drawing the crop box
            if paused:
                cv2.setMouseCallback(WINDOW_NAME, _crop_on_mouse, img)
                keypress = cv2.waitKey(0) & 0xFF
            else:
                cv2.setMouseCallback(WINDOW_NAME, lambda *args: None)
                keypress = cv2.waitKey(50) & 0xFF

            # bounce on ESC
            if keypress == 27:
                raise SystemExit

            # space = pause / unpause
            if keypress == 32:
                paused = not paused

            # enter = return crop if set
            if keypress == 10:
                if crop is not None:
                    done = True
                    break
    # kill our window
    cv2.destroyWindow(WINDOW_NAME)
    pt1, pt2 = sorted(crop)
    scale = fullimg.shape[1]/(img.shape[1]+1.0)

    # return the fullscale coordinates to crop
    return (
        (int(pt1[0]*scale), int(pt1[1]*scale)),
        (int(pt2[0]*scale), int(pt2[1]*scale)),
    )

if __name__ == "__main__":
    # TODO: debug opencv3 weirdness with locale
    os.environ['LANG'] = 'C'

    parser = argparse.ArgumentParser(
        description='Identify and extract images of train cars from a given set of images.'
    )
    parser.add_argument(
        'image',
        nargs='+',
        help='The images to process, if you provide a unix-style glob it will be expanded.'
    )
    parser.add_argument(
        '--no-lensfun',
        default=False,
        action='store_true',
        help="Don't undistort the images with the Lensfun database."
    )
    parser.add_argument(
        '--crop',
        nargs=4,
        type=int,
        metavar=('x1', 'y1', 'x2', 'y2'),
        help="Defines the region containing the train. If not provided, you will be prompted to" \
        " select a region."
    )
    parser.add_argument(
        '--output',
        '-o',
        help='A directory to store the segemnted images. It must exist and be writable.'
    )
    parser.add_argument(
        '--detect',
        '-d',
        nargs='?',
        const=detectors.__all__[0],
        choices=detectors.__all__,
        help='Only display/output images that contain graffiti according to the selected detector'
    )

    parser.add_argument(
        '--quiet',
        '-q',
        default=False,
        action='store_true',
        help="Don't display images as they are processed."
    )
    parser.add_argument(
        '-y',
        default=False,
        action='store_true',
        help='Overwrite exist images in OUTPUT directory.'
    )

    args = parser.parse_args()

    if args.detect:
        args.detect = getattr(detectors, args.detect)

    if args.quiet and args.output is None:
        parser.error("You should specify output with --quiet/-q, otherwise, what's the point?")

    if args.output is not None and not os.path.exists(args.output):
        parser.error('The output directory {0} does not exist'.format(args.output))

    images = []

    # unpack any globs the shell might not have done for us
    for i in args.image:
        images += glob.glob(i)

    # dedupe
    images = sorted(list(set(images)))

    if len(images) == 0:
        parser.error('No images found')

    # see if we need to undistort images
    undist_coords = None
    if args.no_lensfun is False:
        print "Calculating Lens distortion based on {0}...".format(os.path.basename(images[0]))
        try:
            undist_coords = _lensfun(images[0])
            print
        except ValueError as exc:
            parser.error(exc)

    # crop out excess vertical images
    if args.crop is None:
        print "Crop is not set.  Please draw a rectangle around the area containing train cars."
        print
        print "For best results, The rect should be large enough to contain the largest car on the "
        print "train, but the rect should not be so large that multiple cars fit fully inside it"
        print
        print "Controls:"
        print "---------"
        print "\tLeft click/drag to select the crop area."
        print "\tRight click to reset crop area."
        print
        print "\tPress SPACE to play/pause images;  You must pause to select an area."
        print "\tPress ENTER when completed"
        print "\tPress ESC to quit."
        print

        args.crop = get_crop(images, undist_coords)

        print "Crop set to {0},{1} {2},{3}".format(
            args.crop[0][0],
            args.crop[0][1],
            args.crop[1][0],
            args.crop[1][1]
        )
        print "You can use'--crop {0} {1} {2} {3}' to skip this step next time.".format(
            args.crop[0][0],
            args.crop[0][1],
            args.crop[1][0],
            args.crop[1][1]
        )
    else:
        args.crop = ((args.crop[0], args.crop[1]), (args.crop[2], args.crop[3]))

    # loop through each image, crop it, and look for train cars
    with click.progressbar(images, label="Segmenting...") as bar:
        for filename in bar:
            bar.label = os.path.basename(filename)

            fullimg = _open(filename, undist_coords)
            if fullimg is None:
                print "Unable to open {0}".format(os.path.basename(filename))
                continue

            # crop it
            pt1 = args.crop[0]
            pt2 = args.crop[1]
            cropimg = fullimg[pt1[1]:pt2[1], pt1[0]:pt2[0]]
            ch, cw = cropimg.shape[:2]

            # scale it
            img, scale = _scale(cropimg, width=640)
            sh, sw = img.shape[:2]

            gray = grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # find edges
            edges = cv2.Canny(gray, sh*1.5, sh*2)
            edges = cv2.dilate(
                edges,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=2
            )
            edges = cv2.erode(
                edges,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)),
                iterations=1
            )

            # find lines
            lines = cv2.HoughLinesP(edges, 1, math.pi/4, 10, None, sh*.2, sh*0.05)

            # find vertical lines
            vert = []
            if lines is not None:
                for ll in lines:
                    for line in ll:
                        if abs(line[0] - line[2]) < sh*.05:
                            vert.append(line[0])
            vert = sorted(list(set(vert)))

            # if we have more than two, look for a traincar sized sspace
            if len(vert) < 2:
                continue

            # calculate the distance between lines
            results = []
            last = vert[0]
            for v in vert:
                results.append((last, v, v-last))
                last = v

            # biggest gaps first
            results = sorted(results, key=lambda x: x[2], reverse=True)
            train_car = None
            has_graffiti = False

            # is it train car sized?
            # TODO: Learn this size automatically, or have the user specify during interactive crop
            if results and results[0][2] > sw*.45:

                # if our biggest space is big enough, then this is our train car
                x1 = int(results[0][0] * (cw/(sw+1.0)))
                x2 = int(results[0][1] * (cw/(sw+1.0)))

                train_car = cropimg[0:ch, x1:x2]

                # if they asked us, check for graffiti
                if args.detect:
                    rects = args.detect(train_car)
                    has_graffiti = bool(rects)

                # did they ask us to save image, and possibly only ones with graffiti?
                save_image = args.output
                if save_image:
                    save_image = has_graffiti = bool(rects)

                if save_image:
                    target = os.path.join(args.output, os.path.basename(filename))

                    if os.path.exists(target) and not args.y:
                        print
                        parser.error(
                            '{0} already exists.  Use -y to overwrite existing files'.format(target)
                        )

                    cv2.imwrite(target, train_car)

                # if we are detecting, and have graffiti, then highlight that!
                if args.detect:
                    for rect in rects:
                        cv2.rectangle(train_car, rect[0], rect[1], (0, 255, 0), 2)
                else:
                    # if we aren't detecting graffiti, highlight the segmented car
                    cv2.rectangle(cropimg, (x1, 20), (x2, ch-10), (0, 0, 255), 10)


            # skip the rest if they told us to shutup
            if args.quiet:
                continue

            out = None
            if args.detect:
                if has_graffiti:
                    out = _scale(train_car, width=1280)[0]
            else:
                out = _scale(fullimg, width=1280)[0]

            if out is not None:
                cv2.imshow(WINDOW_NAME, out)
                cv2.waitKey(10)
