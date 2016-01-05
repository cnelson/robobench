# ROBOBENCH WIP README

robobench has the following requirements:
* opencv (tested with 3.0, should work with 2.0)
* numpy (tested with 1.8, should work with 1.7)
* Python imaging (tested with 1.1, should work with earlier, pillow, etc)
* lensfun (tested with 1.3)
* click (tested with 6.2)

## Installation

#### TL;DR AND I TRUST SOME RANDOM JANK SHELL SCRIPT TO INSTALL STUFF ON MY MACHINE
If there's a script in the _install_deps that matches your platform, run it.

If there isn't one, and you write one for your platform, please submit a PR!

### Manual install

Install each of the following dependencies 

#### OpenCV

[The Installation Guide for your platform](http://docs.opencv.org/2.4/doc/tutorials/introduction/table_of_content_introduction/table_of_content_introduction.html)

#### Numpy 
[Building and installing NumPy](http://docs.scipy.org/doc/numpy-1.8.1/user/install.html) 

#### PIL
This was tested with old school PIL, but pillow and other dropins should work

[PIL Downloads](http://effbot.org/downloads/#imaging)

#### Lensfun
[Lensfun Install Instructions](https://github.com/neothemachine/lensfunpy)

#### click
Use pip: pip install click

[Click Documentation](http://click.pocoo.org/6/)


## Usage

The segmenter provides full help.  Run with --help 

    usage: car_segmenter.py [-h] [--no-lensfun] [--crop x1 y1 x2 y2]
                            [--output OUTPUT] [--quiet] [-y]
                            image [image ...]

    Identify and extract images of train cars from a given set of images.

    positional arguments:
      image                 The images to process, if you provide a unix-style
                            glob it will be expanded.

    optional arguments:
      -h, --help            show this help message and exit
      --no-lensfun          Don't undistort the images with the Lensfun database.
      --crop x1 y1 x2 y2    Defines the region containing the train. If not
                            provided, you will be prompted to select a region.
      --output OUTPUT, -o OUTPUT
                            A directory to store the segemnted images. It must
                            exist and be writable.
      --quiet, -q           Don't display images as they are processed.
      -y                    Overwrite exist images in OUTPUT directory.

## Examples

Extract cars from the sample2 dataset and store the images in /tmp:

    ./car_segmenter.py _sample_data/sample2/*.jpg --crop 680 952 3378 1592 -o /tmp