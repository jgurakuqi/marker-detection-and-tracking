# marker-detection-and-tracking


## Table of Contents

- [Requirements](#Requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)


## Requirements

The goal is to detect the circular marker placed on the rotating turntable, using the given dataset of videos.
Your task is to create a Python program (or notebook) that:
- opens one of the video files cited above
- detects the marker in the turntable, by identifying all the visible marks to create a list of coordinates of each point in the image and in the marker reference system (more details follows)
- save all the debug information into new video files named obj1_marker.mp4, ..., obj4_marker.mp4.
- save point coordinates in CSV files obj1_marker.csv, . . . ,obj2_marker.csv.
See the following Figure for an example of how a debug image might look like.
You are not required to copy exactly the same style, it is just a reference for the kind of information to provide:


## Installation

In order to run this project it's required a Python 3 installation, along with OpenCV and numpy. For the 2 libraries installation the following two *pip* 
commands can be used:
```bash
pip install opencv-python
pip install numpy
```

In order to work, the program requires the related input videos, which must respect the [Requirements](#Requirements) in terms of format.

## Usage

To run the projects it's required an IDE or a bash/shell. In case of a bash/shell, it's sufficient to use the following command:

```bash
python detection_and_tracking.py
```

## Contributing

This project allows to extract the 2D and 3D coordinates of the numbered markers in the video. Its implementation is specific to the problem, requiring a complete re-design for videos with different perspectives and/or different kinds of markers.

```bash
git clone https://github.com/jgurakuqi/marker_detection_and_tracking
```

## License

MIT License

Copyright (c) 2022 Jurgen Gurakuqi

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS," WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
