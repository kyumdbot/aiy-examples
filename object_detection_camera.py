#!/usr/bin/env python3

import argparse

from picamera import PiCamera
from aiy.vision.inference import CameraInference
from aiy.vision.models import object_detection
from aiy.vision.annotator import Annotator


def objectLabel(kind):
    if kind == 1:
        return "Person"
    elif kind == 2:
        return "Cat"
    elif kind == 3:
        return "Dog"
    else:
        return ""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_frames', '-n', type=int, dest='num_frames', default=None,
        help='Sets the number of frames to run for, otherwise runs forever.')
    args = parser.parse_args()

    with PiCamera(sensor_mode=4, resolution=(1640, 1232), framerate=30) as camera:
        camera.start_preview()

        # Annotator renders in software so use a smaller size and scale results
        # for increased performace.
        annotator = Annotator(camera, dimensions=(320, 240))
        scale_x = 320 / 1640
        scale_y = 240 / 1232

        # Incoming boxes are of the form (x, y, width, height). Scale and
        # transform to the form (x1, y1, x2, y2).
        def transform(bounding_box):
            x, y, width, height = bounding_box
            return (scale_x * x, scale_y * y, scale_x * (x + width),
                    scale_y * (y + height))

        with CameraInference(object_detection.model()) as inference:
            for result in inference.run(args.num_frames):
                objects = object_detection.get_objects(result)
                annotator.clear()
                for obj in objects:
                    rect = transform(obj.bounding_box)
                    annotator.bounding_box(rect, fill=0)
                    loc = (rect[0]+4, rect[1])
                    annotator.text(loc, objectLabel(obj.kind))
                annotator.update()
                #print('#%05d (%5.2f fps): num_objects=%d, objects=%s' %
                #       (inference.count, inference.rate, len(objects), objects))
                if len(objects) > 0:
                    print(f"num_objects={len(objects)}, objects={[objectLabel(obj.kind) for obj in objects]}")

        camera.stop_preview()


if __name__ == '__main__':
    main()
