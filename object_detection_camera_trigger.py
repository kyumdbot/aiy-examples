#!/usr/bin/env python3

import argparse

from picamera import PiCamera
from aiy.vision.inference import CameraInference
from aiy.vision.models import object_detection
from aiy.vision.annotator import Annotator
from datetime import datetime


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
        sendtime = datetime.now()
        with CameraInference(object_detection.model()) as inference:
            for result in inference.run(args.num_frames):
                objects = object_detection.get_objects(result)
                #print('#%05d (%5.2f fps): num_objects=%d, objects=%s' %
                #       (inference.count, inference.rate, len(objects), objects))
                if len(objects) > 0:
                    print(f"num_objects={len(objects)}, objects={[objectLabel(obj.kind) for obj in objects]}")
                    difftime = datetime.now() - sendtime
                    if difftime.seconds > 3: 
                        print(difftime)
                        sendtime = datetime.now()
                        camera.capture('/home/pi/Pictures/faces_%d%02d%02d-%02d%02d%02d.jpg' % 
                                (sendtime.year, sendtime.month, sendtime.day, sendtime.hour, sendtime.minute, sendtime.second))

        camera.stop_preview()


if __name__ == '__main__':
    main()

