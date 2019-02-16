import cv2
import numpy as np
import datetime


class VideoAnalysis:
    stopped = False

    def __init__(self):
        pass

    def run(self, path):
        print("Running video analysis...")
        cap = cv2.VideoCapture(path)

        if not cap.isOpened():
            raise IOError("Error opening video file")

        while cap.isOpened():
            ret, orig_frame = cap.read()

            # Resize frame to 70% of orignal size
            frame = cv2.resize(orig_frame, (0, 0), fx=0.70, fy=0.70)

            height, width, channels = frame.shape
            # Crop frame to region of interest
            frame = frame[0:int(height/2), int(width/2):width]

            if ret:
                # Generate binary black and white image
                _, bw = cv2.threshold(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 60, 256, cv2.THRESH_BINARY)
                cv2.imshow('VideoAnalysis', orig_frame)

                if not self.stopped and np.sum(bw == 255) > 33000:
                    self.stop_signal()
                elif self.stopped and np.sum(bw == 255) <= 33000:
                    self.start_signal()

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

    def stop_signal(self):
        self.stopped = True
        print(datetime.datetime.now(), ": Stop Machine!")

    def start_signal(self):
        self.stopped = False
        print(datetime.datetime.now(), ": Start Machine!")
