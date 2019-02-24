import cv2
import numpy as np
import datetime


class VideoAnalysis:

    def __init__(self):
        pass

    def run(self, path):
        ct = 0
        print("Running video analysis...")
        cap = cv2.VideoCapture(path)

        if not cap.isOpened():
            raise IOError("Error opening video file")

        while cap.isOpened():
            ret, orig_frame = cap.read()

            # Resize frame to 70% of orignal size
            frame = cv2.resize(orig_frame, (0, 0), fx=0.70, fy=0.70)

            if ret:
                ct += 1
                # Generate binary black and white image
                cv2.imwrite('data/frames/' + str(ct) + '.png', frame)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
