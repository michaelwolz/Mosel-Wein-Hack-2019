import cv2
import numpy as np

BLUE = (255, 0, 0)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
FONTSCALE = 1
LINETYPE = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX


class VideoAnalysis:
    stopped = False
    stop_counter = 0
    start_counter = 0
    bottomLeftCornerOfText = (0, 0)
    frame_counter = 0
    bw_array = [0, 0, 0, 0]

    def __init__(self):
        pass

    def run(self, path):
        print("Running video analysis...")
        cap = cv2.VideoCapture(path)

        if not cap.isOpened():
            raise IOError("Error opening video file")

        while cap.isOpened():
            ret, orig_frame = cap.read()
            if ret:
                # Resize frame to 70% of original size
                frame = cv2.resize(orig_frame, (0, 0), fx=0.70, fy=0.70)

                # Draw region of interest
                height, width, channels = orig_frame.shape
                cv2.rectangle(orig_frame, (int(width / 2), 0), (width, int(height / 3 * 2)), (255, 0, 0), 2)

                # Crop frame to region of interest
                height, width, channels = frame.shape
                frame = frame[0:int(height / 3 * 2), int(width / 2):width]

                # Set bottomLeftCorner
                self.bottomLeftCornerOfText = (int(width / 2) + 320, int(height / 3 * 2) + 100)

                # Generate binary black and white image from green channel
                _, bw = cv2.threshold(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 60, 256, cv2.THRESH_BINARY)

                # Simple analysis by counting white pixels
                bw_ratio = np.sum(bw == 0) / np.sum(bw == 255)

                if not self.stopped and bw_ratio < 5:
                    self.stop_counter += 1
                    if self.stop_counter > 15:
                        self.stop_signal()
                elif self.stopped and bw_ratio >= 5:
                    self.start_counter += 1
                    if self.start_counter > 15:
                        self.start_signal()

                if self.stopped:
                    self.write_to_image(orig_frame, "Machine stopped " + str(np.round(bw_ratio)), RED)
                else:
                    self.write_to_image(orig_frame, "Machine running " + str(np.round(bw_ratio)), GREEN)

                # Show video
                cv2.imshow('VideoAnalysis', orig_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

    def run_version_2(self, path):
        print("Running video analysis...")
        frame_counter = 0
        bw_array = [0, 0, 0, 0]
        cap = cv2.VideoCapture(path)

        if not cap.isOpened():
            raise IOError("Error opening video file")

        while cap.isOpened():
            ret, orig_frame = cap.read()
            if ret:
                # Resize frame to 70% of original size
                frame = cv2.resize(orig_frame, (0, 0), fx=0.70, fy=0.70)

                # Draw region of interest
                height, width, channels = orig_frame.shape
                cv2.rectangle(orig_frame, (int(width / 2), int(height / 3)),
                              (int(width / 2 * 1.3), int(height / 3 * 1.5)), (255, 0, 0), 2)

                # Crop frame to region of interest
                height, width, channels = frame.shape
                frame = frame[int(height / 3):int(height / 3 * 1.5), int(width / 2):int(width / 2 * 1.3)]

                # Extract BGR channels
                b, g, r = cv2.split(frame)

                # Set bottomLeftCorner
                self.bottomLeftCornerOfText = (int(width * 1.3 / 2 - 200), int(height * 1.3 - 20))

                # Generate binary black and white image
                _, bw = cv2.threshold(b, 60, 256, cv2.THRESH_BINARY)

                # Simple analysis by counting white pixels
                bw_ratio = np.sum(bw == 0) / np.sum(bw == 255)

                if bw_ratio < 30:
                    bw_array[frame_counter % 4] = 1
                else:
                    bw_array[frame_counter % 4] = 0

                frame_check = np.count_nonzero(bw_array)

                if frame_check == 4:
                    if not self.stopped:
                        self.stop_signal()
                else:
                    if self.stopped:
                        self.start_signal()

                if self.stopped:
                    self.write_to_image(orig_frame, "Machine stopped " + str(np.round(bw_ratio)), RED)
                else:
                    self.write_to_image(orig_frame, "Machine running " + str(np.round(bw_ratio)), GREEN)

                frame_counter += 1

                # Show video
                cv2.imshow('VideoAnalysis', orig_frame)
                if cv2.waitKey(50) & 0xFF == ord('q'):
                    break
            else:
                break

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

    def stop_signal(self):
        self.stopped = True
        self.start_counter = 0

    def start_signal(self):
        self.stopped = False
        self.stop_counter = 0

    def write_to_image(self, image, text, color):
        cv2.putText(image, text,
                    self.bottomLeftCornerOfText,
                    FONT,
                    FONTSCALE,
                    color,
                    LINETYPE)
