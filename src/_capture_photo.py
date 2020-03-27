import cv2
import os

IMAGE_SHOW = False


class TakePhoto:
    def __init__(self):
        self.c270 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if self.c270.isOpened() is False:
            raise Exception("IO Error")
        self.c270.set(cv2.CAP_PROP_FPS, 30)
        self.c270.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.c270.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def capture(self, angle):
        while True:
            ret, img = self.c270.read()
            if ret is not None:
                break
        if img is None:
            exit()

        if IMAGE_SHOW:
            disp = cv2.resize(img, None, fx=1.0, fy=1.0)
            cv2.imshow("captured picture.", disp)
            cv2.waitKey(1000)  # milliseconds

        # datadir = './pics'
        #         # if not os.path.isdir(datadir):
        #         #     os.makedirs(datadir)
        cv2.imwrite(str(angle) + '.png', img)
        print("saved " + str(angle) + '.png')
        cv2.waitKey(100)  # milliseconds

    def close(self):
        self.c270.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # angle patterns of the rotating stage
    photo = TakePhoto()
    angle_pattern = range(0, 11, 10)  # [0,10,...,350]
    for ang in angle_pattern:
        photo.capture(ang)
        # count = 0
        # if photo.capture(ang) == -1:
        #     photo.capture()
    photo.close()
