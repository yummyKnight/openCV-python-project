import cv2
import numpy as np


def show_webcam():
    cap = cv2.VideoCapture(0)

    while True:
        suc, img = cap.read()
        cv2.imshow("q", edge_detection(img))
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break


def convert_to_gray_scale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def gaus_blur(img):
    return cv2.GaussianBlur(img, (11, 11), 15)


def dilate(img):
    return cv2.dilate(img, kernel=np.ones((2, 2), dtype=np.uint8), iterations=1)


def edge_detection(img):
    return cv2.Canny(img, 150, 200)


def persp_warp():
    width, height = 300, 300
    pt1 = np.float32([(301, 402), (433, 101), (742, 113), (635, 453)])
    pt2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pt1, pt2)
    img = cv2.imread("b2b53df4-8094-11ea-8fdb-7ec06edeef84.jpeg")
    persp_img = cv2.warpPerspective(img, matrix, (width, height))
    cv2.imshow("Orig_warp", img)
    cv2.imshow("Persp_warp", persp_img)
    if cv2.waitKey(0) & 0xFF == ord("q"):
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # show_webcam() (301, 402) , (433, 101), (742, 113), (635, 453)
    persp_warp()
