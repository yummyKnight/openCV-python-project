import cv2
import numpy as np
import timeit

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
    img = cv2.imread("Resources/b2b53df4-8094-11ea-8fdb-7ec06edeef84.jpeg")
    cv2.namedWindow("Orig_warp")
    cv2.imshow("Orig_warp", img)
    cv2.setMouseCallback("Orig_warp", show_coord)
    width, height = 300, 300
    pt1 = np.float32([[431, 101], [743, 111], [299, 425], [636, 455]])
    pt2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pt1, pt2)
    persp_img = cv2.warpPerspective(img, matrix, (width, height))
    cv2.namedWindow("Persp_warp")
    cv2.imshow("Persp_warp", persp_img)
    if cv2.waitKey(0) & 0xFF == ord("q"):
        cv2.destroyAllWindows()


def conctat_img():
    img = cv2.imread("Resources/cards.jpg")
    if img is None:
        return
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    h_stacked = np.hstack((img, gray_img))
    v_stacked = np.vstack((img, img))
    cv2.imshow("Stacked hor", h_stacked)
    cv2.imshow("Stacked ver", v_stacked)
    if cv2.waitKey(0) & 0xFF == ord("q"):
        cv2.destroyAllWindows()


def show_coord(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)


if __name__ == '__main__':
    # show_webcam() (301, 402) , (433, 101), (742, 113), (635, 453)
    conctat_img()
