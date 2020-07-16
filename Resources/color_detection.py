import cv2
import numpy as np
from custom_stuck_img import stackImages


def create_track_bars():
    cv2.namedWindow("TB")
    cv2.resizeWindow("TB", 640, 480)
    cv2.createTrackbar("Hue min", "TB", 0, 179, lambda x: ())
    cv2.createTrackbar("Hue max", "TB", 45, 179, lambda x: ())
    cv2.createTrackbar("Saturation min", "TB", 122, 255, lambda x: ())
    cv2.createTrackbar("Saturation max", "TB", 255, 255, lambda x: ())
    cv2.createTrackbar("Value min", "TB", 42, 255, lambda x: ())
    cv2.createTrackbar("Value max", "TB", 255, 255, lambda x: ())


path = "../Resources/urus.jpg"
create_track_bars()
while True:
    img = cv2.imread(path)
    if img is None:
        print("None")
    img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
    HSV_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos("Hue min", "TB")
    h_max = cv2.getTrackbarPos("Hue max", "TB")
    s_min = cv2.getTrackbarPos("Saturation min", "TB")
    s_max = cv2.getTrackbarPos("Saturation max", "TB")
    v_min = cv2.getTrackbarPos("Value min", "TB")
    v_max = cv2.getTrackbarPos("Value max", "TB")
    print(h_min, h_max, s_max, s_min, v_max, v_min)
    lower_lim = np.array([h_min, s_min, v_min])
    upper_lim = np.array([h_max, s_max, v_max])
    # cv2.imshow("Orig", img)
    # cv2.imshow("HSV", HSV_img)
    mask = cv2.inRange(HSV_img, lower_lim, upper_lim)
    img_res = cv2.bitwise_and(img, img, mask=mask)
    # cv2.imshow("Mask", mask)
    # cv2.imshow("Res", img_res)
    big_img = stackImages(1, ([img, HSV_img], [img_res, mask]))
    cv2.imshow("Big window", big_img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        break
