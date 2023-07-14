import cv2
import numpy as np

cur_image = cv2.imread("../data/knot_first_frame.png")

contours = []
cur_corner = []
height, width, layers = cur_image.shape
display_scale = 1
mask = np.zeros((int(height*display_scale), int(width*display_scale)))
frame = cv2.resize(cur_image, (int(width*display_scale), int(height*display_scale)))
resized_cur_image = frame.copy()
item = 0

def on_mouse(event, x, y, flags, params):
    global cur_corner
    global contours

    cur_corner = [x, y]
    if event == cv2.EVENT_LBUTTONDOWN:
        contours.append([x, y])

while True:
    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', on_mouse) 
    
    # reset temp frame
    temp_frame = frame.copy()
    temp_contours = contours.copy()
    temp_contours.append(cur_corner)
    temp_contours = np.array(temp_contours)
    if len(temp_contours) >= 2:
        cv2.fillPoly(temp_frame, pts=np.int32([temp_contours]), color=(150, 150, 150))

    disp = cv2.addWeighted(temp_frame, 0.5, resized_cur_image, 0.5, 0)
    cv2.imshow('frame', disp)
    key = cv2.waitKey(10)

    if key == 113:  # q
        break
    elif key == 32:  # space
        if len(contours) >= 1:
            contours = np.array(contours)
            cv2.fillPoly(frame, pts=np.int32([contours]), color=(150, 150, 150))
            cv2.fillPoly(mask, pts=np.int32([contours]), color=(255, 255, 255))
            contours = []
    elif key == 115:  # s
        mask_resized = cv2.resize(mask, (width, height))
        cv2.imwrite('first_frame_segmentations/knot/mask' + str(item) + '.png', mask_resized)
        print('Successfully saved mask {}!'.format(str(item)))
        item += 1
        mask = np.zeros((int(height*display_scale), int(width*display_scale)))
        frame = cv2.resize(cur_image, (int(width*display_scale), int(height*display_scale)))

cv2.destroyAllWindows()