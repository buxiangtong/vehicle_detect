import cv2
import PIL.Image
# im=cv2.imread('icon/swap.png',flags=1)
im = PIL.Image.open('icon/world-book-day.png')
im=cv2.resize(im,[32,32])
cv2.imshow("test",im)
cv2.waitKey(0)
# cv2.imwrite('icon/swap2.png',im)