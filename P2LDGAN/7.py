import cv2
import numpy as np

gray = cv2.imread(r"D:\VScode\VisionLab\P2LDGAN\output4.1.jpg", 0)

# เพิ่ม contrast และลด brightness เล็กน้อย
alpha = 1.5  # ค่ามาก = contrast มาก
beta = -50   # ค่าลบ = ทำให้มืดลง
darker = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
img = cv2.resize(darker, (700, 700))

# เส้นดำ พื้นหลังขาว
thresh = cv2.adaptiveThreshold(img , 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 15, 10)

cv2.imshow("Original", gray)
cv2.imshow("Black Lines", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

