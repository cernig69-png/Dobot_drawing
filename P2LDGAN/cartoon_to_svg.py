import cv2
import subprocess
import numpy as np
from draw_cartoon_df import draw_cartoon1

# โหลดและ threshold
image = r"E:\P2LDGAN\Image\IMG_20250805_151021.jpg"
draw_cartoon1(image)
gray = cv2.imread(r"E:\P2LDGAN\draw_cartoon.jpg", 0)


alpha = 1.5  # ค่ามาก = contrast มาก
beta = -50   # ค่าลบ = ทำให้มืดลง
darker = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
img = cv2.resize(darker, (500, 500))

# เส้นดำ พื้นหลังขาว
thresh = cv2.adaptiveThreshold(img , 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 15, 10)
kernel = np.ones((2,2), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

_, bw = cv2.threshold(thresh, 128, 255, cv2.THRESH_BINARY)

# เซฟเป็น BMP
cv2.imwrite("temp.bmp", bw)

# ระบุ path ของ potrace.exe ที่คุณโหลดมา
potrace_path = r"E:\P2LDGAN\potrace-1.16.win64\potrace.exe"

# แปลง BMP → SVG
subprocess.run([potrace_path, "temp.bmp", "-s", "-o", "output_final.svg"])

print("✅ ได้ไฟล์เวกเตอร์: output_final.svg")
