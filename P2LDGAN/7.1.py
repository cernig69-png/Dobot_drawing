import cv2
import subprocess
import numpy as np


# โหลดและ threshold
gray = cv2.imread(r"D:\VScode\VisionLab\P2LDGAN\4.1.py", 0)

alpha = 1.5  # ค่ามาก = contrast มาก
beta = -50   # ค่าลบ = ทำให้มืดลง
darker = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
img = cv2.resize(darker, (800, 800))

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
potrace_path = r".\potrace-1.16.win64\potrace.exe"

# แปลง BMP → SVG
subprocess.run([potrace_path, "temp.bmp", "-s", "-o", "output.svg"])

print("✅ ได้ไฟล์เวกเตอร์: output.svg")
