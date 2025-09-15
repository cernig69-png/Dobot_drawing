import cv2
import numpy as np
import serial.tools.list_ports
from pydobot import Dobot
import time

# ================== CONFIG ==================
IMAGE_PATH = '/home/bec/Desktop/EduProject/P2LDGAN/draw_cartoon.jpg'
IMAGE_MAX_SIZE = 1000  # เพิ่มขนาดภาพเพื่อเก็บรายละเอียดมากขึ้น

# พิกัดปากกาบน Dobot
PEN_DOWN_Z = -54
PEN_UP_Z = 20

# พื้นที่วาดบนกระดาษ (ควรวัดและปรับให้ตรงกับ Dobot ของคุณ)
# เปลี่ยนจากจุดมุมเป็นจุดศูนย์กลางเพื่อให้ปรับง่ายขึ้น
DRAWING_AREA_CENTER_X = 167.32
DRAWING_AREA_CENTER_Y = -13
DRAWING_AREA_WIDTH = 100
DRAWING_AREA_HEIGHT = 100
DRAWING_MARGIN = 8.0 # ขอบกระดาษเป็นมิลลิเมตร

# ความเร็วของ Dobot
DOBOT_SPEED = 200
DOBOT_ACCELERATION = 100
MOVE_DELAY = 0
EPSILON = 0.001    # ลดค่าลงเพื่อเก็บรายละเอียดมากขึ้น

# จุดเริ่มต้นของหุ่นยนต์ (จุดที่ปลอดภัยก่อนเริ่มวาด)
START_POINT_X = 151.78
START_POINT_Y = 19.46

# จำนวนครั้งที่ลองใหม่เมื่อเกิดข้อผิดพลาด
RETRY_ATTEMPTS = 3
MIN_CONTOUR_AREA = 3 # กรอง contour ที่มีพื้นที่น้อยกว่า 10 พิกเซลออก
# =============================================

def find_dobot_port():
    """
    ค้นหาพอร์ต USB ของ Dobot โดยอัตโนมัติ
    """
    ports = list(serial.tools.list_ports.comports())
    for port in ports:
        if 'USB' in port.description or 'COM' in port.device \
           or "wchusbserial" in port.device.lower() \
           or "usbmodem" in port.device.lower():
            return port.device
    return None

def img_to_dobot_coords(x_px, y_px, img_w, img_h, keep_aspect=True):
    """
    แปลงพิกัดจากภาพ (พิกเซล) ไปยังพิกัดของ Dobot (มิลลิเมตร)
    """
    drawable_w = DRAWING_AREA_WIDTH - 2 * DRAWING_MARGIN
    drawable_h = DRAWING_AREA_HEIGHT - 2 * DRAWING_MARGIN

    if keep_aspect:
        scale = min(drawable_w / img_w, drawable_h / img_h)
        offset_x = (drawable_w - img_w * scale) / 2.0
        offset_y = (drawable_h - img_h * scale) / 2.0
    else:
        scale_x = drawable_w / img_w
        scale_y = drawable_h / img_h
        scale = 1.0
        offset_x = 0
        offset_y = 0

    dob_x = DRAWING_AREA_CENTER_X - drawable_w / 2.0 + offset_x + x_px * scale
    dob_y = DRAWING_AREA_CENTER_Y - drawable_h / 2.0 + offset_y + (img_h - y_px) * scale
    
    return dob_x, dob_y

def safe_move(bot, x, y, z, r=0, wait=True):
    """
    สั่งให้ Dobot เคลื่อนที่อย่างปลอดภัยพร้อมกับการจัดการข้อผิดพลาด
    """
    x = max(DRAWING_AREA_CENTER_X - DRAWING_AREA_WIDTH/2, min(DRAWING_AREA_CENTER_X + DRAWING_AREA_WIDTH/2, x))
    y = max(DRAWING_AREA_CENTER_Y - DRAWING_AREA_HEIGHT/2, min(DRAWING_AREA_CENTER_Y + DRAWING_AREA_HEIGHT/2, y))

    for i in range(RETRY_ATTEMPTS):
        try:
            bot.move_to(x, y, z, r, wait=wait)
            return True # เคลื่อนที่สำเร็จ
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาดในการสื่อสาร: {e}")
            if i < RETRY_ATTEMPTS - 1:
                print(f"  --> กำลังลองใหม่ครั้งที่ {i+1}...")
                time.sleep(1)
            else:
                print("  --> ลองใหม่ครบจำนวนแล้ว จะข้ามการเคลื่อนที่นี้")
                return False # เคลื่อนที่ล้มเหลว
    return False

def main():
    port = find_dobot_port()
    if not port:
        print("❌ ไม่พบ Dobot! โปรดตรวจสอบการเชื่อมต่อ")
        return

    try:
        print(f"✅ กำลังเชื่อมต่อกับ Dobot ที่ {port}")
        bot = Dobot(port=port, verbose=False)
        bot.speed(DOBOT_SPEED, DOBOT_ACCELERATION)
    except Exception as e:
        print(f"❌ ไม่สามารถเชื่อมต่อ Dobot ได้: {e}")
        return

    # --- โหลดและประมวลผลภาพ ---
    print("⏳ กำลังประมวลผลรูปภาพ...")
    img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("❌ ไม่พบรูปภาพ")
        bot.close()
        return

    h, w = img.shape
    scale_resize = IMAGE_MAX_SIZE / max(h, w)
    img = cv2.resize(img, (int(w * scale_resize), int(h * scale_resize)))

    # เพิ่ม Gaussian blur เพื่อลด noise
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # 🔹 ปรับค่า Adaptive Threshold
    thresh = cv2.adaptiveThreshold(img, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV,11, 7)

    # 🔹 หา contour ทั้งหมด
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    # 🔹 กรอง contour ที่เล็กเกินไปออก
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_CONTOUR_AREA]
    print(f"✅ พบ {len(filtered_contours)} contours ที่จะถูกวาด (จากทั้งหมด {len(contours)})")

    # Preview
    preview = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(preview, filtered_contours, -1, (0, 0, 255), 1)
    cv2.imshow("Processed Contours", preview)
    print("👉 กดปุ่ม 'q' เพื่อเริ่มวาด...")
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

    img_h, img_w = img.shape
    print("✏️ เริ่มวาด...")

    # เคลื่อนปากกาไปยังจุดเริ่มต้น
    if not safe_move(bot, START_POINT_X, START_POINT_Y, PEN_UP_Z):
        print("⚠️ ไม่สามารถไปที่จุดเริ่มต้นได้ จะสิ้นสุดการทำงาน")
        bot.close()
        return
    time.sleep(1)

    for ci, cnt in enumerate(filtered_contours, start=1):
        if len(cnt) < 2:
            continue

        approx = cv2.approxPolyDP(cnt, EPSILON, True)
        if len(approx) < 2:
            continue

        x0, y0 = approx[0][0]
        sx, sy = img_to_dobot_coords(x0, y0, img_w, img_h)

        if not safe_move(bot, sx, sy, PEN_UP_Z):
            continue
        time.sleep(MOVE_DELAY)

        if not safe_move(bot, sx, sy, PEN_DOWN_Z):
            continue
        time.sleep(MOVE_DELAY)

        for point in approx[1:]:
            x, y = point[0]
            dx, dy = img_to_dobot_coords(x, y, img_w, img_h)
            if not safe_move(bot, dx, dy, PEN_DOWN_Z, wait=False):
                break

        safe_move(bot, dx, dy, PEN_UP_Z)
        time.sleep(MOVE_DELAY)

        print(f"✅ วาด contour {ci}/{len(filtered_contours)} เสร็จ")

    print("🎉 วาดเสร็จสมบูรณ์!")
    bot.close()

if __name__ == "__main__":
    main()
