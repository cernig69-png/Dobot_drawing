import cv2
import mediapipe as mp
import torch
from torchvision import transforms
from PIL import Image
from model.models import Generator  # ใช้ class Generator จริงจาก models.py

img_path = r"D:\VScode\VisionLab\P2LDGAN\Image\IMG_20250805_151019.jpg"  # ใส่ path รูปของคุณ
img = cv2.imread(img_path)
if img is None:
    print("❌ ไม่พบไฟล์รูป ตรวจสอบ path อีกครั้ง")
    exit()

img = cv2.resize(img, (700, 700))
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

model_path = r"D:\VScode\VisionLab\P2LDGAN\p2ldgan_generator_200.pth"  # ใส่ path checkpoint ของคุณ

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)

checkpoint = torch.load(model_path, map_location=device)
generator.load_state_dict(checkpoint)  # โหลดตรง ๆ เพราะ checkpoint เป็น state_dict ของ generator โดยตรง
generator.eval()
print("✅ Loaded checkpoint successfully")

# ------------------------------
# 3️⃣ เตรียม input
# ------------------------------
input_img = Image.open(img_path).convert("RGB")
transform = transforms.Compose([
    transforms.Resize((256, 256)),   # ขนาดต้องตรงกับโมเดล
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
input_tensor = transform(input_img).unsqueeze(0).to(device)

# ------------------------------
# 4️⃣ Generate output
# ------------------------------
with torch.no_grad():
    output_tensor = generator(input_tensor)
    output_tensor = (output_tensor * 0.5 + 0.5).clamp(0,1)  # denormalize

output_img = transforms.ToPILImage()(output_tensor.squeeze().cpu())
output_img.save("output4.1.jpg")
print("✅ Generated image saved as output.jpg")


