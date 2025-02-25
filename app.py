import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageTk

# ฟังก์ชันในการโหลดโมเดล
def load_fruit_model():
    model = load_model('fruit_model.keras')
    return model

# ฟังก์ชันในการประมวลผลภาพและทำนายผล
def predict_image(model, img_path):
    # โหลดและปรับขนาดรูปภาพให้เหมาะสมกับโมเดล (150x150)
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # เพิ่มมิติให้เป็น (1, 150, 150, 3)
    img_array /= 255.0  # Normalize ข้อมูลให้อยู่ในช่วง [0, 1]

    # ทำนายผล
    predictions = model.predict(img_array)
    classes = ['apple', 'banana', 'strawberry']
    predicted_class = classes[np.argmax(predictions)]  # เลือกคลาสที่มีความน่าจะเป็นสูงสุด

    # คำนวณความน่าจะเป็นของแต่ละคลาส
    probabilities = predictions[0]
    result = {classes[i]: f"{probabilities[i]*100:.2f}%" for i in range(len(classes))}

    return predicted_class, result

# ฟังก์ชันที่ใช้ในการเลือกไฟล์
def browse_file():
    filename = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if filename:
        # แสดงภาพที่เลือกใน GUI
        img = Image.open(filename)
        img = img.resize((300, 300))  # ปรับขนาดให้พอดีกับกรอบแสดง
        img = ImageTk.PhotoImage(img)
        panel.config(image=img)
        panel.image = img

        # ทำนายผล
        predicted_class, result = predict_image(model, filename)
        
        # แสดงผลลัพธ์
        result_label.config(text=f"Predicted: {predicted_class}")
        
        # แสดงเปอร์เซ็นต์ความน่าจะเป็นของแต่ละคลาส
        result_text = "\n".join([f"{fruit}: {prob}" for fruit, prob in result.items()])
        probability_label.config(text=result_text)

# สร้าง GUI
root = tk.Tk()
root.title("Fruit Image Classifier")
root.geometry("500x600")  # ขนาดหน้าต่างที่เหมาะสม

# โหลดโมเดล
model = load_fruit_model()

# สร้างปุ่มและการแสดงผล
browse_button = tk.Button(root, text="Select Image", command=browse_file, font=("Helvetica", 14), bg="#4CAF50", fg="white", relief="raised", bd=3)
browse_button.pack(pady=20)

# กรอบสำหรับแสดงภาพ
panel = tk.Label(root)
panel.pack(pady=10)

# ป้ายแสดงผลลัพธ์การทำนาย
result_label = tk.Label(root, text="Predicted: ", font=("Helvetica", 16))
result_label.pack(pady=10)

# ป้ายแสดงเปอร์เซ็นต์ความน่าจะเป็น
probability_label = tk.Label(root, text="", font=("Helvetica", 14), justify="left")
probability_label.pack(pady=10)

# เริ่มต้น GUI
root.mainloop()
