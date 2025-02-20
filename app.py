import tkinter as tk
from tkinter import messagebox
import numpy as np
from tensorflow.keras.models import load_model
import cv2

model = load_model("mnist_model.keras")

root = tk.Tk()
root.title("Draw a Number")

canvas = tk.Canvas(root, width=280, height=280, bg="white")
canvas.pack()

points = []


def paint(event):
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    canvas.create_oval(x1, y1, x2, y2, fill="black", width=5)
    points.append((event.x, event.y))


canvas.bind("<B1-Motion>", paint)


def clear_canvas():
    canvas.delete("all")
    points.clear()


def recognize_number():
    canvas.postscript(file="temp_image.ps")

    img = cv2.imread("temp_image.ps", cv2.IMREAD_GRAYSCALE)
    if img is None:
        messagebox.showerror("Error", "Unable to process the image")
        return

    img_resized = cv2.resize(img, (28, 28))  # ปรับขนาด
    img_resized = cv2.threshold(img_resized, 127, 255, cv2.THRESH_BINARY)[
        1
    ]  # แปลงเป็นภาพขาวดำ
    img_resized = img_resized / 255.0  # ทำให้ค่าอยู่ในช่วง 0-1
    img_resized = np.reshape(img_resized, (1, 28, 28, 1))  # เพิ่มมิติสำหรับ input ของโมเดล

    prediction = model.predict(img_resized)
    predicted_digit = np.argmax(prediction)  # ค่า index ที่มีความน่าจะเป็นสูงสุด
    confidence = np.max(prediction) * 100  # ความมั่นใจเป็นเปอร์เซ็นต์

    result_label.config(text=f"Prediction: {predicted_digit} ({confidence:.2f}%)")


clear_button = tk.Button(root, text="Clear", command=clear_canvas)
clear_button.pack()

recognize_button = tk.Button(root, text="Recognize", command=recognize_number)
recognize_button.pack()

result_label = tk.Label(root, text="Prediction: ")
result_label.pack()

root.mainloop()
