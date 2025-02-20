import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw
from tensorflow.keras.models import load_model


model = load_model("mnist_model.keras")

root = tk.Tk()
root.title("Draw a Number")

canvas = tk.Canvas(root, width=280, height=280, bg="white")
canvas.pack()

points = []


def paint(event):
    if points:
        x1, y1 = points[-1]
        x2, y2 = event.x, event.y
        canvas.create_line(
            x1, y1, x2, y2, fill="black", width=15, smooth=True, capstyle="round"
        )
    points.append((event.x, event.y))


canvas.bind("<B1-Motion>", paint)


def clear_canvas():
    canvas.delete("all")
    points.clear()


def recognize_number():
    image = Image.new("RGB", (280, 280), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)

    for point in points:
        x, y = point
        size = 7
        draw.ellipse([x - size, y - size, x + size, y + size], fill=(0, 0, 0))

    img = np.array(image.convert("L"))

    _, img_resized = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

    img_resized = cv2.resize(img_resized, (28, 28))

    img_resized = img_resized / 255.0
    img_resized = np.reshape(img_resized, (1, 28, 28, 1))

    prediction = model.predict(img_resized)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    result_label.config(text=f"Prediction: {predicted_digit} ({confidence:.2f}%)")


clear_button = tk.Button(root, text="Clear", command=clear_canvas)
clear_button.pack()

recognize_button = tk.Button(root, text="Recognize", command=recognize_number)
recognize_button.pack()

result_label = tk.Label(root, text="Prediction: ")
result_label.pack()

root.mainloop()
