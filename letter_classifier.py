import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model('letter_model.keras')

# Set up canvas
root = tk.Tk()
root.title("Draw a Letter (A-Z)")

canvas_width = 200
canvas_height = 200
canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg='white')
canvas.pack()

# PIL image to draw
image = Image.new("L", (canvas_width, canvas_height), color=255)
draw = ImageDraw.Draw(image)

# Drawing functions
def paint(event):
    x1, y1 = (event.x-8), (event.y-8)
    x2, y2 = (event.x+8), (event.y+8)
    canvas.create_oval(x1, y1, x2, y2, fill='black', outline='black')
    draw.ellipse([x1, y1, x2, y2], fill=0)

canvas.bind("<B1-Motion>", paint)

# Predict button
def predict():
    img = image.resize((28,28))
    img = ImageOps.invert(img) # invert colors: black background => white
    img = np.array(img) / 255.0
    img = img.reshape(1, 28, 28, 1)
    pred = model.predict(img)
    letter = chr(np.argmax(pred) + 65) # Convert 0-25 -> A-Z
    result_label.config(text=f"Predicted: {letter}")

# Clear button
def clear():
    canvas.delete('all')
    draw.rectangle([0,0,canvas_width,canvas_height], fill=255)
    result_label.config(text='')

btn_predict = tk.Button(root, text="Predict", command=predict)
btn_predict.pack(side='left', padx=10, pady=10)

btn_clear = tk.Button(root, text='Clear', command=clear)
btn_clear.pack(side='right', padx=10, pady=10)

result_label = tk.Label(root, text='', font=('Helvetica', 16))
result_label.pack(pady=10)

root.mainloop()