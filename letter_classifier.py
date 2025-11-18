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
    r = 3
    x1, y1 = (event.x-r), (event.y-r)
    x2, y2 = (event.x+r), (event.y+r)
    canvas.create_oval(x1, y1, x2, y2, fill='black', outline='black')
    draw.ellipse([x1, y1, x2, y2], fill=0)

canvas.bind("<B1-Motion>", paint)

def preprocess_drawing(img, canvas_size=200, target_size=28):
    img = img.convert('L')
    img = ImageOps.invert(img)

    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)

    # Resize while keeping aspect ratio
    img = img.resize((target_size, target_size))

    # Normalize 
    img = np.array(img) / 255.0

    # Reshape for CNN
    img = img.reshape(1, target_size, target_size, 1)
    
    return img

# Predict button
def predict():
    img_array = preprocess_drawing(image)
    pred = model.predict(img_array, verbose=0)
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