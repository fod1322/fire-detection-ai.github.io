import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model

image = None

model = load_model("fire_model.keras")

# Загрузка фото
def load_image():
    global image
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if not file_path:
        return
    try:
        pil_image = Image.open(file_path).convert("RGB")
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        show_image(image)
    except Exception as e:
        messagebox.showerror("Ошибка", f"Не удалось загрузить изображение:\n{e}")

# Проверка на пожар
def check_fire():
    global image

    if image is None:
        messagebox.showwarning("Внимание", "Сначала загрузите изображение")
        return

    img = cv2.resize(image, (128, 128))
    img = np.expand_dims(img, axis=0) / 255.0

    prediction = model.predict(img)
    prob = float(prediction[0][0])

    print("Предсказание модели:", prediction)

    # Вердикт
    if prob > 0.1:
        print("Результат: 🔥 Обнаружен пожар")
    else:
        print("Результат: ✅ Всё безопасно")

# Отображение изображения
def show_image(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_pil.thumbnail((450, 450))
    img_tk = ImageTk.PhotoImage(img_pil)
    panel.config(image=img_tk)
    panel.image = img_tk

root = Tk()
root.title("🔥 Система обнаружения пожара")
root.geometry("550x600")
root.configure(bg="#f2f2f2")

# Заголовок
title = Label(
    root,
    text="Интеллектуальная система\nобнаружения пожара",
    font=("Arial", 16, "bold"),
    bg="#f2f2f2"
)
title.pack(pady=15)

# Блок кнопок
frame = Frame(root, bg="#f2f2f2")
frame.pack(pady=10)

Button(frame, text="📂 Загрузить фото", width=22, command=load_image).grid(row=0, column=0, pady=5)
Button(frame, text="🔍 Проверить", width=22, command=check_fire).grid(row=1, column=0, pady=5)

# Поле изображения
panel = Label(root, bg="#f2f2f2")
panel.pack(pady=15)

root.mainloop()