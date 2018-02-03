from pytesser3 import *

written = Image.open("media/temp.jpg")
text = image_to_string(written)
print(text)
