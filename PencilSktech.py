from google.colab.patches import cv2_imshow
from IPython.display import Image, display, Markdown
import cv2
import numpy as np

def image_to_pencil_sketch(image_path, output_path):
    # Read the image in RGB format
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # Invert the grayscale image
    inverted_image = cv2.bitwise_not(grayscale_image)

    # Blur the inverted image
    blurred_image = cv2.GaussianBlur(inverted_image, (111, 111), 0)

    # Invert the blurred image
    inverted_blurred_image = cv2.bitwise_not(blurred_image)

    # Create the pencil sketch by dividing the grayscale image by the inverted blurred image
    pencil_sketch = cv2.divide(grayscale_image, inverted_blurred_image, scale=256.0)

    # Save the pencil sketch image
    cv2.imwrite(output_path, pencil_sketch)

    # Display the images for comparison with Markdown captions
    display(Markdown(f'**Original Image**'))
    display(Image(filename=image_path, width=300, height=300))

    display(Markdown(f'**Pencil Sketch**'))
    display(Image(filename=output_path, width=300, height=300))

# Specify the correct input and output paths
input_image_path = '/content/FLOWER.jpg'  # Corrected path
output_sketch_path = '/content/ConvertedFlower.jpg'

# Call the function
image_to_pencil_sketch(input_image_path, output_sketch_path)
