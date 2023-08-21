import sys
import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def generate_font_image(width, height, x, y, font_file, letter, noise_level, output_directory):
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Set the font and size
    font_size = min(width, height)
    font = ImageFont.truetype(font_file, font_size)

    # Create a blank white image
    image = Image.new('1', (width, height), color='white')  # Use '1' mode for binary image
    draw = ImageDraw.Draw(image)

    # Calculate the position to draw the letter
    letter_width, letter_height = draw.textsize(letter, font=font)
    letter_x = max(0, min(x, width - letter_width))
    letter_y = max(0, min(y, height - letter_height))

    # Draw the letter on the image
    draw.text((letter_x, letter_y), letter, fill='black', font=font)  # Use 'black' color for text

    # Apply noise to the image
    if noise_level > 0:
        image_noise = np.array(image)
        threshold = noise_level / 100.0
        for i in range(width):
            for j in range(height):
                if np.random.rand() <= threshold:
                    image_noise[i][j] = 1 - image_noise[i][j]  # Invert the binary values

        image = Image.fromarray(image_noise)

    # Save the image as PNG
    image_path = os.path.join(output_directory, f"{letter}.png")
    image.save(image_path)

    # Write the configuration line to description.txt
    description_file = os.path.join(output_directory, "description.txt")
    with open(description_file, "a") as f:
        f.write(f"{letter}.png:letter {letter}, noise level {noise_level}%\n")

if __name__ == "__main__":
    # Read command line arguments
    if len(sys.argv) != 9:
        print("Invalid number of arguments.")
        print("Usage: font_generator.py w h x y font_file letter noise_level output_directory")
        sys.exit(1)

    width = int(sys.argv[1])
    height = int(sys.argv[2])
    x = int(sys.argv[3])
    y = int(sys.argv[4])
    font_file = sys.argv[5]
    letter = sys.argv[6]
    noise_level = int(sys.argv[7])
    output_directory = sys.argv[8]

    # Generate the font image
    generate_font_image(width, height, x, y, font_file, letter, noise_level, output_directory)
