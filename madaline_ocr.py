import os
import sys
import numpy as np
from PIL import Image


def load_description(file_path):
    description = []
    with open(file_path, 'r') as file:
        for line in file:
            description.append(line)
    return description


def load_image(file_path):
    image = Image.open(file_path).convert('1')  # Convert to binary format
    image_array = np.array(image)
    binary_array = image_array.astype(int)  # Convert boolean array to integer array
    return binary_array

def preprocess_image(image):
    preprocessed_image = np.reshape(image, (1, -1))
    
    # Count the number of 1's in the row
    num_ones = np.count_nonzero(preprocessed_image == 1)
    
    # Convert image to floating-point data type
    preprocessed_image = preprocessed_image.astype(float)
    
    # Divide all values in the row by the square root of the number of 1's
    preprocessed_image /= np.sqrt(num_ones)
    
   
    return preprocessed_image

def calculate_confidence(input_vector, weights):
    
    scores = np.dot(input_vector, weights)
    confidence = scores / np.linalg.norm(weights)
    return confidence

def madaline_ocr(train_directory, test_directory):
    train_description = load_description(os.path.join(train_directory, 'description.txt'))
    test_description = load_description(os.path.join(test_directory, 'description.txt'))

    # Load and preprocess training images
    train_images = []
    train_labels = []

    for image_file in train_description:
        
        image_filename = image_file.split(':')[0]  # Extract the filename before the colon
        label = image_file.split(':')[1]
        image_path = os.path.join(train_directory, image_filename.strip() )  
        image = load_image(image_path)
        preprocessed_image = preprocess_image(image)
        train_images.append(preprocessed_image.flatten())
        train_labels.append(label)

    # Load and preprocess test images
    test_images = []
    test_labels = []
    for image_file in test_description:
        image_filename = image_file.split(':')[0]  # Extract the filename before the colon
        label= image_file.split(':')[1]  # Extract the filename before the colon
        image_path = os.path.join(test_directory, image_filename )  
        image = load_image(image_path)
     
        preprocessed_image = preprocess_image(image)
        test_images.append(preprocessed_image.flatten())
        test_labels.append(label)

    # Perform character recognition on test images
    for i in range(len(test_images)):
        test_image = test_images[i]
        test_label = test_labels[i]

        # Calculate confidence scores for each letter
        confidences = []
        for j in range(len(train_images)):
            train_image = train_images[j]
            
            confidence = calculate_confidence(test_image, train_image)
            confidences.append(confidence)

        # Determine the recognized letter and its confidence
        max_index = np.argmax(confidences)
        recognized_label = train_labels[max_index]
        confidence = confidences[max_index]
        recognized_label = recognized_label.rstrip("\n")
        test_label = test_label.rstrip("\n")
        # Display the recognition result
        result = f"{test_label} --> {recognized_label}, confidence: {confidence:.3f}"
        print(result)



if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: madaline_ocr.py train_directory test_directory")
    else:
        
        train_directory = sys.argv[1]
        test_directory = sys.argv[2]
      
        madaline_ocr(train_directory, test_directory)