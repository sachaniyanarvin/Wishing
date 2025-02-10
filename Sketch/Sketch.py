import cv2
import numpy as np
import time
import os

def image_to_sketch(image_path, save_folder='Sketch_Outputs'):
    # Ensure save folder exists
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Unable to load image. Check the file path.")
        return
    
    # Resize image for consistency
    img = cv2.resize(img, (500, 500))
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Invert the grayscale image
    inverted = cv2.bitwise_not(gray)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(inverted, (21, 21), sigmaX=0, sigmaY=0)
    
    # Invert the blurred image
    inverted_blurred = cv2.bitwise_not(blurred)
    
    # Create the pencil sketch
    sketch = cv2.divide(gray, inverted_blurred, scale=256.0)
    
    # Gradually reveal the sketch from top-left to bottom-right
    h, w = sketch.shape
    blank_canvas = np.ones((h, w), dtype=np.uint8) * 255
    
    step = 5  # Smaller step for smoother drawing effect
    for y in range(0, h, step):
        for x in range(0, w, step):
            blank_canvas[y:y+step, x:x+step] = sketch[y:y+step, x:x+step]
            cv2.imshow('Pencil Sketch Drawing', blank_canvas)
            cv2.waitKey(5)  # Small delay to create drawing effect
    
    cv2.imshow('Final Pencil Sketch', sketch)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save the sketch in specified folder
    save_path = os.path.join(save_folder, 'sketch_output.png')
    cv2.imwrite(save_path, sketch)
    print(f"Sketch saved at '{save_path}'")

# Example usage
image_to_sketch(r'D:\Birthday_Gift\Images\Own.jpg')
