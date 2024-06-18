import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def preprocess_image(image_path):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to load image")

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization
    #gray = cv2.equalizeHist(gray)
    _, gray = cv2.threshold(gray, 126, 255, cv2.THRESH_BINARY)
    # Apply GaussianBlur to reduce noise and improve edge detection
    gray = cv2.GaussianBlur(gray, (9, 9), 2)
    
    #plt.imshow(gray, cmap='gray')
    #plt.title("gray")
    #plt.show()

    return img, gray

def detect_craters(gray_img, original_img, original_filename,folder):
    # Detect circles using HoughCircles
    detected_circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                                        param1=300, param2=50, minRadius=25, maxRadius=100)
    
    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))
        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]
            # Draw the circumference of the circle
            cv2.circle(original_img, (a, b), r, (0, 255, 0), 2)
            # Draw a small circle to show the center
            cv2.circle(original_img, (a, b), 1, (0, 0, 255), 3)
        
        # Convert BGR image to RGB for displaying with matplotlib
        img_with_craters = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        filename="o_"+original_filename
        filePath = os.path.join(folder, filename)
        print(filePath)
        cv2.imwrite(filePath, img_with_craters)

        #############################for NN training
        blank = np.zeros(shape=(512,512,1), dtype=np.int16)
        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]
            # Draw the circumference of the circle
            cv2.circle(blank, (a, b), r, (255, 255, 255), -1)
        
        # Convert BGR image to RGB for displaying with matplotlib
        #img_train = cv2.cvtColor(blank, cv2.COLOR_BGR2RGB)
        
        filename="t_"+original_filename
        filePath = os.path.join(folder, filename)
        #print(filePath)
        cv2.imwrite(filePath, blank)

        #plt.imshow(img_with_craters)
        #plt.title("Detected Craters")
        #plt.show()
    else:
        blank = np.zeros(shape=(512,512,1), dtype=np.int16)
        filename="t_"+original_filename
        filePath = os.path.join(folder, filename)
        cv2.imwrite(filePath, blank)
        print("No craters detected")

def process_images(folder):
    os.makedirs("moonCraters", exist_ok=True)
    for filename in os.listdir(folder):
        if filename.endswith('.png'):
            filePath = os.path.join(folder, filename)
            original_img, gray_img = preprocess_image(filePath)
            detect_craters(gray_img,original_img,filename,"moonCraters")

    return 1

# Example usage
#input_file = "test.png"  # Replace with your image file path

#original_img, gray_img = preprocess_image(input_file)
os.makedirs("moonCraters", exist_ok=True)
#detect_craters(gray_img, original_img, input_file, "moonAlt")
process_images("moonAlt")


