import cv2
import os

# Specify the directory containing the images
directory = 'moonAlt'

# Loop through each file in the directory
for filename in os.listdir(directory):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # Add other file extensions if needed
        file_path = os.path.join(directory, filename)
        
        # Read the image using OpenCV
        image = cv2.imread(file_path)
        
        # Display the image
        #cv2.imshow('Image', image)
        #cv2.waitKey(0)  # Wait for a key press to proceed to the next image
        #cv2.destroyAllWindows()  # Close the image window
        
        # Print the filename
        print(f'Processed file: {filename}')

print("Finished processing all images.")
