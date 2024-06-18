from PIL import Image
import os
Image.MAX_IMAGE_PIXELS = 933120000

# Load the large jpg image
image_path = 'moon_slope.jpg'
image = Image.open(image_path)

# Calculate the coordinates of the center area
width, height = image.size
print(image.size)
top = height // 4
bottom = 3 * height // 4
fragment_size = 512
fragment_degrees_lon = 360.0 * fragment_size / width
fragment_degrees_lat = 180.0 * fragment_size / height
top_lat = 90 - 180 * top / height 
bottom_lat = 90 - 180 * bottom / height
print("Fragment lat and lon size in deg: ", fragment_degrees_lat, fragment_degrees_lon)
print("Copied from a box: ", top_lat, "deg to", bottom_lat, "deg")

# Crop the image to the center area
center_area = image.crop((0, top, width, bottom))

# Create the output directory if it doesn't exist
output_dir = 'moonAlt'
os.makedirs(output_dir, exist_ok=True)

# Save 512x512 fragments of the center area
for i in range(0, width, fragment_size):
    for j in range(0, bottom - top, fragment_size):
        # Define the box to crop
        box = (i, j, i + fragment_size, j + fragment_size)
        fragment = center_area.crop(box)
        
        # Calculate latitude and longitude for the fragment
        box_lat = 45-180*j/height
        box_lon = 360*i/width - 180 # Adjust longitude to be in the range of -180 to 180

        # Check if the fragment is smaller than the desired size and skip if true
        if fragment.size[0] < fragment_size or fragment.size[1] < fragment_size:
            continue
        
        if(box_lat-fragment_degrees_lat<bottom_lat):
            break

        file_name = f'fragment_{int(box_lat)}_{int(box_lon)}.png'
        #print(file_name)
        
        
        # Save the fragment as a PNG file
        fragment_path = os.path.join(output_dir, file_name)
        fragment.save(fragment_path)

print("Fragments saved successfully.")
