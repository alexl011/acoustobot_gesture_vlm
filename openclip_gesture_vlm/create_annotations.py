import os

# The root directory of your dataset
root_dir = '../dataset'
# The name of the output annotation file
output_file = './annotations.txt'

# A list to hold all the image-label pairs
annotations = []

# Iterate through each gesture folder (fist, palm, thumbs_up)
for gesture_name in os.listdir(root_dir):
    gesture_dir = os.path.join(root_dir, gesture_name)
    
    # Skip non-directory files like .DS_Store
    if not os.path.isdir(gesture_dir):
        continue
        
    # Iterate through each image in the gesture folder
    for image_name in os.listdir(gesture_dir):
        # Make sure we only process image files
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Create the relative path for the image
            image_path = os.path.join(gesture_name, image_name)
            # Use the folder name as the label, replacing underscores with spaces
            label = gesture_name.replace('_', ' ')
            # Add the image path and label to our list
            annotations.append(f"{image_path}\t{label}")

# Write all the annotations to the output file
with open(output_file, 'w') as f:
    for line in annotations:
        f.write(line + '\n')

print(f"Successfully created '{output_file}' with {len(annotations)} entries.") 