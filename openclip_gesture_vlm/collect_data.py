import cv2
import os
import time

# Path to the dataset directory, going up one level from the script's location
# and then into the 'dataset' folder.
DATASET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset'))

# Camera setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()


print("Starting camera...")
print("Instructions:")
print("- Position your hand in the frame.")
print("- Press the SPACEBAR to capture an image.")
print("- Press the 'q' key to quit the program.")
print("-" * 30)


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    # Display the resulting frame
    cv2.imshow('Camera - Press SPACE to capture, q to quit', frame)

    key = cv2.waitKey(1) & 0xFF

    # If the 'q' key is pressed, break from the loop
    if key == ord('q'):
        break

    # If the spacebar is pressed, capture the image
    if key == ord(' '):
        # Temporarily freeze the frame by showing it in a new window
        capture_display = frame.copy()
        cv2.putText(capture_display, "Image Captured!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Captured Image', capture_display)
        
        print("\nImage captured!")
        
        # Get gesture label from user input
        print("\nEnter the gesture label number:")
        print("  0: Thumbs Up")
        print("  1: Fist")
        print("  2: Palm")
        label_input = input("Enter number (0, 1, or 2) and press Enter: ").strip()

        label_map = {
            "0": "thumbs_up",
            "1": "fist",
            "2": "palm"
        }

        cv2.destroyWindow('Captured Image') # Close the frozen frame window

        if label_input not in label_map:
            print("Invalid input. Please enter 0, 1, or 2. Image not saved.")
            continue
            
        label = label_map[label_input]
            
        # Create the directory for the label if it doesn't exist
        label_dir = os.path.join(DATASET_DIR, label)
        os.makedirs(label_dir, exist_ok=True)
        
        # Find the highest existing image number to avoid overwriting
        highest_num = 0
        for filename in os.listdir(label_dir):
            if filename.startswith('img') and filename.endswith('.jpg'):
                try:
                    # Extract number from filename like 'img001.jpg'
                    num = int(filename[3:-4])
                    if num > highest_num:
                        highest_num = num
                except ValueError:
                    # Ignore files that don't match the pattern
                    continue
        
        new_img_num = highest_num + 1
        img_name = f"img{new_img_num:03d}.jpg"
        img_path = os.path.join(label_dir, img_name)
        
        # Save the captured frame
        cv2.imwrite(img_path, frame)
        print(f"Success! Image saved to: {img_path}")
        print("\nResuming camera. Press SPACE to capture another image or 'q' to quit.")


# When everything done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()
print("Script finished. Your dataset has been updated.") 