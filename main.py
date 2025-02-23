import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Function to identify objects and draw bounding boxes
def detect_objects(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Convert to binary image using adaptive thresholding for better detection
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours left to right (x-coordinate of bounding box)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    # Draw bounding boxes
    result_image = image.copy()
    detected_objects = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        detected_objects.append((x, y, w, h))
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return result_image, detected_objects

# Streamlit app
st.title("Object Size Detection")
st.write("Upload an image with a reference object on the left and a target object on the right.")

# Default values for the reference object (in cm)
# Change these values based on the actual size of the reference object in the real world
reference_real_width_cm = 3  # real-world width of the reference object (in cm)
reference_real_height_cm = 10  # real-world height of the reference object (in cm)

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Detect objects and draw bounding boxes
    processed_image, objects = detect_objects(image)

    # Convert for Streamlit display
    processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
    st.image(processed_image_rgb, caption="Detected 2 Objects", use_column_width=True)

    # Calculating the pixel-to-centimeter ratio using the reference object
    if len(objects) >= 2:
        # Assuming the leftmost object is the reference object
        ref_x, ref_y, ref_w, ref_h = objects[0]  # Reference object coordinates and size in pixels
        # Calculate pixel-to-cm ratio
        pixel_to_cm = reference_real_width_cm / ref_w  # Assuming width of reference object is used for scale

        # Now, display the size of the reference object in centimeters
        ref_size_cm = (ref_w * pixel_to_cm, ref_h * pixel_to_cm)

        # Display detected object positions and sizes in cms
        st.write("Detected objects (Reference(left) and Target(Right)):")
        for i, (x, y, w, h) in enumerate(objects):
            if i == 0:
                st.write(f"Reference Object: X={x}, Y={y}, Width={w}, Height={h} (in pixels) -> Size: {ref_size_cm[0]:.2f} cm x {ref_size_cm[1]:.2f} cm")
            else:
                # For the target object, calculate its size in cm
                target_size_cm = (w * pixel_to_cm, h * pixel_to_cm)
                st.write(f"Target Object {i}: X={x}, Y={y}, Width={w}, Height={h} (in pixels) -> Size: {target_size_cm[0]:.2f} cm x {target_size_cm[1]:.2f} cm")
    else:
        st.write("Could not detect enough objects. Ensure a clear background and distinct objects.")

# python -m streamlit run main.py
