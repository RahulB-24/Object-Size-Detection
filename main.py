import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Function to identify only outermost objects and draw bounding boxes
def detect_objects(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny Edge Detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours left to right (x-coordinate of bounding box)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    # Filter only outermost objects (ignore small inner objects)
    detected_objects = []
    result_image = image.copy()

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 50 and h > 50:  # Ignore small noise
            detected_objects.append((x, y, w, h))

    return result_image, detected_objects

# Streamlit app
st.title("Object Size Detection")
st.write("Upload an image with a reference object and target objects.")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Detect objects and draw bounding boxes
    processed_image, objects = detect_objects(image)

    if len(objects) >= 2:
        # User input: Number of target objects (excluding reference)
        num_targets = st.number_input("How many target objects do you want to measure?", min_value=1, max_value=len(objects) - 1, value=len(objects) - 1)

        # User selects reference object
        reference_idx = st.selectbox("Select Reference Object", list(range(len(objects))), format_func=lambda i: f"Object {i + 1}")

        # User input: Reference object real-world size
        reference_real_width_cm = st.number_input("Reference Object Width (cm)", min_value=0.1, value=3.0, step=0.1)
        reference_real_height_cm = st.number_input("Reference Object Height (cm)", min_value=0.1, value=10.0, step=0.1)

        # Get reference object details
        ref_x, ref_y, ref_w, ref_h = objects[reference_idx]

        # Calculate **both** width-to-cm and height-to-cm ratios
        pixel_to_cm_width = reference_real_width_cm / ref_w
        pixel_to_cm_height = reference_real_height_cm / ref_h

        # **Ensure reference size displays exactly as the user input**
        reference_size_cm = (reference_real_width_cm, reference_real_height_cm)

        # Draw reference object with an arrow and label
        cv2.rectangle(processed_image, (ref_x, ref_y), (ref_x + ref_w, ref_y + ref_h), (255, 0, 0), 3)
        cv2.arrowedLine(processed_image, (ref_x + ref_w // 2, ref_y - 50), (ref_x + ref_w // 2, ref_y), (255, 0, 0), 5, tipLength=0.2)
        cv2.putText(processed_image, "Reference Object", (ref_x, ref_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3, cv2.LINE_AA)

        # Get target objects (excluding reference)
        target_objects = [obj for i, obj in enumerate(objects) if i != reference_idx][:num_targets]

        # Draw only the specified number of target objects
        for i, (x, y, w, h) in enumerate(target_objects):
            cv2.rectangle(processed_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the sizes separately
        st.write(f"**Reference Object Size:** {reference_size_cm[0]:.2f} cm x {reference_size_cm[1]:.2f} cm")

        st.write("**Target Object Sizes:**")
        for i, (x, y, w, h) in enumerate(target_objects):
            size_cm = (w * pixel_to_cm_width, h * pixel_to_cm_height)
            st.write(f"Object {i + 1}: {size_cm[0]:.2f} cm x {size_cm[1]:.2f} cm")

    else:
        st.write("Could not detect enough objects. Try using a clearer image.")

    # Convert for Streamlit display
    processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
    st.image(processed_image_rgb, caption="Detected Objects", use_column_width=True)

# python -m streamlit run main.py
