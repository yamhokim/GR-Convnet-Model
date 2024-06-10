import cv2
import numpy as np

def preprocess_depth_image(depth_image, laplacian_scale=0.2):
    # Apply a median filter to reduce noise
    depth_image = cv2.medianBlur(depth_image, 5)

    # Fill holes in the data using morphological operations
    kernel = np.ones((5, 5), np.uint8)
    depth_image = cv2.morphologyEx(depth_image, cv2.MORPH_CLOSE, kernel)
    depth_image = cv2.morphologyEx(depth_image, cv2.MORPH_OPEN, kernel)

    # Histogram equalization to enhance object visibility
    equalized_image = cv2.equalizeHist(depth_image)

    # Apply Laplacian filter to enhance edges with reduced scale
    # laplacian_image = cv2.Laplacian(equalized_image, cv2.CV_64F, ksize=3)
    # laplacian_image = np.clip(laplacian_image * laplacian_scale, 0, 255).astype(np.uint8)

    # Thresholding to remove low-confidence pixels
    _, thresholded = cv2.threshold(equalized_image, 30, 255, cv2.THRESH_BINARY)

    # Combine the thresholded image with the original depth image
    cleaned_depth_image = cv2.bitwise_and(depth_image, depth_image, mask=thresholded)

    return cleaned_depth_image

def clean_and_save_image(source, destination, filename):
    # Load your depth image (replace '3_Depth.png' with your actual file path)
    depth_image = cv2.imread(source + "/" + filename)

    # Preprocess each channel of the depth image separately
    b, g, r = cv2.split(depth_image)
    cleaned_b = preprocess_depth_image(b, laplacian_scale=0.5)
    cleaned_g = preprocess_depth_image(g, laplacian_scale=0.5)
    cleaned_r = preprocess_depth_image(r, laplacian_scale=0.5)

    # Merge the cleaned channels back together
    cleaned_depth_image = cv2.merge([cleaned_b, cleaned_g, cleaned_r])

    # Save cleaned depth image
    cv2.imwrite(destination + "/" + filename, cleaned_depth_image)

clean_and_save_image(".", "cleaned", "3_Depth.png")

