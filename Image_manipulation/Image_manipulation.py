import cv2
import numpy as np
import matplotlib.pyplot as plt

def edge_detection(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    edges = cv2.Canny(image, 100, 200)
    plt.subplot(1, 3, 1), plt.imshow(sobelx, cmap='gray'), plt.title('Sobel X')
    plt.subplot(1, 3, 2), plt.imshow(sobely, cmap='gray'), plt.title('Sobel Y')
    plt.subplot(1, 3, 3), plt.imshow(edges, cmap='gray'), plt.title('Canny Edges')
    plt.show()

def morphological_operations(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5,5), np.uint8)
    dilation = cv2.dilate(binary, kernel, iterations=1)
    erosion = cv2.erode(binary, kernel, iterations=1)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    plt.subplot(2, 2, 1), plt.imshow(dilation, cmap='gray'), plt.title('Dilation')
    plt.subplot(2, 2, 2), plt.imshow(erosion, cmap='gray'), plt.title('Erosion')
    plt.subplot(2, 2, 3), plt.imshow(opening, cmap='gray'), plt.title('Opening')
    plt.subplot(2, 2, 4), plt.imshow(closing, cmap='gray'), plt.title('Closing')
    plt.show()

def perspective_transformation(image_path):
    image = cv2.imread(image_path)
    rows, cols, ch = image.shape
    pts1 = np.float32([[100, 100], [400, 100], [100, 400], [400, 400]])
    pts2 = np.float32([[50, 50], [500, 50], [100, 400], [400, 500]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    transformed = cv2.warpPerspective(image, M, (cols, rows))
    plt.imshow(cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB)), plt.title('Perspective Transformation')
    plt.show()

def image_pyramids(image_path):
    image = cv2.imread(image_path)
    smaller = cv2.pyrDown(image)
    larger = cv2.pyrUp(image)
    plt.subplot(1, 2, 1), plt.imshow(cv2.cvtColor(smaller, cv2.COLOR_BGR2RGB)), plt.title('PyrDown')
    plt.subplot(1, 2, 2), plt.imshow(cv2.cvtColor(larger, cv2.COLOR_BGR2RGB)), plt.title('PyrUp')
    plt.show()

def cropping(image_path):
    image = cv2.imread(image_path)
    cropped = image[50:200, 50:200]
    plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)), plt.title('Cropped Image')
    plt.show()

def resize_interpolations(image_path):
    image = cv2.imread(image_path)
    nearest = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    linear = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    cubic = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    plt.subplot(1, 3, 1), plt.imshow(cv2.cvtColor(nearest, cv2.COLOR_BGR2RGB)), plt.title('Nearest')
    plt.subplot(1, 3, 2), plt.imshow(cv2.cvtColor(linear, cv2.COLOR_BGR2RGB)), plt.title('Linear')
    plt.subplot(1, 3, 3), plt.imshow(cv2.cvtColor(cubic, cv2.COLOR_BGR2RGB)), plt.title('Cubic')
    plt.show()

def thresholding(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    adaptive = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    plt.subplot(1, 2, 1), plt.imshow(binary, cmap='gray'), plt.title('Binary Threshold')
    plt.subplot(1, 2, 2), plt.imshow(adaptive, cmap='gray'), plt.title('Adaptive Threshold')
    plt.show()

def sharpening(image_path):
    image = cv2.imread(image_path)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    plt.imshow(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)), plt.title('Sharpened Image')
    plt.show()

def blurring(image_path):
    image = cv2.imread(image_path)
    gaussian = cv2.GaussianBlur(image, (7, 7), 0)
    median = cv2.medianBlur(image, 5)
    bilateral = cv2.bilateralFilter(image, 9, 75, 75)
    plt.subplot(1, 3, 1), plt.imshow(cv2.cvtColor(gaussian, cv2.COLOR_BGR2RGB)), plt.title('Gaussian Blur')
    plt.subplot(1, 3, 2), plt.imshow(cv2.cvtColor(median, cv2.COLOR_BGR2RGB)), plt.title('Median Blur')
    plt.subplot(1, 3, 3), plt.imshow(cv2.cvtColor(bilateral, cv2.COLOR_BGR2RGB)), plt.title('Bilateral Blur')
    plt.show()

def contour_detection(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Contours')
    plt.show()

def hough_lines(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a * rho, b * rho
            x1, y1 = int(x0 + 1000 * (-b)), int(y0 + 1000 * a)
            x2, y2 = int(x0 - 1000 * (-b)), int(y0 - 1000 * a)
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    else:
        print("No lines were detected.")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Hough Lines')
    plt.show()


edge_detection("G:\ML Nagpur\Event 2\Images\persons2.jpg")
# morphological_operations("G:\ML Nagpur\Event 2\Images\persons2.jpg")
# perspective_transformation("G:\ML Nagpur\Event 2\Images\persons2.jpg")
# image_pyramids("G:\ML Nagpur\Event 2\Images\persons2.jpg")
# cropping("G:\ML Nagpur\Event 2\Images\persons2.jpg")
# resize_interpolations("G:\ML Nagpur\Event 2\Images\persons2.jpg")
# thresholding("G:\ML Nagpur\Event 2\Images\persons2.jpg")
# sharpening("G:\ML Nagpur\Event 2\Images\persons2.jpg")
# blurring("G:\ML Nagpur\Event 2\Images\persons2.jpg")
# contour_detection("G:\ML Nagpur\Event 2\Images\persons2.jpg")
# hough_lines("G:\ML Nagpur\Event 2\Images\sudoku.png")
