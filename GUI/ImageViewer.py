import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QFileDialog, QVBoxLayout, QGroupBox
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt5.QtCore import Qt, pyqtSignal, QPoint


class ImageViewer(QWidget):
    imageChanged = pyqtSignal(object)
    selectionMade = pyqtSignal(tuple)  # Emits ((x1, y1), (x2, y2))

    def __init__(self,title = "image"):
        super().__init__()

        self.initializeUI(title)
        self.setupLayout()
        self.styleUI()
        self.setReadOnly(False)

        self.start_point = None
        self.end_point = None
        self.selecting = False
        self.image_with_rect = None

    def initializeUI(self, title):
        self.groupBox = QGroupBox(title)

        self.image = None
        self.isReadOnly = True
        self.image_label = QLabel("Double-click to upload an image", self)
        self.image_label.setAlignment(Qt.AlignCenter)

    def setupLayout(self):
        layout = QVBoxLayout()
        self.groupBox.setLayout(layout)  # Set the QGroupBox layout
        layout.addWidget(self.image_label)  # Add QLabel inside QGroupBox

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.groupBox)  # Add QGroupBox to the main layout
        self.setLayout(main_layout)

    def styleUI(self):
        """Applies styling to the QGroupBox"""
        self.groupBox.setStyleSheet("""
            QGroupBox {
                background-color: #f7f7f7;
                font-size: 18px;
                font-weight: bold;
                border: 2px solid #007BFF;
                border-radius: 10px;
                margin-top: 15px;
                padding: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 5px;
                color: #007BFF;
                border-radius: 5px;
            }
        """)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton and not self.isReadOnly:
            self.openImage()

    def openImage(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.xpm *.jpg *.jpeg *.bmp)")
        if file_path:
            self.displayImage(cv2.imread(file_path))

    def displayImage(self, image):
        """Displays an image in the QLabel."""
        if image is not None:
            self.image = image
            self.image_with_rect = image.copy()  # Keep a copy for selection
            self.imageChanged.emit(self.image)

            self.updatePixmap()

    def updatePixmap(self):
        """Updates QLabel with the current image (with or without selection box)."""
        if self.image_with_rect is not None:
            image_rgb = cv2.cvtColor(self.image_with_rect, cv2.COLOR_BGR2RGB)
            height, width, channels = image_rgb.shape
            bytes_per_line = channels * width
            q_image = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.pixmap = QPixmap.fromImage(q_image)
            self.image_label.setPixmap(
                self.pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)
            )

    def mousePressEvent(self, event):
        """Handles mouse press for region selection."""
        if self.image is not None and event.button() == Qt.LeftButton:
            self.start_point = event.pos()
            self.end_point = None
            self.selecting = True

    def mouseMoveEvent(self, event):
        """Handles mouse movement and updates selection rectangle."""
        if self.selecting and self.image is not None:
            self.end_point = event.pos()
            self.drawRectangle()

    def mouseReleaseEvent(self, event):
        """Handles mouse release and finalizes selection."""
        if self.selecting and self.image is not None:
            self.selecting = False
            self.end_point = event.pos()

            # Convert widget coordinates to image coordinates
            start = self.widgetToImageCoords(self.start_point)
            end = self.widgetToImageCoords(self.end_point)

            if start and end:
                self.selectionMade.emit((start, end))  # Emit the selection
                print(f"Selected Region: Start {start}, End {end}")

    def draw_on_image(self, coordinates, color=Qt.green, thickness=4, close_shape=True):
        """
        General function to draw on the image.

        :param coordinates: NumPy array of shape (N, 2) representing points.
        :param color: Color of the drawing (default: red).
        :param thickness: Line thickness (default: 2).
        :param close_shape: If True, it closes the shape (useful for polygons).
        """
        if coordinates is None or coordinates.shape[0] < 2:  # Check for valid input
            print("Error: Not enough points to draw")
            return

        if self.pixmap is None:
            print("Error: No image loaded")
            return

        # Create a copy of the pixmap to draw on
        pixmap_copy = self.pixmap.copy()
        painter = QPainter(pixmap_copy)
        pen = QPen(color, thickness)
        painter.setPen(pen)

        # Draw lines connecting points
        for i in range(coordinates.shape[0] - 1):  # Use NumPy indexing
            x1, y1 = coordinates[i]
            x2, y2 = coordinates[i + 1]
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))

        # Close the shape if required (connect last to first point)
        if close_shape and coordinates.shape[0] > 2:
            x1, y1 = coordinates[-1]
            x2, y2 = coordinates[0]
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))

        painter.end()

        # Set the updated image
        self.image_label.setPixmap(pixmap_copy)  # Ensure correct QLabel reference

    def drawRectangle(self):
        """Draws a rectangle on the image to show selection."""
        if self.image is not None and self.start_point and self.end_point:
            self.image_with_rect = self.image.copy()

            start = self.widgetToImageCoords(self.start_point)
            end = self.widgetToImageCoords(self.end_point)

            if start and end:
                cv2.rectangle(self.image_with_rect, start, end, (0, 0, 255), 2)

            self.updatePixmap()

    def widgetToImageCoords(self, widget_point):
        """Converts QLabel widget coordinates to actual image coordinates."""
        if self.image is None or self.image_label.pixmap() is None:
            return None

        # Get QLabel dimensions
        label_width = self.image_label.width()
        label_height = self.image_label.height()

        # Get Pixmap (scaled image) dimensions
        pixmap = self.image_label.pixmap()
        pixmap_width = pixmap.width()
        pixmap_height = pixmap.height()

        # Get Original Image dimensions
        img_height, img_width, _ = self.image.shape

        # Calculate scaling factors
        scale_x = img_width / pixmap_width
        scale_y = img_height / pixmap_height

        # Find if there are margins due to aspect ratio
        aspect_ratio_label = label_width / label_height
        aspect_ratio_img = img_width / img_height

        if aspect_ratio_img > aspect_ratio_label:
            # Image is wider than QLabel → Padding at the top & bottom
            scaled_width = label_width
            scaled_height = int(label_width / aspect_ratio_img)
            offset_x = 0
            offset_y = (label_height - scaled_height) // 2  # Centered padding
        else:
            # Image is taller than QLabel → Padding at the sides
            scaled_height = label_height
            scaled_width = int(label_height * aspect_ratio_img)
            offset_y = 0
            offset_x = (label_width - scaled_width) // 2  # Centered padding

        # Adjust widget coordinates to exclude padding
        x = widget_point.x() - offset_x
        y = widget_point.y() - offset_y

        # Ensure the point is inside the scaled image
        if x < 0 or y < 0 or x >= scaled_width or y >= scaled_height:
            return None  # Click was outside the actual image area

        # Convert QLabel scaled coordinates to original image coordinates
        img_x = int(x * (img_width / scaled_width))
        img_y = int(y * (img_height / scaled_height))

        return max(0, min(img_x, img_width - 1)), max(0, min(img_y, img_height - 1))

    def getSelection(self):
        """Returns the selected region's start and end points."""
        if self.start_point and self.end_point:
            return self.widgetToImageCoords(self.start_point), self.widgetToImageCoords(self.end_point)
        return None

    def setReadOnly(self, enabled: bool):
        """Enables or disables image uploading."""
        self.isReadOnly = enabled
        if enabled:
            self.image_label.setText("Image viewing mode only")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = ImageViewer()

    def on_selection_made(coords):
        print(f"Selected Region Coordinates: {coords}")
        print(viewer.image.shape[1])  # Width
        print(viewer.image.shape[0])  # Height


    viewer.selectionMade.connect(on_selection_made)

    viewer.show()
    sys.exit(app.exec_())
