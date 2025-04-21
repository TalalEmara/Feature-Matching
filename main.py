import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, \
    QGroupBox, QSpinBox, QDoubleSpinBox, QComboBox

from GUI.styles import GroupBoxStyle, button_style, second_button_style, label_style
from Core.canny import canny
from Core.imageMode import rgb_to_grayscale
from GUI.ImageViewer import ImageViewer


class Outlier(QMainWindow):
    def __init__(self):
        super().__init__()  # Initialize QMainWindow
        self.setWindowTitle("Outlier Detection")
        self.resize(1200, 800)

        self.initializeUI()
        self.createCornerDetectParameters()
        self.setupLayout()
        self.styleUI()
        self.connectUI()

    def initializeUI(self):

        self.processingImage = None
        self.currentMode = "Corner Detection"
        self.logo = QLabel("Fetch Feature")

        def createModePanel():
            self.cornerButton = QPushButton("Corner Detection")
            self.matchingButton = QPushButton("Feature Matching")


            self.cornerButton.clicked.connect(lambda: self.changeMode("Corner Detection"))
            self.matchingButton.clicked.connect(lambda: self.changeMode("Feature Matching"))


        createModePanel()

        self.inputViewer = ImageViewer("Input Image")
        self.outputViewer = ImageViewer("Output Image")
        self.outputViewer.setReadOnly(True)
        self.secondOutputViewer = ImageViewer("Output Image")
        self.secondOutputViewer.setReadOnly(True)


        self.processButton = QPushButton("Process")


    def createCornerDetectParameters(self):
        self.parametersGroupBox = QGroupBox("Corner Detection Parameters")
        self.parametersGroupBox.setStyleSheet(GroupBoxStyle)

        self.detectionMethodLabel = QLabel("Detection method")
        self.detectionMethodLabel.setAlignment(Qt.AlignCenter)
        self.detectionMethod = QComboBox()
        self.detectionMethod.addItem("Harris operator")
        self.detectionMethod.addItem("- lambda method")

        self.windowSizeLabel = QLabel("Window size")
        self.windowSizeLabel.setAlignment(Qt.AlignCenter)
        self.windowSize = QSpinBox()
        self.windowSize.setValue(7)


        self.distanceThresholdLabel = QLabel("Distance Threshold")
        self.distanceThresholdLabel.setAlignment(Qt.AlignCenter)
        self.distanceThreshold = QSpinBox()
        self.distanceThreshold.setValue(50)


        layout = QHBoxLayout()

        layout.addWidget(self.detectionMethodLabel)
        layout.addWidget(self.detectionMethod)
        layout.addWidget(self.windowSizeLabel)
        layout.addWidget(self.windowSize)
        layout.addWidget(self.distanceThresholdLabel)
        layout.addWidget(self.distanceThreshold)

        self.parametersGroupBox.setLayout(layout)
    def createHoughCirclesParameters(self):
        self.parametersGroupBox = QGroupBox("Hough Circles Parameters")
        self.parametersGroupBox.setStyleSheet(GroupBoxStyle)

        self.thresholdLabel = QLabel("Threshold:")
        self.thresholdLabel.setAlignment(Qt.AlignCenter)
        self.threshold = QSpinBox()
        self.threshold.setRange(0, 1000)
        self.threshold.setValue(100)

        self.minradiusLabel = QLabel("Min Radius:")
        self.minradiusLabel.setAlignment(Qt.AlignCenter)
        self.minraduis = QSpinBox()
        self.minraduis.setRange(0, 1000)
        self.minraduis.setValue(1)

        self.maxradiusLabel = QLabel("Max Radius:")
        self.maxradiusLabel.setAlignment(Qt.AlignCenter)
        self.maxraduis = QSpinBox()
        self.maxraduis.setRange(0, 1000)
        self.maxraduis.setValue(100)

        layout = QHBoxLayout()

        layout.addWidget(self.thresholdLabel)
        layout.addWidget(self.threshold)
        layout.addWidget(self.minradiusLabel)
        layout.addWidget(self.minraduis)
        layout.addWidget(self.maxradiusLabel)
        layout.addWidget(self.maxraduis)

        self.parametersGroupBox.setLayout(layout)


    def createHoughEllipseParameters(self):
        self.parametersGroupBox = QGroupBox("Hough Ellipse Parameters")
        self.parametersGroupBox.setStyleSheet(GroupBoxStyle)

        self.thresholdLabel = QLabel("Threshold:")
        self.thresholdLabel.setAlignment(Qt.AlignCenter)
        self.threshold = QSpinBox()
        self.threshold.setRange(0, 1000)
        self.threshold.setValue(150)


        layout = QHBoxLayout()

        layout.addWidget(self.thresholdLabel)
        layout.addWidget(self.threshold)

        self.parametersGroupBox.setLayout(layout)


    def createCannyParameters(self):
        self.parametersGroupBox = QGroupBox("Canny Parameters")
        self.parametersGroupBox.setStyleSheet(GroupBoxStyle)

        # Create QLabel and QSpinBox for Low Threshold
        self.lowThresholdLabel = QLabel("Low Threshold:")
        self.lowThresholdLabel.setAlignment(Qt.AlignCenter)
        self.lowThreshold = QSpinBox()
        self.lowThreshold.setRange(0, 255)
        self.lowThreshold.setValue(100)

        # Create QLabel and QSpinBox for High Threshold
        self.highThresholdLabel = QLabel("High Threshold:")
        self.highThresholdLabel.setAlignment(Qt.AlignCenter)
        self.highThreshold = QSpinBox()
        self.highThreshold.setRange(0, 255)
        self.highThreshold.setValue(150)

        layout = QHBoxLayout()

        lowLayout = QHBoxLayout()
        lowLayout.addWidget(self.lowThresholdLabel)
        lowLayout.addWidget(self.lowThreshold)

        highLayout = QHBoxLayout()
        highLayout.addWidget(self.highThresholdLabel)
        highLayout.addWidget(self.highThreshold)

        layout.addLayout(lowLayout)
        layout.addLayout(highLayout)

        self.parametersGroupBox.setLayout(layout)



    def setupLayout(self):
        mainWidget = QWidget(self)
        self.setCentralWidget(mainWidget)

        mainLayout = QHBoxLayout()
        modesLayout = QVBoxLayout()
        workspace = QVBoxLayout()
        imagesLayout = QHBoxLayout()
        imagesLayoutV = QVBoxLayout()
        self.parametersLayout = QHBoxLayout()

        self.parametersLayout.addWidget(self.parametersGroupBox)
        self.parametersLayout.addWidget(self.processButton)

        # Add widgets to layout
        modesLayout.addWidget(self.logo, alignment=Qt.AlignCenter)
        modesLayout.addWidget(self.cornerButton)
        modesLayout.addWidget(self.matchingButton)
        # modesLayout.addWidget(self.houghCirclesButton)
        # modesLayout.addWidget(self.houghEllipseButton)
        # modesLayout.addWidget(self.snakeButton)
        modesLayout.addStretch()

        imagesLayoutV.addWidget(self.outputViewer,4)
        imagesLayoutV.addWidget(self.secondOutputViewer,3)


        imagesLayout.addWidget(self.inputViewer,1)
        imagesLayout.addLayout(imagesLayoutV,1)
        # Nest layouts
        mainLayout.addLayout(modesLayout,10)
        mainLayout.addLayout(workspace,90)

        workspace.addLayout(imagesLayout)
        workspace.addLayout(self.parametersLayout)

        mainWidget.setLayout(mainLayout)

    def changeMode(self, mode):
        """Change the current mode and update the UI accordingly."""
        self.currentMode = mode

        # Remove existing parametersGroupBox if it exists
        if hasattr(self, "parametersGroupBox"):
            self.parametersLayout.removeWidget(self.parametersGroupBox)
            self.parametersGroupBox.deleteLater()  # Properly delete the widget

        # Create the corresponding parameter panel
        if mode == "Corner Detection":
            self.createCornerDetectParameters()
            self.secondOutputViewer.hide()
            # self.chainCodeLabel.show()

        elif mode == "Hough Lines":
            self.createHoughLinesParameters()
            self.chainCodeLabel.hide()
            self.perimeterLabel.hide()
            self.areaLabel.hide()
            self.secondOutputViewer.show()
        elif mode == "Hough Circles":
            self.createHoughCirclesParameters()
            self.chainCodeLabel.hide()
            self.perimeterLabel.hide()
            self.areaLabel.hide()
            self.secondOutputViewer.show()
        elif mode == "Hough Ellipse":
            self.createHoughEllipseParameters()
            self.chainCodeLabel.hide()
            self.perimeterLabel.hide()
            self.areaLabel.hide()
            self.secondOutputViewer.show()
        else:
            self.createCannyParameters()
            self.secondOutputViewer.show()


        # Add new parameters group box to layout
        self.parametersLayout.insertWidget(0, self.parametersGroupBox)

    def styleUI(self):
        self.logo.setStyleSheet("font-family: 'Franklin Gothic';"
                                " font-size: 32px;"
                                " font-weight:600;"
                                " padding:30px;")


        self.processButton.setFixedWidth(250)
        self.processButton.setFixedHeight(40)
        # self.processButton.setStyleSheet(second_button_style)
        self.cornerButton.setStyleSheet(button_style)
        self.matchingButton.setStyleSheet(button_style)



    def connectUI(self):
        self.processButton.clicked.connect(self.processImage)
        self.inputViewer.selectionMade.connect(self.setSnakePoints)

    def setSnakePoints(self, selection):
        self.snakeStart, self.snakeEnd = selection

    def  processImage(self):
        self.processingImage = self.inputViewer.image.copy()
        if self.currentMode == "Canny":
            self.processingImage = canny(rgb_to_grayscale(self.processingImage), self.lowThreshold.value(), self.highThreshold.value())
            self.outputViewer.displayImage(self.processingImage)
            self.outputViewer.groupBox.setTitle("Canny Edges")





if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = Outlier()
    window.show()
    sys.exit(app.exec_())
