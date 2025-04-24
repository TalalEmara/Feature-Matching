import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, \
    QGroupBox, QSpinBox, QDoubleSpinBox, QComboBox, QBoxLayout

from Core.FeatureMatching import sift_detector, match_features, draw_matches
from Core.HarrisFeatures import extractHarrisFeatures
from Core.lamda import lambda_detector
from GUI.styles import GroupBoxStyle, button_style, second_button_style, label_style
from Core.canny import canny
from Core.imageMode import rgb_to_grayscale
from GUI.ImageViewer import ImageViewer
import time


class FetchFeature(QMainWindow):
    def __init__(self):
        super().__init__()  # Initialize QMainWindow
        self.setWindowTitle("Fetch Features")
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
        self.secondInputViewer = ImageViewer("Input Image")


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
        self.windowSize.setSingleStep(2)
        self.windowSize.valueChanged.connect(
            lambda v: self.windowSize.setValue(v + 1 if v % 2 == 0 else v)
        )
        # inside createCornerDetectParameters()

        # … after windowSize setup …

        # Harris-style (integer) distance threshold
        self.distThreshLabel_int = QLabel("Distance Threshold")
        self.distThreshLabel_int.setAlignment(Qt.AlignCenter)
        self.distThresh_int = QSpinBox()
        self.distThresh_int.setRange(1, 1200)
        self.distThresh_int.setValue(50)

        # Lambda-style (float %) threshold
        self.threshLabel_float = QLabel("Threshold (%)")
        self.threshLabel_float.setAlignment(Qt.AlignCenter)
        self.thresh_float = QDoubleSpinBox()
        self.thresh_float.setRange(0.1, 10.0)
        self.thresh_float.setSingleStep(0.1)
        self.thresh_float.setValue(0.01)
        self.thresh_float.setSuffix(" %")
        self.thresh_float.hide()
        self.threshLabel_float.hide()

        # Layout
        layout = QHBoxLayout()
        layout.addWidget(self.detectionMethodLabel)
        layout.addWidget(self.detectionMethod)
        layout.addWidget(self.windowSizeLabel)
        layout.addWidget(self.windowSize)

        layout.addWidget(self.distThreshLabel_int)
        layout.addWidget(self.distThresh_int)

        layout.addWidget(self.threshLabel_float)
        layout.addWidget(self.thresh_float)

        self.parametersGroupBox.setLayout(layout)

        self.detectionMethod.currentIndexChanged.connect(self.updateDetectionParameters)
        self.updateDetectionParameters()  # Initialize UI correctly

    def updateDetectionParameters(self):
        method = self.detectionMethod.currentText()

        if method == "Harris operator":
            # Show integer distance threshold
            self.distThreshLabel_int.show()
            self.distThresh_int.show()
            # Hide float threshold
            self.threshLabel_float.hide()
            self.thresh_float.hide()

            # Set ranges back if needed
            self.distThresh_int.setRange(1, 1200)
            self.windowSize.setRange(1, 1200)

        elif method == "- lambda method":
            # Hide integer distance threshold
            self.distThreshLabel_int.hide()
            self.distThresh_int.hide()
            self.thresh_float.setValue(0.01)
            # Show float threshold
            self.threshLabel_float.show()
            self.thresh_float.show()

            # Adjust window‑size range for lambda
            self.windowSize.setRange(1, 15)
            self.windowSize.setValue(5)
            # (thresh_float range already 0.1–10% with suffix)

    def createMatchingParameters(self):
        self.parametersGroupBox = QGroupBox("Matching Parameters")
        self.parametersGroupBox.setStyleSheet(GroupBoxStyle)

        self.matchingMethodLabel = QLabel("Matching method")
        self.matchingMethodLabel.setAlignment(Qt.AlignCenter)
        self.matchingMethod = QComboBox()
        self.matchingMethod.addItem("SSD")
        self.matchingMethod.addItem("NCC")

        self.topMatchesLabel = QLabel("Top matches:")
        self.topMatchesLabel.setAlignment(Qt.AlignCenter)
        self.topMatches = QSpinBox()
        self.topMatches.setRange(10, 300)
        self.topMatches.setValue(50)

        self.matchingThresholdLabel = QLabel("Max Radius:")
        self.matchingThresholdLabel.setAlignment(Qt.AlignCenter)
        self.matchingThreshold = QDoubleSpinBox()
        self.matchingThreshold.setRange(0, 1)
        self.matchingThreshold.setSingleStep(.1)
        self.matchingThreshold.setValue(.6)

        layout = QHBoxLayout()

        layout.addWidget(self.matchingMethodLabel)
        layout.addWidget(self.matchingMethod)
        layout.addWidget(self.topMatchesLabel)
        layout.addWidget(self.topMatches)
        layout.addWidget(self.matchingThresholdLabel)
        layout.addWidget(self.matchingThreshold)

        self.parametersGroupBox.setLayout(layout)




    def setupLayout(self):
        mainWidget = QWidget(self)
        self.setCentralWidget(mainWidget)

        mainLayout = QHBoxLayout()
        modesLayout = QVBoxLayout()
        workspace = QVBoxLayout()
        self.imagesLayout = QBoxLayout(QBoxLayout.TopToBottom)
        imagesLayoutH = QHBoxLayout()
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

        imagesLayoutH.addWidget(self.inputViewer,3)
        imagesLayoutH.addWidget(self.secondInputViewer, 3)


        self.imagesLayout.addLayout(imagesLayoutH,1)
        self.imagesLayout.addWidget(self.outputViewer,1)
        # Nest layouts
        mainLayout.addLayout(modesLayout,10)
        mainLayout.addLayout(workspace,90)

        workspace.addLayout(self.imagesLayout)
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
            self.secondInputViewer.hide()
            self.imagesLayout.setDirection(QBoxLayout.LeftToRight)
            # self.chainCodeLabel.show()

        elif mode == "Feature Matching":
            self.createMatchingParameters()
            self.imagesLayout.setDirection(QBoxLayout.TopToBottom)

            self.secondInputViewer.show()



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
        # self.inputViewer.selectionMade.connect(self.setSnakePoints)

    def processImage(self):
        self.processingImage = self.inputViewer.image.copy()
        if self.currentMode == "Corner Detection":
            if self.detectionMethod.currentIndex() == 0:
                _, _, _, self.processingImage = extractHarrisFeatures(
                    self.processingImage,
                    0.04,
                    self.windowSize.value(),
                    dist_threshold= self.distThresh_int.value()
                )
                self.outputViewer.groupBox.setTitle(f"Harris Detection)")

            elif self.detectionMethod.currentIndex() == 1:
                _,_,self.processingImage = lambda_detector(self.processingImage,self.thresh_float.value(),self.windowSize.value())
                self.outputViewer.groupBox.setTitle(f"- lambda")

            self.outputViewer.displayImage(self.processingImage)

        elif self.currentMode == "Feature Matching":
            self.secondProcessingImage = self.secondInputViewer.image.copy()
            kp1, des1 = sift_detector(rgb_to_grayscale(self.processingImage))
            kp2, des2 = sift_detector(rgb_to_grayscale(self.secondProcessingImage))

            if self.matchingMethod.currentIndex() == 0:
                matches = match_features(des1, des2,"ssd",self.topMatches.value(),self.matchingThreshold.value())
            elif self.matchingMethod.currentIndex() == 1:
                matches = match_features(des1, des2,"ncc",self.topMatches.value(),self.matchingThreshold.value())

            self.outputViewer.displayImage(draw_matches(self.processingImage, kp1, self.secondProcessingImage, kp2, matches))







if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = FetchFeature()
    window.show()
    sys.exit(app.exec_())
