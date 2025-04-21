from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, \
    QComboBox, QSpinBox
from fontTools.ttx import process

from Core.HarrisFeatures import extractHarrisFeatures
from Core.canny import canny
from Core.imageMode import rgb_to_grayscale
from Core.snake import snake_active_contour
from GUI.ImageViewer import ImageViewer


class FetchFeature(QMainWindow):
    def __init__(self):
        super().__init__()  # Initialize QMainWindow
        self.setWindowTitle("Fetch Feature")
        self.resize(1600, 700)

        self.initializeUI()
        self.setupLayout()
        self.styleUI()
        self.connectUI()

    def initializeUI(self):

        self.processingImage = None
        self.logo = QLabel("Fetch Feature")

        self.modeComboBoxLabel = QLabel("Mode")
        self.modeComboBox = QComboBox(self)
        self.modeComboBox.addItem("Harris")
        self.modeComboBox.addItem("- lambda")

        self.windowSizeLabel =QLabel("Window Size")
        self.windowSize = QSpinBox(self)
        self.windowSize.setValue(7)
        self.distanceThreshLabel =QLabel("Distance Threshold")
        self.distanceThresh = QSpinBox(self)
        self.distanceThresh.setValue(50)

        self.matchingComboBoxLabel = QLabel("Matching Method")
        self.matchingComboBox = QComboBox(self)
        self.matchingComboBox.addItem("SSD")
        self.matchingComboBox.addItem("NCC")


        self.inputViewer = ImageViewer("Input Image")
        self.inputViewer2 = ImageViewer("Input Image")
        self.secondOutputViewer = ImageViewer("Output Image")
        self.secondOutputViewer.setReadOnly(True)


        self.processButton = QPushButton("Process")


    def setupLayout(self):
        mainWidget = QWidget(self)
        self.setCentralWidget(mainWidget)

        mainLayout = QHBoxLayout()
        modesLayout = QVBoxLayout()
        workspace = QVBoxLayout()
        imagesLayout = QHBoxLayout()
        imagesLayoutV = QVBoxLayout()
        # self.parametersLayout = QHBoxLayout()
        #
        # self.parametersLayout.addStretch()
        # self.parametersLayout.addWidget(self.processButton)

        # Add widgets to layout
        modesLayout.addWidget(self.logo, alignment=Qt.AlignCenter)
        modesLayout.addWidget(self.modeComboBoxLabel)
        modesLayout.addWidget(self.modeComboBox)
        modesLayout.addWidget(self.windowSizeLabel)
        modesLayout.addWidget(self.windowSize)
        modesLayout.addWidget(self.distanceThreshLabel)
        modesLayout.addWidget(self.distanceThresh)
        modesLayout.addWidget(self.matchingComboBoxLabel)
        modesLayout.addWidget(self.matchingComboBox)
        modesLayout.addStretch()
        modesLayout.addWidget(self.processButton)

        # imagesLayoutV.addWidget(self.inputViewer2,1)
        # imagesLayoutV.addWidget(self.secondOutputViewer,1)

        imagesLayout.addWidget(self.inputViewer,1)
        imagesLayout.addWidget(self.inputViewer2,1)
        imagesLayout.addWidget(self.secondOutputViewer,1)
        # Nest layouts
        mainLayout.addLayout(modesLayout,10)
        mainLayout.addLayout(workspace,90)

        workspace.addLayout(imagesLayout)
        # workspace.addLayout(self.parametersLayout)

        mainWidget.setLayout(mainLayout)

    def styleUI(self):
        self.logo.setStyleSheet("font-family: 'Franklin Gothic';"
                                " font-size: 32px;"
                                " font-weight:600;"
                                " padding:30px;")

        button_style = """
              QPushButton {
                  font-family: 'Franklin Gothic';
                  font-size: 18px;
                  color: white;
                  background-color: #007BFF; /* Blue */
                  border-radius: 10px;
                  padding: 10px;
              }
              QPushButton:hover {
                  background-color: #0056b3; /* Darker Blue */
              }
              QPushButton:pressed {
                  background-color: #004494; /* Even Darker Blue */
              }
          """


    def connectUI(self):
        self.processButton.clicked.connect(self.processImage)

    def  processImage(self):
        self.firstprocessingImage = self.inputViewer.image.copy()
        self.secondProcessingImage = self.inputViewer2.image.copy()
        if self.modeComboBox.currentIndex() == 0:
            firstCorners,_,_,firstCornersMa0k = extractHarrisFeatures(self.firstprocessingImage,window_size=self.windowSize.value(), dist_threshold=self.distanceThreshold.value())
            secondCorners,_,_,secondCornersMark = extractHarrisFeatures(self.secondProcessingImage,window_size=self.windowSize.value(), dist_threshold=self.distanceThreshold.value())
        elif self.modeComboBox.currentIndex() == 1:
            pass




if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = Outlier()
    window.show()
    sys.exit(app.exec_())
