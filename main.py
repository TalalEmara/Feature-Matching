from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton
from fontTools.ttx import process

from Core.canny import canny
from Core.imageMode import rgb_to_grayscale
from Core.snake import snake_active_contour
from GUI.ImageViewer import ImageViewer


class Outlier(QMainWindow):
    def __init__(self):
        super().__init__()  # Initialize QMainWindow
        self.setWindowTitle("Outlier Detection")
        self.resize(1200, 800)

        self.initializeUI()
        self.setupLayout()
        self.styleUI()
        self.connectUI()

    def initializeUI(self):

        self.processingImage = None
        self.currentMode = "Snake"
        self.logo = QLabel("Outliner")

        def createModePanel():
            self.houghButton = QPushButton("Hough Transform")
            self.snakeButton = QPushButton("Snake (Active Contour)")

        createModePanel()

        self.inputViewer = ImageViewer("Input Image")
        self.outputViewer = ImageViewer("Output Image")
        self.outputViewer.setReadOnly(True)
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
        self.parametersLayout = QHBoxLayout()

        self.parametersLayout.addStretch()
        self.parametersLayout.addWidget(self.processButton)

        # Add widgets to layout
        modesLayout.addWidget(self.logo, alignment=Qt.AlignCenter)
        modesLayout.addWidget(self.houghButton)
        modesLayout.addWidget(self.snakeButton)
        modesLayout.addStretch()

        imagesLayoutV.addWidget(self.outputViewer,1)
        imagesLayoutV.addWidget(self.secondOutputViewer,1)

        imagesLayout.addWidget(self.inputViewer,1)
        imagesLayout.addLayout(imagesLayoutV,1)
        # Nest layouts
        mainLayout.addLayout(modesLayout,20)
        mainLayout.addLayout(workspace,80)

        workspace.addLayout(imagesLayout)
        workspace.addLayout(self.parametersLayout)

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

        self.houghButton.setStyleSheet(button_style)
        self.snakeButton.setStyleSheet(button_style)

    def connectUI(self):
        self.processButton.clicked.connect(self.processImage)
        self.inputViewer.selectionMade.connect(self.setSnakePoints)

    def setSnakePoints(self, selection):
        self.snakeStart, self.snakeEnd = selection

    def  processImage(self):
        self.processingImage = self.inputViewer.image.copy()
        if self.currentMode == "Hough":
            self.processingImage = canny(rgb_to_grayscale(self.processingImage))
            self.secondOutputViewer.displayImage(self.processingImage)
            self.secondOutputViewer.groupBox.setTitle("Canny Edges")
        elif self.currentMode == "Snake" :
            self.secondOutputViewer.displayImage(canny(rgb_to_grayscale(self.processingImage),30,100))
            self.processingImage, self.initialContour, self.finalContour = snake_active_contour(self.processingImage, self.snakeStart,self.snakeEnd)
            self.outputViewer.displayImage(self.processingImage)
            # self.secondOutputViewer.draw_on_image(self.initialContour,Qt.red, thickness=2)
            self.outputViewer.draw_on_image(self.finalContour)
            self.outputViewer.groupBox.setTitle("Snake Edges")




if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = Outlier()
    window.show()
    sys.exit(app.exec_())
