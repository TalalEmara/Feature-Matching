GroupBoxStyle =  '''
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
            '''

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

second_button_style = """
             QPushButton {
                    font-family: 'Franklin Gothic';
                    font-size: 18px;
                    color: #007BFF;
                    border: 2px solid #007BFF;
                    background-color: white; /* Light gray */
                    border-radius: 10px;
                    padding: 10px;
                }
                
                QPushButton:hover {
                    background-color: #DDEEFF; /* Lighter blue */
                }
                
                QPushButton:pressed {
                    background-color: #CCE0FF; /* Even lighter blue */
                }
          """

label_style = """
            QLabel {
                font-family: 'Franklin Gothic';
                font-size: 16px;
                font-weight: bold;
                color: #0056b3; /* Deep blue */
                background-color: #f7f7f7; /* Light gray background */
                border: 2px solid #007BFF; /* Blue border */
                border-radius: 8px;
                padding: 6px;
                margin: 5px;
                qproperty-alignment: AlignCenter;
            }
        """

