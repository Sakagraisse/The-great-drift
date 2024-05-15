import sys
import os
import numpy as np
import Simulation_Func as SF
import Graph_Code as GC
import time
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QSpinBox, QLabel, \
    QDoubleSpinBox, QCheckBox, QRadioButton, QButtonGroup, QGroupBox,QGridLayout, QHBoxLayout, \
    QSizePolicy, QTextEdit

from PyQt6.QtGui import QPixmap, QColor
from PyQt6.QtCore import Qt, QSize, QProcess


class AspectRatioLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.aspect_ratio = 1.25 # Set the desired aspect ratio here

    def resizeEvent(self, event):
        w = self.width()
        h = int(w / self.aspect_ratio)
        self.setFixedSize(QSize(w, h))

class ParameterChooser(QWidget):
    def __init__(self):
        super().__init__()



        # Create a QGroupBox
        self.base_parameters_group = QGroupBox("Base Parameters")
        self.base_parameters_layout = QGridLayout()

        self.group_size_label = QLabel("Group Size")
        self.group_size_input = QSpinBox()
        self.group_size_input.setRange(1, 100)
        self.group_size_input.setValue(24)  # Set default value
        self.base_parameters_layout.addWidget(self.group_size_label, 0, 0)
        self.base_parameters_layout.addWidget(self.group_size_input, 0, 1)

        self.number_groups_label = QLabel("Number of Groups")
        self.number_groups_input = QSpinBox()
        self.number_groups_input.setRange(1, 100)
        self.number_groups_input.setSingleStep(10)
        self.number_groups_input.setValue(40)  # Set default value
        self.base_parameters_layout.addWidget(self.number_groups_label, 1, 0)
        self.base_parameters_layout.addWidget(self.number_groups_input, 1, 1)

        self.num_interactions_label = QLabel("Number of Interactions")
        self.num_interactions_input = QSpinBox()
        self.num_interactions_input.setRange(1, 1000)
        self.num_interactions_input.setValue(100)  # Set default value
        self.base_parameters_layout.addWidget(self.num_interactions_label, 0, 2)
        self.base_parameters_layout.addWidget(self.num_interactions_input, 0, 3)

        self.period_label = QLabel("Period")
        self.period_input = QSpinBox()
        self.period_input.setRange(100, 200000)
        self.period_input.setValue(1500)  # Set default value
        self.base_parameters_layout.addWidget(self.period_label, 1, 2)
        self.base_parameters_layout.addWidget(self.period_input, 1, 3)

        self.to_migrate_label = QLabel("To Migrate")
        self.to_migrate_input = QSpinBox()
        self.to_migrate_input.setRange(0, 16)
        self.to_migrate_input.setValue(8)  # Set default value
        self.base_parameters_layout.addWidget(self.to_migrate_label, 2, 0)
        self.base_parameters_layout.addWidget(self.to_migrate_input, 2, 1)

        self.mu_label = QLabel("Mu")
        self.mu_input = QDoubleSpinBox()
        self.mu_input.setRange(0.01, 1.00)
        self.mu_input.setSingleStep(0.01)
        self.mu_input.setValue(0.02)  # Set default value
        self.base_parameters_layout.addWidget(self.mu_label, 3, 0)
        self.base_parameters_layout.addWidget(self.mu_input, 3, 1)

        self.step_size_label = QLabel("Step Size")
        self.step_size_input = QDoubleSpinBox()
        self.step_size_input.setRange(0.010, 1.000)
        self.step_size_input.setSingleStep(0.001)
        self.step_size_input.setValue(0.025)  # Set default value
        self.step_size_input.setDecimals(3)
        self.base_parameters_layout.addWidget(self.step_size_label, 2, 2)
        self.base_parameters_layout.addWidget(self.step_size_input, 2, 3)

        self.truc_label = QLabel("Truc")
        self.truc_input = QDoubleSpinBox()
        self.truc_input.setRange(0, 1.00)
        self.truc_input.setSingleStep(0.01)
        self.truc_input.setValue(0.5)  # Set default value
        self.base_parameters_layout.addWidget(self.truc_label, 3, 2)
        self.base_parameters_layout.addWidget(self.truc_input, 3, 3)

        self.transfert_multiplier_label = QLabel("Transfert Multiplier")
        self.transfert_multiplier_input = QDoubleSpinBox()
        self.transfert_multiplier_input.setRange(0, 100)
        self.transfert_multiplier_input.setSingleStep(1)
        self.transfert_multiplier_input.setValue(2)  # Set default value
        self.base_parameters_layout.addWidget(self.transfert_multiplier_label, 5, 0)
        self.base_parameters_layout.addWidget(self.transfert_multiplier_input, 5, 1)

        self.lambda_param_label = QLabel("Lambda Param")
        self.lambda_param_input = QDoubleSpinBox()
        self.lambda_param_input.setEnabled(False)
        self.lambda_param_input.setRange(0, 100)
        self.lambda_param_input.setSingleStep(1)
        self.lambda_param_input.setValue(10)  # Set default value
        self.base_parameters_layout.addWidget(self.lambda_param_label, 4, 2)
        self.base_parameters_layout.addWidget(self.lambda_param_input, 4, 3)

        self.theta_label = QLabel("Theta")
        self.theta_input = QDoubleSpinBox()
        self.theta_input.setEnabled(False)
        self.theta_input.setRange(0.01, 1.00)
        self.theta_input.setSingleStep(0.01)
        self.theta_input.setValue(0.5)  # Set default value
        self.base_parameters_layout.addWidget(self.theta_label, 4, 0)
        self.base_parameters_layout.addWidget(self.theta_input, 4, 1)

        # Create radio buttons for "Coupled" and "Uncoupled"
        self.coupled_button = QRadioButton("Coupled")
        self.coupled_button.setChecked(True)  # Set "Coupled" as the default value
        self.uncoupled_button = QRadioButton("Uncoupled")

        # Add the radio buttons to the layout
        self.base_parameters_layout.addWidget(self.coupled_button, 6, 0)
        self.base_parameters_layout.addWidget(self.uncoupled_button, 6, 2)

        # Create a button group to make the radio buttons exclusive
        self.coupled_button_group = QButtonGroup()
        self.coupled_button_group.addButton(self.coupled_button)
        self.coupled_button_group.addButton(self.uncoupled_button)

        self.base_parameters_group.setLayout(self.base_parameters_layout)

        #Create a QGroupBox for Average over simulations
        self.average_group = QGroupBox("Average over simulations")
        self.average_layout = QVBoxLayout()

        self.number_average_input = QLabel("Step Size")
        self.number_average_input = QSpinBox()
        self.number_average_input.setRange(1, 30)
        self.number_average_input.setValue(1)
        self.number_average_input.setSingleStep(1)
        self.average_layout.addWidget(self.number_average_input)

        self.average_group.setLayout(self.average_layout)



        # Create a QGroupBox for Simulation Type
        self.simulation_type_group = QGroupBox("Simulation Type")
        self.simulation_type_layout = QVBoxLayout()

        self.option1_button = QRadioButton("Repeated Interactions")
        self.option1_button.setChecked(True)
        self.option1_button.toggled.connect(self.update_theta_lambda)
        self.option2_button = QRadioButton("Group Competition")
        #hide the option2_button
        self.option2_button.hide()
        self.option2_button.toggled.connect(self.update_num_interactions)
        self.option3_button = QRadioButton("Joint Scenario")
        self.option3_button.hide()

        self.button_group = QButtonGroup()
        self.button_group.addButton(self.option1_button)
        self.button_group.addButton(self.option2_button)
        self.button_group.addButton(self.option3_button)

        self.simulation_type_layout.addWidget(self.option1_button)
        self.simulation_type_layout.addWidget(self.option2_button)
        self.simulation_type_layout.addWidget(self.option3_button)

        self.simulation_type_group.setLayout(self.simulation_type_layout)

        # Create a QHBoxLayout
        self.main_layout = QHBoxLayout()

        # Create a QVBoxLayout for the left column
        self.left_column_layout = QVBoxLayout()

        # Add the base parameters group and the simulation type group to the left column
        self.left_column_layout.addWidget(self.base_parameters_group)
        self.left_column_layout.addWidget(self.simulation_type_group)
        self.left_column_layout.addWidget(self.average_group)

        # Add the submit button to the left column
        self.submit_button = QPushButton("Submit")
        self.submit_button.clicked.connect(self.submit)
        self.left_column_layout.addWidget(self.submit_button)

        # Add the left column layout to the main layout
        self.main_layout.addLayout(self.left_column_layout)



        # Create a QVBoxLayout for the right column
        self.right_column_layout = QVBoxLayout()

        # Create a QHBoxLayout for the buttons
        self.button_layout = QHBoxLayout()

        # Create the "Graph 1" and "Graph 2" buttons
        self.graph1_button = QPushButton("Graph 1")
        self.graph1_button.clicked.connect(self.graph1)
        self.button_layout.addWidget(self.graph1_button)

        self.graph2_button = QPushButton("Graph 2")
        self.graph2_button.clicked.connect(self.graph2)
        self.button_layout.addWidget(self.graph2_button)

        # Create a QGroupBox for the buttons
        self.graph_type_group = QGroupBox("Type of Graph")
        self.graph_type_group.setLayout(self.button_layout)

        # Add the graph type group to the right column layout
        self.right_column_layout.addWidget(self.graph_type_group)

        pixmap = QPixmap(600, 400)  # Adjust the size to fit your needs
        pixmap.fill(QColor(0, 0, 0, 0))  # Fill the QPixmap with a transparent color

        # Create a QLabel for the graph
        self.graph_label = QLabel()

        self.graph_label.setPixmap(pixmap)


        self.right_column_layout.addWidget(self.graph_label)


        # Add the right column layout to the main layout
        self.main_layout.addLayout(self.right_column_layout)

        # Set the main layout of the window
        self.setLayout(self.main_layout)



        #self.setLayout(self.layout)


    def update_theta_lambda(self, checked):
        if checked:
            self.theta_input.setEnabled(False)
            self.lambda_param_input.setEnabled(False)
        else:
            self.theta_input.setEnabled(True)
            self.lambda_param_input.setEnabled(True)

    def update_num_interactions(self, checked):
        if checked:
            self.num_interactions_input.setValue(1)
            self.num_interactions_input.setEnabled(False)
        else:
            self.num_interactions_input.setValue(100)
            self.num_interactions_input.setEnabled(True)

    def get_dir_size(self, path):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                # Skip if it is symbolic link
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
        return total_size

    def submit(self):
        group_size = self.group_size_input.value()
        number_groups = self.number_groups_input.value()
        num_interactions = self.num_interactions_input.value()
        to_migrate = self.to_migrate_input.value()
        period = self.period_input.value()
        mu = self.mu_input.value()
        step_size = self.step_size_input.value()
        truc = self.truc_input.value()
        coupled = self.coupled_button.isChecked()
        transfert_multiplier = self.transfert_multiplier_input.value()
        lambda_param = self.lambda_param_input.value()
        theta = self.theta_input.value()

        print(f"Group Size: {group_size}")
        print(f"Number of Groups: {number_groups}")
        print(f"Number of Interactions: {num_interactions}")
        print(f"To Migrate: {to_migrate}")
        print(f"Period: {period}")
        print(f"Mu: {mu}")
        print(f"Step Size: {step_size}")
        print(f"Truc: {truc}")
        print(f"Coupled: {coupled}")
        print(f"Transfert Multiplier: {transfert_multiplier}")
        print(f"Lambda Param: {lambda_param}")
        print(f"Theta: {theta}")


        frame_a = np.zeros((period, (group_size * number_groups)))
        frame_x_store = np.zeros((75, (group_size * number_groups)))
        frame_a_store = np.zeros((75, (group_size * number_groups)))
        frame_d_store = np.zeros((75, (group_size * number_groups)))
        frame_x = np.zeros((period, (group_size * number_groups)))
        frame_d = np.zeros((period, (group_size * number_groups)))
        frame_t = np.zeros((period, (group_size * number_groups)))
        frame_u = np.zeros((period, (group_size * number_groups)))
        frame_v = np.zeros((period, (group_size * number_groups)))
        frame_fitnessToT = np.zeros((period, (group_size * number_groups)))
        # create a folder to store the results

        dir_path = os.path.dirname(os.path.abspath(__file__))
        index_to_store = np.linspace(0, frame_x.shape[0] - 1, 75).astype(int)
        for i in range(1,(self.number_average_input.value()+1)):
            if self.option1_button.isChecked():
                start = time.time()
                SF.main_loop_iterated(group_size, number_groups, num_interactions, period, frame_a, frame_x, frame_d, frame_t, \
                               frame_u, frame_v, frame_fitnessToT, mu, step_size, \
                               coupled, to_migrate, transfert_multiplier, truc)
                end = time.time()
                print("Execution time: ", end - start, "for", period, "iterations.")
                if i == 1:
                    frame_x_store = frame_x[index_to_store, :]
                    frame_a_store = frame_a[index_to_store, :]
                    frame_d_store = frame_d[index_to_store, :]
                else:
                    #happend to the array
                    frame_x_store = np.hstack((frame_x_store, frame_x[index_to_store, :]))
                    frame_a_store = np.hstack((frame_a_store, frame_a[index_to_store, :]))
                    frame_d_store = np.hstack((frame_d_store, frame_d[index_to_store, :]))
        print(i)





        #index = np.linspace(0, frame_x.shape[0] - 1, 75).astype(int)
        #frame_x = frame_x[index, :]
        #frame_a = frame_a[index, :]
        #frame_d = frame_d[index, :]
        #frame_t = frame_t[index, :]
        #frame_u = frame_u[index, :]
        #frame_v = frame_v[index, :]
        #frame_fitnessToT = frame_fitnessToT[index, :]

        #find os path compatible windows/linux/mac
        #dir_path = os.path.dirname(os.path.abspath(__file__))
        np.save(os.path.join(dir_path, 'frame_a.npy'), frame_a_store)
        np.save(os.path.join(dir_path, 'frame_x.npy'), frame_x_store)
        np.save(os.path.join(dir_path, 'frame_d.npy'), frame_d_store)

        print("finished")



    def graph1(self):
        # Generate the graph and save it as an image
        GC.create_frame_x_graph_2()
        # Load the image into the QLabel
        dir_path = os.path.dirname(os.path.abspath(__file__))
        pixmap = QPixmap(os.path.join(dir_path, "frame_x.png"))
        self.graph_label.setPixmap(pixmap)
        self.graph_label.resize(pixmap.width(), pixmap.height())
        print("finished")

    def graph2(self):
        # Generate the graph and save it as an image
        GC.create_graph_pop_type_2()
        # Load the image into the QLabel
        dir_path = os.path.dirname(os.path.abspath(__file__))
        pixmap = QPixmap(os.path.join(dir_path, "frame_a.png"))
        self.graph_label.setPixmap(pixmap)
        self.graph_label.resize(pixmap.width(), pixmap.height())
        print("finished")


app = QApplication([])
window = ParameterChooser()
window.show()
app.exec()



