import sys
import os
import numpy as np
import Simulation_Func as SF
import Graph_Code as GC
import time
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QSpinBox, QLabel, QProgressBar,\
    QDoubleSpinBox, QCheckBox, QRadioButton, QButtonGroup, QGroupBox,QGridLayout, QHBoxLayout, \
    QSizePolicy, QTextEdit, QFileDialog

from PyQt6 import QtGui

from PyQt6.QtGui import QPixmap, QColor
from PyQt6.QtCore import Qt, QSize, QProcess,QThread,pyqtSignal, QTimer

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib import pyplot as plt


class SimulationThread(QThread):
    def __init__(self, group_size, number_groups, num_interactions, period, mu, step_size, coupled, to_migrate, transfert_multiplier, truc, to_average, tracking,x_i_value,choice):
        QThread.__init__(self)
        self.group_size = group_size
        self.number_groups = number_groups
        self.num_interactions = num_interactions
        self.period = period
        self.mu = mu
        self.step_size = step_size
        self.coupled = coupled
        self.to_migrate = to_migrate
        self.transfert_multiplier = transfert_multiplier
        self.truc = truc
        self.to_average = to_average
        self.tracking = tracking
        self.x_i_value = x_i_value
        self.choice = choice

    def run(self):
        SF.launch_sim_iterated(self.group_size, self.number_groups, self.num_interactions, self.period, self.mu, self.step_size, self.coupled, self.to_migrate, self.transfert_multiplier, self.truc, self.to_average,self.tracking,self.x_i_value,self.choice)



class ProgressThread(QThread):
    progress_signal = pyqtSignal(int, int)  # Signal with two integer parameters

    def __init__(self, tracking):
        QThread.__init__(self)
        self.tracking = tracking


    def run(self):
        while True:
            self.progress_signal.emit(int(self.tracking[0]*100), int(self.tracking[1]*100))  # Emit the signal with the current progress
            time.sleep(0.1)  # Sleep for 1 second
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

        screen = QtGui.QGuiApplication.primaryScreen()
        screen_geo = screen.availableGeometry()
        screen_width = screen_geo.width()
        screen_height = screen_geo.height()

        # Create a QHBoxLayout
        self.main_layout = QHBoxLayout()

        # Create a QVBoxLayout for the left column
        self.left_column_layout = QVBoxLayout()

        # left size

        #left size group base parameter

        self.base_parameters_group = QGroupBox("Base Parameters")
        self.base_parameters_layout = QGridLayout()

        self.group_size_label = QLabel("Group Size")
        self.group_size_input = QSpinBox()
        self.group_size_input.setRange(1, 240)
        self.group_size_input.setValue(24)  # Set default value
        self.group_size_input.setSingleStep(12)
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

        self.period_label = QLabel("Number of Periods")
        self.period_input = QSpinBox()
        self.period_input.setRange(100, 200000)
        self.period_input.setValue(1500)  # Set default value
        self.period_input.setSingleStep(10000)
        self.base_parameters_layout.addWidget(self.period_label, 1, 2)
        self.base_parameters_layout.addWidget(self.period_input, 1, 3)

        self.to_migrate_label = QLabel("Individuals to Migrate")
        self.to_migrate_input = QSpinBox()
        self.to_migrate_input.setRange(0, 240)
        self.to_migrate_input.setValue(8)  # Set default value
        self.base_parameters_layout.addWidget(self.to_migrate_label, 2, 0)
        self.base_parameters_layout.addWidget(self.to_migrate_input, 2, 1)

        self.mu_label = QLabel("Mutation rate")
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

        self.truc_label = QLabel("Fitness Weight")
        self.truc_input = QDoubleSpinBox()
        self.truc_input.setRange(0, 1.00)
        self.truc_input.setSingleStep(0.01)
        self.truc_input.setValue(0.5)  # Set default value
        self.base_parameters_layout.addWidget(self.truc_label, 3, 2)
        self.base_parameters_layout.addWidget(self.truc_input, 3, 3)

        self.transfert_multiplier_label = QLabel("Transfert Multiplier")
        self.transfert_multiplier_input = QDoubleSpinBox()
        self.transfert_multiplier_input.setRange(0, 100)
        self.transfert_multiplier_input.setSingleStep(0.1)
        self.transfert_multiplier_input.setValue(2)  # Set default value
        self.base_parameters_layout.addWidget(self.transfert_multiplier_label, 5, 0)
        self.base_parameters_layout.addWidget(self.transfert_multiplier_input, 5, 1)

        self.x_i_value_label = QLabel("Initial x_i value")
        self.x_i_value_input = QDoubleSpinBox()
        self.x_i_value_input.setRange(0, 1)
        self.x_i_value_input.setSingleStep(0.01)
        self.x_i_value_input.setValue(1)
        self.base_parameters_layout.addWidget(self.x_i_value_label, 5, 2)


        # Create a checkbox for "Coupled"
        self.coupled_checkbox = QCheckBox("Coupled")
        self.coupled_checkbox.setChecked(True)  # Set "Coupled" as the default value

        # Add the checkbox to the layout
        self.base_parameters_layout.addWidget(self.coupled_checkbox, 6, 1)




        self.number_average_input = QSpinBox()
        self.number_average_input.setRange(1, 30)
        self.number_average_input.setValue(5)
        self.number_average_input.setSingleStep(1)

        # Create a QVBoxLayout for the QGroupBox
        self.base_parameters_box_layout = QVBoxLayout()

        # Add the QGridLayout to the QVBoxLayout
        self.base_parameters_box_layout.addLayout(self.base_parameters_layout)

        # Add the QSpinBox to the QVBoxLayout
        self.base_parameters_box_layout.addWidget(self.number_average_input)

        # Set the QVBoxLayout as the layout for the QGroupBox
        self.base_parameters_group.setLayout(self.base_parameters_box_layout)

        #create a group box for intial conditions
        self.initial_conditions_group = QGroupBox("Initial Conditions")
        self.initial_conditions_layout = QVBoxLayout()

        #create 3 radio button mutually exclusive "Perfect Reciprocators", "Uncoditionnaly selfish" and "Eq degree of Escallation"

        self.perfect_reciprocators_radio = QRadioButton("Perfect Reciprocators")
        self.uncoditionnaly_selfish_radio = QRadioButton("Uncoditionnaly selfish")
        self.eq_degree_of_escallation_radio = QRadioButton("Eq degree of Escallation")

        #create a button group for the radio button
        self.initial_conditions_button_group = QButtonGroup()
        self.initial_conditions_button_group.addButton(self.perfect_reciprocators_radio)
        self.initial_conditions_button_group.addButton(self.uncoditionnaly_selfish_radio)
        self.initial_conditions_button_group.addButton(self.eq_degree_of_escallation_radio)

        #add the radio button to the layout
        self.initial_conditions_layout.addWidget(self.perfect_reciprocators_radio)
        self.initial_conditions_layout.addWidget(self.uncoditionnaly_selfish_radio)
        self.initial_conditions_layout.addWidget(self.eq_degree_of_escallation_radio)

        #by deflaut the perfect reciprocators radio button is checked
        self.perfect_reciprocators_radio.setChecked(True)

        #set the layout for the group box
        self.initial_conditions_group.setLayout(self.initial_conditions_layout)



        # create a group for the graph buttons
        self.graph_group = QGroupBox("Graphs")
        # create 2 by 2 grid layout
        self.graph_grid_layout = QGridLayout()

        #create buttons to choose between the 3 graph and "Save all"
        self.graph1_button = QPushButton("Graph 1")
        self.graph1_button.clicked.connect(self.draw_graph_1)
        self.graph_grid_layout.addWidget(self.graph1_button, 0, 0)

        self.graph2_button = QPushButton("Graph 2")
        self.graph2_button.clicked.connect(self.draw_graph_2)
        self.graph_grid_layout.addWidget(self.graph2_button, 1, 0)

        self.graph3_button = QPushButton("Graph 3")
        self.graph_grid_layout.addWidget(self.graph3_button, 0, 1)

        self.save_all_button = QPushButton("Save All")
        self.save_all_button.clicked.connect(self.save_plots)
        self.graph_grid_layout.addWidget(self.save_all_button, 1, 1)


        self.graph_group.setLayout(self.graph_grid_layout)


        #create a groupbox for Simulation
        self.simulation_group = QGroupBox("Simulation")
        # Create a 2x1 grid layout for the submit and stop buttons
        self.submit_stop_layout = QGridLayout()

        # Create a QPushButton for the submit button
        self.submit_button = QPushButton("Submit")
        self.submit_button.clicked.connect(self.submit)
        self.submit_stop_layout.addWidget(self.submit_button, 0, 0)


        self.simulation_group.setLayout(self.submit_stop_layout)

        # Add layout to the left column layout
        self.left_column_layout.addWidget(self.base_parameters_group)
        self.left_column_layout.addWidget(self.graph_group)
        self.left_column_layout.addWidget(self.initial_conditions_group)
        self.left_column_layout.addWidget(self.simulation_group)



        # Add the left column layout to the main layout
        self.main_layout.addLayout(self.left_column_layout)

        # Create a QVBoxLayout for the right column
        self.right_column_layout = QVBoxLayout()

        # create a group box for the two prgress bar
        self.progress_group = QGroupBox("Progress")
        self.progress_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        self.progress_layout = QVBoxLayout()

        # Create a progress bar for the simulation
        self.simulation_progress = QProgressBar()

        self.simulation_progress.setRange(0, 100)
        self.simulation_progress.setValue(0)

        self.progress_layout.addWidget(self.simulation_progress)

        # Create a progress bar for the graph
        self.graph_progress = QProgressBar()

        self.graph_progress.setRange(0, 100)
        self.graph_progress.setValue(0)

        self.progress_layout.addWidget(self.graph_progress)

        # Add the progress layout to the progress group
        self.progress_group.setLayout(self.progress_layout)

        # Add the progress group to the right column layout
        self.right_column_layout.addWidget(self.progress_group)

        # create a group box for the graph display
        self.graph_group = QGroupBox("Graph Display")
        self.graph_layout = QVBoxLayout()

        self.fig = plt.figure()
        self.canvas = FigureCanvas(self.fig)
        self.graph_layout.addWidget(self.canvas)

        self.graph_group.setLayout(self.graph_layout)

        self.right_column_layout.addWidget(self.graph_group)
        # Donner la priorité à la groupbox "Graph Display"
        self.right_column_layout.setStretchFactor(self.graph_group, 1)

        # Réduire la priorité de la groupbox "Progress"
        self.right_column_layout.setStretchFactor(self.progress_group, 0)
        #add the left column layout to the main layout
        self.main_layout.addLayout(self.right_column_layout)

        # Set the main layout as the layout for the window
        self.setLayout(self.main_layout)

        # Set default window size as a ratio of the screen size
        # Set window size to 80% of screen size
        window_width = int(screen_width * 0.8)
        window_height = int(screen_height * 0.8)
        self.setGeometry(0, 0, window_width, window_height)
        self.setWindowTitle('The Great Drift')




        # Create

    def draw_graph_1(self):
        # use create_graph_1 from Graph_Code.py to create the graph and display it in the canevas
        fig = GC.create_graph_1(self.period_input.value())

        # Clear the existing figure on the canvas
        self.canvas.figure.clear()

        # Add the new figure to the canvas
        self.canvas.figure = fig

        # Draw the canvas
        self.canvas.draw()

    def draw_graph_2(self):
        # use create_graph_2 from Graph_Code.py to create the graph and display it in the canevas
        fig = GC.create_graph_2(self.period_input.value())

        # Clear the existing figure on the canvas
        self.canvas.figure.clear()

        # Add the new figure to the canvas
        self.canvas.figure = fig

        # Draw the canvas
        self.canvas.draw()

    def submit(self):
        group_size = self.group_size_input.value()
        number_groups = self.number_groups_input.value()
        num_interactions = self.num_interactions_input.value()
        to_migrate = self.to_migrate_input.value()
        period = self.period_input.value()
        mu = self.mu_input.value()
        step_size = self.step_size_input.value()
        truc = self.truc_input.value()
        coupled = self.coupled_checkbox.isChecked()
        transfert_multiplier = self.transfert_multiplier_input.value()
        to_average = self.number_average_input.value()
        x_i_value = self.x_i_value_input.value()
        tracking = [0, 0]
        if self.perfect_reciprocators_radio.isChecked():
            choice = 0
        elif self.uncoditionnaly_selfish_radio.isChecked():
            choice = 1
        else:
            choice = 2

        self.simulation_thread = SimulationThread(group_size, number_groups, num_interactions, period, mu, step_size,\
                                                  coupled, to_migrate, transfert_multiplier, truc, to_average,tracking,x_i_value,choice)
        self.simulation_thread.finished.connect(self.on_simulation_finished)

        self.progress_thread = ProgressThread(self.simulation_thread.tracking)
        self.progress_thread.progress_signal.connect(self.update_progress)
        self.progress_thread.start()
        self.simulation_thread.start()

    def on_simulation_finished(self):
        # This method will be called when the simulation is finished
        print("Simulation finished")
        self.progress_thread.terminate()
        print("Progress thread terminated")
        self.update_progress(100, 100)
        #show graph 2
        self.draw_graph_2()


    def update_progress(self, simulation_progress, graph_progress):
        self.simulation_progress.setValue(simulation_progress)  # Update the simulation progress bar with the current progress
        self.graph_progress.setValue(graph_progress)

    def save_plots(self):
        GC.store_all_graphs(self.period_input.value())
        # Open a QFileDialog to choose the save directory
        save_dir = QFileDialog.getExistingDirectory(self, "Select Directory")

        # If a save directory was chosen (i.e., the user didn't cancel the dialog)
        if save_dir:
            # Save the current plots to the chosen directory
            pixmap1 = QPixmap(os.path.join(os.path.dirname(os.path.abspath(__file__)), "frame_x.png"))
            pixmap1.save(os.path.join(save_dir, "frame_x.png"))

            pixmap2 = QPixmap(os.path.join(os.path.dirname(os.path.abspath(__file__)), "frame_a.png"))
            pixmap2.save(os.path.join(save_dir, "frame_a.png"))

            print(f"Plots saved to {save_dir}")


app = QApplication([])
window = ParameterChooser()
window.show()
app.exec()




