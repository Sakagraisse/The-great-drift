import sys
import os
import numpy as np
import Simulation_Func as SF
import Graph_Code as GC
import time
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QSpinBox, QLabel, QProgressBar,\
    QDoubleSpinBox, QCheckBox, QRadioButton, QButtonGroup, QGroupBox,QGridLayout, QHBoxLayout, \
    QSizePolicy, QFileDialog

from PyQt6 import QtGui
from PyQt6.QtCore import QSize, QThread,pyqtSignal

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import shutil



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
        #copy submitted parameters to a file named "last_simulation_parameters.npy"
        submitted_parameters = {"group_size": self.group_size, "number_groups": self.number_groups, "num_interactions": self.num_interactions, "period": self.period, "mu": self.mu, "step_size": self.step_size, "coupled": self.coupled, "to_migrate": self.to_migrate, "transfert_multiplier": self.transfert_multiplier, "truc": self.truc, "to_average": self.to_average, "tracking": self.tracking, "x_i_value": self.x_i_value, "choice": self.choice}
        # Obtenir le chemin du répertoire du script actuel
        dir_path = os.path.dirname(os.path.abspath(__file__))

        # Créer le chemin complet du fichier
        file_path = os.path.join(dir_path, 'last_simulation_parameters.npy')

        # Enregistrer le fichier
        np.save(file_path, submitted_parameters)
        print("Simulation finished")


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
        #si sur mac test est 0.1 si sur windows test est 0.15  
        if sys.platform == "darwin":
            test = int(self.base_parameters_group.width() * 0.1)
        elif sys.platform == "win32":
            test = int(self.base_parameters_group.width() * 0.15)
        self.base_parameters_layout = QGridLayout()

        self.group_size_label = QLabel("Group Size")
        self.group_size_input = QSpinBox()
        self.group_size_input.setRange(0, 240)
        self.group_size_input.setValue(24)  # Set default value
        self.group_size_input.setMinimumWidth(test)
        self.group_size_input.setSingleStep(12)

        self.group_size_input.setMinimumWidth(test)
        self.base_parameters_layout.addWidget(self.group_size_label, 0, 0)
        self.base_parameters_layout.addWidget(self.group_size_input, 0, 1)

        self.number_groups_label = QLabel("Number of Groups")
        self.number_groups_input = QSpinBox()
        self.number_groups_input.setRange(0, 100)
        self.number_groups_input.setSingleStep(10)
        self.number_groups_input.setValue(40)  # Set default value
        self.number_groups_input.setMinimumWidth(test)
        self.base_parameters_layout.addWidget(self.number_groups_label, 1, 0)
        self.base_parameters_layout.addWidget(self.number_groups_input, 1, 1)

        self.num_interactions_label = QLabel("Number of Interactions")
        self.num_interactions_input = QSpinBox()
        self.num_interactions_input.setRange(1, 1000)
        self.num_interactions_input.setValue(100)  # Set default value
        self.num_interactions_input.setMinimumWidth(test)
        self.base_parameters_layout.addWidget(self.num_interactions_label, 0, 2)
        self.base_parameters_layout.addWidget(self.num_interactions_input, 0, 3)

        self.period_label = QLabel("Number of Periods")
        self.period_input = QSpinBox()
        self.period_input.setRange(100, 200000)
        self.period_input.setValue(3000)  # Set default value
        self.period_input.setSingleStep(500)
        self.period_input.setMinimumWidth(test)
        self.base_parameters_layout.addWidget(self.period_label, 1, 2)
        self.base_parameters_layout.addWidget(self.period_input, 1, 3)

        self.to_migrate_label = QLabel("Individuals to Migrate")
        self.to_migrate_input = QSpinBox()
        self.to_migrate_input.setRange(0, 240)
        self.to_migrate_input.setValue(8)  # Set default value
        self.to_migrate_input.setSingleStep(1)
        self.to_migrate_input.setMinimumWidth(test)
        self.base_parameters_layout.addWidget(self.to_migrate_label, 2, 0)
        self.base_parameters_layout.addWidget(self.to_migrate_input, 2, 1)

        self.mu_label = QLabel("Mutation rate in %")
        self.mu_input = QDoubleSpinBox()
        self.mu_input.setRange(0, 100)
        self.mu_input.setSingleStep(0.5)
        self.mu_input.setValue(2)  # Set default value
        self.mu_input.setMinimumWidth(test)
        self.base_parameters_layout.addWidget(self.mu_label, 3, 0)
        self.base_parameters_layout.addWidget(self.mu_input, 3, 1)

        self.step_size_label = QLabel("Step Size x 100")
        self.step_size_input = QDoubleSpinBox()
        self.step_size_input.setRange(0, 100)
        self.step_size_input.setSingleStep(0.5)
        self.step_size_input.setValue(2.5)  # Set default value
        self.step_size_input.setMinimumWidth(test)
        self.base_parameters_layout.addWidget(self.step_size_label, 2, 2)
        self.base_parameters_layout.addWidget(self.step_size_input, 2, 3)

        self.truc_label = QLabel("Fitness Weight")
        self.truc_input = QDoubleSpinBox()
        self.truc_input.setRange(0, 1.00)
        self.truc_input.setSingleStep(0.01)
        self.truc_input.setValue(0.5)  # Set default value
        self.truc_input.setMinimumWidth(test)
        self.base_parameters_layout.addWidget(self.truc_label, 3, 2)
        self.base_parameters_layout.addWidget(self.truc_input, 3, 3)

        self.transfert_multiplier_label = QLabel("Transfert Multiplier")
        self.transfert_multiplier_input = QDoubleSpinBox()
        self.transfert_multiplier_input.setRange(0, 100)
        self.transfert_multiplier_input.setSingleStep(0.1)
        self.transfert_multiplier_input.setValue(2)  # Set default value
        self.transfert_multiplier_input.setMinimumWidth(test)
        self.base_parameters_layout.addWidget(self.transfert_multiplier_label, 5, 0)
        self.base_parameters_layout.addWidget(self.transfert_multiplier_input, 5, 1)

        self.x_i_value_label = QLabel("Initial x_i value")
        self.x_i_value_input = QDoubleSpinBox()
        self.x_i_value_input.setRange(0, 1)
        self.x_i_value_input.setSingleStep(0.01)
        self.x_i_value_input.setValue(1)
        self.x_i_value_input.setMinimumWidth(test)
        self.base_parameters_layout.addWidget(self.x_i_value_label, 5, 2)
        self.base_parameters_layout.addWidget(self.x_i_value_input, 5, 3)


        # Create a checkbox for "Coupled"
        self.coupled_checkbox = QCheckBox("Coupled")
        self.coupled_checkbox.setChecked(True)  # Set "Coupled" as the default value

        # Add the checkbox to the layout
        self.base_parameters_layout.addWidget(self.coupled_checkbox, 6, 1)

        self.number_average_label = QLabel("Number of Simulations to Average")
        self.number_average_input = QSpinBox()
        self.number_average_input.setRange(1, 30)
        self.number_average_input.setValue(5)
        self.number_average_input.setSingleStep(1)

        # Create a QVBoxLayout for the QGroupBox
        self.base_parameters_box_layout = QVBoxLayout()

        # Add the QGridLayout to the QVBoxLayout
        self.base_parameters_box_layout.addLayout(self.base_parameters_layout)

        # Add the QSpinBox to the QVBoxLayout
        self.base_parameters_box_layout.addWidget(self.number_average_label)
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
        self.eq_degree_of_escallation_radio.clicked.connect(self.check_x_i_value)

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
        self.graph1_button = QPushButton("Evolution of First Move")
        self.graph1_button.clicked.connect(self.draw_graph_1)
        self.graph_grid_layout.addWidget(self.graph1_button, 0, 0)

        self.graph2_button = QPushButton("Evolution of Strategies")
        self.graph2_button.clicked.connect(self.draw_graph_2)
        self.graph_grid_layout.addWidget(self.graph2_button, 0, 1)

        self.graph3_button = QPushButton("Surplus per generation")
        self.graph3_button.clicked.connect(self.draw_graph_3)
        self.graph_grid_layout.addWidget(self.graph3_button, 1, 0)

        self.save_all_button = QPushButton("Save All + parameters")
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

        self.canvas = FigureCanvas()
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
        window_width = int(screen_width * 0.9)
        window_height = int(screen_height * 0.8)
        self.setGeometry(0, 0, window_width, window_height)
        self.setWindowTitle('The Great Drift')





        # Create
    def check_x_i_value(self):
        if self.x_i_value_input.value() == 1:
            self.x_i_value_input.setValue(0.95)
        else :
            pass
    def draw_graph_1(self):

        # Définir le nom du fichier
        file_name = 'last_simulation_parameters.npy'

        # Obtenir le chemin du répertoire du script actuel
        dir_path = os.path.dirname(os.path.abspath(__file__))

        # Créer le chemin complet du fichier
        file_path = os.path.join(dir_path, file_name)

        if not os.path.exists(file_path):
            # Le fichier n'existe pas
            return
            # Get the size of the canvas in pixels

        period = np.load(file_path, allow_pickle=True).item()["period"]
        canvas_size = self.canvas.size()

        # Convert the size to inches
        canvas_size_inches = canvas_size.width() / self.canvas.figure.dpi, canvas_size.height() / self.canvas.figure.dpi

        # Use create_graph_1 from Graph_Code.py to create the graph with the size of the canvas
        self.fig1 = GC.create_graph_1(period, figsize=canvas_size_inches)

        # Remove the existing canvas from the layout
        self.graph_layout.removeWidget(self.canvas)

        # Delete the existing canvas
        self.canvas.deleteLater()

        # Create a new canvas with the new figure
        self.canvas = FigureCanvas(self.fig1)

        # Resize the figure to fit the canvas
        self.canvas.figure.set_size_inches(canvas_size_inches)

        # Add the new canvas to the layout
        self.graph_layout.addWidget(self.canvas)

        # Draw the canvas
        self.canvas.draw()


    def draw_graph_2(self):

        # Définir le nom du fichier
        file_name = 'last_simulation_parameters.npy'

        # Obtenir le chemin du répertoire du script actuel
        dir_path = os.path.dirname(os.path.abspath(__file__))

        # Créer le chemin complet du fichier
        file_path = os.path.join(dir_path, file_name)

        if not os.path.exists(file_path):
            # Le fichier n'existe pas
            return
        # Get the size of the canvas in pixels
        canvas_size = self.canvas.size()

        # Convert the size to inches
        canvas_size_inches = canvas_size.width() / self.canvas.figure.dpi, canvas_size.height() / self.canvas.figure.dpi

        # retrieve last simulation parameters
        last_simulation_parameters = np.load(file_path, allow_pickle=True).item()
        period = last_simulation_parameters["period"]

        # Use create_graph_2 from Graph_Code.py to create the graph with the size of the canvas
        self.fig2 = GC.create_graph_2(period, figsize=canvas_size_inches)

        # Remove the existing canvas from the layout
        self.graph_layout.removeWidget(self.canvas)

        # Delete the existing canvas
        self.canvas.deleteLater()

        # Create a new canvas with the new figure
        self.canvas = FigureCanvas(self.fig2)

        # Resize the figure to fit the canvas
        self.canvas.figure.set_size_inches(canvas_size_inches)

        # Add the new canvas to the layout
        self.graph_layout.addWidget(self.canvas)

        # Draw the canvas
        self.canvas.draw()

    def draw_graph_3(self):
        # Définir le nom du fichier
        file_name = 'last_simulation_parameters.npy'

        # Obtenir le chemin du répertoire du script actuel
        dir_path = os.path.dirname(os.path.abspath(__file__))

        # Créer le chemin complet du fichier
        file_path = os.path.join(dir_path, file_name)

        if not os.path.exists(file_path):
            # Le fichier n'existe pas
            return
            # Get the size of the canvas in pixels
        canvas_size = self.canvas.size()

        # Convert the size to inches
        canvas_size_inches = canvas_size.width() / self.canvas.figure.dpi, canvas_size.height() / self.canvas.figure.dpi

        # retrieve last simulation parameters
        last_simulation_parameters = np.load(file_path, allow_pickle=True).item()
        period = last_simulation_parameters["period"]

        # Use create_graph_3 from Graph_Code.py to create the graph with the size of the canvas
        self.fig3 = GC.create_graph_3(period, figsize=canvas_size_inches)

        # Remove the existing canvas from the layout
        self.graph_layout.removeWidget(self.canvas)

        # Delete the existing canvas
        self.canvas.deleteLater()

        # Create a new canvas with the new figure
        self.canvas = FigureCanvas(self.fig3)

        # Resize the figure to fit the canvas
        self.canvas.figure.set_size_inches(canvas_size_inches)

        # Add the new canvas to the layout
        self.graph_layout.addWidget(self.canvas)

        # Draw the canvas
        self.canvas.draw()

    def submit(self):
        group_size = self.group_size_input.value()
        number_groups = self.number_groups_input.value()
        num_interactions = self.num_interactions_input.value()
        to_migrate = self.to_migrate_input.value()
        period = self.period_input.value()
        mu = self.mu_input.value()/100
        step_size = self.step_size_input.value()/100
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

        #store all parameter on a dictionary and then store it in a file named, submitted parameters
        #find path to store the file

        submitted_parameters = {"group_size": group_size, "number_groups": number_groups, "num_interactions": num_interactions, "to_migrate": to_migrate, "period": period, "mu": mu, "step_size": step_size, "truc": truc, "coupled": coupled, "transfert_multiplier": transfert_multiplier, "to_average": to_average, "tracking": tracking, "x_i_value": x_i_value, "choice": choice}
        dir_path = os.path.dirname(os.path.abspath(__file__))

        # Créer le chemin complet du fichier
        file_path = os.path.join(dir_path, 'submitted_parameters.npy')

        # Enregistrer le fichier
        np.save(file_path, submitted_parameters)

        if not hasattr(self, 'simulation_thread') or not self.simulation_thread.isRunning():
            self.simulation_thread = SimulationThread(group_size, number_groups, num_interactions, period, mu, step_size,\
                                                      coupled, to_migrate, transfert_multiplier, truc, to_average,tracking,x_i_value,choice)
            self.simulation_thread.finished.connect(self.on_simulation_finished)

            self.progress_thread = ProgressThread(self.simulation_thread.tracking)
            self.progress_thread.progress_signal.connect(self.update_progress)
            self.progress_thread.start()
            self.simulation_thread.start()
        else:
            print("SimulationThread is already running")

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
        # Définir le nom du fichier
        file_name = 'last_simulation_parameters.npy'

        # Obtenir le chemin du répertoire du script actuel
        dir_path = os.path.dirname(os.path.abspath(__file__))

        # Créer le chemin complet du fichier
        file_path = os.path.join(dir_path, file_name)

        if not os.path.exists(file_path):
            # Le fichier n'existe pas
            return

        # retrieve last simulation parameters
        last_simulation_parameters = np.load(file_path, allow_pickle=True).item()
        period = last_simulation_parameters["period"]

        GC.store_all_graphs(period)
        # Open a QFileDialog to choose the save directory
        save_dir = QFileDialog.getExistingDirectory(self, "Select Directory")

        # If a save directory was chosen (i.e., the user didn't cancel the dialog)
        if save_dir:
            # Save the current plots to the chosen directory

            # Définir le chemin d'origine et de destination
            src_file1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Evolution_of_First_Move.pdf")
            dest_file1 = os.path.join(save_dir, "Evolution_of_First_Move.pdf")

            src_file2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Evolution_of_Strategies.pdf")
            dest_file2 = os.path.join(save_dir, "Evolution_of_Strategies.pdf")

            src_file3 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Surplus_per_generation.pdf")
            dest_file3 = os.path.join(save_dir, "Surplus_per_generation.pdf")

            # Copier les fichiers
            shutil.copy(src_file1, dest_file1)
            shutil.copy(src_file2, dest_file2)
            shutil.copy(src_file3, dest_file3)

            #save the dictionary of submitted parameters in a txt file
            with open(os.path.join(save_dir, "last_simulation_parameters.txt"), "w") as file:
                file.write(str(np.load(file_path, allow_pickle=True).item()))

            print(f"Plots saved to {save_dir}")


app = QApplication([])
window = ParameterChooser()
window.show()
app.exec()





