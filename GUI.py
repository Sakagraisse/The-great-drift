import numpy as np
import Simulation_Func as SF
import Graph_Code as GC
import time
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QSpinBox, QLabel, \
    QDoubleSpinBox, QCheckBox, QRadioButton, QButtonGroup, QGroupBox,QGridLayout

class ParameterChooser(QWidget):
    def __init__(self):
        super().__init__()

        self.layout = QVBoxLayout()

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
        self.period_input.setRange(1, 200000)
        self.period_input.setValue(150000)  # Set default value
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
        self.lambda_param_input.setRange(0, 100)
        self.lambda_param_input.setSingleStep(1)
        self.lambda_param_input.setValue(10)  # Set default value
        self.base_parameters_layout.addWidget(self.lambda_param_label, 4, 2)
        self.base_parameters_layout.addWidget(self.lambda_param_input, 4, 3)

        self.theta_label = QLabel("Theta")
        self.theta_input = QDoubleSpinBox()
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
        self.layout.addWidget(self.base_parameters_group)

        # Create a QGroupBox for Simulation Type
        self.simulation_type_group = QGroupBox("Simulation Type")
        self.simulation_type_layout = QVBoxLayout()

        self.option1_button = QRadioButton("Repeated Interactions")
        self.option1_button.setChecked(True)
        self.option1_button.toggled.connect(self.update_theta_lambda)
        self.option2_button = QRadioButton("Group Competition")
        self.option2_button.toggled.connect(self.update_num_interactions)
        self.option3_button = QRadioButton("Joint Scenario")

        self.button_group = QButtonGroup()
        self.button_group.addButton(self.option1_button)
        self.button_group.addButton(self.option2_button)
        self.button_group.addButton(self.option3_button)

        self.simulation_type_layout.addWidget(self.option1_button)
        self.simulation_type_layout.addWidget(self.option2_button)
        self.simulation_type_layout.addWidget(self.option3_button)

        self.simulation_type_group.setLayout(self.simulation_type_layout)
        self.layout.addWidget(self.simulation_type_group)

        self.submit_button = QPushButton("Submit")
        self.submit_button.clicked.connect(self.submit)
        self.layout.addWidget(self.submit_button)

        self.submit_button = QPushButton("Graph 1")
        self.submit_button.clicked.connect(self.graph1)
        self.layout.addWidget(self.submit_button)

        self.submit_button = QPushButton("Graph 2")
        self.submit_button.clicked.connect(self.graph2)
        self.layout.addWidget(self.submit_button)


        self.setLayout(self.layout)


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

    def submit(self):
        group_size = self.group_size_input.value()
        number_groups = self.number_groups_input.value()
        num_interactions = self.num_interactions_input.value()
        to_migrate = self.to_migrate_input.value()
        period = self.period_input.value()
        mu = self.mu_input.value()
        step_size = self.step_size_input.value()
        truc = self.truc_input.value()

        print(f"Group Size: {group_size}")
        print(f"Number of Groups: {number_groups}")
        print(f"Number of Interactions: {num_interactions}")
        print(f"To Migrate: {to_migrate}")
        print(f"Period: {period}")
        print(f"Mu: {mu}")
        print(f"Step Size: {step_size}")
        print(f"Truc: {truc}")

        frame_a = np.empty((period, (group_size * number_groups)))
        frame_x = np.empty((period, (group_size * number_groups)))
        frame_d = np.empty((period, (group_size * number_groups)))

        x_i, d_i, a_i, fitness = SF.main_loop(period, 2, frame_a, frame_x, frame_d, mu, step_size, to_migrate, truc,
                                              group_size, number_groups, num_interactions)

        np.save('frame_a.npy', frame_a)
        np.save('frame_x.npy', frame_x)
        np.save('frame_d.npy', frame_d)

        print("finished")

    def graph1(self):
        GC.create_frame_x_graph_2()
        print("finished")

    def graph2(self):
        GC.create_graph_pop_type_2()
        print("finished")


app = QApplication([])
window = ParameterChooser()
window.show()
app.exec_()




