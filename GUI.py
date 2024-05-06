import numpy as np
import Simulation_Func as SF
import Graph_Code as GC
import time
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QSpinBox, QLabel, QDoubleSpinBox

class ParameterChooser(QWidget):
    def __init__(self):
        super().__init__()

        self.layout = QVBoxLayout()

        self.group_size_label = QLabel("Group Size")
        self.group_size_input = QSpinBox()
        self.group_size_input.setRange(1, 100)
        self.group_size_input.setValue(24)  # Set default value
        self.layout.addWidget(self.group_size_label)
        self.layout.addWidget(self.group_size_input)

        self.number_groups_label = QLabel("Number of Groups")
        self.number_groups_input = QSpinBox()
        self.number_groups_input.setRange(1, 100)
        self.number_groups_input.setValue(40)  # Set default value
        self.layout.addWidget(self.number_groups_label)
        self.layout.addWidget(self.number_groups_input)

        self.num_interactions_label = QLabel("Number of Interactions")
        self.num_interactions_input = QSpinBox()
        self.num_interactions_input.setRange(1, 100)
        self.num_interactions_input.setValue(10)  # Set default value
        self.layout.addWidget(self.num_interactions_label)
        self.layout.addWidget(self.num_interactions_input)

        self.to_migrate_label = QLabel("To Migrate")
        self.to_migrate_input = QSpinBox()
        self.to_migrate_input.setRange(1, 100)
        self.to_migrate_input.setValue(16)  # Set default value
        self.layout.addWidget(self.to_migrate_label)
        self.layout.addWidget(self.to_migrate_input)

        self.period_label = QLabel("Period")
        self.period_input = QSpinBox()
        self.period_input.setRange(1, 100000)
        self.period_input.setValue(50000)  # Set default value
        self.layout.addWidget(self.period_label)
        self.layout.addWidget(self.period_input)

        self.mu_label = QLabel("Mu")
        self.mu_input = QDoubleSpinBox()
        self.mu_input.setRange(0.01, 1.00)
        self.mu_input.setSingleStep(0.01)
        self.mu_input.setValue(0.02)  # Set default value
        self.layout.addWidget(self.mu_label)
        self.layout.addWidget(self.mu_input)

        self.step_size_label = QLabel("Step Size")
        self.step_size_input = QDoubleSpinBox()
        self.step_size_input.setRange(0.010, 1.000)
        self.step_size_input.setSingleStep(0.001)
        self.step_size_input.setValue(0.025)  # Set default value
        self.layout.addWidget(self.step_size_label)
        self.layout.addWidget(self.step_size_input)

        self.truc_label = QLabel("Truc")
        self.truc_input = QDoubleSpinBox()
        self.truc_input.setRange(0.01, 1.00)
        self.truc_input.setSingleStep(0.01)
        self.truc_input.setValue(0.5)  # Set default value
        self.layout.addWidget(self.truc_label)
        self.layout.addWidget(self.truc_input)

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




