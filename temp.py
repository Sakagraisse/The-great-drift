def save_plot1(self):
    # Open a QFileDialog to choose the save location
    save_path, _ = QFileDialog.getSaveFileName(self, "Save Plot 1", "", "PNG Files (*.png);;All Files (*)")

    # If a save location was chosen (i.e., the user didn't cancel the dialog)
    if save_path:
        # Save the current plot to the chosen location
        # GC.create_frame_x_graph_2(self.period_input.value())
        pixmap = QPixmap(os.path.join(os.path.dirname(os.path.abspath(__file__)), "frame_x.png"))
        pixmap.save(save_path)

        print(f"Plot 1 saved to {save_path}")


def save_plot2(self):
    # Open a QFileDialog to choose the save location
    save_path, _ = QFileDialog.getSaveFileName(self, "Save Plot 2", "", "PNG Files (*.png);;All Files (*)")

    # If a save location was chosen (i.e., the user didn't cancel the dialog)
    if save_path:
        # Save the current plot to the chosen location
        # GC.create_graph_pop_type_2()
        pixmap = QPixmap(os.path.join(os.path.dirname(os.path.abspath(__file__)), "frame_a.png"))
        pixmap.save(save_path)

        print(f"Plot 2 saved to {save_path}")


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


    def graph1(self):
        # Generate the graph and save it as an image
        #GC.create_frame_x_graph_2(self.period_input.value())
        # Load the image into the QLabel
        dir_path = os.path.dirname(os.path.abspath(__file__))
        pixmap = QPixmap(os.path.join(dir_path, "frame_x.png"))
        self.graph_label.setPixmap(pixmap)
        self.graph_label.resize(pixmap.width(), pixmap.height())
        print("finished")

    def graph2(self):
        # Generate the graph and save it as an image
        #GC.create_graph_pop_type_2()
        # Load the image into the QLabel
        dir_path = os.path.dirname(os.path.abspath(__file__))
        pixmap = QPixmap(os.path.join(dir_path, "frame_a.png"))
        self.graph_label.setPixmap(pixmap)
        self.graph_label.resize(pixmap.width(), pixmap.height())
        print("finished")