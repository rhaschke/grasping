#!/usr/bin/env python3

import sys
import threading
import signal
import numpy
import rclpy
from argparse import ArgumentParser
from controller import Controller, tf
from markers import iPoseMarker, addMarker, processFeedback
from interactive_markers.interactive_marker_server import InteractiveMarkerServer

from python_qt_binding.QtCore import Qt, QTimer
from python_qt_binding.QtGui import QFontDatabase
from python_qt_binding.QtWidgets import QApplication, QWidget, QSlider, QLabel, QVBoxLayout


class Slider(QSlider):
    """QSlider variant that jumps back to zero when released"""

    def __init__(self):
        super(Slider, self).__init__(Qt.Orientation.Horizontal)
        self.setRange(-1000, 1000)
        self.setValue(0)
        self.sliderReleased.connect(lambda: self.setValue(0))


class Gui(QWidget):

    def __init__(self, controller, mode):
        super(Gui, self).__init__()
        self.controller = controller
        self.mode = mode
        self.errors = QLabel()
        self.errors.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.sliders = []

        layout = QVBoxLayout()
        layout.addWidget(QLabel('errors:'))
        layout.addWidget(self.errors)
        layout.addWidget(QLabel('nullspace control:'))
        for _ in range(controller.N):
            slider = Slider()
            self.sliders.append(slider)
            layout.addWidget(slider)
        self.setLayout(layout)

    def onValueChanged(self, index, value):
        self.ns[index] = value

    def showErrors(self, tasks):
        numpy.set_printoptions(precision=3, suppress=True)
        self.errors.setText("\n".join([str(t[1]) for t in tasks]))

    def loop(self):
        c = self.controller
        ims = InteractiveMarkerServer(c.node, "controller")
        rate = c.node.create_rate(50)

        mode = self.mode
        if "pose" in mode:
            addMarker(ims, iPoseMarker(c.T), processFeedback(c.setTarget))
        ims.applyChanges()

        ns_old = numpy.zeros((c.N, 0))
        while rclpy.ok():
            if "pose" in mode:  # position + orientation control
                tasks = [c.pose_task(c.targets["pose"], c.T)]

            self.showErrors(tasks)
            q_delta = c.solve(tasks)

            # nullspace control
            N = c.nullspace.shape[1]  # dimensionality of nullspace
            ns = numpy.array([s.value() for s in self.sliders[:N]])  # slider values
            # disable inactive nullspace sliders
            for i, s in enumerate(self.sliders):
                s.setEnabled(i < N)
            # align new nullspace basis to previous one (vectors might be flipped)
            N = min(N, ns_old.shape[1])  # minimum number of columns before and now
            c.nullspace[:, :N] *= numpy.sign(numpy.sum(c.nullspace[:, :N] * ns_old[:, :N], axis=0))
            ns_old = c.nullspace
            # apply nullspace motion
            q_delta += c.nullspace.dot(1e-4 * ns)

            c.actuate(q_delta)
            try:
                rate.sleep()
            except rclpy.exceptions.ROSInterruptException:
                break


parser = ArgumentParser()
parser.add_argument(
    "mode",
    type=str,
    nargs="*",
    default="pose",
    help="control mode",
    choices=[
        "pose",
    ],
)
args = parser.parse_args()

rclpy.init()
node = rclpy.node.Node("ik")

app = QApplication(sys.argv)
app.setApplicationDisplayName("Constrained-based Control Demo")
gui = Gui(Controller(node), args.mode)
gui.show()

executor = rclpy.executors.SingleThreadedExecutor()
executor.add_node(node)
executor_thread = threading.Thread(target=executor.spin, daemon=True)
executor_thread.start()

threading.Thread(target=gui.loop).start()

# Setup signal handler for Ctrl-C to gracefully quit the application
signal.signal(signal.SIGINT, lambda sig, frame: app.quit())
# Start a timer to periodically handover execution to Python
timer = QTimer()
timer.start(200)
timer.timeout.connect(lambda: None)

# Run Qt's event loop (in C++)
app.exec()

executor.shutdown()
executor_thread.join()

node.destroy_node()
rclpy.shutdown()
