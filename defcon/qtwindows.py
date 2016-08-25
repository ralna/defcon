# Try to use the seaborn palette for matplotlib, but fail gracefully if it isn't installed. 
try: 
    import seaborn as sns
    blue, green, red, purple, yellow, cyan = sns.color_palette()
except ImportError: 
    blue, green, red, purple, yellow, cyan = 'blue', 'green', 'red', 'purple', 'yellow', 'cyan'

import matplotlib
matplotlib.use("Qt4Agg")

# QT compatibility for matplotlib.
from matplotlib.backends import qt_compat
use_pyside = qt_compat.QT_API == qt_compat.QT_API_PYSIDE
if use_pyside: from PySide import QtGui, QtCore
else: from PyQt4 import QtGui, QtCore

# Styles for matplotlib.
# See matpoltlib.styles.available for options.
try:
    from matplotlib import style
    style.use('ggplot')
except AttributeError:
    print "\033[91m[Warning] Update to the latest version of matplotlib to use styles.\033[00m\n"
    pass

# For plotting the bifurcation diagram.
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT
from matplotlib.figure import Figure

from datetime import timedelta
import os

# Colours.


MAIN = 'black' # colour for regular points
DEF = blue # colour for points found via deflation
HIGHLIGHT = red # colour for selected points
GRID = 'white' # colour the for grid.
BORDER = 'black' # borders on the UI
SWEEP = red # colour for the sweep line

# Markers and various other styles.
CONTPLOT = '.' # marker to use for points found by continuation
DEFPLOT = 'o' # marker to use for points found by deflation
SWEEPSTYLE = 'dashed' # line style for the sweep line

# Set up the figure.
figure = Figure(figsize=(7,6), dpi=100)
bfdiag = figure.add_subplot(111)
bfdiag.grid(color=GRID)

def rgb2hex(col):
    """ Utility function for converting RGB tuples to hex strings. """
    if isinstance(col, str): return col # if seaborn failed to import, we don't need to convert the colours. 
    else: return '#%02x%02x%02x' % tuple([int(c*255) for c in col])

def teamtext(job):
    """ Utility function for converting a team's job into a colour and a label. """
    if job == "d": colour, label = rgb2hex(blue), 'Deflating'
    if job == "c": colour, label = rgb2hex(green), 'Continuing'
    if job == "i": colour, label = rgb2hex(yellow), 'Idle'
    if job == "q": colour, label = rgb2hex(red), 'Quit'
    return colour, label  

################################
### Custom matplotlib Figure ###
################################
class DynamicCanvas(FigureCanvas):
    """A canvas that updates itself with a new plot."""
    def __init__(self, pc, update_interval, parent=None, width=5, height=4, dpi=100):
        FigureCanvas.__init__(self, figure)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_figure)
        self.timer.start(update_interval)
        self.pc = pc

    def update_figure(self):
        self.timer.stop()
        redraw = self.pc.update() # grab new points.
        if redraw: self.draw() # if we've found something new, redraw the diagram.
        self.pc.seen() # tell the PlotConstructor that we've seen all the new points.
        self.timer.start()

class CustomToolbar(NavigationToolbar2QT):
    """ A custom matplotlib toolbar, so we can remove those pesky extra buttons. """  
    def __init__(self, canvas, parent, pc, resources_dir, working_dir):
        self.toolitems = (
            ('Home', 'Reset original view', 'home', 'home'),
            ('Pan', 'Pan axes with left mouse, zoom with right', 'move', 'pan'),
            ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom'),
            ('Save', 'Save the figure', 'filesave', 'save_figure'),
            )
        NavigationToolbar2QT.__init__(self, canvas, parent)
        self.layout().takeAt(4)

        self.parent = parent
        self.pc = pc
        self.working_dir = working_dir

        # Add new buttons for saving movies and saving to tikz. 
        self.buttonSaveMovie = self.addAction(QtGui.QIcon(resources_dir + "save_movie.png"), "Save Movie", self.save_movie)
        self.buttonSaveMovie.setToolTip("Save the figure as an animation")

        self.buttonSaveTikz= self.addAction(QtGui.QIcon(resources_dir + "save_tikz.png"), "Save Tikz", self.save_tikz)
        self.buttonSaveTikz.setToolTip("Save the figure as tikz")

    def save_movie(self):
        """ A method that saves an animation of the bifurcation diagram. """
        start = self.working_dir + os.path.sep + "bfdiag.mp4" # default name of the file. 
        filters = "FFMPEG Video (*.mp4)" # what kinds of file extension we allow.
        selectedFilter = filters

        # Ask for some input parameters.
        inputter = MovieDialog(self.parent)
        inputter.exec_()
        length = inputter.length.text()
        fps = inputter.fps.text()

        fname = QtGui.QFileDialog.getSaveFileName(self, "Choose a filename to save to", start, filters, selectedFilter)
        if fname:
            try:
                self.pc.save_movie(str(fname), int(length), int(fps))
            # Handle any exceptions by printing a dialogue box. 
            except Exception, e:
                QtGui.QMessageBox.critical(self, "Error saving file", str(e), QtGui.QMessageBox.Ok, QtGui.QMessageBox.NoButton)

    def save_tikz(self):
        """ A method that saves a .tikz of the bifurcation diagram. """
        start = self.working_dir + os.path.sep + "bfdiag.tex"
        filters = "Tikz Image (*.tex)"
        selectedFilter = filters
 
        fname = QtGui.QFileDialog.getSaveFileName(self, "Choose a filename to save to", start, filters, selectedFilter)
        if fname:
            try:
                self.pc.save_tikz(str(fname))
            except Exception, e:
                QtGui.QMessageBox.critical(self, "Error saving file", str(e), QtGui.QMessageBox.Ok, QtGui.QMessageBox.NoButton)

      


############################
### MOVIE INPUT DIALOGUE ###
############################
class MovieDialog(QtGui.QDialog):
    def __init__(self, parent=None):

        QtGui.QWidget.__init__(self, parent)

        # Layout
        mainLayout = QtGui.QVBoxLayout()

        lengthLayout = QtGui.QHBoxLayout()
        self.label = QtGui.QLabel()
        self.label.setText("Desired length of movie in seconds")
        lengthLayout.addWidget(self.label)

        self.length = QtGui.QLineEdit("60")
        self.length.setFixedWidth(80)
        #inputValidator = QtGui.QIntValidator(self)
        #inputValidator.setRange(1, sys.maxint)
        #self.length.setValidator(inputValidator)
        lengthLayout.addWidget(self.length)

        mainLayout.addLayout(lengthLayout)

        fpsLayout = QtGui.QHBoxLayout()
        self.label2 = QtGui.QLabel()
        self.label2.setText("Frames per second")
        fpsLayout.addWidget(self.label2)

        self.fps = QtGui.QLineEdit("24")
        self.fps.setFixedWidth(80)
        #self.fps.setValidator(inputValidator)
        fpsLayout.addWidget(self.fps)

        mainLayout.addLayout(fpsLayout)

        # The Button
        layout = QtGui.QHBoxLayout()
        button = QtGui.QPushButton("Enter")
        button.setFixedWidth(80)
        self.connect(button, QtCore.SIGNAL("clicked()"), self.close)
        layout.addWidget(button)

        mainLayout.addLayout(layout)
        self.setLayout(mainLayout)

        self.resize(400, 60)
        self.setWindowTitle("Movie parameters")


######################
### Main QT Window ###
######################
class ApplicationWindow(QtGui.QMainWindow):
    def __init__(self, pc, update_interval, resources_dir, working_dir):
        QtGui.QMainWindow.__init__(self)     
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        self.pc = pc

        # Use these to add a toolbar, if desired. 
        #self.file_menu = QtGui.QMenu('&File', self)
        #self.file_menu.addAction('&Quit', self.fileQuit, QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        #self.menuBar().addMenu(self.file_menu)
        #self.help_menu = QtGui.QMenu('&Help', self)
        #self.menuBar().addSeparator()
        #self.menuBar().addMenu(self.help_menu)
        #self.help_menu.addAction('&About', self.about)

        # Main widget
        self.main_widget = QtGui.QWidget(self)
        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

        # Keep track of the current time and maxtime.
        self.time = 0
        self.maxtime = 0


        # Layout
        main_layout = QtGui.QHBoxLayout(self.main_widget)
        lVBox = QtGui.QVBoxLayout()
        rVBox = QtGui.QVBoxLayout()
        rVBox.setAlignment(QtCore.Qt.AlignTop)
        main_layout.addLayout(lVBox)
        main_layout.addLayout(rVBox)

        canvasBox = QtGui.QVBoxLayout()
        lVBox.addLayout(canvasBox)
        timeBox = QtGui.QHBoxLayout()
        timeBox.setAlignment(QtCore.Qt.AlignCenter)
        lVBox.addLayout(timeBox)
        lowerBox = QtGui.QHBoxLayout()
        lowerBox.setAlignment(QtCore.Qt.AlignLeft)
        lVBox.addLayout(lowerBox)

        self.functionalBox = QtGui.QVBoxLayout()
        rVBox.addLayout(self.functionalBox)
        infoBox = QtGui.QVBoxLayout()
        infoBox.setContentsMargins(0, 10, 0, 10)
        rVBox.addLayout(infoBox)
        plotBox = QtGui.QHBoxLayout()
        rVBox.addLayout(plotBox)
        plotBox.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignCenter)
        teamBox = QtGui.QVBoxLayout()
        teamBox.setContentsMargins(0, 10, 0, 10)
        rVBox.addLayout(teamBox)


        # Canvas.
        self.dc = DynamicCanvas(self.pc, update_interval, parent=self.main_widget, width=5, height=4, dpi=100)
        canvasBox.addWidget(self.dc)
        self.dc.mpl_connect('button_press_event', self.clicked_diagram)


        # Toolbar, with save_movie and save_tikz buttons.
        toolbar = CustomToolbar(self.dc, self, self.pc, resources_dir, working_dir)
        toolbar.update()
        canvasBox.addWidget(toolbar)


        # Time navigation buttons
        self.buttonStart = QtGui.QPushButton()
        self.buttonStart.setIcon(QtGui.QIcon(resources_dir+'start.png'))
        self.buttonStart.setIconSize(QtCore.QSize(18,18))
        self.buttonStart.clicked.connect(lambda:self.start())
        self.buttonStart.setFixedWidth(30)
        self.buttonStart.setToolTip("Start")
        timeBox.addWidget(self.buttonStart)

        self.buttonBack = QtGui.QPushButton()
        self.buttonBack.setIcon(QtGui.QIcon(resources_dir+'back.png'))
        self.buttonBack.setIconSize(QtCore.QSize(18,18))
        self.buttonBack.clicked.connect(lambda:self.back())
        self.buttonBack.setFixedWidth(30)
        self.buttonBack.setToolTip("Back")
        timeBox.addWidget(self.buttonBack)

        self.jumpInput = QtGui.QLineEdit()
        self.jumpInput.setText(str(self.time))
        self.jumpInput.setFixedWidth(40)
        self.inputValidator = QtGui.QIntValidator(self)
        self.inputValidator.setRange(0, self.maxtime)
        self.jumpInput.setValidator(self.inputValidator)
        self.jumpInput.returnPressed.connect(self.jump)
        timeBox.addWidget(self.jumpInput)

        self.buttonForward = QtGui.QPushButton()
        self.buttonForward.setIcon(QtGui.QIcon(resources_dir+'forward.png'))
        self.buttonForward.setIconSize(QtCore.QSize(18,18))
        self.buttonForward.clicked.connect(lambda:self.forward())
        self.buttonForward.setToolTip("Forward")
        self.buttonForward.setFixedWidth(30)
        timeBox.addWidget(self.buttonForward)

        self.buttonEnd = QtGui.QPushButton()
        self.buttonEnd.setIcon(QtGui.QIcon(resources_dir+'end.png'))
        self.buttonEnd.setIconSize(QtCore.QSize(18,18))
        self.buttonEnd.clicked.connect(lambda:self.end())
        self.buttonEnd.setToolTip("End")
        self.buttonEnd.setFixedWidth(30)
        timeBox.addWidget(self.buttonEnd)


        # Plot Buttons
        self.buttonPlot = QtGui.QPushButton("Plot")
        self.buttonPlot.clicked.connect(lambda:self.plot())
        self.buttonPlot.setEnabled(False)
        self.buttonPlot.setToolTip("Plot currently selected solution")
        self.buttonPlot.setFixedWidth(80)
        plotBox.addWidget(self.buttonPlot)

        # Radio buttons
        label = QtGui.QLabel("Functionals:")
        label.setFixedHeight(20)
        self.functionalBox.addWidget(label)
        self.radio_buttons = []


        # Output Box
        self.infobox = QtGui.QLabel("")
        self.infobox.setFixedHeight(250)
        self.infobox.setFixedWidth(250)
        self.infobox.setAlignment(QtCore.Qt.AlignTop)
        font = QtGui.QFont()
        font.setPointSize(17)
        font.setBold(True)
        font.setWeight(75)
        self.infobox.setFont(font)
        self.infobox.setStyleSheet('border-color: %s; border-style: outset; border-width: 2px' % BORDER)
        infoBox.addWidget(self.infobox)


        # Teamstats Box
        label = QtGui.QLabel("Team Status:")
        label.setFixedHeight(20)
        label.setAlignment(QtCore.Qt.AlignCenter)
        teamBox.addWidget(label)

        self.teambox = QtGui.QLabel("")
        #self.infobox.setFixedHeight(250)
        self.teambox.setFixedWidth(250)
        self.teambox.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignCenter)
        #self.teambox.setStyleSheet('border-color: %s; border-style: outset; border-width: 2px' % BORDER)
        teamBox.addWidget(self.teambox)


        # Elapsed time counter.
        self.elapsedTime = QtGui.QLabel("Runtime: 0:00:00")
        self.elapsedTime.setAlignment(QtCore.Qt.AlignLeft)
        lowerBox.addWidget(self.elapsedTime)


    ## Utility Functions. ##
    def set_time(self, t):
        """ Set the time, and also update the limits of time jump box if we need to. """
        if not t == self.time:
            self.time = t
            self.jumpInput.setText(str(self.time))
        # If this is larger than the current maxtime, update both the variable and the validator
        if t > self.maxtime: 
            self.maxtime = t
            self.inputValidator.setRange(0, self.maxtime)

    def set_output_box(self, text):
        """ Set the text describing our annotated point. """
        self.infobox.setText(text)

    def update_teamstats(self, teamstats):
        """ Update the text that tells us what each team is doing. """
        text = ""
        for i in range(len(teamstats)):
            # For each team, change the colour of the label for that team depedning on what it's doing. 
            colour, label = teamtext(teamstats[i])
            text += "<p style='margin:0;' ><font color=%s size='+2'> Team %d: %s</FONT></p>\n" % (colour, i, label)
        self.teambox.setText(text)

    def make_radio_buttons(self, functionals):
        """ Build the radiobuttons for switching functionals. """
        for i in range(len(functionals)):
            # For each functional, make a radio button and link it to the switch_functionals method. 
            radio_button = QtGui.QRadioButton(text=functionals[i])
            radio_button.clicked.connect(lambda: self.switch_functional())
            self.functionalBox.addWidget(radio_button)
            self.radio_buttons.append(radio_button)
        self.radio_buttons[0].setChecked(True) # Select the radio button corresponding to functional 0. 

    def switch_functional(self):
        """ Switch functionals. Which one we switch to depends on the radiobutton clicked. """
        i = 0 # keep track of the index of the radiobutton.
        for rb in self.radio_buttons:
            if rb.isChecked(): 
                # If this is the radio button that has been clicked, switch to the appropriate functional and jump out of the loop.
                self.pc.switch_functional(i) 
                break
            else: i+=1

    def clicked_diagram(self, event):
        """ Annotates the diagram, by plotting a tooltip with the params and branchid of the point the user clicked.
            If the diagram is already annotated, remove the annotation. """
        annotated = self.pc.annotate(event.xdata, event.ydata)
        if annotated:
            self.buttonPlot.setEnabled(True)
        else:     
            self.buttonPlot.setEnabled(False)

    def start(self):
        """ Set Time=0. """
        t = self.pc.start()
        self.set_time(t)

    def back(self):
        """ Set Time=Time-1. """
        t = self.pc.back()
        self.set_time(t)

    def forward(self):
        """ Set Time=Time+1. """
        t = self.pc.forward()
        self.set_time(t)

    def end(self):
        """ Set Time=Maxtime. """
        t = self.pc.end()
        self.set_time(t)

    def jump(self):
        """ Jump to Time=t. """
        t = int(self.jumpInput.text())
        new_time = self.pc.jump(t)
        self.set_time(new_time)

    def plot(self):
        """ Launch Matplotlib/Paraview to graph the highlighted solution. """
        self.pc.plot()

    def set_elapsed_time(self, elapsed):
        """ Gets the amount of time that has elapsed since defcon started running. """
        t = str(timedelta(seconds=elapsed)).split('.')[0]
        self.elapsedTime.setText("Runtime: " + t)

