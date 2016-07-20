from matplotlib.backends import qt_compat
use_pyside = qt_compat.QT_API == qt_compat.QT_API_PYSIDE
if use_pyside:
    from PySide import QtGui, QtCore
else:
    from PyQt4 import QtGui, QtCore

import sys, getopt, os
from math import sqrt

# Imports for the paraview and hdf5topvd methods
from subprocess import Popen
import h5py as h5
from dolfin import *
from parametertools import parameterstostring

# We'll use literal_eval to get lists and tuples back from the journal. 
# This is not as bad as eval, as it only recognises: strings, bytes, numbers, tuples, lists, dicts, booleans, sets, and None.
from ast import literal_eval 

# For plotting the bifurcation diagram.
import matplotlib
matplotlib.use("Qt4Agg")
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT
from matplotlib.figure import Figure

# Styles for matplotlib.
# See matpoltlib.styles.available for options.
try:
    from matplotlib import style
    style.use('ggplot')
except AttributeError:
    print "Update to the latest version of matplotlib to use styles."

# Colours.
MAIN = 'black' # colour for points.
HIGHLIGHT = 'red' # colour for selected points.
GRID = 'white' # colour the for grid.
BUTTONBG = 'grey' # backgound colour for buttons and labels.
BUTTONTEXT = 'black' # colour for the text on the buttons and labels.
WINDOWBG = 'grey' # background colour for the window.

# Set some defaults.
problem_type = None
problem_class = None
problem_mesh = None
working_dir= "."
output_dir = None 
darkmode = False
plot_with_mpl = False
update_interval = 100

# Get commandline args.
# Example usage: python defcon-gui.py -p unity -c RootsOfUnityProblem -w /home/joseph/defcon/examples/unity
myopts, args = getopt.getopt(sys.argv[1:],"dp:o:w:c:m:i:")

for o, a in myopts:
    if o == '-p':   problem_type = a
    elif o == '-o': output_dir = a
    elif o == '-w': working_dir = a
    elif o == '-c': problem_class = a
    elif o == '-m': problem_mesh = a
    elif o == '-d': darkmode = True
    elif o == '-i': update_interval = int(a)
    else:           print("Usage: %s -d -p <problem_type> -c <problem_class> -w <working_dir> -o <defcon_output_directory> -m <mesh> -i <update interval in ms>" % sys.argv[0])

if output_dir is None: output_dir = working_dir + os.path.sep + "output"
if problem_type is None: problem_type = working_dir.split(os.path.sep)[-1] 
solutions_dir = output_dir + os.path.sep + "solutions" + os.path.sep

# Set up the figure.
figure = Figure(figsize=(7,6), dpi=100)
bfdiag = figure.add_subplot(111)
bfdiag.grid(color=GRID)

# Darkmode colour scheme.
if darkmode: 
    figure.patch.set_facecolor('black')
    bfdiag.set_axis_bgcolor('black')
    bfdiag.xaxis.label.set_color('#76EE00')
    bfdiag.yaxis.label.set_color('#76EE00')
    bfdiag.tick_params(axis='y', colors='#76EE00')
    bfdiag.tick_params(axis='x', colors='#76EE00')
    MAIN = 'w' 
    HIGHLIGHT = '#76EE00'
    GRID = '0.75' 
    BUTTONBG = 'black'
    BUTTONTEXT = '#76EE00'
    WINDOWBG = 'black'

# Put the working directory on our path.
sys.path.insert(0, working_dir) 
sys.path.insert(0, "%s/.." % os.path.dirname(os.path.realpath(sys.argv[0]))) #FIXME: This is ugly, but does always work. It seems to need this, else the problem_type fails to import 'BifurcationProblem'. even though the defcon directory is in PYTHONPATH. Why, and how to get rid of it?

# If we've been told about the problem, then get the name and type of the problem we're dealing with, as well as everything else we're going to need for plotting solutions in paraview
if problem_type and problem_class:
    problem_name = __import__(problem_type)
    globals().update(vars(problem_name))
    globals()["bfprob"] = getattr(problem_name, problem_class)
    problem = bfprob()

    # Get the mesh. If the user has specified a file, then great, otherwise try to get it from the problem. 
    if problem_mesh is not None: mesh = Mesh(mpi_comm_world(), problem_mesh)
    else: mesh = problem.mesh(mpi_comm_world())
    if mesh.geometry().dim() < 2: plot_with_mpl = True # if the mesh is 1D, we don't want to use paraview. 

    V = problem.function_space(mesh)
    problem_parameters = problem.parameters()
else:
    print "In order to use paraview for graphing solutions, you must specify the class of the problem, eg 'NavierStokesProblem'."
    print("Usage: %s -p <problem type> -c <problem_class> -w <working dir> \n" % sys.argv[0])

# TODO:
#     1) Implement plot branch and plot params
#     2) grey out buttons when they can't be used. 


#####################
### Utility Class ###
#####################

class PlotConstructor():
    """ Class for handling everything to do with the bifuraction diagram plot. """

    def __init__(self, app):
        self.points = [] # Keep track of the points we've found, so we can redraw everything if necessary. Also for annotation.

        self.maxtime = 0 # Keep track of the furthest we have got in time. 
        self.time = 0 # Keep track of where we currently are in time.

        self.paused = False # Are we updating we new points, or are we frozen in time?

        self.annotation_highlight = None # The point we've annotated. 
        self.annotated_point = None # The (params, branchid) of the annotated point

        self.path = output_dir + os.path.sep + "journal" + os.path.sep +"journal.txt" # The working directory.

        self.freeindex = None # The index of the free parameter.

        self.current_functional = 0

        self.app = app # The BifurcationPage window, so that we can set the time.
    
    def distance(self, x1, x2, y1, y2):
        """ Return the L2 distance between two points. """
        return(sqrt((x1 - x2)**2 + (y1 - y2)**2))

    def pause(self):
        """ Pause the drawing. """
        self.paused = True

    def unpause(self):
        """ Unpause the drawing. """
        self.paused = False

    def start(self):
        """ Go to Time=0 """
        if not self.paused: self.pause()
        self.time = 0

        if self.annotated_point is not None: self.unannotate()
        bfdiag.clear()
        bfdiag.set_xlabel(self.parameter_name)
        bfdiag.set_ylabel(self.functional_names[self.current_functional])
        bfdiag.grid(color=GRID)
        return self.time

    def end (self):
        if self.time < self.maxtime:
            xs = [float(point[0][self.freeindex]) for point in self.points[self.time:]]
            ys = [float(point[1][self.current_functional]) for point in self.points[self.time:]]
            bfdiag.plot(xs, ys, marker='.', color=MAIN, linestyle='None')
            self.time = self.maxtime
        if self.paused: self.unpause
        return self.maxtime

    def back(self):
        """ Take a step backwards in time. """
        if not self.paused: self.pause()
        if self.time > 0:
            #FIXME: Extremely inefficient to replot everything
            xs = [float(point[0][self.freeindex]) for point in self.points[:self.time]]
            ys = [float(point[1][self.current_functional]) for point in self.points[:self.time]]
            self.time -= 1

            if self.annotated_point is not None: self.unannotate()
            bfdiag.clear()
            bfdiag.set_xlabel(self.parameter_name)
            bfdiag.set_ylabel(self.functional_names[self.current_functional])
            bfdiag.grid(color=GRID)
            bfdiag.plot(xs, ys, marker='.', color=MAIN, linestyle='None') 
        return self.time

    def forward(self):
        """ Take a step forwards in time. """
        if not self.paused: self.pause()
        if self.time < self.maxtime:
            xs, ys, branchid, teamno, cont = self.points[self.time]
            self.time += 1
            bfdiag.plot(float(xs[self.freeindex]), float(ys[self.current_functional]), marker='.', color=MAIN, linestyle='None')
        if self.time==self.maxtime: self.pause
        return self.time

    def jump(self, t):
        """ Jump to time t. """
        if not self.paused: self.pause()
        if t <= self.maxtime:
            #FIXME: Extremely inefficient to replot everything
            xs = [float(point[0][self.freeindex]) for point in self.points[:(t+1)]]
            ys = [float(point[1][self.current_functional]) for point in self.points[:(t+1)]]
            self.time = t

            if self.annotated_point is not None: self.unannotate()
            bfdiag.clear()
            bfdiag.set_xlabel(self.parameter_name)
            bfdiag.set_ylabel(self.functional_names[self.current_functional])
            bfdiag.grid(color=GRID)
            bfdiag.plot(xs, ys, marker='.', color=MAIN, linestyle='None') 
        return self.time

    def switch_functional(self, i):
        self.current_functional = i
        xs = [float(point[0][self.freeindex]) for point in self.points[:self.time]]
        ys = [float(point[1][self.current_functional]) for point in self.points[:self.time]]

        if self.annotated_point is not None: self.unannotate()
        bfdiag.clear()
        bfdiag.set_xlabel(self.parameter_name)
        bfdiag.set_ylabel(self.functional_names[self.current_functional])
        bfdiag.grid(color=GRID)
        bfdiag.plot(xs, ys, marker='.', color=MAIN, linestyle='None') 

    def grab_data(self):
        """ Get data from the file. """
        # If the file doesn't exist, just pass.
        try: pullData = open(self.path, 'r').read()
        except Exception: pullData = None
        return pullData

    def animate(self):
        """ Handles the redrawing of the graph. """
        # If we're in pause mode, or we're done, then do nothing.
        if self.paused:
            pass

        # If we're not paused, we draw all the points that have come in since we last drew something.
        else:   
            # Catch up to the points we have in memory.
            if self.time < self.maxtime:
                for xs, ys, branchid, params in self.points[self.time:]:
                    bfdiag.plot(float(xs[self.freeindex]), float(ys[self.current_functional]), marker='.', color=MAIN, linestyle='None')
                    self.time += 1

            # Get new points, if they exist. If not, just pass. 
            pullData = self.grab_data()
            if pullData is not None:
                dataList = pullData.split('\n')

                # Is this is first time, get the information from the first line of the data. 
                if self.freeindex is None: 
                    freeindex, self.parameter_name, functional_names, unicode_functional_names = dataList[0].split(';')
                    self.freeindex = int(freeindex)
                    self.functional_names = literal_eval(functional_names)
                    self.unicode_functional_names = literal_eval(unicode_functional_names)
                    self.app.make_radio_buttons(self.unicode_functional_names)
                    bfdiag.set_xlabel(self.parameter_name)
                    bfdiag.set_ylabel(self.functional_names[self.current_functional])

                dataList = dataList[1:] # exclude the first line. 

                # Plot new points one at a time.
                for eachLine in dataList[self.time:]:
                    if len(eachLine) > 1:
                        teamno, oldparams, branchid, newparams, functionals, cont = eachLine.split(';')
                        xs = literal_eval(newparams)
                        ys = literal_eval(functionals)
                        x = float(xs[self.freeindex])
                        y = float(ys[self.current_functional])
                        self.points.append((xs, ys, int(branchid), int(teamno), literal_eval(cont)))
                        bfdiag.plot(x, y, marker='.', color=MAIN, linestyle='None')
                        self.time += 1

                # Update the current time.
                self.maxtime = self.time
                self.app.set_time(self.time)         

    def annotate(self, clickX, clickY):
         """ Annotate a point when clicking on it. If there's already an annotation, remove it. """
         if self.annotated_point is None:
             xs = [float(point[0][self.freeindex]) for point in self.points[:self.time]]
             ys = [float(point[1][self.current_functional]) for point in self.points[:self.time]]
 
             # FIXME: The *100 is because these were too small, might need some changing. Also doesn't work on, say allen-cahn, as xtol=0.
             xtol = 100*((max(xs) - min(xs))/float(len(xs)))/2 
             ytol = 100*((max(ys) - min(ys))/float(len(ys)))/2 

             # xtol and ytol might end up being zero, if all points have same x/y values. Do something in this case.
             if xtol==0: xtol = 1
             if ytol==0: ytol = 1

             annotes = []

             # Find the point on the diagram closest to the point the user clicked.
             time = 1
             for xs, ys, branchid, teamno, cont in self.points[:self.time]:
                  x = float(xs[self.freeindex])
                  y = float(ys[self.current_functional])
                  if ((clickX-xtol < x < clickX+xtol) and (clickY-ytol < y < clickY+ytol)):
                      annotes.append((self.distance(x, clickX, y, clickY), x, y, branchid, xs, teamno, cont, time))
                  time += 1

             if annotes:
                 annotes.sort()
                 distance, x, y, branchid, xs, teamno, cont, time = annotes[0]

                 # Plot the annotation, and keep a handle on all the stuff we plot so we can use/remove it later. 
                 self.annotation_highlight = bfdiag.scatter([x], [y], s=[50], marker='o', color=HIGHLIGHT) # Note: change 's' to make the highlight blob bigger/smaller
                 self.annotated_point = (xs, branchid)  

                 if cont: s = "continuation"
                 else: s = "deflation"

                 self.app.set_output_box("Solution on branch %d\nFound by team %d\nUsing %s\nAt time %d\n\nx = %s\ny = %s" % (branchid, teamno, s, time, x, y))

                 return True
             else: return False

         else: self.unannotate()


    def unannotate(self):
        self.annotation_highlight.remove()
        self.annotation_highlight = None
        self.annotated_point = None
        self.app.set_output_box("")
        return False

    def hdf52pvd(self):
        """ Utility function for creating a pvd from hdf5. Uses the point that is annotated. """
        if self.annotated_point is not None:
            # Get the params and branchid of the point.
            params, branchid = self.annotated_point    

            # Make a directory to put solutions in, if it doesn't exist. 
            try: os.mkdir(output_dir + os.path.sep + "solutions")
            except OSError: pass

            # Create the file to which we will write these solutions.
            pvd_filename = solutions_dir +  "SOLUTION$%s$branchid=%d.pvd" % (parameterstostring(problem_parameters, params), branchid)
            pvd = File(pvd_filename)
    
            # Read the file, then write the solution.
            filename = output_dir + os.path.sep + parameterstostring(problem_parameters, params) + ".hdf5"
            with HDF5File(mesh.mpi_comm(), filename, 'r') as f:
                y = Function(V)
                f.read(y, "solution-%d" % branchid)
                f.flush()
                pvd << y
                pvd

            self.launch_paraview(pvd_filename)

    def launch_paraview(self, filename):
        """ Utility function for launching paraview. Popen launches it in a separate process, so we may carry on with whatever we are doing."""
        Popen(["paraview", filename])

    def mpl_plot(self):
        if self.annotated_point is not None:
            # Make a directory to put solutions in, if it doesn't exist. 
            try: os.mkdir(output_dir + os.path.sep + "solutions")
            except OSError: pass

            params, branchid = self.annotated_point
            filename = output_dir + os.path.sep + parameterstostring(problem_parameters, params) + ".hdf5"
            with HDF5File(mesh.mpi_comm(), filename, 'r') as f:
                y = Function(V)
                f.read(y, "solution-%d" % branchid)
                f.flush() 
            p = plot(y, title="Solution on branch %s, params %s" % (branchid, params), hardcopy_prefix=output_dir + os.path.sep + "solutions" + os.path.sep +"branch-%s$params-%s" % (branchid, params), backend='vtk', interactive=True)
            #p.write_png() # write to a file. This dumps the file in the working directory, doesn't save to a nice place. Why???

          

######################################################################
class DynamicCanvas(FigureCanvas):
    """A canvas that updates itself with a new plot."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        FigureCanvas.__init__(self, figure)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.update_figure)
        timer.start(update_interval)

    def update_figure(self):
        pc.animate()
        self.draw()

    # TODO: have defcon write a line to the journal when it's done. Then 
    def finish(self):
        raise NotImplementedError

class CustomToolbar(NavigationToolbar2QT):
    """ A custom matplotlib toolbar, so we can remove those pesky extra buttons. """  
    def __init__(self, canvas, parent):
        self.toolitems = (
            ('Home', 'Reset original view', 'home', 'home'),
            ('Pan', 'Pan axes with left mouse, zoom with right', 'move', 'pan'),
            ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom'),
            ('Save', 'Save the figure', 'filesave', 'save_figure'),
            )
        NavigationToolbar2QT.__init__(self, canvas, parent)    

class ApplicationWindow(QtGui.QMainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("DEFCON")

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

        if darkmode: self.main_widget.setStyleSheet('color: green; background-color: black')

        # Layout
        main_layout = QtGui.QHBoxLayout(self.main_widget)
        lVBox = QtGui.QVBoxLayout()
        rVBox = QtGui.QVBoxLayout()
        main_layout.addLayout(lVBox)
        main_layout.addLayout(rVBox)

        canvasBox = QtGui.QVBoxLayout()
        lVBox.addLayout(canvasBox)
        timeBox = QtGui.QHBoxLayout()
        timeBox.setAlignment(QtCore.Qt.AlignCenter)
        lVBox.addLayout(timeBox)
        lowerBox = QtGui.QHBoxLayout()
        lowerBox.setAlignment(QtCore.Qt.AlignCenter)
        lVBox.addLayout(lowerBox)

        self.functionalBox = QtGui.QVBoxLayout()
        rVBox.addLayout(self.functionalBox)
        infoBox = QtGui.QVBoxLayout()
        infoBox.setContentsMargins(0, 10, 0, 10)
        rVBox.addLayout(infoBox)
        plotBox = QtGui.QHBoxLayout()
        rVBox.addLayout(plotBox)
        plotBox.setAlignment(QtCore.Qt.AlignTop)


        # Canvas.
        self.dc = DynamicCanvas(self.main_widget, width=5, height=4, dpi=100)
        canvasBox.addWidget(self.dc)
        self.dc.mpl_connect('button_press_event', self.clicked_diagram)

        toolbar = CustomToolbar( self.dc, self )
        toolbar.update()
        canvasBox.addWidget(toolbar)

        # Time navigation buttons
        self.buttonStart = QtGui.QPushButton("|<")
        self.buttonStart.clicked.connect(lambda:self.start())
        self.buttonStart.setFixedWidth(30)
        timeBox.addWidget(self.buttonStart)

        self.buttonBack = QtGui.QPushButton("<")
        self.buttonBack.clicked.connect(lambda:self.back())
        self.buttonBack.setFixedWidth(30)
        timeBox.addWidget(self.buttonBack)

        self.timeLabel = QtGui.QLabel("")
        timeBox.addWidget(self.timeLabel)
        self.timeLabel.setFixedWidth(100)
        self.timeLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.buttonForward = QtGui.QPushButton(">")
        self.buttonForward.clicked.connect(lambda:self.forward())
        self.buttonForward.setFixedWidth(30)
        timeBox.addWidget(self.buttonForward)

        self.buttonEnd = QtGui.QPushButton(">|")
        self.buttonEnd.clicked.connect(lambda:self.end())
        self.buttonEnd.setFixedWidth(30)
        timeBox.addWidget(self.buttonEnd)

        # TODO: Define a custom one of these, to do just we want it to do.
        self.jumpInput = QtGui.QLineEdit()
        self.jumpInput.setMaxLength(4) #FIXME: Get appropriate value.
        self.jumpInput.setText("0")
        self.jumpInput.setFixedWidth(40)
        lowerBox.addWidget(self.jumpInput)

        self.buttonJump = QtGui.QPushButton("Jump")
        self.buttonJump.clicked.connect(lambda:self.jump())
        self.buttonJump.setFixedWidth(40)
        lowerBox.addWidget(self.buttonJump)


        # Plot Buttons
        self.buttonPlot = QtGui.QPushButton("Plot")
        self.buttonPlot.clicked.connect(lambda:self.launch_paraview())
        plotBox.addWidget(self.buttonPlot)

        self.buttonPlotBranch = QtGui.QPushButton("Plot Branch")
        self.buttonPlotBranch.clicked.connect(lambda:self.launch_paraview())
        plotBox.addWidget(self.buttonPlotBranch)

        self.buttonParams = QtGui.QPushButton("Plot Params")
        self.buttonParams.clicked.connect(lambda:self.launch_paraview())
        plotBox.addWidget(self.buttonParams)

        # Radio buttons
        label = QtGui.QLabel("Functionals:")
        label.setFixedHeight(20)
        self.functionalBox.addWidget(label)
        self.radio_buttons = []

        # Output Box
        self.infobox = QtGui.QLabel("")
        self.infobox.setFixedHeight(250)
        self.infobox.setAlignment(QtCore.Qt.AlignTop)
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.infobox.setFont(font)
        self.infobox.setStyleSheet('border-color: black; border-style: outset; border-width: 2px')
        infoBox.addWidget(self.infobox)



    # Utility Functions
    def set_time(self, t):
        self.timeLabel.setText("Time = %d" % t)

    def set_output_box(self, text):
        self.infobox.setText(text)

    def make_radio_buttons(self, functionals):
        for i in range(len(functionals)):
            radio_button = QtGui.QRadioButton(text=functionals[i])
            radio_button.clicked.connect(lambda: self.switch_functional())
            self.functionalBox.addWidget(radio_button)
            self.radio_buttons.append(radio_button)
        self.radio_buttons[0].setChecked(True)

    def switch_functional(self):
        for i in range(1000):
            if self.radio_buttons[i].isChecked(): 
                pc.switch_functional(i)
                break


    def clicked_diagram(self, event):
        """ Annotates the diagram, by plotting a tooltip with the params and branchid of the point the user clicked.
            If the diagram is already annotated, remove the annotation. """
        annotated = pc.annotate(event.xdata, event.ydata)
        #if annotated: self.buttonPlot.config(state="normal")
        #else:         self.buttonPlot.config(state="disabled")

    def start(self):
        """ Set Time=0. """
        t = pc.start()
        self.set_time(t)

    def back(self):
        """ Set Time=Time-1. """
        t = pc.back()
        self.set_time(t)

    def forward(self):
        """ Set Time=Time+1. """
        t = pc.forward()
        self.set_time(t)

    def end(self):
        """ Set Time=Maxtime. """
        t = pc.end()
        self.set_time(t)

    def jump(self):
        """ Jump to Time=t. """
        #t = tkSimpleDialog.askinteger("Jump to", "Enter a time to jump to")
        t = None
        if t is not None: 
            new_time = pc.jump(t)
            self.set_time(new_time)

    def launch_paraview(self):
        """ Launch Paraview to graph the highlighted solution. """
        if not plot_with_mpl: pc.hdf52pvd()
        else: pc.mpl_plot()


qApp = QtGui.QApplication(sys.argv)
aw = ApplicationWindow()
pc = PlotConstructor(aw)
aw.setWindowTitle("DEFCON")
aw.show()
sys.exit(qApp.exec_())
