import matplotlib
matplotlib.use("Qt4Agg")
from matplotlib.backends import qt_compat
use_pyside = qt_compat.QT_API == qt_compat.QT_API_PYSIDE
if use_pyside:
    from PySide import QtGui, QtCore
else:
    from PyQt4 import QtGui, QtCore

import sys, getopt, os
from math import sqrt, floor, ceil

# Imports for the paraview and hdf5topvd methods
from subprocess import Popen
from dolfin import *
from parametertools import parameterstostring

# We'll use literal_eval to get lists and tuples back from the journal. 
# This is not as bad as eval, as it only recognises: strings, bytes, numbers, tuples, lists, dicts, booleans, sets, and None.
# It can't do anything bad if you pass it, for example, the string "os.system("rm -rf /")".
from ast import literal_eval 

# For plotting the bifurcation diagram.
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT
from matplotlib.figure import Figure

# For saving movies.
from matplotlib import animation

# For plotting solutions, if we don't use paraview
import matplotlib.pyplot as plt

# Styles for matplotlib.
# See matpoltlib.styles.available for options.
try:
    from matplotlib import style
    style.use('ggplot')
except AttributeError:
    print "\033[91m[Warning] Update to the latest version of matplotlib to use styles.\033[00m\n"
    pass

# Saving tikz pictures.
try: 
    from matplotlib2tikz import save as tikz_save
    use_tikz = True
except Exception: 
    print "\033[91m[Warning] Could not import the library matplotlib2tikz. You will unable to save the file as a .tikz.\nTo use this functionality, install matplotlib2tikz, eg with:\n     # pip install matplotlib2tikz\033[00m\n"
    use_tikz = False

# Colours.
MAIN = 'black' # colour for points
DEF = 'blue' # colour for points found via deflation
HIGHLIGHT = 'red' # colour for selected points
GRID = 'white' # colour the for grid.
BORDER = 'black' # borders on the UI
SWEEP = 'red' # colour for the sweep line

# Markers and various other styles.
CONTPLOT = '.' # marker to use for points found by continuation
DEFPLOT = 'o' # marker to use for points found by deflation
SWEEPSTYLE = 'dashed' # line style for the sweep line

# Set some defaults.
problem_type = None
problem_class = None
problem_mesh = None
working_dir= "."
output_dir = None
solutions_dir = None
darkmode = False
plot_with_mpl = False # where we will try to plot solutions with paraview or matplotlib
update_interval = 100 # update interval for the diagram
resources_dir = os.path.dirname(os.path.realpath(sys.argv[0])) + os.path.sep + 'resources' + os.path.sep # icons, etc. 

# Get commandline args.
# Example usage: python defcon-gui.py -p unity -c RootsOfUnityProblem -w /home/joseph/defcon/examples/unity
myopts, args = getopt.getopt(sys.argv[1:],"dp:o:w:c:m:i:s:")

for o, a in myopts:
    if o == '-p':   problem_type = a
    elif o == '-o': output_dir = os.path.expanduser(a)
    elif o == '-w': working_dir = os.path.expanduser(a)
    elif o == '-s': solutions_dir = os.path.expanduser(a)
    elif o == '-c': problem_class = a
    elif o == '-m': problem_mesh = a
    elif o == '-d': darkmode = True
    elif o == '-i': update_interval = int(a)
    else:           print("Usage: %s -d -p <problem_type> -c <problem_class> -w <working_dir> -o <defcon_output_directory> -m <mesh> -i <update interval in ms>" % sys.argv[0])

# If we didn't specify an output directory, default to the folder "output" in the working directory
if output_dir is None: output_dir = working_dir + os.path.sep + "output"

# If we didn't specify the name of the python file for the problem (eg, elastica), assume it's the same as the directory we're working in. 
if problem_type is None: 
    dirs = working_dir.split(os.path.sep)
    if dirs[-1]: problem_type = dirs[-1] # no trailing slash
    else: problem_type = dirs[-2] # trailing slash

# If we didn't specify a directory for solutions we plot, store them in the "solutions" subdir of the output directory.
if solutions_dir is None: solutions_dir = output_dir + os.path.sep + "solutions" + os.path.sep

# Darkmode colour scheme.
if darkmode: 
    figure.patch.set_facecolor('black')
    bfdiag.set_axis_bgcolor('black')
    bfdiag.xaxis.label.set_color('#76EE00')
    bfdiag.yaxis.label.set_color('#76EE00')
    bfdiag.tick_params(axis='y', colors='#76EE00')
    bfdiag.tick_params(axis='x', colors='#76EE00')
    MAIN = 'w' 
    DEF = 'yellow'
    HIGHLIGHT = '#76EE00'
    GRID = '0.75'
    BORDER = 'white'


# Set up the figure.
figure = Figure(figsize=(7,6), dpi=100)
bfdiag = figure.add_subplot(111)
bfdiag.grid(color=GRID)

# Put the working directory on our path.
sys.path.insert(0, working_dir) 
sys.path.insert(0, "%s/.." % os.path.dirname(os.path.realpath(sys.argv[0]))) #FIXME: This is ugly, but does always work. It seems to need this, else the problem_type fails to import 'BifurcationProblem'. even though the defcon directory is in PYTHONPATH. Why, and how to get rid of it?

# If we've been told about the problem, then get the name and type of the problem we're dealing with, as well as everything else we're going to need for plotting solutions.
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
    io = FileIO(output_dir)
    io.setup(problem_parameters, None, V)
    
else: print "\033[91m[Warning] In order to graph solutions, you must specify the class of the problem, eg 'NavierStokesProblem'.\nUsage: %s -p <problem type> -c <problem_class> -w <working dir> \033[00m \n" % sys.argv[0]

#####################
### Utility Class ###
#####################

class PlotConstructor():
    """ Class for handling everything to do with the bifuraction diagram plot. """

    def __init__(self, app):
        self.points = [] # Keep track of the points we've found, so we can redraw everything if necessary. Also for annotation.
        self.pointers = [] # Pointers to each point on the plot, so we can remove them. 

        self.maxtime = 0 # Keep track of the furthest we have got in time. 
        self.time = 0 # Keep track of where we currently are in time.
        self.lines_read = 0 # Number of lines of the journal fil we've read.

        self.paused = False # Are we updating we new points, or are we frozen in time?
        self.done = False # Are we done?

        self.annotation_highlight = None # The point we've annotated. 
        self.annotated_point = None # The (params, branchid) of the annotated point

        self.path = output_dir + os.path.sep + "journal" + os.path.sep +"journal.txt" # Journal file.

        self.freeindex = None # The index of the free parameter.

        self.current_functional = 0 # The index of the functional we're currently on.

        self.app = app # The BifurcationPage window, so that we can set the time.

        self.teamstats = [] # The status of each team. 

        self.sweep = 0
        self.sweepline = None

        self.changed = False
    
    ## Private utility functions. ##
    def distance(self, x1, x2, y1, y2):
        """ Return the L2 distance between two points. """
        return(sqrt((x1 - x2)**2 + (y1 - y2)**2))

    def redraw(self):
        "Clears the window, redraws the labels and the grid. """
        bfdiag.clear()
        bfdiag.set_xlabel(self.parameter_name)
        bfdiag.set_ylabel(self.functional_names[self.current_functional])
        bfdiag.grid(color=GRID)
        ys = [point[1][self.current_functional] for point in self.points]
        bfdiag.set_ylim([floor(min(ys)), ceil(max(ys))])
        self.sweepline = bfdiag.axvline(x=self.sweep, linewidth=1, linestyle=SWEEPSTYLE, color=SWEEP)

    def launch_paraview(self, filename):
        """ Utility function for launching paraview. Popen launches it in a separate process, so we may carry on with whatever we are doing."""
        Popen(["paraview", filename])

    def animate(self, i):
        """ Utility function for animating a plot. """
        xs, ys, branchid, teamno, cont = self.points[i]
        x = float(xs[self.freeindex])
        y = float(ys[self.func_index])
        if cont: c, m= MAIN, '.'
        else: c, m= DEF, 'o'
        # Let's output a little log of how we're doing, so the user can see that something is in fact being done.
        if i % 50 == 0: print "Completed %d/%d frames" % (i, self.maxtime)
        return self.ax.plot(x, y, marker=m, color=c, linestyle='None')

    ## Controls for moving backwards and forwards in the diagram, or otherwise manipulating it. ##
    def pause(self):
        """ Pause the drawing. """
        self.paused = True

    def unpause(self):
        """ Unpause the drawing. """
        self.paused = False

    def seen(self):
        """ Tell the PlotConstructor we've got all the new points. """
        self.changed = False

    def start(self):
        """ Go to Time=0 """
        if not self.paused: self.pause()
        self.time = 0
        if self.annotated_point is not None: self.unannotate()
        self.redraw()
        self.changed = True
        return self.time

    def end (self):
        """ Go to Time=maxtime. """
        if self.time < self.maxtime:
            for i in range(self.time, self.maxtime):
                xs, ys, branchid, teamno, cont = self.points[i]
                x = float(xs[self.freeindex])
                y = float(ys[self.current_functional])
                if cont: c, m= MAIN, '.'
                else: c, m = DEF, 'o'
                self.pointers[i] = bfdiag.plot(x, y, marker=m, color=c, linestyle='None')
            self.time = self.maxtime
            self.changed = True
            if self.paused: self.unpause()
        return self.maxtime

    def back(self):
        """ Take a step backwards in time. """
        if not self.paused: self.pause()
        if self.time > 0:
            if self.annotated_point is not None: self.unannotate()
            self.time -= 1
            self.pointers[self.time][0].remove()
            self.changed = True
        return self.time

    def forward(self):
        """ Take a step forwards in time. """
        if not self.paused: self.pause()
        if self.time < self.maxtime:
            xs, ys, branchid, teamno, cont = self.points[self.time]
            x = float(xs[self.freeindex])
            y = float(ys[self.current_functional])
            if cont: c, m= MAIN, '.'
            else: c, m= DEF, 'o'
            self.pointers[self.time] = bfdiag.plot(x, y, marker=m, color=c, linestyle='None')
            self.time += 1
            self.changed = True
        if self.time==self.maxtime: self.unpause()
        return self.time

    def jump(self, t):
        """ Jump to time t. """
        if not self.paused: self.pause()
        if self.annotated_point is not None: self.unannotate()
        if t < self.time:
            for i in range(t, self.time):
                self.pointers[i][0].remove()
        elif t > self.time:
            for i in range(self.time, t):
                xs, ys, branchid, teamno, cont = self.points[i]
                x = float(xs[self.freeindex])
                y = float(ys[self.current_functional])
                if cont: c, m= MAIN, '.'
                else: c, m= DEF, 'o'
                self.pointers[i] = bfdiag.plot(x, y, marker=m, color=c, linestyle='None')
        self.time = t 
        self.changed = True
        if self.time==self.maxtime: self.unpause()
        return self.time

    
    def switch_functional(self, i):
        """ Change the functional being plotted. """
        self.current_functional = i
        if self.annotated_point is not None: self.unannotate()
        self.redraw()
        for j in range(0, self.time):
            xs, ys, branchid, teamno, cont = self.points[j]
            x = float(xs[self.freeindex])
            y = float(ys[self.current_functional])
            if cont: c, m= MAIN, '.'
            else: c, m= DEF, 'o'
            self.pointers[j] = bfdiag.plot(x, y, marker=m, color=c, linestyle='None')
            self.changed = True

    ## Functions for getting new points and updating the diagram ##
    def grab_data(self):
        """ Get data from the file. """
        # If the file doesn't exist, just pass.
        try: pullData = open(self.path, 'r').read()
        except Exception: pullData = None
        return pullData

    def update(self):
        """ Handles the redrawing of the graph. """
        # If we're in pause mode, or we're done, then do nothing.
        if self.paused or self.done:
            return self.changed

        # If we're not paused, we draw all the points that have come in since we last drew something.
        else:   
            # Catch up to the points we have in memory.
            if self.time < self.maxtime:
                for xs, ys, branchid, teamno, cont in self.points[self.time:]:
                    bfdiag.plot(float(xs[self.freeindex]), float(ys[self.current_functional]), marker='.', color=MAIN, linestyle='None')
                    self.time += 1

            # Get new points, if they exist. If not, just pass. 
            pullData = self.grab_data()
            if pullData is not None:
                dataList = pullData.split('\n')

                # Is this is first time, get the information from the first line of the data. 
                if self.freeindex is None: 
                    freeindex, self.parameter_name, functional_names, unicode_functional_names, nteams, minparam, maxparam = dataList[0].split(';')

                    # Set up info about what the teams are doing.
                    self.nteams = int(nteams)
                    for team in range(self.nteams): self.teamstats.append('i')

                    # Info about functionals
                    self.freeindex = int(freeindex)
                    self.functional_names = literal_eval(functional_names)
                    self.unicode_functional_names = literal_eval(unicode_functional_names)
                    self.app.make_radio_buttons(self.unicode_functional_names)
                    bfdiag.set_xlabel(self.parameter_name)
                    bfdiag.set_ylabel(self.functional_names[self.current_functional])
                    bfdiag.set_xlim([float(minparam), float(maxparam)])
                    bfdiag.autoscale(axis='y')

                dataList = dataList[1:] # exclude the first line. 

                # Plot new points one at a time.
                for eachLine in dataList[self.lines_read:]:
                    if len(eachLine) > 1:
                        if eachLine[0] == '$':
                            params = eachLine[1:]
                            self.sweep = float(params)
                            if self.sweepline is not None: self.sweepline.remove()
                            self.sweepline = bfdiag.axvline(x=self.sweep, linewidth=1, linestyle=SWEEPSTYLE, color=SWEEP)
                        elif eachLine[0] == '~':
                            team, task = eachLine[1:].split(';')
                            self.teamstats[int(team)] = task
                            self.app.update_teamstats(self.teamstats)
                            if task == 'q': self.changed = False
                        else:
                            teamno, oldparams, branchid, newparams, functionals, cont = eachLine.split(';')
                            xs = literal_eval(newparams)
                            ys = literal_eval(functionals)
                            x = float(xs[self.freeindex])
                            y = float(ys[self.current_functional])
                            
                            # Use different colours/plot styles for points found by continuation/deflation.
                            if literal_eval(cont): c, m= MAIN, '.'
                            else: c, m= DEF, 'o'
                            self.points.append((xs, ys, int(branchid), int(teamno), literal_eval(cont)))                            
                            self.pointers.append(bfdiag.plot(x, y, marker=m, color=c, linestyle='None'))
                            self.time += 1
                        self.lines_read +=1
                # Update the current time.
                self.changed = True
                self.maxtime = self.time
                self.app.set_time(self.time)
                return True         


    ## Functions for handling annotation. ##
    def annotate(self, clickX, clickY):
         """ Annotate a point when clicking on it. """
         if self.annotated_point is None:
             xs = [float(point[0][self.freeindex]) for point in self.points[:self.time]]
             ys = [float(point[1][self.current_functional]) for point in self.points[:self.time]]
 
             # Sets the clickbox to be one 'square' of the diagram.
             xtick = bfdiag.get_xticks()
             ytick = bfdiag.get_yticks()
             xtol = (xtick[1]-xtick[0])/2 #100*((max(xs) - min(xs))/float(len(xs)))/2 
             ytol = (ytick[1]-ytick[0])/2 #100*((max(ys) - min(ys))/float(len(ys)))/2 

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
                 self.app.set_output_box("Solution on branch %d\nFound by team %d\nUsing %s\nAs event #%d\n\nx = %s\ny = %s" % (branchid, teamno, s, time, x, y))
                 self.changed = True

                 return True
             else: return False

         else: self.unannotate()


    def unannotate(self):
        """ Remove annotation from the graph. """
        self.annotation_highlight.remove()
        self.annotation_highlight = None
        self.annotated_point = None
        self.app.set_output_box("")
        self.changed = True
        return False

    ## Functions that handle the plotting of solutions. ##
    def hdf52pvd(self):
        """Function for creating a pvd from hdf5. Uses the point that is annotated. """
        if self.annotated_point is not None:
            # Get the params and branchid of the point.
            params, branchid = self.annotated_point

            # Make a directory to put solutions in, if it doesn't exist. 
            try: os.mkdir(output_dir + os.path.sep + "solutions")
            except OSError: pass

            # Create the file to which we will write these solutions.
            pvd_filename = solutions_dir +  "SOLUTION$%s$branchid=%d.pvd" % (parameterstostring(problem_parameters, params), branchid)
            pvd = File(pvd_filename)
    
            y = io.fetch_solutions(params, [branchid])[0]
            pvd << y
            pvd

            self.launch_paraview(pvd_filename)

    def mpl_plot(self):
        """ Fetch a solution and plot it with matplotlib. Used when the solutions are 1D. """
        if self.annotated_point is not None:
            params, branchid = self.annotated_point
            y = io.fetch_solutions(params, [branchid])[0]

            try:
                x = interpolate(Expression("x[0]", degree=1), V)
                # FIXME: For functions f other than CG1, we might need to sort both arrays so that x is increasing. Check this out!
                plt.plot(x.vector().array(), y.vector().array(), '-', linewidth=3, color='b')
                plt.title("%s: branch %s, params %s" % (problem_class, branchid, params))
                plt.axhline(0, color='k')
                plt.show(False) # false here means the window is non-blocking, so we may continue using defcon while the plot shows. 
            except RuntimeError, e:
                print "\033[91m [Warning] Error plotting expression. Are your solutions numbers rather than functions? If so, this is why I failed. Anyway, the error was: \033[00m"
                print str(e)
                pass

    ## Functions for saving to disk ##
    def save_movie(self, filename):
        """ Creates a matplotlib animation of the plotting up to the current maxtime. """
        print "Saving movie. This may take a little while..."
        print self.maxtime
        print len(self.points)
        # Fix the functional we're currently on, to avoid unplesantness if we try and change it while the movie is writing.
        self.func_index = self.current_functional

        # Set up the animated figure.
        self.anim_fig = plt.figure()
        self.ax = plt.axes()
        self.ax.clear()
        self.ax.set_xlabel(self.parameter_name)
        self.ax.set_ylabel(self.functional_names[self.func_index])
        self.anim = animation.FuncAnimation(self.anim_fig, self.animate, frames=self.maxtime-1, repeat=False, interval=0.1, blit=False)

        # Save it.
        mywriter = animation.FFMpegWriter()
        try: self.anim.save(filename, fps=30, writer=mywriter, extra_args=['-vcodec', 'libx264'])
        except Exception, e: 
            print "\033[91m[Warning] Saving mvoie failed. Perhaps you don't have ffmpeg installed? Anyway, the error was: \033[00m"
            print str(e)
            pass
        print "Movie saved."    

        self.ax.clear()

    def save_tikz(self, filename):
        """ Save the bfdiag window as a tikz plot. """
        if use_tikz:
            fig = plt.figure()
            ax = plt.axes()
            ax.clear()
            ax.set_xlabel(self.parameter_name)
            ax.set_ylabel(self.functional_names[self.current_functional])
            for xs, ys, branchid, teamno, cont in self.points:
                x = float(xs[self.freeindex])
                y = float(ys[self.current_functional])
                if cont: c, m= MAIN, '.'
                else: c, m= DEF, 'o'
                ax.plot(x, y, marker=m, color=c, linestyle='None')
            tikz_save(filename)
            ax.clear()
        else: print "\033[91m[Warning] matplotlib2tikz not installed. I can't save to tikz! \033[00m \n"

################################
### Custom matplotlib Figure ###
################################
class DynamicCanvas(FigureCanvas):
    """A canvas that updates itself with a new plot."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        FigureCanvas.__init__(self, figure)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_figure)
        self.timer.start(update_interval)

    def update_figure(self):
        self.timer.stop()
        redraw = pc.update()
        if redraw: self.draw()
        pc.seen()
        self.timer.start()

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
        self.layout().takeAt(4)

        
        self.buttonSaveMovie = self.addAction(QtGui.QIcon(resources_dir + "save_movie.png"), "Save Movie", self.save_movie)
        self.buttonSaveMovie.setToolTip("Save the figure as an animation")

        self.buttonSaveTikz= self.addAction(QtGui.QIcon(resources_dir + "save_tikz.png"), "Save Tikz", self.save_tikz)
        self.buttonSaveTikz.setToolTip("Save the figure as tikz")

    def save_figure(self):
        pc.start()
        NavigationToolbar2QT.save_figure(self)
        pc.end()

    def save_movie(self):
        pc.start()
        start = "bfdiag.mp4"
        filters = "FFMPEG Video (*.mp4)"
        selectedFilter = filters
 
        fname = QtGui.QFileDialog.getSaveFileName(self, "Choose a filename to save to", start, filters, selectedFilter)
        if fname:
            try:
                pc.save_movie(str(fname))
            except Exception, e:
                QtGui.QMessageBox.critical(self, "Error saving file", str(e), QtGui.QMessageBox.Ok, QtGui.QMessageBox.NoButton)
        pc.end()

    def save_tikz(self):
        pc.start()
        start = "bfdiag.tikz"
        filters = "Tikz Image (*.tex)"
        selectedFilter = filters
 
        fname = QtGui.QFileDialog.getSaveFileName(self, "Choose a filename to save to", start, filters, selectedFilter)
        if fname:
            try:
                pc.save_tikz(str(fname))
            except Exception, e:
                QtGui.QMessageBox.critical(self, "Error saving file", str(e), QtGui.QMessageBox.Ok, QtGui.QMessageBox.NoButton)
        pc.end()

######################
### Main QT Window ###
######################
class ApplicationWindow(QtGui.QMainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)     
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

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
        if darkmode: self.main_widget.setStyleSheet('color: #76EE00; background-color: black')


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
        lowerBox.setAlignment(QtCore.Qt.AlignCenter)
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
        self.dc = DynamicCanvas(self.main_widget, width=5, height=4, dpi=100)
        canvasBox.addWidget(self.dc)
        self.dc.mpl_connect('button_press_event', self.clicked_diagram)


        # Toolbar, with save_movie and save_tikz buttons.
        toolbar = CustomToolbar( self.dc, self )
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
        self.buttonPlot.clicked.connect(lambda:self.launch_paraview())
        self.buttonPlot.setEnabled(False)
        self.buttonPlot.setToolTip("Plot currently selected solution")
        self.buttonPlot.setFixedWidth(80)
        plotBox.addWidget(self.buttonPlot)

        # Unused plot buttons
        self.buttonPlotBranch = QtGui.QPushButton("Plot Branch")
        self.buttonPlotBranch.clicked.connect(lambda:self.launch_paraview())
        self.buttonPlotBranch.setEnabled(False)
        self.buttonPlotBranch.setToolTip("Plot all solutions in currently selected branch")
        #plotBox.addWidget(self.buttonPlotBranch)

        self.buttonParams = QtGui.QPushButton("Plot Params")
        self.buttonParams.clicked.connect(lambda:self.launch_paraview())
        self.buttonParams.setEnabled(False)
        self.buttonParams.setToolTip("Plot all solutions for currently selected parameter value")
        #plotBox.addWidget(self.buttonParams)


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
            if teamstats[i] == "d": colour = 'blue'
            if teamstats[i] == "c": colour = 'green'
            if teamstats[i] == "i": colour = 'yellow'
            if teamstats[i] == "q": colour = 'red'    
            text += "<p style='margin:0;' ><font color=%s size='+2'> Team %d</FONT></p>\n" % (colour, i)
        self.teambox.setText(text)

    def make_radio_buttons(self, functionals):
        """ Build the radio buttons for switching functionals. """
        for i in range(len(functionals)):
            radio_button = QtGui.QRadioButton(text=functionals[i])
            radio_button.clicked.connect(lambda: self.switch_functional())
            self.functionalBox.addWidget(radio_button)
            self.radio_buttons.append(radio_button)
        self.radio_buttons[0].setChecked(True)

    def switch_functional(self):
        """ Switch functionals. Which one we switch to depends on the radio button clicked. """
        i = 0
        for rb in self.radio_buttons:
            if rb.isChecked(): 
                pc.switch_functional(i)
                break
            else: i+=1

    def clicked_diagram(self, event):
        """ Annotates the diagram, by plotting a tooltip with the params and branchid of the point the user clicked.
            If the diagram is already annotated, remove the annotation. """
        annotated = pc.annotate(event.xdata, event.ydata)
        if annotated:
            self.buttonPlot.setEnabled(True)
            self.buttonPlotBranch.setEnabled(True)
            self.buttonParams.setEnabled(True)
        else:     
            self.buttonPlot.setEnabled(False)
            self.buttonPlotBranch.setEnabled(False)
            self.buttonParams.setEnabled(False)    

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
        t = int(self.jumpInput.text())
        new_time = pc.jump(t)
        self.set_time(new_time)

    def launch_paraview(self):
        """ Launch Paraview to graph the highlighted solution. """
        if not plot_with_mpl: pc.hdf52pvd()
        else: pc.mpl_plot()



#################
### Main Loop ###
#################
qApp = QtGui.QApplication(sys.argv)
aw = ApplicationWindow()
pc = PlotConstructor(aw)
aw.setWindowTitle("DEFCON")
aw.setWindowIcon(QtGui.QIcon(resources_dir + 'defcon_icon.png'))
aw.show()
sys.exit(qApp.exec_())
