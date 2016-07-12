import Tkinter as tk
import ttk
import tkSimpleDialog

import sys, getopt, os
from math import sqrt

# Imports for the paraview and hdf5topvd methods
from subprocess import Popen
import h5py as h5
from dolfin import *
from ast import literal_eval # FIXME: this is bad, get rid of it if possible. 
from parametertools import parameterstostring

# For plotting the bifurcation diagram.
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure

# Needed for animating the graph.
import matplotlib.animation as animation

# Styles for matplotlib.
# See matpoltlib.styles.available for options.
try:
    from matplotlib import style
    style.use('ggplot')
except AttributeError:
    print "Update to the latest version of matplotlib to use styles."

# Set some defaults.
problem_type = None
problem_class = None
working_dir="."
output_dir = "output"

# Fonts.
LARGE_FONT= ("Verdana", 12)

# Colours.
MAIN = 'k' # colour for points.
HIGHLIGHT = 'r' # colour for selected points.
GRID = 'w' # Colour for grid.

# Get commandline args.
# Example usage: python defcon-gui.py -p unity -c RootsOfUnityProblem -w /home/joseph/defcon/examples/unity
myopts, args = getopt.getopt(sys.argv[1:],"p:o:w:c:")

for o, a in myopts:
    if o == '-p':
        problem_type = a
    elif o == '-o':
        output_dir = a
    elif o == '-w':
        working_dir = a
    elif o == '-c':
        problem_class = a
    else:
        print("Usage: %s -p <problem_type> -c <problem_class> -w <working_dir> -o <defcon_output_directory>" % sys.argv[0])

output_dir = working_dir + os.path.sep + output_dir

# Set up the figure.
figure = Figure(figsize=(5,4), dpi=100)
bfdiag = figure.add_subplot(111)
bfdiag.grid(color=GRID)

# Put the working directory on our path.
sys.path.insert(0, working_dir) 
sys.path.insert(0, "~/defcon") #FIXME: seems to need this, even though the directory is in PYTHONPATH. Why, and how to get rid of it?

# If we've been told about the problem, then get the name and type of the problem we're dealing with, as well as everything else we're going to need for plotting solutions in paraview
if problem_type and problem_class:
    problem_name = __import__(problem_type)
    globals().update(vars(problem_name))
    globals()["bfprob"] = getattr(problem_name, problem_class)
    problem = bfprob()
    mesh = problem.mesh(mpi_comm_world())
    V    = problem.function_space(mesh)
    problem_parameters = problem.parameters()
else:
    print "In order to use paraview for graphing solutions, you must specify both the name and class of the problem."
    print("Usage: %s -p <problem type> -c <problem_class> -w <working dir> -o <defcon output directory> \n" % sys.argv[0])

# TODO: 1) Make the command line args less clunky, and set some sensible defaults for things (i.e, if we're working in ~/defcon/examples/unity, then problem_type should default to 'unity' if unset).  
#       2) Add support for selecting multiple solutions? 
#       3) Add support for selecting and then graphing all solutions from one branch?
#       4) Tidy up code. Maybe move PlotConstructor to its own file? 
#       5) Grey out buttons that can't be pressed, or do something more sensible if we do press them. 
#       6) There's a lot of lag when the diagram is finished. How about we add some kind of time limit (option?), so that if no points are found in within that limit we stop updating? 


#####################
### Utility Class ###
#####################

class PlotConstructor():
    """ Class for handling everything to do with the bifuraction diagram plot. """

    def __init__(self, app, directory = output_dir):
        self.points = [] # Keep track of the points we've found, so we can redraw everything if necessary. Also for annotation.

        self.maxtime = 0 # Keep track of the furthest we have got in time. 
        self.time = 0 # Keep track of where we currently are in time.

        self.paused = False # Are we updating we new points, or are we frozen in time?

        self.annotation = None # The annotation on the diagram. 
        self.annotation_highlight = None # The point we've annotated. 
        self.annotated_point = None # The (params, branchid) of the annotated point

        self.directory = directory # The working directory.

        self.freeindex = None # The index of the free parameter.

        self.current_sol = None

        self.app = app # The BifurcationPage window, so that we can set the time.
    
    def distance(self, x1, x2, y1, y2):
        """ Return the L2 distance between two points. """
        return(sqrt((x1 - x2)**2 + (y1 - y2)**2))

    def pause(self):
        """ Pause/unpause the drawing. """
        self.paused = not self.paused
        return self.paused

    def back(self):
        """ Take a step backwards in time. """
        if self.paused and self.time > 0:
            #FIXME: Extremely inefficient to replot everything
            xs = [point[0] for point in self.points[:self.time]]
            ys = [point[1] for point in self.points[:self.time]]
            self.time -= 1
            bfdiag.clear()
            bfdiag.grid(color=GRID)
            bfdiag.plot(xs, ys, marker='.', color=MAIN, linestyle='None') 
        return self.time


    def forward(self):
        """ Take a step forwards in time. """
        if self.paused and self.time < self.maxtime:
            x, y, branchid, params = self.points[self.time]
            self.time += 1
            bfdiag.plot(x, y, marker='.', color=MAIN, linestyle='None')
        return self.time

    def jump(self, t):
        """ Jump to time t. """
        if self.paused and t <= self.maxtime:
            #FIXME: Extremely inefficient to replot everything
            xs = [point[0] for point in self.points[:(t+1)]]
            ys = [point[1] for point in self.points[:(t+1)]]
            self.time = t
            bfdiag.clear()
            bfdiag.grid(color=GRID)
            bfdiag.plot(xs, ys, marker='.', color=MAIN, linestyle='None') 
        return self.time

    def grab_data(self):
        """ Get data from the file. """
        # If the file doesn't exist, just pass.
        try: pullData = open(self.directory + os.path.sep + "points_to_plot",'r').read()
        except Exception: pullData = None
        return pullData

    def animate(self, i):
        """ Handles the redrawing of the graph. """
        # If we're in pause mode, we do nothing.
        if self.paused:
            pass

        # If we're not paused, we draw all the points that have come in since we last drew something.
        else:   
            # Catch up to the points we have in memory.
            if self.time < self.maxtime:
                for x, y, branchid, params in self.points[self.time:]:
                    bfdiag.plot(x, y, marker='.', color=MAIN, linestyle='None')
                    self.time += 1

            # Get new points, if they exist. If not, just pass. 
            pullData = self.grab_data()
            if pullData is not None:
                dataList = pullData.split('\n')

                # Is this is first time, get the information from the first line of the data. 
                if self.freeindex is None: 
                    freeindex, xlabel, ylabel = dataList[0].split(';')
                    self.freeindex = int(freeindex)
                    bfdiag.set_xlabel(xlabel)
                    bfdiag.set_ylabel(ylabel)

                dataList = dataList[1:] # exclude the first line. 

                # Plot new points one at a time.
                for eachLine in dataList[self.time:]:
                    if len(eachLine) > 1:
                        params, y, branchid = eachLine.split(';')
                        x = literal_eval(params)[self.freeindex]
                        self.points.append((float(x), float(y), int(branchid), params))
                        bfdiag.plot(float(x), float(y), marker='.', color=MAIN, linestyle='None')
                        self.time += 1

                # Update the current time.
                self.maxtime = self.time
                self.app.set_time(self.time)         

    def annotate(self, clickX, clickY):
         """ Annotate a point when clicking on it. If there's already an annotation, remove it. """
         if self.annotation is None:
             xs = [point[0] for point in self.points[:self.time]]
             ys = [point[1] for point in self.points[:self.time]]
 
             # FIXME: The *10 is because these were too small, might need some changing.
             xtol = 10*((max(xs) - min(xs))/float(len(xs)))/2
             ytol = 10*((max(ys) - min(ys))/float(len(ys)))/2

             annotes = []

             # Find the point on the diagram closest to the point the user clicked.
             for x, y, branchid, params in self.points:
                  if ((clickX-xtol < x < clickX+xtol) and (clickY-ytol < y < clickY+ytol)):
                      annotes.append((self.distance(x, clickX, y, clickY), x, y, branchid, params))

             if annotes:
                 annotes.sort()
                 distance, x, y, branchid, params = annotes[0]

                 # Plot the annotation, and keep a handle on all the stuff we plot so we can use/remove it later. 
                 # FIXME: Make it prettier.
                 self.annotation = bfdiag.annotate("Parameter=%.5f, Branch=%d" % (x, branchid),
                            xy = (x, y), xytext = (-20, 20),
                            textcoords = 'offset points', ha = 'right', va = 'bottom',
                            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
                            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

                 self.annotation_highlight = bfdiag.scatter([x], [y], s=[50], marker='o', color=HIGHLIGHT) # Note: change 's' to make the highlight blob bigger/smaller

                 self.annotated_point = (literal_eval(params), branchid)      


         else:
            self.annotation.remove()
            self.annotation_highlight.remove()
            self.annotation = None
            self.annotation_highlight = None
            self.annotated_point = None

    def hdf5topvd(self):
        """ Utility function for creating a pvd from hdf5. Uses the point that is annotated. """
        if self.annotation is not None:
            # Get the params and branchid of the point.
            params, branchid = self.annotated_point    

            # Make a directory to put solutions in, if it doesn't exist. 
            try: os.mkdir(output_dir + os.path.sep + "solutions")
            except OSError: pass

            solutions_dir = output_dir + os.path.sep + "solutions" + os.path.sep

            # Create the file to which we will write these solutions.
            pvd_filename = solutions_dir +  "SOLUTION$params:%s$branchid=%d.pvd" % (parameterstostring(problem_parameters, params), branchid)
            pvd = File(pvd_filename)
    
            # Read the file, then write the solution.
            filename = output_dir + os.path.sep + parameterstostring(problem_parameters, params) + ".hdf5"
            with HDF5File(mesh.mpi_comm(), filename, 'r') as f:
                y = Function(V)
                f.read(y, "solution-%d" % branchid)
                f.flush()
                pvd << y

            self.launch_paraview(pvd_filename)

    def launch_paraview(self, filename):
        """ Utility function for launching paraview. Popen launches it in a separate process, so we may carry on with whatever we are doing."""
        Popen(["paraview", filename])
    
        
##################
### Tk Classes ###
##################       
      
class BifurcationPage(tk.Tk):
    """ A page with a plot of the bifurcation diagram. """
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self,*args, **kwargs)
        label = tk.Label(self, text="DEFCON", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        # Time label
        self.time_text = tk.StringVar()
        tk.Label(self, textvariable=self.time_text).pack(pady=10,padx=10)
        self.time_text.set("Time = 0")

        # Draw the canvas for the figure.
        canvas = FigureCanvasTkAgg(figure, self)
        canvas.show()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        toolbar = NavigationToolbar2TkAgg( canvas, self )
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Annotator
        canvas.mpl_connect('button_press_event', self.clicked_diagram)

        # Buttons.
        self.pause_text = tk.StringVar()
        self.pause_text.set("Pause")
        buttonPause = ttk.Button(self, textvariable=self.pause_text, command= self.pause)
        buttonPause.pack()

        # FIXME: Have these buttons greyed out when not paused.
        buttonBack = ttk.Button(self, text="Back", command= self.back)
        buttonBack.pack()

        buttonForward = ttk.Button(self, text="Forward", command= self.forward)
        buttonForward.pack()

        buttonJump = ttk.Button(self, text="Jump", command=self.jump)
        buttonJump.pack()

        buttonPlot = ttk.Button(self, text="Plot", command= self.launch_paraview)
        buttonPlot.pack()


        # TODO: Add a method for jumping to a particular time step. 


    def set_time(self, t):
        self.time_text.set("Time = %d" % t)

    def clicked_diagram(self, event):
        """ Annotates the diagram, by plotting a tooltip with the params and branchid of the point the user clicked.
            If the diagram is already annotated, remove the annotation. """
        pc.annotate(event.xdata, event.ydata)

    def pause(self):
        """ Pauses/Unpauses the plotting of the diagram, as well as changing the button text to whatever is appropriate. """
        paused = pc.pause()
        if paused: self.pause_text.set("Resume")
        else: self.pause_text.set("Pause")

    def back(self):
        """ Set Time=Time-1. """
        t = pc.back()
        self.set_time(t)

    def forward(self):
        """ Set Time=Time+1. """
        t = pc.forward()
        self.set_time(t)

    def jump(self):
        """ Jump to Time=t. """
        t = tkSimpleDialog.askinteger("Jump to", "Enter a time to jump to")
        print t
        if t is not None: 
            new_time = pc.jump(t)
            self.set_time(new_time)

    def launch_paraview(self):
        """ Launch Paraview to graph the highlighted solution. """
        pc.hdf5topvd()


############
### Main ###
############

# Construct the app, name it and give it an icon.
app = BifurcationPage()
app.title("DEFCON")
#app.iconbitmap('path/to/icon.ico')

# Build and set up the animation object for the plot
pc = PlotConstructor(app)
ani = animation.FuncAnimation(figure, pc.animate, interval=10) # Change interval to change the frequency of running diagram. FIXME: make this an option.

# Start the app. 
app.mainloop()
