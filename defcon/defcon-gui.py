import Tkinter as tk
import ttk

import sys, getopt
from math import sqrt

from subprocess import Popen
import h5py as h5

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

# TODO: Get command line arguments.

# Fonts.
LARGE_FONT= ("Verdana", 12)

# Colours.
MAIN = 'k' # colour for points.
HIGHLIGHT = 'r' # colour for selected points.
GRID = 'w' # Colour for grid.

# Set up the figure.
figure = Figure(figsize=(5,4), dpi=100)
bfdiag = figure.add_subplot(111)
bfdiag.grid(color=GRID)

# FIXME: Get these as input, or something?
#bfdiag.xlabel="X axis"
#bfdiag.ylabel="Y axis"



####################################
### Utility Function Definitions ###
####################################

# TODO: We need to know what kind of problem we are handling, and what the function space is, so we can change from hdf5 to pvd.

# FIXME: get this and the paraview function working properly
def hdf5topvd(problem_type, params, branchid):
    """ Utility function for creating a pvd from hdf5. """
    problem_name = __import__(problem_type)
    globals().update(vars(problem_name))

    problem = RootsOfUnityProblem() # FIXME: get this to be whatever is correct
    mesh = problem.mesh(mpi_comm_world())
    V    = problem.function_space(mesh)

    dir = "output" + os.path.sep 
    filename = dir 
    pvd = File(dir + "solutions.pvd")

    # Find the keys
    f = h5.File(filename, 'r')
    solns = sorted(f.keys())
    f.close()

    with HDF5File(mesh.mpi_comm(), filename, 'r') as f:
        for soln in solns:
            print soln
            y = Function(V)
            f.read(y, str(soln))
            print y
            f.flush()
            pvd << y

def paraview(filename):
    """ Utility function for launching paraview. Popen launches it in a separate process, so we may carry on with whatever we are doing."""
    Popen("paraview %s" % filename)
    

#####################
### Utility Class ###
#####################

class PlotConstructor():
    """ Class for handling everything to do with the bifuraction diagram plot. """

    def __init__(self, app, directory = "."):
        self.points = [] # Keep track of the points we've found, so we can redraw everything if necessary. Also for annotation.

        self.maxtime = 0 # Keep track of the furthest we have got in time. 
        self.time = 0 # Keep track of where we currently are in time.

        self.paused = False # Are we updating we new points, or are we frozen in time?

        self.annotation = None # The annotation on the diagram. 
        self.annotationpoint = None # The point we've annotated. 

        self.directory = directory # The working directory.

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
        else:
            pass
        return self.time


    def forward(self):
        """ Take a step forwards in time. """
        if self.paused and self.time < self.maxtime:
            x, y, branchid = self.points[self.time]
            self.time += 1
            bfdiag.plot(x, y, marker='.', color=MAIN, linestyle='None')
        else:
            pass
        return self.time

    def grab_data(self):
        """ Get data from the file. """
        # Urgh. Create the file if it doesn't exist.
        try:
            pullData = open("points_to_plot",'r').read()
        except Exception: 
            f = file("points_to_plot", 'w')
            f.close()
            pullData = ""
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
                for x, y, branchid in self.points[self.time:]:
                    bfdiag.plot(x, y, marker='.', color=MAIN, linestyle='None')
                    self.time += 1

            # Get new points.
            pullData = self.grab_data()
            dataList = pullData.split('\n')

            # Plot new points one at a time.
            for eachLine in dataList[self.time:]:
                if len(eachLine) > 1:
                    x, y, branchid = eachLine.split(',')
                    self.points.append((float(x),float(y),int(branchid)))
                    bfdiag.plot(float(x), float(y), marker='.', color=MAIN, linestyle='None')
                    self.time += 1

            # Update the current time.
            self.maxtime = self.time
            self.app.set_time(self.time)         

    def annotate(self, clickX, clickY):
         """ Annotate a point when clicking on it. If there's already an annotation, remove it. """
         if not self.annotation:
             xs = [point[0] for point in self.points[:self.time]]
             ys = [point[1] for point in self.points[:self.time]]
 
             # FIXME: The *10 is because these were too small, might need some changing.
             xtol = 10*((max(xs) - min(xs))/float(len(xs)))/2
             ytol = 10*((max(ys) - min(ys))/float(len(ys)))/2

             annotes = []

             # Find the point on the diagram closest to the point the user clicked.
             for x, y, branchid in self.points:
                  if ((clickX-xtol < x < clickX+xtol) and (clickY-ytol < y < clickY+ytol)):
                      annotes.append((self.distance(x, clickX, y, clickY), x, y, branchid))

             if annotes:
                 annotes.sort()
                 distance, x, y, branchid = annotes[0]

                 # Plot the annotation
                 # FIXME: Make it prettier.
                 self.annotation = bfdiag.annotate("Parameter=%.5f, Branch=%d" % (x, branchid),
                            xy = (x, y), xytext = (-20, 20),
                            textcoords = 'offset points', ha = 'right', va = 'bottom',
                            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
                            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

                 self.annotationpoint = bfdiag.scatter([x], [y], marker='o', color=HIGHLIGHT)        


         else:
            self.annotation.remove()
            self.annotationpoint.remove()
            self.annotation = None
            self.annotationpoint = None
        
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
        paused = pc.pause()
        if paused: self.pause_text.set("Resume")
        else: self.pause_text.set("Pause")

    def back(self):
        t = pc.back()
        self.set_time(t)

    def forward(self):
        t = pc.forward()
        self.set_time(t)

    def jump_to(self, t):
        raise NotImplementedError

    def launch_paraview(self):
        print "Launch paraview."


############
### Main ###
############

# Construct the app, name it and give it an icon.
app = BifurcationPage()
app.title("DEFCON")
#app.iconbitmap('path/to/icon.ico')

# Build and set up the animation object for the plot
pc = PlotConstructor(app)
ani = animation.FuncAnimation(figure, pc.animate, interval=1) # Change interval to change the frequency of running diagram. FIXME: make this an option.

# Start the app. 
app.mainloop()
