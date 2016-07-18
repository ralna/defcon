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

# Fonts.
LARGE_FONT = ("Verdana", 36)

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
update_interval = 20

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
figure = Figure(figsize=(5,4), dpi=100)
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

    V = problem.function_space(mesh)
    problem_parameters = problem.parameters()
else:
    print "In order to use paraview for graphing solutions, you must specify the class of the problem, eg 'NavierStokesProblem'."
    print("Usage: %s -p <problem type> -c <problem_class> -w <working dir> \n" % sys.argv[0])

# TODO: 1) Make the command line args less clunky.  
#       2) Add support for selecting multiple solutions? 
#       3) Add support for selecting and then graphing all solutions from one branch?
#       4) Tidy up code. Maybe move PlotConstructor to its own file? 
#       5) Grey out buttons that can't be pressed, or do something more sensible if we do press them. 
#       6) There's a lot of lag when the diagram is finished. Have defcon write something to the last line of the file to say it's finished???


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
        self.parameter_name = None

        self.current_sol = None

        self.app = app # The BifurcationPage window, so that we can set the time.

        self.done = False # Are we done?
    
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
        bfdiag.clear()
        bfdiag.grid(color=GRID)
        return self.time

    def end (self):
        if self.time < self.maxtime:
            xs = [point[0] for point in self.points[self.time:]]
            ys = [point[1] for point in self.points[self.time:]]
            bfdiag.plot(xs, ys, marker='.', color=MAIN, linestyle='None')
            self.time = self.maxtime
        if self.paused: self.unpause
        return self.maxtime

    def back(self):
        """ Take a step backwards in time. """
        if not self.paused: self.pause()
        if self.time > 0:
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
        if not self.paused: self.pause()
        if self.time < self.maxtime:
            x, y, branchid, params = self.points[self.time]
            self.time += 1
            bfdiag.plot(x, y, marker='.', color=MAIN, linestyle='None')
        if self.time==self.maxtime: self.pause
        return self.time

    def jump(self, t):
        """ Jump to time t. """
        if not self.paused: self.pause()
        if t <= self.maxtime:
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
        # If we're in pause mode, or we're done, then do nothing.
        if self.paused or self.done:
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
                    self.parameter_name = xlabel
                    self.functional_name = ylabel
                    bfdiag.set_xlabel(xlabel)
                    bfdiag.set_ylabel(ylabel)

                dataList = dataList[1:] # exclude the first line. 

                # Plot new points one at a time.
                for eachLine in dataList[self.time:]:
                    if eachLine=="Finished":
                        # FIXME: do something with this.
                        pass
                        #self.done = True
                    elif len(eachLine) > 1:
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
         if self.annotated_point is None:
             xs = [point[0] for point in self.points[:self.time]]
             ys = [point[1] for point in self.points[:self.time]]
 
             # FIXME: The *100 is because these were too small, might need some changing. Also doesn't work on, say allen-cahn, as xtol=0.
             xtol = 100*((max(xs) - min(xs))/float(len(xs)))/2 
             ytol = 100*((max(ys) - min(ys))/float(len(ys)))/2 

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
                 """self.annotation = bfdiag.annotate("%s=%.5f, Branch=%d" % (self.parameter_name, x, branchid),
                            xy = (x, y), xytext = (-20, 20),
                            textcoords = 'offset points', ha = 'right', va = 'bottom',
                            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5))
                            #arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))"""

                 self.annotation_highlight = bfdiag.scatter([x], [y], s=[50], marker='o', color=HIGHLIGHT) # Note: change 's' to make the highlight blob bigger/smaller

                 self.annotated_point = (literal_eval(params), branchid)  

                 app.set_output_box("Branch = %s\nx = %s\ny = %s" % (branchid, x, y))

                 return True
             else: return False

         else:
            self.annotation_highlight.remove()
            self.annotation_highlight = None
            self.annotated_point = None
            app.set_output_box("")
            return False

    def hdf5topvd(self):
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
    
        
##################
### Tk Classes ###
##################   

class CustomToolbar(NavigationToolbar2TkAgg):
    """ A custom matplotlib toolbar, so we can remove those pesky buttons. """  
    def __init__(self, canvas, parent):
        self.toolitems = (
            ('Home', 'Reset original view', 'home', 'home'),
            (None, None, None, None),
            ('Pan', 'Pan axes with left mouse, zoom with right', 'move', 'pan'),
            ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom'),
            (None, None, None, None),
            ('Save', 'Save the figure', 'filesave', 'save_figure'),
            )
        NavigationToolbar2TkAgg.__init__(self, canvas, parent)    
      
class BifurcationPage(tk.Tk):
    """ A page with a plot of the bifurcation diagram. """
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self,*args, **kwargs)
        label = tk.Label(self, text="DEFCON", font=LARGE_FONT, bg=BUTTONBG, fg=BUTTONTEXT)
        label.grid(row=0,column=4, columnspan=3)

        # Time label
        self.time_text = tk.StringVar()
        tk.Label(self, textvariable=self.time_text, bg=BUTTONBG, fg=BUTTONTEXT).grid(row=8, column=3)
        self.time_text.set("Time = 0")

        # Draw the canvas for the figure.
        canvas = FigureCanvasTkAgg(figure, self)
        canvas.show()
        canvas.get_tk_widget().grid(row=1, column=0, rowspan=7, columnspan=7, sticky = "nesw") #.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        #toolbar = CustomToolbar( canvas, self )
        #toolbar.update()
        canvas._tkcanvas.grid(row=1, column=0, rowspan=7, columnspan=7, sticky = "nesw")#pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Annotator
        canvas.mpl_connect('button_press_event', self.clicked_diagram)


        # Time navigation buttons
        self.buttonStart = tk.Button(self, text="|<", bg=BUTTONBG, fg=BUTTONTEXT, width=1, command=self.start)
        self.buttonStart.grid(row=8, column=1, sticky = "nesw")

        self.buttonBack = tk.Button(self, text="<", bg=BUTTONBG, fg=BUTTONTEXT, width=1, command=self.back)
        self.buttonBack.grid(row=8, column=2, sticky = "nesw")

        self.buttonForward = tk.Button(self, text=">", bg=BUTTONBG, fg=BUTTONTEXT, width=1, command=self.forward)
        self.buttonForward.grid(row=8, column=4, sticky = "nesw")

        self.buttonEnd = tk.Button(self, text=">|", bg=BUTTONBG, fg=BUTTONTEXT, width=1, command=self.end)
        self.buttonEnd.grid(row=8, column=5, sticky = "nesw")

        self.buttonJump = tk.Button(self, text="Jump", bg=BUTTONBG, fg=BUTTONTEXT, command=self.jump)
        self.buttonJump.grid(column=3, row=9, padx=10, sticky = "nesw")

        # Plot buttons
        self.buttonPlot = tk.Button(self, text="Plot", state="disabled", bg=BUTTONBG, fg=BUTTONTEXT, command=self.launch_paraview)
        self.buttonPlot.grid(row=8, column=8, sticky = "nesw")

        self.buttonPlotBranch = tk.Button(self, text="Plot branch", state="disabled", bg=BUTTONBG, fg=BUTTONTEXT, command=self.launch_paraview)
        self.buttonPlotBranch.grid(row=8, column=9, sticky = "nesw")

        self.buttonParams = tk.Button(self, text="Plot params", state="disabled", bg=BUTTONBG, fg=BUTTONTEXT, command=self.launch_paraview)
        self.buttonParams.grid(row=8, column=10, sticky = "nesw")

        # Output Box
        self.output_box_text = tk.StringVar()
        tk.Label(self, textvariable=self.output_box_text, bg=BUTTONBG, fg=BUTTONTEXT, justify="left").grid(row=5, column=8, rowspan=3, columnspan=3, sticky = "nesw")
        self.output_box_text.set("")



    def set_time(self, t):
        self.time_text.set("Time = %d" % t)

    def set_output_box(self, text):
        self.output_box_text.set(text)

    def clicked_diagram(self, event):
        """ Annotates the diagram, by plotting a tooltip with the params and branchid of the point the user clicked.
            If the diagram is already annotated, remove the annotation. """
        annotated = pc.annotate(event.xdata, event.ydata)
        if annotated: self.buttonPlot.config(state="normal")
        else:         self.buttonPlot.config(state="disabled")

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
for col in range(12):
    app.columnconfigure(col, weight=1)
for row in range(10):
    app.rowconfigure(row, weight=1)
app.title("DEFCON")
app.configure(bg=WINDOWBG)
#app.iconbitmap('path/to/icon.ico')

# Build and set up the animation object for the plot
pc = PlotConstructor(app)
ani = animation.FuncAnimation(figure, pc.animate, interval=update_interval)

# Start the app. 
app.mainloop()
