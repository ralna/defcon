# Urgh. We need this to ignore matplotlibs warnings.
import warnings
warnings.filterwarnings("ignore", module="matplotlib")

import matplotlib
matplotlib.use("Qt4Agg")

# Get the window and figure code.
# NOTE: The figure, colours and markers, and a couple of utility functions will be imported from here.
# If necessary, make changes in qtwindows.py.
from qtwindows import *

import sys, getopt

# Put the defcon directories on the path.
sys.path.insert(0, os.path.dirname(os.path.realpath(sys.argv[0])) + os.path.sep + "..")
sys.path.insert(0, os.path.dirname(os.path.realpath(sys.argv[0])) + os.path.sep + ".." + os.path.sep + "..")

import time as TimeModule

# Imports for the paraview and hdf5topvd methods.
from subprocess import Popen
from parametertools import parameterstostring

# We'll use literal_eval to get lists and tuples back from the journal. 
# This is not as bad as eval, as it only recognises: strings, bytes, numbers, tuples, lists, dicts, booleans, sets, and None.
# It can't do anything horrible if you pass it, for example, the string "os.system("rm -rf /")".
from ast import literal_eval 

# For saving movies.
from matplotlib import animation

# For plotting solutions if we don't use paraview, as well as for creating mvoies. 
import matplotlib.pyplot as plt

# Saving tikz pictures.
try: 
    from matplotlib2tikz import save as tikz_save
    use_tikz = True
except Exception: 
    issuewarning("Could not import the library matplotlib2tikz. You will unable to save the file as a .tikz.\nTo use this functionality, install matplotlib2tikz, eg with:\n     # pip install matplotlib2tikz")
    use_tikz = False

# Set some defaults.
problem_type = None
working_dir= None
output_dir = None
solutions_dir = None
xscale = None
plot_with_mpl = False # Whether we will try to plot solutions with matplotlib. If false, we use paraview.
update_interval = 100 # Update interval for the diagram.
resources_dir = os.path.dirname(os.path.realpath(sys.argv[0])) + os.path.sep + 'resources' + os.path.sep # The directory with the icons, etc. 

# Get commandline args.
def usage():
    sys.exit("""Usage: %s -p <problem type> -o <defcon output directory> -i <update interval in ms> -x <x scale> <working directory>
Required:
      The working directory. This is the location where your problem script is. 
Options:
      -p: The name of the script you use to run defcon. If not provided, this defualts to the last folder in the working directory.
          i.e, if the working directory is 'defcon/examples/elastica', we assume the name of the problem scipt is 'elastica'.
      -o: The directory that defcon uses for its output. The defaults to the "output" subdir of the working dir.
      -s: The directory to save solutions in. When you use paraview to visualise a solution, this is where it is saved. Defaults to the "solutions" subdir of the output dir.
      -i: The update interval of the bifurcation diagram in milliseconds. Defaults to 100.
      -x: The scale of the x-axis of the bifurcation diagram. This should be a valid matplotlib scale setting, eg 'log'.""" % sys.argv[0])

try: myopts, args = getopt.getopt(sys.argv[1:-1],"p:o:i:s:x:")
except Exception: usage()

for o, a in myopts:
    if   o == '-p': problem_type = a
    elif o == '-o': output_dir = os.path.expanduser(a)
    elif o == '-s': solutions_dir = os.path.expanduser(a)
    elif o == '-i': update_interval = int(a)
    elif o == '-x': xscale = a
    else          : usage()

# Get the working dir from the last command line argument.
working_dir = os.path.realpath(os.path.expanduser(sys.argv[-1]))

if working_dir is None:
    usage()

# If we didn't specify an output directory, default to the folder "output" in the working directory
if output_dir is None: output_dir = "output"

# If we didn't specify the name of the python file for the problem (eg, elastica), assume it's the same as the directory we're working in. 
if problem_type is None: 
    dirs = os.path.realpath(working_dir).split(os.path.sep)
    if dirs[-1]: problem_type = dirs[-1] # no trailing slash
    else: problem_type = dirs[-2] # trailing slash

# If we didn't specify a directory for solutions we plot, store them in the "solutions" subdir of the output directory.
if solutions_dir is None: solutions_dir = output_dir + os.path.sep + "solutions" + os.path.sep

# Get the current directory. 
current_dir = os.path.dirname(os.path.realpath(sys.argv[0]))

# Put the working directory on our path.
sys.path.insert(0, working_dir) 

# Get the name and type of the problem we're dealing with, as well as everything else we're going to need for plotting solutions.
problem_name = __import__(problem_type)
globals().update(vars(problem_name))

# Run through each class we've imported and figure out which one inherits from BifurcationProblem.
classes = []
for key in globals().keys():
    if key is not 'BifurcationProblem': classes.append(key) # remove this to make sure we don't fetch the wrong class.
for c in classes:
    try:
        globals()["bfprob"] = getattr(problem_name, c)
        assert issubclass(bfprob, BifurcationProblem) and bfprob is not BifurcationProblem # check whether the class is a subclass of BifurcationProblem, which would mean it's the class we want. 
        problem = bfprob() # initialise the class.
        break
    except Exception: pass

os.chdir(working_dir)

# Get the mesh.
mesh = problem.mesh(mpi_comm_world())

# If the mesh is 1D, we don't want to use paraview. 
if mesh.geometry().dim() < 2: plot_with_mpl = True 

# Get the function space and set up the I/O module for fetching solutions. 
V = problem.function_space(mesh)
problem_parameters = problem.parameters()
io = problem.io(prefix=working_dir + os.path.sep)
io.setup(problem_parameters, None, V)
os.chdir(current_dir)

#####################
### Utility Class ###
#####################
class PlotConstructor():
    """ Class for handling everything to do with the bifuraction diagram plot. """

    def __init__(self):
        self.points = [] # Keep track of the points we've found, so we can redraw everything if necessary. Also for annotation.
        self.pointers = [] # Pointers to each point on the plot, so we can remove them. 

        self.maxtime = 0 # Keep track of the furthest we have got in time. 
        self.time = 0 # Keep track of where we currently are in time.
        self.lines_read = 0 # Number of lines of the journal file we've read.

        self.paused = False # Are we updating we new points, or are we frozen in time?

        self.annotation_highlight = None # The point we've annotated. 
        self.annotated_point = None # The (params, branchid) of the annotated point

        self.path = working_dir + os.path.sep + output_dir + os.path.sep + "journal" + os.path.sep +"journal.txt" # Journal file.

        self.freeindex = None # The index of the free parameter.

        self.current_functional = 0 # The index of the functional we're currently on.

        self.teamstats = [] # The status of each team. 

        self.sweep = 0 # The parameter value we've got up to.
        self.sweepline = None # A pointer to the line object that shows how far we've got up to.
        self.sweeplines = [] # Keep track of where the sweepline is at each time step, so we can draw it in the movie. 

        self.changed = False # Has the diagram changed since our last update?

        self.start_time = 0 # The (system) time at which defcon started running.
        self.running = False # Is defcon running?
    
    ## Private utility functions. ##
    def distance(self, x1, x2, y1, y2):
        """ Return the L2 distance between two points. """
        return sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def setx(self, ax):
        """ Sets the xscale to the user defined variable. """
        try:
            if xscale is not None: ax.set_xscale(xscale)
        except Exception:
            issuewarning("User-provided xscale variable is not a valid option for matplotlib.")
            pass

    def redraw(self):
        "Clears the window, redraws the labels and the grid. """
        bfdiag.clear()

        # Redraw labels.
        bfdiag.set_xlabel(self.parameter_name)
        bfdiag.set_ylabel(self.functional_names[self.current_functional])
        bfdiag.grid(color=GRID)

        # Reset the x and y limits.
        ys = [point[1][self.current_functional] for point in self.points] 
        try:
            bfdiag.set_xlim([self.minparam, self.maxparam]) # fix the limits of the x-axis
            bfdiag.set_ylim([min(ys), max(ys)]) # reset the y limits, to prevent stretching
        except Exception: pass

        # Redraw the sweepline.
        self.sweepline = bfdiag.axvline(x=self.sweep, linewidth=1, linestyle=SWEEPSTYLE, color=SWEEP)

        # Reset the scale of the x axis. 
        self.setx(bfdiag)

    def animate(self, i):
        """ Utility function for animating a plot. """
        for j in range(0, self.dots_per_frame):
            try:
                xs, ys, branchid, teamno, cont = self.points_iter.next()
                x = float(xs[self.freeindex])
                y = float(ys[self.func_index])
                self.ax.plot(x, y, marker=CONTPLOT, color=MAIN, linestyle='None')
                self.animsweep = self.sweeplines_iter.next()
                if self.animsweepline is not None: self.animsweepline.remove()
                self.animsweepline = self.ax.axvline(x=self.animsweep, linewidth=1, linestyle=SWEEPSTYLE, color=SWEEP)                
            except StopIteration: return

        # Let's output a log of how we're doing, so the user can see that something is in fact being done.
        if i % 50 == 0: print "Completed %d/%d frames" % (i, self.frames)
        return       
         

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
                if cont: c, m= MAIN, CONTPLOT
                else: c, m = DEF, DEFPLOT
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
            if cont: c, m= MAIN, CONTPLOT
            else: c, m= DEF, DEFPLOT
            self.pointers[self.time] = bfdiag.plot(x, y, marker=m, color=c, linestyle='None')
            self.time += 1
            self.changed = True
        if self.time==self.maxtime: self.unpause()
        return self.time

    def jump(self, t):
        """ Jump to time t. """
        if not self.paused: self.pause()
        if self.annotated_point is not None: self.unannotate()

        # Case 1: we're going backwards in time, and need to remove points.
        if t < self.time:
            for i in range(t, self.time):
                self.pointers[i][0].remove()
        
        # Case 2: we're going forwards in time, and need to re-plot some points.
        elif t > self.time:
            for i in range(self.time, t):
                xs, ys, branchid, teamno, cont = self.points[i]
                x = float(xs[self.freeindex])
                y = float(ys[self.current_functional])
                if cont: c, m= MAIN, CONTPLOT
                else: c, m= DEF, DEFPLOT
                self.pointers[i] = bfdiag.plot(x, y, marker=m, color=c, linestyle='None')

        self.time = t 
        self.changed = True
        if self.time==self.maxtime: self.unpause()
        return self.time

    
    def switch_functional(self, funcindex):
        """ Change the functional being plotted. """
        self.current_functional = funcindex
        if self.annotated_point is not None: self.unannotate()
        self.redraw() # wipe the diagram.

        # Redraw all points up to the current time. 
        for j in range(0, self.time):
            xs, ys, branchid, teamno, cont = self.points[j]
            x = float(xs[self.freeindex])
            y = float(ys[self.current_functional])
            if cont: c, m= MAIN, CONTPLOT
            else: c, m= DEF, DEFPLOT
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

        # Update the run time.
        if self.running: aw.set_elapsed_time(TimeModule.time() - self.start_time) # if defcon is still going, update the timer

        # If we're in pause mode then do nothing more.
        if self.paused:
            return self.changed

        # If we're not paused, we draw all the points that have come in since we last drew something.
        else:   
            # Get new points, if they exist. If not, just pass. 
            pullData = self.grab_data()
            if pullData is not None:
                dataList = pullData.split('\n')

                # Is this is first time we've found data, get the information from the first line of the journal. 
                if self.freeindex is None:
                    self.running = True
                    self.start_time = TimeModule.time() # defcon has just started, so get the start time. 

                    freeindex, self.parameter_name, functional_names, unicode_functional_names, nteams, minparam, maxparam = dataList[0].split(';')
                    self.minparam = float(minparam)
                    self.maxparam = float(maxparam)

                    # Set up info about what the teams are doing.
                    self.nteams = int(nteams)
                    for team in range(self.nteams): self.teamstats.append(('i', 'None', 'None'))
                    aw.set_teamstats(self.nteams)

                    # Info about functionals.
                    self.freeindex = int(freeindex)
                    self.functional_names = literal_eval(functional_names)
                    self.unicode_functional_names = literal_eval(unicode_functional_names)
                    aw.make_radio_buttons(self.unicode_functional_names)

                    # Set the labels and scales of the axes.
                    bfdiag.set_xlabel(self.parameter_name)
                    bfdiag.set_ylabel(self.functional_names[self.current_functional])
                    bfdiag.set_xlim([self.minparam, self.maxparam]) # fix the limits of the x-axis
                    bfdiag.autoscale(axis='y')
                    self.setx(bfdiag)

                dataList = dataList[1:] # exclude the first line from now on.  

                # Plot new points one at a time.
                for eachLine in dataList[self.lines_read:]:
                    if len(eachLine) > 1:
                        if eachLine[0] == '$':
                            # This line of the journal is telling us about the sweep line.
                            self.sweep = float(eachLine[1:])
                            if self.sweepline is not None: self.sweepline.remove()
                            self.sweepline = bfdiag.axvline(x=self.sweep, linewidth=1, linestyle=SWEEPSTYLE, color=SWEEP)

                        elif eachLine[0] == '~':
                            # This line of the journal is telling us about what the teams are doing. 
                            team, task, params, branchid = eachLine[1:].split(';')
                            self.teamstats[int(team)] = (task, params, branchid)
                            aw.update_teamstats(self.teamstats)
                            # If this tells us the teams have quit, we know we're not getting any more new points. 
                            if task == 'q': self.running = False
                                                             
                        else:
                            # This is a newly discovered point. Get all the information we need.
                            teamno, oldparams, branchid, newparams, functionals, cont = eachLine.split(';')
                            xs = literal_eval(newparams)
                            ys = literal_eval(functionals)
                            x = float(xs[self.freeindex])
                            y = float(ys[self.current_functional])
                            
                            # Use different colours/plot styles for points found by continuation/deflation.
                            if literal_eval(cont): c, m= MAIN, CONTPLOT
                            else: c, m= DEF, DEFPLOT

                            # Keep track of the points we've discovered, as well as the matplotlib objects. 
                            self.points.append((xs, ys, int(branchid), int(teamno), literal_eval(cont)))                            
                            self.pointers.append(bfdiag.plot(x, y, marker=m, color=c, linestyle='None'))
                            self.sweeplines.append(self.sweep)
                            self.time += 1
                        self.lines_read +=1

                # Update the current time.
                self.changed = True
                self.maxtime = self.time
                aw.set_time(self.time)
                return self.changed         


    ## Functions for handling annotation. ##
    def annotate(self, clickX, clickY):
        """ Annotate a point when clicking on it. """
        if clickX is None and clickY is None:
           if self.annotated_point is not None:
               self.unannotate()
           return True

        # Sets a clickbox. 
        xs = [float(point[0][self.freeindex]) for point in self.points[:self.time]]
        ys = [float(point[1][self.current_functional]) for point in self.points[:self.time]]

        xlen = max(xs) - min(xs)
        ylen = max(ys) - min(ys)

        xtick = bfdiag.get_xticks()
        ytick = bfdiag.get_yticks()
        xtol = (xtick[1]-xtick[0])/(2)
        ytol = (ytick[1]-ytick[0])/(2)

        annotes = []

        # Find the point on the diagram closest to the point the user clicked.
        time = 1
        for xs, ys, branchid, teamno, cont in self.points[:self.time]:
             x = float(xs[self.freeindex])
             y = float(ys[self.current_functional])
             if ((clickX-xtol < x < clickX+xtol) and (clickY-ytol < y < clickY+ytol)):
                 annotes.append((self.distance(x/xlen, clickX/xlen, y/ylen, clickY/ylen), x, y, branchid, xs, teamno, cont, time)) # uses rescaled distance. 
             time += 1

        if annotes:
            annotes.sort()
            distance, x, y, branchid, xs, teamno, cont, time = annotes[0]

            if self.annotated_point is not None:
                self.unannotate()

            # Plot the annotation, and keep a handle on all the stuff we plot so we can use/remove it later. 
            self.annotation_highlight = bfdiag.scatter([x], [y], s=[50], marker='o', color=HIGHLIGHT) # Note: change 's' to make the highlight blob bigger/smaller
            self.annotated_point = (xs, branchid)  
            if cont: s = "continuation"
            else: s = "deflation"
            aw.set_output_box("Solution on branch %d\nFound by team %d\nUsing %s\nAs event #%d\n\nx = %s\ny = %s" % (branchid, teamno, s, time, x, y))
            self.changed = True
        return self.changed

    def unannotate(self):
        """ Remove annotation from the graph. """
        self.annotation_highlight.remove()
        self.annotation_highlight = None
        self.annotated_point = None
        aw.set_output_box("")
        self.changed = True
        return False

    def plot(self):
        """ Fetch a solution and plot it. If the solutions are 1D we use matplotlib, otherwise we use paraview. """
        if self.annotated_point is not None:
            os.chdir(working_dir)
            # Get the solution from the IO module. 
            params, branchid = self.annotated_point
            y = io.fetch_solutions(params, [branchid])[0]

            if plot_with_mpl:
                try:
                    x = interpolate(Expression("x[0]", degree=1), V)
                    # FIXME: For functions f other than CG1, we might need to sort both arrays so that x is increasing. Check this out!
                    plt.plot(x.vector().array(), y.vector().array(), '-', linewidth=3, color='b')
                    plt.title("branch %s, params %s" % (branchid, params))
                    plt.axhline(0, color='k') # Plot a black line through the origin
                    plt.show(False) # False here means the window is non-blocking, so we may continue using the GUI while the plot shows. 
                except RuntimeError, e:
                    issuewarning("Error plotting expression. Are your solutions numbers rather than functions? If so, this is why I failed. The error was:")
                    print str(e)
                    pass
            else:
                # Make a directory to put solutions in, if it doesn't exist. 
                try: os.mkdir(solutions_dir)
                except OSError: pass

                # Create the file to which we will write these solutions.
                pvd_filename = solutions_dir + "SOLUTION$%s$branchid=%d.pvd" % (parameterstostring(problem_parameters, params), branchid)
                pvd = File(pvd_filename)

                # Write the solution.
                problem.save_pvd(y, pvd)

                # Finally, launch paraview with the newly created file. 
                # If this fails, issue a warning. 
                try: Popen(["paraview", pvd_filename])
                except Exception, e:
                    issuewarning("Oops, something went wrong with launching paraview. Are you sure you have it installed and on your PATH? The error was:")
                    print str(e)

            os.chdir(current_dir)

    ## Functions for saving to disk ##
    def save_movie(self, filename, length, fps):
        """ Creates a matplotlib animation of the plotting up to the current maxtime. """

        # Fix the functional we're currently on, to avoid unpleasantness if we try and change it while the movie is writing.
        self.func_index = self.current_functional

        # Make an iterator of the points list and sweepline list.
        self.points_iter = iter(self.points)
        self.sweeplines_iter = iter(self.sweeplines)

        self.animsweepline = None

        # Set up the animated figure.
        self.anim_fig = plt.figure()
        self.ax = plt.axes()
        self.ax.clear()

        self.ax.set_xlabel(self.parameter_name)
        self.ax.set_xlim([self.minparam, self.maxparam]) # fix the x-limits
        self.setx(self.ax)

        self.ax.set_ylabel(self.functional_names[self.func_index])
        ys = [point[1][self.current_functional] for point in self.points] 
        self.ax.set_ylim([min(ys), max(ys)]) # fix the y-limits

        # Work out how many frames we want.
        self.frames = length * fps

        # Sanity check: if the number of desired frames is greater than the number of points, better fix this by adjusting fps.
        if self.frames > self.maxtime:
            self.frames = self.maxtime
            fps = int(self.frames / float(length))

        self.dots_per_frame = int(ceil(float(self.maxtime) / self.frames))

        # Create and save the animation. 
        print "Saving movie. This may take a while..."
        try:
            self.anim = animation.FuncAnimation(self.anim_fig, self.animate, frames=self.frames, interval=1)
            mywriter = animation.FFMpegWriter(fps=fps, bitrate=5000)
            self.anim.save(filename, fps=fps, dpi=200, bitrate=5000, writer=mywriter, extra_args=['-vcodec', 'libx264'])
            print "Movie saved."    
            return
        except Exception, e: 
            issuewarning("Saving movie failed. Perhaps you don't have ffmpeg installed? Anyway, the error was:")
            print str(e)
            return

    def save_tikz(self, filename):
        """ Save the bfdiag window as a tikz plot. """
        if use_tikz:
            fig = plt.figure()

            ax = plt.axes()
            ax.clear()

            # Set up the x-axis.
            ax.set_xlabel(self.parameter_name)
            ax.set_xlim([self.minparam, self.maxparam])
            self.setx(ax)

            # Set up the y-axis.
            ax.set_ylabel(self.functional_names[self.current_functional])
            ys = [point[1][self.current_functional] for point in self.points] 
            ax.set_ylim([floor(min(ys)), ceil(max(ys))])

            for xs, ys, branchid, teamno, cont in self.points:
                x = float(xs[self.freeindex])
                y = float(ys[self.current_functional])
                if cont: c, m= MAIN, '.'
                else: c, m= DEF, 'o'
                ax.plot(x, y, marker=m, color=c, linestyle='None')
            tikz_save(filename)
            ax.clear()
        else: issuewarning("matplotlib2tikz not installed. I can't save to tikz!")

    
#################
### Main Loop ###
#################
qApp = QtGui.QApplication(sys.argv)
pc = PlotConstructor()
aw = ApplicationWindow(pc, update_interval, resources_dir, working_dir)
aw.setWindowTitle("DEFCON")
aw.setWindowIcon(QtGui.QIcon(resources_dir + 'defcon_icon.png'))
aw.show()
sys.exit(qApp.exec_())
