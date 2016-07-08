import Tkinter as tk
import ttk
import sys
import math

# For plotting the bifurcation diagram.
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure

# Needed for animating the graph.
import matplotlib.animation as animation

# Styles for matplotlib. TODO: Play around with this, find something nice. 
#from matplotlib import style
#style.use('ggplot')

# Set up the figure.
figure = Figure(figsize=(5,4), dpi=100)
bfdiag = figure.add_subplot(111)
bfdiag.grid()

# FIXME: Get these as input, or something?
#bfdiag.xlabel="X axis"
#bfdiag.ylabel="Y axis"

# Fonts.
LARGE_FONT= ("Verdana", 12)

class PlotConstructor():
    """ Class for handling everything to do with the bifuraction diagram plot. """

    def __init__(self, directory = ".", label=False):
        self.points = [] # Keep track of the points we've found, so we can redraw everything if necessary. Also for annotation.

        self.maxtime = 0 # Keep track of the furthest we have got in time. 
        self.time = 0 # Keep track of where we currently are in time.

        self.paused = False # Are we updating we new points, or are we frozen in time?
        self.annotation = None # Have we annotated the diagram?

        self.directory = directory
        self.label = label
        

    def distance(self, x1, x2, y1, y2):
        """ Return the distance between two points. """
        return(math.sqrt((x1 - x2)**2 + (y1 - y2)**2))

    def pause(self):
        """ Pause/unpause the drawing. """
        self.paused = not self.paused

    def back(self):
        """ Take a step backwards in time. """
        raise NotImplementedError

    def forward(self):
        """ Take a step forwards in time. """
        raise NotImplementedError

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
            pullData = self.grab_data()
            # Get a list of points
            dataList = pullData.split('\n')

            # Plot points one at a time. We only want new points.
            for eachLine in dataList[self.maxtime:]:
                if len(eachLine) > 1:
                    x, y, branchid = eachLine.split(',')
                    self.points.append((float(x),float(y),int(branchid)))
                    bfdiag.plot(float(x), float(y), label=branchid, marker='.', color='k', linestyle='None')
                    self.maxtime += 1

            # Label all points
            # FIXME: Make this prettier. 
            if self.label:
                for x,y,branchid,self.maxtime in self.points[self.time:]:
                    bfdiag.annotate(
                        branchid, 
                        xy = (x, y), xytext = (-20, 20),
                        textcoords = 'offset points', ha = 'right', va = 'bottom',
                        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
                        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

            # Update the current time.
            self.time =self.maxtime

    def annotate(self, clickX, clickY):
         """ Annotate a point when clicking on it. If there's already an annotation, remove it. """
         if not self.annotation:
             xdata = [point[0] for point in self.points]
             ydata = [point[1] for point in self.points]
 
             # FIXME: The *10 is because these were too small, might need some changing.
             xtol = 10*((max(xdata) - min(xdata))/float(len(xdata)))/2
             ytol = 10*((max(ydata) - min(ydata))/float(len(ydata)))/2

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
             self.annotation = bfdiag.annotate("param=%.10f, branchid=%d" % (x, branchid),
                            xy = (x, y), xytext = (-20, 20),
                            textcoords = 'offset points', ha = 'right', va = 'bottom',
                            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
                            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
         else:
            self.annotation.remove()
            self.annotation = None
        
         

class MainWindow(tk.Tk):
    """ The class defining the window to be drawn. """
    def __init__(self, directory, *args, **kwargs):
        
        tk.Tk.__init__(self, *args, **kwargs)
        container = tk.Frame(self)

        container.pack(side="top", fill="both", expand = True)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        # Add the two pages. If we add more pages, this must be extended. 
        for F in (StartPage, BifurcationPage, SolutionsPage):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


class StartPage(tk.Frame):
    """ The page we see when the gui loads. """
    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
        label = tk.Label(self, text="Welcome to DEFCON", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        # To BD window
        button = ttk.Button(self, text="Bifurcation Diagram",
                            command=lambda: controller.show_frame(BifurcationPage))
        button.pack()

        # To solutions window
        button = ttk.Button(self, text="Solution Plotter",
                            command=lambda: controller.show_frame(SolutionsPage))
        button.pack()

        # Other stuff for this page goes here.

        
class BifurcationPage(tk.Frame):
    """ A page with a plot of the bifurcation diagram. """
    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
        label = tk.Label(self, text="Bifurcation Diagram", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        # Back to main window.
        buttonHome = ttk.Button(self, text="Back to main page",
                            command=lambda: controller.show_frame(StartPage))
        buttonHome.pack()

        # Draw the canvas for the figure.
        canvas = FigureCanvasTkAgg(figure, self)
        canvas.show()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        toolbar = NavigationToolbar2TkAgg( canvas, self )
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Annotator
        canvas.mpl_connect('button_press_event', self.onclick)

        # Buttons.
        buttonPause = ttk.Button(self, text="Pause",
                            command= lambda: sys.stdout.write("Pause!"))
        buttonPause.pack()#grid(row=0, column=0)

        buttonResume = ttk.Button(self, text="Resume",
                            command= lambda: sys.stdout.write("Resume!"))
        buttonResume.pack()#grid(row=0, column=1)

        buttonBack = ttk.Button(self, text="Back",
                            command= lambda: sys.stdout.write("Back!"))
        buttonBack.pack()#grid(row=1, column=0)

        buttonForward = ttk.Button(self, text="Forward",
                            command= lambda: sys.stdout.write("Forward!"))
        buttonForward.pack()#grid(row=1, column=1)

        buttonClear = ttk.Button(self, text="Clear",
                            command= lambda: [bfdiag.clear(), bfdiag.grid()])
        buttonClear.pack()


    def onclick(self, event):
        """ Annotates the diagram, by plotting a tooltip with the params and branchid of the point the user clicked.
            If the diagram is already annotated, remove the annotation. """
        pc.annotate(event.xdata, event.ydata)


# TODO: Maybe define a page where we can see a list of solutions, by branch or params or something, and have a button to launch paraview and visualise that solution.
# Maybe roll all this functionality into the bifurcation diagram page...
class SolutionsPage(tk.Frame):
    """ A page where we can look at the solutions. """
    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
        label = tk.Label(self, text="Solutions", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        # Back to main window.
        buttonHome = ttk.Button(self, text="Back to main page",
                            command=lambda: controller.show_frame(StartPage))
        buttonHome.pack()


# Main loop.
# TODO: Add scope for passing arguments, such as a directory to work in and so on. 

# Construct the app, name it and give it an icon.
app = MainWindow(directory="")
app.title("DEFCON")
#app.iconbitmap('path/to/icon.ico')

# Build and set up the animation object for the plot
pc = PlotConstructor()
ani = animation.FuncAnimation(figure, pc.animate, interval=1) # Change interval to change the frequency of running diagram. FIXME: make this an option.

# Start the app. 
app.mainloop()
