import Tkinter as tk
import ttk
import sys

# For plotting the bifurcation diagram.
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure

# Needed for animating the graph.
import matplotlib.animation as animation

# Styles for matplotlib. 
#from matplotlib import style
#style.use('ggplot')

# Set up the figure.
figure = Figure(figsize=(5,4), dpi=100)
bfdiag = figure.add_subplot(111)
bfdiag.grid()

# Fonts.
LARGE_FONT= ("Verdana", 12)

def animate(i):
    # Urgh. Create the file if it doesn't exist.
    try:
        pullData = open("points_to_plot",'r').read()
    except Exception: 
        f = file("points_to_plot", 'w')
        f.close()
        pullData = ""

    # Plot points one at a time.
    dataList = pullData.split('\n')
    xList = []
    yList = []
    for eachLine in dataList:
        if len(eachLine) > 1:
            x, y = eachLine.split(',')
            bfdiag.plot(float(x), float(y), marker='.', color='k', linestyle='None')

    # TODO: Draw the lines? Actually label the branches? 
    # FIXME: This is slow as hell, use blit to make it so we don't have to redraw everything each time.

class MainWindow(tk.Tk):
    def __init__(self, directory, *args, **kwargs):
        
        tk.Tk.__init__(self, *args, **kwargs)
        container = tk.Frame(self)

        container.pack(side="top", fill="both", expand = True)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        # Add the two pages. If we add more pages, this must be extended. 
        for F in (StartPage, BifurcationPage):
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

# TODO: Maybe define a page where we can see a list of solutions, by branch or params of something, and have a button to open paraview / have an embedded paraview window. 
class SolutionsPage(tk.Frame):
    """ A page where we can look at the solutions. """
    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
        label = tk.Label(self, text="Solutions", font=LARGE_FONT)
        label.pack(pady=10,padx=10)


# Main loop.
# TODO: Add scope for passing arguments, such as a directory to work in and so on. 
app = MainWindow(directory="")
ani = animation.FuncAnimation(figure, animate, interval=1) # Change interval to change the frequency of running diagram. FIXME: make this an option.
app.mainloop()
