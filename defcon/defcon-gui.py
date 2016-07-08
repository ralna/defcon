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

# Styles.
#from matplotlib import style
#style.use('ggplot')

# Set up the figure
figure = Figure(figsize=(5,4), dpi=100)
bfdiag = figure.add_subplot(111)
bfdiag.grid()

LARGE_FONT= ("Verdana", 12)

def animate(i):
    try:
        pullData = open("points_to_plot",'r').read()
    except Exception: 
        f = file("points_to_plot", 'w')
        f.close()
        pullData = ""

    dataList = pullData.split('\n')
    xList = []
    yList = []
    for eachLine in dataList:
        if len(eachLine) > 1:
            x, y = eachLine.split(',')
            bfdiag.plot(float(x), float(y), marker='.', color='k', linestyle='None')

class MainWindow(tk.Tk):
    def __init__(self, directory, *args, **kwargs):
        
        tk.Tk.__init__(self, *args, **kwargs)
        container = tk.Frame(self)

        container.pack(side="top", fill="both", expand = True)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (StartPage, BifurcationPage):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):

        frame = self.frames[cont]
        frame.tkraise()


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
        label = tk.Label(self, text="Start Page", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button = ttk.Button(self, text="Visit Page 1",
                            command=lambda: controller.show_frame(BifurcationPage))
        button.pack()

        
class BifurcationPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
        label = tk.Label(self, text="Bifurcation Diagram", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()


        canvas = FigureCanvasTkAgg(figure, self)
        canvas.show()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        toolbar = NavigationToolbar2TkAgg( canvas, self )
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)


        buttonStart = ttk.Button(self, text="Start",
                            command= lambda: sys.stdout.write("Start!"))
        buttonStart.pack()

        buttonPause = ttk.Button(self, text="Pause",
                            command= lambda: sys.stdout.write("Pause!"))
        buttonPause.pack()

        buttonClear = ttk.Button(self, text="Clear",
                            command= lambda: [bfdiag.clear(), bfdiag.grid()])
        buttonClear.pack()


app = MainWindow(directory=sys.argv[1])
ani = animation.FuncAnimation(figure, animate, interval=1)
app.mainloop()
