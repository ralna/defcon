from bratu import BratuProblem
from defcon import *
import matplotlib.pyplot as plt

ac = ArclengthContinuation(problem=BratuProblem(), teamsize=1, verbose=True, logfiles=False)
ac.run(params=(0.018,), free="lambda", ds=0.01, sign=+1, bounds=(0.018, 3.6), branchids=[0])

ac.bifurcation_diagram("eval", "lambda", branchids=[0])
plt.title("The Bratu problem")
plt.savefig("arclength.pdf")
