import backend
if backend.__name__ == "dolfin":

    from backend import NonlinearProblem, derivative, SystemAssembler
    # I can't believe this isn't in DOLFIN.
    class GeneralProblem(NonlinearProblem):
        def __init__(self, F, y, bcs):
            NonlinearProblem.__init__(self)
            J = derivative(F, y)
            self.ass = SystemAssembler(J, F, bcs)

        def F(self, b, x):
            self.ass.assemble(b, x)

        def J(self, A, x):
            self.ass.assemble(A)

