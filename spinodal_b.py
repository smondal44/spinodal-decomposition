#for problem a of spinodal decomposition#
from __future__ import print_function
from dolfin import *
from fenics import *
import matplotlib.pyplot as plt
from mshr import *
from math import *
import numpy as np
import random
import csv
import os


L,H = 200,200
k,M,rhos,calpha,cbeta,theta = 2,5,5,0.3,0.7,0.5
c0 = 0.5
epsilon = 0.01
dt = 0.1
mesh = RectangleMesh(Point(0.,0.),Point(L,H),400,400,"crossed")
#periodic Boundary condition#
class PeriodicBoundary(SubDomain):

    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        # return True if on left or bottom boundary AND NOT on one of the two corners (0, 1) and (1, 0)
        return bool((near(x[0], 0) or near(x[1], 0)) and 
                (not ((near(x[0], 0) and near(x[1], 1)) or 
                        (near(x[0], 1) and near(x[1], 0)))) and on_boundary)

    def map(self, x, y):
        if near(x[0], 1) and near(x[1], 1):
            y[0] = x[0] - L
            y[1] = x[1] - H
        elif near(x[0], 1):
            y[0] = x[0] - L
            y[1] = x[1]
        else:   # near(x[1], 1)
            y[0] = x[0]
            y[1] = x[1] - H
class CahnHilliard(NonlinearProblem):
    def __init__(self,a,L):
        NonlinearProblem.__init__(self)
        self.L = L
        self.a = a
        #self.bcs = bcs
    def F(self,b,x):
        assemble(self.L,tensor=b)
        #[bc.apply(b) for bc in self.bcs]

    def J(self,A,x):
        assemble(self.a,tensor=A)
        #[bc.apply(A) for bc in self.bcs]
eta0 = Expression(("c0 + epsilon*(cos(0.105*x[0])*cos(0.11*x[1])+pow(cos(0.13*x[0])*cos(0.087*x[1]),2) + cos(0.025*x[0]-0.15*x[1])*cos(0.07*x[0]-0.02*x[1]))","0"),degree = 0,c0 = c0,epsilon = epsilon)
#eta0 = Expression(("c0 + epsilon*(cos(0.105*x[0])*cos(0.11*x[1])+pow(cos(0.13*x[0])*cos(0.0087*x[1]),2)+cos(0.025*x[0]-0.15*x[1])*cos(0.07*x[0]-0.02[x1]))","0"),degree = 0,c0 = c0,epsilon = epsilon)
#eta0 = Expression(("random.random()","0"),degree =0 )
d =1
Vne = FiniteElement('CG',mesh.ufl_cell(),d) #for eta 
Vme = FiniteElement('CG',mesh.ufl_cell(),d) #for mu
#V = FunctionSpace(mesh,MixedElement([Vne,Vme]),constrained_domain=PeriodicBoundary())
V = FunctionSpace(mesh,MixedElement([Vne,Vme]))
V1,V2 = TestFunction(V)
VTrial= TrialFunction(V)
(dc,dmu) = split(VTrial)
VOld = Function(V)

VNew = Function(V)
eta_init = eta0
VOld.interpolate(eta_init)
(c_0,mu_0) = split(VOld)
VNew.interpolate(eta_init)
(c,mu) = split(VNew)
##compute dfdn ###
c = variable(c)
f = rhos*(c-calpha)*(c-calpha)*(cbeta-c)*(cbeta-c)
dfdc = diff(f, c)
solver = NewtonSolver()
solver.parameters["linear_solver"]="mumps"
#solver.parameters["preconditioner"] = "ilu" #jacobi is slower than "ilu"
solver.parameters["convergence_criterion"]="incremental"
solver.parameters["relative_tolerance"]= 1e-8
#solver.parameters["krylov_solver"]["absolute_tolerance"]=1e-12
#solver.parameters["krylov_solver"]["relative_tolerance"]=1e-8
file1 = File ("spi_b/solution_.pvd","compressed")
#file2 = File ("sol_/Energy_.pvd","compressed")
t = 0.0
file1 << (VNew.split()[0], t)

#os.remove("Time_Total_Energy_Spinodal_a.csv")
Total_Energy = []
while True:
    TE = assemble(f*dx)+assemble(k/2*dot(grad(c),grad(c))*dx)
    Total_Energy =[t,TE]
    t += dt
    print(TE)
    VOld.vector()[:] = VNew.vector()
    mu_mid = (1.0-theta)*mu_0 + theta*mu
    cahn1 = (c - c_0)/dt*V1*dx + M*dot(grad(mu_mid),grad(V1))*dx
    cahn2 = mu*V2*dx -dfdc*V2*dx -k*dot(grad(c),grad(V2))*dx
    cahn = cahn1+cahn2
    a = derivative(cahn,VNew,VTrial)
    problem = CahnHilliard(a,cahn)
    (no_of_iterations,b) = solver.solve(problem,VNew.vector())
    if no_of_iterations <5:
        dt += 0.5*dt
    else:
        dt -= 0.5*dt
    with open('Time_Total_Energy_Spinodal_a.csv',mode='a') as csv2_file:
         writer = csv.writer(csv2_file)
         writer.writerow(Total_Energy)
    
    file1 << (VNew.split()[0], t)
    q = assemble(abs(mu - mu_0)*dx)/len(mesh.coordinates())
    print(q)
    if (q < 1e-10):
        break

print("Loop Terminated")
    



