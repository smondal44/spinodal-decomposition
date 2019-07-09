# Spinodal decomposition with square shape domain( prob. a of spinodal decomposition problem of pfhub)  
# Here, periodic boundary condition has been applied.
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

L,H = 200,200 #size of the domain
# model parameter
k,M,rhos,calpha,cbeta,theta = 2,5,5,0.3,0.7,0.5 
c0 = 0.5
epsilon = 0.01
dt = 0.1 #initial dt
Nx,Ny = 400,400 #number of nodes in the x and y direction o fthe domain
mesh = RectangleMesh(Point(0.,0.),Point(L,H),Nx,Ny,"crossed") #mesh generation
#periodic Boundary condition#
class PeriodicBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return bool((near(x[0], 0) or near(x[1], 0)) and 
                (not ((near(x[0], 0) and near(x[1], H)) or 
                        (near(x[0], L) and near(x[1], 0)))) and on_boundary)
    def map(self, x, y):
        if near(x[0], L) and near(x[1], H):
            y[0] = x[0] - L
            y[1] = x[1] - H
        elif near(x[0], 1):
            y[0] = x[0] - L
            y[1] = x[1]
        else:
            y[0] = x[0]
            y[1] = x[1] - H
# To assemble the jacobian and non-linear form the problem            
class CahnHilliard(NonlinearProblem):  
    def __init__(self,a,L):
        NonlinearProblem.__init__(self)
        self.L = L
        self.a = a
    def F(self,b,x):
        assemble(self.L,tensor=b)
    def J(self,A,x):
        assemble(self.a,tensor=A)
#initial codition for the domain is expressed using C++ expression, all the constants other than x[0],x[1] have to mentioned 
#just after mentioning the degree of the expression
eta0 = Expression(("c0 + epsilon*(cos(0.105*x[0])*cos(0.11*x[1])+pow(cos(0.13*x[0])*cos(0.087*x[1]),2) + cos(0.025*x[0]-0.15*x[1])*cos(0.07*x[0]-0.02*x[1]))","0"),degree = 0,c0 = c0,epsilon = epsilon)
d =1   # degree of the functionspace of the varible to be solved
Vne = FiniteElement('CG',mesh.ufl_cell(),d)  #scalar finite elment for eta 
Vme = FiniteElement('CG',mesh.ufl_cell(),d)  #scalar finite elment for mu
# mixing the two finite elment to make the functionspace, here sequence of mixing is important
V = FunctionSpace(mesh,MixedElement([Vne,Vme]),constrained_domain=PeriodicBoundary())
V1,V2 = TestFunction(V) #V1,V2 are the test function for eta and mu resspectively
VTrial= TrialFunction(V)
(dc,dmu) = split(VTrial) # splitting the trial functionn for c and mu
VOld = Function(V)       # Function to store the solution of the previous time step
VNew = Function(V)       # Function to store the solution of the current time step
VOld.interpolate(eta0)   # interpolating the initial condition to the functuion of previous time step
(c_0,mu_0) = split(VOld) # splitting
VNew.interpolate(eta0)   # interpolating the initial condition to the functuion of current time step
(c,mu) = split(VNew)     # splitting

# compute the bulk free energy and the variational derivateive of bulk free energy
c = variable(c)
f = rhos*(c-calpha)*(c-calpha)*(cbeta-c)*(cbeta-c)
dfdc = diff(f, c)

# To have control over the solver (newton solver)
solver = NewtonSolver()
solver.parameters["linear_solver"]="lu"
#solver.parameters["preconditioner"] = "ilu" #jacobi is slower than "ilu"
solver.parameters["convergence_criterion"]="incremental"
solver.parameters["relative_tolerance"]= 1e-8
#solver.parameters["krylov_solver"]["absolute_tolerance"]=1e-12
#solver.parameters["krylov_solver"]["relative_tolerance"]=1e-8
file1 = File ("spinodal_sol_c_conservative/solution_.pvd","compressed") # name of the filw where the solution of each time will be stored
t = 0.0 # initial time
file1 << (VNew.split()[0], t)   # storing the initial value in the file
Total_Energy = []
#Time stepping loop
while True:
    TE = assemble(f*dx)+assemble(k/2*dot(grad(c),grad(c))*dx) # to calculate the total energy of the system
    Total_Energy =[t,TE]
    t += dt
    VOld.vector()[:] = VNew.vector()
    mu_mid = (1.0-theta)*mu_0 + theta*mu                      # crank-nicholson scheme
    cahn1 = (c - c_0)/dt*V1*dx + M*dot(grad(mu_mid),grad(V1))*dx #cahn-hilliard equation is splitted into two couple pde
    cahn2 = mu*V2*dx -dfdc*V2*dx -k*dot(grad(c),grad(V2))*dx
    cahn = cahn1+cahn2
    a = derivative(cahn,VNew,VTrial)                      # finding the jacobian of the cahn
    problem = CahnHilliard(a,cahn)                       
    (no_of_iterations,b) = solver.solve(problem,VNew.vector())
    #if no_of_iterations <5:
     #   dt = 2*dt
    #else:
    #    dt = dt/2
    with open('Time_Total_Energy_Spinodal_c_conservative.csv',mode='a') as csv2_file: # storing total energy of the system inn the .csv file
         writer = csv.writer(csv2_file)
         writer.writerow(Total_Energy)
    
    file1 << (VNew.split()[0], t)  # solution will be stored at each time step
    q = assemble(abs(mu - mu_0)*dx)/len(mesh.coordinates()) # Finiding the difference in the chemical potential from previous time step
    print(q)
    if (q < 1e-10):    # if the condition is true the programme will be the out of the time stepping loop
        break

print("Loop Terminated as the difference of chemical potential reached to the desired small value")
