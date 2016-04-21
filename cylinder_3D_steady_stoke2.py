from dolfin import *
import numpy as np
import matplotlib.pylab as plt

# Test for PETSc or Tpetra
if not has_linear_algebra_backend("PETSc") and not has_linear_algebra_backend("Tpetra"):
    info("DOLFIN has not been configured with Trilinos or PETSc. Exiting.")
    exit()

if not has_krylov_solver_preconditioner("amg"):
    info("Sorry, this demo is only available when DOLFIN is compiled with AMG "
         "preconditioner, Hypre or ML.")
    exit()

if has_krylov_solver_method("minres"):
    krylov_method = "minres"
elif has_krylov_solver_method("tfqmr"):
    krylov_method = "tfqmr"
else:
    info("Default linear algebra backend was not compiled with MINRES or TFQMR "
         "Krylov subspace method. Terminating.")
    exit()

# Verbose ? 
show = 0

##########################################
## Domain and mesh #######################
##########################################

mesh = Mesh("meshes/cylinder_fluid_smooth_changed.xml.gz")

if(show):
     plot(mesh, title="Mesh_type?")
     interactive()

print "\n","printing size of mesh; \n","x: ",mesh.coordinates()[:,0].min(), mesh.coordinates()[:,0].max(),"\n","y: ", mesh.coordinates()[:,1].min(), mesh.coordinates()[:,1].max(),"\n","z: ", mesh.coordinates()[:,2].min(), mesh.coordinates()[:,2].max(),"\n"

# ##########################################
# ## Mark Boundaries #######################
# ##########################################
mark = {"generic": 0,
        "wall": 10,
        "inlet": 5,
        "outlet": 4}
subdomains = MeshFunction("size_t", mesh, mesh.topology().dim() -1)
my_eps = 1.01
# define outer part of boundary
class Wall(SubDomain): 
    def inside(self, x, on_boundary): 
        return on_boundary and x[2]>DOLFIN_EPS and x[2]<mesh.coordinates()[:,2].max()

class Inlet(SubDomain): 
    def inside(self, x, on_boundary): 
        return on_boundary and x[2]<my_eps

class Outlet(SubDomain): 
    def inside(self, x, on_boundary): 
        return on_boundary and x[2] > (mesh.coordinates()[:,2].max()-my_eps)

# Apply Mark (defined above)
subdomains.set_all(mark["generic"])  # Entire mesh is marked "generic"
wall = Wall() 
wall.mark(subdomains, mark["wall"]) 
inlet = Inlet() 
inlet.mark(subdomains, mark["inlet"]) 
outlet = Outlet()
outlet.mark(subdomains, mark["outlet"]) 

if(show):
    plot(subdomains, title="Pipe Flow Setup", scalarbar = True, wireframe = True), interactive()

# Apply BC:
# set parameters:
u_0 = Constant((0.0,0.0,0.0))
p_in = Constant(1.0)
p_out = Constant(0.0)

# Define function spaces (P2-P1) ... we need these to impose BC
V = VectorFunctionSpace(mesh, "CG", 2)
P = FunctionSpace(mesh, "CG", 1)
W = V * P # our mixed space

# Define Variational Problem
(u, p) = TrialFunctions(W) # variables: u and p make a trialfunction in W-space
(v, q) = TestFunctions(W) # the test function has variables "v" and "q"

# to perform integrals we need:
dx = Measure("dx")[subdomains]    # volume integration
ds = Measure("ds")[subdomains]    # surface integration
n = FacetNormal(mesh)             # surface normal
force = Constant((0.0, 0.0, 0.0)) # a (dummy) body force.

bc_wall = DirichletBC(W.sub(0), u_0, subdomains, mark["wall"])
bc_inlet = DirichletBC(W.sub(1), p_in, subdomains, mark["inlet"])
bc_outlet = DirichletBC(W.sub(1), p_out, subdomains, mark["outlet"])
bcs = [bc_wall,bc_inlet,bc_outlet]
############################################# 
## VARIATIONAL PROBLEM: #####################
############################################# 
a = inner(grad(u), grad(v)) * dx - p*div(v)*dx - q * div(u)*dx 
L = inner(force, v) * dx \
    - p_in * inner(n, v) * ds(mark["inlet"]) \
    - p_out * inner(n, v) * ds(mark["outlet"])

# Form for use in constructing preconditioner matrix
b = inner(grad(u), grad(v))*dx + p*q*dx

# Assemble system
A, bb = assemble_system(a, L, bcs)

# Assemble preconditioner system
P, btmp = assemble_system(b, L, bcs)

# Create Krylov solver and AMG preconditioner
solver = KrylovSolver(krylov_method, "amg")

# Associate operator (A) and preconditioner matrix (P)
solver.set_operators(A, P)

# Solve
w = Function(W)
solver.solve(w.vector(), bb)

# # compute solution
# w = Function(W) # temporary solution vector w... lives in mixed space: W
# solve(a == L, w, bcs)

# split solution into velocity and pressure part
(u, p) = w.split(True)

# Create files for storing solution
ufile = File("results/steady/velocity.pvd")
pfile = File("results/steady/pressure.pvd")

# Save to file
# for ParaView:
ufile << u
pfile << p

# Visualization
if(show):
    plot(u, title="Velocity")
    plot(p, title="Pressure")
    interactive()

print "a has different signs!!"
print "\n"
print "check this site:"
print "http://fenicsproject.org/documentation/dolfin/1.6.0/python/demo/documented/stokes-iterative/python/documentation.html"
raw_input("Done, press <enter>")
