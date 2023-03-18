from os import kill
from platform import node
import taichi as ti
import numpy as np
import datetime

import discretization

import warnings
warnings.filterwarnings("ignore")

# global parameters
AUTOMATIC_DIFFERENTIATION = False
MAX_FEPR_ITERATIONS = 10

MAX_VELOCITY_CORRECTION = 1e-3
MAX_POSITION_CORRECTION = 1e-1

MEASURE_TIME = False

@ti.data_oriented
class MeshObject:
    def __init__(self, FEPR=True, dt=0.005, damping=0.999, ground_height=-5, iterations_per_frame=2, filename="ellell.1", G=9.81, node_mass=1.0, E=1e3, nu=0.3, demo=1):
        # Meta Parameters
        self.dt = dt
        self.damping = damping
        self.ground_height = ground_height
        self.iterations_per_frame = iterations_per_frame

        self.G = G # Gravity

        self.E = E  # Young's modulus
        self.nu = nu  # Poisson's ratio: nu \in [0, 0.5)

        self.node_mass = node_mass

        # create the mesh from tetrahedrons
        self.mesh = discretization.create_mesh(filename)
        v, e, f, elements = discretization.get_vertices_edges_faces_elements()
        
        self.num_vertices = len(v)
        self.num_edges = len(e)
        self.num_faces = len(f)
        self.num_elements = len(elements)

        self.mu = ti.var(dt=ti.f32, shape=())
        self.la = ti.var(dt=ti.f32, shape=())

        # initialize taichi vertices, edges and faces
        self.x = ti.Vector.field(3, float, len(v), needs_grad=True)
        self.e = ti.Vector.field(2, float, len(e))
        self.f = ti.Vector.field(3, float, len(f))

        self.elements = ti.Vector.field(4, float, len(elements))

        # Initialize per vertice (velocity, external & internal forces)
        self.mass = ti.Vector.field(1, float, len(v))
        self.velocity = ti.Vector.field(3, float, len(v), needs_grad=True)
        self.f_ext = ti.Vector.field(3, float, len(v))
        self.f_int = ti.Vector.field(3, float, len(v))

        self.A_inv_list = ti.Matrix.field(3, 3, float, len(elements))

        self.F_list = ti.Matrix.field(3, 3, float, len(elements))
        self.Volume_list = ti.Vector.field(1, float, len(elements))

        self.total_energy = ti.field(float, (), needs_grad=True)
        self.total_energy_recomputed = ti.field(float, (), needs_grad=True)

        # ===================================FEPR===========================================
        if FEPR:
            print("Using FEPR for MeshObject")
        else:
            print("Not using FEPR for MeshObject")

        self.FEPR = FEPR
        self.epsilon = 0.001 # in theory cmr^2
        # self.r_squared = None
        self.h = self.dt
        self.c = ti.Vector.field(7, float, 1)
        self.s = ti.field(float, (), needs_grad=True)

      
        # constraints
        self.P = ti.Vector.field(3, float, 1)
        self.H = ti.field(float, ())
        self.L = ti.Vector.field(3, float, 1)

        self.helper_H = ti.field(float, (), needs_grad=True)
        self.helper_P0 = ti.field(float, (), needs_grad=True)
        self.helper_P1 = ti.field(float, (), needs_grad=True)
        self.helper_P2 = ti.field(float, (), needs_grad=True)
        self.helper_L0 = ti.field(float, (), needs_grad=True)
        self.helper_L1 = ti.field(float, (), needs_grad=True)
        self.helper_L2 = ti.field(float, (), needs_grad=True)

        self.constraint_li = [self.helper_P0, self.helper_P1, self.helper_P2, self.helper_L0,self.helper_L1, self.helper_L2, self.helper_H]

        # gradient of constraint
        self.grad_c = ti.Vector.field(7, float, (6*self.num_vertices) + 2)
        self.grad_c_np = np.zeros(shape=(6*self.num_vertices + 2, 7), dtype=float)

        # D
        self.dinv = ti.Vector.field(6*self.num_vertices, float, 1)
        self.Dinv = ti.Vector.field(6*self.num_vertices + 2, float, 6*self.num_vertices + 2)

        self.Dinverse_gradient_product = ti.Vector.field(7, float, (6*self.num_vertices) + 2)
        self.lam = ti.Vector.field(7, float, 1)

        # Schur-Complement to solve for lambda
        self.S = ti.Vector.field(7, float, 7)

        # Second part of equation (14) that needs to be a taichi field
        #self.eq14_sub = ti.Vector.field(6*self.num_vertices, float, 1)
        
        # ==================================================================================

        # Do not use from_numpy() if you still intend to create new taichi vector fields
        self.x.from_numpy(v)
        self.e.from_numpy(e)
        self.f.from_numpy(f)
        self.elements.from_numpy(np.array(elements))
        self.mass.from_numpy(np.ones((1, self.num_vertices), np.float32))

        self.compute_A_inverse()
        self.fepr_computeDinv_as_matrix()  
        self.compute_volume()  

        # Lamé Parameters
        self.mu[None] = self.E / (2 * (1 + self.nu))
        self.la[None] = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        
        # start demo
        if demo == 0:
            self.demo0()
        
        elif demo == 1:
            self.demo1()       # Wobbly fall
            self.demo2(1.5)    # + stretch
        
        elif demo == 2:
            self.demo2(1.1)    # Stretch only
        
        elif demo == 3:
            self.demo2(1.5)    # Stretch only

        elif demo == 4:
            self.demo1()       # Wobbly fall
            self.demo2(1.5)    # + stretch
        

    # Static / do not apply any initial velocities
    def demo0(self):
        pass
    

    def demo1(self):
        # WOBBLY FALL
        for j in range(self.num_vertices):
            if j % 2 == 0:
                vel = -5
            else:
                vel = 0
                
            self.velocity[j][2] = vel


    def demo2(self, stretch):
        # INITIAL STRETCH
        for j in range(self.num_vertices):
            self.x[j][0] = stretch * self.x[j][0]
            self.x[j][1] = stretch * self.x[j][1]
            self.x[j][2] = stretch * self.x[j][2]


    def getMesh(self):
        # Returns the (initial) mesh belonging to this MeshObject
        return self.mesh


    @ti.kernel
    def compute_volume(self):
        # Compute Resting Volume
        for element_i in range(self.num_elements):
            displacement = self.displacement_matrix(element_i)
            self.Volume_list[element_i][0] = abs(displacement.determinant()) / 6.0


    @ti.kernel
    def compute_A_inverse(self):
        # Compute inverse of resting pose displacement matrix
        for element_i in range(self.num_elements):
            A = self.displacement_matrix(element_i)
            A_inverse = A.inverse()
            self.A_inv_list[element_i] = A_inverse


    @ti.func
    def displacement_matrix(self, element_i: ti.i32):
        # M = ((e^1_x, e^2_x, e^3_x), (e^1_y, e^2_y, e^3_y), (e^1_z, e^2_z, e^3_z))
        # e^1 = x^2 - x^1
        # e^2 = x^3 - x^1
        # e^3 = x^4 - x^1
        element = self.elements[element_i]

        x1 = self.x[element[0]]
        x2 = self.x[element[1]]
        x3 = self.x[element[2]]
        x4 = self.x[element[3]]        

        col_0 = (x2 - x1)
        col_1 = (x3 - x1)
        col_2 = (x4 - x1)

        return ti.Matrix.cols([col_0, col_1, col_2]) # this works

    @ti.func
    def compute_B(self, element_i):
        # Compute deformed displacement matrix
        B = self.displacement_matrix(element_i)
        return B


    @ti.kernel
    def compute_neo_hookean(self):
        # Compute strain energy density
        #   ContinuumMechanicsFEMPart3.pdf : PDF page 17 (Lecture Slides)
        for element_i in range(self.num_elements):
            B = self.compute_B(element_i)           # deformed displacement
            F = B @ self.A_inv_list[element_i]      # deformation gradient
            J = max(0.1, F.determinant())           # avoid zero and negative det
            
            F_2 = F.transpose() @ F
            trace_F2 = F_2.trace()

            psi = 0.5 * self.mu * (trace_F2 - 3) \
                    - self.mu * ti.log(J) \
                    + 0.5 * self.la * ti.log(J)**2
            
            # Total energy = sum of element energies
            self.total_energy[None] += psi * self.Volume_list[element_i][0]



    @ti.kernel
    def substep_symplectic(self):
         # Semi-Implicit Euler
        for i in range(self.num_vertices):
            self.velocity[i] = self.damping * (self.velocity[i] + (self.f_int[i] + self.f_ext[i]) / self.node_mass * self.dt)
            self.x[i] += self.dt * self.velocity[i]


    def update_step(self):
        # update the vertices, edges, faces. The taichi visualizer will then use those to update the blender visualization
        for _ in range(self.iterations_per_frame):

            self.total_energy[None] = 0.0

            # Interior forces
            # https://dongqing-wang.com/blog/games201l3/
            with ti.Tape(self.total_energy):
                self.compute_neo_hookean()

            for v_i in range(self.num_vertices):
                self.f_int[v_i][0] = -self.x.grad[v_i][0]
                self.f_int[v_i][1] = -self.x.grad[v_i][1]
                self.f_int[v_i][2] = -self.x.grad[v_i][2]
            

            # Time Integration
            self.substep_symplectic()


            # FEPR iterations
            if self.FEPR:
                v_norm = 1
                j = 0
                while np.abs(v_norm) > 10e-7 and j < MAX_FEPR_ITERATIONS:
 
                    self.fepr_compute_iterate()
                    v_norm = np.linalg.norm(self.c.to_numpy().flatten(), ord=1)
        
                    j += 1
   

    @ti.kernel
    def reset(self):
        self.P[0][0] = 0.0
        self.P[0][1] = 0.0
        self.P[0][2] = 0.0

        self.L[0][0] = 0.0
        self.L[0][1] = 0.0
        self.L[0][2] = 0.0

        self.H[None] = 0
        
        self.helper_H[None]  = 0
        self.helper_P0[None] = 0
        self.helper_P1[None] = 0
        self.helper_P2[None] = 0
        self.helper_L0[None] = 0
        self.helper_L1[None] = 0
        self.helper_L2[None] = 0
        

    def fepr_compute_iterate(self): # qk+1 = ... Eq (14)
        # x = x - D^-1∇c(q)lambda[:3]
        # velocity = velocity - D^-1∇c(q)lambda[]
        # q = [x, v]
        self.reset()
        self.grad_c_np[:, :] = 0.0

        # One FEPR iteration

        self.fepr_computeDinv_as_matrix()   # D as defined in paper
        self.fepr_compute_grad_c()          # ∇c   
        self.fepr_construct_c()             
        self.fepr_compute_lambda()          # build and solve equation (13) for λ
        self.fepr_compute_q()   


    def regularize_matrix(self, sub):
        #Cap of position corrections

        max_position_change = 10e-30
        max_velocity_change = 10e-30

        # find max value of position and velocity
        for i in range(3*self.num_vertices):  
            if abs(sub[i]) > max_position_change:
                max_position_change = abs(sub[i])

            if abs(sub[3 * self.num_vertices + i] > max_velocity_change):
                max_velocity_change = abs(sub[3 * self.num_vertices + i])

        # correct positions if necessary
        if max_position_change > MAX_POSITION_CORRECTION:
            for i in range(3 * self.num_vertices):
                sign = 1 if sub[i] > 0 else -1

                #val = abs(sub[i])
                val = abs(sub[i]) / max_position_change * MAX_POSITION_CORRECTION
                sub[i] = sign * val 
        

        # correct velocities if necessary
        if max_velocity_change > MAX_VELOCITY_CORRECTION:
            for i in range(3*self.num_vertices, 6*self.num_vertices):
                sign = 1 if sub[i] > 0 else -1

                val = abs(sub[i]) / max_velocity_change * MAX_VELOCITY_CORRECTION
                sub[i] = sign * val 
        


    def fepr_compute_q(self):
        # update q at timestep t+1
        sub = np.matmul(self.Dinverse_gradient_product, self.lam)
        self.regularize_matrix(sub)

        for j in range(self.num_vertices):
            self.x[j][0] = self.x[j][0] - sub[j*3 + 0]
            self.x[j][1] = self.x[j][1] - sub[j*3 + 1]
            self.x[j][2] = self.x[j][2] - sub[j*3 + 2]

        # Regularize only after changing the positions of the vertices
        for j in range(self.num_vertices):
            self.velocity[j][0] = self.velocity[j][0] - sub[(self.num_vertices + j)*3 + 0]
            self.velocity[j][1] = self.velocity[j][1] - sub[(self.num_vertices + j)*3 + 1]
            self.velocity[j][2] = self.velocity[j][2] - sub[(self.num_vertices + j)*3 + 2]
        

    def fepr_set_P_L_H(self):
        # Set the constraints from the helpers
        self.H[None] = self.helper_H[None]
        self.L[0][0] = self.helper_L0[None]
        self.L[0][1] = self.helper_L1[None]
        self.L[0][2] = self.helper_L2[None]
        self.P[0][0] = self.helper_P0[None]
        self.P[0][1] = self.helper_P1[None]
        self.P[0][2] = self.helper_P2[None]


    @ti.kernel
    def fepr_compute_P0(self):
        # P(v) = sum_i m_i * v_i
        for j in range(self.num_vertices):
            self.helper_P0[None] +=  self.node_mass * self.velocity[j][0]

    @ti.kernel
    def fepr_compute_P1(self):
        # P(v) = sum_i m_i * v_i
        for j in range(self.num_vertices):
            self.helper_P1[None] +=  self.node_mass * self.velocity[j][1]

    @ti.kernel
    def fepr_compute_P2(self):
        # P(v) = sum_i m_i * v_i
        for j in range(self.num_vertices):
            self.helper_P2[None] +=  self.node_mass * self.velocity[j][2]

    
    @ti.kernel
    def fepr_compute_L0(self):
        #  L(x, v) = sum_i x_i cross m_i * v_i
        for j in range(self.num_vertices):
            h = self.x[j].cross(self.node_mass*self.velocity[j])
            self.helper_L0[None] += h[0]


    @ti.kernel
    def fepr_compute_L1(self):
        #  L(x, v) = sum_i x_i cross m_i * v_i
        for j in range(self.num_vertices):
            h = self.x[j].cross(self.node_mass*self.velocity[j])
            self.helper_L1[None] += h[1]

    @ti.kernel
    def fepr_compute_L2(self):
        #  L(x, v) = sum_i x_i cross m_i * v_i
        for j in range(self.num_vertices):
            h = self.x[j].cross(self.node_mass*self.velocity[j])
            self.helper_L2[None] += h[2]

    
    @ti.kernel
    def fepr_compute_H_(self):
        # H(x,v) = E(x) + 0.5 * v^T * M * v

        # E(x) using Neo-Hookean
        for element_i in range(self.num_elements):
            B = self.compute_B(element_i)
            F = B @ self.A_inv_list[element_i]
            J = max(0.01, F.determinant())  # avoid zero and negative det
            
            F_2 = F.transpose() @ F
            trace_F2 = F_2.trace()

            psi = 0.5 * self.mu * (trace_F2 - 3) \
                    - self.mu * ti.log(J) \
                    + 0.5 * self.la * ti.log(J)**2

            self.total_energy_recomputed[None] += psi * self.Volume_list[element_i][0]

        # v^T * M * v
        for j in range(self.num_vertices):
            self.s[None] += self.node_mass * (self.velocity[j][0]**2 + self.velocity[j][1]**2 + self.velocity[j][2]**2)
        
        # E(x) + 0.5 * v^T * M * v
        for _ in range(1):
                self.helper_H[None] = self.total_energy_recomputed[None] + 0.5 * self.s[None]
                


    def construct_grad(self, constraint_i):
        for j in range(self.num_vertices):
            self.grad_c[3 * j + 0][constraint_i] = self.x.grad[j][0] 
            self.grad_c[3 * j + 1][constraint_i] = self.x.grad[j][1]
            self.grad_c[3 * j + 2][constraint_i] = self.x.grad[j][2]

            self.grad_c[3 * self.num_vertices + 3 * j + 0][constraint_i] = self.velocity.grad[j][0] 
            self.grad_c[3 * self.num_vertices + 3 * j + 1][constraint_i] = self.velocity.grad[j][1]
            self.grad_c[3 * self.num_vertices + 3 * j + 2][constraint_i] = self.velocity.grad[j][2]

        self.grad_c[6 * self.num_vertices][constraint_i] = 0.0
        self.grad_c[6 * self.num_vertices + 1][constraint_i] = 0.0

    
    def grad_P(self):
        self.grad_c_np[3 * self.num_vertices + 0::3, 1] = self.node_mass
        self.grad_c_np[3 * self.num_vertices + 1::3, 2] = self.node_mass
        self.grad_c_np[3 * self.num_vertices + 2::3, 3] = self.node_mass


    def grad_L(self):
        for j in range(self.num_vertices):
            # x
            self.grad_c_np[3 * j + 1, 4] = self.node_mass * self.velocity[j][2]
            self.grad_c_np[3 * j + 2, 4] = - self.node_mass * self.velocity[j][1]

            self.grad_c_np[3 * j + 0, 5] = -self.node_mass * self.velocity[j][2]
            self.grad_c_np[3 * j + 2, 5] = self.node_mass * self.velocity[j][0]

            self.grad_c_np[3 * j + 0, 6] = self.node_mass * self.velocity[j][1]
            self.grad_c_np[3 * j + 1, 6] = -self.node_mass * self.velocity[j][0]

            # v
            self.grad_c_np[3 * self.num_vertices + 3 * j + 1, 4] = - self.node_mass * self.x[j][2]
            self.grad_c_np[3 * self.num_vertices + 3 * j + 2, 4] = self.node_mass * self.x[j][1]

            self.grad_c_np[3 * self.num_vertices + 3 * j + 0, 5] = self.node_mass * self.x[j][2]
            self.grad_c_np[3 * self.num_vertices + 3 * j + 2, 5] = - self.node_mass * self.x[j][0]

            self.grad_c_np[3 * self.num_vertices + 3 * j + 0, 6] = -self.node_mass * self.x[j][1]
            self.grad_c_np[3 * self.num_vertices + 3 * j + 1, 6] = self.node_mass * self.x[j][0]


    def grad_H(self):
        for j in range(self.num_vertices):
            self.grad_c_np[3 * j + 0, 0] = self.x.grad[j][0] 
            self.grad_c_np[3 * j + 1, 0] = self.x.grad[j][1]
            self.grad_c_np[3 * j + 2, 0] = self.x.grad[j][2]

            self.grad_c_np[3 * self.num_vertices + 3 * j + 0, 0] = self.velocity.grad[j][0] # + 2 * self.node_mass * self.velocity[j][0]
            self.grad_c_np[3 * self.num_vertices + 3 * j + 1, 0] = self.velocity.grad[j][1] # + 2 * self.node_mass * self.velocity[j][1]
            self.grad_c_np[3 * self.num_vertices + 3 * j + 2, 0] = self.velocity.grad[j][2] # + 2 * self.node_mass * self.velocity[j][2]


    def fepr_compute_grad_c(self):
        # ∇c(q) = [∇c1(q), ∇c2(q), . . . , ∇c7(q)] ∈ R^(6m+2)×7

        self.s[None] = 0.0
        for constraint_i in range(7):           
            self.constraint_li[constraint_i][None] = 0  # this works
        

        start_of_grad_computation = datetime.datetime.now()
        
        self.total_energy_recomputed[None] = 0.0

        # Taichi can only track 1 scalar fields reliably
        if AUTOMATIC_DIFFERENTIATION:
            # Compute separate gradients and combine into grad_c
            with ti.Tape(self.helper_H):
                self.fepr_compute_H_()
            self.construct_grad(0)

            with ti.Tape(self.helper_P0):
                self.fepr_compute_P0()
            self.construct_grad(1)    

            with ti.Tape(self.helper_P1):
                self.fepr_compute_P1()
            self.construct_grad(2)   

            with ti.Tape(self.helper_P2):
                self.fepr_compute_P2()
            self.construct_grad(3)   
            
            with ti.Tape(self.helper_L0):
                self.fepr_compute_L0()
            self.construct_grad(4)   

            with ti.Tape(self.helper_L1):
                self.fepr_compute_L1()
            self.construct_grad(5)   

            with ti.Tape(self.helper_L2):
                self.fepr_compute_L2()
            self.construct_grad(6)   


        else:    
            # Compute gradients of P and L by hand
            self.fepr_compute_P0()
            self.fepr_compute_P1()
            self.fepr_compute_P2()

            self.fepr_compute_L0()
            self.fepr_compute_L1()
            self.fepr_compute_L2()


            # compute gradient of H automatically
            with ti.Tape(self.helper_H):
                self.fepr_compute_H_()
            
            # Contruct gradient from results
            self.grad_H()
            self.grad_P()
            self.grad_L()
            self.grad_c.from_numpy(self.grad_c_np)


        end_of_grad_computation = datetime.datetime.now()

        if MEASURE_TIME:
            print("Grad computation finished:")
            print("AUTODIFFERENTIATION:", AUTOMATIC_DIFFERENTIATION)
            print("Time for gradient computation: ", (end_of_grad_computation - start_of_grad_computation))

        # update constraints from helpers
        self.fepr_set_P_L_H()


           
    def fepr_compute_lambda(self):
        Dinv_np = self.Dinv.to_numpy()
        grad_np = self.grad_c.to_numpy()

        # Store instead of recomputing everytime for eqn (13) and (14)
        self.Dinverse_gradient_product = np.matmul(Dinv_np, grad_np)
        self.S = np.matmul(grad_np.transpose(), self.Dinverse_gradient_product)

        # as proposed in paper: Correct S if determinant is close to 0
        if (abs(np.linalg.det(self.S)) < 1e-7):
            self.S = self.S + 1e-7

        # (∇c(q(k))TD^−1∇c(q(k)) * λ(k+1) = c(q(k))
        self.lam = np.linalg.lstsq(self.S, self.c.to_numpy().transpose())[0]



    def fepr_construct_c(self):
        # combined constraints
        self.c[0][0] = self.H[None]
        self.c[0][1] = self.P[0][0]
        self.c[0][2] = self.P[0][1]
        self.c[0][3] = self.P[0][2]
        self.c[0][4] = self.L[0][0]
        self.c[0][5] = self.L[0][1]
        self.c[0][6] = self.L[0][2]


    def fepr_computeDinv_as_matrix(self):
        h_squared = self.h ** 2
        nodemass_inv = 1. / self.node_mass


        h_squared_times_nodemass_inv = 1/(h_squared * self.node_mass)

        for n in range(3*self.num_vertices):
            self.Dinv[n][n] = nodemass_inv
        
        for n in range(3*self.num_vertices, 6*self.num_vertices):
            self.Dinv[n][n] = h_squared_times_nodemass_inv

        self.Dinv[6*self.num_vertices][6*self.num_vertices] = 1/self.epsilon
        self.Dinv[6*self.num_vertices + 1][6*self.num_vertices + 1] = 1/self.epsilon

   