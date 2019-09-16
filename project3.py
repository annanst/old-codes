import numpy as np
import FVis3 as FVis # visulaliser

pertubatuions = True #turn on or off gaussian pertuabtions

class convection:
    def __init__(self):
        """defines variables"""
        self.mu = 0.61
        self.m_u = 1.6605e-27       #[kg] atomic mass constant
        self.N_A = 6.022e23         #[1/mol] Avogardos constant
        self.M_S = 1.989e30         #[kg] solar mass
        self.R_S = 6.96e8           #[m] solar radius
        self.G = 6.627e-11          #[m^3/kg s^2] graviational constant
        self.k_B = 1.382e-23        #[m^2 kg/s^2 K] Boltzmann constant
        self.gamma = 5/3            #1+2/degrees of freedom
        self.nx = 300               #number of elements in x
        self.ny = 100               #number of elements in y
        self.nabla = 0.40001        #constant slightly lager than 2/5
        self.x = 12e6               #[m] x length of box
        self.y = 4e6                #[m] y length of box
        self.dx = self.x/self.nx    #[m] length between each x element
        self.dy = self.y/self.ny    #[m] length between each y element
        self.g = -self.G*self.M_S/self.R_S**2    #[m/s^2] graviational acceleration

        self.T_0 = 5778         #[K] temperature
        self.P_0 = 1.8e8        #[Pa] pressure

    def initialise(self):
        """initialises temperature, pressure, density and internal energy"""
        nx, ny = self.nx, self.ny
        self.T = np.zeros((ny, nx))
        self.u = np.zeros((ny, nx))
        self.w = np.zeros((ny, nx))
        self.P = np.zeros((ny, nx))
        self.rho = np.zeros((ny, nx))
        self.e = np.zeros((ny, nx))

        #initializing T
        for i in range(ny):
            self.T[i,:] = -self.nabla*self.g*(self.y-self.dy*i)*self.mu*self.m_u/self.k_B+self.T_0

        #initializing P
        self.P[:] = self.P_0*(self.T[:]/self.T_0)**(1/self.nabla)

        #adding gaussian pertubations to the initial values of T.
        if pertubatuions:
            self.T += self.T_0*self.gaussian_pertubation()

        #Initialising rho and e
        self.rho[:] = self.P[:]*self.mu*self.m_u/(self.k_B*self.T[:])
        self.e[:] = self.P[:]/(self.gamma-1)


    def gaussian_pertubation(self):
        """creates gaussian pertuabtions in the initial temperature"""
        x_arr = np.linspace(0, self.x, self.nx)
        y_arr = np.linspace(0, self.y, self.ny)
        x_mid, y_mid = np.meshgrid(x_arr, y_arr)
        sigma = self.y/4
        return np.exp(-((x_mid-self.x/2)**2+(y_mid-self.y/2)**2)/(2*sigma**2))*0.5

    def timestep(self):
        """calculates timestep"""
        rho_rel = self.drhodt/self.rho
        e_rel = self.dedt/self.e
        x_rel = self.u/self.dx
        y_rel = self.w/self.dy

        vals = np.array([rho_rel, e_rel, x_rel, y_rel])
        delta = np.max(np.absolute(vals))

        if delta >= 0.10:
            p = 0.1     #can be changed to a smaller value
            self.dt = p/delta
        else:
            self.dt = 0.10

    def boundary_conditions(self):
        """boundary conditions for energy, density and velocity"""
        constant = self.mu*self.m_u/(self.k_B)

        #Vertical conditions
        self.u[0,:] = (4*self.u[1,:]-self.u[2,:])/3      #upper u
        self.u[-1,:] = (4*self.u[-2,:]-self.u[-3,:])/3   #lower u

        self.w[0,:] = 0     #upper w
        self.w[-1,:] = 0    #lower w

        self.e[0,:] = (4*self.e[1,:]-self.e[2,:])/(3+self.g*constant/(self.T[0,:])*2*self.dy)      #upper e
        self.e[-1,:] = (-4*self.e[-2,:]+self.e[-3,:])/(-3+self.g*constant/(self.T[-1,:])*2*self.dy)  #lower e

        self.rho[0,:] = self.e[0,:]*(self.gamma-1)*constant/self.T[0,:]     #upper rho
        self.rho[-1,:] = self.e[-1,:]*(self.gamma-1)*constant/self.T[-1,:]  #lower rho

    def central_x(self,func):
        """central difference scheme in x-direction"""
        roll_right = np.roll(func, 1, axis=1)
        roll_left = np.roll(func, -1, axis=1)
        return (roll_left-roll_right)/(2*self.dx)

    def central_y(self,func):
        """central difference scheme in y-direction"""
        roll_down = np.roll(func, 1, axis=0)
        roll_up = np.roll(func, -1, axis=0)
        return (roll_up - roll_down)/(2*self.dy)

    def upwind_x(self,func,u):
        """upwind difference scheme in x-direction"""
        new_func = np.zeros((self.ny,self.nx))
        func_le0 =(func-np.roll(func, -1, axis=1))/self.dx
        func_s0 =(np.roll(func, 1, axis=1)-func)/self.dx
        new_func[np.where(u>=0)] = func_le0[np.where(u>=0)]
        new_func[np.where(u<0)] = func_le0[np.where(u<0)]
        return new_func

    def upwind_y(self,func,w):
        """upwind difference scheme in y-direction"""
        new_func = np.zeros((self.ny,self.nx))
        func_le0 =(func-np.roll(func, -1, axis=0))/self.dy
        func_s0 =(np.roll(func, 1, axis=0)-func)/self.dy
        new_func[np.where(w>=0)] = func_le0[np.where(w>=0)]
        new_func[np.where(w<0)] = func_le0[np.where(w<0)]
        return new_func

    def hydro_solver(self):
        """hydrodynamic equations solver"""
        #solve derivatives for primary variables
        c_u_x = self.central_x(self.u)
        c_w_y = self.central_y(self.w)
        c_P_x = self.central_x(self.P)
        c_P_y = self.central_y(self.P)

        self.drhodt = -self.rho*(c_u_x+c_w_y)-self.u*self.upwind_x(self.rho,self.u)-self.w*self.upwind_y(self.rho,self.w)

        self.dedt = -(self.e+self.P)*(c_u_x+c_w_y)-self.u*self.upwind_x(self.e,self.u)-self.w*self.upwind_y(self.e,self.w)

        self.drhoudt = -self.rho*self.u*(self.upwind_x(self.u,self.u)+self.upwind_y(self.w,self.u))-self.u*self.upwind_x(self.rho*self.u,self.u)-self.w*self.upwind_y(self.rho*self.u,self.w)-c_P_x

        self.drhowdt = -self.rho*self.w*(self.upwind_y(self.w,self.w)+self.upwind_x(self.u,self.w))-self.w*self.upwind_y(self.rho*self.w,self.w)-self.u*self.upwind_x(self.rho*self.w,self.u)-c_P_y+self.rho*self.g

        #creating dt
        self.timestep()

        #saving the previous rho and e to be used in u, w, T and P
        rho_prev = self.rho
        e_prev = self.e

        #solve next step for primary variables
        self.rho[:] = self.rho+self.drhodt*self.dt
        self.e[:] = self.e+self.dedt*self.dt
        self.u[:] = (rho_prev*self.u+self.drhoudt*self.dt)/self.rho
        self.w[:] = (rho_prev*self.w+self.drhowdt*self.dt)/self.rho

        #fixing boundary conditions for u, w, rho and e
        self.boundary_conditions()

        #solve for T
        self.T[:] = (self.mu*self.m_u*e_prev*(self.gamma-1))/(rho_prev*self.k_B)

        #solve for P
        self.P[:] = (self.gamma-1)*e_prev

        return self.dt

if __name__ == '__main__':
    solver = convection()
    vis = FVis.FluidVisualiser()
    solver.initialise()
    vis.save_data(216, solver.hydro_solver, rho=solver.rho, u=solver.u, w=solver.w, e=solver.e, P=solver.P, T=solver.T, sim_fps=1.0, folder='T') #name of the folder decides name of the video that is being made
    list = [5, 50, 100, 150, 200, 215]      #list of times to take snapshots, must be added to the next line
    vis.animate_2D('T', save=True)          #makes video of convection box for temperature
    vis.delete_current_data()   #deletes unnessecarry data
