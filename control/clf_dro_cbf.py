import numpy as np
import casadi as ca

class ClfCbfDrccpController:
    def __init__(self, p1=1e0, p2=1e3, clf_rate=1.0, cbf_rate=1.0, 
                 wasserstein_r=0.02, epsilon=0.1, num_samples=5,
                 state_dim=6, control_limits=2.0, stability_threshold=0.1):
        # Control parameters
        self.p1 = p1  # Control effort penalty
        self.p2 = p2  # CLF slack variable penalty
        self.clf_rate = clf_rate
        self.cbf_rate = cbf_rate
        self.wasserstein_r = wasserstein_r
        self.epsilon = epsilon
        self.num_samples = num_samples
        self.state_dim = state_dim  # Number of joints/states (same as control dim for arm)
        self.control_limits = control_limits
        self.stability_threshold = stability_threshold  # Threshold for CLF slack variable
        self.unstable_flag = False  # Flag to indicate if control is unstable
        
        
        # Define Q matrix for CLF based on state dimension
        Q_diag = np.ones(state_dim)
        if state_dim == 2:
            Q_diag[0] = 1.0  # Weight for first joint in 2D case
        elif state_dim == 6:
            # Linearly decreasing weights from 2.0 to 1.0
            Q_diag = np.linspace(1.0, 1.0, 6)
        self.Q = np.diag(Q_diag)
        
        self.prev_u = None
        self.solve_fail = False
        
        
        # Setup the optimization solver
        self.setup_solver()
        
    def setup_solver(self):
        # Decision variables (total: state_dim + 1 + 1 + num_samples)
        u = ca.SX.sym('u', self.state_dim)  # Control inputs
        delta = ca.SX.sym('delta')     # CLF relaxation
        s = ca.SX.sym('s')             # DRO variable
        beta = ca.SX.sym('beta', self.num_samples)    # DRO multipliers
        
        # Parameters
        current_config = ca.SX.sym('current_config', self.state_dim)
        reference_config = ca.SX.sym('reference_config', self.state_dim)
        h_samples = ca.SX.sym('h', self.num_samples)
        h_grads = ca.SX.sym('h_grad', self.num_samples, self.state_dim)
        dh_dt = ca.SX.sym('dh_dt', self.num_samples)
        u_nominal = ca.SX.sym('u_nominal', self.state_dim)  # Add u_nominal as parameter
        
        # CLF computation
        error = current_config - reference_config
        # Convert Q matrix to CasADi symbolic
        Q = ca.DM(self.Q)
        V = 0.5 * ca.mtimes([error.T, Q, error])  # Quadratic form with Q matrix
        dV = ca.mtimes([error.T, Q, u])  # Derivative also includes Q matrix
        
        # Constraints
        g = []
        lbg = []
        ubg = []
        
        # CLF constraint
        g.append(dV + self.clf_rate * V - delta)
        lbg.append(-ca.inf)
        ubg.append(0)
        
        
        # Following Proposition 1 in the theory:
        # Construct u_bar = [u; 1; 1]
        u_bar = ca.vertcat(*[u[i] for i in range(self.state_dim)])  # First add all elements of u
        u_bar = ca.vertcat(u_bar, 1.0, 1.0)  # Then append the two ones
        
        
        # DRO constraints for each sample
        for i in range(self.num_samples):
            try:
                # Get the gradient for the current sample
                grad_i = h_grads[i, :]
                
                # Construct xi = [grad_h; h; dh_dt] for each sample
                # First create a list of all components
                xi_components = []
                for j in range(self.state_dim):  # Add gradient components
                    xi_components.append(grad_i[j])
                xi_components.append(h_samples[i])  # Add CBF value
                xi_components.append(dh_dt[i])      # Add time derivative
                
                # Concatenate all components
                xi_i = ca.vertcat(*xi_components)
                
                #print(f"Sample {i}: Created xi_i with shape: {xi_i.shape}")
                
                # beta_i >= s - xi_i^T * u_bar
                constraint_value = s - ca.dot(xi_i, u_bar) - beta[i]
                g.append(constraint_value)
                lbg.append(-ca.inf)
                ubg.append(0)
                
            except Exception as e:
                print(f"Error processing sample {i}: {str(e)}")
                raise
        
        
        # Wasserstein radius constraint
        g.append(self.wasserstein_r * ca.mmax(ca.vertcat(1, ca.norm_inf(u))) - 
                s * self.epsilon + (1/self.num_samples) * ca.sum1(beta))
        lbg.append(-ca.inf)
        ubg.append(0)
        
        # Non-negativity constraints for beta
        g.append(beta)
        lbg.extend([0] * self.num_samples)
        ubg.extend([ca.inf] * self.num_samples)
        
        # Control limits - now using state_dim
        g.append(u)
        lbg.extend([-self.control_limits] * self.state_dim)
        ubg.extend([self.control_limits] * self.state_dim)
        
        # Objective function
        obj = (self.p1 * ca.sumsqr(u) + 
               self.p2 * delta**2)
        
        
        # Create solver
        opts = {
            'print_time': 0,
            'ipopt': {
                'print_level': 0,
                'max_iter': 100,
                'tol': 1e-3,
                'acceptable_tol': 1e-3,
                'warm_start_init_point': 'yes'
            }
        }
        
        try:
            nlp = {
                'x': ca.vertcat(u, delta, s, beta),
                'p': ca.vertcat(current_config, reference_config, h_samples, 
                               h_grads.reshape((-1, 1)), dh_dt, u_nominal),  # Add u_nominal to parameters
                'f': obj,
                'g': ca.vertcat(*g)
            }
            
            self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
            self.lbg = lbg
            self.ubg = ubg
            print("Solver setup complete!")
            
        except Exception as e:
            print(f"Error creating solver: {str(e)}")
            raise
        
        # Initialize solution guess with correct dimensions
        self.x0 = np.zeros(self.state_dim + 1 + 1 + self.num_samples)  # [u, delta, s, beta]
        
    def generate_controller(self, current_config, reference_config, 
                          h_samples, h_grad_samples, dh_dt_samples, u_nominal):
        try:
            slack = 0.05
            # Pack parameters
            p = np.concatenate([
                current_config,
                reference_config,
                h_samples - slack,
                h_grad_samples.flatten(),
                dh_dt_samples,
                u_nominal  # Add u_nominal to parameters
            ])
            
            # Initial guess should match the number of decision variables
            x0 = np.zeros(self.state_dim + 1 + 1 + self.num_samples)  # [u, delta, s, beta]
            
            # Solve optimization
            sol = self.solver(
                x0=x0,
                lbg=self.lbg,
                ubg=self.ubg,
                p=p
            )
            
            # Extract solution
            x_opt = sol['x'].full().flatten()
            u_new = x_opt[:self.state_dim]  # Control inputs
            delta = x_opt[self.state_dim]   # CLF slack variable
            
            # Check if delta is too large
            self.unstable_flag = delta > self.stability_threshold
            if self.unstable_flag:
                print(f"Warning: Large CLF slack variable (delta = {delta:.3f})")
            
            return u_new
            
        except Exception as e:
            print(f"[DR-CLF-CBF-QP] Solver failed: {str(e)}")
            self.unstable_flag = True  # Set flag to True if solver fails
            return np.zeros(self.state_dim) 