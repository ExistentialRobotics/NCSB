import numpy as np
import casadi as ca

class ClfCbfQpController:
    def __init__(self, p1=1e0, p2=1e3, clf_rate=1.0, cbf_rate=0.2, safety_margin=0.1, state_dim=6, control_limits=2.0):
        self.p1 = p1  # Control effort penalty
        self.p2 = p2  # CLF slack variable penalty
        self.clf_rate = clf_rate
        self.cbf_rate = cbf_rate
        self.safety_margin = safety_margin
        self.prev_u = None

        self.state_dim = state_dim
        self.control_limits = control_limits

    
        # Define Q matrix for CLF based on state dimension
        Q_diag = np.ones(state_dim)
        if state_dim == 2:
            Q_diag[0] = 3.0  # Weight for first joint in 2D case
        elif state_dim == 6:
            # Linearly decreasing weights from 2.0 to 1.0
            Q_diag = np.linspace(2.0, 1.0, 6)
        self.Q = np.diag(Q_diag)
        
        # Setup the optimization solver
        self.setup_solver()
        
    def setup_solver(self):
        # Decision variables
        u = ca.SX.sym('u', self.state_dim)  # Control inputs
        delta = ca.SX.sym('delta')  # CLF slack variable
        
        # Parameters
        current_config = ca.SX.sym('current_config', self.state_dim)
        reference_config = ca.SX.sym('reference_config', self.state_dim)
        h = ca.SX.sym('h')  # CBF value
        dh_dtheta = ca.SX.sym('dh_dtheta', self.state_dim)  # CBF gradient
        dh_dt = ca.SX.sym('dh_dt')  # Time derivative of CBF
        u_nominal = ca.SX.sym('u_nominal', self.state_dim)  # Add nominal control input
        
        # CLF computation
        # error = current_config - reference_config
        # Q = ca.DM(self.Q)
        # V = 0.5 * ca.mtimes([error.T, Q, error])  # Quadratic form with Q matrix
        # dV = ca.mtimes([error.T, Q, u])  # Derivative also includes Q matrix
        
        # Constraints
        g = []
        lbg = []
        ubg = []
        
        # CLF constraint with slack
        # g.append(dV + self.clf_rate * V - delta)
        # lbg.append(-ca.inf)
        # ubg.append(0)
        
        # CBF constraint
        g.append(ca.dot(dh_dtheta, u) + self.cbf_rate * (h - self.safety_margin) + dh_dt)
        lbg.append(0)
        ubg.append(ca.inf)
        
        # Input bounds
        g.append(u)
        lbg.extend([-self.control_limits] * self.state_dim)
        ubg.extend([self.control_limits] * self.state_dim)
        
        # Objective function - modified to track nominal control
        obj = self.p1 * ca.sumsqr(u - u_nominal) + self.p2 * delta**2
        
        # Create solver
        opts = {
            'print_time': 0,
            'ipopt': {
                'print_level': 0,
                'max_iter': 100,
                'tol': 1e-3,
                'acceptable_tol': 1e-3
            }
        }
        
        nlp = {
            'x': ca.vertcat(u, delta),
            'p': ca.vertcat(current_config, reference_config, h, dh_dtheta, dh_dt, u_nominal),  # Add u_nominal
            'f': obj,
            'g': ca.vertcat(*g)
        }
        
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        self.lbg = lbg
        self.ubg = ubg
        
    def generate_controller(self, current_config, reference_config, h, dh_dtheta, dh_dt, u_nominal=None):
        num_links = len(current_config)
        if self.prev_u is None:
            self.prev_u = np.zeros(num_links)
            
        # If no nominal control provided, use zero
        if u_nominal is None:
            u_nominal = np.zeros(num_links)
            
        try:
            # Pack parameters
            p = np.concatenate([
                current_config,
                reference_config,
                [h],
                dh_dtheta,
                [dh_dt],
                u_nominal  # Add nominal control to parameters
            ])

            
            # Initial guess
            x0 = np.zeros(self.state_dim + 1)  # state_dim for u, +1 for delta
            
            # Solve optimization
            sol = self.solver(
                x0=x0,
                lbg=self.lbg,
                ubg=self.ubg,
                p=p
            )
            
            # Extract solution
            x_opt = sol['x'].full().flatten()
            u_new = x_opt[:num_links]
            
            # Add diagnostic prints for CBF
            # print("\n=== CBF Diagnostics ===")
            # print(f"CBF value (h): {h:.4f}")
            # print(f"CBF gradient (dh_dtheta): {dh_dtheta}")
            # print(f"Safety constraint (should be ≥ 0): {np.dot(dh_dtheta, u_new) + self.cbf_rate * h:.4f}")
            # print("=====================\n")
            
            self.prev_u = u_new
            return u_new
            
        except Exception as e:
            print(f"[CLF-CBF-QP] Solver failed: {str(e)}")
            return np.zeros(num_links) 