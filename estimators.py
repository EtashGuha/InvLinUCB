import gurobipy as gp
import numpy as np
from algorithms import UCB

def Baseline1(alg):
    sample_means = np.random.rand(len(alg.sample_means))
    return sample_means

def Baseline2(alg):
    taus = [0] * alg.arm.shape[0]
    num_pulls = {}
    for t, action in enumerate(alg.action_idxs):
        taus[action] = t

    for i in range(len(alg.arm)):
        num_pulls[i] = []

    for t in range(alg.T):
        for key in num_pulls.keys():
            if alg.action_idxs[t] != key:
                if t != 0:
                    num_pulls[key].append(num_pulls[key][ - 1])
                else:
                    num_pulls[key].append(0)
            else:
                if t != 0:
                    num_pulls[key].append(num_pulls[key][ - 1] + 1)
                else:
                    num_pulls[key].append(1)

    optimal_arm = None
    most_pulls = -1
    for i in range(len(alg.arm)):
        if num_pulls[i][alg.T - 1] > most_pulls:
            most_pulls = num_pulls[i][alg.T - 1] 
            optimal_arm = i
    sample_means = []
    for idx, tau in enumerate(taus):
        if idx is not optimal_arm:
            sample_means.append(UCB.gcb(alg.T, alg.alpha, num_pulls[idx][tau]) - UCB.gcb(alg.T, alg.alpha, num_pulls[optimal_arm][tau]))
        else:
            sample_means.append(0)
    
    return np.asarray(sample_means) * -1

def estimate_ucb_means_lp(alg):
    m = gp.Model()
    m.Params.LogToConsole = 0
    all_vars = {}
    num_pulls = {}
    T = alg.T
    for i in range(len(alg.arm)):
        num_pulls[i] = []

    for t in range(T):
        for key in num_pulls.keys():
            if alg.action_idxs[t] != key:
                if t != 0:
                    num_pulls[key].append(num_pulls[key][ - 1])
                else:
                    num_pulls[key].append(0)
            else:
                if t != 0:
                    num_pulls[key].append(num_pulls[key][ - 1] + 1)
                else:
                    num_pulls[key].append(1)

    for t in range(T):
        for idx, ele in enumerate(alg.arm):
            if t not in all_vars:
                all_vars[t] = {}
            all_vars[t][idx] = m.addVar(name="u_{}_{}".format(t, idx))

    expr = gp.LinExpr()
    list_of_all_vars = []
    for t in range(T-1,T):
        for i in range(len(alg.arm)):
            list_of_all_vars.append(all_vars[t][i])


    expr.addTerms([1.0] * len(list_of_all_vars), list_of_all_vars)

#     m.setObjective(expr, gp.GRB.MAXIMIZE)



    for t, ele in enumerate(alg.action_idxs):
        for i in range(len(alg.arm)):
            if i != ele and t >= len(alg.arm):
                m.addConstr(all_vars[t][ele] + UCB.gcb(T, alg.alpha, num_pulls[ele][t-1]) - all_vars[t][i] - UCB.gcb(T, alg.alpha, num_pulls[i][t - 1]) >= 0)
                if t + 1 < T:
                    m.addConstr(all_vars[t + 1][i] - all_vars[t][i] == 0)
            m.addConstr(all_vars[t][i] >= 0)
            m.addConstr(all_vars[t][i] <= 1)



    m.optimize()
    lp_vals = []
    for i in range(len(alg.arm)):
        lp_vals.append(all_vars[T-1][i].X)
        
    return lp_vals


def calc_Vs(alg):
    Vs = []
    Vs.append(alg.lamda * np.identity(alg.dim))
    for t in range(alg.T):
        Vs.append(Vs[-1] + (alg.actions[t].T * alg.actions[t]))
    return(Vs)

def get_orthogonal_matrix(vec):
    first = np.identity(len(vec) - 1)
    second = -1 * vec[1:] / vec[0]

    others = np.concatenate((second, first),axis=0)

    banana = others @ (others.T @ others) @ others.T

    return banana

def estimate_linucb_means_lp(alg, normalize=True, tolerance=1e-5):
    m = gp.Model()
    m.Params.LogToConsole = 0
    all_vars = {}
    T = alg.T

    Vs = calc_Vs(alg)
    for t in range(alg.T):
        all_vars[t] = []
        for i in range(alg.dim):
            all_vars[t].append(m.addVar(name="y_{}_{}".format(t, i)))

    expr = gp.LinExpr()
    expr.addTerms([1.0] * len(all_vars[alg.T - 1]),all_vars[alg.T - 1])

#     m.setObjective(expr, gp.GRB.MAXIMIZE) 
    for t in range(T):
        for action in alg.arm:
            V = Vs[t]
            invV = np.linalg.inv(V)
            if action is not alg.actions[t]:
                expr = gp.LinExpr()

                mult = np.asarray(action @ invV)[0] * -1
                expr.addTerms(mult, all_vars[t])
                constant = np.sqrt(action @ invV @ action.T) * alg.beta * -1
                expr.addConstant(constant)

                opt_mult = np.asarray(alg.actions[t] @ invV)[0]
                expr.addTerms(opt_mult, all_vars[t])
                opt_constant = np.sqrt(alg.actions[t] @ invV @ alg.actions[t].T) * alg.beta
                expr.addConstant(opt_constant)

                m.addConstr(expr >= 0)

                
    for t in range(T):
        if t == T - 1:
            continue
        a_t = alg.actions[t]
        V_t = Vs[t]
        V_tplus1 = Vs[t+1]
        
        V_t_inv = np.linalg.pinv(V_t)
        V_tplus1_inv = np.linalg.pinv(V_tplus1)
        
        orthogonal_projection = get_orthogonal_matrix(V_t_inv @ a_t.T)
        first_coeff = np.asarray(orthogonal_projection @ V_tplus1_inv)
        second_coeff = np.asarray(orthogonal_projection @ V_t_inv)
        
        for curr_dim in range(first_coeff.shape[0]):
            firstexpr = gp.LinExpr()
            firstexpr.addTerms(first_coeff[curr_dim], all_vars[t+1])
            firstexpr.addTerms(-1 * second_coeff[curr_dim], all_vars[t])
            m.addConstr(firstexpr >= -1 * tolerance)
            m.addConstr(firstexpr <= tolerance)
            
        
    for i in range(alg.dim):
        m.addConstr(all_vars[0][i] == 0)



    for i in range(alg.dim):
        for t in range(alg.T):
            if t + 1 < alg.T:
                m.addConstr(all_vars[t][i] + 1 >= all_vars[t + 1][i])
                m.addConstr(all_vars[t][i] <= all_vars[t+1][i])

    m.optimize()

    final_Vinv = np.linalg.inv(Vs[-1])
    final_y = np.matrix([ele.X for ele in all_vars[alg.T - 1]]).T
    
    
    theta_estimate = np.linalg.inv(Vs[-1]) @ np.matrix(final_y)
    
    if normalize:
        theta_estimate = theta_estimate/np.linalg.norm(theta_estimate)
        
    return theta_estimate