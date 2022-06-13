import gurobipy as gp
import numpy as np
from algorithms import UCB
import math
from estimator_util import calc_Vs, get_orthogonal_matrix, initialize_taus_np_optarm

def Baseline1(alg):
    sample_means = np.random.rand(len(alg.sample_means))/np.linalg.norm(sample_means)
    return sample_means

def Baseline2(alg):
    optimal_arm, taus, num_pulls, tau_bars = initialize_taus_np_optarm(alg)

    sample_means = []
    for idx, tau in enumerate(taus):
        if idx is not optimal_arm:
            sample_means.append(UCB.gcb(alg.T, alg.alpha, num_pulls[idx][tau]) - UCB.gcb(alg.T, alg.alpha, num_pulls[optimal_arm][tau]))
        else:
            sample_means.append(0)
    
    return np.asarray(sample_means) * -1

def Baseline2_LP(alg, timelimit=None):
    optimal_arm, taus, num_pulls, tau_bars = initialize_taus_np_optarm(alg)

    m = gp.Model()
    m.Params.LogToConsole = 0
    if timelimit is not None:
        m.setParam('TimeLimit', timelimit)
    all_vars = {}
    for idx, arm in enumerate(alg.arm):
        all_vars[idx] = m.addVar(name="u_{}".format(idx))

    for idx, tau in enumerate(taus):
        if idx is not optimal_arm:
            m.addConstr(all_vars[idx] - all_vars[optimal_arm] >= UCB.gcb(alg.T, alg.alpha, num_pulls[optimal_arm][tau-1]) - UCB.gcb(alg.T, alg.alpha, num_pulls[idx][tau-1]))
            m.addConstr(all_vars[idx] - all_vars[optimal_arm] <= UCB.gcb(alg.T, alg.alpha, num_pulls[optimal_arm][tau_bars[idx] - 1]) - UCB.gcb(alg.T, alg.alpha, num_pulls[idx][tau_bars[idx] - 1]))
    
    m.optimize()
    lp_vals = []

    for i in range(len(alg.arm)):
        try:
            lp_vals.append(all_vars[i].X)
        except:
            return None
    

    return lp_vals

                

def estimate_ucb_means_lp(alg, timelimit=None):
    m = gp.Model()
    m.Params.LogToConsole = 0
    if timelimit is not None:
        m.setParam('TimeLimit', timelimit)
    all_vars = {}
    T = alg.T
    
    optimal_arm, taus, num_pulls, tau_bars = initialize_taus_np_optarm(alg)

    
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
    
    for t, ele in enumerate(alg.action_idxs):
        for i in range(len(alg.arm)):
            if i != ele and t >= len(alg.arm):
                if not (t > taus[ele]):
                    m.addConstr(all_vars[t-1][ele] + UCB.gcb(T, alg.alpha, num_pulls[ele][t-1]) - all_vars[t-1][i] - UCB.gcb(T, alg.alpha, num_pulls[i][t - 1]) >= 0)
                if t-1 > 0:
                    m.addConstr(all_vars[t][i] - all_vars[t - 1][i] == 0)
            m.addConstr(all_vars[t][i] >= 0)
            m.addConstr(all_vars[t][i] <= 1)

    for t, ele in enumerate(alg.action_idxs):
        if t > taus[ele]:
            continue
        if t - 1 >= 0:
            m.addConstr(num_pulls[ele][t] * all_vars[t][ele] - num_pulls[ele][t - 1] * all_vars[t - 1][ele] <= 1)
            m.addConstr(num_pulls[ele][t] * all_vars[t][ele] - num_pulls[ele][t - 1] * all_vars[t - 1][ele] >= 0)


    for idx, ele in enumerate(alg.arm):
        for t in range(taus[idx], alg.T):
            m.addConstr(all_vars[t][idx] - all_vars[t - 1][idx] == 0)
    m.optimize()
    

    lp_vals = []

    for i in range(len(alg.arm)):
        try:
            lp_vals.append(all_vars[T-1][i].X)
        except:
            breakpoint()

    return lp_vals

def estimate_linucb_means_simple(alg,  normalize=True, tolerance=1e-5, timelimit=None):
 
    actions = np.asarray(alg.actions).squeeze(axis=1)
    y = np.sum(actions, axis=0)
    Vs = calc_Vs(alg)
    Vinv = np.linalg.inv(Vs[-1])
    theta_estimate = (Vinv @ y).squeeze(axis=0)

    return theta_estimate

def estimate_linucb_means_lp(alg, normalize=True, tolerance=1e-5, timelimit=None):
    m = gp.Model()
    # m.Params.LogToConsole = 0
    if timelimit is not None:
        m.setParam('TimeLimit', timelimit)
        m.setParam("PoolSearchMode", 2)
    all_vars = {}
    T = alg.T

    Vs = calc_Vs(alg)
    for t in range(alg.T):
        all_vars[t] = []
        for i in range(alg.dim):
            all_vars[t].append(m.addVar(name="y_{}_{}".format(t, i)))

    expr = gp.LinExpr()
    expr.addTerms([1.0] * len(all_vars[alg.T - 1]),all_vars[alg.T - 1])
    for t in range(T- 1, T):
        V = Vs[t]
        invV = np.linalg.inv(V)

        lhs = np.dot(alg.actions[t], alg.hat_theta)
        rhs = np.sqrt(alg.actions[t] @ invV @ alg.actions[t].T) * alg.beta 

        true_value = np.dot(alg.actions[t], alg.hat_theta)
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
        orthogonal_projection = orthogonal_projection/np.max(orthogonal_projection)
        first_coeff = np.asarray(orthogonal_projection @ V_tplus1_inv)
        second_coeff = np.asarray(orthogonal_projection @ V_t_inv)
        
        for curr_dim in range(first_coeff.shape[0]):
            firstexpr = gp.LinExpr()

            if np.any(first_coeff[curr_dim]):
                firstexpr.addTerms(first_coeff[curr_dim], all_vars[t+1])
            if np.any(second_coeff[curr_dim]):
                firstexpr.addTerms(-1 * second_coeff[curr_dim], all_vars[t])
            # breakpoint()
            # if np.any(first_coeff[curr_dim]) or np.any(second_coeff[curr_dim]):
            #     m.addConstr(firstexpr >= -1 * tolerance)
            #     m.addConstr(firstexpr <= tolerance)
            #     pass

    for i in range(alg.dim):
        m.addConstr(all_vars[0][i] == 0)



    for i in range(alg.dim):
        for t in range(alg.T):
            if t + 1 < alg.T:
                m.addConstr(all_vars[t][i] + 1 >= all_vars[t + 1][i])
                m.addConstr(all_vars[t][i] <= all_vars[t+1][i])
    m.write("example.lp")
    m.optimize()


    final_Vinv = np.linalg.inv(Vs[-1])
    all_answers = np.empty((alg.T, alg.dim))
    
    for i in range(alg.T):
        for j in range(alg.dim):
            all_answers[i, j] = all_vars[i][j].X

    try:
        final_y = np.matrix([ele.X for ele in all_vars[alg.T - 1]]).T
    except:
        return None
    
    not_final_y = np.matrix([ele.X for ele in all_vars[alg.T - 100]]).T
    
    theta_estimate = np.linalg.inv(Vs[-1]) @ np.matrix(final_y)
    
    breakpoint()
    # if normalize:
    #     if np.linalg.norm(theta_estimate) != 0:
    #         theta_estimate = theta_estimate/np.linalg.norm(theta_estimate)
    
    
    # print("TESTING")
    # bounded_values = []
    # for t in range(40, alg.T):
    #     our_y = np.matrix([ele.X for ele in all_vars[t]])
    #     K = np.linalg.norm(alg.debug_y[t])/np.linalg.norm(alg.arm[alg.action_idxs[t]])
    #     lambda_min = min(np.linalg.eig(alg.Vs[t])[0])
    #     lambda_max = max(np.linalg.eig(alg.Vs[t])[0])

    #     mag_y_hat = np.linalg.norm(our_y)
    #     bounded_value = 1/(math.pow(K,2) * mag_y_hat) - lambda_min/(math.pow(K, 2) * lambda_max * mag_y_hat) + lambda_min/(math.pow(K, 2) * lambda_max)

    #     bounded_values.append(np.dot(our_y,alg.y)/(mag_y_hat * np.linalg.norm(alg.y)) > bounded_value)
    # print(bounded_values)
    print("Solution Count: {}".format(m.SolCount))
    # print(final_y)
    return theta_estimate, all_answers

def is_baseline_3_feasible_baseline_4(alg, lp_vals):
    
    optimal_arm, taus, num_pulls, tau_bars = initialize_taus_np_optarm(alg)

    values_of_constraints = []
    for idx, tau in enumerate(taus):
        try:
            if idx is not optimal_arm:
                values_of_constraints.append(lp_vals[idx] - lp_vals[optimal_arm] >= UCB.gcb(alg.T, alg.alpha, num_pulls[optimal_arm][tau-1]) - UCB.gcb(alg.T, alg.alpha, num_pulls[idx][tau-1]))
                values_of_constraints.append(lp_vals[idx] - lp_vals[optimal_arm] <= UCB.gcb(alg.T, alg.alpha, num_pulls[optimal_arm][tau_bars[idx] - 1]) - UCB.gcb(alg.T, alg.alpha, num_pulls[idx][tau_bars[idx] - 1]))
        except:
            print(num_pulls[optimal_arm][tau-1])
            print(num_pulls[optimal_arm][tau])

            print(num_pulls[idx][tau-1])
    return all(values_of_constraints)