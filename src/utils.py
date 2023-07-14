import gurobipy
import operator
import pickle as pkl
import numpy as np

# from stackoverflow: https://stackoverflow.com/a/18994296
def shortestDistanceBetweenLines(a0,a1,b0,b1,clampAll=False,clampA0=False,clampA1=False,clampB0=False,clampB1=False):

    ''' Given two lines defined by numpy.array pairs (a0,a1,b0,b1)
        Return the closest points on each segment and their distance
    '''

    # If clampAll=True, set all clamps to True
    if clampAll:
        clampA0=True
        clampA1=True
        clampB0=True
        clampB1=True


    # Calculate denomitator
    A = a1 - a0
    B = b1 - b0
    magA = np.linalg.norm(A)
    magB = np.linalg.norm(B)
    
    _A = A / magA
    _B = B / magB
    
    cross = np.cross(_A, _B)
    denom = np.linalg.norm(cross)**2
    
    
    # If lines are parallel (denom=0) test if lines overlap.
    # If they don't overlap then there is a closest point solution.
    # If they do overlap, there are infinite closest positions, but there is a closest distance
    if not denom:
        d0 = np.dot(_A,(b0-a0))
        
        # Overlap only possible with clamping
        if clampA0 or clampA1 or clampB0 or clampB1:
            d1 = np.dot(_A,(b1-a0))
            
            # Is segment B before A?
            if d0 <= 0 >= d1:
                if clampA0 and clampB1:
                    if np.absolute(d0) < np.absolute(d1):
                        return a0,b0,np.linalg.norm(a0-b0)
                    return a0,b1,np.linalg.norm(a0-b1)
                
                
            # Is segment B after A?
            elif d0 >= magA <= d1:
                if clampA1 and clampB0:
                    if np.absolute(d0) < np.absolute(d1):
                        return a1,b0,np.linalg.norm(a1-b0)
                    return a1,b1,np.linalg.norm(a1-b1)
                
                
        # Segments overlap, return distance between parallel segments
        return None,None,np.linalg.norm(((d0*_A)+a0)-b0)
        
    
    
    # Lines criss-cross: Calculate the projected closest points
    t = (b0 - a0)
    detA = np.linalg.det([t, _B, cross])
    detB = np.linalg.det([t, _A, cross])

    t0 = detA/denom
    t1 = detB/denom

    pA = a0 + (_A * t0) # Projected closest point on segment A
    pB = b0 + (_B * t1) # Projected closest point on segment B


    # Clamp projections
    if clampA0 or clampA1 or clampB0 or clampB1:
        if clampA0 and t0 < 0:
            pA = a0
        elif clampA1 and t0 > magA:
            pA = a1
        
        if clampB0 and t1 < 0:
            pB = b0
        elif clampB1 and t1 > magB:
            pB = b1
            
        # Clamp projection A
        if (clampA0 and t0 < 0) or (clampA1 and t0 > magA):
            dot = np.dot(_B,(pA-b0))
            if clampB0 and dot < 0:
                dot = 0
            elif clampB1 and dot > magB:
                dot = magB
            pB = b0 + (_B * dot)
    
        # Clamp projection B
        if (clampB0 and t1 < 0) or (clampB1 and t1 > magB):
            dot = np.dot(_A,(pB-a0))
            if clampA0 and dot < 0:
                dot = 0
            elif clampA1 and dot > magA:
                dot = magA
            pA = a0 + (_A * dot)

    
    return pA,pB,np.linalg.norm(pA-pB)

def edge_squared_distances(points):
    diff = points[0:-1] - points[1:]
    sqr_dist = np.sum(np.square(diff), axis=1)
    return sqr_dist

def add_stretching_constraints(model, lhs, operation_type="<=", rhs=None, name=''):
    python_operator = None
    if operation_type == "<=":
        python_operator = operator.le
    elif operation_type == "==":
        python_operator = operator.eq
    elif operation_type == ">=":
        python_operator = operator.ge
    else:
        raise RuntimeError("Unsupported operator {}".format(operation_type))

    if type(rhs) is not np.ndarray:
        raise RuntimeError("rhs is not numpy.ndarray!")

    if lhs.shape != rhs.shape:
        raise RuntimeError("lhs and rhs have different shape!")

    gen_temp_constrs = (python_operator(lhs[index], rhs[index]) for index in np.ndindex(lhs.shape))
    model.addConstrs(gen_temp_constrs, name=name)

def add_self_intersection_constraints (model, vars, dlo_diameter, Y, num_of_dlos=1, nodes_per_dlo=0):
    for i in range (0, len(Y)-1):
        for j in range (i, len(Y)-1):
            # edge 1: y_i, y_{i+1}
            # edge 2: y_j, y_{j+1}
            if np.abs(i - j) <= 1:
                continue

            # for multiple dlos
            if num_of_dlos > 1:
                if (i+1) % nodes_per_dlo == 0 or (j+1) % nodes_per_dlo == 0:
                    continue

            pA, pB, dist = shortestDistanceBetweenLines(Y[i], Y[i+1], Y[j], Y[j+1], True)
            if dist >= dlo_diameter:
                continue
            
            # recompute with clamping turned off
            # pA, pB, dist = shortestDistanceBetweenLines(Y[i], Y[i+1], Y[j], Y[j+1], False)

            print('adding self-intersection constraint between E({}, {}) and E({}, {})'.format(i, i+1, j, j+1))
            # pA is the point on edge y_i, y_{i+1}
            # pB is the point on edge y_j, y_{j+1}
            # the below definition should be consistent with CDCPD2's Eq 18-21
            r_i = (pA - Y[i+1]) / (Y[i] - Y[i+1])
            r_j = (pB - Y[j+1]) / (Y[j] - Y[j+1])

            pA_var = r_i*vars[i] + (1 - r_i)*vars[i+1]
            pB_var = r_j*vars[j] + (1 - r_j)*vars[j+1]
            # model.addConstr(operator.ge(np.sum(np.square(pA_var - pB_var)), dlo_diameter**2))
            model.addConstr(operator.ge(((pA_var[0] - pB_var[0])*(pA[0] - pB[0]) +
                                         (pA_var[1] - pB_var[1])*(pA[1] - pB[1]) +
                                         (pA_var[2] - pB_var[2])*(pA[2] - pB[2])) / np.linalg.norm(pA - pB), dlo_diameter))
            # model.addConstr(operator.ge(((pA_var[0] - pB_var[0])*(pB[0] - pA[0]) +
            #                              (pA_var[1] - pB_var[1])*(pB[1] - pA[1]) +
            #                              (pA_var[2] - pB_var[2])*(pB[2] - pA[2])) / np.linalg.norm(pA - pB), dlo_diameter))

# generates numpy element-wise function
_get_value = np.vectorize(lambda var: var.X)
def get_value(g_arr):
    return _get_value(g_arr)


def post_processing (Y, Y_0, dlo_diameter, num_of_dlos=1, nodes_per_dlo=0):
    # ===== gurobi initialization =====
    model = gurobipy.Model('test_model')
    model.setParam('OutputFlag', False)

    # directly copied from CDCPD2
    # model.setParam('ScaleFlag', 0)
    # model.setParam('NonConvex', 2)
    # model.setParam('FeasibilityTol', 0.000001)

    vars = model.addVars(*(Y.shape), lb=-gurobipy.GRB.INFINITY, ub=gurobipy.GRB.INFINITY, name='nodes')
    g_verts = np.empty(Y.shape, dtype=object)
    for index in np.ndindex(Y.shape):
        g_verts[index] = vars[index]

    # ===== distance constraint =====
    rhs = np.full((len(Y)-1,), 0.03**2)
    lhs = edge_squared_distances(g_verts)
    # add_stretching_constraints(model, lhs, "<=", rhs, name="edge")
    add_self_intersection_constraints(model, g_verts, dlo_diameter, Y_0, num_of_dlos, nodes_per_dlo)

    # ===== objective function =====
    g_objective = np.sum(np.square(g_verts - Y))
    model.setObjective(g_objective, gurobipy.GRB.MINIMIZE)
    model.update()
    model.optimize()

    result = get_value(g_verts)
    Y_opt = result.astype(Y.dtype)

    return Y_opt