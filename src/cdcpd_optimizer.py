import operator
import numpy as np
import gurobipy


# generate object array from dictionary with tuple key
def _gen_obj_arr(vars, shape):
    arr = np.empty(shape, dtype=np.object)
    for index in np.ndindex(shape):
        arr[index] = vars[index]
    return arr


def create_gurobi_arr(model, shape, name='', lb=-gurobipy.GRB.INFINITY):
    """
    Create gurobi variables as a numpy object array.
    :param model: instance of gurobipy.Model
    :param shape: tuple of numpy array shape
    :param name: name for gurobi variable
    :param lb: lower bound for the variable
    :return: numpy array of gurobi variables
    """
    vars = model.addVars(*shape, lb=lb, name=name)
    g_vars = _gen_obj_arr(vars, shape)
    return g_vars


def add_constraints(model, lhs, operation_type="<=", rhs=None, name=''):
    """
    Create element-wise constraint between gurobi variables and numpy array
    :param model: instance of gurobipy.Model
    :param lhs: A numpy object array of gurobi expressions
    :param operation_type: Comparison type, one of ["<=", "==", ">="]
    :param rhs: A numpy array of real values, same shape as lhs
    :param name: gurobi name for constraint objects
    :return: numpy array of gurobi constraints
    """
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
    constr_dict = model.addConstrs(gen_temp_constrs, name=name)

    constr_arr = _gen_obj_arr(constr_dict, lhs.shape)
    return constr_arr


# generates numpy element-wise function
_get_value = np.vectorize(lambda var: var.X)


def get_value(g_arr):
    """
    Get optimization result from numpy array of gurobi variables
    :param g_arr: numpy array of gurobi variables
    :return: numpy array of float, same shape as g_arr
    """
    return _get_value(g_arr)

class Optimizer:
    def run(self, verts):
        """
        Optimization method called by CDCPD.
        :param verts: (M, 3) vertices to be optimized. Generally use output of CPD.
        :return: (M, 3) optimization result. Same shape as input.
        """
        return verts


def edge_squared_distances(points, edges):
    diff = points[edges[:, 0]] - points[edges[:, 1]]
    sqr_dist = np.sum(np.square(diff), axis=1)
    return sqr_dist

class DistanceConstrainedOptimizer(Optimizer):
    """
    Performs constrained optimization that optimizes MSE between output and verts,
    subject to constraint that edges length in output is within stretch_coefficient
    of original distance.
    """
    def __init__(self, stretch_coefficient=1.0):
        """
        Constructor.
        :param template: (M, 3) template whose edge length used as reference.
        :param edges: (E, 2) integer vertices index list, represent edges in template.
        :param stretch_coefficient: Maximum ratio of out output edge length and
         reference edge length.
        """
        self.stretch_coefficient = stretch_coefficient

    def add_edges(self, template, edges):
        self.template = template
        self.edges = edges

    def run(self, verts):
        model = gurobipy.Model()
        model.setParam('OutputFlag', False)
        g_verts = create_gurobi_arr(model, verts.shape, name="verts")

        # distance constraint
        rhs = (self.stretch_coefficient ** 2) * edge_squared_distances(self.template, self.edges)
        lhs = edge_squared_distances(g_verts, self.edges)
        add_constraints(model, lhs, "<=", rhs, name="edge")

        # objective function
        g_objective = np.sum(np.square(g_verts - verts))
        model.setObjective(g_objective, gurobipy.GRB.MINIMIZE)
        model.update()
        model.optimize()

        verts_result = get_value(g_verts)
        return verts_result