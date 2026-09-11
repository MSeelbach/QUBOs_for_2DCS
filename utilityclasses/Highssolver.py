
# NB Must have installed highspy, either by using pip install highspy,
# or by following the instructions on
# https://github.com/ERGO-Code/HiGHS#python
#
# The paths to MPS file instances assumes that this is run in the
# directory of this file (ie highs/examples) or any other subdirectory
# of HiGHS
import highspy
import numpy as np
from scipy.sparse import csc_matrix

h = highspy.Highs()

def  linprog(c,A, b ,Aeq ,beq, bounds ,integrality,Inputoptions,SaveStr=None):
    '''
    

    Parameters
    ----------
    c : numpy array
        Vector in objective.
    A : numpy array
        Matrix for inequality constraints.
    b : numpy array
        Vector for inequality constraints.
    Aeq : numpy array
        Matrix for system of linear equations.
    beq : numpy array
        Vector for equalities.
    bounds :List
        List consisting of upper and lower bounds.
    integrality : List
        Which variables are integers.
    Inputoptions : HiGHS solver options dict
        Options for the solver.
    SaveStr : str, optional
        Where to save the output. The default is None.

    Returns: binary vector
    -------
    Solution of the linear program via the HiGHS solver. The endresult is rounded
    so that one obtains binary values.

    '''

    inf = highspy.kHighsInf
    
    h = highspy.Highs()
    alt_inf = h.getInfinity()
    print('highspy.kHighsInf = ', inf, '; h.getInfinity() = ', alt_inf)
    

    num_col = A.shape[1]
    num_row = A.shape[0]
    sense = highspy.ObjSense.kMinimize #minimize objective
    offset = 0
    col_cost = c.toarray()
    col_lower = bounds[0]
    col_upper = bounds[1]
    row_lower = np.array([-inf]* num_row, dtype=np.double)
    row_upper = b.toarray()
    
    
    a_matrix_format = highspy.MatrixFormat.kColwise
    
    Afin = csc_matrix(A)

    
    
    a_matrix_start =Afin.indptr 
    a_matrix_index = Afin.indices
    a_matrix_value =Afin.data
    a_matrix_num_nz = a_matrix_start[num_col]
  
    integralityNew = []
    for s in integrality:
        if s==1:
            integralityNew.append(highspy.HighsVarType.kInteger)
    
    
    print('test-semi-definite0 as pointers')
    h.passModel(num_col, num_row, a_matrix_num_nz, 
                a_matrix_format, sense, offset,
                col_cost, col_lower, col_upper, row_lower, row_upper,
                a_matrix_start, a_matrix_index, a_matrix_value,
    #            hessian_start, hessian_index, hessian_value,
              integralityNew)
    
    options = h.getOptions()
    #options.presolve = 'off'# One can remove presolving capabilities
    #options.solver = 'ipm' # Alternative solver interior point method
    
    
    options.mip_heuristic_effort= Inputoptions['mip_heuristic_effort']
    options.time_limit= Inputoptions['time_limit']
    options.output_flag= Inputoptions['disp']
    if Inputoptions['disp']==False:
        options.mip_report_level=0
        
    
    h.passOptions(options)
    #h.writeOptions(SaveStr)
    h.run()
    solution = h.getSolution()
    h.writeSolution(SaveStr,3)

    return np.round(np.array(solution.col_value)).astype(int)  

 