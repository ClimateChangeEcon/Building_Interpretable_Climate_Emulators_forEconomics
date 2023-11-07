import numpy as np
import time  
import random

_m_A_eq   = 589
_m_O_1_eq = 900
_m_O_2_eq = 37100
_m_L_eq   = 550

_eps      = 1e-6 

_bounds_free={
    'm_AT' :[(_m_A_eq * (1-_eps) , _m_A_eq * (1+_eps))     ],
    'm_O_1':[(_eps,_m_O_1_eq*2)],
    'm_O_2':[(_eps,_m_O_2_eq*2)],
    'm_L'  :[(_eps,_m_L_eq*2)],
    'a_1':[(_eps,.3)],
    'a_2':[(_eps,.3)]
}

_bounds_fixed={
    'm_AT' :[(_m_A_eq   * (1-_eps) , _m_A_eq   * (1+_eps)) ],
    'm_O_1':[(_m_O_1_eq * (1-_eps) , _m_O_1_eq * (1+_eps)) ],
    'm_O_2':[(_m_O_2_eq * (1-_eps) , _m_O_2_eq * (1+_eps)) ],
    'm_L'  :[(_m_L_eq   * (1-_eps) , _m_L_eq   * (1+_eps)) ],
    'a_1':[(_eps,.3)],
    'a_2':[(_eps,.3)]
}

_bounds_3sr = _bounds_free
_bounds_4pr = _bounds_free

###########################################
# AT -> X_1 -> [X_2]
###########################################
def model_3sr(a=np.empty(0),m_eq=np.empty(0)): 
    global _bounds_3sr
    p = 3

    if(len(a)==0):
        info             = {}
        info['p']        = p
        info['name']     = '3SR' 
        info['m_names']  = ['AT','O_1','O_2']
        info['a_bounds'] = _bounds_3sr['a_1']  + _bounds_3sr['a_2']
        info['m_bounds'] = _bounds_3sr['m_AT'] + _bounds_3sr['m_O_1'] + _bounds_3sr['m_O_2']

        return info

    ##################################
    # Assign variable
    ##################################
    A_21 = a[0] #AT-X_1
    A_32 = a[1] #X_1-[O+L]
    
    m_1 = m_eq[0] #AT
    m_2 = m_eq[1] #X_1
    m_3 = m_eq[2] #O+L
    
    ##################################
    # Generate matrix
    ##################################
    A = np.zeros(shape=(p,p))
    
    A[1,0] = A_21
    A[2,1] = A_32
    
    A[0,1] = (A_21*m_1)/m_2
    A[1,2] = (A_32*m_2)/m_3

    temp = -np.sum(A,0)
    for i in range(0,p):
        A[i,i] = temp[i]
        
    x_vec = np.array(list(a)+list(m_eq))

    return [A,m_eq,x_vec]

###########################################
#  - -> [L_1]
# |
# AT -> O_1 -> [O_2]  
###########################################
def model_4pr(a=np.empty(0),m_eq=np.empty(0)): 
    global _bounds_4pr
    p = 4
    
    if(len(a)==0):
        info             = {}
        info['p']        = p
        info['name']     = '4PR' 
        info['m_names']  = ['AT','O_1','O_2','L_1'] 
        info['a_bounds'] = _bounds_4pr['a_1']  + _bounds_4pr['a_2']   + _bounds_4pr['a_1'] 
        info['m_bounds'] = _bounds_4pr['m_AT'] + _bounds_4pr['m_O_1'] + _bounds_4pr['m_O_2'] + _bounds_4pr['m_L']

        return info

    ##################################
    # Assign variable
    ##################################
    A_21 = a[0] #A-O_1
    A_32 = a[1] #O_1-O_2
    A_41 = a[2] #A-L_1
    
    m_1 =  m_eq[0] #A
    m_2 =  m_eq[1] #O_1
    m_3 =  m_eq[2] #O_3
    m_4 =  m_eq[3] #L_1
    
    ##################################
    # Generate matrix
    ##################################
    A = np.zeros(shape=(p,p))
    
    A[1,0] = A_21
    A[2,1] = A_32
    A[3,0] = A_41
    
    A[0,1] = (A_21*m_1)/m_2
    A[1,2] = (A_32*m_2)/m_3
    A[0,3] = (A_41*m_1)/m_4

    temp = -np.sum(A,0)
    for i in range(0,p):
        A[i,i] = temp[i]
        
    x_vec = np.array(list(a)+list(m_eq))

    return [A,m_eq,x_vec]

def check_error():  
    model_set = [model_3sr,model_4pr]
    
    N=int(1e4)
    
    for model in model_set:
        
        temp=[0,0,0]
        info=model()

        t=0
            
        for i in range(0,N):

            a = []
            for j in range(0,len(info['a_bounds'])):
                a.append(np.random.uniform(info['a_bounds'][j][0],info['a_bounds'][j][1]))

            m  = []
            for j in range(0,len(info['m_bounds'])):
                m.append(np.random.uniform(info['m_bounds'][j][0],info['m_bounds'][j][1]))
            
            t += -time.time()
            [A,m_eq,x_vec]=model(a,m)
            t += time.time()

            temp[0] += A@m_eq/N
            temp[1] += np.ones(A.shape[0]).T@A/N
            temp[2] += (x_vec - np.array(list(a)+list(m)))/N
            
           
        if np.linalg.norm(temp[0])/len(temp[0])<1e-12 and np.linalg.norm(temp[1])/len(temp[1])<1e-12 and np.linalg.norm(temp[2])/len(temp[2])<1e-12:
            print(info['name'],'- OK [',np.round(t,2),'s for ',N,' calls]')
        else:
            
            print(info['name'],'- Error')
            print('-----------------------------------------------------------')
            print('a=',a)
            print('m=',m)
            print('A=',A)
            print('m_eq=',m_eq)
            print('A@m_eq=',temp[0])
            print('np.ones(A.shape[0]).T@A=',temp[1])
            print('x_vec - np.array(list(a)+list(m))=',temp[2])
            print('-----------------------------------------------------------')
        
if __name__ == "__main__":
    check_error()  
    
        
    
    
