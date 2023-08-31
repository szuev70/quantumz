#!/usr/bin/env python
# coding: utf-8

import cmath as cm
from math import sqrt,sin,cos

def cround(z: complex, n: int = 5):
    """
    Rounds a complex number to the appointed degrees of 10**(-1).

    Parameters
    ----------
        z (complex): a complex number
        n (int): resulted number of valued digits

    Returns
    -------
        Complex number with rounded real and imaginary parts
    """
    return complex(round(z.real,n),round(z.imag,n))
def dism(d: dict):
    """
    Erases zero values from the dictionary.

    Parameters
    ----------
        d (dict): a dictionary

    Returns
    -------
        The dictionary without zero values
    """
    dc = {}
    for k in d:
        if d[k] != 0:
            dc[k] = d[k]
    return dc
def initq(n: int):
    """
    Produces initial n-dimentional quantum state.

    Parameters
    ----------
        n (int): a quantum state's dimension

    Returns
    -------
        The |0...0> quantum state 
    """
    s = '|'+'0'*n
    return {s+'>':1}
def chm(q: dict, m: int):
    """
    Checks the dimension to be appropriate.

    Parameters
    ----------
        q (dict): a quantum state
        m (int): the dimension to be checked

    Returns
    -------
        Boolean
    """
    n = len(list(q.keys())[0][1:-1])
    if m+1>n:
        return False
    else: 
        return True
def U(q: dict, m: int, alpha: float, phi: float, psi: float, prec: int = 5):
    """
    The quantum gate U acts on the m-th register of the quantum state q.

    Parameters
    ----------
        q (dict): a quantum state
        m (int): register number
        alpha, phi, psi (float): gate's parameters
        prec (int): resulted precision (number of valued digits)

    Returns
    -------
        Quantum state in the form of dictionary with frame vector as a key and it's amplitude as a value
    """
    if chm(q, m):
        nq = {}
        for k in q.keys():
            u = k[m + 1]
            if u == '0':
                k0, k1 = k, k[:m + 1] + '1' + k[m + 2:]
                if k0 in nq.keys():
                    nq[k0] += cround(q[k]*cos(alpha), prec)
                else:
                    nq[k0] = cround(q[k]*cos(alpha), prec)
                if k1 in nq.keys():
                    nq[k1] += cround(q[k]*sin(alpha)*cm.rect(1, psi), prec)
                else:
                    nq[k1] = cround(q[k]*sin(alpha)*cm.rect(1, psi), prec)
            else:
                k0, k1 = k[:m + 1] + '0' + k[m + 2:], k
                if k0 in nq.keys():
                    nq[k0] -= cround(q[k]*sin(alpha)*cm.rect(1, phi), prec)
                else:
                    nq[k0] = -cround(q[k]*sin(alpha)*cm.rect(1, phi), prec)
                if k1 in nq.keys():
                    nq[k1] += cround(q[k]*cos(alpha)*cm.rect(1, phi + psi), prec)
                else:
                    nq[k1] =  cround(q[k]*cos(alpha)*cm.rect(1, phi + psi), prec)
        q = dism(nq)
    return q
def X(q: dict, m: int):
    """
    The quantum gate X acts on the m-th register of the quantum state q.

    Parameters
    ----------
        q (dict): a quantum state
        m (int): register number

    Returns
    -------
        Resulting quantum state
    """
    return U(q, m, cm.pi/2, cm.pi, 0)
def Y(q: dict, m: int):
    """
    The quantum gate Y acts on the m-th register of the quantum state q.

    Parameters
    ----------
        q (dict): a quantum state
        m (int): register number

    Returns
    -------
        Resulting quantum state
    """
    return U(q, m, cm.pi/2, -cm.pi/2, cm.pi/2)
def Z(q: dict, m: int):
    """
    The quantum gate Z acts on the m-th register of the quantum state q.

    Parameters
    ----------
        q (dict): a quantum state
        m (int): register number

    Returns
    -------
        Resulting quantum state
    """
    return U(q, m, 0, cm.pi, 0)
def H(q: dict, m: int):
    """
    The quantum gate H acts on the m-th register of the quantum state q.

    Parameters
    ----------
        q (dict): a quantum state
        m (int): register number

    Returns
    -------
        Resulting quantum state
    """
    return U(q, m, cm.pi/4, cm.pi, 0)
def S(q: dict, m: int):
    """
    The quantum gate S acts on the m-th register of the quantum state q.

    Parameters
    ----------
        q (dict): a quantum state
        m (int): register number

    Returns
    -------
        Resulting quantum state
    """
    return U(q, m, 0, cm.pi/2, 0)
def T(q: dict, m: int):
    """
    The quantum gate T acts on the m-th register of the quantum state q.

    Parameters
    ----------
        q (dict): a quantum state
        m (int): register number

    Returns
    -------
        Resulting quantum state
    """
    return U(q, m, 0, cm.pi/8, 0)
def CU(q: dict, mlist: list, t: int, alpha: float, phi: float, psi: float, prec: int = 5):
    """
    The quantum gate U, controlled by registers listed in mlist, acts on the t register of the quantum state q.

    Parameters
    ----------
        q (dict): a quantum state
        mlist (list): list of controlling registers' numbers
        t (int): register under act 
        alpha, phi, psi (float): gate's parameters
        prec (int): resulted precision (number of valued digits)

    Returns
    -------
        Quantum state in the form of dictionary with frame vector as a key and it's amplitude as a value
    """
    if (all([chm(q,m) for m in mlist]) and chm(q,t)):
        nq = {}
        for k in q.keys():
            if all([k[m+1] == '1' for m in mlist]):
                u = k[t+1]
                if u == '0':
                    k0, k1 = k, k[:t+1]+'1'+k[t+2:]
                    if k0 in nq.keys():
                        nq[k0]+=cround(q[k]*cos(alpha),prec)
                    else:
                        nq[k0]=cround(q[k]*cos(alpha),prec)
                    if k1 in nq.keys():
                        nq[k1]+=cround(q[k]*sin(alpha)*cm.rect(1,psi),prec)
                    else:
                        nq[k1]=cround(q[k]*sin(alpha)*cm.rect(1,psi),prec)
                else:
                    k0, k1 = k[:t+1]+'0'+k[t+2:], k
                    if k0 in nq.keys():
                        nq[k0]-=cround(q[k]*sin(alpha)*cm.rect(1,phi),prec)
                    else:
                        nq[k0]=-cround(q[k]*sin(alpha)*cm.rect(1,phi),prec)
                    if k1 in nq.keys():
                        nq[k1]+=cround(q[k]*cos(alpha)*cm.rect(1,phi+psi),prec)
                    else:
                        nq[k1]=cround(q[k]*cos(alpha)*cm.rect(1,phi+psi),prec)
            else:
                nq[k] = q[k]
        q = dism(nq)
    return q
def CX(q: dict, mlist: list, t: int):
    """
    The quantum gate X, controlled by registers listed in mlist, acts on the t register of the quantum state q.

    Parameters
    ----------
        q (dict): a quantum state
        mlist (list): list of controlling registers' numbers
        t (int): register under act 

    Returns
    -------
        Resulted quantum state
    """
    return CU(q, mlist, t, cm.pi/2, cm.pi, 0)
def CY(q: dict, mlist: list, t: int):
    """
    The quantum gate Y, controlled by registers listed in mlist, acts on the t register of the quantum state q.

    Parameters
    ----------
        q (dict): a quantum state
        mlist (list): list of controlling registers' numbers
        t (int): register under act 

    Returns
    -------
        Resulted quantum state
    """
    return CU(q, mlist, t, cm.pi/2, -cm.pi/2, cm.pi/2)
def CZ(q: dict, mlist: list, t: int):
    """
    The quantum gate Z, controlled by registers listed in mlist, acts on the t register of the quantum state q.

    Parameters
    ----------
        q (dict): a quantum state
        mlist (list): list of controlling registers' numbers
        t (int): register under act 

    Returns
    -------
        Resulted quantum state
    """
    return CU(q, mlist, t, 0, cm.pi, 0)
def CH(q: dict, mlist: list, t: int):
    """
    The quantum gate H, controlled by registers listed in mlist, acts on the t register of the quantum state q.

    Parameters
    ----------
        q (dict): a quantum state
        mlist (list): list of controlling registers' numbers
        t (int): register under act 

    Returns
    -------
        Resulted quantum state
    """
    return CU(q, mlist, t, cm.pi/4, cm.pi, 0)
def CS(q: dict, mlist: list, t: int):
    """
    The quantum gate S, controlled by registers listed in mlist, acts on the t register of the quantum state q.

    Parameters
    ----------
        q (dict): a quantum state
        mlist (list): list of controlling registers' numbers
        t (int): register under act 

    Returns
    -------
        Resulted quantum state
    """
    return CU(q, mlist, t, 0, cm.pi/2, 0)
def CT(q: dict, mlist: list, t: int):
    """
    The quantum gate T, controlled by registers listed in mlist, acts on the t register of the quantum state q.

    Parameters
    ----------
        q (dict): a quantum state
        mlist (list): list of controlling registers' numbers
        t (int): register under act 

    Returns
    -------
        Resulted quantum state
    """
    return CU(q, mlist, t, 0, cm.pi/8, 0)
# новые
def CUB(q: dict, mdict: dict, t: int, alpha: float, phi: float, psi: float, prec: int = 5): 
    """
    The quantum gate U, multi-controlled by registers described in mdict, acts on the t register of the quantum state q.

    Parameters
    ----------
        q (dict): a quantum state
        mdict (dict): a dictionary of 0 or 1 as a key and a list of controlling registers' as a value
        t (int): register under act
        alpha, phi, psi (float): gate's parameters
        prec (int): number of valued digits

    Returns
    -------
        Resulted quantum state
    """
    for z in mdict['0']:
        q = X(q,z)
    q = CU(q,mdict['0']+mdict['1'],t,alpha,phi,psi,prec)
    for z in mdict['0']:
        q = X(q,z)
    return q
def chkl(s: list, k: int, l: int):
    """
    Swaps k-th and l-th values in the list s.

    Parameters
    ----------
        s (list): a list
        k, t (int): list entries indexes 

    Returns
    -------
        The list with swapped values
    """
    return s[:k]+s[l]+s[k+1:l]+s[k]+s[l+1:]
def sw2(q: dict, k: int, l: int): 
    """
    Swaps k-th and l-th registers in the quantum state q.

    Parameters
    ----------
        q (dict): a quantum state
        k, l (int): list entries indexes 

    Returns
    -------
        The resulted quantum state
    """
    if k<l:
        k+=1
        l+=1
        tl = []
        klist = list(q.keys())
        for key in klist:
            tl.append(key)
            ckey = chkl(key,k,l)
            if ckey in tl:
                continue
            if ckey in q.keys():
                mv = q[key]
                q[key] = q[ckey]
                q[ckey] = mv
            else:
                q[ckey] = q[key]
                q[key] = 0
        q = dism(q)
    return q
def swap(q: dict, clist: list): 
    """
    Makes complete SWAP in the registers listed in clist in the quantum state q.

    Parameters
    ----------
        q (dict): a quantum state
        clist (list): sorted list of the swapping registers 

    Returns
    -------
        The quantum state with swapped registers
    """
    lc = len(clist)
    if lc>1:
        rlist = sorted(clist, reverse = True)
        for j,_ in enumerate(rlist[:-1]):
            q = sw2(q,rlist[j+1],rlist[j])
        if lc>2:
            for j,_ in enumerate(clist[2:]):
                q = sw2(q,clist[j+1],clist[j+2])
    return q
def CR(q: dict, mlist: list, t: int, k: int):
    """
    The quantum rotation gate R of k-th degree, controlled by registers listed in mlist, acts on the t register of the quantum state q.

    Parameters
    ----------
        q (dict): a quantum state
        mlist (list): list of controlling registers' numbers
        t (int): register under act 
        k (int): a degree of the rotation

    Returns
    -------
        Resulted quantum state
    """
    return CU(q, mlist, t, 0, 2*cm.pi/2**k, 0)
def CRt(q: dict, mlist: list, t: int, k: int):
    """
    The backward quantum rotation gate R of k-th degree, controlled by registers listed in mlist, acts on the t register of the quantum state q.

    Parameters
    ----------
        q (dict): a quantum state
        mlist (list): list of controlling registers' numbers
        t (int): register under act 
        k (int): a degree of the rotation

    Returns
    -------
        Resulted quantum state
    """
    return CU(q, mlist, t, 0, -2*cm.pi/2**k, 0)
def QFT(q: dict):
    """
    The quantum Fourier transformation of the quantum state q.

    Parameters
    ----------
        q (dict): a quantum state

    Returns
    -------
        Resulted quantum state
    """
    n = len(list(q.keys())[0])-2
    for i in range(n):
        q = H(q,i)
        for k in range(n-i-1):
            q = CR(q,[i+1+k],i,k+2)
    return swap(q,[i for i in range(n)])
def rQFT(q): 
    """
    The backward quantum Fourier transformation of the quantum state q.

    Parameters
    ----------
        q (dict): a quantum state

    Returns
    -------
        Resulted quantum state
    """
    n = len(list(q.keys())[0])-2
    q = swap(q,[i for i in range(n)])
    for i in range(n-1,-1,-1):
        for k in range(n-i,1,-1):
            q = CRt(q,[i+k-1],i,k)
        q = H(q,i)
    return q

def measure(q: dict, m: int):
    """
    The measurement of the m-th register in the quantum state q.

    Parameters
    ----------
        q (dict): a quantum state
        m (int): a register

    Returns
    -------
        A dictionary with 0 and 1 as keys and the probabilities as corresponding values
    """
    mes = [0,0]
    if chm(q,m):
        for k in q.keys():
            if k[m+1] == '0':
                mes[0] += cm.polar(q[k])[0]**2
            else:
                mes[1] += cm.polar(q[k])[0]**2
    return mes

def multi_measure(q: dict, mlist: list): 
    """
    The measurement of the registers in the quantum state q, given in the mlist.

    Parameters
    ----------
        q (dict): a quantum state
        mlist (list): a list of registers to be measure

    Returns
    -------
        Two dictionaries: the measurement result and the resulted quantum states with their probabilities
    """
    mes, res = {}, {}
    if all([chm(q,m) for m in mlist]):
        for k in q.keys():
            k_mes, k_res = '|', k
            for m in mlist:
                k_mes += k[m+1]
                k_res = k_res[:m+1]+'!'+k_res[m+2:]
            k_mes += '>'
            k_res = k_res.replace('!','')
            if k_mes in mes.keys():
                mes[k_mes] += cm.polar(q[k])[0]**2
                res[k_mes][k_res] = q[k]
            else:
                mes[k_mes] = cm.polar(q[k])[0]**2
                res[k_mes] = {k_res: q[k]}
        for kr in res.keys():
            for kkr in res[kr].keys():
                res[kr][kkr] *= 1/sqrt(mes[kr])
    return mes, res
def qpsi(alpha: float, phi: float):
    """
    Forms a qubit.

    Parameters
    ----------
        alpha, phi (float): a qubit's parameters
        
    Returns
    -------
        1-particle quantum state
    """
    return {'|0>': cround(cos(alpha)),'|1>': cround(sin(alpha)*cm.rect(1, phi))}
def common(q1: dict, q2: dict):
    """
    Unifies two quantum states into the whole quantum system.

    Parameters
    ----------
        q1, q2 (dict): given quantum states

    Returns
    -------
        The quantum state of the joint quantum system
    """
    nq = {}
    for k1 in q1.keys():
        for k2 in q2.keys():
            nq[k1[:-1]+k2[1:]] = q1[k1]*q2[k2]
    return nq

def supp(n):
    """
    Prepares the complete superposition of the n-th dimension.

    Parameters
    ----------
        n (int): target dimension
        
    Returns
    -------
        Complete superposition quantum state
    """
    q = initq(n)
    for i in range(n):
        q = H(q,i)
    return q
def partially(Ug: dict, q: dict, n: int): 
    """
    Ug gate acts on the first n registers of the quantum state q.

    Parameters
    ----------
        Ug (dict): a n-dimensional quantum gate
        q (dict): a quantum state
        n (int): number of registers under action

    Returns
    -------
        The resulted quantum state
    """ 
    keys_list = list(q.keys())
    naked_keys = [kl[1:-1] for kl in keys_list]
    naked_keys_n = [nk[:n] for nk in naked_keys]
    naked_keys_rest = [nk[n:] for nk in naked_keys]
    naked_keys_n_unique = list(set(naked_keys_n))
    naked_keys_rest_unique = list(set(naked_keys_rest))
    res_q = {}
    for kn in naked_keys_n_unique:
        tqn = Ug({'|'+kn+'>':1})
        for tk in list(tqn.keys()):
            ltk = []
            f = 0
            for nkn,nkr in zip(naked_keys_n,naked_keys_rest):
                if nkn == tk[1:-1]:
                    ltk.append('|'+nkn+nkr+'>')
                    f = 1
            if not f:
                ltk.extend([tk[:-1]+nrkui+'>' for nrkui in naked_keys_rest_unique])
            for ltki in ltk:
                if ((ltki in res_q.keys()) and (ltki in q.keys())):
                    res_q[ltki] += tqn[tk]*q[ltki]
                elif ltki in q.keys():
                    res_q[ltki] = tqn[tk]*q[ltki]
                else:
                    res_q[ltki] = tqn[tk]
    return res_q
