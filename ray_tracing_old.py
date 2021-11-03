import numpy as np

class NullVector(Exception):
    pass

def normalize(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise NullVector
    return vector / norm



def interpolate(pt,tab): ## Attention, ici tab doit etre un np.array de dim 3
    (a,b,c)=pt
    x=int(a)
    y=int(b)
    z=int(c)
    c000=tab[x,y,z]
    c100=tab[x+1,y,z]
    c001=tab[x,y,z+1]
    c101=tab[x+1,y,z+1]
    c010=tab[x,y+1,z]
    c110=tab[x+1,y+1,z]
    c011=tab[x,y+1,z+1]
    c111=tab[x+1,y+1,z+1]
    L=[c000,c100,c010,c110,c001,c101,c011,c111]
    L1=np.array([1,x,y,z,(x)*(y),(x)*(z),(y)*(z),(x)*(y)*(z)])
    L2=np.array([1,x+1,y,z,(x+1)*(y),(x+1)*(z),(y)*(z),(x+1)*(y)*(z)])
    L3=np.array([1,x,y+1,z,(x)*(y+1),(x)*(z),(y+1)*(z),(x)*(y+1)*(z)])
    L4=np.array([1,x+1,y+1,z,(x+1)*(y+1),(x+1)*(z),(y+1)*(z),(x+1)*(y+1)*(z)])
    L5=np.array([1,x,y,z+1,(x)*(y),(x)*(z+1),(y)*(z+1),(x)*(y)*(z+1)])
    L6=np.array([1,x+1,y,z+1,(x+1)*(y),(x+1)*(z+1),(y)*(z+1),(x+1)*(y)*(z+1)])
    L7=np.array([1,x,y+1,z+1,(x)*(y+1),(x)*(z+1),(y+1)*(z+1),(x)*(y+1)*(z+1)])
    L8=np.array([1,x+1,y+1,z+1,(x+1)*(y+1),(x+1)*(z+1),(y+1)*(z+1),(x+1)*(y+1)*(z+1)])
    M=np.array([L1,L2,L3,L4,L5,L6,L7,L8])
    X=np.linalg.solve(M,L)
    return X[0]+X[1]*a+X[2]*b+X[3]*c+X[4]*a*b+X[5]*a*c+X[6]*b*c+X[7]*a*b*c

def N(u,tab):
    (x,y,z)=interpolate(u,tab)
    g_x = (interpolate((x-1,y,z),tab)-interpolate((x+1,y,z),tab))/2
    g_y = (interpolate((x,y-1,z),tab)-interpolate((x,y+1,z),tab))/2
    g_z = (interpolate((x,y,z-1),tab)-interpolate((x,y,z+1),tab))/2
    return normalize((g_x,g_y,g_z))

def V(u,dir_r):
    (a,b,c)=u
    (e,f,g)=dir_r
    return normalize((e-a,f-b,g-c))

def S(u,lum):
    (a,b,c)=u
    (e,f,g)=lum
    return normalize((e-a,f-b,g-c))

def R(u,lum,tab):
    return 2*np.dot(N(u,tab),S(u,lum))*N(u,tab)/np.linalg.norm(S(u,lum))**2-S(u,lum)



def C_amb(k,I):
    return k*I

def C_diff(k,I,N,L):
    return k*I*max(0,np.dot(N,L))


def C_spec(k,I,V,R,S):
    return k*I*max(0,np.dot(V,R))**S


def compute_color(dir_r, norm_e, pt_d, pt_f,step_size,skycolor,tab,fct,lum):
    d=np.linalg.norm(pt_d,pt_f)
    n=d/step_size
    (x,y,z)=pt_d
    (u,v,w)=pt_f
    (a,b,c)=dir_r
    L=[]
    X=[]
    # creation des valeurs de chaque pixel
    for t in range(0,n):
        new_pt = interpolate((-a*t+u,-b*t+v,-c*t+w))
        L.append(new_pt)
        X.append((-a*t+u,-b*t+v,-c*t+w))
    # transformation de la couleur pour chaque pixel
    tab_des_alpha=[]
    tab_des_couleurs=[]
    for i in range(n):
        (x,y,z,alpha)=fct(L[i])
        tab_des_couleurs.append((x,y,z).dot(I_amb + I_diff*max(0,np.dot(N,L)))+ I_spec*max(0,np.dot(V,T))**S)
        tab_des_alpha.append(alpha)
    # Computation de la couleur finale
    color = skycolor[:3]
    alpha = skycolor[3]
    for i in range(0,n):
        color = tab_des_alpha[i]*tab_des_couleurs[i]+(1-tab_des_alpha[i])*alpha*color
        alpha = tab_des_alpha[i]+ (1-tab_des_alpha[i])*alpha

    return (color,alpha)




def test(tab_des_alpha,tab_des_couleurs,skycolor):
    color = skycolor[:3]
    alpha = skycolor[3]
    for i in range(0,n):
        color = tab_des_alpha[i]*tab_des_couleurs[i]+(1-tab_des_alpha[i])*tab_des_alpha[i-1]*tab_des_couleurs[i-1]
        alpha = tab_des_alpha[i]+ (1-tab_des_alpha[i])*tab_des_alpha[i-1]

    return (color,alpha)

tab_c = [(1,0,0),(0,1,1)]
tab_alpha =[1,1]
skycolor =[0,0,0,1]

return test(tab_c,tab_alpha,skycolor)










