def Grovers_Simulator(a):
    """a is the unknown number that is searched """
    A=bin(a)[2:]
    n=len(A)
    N=2**n
    
    a1=Dict[A]
    a2=a1[:, np.newaxis]
    Ufa=np.eye(N)-2*a2.dot(a2.T)
    
    b='0'*n
    b1=Dict[b]
    b2=b1[:, np.newaxis]
    Uf0=np.eye(N)-2*b2.dot(b2.T)

    w={1:Walsh(I2), 2:Walsh4(I4), 3: Walsh8(I8), 4: Walsh16(I16), 5: Walsh32(I32)}
    W=w[n]
    
    l1=-W.dot(Uf0)
    l2=l1.dot(np.linalg.inv(W))
    RsRa=l2.dot(Ufa)
    
    i={1:Q0, 2:Q00, 3:Q000, 4:Q0000, 5:Q00000}
    initial_prep=W.dot(i[n])
    
    k=int(np.rint(np.pi*np.sqrt(N)/4.-0.5))
    
    I=np.eye(N)
    p=RsRa
    for i in range(1, k+1):
        f=p.dot(initial_prep)
        p=p.dot(p)
    
    plot_measure(measure(f))
