""" Module: test_dbscan
    ===================
    This is a collection of tests to check
    that the DBSCAN behavior is kept consistent

    >>> import MDAnalysis as mda
    >>> import pytim
    >>> import numpy as np
    >>> from pytim.datafiles import ILBENZENE_GRO
    >>> from pytim.utilities import do_cluster_analysis_dbscan as DBScan
    >>> u = mda.Universe(ILBENZENE_GRO)
    >>> benzene = u.select_atoms('name C and resname LIG')
    >>> u.atoms.positions = u.atoms.pack_into_box()
    >>> l,c,n =  DBScan(benzene, cluster_cut = 4.5, threshold_density = None)
    >>> l1,c1,n1 = DBScan(benzene, cluster_cut = 8.5, threshold_density = 'auto')
    >>> td = 0.009
    >>> l2,c2,n2 = DBScan(benzene, cluster_cut = 8.5, threshold_density = td)
    >>> print (np.sort(c)[-2:])
    [   12 14904]

    >>> print (np.sort(c2)[-2:])
    [   0 9335]

    >>> print (np.all(c1==c2), np.all(l1==l2))
    (True, True)


"""
