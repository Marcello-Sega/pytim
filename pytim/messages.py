# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4

ALPHA_NEGATIVE = "parameter alpha must be positive"
ALPHA_LARGE = "parameter alpha must be smaller than the smaller box side"
MESH_NEGATIVE = "parameter mesh must be positive"
MESH_LARGE = "parameter mesh must be smaller than the smaller box side"
UNDEFINED_RADIUS = "one or more atoms do not have a corresponding radius in the default or provided dictionary"
UNDEFINED_CLUSTER_SEARCH = "If extra_cluster_groups is defined, a cluster_cut should e provided"
MISMATCH_CLUSTER_SEARCH = "cluster_cut should be either a scalar or an array matching the number of groups (including itim_group)"
EMPTY_LAYER = "One or more layers are empty"
CLUSTER_FAILURE = "Cluster algorithm failed: too small cluster cutoff provided?"
UNDEFINED_LAYER = "No layer defined: forgot to call _assign_layers() or not enough layers requested"
WRONG_UNIVERSE = "Wrong Universe passed to ITIM class"
UNDEFINED_ITIM_GROUP = "No itim_group defined, or empty"
WRONG_DIRECTION = "Wrong direction supplied. Use 'x','y','z' , 'X', 'Y', 'Z' or 0, 1, 2"
CENTERING_FAILURE = "Cannot center the group in the box. Wrong direction supplied?"
