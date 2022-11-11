import numpy as np
import matplotlib.pyplot as plt
import networkx as nx



def plot_avg_density(densities, register, with_labels = True, custom_axes = None, cmap=None):
    
    # get atom coordinates
    atom_coords = list(zip(register.coordinate_list(0), register.coordinate_list(1)))
    # convert all to micrometers
    atom_coords = [(atom_coord[0] * 10**6, atom_coord[1] * 10**6) for atom_coord in atom_coords]

    pos = {i:coord for i,coord in enumerate(atom_coords)}
           
    # get colors
    vmin = 0
    vmax = 1
    if cmap is None:
        cmap = plt.cm.bwr
    
    # construct graph
    g = nx.Graph()
    g.add_nodes_from(list(range(len(densities))))
    
    # construct plot
    if custom_axes is None:
        fig, ax = plt.subplots()
    else:
        ax = custom_axes
    
    nx.draw(g, 
            pos,
            node_color=densities,
            cmap=cmap,
            node_shape="o",
            vmin=vmin,
            vmax=vmax,
            font_size=9,
            with_labels=with_labels,
            ax = custom_axes if custom_axes is not None else ax)
        
    ## Set axes
    ax.set_axis_on()
    ax.tick_params(left=True, 
                   bottom=True, 
                   top=True,
                   right=True,
                   labelleft=True, 
                   labelbottom=True, 
                   # labeltop=True,
                   # labelright=True,
                   direction="in")
    ## Set colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])

    
    ax.ticklabel_format(style="sci", useOffset=False)
    
    # set titles on x and y axes
    plt.xlabel("x [μm]")
    plt.ylabel("y [μm]")
    
    

    cbar_label = "Rydberg Density"
        
    plt.colorbar(sm, ax=ax, label=cbar_label)

