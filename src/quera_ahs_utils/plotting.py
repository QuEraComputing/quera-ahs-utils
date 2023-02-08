import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from braket.ahs.atom_arrangement import SiteType
from braket.ahs.driving_field import DrivingField
from braket.ahs.shifting_field import ShiftingField

from braket.tasks.analog_hamiltonian_simulation_quantum_task_result import AnalogHamiltonianSimulationQuantumTaskResult
from braket.ahs.atom_arrangement import AtomArrangement

__all__ = [
    "show_register",
    "show_global_drive",
    'show_local_shift',
    'show_drive_and_shift',
    'show_final_avg_density',
    'plot_avg_density',
]

def show_register(
    register: AtomArrangement, 
    blockade_radius: float=0.0, 
    what_to_draw: str="bond", 
    show_atom_index:bool=True
):
    """Plot the given register 

        Args:
            register (AtomArrangement): A given register
            blockade_radius (float): Default is 0. The blockade radius for the register.
            what_to_draw (str): Default is "bond". Either "bond" or "circle" to indicate the blockade region. 
            show_atom_index (bool): Default is True. Choose if each atom's index is displayed over the atom itself in the resulting figure.  
    """
    
    positions = np.array([site.coordinate for site in register])
    is_filled = [site.site_type == SiteType.FILLED for site in register]
    is_empty  = [site.site_type == SiteType.EMPTY for site in register]
    
    
    if what_to_draw=="bond":
        ax, _ = visualize_UDG(positions = positions[is_filled,:],
                          radii = blockade_radius/2,
                          label_nodes=show_atom_index)
    elif what_to_draw=="circle":
        ax, G = visualize_UDG(positions = positions[is_filled,:],
                          radii = blockade_radius/2,
                          label_nodes=show_atom_index,
                          color_edges = 1,
                          draw_disks=True)
    
    ax.scatter(positions[is_empty,0],positions[is_empty,1],c='k',s=100,market="*")
    
    ax.axis("on")
    ax.tick_params(axis="x",direction="in",pad=-15)
    ax.tick_params(axis="y",direction="in",pad=-20)
    
    plt.show()


def show_global_drive(drive, axes=None, **plot_ops):
    """Plot the driving field
        Args:
            drive (DrivingField): The driving field to be plot
            axes: Default is None. matplotlib axis to draw on
            **plot_ops: options passed to matplitlib.pyplot.plot
    """   

    data = {
        'amplitude [rad/s]': drive.amplitude.time_series,
        'detuning [rad/s]': drive.detuning.time_series,
        'phase [rad]': drive.phase.time_series,
    }


    if axes is None:
        fig, axes = plt.subplots(3, 1, figsize=(7, 7), sharex=True)

    for ax, data_name in zip(axes, data.keys()):
        if data_name == 'phase [rad]':
            ax.step(data[data_name].times(), data[data_name].values(), '.-', where='post',**plot_ops)
        else:
            ax.plot(data[data_name].times(), data[data_name].values(), '.-',**plot_ops)
        ax.set_ylabel(data_name)
        ax.grid(ls=':')
    axes[-1].set_xlabel('time [s]')
    plt.tight_layout()
    plt.show()


def show_local_shift(shift:ShiftingField):
    """Plot the shifting field
        Args:
            shift (ShiftingField): The shifting field to be plot
    """       
    data = shift.magnitude.time_series
    pattern = shift.magnitude.pattern.series
    
    plt.plot(data.times(), data.values(), '.-', label="pattern: " + str(pattern))
    plt.xlabel('time [s]')
    plt.ylabel('shift [rad/s]')
    plt.legend()
    plt.tight_layout()
    plt.show()

    
def show_drive_and_shift(drive: DrivingField, shift: ShiftingField):
    """Plot the driving and shifting fields
    
        Args:
            drive (DrivingField): The driving field to be plot
            shift (ShiftingField): The shifting field to be plot
    """        
    drive_data = {
        'amplitude [rad/s]': drive.amplitude.time_series,
        'detuning [rad/s]': drive.detuning.time_series,
        'phase [rad]': drive.phase.time_series,
    }
    
    fig, axes = plt.subplots(4, 1, figsize=(7, 7), sharex=True)
    for ax, data_name in zip(axes, drive_data.keys()):
        if data_name == 'phase [rad]':
            ax.step(drive_data[data_name].times(), drive_data[data_name].values(), '.-', where='post')
        else:
            ax.plot(drive_data[data_name].times(), drive_data[data_name].values(), '.-')
        ax.set_ylabel(data_name)
        ax.grid(ls=':')
        
    shift_data = shift.magnitude.time_series
    pattern = shift.magnitude.pattern.series   
    axes[-1].plot(shift_data.times(), shift_data.values(), '.-', label="pattern: " + str(pattern))
    axes[-1].set_ylabel('shift [rad/s]')
    axes[-1].set_xlabel('time [s]')
    axes[-1].legend()
    axes[-1].grid()
    plt.tight_layout()
    plt.show()


def show_final_avg_density(result: AnalogHamiltonianSimulationQuantumTaskResult):
    """Showing a bar plot for the average Rydberg densities from the result

        Args:
            result (AnalogHamiltonianSimulationQuantumTaskResult): The result 
                from which the aggregated state counts are obtained
    """    
    avg_density = get_avg_density(result)
    
    plt.bar(range(len(avg_density)), avg_density)
    plt.xlabel("Indices of atoms")
    plt.ylabel("Average Rydberg density")
    plt.show()


def plot_avg_density(densities, register, with_labels = True, custom_axes = None, cmap = None):
    """Plot the average Rydberg densities for any 2D geometry from the result

        Args:
            densities (List[Float]): The average Rydberg densities from the result, obtainable via
               invoking get_avg_density() on an AnalogHamiltonianSimulationQuantumTaskResult instance.

            register (AtomArrangement): The register used in creating the Hamiltonian.

            with_labels (Boolean): Default is True. Choose if each atom's index is displayed over the atom itself in the resulting figure. 
                Default is True.

            custom_axes (matplotlib.axes.Axes): Default is None. If argument is given, the plot will use the supplied
                axis for displaying data and the function will not return anything. Otherwise, a new matplotlib Figure and
                Axes will be generated and returned.

            cmap (matplotlib.colors.Colormap): Default is None. Defines the colormap that the plot uses to map the average density values
                to the colors of each plotted atom. When Default value is used a the resulting plot uses a Colormap that is given by 
                `matplotlib.pyplot.cm.bwr` which is gradient from red to blue with white in the middle.  
                
        Returns:
            Tuple[Optional[matplotlib.figure.Figure],matplotlib.axes.Axes]]: returns the Figure and the Axes object used to create the plot if `custom_axes`
                is not given, otherwise the function returns None
    """
    
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
        return_fig = True
        fig, ax = plt.subplots()
    else:
        ax = custom_axes
        return_fig = False
    
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
    
    if return_fig:
        return fig,ax
    else:
        return None,ax




def visualize_UDG(positions,
                  radii,
                  ax=None,
                  label_nodes=False,
                  draw_disks=False,
                  color_nodes=3,
                  color_edges=0,
                  scale=1,
                  vertex_edge = False):
    """
    Draws a unit disk graph.
    
    Args:
        positions  : 2d positions of each vertex
        radii (float) : Float. radius of each vertex. Vertices are connected if the unit disks overlap.
                    !! This is a factor of 2 from usual !!
        draw_disks : If unit disks should be drawn. Can be:
                bool - Decision for every vertex
                    True  : Draw disks for all vertices
                    False : Do not draw disks
                list - list of bool for each vertex
        color_nodes : The color for each vertex. Can be:
                int - color assignment for all vertices.
                    0   > Black.            Indicates an inert / ground state Rydberg atom
                    1   > "Red"  #C2477F.   Indicates a Rydberg independent set excited atom
                    2   > "Pink" #FFE5E5    Indicates an inert / ground state Rydberg atom against a dark background
                    3   > "Blue" #6437FF    Default; used when a vertex is stateless
                str  - a color descriptor for every vertex. eg #8ad679
                list - color assignment for each vertex. Each element may be an int or a string.
        ax : axis to plot on. If None, generates a new axis that scales
                        so that every edge is a consistent size.
        color_edges : The color for each edge. Can be:
                int - A style label for every edge
                    0   > Thick black line.   Used for geometric connections.
                    1   > Thin grey line.     Used for nongeometric connections.
                    2   > Thick pink line.    Used against dark backgrounds
                   -1   > No line.
                tuple - (str, int). str indicates line color, int indicates line weight
                dict  - keys are edge labels, and values are style labels (int) or tuples.
        label_nodes (bool) : label each vertex.
        scale (float) : Float. scaling of vertex size. Smaller number means more area
        vertex_edge (bool): Draw an edge around each vertex.
    
    Returns:
        ax (matplotlib.axes._subplots.AxesSubplot) : Axis which the graph is plotted on
        G (networkx.classes.graph.Graph) : The unit disk graph generated by positions and radii.
    """
    
    positions = np.array(positions)
    N = len(positions)
    

    # Construct the unit disk graph
    G = nx.Graph((np.sqrt((positions[:,0:1] - positions[:,0:1].T)**2 + (positions[:,1::] - positions[:,1::].T)**2) + np.diag(np.ones(N))*1e99)<(2*radii))
    # Relabel vertices
    relabeling = {i:tuple(positions[i]) for i in range(N)}
    G = nx.relabel.relabel_nodes(G, relabeling)
    
    pos = {tuple(a):a for a in positions}
    
    if hasattr(color_nodes,'__len__'):
        color_nodes = {tuple(positions[i]):color_nodes[i] for i in range(N)}
    

    X = positions
    position_extent = X.max(0) - X.min(0)
    
    
    # If no axis given, generate a new plot. The size will be chosen such that each
    #  edge is a consistent size across plots.
    if ax==None:
        figsize = position_extent / radii + 2
    
        scaling = 0.5**np.ceil(np.log2(max(figsize)/16))
        figsize *= scaling
        
        if scaling!=1:
            print('Figure scaled by 1/{:0.2f}x'.format(1/scaling))
        
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(1,1,1)
        fig.subplots_adjust(left=0,right=1,top=1,bottom=0)
    else:
        figsize = position_extent / radii + 2
        scaling = 0.5**np.ceil(np.log2(max(figsize)/16))
    
    # Determine scaling.
    ax.axis([X.min(0)[0] - radii,X.max(0)[0] + radii,X.min(0)[1] - radii,X.max(0)[1] + radii])
    ax.set_aspect('equal')
    ax.axis('off')
    
    nodesize = 2.5*400*scaling * scale
    edgesize = 4*scaling    * scale
    
    
    # Define vertex colors from a palette
    VERTEX_COLOR_LOOKUP = {0:'#333333',2:"#FF505D",3:"#6437FF",1:"#C2477F"}
    
    # Define edge styles from a pallite
    EDGE_WIDTH_LOOKUP = {0:3  ,1:1     ,2:3        ,-1:0}
    EDGE_COLOR_LOOKUP = {0:'#333333',1:'grey',2:'#E6E6E6',-1:'white'}
    
    # Clean the node color list
    if type(color_nodes)==int:
        color_nodes2 = VERTEX_COLOR_LOOKUP[color_nodes]
    
    elif type(color_nodes)==type('string'):
        color_nodes2 = color_nodes
    elif len(color_nodes)>=len(G.nodes):
        color_nodes2 = []
        for node in G.nodes:
            color_nodes2.append(VERTEX_COLOR_LOOKUP.get(color_nodes[node],color_nodes[node]))
    else:
        print("WARNING: Color Nodes assignment not understood! Defaulting to black.")
        color_nodes2 = 'k'
    
    
    # Clean the edge color list
    if type(color_edges)==int:
        color_edges2 = EDGE_COLOR_LOOKUP[color_edges]
        edge_weights = EDGE_WIDTH_LOOKUP[color_edges]
    elif type(color_edges)==tuple:
        color_edges2 = color_edges[0]
        edge_weights = color_edges[1]
    elif len(color_edges)>=len(G.edges):
        color_edges2 = []
        edge_weights = []
        for edge in G.edges:
          if type(color_edges[edge])==int:
              color_edges2.append(EDGE_COLOR_LOOKUP[color_edges[edge]])
              edge_weights.append(EDGE_WIDTH_LOOKUP[color_edges[edge]])
          elif type(color_edges[edge])==tuple:
              color_edges2.append(color_edges[edge][0])
              edge_weights.append(color_edges[edge][1])
          else:
              raise BaseException('Bad color_edges specifier! '+repr(color_edges[edge]))
    else:
        print("WARNING: Color Edges assignment not understood! Defaulting to black.")     
        color_edges2 = EDGE_COLOR_LOOKUP[0]
        edge_weights = EDGE_WIDTH_LOOKUP[0]
    
    
    if vertex_edge==False:
        vertex_edge_radius = 0
    else:
        vertex_edge_radius = edgesize*np.array(edge_weights)
    
    nodes = nx.draw_networkx_nodes(G,pos,node_size = nodesize**2/400,node_color = color_nodes2,linewidths=vertex_edge_radius,edgecolors='k')
    edges = nx.draw_networkx_edges(G,pos,width=edgesize*np.array(edge_weights),edge_color=color_edges2)
    
    if vertex_edge:
        nodes.set_edgecolor('w')
        nodes.set_linewidth(0.25*edgesize*EDGE_WIDTH_LOOKUP[0])
    
    
    # Clean draw_disks variable
    if not hasattr(draw_disks,'__len__'):
        draw_disks = np.array([draw_disks]*N)
    
    for i in range(N):
        if draw_disks[i]==False:
            continue
        x1 = np.sin(np.linspace(0,2*np.pi,101))*radii
        y1 = np.cos(np.linspace(0,2*np.pi,101))*radii
        #print(x1)
        ax.plot(x1+positions[i,0],y1+positions[i,1],'--',color=VERTEX_COLOR_LOOKUP[1],linewidth=1,alpha=0.5)
        ax.fill(x1+positions[i,0],y1+positions[i,1],color=VERTEX_COLOR_LOOKUP[1],zorder=-100,alpha=0.1)

    
    return ax,G
