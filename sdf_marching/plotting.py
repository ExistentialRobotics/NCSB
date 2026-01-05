import matplotlib.pyplot as plt
import matplotlib.patches as patches
import igraph as ig
import numpy as np
from matplotlib.patches import Circle as mplCircle

def get_circle_patch(circle_or_center, radius=None, fill=False, **kwargs):
    """
    Create a matplotlib patch for a circle.
    
    Args:
        circle_or_center: Either a Circle object or a numpy array of center coordinates
        radius: If circle_or_center is a center point, this is the radius
        fill: Whether to fill the circle
        **kwargs: Additional arguments to pass to matplotlib Circle
    """
    if radius is None:
        # Assume circle_or_center is a Circle object
        center = circle_or_center.centre
        radius = circle_or_center.radius
    else:
        # Use provided center and radius
        center = circle_or_center

    return mplCircle(
        center,
        radius,
        fill=fill,
        **kwargs
    )

def plot_graph(
    ax,
    graph,
    **kwargs
):
    layout = ig.Layout(coords = graph.vs["position"])

    ig.plot(
        graph, 
        target=ax,
        layout=layout,
        **kwargs
    )
