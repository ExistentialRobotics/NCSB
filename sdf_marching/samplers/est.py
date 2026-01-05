from sdf_marching.circles import Circle
import numpy as np
from sdf_marching.overlap import get_overlaps_graph


def get_expansive(
    dist_function,
    epsilon,
    minimum_radius,
    num_samples,
    mins,
    maxs,
    start_point,
    num_directions_per_dim=4,
    overlap_factor=0.1,
    max_num_iterations=np.inf,
    end_point=None,
    rng=None,
):
    directions = get_expanding_directions(num_directions_per_dim, ndim=start_point.shape[0])

    queue = [Circle(start_point, float(dist_function(start_point) - epsilon))]
    circles_to_return = []

    num_iterations = 0

    end_point_sdf = None
    if end_point is not None:
        end_point_sdf = dist_function(end_point)[0]

    end_point_reached = False
    while len(queue) > 0 and not end_point_reached:
        if len(circles_to_return) > num_samples:
            print("reached num samples")
            break
        if num_iterations >= max_num_iterations:
            print("reached max num iterations")
            break

        queue, circles_to_return, current_circle, new_circles, end_point_reached = get_expansive_loop(
            queue,
            circles_to_return,
            dist_function,
            mins,
            maxs,
            overlap_factor,
            epsilon,
            minimum_radius,
            directions,
            end_point=end_point,
            end_point_sdf=end_point_sdf,
            rng=rng,
        )
        num_iterations += 1

    return get_overlaps_graph(circles_to_return), circles_to_return, num_iterations


def get_expansive_loop(
    queue,
    circles_to_return,
    dist_function,
    mins,
    maxs,
    overlap_factor,
    epsilon,
    minimum_radius,
    directions,
    end_point=None,
    end_point_sdf=None,
    rng=None,
):
    queue_key = lambda c: -c.radius  # sort by negative radius

    current_circle = heappop(queue, key=queue_key)

    # if not contained by an existing circle, do not append or expand
    if not get_min_dist_to_circles(circles_to_return, current_circle) > -overlap_factor * current_circle.radius:
        return queue, circles_to_return, current_circle, [], False

    # else, append and expand
    circles_to_return.append(current_circle)  # append
    # check if end_point is reached
    if end_point is not None:
        if current_circle.contains_point(end_point):
            return queue, circles_to_return, current_circle, [], True
        assert end_point_sdf is not None, "end_point_sdf must be provided if end_point is provided"
        goal_circle = Circle(end_point, end_point_sdf - epsilon)
        overlap_dist = current_circle.radius + goal_circle.radius - current_circle.distance_to(goal_circle)
        if overlap_dist > overlap_factor * min(current_circle.radius, goal_circle.radius):  # overlap enough
            circles_to_return.append(goal_circle)  # append goal circle
            return queue, circles_to_return, current_circle, [], True

    # randomly rotate directions
    ndim = current_circle.centre.shape[0]
    if rng is not None:
        if ndim == 2:
            theta = rng.uniform(0, 2 * np.pi)
            rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            rotated_directions = np.dot(directions, rot)
        elif ndim == 3:
            theta = rng.uniform(0, 2 * np.pi)
            phi = rng.uniform(-np.pi / 2, np.pi / 2)
            rot = np.array(
                [
                    [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)],
                    [np.cos(phi) * np.cos(theta), np.cos(phi) * np.sin(theta), -np.sin(phi)],
                    [-np.sin(theta), np.cos(theta), 0],
                ]
            )
            rotated_directions = np.dot(directions, rot)
        else:
            raise NotImplementedError("Only 2D and 3D are supported for random rotation")
    else:
        rotated_directions = directions

    # expand in rotated directions
    candidate_centres = current_circle.centre + current_circle.radius * rotated_directions
    is_in_box = np.all((candidate_centres > mins) & (candidate_centres < maxs), axis=-1)
    candidate_centres = candidate_centres[is_in_box]  # only keep points in box, reduce query of dist_function
    if candidate_centres.size == 0:  # no valid candidates
        return queue, circles_to_return, current_circle, [], False
    candidate_dists = dist_function(candidate_centres)
    is_big_enough = (candidate_dists - epsilon) > minimum_radius
    valid_idxs = np.where(is_big_enough)

    new_circles = []
    for idx in valid_idxs[0].tolist():
        new_circle = Circle(candidate_centres[idx, :], float(candidate_dists[idx]) - epsilon)
        new_circles.append(new_circle)

        heappush(queue, new_circle, key=queue_key)

    return queue, circles_to_return, current_circle, new_circles, False


def get_min_dist_to_circles(circles, target_circle):
    if circles:
        # this measures the (signed) distance between circle and new centre
        dists = np.fromiter(map(lambda circle: circle.distance_to_point(target_circle.centre), circles), np.float32)
        return dists.min()
    else:
        return 0.0


def get_expanding_directions(num_directions_per_dim, ndim):
    if ndim == 3:
        coordinates_list = np.meshgrid(
            np.linspace(0, 2 * np.pi, num_directions_per_dim, endpoint=False),
            np.linspace(-np.pi / 2, np.pi / 2, num_directions_per_dim, endpoint=True),
        )
    else:
        coordinates_list = np.meshgrid(
            *((np.linspace(0, 2 * np.pi, num_directions_per_dim, endpoint=False),) * (ndim - 1))
        )  # add 1 to number of directions so that 2 * np.pi is excluded

    points = np.zeros((coordinates_list[0].size, ndim))

    # set the x-axis as 1 for rotation
    points[:, 0] = 1.0

    # sequentially apply rotation in (idx, idx+1)-th plane
    # assuming the idx+1-th coordinate is zero
    for idx, angle in enumerate(coordinates_list):
        reshaped_angle = angle.flatten()
        points[:, idx + 1] = points[:, idx] * np.sin(reshaped_angle)
        points[:, idx] = points[:, idx] * np.cos(reshaped_angle)

    return points


####### adapted from Lib/heapq.py
def heapify(x, key=None):
    """Transform list into a heap, in-place, in O(len(x)) time."""
    n = len(x)
    # Transform bottom-up.  The largest index there's any point to looking at
    # is the largest with a child index in-range, so must have 2*i + 1 < n,
    # or i < (n-1)/2.  If n is even = 2*j, this is (2*j-1)/2 = j-1/2 so
    # j-1 is the largest, which is n//2 - 1.  If n is odd = 2*j+1, this is
    # (2*j+1-1)/2 = j so j-1 is the largest, and that's again n//2-1.
    for i in reversed(range(n // 2)):
        _siftup(x, i, key=key)


def heappush(heap, item, key=None):
    """Push item onto heap, maintaining the heap invariant."""
    heap.append(item)
    _siftdown(heap, 0, len(heap) - 1, key=key)


def heappop(heap, key=None):
    """Pop the smallest item off the heap, maintaining the heap invariant."""
    lastelt = heap.pop()  # raises appropriate IndexError if heap is empty
    if heap:
        returnitem = heap[0]
        heap[0] = lastelt
        _siftup(heap, 0, key=key)
        return returnitem
    return lastelt


def _siftdown(heap, startpos, pos, key=None):
    if key is None:
        _key = lambda x: x
    else:
        _key = key

    newitem = heap[pos]
    # Follow the path to the root, moving parents down until finding a place
    # newitem fits.
    while pos > startpos:
        parentpos = (pos - 1) >> 1
        parent = heap[parentpos]
        if _key(newitem) < _key(parent):
            heap[pos] = parent
            pos = parentpos
            continue
        break
    heap[pos] = newitem


def _siftup(heap, pos, key=None):
    if key is None:
        _key = lambda x: x
    else:
        _key = key

    endpos = len(heap)
    startpos = pos
    newitem = heap[pos]
    # Bubble up the smaller child until hitting a leaf.
    childpos = 2 * pos + 1  # leftmost child position
    while childpos < endpos:
        # Set childpos to index of smaller child.
        rightpos = childpos + 1
        if rightpos < endpos and not _key(heap[childpos]) < _key(heap[rightpos]):
            childpos = rightpos
        # Move the smaller child up.
        heap[pos] = heap[childpos]
        pos = childpos
        childpos = 2 * pos + 1
    # The leaf at pos is empty now.  Put newitem there, and bubble it up
    # to its final resting place (by sifting its parents down).
    heap[pos] = newitem
    _siftdown(heap, startpos, pos, key=key)
