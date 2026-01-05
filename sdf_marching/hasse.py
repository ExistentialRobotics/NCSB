from collections import defaultdict
from operator import attrgetter

def find_containment_hierarchy_full(
        circles, 
        sort_in_place=True, 
        exclude_contained=False
    ):

    containment_dict = defaultdict(list) # maps idx of smaller circle to idxs of bigger containing circle

    # sort by radius
    # only bigger circles "can" contain smaller circles

    if sort_in_place:
        #NOTE: side effect
        circles.sort()
        sorted_circles = circles
    else:
        sorted_circles = sorted(circles)

    for idx_bigger in range(1, len(sorted_circles)):
        for idx_smaller in range(0, idx_bigger):
            # TODO: check if this works
            # if exclude_contained and idx_bigger in containment_dict:
            #     continue

            bigger_circle = sorted_circles[idx_bigger]
            smaller_circle = sorted_circles[idx_smaller]

            if bigger_circle.contains(smaller_circle): # overlap factor doesn't make sense here
                containment_dict[idx_smaller].append(idx_bigger)
    
    return containment_dict

def find_containment_hierarchy_min(circles, sort_in_place=True, overlap_factor=1.0):

    containment_dict = {}

    if sort_in_place:
        #NOTE: side effect
        circles.sort()
        sorted_circles = circles
    else:
        sorted_circles = sorted(circles)

    for idx_bigger in range(1, len(sorted_circles)):
        for idx_smaller in range(0, idx_bigger):
            if idx_smaller in containment_dict:
                continue

            bigger_circle = sorted_circles[idx_bigger]
            smaller_circle = sorted_circles[idx_smaller]


            if bigger_circle.contains(smaller_circle, overlap_factor=overlap_factor):
                containment_dict[idx_smaller] = idx_bigger
    
    return containment_dict

def find_containment_hierarchy_max(circles, sort_in_place=True, overlap_factor=1.0):

    containment_dict = {}

    if sort_in_place:
        #NOTE: side effect
        circles.sort()
        sorted_circles = circles
    else:
        sorted_circles = sorted(circles)

    for idx_bigger in reversed(range(1, len(sorted_circles))):
        for idx_smaller in range(0, idx_bigger):
            if idx_smaller in containment_dict:
                continue

            bigger_circle = sorted_circles[idx_bigger]
            smaller_circle = sorted_circles[idx_smaller]


            if bigger_circle.contains(smaller_circle, overlap_factor=overlap_factor):
                containment_dict[idx_smaller] = idx_bigger
    
    return containment_dict



def find_containment_hierarchy_naive(circles, overlap_factor=1.0):
    containment_dict = defaultdict(list)

    for idx_1, circle_1 in enumerate(circles):
        for idx_2, circle_2 in enumerate(circles):
            if idx_1 == idx_2:
                continue
            if circle_1.contains(circle_2, overlap_factor=overlap_factor):
                containment_dict[idx_2].append(idx_1)
    
    return containment_dict

if __name__ == "__main__":
    pass

