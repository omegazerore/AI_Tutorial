'''
AC 1.3s, 39.4MB
'''
def triangle_area(n):

    return (n * (n+1)) // 2


def get_node_height(node):

    if node not in nodes_dict:
        return 0, 0

    node_temp = node
    linear_counting = 0
    while len(nodes_dict[node_temp]) == 1:
        node_temp = nodes_dict[node_temp][0]
        linear_counting += 1
        if node_temp not in nodes_dict:
            return triangle_area(linear_counting), linear_counting

    if linear_counting != 0:
        children_node_heights = []
        children_H = []
        for child_node in nodes_dict[node_temp]:
            child_H, child_local_height = get_node_height(child_node)
            children_node_heights.append(child_local_height)
            children_H.append(child_H)
        H_child = sum(children_H)
        h_child = max(children_node_heights)
        return H_child + (linear_counting + 1) * h_child + triangle_area(linear_counting + 1), h_child + linear_counting + 1

    children_node_heights = []
    children_H = []

    for child_node in nodes_dict[node]:
        child_H, child_local_height = get_node_height(child_node)
        children_node_heights.append(child_local_height)
        children_H.append(child_H)

    H = max(children_node_heights) + 1 + sum(children_H)
    local_height = max(children_node_heights) + 1

    return H, local_height

# DP

total_number_of_nodes = int(input())

nodes_dict = {}
leaf_to_root_dict = {}
leafs = []

t_root = 1

for i in range(total_number_of_nodes):

    children_nodes_info = list(map(int, input().split()))

    if children_nodes_info[0] != 0:
        nodes_dict[i+1] = children_nodes_info[1:]
        for leaf in children_nodes_info[1:]:
            leaf_to_root_dict[leaf] = i+1
    else:
        leafs.append(i+1)

# 1. from a leaf to find the root_node
leaf = leafs[0]
root_node = None

while leaf is not None:
    root_node = leaf
    leaf = leaf_to_root_dict.get(leaf, None)
print(root_node)

# Dynamic programming

if len(leaf_to_root_dict) == total_number_of_nodes:
    H = triangle_area(total_number_of_nodes - 1)
else:
    H, _ = get_node_height(root_node)

print(H)
