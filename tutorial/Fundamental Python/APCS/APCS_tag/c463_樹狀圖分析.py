'''
RecursionError: maximum recursion depth exceeded in comparison

95%
'''
def get_node_height(node):

    if node not in nodes_dict:
        return 0, 0

    children_node_heights = []
    children_H = []

    for child_node in nodes_dict[node]:
        child_H, child_local_height = get_node_height(child_node)
        children_node_heights.append(child_local_height)
        children_H.append(child_H)

    H = max(children_node_heights) + 1 + sum(children_H)
    local_height = max(children_node_heights) + 1

    return  H, local_height

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

H, _ = get_node_height(root_node)

print(H)