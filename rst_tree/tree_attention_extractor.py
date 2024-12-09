from collections import defaultdict
import math

class RST_Tree_AttentionSpan_Parser:
  def __init__():
    pass

  @staticmethod
  def _get_height_map_of_tree(root):
    height_map = defaultdict(list)
    def dfs(node):
      height = 0
      if  node.left and  node.right:
        height_left = dfs(node.left)
        height_right = dfs(node.right)
        height = max(height_left, height_right) + 1
      height_map[height].append(node)
      return height
    tree_height = dfs(root)
    return height_map, tree_height

  @staticmethod
  def parse_attention_spans(root, height_map = None, tree_height = None, n_layers = 6):
    if not height_map or not tree_height:
      height_map, tree_height = RST_Tree_AttentionSpan_Parser._get_height_map_of_tree(root)

    layers_so_far = -1
    prev_elements = set()
    attention_spans = {}
    for curr_layer in range(n_layers):
      candidates = prev_elements
      height_bound = math.floor(curr_layer/(n_layers-1) * tree_height)
      for h in range(height_bound, layers_so_far, -1):
        candidates |= set(height_map[h])

      filtered_elements = set()
      for candidate in candidates:
        if candidate.parent not in candidates:
          filtered_elements.add(candidate)

      attention_spans[curr_layer] = sorted([(n.start, n.end) for n in filtered_elements])
      prev_elements = filtered_elements
    return attention_spans