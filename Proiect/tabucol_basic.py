import random

def tabucol(G, k, iterations=50000, tabu_size=None):
	nodes = list(G.nodes())
	if not nodes:
		return True, {}

	# Initial random coloring with k colors
	coloring = {node: random.randint(0, k-1) for node in nodes}

	# Fixed tabu tenure (no dynamic formula)
	if tabu_size is None:
		tabu_size = 7  # fixed value, no dependency on k

	# Array tracking until which iteration a color is tabu for a given node
	tabu_until = {node: [0] * k for node in nodes}

	for i in range(iterations):
		# Find all nodes in conflict (no O(1) conflict list)
		conflict_nodes = [node for node in nodes if any(
			coloring[node] == coloring[nbr] for nbr in G.neighbors(node))]
		if not conflict_nodes:
			return True, coloring  # Valid k-coloring found

		# Pick a node involved in a conflict (uniform random from conflicted nodes)
		node_to_move = random.choice(conflict_nodes)

		best_move = None
		min_conflicts = float('inf')

		# Try moving node_to_move to a different color
		for new_color in range(k):
			if new_color == coloring[node_to_move]:
				continue
			if tabu_until[node_to_move][new_color] > i:
				continue

			# Count number of neighbors with new_color (no O(1) adj_color_table)
			current_count = sum(1 for nbr in G.neighbors(node_to_move)
								if coloring[nbr] == new_color)

			if current_count < min_conflicts:
				min_conflicts = current_count
				best_move = new_color

		if best_move is not None:
			old_color = coloring[node_to_move]
			# Ban moving back to old_color for tabu_size iterations
			tabu_until[node_to_move][old_color] = i + tabu_size
			coloring[node_to_move] = best_move

	return False, None
