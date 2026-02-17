"""Lightweight DAG with typed nodes for tiling."""

from __future__ import annotations


class Graph:
    """Directed acyclic graph with typed nodes and ordered inputs.

    Nodes have string IDs and string op types.
    Edges are ordered: inputs[node] is an ordered list of predecessor node IDs,
    where position in the list corresponds to the input slot.
    """

    def __init__(self):
        self.nodes: dict[str, str] = {}  # node_id -> op_type
        self.inputs: dict[str, list[str]] = {}  # node_id -> [input_node_ids]
        self._outputs: list[str] | None = None

    # ── Construction ──────────────────────────────────────────────

    def add_node(self, node_id: str, op_type: str) -> Graph:
        node_id = str(node_id)
        if node_id in self.nodes:
            if self.nodes[node_id] != op_type:
                raise ValueError(
                    f"Node '{node_id}' already defined as '{self.nodes[node_id]}', "
                    f"cannot redefine as '{op_type}'"
                )
            return self
        self.nodes[node_id] = op_type
        self.inputs[node_id] = []
        return self

    def add_edge(self, from_id: str, to_id: str) -> Graph:
        from_id, to_id = str(from_id), str(to_id)
        if from_id not in self.nodes:
            raise ValueError(f"Node '{from_id}' not defined")
        if to_id not in self.nodes:
            raise ValueError(f"Node '{to_id}' not defined")
        self.inputs[to_id].append(from_id)
        return self

    def set_outputs(self, output_ids: list[str]) -> Graph:
        self._outputs = [str(x) for x in output_ids]
        return self

    @property
    def outputs(self) -> list[str]:
        if self._outputs is not None:
            return self._outputs
        # Auto-detect: sink nodes (nodes with no outgoing edges).
        # A node has an outgoing edge if it appears in some other node's input list.
        has_outgoing = set()
        for inp_list in self.inputs.values():
            has_outgoing.update(inp_list)
        sinks = [n for n in self.nodes if n not in has_outgoing]
        return sinks if sinks else list(self.nodes.keys())

    def consumers(self, node_id: str) -> list[str]:
        """Return nodes that have *node_id* in their input list."""
        return [n for n in self.nodes if node_id in self.inputs.get(n, [])]

    # ── DSL parser ────────────────────────────────────────────────

    @classmethod
    def from_spec(cls, spec: str, outputs: list[str] | None = None) -> Graph:
        """Build a graph from a compact DSL string.

        Format::

            "1:mm->2:relu->3:add; 4:sigmoid->3"

        - ``id:type`` defines a node (first occurrence).
        - ``id`` alone references an already-defined node.
        - ``->`` creates an edge between consecutive items in a chain.
        - ``;`` separates independent chains.

        Edge order determines input slot position:
        the first edge into a node occupies slot 0, the second slot 1, etc.
        """
        g = cls()
        spec = spec.strip()
        if not spec:
            return g

        chains = [c.strip() for c in spec.split(";")]

        for chain in chains:
            if not chain:
                continue
            parts = [p.strip() for p in chain.split("->")]
            prev_id = None

            for part in parts:
                if ":" in part:
                    node_id, op_type = part.split(":", 1)
                    node_id = node_id.strip()
                    op_type = op_type.strip()
                    g.add_node(node_id, op_type)
                else:
                    node_id = part.strip()
                    if node_id not in g.nodes:
                        raise ValueError(
                            f"Node '{node_id}' referenced but not defined"
                        )

                if prev_id is not None:
                    g.add_edge(prev_id, node_id)

                prev_id = node_id

        if outputs is not None:
            g.set_outputs([str(x) for x in outputs])

        return g

    # ── Visualisation (Graphviz DOT) ──────────────────────────────

    def to_dot(self, title: str = "Graph") -> str:
        lines = ["digraph {", f'  label="{title}"']
        for nid, op in self.nodes.items():
            lines.append(f'  n{nid} [label="{nid}:{op}"]')
        for nid, inps in self.inputs.items():
            for inp in inps:
                lines.append(f"  n{inp} -> n{nid}")
        lines.append("}")
        return "\n".join(lines)

    def to_dot_with_tiling(self, tiling_result, title: str = "Tiled Graph") -> str:
        """Render the graph with tiles as coloured clusters."""
        colors = [
            "#cce5ff", "#d4edda", "#fff3cd", "#f8d7da", "#d1ecf1",
            "#e2d5f1", "#fce4ec", "#e8f5e9", "#fff8e1", "#e3f2fd",
        ]
        lines = ["digraph {", f'  label="{title}"', "  compound=true"]

        # Map each node to its tile index (if any)
        node_to_tile: dict[str, int] = {}
        for i, t in enumerate(tiling_result.tiles):
            for n in t.covered_nodes:
                node_to_tile[n] = i

        # Tile clusters
        for i, t in enumerate(tiling_result.tiles):
            color = colors[i % len(colors)]
            lines.append(f"  subgraph cluster_{i} {{")
            lines.append(f'    label="tile {i} (pat {t.pattern_id})"')
            lines.append(f'    style=filled; fillcolor="{color}"')
            for n in sorted(t.covered_nodes):
                lines.append(f'    n{n} [label="{n}:{self.nodes[n]}"]')
            lines.append("  }")

        # Uncovered nodes
        for n in sorted(tiling_result.uncovered):
            lines.append(
                f'  n{n} [label="{n}:{self.nodes[n]}" '
                f"style=dashed color=gray fontcolor=gray]"
            )

        # Edges
        for nid, inps in self.inputs.items():
            for inp in inps:
                lines.append(f"  n{inp} -> n{nid}")

        lines.append("}")
        return "\n".join(lines)

    # ── Dunder helpers ────────────────────────────────────────────

    def __len__(self):
        return len(self.nodes)

    def __repr__(self):
        return f"Graph(nodes={self.nodes}, inputs={self.inputs}, outputs={self.outputs})"
