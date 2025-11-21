"""
Graph visualization utilities for debugging and documentation.

Provides stateless, pure functions to export graph structure to various formats.
These utilities are read-only and have zero effect on graph execution.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .graph import GraphRunner, GraphNode


def to_mermaid(graph: 'GraphRunner', include_metadata: bool = False) -> str:
    """
    Export graph structure to Mermaid flowchart format.

    Generates a Mermaid flowchart diagram showing nodes and dependencies.
    Useful for debugging, documentation, and visualizing complex DAGs.

    Args:
        graph: GraphRunner instance to visualize
        include_metadata: If True, include node type and flags in labels

    Returns:
        Mermaid flowchart syntax as string
    """
    lines = ["flowchart TD"]

    # Generate node declarations with optional metadata
    for node_id, node in graph._nodes.items():
        label = node_id

        if include_metadata:
            parts = [node_id]

            # Node type
            node_type = type(node.executable).__name__
            if node_type not in ("function", "method"):
                parts.append(node_type)

            # Flags
            if node.run_on_failure:
                parts.append("cleanup")
            if node.failure_mode == "soft_fail":
                parts.append("soft")
            if node.output_key:
                parts.append(f"→{node.output_key}")

            label = " | ".join(parts)

        lines.append(f'    {node_id}["{label}"]')

    # Generate edges from parent -> child relationships
    for node_id, parents in graph._parents.items():
        for parent_id in parents:
            lines.append(f"    {parent_id} --> {node_id}")

    return "\n".join(lines)


def to_dot(graph: 'GraphRunner', include_metadata: bool = False) -> str:
    """
    Export graph structure to Graphviz DOT format.

    Generates a DOT file for rendering with Graphviz tools.

    Args:
        graph: GraphRunner instance to visualize
        include_metadata: If True, include node type and flags in labels

    Returns:
        DOT format as string
    """
    lines = ["digraph G {", "    rankdir=TD;"]

    # Node declarations
    for node_id, node in graph._nodes.items():
        label = node_id
        attrs = []

        if include_metadata:
            parts = [node_id]

            node_type = type(node.executable).__name__
            if node_type not in ("function", "method"):
                parts.append(node_type)

            if node.run_on_failure:
                parts.append("cleanup")
                attrs.append('color=orange')
            if node.failure_mode == "soft_fail":
                parts.append("soft")
                attrs.append('style=dashed')
            if node.output_key:
                parts.append(f"→{node.output_key}")

            label = "\\n".join(parts)

        attr_str = f", {', '.join(attrs)}" if attrs else ""
        lines.append(f'    {node_id} [label="{label}"{attr_str}];')

    # Edges
    for node_id, parents in graph._parents.items():
        for parent_id in parents:
            lines.append(f"    {parent_id} -> {node_id};")

    lines.append("}")
    return "\n".join(lines)
