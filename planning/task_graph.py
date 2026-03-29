"""Task graph / Behavior Tree interpreter.

M0 stub: placeholder for M3 implementation.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any


class NodeStatus(Enum):
    """Behavior tree node status."""
    SUCCESS = "success"
    RUNNING = "running"
    FAILURE = "failure"


@dataclass
class TaskNode:
    """Base task node definition."""
    name: str
    preconditions: Optional[Dict[str, Any]] = None
    postconditions: Optional[Dict[str, Any]] = None

    def tick(self, state: Dict) -> NodeStatus:
        """Execute one tick of the node.

        Args:
            state: Current environment state

        Returns:
            NodeStatus indicating progress
        """
        # M0 stub - to be implemented in M3
        return NodeStatus.SUCCESS


class TaskGraph:
    """Task graph / Behavior Tree executor."""

    def __init__(self, nodes: List[TaskNode] = None):
        """Initialize task graph.

        Args:
            nodes: List of task nodes in execution order
        """
        self.nodes = nodes or []
        self.current_index = 0

    def add_node(self, node: TaskNode):
        """Add a node to the graph."""
        self.nodes.append(node)

    def tick(self, state: Dict) -> NodeStatus:
        """Execute current node.

        Args:
            state: Current environment state

        Returns:
            NodeStatus of current node
        """
        if not self.nodes:
            return NodeStatus.SUCCESS

        if self.current_index >= len(self.nodes):
            return NodeStatus.SUCCESS

        node = self.nodes[self.current_index]
        status = node.tick(state)

        if status == NodeStatus.SUCCESS:
            self.current_index += 1
        elif status == NodeStatus.FAILURE:
            pass  # Handle failure

        return status

    def reset(self):
        """Reset task graph to initial state."""
        self.current_index = 0
