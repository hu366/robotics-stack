from __future__ import annotations

from dataclasses import dataclass, field

from interfaces.common import SourceAttribution
from interfaces.scene_graph import SceneEdge, SceneGraph, SceneNode


@dataclass(slots=True)
class SceneObjectObservation:
    object_id: str
    label: str
    category: str
    position: list[float]
    bbox_min: list[float]
    bbox_max: list[float]
    static: bool
    attributes: dict[str, str] = field(default_factory=dict)
    provenance: list[SourceAttribution] = field(default_factory=list)


class SceneGraphBuilder:
    """Build a deterministic scene graph from simulator-known objects and geometry."""

    def __init__(
        self,
        support_gap: float = 0.03,
        contact_gap: float = 0.02,
        reachable_radius: float = 0.9,
    ) -> None:
        self._support_gap = support_gap
        self._contact_gap = contact_gap
        self._reachable_radius = reachable_radius

    def build(
        self,
        scene_id: str,
        objects: list[SceneObjectObservation],
        robot_position: list[float],
    ) -> SceneGraph:
        nodes = [self._node_from_object(obj) for obj in objects]
        nodes.append(self._robot_node(robot_position))

        edges: list[SceneEdge] = []
        for lower in objects:
            for upper in objects:
                if lower.object_id == upper.object_id:
                    continue
                if self._supports(lower, upper):
                    edges.append(
                        self._edge(
                            edge_id=f"{lower.object_id}_supports_{upper.object_id}",
                            source=lower.object_id,
                            target=upper.object_id,
                            relation_type="supports",
                            relation_family="support",
                        )
                    )
                    edges.append(
                        self._edge(
                            edge_id=f"{upper.object_id}_supported_by_{lower.object_id}",
                            source=upper.object_id,
                            target=lower.object_id,
                            relation_type="supported_by",
                            relation_family="support",
                        )
                    )
                elif self._contacts(lower, upper):
                    edges.append(
                        self._edge(
                            edge_id=f"{lower.object_id}_contacts_{upper.object_id}",
                            source=lower.object_id,
                            target=upper.object_id,
                            relation_type="contacts",
                            relation_family="contact",
                        )
                    )

        for obj in objects:
            if self._reachable(robot_position, obj.position):
                edges.append(
                    self._edge(
                        edge_id=f"robot_base_can_reach_{obj.object_id}",
                        source="robot_base",
                        target=obj.object_id,
                        relation_type="can_reach",
                        relation_family="reachability",
                    )
                )
                edges.append(
                    self._edge(
                        edge_id=f"{obj.object_id}_reachable_by_robot_base",
                        source=obj.object_id,
                        target="robot_base",
                        relation_type="reachable_by",
                        relation_family="reachability",
                    )
                )

        return SceneGraph(
            scene_id=scene_id,
            root_frame_id="world",
            nodes=nodes,
            edges=edges,
            summary={
                "object_count": len(nodes),
                "edge_count": len(edges),
                "support_gap_m": self._support_gap,
                "contact_gap_m": self._contact_gap,
                "reachable_radius_m": self._reachable_radius,
            },
        )

    def _node_from_object(self, obj: SceneObjectObservation) -> SceneNode:
        bbox_extent = [
            max_value - min_value
            for min_value, max_value in zip(obj.bbox_min, obj.bbox_max, strict=True)
        ]
        return SceneNode(
            node_id=obj.object_id,
            label=obj.label,
            node_type=obj.category,
            center_pose=[*obj.position, 0.0, 0.0, 0.0, 1.0],
            bbox_extent_m=bbox_extent,
            geometry_anchor_id=obj.object_id,
            semantic_tags=[obj.category],
            state_tags=["static"] if obj.static else ["dynamic"],
            properties=dict(obj.attributes),
            provenance=list(obj.provenance),
        )

    def _robot_node(self, robot_position: list[float]) -> SceneNode:
        return SceneNode(
            node_id="robot_base",
            label="robot_base",
            node_type="robot",
            center_pose=[*robot_position, 0.0, 0.0, 0.0, 1.0],
            bbox_extent_m=[0.0, 0.0, 0.0],
            geometry_anchor_id="robot_base",
            semantic_tags=["robot"],
            state_tags=["static"],
        )

    def _edge(
        self,
        edge_id: str,
        source: str,
        target: str,
        relation_type: str,
        relation_family: str,
    ) -> SceneEdge:
        return SceneEdge(
            edge_id=edge_id,
            source_node_id=source,
            target_node_id=target,
            relation_type=relation_type,
            relation_family=relation_family,
            provenance=[
                SourceAttribution(
                    source_type="inferred_rule",
                    source_id="scene_graph_builder",
                    confidence=1.0,
                )
            ],
        )

    def _supports(self, lower: SceneObjectObservation, upper: SceneObjectObservation) -> bool:
        lower_top = float(lower.bbox_max[2])
        upper_bottom = float(upper.bbox_min[2])
        vertical_gap = upper_bottom - lower_top
        return 0.0 <= vertical_gap <= self._support_gap and self._xy_overlap(lower, upper)

    def _contacts(self, left: SceneObjectObservation, right: SceneObjectObservation) -> bool:
        return self._bbox_distance(left, right) <= self._contact_gap

    def _bbox_distance(self, left: SceneObjectObservation, right: SceneObjectObservation) -> float:
        distance = 0.0
        for axis in range(3):
            left_min = float(left.bbox_min[axis])
            left_max = float(left.bbox_max[axis])
            right_min = float(right.bbox_min[axis])
            right_max = float(right.bbox_max[axis])
            if left_max < right_min:
                gap = right_min - left_max
            elif right_max < left_min:
                gap = left_min - right_max
            else:
                gap = 0.0
            distance += gap * gap
        return float(distance ** 0.5)

    def _xy_overlap(self, left: SceneObjectObservation, right: SceneObjectObservation) -> bool:
        left_min_x = float(left.bbox_min[0])
        left_max_x = float(left.bbox_max[0])
        right_min_x = float(right.bbox_min[0])
        right_max_x = float(right.bbox_max[0])
        left_min_y = float(left.bbox_min[1])
        left_max_y = float(left.bbox_max[1])
        right_min_y = float(right.bbox_min[1])
        right_max_y = float(right.bbox_max[1])
        overlap_x = left_min_x <= right_max_x and right_min_x <= left_max_x
        overlap_y = left_min_y <= right_max_y and right_min_y <= left_max_y
        return bool(overlap_x and overlap_y)

    def _reachable(self, robot_position: list[float], object_position: list[float]) -> bool:
        dx = float(robot_position[0]) - float(object_position[0])
        dy = float(robot_position[1]) - float(object_position[1])
        dz = float(robot_position[2]) - float(object_position[2])
        return bool((dx * dx + dy * dy + dz * dz) ** 0.5 <= self._reachable_radius)
