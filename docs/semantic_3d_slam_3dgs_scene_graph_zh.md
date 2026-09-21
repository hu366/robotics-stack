# 语义 3D-SLAM、3DGS 与 3D 场景图需求文档

## 1. 目标

本文档把“完成 3D 场景图的创建”细化为一条可实现、可验证的感知与世界模型主线。

目标链路是：

```text
RGB-D 序列 -> 位姿估计 / 3D-SLAM -> 快速 3DGS 建图 -> 3DGS 实例切割 -> 开放词汇语义理解 -> 3D 场景图 -> 世界模型 -> grounding / planner
```

该链路可参考 DovSG 的系统思路：先从 RGB-D 扫描和 SLAM 位姿构建 3D scene graph，再用视觉语言模型或大语言模型补足对象语义、空间关系和任务可用的符号信息。

本文档是开发需求与架构约束，不代表当前仓库已经实现完整 3D-SLAM、3DGS 或 DovSG 级别能力。

## 2. 当前代码现状

当前仓库已有的基础：

- `interfaces/scene_graph.py` 已定义 `SceneGraph`、`SceneNode`、`SceneEdge`、`ConstraintState`。
- `interfaces/perception.py` 已定义 RGB-D 帧、相机内参、相机位姿、点云帧、几何地图摘要和 artifact 摘要。
- `modules/world_model/map_fuser.py` 已有轻量点云融合基线。
- `modules/world_model/scene_graph_builder.py` 已能从仿真已知对象和包围盒推导 `supports`、`contacts`、`can_reach` 等关系。
- `docs/rgbd_mapping_grounding_architecture.md` 已定义 stage-1 的 RGB-D 融合、场景图和 grounding 合同。

当前仍缺失：

- 真正的视觉里程计或 3D-SLAM 位姿估计。
- RGB-D 到 3D Gaussian Splatting 的快速建图后端。
- 3DGS 级别的实例切割、语义标注和 object-level map。
- 从 3DGS object map 构建可查询 3D scene graph 的完整管线。
- 语义理解结果与 `WorldState` / `GroundingResult` 的稳定接线。

## 3. 模块边界

### 3.1 `modules/world_model/`

负责保存和更新世界状态，是该能力的主归属模块。

应新增的子能力：

- `slam_frontend`：接收 RGB-D 序列，输出每帧位姿、关键帧、轨迹质量和回环信息。
- `gaussian_map`：管理 3DGS 地图 artifact、训练状态、地图摘要和渲染入口。
- `gaussian_segmentation`：把 3DGS 切割成 object-level / part-level anchors。
- `semantic_scene_graph`：把切割结果、语义标签和空间关系转成 `SceneGraph`。

`world_model` 可以保存 3DGS artifact 引用和摘要，但不应把密集 Gaussian 参数直接塞进 `WorldState`。

### 3.2 `modules/grounding/`

负责把自然语言目标绑定到场景图中的具体节点。

它可以查询：

- `SceneGraph.nodes`
- `SceneGraph.edges`
- 节点的语义标签、属性、置信度和来源
- 节点对应的 geometry anchor

它不应该直接扫描 3DGS 参数、训练日志或原始 RGB-D 帧。

### 3.3 `modules/planner/`

负责消费 grounding 后的候选对象、候选表面、候选位姿和约束。

planner 不应该依赖 3DGS 后端细节。3DGS 只通过场景图、几何摘要、artifact 引用和约束状态影响 planner。

### 3.4 `apps/`

应提供可运行入口，例如：

- `apps/build_semantic_3d_map.py`
- `apps/export_scene_graph.py`
- `apps/query_scene_graph.py`

入口需要写出结构化 artifact 和 trace，不能只在屏幕上打印非结构化结果。

## 4. 推荐管线

### 4.1 RGB-D 输入与同步

输入应至少包含：

- RGB 图像序列
- 深度图序列
- 相机内参
- 时间戳
- 可选初始位姿或仿真真值位姿

输出：

- `RGBDFrame` manifest
- 有效深度比例
- 图像和深度 artifact 引用
- 数据质量诊断

### 4.2 3D-SLAM / 位姿估计

近期可先保留两种模式：

- `sim_ground_truth`：仿真真值位姿，适合最小闭环和测试。
- `slam_estimated`：DROID-SLAM / DovSG 风格视觉 SLAM 位姿，适合真实 RGB-D 序列。

SLAM 前端输出：

- 每帧 `CameraPose`
- 关键帧列表
- 轨迹 artifact
- 位姿协方差或质量分数
- 回环和重定位诊断

SLAM 只负责位姿和轨迹，不负责语义 grounding。

### 4.3 快速 RGB-D 到 3DGS 建图

3DGS 建图后端的目标是快速生成可渲染、可切割、可绑定对象的三维表示。

输出不直接进入 planner，而是写成 artifact：

- Gaussian checkpoint
- 稀疏或稠密点初始化 artifact
- camera trajectory
- rendered debug views
- map quality metrics

世界模型只保存：

- `map_id`
- frame count
- bounds
- Gaussian 数量
- 训练耗时
- artifact 路径
- provenance

### 4.4 3DGS 切割

切割目标是把全局 3DGS 地图分成可作为场景图节点的几何 anchor。

推荐输出：

- `segment_id`
- 对应 Gaussian indices 或 mask artifact
- 3D bbox
- centroid
- support surface estimate
- visible frames
- segmentation confidence

切割可以来自多种来源：

- RGB 图像 2D mask 多视角提升到 3D
- 3D clustering
- 语义相似度聚类
- DovSG 风格 object map 构建

无论来源如何，都必须通过统一结构写出，不允许在下游临时解释 raw mask。

### 4.5 语义理解

语义理解负责给 segment / object anchor 添加语义标签和属性。

输入：

- 每个 segment 的多视角 crop
- 3D bbox 和上下文关系
- 可选 open-vocabulary detector / VLM / LLM 输出

输出：

- `label`
- `semantic_tags`
- `properties`
- `confidence`
- `source_attribution`

语义理解不能直接改写任务计划，只能丰富场景图节点和边。

### 4.6 3D 场景图构建

场景图节点应至少包括：

- `object`
- `surface`
- `region`
- `agent`
- `frame`

边应至少支持：

- `supports` / `supported_by`
- `inside`
- `near`
- `contacts`
- `left_of` / `right_of` / `in_front_of` / `behind`
- `visible_from`
- `semantic_related`

场景图必须保留 provenance，例如：

- `source_type=slam_estimated_pose`
- `source_type=rgbd_3dgs`
- `source_type=gaussian_segmentation`
- `source_type=open_vocab_detector`
- `source_type=vlm_labeler`
- `source_type=inferred_rule`

## 5. Artifact 合同

推荐目录：

```text
artifacts/semantic_maps/<mapping_run_id>/
  manifest.json
  frames/
  slam/
    trajectory.json
    keyframes.json
    diagnostics.json
  gaussian/
    checkpoint/
    map_summary.json
    renders/
  segments/
    segments.json
    masks/
  semantics/
    labels.json
    crops/
  scene_graph/
    scene_graph.json
    constraints.json
  traces/
    semantic_mapping_trace.json
```

`manifest.json` 至少包含：

- `mapping_run_id`
- `scene_id`
- `input_source`
- `pose_source`
- `frame_count`
- `backend_versions`
- 各阶段 artifact 路径

trace 至少记录：

- `rgbd_loaded`
- `slam_completed`
- `gaussian_map_built`
- `gaussian_segmented`
- `semantics_labeled`
- `scene_graph_built`
- `world_state_updated`

每个事件应包含耗时、输入输出数量、失败原因和 artifact 引用。

## 6. 接口扩展建议

应在 `interfaces/perception.py` 或新文件中补充以下结构：

- `SlamTrajectorySummary`
- `GaussianMapSummary`
- `GaussianSegment`
- `SemanticLabelObservation`
- `SemanticMappingTracePayload`

这些结构应满足：

- 可序列化为 JSON。
- 包含 `provenance`。
- 只携带摘要和 artifact 引用，不内联大数组。
- 可被 `WorldState` 持有或引用。

## 7. 阶段验收

### M0：仿真真值 + 点云场景图

目标：

- 用现有 MuJoCo RGB-D 和真值位姿生成点云地图。
- 构建 `SceneGraph`。
- 写出 `WorldState` snapshot。

验收：

- `scene_graph.json` 中有对象节点、表面节点和支持关系。
- grounding 能从场景图绑定“瓶子”“托盘”等对象。
- 测试覆盖序列化、场景图构建和 grounding 查询。

### M1：接入 SLAM 位姿

目标：

- 支持 `slam_estimated` pose source。
- 保存轨迹、关键帧和诊断。

验收：

- 同一 RGB-D 序列可用真值位姿和 SLAM 位姿分别建图。
- trace 中能看出每帧位姿来源和失败帧。

### M2：快速 3DGS 建图

目标：

- 用 RGB-D 序列和位姿训练或增量构建 3DGS artifact。
- 生成地图摘要和 debug renders。

验收：

- `gaussian/map_summary.json` 存在。
- 能从固定视角渲染非空图像。
- `WorldState` 只引用 artifact，不内联 Gaussian 参数。

### M3：3DGS 切割

目标：

- 将 3DGS 切割成 object-level segments。
- 输出 bbox、centroid、mask artifact 和置信度。

验收：

- `segments/segments.json` 中每个 segment 可追溯到 Gaussian mask 或 indices。
- 每个 object segment 都能映射为 `SceneNode.geometry_anchor_id`。

### M4：开放词汇语义理解

目标：

- 对 segment 做 open-vocabulary label。
- 生成语义标签、属性和置信度。

验收：

- `semantics/labels.json` 中有 label、confidence 和 source。
- 场景图节点包含 `semantic_tags` 和 `properties`。
- grounding 能用中文或英文别名查到同一对象。

### M5：DovSG 风格任务可用场景图

目标：

- 场景图不仅包含对象，还包含空间关系、支持关系、可见性和任务相关属性。

验收：

- `query_scene_graph` 能回答“桌子上的瓶子”“托盘附近的杯子”等查询。
- planner 只消费 grounding 输出，不接触 3DGS 内部 artifact。

## 8. 风险与约束

- 3DGS、DROID-SLAM、开放词汇分割模型依赖重，不应塞进默认轻量开发环境。
- 建议把重型后端做成 optional worker / adapter，主仓库保留 typed contract、artifact protocol 和 mock / fixture 测试。
- 真实实时性能必须单独评测，不能用离线 demo 冒充高速在线建图。
- 语义标签存在不确定性，必须保留 confidence 和 provenance。
- 任何下游规划成功率宣称都必须说明使用的 pose source、semantic source 和 map backend。
