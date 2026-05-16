from __future__ import annotations


def build_task_parse_prompt(instruction: str) -> str:
    return (
        "将用户指令压缩为单个受控 ParsedIntent JSON 对象。"
        "只能输出 JSON，不要使用 markdown，不要解释。\n"
        "要求：\n"
        "1. 只能输出以下字段：goal, action, arguments, spatial_relation, confidence, "
        "ambiguities, omitted_details。\n"
        "2. goal 只能是 place_object, open_object, close_object, insert_object, inspect_scene。\n"
        "3. action 只能是 place, move, open, close, insert, inspect。\n"
        "4. arguments 是数组，元素字段只能是 role, text, entity_type；"
        "role 只能是 target_object 或 target_location。\n"
        "5. spatial_relation 只能是 on, in, to, inside 或 null。\n"
        "6. 如果原始指令包含多步、条件分支、指代或附加约束，但当前 schema 无法完整表示，"
        "请把被忽略的信息写入 omitted_details。\n"
        "7. confidence 是 0 到 1 的数字；ambiguities 和 omitted_details 都是字符串数组。\n"
        "8. 不允许新增任何其他键。\n"
        f"用户指令：{instruction}\n"
        "输出示例："
        '{"goal":"place_object","action":"place","arguments":[{"role":"target_object",'
        '"text":"瓶子","entity_type":"object"},{"role":"target_location","text":"托盘",'
        '"entity_type":"object"}],"spatial_relation":"on","confidence":0.92,'
        '"ambiguities":[],"omitted_details":[]}'
    )
