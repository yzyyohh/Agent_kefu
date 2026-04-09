from __future__ import annotations


class RobotActionTool:
    name = "robot_action"
    description = "给出机器人可执行动作建议，例如回充、清洁滚刷、重置地图"

    def run(self, issue_type: str, severity: str = "normal") -> str:
        base = {
            "stuck": "建议动作: 暂停任务 -> 抬起机器检查主刷和万向轮 -> 清理缠绕物 -> 继续任务。",
            "low_suction": "建议动作: 清空尘盒 -> 清洗/更换滤网 -> 检查吸口堵塞。",
            "odor": "建议动作: 取下拖布清洗晾干 -> 清理污水箱 -> 启动自清洁。",
            "navigation": "建议动作: 擦拭传感器 -> 重新建图 -> 检查禁区设置。",
        }
        action = base.get(issue_type, "建议动作: 先执行基础保养并重新启动设备。")
        if severity == "high":
            return action + " 若仍失败，建议联系售后并提供错误码。"
        return action
