from nuscenes.nuscenes import NuScenes


def find_scene_by_timestamp(nusc, timestamp):
    """
    根据时间戳定位场景。
    Args:
        nusc: NuScenes 数据集实例
        timestamp: 图像帧的时间戳（整数，单位：微秒）
    Returns:
        scene_name: 所属场景名称（字符串）
        scene: 所属场景信息（字典）
    """
    for scene in nusc.scene:
        # 获取场景的第一个样本和最后一个样本的时间戳
        first_sample = nusc.get('sample', scene['first_sample_token'])
        last_sample = nusc.get('sample', scene['last_sample_token'])

        start_time = first_sample['timestamp']  # 起始时间
        end_time = last_sample['timestamp']  # 结束时间

        # 判断时间戳是否在该场景的时间范围内
        if start_time <= timestamp <= end_time:
            return scene['name'], scene  # 返回场景名称和信息
    return None, None  # 如果没有匹配的场景，返回 None


# 使用示例
if __name__ == "__main__":
    # 初始化 NuScenes 数据集
    nusc = NuScenes(version='v1.0-trainval', dataroot='./data/nuscenes', verbose=True)

    # 给定帧的时间戳（单位：微秒）
    frame_timestamp = 1526915645412465  # 示例时间戳

    # 定位场景
    scene_name, scene_info = find_scene_by_timestamp(nusc, frame_timestamp)

    if scene_info:
        print(f"Timestamp {frame_timestamp} belongs to scene: {scene_name}")
        print(f"Scene info: {scene_info}")
    else:
        print(f"Timestamp {frame_timestamp} does not belong to any scene.")
