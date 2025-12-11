import os
from collections import defaultdict

def count_european_records():
    """
    统计european-st-t-database-1.0.0文件夹中不同的记录文件名

    统计规则：
    2. 只处理.atr、.hea、.dat扩展名的文件
    3. 去掉扩展名后统计不同的文件名
    4. e103.hea和e103.dat算作同一个记录e103
    """

    # 数据库路径
    db_path = "/home/yogsothoth/DataSet/mit-bih-noise-stress-test-database-1.0.0"

    # 允许的文件扩展名
    allowed_extensions = {'.atr', '.hea', '.dat'}

    # 存储文件名和对应的扩展名
    record_dict = defaultdict(set)

    print("正在扫描文件夹...")
    print(f"数据库路径: {db_path}")
    print("-" * 60)

    if not os.path.exists(db_path):
        print(f"错误：路径 {db_path} 不存在")
        return

    # 遍历文件夹中的所有文件
    for filename in os.listdir(db_path):
        # # 检查是否以'e'开头
        # if not filename.startswith('cu'):
        #     continue

        # 获取文件扩展名
        _, ext = os.path.splitext(filename)

        # 检查扩展名是否在允许的列表中
        if ext.lower() in allowed_extensions:
            # 去掉扩展名，获取基础文件名
            base_name = os.path.splitext(filename)[0]
            record_dict[base_name].add(ext.lower())

    # 统计结果
    total_records = len(record_dict)
    print(f"\n统计结果:")
    print(f"总共找到 {total_records} 个不同的记录文件")
    print("-" * 60)

    # 按文件名排序输出
    sorted_records = sorted(record_dict.keys())

    print("\n详细的文件列表:")
    print("文件名\t\t包含的文件类型")
    print("-" * 40)

    for record_name in sorted_records:
        extensions = sorted(record_dict[record_name])
        ext_str = ", ".join(extensions)
        print(f"{record_name}\t\t{ext_str}")

    # 统计每种扩展名的数量
    ext_count = defaultdict(int)
    for extensions in record_dict.values():
        for ext in extensions:
            ext_count[ext] += 1

    print("\n扩展名统计:")
    print("扩展名\t\t文件数量")
    print("-" * 30)
    for ext in sorted(ext_count.keys()):
        print(f"{ext}\t\t{ext_count[ext]}")

    # 检查缺失的文件类型
    print("\n文件完整性检查:")
    print("-" * 30)
    complete_records = 0
    incomplete_records = []

    for record_name in sorted_records:
        extensions = record_dict[record_name]
        if len(extensions) == 3:  # 有三种类型的文件
            complete_records += 1
        else:
            incomplete_records.append((record_name, extensions))

    print(f"完整记录(3种文件类型): {complete_records}")

    if incomplete_records:
        print(f"不完整记录: {len(incomplete_records)}")
        for record_name, extensions in incomplete_records:
            missing = set(allowed_extensions) - extensions
            missing_str = ", ".join(sorted(missing))
            print(f"  {record_name}: 缺失 {missing_str}")

    return sorted_records

if __name__ == "__main__":
    print("Database 记录统计工具")
    print("=" * 60)

    records = count_european_records()

    print(f"\n统计完成！")
    print(f"可以处理的记录列表: {records}")