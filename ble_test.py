import asyncio
from bleak import BleakScanner
from bleak import BleakClient
from bleak.backends.device import BLEDevice
from bleak.backends.scanner import AdvertisementData
import struct
import matplotlib.pyplot as plt
from collections import deque
import struct

# E2:1B:A5:DB:DE:EA: PW-ECG-SL
# address = "E2:1B:A5:DB:DE:EA"

UART_SERVICE_UUID = "6e400001-b5a3-f393-e0a9-68716563686f"
UART_RX_CHAR_UUID = "6e400002-b5a3-f393-e0a9-68716563686f"
UART_TX_CHAR_UUID = "6e400003-b5a3-f393-e0a9-68716563686f"

# ============================================
# 测试功能标志位设置 (将对应的标志位设为1即可启用该测试)
# ============================================
TEST_SCAN_DEVICES = 0          # 测试1: 简单扫描蓝牙设备
TEST_CONNECT_AND_VIEW = 0      # 测试2: 连接设备并查看服务
TEST_DATA_PARSE = 0            # 测试3: 数据解析测试
TEST_ECG_COLLECTION = 0        # 测试4: ECG数据采集与实时绘图(简单版)
TEST_QINGXUN_COLLECTOR = 0     # 测试5: QingXunBlueToothCollector类完整测试


# ============================================
# 测试1: 简单扫描蓝牙设备
# ============================================
if TEST_SCAN_DEVICES:

    async def main():
        devices = await BleakScanner.discover()
        for d in devices:
            print(d)

    asyncio.run(main())


# ============================================
# 测试2: 连接设备并查看服务
# ============================================
if TEST_CONNECT_AND_VIEW:

    async def main(address):
        try:
            async with BleakClient(address) as client:
                print(f"已连接到设备: {address}")

                # 检查服务
                services = client.services
                print("可用服务:")
                for service in services:
                    print(f"  {service.uuid}: {service.description}")

                # 尝试读取型号
                # if MODEL_NBR_UUID in [char.uuid for service in services for char in service.characteristics]:
                #     model_number = await client.read_gatt_char(MODEL_NBR_UUID)
                #     print("Model Number: {0}".format("".join(map(chr, model_number))))
                # else:
                #     print(f"设备不支持型号特征: {MODEL_NBR_UUID}")

        except Exception as e:
            print(f"连接或读取失败: {e}")

    # asyncio.run(main("E2:1B:A5:DB:DE:EA"))  # 需要指定设备地址


# ============================================
# 测试3: 数据解析测试
# ============================================
if TEST_DATA_PARSE:
    # 包含全部解包结构，不一定使用
    # 解析协议包结构: 功能码(2) + 数据长度(2) + 数据内容(238) + CRC16(2) = 244字节
    # 1. 提取功能码 (2)
    feature_code = data[0:2]

    # 2. 提取数据长度 (2)
    data_len = struct.unpack('<H', data[2:4])[0]

    # 3. 提取数据内容
    samples = []
    sample_data_start = 4  # 采样数据起始位置
    for i in range(119):
        sample_offset = sample_data_start + i * 2
        if sample_offset + 2 > len(data) - 2:  # 减去校验和
            break

        # 读取小端格式的16位整数
        sample_value = struct.unpack('<H', data[sample_offset:sample_offset + 2])[0]

        # 转换为电压值 (μV) - 单导联 0.288 12导联 0.318
        voltage_uV = sample_value * 0.288
        samples.append({
            'raw_adc': sample_value,
            'voltage_uV': voltage_uV,
            'voltage_mV': voltage_uV / 1000.0
        })
    # CRC校验和 (最后2字节)
    crc_checksum = struct.unpack('<H', data[-2:])[0]

    parsed_data = {
        'feature_code': feature_code.hex(),
        'data_len': data_len,
        'samples': samples,
        'sample_count': len(samples),
        'crc_checksum': f'{crc_checksum:04x}',
    }
    for key, value in parsed_data.items():
        print(f"{key}: {value}")
    hex_str = ' '.join(f'{b:02x}' for b in data)
    print(f"十六进制: {hex_str} (共{len(data)}字节)")

    # 假设 samples 是一个字典列表
    voltage_mV_max = max([sample.get('voltage_mV') for sample in samples if 'voltage_mV' in sample])
    voltage_mV_min = min([sample.get('voltage_mV') for sample in samples if 'voltage_mV' in sample])

    try:
        for sample in samples:
            value = sample['voltage_mV']
            data_list.append(value)
            line.set_ydata(data_list)
            line.set_xdata(range(len(data_list)))
            ax.set_ylim(voltage_mV_min - 0.1 * abs(voltage_mV_min), voltage_mV_max + 0.1 * abs(voltage_mV_max))
            ax.relim()
            ax.autoscale_view()
            # plt.pause(0.01)  # 更新图形，可以根据需要调整刷新频率
    except ValueError:
        pass


# ============================================
# 测试4: ECG数据采集与实时绘图(简单版)
# ============================================
if TEST_ECG_COLLECTION:
    # 创建空的数据列表
    data_list = deque(maxlen=100)  # 保持最新的100个数据点

    # 创建一个图形窗口
    plt.ion()  # 开启交互模式
    fig, ax = plt.subplots()
    line, = ax.plot([])
    ax.set_ylim(0.0, 0.05)  # 设置y轴范围，根据你的数据范围进行调整


    # 断开连接回调函数
    def handle_disconnect(client):
        print("设备已断开连接")

    # 接收数据回调函数
    def handle_rx(sender, data):
        hex_str = ' '.join(f'{b:02x}' for b in data)  # 只显示前32字节
        print(data)
        if 1:
            # 解析固定结构
            # 协议包结构: 功能码(2) + 数据长度(2) + 数据内容(238) + CRC16(2) = 244字节
            feature_code = data[0:2]  # 功能码
            data_len = struct.unpack('<H', data[2:4])[0]

            samples = []
            sample_data_start = 4  # 采样数据起始位置

            for i in range(119):
                sample_offset = sample_data_start + i * 2
                if sample_offset + 2 > len(data) - 2:  # 减去校验和
                    break

                # 读取小端格式的16位整数
                sample_value = struct.unpack('<H', data[sample_offset:sample_offset + 2])[0]

                # 转换为电压值 (μV)，系数0.288需要根据设备校准
                voltage_uV = sample_value * 0.288
                samples.append({
                    'raw_adc': sample_value,
                    'voltage_uV': voltage_uV,
                    'voltage_mV': voltage_uV / 1000.0
                })
            # CRC校验和 (最后2字节)
            crc_checksum = struct.unpack('<H', data[-2:])[0]

            hex_str += f" (共{len(data)}字节)"

            parsed_data = {
                'feature_code': feature_code.hex(),
                'data_len': data_len,
                'samples': samples,
                'sample_count': len(samples),
                'crc_checksum': f'{crc_checksum:04x}',
            }

        print(f"  十六进制: {hex_str}")
        for key, value in parsed_data.items():
            print(f"{key}: {value}")
        print("=" * 60)
        print(parsed_data["samples"])

        # 假设 samples 是一个字典列表
        voltage_mV_max = max([sample.get('voltage_mV') for sample in samples if 'voltage_mV' in sample])
        voltage_mV_min = min([sample.get('voltage_mV') for sample in samples if 'voltage_mV' in sample])

        try:
            for sample in samples:
                value = sample['voltage_mV']
                data_list.append(value)
                line.set_ydata(data_list)
                line.set_xdata(range(len(data_list)))
                ax.set_ylim(voltage_mV_min - 0.1 * abs(voltage_mV_min), voltage_mV_max + 0.1 * abs(voltage_mV_max))
                ax.relim()
                ax.autoscale_view()
                plt.pause(0.01)  # 更新图形，可以根据需要调整刷新频率
        except ValueError:
            pass

        print(data_list)

    def build_protocol_packet(func_code, data):
        """
        构建协议包
        """
        # 这里需要根据实际协议格式实现
        # 通常包含包头、功能码、数据长度、数据、校验和等
        packet = bytearray()

        # # 包头 (根据实际协议调整)
        # packet.extend([0xAA, 0x55])  # 示例包头
        #
        # # 功能码 (2字节小端)
        # packet.extend(struct.pack('<H', func_code))
        #
        # # 数据长度 (2字节小端)
        # packet.extend(struct.pack('<H', len(data)))

        # 数据
        packet.extend(data)

        # # 计算校验和 (根据实际协议调整)
        # checksum = sum(packet) & 0xFF
        # packet.append(checksum)

        return packet

    async def start_collection(client, timestamp=0):
        """
        开启采集
        功能码: 0x0001
        数据格式: [采集开关(1字节)] [时间戳(8字节)]
        """
        try:
            data = bytearray(9)
            index = 0

            # 采集开关: 0x01 (开启)
            data[index] = 0x01
            index += 1

            # 时间戳: 8字节 (小端格式)
            for i in range(8):
                data[index] = (timestamp >> (i * 8)) & 0xFF
                index += 1

            # 构建协议包 (功能码 0x0001)
            packet = build_protocol_packet(0x0001, data)

            print(f"发送开启采集指令: {[f'0x{b:02X}' for b in packet]}")
            await client.write_gatt_char(UART_RX_CHAR_UUID, packet)
            print("开始采集指令发送成功")

        except Exception as e:
            print(f"构建开启采集指令失败: {e}")

    def match_nus_uuid(device: BLEDevice, adv: AdvertisementData):
        if adv and adv.service_uuids and UART_SERVICE_UUID.lower() in [uuid.lower() for uuid in adv.service_uuids]:
            return True
        return False

    async def main():
        # 搜索设备, 查看是否匹配NUS UUID，找到后可尝试建立连接，进行读写。

        device = await BleakScanner.find_device_by_filter(match_nus_uuid)

        if not device:
            print("未找到支持NUS的设备")
            return

        print(f"找到设备: {device.address}")

        # 创建BleakClient客户端，连接后进行串口操作
        async with BleakClient(device, disconnected_callback=handle_disconnect) as client:
            await client.start_notify(UART_TX_CHAR_UUID, handle_rx)

            print("Connected, start typing and press ENTER...")

            loop = asyncio.get_running_loop()
            nus = client.services.get_service(UART_SERVICE_UUID)
            # 接收蓝牙串口信息
            rx_char = nus.get_characteristic(UART_RX_CHAR_UUID)

            # # 发送开始采集指令
            # await start_collection(client, 0)

            # 持续接收数据的循环
            try:
                print("开始持续接收数据")
                while True:
                    # 保持连接并等待数据
                    await asyncio.sleep(0.1)  # 防止CPU占用过高，同时维持连接

                    exit()

            except KeyboardInterrupt:
                print("\n收到中断信号，正在断开连接...")
            except Exception as e:
                print(f"发生错误: {e}")
            finally:
                print("连接已断开")

    if __name__ == "__main__":
        asyncio.run(main())


# ============================================
# 测试5: QingXunBlueToothCollector类完整测试
# ============================================
if TEST_QINGXUN_COLLECTOR:

    QINGXUN_UART_SERVICE_UUID = "6e400001-b5a3-f393-e0a9-68716563686f"
    QINGXUN_UART_RX_CHAR_UUID = "6e400002-b5a3-f393-e0a9-68716563686f"
    QINGXUN_UART_TX_CHAR_UUID = "6e400003-b5a3-f393-e0a9-68716563686f"

    device = "AAA-TEST"
    if device == "AAA-TEST":
        device_param = {
            "name": device,
            "address": "EC:7A:26:9D:81:3F",
            "service_uuid": QINGXUN_UART_SERVICE_UUID,
            "rx_uuid": QINGXUN_UART_RX_CHAR_UUID,
            "tx_uuid": QINGXUN_UART_TX_CHAR_UUID,
        }
    elif device == "PW-ECG-SL":
        device_param = {
            "name": device,
            "address": "E2:1B:A5:DB:DE:EA",
            "service_uuid": QINGXUN_UART_SERVICE_UUID,
            "rx_uuid": QINGXUN_UART_RX_CHAR_UUID,
            "tx_uuid": QINGXUN_UART_TX_CHAR_UUID,
        }


    voltage_mV_max = -0xffffff
    voltage_mV_min = 0xffffff

    # 创建空的数据列表
    data_list = deque(maxlen=300)  # 保持最新的100个数据点
    # 创建一个图形窗口
    plt.ion()  # 开启交互模式
    fig, ax = plt.subplots()
    line, = ax.plot([])
    ax.set_ylim(0.0, 0.05)  # 设置y轴范围，根据你的数据范围进行调整


    class QingXunBlueToothCollector:
        def __init__(self, client=None):
            self.data_queue = deque(maxlen=300)
            self.latest_samples = []
            self.data = []
            self.qrs_detector = None  # QRS检测器实例

        def set_qrs_detector(self, qrs_detector):
            """
            设置QRS检测器实例
            参数:
                qrs_detector: PanTomkinsQRSDetectorOnline实例
            """
            self.qrs_detector = qrs_detector

        # 断开连接回调函数
        def handle_disconnect(self, client):
            print(f"设备已断开连接")

        def match_nus_device(self, device: BLEDevice, adv: AdvertisementData):
            # 优先通过MAC地址匹配
            if device.address == "EC:7A:26:9D:81:3F":
                print(f"通过MAC地址匹配到设备: {device.name or '未知'} ({device.address})")
                return True
            # 优先通过设备名称匹配
            if device.name and "AAA-TEST" in device.name:
                print(f"通过名称匹配到设备: {device.name} ({device.address})")
                return True
            # 如果名称和MAC地址都匹配失败，尝试UUID匹配
            if adv and adv.service_uuids and device_param["service_uuid"].lower() in [uuid.lower() for uuid in adv.service_uuids]:
                print(f"通过UUID匹配到设备: {device.name or '未知'} ({device.address})")
                print(f"  服务UUIDs: {adv.service_uuids}")
                return True
            return False

        def build_protocol_packet(self, func_code, data):
            """
            构建轻迅协议V1.0.1数据包
            格式: [功能码(2字节)] [数据长度(2字节)] [数据内容] [CRC16(2字节)]

            Args:
                func_code: 功能码 (int)
                data: 数据内容 (bytes/bytearray)

            Returns:
                完整的协议包 (bytearray)
            """
            packet = bytearray()
            # 1. 功能码 (2字节小端格式)
            packet.extend(struct.pack('<H', func_code))
            # 2. 数据长度 (2字节小端格式)
            packet.extend(struct.pack('<H', len(data)))
            # 3. 数据内容
            packet.extend(data)

            def calculate_crc16(data, offset=0, length=None):
                """
                计算CRC16-CCITT-FALSE校验值
                多项式: 0x1021
                初始值: 0xFFFF
                结果异或值: 0x0000
                输入输出反转: 无
                """

                wCRCin = 0xFFFF
                wCPoly = 0x1021

                for i in range(offset, offset + length):
                    byte = data[i]
                    for j in range(8):
                        bit = ((byte >> (7 - j)) & 1) == 1
                        c15 = ((wCRCin >> 15) & 1) == 1
                        wCRCin = wCRCin << 1
                        if c15 ^ bit:
                            wCRCin = wCRCin ^ wCPoly

                return wCRCin & 0xFFFF

            # 4. CRC检验
            crc_value = calculate_crc16(packet, 0, len(packet))
            packet.extend(struct.pack('<H', crc_value))

            return packet

        async def start_collection(self, client, collect_enable=1, timestamp=0):
            """
            开启采集
            功能码: 0x0001
            数据格式: [功能码(2字节)] [数据长度(2字节)] [数据内容 [采集开关(1字节)] [时间戳(8字节)] (9字节)] [CRC16(2字节)]
            """
            data = bytearray()
            # 采集开关(1字节)
            data.extend(struct.pack('B', collect_enable))
            # 时间戳(8字节)
            data.extend(struct.pack('<Q', timestamp))
            packet = self.build_protocol_packet(0x0001, data)
            print(f"发送开启采集指令: {[f'0x{b:02X}' for b in packet]}")
            respones = await client.write_gatt_char(device_param["rx_uuid"], packet)
            print("开始采集指令发送成功")
            return respones

        # 接收数据回调函数
        def handle_rx(self, sender, data):
            global voltage_mV_max
            global voltage_mV_min

            # 只提取数据内容
            samples = []
            sample_data_start = 4  # 采样数据起始位置
            for i in range(119):
                sample_offset = sample_data_start + i * 2
                if sample_offset + 2 > len(data) - 2:  # 减去校验和
                    break

                # 读取小端格式的16位整数
                sample_value = struct.unpack('<H', data[sample_offset:sample_offset + 2])[0]

                # 转换为电压值 (μV) - 单导联 0.288 12导联 0.318
                voltage_mV = sample_value * 0.288 / 1000.0
                samples.append(voltage_mV)

            # 将数据传递给QRS检测器
            if self.qrs_detector and len(samples) > 0:
                self.qrs_detector.update_signal(samples)

            try:
                voltage_mV_max = max(samples) if voltage_mV_max < max(samples) else voltage_mV_max
                voltage_mV_min = min(samples) if voltage_mV_min > min(samples) else voltage_mV_min
                # print(voltage_mV_max, voltage_mV_min)
                voltage_delta = voltage_mV_max - voltage_mV_min

                for sample in samples:
                    value = sample
                    data_list.append(value)
                    line.set_ydata(data_list)
                    line.set_xdata(range(len(data_list)))
                    ax.set_ylim(voltage_mV_min - 0.2 * voltage_delta, voltage_mV_max + 0.2 * voltage_delta)
                    ax.relim()
                    ax.autoscale_view()
                    plt.pause(0.01)  # 更新图形，可以根据需要调整刷新频率
            except ValueError:
                pass





    async def main():
        # 首先扫描并输出所有附近的蓝牙设备
        print("正在扫描所有附近的蓝牙设备...")
        all_devices = await BleakScanner.discover(timeout=10.0)
        print(f"\n找到 {len(all_devices)} 个蓝牙设备:\n")

        for d in all_devices:
            print(f"设备名称: {d.name or '未知'}")
            print(f"MAC地址: {d.address}")
            print("-" * 60)

        print("\n" + "=" * 60)
        print("开始搜索目标设备...")

        # 搜索设备, 查看是否匹配NUS UUID，找到后可尝试建立连接，进行读写。
        Collector = QingXunBlueToothCollector()
        device = await BleakScanner.find_device_by_filter(Collector.match_nus_device)
        if not device:
            print("未找到目标设备")
            print("\n提示: 请检查:")
            print("  1. 设备是否已开启")
            print("  2. 设备名称是否为 'AAA-TEST'")
            print("  3. MAC地址是否为 'EC:7A:26:9D:81:3F'")
            return
        else:
            print(f"\n成功找到设备: {device.address}")

        # 创建BleakClient客户端，连接后进行串口操作
        async with BleakClient(device, disconnected_callback=Collector.handle_disconnect) as client:
            # 发送开始监听指令
            await client.start_notify(device_param["tx_uuid"], Collector.handle_rx)
            print("Enable listening Callback Function")
            # 发送开始采集指令
            await Collector.start_collection(client, collect_enable=1, timestamp=0)
            print("Enable Collector Callback Function")
            # 持续接收数据的循环
            try:
                print("开始持续接收数据")
                while True:
                    # 保持连接并等待数据
                    await asyncio.sleep(0.01)  # 防止CPU占用过高，同时维持连接

            except KeyboardInterrupt:
                print("\n收到中断信号，正在断开连接...")
            except Exception as e:
                print(f"发生错误: {e}")
            finally:
                print("连接已断开")


    if __name__ == "__main__":
        asyncio.run(main())
