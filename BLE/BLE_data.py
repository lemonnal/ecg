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


# async def main():
#     devices = await BleakScanner.discover()
#     for d in devices:
#         print(d)
#
# asyncio.run(main())

# async def main(address):
#     try:
#         async with BleakClient(address) as client:
#             print(f"已连接到设备: {address}")
#
#             # 检查服务
#             services = client.services
#             print("可用服务:")
#             for service in services:
#                 print(f"  {service.uuid}: {service.description}")
#
#             # 尝试读取型号
#             if MODEL_NBR_UUID in [char.uuid for service in services for char in service.characteristics]:
#                 model_number = await client.read_gatt_char(MODEL_NBR_UUID)
#                 print("Model Number: {0}".format("".join(map(chr, model_number))))
#             else:
#                 print(f"设备不支持型号特征: {MODEL_NBR_UUID}")
#
#     except Exception as e:
#         print(f"连接或读取失败: {e}")
#
# asyncio.run(main(address))

