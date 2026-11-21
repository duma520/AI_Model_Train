import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
import os
from datetime import datetime

class CompactDLTrainingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("紧凑型深度学习模型工具 v1.0")
        self.root.geometry("900x650")
        
        # 状态变量
        self.training = False
        self.current_model = None
        
        # 创建紧凑型布局
        self.create_compact_layout()
        
    def create_compact_layout(self):
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 模型选择标签页
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # 创建各模型标签页
        self.create_tensorrt_tab()
        self.create_onnx_tab()
        self.create_yolo_tab()
        self.create_mobilenet_tab()
        self.create_openvino_tab()
        
        # 底部状态栏和控制按钮
        self.create_bottom_controls(main_frame)
    
    def create_tensorrt_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="TensorRT")
        
        # 参数框架
        param_frame = ttk.LabelFrame(tab, text="TensorRT 参数", padding=(5, 2, 5, 5))
        param_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # 紧凑布局 - 使用网格
        ttk.Label(param_frame, text="精度模式:").grid(row=0, column=0, sticky="e", padx=2, pady=1)
        self.trt_precision = ttk.Combobox(param_frame, values=["FP32", "FP16", "INT8"], width=8)
        self.trt_precision.grid(row=0, column=1, sticky="w", padx=2, pady=1)
        self.trt_precision.set("FP16")
        
        ttk.Label(param_frame, text="批量大小:").grid(row=0, column=2, sticky="e", padx=2, pady=1)
        self.trt_batch = ttk.Spinbox(param_frame, from_=1, to=256, width=6)
        self.trt_batch.grid(row=0, column=3, sticky="w", padx=2, pady=1)
        self.trt_batch.set(16)
        
        ttk.Label(param_frame, text="最大工作空间(GB):").grid(row=1, column=0, sticky="e", padx=2, pady=1)
        self.trt_workspace = ttk.Spinbox(param_frame, from_=1, to=32, width=6)
        self.trt_workspace.grid(row=1, column=1, sticky="w", padx=2, pady=1)
        self.trt_workspace.set(2)
        
        ttk.Label(param_frame, text="优化级别:").grid(row=1, column=2, sticky="e", padx=2, pady=1)
        self.trt_opt_level = ttk.Combobox(param_frame, values=["0", "1", "2", "3"], width=8)
        self.trt_opt_level.grid(row=1, column=3, sticky="w", padx=2, pady=1)
        self.trt_opt_level.set("3")
        
        # 数据路径
        ttk.Label(param_frame, text="训练数据:").grid(row=2, column=0, sticky="e", padx=2, pady=1)
        self.trt_train_data = ttk.Entry(param_frame, width=20)
        self.trt_train_data.grid(row=2, column=1, sticky="we", padx=2, pady=1)
        ttk.Button(param_frame, text="浏览", width=4, command=lambda: self.browse_file(self.trt_train_data)).grid(row=2, column=2, padx=2, pady=1)
        
        ttk.Label(param_frame, text="验证数据:").grid(row=2, column=3, sticky="e", padx=2, pady=1)
        self.trt_val_data = ttk.Entry(param_frame, width=20)
        self.trt_val_data.grid(row=2, column=4, sticky="we", padx=2, pady=1)
        ttk.Button(param_frame, text="浏览", width=4, command=lambda: self.browse_file(self.trt_val_data)).grid(row=2, column=5, padx=2, pady=1)
        
        ttk.Label(param_frame, text="测试数据:").grid(row=3, column=0, sticky="e", padx=2, pady=1)
        self.trt_test_data = ttk.Entry(param_frame, width=20)
        self.trt_test_data.grid(row=3, column=1, sticky="we", padx=2, pady=1)
        ttk.Button(param_frame, text="浏览", width=4, command=lambda: self.browse_file(self.trt_test_data)).grid(row=3, column=2, padx=2, pady=1)
        
        ttk.Label(param_frame, text="输出路径:").grid(row=3, column=3, sticky="e", padx=2, pady=1)
        self.trt_output = ttk.Entry(param_frame, width=20)
        self.trt_output.grid(row=3, column=4, sticky="we", padx=2, pady=1)
        ttk.Button(param_frame, text="浏览", width=4, command=lambda: self.browse_dir(self.trt_output)).grid(row=3, column=5, padx=2, pady=1)
        
        # 训练参数
        ttk.Label(param_frame, text="学习率:").grid(row=4, column=0, sticky="e", padx=2, pady=1)
        self.trt_lr = ttk.Entry(param_frame, width=8)
        self.trt_lr.grid(row=4, column=1, sticky="w", padx=2, pady=1)
        self.trt_lr.insert(0, "0.001")
        
        ttk.Label(param_frame, text="迭代次数:").grid(row=4, column=2, sticky="e", padx=2, pady=1)
        self.trt_epochs = ttk.Spinbox(param_frame, from_=1, to=1000, width=6)
        self.trt_epochs.grid(row=4, column=3, sticky="w", padx=2, pady=1)
        self.trt_epochs.set(50)
        
        # 操作按钮 - 紧凑排列
        btn_frame = ttk.Frame(param_frame)
        btn_frame.grid(row=5, column=0, columnspan=6, pady=3)
        
        ttk.Button(btn_frame, text="训练", width=6, command=lambda: self.start_training("TensorRT")).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="验证", width=6, command=lambda: self.start_validation("TensorRT")).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="测试", width=6, command=lambda: self.start_testing("TensorRT")).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="导出", width=6, command=lambda: self.export_model("TensorRT")).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="加载", width=6, command=lambda: self.load_model("TensorRT")).pack(side=tk.LEFT, padx=2)
        
        # 网格配置
        for i in range(6):
            param_frame.grid_columnconfigure(i, weight=1)
    
    def create_onnx_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="ONNX")
        
        param_frame = ttk.LabelFrame(tab, text="ONNX 参数", padding=(5, 2, 5, 5))
        param_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # 紧凑布局
        ttk.Label(param_frame, text="ONNX opset:").grid(row=0, column=0, sticky="e", padx=2, pady=1)
        self.onnx_opset = ttk.Spinbox(param_frame, from_=7, to=15, width=6)
        self.onnx_opset.grid(row=0, column=1, sticky="w", padx=2, pady=1)
        self.onnx_opset.set(11)
        
        ttk.Label(param_frame, text="动态轴:").grid(row=0, column=2, sticky="e", padx=2, pady=1)
        self.onnx_dynamic = ttk.Combobox(param_frame, values=["无", "批量", "宽高", "全部"], width=8)
        self.onnx_dynamic.grid(row=0, column=3, sticky="w", padx=2, pady=1)
        self.onnx_dynamic.set("无")
        
        # 数据路径
        ttk.Label(param_frame, text="训练数据:").grid(row=1, column=0, sticky="e", padx=2, pady=1)
        self.onnx_train_data = ttk.Entry(param_frame, width=20)
        self.onnx_train_data.grid(row=1, column=1, sticky="we", padx=2, pady=1)
        ttk.Button(param_frame, text="浏览", width=4, command=lambda: self.browse_file(self.onnx_train_data)).grid(row=1, column=2, padx=2, pady=1)
        
        ttk.Label(param_frame, text="验证数据:").grid(row=1, column=3, sticky="e", padx=2, pady=1)
        self.onnx_val_data = ttk.Entry(param_frame, width=20)
        self.onnx_val_data.grid(row=1, column=4, sticky="we", padx=2, pady=1)
        ttk.Button(param_frame, text="浏览", width=4, command=lambda: self.browse_file(self.onnx_val_data)).grid(row=1, column=5, padx=2, pady=1)
        
        ttk.Label(param_frame, text="测试数据:").grid(row=2, column=0, sticky="e", padx=2, pady=1)
        self.onnx_test_data = ttk.Entry(param_frame, width=20)
        self.onnx_test_data.grid(row=2, column=1, sticky="we", padx=2, pady=1)
        ttk.Button(param_frame, text="浏览", width=4, command=lambda: self.browse_file(self.onnx_test_data)).grid(row=2, column=2, padx=2, pady=1)
        
        ttk.Label(param_frame, text="输出路径:").grid(row=2, column=3, sticky="e", padx=2, pady=1)
        self.onnx_output = ttk.Entry(param_frame, width=20)
        self.onnx_output.grid(row=2, column=4, sticky="we", padx=2, pady=1)
        ttk.Button(param_frame, text="浏览", width=4, command=lambda: self.browse_dir(self.onnx_output)).grid(row=2, column=5, padx=2, pady=1)
        
        # 训练参数
        ttk.Label(param_frame, text="学习率:").grid(row=3, column=0, sticky="e", padx=2, pady=1)
        self.onnx_lr = ttk.Entry(param_frame, width=8)
        self.onnx_lr.grid(row=3, column=1, sticky="w", padx=2, pady=1)
        self.onnx_lr.insert(0, "0.001")
        
        ttk.Label(param_frame, text="迭代次数:").grid(row=3, column=2, sticky="e", padx=2, pady=1)
        self.onnx_epochs = ttk.Spinbox(param_frame, from_=1, to=1000, width=6)
        self.onnx_epochs.grid(row=3, column=3, sticky="w", padx=2, pady=1)
        self.onnx_epochs.set(50)
        
        # 操作按钮
        btn_frame = ttk.Frame(param_frame)
        btn_frame.grid(row=4, column=0, columnspan=6, pady=3)
        
        ttk.Button(btn_frame, text="训练", width=6, command=lambda: self.start_training("ONNX")).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="验证", width=6, command=lambda: self.start_validation("ONNX")).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="测试", width=6, command=lambda: self.start_testing("ONNX")).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="导出", width=6, command=lambda: self.export_model("ONNX")).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="加载", width=6, command=lambda: self.load_model("ONNX")).pack(side=tk.LEFT, padx=2)
        
        # 网格配置
        for i in range(6):
            param_frame.grid_columnconfigure(i, weight=1)
    
    def create_yolo_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="YOLO")
        
        param_frame = ttk.LabelFrame(tab, text="YOLO 参数", padding=(5, 2, 5, 5))
        param_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # 紧凑布局
        ttk.Label(param_frame, text="YOLO版本:").grid(row=0, column=0, sticky="e", padx=2, pady=1)
        self.yolo_version = ttk.Combobox(param_frame, values=["YOLOv3", "YOLOv4", "YOLOv5", "YOLOv7", "YOLOv8"], width=8)
        self.yolo_version.grid(row=0, column=1, sticky="w", padx=2, pady=1)
        self.yolo_version.set("YOLOv5")
        
        ttk.Label(param_frame, text="模型大小:").grid(row=0, column=2, sticky="e", padx=2, pady=1)
        self.yolo_size = ttk.Combobox(param_frame, values=["nano", "small", "medium", "large", "xlarge"], width=8)
        self.yolo_size.grid(row=0, column=3, sticky="w", padx=2, pady=1)
        self.yolo_size.set("medium")
        
        # 数据路径
        ttk.Label(param_frame, text="训练数据:").grid(row=1, column=0, sticky="e", padx=2, pady=1)
        self.yolo_train_data = ttk.Entry(param_frame, width=20)
        self.yolo_train_data.grid(row=1, column=1, sticky="we", padx=2, pady=1)
        ttk.Button(param_frame, text="浏览", width=4, command=lambda: self.browse_file(self.yolo_train_data)).grid(row=1, column=2, padx=2, pady=1)
        
        ttk.Label(param_frame, text="验证数据:").grid(row=1, column=3, sticky="e", padx=2, pady=1)
        self.yolo_val_data = ttk.Entry(param_frame, width=20)
        self.yolo_val_data.grid(row=1, column=4, sticky="we", padx=2, pady=1)
        ttk.Button(param_frame, text="浏览", width=4, command=lambda: self.browse_file(self.yolo_val_data)).grid(row=1, column=5, padx=2, pady=1)
        
        ttk.Label(param_frame, text="测试数据:").grid(row=2, column=0, sticky="e", padx=2, pady=1)
        self.yolo_test_data = ttk.Entry(param_frame, width=20)
        self.yolo_test_data.grid(row=2, column=1, sticky="we", padx=2, pady=1)
        ttk.Button(param_frame, text="浏览", width=4, command=lambda: self.browse_file(self.yolo_test_data)).grid(row=2, column=2, padx=2, pady=1)
        
        ttk.Label(param_frame, text="输出路径:").grid(row=2, column=3, sticky="e", padx=2, pady=1)
        self.yolo_output = ttk.Entry(param_frame, width=20)
        self.yolo_output.grid(row=2, column=4, sticky="we", padx=2, pady=1)
        ttk.Button(param_frame, text="浏览", width=4, command=lambda: self.browse_dir(self.yolo_output)).grid(row=2, column=5, padx=2, pady=1)
        
        # 训练参数
        ttk.Label(param_frame, text="学习率:").grid(row=3, column=0, sticky="e", padx=2, pady=1)
        self.yolo_lr = ttk.Entry(param_frame, width=8)
        self.yolo_lr.grid(row=3, column=1, sticky="w", padx=2, pady=1)
        self.yolo_lr.insert(0, "0.01")
        
        ttk.Label(param_frame, text="迭代次数:").grid(row=3, column=2, sticky="e", padx=2, pady=1)
        self.yolo_epochs = ttk.Spinbox(param_frame, from_=1, to=1000, width=6)
        self.yolo_epochs.grid(row=3, column=3, sticky="w", padx=2, pady=1)
        self.yolo_epochs.set(100)
        
        # 操作按钮
        btn_frame = ttk.Frame(param_frame)
        btn_frame.grid(row=4, column=0, columnspan=6, pady=3)
        
        ttk.Button(btn_frame, text="训练", width=6, command=lambda: self.start_training("YOLO")).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="验证", width=6, command=lambda: self.start_validation("YOLO")).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="测试", width=6, command=lambda: self.start_testing("YOLO")).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="导出", width=6, command=lambda: self.export_model("YOLO")).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="加载", width=6, command=lambda: self.load_model("YOLO")).pack(side=tk.LEFT, padx=2)
        
        # 网格配置
        for i in range(6):
            param_frame.grid_columnconfigure(i, weight=1)
    
    def create_mobilenet_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="MobileNet")
        
        param_frame = ttk.LabelFrame(tab, text="MobileNet 参数", padding=(5, 2, 5, 5))
        param_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # 紧凑布局
        ttk.Label(param_frame, text="版本:").grid(row=0, column=0, sticky="e", padx=2, pady=1)
        self.mn_version = ttk.Combobox(param_frame, values=["v1", "v2", "v3"], width=8)
        self.mn_version.grid(row=0, column=1, sticky="w", padx=2, pady=1)
        self.mn_version.set("v2")
        
        ttk.Label(param_frame, text="宽度倍数:").grid(row=0, column=2, sticky="e", padx=2, pady=1)
        self.mn_width = ttk.Spinbox(param_frame, from_=0.1, to=2.0, increment=0.1, width=6)
        self.mn_width.grid(row=0, column=3, sticky="w", padx=2, pady=1)
        self.mn_width.set(1.0)
        
        # 数据路径
        ttk.Label(param_frame, text="训练数据:").grid(row=1, column=0, sticky="e", padx=2, pady=1)
        self.mn_train_data = ttk.Entry(param_frame, width=20)
        self.mn_train_data.grid(row=1, column=1, sticky="we", padx=2, pady=1)
        ttk.Button(param_frame, text="浏览", width=4, command=lambda: self.browse_file(self.mn_train_data)).grid(row=1, column=2, padx=2, pady=1)
        
        ttk.Label(param_frame, text="验证数据:").grid(row=1, column=3, sticky="e", padx=2, pady=1)
        self.mn_val_data = ttk.Entry(param_frame, width=20)
        self.mn_val_data.grid(row=1, column=4, sticky="we", padx=2, pady=1)
        ttk.Button(param_frame, text="浏览", width=4, command=lambda: self.browse_file(self.mn_val_data)).grid(row=1, column=5, padx=2, pady=1)
        
        ttk.Label(param_frame, text="测试数据:").grid(row=2, column=0, sticky="e", padx=2, pady=1)
        self.mn_test_data = ttk.Entry(param_frame, width=20)
        self.mn_test_data.grid(row=2, column=1, sticky="we", padx=2, pady=1)
        ttk.Button(param_frame, text="浏览", width=4, command=lambda: self.browse_file(self.mn_test_data)).grid(row=2, column=2, padx=2, pady=1)
        
        ttk.Label(param_frame, text="输出路径:").grid(row=2, column=3, sticky="e", padx=2, pady=1)
        self.mn_output = ttk.Entry(param_frame, width=20)
        self.mn_output.grid(row=2, column=4, sticky="we", padx=2, pady=1)
        ttk.Button(param_frame, text="浏览", width=4, command=lambda: self.browse_dir(self.mn_output)).grid(row=2, column=5, padx=2, pady=1)
        
        # 训练参数
        ttk.Label(param_frame, text="学习率:").grid(row=3, column=0, sticky="e", padx=2, pady=1)
        self.mn_lr = ttk.Entry(param_frame, width=8)
        self.mn_lr.grid(row=3, column=1, sticky="w", padx=2, pady=1)
        self.mn_lr.insert(0, "0.001")
        
        ttk.Label(param_frame, text="迭代次数:").grid(row=3, column=2, sticky="e", padx=2, pady=1)
        self.mn_epochs = ttk.Spinbox(param_frame, from_=1, to=1000, width=6)
        self.mn_epochs.grid(row=3, column=3, sticky="w", padx=2, pady=1)
        self.mn_epochs.set(50)
        
        # 操作按钮
        btn_frame = ttk.Frame(param_frame)
        btn_frame.grid(row=4, column=0, columnspan=6, pady=3)
        
        ttk.Button(btn_frame, text="训练", width=6, command=lambda: self.start_training("MobileNet")).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="验证", width=6, command=lambda: self.start_validation("MobileNet")).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="测试", width=6, command=lambda: self.start_testing("MobileNet")).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="导出", width=6, command=lambda: self.export_model("MobileNet")).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="加载", width=6, command=lambda: self.load_model("MobileNet")).pack(side=tk.LEFT, padx=2)
        
        # 网格配置
        for i in range(6):
            param_frame.grid_columnconfigure(i, weight=1)
    
    def create_openvino_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="OpenVINO")
        
        param_frame = ttk.LabelFrame(tab, text="OpenVINO 参数", padding=(5, 2, 5, 5))
        param_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # 紧凑布局
        ttk.Label(param_frame, text="设备:").grid(row=0, column=0, sticky="e", padx=2, pady=1)
        self.ov_device = ttk.Combobox(param_frame, values=["CPU", "GPU", "MYRIAD", "HDDL"], width=8)
        self.ov_device.grid(row=0, column=1, sticky="w", padx=2, pady=1)
        self.ov_device.set("CPU")
        
        ttk.Label(param_frame, text="精度:").grid(row=0, column=2, sticky="e", padx=2, pady=1)
        self.ov_precision = ttk.Combobox(param_frame, values=["FP32", "FP16", "INT8"], width=8)
        self.ov_precision.grid(row=0, column=3, sticky="w", padx=2, pady=1)
        self.ov_precision.set("FP32")
        
        # 数据路径
        ttk.Label(param_frame, text="训练数据:").grid(row=1, column=0, sticky="e", padx=2, pady=1)
        self.ov_train_data = ttk.Entry(param_frame, width=20)
        self.ov_train_data.grid(row=1, column=1, sticky="we", padx=2, pady=1)
        ttk.Button(param_frame, text="浏览", width=4, command=lambda: self.browse_file(self.ov_train_data)).grid(row=1, column=2, padx=2, pady=1)
        
        ttk.Label(param_frame, text="验证数据:").grid(row=1, column=3, sticky="e", padx=2, pady=1)
        self.ov_val_data = ttk.Entry(param_frame, width=20)
        self.ov_val_data.grid(row=1, column=4, sticky="we", padx=2, pady=1)
        ttk.Button(param_frame, text="浏览", width=4, command=lambda: self.browse_file(self.ov_val_data)).grid(row=1, column=5, padx=2, pady=1)
        
        ttk.Label(param_frame, text="测试数据:").grid(row=2, column=0, sticky="e", padx=2, pady=1)
        self.ov_test_data = ttk.Entry(param_frame, width=20)
        self.ov_test_data.grid(row=2, column=1, sticky="we", padx=2, pady=1)
        ttk.Button(param_frame, text="浏览", width=4, command=lambda: self.browse_file(self.ov_test_data)).grid(row=2, column=2, padx=2, pady=1)
        
        ttk.Label(param_frame, text="输出路径:").grid(row=2, column=3, sticky="e", padx=2, pady=1)
        self.ov_output = ttk.Entry(param_frame, width=20)
        self.ov_output.grid(row=2, column=4, sticky="we", padx=2, pady=1)
        ttk.Button(param_frame, text="浏览", width=4, command=lambda: self.browse_dir(self.ov_output)).grid(row=2, column=5, padx=2, pady=1)
        
        # 训练参数
        ttk.Label(param_frame, text="学习率:").grid(row=3, column=0, sticky="e", padx=2, pady=1)
        self.ov_lr = ttk.Entry(param_frame, width=8)
        self.ov_lr.grid(row=3, column=1, sticky="w", padx=2, pady=1)
        self.ov_lr.insert(0, "0.001")
        
        ttk.Label(param_frame, text="迭代次数:").grid(row=3, column=2, sticky="e", padx=2, pady=1)
        self.ov_epochs = ttk.Spinbox(param_frame, from_=1, to=1000, width=6)
        self.ov_epochs.grid(row=3, column=3, sticky="w", padx=2, pady=1)
        self.ov_epochs.set(50)
        
        # 操作按钮
        btn_frame = ttk.Frame(param_frame)
        btn_frame.grid(row=4, column=0, columnspan=6, pady=3)
        
        ttk.Button(btn_frame, text="训练", width=6, command=lambda: self.start_training("OpenVINO")).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="验证", width=6, command=lambda: self.start_validation("OpenVINO")).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="测试", width=6, command=lambda: self.start_testing("OpenVINO")).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="导出", width=6, command=lambda: self.export_model("OpenVINO")).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="加载", width=6, command=lambda: self.load_model("OpenVINO")).pack(side=tk.LEFT, padx=2)
        
        # 网格配置
        for i in range(6):
            param_frame.grid_columnconfigure(i, weight=1)
    
    def create_bottom_controls(self, parent):
        # 底部控制面板
        bottom_frame = ttk.Frame(parent)
        bottom_frame.pack(fill=tk.X, padx=5, pady=(0,5))
        
        # 进度条
        self.progress = ttk.Progressbar(bottom_frame, orient=tk.HORIZONTAL, mode='determinate')
        self.progress.pack(fill=tk.X, expand=True, side=tk.LEFT, padx=(0,5))
        
        # 进度标签
        self.progress_label = ttk.Label(bottom_frame, text="就绪", width=10)
        self.progress_label.pack(side=tk.LEFT, padx=(0,5))
        
        # 停止按钮
        self.stop_btn = ttk.Button(bottom_frame, text="停止", command=self.stop_operation, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT)
        
        # 日志框架
        log_frame = ttk.LabelFrame(parent, text="日志", padding=(5, 2, 5, 5))
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0,5))
        
        # 日志文本框
        self.log_text = tk.Text(log_frame, height=8, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # 滚动条
        scrollbar = ttk.Scrollbar(self.log_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.log_text.yview)
        
        # 日志级别选择
        log_level_frame = ttk.Frame(log_frame)
        log_level_frame.pack(fill=tk.X, pady=(2,0))
        
        ttk.Label(log_level_frame, text="日志级别:").pack(side=tk.LEFT, padx=(0,5))
        self.log_level = ttk.Combobox(log_level_frame, values=["DEBUG", "INFO", "WARNING", "ERROR"], width=8)
        self.log_level.pack(side=tk.LEFT)
        self.log_level.set("INFO")
        
        # 清空日志按钮
        ttk.Button(log_level_frame, text="清空日志", command=self.clear_log).pack(side=tk.RIGHT)
    
    def browse_file(self, entry_widget):
        filename = filedialog.askopenfilename()
        if filename:
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, filename)
    
    def browse_dir(self, entry_widget):
        dirname = filedialog.askdirectory()
        if dirname:
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, dirname)
    
    def start_training(self, model_name):
        if self.training:
            messagebox.showwarning("警告", "已有任务正在运行，请等待完成或停止当前任务")
            return
        
        self.current_model = model_name
        self.training = True
        self.stop_btn.config(state=tk.NORMAL)
        self.log_message(f"开始 {model_name} 模型训练...")
        
        # 获取训练参数
        if model_name == "TensorRT":
            epochs = int(self.trt_epochs.get())
        elif model_name == "ONNX":
            epochs = int(self.onnx_epochs.get())
        elif model_name == "YOLO":
            epochs = int(self.yolo_epochs.get())
        elif model_name == "MobileNet":
            epochs = int(self.mn_epochs.get())
        elif model_name == "OpenVINO":
            epochs = int(self.ov_epochs.get())
        
        # 模拟训练过程
        self.progress["maximum"] = epochs
        self.progress["value"] = 0
        
        # 在新线程中运行训练
        training_thread = threading.Thread(target=self.simulate_training, args=(epochs,))
        training_thread.start()
    
    def start_validation(self, model_name):
        if self.training:
            messagebox.showwarning("警告", "已有任务正在运行，请等待完成或停止当前任务")
            return
        
        self.current_model = model_name
        self.training = True
        self.stop_btn.config(state=tk.NORMAL)
        self.log_message(f"开始 {model_name} 模型验证...")
        
        # 模拟验证过程
        self.progress["maximum"] = 100
        self.progress["value"] = 0
        
        # 在新线程中运行验证
        validation_thread = threading.Thread(target=self.simulate_validation)
        validation_thread.start()
    
    def start_testing(self, model_name):
        if self.training:
            messagebox.showwarning("警告", "已有任务正在运行，请等待完成或停止当前任务")
            return
        
        self.current_model = model_name
        self.training = True
        self.stop_btn.config(state=tk.NORMAL)
        self.log_message(f"开始 {model_name} 模型测试...")
        
        # 模拟测试过程
        self.progress["maximum"] = 100
        self.progress["value"] = 0
        
        # 在新线程中运行测试
        testing_thread = threading.Thread(target=self.simulate_testing)
        testing_thread.start()
    
    def export_model(self, model_name):
        if self.training:
            messagebox.showwarning("警告", "已有任务正在运行，请等待完成或停止当前任务")
            return
        
        self.current_model = model_name
        self.training = True
        self.stop_btn.config(state=tk.NORMAL)
        self.log_message(f"开始导出 {model_name} 模型...")
        
        # 模拟导出过程
        self.progress["maximum"] = 100
        self.progress["value"] = 0
        
        # 在新线程中运行导出
        export_thread = threading.Thread(target=self.simulate_export)
        export_thread.start()
    
    def load_model(self, model_name):
        if self.training:
            messagebox.showwarning("警告", "已有任务正在运行，请等待完成或停止当前任务")
            return
        
        filename = filedialog.askopenfilename(title=f"加载 {model_name} 模型")
        if filename:
            self.log_message(f"已加载 {model_name} 模型: {filename}")
            messagebox.showinfo("成功", f"{model_name} 模型加载成功")
    
    def simulate_training(self, epochs):
        try:
            for epoch in range(1, epochs+1):
                if not self.training:
                    break
                
                # 模拟训练步骤
                time.sleep(0.1)
                
                # 更新进度
                self.progress["value"] = epoch
                self.progress_label.config(text=f"{epoch}/{epochs}")
                
                # 模拟日志输出
                if epoch % 5 == 0:
                    self.log_message(f"Epoch {epoch}/{epochs} - 损失: {0.1 + 0.9 * (1 - epoch/epochs):.4f} - 准确率: {0.1 + 0.8 * (epoch/epochs):.4f}")
            
            if self.training:
                self.log_message(f"{self.current_model} 训练完成!")
                
                # 保存训练结果
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_filename = f"{self.current_model}_model_{timestamp}.pt"
                self.log_message(f"模型已保存为: {model_filename}")
                
                messagebox.showinfo("完成", f"{self.current_model} 训练完成!")
        except Exception as e:
            self.log_message(f"训练出错: {str(e)}", level="ERROR")
        finally:
            self.training = False
            self.stop_btn.config(state=tk.DISABLED)
    
    def simulate_validation(self):
        try:
            for i in range(1, 101):
                if not self.training:
                    break
                
                time.sleep(0.05)
                self.progress["value"] = i
                self.progress_label.config(text=f"{i}%")
                
                if i % 20 == 0:
                    self.log_message(f"验证进度: {i}% - 准确率: {0.7 + 0.2 * (i/100):.4f}")
            
            if self.training:
                self.log_message(f"{self.current_model} 验证完成!")
                messagebox.showinfo("完成", f"{self.current_model} 验证完成!")
        except Exception as e:
            self.log_message(f"验证出错: {str(e)}", level="ERROR")
        finally:
            self.training = False
            self.stop_btn.config(state=tk.DISABLED)
    
    def simulate_testing(self):
        try:
            for i in range(1, 101):
                if not self.training:
                    break
                
                time.sleep(0.03)
                self.progress["value"] = i
                self.progress_label.config(text=f"{i}%")
                
                if i % 25 == 0:
                    self.log_message(f"测试进度: {i}% - mAP: {0.5 + 0.4 * (i/100):.4f}")
            
            if self.training:
                self.log_message(f"{self.current_model} 测试完成!")
                messagebox.showinfo("完成", f"{self.current_model} 测试完成!")
        except Exception as e:
            self.log_message(f"测试出错: {str(e)}", level="ERROR")
        finally:
            self.training = False
            self.stop_btn.config(state=tk.DISABLED)
    
    def simulate_export(self):
        try:
            for i in range(1, 101):
                if not self.training:
                    break
                
                time.sleep(0.02)
                self.progress["value"] = i
                self.progress_label.config(text=f"{i}%")
                
                if i % 33 == 0:
                    self.log_message(f"导出进度: {i}% - 优化中...")
            
            if self.training:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_filename = f"{self.current_model}_exported_{timestamp}.onnx"
                self.log_message(f"{self.current_model} 导出完成! 保存为: {model_filename}")
                messagebox.showinfo("完成", f"{self.current_model} 导出完成!")
        except Exception as e:
            self.log_message(f"导出出错: {str(e)}", level="ERROR")
        finally:
            self.training = False
            self.stop_btn.config(state=tk.DISABLED)
    
    def stop_operation(self):
        self.training = False
        self.log_message(f"{self.current_model} 操作已停止")
        self.stop_btn.config(state=tk.DISABLED)
    
    def log_message(self, message, level=None):
        if level is None:
            level = self.log_level.get()
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}\n"
        
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        
        # 根据日志级别设置颜色
        if level == "ERROR":
            self.log_text.tag_add("error", "end-2l linestart", "end-2l lineend")
            self.log_text.tag_config("error", foreground="red")
        elif level == "WARNING":
            self.log_text.tag_add("warning", "end-2l linestart", "end-2l lineend")
            self.log_text.tag_config("warning", foreground="orange")
        elif level == "DEBUG":
            self.log_text.tag_add("debug", "end-2l linestart", "end-2l lineend")
            self.log_text.tag_config("debug", foreground="gray")
    
    def clear_log(self):
        self.log_text.delete(1.0, tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = CompactDLTrainingGUI(root)
    root.mainloop()