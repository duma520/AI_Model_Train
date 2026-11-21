import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, 
                            QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, 
                            QLineEdit, QPushButton, QCheckBox, QComboBox, 
                            QSpinBox, QDoubleSpinBox, QScrollArea)
from PyQt5.QtCore import Qt

class CompactModelConfigurator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("极致紧凑模型配置器")
        self.setGeometry(100, 100, 800, 600)
        
        # 主布局
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        # 创建标签页
        self.tab_widget = QTabWidget()
        
        # 添加各模型标签页
        self.create_tensorrt_tab()
        self.create_onnx_tab()
        self.create_yolo_tab()
        self.create_mobilenet_tab()
        self.create_openvino_tab()
        
        main_layout.addWidget(self.tab_widget)
        
        # 底部按钮
        btn_layout = QHBoxLayout()
        self.run_btn = QPushButton("运行")
        self.save_btn = QPushButton("保存配置")
        self.load_btn = QPushButton("加载配置")
        btn_layout.addWidget(self.run_btn)
        btn_layout.addWidget(self.save_btn)
        btn_layout.addWidget(self.load_btn)
        
        main_layout.addLayout(btn_layout)
        
        self.setCentralWidget(main_widget)
        
        # 连接信号
        self.run_btn.clicked.connect(self.run_configuration)
        self.save_btn.clicked.connect(self.save_configuration)
        self.load_btn.clicked.connect(self.load_configuration)
    
    def create_compact_group(self, title, params):
        """创建极致紧凑的参数组"""
        group = QGroupBox(title)
        layout = QHBoxLayout()
        
        for param in params:
            if param["type"] == "label":
                widget = QLabel(param["text"])
            elif param["type"] == "lineedit":
                widget = QLineEdit(param["default"])
                widget.setMaximumWidth(80)
            elif param["type"] == "spinbox":
                widget = QSpinBox()
                widget.setRange(param.get("min", 0), param.get("max", 100))
                widget.setValue(param.get("default", 0))
                widget.setMaximumWidth(80)
            elif param["type"] == "dspinbox":
                widget = QDoubleSpinBox()
                widget.setRange(param.get("min", 0), param.get("max", 100))
                widget.setValue(param.get("default", 0))
                widget.setSingleStep(param.get("step", 0.1))
                widget.setMaximumWidth(80)
            elif param["type"] == "checkbox":
                widget = QCheckBox(param["text"])
                widget.setChecked(param.get("checked", False))
            elif param["type"] == "combobox":
                widget = QComboBox()
                widget.addItems(param["items"])
                widget.setCurrentText(param.get("default", ""))
                widget.setMaximumWidth(120)
            
            if "tooltip" in param:
                widget.setToolTip(param["tooltip"])
            
            layout.addWidget(widget)
            param["widget"] = widget
        
        group.setLayout(layout)
        return group, params
    
    def create_tensorrt_tab(self):
        """创建TensorRT标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 模型配置
        model_params = [
            {"type": "label", "text": "模型路径:"},
            {"type": "lineedit", "default": "model.trt", "tooltip": "TensorRT模型文件路径"},
            {"type": "label", "text": "精度:"},
            {"type": "combobox", "items": ["FP32", "FP16", "INT8"], "default": "FP16"},
            {"type": "label", "text": "批大小:"},
            {"type": "spinbox", "default": 8, "min": 1, "max": 256},
        ]
        model_group, self.tensorrt_model_params = self.create_compact_group("模型配置", model_params)
        
        # 训练配置
        train_params = [
            {"type": "checkbox", "text": "启用训练", "checked": True},
            {"type": "label", "text": "学习率:"},
            {"type": "dspinbox", "default": 0.001, "min": 0, "max": 1, "step": 0.0001},
            {"type": "label", "text": "迭代:"},
            {"type": "spinbox", "default": 1000, "min": 1, "max": 100000},
        ]
        train_group, self.tensorrt_train_params = self.create_compact_group("训练配置", train_params)
        
        # 验证配置
        val_params = [
            {"type": "checkbox", "text": "启用验证", "checked": True},
            {"type": "label", "text": "间隔:"},
            {"type": "spinbox", "default": 10, "min": 1, "max": 100, "tooltip": "每多少轮验证一次"},
            {"type": "label", "text": "数据集:"},
            {"type": "lineedit", "default": "val_data", "tooltip": "验证数据集路径"},
        ]
        val_group, self.tensorrt_val_params = self.create_compact_group("验证配置", val_params)
        
        # 测试配置
        test_params = [
            {"type": "checkbox", "text": "启用测试", "checked": True},
            {"type": "label", "text": "数据集:"},
            {"type": "lineedit", "default": "test_data", "tooltip": "测试数据集路径"},
            {"type": "label", "text": "指标:"},
            {"type": "combobox", "items": ["准确率", "mAP", "F1"], "default": "mAP"},
        ]
        test_group, self.tensorrt_test_params = self.create_compact_group("测试配置", test_params)
        
        # 高级配置
        adv_params = [
            {"type": "label", "text": "工作空间(MB):"},
            {"type": "spinbox", "default": 1024, "min": 128, "max": 8192},
            {"type": "label", "text": "最大批:"},
            {"type": "spinbox", "default": 16, "min": 1, "max": 256},
            {"type": "checkbox", "text": "动态形状", "checked": False},
        ]
        adv_group, self.tensorrt_adv_params = self.create_compact_group("高级配置", adv_params)
        
        layout.addWidget(model_group)
        layout.addWidget(train_group)
        layout.addWidget(val_group)
        layout.addWidget(test_group)
        layout.addWidget(adv_group)
        layout.addStretch()
        
        self.tab_widget.addTab(tab, "TensorRT")
    
    def create_onnx_tab(self):
        """创建ONNX标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 模型配置
        model_params = [
            {"type": "label", "text": "模型路径:"},
            {"type": "lineedit", "default": "model.onnx"},
            {"type": "label", "text": "输入形状:"},
            {"type": "lineedit", "default": "1,3,224,224", "tooltip": "格式: N,C,H,W"},
            {"type": "label", "text": "输出:"},
            {"type": "lineedit", "default": "output", "tooltip": "输出节点名称"},
        ]
        model_group, self.onnx_model_params = self.create_compact_group("模型配置", model_params)
        
        # 训练配置
        train_params = [
            {"type": "checkbox", "text": "启用训练", "checked": True},
            {"type": "label", "text": "优化器:"},
            {"type": "combobox", "items": ["SGD", "Adam", "RMSprop"], "default": "Adam"},
            {"type": "label", "text": "学习率:"},
            {"type": "dspinbox", "default": 0.001, "min": 0, "max": 1, "step": 0.0001},
        ]
        train_group, self.onnx_train_params = self.create_compact_group("训练配置", train_params)
        
        # 验证配置
        val_params = [
            {"type": "checkbox", "text": "启用验证", "checked": True},
            {"type": "label", "text": "指标:"},
            {"type": "combobox", "items": ["准确率", "AUC", "混淆矩阵"], "default": "准确率"},
            {"type": "label", "text": "批大小:"},
            {"type": "spinbox", "default": 32, "min": 1, "max": 256},
        ]
        val_group, self.onnx_val_params = self.create_compact_group("验证配置", val_params)
        
        # 测试配置
        test_params = [
            {"type": "checkbox", "text": "启用测试", "checked": True},
            {"type": "label", "text": "数据集:"},
            {"type": "lineedit", "default": "test_data"},
            {"type": "label", "text": "可视化:"},
            {"type": "checkbox", "text": "启用", "checked": False},
        ]
        test_group, self.onnx_test_params = self.create_compact_group("测试配置", test_params)
        
        # 优化配置
        opt_params = [
            {"type": "label", "text": "优化级别:"},
            {"type": "combobox", "items": ["无", "基础", "扩展", "全部"], "default": "基础"},
            {"type": "checkbox", "text": "常量折叠", "checked": True},
            {"type": "checkbox", "text": "冗余消除", "checked": True},
        ]
        opt_group, self.onnx_opt_params = self.create_compact_group("优化配置", opt_params)
        
        layout.addWidget(model_group)
        layout.addWidget(train_group)
        layout.addWidget(val_group)
        layout.addWidget(test_group)
        layout.addWidget(opt_group)
        layout.addStretch()
        
        self.tab_widget.addTab(tab, "ONNX")
    
    def create_yolo_tab(self):
        """创建YOLO标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 模型配置
        model_params = [
            {"type": "label", "text": "版本:"},
            {"type": "combobox", "items": ["YOLOv3", "YOLOv4", "YOLOv5", "YOLOv8"], "default": "YOLOv5"},
            {"type": "label", "text": "尺寸:"},
            {"type": "spinbox", "default": 640, "min": 320, "max": 1280, "step": 32},
            {"type": "label", "text": "预训练:"},
            {"type": "combobox", "items": ["无", "COCO", "VOC", "自定义"], "default": "COCO"},
        ]
        model_group, self.yolo_model_params = self.create_compact_group("模型配置", model_params)
        
        # 训练配置
        train_params = [
            {"type": "checkbox", "text": "启用训练", "checked": True},
            {"type": "label", "text": "轮次:"},
            {"type": "spinbox", "default": 300, "min": 1, "max": 1000},
            {"type": "label", "text": "批大小:"},
            {"type": "spinbox", "default": 16, "min": 1, "max": 64},
            {"type": "label", "text": "学习率:"},
            {"type": "dspinbox", "default": 0.01, "min": 0, "max": 1, "step": 0.001},
        ]
        train_group, self.yolo_train_params = self.create_compact_group("训练配置", train_params)
        
        # 验证配置
        val_params = [
            {"type": "checkbox", "text": "启用验证", "checked": True},
            {"type": "label", "text": "间隔:"},
            {"type": "spinbox", "default": 10, "min": 1, "max": 100},
            {"type": "label", "text": "mAP阈值:"},
            {"type": "dspinbox", "default": 0.5, "min": 0.1, "max": 0.9, "step": 0.05},
        ]
        val_group, self.yolo_val_params = self.create_compact_group("验证配置", val_params)
        
        # 测试配置
        test_params = [
            {"type": "checkbox", "text": "启用测试", "checked": True},
            {"type": "label", "text": "置信度:"},
            {"type": "dspinbox", "default": 0.25, "min": 0.1, "max": 0.9, "step": 0.05},
            {"type": "label", "text": "NMS阈值:"},
            {"type": "dspinbox", "default": 0.45, "min": 0.1, "max": 0.9, "step": 0.05},
        ]
        test_group, self.yolo_test_params = self.create_compact_group("测试配置", test_params)
        
        # 数据配置
        data_params = [
            {"type": "label", "text": "类别数:"},
            {"type": "spinbox", "default": 80, "min": 1, "max": 1000},
            {"type": "label", "text": "数据文件:"},
            {"type": "lineedit", "default": "data.yaml"},
            {"type": "checkbox", "text": "马赛克增强", "checked": True},
        ]
        data_group, self.yolo_data_params = self.create_compact_group("数据配置", data_params)
        
        layout.addWidget(model_group)
        layout.addWidget(train_group)
        layout.addWidget(val_group)
        layout.addWidget(test_group)
        layout.addWidget(data_group)
        layout.addStretch()
        
        self.tab_widget.addTab(tab, "YOLO")
    
    def create_mobilenet_tab(self):
        """创建MobileNet标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 模型配置
        model_params = [
            {"type": "label", "text": "版本:"},
            {"type": "combobox", "items": ["MobileNetV1", "MobileNetV2", "MobileNetV3"], "default": "MobileNetV2"},
            {"type": "label", "text": "宽度因子:"},
            {"type": "dspinbox", "default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1},
            {"type": "label", "text": "分辨率:"},
            {"type": "spinbox", "default": 224, "min": 128, "max": 512, "step": 32},
        ]
        model_group, self.mobilenet_model_params = self.create_compact_group("模型配置", model_params)
        
        # 训练配置
        train_params = [
            {"type": "checkbox", "text": "启用训练", "checked": True},
            {"type": "label", "text": "轮次:"},
            {"type": "spinbox", "default": 100, "min": 1, "max": 500},
            {"type": "label", "text": "批大小:"},
            {"type": "spinbox", "default": 32, "min": 1, "max": 256},
            {"type": "label", "text": "学习率:"},
            {"type": "dspinbox", "default": 0.001, "min": 0, "max": 1, "step": 0.0001},
        ]
        train_group, self.mobilenet_train_params = self.create_compact_group("训练配置", train_params)
        
        # 验证配置
        val_params = [
            {"type": "checkbox", "text": "启用验证", "checked": True},
            {"type": "label", "text": "间隔:"},
            {"type": "spinbox", "default": 5, "min": 1, "max": 100},
            {"type": "label", "text": "Top-K:"},
            {"type": "spinbox", "default": 5, "min": 1, "max": 10},
        ]
        val_group, self.mobilenet_val_params = self.create_compact_group("验证配置", val_params)
        
        # 测试配置
        test_params = [
            {"type": "checkbox", "text": "启用测试", "checked": True},
            {"type": "label", "text": "数据集:"},
            {"type": "lineedit", "default": "test_data"},
            {"type": "label", "text": "指标:"},
            {"type": "combobox", "items": ["准确率", "Top-1", "Top-5"], "default": "准确率"},
        ]
        test_group, self.mobilenet_test_params = self.create_compact_group("测试配置", test_params)
        
        # 优化配置
        opt_params = [
            {"type": "checkbox", "text": "量化训练", "checked": False},
            {"type": "label", "text": "量化位:"},
            {"type": "combobox", "items": ["8位", "16位"], "default": "8位"},
            {"type": "checkbox", "text": "剪枝", "checked": False},
        ]
        opt_group, self.mobilenet_opt_params = self.create_compact_group("优化配置", opt_params)
        
        layout.addWidget(model_group)
        layout.addWidget(train_group)
        layout.addWidget(val_group)
        layout.addWidget(test_group)
        layout.addWidget(opt_group)
        layout.addStretch()
        
        self.tab_widget.addTab(tab, "MobileNet")
    
    def create_openvino_tab(self):
        """创建OpenVINO标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 模型配置
        model_params = [
            {"type": "label", "text": "XML路径:"},
            {"type": "lineedit", "default": "model.xml"},
            {"type": "label", "text": "BIN路径:"},
            {"type": "lineedit", "default": "model.bin"},
            {"type": "label", "text": "设备:"},
            {"type": "combobox", "items": ["CPU", "GPU", "MYRIAD"], "default": "CPU"},
        ]
        model_group, self.openvino_model_params = self.create_compact_group("模型配置", model_params)
        
        # 训练配置
        train_params = [
            {"type": "checkbox", "text": "启用训练", "checked": True},
            {"type": "label", "text": "学习率:"},
            {"type": "dspinbox", "default": 0.001, "min": 0, "max": 1, "step": 0.0001},
            {"type": "label", "text": "迭代:"},
            {"type": "spinbox", "default": 1000, "min": 1, "max": 100000},
            {"type": "label", "text": "优化器:"},
            {"type": "combobox", "items": ["SGD", "Adam"], "default": "Adam"},
        ]
        train_group, self.openvino_train_params = self.create_compact_group("训练配置", train_params)
        
        # 验证配置
        val_params = [
            {"type": "checkbox", "text": "启用验证", "checked": True},
            {"type": "label", "text": "间隔:"},
            {"type": "spinbox", "default": 10, "min": 1, "max": 100},
            {"type": "label", "text": "批大小:"},
            {"type": "spinbox", "default": 16, "min": 1, "max": 256},
        ]
        val_group, self.openvino_val_params = self.create_compact_group("验证配置", val_params)
        
        # 测试配置
        test_params = [
            {"type": "checkbox", "text": "启用测试", "checked": True},
            {"type": "label", "text": "数据集:"},
            {"type": "lineedit", "default": "test_data"},
            {"type": "label", "text": "推理线程:"},
            {"type": "spinbox", "default": 4, "min": 1, "max": 32},
        ]
        test_group, self.openvino_test_params = self.create_compact_group("测试配置", test_params)
        
        # 高级配置
        adv_params = [
            {"type": "checkbox", "text": "异步推理", "checked": True},
            {"type": "label", "text": "缓存路径:"},
            {"type": "lineedit", "default": "cache"},
            {"type": "checkbox", "text": "启用FP16", "checked": False},
        ]
        adv_group, self.openvino_adv_params = self.create_compact_group("高级配置", adv_params)
        
        layout.addWidget(model_group)
        layout.addWidget(train_group)
        layout.addWidget(val_group)
        layout.addWidget(test_group)
        layout.addWidget(adv_group)
        layout.addStretch()
        
        self.tab_widget.addTab(tab, "OpenVINO")
    
    def run_configuration(self):
        """运行配置"""
        current_tab = self.tab_widget.tabText(self.tab_widget.currentIndex())
        print(f"正在运行 {current_tab} 配置...")
        # 这里应该添加实际运行模型的代码
    
    def save_configuration(self):
        """保存配置"""
        current_tab = self.tab_widget.tabText(self.tab_widget.currentIndex())
        print(f"正在保存 {current_tab} 配置...")
        # 这里应该添加实际保存配置的代码
    
    def load_configuration(self):
        """加载配置"""
        current_tab = self.tab_widget.tabText(self.tab_widget.currentIndex())
        print(f"正在加载 {current_tab} 配置...")
        # 这里应该添加实际加载配置的代码

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CompactModelConfigurator()
    window.show()
    sys.exit(app.exec_())