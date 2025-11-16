import onnx

def simple_inspect(model_path):
    model = onnx.load(model_path)
    
    print("=== 简化模型信息 ===")
    
    # 输入信息
    inputs = model.graph.input
    print("输入节点:")
    for inp in inputs:
        shape = []
        for dim in inp.type.tensor_type.shape.dim:
            if dim.dim_value > 0:
                shape.append(dim.dim_value)
            else:
                shape.append("?")
        print(f"  - {inp.name}: 形状{shape}")
    
    # 输出信息
    outputs = model.graph.output
    print("输出节点:")
    for out in outputs:
        shape = []
        for dim in out.type.tensor_type.shape.dim:
            if dim.dim_value > 0:
                shape.append(dim.dim_value)
            else:
                shape.append("?")
        print(f"  - {out.name}: 形状{shape}")

# 使用
simple_inspect("/home/yksilvery/RM/Aimbot/src/model/best.onnx")