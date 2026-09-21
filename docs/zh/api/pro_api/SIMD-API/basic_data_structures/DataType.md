# pypto_pro.language.DataType

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR&950DT系列产品：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3系列产品：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2系列产品：不支持
<!-- end id3 -->

## 功能说明

数据类型的枚举。

## 原型定义

```python
DT_FP4 = DataType.FP4               # 4位浮点数，2位指数，1位尾数，两个占用1字节内存
DT_FP8E4M3FN = DataType.FP8E4M3FN   # 8位浮点数，4位指数，3位尾数，占用1字节内存
DT_FP8E5M2 = DataType.FP8E5M2       # 8位浮点数，5位指数，2位尾数，占用1字节内存
DT_FP8E8M0 = DataType.FP8E8M0       # 8位浮点数，8位指数，0位尾数，占用1字节内存
DT_FP4E2M1 = DataType.FP4E2M1       # 4位浮点数，2位指数，1位尾数，两个占用1字节内存
DT_FP4E1M2 = DataType.FP4E1M2       # 4位浮点数，1位指数，2位尾数，两个占用1字节内存
DT_FP16 = DataType.FP16             # 16位半精度浮点数，占用2字节内存
DT_FP32 = DataType.FP32             # 32位单精度浮点数，占用4字节内存
DT_BF16 = DataType.BF16             # 16位Brain Float格式，占用2字节内存
DT_HF4 = DataType.HF4               # 4位HiFloat格式，两个占用1字节内存
DT_HF8 = DataType.HF8               # 8位HiFloat格式，占用1字节内存
DT_INT4 = DataType.INT4             # 4位有符号整数，占用1字节内存
DT_INT8 = DataType.INT8             # 8位有符号整数，占用1字节内存
DT_INT16 = DataType.INT16           # 16位有符号整数，占用2字节内存
DT_INT32 = DataType.INT32           # 32位有符号整数，占用4字节内存
DT_INT64 = DataType.INT64           # 64位有符号整数，占用8字节内存
DT_UINT4 = DataType.UINT4           # 4位无符号整数，占用1字节内存
DT_UINT8 = DataType.UINT8           # 8位无符号整数，占用1字节内存
DT_UINT16 = DataType.UINT16         # 16位无符号整数，占用2字节内存
DT_UINT32 = DataType.UINT32         # 32位无符号整数，占用4字节内存
DT_UINT64 = DataType.UINT64         # 64位无符号整数，占用8字节内存
DT_BOOL = DataType.BOOL             # 布尔类型，占用1字节内存
```

上述DT_XXX常量是Python侧的别名，DataType本身并非Python枚举，而是C++类经绑定后暴露到Python的同名类型。每个DataType.XXX都是该类的一个静态常量实例。

```C++
class DataType {
public:
    static const DataType BOOL;      // Boolean (true/false)
    static const DataType INT4;      // 4-bit signed integer
    static const DataType INT8;      // 8-bit signed integer
    static const DataType INT16;     // 16-bit signed integer
    static const DataType INT32;     // 32-bit signed integer
    static const DataType INT64;     // 64-bit signed integer
    static const DataType UINT4;     // 4-bit unsigned integer
    static const DataType UINT8;     // 8-bit unsigned integer
    static const DataType UINT16;    // 16-bit unsigned integer
    static const DataType UINT32;    // 32-bit unsigned integer
    static const DataType UINT64;    // 64-bit unsigned integer
    static const DataType FP4;       // 4-bit floating point
    static const DataType FP8E4M3FN; // 8-bit floating point (IEEE 754 e4m3fn format)
    static const DataType FP8E5M2;   // 8-bit floating point (IEEE 754 e5m2 format)
    static const DataType FP8;       // 8-bit floating point (backward compatibility alias)
    static const DataType FP8E8M0;   // 8-bit floating point (8-bit exponent, 0-bit mantissa)
    static const DataType FP4E2M1;   // 4-bit floating point (2-bit exponent, 1-bit mantissa)
    static const DataType FP4E1M2;   // 4-bit floating point (1-bit exponent, 2-bit mantissa)
    static const DataType FP16;      // 16-bit floating point (IEEE 754 half precision)
    static const DataType FP32;      // 32-bit floating point (IEEE 754 single precision)
    static const DataType FP64;      // 64-bit floating point (IEEE 754 double precision)
    static const DataType BF16;      // 16-bit brain floating point
    static const DataType HF4;       // 4-bit Hisilicon float
    static const DataType HF8;       // 8-bit Hisilicon float
    static const DataType INDEX;     // 64-bit index type
}
```

## 约束说明

- 裸整数常量（如42）使用DT_INT64。只有数值超过DT_INT64的上限时，才使用DT_UINT64。裸浮点常量（如3.14）使用DT_FP32。

- MXFP8的数据元素使用DT_FP8E4M3FN或DT_FP8E5M2，MXFP4的数据元素使用DT_FP4E2M1或DT_FP4E1M2；两者均使用DT_FP8E8M0保存分组缩放因子。详见[matmul_mx](../cube_computation/matmul_mx.md)和[matmul_mx_acc](../cube_computation/matmul_mx_acc.md)。

## 常用接口

| 方法 | 说明 |
|---|---|
| dtype.get_bit() / dtype.bits() | 取位宽（如pypto_pro.language.DT_FP16.get_bit()返回16） |
| dtype.is_float() | 是否浮点类型 |
| dtype.is_int() | 是否整型（有符号或无符号） |
| dtype.is_signed_int() / dtype.is_signed() | 是否有符号整型 |
| dtype.is_unsigned_int() / dtype.is_unsigned() | 是否无符号整型 |
| dtype.to_string() / str(dtype) | 人类可读名称（如"fp16"） |
| dtype.to_c_type_string() | C类型字符串（如"half"） |
