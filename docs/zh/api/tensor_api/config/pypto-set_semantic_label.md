# pypto.set\_semantic\_label

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR&950DT系列产品：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3系列产品：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2系列产品：支持
<!-- end id3 -->

## 功能说明

设置代码段的语义标签，PyPTO Toolkit将识别该标签后的代码行，直到遇到下一个语义标签，这有助于用户更方便地定位问题。

## 函数原型

```python
set_semantic_label(label: str) -> None
```

## 参数说明

| 参数名  | 输入/输出 | 说明                  |
|---------|-----------|-----------------------|
| label   | 输入      | 设置语义标签的名称，支持任意字符串。 |

## 返回值说明

无返回值，设置操作成功即生效。

## 约束说明

无。

## 调用示例

```python
pypto.set_semantic_label("kv")
compressed_kv = pypto.view(kv_tmp, [tile_b, s, kv_lora_rank], [0, 0, 0])
...
```
