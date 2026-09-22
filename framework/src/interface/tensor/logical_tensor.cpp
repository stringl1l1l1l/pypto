/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file logical_tensor.cpp
 * \brief
 */

#include "interface/configs/config_manager.h"
#include "logical_tensor.h"

#include "irbuilder.h"
#include "raw_tensor.h"
#include "tilefwk/data_type.h"
#include "tilefwk/symbolic_scalar.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/utils/error.h"
#include "interface/program/program.h"
#include "interface/utils/id_gen.h"
#include "interface/function/function.h"
#include "interface/utils/serialization.h"
#include "passes/pass_utils/subgraph_utils.h"
#include "passes/pass_utils/pass_utils.h"

#include "irbuilder.h"

using namespace npu::tile_fwk;

LogicalTensor::LogicalTensor(Function& function, DataType t, Shape tshape, TileOpFormat format, std::string tname)
    : ir::Var(IRContext::Get().GetVarName(tname), ir::GetLogicalTensorType(), ir::Span::Unknown()),
      tensor(std::make_shared<RawTensor>(t, tshape, format, std::move(tname))),
      offset(Offset(tshape.size(), 0)),
      shape(tshape),
      oriShape(tshape),
      magic(IdGen<IdType::LOGICAL_TENSOR>::Inst().NewId()),
      function_(&function)
{}

LogicalTensor::LogicalTensor(Function& function, DataType t, Shape tshape, std::vector<SymbolicScalar> tValidShape,
                             TileOpFormat format, std::string tname)
    : ir::Var(IRContext::Get().GetVarName(tname), ir::GetLogicalTensorType(), ir::Span::Unknown()),
      tensor(std::make_shared<RawTensor>(t, tshape, format, std::move(tname))),
      offset(Offset(tshape.size(), 0)),
      shape(tshape),
      oriShape(tshape),
      dynValidShape_(tValidShape),
      storageShape(tshape),
      magic(IdGen<IdType::LOGICAL_TENSOR>::Inst().NewId()),
      function_(&function)
{}

LogicalTensor::LogicalTensor(Function& function, std::shared_ptr<RawTensor> rawTensor, Offset toffset, Shape tshape)
    : ir::Var(IRContext::Get().GetVarName(rawTensor->GetSymbol()), ir::GetLogicalTensorType(), ir::Span::Unknown()),
      tensor(rawTensor),
      offset(toffset),
      shape(tshape),
      oriShape(tshape),
      magic(IdGen<IdType::LOGICAL_TENSOR>::Inst().NewId()),
      function_(&function)
{
    // Initialize other members if necessary
    FE_ASSERT(FeError::INVALID_VAL, shape.size() == offset.size())
        << "shape.size(): " << shape.size() << ", offset.size(): " << offset.size();
}

LogicalTensor::LogicalTensor(Function& function, std::shared_ptr<RawTensor> rawTensor, Offset toffset, Shape tshape,
                             std::vector<SymbolicScalar> tValidShape)
    : ir::Var(IRContext::Get().GetVarName(rawTensor->GetSymbol()), ir::GetLogicalTensorType(), ir::Span::Unknown()),
      tensor(rawTensor),
      offset(toffset),
      shape(tshape),
      oriShape(tshape),
      dynValidShape_(tValidShape),
      magic(IdGen<IdType::LOGICAL_TENSOR>::Inst().NewId()),
      function_(&function)
{
    // Initialize other members if necessary
    FE_ASSERT(FeError::INVALID_VAL, shape.size() == offset.size())
        << "shape.size(): " << shape.size() << ", offset.size(): " << offset.size();
}

void LogicalTensor::UpdateBelongFunction(Function& newFunc)
{
    if (function_ == &newFunc) {
        return;
    }
    auto self = shared_from_this();
    function_ = &newFunc;
}

std::shared_ptr<LogicalTensor> LogicalTensor::Clone(Function& dstFunc, bool create) const
{
    /* Clone is only for dstFunc to simplify the process of creating OP_CALL's input and output. */
    if (!create) {
        auto cloned = dstFunc.GetTensorMap().GetTensorByMagic(magic);
        if (cloned != nullptr) {
            return cloned;
        }
    }

    std::shared_ptr<RawTensor> rawTensor = dstFunc.GetTensorMap().GetRawTensorByRawMagic(tensor->rawmagic);
    if (rawTensor == nullptr || create) {
        if (create) {
            rawTensor = std::make_shared<RawTensor>(tensor->datatype, tensor->rawshape, tensor->format, tensor->symbol);
        } else {
            rawTensor = std::make_shared<RawTensor>(tensor->datatype, tensor->rawshape, tensor->format, tensor->symbol,
                                                    tensor->rawmagic);
            rawTensor->actualRawmagic = tensor->actualRawmagic;
            rawTensor->memoryId = tensor->memoryId;
        }
        rawTensor->SetSymbol(tensor->GetSymbol());
        rawTensor->UpdateDynRawShape(tensor->GetDynRawShape());
    }

    std::shared_ptr<LogicalTensor> newTensor = std::make_shared<LogicalTensor>(dstFunc, rawTensor, offset, shape,
                                                                               dynValidShape_);
    if (!create) {
        newTensor->magic = magic;
    } else {
        newTensor->magic = IdGen<IdType::LOGICAL_TENSOR>::Inst().NewId();
    }

    newTensor->memoryrange = memoryrange;
    newTensor->memoryTypeOriginal_ = memoryTypeOriginal_;
    newTensor->memoryTypeToBe_ = memoryTypeToBe_;
    newTensor->readyTime_ = readyTime_;
    newTensor->remainingTime_ = remainingTime_;
    newTensor->dynOffset_ = dynOffset_;
    if (HasAttr("NORMAL")) {
        newTensor->SetAttr("NORMAL", true);
    }
    dstFunc.GetTensorMap().Insert(newTensor, false);
    return newTensor;
}

ir::VarPtr LogicalTensor::Clone() const
{
    auto cloned = Clone(*function_, true);
    cloned->name_ = IRContext::Get().GetVarName(name_);
    return std::static_pointer_cast<const ir::Var>(cloned);
}

ir::VarPtr LogicalTensor::Clone(bool shareRawTensor) const
{
    if (!shareRawTensor) {
        return Clone();
    }

    auto newTensor = std::make_shared<LogicalTensor>(*function_, tensor, offset, shape, dynValidShape_);
    newTensor->magic = IdGen<IdType::LOGICAL_TENSOR>::Inst().NewId();
    newTensor->name_ = IRContext::Get().GetVarName(name_);
    newTensor->oriShape = oriShape;
    newTensor->memoryrange = memoryrange;
    newTensor->memoryTypeOriginal_ = memoryTypeOriginal_;
    newTensor->memoryTypeToBe_ = memoryTypeToBe_;
    newTensor->readyTime_ = readyTime_;
    newTensor->remainingTime_ = remainingTime_;
    newTensor->dynOffset_ = dynOffset_;
    if (HasAttr("NORMAL")) {
        newTensor->SetAttr("NORMAL", true);
    }

    return std::static_pointer_cast<const ir::Var>(newTensor);
}

LogicalTensorPtr LogicalTensor::NextVersion(Function& func, std::vector<ir::VarPtr>& tokens) const
{
    (void)tokens;
    IRBuilder builder;
    auto newTensor = builder.CreateTensorVar(func, tensor, offset, shape, dynValidShape_);
    newTensor->oriShape = oriShape;
    newTensor->dynOffset_ = dynOffset_;
    newTensor->storageShape = storageShape;
    newTensor->storage_ = storage_;
    newTensor->storageOffset_ = storageOffset_;
    newTensor->memoryrange = memoryrange;
    newTensor->memoryTypeOriginal_ = memoryTypeOriginal_;
    newTensor->memoryTypeToBe_ = memoryTypeToBe_;
    newTensor->GetAllAttr() = GetAllAttr();
    return newTensor;
}

Json LogicalTensor::DumpJson(Function& func, bool dumpRawTensor) const
{
    Json result;
    result[T_FIELD_KIND] = static_cast<int>(Kind::T_KIND_TENSOR);
    result["offset"] = offset;
    result["shape"] = shape;
    result["validshape"] = oriShape;
    result["nodetype"] = static_cast<int>(FunctionUtils::GetNodeType(*this, func));
    if (dumpRawTensor) {
        result[T_FIELD_RAWTENSOR] = tensor->DumpJson();
    } else {
        result[T_FIELD_RAWTENSOR] = tensor->rawmagic;
    }
    result["magic"] = magic;
    if (storage_ != nullptr) {
        result["storage"] = storage_->DumpJson();
    }
    if (HasAttr(OpAttributeKey::needAlloc)) {
        bool allocValue = false;
        GetAttr(OpAttributeKey::needAlloc, allocValue);
        result["need_alloc"] = allocValue;
    }

    result["mem_range"] = Json(std::vector<std::size_t>({memoryrange.start, memoryrange.end}));
    result["life_range"] = Json(std::vector<int>({memoryrange.lifeStart, memoryrange.lifeEnd}));
    result["mem_id"] = Json(memoryrange.memId);

    if (GetMemoryTypeOriginal() != MemoryType::MEM_UNKNOWN || GetMemoryTypeToBe() != MemoryType::MEM_UNKNOWN) {
        Json memorytype = Json::object();
        if (GetMemoryTypeOriginal() != MemoryType::MEM_UNKNOWN) {
            memorytype["asis"] = static_cast<int>(GetMemoryTypeOriginal());
        }
        if (GetMemoryTypeToBe() != MemoryType::MEM_UNKNOWN) {
            memorytype["tobe"] = static_cast<int>(GetMemoryTypeToBe());
        }
        result["mem_type"] = memorytype;
    }
    Json offsetJson = Json::array();
    for (auto dynOffset : dynOffset_) {
        auto joffset = ToJson(dynOffset);
        if (joffset.size() > 0) {
            offsetJson.push_back(joffset);
        }
    }
    if (offsetJson.size() > 0) {
        result["dynoffset"] = offsetJson;
    }
    Json dynValidShapeJson = Json::array();
    for (auto dynValidShape : dynValidShape_) {
        auto jValidShape = ToJson(dynValidShape);
        if (jValidShape.size() > 0) {
            dynValidShapeJson.push_back(jValidShape);
        }
    }
    if (dynValidShapeJson.size() > 0) {
        result["dynvalidshape"] = dynValidShapeJson;
    }
    return result;
}

std::shared_ptr<LogicalTensor> LogicalTensor::LoadJson(
    Function& function, const std::unordered_map<int, std::shared_ptr<RawTensor>>& rawTensorDict,
    const Json& tensorDump)
{
    FE_ASSERT(tensorDump[T_FIELD_KIND].get<int>() == static_cast<int>(Kind::T_KIND_TENSOR))
        << "[tensorDump]json field<" << T_FIELD_KIND << "> doesn't match T_KIND_TENSOR.";

    Offset toffset = tensorDump["offset"].get<std::vector<int64_t>>();
    Shape tshape = tensorDump["shape"].get<std::vector<int64_t>>();

    std::shared_ptr<RawTensor> rawTensor;
    if (tensorDump[T_FIELD_RAWTENSOR].is_number()) {
        int rawTensorMagic = tensorDump[T_FIELD_RAWTENSOR].get<int>();
        FE_ASSERT(FeError::NOT_EXIST, rawTensorDict.count(rawTensorMagic))
            << "rawTensorDict doesn't have magic " << rawTensorMagic;
        rawTensor = rawTensorDict.find(rawTensorMagic)->second;
    } else {
        rawTensor = RawTensor::LoadJson(tensorDump[T_FIELD_RAWTENSOR]);
    }
    int tensorMagic = tensorDump["magic"].get<int>();

    std::shared_ptr<LogicalTensor> tensorJson = std::make_shared<LogicalTensor>(function, rawTensor, toffset, tshape);
    tensorJson->magic = tensorMagic;

    if (tensorDump.count("need_alloc") != 0) {
        bool needAlloc = tensorDump["need_alloc"].get<bool>();
        tensorJson->SetAttr(OpAttributeKey::needAlloc, needAlloc);
    }

    if (tensorDump.count("mem_range")) {
        tensorJson->memoryrange = TileRange(tensorDump["mem_range"][0].get<int>(),
                                            tensorDump["mem_range"][1].get<int>());
    }
    if (tensorDump.count("life_range")) {
        tensorJson->memoryrange.lifeStart = tensorDump["life_range"][0].get<int>();
        tensorJson->memoryrange.lifeEnd = tensorDump["life_range"][1].get<int>();
    }
    if (tensorDump.count("mem_id")) {
        tensorJson->memoryrange.memId = tensorDump["mem_id"].get<int>();
    }
    if (tensorDump.count("mem_type")) {
        auto& memorytype = tensorDump["mem_type"];
        if (memorytype.count("asis")) {
            tensorJson->memoryTypeOriginal_ = static_cast<MemoryType>(memorytype["asis"].get<int>());
        }
        if (memorytype.count("tobe")) {
            tensorJson->memoryTypeToBe_ = static_cast<MemoryType>(memorytype["tobe"].get<int>());
        }
    }
    if (tensorDump.count("storage")) {
        tensorJson->storage_ = Storage::LoadJson(tensorDump["storage"]);
    }
    if (tensorDump.count("validshape")) {
        tensorJson->oriShape = tensorDump["validshape"].get<std::vector<int64_t>>();
    }
    if (tensorDump.count("dynoffset")) {
        auto dynoffsetJson = tensorDump["dynoffset"];
        std::vector<SymbolicScalar> dynOffset;
        for (auto offsetJson : dynoffsetJson) {
            dynOffset.push_back(LoadSymbolicScalar(offsetJson));
        }
        tensorJson->dynOffset_ = dynOffset;
    }
    if (tensorDump.count("dynvalidshape")) {
        auto dynvalidJson = tensorDump["dynvalidshape"];
        std::vector<SymbolicScalar> dynValidShape;
        for (auto validJson : dynvalidJson) {
            dynValidShape.push_back(LoadSymbolicScalar(validJson));
        }
        tensorJson->UpdateDynValidShape(dynValidShape);
    }
    return tensorJson;
}

std::string LogicalTensor::DumpType() const
{
    std::string result = "<";
    for (auto& value : shape) {
        result += std::to_string(value) + "x";
    }
    result += DataType2String(Datatype());
    if (dynValidShape_.size() != 0) {
        result += " / ";
        for (auto& value : dynValidShape_) {
            result += value.Dump() + "x";
        }
        result += DataType2String(Datatype());
        if (tensor->format == TileOpFormat::TILEOP_NZ) {
            result += "_NZ";
        }
    }
    result += ">";
    return result;
}

std::string LogicalTensor::DumpSSA([[maybe_unused]] bool showFrom, bool showMem, bool showType) const
{
    std::ostringstream oss;
    if (showType) {
        oss << DumpType() << " ";
    }
    oss << "%" << GetMagic() << GetRawTensor()->DumpSSA(false, false);

    if (not std::all_of(offset.begin(), offset.end(), [](int ox) { return ox == 0; })) {
        oss << "(";
        for (size_t i = 0; i < offset.size(); ++i) {
            oss << offset[i];
            if (i != offset.size() - 1) {
                oss << ", ";
            }
        }
        oss << ")";
    }
    if (dynOffset_.size() != 0) {
        oss << "(";
        for (size_t i = 0; i < dynOffset_.size(); ++i) {
            oss << dynOffset_[i].Dump();
            if (i != offset.size() - 1) {
                oss << ", ";
            }
        }
        oss << ")";
    }
    oss << "#"
        << "(" << CommonUtils::GetTensorSubgraphID(this) << ")";
    if (showMem) {
        oss << MemoryTypeToString(GetMemoryTypeOriginal()) << "::" << MemoryTypeToString(GetMemoryTypeToBe());
        if (IsDummy()) {
            oss << "::IsDummy";
        }
    }
    return oss.str();
}

std::string LogicalTensor::Dump(bool showFrom, bool showMem) const { return DumpSSA(showFrom, showMem); }

std::shared_ptr<LogicalTensor> LogicalTensor::View(Function& function, const Shape& newShape,
                                                   const Offset& newOffset) const
{
    FE_ASSERT(FeError::INVALID_VAL, shape.size() == newShape.size()) << "Tensor.view, shape must be the same dimension";
    FE_ASSERT(FeError::INVALID_VAL, offset.size() == newOffset.size())
        << "Tensor.view, offset must be the same dimension";

    auto view = std::make_shared<LogicalTensor>(function, this->tensor, this->offset, this->shape);
    for (size_t i = 0; i < shape.size(); i++) {
        FE_ASSERT(FeError::OUT_OF_RANGE, shape[i] >= (newShape[i] + newOffset[i]))
            << "i: " << i << ", shape[i]: " << shape[i] << ", newShape[i]: " << newShape[i]
            << ", newOffset[i]: " << newOffset[i];
    }

    view->shape = newShape;
    view->oriShape = newShape;
    view->offset = TensorOffset::Add(offset, newOffset);

    if (dynOffset_.size() != 0) {
        view->dynOffset_ = TensorOffset::Add(dynOffset_, newOffset);
    }
    view->dynValidShape_ = GetViewValidShape(dynValidShape_, newOffset, {}, newShape);

    auto parent = const_cast<LogicalTensor*>(this)->shared_from_this();
    if (HasAttr("SLICE_PARENT")) {
        GetAttr("SLICE_PARENT", parent);
    }
    view->SetAttr("SLICE_PARENT", parent);
    return view;
}

std::string LogicalTensor::Symbol() const { return tensor->symbol; }

DataType LogicalTensor::Datatype() const { return tensor->datatype; }

MemoryType LogicalTensor::GetMemoryTypeOriginal() const { return memoryTypeOriginal_; }

MemoryType LogicalTensor::GetMemoryTypeToBe() const { return memoryTypeToBe_; }

void LogicalTensor::CopyMemoryType(const std::shared_ptr<LogicalTensor>& other)
{
    memoryTypeOriginal_ = other->GetMemoryTypeOriginal();
    memoryTypeToBe_ = other->GetMemoryTypeToBe();
}

void LogicalTensor::SetMemoryTypeBoth(MemoryType t, bool force)
{
    SetMemoryTypeOriginal(t, force);
    SetMemoryTypeToBe(t);
}

void LogicalTensor::SetMemoryTypeOriginal(MemoryType t, bool force)
{
    if (t == MemoryType::MEM_UNKNOWN) {
        return;
    }

    if (memoryTypeOriginal_ == MemoryType::MEM_UNKNOWN) {
        memoryTypeOriginal_ = t;
    } else if (memoryTypeOriginal_ != t) {
        if (force) {
            memoryTypeOriginal_ = t;
        }
    }
}

void LogicalTensor::SetMemoryTypeToBe(MemoryType t)
{
    if (t == MemoryType::MEM_UNKNOWN) {
        return;
    }
    memoryTypeToBe_ = t;
}

bool LogicalTensor::MemoryConflict() const
{
    return memoryTypeOriginal_ != MemoryType::MEM_UNKNOWN && memoryTypeToBe_ != MemoryType::MEM_UNKNOWN &&
           memoryTypeOriginal_ != memoryTypeToBe_;
}

size_t LogicalTensor::MemorySize() const
{
    if (IsDummy()) {
        return 0;
    }

    int64_t numel = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<>());
    size_t baseMemorySize = static_cast<size_t>(DataSizeOf(numel, Datatype()));

    switch (GetMemoryTypeToBe()) {
        case MemoryType::MEM_UB: // 32B align
            return (baseMemorySize + ALIGN_SIZE_32 - 1) / ALIGN_SIZE_32 * ALIGN_SIZE_32;
        case MemoryType::MEM_L0A:
        case MemoryType::MEM_L0B:
        case MemoryType::MEM_L0C: // 512B align
            return (baseMemorySize + ALIGN_SIZE_512 - 1) / ALIGN_SIZE_512 * ALIGN_SIZE_512;
        default:
            return baseMemorySize;
    }
}

bool LogicalTensor::IsDummy() const { return tensor->IsDummy(); }

void LogicalTensor::SetIsDummy(bool dummy) { tensor->SetIsDummy(dummy); }

bool LogicalTensor::Overlap(const std::shared_ptr<LogicalTensor>& other) const
{
    if (tensor->rawmagic != other->tensor->rawmagic) {
        return false;
    }
    // Check if the shape and offsets overlap
    for (size_t i = 0; i < shape.size(); ++i) {
        if (offset[i] + shape[i] <= other->offset[i] || other->offset[i] + other->shape[i] <= offset[i]) {
            return false;
        }
    }
    return true;
}

int64_t LogicalTensor::GetDataSize() const
{
    if (HasNegativeNum<int64_t>(shape)) {
        FE_LOGD("Logical tensor shape has negative dimension(s), which indicates a dynamic axis.");
        return INT64_MAX;
    }
    int64_t shapeSize = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<>());
    return DataSizeOf(shapeSize, tensor->GetDataType());
}

bool LogicalTensor::CompareOp::operator()(const Operation* a, const Operation* b) const
{
    int funcMagicA = a->BelongTo()->GetFuncMagic();
    int funcMagicB = b->BelongTo()->GetFuncMagic();
    if (funcMagicA != funcMagicB) {
        return funcMagicA < funcMagicB;
    }
    int opmagicA = a->opmagic;
    int opmagicB = b->opmagic;
    return opmagicA < opmagicB;
}

bool LogicalTensor::IsGetTensorDataOutcast()
{
    for (auto& prod : GetProducers()) {
        if (prod->GetOpcode() == Opcode::OP_ASSEMBLE) {
            for (auto& prodProd : prod->iOperand[0]->GetProducers()) {
                if (CheckEmuOpcode(prodProd, EMUOP_TENSOR_EXTRACT)) {
                    return true;
                }
            }
        }
    }
    return false;
}

SymbolicScalar npu::tile_fwk::GetViewValidShapeDim(const SymbolicScalar& validShapeDim,
                                                   const SymbolicScalar& viewOffsetDim,
                                                   const SymbolicScalar& viewShapeDim)
{
    auto result = std::min(std::max(validShapeDim - viewOffsetDim, 0L), viewShapeDim);
    return result.Simplify();
}

std::vector<SymbolicScalar> npu::tile_fwk::GetViewValidShape(const std::vector<SymbolicScalar>& validShape,
                                                             const Offset& viewOffset,
                                                             const std::vector<SymbolicScalar>& viewDynOffset,
                                                             const Shape& viewShape)
{
    if (validShape.size() == 0) {
        return {};
    }
    FE_ASSERT(FeError::INVALID_VAL, validShape.size() == viewShape.size())
        << "View rank mismatch: input tensor rank is " << validShape.size() << ", view shape rank is "
        << viewShape.size() << ". pypto.view cannot change tensor rank; use pypto.unsqueeze to add a dimension.";

    std::vector<SymbolicScalar> result;
    for (size_t i = 0; i < validShape.size(); i++) {
        SymbolicScalar validShapeDim;
        if (viewDynOffset.size() != 0) {
            validShapeDim = GetViewValidShapeDim(validShape[i], viewDynOffset[i], viewShape[i]);
        } else {
            validShapeDim = GetViewValidShapeDim(validShape[i], viewOffset[i], viewShape[i]);
        }
        result.push_back(validShapeDim);
    }
    return result;
}

namespace npu::tile_fwk {

bool CheckEmuOpcode(const Operation* op, EmuOpcode opcode)
{
    if (!op->HasAttr(OP_EMUOP_PREFIX + "opc")) {
        return false;
    }
    if (op->GetIntAttribute(OP_EMUOP_PREFIX + "opc") != opcode) {
        return false;
    }
    return true;
}

void SetEmuOpcode(Operation* op, EmuOpcode opcode) { op->SetAttr<int64_t>(OP_EMUOP_PREFIX + "opc", opcode); }

int GetTensorDataGetIndex(const Operation* op)
{
    if (op->HasAttr(OP_EMUOP_PREFIX + "GetTensorData_index")) {
        int index = op->GetIntAttribute(OP_EMUOP_PREFIX + "GetTensorData_index");
        return index;
    } else {
        return -1;
    }
}

void GetTensorDataSetIndex(Operation* op, int index)
{
    op->SetAttr<int64_t>(OP_EMUOP_PREFIX + "GetTensorData_index", index);
}

int GetTensorDataGetCoaIndex(const Operation* op)
{
    if (op->HasAttr(OP_EMUOP_PREFIX + "GetTensorData_coaIndex")) {
        int index = op->GetIntAttribute(OP_EMUOP_PREFIX + "GetTensorData_coaIndex");
        return index;
    } else {
        return -1;
    }
}

void GetTensorDataSetCoaIndex(Operation* op, int index)
{
    op->SetAttr<int64_t>(OP_EMUOP_PREFIX + "GetTensorData_coaIndex", index);
}

Tensor TensorExtract(const Tensor& src, const std::vector<SymbolicScalar>& offset)
{
    FE_ASSERT(FeError::INVALID_VAL, src.GetShape().size() == offset.size())
        << "src.GetShape().size(): " << src.GetShape().size() << ", offset.size(): " << offset.size();
    auto currFunc = Program::GetInstance().GetCurrentFunction();

    Shape dstShape(src.GetShape().size(), 1);
    // minimal size is 32
    dstShape.back() = 32;
    Tensor dst(src.GetDataType(), dstShape, currFunc->GetRawName() + "_TensorExtract");

    Shape viewShape(src.GetShape().size(), 1);
    Tensor view = View(src, viewShape, offset);
    Operation& emuopView = **view.GetStorage()->GetProducers().begin();

    // Force to UB
    Tensor mark = Add(view, Element(view.GetDataType(), (int64_t)0));
    Operation& emuopMark = **mark.GetStorage()->GetProducers().begin();

    Offset assembleOffset(src.GetShape().size(), 0);
    Operation& emuopAssemble = currFunc->AddOperation(Opcode::OP_ASSEMBLE, {mark.GetStorage()}, {dst.GetStorage()});
    emuopAssemble.SetOpAttribute(std::make_shared<AssembleOpAttribute>(assembleOffset));

    SetEmuOpcode(&emuopView, EMUOP_TENSOR_EXTRACT);
    SetEmuOpcode(&emuopMark, EMUOP_TENSOR_EXTRACT);
    SetEmuOpcode(&emuopAssemble, EMUOP_TENSOR_EXTRACT);
    return dst;
}

void TensorInsert(const Tensor& src, const std::vector<SymbolicScalar>& offset, Tensor& dst)
{
    FE_ASSERT(FeError::INVALID_VAL, src.GetShape() == Shape(src.GetShape().size(), 1))
        << "src.GetShape(): " << src.GetShape()
        << ", Shape(src.GetShape().size(), 1): " << Shape(src.GetShape().size(), 1);
    FE_ASSERT(FeError::INVALID_VAL, src.GetShape().size() == dst.GetShape().size())
        << "src.GetShape().size(): " << src.GetShape().size() << ", dst.GetShape().size(): " << dst.GetShape().size();
    FE_ASSERT(FeError::INVALID_VAL, src.GetShape().size() == offset.size())
        << "src.GetShape().size(): " << src.GetShape().size() << ", offset.size(): " << offset.size();

    // Force to UB
    Tensor mark = Add(src, Element(src.GetDataType(), (int64_t)0));
    Operation& emuopMark = **mark.GetStorage()->GetProducers().begin();

    Assemble(mark, offset, dst);
    Operation& emuopAssemble = **mark.GetStorage()->GetConsumers().begin();

    emuopMark.SetAttribute(OP_EMUOP_PREFIX + "opc", EMUOP_TENSOR_INSERT);
    emuopAssemble.SetAttribute(OP_EMUOP_PREFIX + "opc", EMUOP_TENSOR_INSERT);
}

RawSymbolicScalarPtr ReplaceExpression(const RawSymbolicScalarPtr& expr, const RawSymbolicScalarPtr& src,
                                       const RawSymbolicScalarPtr& dst)
{
    if (expr == src) {
        return dst;
    }
    RawSymbolicScalarPtr result;
    switch (expr->Kind()) {
        case SymbolicScalarKind::T_SCALAR_SYMBOLIC_IMMEDIATE:
        case SymbolicScalarKind::T_SCALAR_SYMBOLIC_SYMBOL:
            result = expr;
            break;
        case SymbolicScalarKind::T_SCALAR_SYMBOLIC_EXPRESSION: {
            std::vector<RawSymbolicScalarPtr> subexprList;
            bool allreuse = true;
            for (auto& subexpr : expr->GetExpressionOperandList()) {
                RawSymbolicScalarPtr sub = ReplaceExpression(subexpr, src, dst);
                subexprList.push_back(sub);
                allreuse = allreuse && (sub == subexpr);
            }
            if (allreuse) {
                result = expr;
            } else {
                result = std::make_shared<RawSymbolicExpression>(expr->GetExpressionOpcode(), subexprList);
            }
        } break;
        default:
            FE_ASSERT(false) << "unexpected behavior.";
            break;
    }
    return result;
}

std::map<int, std::vector<RawSymbolicScalarPtr>> GetTensorDataDict(const RawSymbolicScalarPtr& dimOffset)
{
    std::map<int, std::vector<RawSymbolicScalarPtr>> getTensorDataDict;
    std::vector<RawSymbolicScalarPtr> mopCall = LookupExpressionByOpcode(dimOffset, SymbolicOpcode::T_MOP_CALL);
    for (auto mop : mopCall) {
        auto callee = mop->GetExpressionOperandList()[0];
        if (!callee->IsSymbol()) {
            continue;
        }
        auto name = callee->GetSymbolName();
        if (StringUtils::StartsWith(name, AddRuntimePrefix("GetTensorData"))) {
            auto getTensorDataIndex = mop->GetExpressionOperandList()[GET_TENSOR_DATA_OPERAND_INDEX_INDEX]
                                          ->GetImmediateValue();
            getTensorDataDict[getTensorDataIndex].push_back(mop);
        }
    }
    return getTensorDataDict;
}

std::map<int, std::vector<RawSymbolicScalarPtr>> GetTensorDataDict(const SymbolicScalar& dimOffset)
{
    return GetTensorDataDict(dimOffset.Raw());
}

std::map<int, std::vector<RawSymbolicScalarPtr>> GetTensorDataDict(const std::vector<SymbolicScalar>& offset)
{
    std::map<int, std::vector<RawSymbolicScalarPtr>> getTensorDataDict;
    for (auto& off : offset) {
        auto perOffsetDict = GetTensorDataDict(off);
        for (auto& [index, callList] : perOffsetDict) {
            for (auto& call : callList) {
                getTensorDataDict[index].push_back(call);
            }
        }
    }
    return getTensorDataDict;
}

std::map<int, std::vector<RawSymbolicScalarPtr>> GetTensorDataDict(
    const std::vector<std::reference_wrapper<SymbolicScalar>>& offset)
{
    std::map<int, std::vector<RawSymbolicScalarPtr>> getTensorDataDict;
    for (auto& off : offset) {
        auto perOffsetDict = GetTensorDataDict(off.get());
        for (auto& [index, callList] : perOffsetDict) {
            for (auto& call : callList) {
                getTensorDataDict[index].push_back(call);
            }
        }
    }
    return getTensorDataDict;
}

std::string GetTensorDataIODescDict::Dump() const
{
    std::ostringstream oss;
    for (auto [index, desc] : *this) {
        oss << index << ":"
            << "GetTensorDataIODesc(" << desc.ioType << "," << desc.ioTypeIndex << "," << desc.address.Dump() << ")\n";
    }
    return oss.str();
}

std::set<std::pair<int, int>> GetTensorDataUsage(const std::vector<std::reference_wrapper<SymbolicScalar>>& scalars)
{
    std::set<std::pair<int, int>> usage;
    for (auto& scalar : scalars) {
        auto mopcall = LookupExpressionByOpcode(scalar.get().Raw(), SymbolicOpcode::T_MOP_CALL);
        for (auto mop : mopcall) {
            auto callee = mop->GetExpressionOperandList()[0];
            if (!callee->IsSymbol()) {
                continue;
            }
            auto name = callee->GetSymbolName();
            if (StringUtils::StartsWith(name, AddRuntimePrefix("GetTensorData"))) {
                auto ioType = mop->GetExpressionOperandList()[GET_TENSOR_DATA_OPERAND_INDEX_IOTYPE]
                                  ->GetImmediateValue();
                auto ioIndex = mop->GetExpressionOperandList()[GET_TENSOR_DATA_OPERAND_INDEX_IOTYPE_INDEX]
                                   ->GetImmediateValue();
                usage.insert({ioType, ioIndex});
            }
        }
    }
    return usage;
}

SymbolicScalar UpdateGetTensorDataIOIndex(size_t currOutcastIdx, size_t newOutcastIdx, const SymbolicScalar& scalar)
{
    FE_ASSERT(currOutcastIdx != newOutcastIdx)
        << "currOutcastIdx == currOutcastIdx, should not be updated. Their value are " << currOutcastIdx;
    RawSymbolicScalarPtr curr = scalar.Raw();
    // when updating multilple outcastIdx, should ensure the currOutcastIdx of multiple calls is in ascending order
    bool filledFound = true;
    while (filledFound) {
        filledFound = false;
        for (auto [index, callList] : GetTensorDataDict(curr)) {
            (void)index;
            for (auto& call : callList) {
                std::vector<RawSymbolicScalarPtr> operandList = call->GetExpressionOperandList();
                auto currIOType = operandList[GET_TENSOR_DATA_OPERAND_INDEX_IOTYPE];
                auto currIOTypeIndex = operandList[GET_TENSOR_DATA_OPERAND_INDEX_IOTYPE_INDEX];
                FE_ASSERT(currIOType->IsImmediate()) << "its kind: " << SymbolicScalarKind2Name(currIOType->kind);
                FE_ASSERT(currIOTypeIndex->IsImmediate())
                    << "its kind: " << SymbolicScalarKind2Name(currIOTypeIndex->kind);
                if (currIOType->GetImmediateValue() != GET_TENSOR_DATA_OPERAND_IOTYPE_OUTCAST)
                    continue;
                size_t outcastIndex = currIOTypeIndex->GetImmediateValue();
                if (outcastIndex == newOutcastIdx || outcastIndex != currOutcastIdx)
                    continue;
                filledFound = true;
                operandList[GET_TENSOR_DATA_OPERAND_INDEX_IOTYPE_INDEX] = std::make_shared<RawSymbolicImmediate>(
                    newOutcastIdx);
                auto ptrNext = std::make_shared<RawSymbolicExpression>(call->GetExpressionOpcode(), operandList);
                auto currNext = ReplaceExpression(curr, call, ptrNext);
                curr = currNext;
                break;
            }
        }
    }
    return SymbolicScalar(curr);
}

SymbolicScalar GetTensorDataFillIO(const GetTensorDataIODescDict& iodescDict, const SymbolicScalar& dimOffset)
{
    RawSymbolicScalarPtr curr = dimOffset.Raw();
    bool filledFound = true;
    while (filledFound) {
        // There might be nesting GetTensorData call, so it's replaced iteratively.
        std::map<int, std::vector<RawSymbolicScalarPtr>> getDict = GetTensorDataDict(curr);
        filledFound = false;
        for (auto [index, callList] : getDict) {
            if (!iodescDict.count(index)) {
                continue;
            }
            // The same index always result in the same io type and io type index
            auto [ioTypeValue, ioTypeIndexValue, address] = iodescDict.find(index)->second;
            for (auto& call : callList) {
                std::vector<RawSymbolicScalarPtr> operandList = call->GetExpressionOperandList();
                auto currIOType = operandList[GET_TENSOR_DATA_OPERAND_INDEX_IOTYPE];
                auto currIOTypeIndex = operandList[GET_TENSOR_DATA_OPERAND_INDEX_IOTYPE_INDEX];
                FE_ASSERT(currIOType->IsImmediate()) << "its kind: " << SymbolicScalarKind2Name(currIOType->kind);
                FE_ASSERT(currIOTypeIndex->IsImmediate())
                    << "its kind: " << SymbolicScalarKind2Name(currIOTypeIndex->kind);
                if (currIOType->GetImmediateValue() == ioTypeValue &&
                    currIOTypeIndex->GetImmediateValue() == ioTypeIndexValue) {
                    continue;
                }
                operandList[GET_TENSOR_DATA_OPERAND_INDEX_IOTYPE] = std::make_shared<RawSymbolicImmediate>(ioTypeValue);
                operandList[GET_TENSOR_DATA_OPERAND_INDEX_IOTYPE_INDEX] = std::make_shared<RawSymbolicImmediate>(
                    ioTypeIndexValue);
                operandList[GET_TENSOR_DATA_OPERAND_INDEX_ADDRESS] = address.Raw();
                auto ptrNext = std::make_shared<RawSymbolicExpression>(call->GetExpressionOpcode(), operandList);
                auto currNext = ReplaceExpression(curr, call, ptrNext);
                curr = currNext;
                filledFound = true;
                break;
            }
        }
    }
    return SymbolicScalar(curr);
}

bool TypeEqual(const LogicalTensorPtr a, const LogicalTensorPtr b)
{
    auto ra = a->tensor;
    auto rb = b->tensor;
    if (ra->datatype != rb->datatype || ra->format != rb->format || ra->rawshape != rb->rawshape) {
        return false;
    }
    if (a->shape != b->shape) {
        return false;
    }
    return true;
}

} // namespace npu::tile_fwk
