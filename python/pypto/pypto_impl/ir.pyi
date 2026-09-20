# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Final, Union, Any, Optional, Sequence, overload
import enum
from .. import pypto_impl
from .. import SymbolicScalar


class DataType:
    """Data type representation for PyPTO tensors and operations"""

    # Static type constants
    BOOL: DataType  # Boolean (true/false)
    INT4: DataType  # 4-bit signed integer
    INT8: DataType  # 8-bit signed integer
    INT16: DataType  # 16-bit signed integer
    INT32: DataType  # 32-bit signed integer
    INT64: DataType  # 64-bit signed integer
    UINT4: DataType  # 4-bit unsigned integer
    UINT8: DataType  # 8-bit unsigned integer
    UINT16: DataType  # 16-bit unsigned integer
    UINT32: DataType  # 32-bit unsigned integer
    UINT64: DataType  # 64-bit unsigned integer
    FP4: DataType  # 4-bit floating point
    FP8E4M3FN: DataType  # 8-bit floating point (IEEE 754 e4m3fn format)
    FP8E5M2: DataType  # 8-bit floating point (IEEE 754 e5m2 format)
    FP8E8M0: DataType  # 8-bit floating point (8-bit exponent, 0-bit mantissa)
    FP4E2M1: DataType  # 4-bit floating point (2-bit exponent, 1-bit mantissa)
    FP4E1M2: DataType  # 4-bit floating point (1-bit exponent, 2-bit mantissa)
    FP16: DataType  # 16-bit floating point (IEEE 754 half precision)
    FP32: DataType  # 32-bit floating point (IEEE 754 single precision)
    FP64: DataType  # 64-bit floating point (IEEE 754 double precision)
    BF16: DataType  # 16-bit brain floating point
    FP64: DataType  # 64-bit floating point (IEEE 754 double precision)
    HF4: DataType  # 4-bit Hisilicon float
    HF8: DataType  # 8-bit Hisilicon float
    INDEX: DataType  # Machine-word integer for index computations (loop vars, dims, valid shapes)

    def bits(self) -> int:
        """
        Get the size in bits of this data type. Returns the actual bit size for sub-byte types
        (e.g., 4 bits for INT4, 8 bits for INT8, etc.).

        Returns:
            The size in bits of the data type
        """

    def to_c_type_string(self) -> str:
        """
        Get the C-style type string.

        Returns:
            The C-style type string
        """

    def is_float(self) -> bool:
        """
        Check if this data type is a floating point type (FP4, FP4E2M1, FP4E1M2, FP8, FP8E4M3FN, FP8E5M2, FP8E8M0, FP16, FP32, BF16, HF4, HF8).

        Returns:
            True if the data type is a floating point type, False otherwise
        """

    def is_signed(self) -> bool:
        """
        Check if this data type is a signed integer type (INT4, INT8, INT16, INT32, INT64).

        Returns:
            True if the data type is a signed integer type, False otherwise
        """

    def is_unsigned(self) -> bool:
        """
        Check if this data type is an unsigned integer type (UINT4, UINT8, UINT16, UINT32, UINT64).

        Returns:
            True if the data type is an unsigned integer type, False otherwise
        """

    def is_int(self) -> bool:
        """
        Check if this data type is any integer type (signed or unsigned).

        Returns:
            True if the data type is any integer type, False otherwise
        """

    def __int__(self) -> int:
        """
        Get the underlying type code as uint8_t.

        Returns:
            The type code as an integer
        """

    def __eq__(self, other: DataType) -> bool:
        """Equality comparison operator"""

    def __ne__(self, other: DataType) -> bool:
        """Inequality comparison operator"""

    def __repr__(self) -> str:
        """String representation for debugging"""

    def __str__(self) -> str:
        """String representation for printing"""


class Span:
    """Source location information tracking file, line, and column positions."""

    filename: Final[str]
    """Source filename."""

    begin_line: Final[int]
    """Beginning line (1-indexed)."""

    begin_column: Final[int]
    """Beginning column (1-indexed)."""

    end_line: Final[int]
    """Ending line (1-indexed)."""

    end_column: Final[int]
    """Ending column (1-indexed)."""

    def __init__(
        self,
        filename: str,
        begin_line: int,
        begin_column: int,
        end_line: int = -1,
        end_column: int = -1,
    ) -> None:
        """Create a source span.

        Args:
            filename: Source filename
            begin_line: Beginning line (1-indexed)
            begin_column: Beginning column (1-indexed)
            end_line: Ending line (1-indexed, -1 means unknown)
            end_column: Ending column (1-indexed, -1 means unknown)
        """

    def is_unknown(self) -> bool:
        """Check if the span is unknown.

        Returns:
            True if the span is unknown, False otherwise
        """

    @staticmethod
    def unknown() -> Span:
        """Create an unknown/invalid span for cases where source location is unavailable.

        Returns:
            Span with empty filename and invalid coordinates
        """

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...


class IRNode:
    """Base class for all IR nodes."""

    span: Final[Span]
    """Source location of this IR node."""


class Expr(IRNode):
    """Base class for all expressions."""

    type: Final[Type]
    """Type of the expression result."""

    # Binary operators (only work with ScalarType)
    def __add__(self, other: ExprType) -> Expr:
        """Addition operator (self + other). Only works with ScalarType variables."""

    def __sub__(self, other: ExprType) -> Expr:
        """Subtraction operator (self - other). Only works with ScalarType variables."""

    def __mul__(self, other: ExprType) -> Expr:
        """Multiplication operator (self * other). Only works with ScalarType variables."""

    def __truediv__(self, other: ExprType) -> Expr:
        """Division operator (self / other). Only works with ScalarType variables."""

    def __floordiv__(self, other: ExprType) -> Expr:
        """Floor division operator (self // other). Only works with ScalarType variables."""

    def __mod__(self, other: ExprType) -> Expr:
        """Modulo operator (self % other). Only works with ScalarType variables."""

    def __pow__(self, other: ExprType) -> Expr:
        """Power operator (self ** other). Only works with ScalarType variables."""

    # Comparison operators (only work with ScalarType)
    def __eq__(self, other: ExprType) -> Expr:  # type: ignore[override]
        """Equality operator (self == other). Only works with ScalarType variables."""

    def __ne__(self, other: ExprType) -> Expr:  # type: ignore[override]
        """Inequality operator (self != other). Only works with ScalarType variables."""

    def __lt__(self, other: ExprType) -> Expr:
        """Less than operator (self < other). Only works with ScalarType variables."""

    def __le__(self, other: ExprType) -> Expr:
        """Less than or equal operator (self <= other). Only works with ScalarType variables."""

    def __gt__(self, other: ExprType) -> Expr:
        """Greater than operator (self > other). Only works with ScalarType variables."""

    def __ge__(self, other: ExprType) -> Expr:
        """Greater than or equal operator (self >= other). Only works with ScalarType variables."""

    # Bitwise operators (only work with ScalarType)
    def __and__(self, other: ExprType) -> Expr:
        """Bitwise and operator (self & other). Only works with ScalarType variables."""

    def __or__(self, other: ExprType) -> Expr:
        """Bitwise or operator (self | other). Only works with ScalarType variables."""

    def __xor__(self, other: ExprType) -> Expr:
        """Bitwise xor operator (self ^ other). Only works with ScalarType variables."""

    def __lshift__(self, other: ExprType) -> Expr:
        """Bitwise left shift operator (self << other). Only works with ScalarType variables."""

    def __rshift__(self, other: ExprType) -> Expr:
        """Bitwise right shift operator (self >> other). Only works with ScalarType variables."""

    # Unary operators (only work with ScalarType)
    def __neg__(self) -> Expr:
        """Negation operator (-self). Only works with ScalarType variables."""

    def __invert__(self) -> Expr:
        """Bitwise not operator (~self). Only works with ScalarType variables."""

    # Reverse operators (only work with ScalarType)
    def __radd__(self, other: ExprType) -> Expr:
        """Reverse addition operator (other + self). Only works with ScalarType variables."""

    def __rsub__(self, other: ExprType) -> Expr:
        """Reverse subtraction operator (other - self). Only works with ScalarType variables."""

    def __rmul__(self, other: ExprType) -> Expr:
        """Reverse multiplication operator (other * self). Only works with ScalarType variables."""

    def __rtruediv__(self, other: ExprType) -> Expr:
        """Reverse division operator (other / self). Only works with ScalarType variables."""

    def __rfloordiv__(self, other: ExprType) -> Expr:
        """Reverse floor division operator (other // self). Only works with ScalarType variables."""

    def __rmod__(self, other: ExprType) -> Expr:
        """Reverse modulo operator (other % self). Only works with ScalarType variables."""

    def __rpow__(self, other: ExprType) -> Expr:
        """Reverse power operator (other ** self). Only works with ScalarType variables."""

    def __rand__(self, other: ExprType) -> Expr:
        """Reverse bitwise and operator (other & self). Only works with ScalarType variables."""

    def __ror__(self, other: ExprType) -> Expr:
        """Reverse bitwise or operator (other | self). Only works with ScalarType variables."""

    def __rxor__(self, other: ExprType) -> Expr:
        """Reverse bitwise xor operator (other ^ self). Only works with ScalarType variables."""

    def __rlshift__(self, other: ExprType) -> Expr:
        """Reverse bitwise left shift operator (other << self). Only works with ScalarType variables."""

    def __rrshift__(self, other: ExprType) -> Expr:
        """Reverse bitwise right shift operator (other >> self). Only works with ScalarType variables."""


ExprType = Expr | int | float


class BinaryExpr(Expr):
    """Base class for binary operations."""

    dtype: Final[DataType]
    """Data type of the expression."""

    left: Final[Expr]
    """Left operand."""

    right: Final[Expr]
    """Right operand."""


class UnaryExpr(Expr):
    """Base class for unary operations."""

    dtype: Final[DataType]
    """Data type of the expression."""

    operand: Final[Expr]
    """Operand."""


class Add(BinaryExpr):
    """Addition expression (left + right)."""

    def __init__(self, left: Expr, right: Expr, dtype: DataType, span: Span) -> None:
        """Create an addition expression.

        Args:
            left: Left operand
            right: Right operand
            dtype: Data type
            span: Source location
        """


class Sub(BinaryExpr):
    """Subtraction expression (left - right)."""

    def __init__(self, left: Expr, right: Expr, dtype: DataType, span: Span) -> None:
        """Create a subtraction expression.

        Args:
            left: Left operand
            right: Right operand
            dtype: Data type
            span: Source location
        """


class Mul(BinaryExpr):
    """Multiplication expression (left * right)."""

    def __init__(self, left: Expr, right: Expr, dtype: DataType, span: Span) -> None:
        """Create a multiplication expression.

        Args:
            left: Left operand
            right: Right operand
            dtype: Data type
            span: Source location
        """


class FloorDiv(BinaryExpr):
    """Floor division expression (left // right)."""

    def __init__(self, left: Expr, right: Expr, dtype: DataType, span: Span) -> None:
        """Create a floor division expression.

        Args:
            left: Left operand
            right: Right operand
            dtype: Data type
            span: Source location
        """


class FloorMod(BinaryExpr):
    """Floor modulo expression (left % right)."""

    def __init__(self, left: Expr, right: Expr, dtype: DataType, span: Span) -> None:
        """Create a floor modulo expression.

        Args:
            left: Left operand
            right: Right operand
            dtype: Data type
            span: Source location
        """


class FloatDiv(BinaryExpr):
    """Float division expression (left / right)."""

    def __init__(self, left: Expr, right: Expr, dtype: DataType, span: Span) -> None:
        """Create a float division expression.

        Args:
            left: Left operand
            right: Right operand
            dtype: Data type
            span: Source location
        """


class Min(BinaryExpr):
    """Minimum expression (min(left, right))."""

    def __init__(self, left: Expr, right: Expr, dtype: DataType, span: Span) -> None:
        """Create a minimum expression.

        Args:
            left: Left operand
            right: Right operand
            dtype: Data type
            span: Source location
        """


class Max(BinaryExpr):
    """Maximum expression (max(left, right))."""

    def __init__(self, left: Expr, right: Expr, dtype: DataType, span: Span) -> None:
        """Create a maximum expression.

        Args:
            left: Left operand
            right: Right operand
            dtype: Data type
            span: Source location
        """


class Pow(BinaryExpr):
    """Power expression (left ** right)."""

    def __init__(self, left: Expr, right: Expr, dtype: DataType, span: Span) -> None:
        """Create a power expression.

        Args:
            left: Left operand
            right: Right operand
            dtype: Data type
            span: Source location
        """


class Eq(BinaryExpr):
    """Equality expression (left == right)."""

    def __init__(self, left: Expr, right: Expr, dtype: DataType, span: Span) -> None:
        """Create an equality expression.

        Args:
            left: Left operand
            right: Right operand
            dtype: Data type
            span: Source location
        """


class Ne(BinaryExpr):
    """Inequality expression (left != right)."""

    def __init__(self, left: Expr, right: Expr, dtype: DataType, span: Span) -> None:
        """Create an inequality expression.

        Args:
            left: Left operand
            right: Right operand
            dtype: Data type
            span: Source location
        """


class Lt(BinaryExpr):
    """Less than expression (left < right)."""

    def __init__(self, left: Expr, right: Expr, dtype: DataType, span: Span) -> None:
        """Create a less than expression.

        Args:
            left: Left operand
            right: Right operand
            dtype: Data type
            span: Source location
        """


class Le(BinaryExpr):
    """Less than or equal to expression (left <= right)."""

    def __init__(self, left: Expr, right: Expr, dtype: DataType, span: Span) -> None:
        """Create a less than or equal to expression.

        Args:
            left: Left operand
            right: Right operand
            dtype: Data type
            span: Source location
        """


class Gt(BinaryExpr):
    """Greater than expression (left > right)."""

    def __init__(self, left: Expr, right: Expr, dtype: DataType, span: Span) -> None:
        """Create a greater than expression.

        Args:
            left: Left operand
            right: Right operand
            dtype: Data type
            span: Source location
        """


class Ge(BinaryExpr):
    """Greater than or equal to expression (left >= right)."""

    def __init__(self, left: Expr, right: Expr, dtype: DataType, span: Span) -> None:
        """Create a greater than or equal to expression.

        Args:
            left: Left operand
            right: Right operand
            dtype: Data type
            span: Source location
        """


class And(BinaryExpr):
    """Logical and expression (left and right)."""

    def __init__(self, left: Expr, right: Expr, dtype: DataType, span: Span) -> None:
        """Create a logical and expression.

        Args:
            left: Left operand
            right: Right operand
            dtype: Data type
            span: Source location
        """


class Or(BinaryExpr):
    """Logical or expression (left or right)."""

    def __init__(self, left: Expr, right: Expr, dtype: DataType, span: Span) -> None:
        """Create a logical or expression.

        Args:
            left: Left operand
            right: Right operand
            dtype: Data type
            span: Source location
        """


class Xor(BinaryExpr):
    """Logical xor expression (left xor right)."""

    def __init__(self, left: Expr, right: Expr, dtype: DataType, span: Span) -> None:
        """Create a logical xor expression.

        Args:
            left: Left operand
            right: Right operand
            dtype: Data type
            span: Source location
        """


class BitAnd(BinaryExpr):
    """Bitwise and expression (left & right)."""

    def __init__(self, left: Expr, right: Expr, dtype: DataType, span: Span) -> None:
        """Create a bitwise and expression.

        Args:
            left: Left operand
            right: Right operand
            dtype: Data type
            span: Source location
        """


class BitOr(BinaryExpr):
    """Bitwise or expression (left | right)."""

    def __init__(self, left: Expr, right: Expr, dtype: DataType, span: Span) -> None:
        """Create a bitwise or expression.

        Args:
            left: Left operand
            right: Right operand
            dtype: Data type
            span: Source location
        """


class BitXor(BinaryExpr):
    """Bitwise xor expression (left ^ right)."""

    def __init__(self, left: Expr, right: Expr, dtype: DataType, span: Span) -> None:
        """Create a bitwise xor expression.

        Args:
            left: Left operand
            right: Right operand
            dtype: Data type
            span: Source location
        """


class BitShiftLeft(BinaryExpr):
    """Bitwise left shift expression (left << right)."""

    def __init__(self, left: Expr, right: Expr, dtype: DataType, span: Span) -> None:
        """Create a bitwise left shift expression.

        Args:
            left: Left operand
            right: Right operand
            dtype: Data type
            span: Source location
        """


class BitShiftRight(BinaryExpr):
    """Bitwise right shift expression (left >> right)."""

    def __init__(self, left: Expr, right: Expr, dtype: DataType, span: Span) -> None:
        """Create a bitwise right shift expression.

        Args:
            left: Left operand
            right: Right operand
            dtype: Data type
            span: Source location
        """


class Abs(UnaryExpr):
    """Absolute value expression (abs(operand))."""

    def __init__(self, operand: Expr, dtype: DataType, span: Span) -> None:
        """Create an absolute value expression.

        Args:
            operand: Operand expression
            dtype: Data type
            span: Source location
        """


class Neg(UnaryExpr):
    """Negation expression (-operand)."""

    def __init__(self, operand: Expr, dtype: DataType, span: Span) -> None:
        """Create a negation expression.

        Args:
            operand: Operand expression
            dtype: Data type
            span: Source location
        """


class Not(UnaryExpr):
    """Logical not expression (not operand)."""

    def __init__(self, operand: Expr, dtype: DataType, span: Span) -> None:
        """Create a logical not expression.

        Args:
            operand: Operand expression
            dtype: Data type
            span: Source location
        """


class BitNot(UnaryExpr):
    """Bitwise not expression (~operand)."""

    def __init__(self, operand: Expr, dtype: DataType, span: Span) -> None:
        """Create a bitwise not expression.

        Args:
            operand: Operand expression
            dtype: Data type
            span: Source location
        """


class Cast(UnaryExpr):
    """Cast expression (cast operand to dtype)."""

    def __init__(self, operand: Expr, dtype: DataType, span: Span) -> None:
        """Create a cast expression.

        Args:
            operand: Operand expression
            dtype: Target data type
            span: Source location
        """


class Type:
    """Base class for type representations."""


class UnknownType(Type):
    """Unknown or unspecified type representation.

    Used as the default type for expressions when type information is not available.
    """

    def __init__(self) -> None:
        """Create an unknown type."""

    @staticmethod
    def get() -> UnknownType:
        """Get the singleton UnknownType instance.

        Returns:
            The singleton UnknownType instance
        """


class NoneType(Type):
    """None (void-like) type representation.

    Represents the absence of a value.
    """

    def __init__(self) -> None:
        """Create a none type."""

    @staticmethod
    def get() -> NoneType:
        """Get the singleton NoneType instance.

        Returns:
            The singleton NoneType instance
        """


class ScalarType(Type):
    """Scalar type representation."""

    dtype: Final[DataType]
    """Data type."""

    def __init__(self, dtype: DataType) -> None:
        """Create a scalar type.

        Args:
            dtype: Data type
        """


class ShapedType(Type):
    """Base class for shaped types (tensors and tiles)."""

    dtype: Final[DataType]
    """Element data type."""

    shape: Final[Sequence[Expr]]
    """Shape dimensions."""

    memref: Final[MemRef | None]
    """Optional memory reference."""

    @property
    def memory_space(self) -> MemorySpace | None:
        """Canonical memory space for this shaped type."""
        ...

    def shares_memref_with(self, other: ShapedType) -> bool:
        """Check if this ShapedType shares the same MemRef object with another ShapedType.

        Args:
            other: Another ShapedType to compare with

        Returns:
            True if both have MemRef and they point to the same object, False otherwise
        """
        ...


class MemorySpace(enum.IntEnum):
    """Memory space enumeration."""

    DDR = ...
    """DDR memory (off-chip)."""

    Vec = ...
    """Vector/unified buffer (on-chip)."""

    Mat = ...
    """Matrix/L1 buffer."""

    Left = ...
    """Left matrix operand buffer."""

    Right = ...
    """Right matrix operand buffer."""

    Scaling = ...
    """Scaling/FBuffer tile buffer."""

    Acc = ...
    """Accumulator buffer."""

    Bias = ...
    """Bias buffer."""

    ScaleLeft = ...
    """MX scale buffer in L0A."""

    ScaleRight = ...
    """MX scale buffer in L0B."""


class TensorLayout(enum.IntEnum):
    """Tensor layout type enumeration."""

    ND = ...
    """ND layout."""

    DN = ...
    """DN layout."""

    NZ = ...
    """NZ layout."""


class PipeType(enum.IntEnum):
    """Pipeline type enumeration for hardware execution units."""

    MTE1 = ...
    MTE2 = ...
    MTE3 = ...
    M = ...
    V = ...
    S = ...
    FIX = ...
    ALL = ...


class CoreType(enum.IntEnum):
    """Core type enumeration."""

    VECTOR = ...
    CUBE = ...


class FunctionType(enum.IntEnum):
    """Function type classification.

    Categorizes functions by their execution context and purpose:
    - Opaque: Unspecified (default)
    - Orchestration: Host/AICPU control and coordination
    - InCore: AICore sub-graph execution
    - Helper: Scalar helper function callable from kernels
    - SimtVF: A5 SIMT vector function launched from an AIV kernel
    - SimtCallee: A5 helper callable from SIMT functions
    """

    Opaque = ...
    Orchestration = ...
    InCore = ...
    Helper = ...
    SimtVF = ...
    SimtCallee = ...


class PtrType(Type):
    """Pointer type with element dtype."""

    dtype: Final[DataType]
    """Element data type pointed to."""

    def __init__(self, dtype: DataType = DataType.INT8) -> None: ...


class TokenKind(enum.IntEnum):
    """Token semantic kind."""

    NORMAL = ...
    READ = ...
    WRITE = ...


class TokenType(Type):
    """Opaque token type for side-effect ordering."""

    kind: Final[TokenKind]

    def __init__(self, kind: TokenKind = TokenKind.NORMAL) -> None: ...

    @staticmethod
    def get(kind: TokenKind = TokenKind.NORMAL) -> TokenType:
        """Get the singleton TokenType instance."""
        ...


class LogicalTensorType(Type):
    """Logical tensor type"""

    def __init__(self) -> None: ...


class Var(Expr):
    """Variable reference expression."""

    name: Final[str]
    """Variable name (cosmetic label, not an identifier)."""

    def __init__(self, name: str, type: Optional[Type], span: Span) -> None:
        """Create a variable reference.

        Args:
            name: Variable name (cosmetic label, not an identifier)
            type: Type of the variable (ScalarType, TensorType, or TileType)
                  Memory reference information is stored in ShapedType for Tensor/Tile types
            span: Source location
        """

    def __str__(self) -> str:
        """String representation of the variable."""

    def __repr__(self) -> str:
        """Detailed representation of the variable."""


class MemRef(Expr):
    """Memory reference variable for shaped types (inherits from Expr)."""

    memory_space: MemorySpace
    """Base Ptr variable (allocation identity token)."""

    addr: Expr
    """Starting address expression."""

    size: int
    """Size in bytes (64-bit unsigned)."""

    def __init__(
        self, memory_space: MemorySpace, addr: Expr | int, size: int, span: Optional[Span] = None
    ) -> None:
        """Create a memory reference.

        New API: MemRef(memory_space, addr, size)
        """

    @staticmethod
    def same_allocation(a: MemRef, b: MemRef) -> bool:
        """Check if two MemRefs share the same allocation (same base_ Ptr)."""
        ...

    @staticmethod
    def may_alias(a: MemRef, b: MemRef) -> bool:
        """Check if two MemRefs may alias (same base + overlapping byte ranges)."""
        ...


class IterArg:
    """Iteration argument variable."""

    initValue: Final[Expr]
    """Initial value expression (can be any Expr)."""

    iterVar: Final[Var]
    """Iteration argument variable."""

    @overload
    def __init__(self, name: str, type: Type, initValue: Expr, span: Span) -> None:
        """Create an iteration argument.

        Args:
            name: Variable name (cosmetic label, not an identifier)
            type: Type of the variable (ScalarType, TensorType, or TileType)
                  Memory reference information is stored in ShapedType for Tensor/Tile types
            initValue: Initial value expression (can be any Expr)
            span: Source location
        """

    @overload
    def __init__(self, iterVar: Var, initValue: Expr) -> None:
        """Create an iteration argument.

        Args:
            iterVar: Iteration argument variable
            initValue: Initial value expression (can be any Expr)
        """

    def __str__(self) -> str:
        """String representation of the iteration argument."""

    def __repr__(self) -> str:
        """Detailed representation of the iteration argument."""


class ConstInt(Expr):
    """Constant integer expression."""

    value: Final[int]
    """Constant integer value."""

    def __init__(self, value: int, dtype: DataType, span: Span) -> None:
        """Create a constant integer expression.

        Args:
            value: Integer value
            dtype: Data type
            span: Source location
        """

    @property
    def dtype(self) -> DataType:
        """Data type of the expression."""


class ConstFloat(Expr):
    """Constant floating-point expression."""

    value: Final[float]
    """Constant floating-point value."""

    def __init__(self, value: float, dtype: DataType, span: Span) -> None:
        """Create a constant floating-point expression.

        Args:
            value: Floating-point value
            dtype: Data type
            span: Source location
        """

    @property
    def dtype(self) -> DataType:
        """Data type of the expression."""


class ConstBool(Expr):
    """Constant boolean expression."""

    value: Final[bool]
    """Constant boolean value."""

    def __init__(self, value: bool, span: Span) -> None:
        """Create a constant boolean expression.

        Args:
            value: Boolean value
            span: Source location

        Note:
            dtype is always DataType.BOOL - no need to specify.
        """

    @property
    def dtype(self) -> DataType:
        """Data type of the expression (always DataType.BOOL)."""


class Call(Expr):
    """Function call expression."""

    name: Final[str]
    """Operation/function."""

    args: Final[Sequence[Expr]]
    """Positional arguments."""

    def __init__(self, op: str, args: Sequence[Expr], span: Span) -> None:
        """Create a function call expression.

        Args:
            op: Operation/function to call
            args: List of argument expressions
            kwargs: Keyword arguments (metadata)
            span: Source location
        """
        ...

    def __str__(self) -> str:
        """String representation of the call expression."""

    def __repr__(self) -> str:
        """Detailed representation of the call expression."""


class TensorType(Type):
    """Tensor type representation."""

    dtype: Final[DataType]
    """Element data type."""

    shape: Final[Sequence[Expr]]
    """Shape dimensions as Expr nodes."""

    memref: Optional[MemRef]
    """Optional memory reference."""

    @overload
    def __init__(self, shape: Sequence[Expr], dtype: DataType, memref: Optional[MemRef] = None) -> None:
        """Create a tensor type with memory reference.

        Args:
            shape: Shape dimensions as Expr nodes
            dtype: Element data type
            memref: Optional memory reference
        """

    @overload
    def __init__(self, shape: Sequence[int], dtype: DataType, memref: Optional[MemRef] = None) -> None:
        """Create a tensor type with memory reference.

        Args:
            shape: Shape dimensions as Expr nodes
            dtype: Element data type
            memref: Optional memory reference
        """


class TupleType(Type):
    """Tuple type representation (contains multiple types)."""

    types: Final[Sequence[Type]]
    """Types in the tuple."""

    def __init__(self, types: Sequence[Type]) -> None:
        """Create a tuple type from a list of types.

        Args:
            types: List of types in the tuple
        """


class TupleTypeKind(enum.IntEnum):
    """Semantic classification for a TupleType."""

    TUPLE = ...
    NAMED_TUPLE = ...
    STRUCT = ...


class TupleTypeInfo:
    """Semantic metadata associated with one exact TupleType."""

    kind: Final[TupleTypeKind]
    name: Final[str | None]
    fields: Final[list[str]]

    def __init__(
        self,
        kind: TupleTypeKind,
        name: str | None = None,
        fields: Sequence[str] = (),
    ) -> None: ...

    def __eq__(self, other: TupleTypeInfo) -> bool: ...

    def __ne__(self, other: TupleTypeInfo) -> bool: ...


class IRDebugInfo:
    """Program-owned side table for semantic tuple metadata."""

    def __init__(self) -> None: ...

    def register_tuple_type_info(self, type: TupleType, info: TupleTypeInfo) -> None: ...

    def get_tuple_type_info(self, type: TupleType) -> TupleTypeInfo | None: ...


class MakeTuple(Expr):
    """Tuple construction expression."""

    elements: Final[Sequence[Expr]]
    """Elements of the tuple."""

    def __init__(self, elements: Sequence[Expr], span: Span) -> None:
        """Create a tuple construction expression.

        Args:
            elements: Expressions to be tuple elements
            span: Source location

        The result type is automatically set to TupleType containing
        the types of all input expressions.
        """

    def __str__(self) -> str:
        """String representation of the tuple construction expression."""

    def __repr__(self) -> str:
        """Detailed representation of the tuple construction expression."""


class GetItemExpr(Expr):
    """Subscript expression for tuple element access or tile offset."""

    value: Final[Expr]
    """Base expression (must have TupleType or TileType)."""

    slice: Final[Expr]
    """Subscript expression."""

    def __init__(self, value: Expr, slice: Expr, span: Span) -> None:
        """Create a subscript expression.

        Args:
            value: Base expression
            slice: Subscript expression
            span: Source location
        """

    def __str__(self) -> str:
        """String representation of the tuple access expression."""

    def __repr__(self) -> str:
        """Detailed representation of the tuple access expression."""


class Stmt(IRNode):
    """Base class for all statements."""

    def __init__(self, span: Span) -> None:
        """Create a statement.

        Args:
            span: Source location
        """


class AssignStmt(Stmt):
    """Assignment statement: var = value."""

    var: Final[Var]
    """Variable."""

    value: Final[Expr]
    """Expression."""

    def __init__(self, var: Var, value: Expr, span: Span) -> None:
        """Create an assignment statement.

        Args:
            var: Variable
            value: Expression
            span: Source location
        """


class IfStmt(Stmt):
    """Conditional statement: if condition then then_body else else_body."""

    condition: Final[Expr]
    """Condition expression."""

    then_body: Final[SeqStmts]
    """Then branch statement."""

    else_body: Final[SeqStmts]
    """Else branch statement (can be None)."""

    return_vars: Final[list[Var]]
    """Return variables (can be empty)."""

    def __init__(
        self,
        condition: Expr,
        then_body: Stmt,
        else_body: Stmt | None,
        return_vars: list[Var],
        span: Span,
    ) -> None:
        """Create a conditional statement with then and else branches.

        Args:
            condition: Condition expression
            then_body: Then branch statement
            else_body: Else branch statement (can be None)
            return_vars: Return variables (can be empty)
            span: Source location
        """
        ...


class YieldStmt(Stmt):
    """Yield statement: yield value."""

    value: Final[list[Expr]]
    """List of variables to yield (can be empty)."""

    @overload
    def __init__(self, value: list[Expr], span: Span) -> None:
        """Create a yield statement with a list of variables.

        Args:
            value: List of variables to yield
            span: Source location
        """
        ...

    @overload
    def __init__(self, span: Span) -> None:
        """Create a yield statement without values.

        Args:
            span: Source location
        """
        ...


class ReturnStmt(Stmt):
    """Return statement: return value."""

    value: Final[list[Expr]]
    """List of expressions to return (can be empty)."""

    @overload
    def __init__(self, value: list[Expr], span: Span) -> None:
        """Create a return statement with a list of expressions.

        Args:
            value: List of expressions to return
            span: Source location
        """
        ...

    @overload
    def __init__(self, span: Span) -> None:
        """Create a return statement without values.

        Args:
            span: Source location
        """
        ...


class ForStmt(Stmt):
    """For loop statement: for loop_var in range(start, stop, step): body."""

    loop_var: Final[Var]
    """Loop variable."""

    start: Final[Expr]
    """Start value expression."""

    stop: Final[Expr]
    """Stop value expression."""

    step: Final[Expr]
    """Step value expression."""

    iter_args: Final[list[IterArg]]
    """Iteration arguments (can be empty)."""

    body: Final[SeqStmts]
    """Loop body statement."""

    return_vars: Final[list[Var]]
    """Return variables (can be empty)."""

    attrs: Final[dict[str, object]]
    """Loop-level attributes (key-value metadata)."""

    def __init__(
        self,
        loop_var: Var,
        start: Expr,
        stop: Expr,
        step: Expr,
        iter_args: list[IterArg],
        body: Stmt,
        return_vars: list[Var],
        span: Span,
        attrs: dict[str, object] | None = None,
    ) -> None:
        """Create a for loop statement.

        Args:
            loop_var: Loop variable
            start: Start value expression
            stop: Stop value expression
            step: Step value expression
            iter_args: Iteration arguments (can be empty)
            body: Loop body statements
            return_vars: Return variables (can be empty)
            span: Source location
        """


class WhileStmt(Stmt):
    """While loop statement: while condition: body."""

    condition: Final[Expr]
    """Condition expression."""

    iter_args: Final[list[IterArg]]
    """Iteration arguments (can be empty)."""

    body: Final[SeqStmts]
    """Loop body statement."""

    return_vars: Final[list[Var]]
    """Return variables (can be empty)."""

    def __init__(
        self,
        condition: Expr,
        iter_args: list[IterArg],
        body: Stmt,
        return_vars: list[Var],
        span: Span,
    ) -> None:
        """Create a while loop statement.

        Args:
            condition: Condition expression
            iter_args: Iteration arguments (can be empty)
            body: Loop body statement
            return_vars: Return variables (can be empty)
            span: Source location
        """


class SectionKind(enum.IntEnum):
    """Section kind classification."""

    Vector = ...
    Cube = ...
    VF = ...


class SectionStmt(Stmt):
    """Section statement: with section_vector/section_cube."""

    section_kind: Final[SectionKind]
    """Section kind."""

    body: Final[Stmt]
    """Nested statement body."""

    def __init__(self, section_kind: SectionKind, body: Stmt, span: Span) -> None:
        """Create a section statement.

        Args:
            section_kind: Section kind
            body: Nested statement body
            span: Source location
        """


class SeqStmts(Stmt):
    """Sequence of statements: a sequence of statements."""

    stmts: list[Stmt]
    """List of statements."""

    @overload
    def __init__(self, stmts: list[Stmt], span: Span) -> None:
        """Create a sequence of statements.

        Args:
            stmts: List of statements
            span: Source location
        """

    @overload
    def __init__(self, span: Span) -> None:
        """Create an empty sequence of statements.

        Args:
            span: Source location
        """

    def __getitem__(self, index: int) -> Stmt:
        """Get statement by index, supports negative indexing.

        Args:
            index: Statement index (negative indices count from end)

        Returns:
            Statement at the given index

        Raises:
            IndexError: If index is out of range
        """


class EvalStmt(Stmt):
    """Evaluation statement: expr."""

    expr: Final[Expr]
    """Expression."""

    def __init__(self, expr: Expr, span: Span) -> None:
        """Create an evaluation statement.

        Args:
            expr: Expression to execute
            span: Source location
        """


class BreakStmt(Stmt):
    """Break statement: break."""

    value: Final[list[Expr]]
    """Value to break."""

    @overload
    def __init__(self, span: Span) -> None:
        """Create a break statement.

        Args:
            span: Source location
        """

    @overload
    def __init__(self, operands: list[Expr], span: Span) -> None:
        """Create a break statement with operands.

        Args:
            operands: Operands to break
            span: Source location
        """


class ContinueStmt(Stmt):
    """Continue statement: continue."""

    value: Final[list[Expr]]
    """Value to continue."""

    @overload
    def __init__(self, span: Span) -> None:
        """Create a continue statement.

        Args:
            span: Source location
        """

    @overload
    def __init__(self, operands: list[Expr], span: Span) -> None:
        """Create a continue statement.

        Args:
            operands: Operands to continue
            span: Source location
        """


class ScalarOpStmt(Stmt):
    """Scalar operation statement"""

    result: Final[Var]
    """Result expression."""

    result_token: Final[Var]
    """Second operand."""

    opcode: Final[str]
    """Scalar operation."""

    args: Final[list[Expr]]
    """Operands to the operation."""

    def __init__(self, result: Var, result_token: Var, opcode: str, args: list[Expr], span: Span) -> None:
        """Create a scalar operation statement.

        Args:
            result: Result expression
            result_token: Second operand
            opcode: Scalar operation
            args: Operands to the operation
            span: Source location
        """


class TensorOpStmt(Stmt):
    """Tensor operation statement"""

    result: Final[list[Var]]
    """Result expression."""

    result_token: Final[list[Var]]
    """Result tokens (can be empty)."""

    opcode: Final[str]
    """Tensor operation."""

    args: Final[list[Expr]]
    """Operands to the operation."""

    tokens: Final[list[Var]]
    """Tokens (can be empty)."""

    attrs: Final[dict[str, Any]]
    """Attributes (key-value metadata)."""

    def __init__(self, result: list[Var], result_tokens: list[Var], opcode: str, args: list[Expr],
                 tokens: list[Var], attrs: dict[str, Any], span: Span) -> None:
        """Create a tensor operation statement.

        Args:
            result: Result expression
            result_tokens: Result tokens
            opcode: Tensor operation
            args: Operands to the operation
            tokens: Tokens (can be empty)
            attrs: Attributes (key-value metadata)
            span: Source location
        """


class Function(IRNode):
    """Function definition with name, parameters, return types, and body."""

    name: Final[str]
    """Function name."""

    func_type: Final[FunctionType]
    """Function type (Opaque, Orchestration, InCore, Helper, SimtVF, or SimtCallee)."""

    entry: Final[bool]
    """Whether this is the program entry function."""

    attrs: Final[dict[str, object]]
    """Function attributes (key-value metadata)."""

    params: Final[list[Var]]
    """Parameter variables."""

    return_types: Final[list[Type]]
    """Return types."""

    body: Final[SeqStmts]
    """Function body statement (use SeqStmts for multiple statements)."""

    def has_attr(self, key: str) -> bool:
        """Check whether a function attribute exists."""
        ...

    def get_attr(self, key: str, default_value: int = 0) -> int:
        """Get an integer function attribute."""
        ...

    def __init__(
        self,
        name: str,
        params: list[Var],
        return_types: list[Type],
        body: Stmt,
        span: Span,
        type: FunctionType = FunctionType.Opaque,
        entry: bool = False,
        attrs: dict[str, object] | None = None,
    ) -> None:
        """Create a function definition.

        Args:
            name: Function name
            params: Parameter variables
            return_types: Return types
            body: Function body statement (use SeqStmts for multiple statements)
            span: Source location
            type: Function type (default: Opaque)
            entry: Whether this is the program entry function
            attrs: Function attributes (key-value metadata)
        """

    def __str__(self) -> str:
        """String representation of the function.

        Returns:
            Function as a string
        """

    def __repr__(self) -> str:
        """Detailed representation of the function.

        Returns:
            Function with type information
        """


class Program(IRNode):
    """Program definition with functions.

    Functions are automatically sorted by name for deterministic ordering.
    The GlobalVar name must match the function name and be unique within the program.
    """

    name: Final[str]
    """Program name."""

    def __init__(
        self,
        functions: list[Function],
        name: str,
        span: Span,
        debug_info: IRDebugInfo | None = None,
    ) -> None:
        """Create a program from a list of functions.

        GlobalVar references are created automatically from function names.

        Args:
            functions: List of functions
            name: Program name (optional)
            span: Source location
            debug_info: Semantic tuple metadata side table
        """

    def __getitem__(self, name: str) -> Function | None:
        """Get function by name, returns None if not found.

        Enables copy-paste navigation of structural equality error paths:
            program['main'].body[1].var

        Args:
            name: Function name to look up

        Returns:
            Function if found, None otherwise
        """

    @property
    def functions(self) -> dict[str, Function]:
        """List of functions, sorted by GlobalVar name."""


# ========== IR Builder ==========

def set_assemble_new_logical_tensor(enabled: bool) -> None:
    """Control whether Assemble creates a new LogicalTensor version."""


def assemble_new_logical_tensor() -> bool:
    """Get whether Assemble creates a new LogicalTensor version."""


class IRBuilder:
    """IR Builder for incremental IR construction with context management.

    The IRBuilder provides a stateful API for building IR incrementally using
    Begin/End patterns. It maintains a context stack to track nested scopes
    and validates proper construction.
    """

    def __init__(self) -> None:
        """Create an IR builder."""

    # Function building
    def begin_function(
        self,
        name: str,
        span: Span,
        type: FunctionType = FunctionType.Opaque,
    ) -> None:
        """Begin building a function.

        Args:
            name: Function name
            span: Source location for function definition
            type: Function type (default: Opaque)
            level: Hierarchy level (default: None)
            role: Function role (default: None)
            attrs: Function-level attributes dict (default: None)
        """

    def func_arg(
        self,
        name: str,
        type: Type,
        span: Span,
    ) -> Var:
        """Add a function parameter.

        Args:
            name: Parameter name
            type: Parameter type
            span: Source location for parameter

        Returns:
            Variable representing the parameter
        """

    def return_type(self, type: Type) -> None:
        """Add a return type to the current function.

        Args:
            type: Return type
        """

    def end_function(self, end_span: Span) -> Function:
        """End building a function.

        Args:
            end_span: Source location for end of function

        Returns:
            The built function
        """

    # For loop building
    def begin_for_loop(
        self,
        loop_var: Var,
        start: Expr,
        stop: Expr,
        step: Expr,
        span: Span,
    ) -> None:
        """Begin building a for loop.

        Args:
            loop_var: Loop variable
            start: Start value expression
            stop: Stop value expression
            step: Step value expression
            span: Source location for loop definition
        """

    def add_iter_arg(self, iter_arg: IterArg) -> None:
        """Add an iteration argument to the current for loop.

        Args:
            iter_arg: Iteration argument with initial value
        """

    def add_return_var(self, var: Var) -> None:
        """Add a return variable to the current for loop.

        Args:
            var: Return variable
        """

    def end_for_loop(self, end_span: Span) -> ForStmt:
        """End building a for loop.

        Args:
            end_span: Source location for end of loop

        Returns:
            The built for statement
        """

    # While loop building
    def begin_while_loop(self, condition: Expr, span: Span) -> None:
        """Begin building a while loop.

        Creates a new while loop context. Must be closed with end_while_loop().

        Args:
            condition: Condition expression
            span: Source location for loop definition
        """

    def add_while_iter_arg(self, iter_arg: IterArg) -> None:
        """Add an iteration argument to the current while loop.

        Iteration arguments are loop-carried values (SSA-style).

        Args:
            iter_arg: Iteration argument with initial value
        """

    def add_while_return_var(self, var: Var) -> None:
        """Add a return variable to the current while loop.

        Return variables capture the final values of iteration arguments.

        Args:
            var: Return variable
        """

    def set_while_loop_condition(self, condition: Expr) -> None:
        """Set the condition for the current while loop.

        Used to update the loop condition after setting up iter_args. This allows
        the condition to reference iter_arg variables that are defined in the loop.

        Args:
            condition: New condition expression
        """

    def end_while_loop(self, end_span: Span) -> WhileStmt:
        """End building a while loop.

        Finalizes the loop and returns it.

        Args:
            end_span: Source location for end of loop

        Returns:
            The built while statement
        """

    # If statement building
    def begin_if(self, condition: Expr, span: Span) -> None:
        """Begin building an if statement.

        Args:
            condition: Condition expression
            span: Source location for if statement
        """

    def begin_else(self, span: Span) -> None:
        """Begin the else branch of the current if statement.

        Args:
            span: Source location for else keyword
        """

    def add_if_return_var(self, var: Var) -> None:
        """Add a return variable to the current if statement.

        Args:
            var: Return variable
        """

    def end_if(self, end_span: Span) -> IfStmt:
        """End building an if statement.

        Args:
            end_span: Source location for end of if

        Returns:
            The built if statement
        """

    # Section building
    def begin_section(self, section_kind: SectionKind, span: Span) -> None:
        """Begin building a section statement.

        Args:
            section_kind: Section kind
            span: Source location for section statement
        """

    def end_section(self, end_span: Span) -> SectionStmt:
        """End building a section statement.

        Args:
            end_span: Source location for end of section

        Returns:
            The built section statement
        """

    # Program building
    def begin_program(self, name: str, span: Span) -> None:
        """Begin building a program.

        Args:
            name: Program name
            span: Source location for program definition
        """

    def add_function(self, func: Function) -> None:
        """Add a completed function to the current program.

        Args:
            func: Function to add
        """

    def end_program(self, end_span: Span) -> Program:
        """End building a program.

        Args:
            end_span: Source location for end of program

        Returns:
            The built program
        """

    def get_function_return_types(self, func_name: str) -> list[Type]:
        """Get return types for a function by its name.

        Returns the return types for a function if it has been added to the program.
        Returns empty list if not inside a program or function not yet added.

        Args:
            gvar: GlobalVar for the function

        Returns:
            Vector of return types
        """

    # Statement recording
    def emit(self, stmt: Stmt) -> None:
        """Emit a statement in the current context.

        Args:
            stmt: Statement to emit
        """

    def assign(self, var: Var, value: Expr, span: Span) -> AssignStmt:
        """Create an assignment statement and emit it.

        Args:
            var: Variable to assign to
            value: Expression value
            span: Source location for assignment

        Returns:
            The created assignment statement
        """

    def var(self, name: str, type: Type, span: Span) -> Var:
        """Create a variable (does not emit).

        Args:
            name: Variable name
            type: Variable type
            span: Source location

        Returns:
            The created variable
        """

    @overload
    def return_(self, values: list[Expr], span: Span) -> ReturnStmt:
        """Create a return statement and emit it.

        Args:
            values: List of expressions to return
            span: Source location for return statement

        Returns:
            The created return statement
        """

    @overload
    def return_(self, span: Span) -> ReturnStmt:
        """Create an empty return statement and emit it.

        Args:
            span: Source location for return statement

        Returns:
            The created return statement
        """

    # Context state queries
    def in_function(self) -> bool:
        """Check if currently inside a function.

        Returns:
            True if inside a function context
        """

    def in_loop(self) -> bool:
        """Check if currently inside a for loop.

        Returns:
            True if inside a for loop context
        """

    def in_if(self) -> bool:
        """Check if currently inside an if statement.

        Returns:
            True if inside an if statement context
        """

    def in_program(self) -> bool:
        """Check if currently inside a program.

        Returns:
            True if inside a program context
        """

    def create_tensor_var(
        self,
        t: pypto_impl.DataType,
        shape: list[SymbolicScalar | int],
        format: pypto_impl.TileOpFormat = pypto_impl.TileOpFormat.TILEOP_ND,
        name: str = "",
    ) -> pypto_impl.LogicalTensor:
        """Create a tensor variable with static shape.

        Args:
            t: Data type of the tensor
            shape: Shape of the tensor
            format: Data format of the tensor
            name: Name of the tensor

        Returns:
            The created tensor variable
        """

    def create_tensor_op_stmt(
        self,
        result: list[Var],
        result_token: Var,
        opcode: str,
        args: list[Expr],
        tokens: list[Var],
        attrs: dict[str, Any],
        span: Span,
    ) -> TensorOpStmt:
        """Create a tensor operation statement.

        Args:
            result: Result variables of the operation
            result_token: Result token of the operation
            opcode: Opcode of the operation
            args: Arguments of the operation
            tokens: Tokens of the operation
            attrs: Attributes of the operation
            span: Source location

        Returns:
            The created tensor operation statement
        """

    def create_scalar_var(self, value: str = "") -> 'SymbolicScalar':
        """Create a scalar variable from a symbol name.

        Args:
            value: Symbol name

        Returns:
            The created symbolic scalar
        """

    def create_const_int(self, value: int) -> 'SymbolicScalar':
        """Create a constant integer scalar.

        Args:
            value: Integer value

        Returns:
            The created constant integer scalar
        """

    def create_assign_stmt(self, var: Var, value: Expr, span: Span) -> AssignStmt:
        """Create an assignment statement.

        Args:
            var: Variable to assign to
            value: Right-hand side expression
            span: Source location

        Returns:
            The created assignment statement
        """

    def create_seq_stmts(self, stmts: list[Stmt], span: Span) -> SeqStmts:
        """Create a sequence of statements.

        Args:
            stmts: Statements in the sequence
            span: Source location

        Returns:
            The created sequence of statements
        """

    def create_if_stmt(
        self,
        cond: Expr,
        then_body: Stmt,
        else_body: Optional[Stmt],
        return_vars: list[Var],
        span: Span,
    ) -> IfStmt:
        """Create an if statement.

        Args:
            cond: Condition expression
            then_body: Then branch body
            else_body: Else branch body (can be None)
            return_vars: Return variables for SSA phi nodes
            span: Source location

        Returns:
            The created if statement
        """

    def create_return_stmt(self, return_vars: list[Expr], span: Span) -> ReturnStmt:
        """Create a return statement.

        Args:
            return_vars: Expressions to return
            span: Source location

        Returns:
            The created return statement
        """

    def create_yield_stmt(self, return_vars: list[Expr], span: Span) -> YieldStmt:
        """Create a yield statement.

        Args:
            return_vars: Expressions to yield
            span: Source location

        Returns:
            The created yield statement
        """

    def create_for_stmt(
        self,
        loop_var: Var,
        start: Expr,
        stop: Expr,
        step: Expr,
        iter_args: list[IterArg],
        body: Stmt,
        return_vars: list[Var],
        span: Span,
        attrs: dict[str, Any] | None = None,
    ) -> ForStmt:
        """Create a for statement.

        Args:
            loop_var: Loop variable
            start: Start value expression
            stop: Stop value expression
            step: Step value expression
            iter_args: Iteration arguments (loop-carried values)
            body: Loop body
            return_vars: Return variables
            span: Source location

        Returns:
            The created for statement
        """

    def create_while_stmt(
        self,
        cond: Expr,
        iter_args: list[IterArg],
        body: Stmt,
        return_vars: list[Var],
        span: Span,
    ) -> WhileStmt:
        """Create a while statement.

        Args:
            cond: Condition expression
            iter_args: Iteration arguments (loop-carried values)
            body: Loop body
            return_vars: Return variables
            span: Source location

        Returns:
            The created while statement
        """

    def create_break_stmt(self, return_vars: list[Expr], span: Span) -> BreakStmt:
        """Create a break statement.

        Args:
            return_vars: Expressions carried by the break
            span: Source location

        Returns:
            The created break statement
        """

    def create_continue_stmt(self, return_vars: list[Expr], span: Span) -> ContinueStmt:
        """Create a continue statement.

        Args:
            return_vars: Expressions carried by the continue
            span: Source location

        Returns:
            The created continue statement
        """

    def update_jump_values(
        self, jump_op: YieldStmt | BreakStmt | ContinueStmt, values: list[Expr]
    ) -> None:
        """Replace values carried by a control-flow jump during IR construction."""

    @overload
    def create_function(
        self,
        name: str,
        params: list[Var],
        body: Stmt,
    ) -> Function:
        """Create a function with variable parameters.

        Args:
            name: Function name
            params: Parameter variables
            body: Function body statement

        Returns:
            The created function
        """

    @overload
    def create_function(
        self,
        name: str,
        params: list[Var],
        returnTypes: list[Type],
        body: Stmt,
        span: Span,
    ) -> Function:
        """Create a function.

        Args:
            name: Function name
            params: Parameter variables
            returnTypes: Return types
            body: Function body statement
            span: Source location

        Returns:
            The created function
        """

    def create_program(
        self,
        functions: list[Function],
        name: str,
        span: Span,
    ) -> Program:
        """Create a program.

        Args:
            functions: List of functions
            name: Program name
            span: Source location

        Returns:
            The created program
        """

    def set_insert_point(self, insert_point: InsertPoint):
        """Set the insert point for the builder.

        Args:
            insert_point: Insert point to set

        Returns:
            None
        """

    def clear_insert_point(self):
        """Clear the insert point for the builder.

        Returns:
            None

        """

    def create_var_like(self, name: str, value: Expr) -> Var:
        """Create a variable like the given value.

        Args:
            name: Name of the variable
            value: Value to base the variable on

        Returns:
            The created variable
        """

    @overload
    def create_iter_arg(self, var: Var, initValue: Optional[Expr]) -> IterArg:
        """Create an iteration argument.

        Args:
            var: Variable of the iteration argument
            initValue: Initial value of the iteration argument

        Returns:
            The created iteration argument
        """

    @overload
    def create_iter_arg(self, name: str, type: Type, initValue: Expr, span: Span) -> IterArg:
        """Create an iteration argument.

        Args:
            name: Name of the iteration argument
            initValue: Initial value of the iteration argument
            span: Span of the iteration argument

        Returns:
            The created iteration argument
        """

    def emit_tensor_stmts(self):
        """Emit tensor statements.

        Returns:
            None
        """

    def checkpoint(self):
        """Create a checkpoint.

        Returns:
            None
        """

    def restore(self):
        """Restore the last checkpoint.

        Returns:
            None
        """

    def none(self) -> Expr:
        """Get the None value.

        Returns:
            The None value
        """


class InsertPoint:
    def __init__(self, seq_stmts: SeqStmts):
        """Create an insert point.

        Args:
            seq_stmts: Sequence of statements
            span: Source location
        """


def type_equal(a: Expr, b: Expr) -> bool:
    """Check if the types of two expressions are equal.

    Args:
        a: First expression
        b: Second expression

    Returns:
        True if the types are equal, False otherwise
    """


class Pass:

    def __call__(self, program: Program):
        """Execute the pass on the program.

        Args:
            program: Program to apply the pass on
        """

    @staticmethod
    def init_mem_ref() -> Pass:
        """Initialize memory references."""

    @staticmethod
    def aggressive_dce() -> Pass:
        """Eliminate dead code."""

    @staticmethod
    def canonicalize() -> Pass:
        """Canonicalize the IR."""

    @staticmethod
    def infer_token_pass() -> Pass:
        """Infer token dependencies."""

    @staticmethod
    def remove_redundant_token_pass() -> Pass:
        """Remove redundant token dependencies."""

    @staticmethod
    def merge_stmts_into_if() -> Pass:
        """Merge statements into if statements."""

    @staticmethod
    def simplify_symbolic_scalar() -> Pass:
        """Simplify symbolic scalars under their enclosing branch condition."""

    @staticmethod
    def create_root_functions() -> Pass:
        """Create root functions."""

    @staticmethod
    def infer_multi_iter_overlap() -> Pass:
        """Mark outcast multi_iter_no_overlap for assemble-level loop-carried non-overlap."""

    @staticmethod
    def finalize_dynamic_function() -> Pass:
        """Finalize dynamic functions."""

    @staticmethod
    def run_verifier() -> Pass:
        """Run the verifier on the program."""
