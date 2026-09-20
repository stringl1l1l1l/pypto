# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
import enum
import inspect
import textwrap
import threading
from typing import Any, Optional, Union, cast
import weakref

import pypto
from pypto import ir


@dataclass
class LoopRange:
    start: Any
    stop: Any
    step: Any
    unroll_list: list[int]
    batch: bool
    parallel: bool = False
    submit_before_loop: bool = False
    name: Optional[str] = None
    idx_name: Optional[str] = None


@dataclass(frozen=True)
class Value:
    id: int

    def __str__(self) -> str:
        return f"%{self.id}"


Operand = Union[Value, Any]


@dataclass
class Starred:
    """A ``*x`` unpacking marker wrapping an operand within a Call's args."""

    value: Operand


@dataclass
class DoubleStarred:
    """A ``**d`` mapping-unpack marker wrapping an operand within a Call's
    kwargs or a dict literal's items.
    """

    value: Operand


class Formatter:
    def __init__(self, show_blocks):
        self.blocks = []
        self.show_blocks = show_blocks

    def format(self, x: Operand) -> str:
        if isinstance(x, Starred):
            return f"*{self.format(x.value)}"
        if isinstance(x, DoubleStarred):
            return f"**{self.format(x.value)}"
        if isinstance(x, Block):
            if self.show_blocks:
                self.blocks.append(x)
            return f"^{x.id}"
        elif isinstance(x, list):
            return f"[{', '.join(self.format(x) for x in x)}]"
        elif isinstance(x, tuple):
            return f"({', '.join(self.format(x) for x in x)})"
        elif callable(x):
            return f"@{x.__name__}"
        else:
            return f"{x}"


class Jump(enum.Enum):
    END_BRANCH = 1  # used in if_else
    RETURN = 2
    CONTINUE = 3
    BREAK = 4


class LoopKind(enum.Enum):
    FOR = 0
    WHILE = 1
    DYNAMIC_FOR = 2  # pypto.loop, compiled to a hardware for; break/continue unsupported


@dataclass
class Call:
    result: Optional[Value]
    callee: Operand
    args: tuple[Operand, ...]
    kwargs: dict[str, Operand]
    span: ir.Span

    def __str__(self) -> str:
        return self.dump(show_blocks=True)

    def dump(self, show_blocks: bool = False):
        fmt = Formatter(show_blocks)
        result_str = f"{self.result} = " if self.result else ""
        line_info = f"  # Line {self.span.begin_line}"
        kw_strs = [
            f"**{fmt.format(v.value)}" if isinstance(v, DoubleStarred) else f"{k}={fmt.format(v)}"
            for k, v in self.kwargs
        ]
        args_all = [fmt.format(x) for x in self.args] + kw_strs
        args_str = ", ".join(args_all)
        block_str = "".join(textwrap.indent(f"\n{b}", "  ") for b in fmt.blocks)
        return f"{result_str}{fmt.format(self.callee)}({args_str}){line_info}{block_str}"


@dataclass
class Block:
    id: int
    args: tuple[Operand, ...]
    calls: list[Call]
    result: Operand
    jump: Optional[Jump] = None
    jump_loc: ir.Span = ir.Span.unknown()
    store_names: set[str] = field(default_factory=set)
    span: ir.Span = ir.Span.unknown()

    def __str__(self):
        arg_str = ", ".join(str(p) for p in self.args)
        stmt_str = "".join(f"\n{c}" for c in self.calls)
        if self.jump is not None:
            stmt_str += f"\n{self.jump.name}  # Line {self.jump_loc.begin_line}"
        stmt_str = textwrap.indent(stmt_str, "    ")
        return f"^{self.id}({arg_str}):{stmt_str}"


@dataclass
class Function:
    # Function name
    name: str

    # source location
    span: ir.Span

    # Function signature
    signature: inspect.Signature

    # Function body
    body: Block

    # Global variables
    global_vars: tuple[str, ...]

    # Global values
    global_values: tuple[Any, ...]

    # Parameter names (positional), used to bind call arguments for nested calls
    params: tuple[str, ...] = ()

    # Default values aligned with `params` (None where no default); applied when
    # a nested call omits the argument.
    param_defaults: tuple = ()

    def __str__(self):
        param_str = []
        for k, v in zip(self.params, self.param_defaults):
            if v is None:
                param_str.append(f"{k}")
            else:
                param_str.append(f"{k}={v}")
        return f"lambda {', '.join(param_str)}:\n" + str(self.body)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Function should not be called directly")


class ReturnSignal(Exception):  # noqa: N818
    def __init__(self, value=None):
        super().__init__()
        self.value = value


class BreakSignal(Exception):  # noqa: N818
    pass


class ContinueSignal(Exception):  # noqa: N818
    pass


def in_(item: Any, container: Any) -> bool:
    return item in container


def not_in(item: Any, container: Any) -> bool:
    return item not in container


class _Poison:
    """Value for a conflicted yield var.

    Any value-use (arithmetic, indexing, attribute access, ...) raises so the
    conflict is reported at the point of consumption rather than at if-construction
    time.
    """

    __slots__ = ("_name", "_span")

    def __init__(self, name, span):
        self._name = name
        self._span = span

    def _conflict(self, *args, **kwargs):
        raise RuntimeError(
            f"Conflicted var {self._name}: {self._span.filename}:{self._span.begin_line}"
        )

    __getattr__ = _conflict
    __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = _conflict
    __truediv__ = __rtruediv__ = __floordiv__ = __rfloordiv__ = __matmul__ = _conflict
    __mod__ = __rmod__ = __pow__ = __rpow__ = _conflict
    __neg__ = __pos__ = __abs__ = __invert__ = _conflict
    __getitem__ = __setitem__ = __iter__ = __len__ = _conflict
    __eq__ = __ne__ = __lt__ = __le__ = __gt__ = __ge__ = _conflict


class BuildContext(ir.IRBuilder):
    def __init__(self, span: ir.Span):
        super().__init__()
        self.parent = None
        self.span = span
        self.return_var_names = []
        self.loop_stack = []  # used by legacy is_loop_begin() and is_loop_end()
        self.call_stack = []
        self.in_loop_unroll = False

    def __enter__(self):
        self.parent = _current.build_context
        _current.build_context = self
        return self

    def __exit__(self, exc_type, exc, tb):
        _current.build_context = self.parent

    @staticmethod
    def current() -> "BuildContext":
        if _current.build_context is None:
            raise ValueError("BuildContext is not initialized")
        return _current.build_context

    @contextmanager
    def change_span(self, span: ir.Span):
        old_span, self.span = self.span, span
        try:
            yield
        finally:
            self.span = old_span

    @contextmanager
    def change_return_vars(self, names: list[str]):
        old_names, self.return_var_names = self.return_var_names, names
        try:
            yield
        finally:
            self.return_var_names = old_names

    def poison(self, name: str, span: ir.Span):
        # An UnknownType-typed var that wrap() turns into a _Poison: consuming it
        # raises "Conflicted var <name>", so a deferred type conflict is reported
        # at the point of use rather than at if-construction time.
        return ir.Var(name, ir.UnknownType.get(), span)

    def unwrap(self, val: Any) -> ir.Expr:
        if val is None:
            return self.none()
        if isinstance(val, bool):
            return ir.ConstBool(val, ir.Span.unknown())
        if isinstance(val, int):
            return self.create_const_int(val).as_expr()
        elif isinstance(val, float):
            return ir.ConstFloat(val, ir.DataType.FP64, ir.Span.unknown())
        elif isinstance(val, pypto.SymbolicScalar):
            return val.as_expr()
        elif isinstance(val, pypto.Tensor):
            if val.is_empty():
                return self.none()
            return val.logical_tensor()
        elif isinstance(val, (list, tuple)):
            return ir.MakeTuple([self.unwrap(v) for v in val], ir.Span.unknown())
        else:
            # not ir types, treated as none, it should be removed in canonicalize pass
            return self.none()

    def wrap(self, val: ir.Expr) -> Any:
        if isinstance(val, (ir.ConstInt, ir.ConstFloat, ir.ConstBool)):
            return val.value
        if isinstance(val, ir.Expr):
            if isinstance(val.type, ir.ScalarType):
                return pypto.SymbolicScalar(val)
            elif isinstance(val.type, ir.LogicalTensorType):
                return pypto.Tensor.from_logical_tensor(val)
            elif isinstance(val.type, ir.NoneType):
                return None
            elif isinstance(val.type, ir.UnknownType):
                var = cast(ir.Var, val)
                return _Poison(var.name, var.span)
            else:
                raise TypeError(f"Invalid type {type(val)} for wrap")
        return val


class CollectContext(BuildContext):
    pass


class InsertPoint:
    ctx: BuildContext

    def __init__(self, body: ir.SeqStmts):
        self.insert_point = ir.InsertPoint(body)
        self.ctx: BuildContext

    def __enter__(self):
        self.ctx = BuildContext.current()
        self.ctx.set_insert_point(self.insert_point)

    def __exit__(self, exc_type, exc, tb):
        self.ctx.clear_insert_point()


_ATOMIC_TYPES = (
    type(None),
    bool,
    int,
    float,
    complex,
    str,
    bytes,
    pypto.SymbolicScalar,
)

def _is_atomic(value) -> bool:
    return isinstance(value, _ATOMIC_TYPES) or value is pypto


def _id_guard(value: Any):
    """Callable proving an id key is still owned by its original object:
    a weakref when possible, else a strong pin (a pinned object never dies,
    so its id can never be recycled). Calling it returns None once dead."""
    try:
        return weakref.ref(value)
    except TypeError:
        return lambda: value


class Scope:
    def __init__(self, parent: Optional["Scope"] = None):
        self.locals: dict[str, Union[ir.Var, Any]] = {}
        self.parent = parent
        self.varmap = {}
        self.eval = True
        self.name_aliases: dict[int, set[str]] = defaultdict(set)
        self.canonical_name: dict[int, str] = {}
        # id(value) keys outlive their objects: a dead object's address can be
        # recycled by a new allocation, whose names must not merge with the
        # stale entry. The guard lets us detect and drop recycled ids.
        self._id_guards: dict[int, Any] = {}

    def _drop_if_dead(self, key: int) -> None:
        guard = self._id_guards.get(key)
        if guard is not None and guard() is None:
            self._id_guards.pop(key, None)
            self.name_aliases.pop(key, None)
            self.canonical_name.pop(key, None)

    def __getitem__(self, name: str) -> Union[ir.Var, Any]:
        var = self.get_local(name)
        if var is None:
            if self.parent:
                return self.parent[name]
            return None
        return var

    def __setitem__(self, key: str, value: Union[ir.Var, Any]):
        self.bind_name(key, value)
        self.set_local(key, value)

    def bind_name(self, name: str, value: Any):
        if self.is_global_scope():
            return
        if not _is_atomic(value):
            key = id(value)
            self._drop_if_dead(key)
            self.name_aliases[key].add(name)
            if key not in self.canonical_name:
                self.canonical_name[key] = name
                self._id_guards[key] = _id_guard(value)

    def get_canonical_name(self, value: Any) -> str:
        key = id(value)
        self._drop_if_dead(key)
        name = self.canonical_name.get(key, "")
        if name:
            return name
        if self.parent is not None:
            name = self.parent.get_canonical_name(value)
            if name and name.split(".", 1)[0] not in self.locals:
                return name
            # The parent's canonical name may be shadowed, but another alias
            # can still name the same object in this scope.
            return next(iter(sorted(self.aliases(value))), "")
        return ""

    def aliases(self, value: Any) -> set[str]:
        """Names bound to this object, excluding shadowed parent aliases."""
        key = id(value)
        self._drop_if_dead(key)
        names = set(self.name_aliases.get(key, set()))
        if self.parent is not None:
            names.update(name for name in self.parent.aliases(value)
                         if name.split(".", 1)[0] not in self.locals)
        return names

    def is_global_scope(self) -> bool:
        return self.parent is None

    @staticmethod
    def store(name: str, value: Union[ir.Var, Any]):
        scope = Scope.current()
        scope[name] = value

    @staticmethod
    def current() -> "Scope":
        if _current.scope is None:
            raise ValueError("Scope is not initialized")
        return _current.scope

    @contextmanager
    def make_current(self):
        old, _current.scope = _current.scope, self
        try:
            yield
        finally:
            _current.scope = old

    def resolve(self, val) -> Any:
        if isinstance(val, Value):
            return self.varmap[val.id]
        if isinstance(val, (list, tuple)):
            result = []
            for v in val:
                if isinstance(v, Starred):
                    result.extend(self.resolve(v.value))
                else:
                    result.append(self.resolve(v))
            if isinstance(val, tuple):
                return tuple(result)
            return result
        return val

    def _resolve_slot(self, parts: list[str]):
        """Resolve a dotted name's container: (object holding the last part, last part).
        """
        obj = self.locals.get(parts[0], None)
        for part in parts[1:-1]:
            if obj is None:
                return None, None
            obj = getattr(obj, part, None)
        return obj, parts[-1]

    def get_local(self, name: str):
        if "." not in name:
            return self.locals.get(name)
        obj, key = self._resolve_slot(name.split("."))
        if obj is not None and key is not None:
            return getattr(obj, key, None)
        return None

    def resolve_local(self, name: str):
        """Resolve a name against THIS scope only (locals + dotted slots).

        Unlike __getitem__, a miss does not fall through to parent scopes or
        globals: carry collection must not pick up a same-named global/builtin
        just because a local currently holds None.
        """
        return self.get_local(name)

    def set_local(self, name: str, value: Union[ir.Var, Any]):
        if "." not in name:
            self.locals[name] = value
            return
        obj, key = self._resolve_slot(name.split("."))
        if obj is not None and key is not None:
            current = getattr(obj, key, None)
            if isinstance(current, pypto.Tensor) and isinstance(value, pypto.Tensor):
                Journal.record(current)
                current.set_logical_tensor(value.logical_tensor())
            else:
                Journal.record(obj, key)
                setattr(obj, key, value)


class _Missing:
    """Sentinel: attribute slot did not exist before the recorded write."""

    def __repr__(self):
        return "<missing>"


_MISSING = _Missing()


class Journal:
    """Undo log for attribute-slot writes (``d.a = x``).
    """

    _state = threading.local()

    @classmethod
    def _active(cls) -> list["Journal"]:
        stack = getattr(cls._state, "stack", None)
        if stack is None:
            stack = cls._state.stack = []
        return stack

    def __init__(self):
        self._entries = []  # (obj, key, old value) in write order

    @classmethod
    def record(cls, obj, key = ""):
        """Record an attribute-slot write before it happens; no-op unless a
        Journal is active."""
        stack = cls._active()
        if stack:
            entries = stack[-1]._entries
            if isinstance(obj, pypto.Tensor):
                entries.append((obj, key, obj.logical_tensor()))
            else:
                entries.append((obj, key, getattr(obj, key, _MISSING)))

    def __enter__(self):
        Journal._active().append(self)
        return self

    def __exit__(self, exc_type, exc, tb):
        Journal._active().pop()
        for obj, key, old in reversed(self._entries):
            if isinstance(obj, pypto.Tensor):
                obj.set_logical_tensor(old)
            elif old is _MISSING:
                # Slot was created inside the branch: roll the creation back.
                if key and getattr(obj, key, _MISSING) is not _MISSING:
                    delattr(obj, key)
            else:
                setattr(obj, key, old)


def slot_read(obj, key, value) -> None:
    scope = _current.scope
    assert scope is not None

    name = scope.get_canonical_name(obj)
    if name:
        scope.bind_name(f"{name}.{key}", value)


def slot_write(obj, key, value) -> None:
    scope = _current.scope
    assert scope is not None

    name = scope.get_canonical_name(obj)
    scope.bind_name(f"{name}.{key}", value)

    block = _current.collector_block
    if block is None:
        return
    block.store_names.add(f"{name}.{key}")

    Journal.record(obj, key)


def slot_write_inplace(obj) -> None:
    block = _current.collector_block
    if block is None:
        return

    scope = _current.scope
    assert scope is not None

    name = scope.get_canonical_name(obj)
    if name:
        block.store_names.add(name)
    block.store_names.update(scope.aliases(obj))

    Journal.record(obj)

class _Current(threading.local):
    scope: Optional[Scope] = None
    build_context: Optional[BuildContext] = None
    collector_block: Optional[Block] = None


_current = _Current()
