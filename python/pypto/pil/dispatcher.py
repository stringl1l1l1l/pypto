# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
from functools import lru_cache
import inspect
import logging
import os
import sys
import sysconfig
import traceback
from typing import Optional

from .. import pypto_impl
from .op_registry import dispatch
from .parser import ast2pil
from .pir import (
    Block,
    BreakSignal,
    BuildContext,
    Call,
    CollectContext,
    ContinueSignal,
    DoubleStarred,
    Function,
    Jump,
    ReturnSignal,
    Scope,
    _current,
)

# used to skip "std python package".
_STDLIB_DIRS = (sysconfig.get_paths().get("stdlib", ""), sysconfig.get_paths().get("platstdlib", ""))


def _is_stdlib(mod_name: str) -> bool:
    mod = sys.modules.get(mod_name)
    if mod is None:
        return False
    f = getattr(mod, "__file__", "")
    if not f:
        # built-in module (no source file) -> skip
        return True
    f = os.path.normpath(f)
    return any(f.startswith(d) for d in _STDLIB_DIRS if d)


def _is_pypto(mod_name: str) -> bool:
    return mod_name == "pypto" or mod_name.startswith("pypto.")


@lru_cache(maxsize=None)
def _get_or_compile(pyfunc):
    """Return a cached pil Function for a user-defined helper, else `None`."""
    if not inspect.isfunction(pyfunc):
        return None

    mod = getattr(pyfunc, "__module__", "")
    if not mod or _is_pypto(mod) or _is_stdlib(mod):
        return None

    try:
        return ast2pil(pyfunc, entry_point=False)
    except Exception as e:
        logging.error("Failed to compile %s: %s", pyfunc.__name__, e)
        return None


class DispatchError(RuntimeError):
    pass


def dispatch_call(call: Call, scope: Scope, ctx: BuildContext):
    callee = scope.resolve(call.callee)
    args = tuple(scope.resolve(call.args))

    kwargs = {}
    for k, v in call.kwargs:
        if isinstance(v, DoubleStarred):
            kwargs.update(scope.resolve(v.value))
        else:
            kwargs[k] = scope.resolve(v)

    func = callee if isinstance(callee, Function) else _get_or_compile(callee)
    if func is not None:
        ctx.call_stack.append(call.span)
        try:
            ret = call_function(func, args, kwargs, ctx)
        finally:
            ctx.call_stack.pop()
    else:
        try:
            ret = dispatch(callee, ctx, *args, **kwargs)
        except (ReturnSignal, BreakSignal, ContinueSignal):
            # control-flow signals are not dispatch errors; let them propagate.
            raise
        except DispatchError:
            # already built by an inner dispatch_call; don't duplicate the stack.
            raise
        except Exception as e:
            frames = [traceback.FrameSummary(s.filename, s.begin_line, "")
                      for s in ctx.call_stack + [call.span]]
            stack = "".join(traceback.format_list(frames))
            raise DispatchError(f"{e}\n{callee}\n{stack}") from None

    if call.result is not None:
        scope.varmap[call.result.id] = ret

    pypto_impl.SetSpan(call.span.filename, call.span.begin_line)
    ctx.emit_tensor_stmts()
    pypto_impl.ClearSpan()


def call_function(func: Function, args: tuple, kwargs: dict, ctx: BuildContext):
    caller = Scope.current()
    if func.global_vars:
        # Standalone function (built by ast2pil from a real Python function
        root = Scope()
        for name, val in zip(func.global_vars, func.global_values):
            root[name] = val
    else:
        # Inline nested def/lambda/comprehension: lowered within the caller, so
        # its free names resolve lexically through the caller's scope chain.
        root = caller

    scope = Scope(parent=root)
    namemap = {}

    with scope.make_current():
        supplied = set()
        for name, val in zip(func.params, args):
            scope[name] = val
            namemap[name] = caller.get_canonical_name(val)
            supplied.add(name)
        for name, val in kwargs.items():
            scope[name] = val
            namemap[name] = caller.get_canonical_name(val)
            supplied.add(name)

        # apply default values for any params not supplied by the call
        for name, defval in zip(func.params, func.param_defaults):
            if name not in supplied and defval is not None:
                scope[name] = caller.resolve(defval)
        orig_vals = {name: scope.locals.get(name) for name in func.params}

        try:
            if isinstance(ctx, CollectContext):
                collect(func.body, rewriter=frame_rewrite(namemap, caller, scope))
            else:
                dispatch_block(func.body, True)
        except ReturnSignal as sig:
            retval = sig.value
        else:
            retval = None

    if not isinstance(ctx, CollectContext):
        for name in func.params:
            orig_val = orig_vals.get(name)
            current_val = scope.locals.get(name)
            if current_val is None or current_val is orig_val:
                continue
            for caller_name, caller_val in caller.locals.items():
                if caller_val is orig_val:
                    caller[caller_name] = current_val
                    break

    return retval


def block_jump(scope, ctx, block: Block):
    if block.jump is Jump.CONTINUE:
        raise ContinueSignal
    if block.jump is Jump.BREAK:
        raise BreakSignal
    if block.jump is Jump.RETURN:
        raise ReturnSignal(ctx.wrap(scope["$retval"]))
    if block.jump is Jump.END_BRANCH and block.result is not None:
        return scope.resolve(block.result)


def frame_rewrite(namemap: dict, caller: Scope, callee: Scope):
    def rewrite(name: str) -> Optional[str]:
        parts = name.split(".", 1)
        if parts[0] in namemap:
            # An arg with no canonical caller name (atomic value, unbound
            # temporary) has nothing to rewrite to — drop, don't emit "".
            if not namemap[parts[0]]:
                return None
            parts[0] = namemap[parts[0]]
            return ".".join(parts)
        # A free name must refer to the same object in both frames. A standalone
        # helper can have a separate global binding with the same spelling.
        if (
            parts[0] not in callee.locals
            and caller[parts[0]] is not None
            and caller[parts[0]] is callee[parts[0]]
        ):
            return name
        return None

    return rewrite


def dispatch_block(block: Block, is_static: bool, rewriter=None):
    scope = Scope.current()
    ctx = BuildContext.current()

    prev_block = _current.collector_block
    _current.collector_block = block
    try:
        for call in block.calls:
            with ctx.change_span(call.span):
                dispatch_call(call, scope, ctx)
    finally:
        if prev_block is not None and isinstance(ctx, CollectContext):
            if rewriter is None:
                prev_block.store_names.update(block.store_names)
            else:
                for name in block.store_names:
                    rewritten = rewriter(name)
                    if rewritten is not None:
                        prev_block.store_names.add(rewritten)
        _current.collector_block = prev_block

    if is_static:
        return block_jump(scope, ctx, block)

    return None


def collect(block: Block, rewriter=None):
    dispatch_block(block, True, rewriter=rewriter)
