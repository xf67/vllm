import os
import functools
from datetime import datetime

import torch


def trace_cuda_memory(
    output_dir="./cuda_mem_traces",
    enabled=True,
    max_entries=200000,
    device=None,
    dump_snapshot=True,
    print_summary=True,
):
    """
    装饰器：追踪被包装函数执行期间的 CUDA memory trace。

    参数：
        output_dir: 导出 trace 文件的目录
        enabled: 是否启用
        max_entries: memory history 最大记录条目数
        device: 指定设备，如 "cuda:0"；默认使用当前设备
        dump_snapshot: 是否在结束时导出 snapshot pickle
        print_summary: 是否打印前后显存统计

    导出文件：
        - *.pickle: 可上传到 https://pytorch.org/memory_viz 查看
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not enabled:
                return func(*args, **kwargs)

            if not torch.cuda.is_available():
                print(f"[cuda-trace] CUDA 不可用，跳过 {func.__name__}")
                return func(*args, **kwargs)

            os.makedirs(output_dir, exist_ok=True)

            if device is not None:
                dev = torch.device(device)
                torch.cuda.set_device(dev)
            else:
                dev = torch.device(torch.cuda.current_device())

            pid = os.getpid()
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = f"{func.__name__}_pid{pid}_{ts}"
            snapshot_path = os.path.join(output_dir, f"{base}.pickle")

            try:
                torch.cuda.synchronize(dev)
            except Exception:
                pass

            alloc_before = torch.cuda.memory_allocated(dev)
            reserved_before = torch.cuda.memory_reserved(dev)
            max_alloc_before = torch.cuda.max_memory_allocated(dev)
            max_reserved_before = torch.cuda.max_memory_reserved(dev)

            if print_summary:
                print(f"[cuda-trace] start tracing: {func.__name__}")
                print(f"[cuda-trace] device={dev}, pid={pid}")
                print(
                    f"[cuda-trace] before | "
                    f"allocated={alloc_before / 1024**2:.2f} MB, "
                    f"reserved={reserved_before / 1024**2:.2f} MB, "
                    f"max_allocated={max_alloc_before / 1024**2:.2f} MB, "
                    f"max_reserved={max_reserved_before / 1024**2:.2f} MB"
                )

            torch.cuda.reset_peak_memory_stats(dev)

            # 开启 allocator history
            torch.cuda.memory._record_memory_history(
                enabled="all",
                context="all",
                stacks="all",
                max_entries=max_entries,
            )

            result = None
            exc = None

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                exc = e
                raise
            finally:
                try:
                    torch.cuda.synchronize(dev)
                except Exception:
                    pass

                alloc_after = torch.cuda.memory_allocated(dev)
                reserved_after = torch.cuda.memory_reserved(dev)
                max_alloc_after = torch.cuda.max_memory_allocated(dev)
                max_reserved_after = torch.cuda.max_memory_reserved(dev)

                if dump_snapshot:
                    try:
                        torch.cuda.memory._dump_snapshot(snapshot_path)
                        print(f"[cuda-trace] snapshot saved: {snapshot_path}")
                        print("[cuda-trace] open with: https://pytorch.org/memory_viz")
                    except Exception as dump_e:
                        print(f"[cuda-trace] dump snapshot failed: {dump_e}")

                try:
                    torch.cuda.memory._record_memory_history(enabled=None)
                except Exception as stop_e:
                    print(f"[cuda-trace] stop history failed: {stop_e}")

                if print_summary:
                    print(
                        f"[cuda-trace] after  | "
                        f"allocated={alloc_after / 1024**2:.2f} MB, "
                        f"reserved={reserved_after / 1024**2:.2f} MB, "
                        f"peak_allocated={max_alloc_after / 1024**2:.2f} MB, "
                        f"peak_reserved={max_reserved_after / 1024**2:.2f} MB"
                    )
                    print(
                        f"[cuda-trace] delta  | "
                        f"allocated={((alloc_after - alloc_before) / 1024**2):.2f} MB, "
                        f"reserved={((reserved_after - reserved_before) / 1024**2):.2f} MB"
                    )
                    if exc is not None:
                        print(f"[cuda-trace] function exited with exception: {type(exc).__name__}: {exc}")

        return wrapper
    return decorator