from scripts.monitor_v42_formal_resources import _classify


def _row(**values):
    return {
        "available_ram_gb": 5.0,
        "total_ram_used_gb": 10.0,
        "pagefile_used_gb": 1.0,
        "gpu_util_percent": 40.0,
        "gpu_memory_used_mb": 2500.0,
        "gpu_memory_total_mb": 8192.0,
        "total_cpu_percent": 30.0,
        "disk_read_MBps": 0.0,
        "disk_write_MBps": 0.0,
        **values,
    }


def test_formal_resource_bottleneck_classification():
    assert _classify([_row(gpu_util_percent=90.0)] * 4) == "GPU_COMPUTE_BOUND"
    assert _classify([_row(available_ram_gb=2.0)] * 4) == "RAM_BOUND"
    assert _classify([_row(gpu_util_percent=40.0)] * 4) == "GPU_STARVED"
