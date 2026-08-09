import triton

if triton.__version__ >= "3.4.0":
    from .k_grouped_gemm_TMA_triton3_4 import k_grouped_gemm
    from .m_grouped_gemm_TMA_triton3_4 import m_grouped_gemm
elif triton.__version__ >= "3.2.0":
    from .k_grouped_gemm_TMA import k_grouped_gemm
    from .m_grouped_gemm_TMA import m_grouped_gemm
else:
    raise ImportError(
        f"Triton version {triton.__version__} is not supported. Please install Triton version 3.2.0 or higher."
    )
__all__ = ["k_grouped_gemm", "m_grouped_gemm"]
