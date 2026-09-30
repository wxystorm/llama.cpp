#ifndef GGML_OPENCL_H
#define GGML_OPENCL_H

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

//
// backend API
//
GGML_BACKEND_API ggml_backend_t ggml_backend_opencl_init(void);
GGML_BACKEND_API bool ggml_backend_is_opencl(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_opencl_buffer_type(void);
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_opencl_host_buffer_type(void);

#define GGML_BACKEND_OPENCL_GET_TENSOR_BATCH3_PROC \
    "ggml_backend_opencl_get_tensor_batch3"

typedef bool (*ggml_backend_opencl_get_tensor_batch3_t)(
        const struct ggml_tensor * tensor0,
        void * data0,
        size_t offset0,
        size_t size0,
        const struct ggml_tensor * tensor1,
        void * data1,
        size_t offset1,
        size_t size1,
        const struct ggml_tensor * tensor2,
        void * data2,
        size_t offset2,
        size_t size2);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_opencl_reg(void);

#ifdef  __cplusplus
}
#endif

#endif // GGML_OPENCL_H
