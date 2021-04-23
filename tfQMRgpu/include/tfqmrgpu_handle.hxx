#pragma once

typedef struct {
    cudaStream_t streamId; // use setStream and getStream for access
} tfq_handle_t;
