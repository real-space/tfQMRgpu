#pragma once
// This file is part of tfQMRgpu under MIT-License

typedef struct {
    cudaStream_t streamId; // use setStream and getStream for access
} tfq_handle_t;
