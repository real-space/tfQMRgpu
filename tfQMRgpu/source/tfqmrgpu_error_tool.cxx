// This file is part of tfQMRgpu under MIT-License

#ifndef __NO_MAIN__
    // To compile a standalone executable tool that decyphers tfQMRgpu error codes:
    // g++ -g -O0 -Wall -Werror -I include -D __MAIN__ source/tfqmrgpu_error_tool.cxx
    #include <cstdint> // uint32_t
    #include <cstdlib> // std::atoi, int64_t
    #include <cstdio> // std::printf

    #define cudaStream_t int64_t // needed to include tfqmrgpu.h without cuda.h
    #include "tfqmrgpu.h" // declaration of tfqmrgpuPrintError

    #define debug_printf std::printf
    // To compile a standalone executable tool that decyphers tfQMRgpu error codes
    int main(int argc, char **argv) {

        char const *const exe = argv[0]; // name of the executable
        if (argc < 2) {
            std::printf("# %s usage:\n# %s <int error_code>\n\n", __FILE__, exe);
            return 0;
        } else {
            char const *const arg1 = argv[1]; // first argument
            int const error_code = std::atoi(arg1);
            debug_printf("# %s %s --> %i\n", exe, arg1, error_code);
            int const stat = tfqmrgpuPrintError(error_code);
            if (TFQMRGPU_STATUS_SUCCESS != stat) std::printf("# %s tfqmrgpuPrintError returned status %i\n\n", __FILE__, stat);
            return error_code;
        } // argument given

    } // main
#endif // __NO_MAIN__

    char const* tfqmrgpuGetErrorString(tfqmrgpuStatus_t const status) {
        unsigned const nc = 128; // max number of characters
        static char tfqmrgpuErrorString[nc];
        char* str = tfqmrgpuErrorString; // abbreviation
        tfqmrgpuStatus_t stat = status;
        char const key = stat / TFQMRGPU_CODE_CHAR;
        stat -= key * TFQMRGPU_CODE_CHAR;
        uint32_t const line = stat / TFQMRGPU_CODE_LINE;
        stat -= line * TFQMRGPU_CODE_LINE;
        debug_printf("# tfqmrgpuGetErrorString: status= %d, key= \'%c\', line= %d, stat= %i!\n", status, key, line, stat);
        switch (stat) {
            case TFQMRGPU_STATUS_SUCCESS:           for (int c = 0; c < nc; ++c) { str[c] = '\0'; } break; // clear the string
            case TFQMRGPU_STATUS_MAX_ITERATIONS:    std::snprintf(str, nc, "tfQMRgpu: Max number of iterations exceeded!");       break;
            case TFQMRGPU_STATUS_BREAKDOWN:         std::snprintf(str, nc, "tfQMRgpu: All components have broken down!");         break;
            case TFQMRGPU_STATUS_NO_INFO_PASSED:    std::snprintf(str, nc, "tfQMRgpu: getInfo did not provide any information!"); break;
            case TFQMRGPU_POINTER_INVALID:          std::snprintf(str, nc, "tfQMRgpu: Pointer invalid at line %d!",        line); break;
            case TFQMRGPU_STATUS_ALLOCATION_FAILED: std::snprintf(str, nc, "tfQMRgpu: Allocation failed at line %d!",      line); break;
            case TFQMRGPU_STATUS_RANDOM_GEN_FAILED: std::snprintf(str, nc, "tfQMRgpu: Random number generation line %d!",  line); break;
            case TFQMRGPU_NO_IMPLEMENTATION:        std::snprintf(str, nc, "tfQMRgpu: Missing implementation at line %d!", line); break;
            case TFQMRGPU_UNDOCUMENTED_ERROR:       std::snprintf(str, nc, "tfQMRgpu: Undocumented error at line %d!",     line); break;
            case TFQMRGPU_STATUS_LAUNCH_FAILED:     std::snprintf(str, nc, "tfQMRgpu: Device launch failed at line %d!",   line); break;
            case TFQMRGPU_DATALAYOUT_UNKNOWN:       std::snprintf(str, nc, "tfQMRgpu: Unknown data layout \'0x%2.2x\'!",   line); break;
            case TFQMRGPU_B_IS_NOT_SUBSET_OF_X:     std::snprintf(str, nc, "tfQMRgpu: B is not a subset of X in row %d!",  line); break;
            case TFQMRGPU_B_HAS_A_ZERO_COLUMN:      std::snprintf(str, nc, "tfQMRgpu: B has %d zero columns, will break!", line); break;
            case TFQMRGPU_BLOCKSIZE_MISSING:        std::snprintf(str, nc, "tfQMRgpu: Missing blocksize %d x %d!",         int(key),   line); break;
            case TFQMRGPU_TANSPOSITION_UNKNOWN:     std::snprintf(str, nc, "tfQMRgpu: Unknown transposition \'%c\' at line %d!",  key, line); break;
            case TFQMRGPU_VARIABLENAME_UNKNOWN:     std::snprintf(str, nc, "tfQMRgpu: Unknown variable name \'%c\' at line %d!",  key, line); break;
            case TFQMRGPU_PRECISION_MISSMATCH:      std::snprintf(str, nc, "tfQMRgpu: Missmatch in precision \'%c\' at line %d!", key, line); break;
            default:                                std::snprintf(str, nc, "tfQMRgpu: Unknown status= %d at line %d, key \'%c\', stat= %d!",
                                                                                              status, line, (key > 31)?key:'?',  stat); break;
        } // switch status
        return str;
    } // tfqmrgpuGetErrorString

    tfqmrgpuStatus_t tfqmrgpuPrintError(tfqmrgpuStatus_t const status) {
        std::fflush(stdout);
        if (TFQMRGPU_STATUS_SUCCESS == status) {
            debug_printf("# tfQMRgpu: Success!\n");
        } else {
            std::printf("\n%s\n\n", tfqmrgpuGetErrorString(status));
        }
        std::fflush(stdout);
        return TFQMRGPU_STATUS_SUCCESS;
    } // printError

