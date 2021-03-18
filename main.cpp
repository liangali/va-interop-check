#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include <new>
#include <stdlib.h>
#include <assert.h>

#include <iostream>
#include <string>
#include <sstream>
#include <utility>
#include <vector>
#include <fstream>
#include <memory>
#include <iomanip>

#include "va/va.h"
#include "va/va_drm.h"
#include "video.h"

#include <inference_engine.hpp>
#include <ie_compound_blob.h>
#include <gpu/gpu_context_api_va.hpp>

using namespace InferenceEngine;

#define CHECK_VASTATUS(va_status,func)                                     \
if (va_status != VA_STATUS_SUCCESS) {                                      \
    fprintf(stderr,"%s:%s (%d) failed, exit\n", __func__, func, __LINE__); \
    exit(1);                                                               \
}

VADisplay va_dpy = NULL;
int drm_fd = -1;
const size_t batch_size = 2;
const std::string device_name = "DNNL";
const std::string input_model = "/home/fresh/data/work/va_solution_innersource/source/intel-visual-analytics/assets/models/resnet_v1.5_50_i8.xml";

void setBatchSize(CNNNetwork& network, size_t batch) {
    ICNNNetwork::InputShapes inputShapes = network.getInputShapes();
    for (auto& shape : inputShapes) {
        auto& dims = shape.second;
        if (dims.empty()) {
            throw std::runtime_error("Network's input shapes have empty dimensions");
        }
        dims[0] = batch;
    }
    network.reshape(inputShapes);
}

int initVA()
{
    VADisplay disp;
    VAStatus va_res = VA_STATUS_SUCCESS;
    int major_version = 0, minor_version = 0;

    int adapter_num = 0;
    char adapterpath[256];
    snprintf(adapterpath,sizeof(adapterpath),"/dev/dri/renderD%d", adapter_num + 128);

    drm_fd = open(adapterpath, O_RDWR);
    if (drm_fd < 0) {
        printf("ERROR: failed to open adapter in %s\n", adapterpath);
        return -1;
    }
    printf("INFO: drm_fd = 0x%08x\n", drm_fd);

    va_dpy = vaGetDisplayDRM(drm_fd);
    if (!va_dpy) {
        close(drm_fd);
        drm_fd = -1;
        printf("ERROR: failed in vaGetDisplayDRM\n");
        return -1;
    }
    printf("INFO: drm_fd = 0x%08lx\n", (uint64_t)va_dpy);

    va_res = vaInitialize(va_dpy, &major_version, &minor_version);
    if (VA_STATUS_SUCCESS != va_res) {
        close(drm_fd);
        drm_fd = -1;
        printf("ERROR: failed in vaInitialize with err = %d\n", va_res);
        return -1;
    }
    printf("INFO: vaInitialize done\n");

    return 0;
}

void closeVA()
{
    vaTerminate(va_dpy);
    printf("INFO: vaInitialize done\n");

    if (drm_fd < 0)
        return;

    close(drm_fd);
    drm_fd = -1;
    printf("INFO: drm_fd closed\n");
}

int decodeFrame(VASurfaceID& frame)
{
    VAEntrypoint entrypoints[5];
    int num_entrypoints, vld_entrypoint;
    VAConfigAttrib attrib;
    VAConfigID config_id;
    VASurfaceID surface_id[RT_NUM];
    VAContextID context_id;
    VABufferID pic_param_buf, iqmatrix_buf, slice_param_buf, slice_data_buf;
    int major_ver, minor_ver;
    VAStatus va_status;
    int putsurface=0;

    va_status = vaQueryConfigEntrypoints(va_dpy, VAProfileH264Main, entrypoints, 
                             &num_entrypoints);
    CHECK_VASTATUS(va_status, "vaQueryConfigEntrypoints");

    for	(vld_entrypoint = 0; vld_entrypoint < num_entrypoints; vld_entrypoint++) {
        if (entrypoints[vld_entrypoint] == VAEntrypointVLD)
            break;
    }
    if (vld_entrypoint == num_entrypoints) {
        printf("ERROR: not find AVC VLD entry point\n");
        return -1;
    }

    /* find out the format for the render target */
    attrib.type = VAConfigAttribRTFormat;
    vaGetConfigAttributes(va_dpy, VAProfileH264Main, VAEntrypointVLD, &attrib, 1);
    if ((attrib.value & VA_RT_FORMAT_YUV420) == 0) {
        printf("ERROR: not find desired YUV420 RT format\n");
        return -1;
    }

    va_status = vaCreateConfig(va_dpy, VAProfileH264Main, VAEntrypointVLD, &attrib, 1, &config_id);
    CHECK_VASTATUS(va_status, "vaQueryConfigEntrypoints");

    for (size_t i = 0; i < RT_NUM; i++)
    {
        va_status = vaCreateSurfaces(va_dpy, VA_RT_FORMAT_YUV420, CLIP_WIDTH, CLIP_HEIGHT,  &surface_id[i], 1, NULL, 0 );
        CHECK_VASTATUS(va_status, "vaCreateSurfaces");
    }

    /* Create a context for this decode pipe */
    va_status = vaCreateContext(va_dpy, config_id, CLIP_WIDTH, ((CLIP_HEIGHT+15)/16)*16, VA_PROGRESSIVE, surface_id, RT_NUM, &context_id);
    CHECK_VASTATUS(va_status, "vaCreateContext");

    va_status = vaCreateBuffer(va_dpy, context_id, VAPictureParameterBufferType, pic_size, 1, &pic_param, &pic_param_buf);
    CHECK_VASTATUS(va_status, "vaCreateBuffer type = VAPictureParameterBufferType");
    va_status = vaCreateBuffer(va_dpy, context_id, VAIQMatrixBufferType, iq_size, 1, &iq_matrix, &iqmatrix_buf );
    CHECK_VASTATUS(va_status, "vaCreateBuffer type = VAIQMatrixBufferType");
    va_status = vaCreateBuffer(va_dpy, context_id, VASliceParameterBufferType, slc_size, 1, &slc_param, &slice_param_buf);
    CHECK_VASTATUS(va_status, "vaCreateBuffer type = VASliceParameterBufferType");
    va_status = vaCreateBuffer(va_dpy, context_id, VASliceDataBufferType, bs_size, 1, bs_data, &slice_data_buf);
    CHECK_VASTATUS(va_status, "vaCreateBuffer type = VASliceDataBufferType");

    /* send decode workload to GPU */
    va_status = vaBeginPicture(va_dpy, context_id, surface_id[RT_ID]);
    CHECK_VASTATUS(va_status, "vaBeginPicture");
    va_status = vaRenderPicture(va_dpy,context_id, &pic_param_buf, 1);
    CHECK_VASTATUS(va_status, "vaRenderPicture");
    va_status = vaRenderPicture(va_dpy,context_id, &iqmatrix_buf, 1);
    CHECK_VASTATUS(va_status, "vaRenderPicture");
    va_status = vaRenderPicture(va_dpy,context_id, &slice_param_buf, 1);
    CHECK_VASTATUS(va_status, "vaRenderPicture");
    va_status = vaRenderPicture(va_dpy,context_id, &slice_data_buf, 1);
    CHECK_VASTATUS(va_status, "vaRenderPicture");
    va_status = vaEndPicture(va_dpy,context_id);
    CHECK_VASTATUS(va_status, "vaEndPicture");

    va_status = vaSyncSurface(va_dpy, surface_id[RT_ID]);
    CHECK_VASTATUS(va_status, "vaSyncSurface");

    frame = surface_id[RT_ID];

    if (1)
    {
        VAImage output_image;
        va_status = vaDeriveImage(va_dpy, surface_id[RT_ID], &output_image);
        CHECK_VASTATUS(va_status, "vaDeriveImage");

        void *out_buf = nullptr;
        va_status = vaMapBuffer(va_dpy, output_image.buf, &out_buf);
        CHECK_VASTATUS(va_status, "vaMapBuffer");

        FILE *fp = fopen("out_224x224.nv12", "wb+");
        // dump y_plane
        char *y_buf = (char*)out_buf;
        int y_pitch = output_image.pitches[0];
        for (size_t i = 0; i < CLIP_HEIGHT; i++)
        {
            fwrite(y_buf + y_pitch*i, CLIP_WIDTH, 1, fp);
        }
        // dump uv_plane
        char *uv_buf = (char*)out_buf + output_image.offsets[1];
        int uv_pitch = output_image.pitches[1];
        for (size_t i = 0; i < CLIP_HEIGHT/2; i++)
        {
            fwrite(uv_buf + uv_pitch*i, CLIP_WIDTH, 1, fp);
        }

        // use below command line to convert nv12 surface as bmp image
        /* ffmpeg -s 224x224 -pix_fmt nv12 -f rawvideo -i out.nv12 out.bmp */

        fclose(fp);
    }

    vaDestroyConfig(va_dpy,config_id);
    vaDestroyContext(va_dpy,context_id);

    return 0;
}

int main (int argc, char **argv) 
{
    if (initVA()) {
        printf("ERROR: initVA failed!\n");
        return -1;
    }

    VASurfaceID va_frame;
    if(decodeFrame(va_frame)) {
        printf("ERROR: decode failed\n");
        return -1;
    }

    Core ie;
    CNNNetwork network = ie.ReadNetwork(input_model);
    setBatchSize(network, batch_size);

    // set input info
    if (network.getInputsInfo().empty()) {
        std::cerr << "Network inputs info is empty" << std::endl;
        return -1;
    }
    InputInfo::Ptr input_info = network.getInputsInfo().begin()->second;
    std::string input_name = network.getInputsInfo().begin()->first;
    input_info->setLayout(Layout::NCHW);
    input_info->setPrecision(Precision::U8);
    input_info->getPreProcess().setColorFormat(ColorFormat::NV12);
    InferenceEngine::InputsDataMap inputInfo(network.getInputsInfo());
    auto& inputInfoFirst = inputInfo.begin()->second;
    const InferenceEngine::SizeVector inputDims = inputInfoFirst->getTensorDesc().getDims();
    size_t inputBatch = inputDims[0];
    size_t inputChannel = inputDims[1];
    size_t inputWidth = inputDims[3];
    size_t inputHeight = inputDims[2];
    printf("INFO: input_name = %s, input_batch = %ld, input_channel = %ld, input_width = %ld, input_height = %ld\n", 
        input_name.c_str(), inputBatch, inputChannel, inputWidth, inputHeight);

    // set output info
    if (network.getOutputsInfo().empty()) {
        std::cerr << "Network outputs info is empty" << std::endl;
        return -1;
    }
    DataPtr output_info = network.getOutputsInfo().begin()->second;
    std::string output_name = network.getOutputsInfo().begin()->first;
    output_info->setPrecision(Precision::FP32);
    InferenceEngine::OutputsDataMap outputInfo(network.getOutputsInfo());
    auto& _output = outputInfo.begin()->second;
    const InferenceEngine::SizeVector outputDims = _output->getTensorDesc().getDims();
    size_t outputSize = outputDims[1];
    printf("INFO: output_name = %s, outputSize = %ld\n", output_name.c_str(), outputSize);

    auto shared_va_context = gpu::make_shared_context(ie, device_name, va_dpy);
    ExecutableNetwork executable_network = ie.LoadNetwork(network, shared_va_context);
    InferRequest infer_request = executable_network.CreateInferRequest();

    VASurfaceID va_frame1 = va_frame;
    VASurfaceID va_frame2 = 0; // VASurfaceID=0 is valid decode RT, although empty conent, just use it for test. 

    std::vector<Blob::Ptr> blobs;
    auto image1 = gpu::make_shared_blob_nv12(CLIP_HEIGHT, CLIP_WIDTH, shared_va_context, va_frame1); 
    auto image2 = gpu::make_shared_blob_nv12(CLIP_HEIGHT, CLIP_WIDTH, shared_va_context, va_frame2); 
    blobs.push_back(image1);
    blobs.push_back(image2);
    auto batchedBlob = make_shared_blob<BatchedBlob>(blobs);
    infer_request.SetBlob(input_name, batchedBlob);

#ifndef ENABLE_ASYNC
    infer_request.Infer();
#else
    infer_request.StartAsync();
    infer_request.Wait(InferenceEngine::IInferRequest::RESULT_READY);
#endif

    float *result = infer_request.GetBlob(output_name)->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
    for (int i = 0; i < inputBatch; i++) {
        int c = -1;
        float conf = 0;
        for (int j = 0; j < outputSize; j ++) {
            if (result[j] > conf) {
                c = j;
                conf = result[j];
            }
        }
        result += outputSize;
        printf("INFO: Resnet50 inference output: batchIndex = %d, classID = %d, confidence = %f\n", i, c, conf);
    }

    closeVA();
    printf("done!\n");
    return 0;
}
