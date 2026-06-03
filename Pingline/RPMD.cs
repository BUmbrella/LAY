using System.IO;
using LAY.Project.Models;
using LAY.Project.Services;
using OpenCvSharp;

namespace LAY.Pingline
{
    // RPMD 是文件名最后一段为 R 时调用的 pipeline。
    // 它返回 PipelineService 自己的 PipelineResult，不和 BLT 的 PinglineResult 混用。
    public class RPMD
    {
        private static readonly RPMD _instance = new RPMD();

        public static RPMD Instance
        {
            get
            {
                return _instance;
            }
        }

        private readonly PipelineService _pipeline;

        private RPMD()
        {
            bool preferGpu = true;

            string yoloOnnx = Path.Combine(AppContext.BaseDirectory, "onnx", "yolo", "best_ROI.onnx");
            
            string unetOnnx = Path.Combine(AppContext.BaseDirectory, "onnx", "unet", "best_epoch_weights.onnx");

            _pipeline = new PipelineService(yoloOnnx, unetOnnx, preferGpu);
        }

        // R pipeline 的统一入口。
        // PipelineService.Run 返回 PipelineResult，里面有 Success、Error、FullOverlay、Measurements。
        public PipelineResult Process(Mat image)
        {
            if (image == null || image.Empty())
            {
                PipelineResult emptyResult = new PipelineResult();
                emptyResult.Success = false;
                emptyResult.Error = "image is null or empty";
                return emptyResult;
            }

            return _pipeline.Run(image);
        }
    }
}
