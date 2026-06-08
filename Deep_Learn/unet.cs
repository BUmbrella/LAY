using System.Diagnostics;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System.IO;

namespace LAY.Deep_Learn
{
    /// <summary>
    /// UNet ONNX 语义分割推理器。
    ///
    /// 为什么需要这个类：UNet 和 YOLO 的任务不一样。YOLO 输出的是目标框，UNet 输出的是每个像素的类别。
    /// 所以 UNet 推理流程的核心不是 NMS，而是把模型输出的像素级分数解析成一张 mask。
    ///
    /// 完整流程：
    /// 1. 加载 onnx/unet/best.onnx。
    /// 2. 把输入图像统一成模型训练时接近的格式。
    /// 3. resize 到 ONNX 模型要求的输入尺寸。
    /// 4. Mat -> [1,3,H,W] Tensor。
    /// 5. 调用 ONNX Runtime 前向推理。
    /// 6. 识别输出 Tensor 的布局。
    /// 7. 对每个像素做 argmax 或阈值判断，生成前景/背景 mask。
    /// 8. 把 mask resize 回原图大小，方便后续 OpenCV 测量或显示。
    /// </summary>
    public class unet : IDisposable
    {
        // 使用单例。为什么这么做：ONNX 模型加载很重，如果每次分割都 new 一个 session，速度会慢很多，显存/内存也会反复申请释放。
        private static readonly unet _instance = new unet();

        // 推理锁。为什么这么做：这个类是单例，多个地方可能同时调用；加锁可以避免共享 session 和 Mat 中间对象在并发时出现难排查的问题。
        private static readonly object _predictLock = new object();

        public static unet Instance
        {
            get
            {
                return _instance;
            }
        }

        // ONNX Runtime 的模型会话。为什么保存成字段：模型只需要加载一次，后续每张图直接复用这个 session 做推理。
        private readonly InferenceSession _session;

        // 模型输入名。为什么不写死：不同导出脚本可能把输入叫 input、images、input.1；从模型元数据读取更稳。
        private readonly string _inputName;

        // 模型输入宽高。为什么从模型里读：UNet 通常固定输入尺寸，尺寸不一致会导致推理失败或结果错位。
        private readonly int _inputWidth;
        private readonly int _inputHeight;

        private unet()
        {
            // 拼出模型路径。为什么用 AppContext.BaseDirectory：发布后的 exe 运行目录才是程序真正找模型的基准目录。
            string unetModelPath = Path.Combine(
                AppContext.BaseDirectory,
                "onnx",
                "unet",
                "best.onnx"
            );

            // 创建推理会话。为什么做 CUDA fallback：有 GPU 就加速，没有 GPU 也不让程序直接不可用。
            _session = CreateSessionWithFallback(unetModelPath);

            // 读取输入节点名。为什么需要：ONNX Runtime 喂输入时必须使用模型定义的输入名，否则 Run 找不到输入。
            _inputName = _session.InputMetadata.Keys.First();

            // 读取输入维度。常见是 [1,3,H,W]。
            // 为什么 fallback 到 256：有些 ONNX 是动态尺寸，元数据可能是 -1；给一个默认值可以让代码继续运行。
            var inputDims = _session.InputMetadata[_inputName].Dimensions;
            _inputHeight = inputDims.Length >= 4 && inputDims[2] > 0 ? inputDims[2] : 256;
            _inputWidth = inputDims.Length >= 4 && inputDims[3] > 0 ? inputDims[3] : 256;
        }

        private static InferenceSession CreateSessionWithFallback(string modelPath)
        {
            try
            {
                var gpuOptions = new SessionOptions();

                // 添加 CUDA 执行器。为什么这么做：UNet 是像素级推理，计算量通常比普通分类大，用 GPU 可以明显加速。
                gpuOptions.AppendExecutionProvider_CUDA(0);

                Debug.WriteLine("UNet using CUDA execution provider.");
                return new InferenceSession(modelPath, gpuOptions);
            }
            catch (Exception ex)
            {
                // CUDA 不可用时回退 CPU。为什么这么做：部署机器可能没有显卡或 CUDA 版本不匹配，CPU fallback 可以提高程序健壮性。
                Debug.WriteLine($"UNet CUDA unavailable, fallback to CPU: {ex.Message}");
                return new InferenceSession(modelPath, new SessionOptions());
            }
        }

        /// <summary>
        /// 返回和原图同尺寸的二值分割图，前景为 255，背景为 0。
        ///
        /// 为什么用 argmax：训练代码使用 CrossEntropyLoss 时，模型通常输出每个类别的 logit 分数；推理时应该选分数最大的类别。
        /// 为什么输出 0/255：OpenCV 二值图习惯用 0 表示背景、255 表示前景，后续找轮廓、面积计算、形态学操作都方便。
        /// </summary>
        public Mat Predict(Mat image)
        {
            // 空图直接返回空 Mat。为什么这么做：后面的 Resize、Tensor 转换、ONNX 推理都不能处理空图。
            if (image == null || image.Empty())
                return new Mat();

            lock (_predictLock)
            {
                // 把输入统一成 BGR 8-bit 三通道。为什么需要这一步：ImgToTensor 用 Vec3b 读取像素，要求每个像素正好 3 个 byte；如果是灰度、BGRA、16 位或 float 图，读取方式和模型训练输入都会不匹配。
                using Mat input = EnsureBgr8(image);
                if (input.Empty())
                    return new Mat();

                // resize 到模型输入尺寸。为什么需要这一步：UNet 的 ONNX 输入通常固定 H/W；如果直接传原图，尺寸不匹配会推理失败。
                using Mat resized = ResizeImage(input, _inputWidth, _inputHeight);
                if (resized.Empty())
                    return new Mat();

                // Mat 转 Tensor。为什么需要这一步：ONNX 模型不能直接读取 OpenCV Mat，必须给它 NCHW float 数组，并且数值范围要接近训练时的输入。
                DenseTensor<float> tensor = ImgToTensor(resized);

                IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results;
                try
                {
                    // 执行 ONNX 推理。为什么需要这一步：前面只是预处理，真正的神经网络计算在 _session.Run 中完成。
                    results = _session.Run(new[]
                    {
                        NamedOnnxValue.CreateFromTensor(_inputName, tensor)
                    });
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"UNet inference failed: {ex}");
                    return new Mat();
                }

                using (results)
                {
                    // 取第一个输出。为什么这么做：分割模型一般只有一个主输出，它保存每个像素的类别分数。
                    DisposableNamedOnnxValue? firstOutput = results.FirstOrDefault();
                    if (firstOutput == null)
                        return new Mat();

                    Tensor<float> output = firstOutput.AsTensor<float>();

                    // 判断输出布局。为什么需要这一步：不同导出方式可能是 NCHW、NHWC、NHW 或 HW；布局读错会把通道当坐标或把坐标当通道，mask 会完全错误。
                    if (!TryGetUnetOutputLayout(output, out int channels, out int maskHeight, out int maskWidth, out OutputLayout layout))
                        return new Mat();

                    // 转成数组。为什么这么做：后处理要逐像素读取分数，float[] 加上 ReadOutput 的索引公式更直接。
                    float[] buffer = output.ToArray();

                    // 根据每个像素的类别分数生成小尺寸 mask。为什么叫 maskSmall：它的尺寸是模型输出尺寸，不一定等于原图尺寸。
                    using Mat maskSmall = BuildMaskByArgmax(buffer, channels, maskHeight, maskWidth, layout);

                    // resize 回原图尺寸。为什么用 Nearest：mask 是离散类别图，用线性插值会产生 0~255 的中间灰度，破坏二值含义。
                    Mat mask = new Mat();
                    Cv2.Resize(maskSmall, mask, new Size(image.Width, image.Height), 0, 0, InterpolationFlags.Nearest);
                    return mask;
                }
            }
        }

        /// <summary>
        /// 返回 class1 的概率图，和原图同尺寸，类型为 CV_32FC1，范围 0~1。
        ///
        /// 为什么要单独提供概率图：二值 mask 只告诉你前景/背景，但概率图能告诉你模型有多确定。
        /// 它适合调阈值、做热力图、检查边缘不确定区域，或者后续根据概率做更细的规则处理。
        /// </summary>
        public Mat PredictProbability(Mat image)
        {
            if (image == null || image.Empty())
                return new Mat();

            lock (_predictLock)
            {
                // 和 Predict 一样，先统一输入格式。为什么这么做：保证概率图和二值 mask 使用完全一致的预处理，便于对比。
                using Mat input = EnsureBgr8(image);
                if (input.Empty())
                    return new Mat();

                using Mat resized = ResizeImage(input, _inputWidth, _inputHeight);
                DenseTensor<float> tensor = ImgToTensor(resized);

                IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results;
                try
                {
                    results = _session.Run(new[]
                    {
                        NamedOnnxValue.CreateFromTensor(_inputName, tensor)
                    });
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"UNet inference failed: {ex}");
                    return new Mat();
                }

                using (results)
                {
                    DisposableNamedOnnxValue? firstOutput = results.FirstOrDefault();
                    if (firstOutput == null)
                        return new Mat();

                    Tensor<float> output = firstOutput.AsTensor<float>();
                    if (!TryGetUnetOutputLayout(output, out int channels, out int maskHeight, out int maskWidth, out OutputLayout layout))
                        return new Mat();

                    float[] buffer = output.ToArray();

                    // 构建 class1 概率图。为什么不是直接 argmax：概率图要保留连续置信度，所以多通道用 softmax，单通道用 sigmoid/原概率归一。
                    using Mat probSmall = BuildClass1Probability(buffer, channels, maskHeight, maskWidth, layout);

                    // 概率图是连续值，resize 时用 Linear。为什么不用 Nearest：概率不是离散类别，线性插值能让概率过渡更平滑。
                    Mat prob = new Mat();
                    Cv2.Resize(probSmall, prob, new Size(image.Width, image.Height), 0, 0, InterpolationFlags.Linear);
                    return prob;
                }
            }
        }

        private DenseTensor<float> ImgToTensor(Mat img)
        {
            // 再次确保 BGR8。为什么这里还要做：即使调用方忘记预处理，这个函数也能保护自己，避免错误格式进入像素读取循环。
            using Mat safe = EnsureBgr8(img);

            int h = safe.Rows;
            int w = safe.Cols;
            int hw = h * w;

            // 准备 CHW 数组。为什么不是 HWC：PyTorch/UNet ONNX 常见输入是 [N,C,H,W]，通道维在前。
            float[] chw = new float[3 * hw];

            for (int y = 0; y < h; y++)
            {
                int rowOffset = y * w;

                for (int x = 0; x < w; x++)
                {
                    int dstIdx = rowOffset + x;
                    Vec3b pixel = safe.At<Vec3b>(y, x);

                    // BGR -> RGB，并除以 255。为什么需要：OpenCV 默认 BGR，但多数训练代码用 RGB；归一化到 0~1 是为了和训练时的输入尺度一致。
                    chw[dstIdx] = pixel[2] / 255.0f;
                    chw[hw + dstIdx] = pixel[1] / 255.0f;
                    chw[2 * hw + dstIdx] = pixel[0] / 255.0f;
                }
            }

            return new DenseTensor<float>(chw, new[] { 1, 3, h, w });
        }

        private static Mat ResizeImage(Mat image, int width, int height)
        {
            Mat resized = new Mat();

            // 直接缩放到模型输入尺寸。为什么用 Linear：这是普通图像缩放，双线性插值比最近邻更平滑，能减少输入锯齿。
            Cv2.Resize(image, resized, new Size(width, height), 0, 0, InterpolationFlags.Linear);
            return resized;
        }

        private static Mat EnsureBgr8(Mat image)
        {
            if (image == null || image.Empty())
                throw new ArgumentException("image is null or empty.");

            Mat source = image;
            Mat? converted = null;

            // 如果不是 8-bit，就转成 CV_8U。为什么需要：后面用 Vec3b 读取，一个通道必须是 byte；非 8 位图用 Vec3b 读会语义错误。
            if (image.Depth() != MatType.CV_8U)
            {
                converted = new Mat();
                image.ConvertTo(converted, MatType.CV_8U);
                source = converted;
            }

            try
            {
                int channels = source.Channels();
                if (channels == 1)
                {
                    // 灰度转 BGR。为什么需要：模型输入固定 3 通道，灰度图只有 1 通道，必须复制成 3 通道才能组成 [1,3,H,W]。
                    return source.CvtColor(ColorConversionCodes.GRAY2BGR);
                }

                if (channels == 3)
                {
                    // 三通道直接 Clone。为什么 Clone：返回一份独立 Mat，避免外部图像生命周期或后续修改影响当前推理。
                    return source.Clone();
                }

                if (channels == 4)
                {
                    // BGRA 转 BGR。为什么去掉 Alpha：模型训练一般没有透明度通道，保留 4 通道会和 [1,3,H,W] 输入不匹配。
                    return source.CvtColor(ColorConversionCodes.BGRA2BGR);
                }

                throw new InvalidOperationException("Unsupported Mat channels: " + source.Channels() + ", type: " + source.Type());
            }
            finally
            {
                if (converted != null)
                {
                    converted.Dispose();
                }
            }
        }

        private static bool TryGetUnetOutputLayout(Tensor<float> output, out int channels, out int height, out int width, out OutputLayout layout)
        {
            channels = 0;
            height = 0;
            width = 0;
            layout = OutputLayout.NCHW;

            ReadOnlySpan<int> dims = output.Dimensions;

            if (dims.Length == 4)
            {
                // 判断 NCHW。为什么用 channels <= 256：类别通道通常很小，而 H/W 通常较大，用这个经验规则区分通道维和空间维。
                if (dims[1] > 0 && dims[2] > 0 && dims[3] > 0 && dims[1] <= 256)
                {
                    channels = dims[1];
                    height = dims[2];
                    width = dims[3];
                    layout = OutputLayout.NCHW;
                    return true;
                }

                // 判断 NHWC。为什么也要支持：不同框架或导出设置可能把通道放在最后，如果只支持 NCHW，会导致这类模型输出无法解析。
                if (dims[1] > 0 && dims[2] > 0 && dims[3] > 0 && dims[3] <= 256)
                {
                    channels = dims[3];
                    height = dims[1];
                    width = dims[2];
                    layout = OutputLayout.NHWC;
                    return true;
                }
            }

            if (dims.Length == 3 && dims[1] > 0 && dims[2] > 0)
            {
                // [1,H,W] 单通道。为什么 channels=1：输出本身没有类别维，只能按单通道前景概率/logit 解析。
                channels = 1;
                height = dims[1];
                width = dims[2];
                layout = OutputLayout.NHW;
                return true;
            }

            if (dims.Length == 2 && dims[0] > 0 && dims[1] > 0)
            {
                // [H,W] 单通道。为什么支持：有些导出会去掉 batch/channel 维，代码兼容后更不容易因模型导出差异失败。
                channels = 1;
                height = dims[0];
                width = dims[1];
                layout = OutputLayout.HW;
                return true;
            }

            return false;
        }

        private static Mat BuildMaskByArgmax(float[] buffer, int channels, int height, int width, OutputLayout layout)
        {
            // 创建单通道二值图。为什么用 CV_8UC1：OpenCV 的 mask、找轮廓、形态学操作通常都用 8 位单通道图。
            Mat mask = new(height, width, MatType.CV_8UC1, Scalar.All(0));

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    byte value;

                    if (channels == 1)
                    {
                        float score = ReadOutput(buffer, 0, y, x, channels, height, width, layout);

                        // 单通道用 0.5 阈值。为什么这么做：单通道通常表示前景概率，超过 0.5 就认为前景更可能。
                        value = score > 0.5f ? (byte)255 : (byte)0;
                    }
                    else
                    {
                        int bestClass = 0;
                        float bestScore = float.MinValue;

                        // 多通道做 argmax。为什么这么做：CrossEntropyLoss 训练的输出是每类一个分数，分数最大的类别就是模型判断的像素类别。
                        for (int c = 0; c < channels; c++)
                        {
                            float score = ReadOutput(buffer, c, y, x, channels, height, width, layout);
                            if (score > bestScore)
                            {
                                bestScore = score;
                                bestClass = c;
                            }
                        }

                        // class 0 当背景，其它类当前都当前景。为什么这么做：二值 mask 只需要区分背景/目标，不关心更多类别细分。
                        value = bestClass == 0 ? (byte)0 : (byte)255;
                    }

                    mask.Set(y, x, value);
                }
            }

            return mask;
        }

        private static Mat BuildClass1Probability(float[] buffer, int channels, int height, int width, OutputLayout layout)
        {
            // CV_32FC1 保存 0~1 概率。为什么不用 byte：概率是连续小数，byte 会丢失大量置信度信息。
            Mat probability = new(height, width, MatType.CV_32FC1, Scalar.All(0));

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    float value;

                    if (channels == 1)
                    {
                        // 单通道输出可能是概率，也可能是 logit。为什么要 NormalizeScore：兼容不同训练/导出方式。
                        value = NormalizeScore(ReadOutput(buffer, 0, y, x, channels, height, width, layout));
                    }
                    else
                    {
                        float bg = ReadOutput(buffer, 0, y, x, channels, height, width, layout);
                        float fg = ReadOutput(buffer, 1, y, x, channels, height, width, layout);

                        // 双通道用 softmax。为什么这么做：CrossEntropyLoss 输出常是未归一化 logit，需要 softmax 才能解释成概率。
                        value = SoftmaxClass1(bg, fg);
                    }

                    probability.Set(y, x, value);
                }
            }

            return probability;
        }

        private static float ReadOutput(float[] buffer, int channel, int y, int x, int channels, int height, int width, OutputLayout layout)
        {
            if (layout == OutputLayout.NCHW)
            {
                // NCHW 索引公式。为什么这样算：通道是一整张 H*W 平面，先跳到 channel 平面，再跳到 y 行 x 列。
                return buffer[channel * height * width + y * width + x];
            }

            if (layout == OutputLayout.NHWC)
            {
                // NHWC 索引公式。为什么这样算：每个像素后面紧跟所有类别通道，所以先定位像素，再加 channel。
                return buffer[(y * width + x) * channels + channel];
            }

            if (layout == OutputLayout.NHW)
            {
                // 单通道 NHW 没有 channel 维。为什么忽略 channel：整个输出只有一张图。
                return buffer[y * width + x];
            }

            if (layout == OutputLayout.HW)
            {
                // 单通道 HW 没有 batch/channel 维，直接按二维图索引。
                return buffer[y * width + x];
            }

            return 0f;
        }

        private static float NormalizeScore(float value)
        {
            // 已经在 0~1 就直接返回。为什么这么做：有些模型导出时最后已经带 sigmoid，重复 sigmoid 会改变概率含义。
            if (value >= 0f && value <= 1f)
                return value;

            // 不在 0~1 就认为是 logit，用 sigmoid 转概率。为什么用 sigmoid：二分类单通道 logit 转概率的标准函数就是 sigmoid。
            return 1f / (1f + MathF.Exp(-value));
        }

        private static float SoftmaxClass1(float class0, float class1)
        {
            // 减 max 是数值稳定技巧。为什么需要：直接 Exp(很大的数) 可能溢出，减去最大值不改变 softmax 结果。
            float max = MathF.Max(class0, class1);
            float e0 = MathF.Exp(class0 - max);
            float e1 = MathF.Exp(class1 - max);
            return e1 / (e0 + e1 + 1e-6f);
        }

        private enum OutputLayout
        {
            // [N,C,H,W]。为什么常见：PyTorch 默认就是 channel first。
            NCHW,

            // [N,H,W,C]。为什么支持：TensorFlow 和部分 ONNX 导出可能是 channel last。
            NHWC,

            // [N,H,W]。为什么支持：单通道分割可能省掉 channel 维。
            NHW,

            // [H,W]。为什么支持：有些导出甚至省掉 batch 维。
            HW
        }

        public void Dispose()
        {
            // 释放 ONNX Runtime session。为什么需要：session 持有 native 内存/GPU 资源，不释放可能造成资源占用。
            _session.Dispose();
        }
    }
}
