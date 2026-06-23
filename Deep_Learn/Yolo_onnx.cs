using System.Diagnostics;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System.IO;

namespace YOLOv8
{
    /// <summary>
    /// YOLOv8 ONNX 推理器。
    ///
    /// 这个类负责完整的一次目标检测流程：
    /// 1. 加载 best.onnx 模型。
    /// 2. 把 OpenCV 的 Mat 图像预处理成模型需要的 Tensor。
    /// 3. 调用 ONNX Runtime 执行推理。
    /// 4. 解析 YOLO 输出：[中心点 x, 中心点 y, 宽, 高, 各类别置信度...]。
    /// 5. 把 640x640 letterbox 坐标还原回原图坐标。
    /// 6. 使用 NMS 去掉重复框。
    /// 7. 返回 [centerX, centerY, width, height, score, classId]。
    /// </summary>
    public class YoloV8Predictor : IDisposable
    {
        // ---------------- 单例 ----------------
        // 这里使用饿汉式单例：程序启动后第一次访问 Instance 时，整个进程只会创建一个 YoloV8Predictor。
        // 好处：ONNX 模型只加载一次，避免每次检测都重新创建 InferenceSession，速度更快。
        // 注意：如果你后续想支持多个模型路径，就不能再用这种固定单例写法。
        private static readonly YoloV8Predictor _instance = new YoloV8Predictor();

        // Predict 方法的线程锁。
        // 作用：如果多个线程同时调用 Predict，就让它们排队执行，避免 session / OpenCV Mat / 中间数据出现并发问题。
        // 代价：同一时间只能跑一个推理请求，吞吐量会降低。
        private static readonly object _predictLock = new object();

        /// <summary>
        /// 对外暴露的唯一推理器实例。
        /// 使用方式：YoloV8Predictor.Instance.Predict(image)
        /// </summary>
        public static YoloV8Predictor Instance
        {
            get
            {
                return _instance;
            }
        }

        // ONNX Runtime 推理会话。它内部持有模型、执行图、CPU/GPU 执行器等资源。
        private readonly InferenceSession _session;

        // 模型输入节点名称。
        // 不同 ONNX 导出时输入名可能不同，比如 images、input、input.1。
        // 这里从模型元数据里取第一个输入名，避免写死。
        private readonly string _inputName;

        // YOLOv8 常见输入尺寸是 640x640。
        // 训练/导出模型时如果使用了其他 imgsz，这里也应该对应调整。
        private readonly int _inputWidth = 640;
        private readonly int _inputHeight = 640;

        // ---------------- 私有构造函数 ----------------
        // 构造函数是 private，说明外部不能 new YoloV8Predictor，只能通过 Instance 获取单例。
        private YoloV8Predictor()
        {
            // 模型路径：程序运行目录/onnx/yolo/best.onnx
            // AppContext.BaseDirectory 通常是 exe 所在目录，比如 bin/Debug/net...。
            string yoloModelPath = Path.Combine(
                AppContext.BaseDirectory,
                "onnx",
                "yolo",
                "best.onnx"
            );

            // 创建 ONNX Runtime session。
            // 优先尝试 CUDA，如果 CUDA 不可用则自动退回 CPU。
            _session = CreateSessionWithFallback(yoloModelPath);

            // 读取模型第一个输入节点名称，后面 Run 时需要用这个名字喂入 Tensor。
            _inputName = _session.InputMetadata.Keys.First();
        }

        /// <summary>
        /// 创建 ONNX Runtime 推理会话。
        /// 优先使用 CUDA GPU；如果机器没有 CUDA、驱动不匹配、onnxruntime-gpu 不可用，就退回 CPU。
        /// </summary>
        private static InferenceSession CreateSessionWithFallback(string modelPath)
        {
            try
            {
                // SessionOptions 用来配置 ONNX Runtime 的执行方式。
                var gpuOptions = new SessionOptions();

                // 添加 CUDA 执行提供器，参数 0 表示使用第 0 张显卡。
                // 如果项目引用的是 Microsoft.ML.OnnxRuntime.Gpu，并且 CUDA 环境正确，这里会走 GPU。
                gpuOptions.AppendExecutionProvider_CUDA(0);

                Debug.WriteLine("YOLO using CUDA execution provider.");
                return new InferenceSession(modelPath, gpuOptions);
            }
            catch (Exception ex)
            {
                // GPU 初始化失败时，不让程序直接崩溃，而是记录日志并使用 CPU。
                // 常见原因：没有 NVIDIA 显卡、CUDA/cuDNN 版本不匹配、缺少 onnxruntime-gpu 依赖。
                Debug.WriteLine($"YOLO CUDA unavailable, fallback to CPU: {ex.Message}");
                return new InferenceSession(modelPath, new SessionOptions());
            }
        }

        /// <summary>
        /// 把 OpenCV Mat 转成 YOLOv8 ONNX 需要的输入 Tensor。
        ///
        /// 输入 Mat 格式：BGR、8-bit、3 通道，即 OpenCV 常用的 CV_8UC3。
        /// 输出 Tensor 格式：NCHW = [1, 3, H, W]。
        /// 数值范围：0~255 转成 0~1。
        /// 通道顺序：BGR 转 RGB。
        ///
        /// YOLO/PyTorch 通常使用 RGB + CHW 格式，而 OpenCV 默认读进来是 BGR + HWC，
        /// 所以这里做了两个关键转换：
        /// 1. HWC -> CHW。
        /// 2. BGR -> RGB。
        /// </summary>
        DenseTensor<float> ImgToTensorFast(Mat img)
        {
            if (img == null || img.Empty())
                throw new ArgumentException("img is null or empty.");

            // 确保输入一定是 8-bit BGR 三通道，避免灰度图、BGRA 图、非 8 位图导致后面 At<Vec3b> 读取错误。
            using Mat safe = EnsureBgr8(img);

            if (safe.Empty())
                throw new ArgumentException("safe image is empty.");

            if (safe.Type() != MatType.CV_8UC3)
                throw new ArgumentException(
                    $"ImgToTensorFast requires CV_8UC3, actual: {safe.Type()}"
                );

            int h = safe.Rows;
            int w = safe.Cols;
            int hw = h * w;

            // YOLO 输入是 [batch, channel, height, width]。
            // 这里 batch 固定为 1，所以只需要准备 3 * H * W 的一维数组。
            // 数组布局：
            // R 通道：chw[0 ... H*W-1]
            // G 通道：chw[H*W ... 2*H*W-1]
            // B 通道：chw[2*H*W ... 3*H*W-1]
            float[] chw = new float[3 * hw];

            for (int y = 0; y < h; y++)
            {
                int rowOffset = y * w;

                for (int x = 0; x < w; x++)
                {
                    int dstIdx = rowOffset + x;

                    // OpenCV Vec3b 的顺序是 B, G, R。
                    Vec3b pixel = safe.At<Vec3b>(y, x);

                    // 写入 Tensor 时改成 RGB 顺序，并除以 255 归一化到 0~1。
                    chw[dstIdx] = pixel[2] / 255.0f;          // R
                    chw[hw + dstIdx] = pixel[1] / 255.0f;     // G
                    chw[2 * hw + dstIdx] = pixel[0] / 255.0f; // B
                }
            }

            // DenseTensor 维度为 [1, 3, H, W]。
            // 1 表示 batch size；3 表示 RGB 三通道。
            return new DenseTensor<float>(chw, new[] { 1, 3, h, w });
        }

        /// <summary>
        /// 对一张图执行 YOLOv8 目标检测。
        ///
        /// 参数：
        /// image：原始 OpenCV Mat。
        /// confThresh：置信度阈值，低于这个分数的候选框会被直接丢弃。
        /// iouThresh：NMS 阈值，两个框重叠超过这个 IoU 时会认为是重复框。
        ///
        /// 返回：
        /// 每个 float[6] 表示一个检测框：
        /// [0] centerX：框中心点 X，原图坐标。
        /// [1] centerY：框中心点 Y，原图坐标。
        /// [2] width：框宽度，原图尺度。
        /// [3] height：框高度，原图尺度。
        /// [4] score：该类别置信度。
        /// [5] cls：类别 ID。
        /// </summary>
        public float[][] Predict(Mat image, float confThresh = 0.25f, float iouThresh = 0.4f)
        {
            // 空图直接返回空结果。为什么需要这一步：OpenCV 的 Resize、At<Vec3b>、ONNX 推理都不能处理空图；提前返回可以让上层业务把“未检测到”当成正常结果，而不是让程序崩溃。
            if (image == null || image.Empty())
                return Array.Empty<float[]>();

            // 推理加锁：保证同一时间只有一个线程进入完整的预处理、推理、后处理流程。为什么需要这一步：ONNX Runtime session 虽然通常可并发，但这里还混合了 Mat 生命周期和共享单例资源；加锁可以牺牲一点吞吐量，换取推理过程更稳定、排查问题更简单。
            lock (_predictLock)
            {
                // 把输入统一成 BGR 8-bit 三通道。为什么需要这一步：后面的 ImgToTensorFast 使用 Vec3b 逐像素读取，要求图像必须是 8 位、3 通道；如果传入灰度图、BGRA 图或 float 图，通道数/数据类型不一致会导致读取错误，或者模型看到的颜色分布和训练时不一致。
                using Mat input = EnsureBgr8(image);

                if (input.Empty())
                    return Array.Empty<float[]>();

                // 将原图 letterbox 到 640x640。为什么需要这一步：YOLO 模型训练/导出时通常固定输入尺寸是 640x640，直接把原图强行拉伸会改变目标形状；letterbox 保持宽高比，只在边缘补灰边，可以减少形变带来的定位误差。
                // letterbox 的意思是：保持原图宽高比缩放，不足的地方用灰色填充。
                // 这样不会把目标拉伸变形，检测框坐标也可以按 scale/dx/dy 还原回原图
                //
                // 。
                using Mat resized =
                    ResizeImage(input, _inputWidth, _inputHeight, true);

                if (resized.Empty())
                    return Array.Empty<float[]>();

                // 把 640x640 BGR Mat 转成 [1,3,640,640] 的 float Tensor。为什么需要这一步：ONNX 模型不能直接吃 OpenCV 的 Mat，它需要和训练时一致的 NCHW float 输入；如果维度顺序、通道顺序或数值范围错了，模型不会报错但预测会明显变差。
                DenseTensor<float> tensor = ImgToTensorFast(resized);

                IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results;
                try
                {
                    // 执行 ONNX 推理。为什么需要这一步：前面的步骤只是把图像整理成模型能理解的输入，真正的神经网络前向计算发生在 _session.Run；它会输出每个候选框的位置和类别分数。
                    // NamedOnnxValue.CreateFromTensor 的名字必须等于模型输入节点名。
                    // 返回 results，里面通常有一个输出 Tensor。
                    results = _session.Run(new[]
                    {
                        NamedOnnxValue.CreateFromTensor(_inputName, tensor)
                    });
                }
                catch (Exception ex)
                {
                    // 推理失败时记录日志并返回空结果。
                    // 常见原因：模型路径错误、输入尺寸不匹配、CUDA 运行时错误、ONNX 节点不支持等。
                    Debug.WriteLine($"YOLO inference failed: {ex}");
                    return Array.Empty<float[]>();
                }

                // results 持有 unmanaged/native 资源，用 using 确保及时释放。
                using (results)
                {
                    // 取第一个输出。为什么需要这一步：YOLOv8 检测模型一般只有一个检测输出；ONNX Runtime 返回的是输出集合，所以要先拿到第一个 Tensor，再开始解析框坐标和类别分数。
                    // YOLOv8 检测模型一般只有一个主输出。
                    DisposableNamedOnnxValue? firstOutput = results.FirstOrDefault();
                    if (firstOutput == null)
                        return Array.Empty<float[]>();

                    var output = firstOutput.AsTensor<float>();

                    // 判断输出布局。为什么需要这一步：不同 YOLO 导出版本可能把输出排成 [1,通道数,框数量] 或 [1,框数量,通道数]；如果不先识别布局，读取 cx/cy/w/h 和类别分数时会把维度读反，结果框会完全错误。
                    // 常见 YOLOv8 ONNX 输出可能是：
                    // [1, channels, numBoxes]，例如 [1, 6, 8400]
                    // 或 [1, numBoxes, channels]，例如 [1, 8400, 6]
                    // channels = 4 + 类别数。
                    // 前 4 个通道是 cx, cy, w, h；后面是每个类别的置信度。
                    if (!TryGetYoloOutputLayout(output, out int numBoxes, out int channels, out bool channelsFirst))
                        return Array.Empty<float[]>();

                    // YOLOv8 检测输出一般没有单独 objectness，类别分数直接作为该类别置信度。
                    // 所以类别数 = 总通道数 - 4 个框坐标通道。
                    int numClasses = channels - 4;

                    // 转成数组，后面用 ReadOutput 按不同 layout 读取指定 box / channel 的值。为什么需要这一步：Tensor 直接访问不如数组索引直观，这里统一展平成 float[]，再用 ReadOutput 封装不同布局的索引公式，降低后处理写错的概率。
                    float[] buffer = output.ToArray();

                    // 计算 letterbox 的缩放比例和填充偏移。为什么需要这一步：模型输出坐标属于 640x640 的输入图，而业务需要原图坐标；只有知道当时缩放了多少、补了多少灰边，才能把检测框准确还原回原图。
                    // 这和 ResizeImage 里面的逻辑必须保持一致。
                    // scale：原图缩放到 640x640 时使用的比例。
                    // dx/dy：左右/上下填充的灰边大小。
                    float scale = Math.Min(
                        (float)_inputWidth / image.Width,
                        (float)_inputHeight / image.Height
                    );
                    float dx = (_inputWidth - image.Width * scale) / 2;
                    float dy = (_inputHeight - image.Height * scale) / 2;

                    // 存放通过 confThresh 初筛后的候选框。
                    List<Box> boxes = new List<Box>();

                    // 遍历模型输出中的所有候选框。
                    // 以 640x640 YOLOv8 为例，numBoxes 常见是 8400：
                    // 80x80 + 40x40 + 20x20 三个尺度上的候选点总数。
                    for (int i = 0; i < numBoxes; i++)
                    {
                        // 保存当前候选框对每个类别的分数。
                        // 后面的自定义 NMS 会用到 clsScores，而不只是最大类别分数。
                        float[] clsScores = new float[numClasses];

                        float maxCls = 0f;
                        int cls = 0;

                        // 找出当前框分数最高的类别，也就是 argmax(class scores)。
                        for (int j = 0; j < numClasses; j++)
                        {
                            // 第 j 个类别在输出通道中的位置是 j + 4，因为前 4 个是框坐标。
                            float v = ReadOutput(buffer, i, j + 4, numBoxes, channels, channelsFirst);
                            clsScores[j] = v;

                            if (v > maxCls)
                            {
                                maxCls = v;
                                cls = j;
                            }
                        }

                        // 置信度过滤：最高类别分数低于阈值的框直接丢弃。为什么需要这一步：YOLO 会产生大量候选框，其中很多只是背景噪声；先用 confThresh 过滤低分框，可以减少误检，也能减少后面 NMS 的计算量。
                        // confThresh 越高，误检通常越少，但漏检可能增加。
                        if (maxCls < confThresh)
                            continue;

                        // 读取 YOLO 输出的框坐标。
                        // YOLOv8 输出的是中心点格式：cx, cy, w, h，坐标属于 letterbox 后的 640x640 图。
                        float cx = ReadOutput(buffer, i, 0, numBoxes, channels, channelsFirst);
                        float cy = ReadOutput(buffer, i, 1, numBoxes, channels, channelsFirst);
                        float w = ReadOutput(buffer, i, 2, numBoxes, channels, channelsFirst);
                        float h = ReadOutput(buffer, i, 3, numBoxes, channels, channelsFirst);

                        // 中心点格式转成左上角/右下角格式。为什么需要这一步：YOLO 输出是 cx/cy/w/h，适合网络回归；但 IoU/NMS 更容易用 x1/y1/x2/y2 计算交集和面积，所以这里先转换格式。
                        // NMS 计算 IoU 时更适合用 x1,y1,x2,y2。
                        float x1 = cx - w / 2;
                        float y1 = cy - h / 2;
                        float x2 = cx + w / 2;
                        float y2 = cy + h / 2;

                        // 把 640x640 letterbox 坐标还原回原图坐标。
                        // 先减掉灰边 dx/dy，再除以缩放比例 scale。
                        float rx1 = (x1 - dx) / scale;
                        float ry1 = (y1 - dy) / scale;
                        float rx2 = (x2 - dx) / scale;
                        float ry2 = (y2 - dy) / scale;

                        // 坐标裁剪到原图范围内，防止框越界。为什么需要这一步：模型预测可能落在图像边界外，尤其是贴边目标；如果不裁剪，后续画框、裁 ROI 或测量时可能访问非法区域。
                        rx1 = Math.Max(0, Math.Min(image.Width, rx1));
                        ry1 = Math.Max(0, Math.Min(image.Height, ry1));
                        rx2 = Math.Max(0, Math.Min(image.Width, rx2));
                        ry2 = Math.Max(0, Math.Min(image.Height, ry2));

                        boxes.Add(new Box
                        {
                            x1 = rx1,
                            y1 = ry1,
                            x2 = rx2,
                            y2 = ry2,
                            cls = cls,
                            score = maxCls,
                            clsScores = clsScores
                        });
                    }

                    // 没有任何候选框通过置信度阈值，就返回空数组。
                    if (boxes.Count == 0)
                        return Array.Empty<float[]>();

                    // 按置信度从高到低排序。为什么需要这一步：NMS 的基本思想是优先保留最可信的框，再删除和它高度重叠的重复框；如果不排序，低分框可能先占位，把更好的高分框压掉。
                    // NMS 通常先保留高分框，再抑制和它高度重叠的低分框。
                    boxes.Sort(delegate (Box a, Box b)
                    {
                        return b.score.CompareTo(a.score);
                    });

                    // used[i] 表示第 i 个候选框已经被 NMS 抑制。
                    // 为什么需要这个数组：boxes 已经按分数从高到低排序，某个低分框一旦被高分框判定为重复框，就不应该再作为独立结果输出。
                    bool[] used = new bool[boxes.Count];

                    // finalBoxes 存最终保留下来的框。
                    List<Box> finalBoxes = new List<Box>();

                    // 标准 NMS。为什么需要这一步：同一个目标附近通常会有多个预测框，如果全部输出，后续业务会重复计数或重复测量。
                    // 正常流程：
                    // 1. boxes 已经按 score 从高到低排序，所以先遇到的框更可信。
                    // 2. 保留当前最高分框。
                    // 3. 只在同类别框之间计算 IoU 并抑制重复框。
                    // 4. 不再写死 TYPE1/TYPE2，也不让某个类别天然压过另一个类别。
                    //
                    // 为什么只抑制同类别：标准目标检测 NMS 通常是 class-aware NMS。
                    // 例如同一位置可能既有“缺陷A”又有“缺陷B”的候选，类别不同不能简单互相删除；
                    // 是否跨类别互斥应该由业务规则决定，而不是由通用 NMS 默认处理。
                    for (int i = 0; i < boxes.Count; i++)
                    {
                        if (used[i])
                            continue;

                        Box best = boxes[i];
                        finalBoxes.Add(best);

                        for (int j = i + 1; j < boxes.Count; j++)
                        {
                            if (used[j])
                                continue;

                            // 标准 class-aware NMS：不同类别不互相抑制。
                            // 为什么需要这个判断：如果不判断类别，两个重叠但类别不同的目标可能被误删。
                            if (best.cls != boxes[j].cls)
                                continue;

                            // IoU 大于阈值，说明同类别的两个框高度重叠，通常是在预测同一个目标。
                            // 因为 best 分数更高，所以抑制当前较低分的 boxes[j]。
                            if (IoU(best, boxes[j]) > iouThresh)
                            {
                                used[j] = true;
                            }
                        }
                    }

                    // 把内部 Box 结构转换成调用方需要的 float[6]。为什么需要这一步：内部用 x1/y1/x2/y2 是为了方便 NMS 和 IoU，外部业务可能习惯中心点+宽高格式，所以最后统一转换成调用方约定的数据结构。
                    List<float[]> resultBoxes = new List<float[]>();
                    foreach (Box b in finalBoxes)
                    {
                        float[] resultBox = new float[6];

                        // 返回中心点格式，而不是 x1/y1/x2/y2。
                        resultBox[0] = (b.x1 + b.x2) / 2; // centerX
                        resultBox[1] = (b.y1 + b.y2) / 2; // centerY
                        resultBox[2] = b.x2 - b.x1;       // width
                        resultBox[3] = b.y2 - b.y1;       // height
                        resultBox[4] = b.clsScores[b.cls]; // score
                        resultBox[5] = b.cls;             // classId

                        resultBoxes.Add(resultBox);
                    }

                    return resultBoxes.ToArray();
                }
            }
        }

        /// <summary>
        /// 把图像缩放到模型输入尺寸。
        ///
        /// letterbox = true：保持宽高比缩放，然后用灰色填充到目标尺寸。
        /// letterbox = false：当前代码直接返回原图，不做 resize。
        ///
        /// 注意：如果返回原图给模型，而模型固定要求 640x640，可能会导致输入尺寸不匹配。
        /// 当前 Predict 固定传 true，所以实际走的是 letterbox 分支。
        /// </summary>
        private Mat ResizeImage(Mat image, int width, int height, bool letterbox)
        {
            int ih = image.Rows;
            int iw = image.Cols;

            // 不使用 letterbox 时直接返回原图引用。
            // 当前调用方会 using Mat resized = ResizeImage(...)，所以如果这里真的返回 image，可能会导致调用方 Dispose 掉外部传入图。
            // 目前 Predict 传 true，不会触发这个分支。
            if (!letterbox) return image;

            // 计算保持宽高比的缩放比例取最小的一个比例。
            // 取 min 是为了保证缩放后的图一定能放进 width x height 画布里。
            float scale = Math.Min((float)width / iw, (float)height / ih);
            int nw = Math.Max(1, (int)(iw * scale));
            int nh = Math.Max(1, (int)(ih * scale));

            // 先把原图缩放到 nw x nh。
            Mat resized = new Mat();
            Cv2.Resize(image, resized, new Size(nw, nh));

            // 创建 640x640 灰色画布。
            // Scalar(128,128,128) 是常见 letterbox 填充值  。
            Mat canvas = new(height, width, MatType.CV_8UC3, new Scalar(128, 128, 128));

            // 计算居中贴图的左上角偏移。
            int xOffset = (width - nw) / 2;
            int yOffset = (height - nh) / 2;

            // 把缩放后的图复制到画布中间  把缩小后原图贴在画布中间  其他的就是灰色填充。
            resized.CopyTo(canvas[new Rect(xOffset, yOffset, nw, nh)]);

            return canvas;
        }

        /// <summary>
        /// 确保输入 Mat 是 8-bit BGR 三通道。
        ///
        /// 为什么需要这个函数：
        /// OpenCV 图像可能来自不同来源：灰度图、BGRA 图、16 位图、float 图等。
        /// 但 ImgToTensorFast 使用 Vec3b 读取像素，要求必须是 CV_8UC3。
        /// </summary>
        private static Mat EnsureBgr8(Mat image)
        {
            if (image == null || image.Empty())
                throw new ArgumentException("image is null or empty.");

            Mat source = image;
            Mat? converted = null;

            // 如果图像深度不是 8-bit unsigned，就转换成 CV_8U。
            // 注意：ConvertTo 没有做归一化拉伸，只是类型转换；如果原图是 0~1 float，这里可能会变得很暗。
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
                    // 灰度图转 BGR 三通道。
                    return source.CvtColor(ColorConversionCodes.GRAY2BGR);
                }

                if (channels == 3)
                {
                    // 已经是三通道，Clone 一份返回，避免外部 Mat 生命周期影响这里。
                    return source.Clone();
                }

                if (channels == 4)
                {
                    // BGRA 去掉 alpha 通道，变成 BGR。
                    return source.CvtColor(ColorConversionCodes.BGRA2BGR);
                }

                // 其它通道数不支持，比如 2 通道、5 通道。
                throw new InvalidOperationException("Unsupported Mat channels: " + source.Channels() + ", type: " + source.Type());
            }
            finally
            {
                // converted 是临时 Mat，函数返回前释放。
                // 返回值是 Clone/CvtColor 生成的新 Mat，不依赖 converted。
                if (converted != null)
                {
                    converted.Dispose();
                }
            }
        }

        /// <summary>
        /// 判断 YOLO ONNX 输出 Tensor 的布局。
        ///
        /// 支持两种常见格式：
        /// 1. channelsFirst = true： [1, channels, numBoxes]
        /// 2. channelsFirst = false：[1, numBoxes, channels]
        ///
        /// channels = 4 + numClasses。
        /// numBoxes = 候选框数量，例如 8400。
        /// </summary>
        private static bool TryGetYoloOutputLayout(Tensor<float> output, out int numBoxes, out int channels, out bool channelsFirst)
        {
            numBoxes = 0;
            channels = 0;
            channelsFirst = true;

            ReadOnlySpan<int> dims = output.Dimensions;

            // 这里要求输出必须是三维 Tensor。
            // 第 0 维通常是 batch size，当前代码默认 batch=1。
            if (dims.Length != 3)
                return false;

            int d1 = dims[1];
            int d2 = dims[2];

            // 格式 [1, channels, numBoxes]。
            // channels 一般较小，比如 5、6、84；numBoxes 一般很大，比如 8400。
            if (d1 > 4 && d2 > d1)
            {
                channels = d1;
                numBoxes = d2;
                channelsFirst = true;
                return true;
            }

            // 格式 [1, numBoxes, channels]。
            if (d2 > 4 && d1 > d2)
            {
                channels = d2;
                numBoxes = d1;
                channelsFirst = false;
                return true;
            }

            // 不符合预期布局，调用方会返回空结果。
            return false;
        }

        /// <summary>
        /// 从 YOLO 输出数组中读取某个候选框、某个通道的值。
        ///
        /// 参数含义：
        /// boxIndex：第几个候选框。
        /// channelIndex：第几个通道。0=cx, 1=cy, 2=w, 3=h, 4 以后是类别分数。
        /// numBoxes：候选框总数。
        /// channels：每个候选框的通道数。
        /// channelsFirst：输出布局是否为 [1, channels, numBoxes]。
        /// </summary>
        private static float ReadOutput(float[] buffer, int boxIndex, int channelIndex, int numBoxes, int channels, bool channelsFirst)
        {
            if (channelsFirst)
            {
                // [1, channels, numBoxes] 展平成一维后，索引 = channelIndex * numBoxes + boxIndex。
                return buffer[channelIndex * numBoxes + boxIndex];
            }

            // [1, numBoxes, channels] 展平成一维后，索引 = boxIndex * channels + channelIndex。
            return buffer[boxIndex * channels + channelIndex];
        }

        /// <summary>
        /// 旧版/备用的 Mat 转 Tensor 方法。
        ///
        /// 当前 Predict 使用的是 ImgToTensorFast，不使用这个函数。
        /// 这个函数先 ConvertTo float，再按 c/i/j 读取 Vec3f。
        /// 注意：它没有像 ImgToTensorFast 那样显式 BGR -> RGB，
        /// 如果直接用于 YOLOv8，可能会造成颜色通道顺序和训练时不一致。
        /// </summary>
        private DenseTensor<float> ImgToTensor(Mat img)
        {
            Mat imgF = new Mat();

            // 转 float，并把 0~255 缩放到 0~1。
            img.ConvertTo(imgF, MatType.CV_32FC3, 1.0 / 255.0);

            int H = img.Rows;
            int W = img.Cols;
            float[] chw = new float[3 * H * W];

            // HWC -> CHW。
            // c 是通道，i 是行，j 是列。
            for (int c = 0; c < 3; c++)
                for (int i = 0; i < H; i++)
                    for (int j = 0; j < W; j++)
                        chw[c * H * W + i * W + j] = imgF.At<Vec3f>(i, j)[c];

            return new DenseTensor<float>(chw, new[] { 1, 3, H, W });
        }

        /// <summary>
        /// 计算两个框的 IoU：Intersection over Union，交并比。
        ///
        /// IoU = 交集面积 / 并集面积。
        /// NMS 用它判断两个框是否高度重叠。
        /// IoU 越大，说明两个框越像是在框同一个目标。
        /// </summary>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        private static float IoU(in Box a, in Box b)
        {
            // 交集矩形的左上角：取两个框左上角坐标中更靠右、更靠下的位置。
            float xx1 = Math.Max(a.x1, b.x1);
            float yy1 = Math.Max(a.y1, b.y1);

            // 交集矩形的右下角：取两个框右下角坐标中更靠左、更靠上的位置。
            float xx2 = Math.Min(a.x2, b.x2);
            float yy2 = Math.Min(a.y2, b.y2);

            // 如果两个框不相交，xx2-xx1 或 yy2-yy1 会是负数，这里用 Max(0, ...) 变成 0。
            float w = Math.Max(0, xx2 - xx1);
            float h = Math.Max(0, yy2 - yy1);
            float inter = w * h;

            // 分别计算两个框的面积。
            float areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
            float areaB = (b.x2 - b.x1) * (b.y2 - b.y1);

            // 并集 = A 面积 + B 面积 - 交集面积。
            // 加 1e-6f 是为了避免分母为 0。
            return inter / (areaA + areaB - inter + 1e-6f);
        }

        /// <summary>
        /// 内部检测框结构。
        /// 使用 x1/y1/x2/y2 是为了方便 NMS 计算 IoU；
        /// 最终返回给外部时再转成 centerX/centerY/width/height。
        /// </summary>
        struct Box
        {
            // 左上角和右下角坐标，已经是原图坐标，不是 640x640 输入图坐标。
            public float x1, y1, x2, y2;

            // argmax 后的最高类别分数，用于置信度初筛和排序。
            public float score;

            // argmax 后的类别索引。
            public int cls;

            // 每个类别的原始置信度。
            // 这个数组在自定义 NMS 里很关键，因为代码会比较不同类别各自的分数。
            public float[] clsScores;
        }

        /// <summary>
        /// 释放 ONNX Runtime session。
        /// 如果程序生命周期内一直用单例，通常程序退出时才释放。
        /// </summary>
        public void Dispose()
        {
            _session.Dispose();
        }
    }
}


