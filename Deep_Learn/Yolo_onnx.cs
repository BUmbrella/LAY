using System.Diagnostics;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System.IO;
namespace YOLOv8
{
    public class YoloV8Predictor : IDisposable
    {
        // ---------------- 单例 ----------------
        private static readonly YoloV8Predictor _instance = new YoloV8Predictor();
        private static readonly object _predictLock = new object(); // 用于 Predict 方法线程锁

        public static YoloV8Predictor Instance
        {
            get
            {
                return _instance;
            }
        }

        private readonly InferenceSession _session;
        private readonly string _inputName;
        private readonly int _inputWidth = 640;
        private readonly int _inputHeight = 640;

        // ---------------- 私有构造函数 ----------------
        private YoloV8Predictor()
        {


            string yoloModelPath = Path.Combine(
                AppContext.BaseDirectory,
                "onnx",
                "yolo",
                "best.onnx"
            );



            _session = CreateSessionWithFallback(yoloModelPath);
            _inputName = _session.InputMetadata.Keys.First();
        }

        private static InferenceSession CreateSessionWithFallback(string modelPath)
        {
            try
            {
                var gpuOptions = new SessionOptions();
                gpuOptions.AppendExecutionProvider_CUDA(0);
                Debug.WriteLine("YOLO using CUDA execution provider.");
                return new InferenceSession(modelPath, gpuOptions);
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"YOLO CUDA unavailable, fallback to CPU: {ex.Message}");
                return new InferenceSession(modelPath, new SessionOptions());
            }
        }

        DenseTensor<float> ImgToTensorFast(Mat img)
        {
            if (img == null || img.Empty())
                throw new ArgumentException("img is null or empty.");

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

            float[] chw = new float[3 * hw];

            for (int y = 0; y < h; y++)
            {
                int rowOffset = y * w;

                for (int x = 0; x < w; x++)
                {
                    int dstIdx = rowOffset + x;
                    Vec3b pixel = safe.At<Vec3b>(y, x);

                    chw[dstIdx] = pixel[2] / 255.0f;
                    chw[hw + dstIdx] = pixel[1] / 255.0f;
                    chw[2 * hw + dstIdx] = pixel[0] / 255.0f;
                }
            }

            return new DenseTensor<float>(chw, new[] { 1, 3, h, w });
        }

        public float[][] Predict(Mat image, float confThresh = 0.25f, float iouThresh = 0.4f)
        {
            if (image == null || image.Empty())
                return Array.Empty<float[]>();

            lock (_predictLock)
            {
                using Mat input = EnsureBgr8(image);

                if (input.Empty())
                    return Array.Empty<float[]>();

                using Mat resized =
                    ResizeImage(input, _inputWidth, _inputHeight, true);

                if (resized.Empty())
                    return Array.Empty<float[]>();

                DenseTensor<float> tensor = ImgToTensorFast(resized);

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
                    Debug.WriteLine($"YOLO inference failed: {ex}");
                    return Array.Empty<float[]>();
                }

                using (results)
                {
                    DisposableNamedOnnxValue? firstOutput = results.FirstOrDefault();
                    if (firstOutput == null)
                        return Array.Empty<float[]>();

                    var output = firstOutput.AsTensor<float>();
                    if (!TryGetYoloOutputLayout(output, out int numBoxes, out int channels, out bool channelsFirst))
                        return Array.Empty<float[]>();

                    int numClasses = channels - 4;
                    float[] buffer = output.ToArray();

                float scale = Math.Min(
                    (float)_inputWidth / image.Width,
                    (float)_inputHeight / image.Height
                );
                float dx = (_inputWidth - image.Width * scale) / 2;
                float dy = (_inputHeight - image.Height * scale) / 2;

                List<Box> boxes = new List<Box>();

                for (int i = 0; i < numBoxes; i++)
                {
                    float[] clsScores = new float[numClasses];

                    float maxCls = 0f;
                    int cls = 0;

                    for (int j = 0; j < numClasses; j++)
                    {
                        float v = ReadOutput(buffer, i, j + 4, numBoxes, channels, channelsFirst);
                        clsScores[j] = v;

                        if (v > maxCls)
                        {
                            maxCls = v;
                            cls = j;
                        }
                    }

                    if (maxCls < confThresh)
                        continue;

                    float cx = ReadOutput(buffer, i, 0, numBoxes, channels, channelsFirst);
                    float cy = ReadOutput(buffer, i, 1, numBoxes, channels, channelsFirst);
                    float w = ReadOutput(buffer, i, 2, numBoxes, channels, channelsFirst);
                    float h = ReadOutput(buffer, i, 3, numBoxes, channels, channelsFirst);

                    float x1 = cx - w / 2;
                    float y1 = cy - h / 2;
                    float x2 = cx + w / 2;
                    float y2 = cy + h / 2;

                    float rx1 = (x1 - dx) / scale;
                    float ry1 = (y1 - dy) / scale;
                    float rx2 = (x2 - dx) / scale;
                    float ry2 = (y2 - dy) / scale;

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

                if (boxes.Count == 0)
                    return Array.Empty<float[]>();

                boxes.Sort(delegate (Box a, Box b)
                {
                    return b.score.CompareTo(a.score);
                });

                int TYPE1 = 0;
                int TYPE2 = 1;

                bool[] used = new bool[boxes.Count];
                List<Box> finalBoxes = new List<Box>();

                for (int i = 0; i < boxes.Count; i++)
                {
                    if (used[i])
                        continue;

                    Box best = boxes[i];
                    used[i] = true;

                    for (int j = i + 1; j < boxes.Count; j++)
                    {
                        if (used[j])
                            continue;

                        if (IoU(best, boxes[j]) > iouThresh)
                        {
                            float scoreBest = best.clsScores[best.cls];
                            float scoreCur = boxes[j].clsScores[boxes[j].cls];

                            if (best.cls == TYPE1 && boxes[j].cls == TYPE2)
                            {
                            }
                            else if (best.cls == TYPE2 && boxes[j].cls == TYPE1)
                            {
                                best = boxes[j];
                            }
                            else if (scoreCur > scoreBest)
                            {
                                best = boxes[j];
                            }

                            used[j] = true;
                        }
                    }

                    finalBoxes.Add(best);
                }

                List<float[]> resultBoxes = new List<float[]>();
                foreach (Box b in finalBoxes)
                {
                    float[] resultBox = new float[6];
                    resultBox[0] = (b.x1 + b.x2) / 2;
                    resultBox[1] = (b.y1 + b.y2) / 2;
                    resultBox[2] = b.x2 - b.x1;
                    resultBox[3] = b.y2 - b.y1;
                    resultBox[4] = b.clsScores[b.cls];
                    resultBox[5] = b.cls;
                    resultBoxes.Add(resultBox);
                }

                return resultBoxes.ToArray();
                }
            }
        }




        private Mat ResizeImage(Mat image, int width, int height, bool letterbox)
        {
            int ih = image.Rows;
            int iw = image.Cols;

            if (!letterbox) return image;

            float scale = Math.Min((float)width / iw, (float)height / ih);
            int nw = Math.Max(1, (int)(iw * scale));
            int nh = Math.Max(1, (int)(ih * scale));

            Mat resized = new Mat();
            Cv2.Resize(image, resized, new Size(nw, nh));

            Mat canvas = new(height, width, MatType.CV_8UC3, new Scalar(128, 128, 128));
            int xOffset = (width - nw) / 2;
            int yOffset = (height - nh) / 2;
            resized.CopyTo(canvas[new Rect(xOffset, yOffset, nw, nh)]);
            return canvas;
        }

        private static Mat EnsureBgr8(Mat image)
        {
            if (image == null || image.Empty())
                throw new ArgumentException("image is null or empty.");

            Mat source = image;
            Mat? converted = null;

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
                    return source.CvtColor(ColorConversionCodes.GRAY2BGR);
                }

                if (channels == 3)
                {
                    return source.Clone();
                }

                if (channels == 4)
                {
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

        private static bool TryGetYoloOutputLayout(Tensor<float> output, out int numBoxes, out int channels, out bool channelsFirst)
        {
            numBoxes = 0;
            channels = 0;
            channelsFirst = true;

            ReadOnlySpan<int> dims = output.Dimensions;
            if (dims.Length != 3)
                return false;

            int d1 = dims[1];
            int d2 = dims[2];

            if (d1 > 4 && d2 > d1)
            {
                channels = d1;
                numBoxes = d2;
                channelsFirst = true;
                return true;
            }

            if (d2 > 4 && d1 > d2)
            {
                channels = d2;
                numBoxes = d1;
                channelsFirst = false;
                return true;
            }

            return false;
        }

        private static float ReadOutput(float[] buffer, int boxIndex, int channelIndex, int numBoxes, int channels, bool channelsFirst)
        {
            if (channelsFirst)
            {
                return buffer[channelIndex * numBoxes + boxIndex];
            }

            return buffer[boxIndex * channels + channelIndex];
        }

        private DenseTensor<float> ImgToTensor(Mat img)
        {
            Mat imgF = new Mat();
            img.ConvertTo(imgF, MatType.CV_32FC3, 1.0 / 255.0);
            int H = img.Rows;
            int W = img.Cols;
            float[] chw = new float[3 * H * W];

            for (int c = 0; c < 3; c++)
                for (int i = 0; i < H; i++)
                    for (int j = 0; j < W; j++)
                        chw[c * H * W + i * W + j] = imgF.At<Vec3f>(i, j)[c];

            return new DenseTensor<float>(chw, new[] { 1, 3, H, W });
        }

        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        private static float IoU(in Box a, in Box b)
        {
            float xx1 = Math.Max(a.x1, b.x1);
            float yy1 = Math.Max(a.y1, b.y1);
            float xx2 = Math.Min(a.x2, b.x2);
            float yy2 = Math.Min(a.y2, b.y2);

            float w = Math.Max(0, xx2 - xx1);
            float h = Math.Max(0, yy2 - yy1);
            float inter = w * h;
            float areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
            float areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
            return inter / (areaA + areaB - inter + 1e-6f);
        }

        struct Box
        {
            public float x1, y1, x2, y2;

            public float score;        // argmax 后的分数（用于初筛/排序）
            public int cls;            // argmax 类别索引

            public float[] clsScores;  // ⭐ 每个类别的原始置信度（关键）
        }


        public void Dispose()
        {
            _session.Dispose();
        }
    }
}
