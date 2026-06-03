using System.Diagnostics;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System.IO;

namespace LAY.Deep_Learn
{
    public class unet : IDisposable
    {
        private static readonly unet _instance = new unet();
        private static readonly object _predictLock = new object();

        public static unet Instance
        {
            get
            {
                return _instance;
            }
        }

        private readonly InferenceSession _session;
        private readonly string _inputName;
        private readonly int _inputWidth;
        private readonly int _inputHeight;

        private unet()
        {
            string unetModelPath = Path.Combine(
                AppContext.BaseDirectory,
                "onnx",
                "unet",
                "best.onnx"
            );

            _session = CreateSessionWithFallback(unetModelPath);
            _inputName = _session.InputMetadata.Keys.First();

            var inputDims = _session.InputMetadata[_inputName].Dimensions;
            _inputHeight = inputDims.Length >= 4 && inputDims[2] > 0 ? inputDims[2] : 256;
            _inputWidth = inputDims.Length >= 4 && inputDims[3] > 0 ? inputDims[3] : 256;
        }

        private static InferenceSession CreateSessionWithFallback(string modelPath)
        {
            try
            {
                var gpuOptions = new SessionOptions();
                gpuOptions.AppendExecutionProvider_CUDA(0);
                Debug.WriteLine("UNet using CUDA execution provider.");
                return new InferenceSession(modelPath, gpuOptions);
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"UNet CUDA unavailable, fallback to CPU: {ex.Message}");
                return new InferenceSession(modelPath, new SessionOptions());
            }
        }

        /// <summary>
        /// 返回和原图同尺寸的二值分割图，前景为 255，背景为 0。
        /// 训练代码使用 CrossEntropyLoss + argmax，所以这里也按 argmax 解析输出。
        /// </summary>
        public Mat Predict(Mat image)
        {
            if (image == null || image.Empty())
                return new Mat();

            lock (_predictLock)
            {
                using Mat input = EnsureBgr8(image);
                if (input.Empty())
                    return new Mat();

                using Mat resized = ResizeImage(input, _inputWidth, _inputHeight);
                if (resized.Empty())
                    return new Mat();

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
                    using Mat maskSmall = BuildMaskByArgmax(buffer, channels, maskHeight, maskWidth, layout);

                    Mat mask = new Mat();
                    Cv2.Resize(maskSmall, mask, new Size(image.Width, image.Height), 0, 0, InterpolationFlags.Nearest);
                    return mask;
                }
            }
        }

        /// <summary>
        /// 返回 class1 的概率图，和原图同尺寸，类型为 CV_32FC1，范围 0~1。
        /// 如果只需要最终分割图，直接调用 Predict 即可。
        /// </summary>
        public Mat PredictProbability(Mat image)
        {
            if (image == null || image.Empty())
                return new Mat();

            lock (_predictLock)
            {
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
                    using Mat probSmall = BuildClass1Probability(buffer, channels, maskHeight, maskWidth, layout);

                    Mat prob = new Mat();
                    Cv2.Resize(probSmall, prob, new Size(image.Width, image.Height), 0, 0, InterpolationFlags.Linear);
                    return prob;
                }
            }
        }

        private DenseTensor<float> ImgToTensor(Mat img)
        {
            using Mat safe = EnsureBgr8(img);

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

        private static Mat ResizeImage(Mat image, int width, int height)
        {
            Mat resized = new Mat();
            Cv2.Resize(image, resized, new Size(width, height), 0, 0, InterpolationFlags.Linear);
            return resized;
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

        private static bool TryGetUnetOutputLayout(Tensor<float> output, out int channels, out int height, out int width, out OutputLayout layout)
        {
            channels = 0;
            height = 0;
            width = 0;
            layout = OutputLayout.NCHW;

            ReadOnlySpan<int> dims = output.Dimensions;

            if (dims.Length == 4)
            {
                if (dims[1] > 0 && dims[2] > 0 && dims[3] > 0 && dims[1] <= 256)
                {
                    channels = dims[1];
                    height = dims[2];
                    width = dims[3];
                    layout = OutputLayout.NCHW;
                    return true;
                }

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
                channels = 1;
                height = dims[1];
                width = dims[2];
                layout = OutputLayout.NHW;
                return true;
            }

            if (dims.Length == 2 && dims[0] > 0 && dims[1] > 0)
            {
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
            Mat mask = new(height, width, MatType.CV_8UC1, Scalar.All(0));

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    byte value;

                    if (channels == 1)
                    {
                        float score = ReadOutput(buffer, 0, y, x, channels, height, width, layout);
                        value = score > 0.5f ? (byte)255 : (byte)0;
                    }
                    else
                    {
                        int bestClass = 0;
                        float bestScore = float.MinValue;

                        for (int c = 0; c < channels; c++)
                        {
                            float score = ReadOutput(buffer, c, y, x, channels, height, width, layout);
                            if (score > bestScore)
                            {
                                bestScore = score;
                                bestClass = c;
                            }
                        }

                        value = bestClass == 0 ? (byte)0 : (byte)255;
                    }

                    mask.Set(y, x, value);
                }
            }

            return mask;
        }

        private static Mat BuildClass1Probability(float[] buffer, int channels, int height, int width, OutputLayout layout)
        {
            Mat probability = new(height, width, MatType.CV_32FC1, Scalar.All(0));

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    float value;

                    if (channels == 1)
                    {
                        value = NormalizeScore(ReadOutput(buffer, 0, y, x, channels, height, width, layout));
                    }
                    else
                    {
                        float bg = ReadOutput(buffer, 0, y, x, channels, height, width, layout);
                        float fg = ReadOutput(buffer, 1, y, x, channels, height, width, layout);
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
                return buffer[channel * height * width + y * width + x];
            }

            if (layout == OutputLayout.NHWC)
            {
                return buffer[(y * width + x) * channels + channel];
            }

            if (layout == OutputLayout.NHW)
            {
                return buffer[y * width + x];
            }

            if (layout == OutputLayout.HW)
            {
                return buffer[y * width + x];
            }

            return 0f;
        }

        private static float NormalizeScore(float value)
        {
            if (value >= 0f && value <= 1f)
                return value;

            return 1f / (1f + MathF.Exp(-value));
        }

        private static float SoftmaxClass1(float class0, float class1)
        {
            float max = MathF.Max(class0, class1);
            float e0 = MathF.Exp(class0 - max);
            float e1 = MathF.Exp(class1 - max);
            return e1 / (e0 + e1 + 1e-6f);
        }

        private enum OutputLayout
        {
            NCHW,
            NHWC,
            NHW,
            HW
        }

        public void Dispose()
        {
            _session.Dispose();
        }
    }
}
