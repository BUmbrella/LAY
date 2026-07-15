using LAY.Deep_Learn;
using OpenCvSharp;
using YOLOv8;

namespace LAY.Pingline
{
    public class BLT
    {
        private static readonly BLT _instance = new BLT();

        public static BLT Instance
        {
            get
            {
                return _instance;
            }
        }

        public float ConfThreshold { get; set; } = 0.25f;
        public float IouThreshold { get; set; } = 0.4f;
        public float BottomCropRatio { get; set; } = 0.3f;
        //偏移量
        public int HeightOffset { get; set; } = 3;
        public double CenterDistanceTieToleranceRatio { get; set; } = 0.03;

        private BLT()
        {
        }

        public PinglineResult Process(Mat image, bool drawResult = true, double micronScale = 1.0)
        {
            if (image == null || image.Empty())
            {
                return PinglineResult.Fail("image is null or empty");
            }

            Mat output;
            if (drawResult)
            {
                output = image.Clone();
            }
            else
            {
                output = new Mat();
            }

            // 第一步：YOLO 找目标。
            float[][] yoloBoxes = YoloV8Predictor.Instance.Predict(image, ConfThreshold, IouThreshold);
            if (yoloBoxes.Length == 0)
            {
                return PinglineResult.Fail("yolo target not found", output);
            }

            // 第二步：把 YOLO 输出转换成候选目标，并过滤掉无效框。
            List<TargetCandidate> candidates = new List<TargetCandidate>();
            foreach (float[] box in yoloBoxes)
            {
                TargetCandidate candidate = ToCandidate(image, box);
                if (candidate.Box.Width > 0 && candidate.Box.Height > 0)
                {
                    candidates.Add(candidate);
                }
            }

            if (candidates.Count == 0)
            {
                return PinglineResult.Fail("valid yolo target not found", output);
            }

            // 第三步：按距离图像中心最近、清晰度最高的规则选出唯一目标。
            TargetCandidate selected = SelectTarget(image, candidates);
            Rect roiRect = BuildBottomRoi(image, selected.Box, BottomCropRatio);
            if (roiRect.Width <= 0 || roiRect.Height <= 0)
            {
                return PinglineResult.Fail("roi is empty", output, selected.Box);
            }

            using Mat roi = new Mat(image, roiRect);
            if (roi.Empty())
            {
                return PinglineResult.Fail("roi is empty", output, selected.Box, roiRect);
            }

            // 第四步：对截取的底部区域做 UNet 分割。
            using Mat mask = unet.Instance.Predict(roi);
            if (mask.Empty())
            {
                return PinglineResult.Fail("unet mask is empty", output, selected.Box, roiRect);
            }

            // 第五步：只保留最大连通域，减少小噪点影响。
            using Mat binaryMask = ToBinaryMask(mask);
            using Mat? largestMask = LargestComponent(binaryMask);
            if (largestMask == null || largestMask.Empty())
            {
                return PinglineResult.Fail("largest component not found", output, selected.Box, roiRect);
            }

            // 第六步：在 25% 和 75% 的位置测两根竖线长度。
            Measurement? measurement = Measure(largestMask, roiRect, HeightOffset);
            if (measurement == null)
            {
                return PinglineResult.Fail("measure failed", output, selected.Box, roiRect);
            }

            if (drawResult)
            {
                Draw(output, selected.Box, roiRect, measurement.Value, micronScale);
            }

            return PinglineResult.Success(output, selected.Box, roiRect, measurement.Value, selected);
        }

        private static TargetCandidate ToCandidate(Mat image, float[] box)
        {
            float cx = 0;
            float cy = 0;
            float w = 0;
            float h = 0;
            float score = 0;
            int cls = 0;

            if (box.Length > 0) cx = box[0];
            if (box.Length > 1) cy = box[1];
            if (box.Length > 2) w = box[2];
            if (box.Length > 3) h = box[3];
            if (box.Length > 4) score = box[4];
            if (box.Length > 5) cls = (int)box[5];

            int x1 = ClampToInt(cx - w / 2f, 0, image.Width - 1);
            int y1 = ClampToInt(cy - h / 2f, 0, image.Height - 1);
            int x2 = ClampToInt(cx + w / 2f, 0, image.Width);
            int y2 = ClampToInt(cy + h / 2f, 0, image.Height);

            Rect rect = NormalizeRect(x1, y1, x2, y2);
            double imageCx = image.Width / 2.0;
            double imageCy = image.Height / 2.0;
            double dx = cx - imageCx;
            double dy = cy - imageCy;

            TargetCandidate candidate = new TargetCandidate();
            candidate.Box = rect;
            candidate.CenterX = cx;
            candidate.CenterY = cy;
            candidate.CenterDistance = Math.Sqrt(dx * dx + dy * dy);
            candidate.Sharpness = CalcSharpness(image, rect);
            candidate.Score = score;
            candidate.ClassId = cls;
            return candidate;
        }

        private TargetCandidate SelectTarget(Mat image, IReadOnlyList<TargetCandidate> candidates)
        {
            TargetCandidate best = candidates[0];
            double centerDistanceTieTolerance = Math.Sqrt(image.Width * image.Width + image.Height * image.Height) * CenterDistanceTieToleranceRatio;

            for (int i = 1; i < candidates.Count; i++)
            {
                TargetCandidate current = candidates[i];
                double distanceDiff = current.CenterDistance - best.CenterDistance;

                if (distanceDiff < -centerDistanceTieTolerance)
                {
                    best = current;
                    continue;
                }

                if (Math.Abs(distanceDiff) <= centerDistanceTieTolerance)
                {
                    if (current.Sharpness > best.Sharpness)
                    {
                        best = current;
                        continue;
                    }

                    if (Math.Abs(current.Sharpness - best.Sharpness) < 1e-6 && current.Score > best.Score)
                    {
                        best = current;
                    }
                }
            }

            return best;
        }

        private static double CalcSharpness(Mat image, Rect rect)
        {
            Rect safeRect = ClampRect(rect, image.Width, image.Height);
            if (safeRect.Width <= 0 || safeRect.Height <= 0)
            {
                return 0;
            }

            using Mat roi = new Mat(image, safeRect);
            using Mat gray = new Mat();
            using Mat lap = new Mat();

            if (roi.Channels() == 1)
            {
                roi.CopyTo(gray);
            }
            else
            {
                Cv2.CvtColor(roi, gray, ColorConversionCodes.BGR2GRAY);
            }

            Cv2.Laplacian(gray, lap, MatType.CV_64F);
            Cv2.MeanStdDev(lap, out _, out Scalar stddev);
            return stddev.Val0 * stddev.Val0;
        }

        private static Rect BuildBottomRoi(Mat image, Rect box, float ratio)
        {
            int offset = Math.Max(1, (int)Math.Round(box.Height * ratio));
            int x1 = box.Left;
            int x2 = box.Right;
            int y1 = box.Bottom - offset;
            int y2 = box.Bottom + offset;

            return ClampRect(NormalizeRect(x1, y1, x2, y2), image.Width, image.Height);
        }

        private static Mat ToBinaryMask(Mat mask)
        {
            Mat binary = new Mat();

            if (mask.Type() != MatType.CV_8UC1)
            {
                using Mat converted = new Mat();
                mask.ConvertTo(converted, MatType.CV_8UC1);
                Cv2.Threshold(converted, binary, 0, 1, ThresholdTypes.Binary);
            }
            else
            {
                Cv2.Threshold(mask, binary, 0, 1, ThresholdTypes.Binary);
            }

            return binary;
        }

        private static Mat? LargestComponent(Mat mask)
        {
            using Mat labels = new Mat();
            using Mat stats = new Mat();
            using Mat centroids = new Mat();

            int count = Cv2.ConnectedComponentsWithStats(mask, labels, stats, centroids, PixelConnectivity.Connectivity8);
            if (count <= 1)
            {
                return null;
            }

            int bestIndex = 1;
            int bestArea = stats.At<int>(1, (int)ConnectedComponentsTypes.Area);

            for (int i = 2; i < count; i++)
            {
                int area = stats.At<int>(i, (int)ConnectedComponentsTypes.Area);
                if (area > bestArea)
                {
                    bestArea = area;
                    bestIndex = i;
                }
            }

            int rows = labels.Rows;
            int cols = labels.Cols;

            Mat largest = new Mat(mask.Rows, mask.Cols, MatType.CV_8UC1, Scalar.All(0));
            for (int y = 0; y < rows; y++)
            {
                for (int x = 0; x < cols; x++)
                {
                    if (labels.At<int>(y, x) == bestIndex)
                        largest.Set(y, x, (byte)1);
                }
            }

            return largest;
        }
        /// <summary>
        /// 竖线长度
        /// </summary>
        /// <param name="mask"></param>
        /// <param name="roiRect"></param>
        /// <param name="offset"></param>
        /// <returns></returns>
        private static Measurement? Measure(Mat mask, Rect roiRect, int offset)
        {
            using Mat points = new Mat();
            Cv2.FindNonZero(mask, points);
            if (points.Empty())
            {
                return null;
            }

            Rect bounds = Cv2.BoundingRect(points);
            if (bounds.Width <= 0)
            {
                return null;
            }

            int localX1 = ClampToInt(bounds.Left + bounds.Width * 0.1, 0, mask.Width - 1);
            int localX2 = ClampToInt(bounds.Left + bounds.Width * 0.85, 0, mask.Width - 1);

            LineMeasure? left = GetHeight(mask, localX1, roiRect, offset);
            LineMeasure? right = GetHeight(mask, localX2, roiRect, offset);

            if (left == null || right == null)
                return null;

            Measurement measurement = new Measurement();
            measurement.Left = left.Value;
            measurement.Right = right.Value;
            return measurement;
        }

        private static LineMeasure? GetHeight(Mat mask, int localX, Rect roiRect, int offset)
        {
            int minY = -1;
            int maxY = -1;

            int rows = mask.Rows;
            for (int y = 0; y < rows; y++)
            {
                if (mask.At<byte>(y, localX) == 0)
                {
                    continue;
                }

                if (minY < 0)
                {
                    minY = y;
                }

                maxY = y;
            }

            if (minY < 0 || maxY < 0)
            {
                return null;
            }

            int x = roiRect.Left + localX;
            int y1 = roiRect.Top + minY;
            int y2 = roiRect.Top + maxY - offset;
            int height = Math.Max(0, maxY - minY - offset);

            LineMeasure lineMeasure = new LineMeasure();
            lineMeasure.X = x;
            lineMeasure.Y1 = y1;
            lineMeasure.Y2 = y2;
            lineMeasure.Height = height;
            return lineMeasure;
        }

        private static void Draw(Mat image, Rect selectedBox, Rect roiRect, Measurement measurement, double micronScale)
        {
            DrawLineMeasure(image, measurement.Left, "L", micronScale);
            DrawLineMeasure(image, measurement.Right, "L", micronScale);
            DrawSummaryText(image, measurement, micronScale);
        }

        private static void DrawSummaryText(Mat image, Measurement measurement, double micronScale)
        {
            Scalar green = new Scalar(0, 255, 0);
            double fontScale = 3.0;
            int thickness = 3;
            int x = 20;
            int y = 20;
            int baseline;
            Size textSize = Cv2.GetTextSize("LR:0000um", HersheyFonts.HersheySimplex, fontScale, thickness, out baseline);
            int lineGap = textSize.Height + baseline + 16;

            Cv2.PutText(
                image,
                "LF:" + FormatMicronValue(measurement.Left.Height, micronScale) + "um",
                new Point(x, y + textSize.Height),
                HersheyFonts.HersheySimplex,
                fontScale,
                green,
                thickness
            );
            Cv2.PutText(
                image,
                "LR:" + FormatMicronValue(measurement.Right.Height, micronScale) + "um",
                new Point(x, y + textSize.Height + lineGap),
                HersheyFonts.HersheySimplex,
                fontScale,
                green,
                thickness
            );
        }
        /// <summary>
        /// 画竖线和写字
        /// </summary>
        /// <param name="image"></param>
        /// <param name="line"></param>
        /// <param name="label"></param>
        private static void DrawLineMeasure(Mat image, LineMeasure line, string label, double micronScale)
        {
            Point p1 = new Point(line.X, line.Y1);
            Point p2 = new Point(line.X, line.Y2);

            Scalar color = new Scalar(0, 0, 255);
            string text = label + "=" + FormatMicronValue(line.Height, micronScale) + "um";
            int baseline;
            Size textSize = Cv2.GetTextSize(text, HersheyFonts.HersheySimplex, 0.6, 2, out baseline);
            int textX = Math.Max(0, Math.Min(image.Width - textSize.Width, line.X - textSize.Width / 2));
            int textY = Math.Max(textSize.Height + 2, line.Y1 - 8);
            //画线
            Cv2.Line(image, p1, p2, color, 2);

            Cv2.PutText(
                image,
                text,
                new Point(textX, textY),
                HersheyFonts.HersheySimplex,
                2,
                color,
                2
            );
        }

        private static string FormatMicronValue(int pixelValue, double micronScale)
        {
            double micronValue = pixelValue * micronScale;
            return micronValue.ToString("0.###", System.Globalization.CultureInfo.InvariantCulture);
        }

        private static Rect NormalizeRect(int x1, int y1, int x2, int y2)
        {
            int left = Math.Min(x1, x2);
            int top = Math.Min(y1, y2);
            int right = Math.Max(x1, x2);
            int bottom = Math.Max(y1, y2);
            return new Rect(left, top, right - left, bottom - top);
        }

        private static Rect ClampRect(Rect rect, int width, int height)
        {
            int x1 = ClampToInt(rect.Left, 0, width);
            int y1 = ClampToInt(rect.Top, 0, height);
            int x2 = ClampToInt(rect.Right, 0, width);
            int y2 = ClampToInt(rect.Bottom, 0, height);
            return NormalizeRect(x1, y1, x2, y2);
        }

        private static int ClampToInt(double value, int min, int max)
        {
            return Math.Max(min, Math.Min(max, (int)Math.Round(value)));
        }
    }

    public class PinglineResult
    {
        public bool IsSuccess { get; private set; }
        public string Message { get; private set; } = string.Empty;
        public Mat OutputImage { get; private set; } = new Mat();
        public Rect SelectedBox { get; private set; }
        public Rect RoiRect { get; private set; }
        public Measurement? Measurement { get; private set; }
        public TargetCandidate? SelectedTarget { get; private set; }

        public static PinglineResult Success(Mat outputImage, Rect selectedBox, Rect roiRect, Measurement measurement, TargetCandidate selectedTarget)
        {
            PinglineResult result = new PinglineResult();
            result.IsSuccess = true;
            result.Message = "ok";
            result.OutputImage = outputImage;
            result.SelectedBox = selectedBox;
            result.RoiRect = roiRect;
            result.Measurement = measurement;
            result.SelectedTarget = selectedTarget;
            return result;
        }

        public static PinglineResult SuccessWithoutMeasurement(Mat outputImage, string message)
        {
            PinglineResult result = new PinglineResult();
            result.IsSuccess = true;
            result.Message = message;
            result.OutputImage = outputImage;
            result.SelectedBox = default;
            result.RoiRect = default;
            result.Measurement = null;
            result.SelectedTarget = null;
            return result;
        }

        public static PinglineResult Fail(string message, Mat? outputImage = null, Rect selectedBox = default, Rect roiRect = default)
        {
            PinglineResult result = new PinglineResult();
            result.IsSuccess = false;
            result.Message = message;

            if (outputImage == null)
            {
                result.OutputImage = new Mat();
            }
            else
            {
                result.OutputImage = outputImage;
            }

            result.SelectedBox = selectedBox;
            result.RoiRect = roiRect;
            return result;
        }
    }

    public struct Measurement
    {
        public LineMeasure Left { get; set; }
        public LineMeasure Right { get; set; }
    }

    public struct LineMeasure
    {
        public int X { get; set; }
        public int Y1 { get; set; }
        public int Y2 { get; set; }
        public int Height { get; set; }
    }

    public struct TargetCandidate
    {
        public Rect Box { get; set; }
        public float CenterX { get; set; }
        public float CenterY { get; set; }
        public double CenterDistance { get; set; }
        public double Sharpness { get; set; }
        public float Score { get; set; }
        public int ClassId { get; set; }
    }
}
