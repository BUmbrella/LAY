using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Text;
using LAY.Pingline;
using LAY.Project.Models;
using OpenCvSharp;

namespace LAY.Main
{
    // sysmain 是界面点击“启动”后调用的主流程类。
    // 它负责遍历输入文件夹、按文件名选择 B/R pipeline、保存结果图、生成 xlsx。
    internal class sysmain
    {
        private static readonly HashSet<string> SupportedExtensions = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            ".bmp",
            ".gif",
            ".jpeg",
            ".jpg",
            ".png",
            ".tif",
            ".tiff",
            ".webp"
        };

        private enum FolderProcessMode
        {
            Legacy,
            Gc,
            Bf
        }

        private sealed class FolderProcessOptions
        {
            /// <summary>
            /// 文件夹名中的放大倍数部分。
            /// </summary>
            public string Magnification { get; set; } = string.Empty;

            /// <summary>
            /// 当前文件夹使用的处理模式。
            /// </summary>
            public FolderProcessMode Mode { get; set; }

            /// <summary>
            /// 新模式中由文件夹名强制指定的流程类型。
            /// </summary>
            public string? ForcedPipelineCode { get; set; }

            /// <summary>
            /// 是否只输出结果图片。。
            /// </summary>
            public bool OutputImagesOnly
            {
                get
                {
                    return Mode != FolderProcessMode.Legacy;
                }
            }
        }

        private volatile bool _stopRequested;
        /// <summary>
        ///基础放大倍率
        /// </summary>
        public double BaseMagnification { get; set; } = 5.0;
        /// <summary>
        /// 基础像素大小
        /// </summary>
        public double BaseMicronScale { get; set; } = 0.9129;

        public void Stop()
        {
            _stopRequested = true;
        }

        public SysmainProcessResult Start(string inputFolderPath, Action<string>? log = null)
        {
            // 每次点击“启动”都重新开始一轮检测。
            // 上一次如果点过“停止”，这里必须先把停止标记清掉，否则新一轮会直接退出。
            _stopRequested = false;

            // 输入必须是一个真实存在的文件夹。
            // 后续所有规则解析、图片查找、结果保存都依赖这个根目录。
            if (string.IsNullOrWhiteSpace(inputFolderPath) || !Directory.Exists(inputFolderPath))
            {
                throw new DirectoryNotFoundException("Input folder not found: " + inputFolderPath);
            }

            // 解析文件夹名称中的处理规则。
            // 支持两类：
            // 1. 旧规则：文件夹名只有倍数，例如“50”。按图片文件名自动判断 B/R，输出到 Result，并写 Excel。
            // 2. 新规则：例如“50-R-GC”“50-B-GC”“50-BLT-GC”“50-R-BF”“50-B-BF”。
            //    - 第一段是倍数。
            //    - 第二段强制指定流程：R、B 或 BLT；BLT 会按 B 流程处理。
            //    - 第三段指定模式：GC 或 BF。
            FolderProcessOptions options = ParseFolderProcessOptions(inputFolderPath);

            // 放大倍数只取规则中的第一段数字。
            // 例如“50-R-GC”最终使用 50 计算像素到微米的换算系数。
            string magnification = options.Magnification;

            // 根据当前倍数计算微米/像素系数。
            // 后面的 B/R pipeline 都使用同一个 micronScale，保证新旧模式测量单位一致。
            double micronScale = CalculateMicronScale(magnification, log);

            // 旧模式会把结果统一放到 inputFolderPath\Result。
            // GC/BF 新模式只保存 Checked 图片，不生成 Excel，也不创建 Result 文件夹。
            // 新模式的结果图会保存到原图所在目录，这样子文件夹里的 BF 图片也留在自己的文件夹中。
            string resultFolderPath = options.OutputImagesOnly ? inputFolderPath : Path.Combine(inputFolderPath, "Result");
            if (!options.OutputImagesOnly)
            {
                Directory.CreateDirectory(resultFolderPath);
            }

            // Excel 只服务旧模式。
            // 新模式虽然也计算 bXlsxPath/rXlsxPath，后面不会删除旧表，也不会写新表。
            string bXlsxPath = Path.Combine(resultFolderPath, "BLT_measure_results.xlsx");
            string rXlsxPath = Path.Combine(resultFolderPath, "R_measure_results.xlsx");

            // 旧模式每次重新检测时清理旧 Excel，避免新旧数据混在一起。
            // GC/BF 新模式不动 Excel，因为需求是“只保存一个结果图像”。
            if (!options.OutputImagesOnly)
            {
                DeleteExistingResultFile(Path.Combine(resultFolderPath, "measure_results.xlsx"));
                DeleteExistingResultFile(bXlsxPath);
                DeleteExistingResultFile(rXlsxPath);
            }

            // B/R 测量记录集合。
            // 旧模式检测完成后会写入对应 Excel。
            // 新模式不会写 Excel，但仍复用 ProcessBImage/ProcessRImage，所以这里继续传入集合。
            List<MeasureRecord> bRecords = new List<MeasureRecord>();
            List<MeasureRecord> rRecords = new List<MeasureRecord>();

            // 记录有问题的图片名称。
            // 旧模式界面会根据 Excel 空值标红；这里保留这个集合用于返回过程结果。
            HashSet<string> problemImageFileNames = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

            // 按文件夹规则获取待检测图片：
            // - 旧模式：只取当前根目录图片。
            // - GC / BF：只递归扫描子目录，所有图片都按文件夹名指定的 R/B 流程处理。
            //   直接放在拖入根目录下的图片不显示、不检测。
            // 同时会跳过 Result 文件夹和已经带 -Checked 后缀的图片。
            string[] imagePaths = GetImagePaths(inputFolderPath, options);

            foreach (string imagePath in imagePaths)
            {
                // 停止按钮只是设置 _stopRequested。
                // 真正停止发生在处理下一张图片之前，因此不会打断正在处理中的图片。
                if (_stopRequested)
                {
                    WriteLog(log, "Stop requested.");
                    break;
                }

                // 处理单张图片。
                // 旧模式：按文件名解析 B/R，并把结果图写到 Result 文件夹。
                // GC/BF：强制使用文件夹规则指定的 B/R 流程，并生成 -Checked 结果图。
                ProcessOneImage(imagePath, resultFolderPath, bRecords, rRecords, problemImageFileNames, micronScale, log, options);
            }

            // 保存本轮生成的 Excel 路径。
            // 新模式不写 Excel，所以这个列表会保持为空。
            List<string> xlsxPaths = new List<string>();

            // 旧模式：如果有 B 流程记录，写 BLT_measure_results.xlsx。
            if (!options.OutputImagesOnly && bRecords.Count > 0)
            {
                WriteXlsx(bXlsxPath, bRecords, XlsxSheetType.Blt);
                xlsxPaths.Add(bXlsxPath);
            }

            // 旧模式：如果有 R 流程记录，写 R_measure_results.xlsx。
            if (!options.OutputImagesOnly && rRecords.Count > 0)
            {
                WriteXlsx(rXlsxPath, rRecords, XlsxSheetType.R);
                xlsxPaths.Add(rXlsxPath);
            }

            // 把本轮处理结果返回给 ViewModel。
            // ResultFolderPath：
            // - 旧模式是 Result 文件夹，界面“查看结果”会切过去。
            // - 新模式是根目录，界面会递归读取各目录下的 -Checked 图片。
            SysmainProcessResult result = new SysmainProcessResult();
            result.ResultFolderPath = resultFolderPath;
            result.XlsxPath = string.Join("; ", xlsxPaths);
            result.XlsxPaths = xlsxPaths;
            result.Records = bRecords.Concat(rRecords).ToList();
            result.ProblemImageFileNames = problemImageFileNames.ToList();
            result.OutputImagesOnly = options.OutputImagesOnly;
            return result;
        }

        private static void DeleteExistingResultFile(string filePath)
        {
            if (File.Exists(filePath))
            {
                File.Delete(filePath);
            }
        }

        private static string[] GetImagePaths(string inputFolderPath, FolderProcessOptions options)
        {
            List<string> imagePaths = new List<string>();

            // 旧模式只扫描根目录，保持原来的检测方式。
            // GC / BF 新模式递归扫描子目录，但跳过直接放在根目录下的图片。
            SearchOption searchOption = options.OutputImagesOnly ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly;
            string[] allFiles = Directory.GetFiles(inputFolderPath, "*.*", searchOption);

            foreach (string filePath in allFiles)
            {
                string extension = Path.GetExtension(filePath);

                // 只处理支持的图片格式。
                // 同时跳过已经生成过的 -Checked 图片，避免下一次检测把结果图再次当作输入。
                // Result 文件夹也跳过，避免旧模式输出结果被再次处理。
                if (!SupportedExtensions.Contains(extension) ||
                    IsCheckedImage(filePath) ||
                    IsInResultFolder(inputFolderPath, filePath) ||
                    HasExistingResultImage(inputFolderPath, filePath, options) ||
                    (options.OutputImagesOnly && IsDirectlyInFolder(inputFolderPath, filePath)))
                {
                    continue;
                }

                imagePaths.Add(filePath);
            }

            imagePaths.Sort(delegate (string left, string right)
            {
                string leftRelative = Path.GetRelativePath(inputFolderPath, left);
                string rightRelative = Path.GetRelativePath(inputFolderPath, right);
                return string.Compare(leftRelative, rightRelative, StringComparison.OrdinalIgnoreCase);
            });

            return imagePaths.ToArray();
        }
        //进行文件夹名字类型判断
        public static bool TryGetMagnificationFromFolderName(string folderPath, out string magnification)
        {
            magnification = ParseFolderProcessOptions(folderPath).Magnification;
            return magnification.Length > 0;
        }

        public static bool HasProcessablePhotos(string folderPath)
        {
            if (string.IsNullOrWhiteSpace(folderPath) || !Directory.Exists(folderPath))
            {
                return false;
            }

            FolderProcessOptions options = ParseFolderProcessOptions(folderPath);
            return GetImagePaths(folderPath, options).Length > 0;
        }

        public static IReadOnlyList<string> GetProcessableImagePaths(string folderPath)
        {
            if (string.IsNullOrWhiteSpace(folderPath) || !Directory.Exists(folderPath))
            {
                return Array.Empty<string>();
            }

            FolderProcessOptions options = ParseFolderProcessOptions(folderPath);
            return GetImagePaths(folderPath, options);
        }

        public static IReadOnlyList<string> GetCheckedImagePaths(string folderPath)
        {
            List<string> imagePaths = new List<string>();

            if (string.IsNullOrWhiteSpace(folderPath) || !Directory.Exists(folderPath))
            {
                return imagePaths;
            }

            string[] allFiles = Directory.GetFiles(folderPath, "*.*", SearchOption.AllDirectories);
            foreach (string filePath in allFiles)
            {
                string extension = Path.GetExtension(filePath);
                if (SupportedExtensions.Contains(extension) &&
                    IsCheckedImage(filePath) &&
                    !IsDirectlyInFolder(folderPath, filePath))
                {
                    imagePaths.Add(filePath);
                }
            }

            imagePaths.Sort(delegate (string left, string right)
            {
                string leftRelative = Path.GetRelativePath(folderPath, left);
                string rightRelative = Path.GetRelativePath(folderPath, right);
                return string.Compare(leftRelative, rightRelative, StringComparison.OrdinalIgnoreCase);
            });

            return imagePaths;
        }

        public static IReadOnlyList<string> GetExistingResultImagePaths(string folderPath)
        {
            if (string.IsNullOrWhiteSpace(folderPath) || !Directory.Exists(folderPath))
            {
                return Array.Empty<string>();
            }

            FolderProcessOptions options = ParseFolderProcessOptions(folderPath);
            if (options.OutputImagesOnly)
            {
                return GetCheckedImagePaths(folderPath);
            }

            string resultFolderPath = Path.Combine(folderPath, "Result");
            if (!Directory.Exists(resultFolderPath))
            {
                return Array.Empty<string>();
            }

            List<string> imagePaths = new List<string>();
            string[] allFiles = Directory.GetFiles(resultFolderPath, "*.*", SearchOption.TopDirectoryOnly);
            foreach (string filePath in allFiles)
            {
                string extension = Path.GetExtension(filePath);
                if (SupportedExtensions.Contains(extension))
                {
                    imagePaths.Add(filePath);
                }
            }

            imagePaths.Sort(delegate (string left, string right)
            {
                return string.Compare(Path.GetFileName(left), Path.GetFileName(right), StringComparison.OrdinalIgnoreCase);
            });

            return imagePaths;
        }

        public static bool IsOutputImagesOnlyFolder(string folderPath)
        {
            if (string.IsNullOrWhiteSpace(folderPath) || !Directory.Exists(folderPath))
            {
                return false;
            }

            return ParseFolderProcessOptions(folderPath).OutputImagesOnly;
        }

        private static string GetMagnificationFromFolderName(string inputFolderPath)
        {
            string magnification = ParseFolderProcessOptions(inputFolderPath).Magnification;
            if (magnification.Length > 0)
            {
                return magnification;
            }

            throw new InvalidOperationException("请修改文件夹名字，提供放大倍数");
        }

        private static FolderProcessOptions ParseFolderProcessOptions(string folderPath)
        {
            string folderName = new DirectoryInfo(folderPath).Name.Trim();
            string[] parts = folderName.Split('-', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

            if (parts.Length == 1)
            {
                return new FolderProcessOptions
                {
                    Magnification = GetTrailingDigits(folderName),
                    Mode = FolderProcessMode.Legacy
                };
            }

            if (parts.Length == 3 &&
                TryParsePipelineCode(parts[1], out string pipelineCode) &&
                TryParseFolderMode(parts[2], out FolderProcessMode mode))
            {
                return new FolderProcessOptions
                {
                    Magnification = parts[0],
                    Mode = mode,
                    ForcedPipelineCode = pipelineCode
                };
            }

            return new FolderProcessOptions
            {
                Magnification = GetTrailingDigits(folderName),
                Mode = FolderProcessMode.Legacy
            };
        }

        private static bool TryParsePipelineCode(string text, out string pipelineCode)
        {
            pipelineCode = string.Empty;
            if (string.Equals(text, "R", StringComparison.OrdinalIgnoreCase))
            {
                pipelineCode = "R";
                return true;
            }

            if (string.Equals(text, "B", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(text, "BLT", StringComparison.OrdinalIgnoreCase))
            {
                pipelineCode = "B";
                return true;
            }

            return false;
        }

        private static bool TryParseFolderMode(string text, out FolderProcessMode mode)
        {
            if (string.Equals(text, "GC", StringComparison.OrdinalIgnoreCase))
            {
                mode = FolderProcessMode.Gc;
                return true;
            }

            if (string.Equals(text, "BF", StringComparison.OrdinalIgnoreCase))
            {
                mode = FolderProcessMode.Bf;
                return true;
            }

            mode = FolderProcessMode.Legacy;
            return false;
        }

        private static bool IsCheckedImage(string filePath)
        {
            string name = Path.GetFileNameWithoutExtension(filePath);
            return name.EndsWith("-Checked", StringComparison.OrdinalIgnoreCase);
        }

        private static bool IsInResultFolder(string inputFolderPath, string filePath)
        {
            string relativePath = Path.GetRelativePath(inputFolderPath, filePath);
            string[] parts = relativePath.Split(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            return parts.Any(part => string.Equals(part, "Result", StringComparison.OrdinalIgnoreCase));
        }

        private static bool IsDirectlyInFolder(string folderPath, string filePath)
        {
            string? fileFolderPath = Path.GetDirectoryName(filePath);
            return string.Equals(fileFolderPath, folderPath, StringComparison.OrdinalIgnoreCase);
        }

        private static bool FileNameContainsPipelineCode(string filePath, string? pipelineCode)
        {
            if (string.IsNullOrWhiteSpace(pipelineCode))
            {
                return true;
            }

            string name = Path.GetFileNameWithoutExtension(filePath);
            return name.IndexOf(pipelineCode, StringComparison.OrdinalIgnoreCase) >= 0;
        }

        private static string GetTrailingDigits(string text)
        {
            int end = text.Length - 1;
            while (end >= 0 && char.IsWhiteSpace(text[end]))
            {
                end--;
            }

            int start = end;
            while (start >= 0 && char.IsDigit(text[start]))
            {
                start--;
            }

            if (start == end)
            {
                return string.Empty;
            }

            return text.Substring(start + 1, end - start);
        }
        /// <summary>
        /// 根据基准50倍的像素大小0.4555微米/px  计算其他倍率下的像素大小
        /// </summary>
        /// <param name="magnification"></param>
        /// <param name="log"></param>
        /// <returns></returns>
        /// <exception cref="InvalidOperationException"></exception>
        private double CalculateMicronScale(string magnification, Action<string>? log)
        {
            if (!TryParseDouble(magnification, out double currentMagnification) || currentMagnification <= 0)
            {
                throw new InvalidOperationException("放大倍数格式错误: " + magnification);
            }

            if (BaseMagnification <= 0 || BaseMicronScale <= 0)
            {
                throw new InvalidOperationException("基础倍数或基础系数必须大于 0");
            }

            double scale = BaseMicronScale * BaseMagnification / currentMagnification;
            WriteLog(log, "Magnification " + magnification + " micron scale: " + scale.ToString("0.######", CultureInfo.InvariantCulture));
            return scale;
        }

       

       

        private static bool TryParseDouble(string text, out double value)
        {
            if (double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out value))
            {
                return true;
            }

            return double.TryParse(text, NumberStyles.Float, CultureInfo.CurrentCulture, out value);
        }
        /// <summary>
        /// 进行检测
        /// </summary>
        /// <param name="imagePath"></param>
        /// <param name="resultFolderPath"></param>
        /// <param name="bRecords"></param>
        /// <param name="rRecords"></param>
        /// <param name="micronScale"></param>
        /// <param name="log"></param>
        private static void ProcessOneImage(string imagePath, string resultFolderPath, List<MeasureRecord> bRecords, List<MeasureRecord> rRecords, HashSet<string> problemImageFileNames, double micronScale, Action<string>? log, FolderProcessOptions options)
        {
            string fileName = Path.GetFileName(imagePath);
            FileNameInfo fileNameInfo = ParseFileName(fileName);
            if (!string.IsNullOrWhiteSpace(options.ForcedPipelineCode))
            {
                fileNameInfo.PipelineCode = options.ForcedPipelineCode;
            }

            string resultImagePath = GetResultImagePath(imagePath, resultFolderPath, options);

            using (Mat image = Cv2.ImRead(imagePath, ImreadModes.Color))
            {
                if (image.Empty())
                {
                    WriteLog(log, "Image read failed: " + fileName);
                    return;
                }

                // 去掉图片上叠加的彩色字符，再传给 pipeline。
                using Mat cleanImage = RemoveColorText(image);

                if (string.Equals(fileNameInfo.PipelineCode, "R", StringComparison.OrdinalIgnoreCase))

                {//R
                    
                    ProcessRImage(cleanImage, image, fileName, fileNameInfo, resultImagePath, rRecords, problemImageFileNames, micronScale, log);
                    
                    
                }
                else
                {//B
                    ProcessBImage(cleanImage, image, fileName, fileNameInfo, resultImagePath, bRecords, problemImageFileNames, micronScale, log);
                }
            }
        }

        private static string GetResultImagePath(string imagePath, string resultFolderPath, FolderProcessOptions options)
        {
            string fileName = Path.GetFileName(imagePath);
            if (!options.OutputImagesOnly)
            {
                return Path.Combine(resultFolderPath, fileName);
            }

            string checkedFileName = BuildCheckedFileName(fileName);
            string? imageFolderPath = Path.GetDirectoryName(imagePath);
            if (!string.IsNullOrWhiteSpace(imageFolderPath))
            {
                return Path.Combine(imageFolderPath, checkedFileName);
            }

            return Path.Combine(resultFolderPath, checkedFileName);
        }

        private static bool HasExistingResultImage(string inputFolderPath, string imagePath, FolderProcessOptions options)
        {
            string resultFolderPath = options.OutputImagesOnly
                ? inputFolderPath
                : Path.Combine(inputFolderPath, "Result");
            string resultImagePath = GetResultImagePath(imagePath, resultFolderPath, options);
            return File.Exists(resultImagePath);
        }

        private static string BuildCheckedFileName(string fileName)
        {
            string extension = Path.GetExtension(fileName);
            string nameWithoutExtension = Path.GetFileNameWithoutExtension(fileName);
            return nameWithoutExtension + "-Checked" + extension;
        }

        // 去掉图片上叠加的彩色字符。
        // BGR 通道优势精确找字
        private static Mat RemoveColorText(Mat image)
        {
            if (image == null || image.Empty())
            {
                return new Mat();
            }

            using Mat colorMask = BuildDominantColorTextMask(image);
            if (Cv2.CountNonZero(colorMask) == 0)
            {
                return image.Clone();
            }

            // 连续两次 MaxFilter(3)，3x3 膨胀两次。
            using Mat kernel = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(3, 3));
            using Mat dilatedMask = new Mat();
            Cv2.Dilate(colorMask, dilatedMask, kernel, iterations: 2);

            Mat result = new Mat();
            Cv2.Inpaint(image, dilatedMask, result, 3, InpaintTypes.Telea);
            return result;
        }

        // 生成彩色文字 mask。OpenCV 读进来的图片是 BGR 顺序。
        private static Mat BuildDominantColorTextMask(Mat image)
        {
            int rows = image.Rows;
            int cols = image.Cols;
            Mat mask = new Mat(rows, cols, MatType.CV_8UC1, Scalar.All(0));

            for (int y = 0; y < rows; y++)
            {
                for (int x = 0; x < cols; x++)
                {
                    Vec3b pixel = image.At<Vec3b>(y, x);
                    int blue = pixel.Item0;
                    int green = pixel.Item1;
                    int red = pixel.Item2;

                    bool isGreenText =
                        green >= 110 &&
                        green - red >= 45 &&
                        green - blue >= 45 &&
                        red <= 180 &&
                        blue <= 180;

                    bool isRedText =
                        red >= 110 &&
                        red - green >= 45 &&
                        red - blue >= 45 &&
                        green <= 190 &&
                        blue <= 190;

                    bool isBlueText =
                        blue >= 110 &&
                        blue - red >= 45 &&
                        blue - green >= 45 &&
                        red <= 190 &&
                        green <= 190;

                    bool isPinkText =
                        red >= 130 &&
                        blue >= 100 &&
                        red - green >= 45 &&
                        blue - green >= 35 &&
                        green <= 190;

                    if (isGreenText || isRedText || isBlueText || isPinkText)
                    {
                        mask.Set(y, x, (byte)255);
                    }
                }
            }

            return mask;
        }

        private static void ProcessBImage(Mat image, Mat originalImage, string fileName, FileNameInfo fileNameInfo, string resultImagePath, List<MeasureRecord> records, HashSet<string> problemImageFileNames, double micronScale, Action<string>? log)
        {
            PinglineResult result = BLT.Instance.Process(image, true, micronScale);

            using (Mat output = GetBOutputImage(originalImage, image, result))
            {
                Cv2.ImWrite(resultImagePath, output);
            }

            if (result.IsSuccess && result.Measurement.HasValue)
            {
                MeasureRecord record = MeasureRecord.CreateFromB(fileNameInfo, result.Measurement.Value, micronScale);
                records.Add(record);
                if (IsBProblem(record))
                {
                    problemImageFileNames.Add(fileName);
                }
                WriteLog(log, fileName + " -> B OK");
                return;
            }

            problemImageFileNames.Add(fileName);
            WriteLog(log, fileName + " -> " + result.Message);
        }

        private static void ProcessRImage(Mat image, Mat originalImage, string fileName, FileNameInfo fileNameInfo, string resultImagePath, List<MeasureRecord> records, HashSet<string> problemImageFileNames, double micronScale, Action<string>? log)
        {
            PipelineResult result = RPMD.Instance.Process(micronScale, image);

            using (Mat output = GetROutputImage(originalImage, image, result))
            {
                Cv2.ImWrite(resultImagePath, output);
            }

            if (!result.Success)
            {
                problemImageFileNames.Add(fileName);
                WriteLog(log, fileName + " -> " + result.Error);
                return;
            }

            if (result.Measurements == null || result.Measurements.Count == 0)
            {
                problemImageFileNames.Add(fileName);
                WriteLog(log, fileName + " -> R OK，但是没有 Measurements，未写入 xlsx。");
                return;
            }

            foreach (RoiMeasurement measurement in result.Measurements)
            {
                MeasureRecord record = MeasureRecord.CreateFromR(fileNameInfo, measurement);
                records.Add(record);
                if (IsRProblem(record))
                {
                    problemImageFileNames.Add(fileName);
                }
            }

            WriteLog(log, fileName + " -> R OK，写入 " + result.Measurements.Count + " 条记录。");
        }

        private static bool IsBProblem(MeasureRecord record)
        {
            return !record.LowestPoint.HasValue || !record.NormalPoint.HasValue;
        }

        private static bool IsRProblem(MeasureRecord record)
        {
            return !record.LowestPoint.HasValue ||
                   !record.NormalPoint.HasValue ||
                   !record.SolderBallThickness.HasValue ||
                   !record.SolderBallSize.HasValue;
        }

        private static FileNameInfo ParseFileName(string fileName)
        {
            string nameWithoutExtension = Path.GetFileNameWithoutExtension(fileName);
            string[] parts = nameWithoutExtension.Split(' ', StringSplitOptions.RemoveEmptyEntries);

            FileNameInfo info = new FileNameInfo();
            info.SourceFileName = fileName;
            info.DateText = string.Empty;
            info.MachineNo = string.Empty;
            info.BatchNo = nameWithoutExtension;
            info.PipelineCode = "B";

            if (parts.Length >= 1)
            {
                info.DateText = parts[0];
            }

            if (parts.Length >= 2)
            {
                info.MachineNo = parts[1];
            }

            if (parts.Length >= 3)
            {
                info.BatchNo = parts[2];
            }

            if (parts.Length >= 4 &&
                (string.Equals(parts[parts.Length - 1], "B", StringComparison.OrdinalIgnoreCase) ||
                 string.Equals(parts[parts.Length - 1], "R", StringComparison.OrdinalIgnoreCase)))
            {
                info.PipelineCode = parts[parts.Length - 1];
            }

            return info;
        }

        private static Mat GetBOutputImage(Mat originalImage, Mat pipelineInputImage, PinglineResult result)
        {
            if (result.OutputImage.Empty())
            {
                return originalImage.Clone();
            }

            return ApplyPipelineOverlayToOriginal(originalImage, pipelineInputImage, result.OutputImage);
        }

        private static Mat GetROutputImage(Mat originalImage, Mat pipelineInputImage, PipelineResult result)
        {
            if (result.FullOverlay != null && !result.FullOverlay.Empty())
            {
                return ApplyPipelineOverlayToOriginal(originalImage, pipelineInputImage, result.FullOverlay);
            }

            return originalImage.Clone();
        }

        private static Mat ApplyPipelineOverlayToOriginal(Mat originalImage, Mat pipelineInputImage, Mat pipelineOutputImage)
        {
            if (originalImage.Empty())
            {
                return pipelineOutputImage.Clone();
            }

            if (pipelineInputImage.Empty() ||
                pipelineOutputImage.Empty() ||
                originalImage.Size() != pipelineInputImage.Size() ||
                originalImage.Size() != pipelineOutputImage.Size() ||
                originalImage.Type() != pipelineInputImage.Type() ||
                originalImage.Type() != pipelineOutputImage.Type())
            {
                return pipelineOutputImage.Clone();
            }

            Mat output = originalImage.Clone();

            using Mat diff = new Mat();
            using Mat grayDiff = new Mat();
            using Mat overlayMask = new Mat();
            Cv2.Absdiff(pipelineOutputImage, pipelineInputImage, diff);
            Cv2.CvtColor(diff, grayDiff, ColorConversionCodes.BGR2GRAY);
            Cv2.Threshold(grayDiff, overlayMask, 0, 255, ThresholdTypes.Binary);
            pipelineOutputImage.CopyTo(output, overlayMask);

            return output;
        }

        private static void WriteXlsx(string xlsxPath, IReadOnlyList<MeasureRecord> records, XlsxSheetType sheetType)
        {
            if (File.Exists(xlsxPath))
            {
                File.Delete(xlsxPath);
            }

            using (ZipArchive archive = ZipFile.Open(xlsxPath, ZipArchiveMode.Create))
            {
                AddTextEntry(archive, "[Content_Types].xml", BuildContentTypesXml());
                AddTextEntry(archive, "_rels/.rels", BuildRootRelsXml());
                AddTextEntry(archive, "xl/workbook.xml", BuildWorkbookXml());
                AddTextEntry(archive, "xl/_rels/workbook.xml.rels", BuildWorkbookRelsXml());
                AddTextEntry(archive, "xl/worksheets/sheet1.xml", BuildSheetXml(records, sheetType));
            }
        }

        private static string BuildContentTypesXml()
        {
            return "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" +
                   "<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">" +
                   "<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>" +
                   "<Default Extension=\"xml\" ContentType=\"application/xml\"/>" +
                   "<Override PartName=\"/xl/workbook.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml\"/>" +
                   "<Override PartName=\"/xl/worksheets/sheet1.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml\"/>" +
                   "</Types>";
        }

        private static string BuildRootRelsXml()
        {
            return "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" +
                   "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">" +
                   "<Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" Target=\"xl/workbook.xml\"/>" +
                   "</Relationships>";
        }

        private static string BuildWorkbookXml()
        {
            return "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" +
                   "<workbook xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" " +
                   "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\">" +
                   "<sheets><sheet name=\"结果\" sheetId=\"1\" r:id=\"rId1\"/></sheets>" +
                   "</workbook>";
        }

        private static string BuildWorkbookRelsXml()
        {
            return "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" +
                   "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">" +
                   "<Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet\" Target=\"worksheets/sheet1.xml\"/>" +
                   "</Relationships>";
        }

        private static string BuildSheetXml(IReadOnlyList<MeasureRecord> records, XlsxSheetType sheetType)
        {
            if (sheetType == XlsxSheetType.Blt)
            {
                return BuildBltSheetXml(records);
            }

            return BuildRSheetXml(records);
        }

        private static string BuildBltSheetXml(IReadOnlyList<MeasureRecord> records)
        {
            StringBuilder builder = new StringBuilder();
            builder.Append("<?xml version=\"1.0\" encoding=\"UTF-8\"?>");
            builder.Append("<worksheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\">");
            builder.Append("<cols>");
            builder.Append("<col min=\"1\" max=\"3\" width=\"18\" customWidth=\"1\"/>");
            builder.Append("<col min=\"4\" max=\"5\" width=\"14\" customWidth=\"1\"/>");
            builder.Append("</cols>");
            builder.Append("<sheetData>");

            builder.Append("<row r=\"1\">");
            AppendTextCell(builder, "A1", "日期");
            AppendTextCell(builder, "B1", "机台号");
            AppendTextCell(builder, "C1", "批次号");
            AppendTextCell(builder, "D1", "最小值(微米)");
            AppendTextCell(builder, "E1", "最大值(微米)");
            builder.Append("</row>");

            for (int index = 0; index < records.Count; index++)
            {
                int rowNumber = index + 2;
                MeasureRecord record = records[index];

                builder.Append("<row r=\"");
                builder.Append(rowNumber.ToString(CultureInfo.InvariantCulture));
                builder.Append("\">");
                AppendTextCell(builder, "A" + rowNumber, record.DateText);
                AppendTextCell(builder, "B" + rowNumber, record.MachineNo);
                AppendTextCell(builder, "C" + rowNumber, record.BatchNo);
                AppendNullableNumberCell(builder, "D" + rowNumber, record.LowestPoint);
                AppendNullableNumberCell(builder, "E" + rowNumber, record.NormalPoint);
                builder.Append("</row>");
            }

            builder.Append("</sheetData>");
            builder.Append("</worksheet>");
            return builder.ToString();
        }

        private static string BuildRSheetXml(IReadOnlyList<MeasureRecord> records)
        {
            StringBuilder builder = new StringBuilder();
            builder.Append("<?xml version=\"1.0\" encoding=\"UTF-8\"?>");
            builder.Append("<worksheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\">");
            builder.Append("<cols>");
            builder.Append("<col min=\"1\" max=\"3\" width=\"18\" customWidth=\"1\"/>");
            builder.Append("<col min=\"4\" max=\"7\" width=\"14\" customWidth=\"1\"/>");
            builder.Append("</cols>");
            builder.Append("<sheetData>");

            builder.Append("<row r=\"1\">");
            AppendTextCell(builder, "A1", "日期");
            AppendTextCell(builder, "B1", "机台号");
            AppendTextCell(builder, "C1", "批次号");
            AppendTextCell(builder, "D1", "最低点(微米)");
            AppendTextCell(builder, "E1", "正常点(微米)");
            AppendTextCell(builder, "F1", "焊球厚度(微米)");
            AppendTextCell(builder, "G1", "焊球大小(微米)");
            builder.Append("</row>");

            for (int index = 0; index < records.Count; index++)
            {
                int rowNumber = index + 2;
                MeasureRecord record = records[index];

                builder.Append("<row r=\"");
                builder.Append(rowNumber.ToString(CultureInfo.InvariantCulture));
                builder.Append("\">");
                AppendTextCell(builder, "A" + rowNumber, record.DateText);
                AppendTextCell(builder, "B" + rowNumber, record.MachineNo);
                AppendTextCell(builder, "C" + rowNumber, record.BatchNo);
                AppendNullableNumberCell(builder, "D" + rowNumber, record.LowestPoint);
                AppendNullableNumberCell(builder, "E" + rowNumber, record.NormalPoint);
                AppendNullableNumberCell(builder, "F" + rowNumber, record.SolderBallThickness);
                AppendNullableNumberCell(builder, "G" + rowNumber, record.SolderBallSize);
                builder.Append("</row>");
            }

            builder.Append("</sheetData>");
            builder.Append("</worksheet>");
            return builder.ToString();
        }

        private static void AppendTextCell(StringBuilder builder, string cellName, string value)
        {
            builder.Append("<c r=\"");
            builder.Append(cellName);
            builder.Append("\" t=\"inlineStr\"><is><t>");
            builder.Append(EscapeXml(value));
            builder.Append("</t></is></c>");
        }

        private static void AppendNullableNumberCell(StringBuilder builder, string cellName, double? value)
        {
            if (!value.HasValue)
            {
                builder.Append("<c r=\"");
                builder.Append(cellName);
                builder.Append("\"/>");
                return;
            }

            builder.Append("<c r=\"");
            builder.Append(cellName);
            builder.Append("\"><v>");
            builder.Append(value.Value.ToString("0.###", CultureInfo.InvariantCulture));
            builder.Append("</v></c>");
        }

        private static void AddTextEntry(ZipArchive archive, string entryName, string text)
        {
            ZipArchiveEntry entry = archive.CreateEntry(entryName);
            using (Stream stream = entry.Open())
            using (StreamWriter writer = new StreamWriter(stream, new UTF8Encoding(false)))
            {
                writer.Write(text);
            }
        }

        private static string EscapeXml(string value)
        {
            string text = value;
            text = text.Replace("&", "&amp;");
            text = text.Replace("<", "&lt;");
            text = text.Replace(">", "&gt;");
            text = text.Replace("\"", "&quot;");
            text = text.Replace("'", "&apos;");
            return text;
        }

        private static void WriteLog(Action<string>? log, string message)
        {
            if (log != null)
            {
                log(message);
            }
        }
    }

    internal enum XlsxSheetType
    {
        Blt,
        R
    }

    internal class SysmainProcessResult
    {
        public string ResultFolderPath { get; set; } = string.Empty;
        public string XlsxPath { get; set; } = string.Empty;
        public IReadOnlyList<string> XlsxPaths { get; set; } = new List<string>();
        public IReadOnlyList<MeasureRecord> Records { get; set; } = new List<MeasureRecord>();
        public IReadOnlyList<string> ProblemImageFileNames { get; set; } = new List<string>();
        public bool OutputImagesOnly { get; set; }
    }

    internal class FileNameInfo
    {
        public string SourceFileName { get; set; } = string.Empty;
        public string DateText { get; set; } = string.Empty;
        public string MachineNo { get; set; } = string.Empty;
        public string BatchNo { get; set; } = string.Empty;
        public string PipelineCode { get; set; } = string.Empty;
    }

    internal class MeasureRecord
    {
        public string SourceFileName { get; set; } = string.Empty;
        public string PipelineCode { get; set; } = string.Empty;
        public string DateText { get; set; } = string.Empty;
        public string MachineNo { get; set; } = string.Empty;
        public string BatchNo { get; set; } = string.Empty;
        public double? LowestPoint { get; set; }
        public double? NormalPoint { get; set; }
        public double? SolderBallThickness { get; set; }
        public double? SolderBallSize { get; set; }

        public static MeasureRecord CreateFromB(FileNameInfo fileNameInfo, Measurement measurement, double micronScale)
        {
            int leftHeight = measurement.Left.Height;
            int rightHeight = measurement.Right.Height;

            MeasureRecord record = CreateBase(fileNameInfo);
            record.LowestPoint = ScaleValue(Math.Min(leftHeight, rightHeight), micronScale);
            record.NormalPoint = ScaleValue(Math.Max(leftHeight, rightHeight), micronScale);
            return record;
        }

        public static MeasureRecord CreateFromR(FileNameInfo fileNameInfo, RoiMeasurement measurement, double micronScale=1)
        {
            MeasureRecord record = CreateBase(fileNameInfo);
            record.LowestPoint = ScaleValue(measurement.LowestRedTopHeight, micronScale);
            record.NormalPoint = ScaleValue(measurement.LeftSmoothHeight, micronScale);

            if (!record.NormalPoint.HasValue)
            {
                record.NormalPoint = ScaleValue(measurement.RightSmoothHeight, micronScale);
            }

            record.SolderBallThickness = ScaleValue(measurement.H1, micronScale);
            record.SolderBallSize = ScaleValue(measurement.H2, micronScale);
            return record;
        }

        private static double ScaleValue(double value, double micronScale)
        {
            return value * micronScale;
        }

        private static double? ScaleValue(double? value, double micronScale)
        {
            if (!value.HasValue)
            {
                return null;
            }

            return value.Value * micronScale;
        }

        private static MeasureRecord CreateBase(FileNameInfo fileNameInfo)
        {
            MeasureRecord record = new MeasureRecord();
            record.SourceFileName = fileNameInfo.SourceFileName;
            record.PipelineCode = fileNameInfo.PipelineCode;
            record.DateText = fileNameInfo.DateText;
            record.MachineNo = fileNameInfo.MachineNo;
            record.BatchNo = fileNameInfo.BatchNo;
            return record;
        }
    }
}
