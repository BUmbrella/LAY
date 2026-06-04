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
            _stopRequested = false;

            if (string.IsNullOrWhiteSpace(inputFolderPath) || !Directory.Exists(inputFolderPath))
            {
                throw new DirectoryNotFoundException("Input folder not found: " + inputFolderPath);
            }
            //读取倍数设置文件
            string magnification = GetMagnificationFromFolderName(inputFolderPath);

            double micronScale = CalculateMicronScale(magnification, log);
            string resultFolderPath = Path.Combine(inputFolderPath, "Result");
            Directory.CreateDirectory(resultFolderPath);

            string bXlsxPath = Path.Combine(resultFolderPath, "BLT_measure_results.xlsx");
            string rXlsxPath = Path.Combine(resultFolderPath, "R_measure_results.xlsx");
            DeleteExistingResultFile(Path.Combine(resultFolderPath, "measure_results.xlsx"));
            DeleteExistingResultFile(bXlsxPath);
            DeleteExistingResultFile(rXlsxPath);

            List<MeasureRecord> bRecords = new List<MeasureRecord>();
            List<MeasureRecord> rRecords = new List<MeasureRecord>();
            HashSet<string> problemImageFileNames = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            string[] imagePaths = GetImagePaths(inputFolderPath);

            foreach (string imagePath in imagePaths)
            {
                if (_stopRequested)
                {
                    WriteLog(log, "Stop requested.");
                    break;
                }

                ProcessOneImage(imagePath, resultFolderPath, bRecords, rRecords, problemImageFileNames, micronScale, log);
            }

            List<string> xlsxPaths = new List<string>();
            if (bRecords.Count > 0)
            {
                WriteXlsx(bXlsxPath, bRecords, XlsxSheetType.Blt);
                xlsxPaths.Add(bXlsxPath);
            }

            if (rRecords.Count > 0)
            {
                WriteXlsx(rXlsxPath, rRecords, XlsxSheetType.R);
                xlsxPaths.Add(rXlsxPath);
            }

            SysmainProcessResult result = new SysmainProcessResult();
            result.ResultFolderPath = resultFolderPath;
            result.XlsxPath = string.Join("; ", xlsxPaths);
            result.XlsxPaths = xlsxPaths;
            result.Records = bRecords.Concat(rRecords).ToList();
            result.ProblemImageFileNames = problemImageFileNames.ToList();
            return result;
        }

        private static void DeleteExistingResultFile(string filePath)
        {
            if (File.Exists(filePath))
            {
                File.Delete(filePath);
            }
        }

        private static string[] GetImagePaths(string inputFolderPath)
        {
            List<string> imagePaths = new List<string>();
            string[] allFiles = Directory.GetFiles(inputFolderPath, "*.*", SearchOption.TopDirectoryOnly);

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

            return imagePaths.ToArray();
        }

        public static bool TryGetMagnificationFromFolderName(string folderPath, out string magnification)
        {
            magnification = GetTrailingDigits(new DirectoryInfo(folderPath).Name);
            return magnification.Length > 0;
        }

        private static string GetMagnificationFromFolderName(string inputFolderPath)
        {
            if (TryGetMagnificationFromFolderName(inputFolderPath, out string magnification))
            {
                return magnification;
            }

            throw new InvalidOperationException("请修改文件夹名字，提供放大倍数");
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
        private static void ProcessOneImage(string imagePath, string resultFolderPath, List<MeasureRecord> bRecords, List<MeasureRecord> rRecords, HashSet<string> problemImageFileNames, double micronScale, Action<string>? log)
        {
            string fileName = Path.GetFileName(imagePath);
            FileNameInfo fileNameInfo = ParseFileName(fileName);
            string resultImagePath = Path.Combine(resultFolderPath, fileName);

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
