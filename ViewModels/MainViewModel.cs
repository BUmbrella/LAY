using System.Collections.ObjectModel;
using System.IO;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using LAY.Main;
using LAY.Models;
using LAY.Services;
using Prism.Commands;
using Prism.Mvvm;

namespace LAY.ViewModels
{
    // MainViewModel 是主界面的数据和命令控制类。
    // XAML 界面上的图片列表、预览图、日志、启动/停止按钮、缩放按钮，基本都绑定到这个类。
    public class MainViewModel : BindableBase
    {
        // 图片文件夹服务。它只负责从某个文件夹中读取图片文件列表，供界面左侧列表显示。
        private readonly IPhotoFolderService _photoFolderService;

        // 检测主流程对象。界面点击启动后，最终会调用 sysmain.Start 处理整个输入文件夹。
        private readonly sysmain _sysmain;

        // 用户拖入或选择的原始输入文件夹路径。后续启动检测、切回输入图，都依赖这个路径。
        private string? _sourceFolderPath;

        // 当前界面正在显示的文件夹路径。可能是输入文件夹，也可能是结果文件夹。
        private string? _currentFolderPath;

        // 当前选中图片的文件名。这个属性通常显示在界面上，用来提示用户正在看哪张图。
        private string? _selectedPhotoPath;

        // 当前图片列表中被选中的图片对象。设置它时会同步刷新右侧预览图。
        private PhotoItem? _selectedPhoto;

        // 右侧图片预览控件使用的图片源。这里保存的是已经加载到内存里的 BitmapImage。
        private ImageSource? _previewImage;

        // 是否有可显示的预览图。界面可以用它控制图片区域或空状态提示的显示。
        private bool _hasPreviewImage;

        // 当前是否正在检测。它会影响启动/停止按钮文字，也用于防止重复启动。
        private bool _isRunning;

        // 当前图片预览缩放比例。1.0 表示原始比例，数值越大图片越放大。
        private double _zoomFactor;

        // 构造函数由 Prism 容器创建。
        // 这里完成服务注入、主流程对象创建、集合初始化和按钮命令绑定。
        public MainViewModel(IPhotoFolderService photoFolderService)
        {
            _photoFolderService = photoFolderService;
            _sysmain = new sysmain();
            _zoomFactor = 1.0;

            PhotoItems = new ObservableCollection<PhotoItem>();
            Logs = new ObservableCollection<string>();

            ToggleRunCommand = new DelegateCommand(ToggleRun);
            ShowInputCommand = new DelegateCommand(ShowInput);
            ShowResultsCommand = new DelegateCommand(ShowResults);
            ZoomInCommand = new DelegateCommand(ZoomIn);
            ZoomOutCommand = new DelegateCommand(ZoomOut);
            ResetZoomCommand = new DelegateCommand(ResetZoom);

            Log("准备就绪，拖入图片文件夹开始使用。");
        }

        // 图片列表数据源。界面左侧列表绑定它，集合变化后界面会自动刷新。
        public ObservableCollection<PhotoItem> PhotoItems { get; private set; }

        // 日志列表数据源。检测过程中的提示、错误、保存路径都会写到这里。
        public ObservableCollection<string> Logs { get; private set; }

        // 启动/停止按钮命令。按钮点击后会进入 ToggleRun 方法。
        public DelegateCommand ToggleRunCommand { get; private set; }

        // “查看输入图”按钮命令。用于从结果图列表切回原始输入图列表。
        public DelegateCommand ShowInputCommand { get; private set; }

        // “查看结果图”按钮命令。用于切换到 Result 文件夹查看检测后的图片。
        public DelegateCommand ShowResultsCommand { get; private set; }

        // 放大按钮命令。每次点击会把预览图放大一点。
        public DelegateCommand ZoomInCommand { get; private set; }

        // 缩小按钮命令。每次点击会把预览图缩小一点。
        public DelegateCommand ZoomOutCommand { get; private set; }

        // 重置缩放按钮命令。点击后把预览图缩放比例恢复到 1.0。
        public DelegateCommand ResetZoomCommand { get; private set; }

        // 启动/停止按钮显示的文字。
        // 检测中显示“停止”，空闲时显示“启动”。
        public string StartStopText
        {
            get
            {
                if (IsRunning)
                {
                    return "停止";
                }

                return "启动";
            }
        }

        // 当前界面显示的文件夹路径。
        // set 是 private，避免界面外部随便改路径，统一由 LoadPhotos / ShowResults / ShowInput 控制。
        public string? CurrentFolderPath
        {
            get
            {
                return _currentFolderPath;
            }
            private set
            {
                SetProperty(ref _currentFolderPath, value);
            }
        }

        // 当前选中的图片文件名。
        // 这里保存文件名而不是完整路径，是为了界面显示更简洁。
        public string? SelectedPhotoPath
        {
            get
            {
                return _selectedPhotoPath;
            }
            private set
            {
                SetProperty(ref _selectedPhotoPath, value);
            }
        }

        // 当前选中的图片项。
        // 当用户在列表里选中另一张图时，这里会自动加载对应图片到右侧预览区。
        public PhotoItem? SelectedPhoto
        {
            get
            {
                return _selectedPhoto;
            }
            set
            {
                if (SetProperty(ref _selectedPhoto, value))
                {
                    if (value == null)
                    {
                        SelectedPhotoPath = null;
                        UpdatePreviewImage(null);
                    }
                    else
                    {
                        SelectedPhotoPath = value.FileName;
                        UpdatePreviewImage(value.FullPath);
                    }
                }
            }
        }

        // 当前右侧预览图。
        // WPF 图片控件绑定这个属性后，只要它变化，界面就会自动换图。
        public ImageSource? PreviewImage
        {
            get
            {
                return _previewImage;
            }
            private set
            {
                SetProperty(ref _previewImage, value);
            }
        }

        // 是否存在可预览图片。
        // 主要用于界面判断要显示图片，还是显示“没有图片”的提示。
        public bool HasPreviewImage
        {
            get
            {
                return _hasPreviewImage;
            }
            private set
            {
                SetProperty(ref _hasPreviewImage, value);
            }
        }

        // 当前是否正在检测。
        // 修改这个属性时，要同时通知 StartStopText 也变了，因为按钮文字依赖它。
        public bool IsRunning
        {
            get
            {
                return _isRunning;
            }
            private set
            {
                if (SetProperty(ref _isRunning, value))
                {
                    RaisePropertyChanged(nameof(StartStopText));
                }
            }
        }

        // 当前预览缩放比例。
        // 界面图片控件会根据这个值做缩放显示。
        public double ZoomFactor
        {
            get
            {
                return _zoomFactor;
            }
            private set
            {
                SetProperty(ref _zoomFactor, value);
            }
        }

        // 外部调用的加载入口。
        // 一般是用户拖入文件夹后，窗口代码把文件夹路径传进来。
        public void LoadFolder(string folderPath)
        {
            if (!sysmain.TryGetMagnificationFromFolderName(folderPath, out _))
            {
                ShowPrompt("请修改文件夹名字，提供放大倍数");
                return;
            }

            _sourceFolderPath = folderPath;
            LoadPhotos(folderPath, folderPath);
            Log("已加载输入文件夹：" + folderPath);
            Log("图片数量：" + PhotoItems.Count);
        }

        // 启动/停止按钮的核心逻辑。
        // 空闲时点击会启动检测；检测中点击会请求 sysmain 停止。
        private async void ToggleRun()
        {
            // 如果正在运行，再次点击按钮表示请求停止。
            // Stop 只是设置停止标记，真正退出发生在 sysmain 处理下一张图片之前。
            if (IsRunning)
            {
                _sysmain.Stop();
                Log("正在请求停止检测...");
                return;
            }

            if (!HasInputPhotos())
            {
                ShowPrompt("请先选择检测照片");
                return;
            }

            IsRunning = true;
            Log("开始检测。");

            try
            {
                // 检测图片比较耗时，所以放到后台线程执行，避免界面卡死。
                SysmainProcessResult result = await Task.Run(ProcessCurrentFolder);

                Log("距离结果已保存：" + result.XlsxPath);

                // 检测完成后，自动切换到 Result 文件夹，方便用户直接看结果图。
                LoadPhotos(result.ResultFolderPath, result.ResultFolderPath);
                Log("检测完成。");
            }
            catch (Exception ex)
            {
                Log("检测失败：" + ex.Message);
            }
            finally
            {
                IsRunning = false;
            }
        }

        // 切换回输入文件夹显示原图。
        private void ShowInput()
        {
            if (string.IsNullOrWhiteSpace(_sourceFolderPath) || !Directory.Exists(_sourceFolderPath))
            {
                ShowPrompt("请先选择检测照片");
                return;
            }

            LoadPhotos(_sourceFolderPath, _sourceFolderPath);

            if (PhotoItems.Count == 0)
            {
                ShowPrompt("请先选择检测照片");
            }
        }

        // 切换到 Result 文件夹显示检测结果图。
        private void ShowResults()
        {
            if (string.IsNullOrWhiteSpace(_sourceFolderPath))
            {
                ShowPrompt("请先进行检测");
                return;
            }

            string resultFolderPath = Path.Combine(_sourceFolderPath, "Result");
            IReadOnlyList<PhotoItem> photos = _photoFolderService.GetPhotos(resultFolderPath);
            if (photos.Count == 0)
            {
                ShowPrompt("请先进行检测");
                return;
            }

            ReplacePhotos(photos);
            CurrentFolderPath = resultFolderPath;
            SelectedPhoto = GetFirstPhoto();
            SetZoom(1.0);

            Log("已切换到输出文件夹：" + resultFolderPath);
            Log("结果图片数量：" + PhotoItems.Count);
        }

        // 从指定文件夹读取图片，然后刷新界面列表、当前路径、选中图片和缩放比例。
        private void LoadPhotos(string folderPath, string displayPath)
        {
            IReadOnlyList<PhotoItem> photos = _photoFolderService.GetPhotos(folderPath);
            ReplacePhotos(photos);

            CurrentFolderPath = displayPath;
            SelectedPhoto = GetFirstPhoto();
            SetZoom(1.0);
        }

        // 判断当前是否有可检测的输入图片。
        // 启动检测前必须通过这个检查，否则 sysmain 没有有效输入。
        private bool HasInputPhotos()
        {
            if (string.IsNullOrWhiteSpace(_sourceFolderPath))
            {
                return false;
            }

            if (!Directory.Exists(_sourceFolderPath))
            {
                return false;
            }

            IReadOnlyList<PhotoItem> photos = _photoFolderService.GetPhotos(_sourceFolderPath);
            return photos.Count > 0;
        }

        // 用新的图片列表替换界面当前列表。
        // ObservableCollection 不能直接替换绑定对象，所以这里清空后逐个添加。
        private void ReplacePhotos(IReadOnlyList<PhotoItem> photos)
        {
            PhotoItems.Clear();

            foreach (PhotoItem photo in photos)
            {
                PhotoItems.Add(photo);
            }
        }

        // 获取当前图片列表中的第一张图片。
        // 切换文件夹后默认选中第一张，用户就能马上看到预览。
        private PhotoItem? GetFirstPhoto()
        {
            if (PhotoItems.Count == 0)
            {
                return null;
            }

            return PhotoItems[0];
        }

        // 后台线程实际调用的检测方法。
        // 这里单独封装，是为了 Task.Run 调用时逻辑更清楚。
        private SysmainProcessResult ProcessCurrentFolder()
        {
            if (string.IsNullOrWhiteSpace(_sourceFolderPath))
            {
                throw new InvalidOperationException("Input folder is empty.");
            }

            return _sysmain.Start(_sourceFolderPath, Log);
        }

        // 根据图片路径刷新右侧预览图。
        // 使用 OnLoad 是为了加载完成后释放文件占用，避免结果图被覆盖时文件仍被锁住。
        private void UpdatePreviewImage(string? imagePath)
        {
            if (string.IsNullOrWhiteSpace(imagePath) || !File.Exists(imagePath))
            {
                PreviewImage = null;
                HasPreviewImage = false;
                return;
            }

            try
            {
                BitmapImage bitmap = new BitmapImage();
                bitmap.BeginInit();
                bitmap.CacheOption = BitmapCacheOption.OnLoad;
                bitmap.UriSource = new Uri(imagePath, UriKind.Absolute);
                bitmap.EndInit();
                bitmap.Freeze();

                PreviewImage = bitmap;
                HasPreviewImage = true;
            }
            catch (Exception ex)
            {
                PreviewImage = null;
                HasPreviewImage = false;
                Log("图片加载失败：" + Path.GetFileName(imagePath) + "，" + ex.Message);
            }
        }

        // 鼠标滚轮缩放入口。
        // delta 大于 0 表示向上滚动，放大；小于等于 0 表示向下滚动，缩小。
        public void ZoomByWheel(int delta)
        {
            double nextZoom;

            if (delta > 0)
            {
                nextZoom = ZoomFactor + 0.1;
            }
            else
            {
                nextZoom = ZoomFactor - 0.1;
            }

            SetZoom(nextZoom);
        }

        // 设置缩放比例，并限制最小和最大范围。
        // 这里限制在 0.2 到 4.0，避免图片太小看不到，或者太大导致操作不方便。
        public void SetZoom(double zoomFactor)
        {
            double clamped = Math.Max(0.2, Math.Min(4.0, zoomFactor));
            ZoomFactor = clamped;
        }

        // 放大按钮调用的方法。
        private void ZoomIn()
        {
            SetZoom(ZoomFactor + 0.1);
        }

        // 缩小按钮调用的方法。
        private void ZoomOut()
        {
            SetZoom(ZoomFactor - 0.1);
        }

        // 重置缩放按钮调用的方法。
        private void ResetZoom()
        {
            SetZoom(1.0);
        }

        // 写界面日志。
        // 因为检测流程在后台线程运行，后台线程不能直接修改 WPF 集合，所以要判断并切回 UI 线程。
        private void Log(string message)
        {
            if (Application.Current != null && !Application.Current.Dispatcher.CheckAccess())
            {
                Application.Current.Dispatcher.Invoke(delegate
                {
                    AddLog(message);
                });
                return;
            }

            AddLog(message);
        }

        // 真正往日志集合里添加一行文字。
        // 前面加当前时间，方便看每一步检测发生的先后顺序。
        private void AddLog(string message)
        {
            Logs.Insert(0, DateTime.Now.ToString("HH:mm:ss") + "  " + message);

            while (Logs.Count > 200)
            {
                Logs.RemoveAt(Logs.Count - 1);
            }
        }

        // 统一弹出提示框。
        // 这里集中封装，后面如果要改提示框标题或图标，只改一个地方。
        private static void ShowPrompt(string message)
        {
            MessageBox.Show(message, "提示", MessageBoxButton.OK, MessageBoxImage.Information);
        }
    }
}
