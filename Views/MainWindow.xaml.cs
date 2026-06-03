using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using LAY.ViewModels;

namespace LAY.Views
{
    public partial class MainWindow : Window
    {
        private static readonly HashSet<string> SupportedPhotoExtensions = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
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

        private bool _isPreviewPanning;
        private Point _lastPreviewPanPoint;

        public MainWindow(MainViewModel viewModel)
        {
            InitializeComponent();
            DataContext = viewModel;
        }

        private void RootWindow_PreviewDragEnter(object sender, DragEventArgs e)
        {
            // 拖入文件夹或图片时，显示可复制的拖拽效果。
            if (IsDropSupported(e))
            {
                e.Effects = DragDropEffects.Copy;
            }
            else
            {
                e.Effects = DragDropEffects.None;
            }

            e.Handled = true;
        }

        private void RootWindow_PreviewDragOver(object sender, DragEventArgs e)
        {
            if (IsDropSupported(e))
            {
                e.Effects = DragDropEffects.Copy;
            }
            else
            {
                e.Effects = DragDropEffects.None;
            }

            e.Handled = true;
        }

        private void RootWindow_PreviewDrop(object sender, DragEventArgs e)
        {
            if (!IsDropSupported(e))
            {
                return;
            }

            string[]? paths = e.Data.GetData(DataFormats.FileDrop) as string[];
            if (paths == null || paths.Length == 0)
            {
                return;
            }

            string firstPath = paths[0];
            string? folderPath;

            if (Directory.Exists(firstPath))
            {
                folderPath = firstPath;
            }
            else
            {
                folderPath = Path.GetDirectoryName(firstPath);
            }

            if (string.IsNullOrWhiteSpace(folderPath))
            {
                return;
            }

            MainViewModel? viewModel = DataContext as MainViewModel;
            if (viewModel != null)
            {
                viewModel.LoadFolder(folderPath);
            }

            e.Handled = true;
        }

        private void PreviewScrollViewer_PreviewMouseWheel(object sender, MouseWheelEventArgs e)
        {
            ScrollViewer? scrollViewer = sender as ScrollViewer;
            MainViewModel? viewModel = DataContext as MainViewModel;

            if (scrollViewer == null || viewModel == null)
            {
                return;
            }

            double oldZoom = viewModel.ZoomFactor;
            Point mousePosition = e.GetPosition(PreviewContent);

            viewModel.ZoomByWheel(e.Delta);
            double newZoom = viewModel.ZoomFactor;

            // 缩放后保持鼠标所在位置尽量不动，预览体验会更自然。
            if (Math.Abs(newZoom - oldZoom) > double.Epsilon)
            {
                double zoomRatio = newZoom / oldZoom;
                double horizontalOffset = ((scrollViewer.HorizontalOffset + mousePosition.X) * zoomRatio) - mousePosition.X;
                double verticalOffset = ((scrollViewer.VerticalOffset + mousePosition.Y) * zoomRatio) - mousePosition.Y;

                scrollViewer.UpdateLayout();
                scrollViewer.ScrollToHorizontalOffset(horizontalOffset);
                scrollViewer.ScrollToVerticalOffset(verticalOffset);
            }

            e.Handled = true;
        }

        private void PreviewScrollViewer_PreviewMouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            ScrollViewer? scrollViewer = sender as ScrollViewer;
            if (scrollViewer == null)
            {
                return;
            }

            _isPreviewPanning = true;
            _lastPreviewPanPoint = e.GetPosition(scrollViewer);
            scrollViewer.CaptureMouse();
            scrollViewer.Cursor = Cursors.ScrollAll;
            e.Handled = true;
        }

        private void PreviewScrollViewer_PreviewMouseLeftButtonUp(object sender, MouseButtonEventArgs e)
        {
            ScrollViewer? scrollViewer = sender as ScrollViewer;
            StopPreviewPanning(scrollViewer);
            e.Handled = true;
        }

        private void PreviewScrollViewer_PreviewMouseMove(object sender, MouseEventArgs e)
        {
            ScrollViewer? scrollViewer = sender as ScrollViewer;
            if (!_isPreviewPanning || scrollViewer == null)
            {
                return;
            }

            Point currentPoint = e.GetPosition(scrollViewer);
            Vector delta = currentPoint - _lastPreviewPanPoint;

            scrollViewer.ScrollToHorizontalOffset(scrollViewer.HorizontalOffset - delta.X);
            scrollViewer.ScrollToVerticalOffset(scrollViewer.VerticalOffset - delta.Y);

            _lastPreviewPanPoint = currentPoint;
            e.Handled = true;
        }

        private void StopPreviewPanning(ScrollViewer? scrollViewer)
        {
            _isPreviewPanning = false;

            if (scrollViewer == null)
            {
                return;
            }

            scrollViewer.ReleaseMouseCapture();
            scrollViewer.Cursor = Cursors.SizeAll;
        }

        private static bool IsDropSupported(DragEventArgs e)
        {
            if (!e.Data.GetDataPresent(DataFormats.FileDrop))
            {
                return false;
            }

            string[]? paths = e.Data.GetData(DataFormats.FileDrop) as string[];
            if (paths == null || paths.Length == 0)
            {
                return false;
            }

            string firstPath = paths[0];
            if (Directory.Exists(firstPath))
            {
                return true;
            }

            if (!File.Exists(firstPath))
            {
                return false;
            }

            string extension = Path.GetExtension(firstPath);
            return SupportedPhotoExtensions.Contains(extension);
        }
    }
}
