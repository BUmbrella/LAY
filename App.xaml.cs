using System.Threading;
using System.Windows;
using LAY.Services;
using LAY.Views;
using Prism.Ioc;
using Prism.Unity;

namespace LAY
{
    public partial class App : PrismApplication
    {
        private const string SingleInstanceMutexName = "LAY_GrindingMeasurement_SingleInstance";
        private Mutex? _singleInstanceMutex;

        protected override void OnStartup(StartupEventArgs e)
        {
            _singleInstanceMutex = new Mutex(true, SingleInstanceMutexName, out bool createdNew);
            if (!createdNew)
            {
                MessageBox.Show("软件已经打开", "提示", MessageBoxButton.OK, MessageBoxImage.Information);
                Shutdown();
                return;
            }

            base.OnStartup(e);
        }

        protected override void OnExit(ExitEventArgs e)
        {
            _singleInstanceMutex?.ReleaseMutex();
            _singleInstanceMutex?.Dispose();
            _singleInstanceMutex = null;

            base.OnExit(e);
        }

        // 创建程序启动后显示的主窗口。
        protected override Window CreateShell()
        {
            return Container.Resolve<MainWindow>();
        }

        // 注册全局服务，ViewModel 可以通过构造函数拿到这些服务。
        protected override void RegisterTypes(IContainerRegistry containerRegistry)
        {
            containerRegistry.RegisterSingleton<IPhotoFolderService, PhotoFolderService>();
        }
    }
}
