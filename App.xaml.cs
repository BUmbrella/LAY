using System.Windows;
using LAY.Services;
using LAY.Views;
using Prism.Ioc;
using Prism.Unity;

namespace LAY
{
    public partial class App : PrismApplication
    {
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
