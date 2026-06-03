using LAY.Models;

namespace LAY.Services
{
    public interface IPhotoFolderService
    {
        // 读取文件夹中的所有支持图片。
        IReadOnlyList<PhotoItem> GetPhotos(string folderPath);

        // 判断单个文件是否是支持的图片格式。
        bool IsSupportedPhoto(string filePath);
    }
}
