using System.IO;
using LAY.Models;

namespace LAY.Services
{
    public class PhotoFolderService : IPhotoFolderService
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

        public IReadOnlyList<PhotoItem> GetPhotos(string folderPath)
        {
            List<PhotoItem> photos = new List<PhotoItem>();

            if (string.IsNullOrWhiteSpace(folderPath) || !Directory.Exists(folderPath))
            {
                return photos;
            }

            // 先排序，再逐个包装成界面使用的 PhotoItem。
            string[] filePaths = Directory.GetFiles(folderPath, "*.*", SearchOption.TopDirectoryOnly);
            Array.Sort(filePaths, delegate (string left, string right)
            {
                return string.Compare(Path.GetFileName(left), Path.GetFileName(right), StringComparison.OrdinalIgnoreCase);
            });

            foreach (string filePath in filePaths)
            {
                string extension = Path.GetExtension(filePath);
                if (SupportedExtensions.Contains(extension))
                {
                    photos.Add(new PhotoItem(filePath));
                }
            }

            return photos;
        }

        public bool IsSupportedPhoto(string filePath)
        {
            if (string.IsNullOrWhiteSpace(filePath))
            {
                return false;
            }

            if (!File.Exists(filePath))
            {
                return false;
            }

            string extension = Path.GetExtension(filePath);
            return SupportedExtensions.Contains(extension);
        }
    }
}
