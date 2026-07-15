using System.IO;

namespace LAY.Models
{
    public class PhotoItem
    {
        public PhotoItem(string fullPath, string? displayName = null)
        {
            FullPath = fullPath;
            FileName = string.IsNullOrWhiteSpace(displayName)
                ? Path.GetFileName(fullPath)
                : displayName;
        }

        // 图片完整路径，用于读取和预览。
        public string FullPath { get; private set; }

        // 界面列表中显示的文件名。
        public string FileName { get; private set; }

        public bool HasProblem { get; set; }
    }
}
