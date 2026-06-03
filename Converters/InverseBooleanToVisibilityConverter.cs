using System.Globalization;
using System.Windows;
using System.Windows.Data;

namespace LAY.Converters
{
    public sealed class InverseBooleanToVisibilityConverter : IValueConverter
    {
        // true 时隐藏，false 时显示。
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            bool isTrue = false;

            if (value is bool)
            {
                isTrue = (bool)value;
            }

            if (isTrue)
            {
                return Visibility.Collapsed;
            }

            return Visibility.Visible;
        }

        // 界面一般不会反向写这个值，这里保留基础转换。
        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is Visibility)
            {
                Visibility visibility = (Visibility)value;
                return visibility != Visibility.Visible;
            }

            return false;
        }
    }
}
