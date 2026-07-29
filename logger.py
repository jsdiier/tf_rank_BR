# file: scripts/logger.py
import os
import datetime
import sys
import os.path


class LogLevel:
    DEBUG = 1
    INFO = 2
    WARNING = 3
    ERROR = 4


class Logger:
    _instance = None

    def __new__(cls, log_dir="./log"):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, log_dir="./log"):
        if self._initialized:
            return

        self.log_dir = log_dir

        # 检查标准输出是否被重定向
        # 如果标准输出不是终端设备，说明已被重定向到文件
        self.is_stdout_redirected = not sys.stdout.isatty()

        # 只有在标准输出未被重定向时才创建日志文件
        if not self.is_stdout_redirected:
            # 创建日志目录
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            # 获取调用的脚本名称（优先从环境变量获取，否则使用当前脚本名）
            script_name = os.environ.get('CALLER_SCRIPT_NAME')
            if not script_name:
                script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]

            # 根据当前日期生成日志文件名
            current_datetime = datetime.datetime.now().strftime("%Y%m%d%H%M")
            # Python 2 不支持 f-string，使用 % 格式化
            self.main_log_file = os.path.join(log_dir, "%s_%s.log" % (script_name, current_datetime))
            self.debug_log_file = os.path.join(log_dir, "%s_debug_%s.log" % (script_name, current_datetime))
        else:
            self.main_log_file = None
            self.debug_log_file = None

        self._initialized = True

    def _write_log(self, level, message):
        """写入日志到文件"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Python 2 不支持 f-string，使用 % 格式化
        if level == LogLevel.DEBUG:
            level_name = "DEBUG"
        elif level == LogLevel.INFO:
            level_name = "INFO"
        elif level == LogLevel.WARNING:
            level_name = "WARNING"
        elif level == LogLevel.ERROR:
            level_name = "ERROR"
        else:
            level_name = "UNKNOWN"

        log_message = "[{}] [{}] {}\n".format(timestamp, level_name, message)

        # 只有在未重定向标准输出时才写入文件
        if not self.is_stdout_redirected:
            # DEBUG信息单独写入debug文件
            if level == LogLevel.DEBUG:
                with open(self.debug_log_file, "a") as f:
                    f.write(log_message)
            else:
                # INFO, WARNING, ERROR信息写入主日志文件
                with open(self.main_log_file, "a") as f:
                    f.write(log_message)

        # 始终打印到标准输出（会被shell脚本重定向）
        if level >= LogLevel.INFO:
            print("[{}] [{}] {}".format(timestamp, level_name, message))

    def debug(self, message):
        self._write_log(LogLevel.DEBUG, message)

    def info(self, message):
        self._write_log(LogLevel.INFO, message)

    def warning(self, message):
        self._write_log(LogLevel.WARNING, message)

    def error(self, message):
        self._write_log(LogLevel.ERROR, message)


# 全局日志实例
logger = Logger()
