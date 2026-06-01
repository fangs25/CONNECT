import logging, sys
import logging.config
from pathlib import Path


def setup_logging(save_dir, default_level=logging.INFO):
    """Configure console and rotating-file logging for CONNECT runs.

    The function creates ``save_dir`` if needed, writes INFO-and-above records
    to ``info.log``, and mirrors logs to standard output.  If dictionary-based
    logging configuration fails, it falls back to ``logging.basicConfig`` and
    writes ``default_info.log``.

    Parameters
    ----------
    save_dir
        Directory used for log files.  It can be a string or
        :class:`pathlib.Path`.
    default_level
        Logging level used by the fallback configuration.
    """

    log_config = {
        "version": 1,
        "disable_existing_loggers": False,  # 不禁用已存在的日志器
        "formatters": {
            "simple": {"format": "%(message)s"},  # 简化格式（仅日志内容）
            "datetime": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"}  # 带时间格式
        },
        "handlers": {
            # 控制台处理器（输出到stdout）
            "console": {
                "class": "logging.StreamHandler",
                "level": "DEBUG",
                "formatter": "datetime",
                "stream": "ext://sys.stdout"
            },
            # 文件处理器（轮转日志，避免单个文件过大）
            "info_file_handler": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",
                "formatter": "datetime",
                "filename": "info.log",  # 文件名（后续会拼接save_dir路径）
                "maxBytes": 10485760,  # 单个日志文件最大10MB（10*1024*1024）
                "backupCount": 20,  # 最多保留20个备份日志
                "encoding": "utf8"  # 避免中文乱码
            }
        },
        "root": {  # 根日志器配置
            "level": "INFO",
            "handlers": ["console", "info_file_handler"]  # 同时输出到控制台和文件
        }
    }

    # 确保save_dir目录存在（不存在则创建）
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)  # parents=True：创建多级目录；exist_ok=True：目录存在不报错

    # 遍历所有处理器，修改文件路径
    for _, handler in log_config["handlers"].items():
        if "filename" in handler:  # 仅处理带文件输出的处理器（如info_file_handler）
            handler["filename"] = str(save_dir / handler["filename"])  # 拼接目录与文件名

    try:
        logging.config.dictConfig(log_config)
        # 验证配置生效（可选，输出初始化日志）

    except Exception as e:
        # 配置加载失败时，使用基础日志配置
        print(f"Warning: 日志配置加载失败 - {str(e)}，启用默认配置")
        logging.basicConfig(
            level=default_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(stream=sys.stdout),  # 控制台输出
                logging.FileHandler(save_dir / "default_info.log", encoding="utf8")  # 默认文件输出
            ]
        )

if __name__ == "__main__":
    setup_logging(save_dir=Path("./logs"))
    
    logger = logging.getLogger("test_logger")
    logger.debug("调试信息（仅控制台输出，文件不记录）")
    logger.info("普通信息（控制台+文件均记录）")
    logger.warning("警告信息")
    logger.error("错误信息")
    logger.critical("严重错误信息")
