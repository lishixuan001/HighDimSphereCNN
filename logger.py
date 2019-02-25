import logging
import datetime
import sys


def setup_logger(name):
    now = datetime.datetime.now().strftime('%m%d%H%M%S')
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    handler = logging.FileHandler('log/{}.log'.format(now), mode='w')
    handler.setFormatter(formatter)

    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)

<<<<<<< HEAD
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(screen_handler)

    return logger
=======
    def scalar_summary(self, tag, value, step):
        """Add scalar summary."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def close(self):
        # self.writer.export_scalars_to_json("./all_scalars.json")
        self.writer.close()
>>>>>>> 53123aaf1be83633dff0ba85523c13718e2e8f88
