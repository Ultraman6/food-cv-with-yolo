import yaml
import pickle
import torch
from theseus.utilities.loggers.observer import LoggerObserver
LOGGER = LoggerObserver.getLogger("main")


def load_yaml(path):
    with open(path, 'rt') as f:
        return yaml.safe_load(f)


def load_state_dict(instance, state_dict, key=None, is_detection=False):
    """
    Load trained model checkpoint
    :param model: (nn.Module)
    :param path: (string) checkpoint path
    """

    if isinstance(instance, torch.nn.Module):
        try:
            if is_detection and key is not None:
                instance.load_state_dict(state_dict[key].state_dict())
            else:
                if key is not None:
                    instance.load_state_dict(state_dict[key])
                else:
                    instance.load_state_dict(state_dict)

            LOGGER.text("Loaded Successfully!", level=LoggerObserver.INFO)
        except RuntimeError as e:
            LOGGER.text(
                f'Loaded Successfully. Ignoring {e}', level=LoggerObserver.WARN)
        return instance
    else:
        if key in state_dict.keys():
            return state_dict[key]
        else:
            LOGGER.text(
                f"Cannot load key={key} from state_dict", LoggerObserver.WARN)


class RenameUnpickler(pickle.Unpickler):
    def __init__(self, *args, **kwargs):
        super(RenameUnpickler, self).__init__(*args, **kwargs)
        self.persistent_load = self._persistent_load
        self.deserialized_objects = {}

    def find_class(self, module, name):
        # 在这里替换旧的模块名为新的模块名
        if module == 'old_module':
            module = 'new_module'
        return super(RenameUnpickler, self).find_class(module, name)

    def _persistent_load(self, saved_id):
        # 这是 PyTorch 的内部实现，用于处理持久化 ID
        assert isinstance(saved_id, tuple)
        typename = saved_id[0]
        data = saved_id[1:]
        if typename == 'storage':
            storage_type, key, location, size = data
            if key not in self.deserialized_objects:
                # 创建一个空的存储器，并将其添加到已反序列化的对象中
                storage = torch.storage._get_third_party_storage()[storage_type](size)
                self.deserialized_objects[key] = storage
            else:
                storage = self.deserialized_objects[key]
            return storage
        else:
            raise pickle.UnpicklingError("Unknown persistent_load type: {}".format(typename))

def renamed_load(filename):
    with open(filename, 'rb') as f:
        unpickler = RenameUnpickler(f)
        result = unpickler.load()
    return result

