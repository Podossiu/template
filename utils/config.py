### config utilities for yml file
import os
import sys
import yaml

# singletone
FLAGS = None

class AttrDict(dict):
    """ Dict as attribute trick.
    """
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        # 같은 namespace 설정
        self.__dict__ = self
        # key값에 대해서
        for key in self.__dict__:
            # value값 설정 뒤
            value = self.__dict__[key]
            # 만약에 value도 dictionary면
            if isinstance(value, dict):
                # AttrDict로 설정해
                self.__dict__[key] = AttrDict(value)
            # 만약에 value가 list면
            elif isinstance(value, list):
                # value 0가 dictionary면 모든 값에 대해 AttrDict로 설정해
                if isinstance(value[0], dict):
                    self.__dict__[key] = [AttrDict(item) for item in value]
                # 걍 값이면 그냥 넣어
                else:
                    self.__dict__[key] = value
    # object를 yaml dict로 바꿔
    def yaml(self):
        """ Convert object to yaml dict and return.
        """
        yaml_dict = {}
        # 모든 key에 대해서
        for key in self.__dict__:
            # value값 받은 뒤
            value = self.__dict__[key]
            # 아까 설정한 AttrDict라면 
            if isinstance(value, AttrDict):
                # 재귀적으로 yaml 반복
                yaml_dict[key] = value.yaml()
            # list라면 
            elif isinstance(value, list):
                # 각원소에 대해서 AttrDict면
                if isinstance(value[0], AttrDict):
                    # 다시 list만들어서
                    new_l = []
                    for item in value:
                        # item들을 바꿔서 넣어줘
                        new_l.append(item.yaml())
                    # key의 value 설정해줘
                    yaml_dict[key] = new_l
                else:
                    # list에 들어있는 값이 그냥 data면 key에 value 대입
                    yaml_dict[key] = value
            # 그냥 value라면 그냥 key에 대입
            else:
                yaml_dict[key] = value
        return yaml_dict

    def __repr__(self):
        """ Print all variables.
        """
        # representation 정의
        ret_str = []
        # dict안에 들어있는 key에 대해서
        for key in self.__dict__:
            # value값 가져옴
            value = self.__dict__[key]
            # 만약 AttrDict면
            if isinstance(value, AttrDict):
                ret_str.append('{}:'.format(key))
                child_ret_str = value.__repr__().split('\n')
                for item in child_ret_str:
                    ret_str.append('    '+item)
            elif isinstance(value, list):
                if isinstance(value[0], AttrDict):
                    ret_str.append('{}:'.format(key))
                    for item in child_ret_str:
                        ret_str.append('    '+item)
                else:
                    ret_str.append('{}: {}'.format(key, value))
            else:
                ret_str.append('{}: {}'.format(key, value))
        return '\n'.join(ret_str)

class Config(AttrDict):
    """ Config with yaml file.
        This class is used to config model hyper-parameter, global constants, and
        other settings with yaml file. All settings in yaml file will be 
        automatically logged into file.

        Args:
            filename(str) : File name

        Examples:
            
            yaml file ''model.yml''::
                NAME: 'neuralgym'
                ALPHA: 1.0
                DATASET: '/mnt/data/imagenet'
            Usage in .py:
            >>> from neuralgym import Config
            >>> config = Config('model.yml')
            >>> print(config.Name)
                neuralgym
            >>> print(config.ALPHA)
                1.0
            >>> print(config.DATASET)
                /mnt/data/imagenet
    """
    def __init__(self, filename=None, verbose = False):
        assert os.path.exists(filename), 'File{} not exist.'.format(filename)
        try:
            with open(filename, 'r') as f:
                try:
                    cfg_dict = yaml.safe_load(f)
                except yaml.YAMLError as exc:
                    print(exc)
        except EnvironmentError:
            print('Please check the file with name of "%s"', filename)
        super(Config, self).__init__(cfg_dict)
        if verbose:    
            print(' pi.cfg '.center(80, '-'))
            print(self.__repr__())
            print(''.center(80, '-'))

def app():
    """ Load app via stdin from subprocess"""
    global FLAGS
    if FLAGS is None:
        job_yaml_file = None
        batch_size = None
        kappa = None
        for arg in sys.argv:
            if arg.startswith('app:'):
                job_yaml_file = arg[4:]
        if job_yaml_file is None:
            job_yaml_file = sys.stdin.readline()
        
        FLAGS = Config(job_yaml_file)
        return FLAGS
app()
print(FLAGS)
print(FLAGS.lr)
