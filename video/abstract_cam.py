'''
Abstract base class (aka 'interface') for video interfaces
'''
import abc


class AbstractCam(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def start(self):
        raise NotImplementedError('You must define a start method')

    @abc.abstractmethod
    def stop(self):
        raise NotImplementedError('You must define a stop method')

    @abc.abstractmethod
    def read_frame(self):
        raise NotImplementedError('You must define a read_frame method')
