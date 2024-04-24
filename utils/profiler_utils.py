import argparse
import cProfile
import logging
import pstats
import io
import os
import functools


class ProfilerContext:
    is_profiling_active = False  # Default to off

    def __init__(self, name=None, save_after=1, function=None):
        self.name = name
        self.profiler = cProfile.Profile()
        self.save_after = save_after if (save_after is not 0 and save_after is not None) else 1
        self.counter = 0
        self.function = function

        assert self.name is not None or self.function is not None, "Either name or function must be provided"

    def __enter__(self):
        if self.is_profiling_active:
            self.profiler.enable()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        if self.is_profiling_active:
            self.profiler.disable()

            # Use the function_save_after_executions from the wrapper function's closure
            if self.counter % self.save_after == 0:
                self.save()

            self.counter += 1

    @classmethod
    def profile_function(cls, save_after=None, **kwargs):
        def decorator(func):

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if cls.is_profiling_active:
                    with cls(save_after=save_after, function=func):
                        return func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)

            return wrapper

        return decorator

    def save(self):

        if self.name is not None:
            filename = self.name + "_profile.prof"
        else:
            function_key = (self.function.__module__, self.function.__name__)
            filename = f"{function_key[0]}_{function_key[1]}_profile.prof"

        self.profiler.dump_stats(filename)
        logging.info(f"Saved profile to {filename}")
