"""Install storage adapter in every serving worker, only when explicitly enabled."""
import importlib.abc
import importlib.machinery
import os
import sys

class EngramLoader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        if module.__name__ == 'sglang.srt.layers.engram':
            from engram_backend import install
            install(module)
        else:
            # V4.1 ratio-1/2 indexers always call the FP4 DeepGEMM kernel.
            # SM120 needs its split-128 planner even when the legacy FP8
            # indexer uses the torch path. The upstream guard misses this case.
            cls = module.PagedIndexerMetadata
            original = cls.__post_init__
            def post_init(self):
                if module._IS_SM120 and self.compress_ratio in (1, 2):
                    self.force_deep_gemm_metadata = True
                original(self)
            cls.__post_init__ = post_init

class EngramFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in ('sglang.srt.layers.engram',
                            'sglang.srt.layers.attention.dsv4.metadata'):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None:
            spec.loader = EngramLoader(spec.loader)
        return spec

if os.environ.get('DSV41_SOURCE'):
    sys.meta_path.insert(0, EngramFinder())


# onprem.ai addition: refusal-direction work (abl41.py). ABL_RUNTIME=1 hooks the attention
# output projections and adds the /abl routes; ABL_BAKE_DIRECTIONS edits them at load time.
class AblLoader(importlib.abc.Loader):
    MODELS = {
        'sglang.srt.models.deepseek_v4': ('DeepseekV4ForCausalLM', False),
        'sglang.srt.models.deepseek_v4_dspark': ('DeepseekV4ForCausalLMDSpark', True),
    }

    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        import abl41
        if module.__name__ == 'sglang.srt.entrypoints.http_server':
            if os.environ.get('ABL_RUNTIME') == '1':
                abl41.add_routes(module)
            return
        cls_name, drafter = self.MODELS[module.__name__]
        cls = getattr(module, cls_name)
        original = cls.__dict__['load_weights']

        def load_weights(self, weights, *args, **kwargs):
            if os.environ.get('ABL_BAKE_DIRECTIONS'):
                weights = abl41.bake_weights(weights)
            if os.environ.get('ABL_OVERLAY'):
                weights = abl41.overlay_weights(weights)
            out = original(self, weights, *args, **kwargs)
            if os.environ.get('ABL_RUNTIME') == '1':
                abl41.RUNTIME.attach(self, drafter)
            return out
        cls.load_weights = load_weights


class AblFinder(importlib.abc.MetaPathFinder):
    NAMES = ('sglang.srt.models.deepseek_v4', 'sglang.srt.models.deepseek_v4_dspark',
             'sglang.srt.entrypoints.http_server')

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self.NAMES:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None:
            spec.loader = AblLoader(spec.loader)
        return spec


if os.environ.get('ABL_RUNTIME') == '1' or os.environ.get('ABL_BAKE_DIRECTIONS') or os.environ.get('ABL_OVERLAY'):
    sys.meta_path.insert(0, AblFinder())
