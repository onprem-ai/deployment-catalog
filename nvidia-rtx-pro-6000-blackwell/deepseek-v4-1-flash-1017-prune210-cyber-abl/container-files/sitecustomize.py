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


# onprem.ai addition: expert pruning at load time (expert_keep.py), only when
# DSV41_EXPERT_KEEP_SET names a keep set, and the abliteration overlay (abl41.py), only when
# ABL_OVERLAY names the overlay file. Both wrap the load_weights of the V4.1 target model and
# of its DSpark drafter in one loader (a second finder for the same modules would never run):
# the overlay replaces the edited attention tensors by name, then the keep set drops and
# renumbers the routed experts; the two touch disjoint tensors.
class ExpertKeepLoader(importlib.abc.Loader):
    CLASSES = {
        'sglang.srt.models.deepseek_v4': 'DeepseekV4ForCausalLM',
        'sglang.srt.models.deepseek_v4_dspark': 'DeepseekV4ForCausalLMDSpark',
    }

    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        cls = getattr(module, self.CLASSES[module.__name__])
        from expert_keep import wrap_load_weights
        wrap_load_weights(cls)
        if os.environ.get('ABL_OVERLAY'):
            import abl41
            pruned = cls.load_weights

            def load_weights(self, weights, *args, **kwargs):
                return pruned(self, abl41.overlay_weights(weights), *args, **kwargs)
            cls.load_weights = load_weights


class ExpertKeepFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in ExpertKeepLoader.CLASSES:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None:
            spec.loader = ExpertKeepLoader(spec.loader)
        return spec


if os.environ.get('DSV41_EXPERT_KEEP_SET') or os.environ.get('ABL_OVERLAY'):
    sys.meta_path.insert(0, ExpertKeepFinder())
