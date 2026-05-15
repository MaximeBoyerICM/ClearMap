# -*- coding: utf-8 -*-
"""
Modular pipelines for ClearMap Experts
=================================================================================

This file provides drop‑in, picklable, modular replacements for the
monolithic pipeline entrypoint, aligned with your newer file:

- `Vasculature.binarize` → `tube_map_pipeline.binarize_modular`
"""

import traceback
from typing import Any, Dict, List, Optional

import ClearMap.IO.IO as cmp_io
import ClearMap.ParallelProcessing.BlockProcessing as block_processing
import ClearMap.ParallelProcessing.DataProcessing.ArrayProcessing as array_processing

from ClearMap.ImageProcessing.Experts.utils import (initialize_sinks as _initialize_sinks)
from ClearMap.ImageProcessing.Experts.registry import Registry, DataCtx

DEFAULT_TUBE_MAP_PIPELINE_STEPS = ["clip", "lightsheet", "median", "deconvolve", "threshold_bin", "adaptive",
                                   "equalize", "bin_equalized", "vesselize"]


def binarize_modular(source, sink=None,
                     binarization_parameter: Optional[Dict[str, Any]] = None,
                     processing_parameter: Optional[Dict[str, Any]] = None):
    """
    source (Source): source specification
        The source of the stitched raw data.
    sink (Source): sink specification or None
        The sink to write the result to.
    binarization_parameter : dict
        Parameter for the binarization. See below for details.
    processing_parameter : dict
        Parameter for the parallel processing.
        See :func:`ClearMap.ParallelProcessing.BlockProcessing.process` for
        description of all the parameter.

    General parameter
    -----------------
    binary_status : str or None
        File name to save the information about which part of the multi-path
        binarization contributed to the final result.

    max_bin : int
        Number of intensity levels to use for the data after preprocessing.
        Higher values will increase the intensity resolution but slow down
        processing.

        For the vasculature a typical value is 2**12.

    pipeline : list
        Step keys to execute.

    custom_steps (optional) : dict
        You can insert brand-new steps or override
        built-ins by providing
        binarization_parameter["custom_steps"] = {name:
                                                  requires:[],
                                                  produces:[],
                                                  fn:,
                                                  sink_spec:}
        where fn is a top-level pickable BlockFunction (cf. StepFunction class for more details)
    """
    binarization_parameter = binarization_parameter or {}
    processing_parameter = processing_parameter or {}

    # initialize sinks as in original
    shape, order = cmp_io.shape(source), cmp_io.order(source)
    sink, _ = array_processing.initialize_sink(sink=sink, shape=shape, order=order, dtype=bool)
    if binarization_parameter.get('binary_status'):
        array_processing.initialize_sink(sink=binarization_parameter['binary_status'],
                                         shape=shape, order=order, dtype='uint16')
    _initialize_sinks(binarization_parameter, shape, order)

    default_steps= binarization_parameter.get('default_steps') or {}
    custom_steps = binarization_parameter.get('custom_steps') or {}

    registry = Registry(default_steps)
    registry.submit_steps(custom_steps)

    block_processing.process(_block_fn, source, sink, function_type='block',
                             parameter={**binarization_parameter,
                                        'registry': registry,
                                        'verbose': processing_parameter.get('verbose', False)},
                                        **processing_parameter)
    return sink

def _extract_algo_params(pipeline, parameter):
    """Extracts algo params from parameter dictionary for easier parsing later"""
    pipeline_algo_params = {'global_params': {}, 'steps_params': {}}
    for step_name in pipeline:
        pipeline_algo_params['steps_params'][step_name] = parameter.get(step_name)
    pipeline_algo_params['global_params']['max_bin'] = parameter.get('max_bin')
    return pipeline_algo_params


def _block_fn(_source, _sink, parameter):
    data_ctx = DataCtx(_source, _sink, parameter)
    pipeline: List[str] = parameter.get('pipeline')
    pipeline_algo_params = _extract_algo_params(pipeline, parameter)
    tmp: Dict[str, Any] = {}
    registry = parameter.get('registry')
    completed_steps = set()

    def _validate_running_step(step_name):
        step_fn = registry.step_functions[step_name]

        if step_name in completed_steps:
            print(f"Step {step_name} already completed. Skipping.")
            return False

        reqs = registry.get_requirements(set(pipeline), as_set=True)
        if not step_fn.has_sink and set(step_fn.produces).isdisjoint(reqs):
            print(f"Step {step_name} does not write to disk and is not required by other steps. Skipping.")
            return False

        return True


    def _collect_garbage():
        """Delete results from temporary storage (tmp) that are not useful to produce future steps to free memory."""
        remaining_steps = set(pipeline) - completed_steps
        reqs = registry.get_requirements(remaining_steps, as_set=True)
        prod_to_del = set(tmp.keys()) - reqs
        for prod in prod_to_del:
            del tmp[prod]

    def _ensure(key: str):
        """Ensure a result has been produced. If it has not been produced, run the step producing it"""
        if key in tmp:
            return
        producer = registry.produced_by.get(key)
        if not producer:
            raise KeyError(f'No step produces required {key=}')
        _run(producer)
        if key not in tmp:
            raise KeyError(f'Producer {producer} did not provide {key=}')

    def _ensure_any_of(keys: tuple[str, ...]):
        """Ensure all the given results are present."""
        for k in keys:
            try:
                _ensure(k)
                return
            except KeyError:
                pass
        raise KeyError(f'Could not satisfy any of {keys} for current step')

    def _run(step_name: str):
        """Run a step. First check if required results are present."""
        try:
            step_fn = registry.step_functions[step_name]

            # ensure requirements first
            for req in step_fn.requires:
                if isinstance(req, tuple):
                    if not any(r in tmp for r in req):
                        _ensure_any_of(req)
                else:
                    _ensure(req)
            step_algo_params =  pipeline_algo_params['steps_params'][step_name] | pipeline_algo_params['global_params']
            out = step_fn(data_ctx, tmp, step_algo_params) or {}
            completed_steps.add(step_name)
            if out:
                tmp.update(out)
            return out
        except Exception as e:  # TODO fix traceback not coming up without that
            traceback.print_exc()
            raise RuntimeError(f"Step '{step_name}' failed") from e

    # execute declared pipeline
    for name in pipeline:
        try:
            if name not in registry.step_functions:
                raise KeyError(f'Unknown vasculature step "{name}".')
            step_has_to_run = _validate_running_step(name)
            if step_has_to_run:
                _run(name)
                _collect_garbage()
        except Exception as e:  # TODO fix traceback not coming up without that
            traceback.print_exc()
            raise RuntimeError(f"Step '{name}' failed") from e

    return None  # block writes into sink

