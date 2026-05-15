from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable, TYPE_CHECKING
from functools import wraps

import ClearMap.IO.IO as cmp_io

if TYPE_CHECKING:
    from ClearMap.ImageProcessing.Experts.registry import SinkSpec, DataCtx
    from ClearMap.ImageProcessing.Experts.image_processing_func_library import ImageProcessingFunction


@runtime_checkable
class BlockFunction(Protocol):
    """
    data_ctx : DataCtx
        Block-level context (source, sink, slicing, etc.).
    sink_spec : SinkSpec or None
        Specification for writing results to sink/debug_sink.
        If None, nothing is written to disk.
    req : dict
        Required results from previous pipeline steps.
    algo_params : dict
        User-provided algorithm parameters.
    """
    def __call__(self, data_ctx: DataCtx, sink_spec: SinkSpec, req: Dict[str, Any], algo_params: Dict[str, Any]) -> Dict[str, Any]: ...


def block_function(fn: ImageProcessingFunction) -> BlockFunction:
    """
    Decorator that wraps an ImageProcessingFunction into a BlockFunction.

    Adds block-level awareness (DataCtx) around a pure image processing
    function, handling:

    - Requirement injection: if req is empty, feeds data_ctx.source
      as the first argument (first step of the pipeline). Otherwise, unpacks
      req.values() as positional arguments.

    - Saving to disk (optional): if algo_params contains a 'save'
      key, writes the specified result to a file. The save dict expects:
          - 'path': file path for the output sink.
          - 'to_save': key in results to save.
          - 'dtype' (optional): dtype for the output sink.
          - 'presave_parser' (optional): callable applied to the result
            before saving. Defaults to identity.

    - Writing to sink: if sink_spec is provided, writes the specified
      result to data_ctx.sink using sink_spec.operator.

    - Debug sink: if data_ctx.debug_sink exists, writes
      sink_spec.debug_label to the debug sink instead of the main sink.

    Parameters
    ----------
    fn : ImageProcessingFunction
        A pure image processing function to wrap.

    Returns
    -------
    BlockFunction
        Wrapped function with block-level context handling.
    """
    @wraps(fn)
    def wrapper(data_ctx: DataCtx, sink_spec: SinkSpec | None, req: Dict[str, Any], algo_params: Dict[str, Any]) -> Dict[str, Any]:
        save = algo_params.pop('save', None)  # TODO make save a dictionary containing path, to_save, dtype, presave_parser
        if not req:
            # if no requirements then it must be (one of) the first step(s) of the pipeline
            # thus feed the ImageProcessingFunction with the source
            results = fn(data_ctx.source, **algo_params) or {}
        else:
            results = fn(*req.values(), **algo_params) or {}

        if save:
            save_dtype = save.get('dtype', None)
            presave_parser = save.get('presave_parser', lambda t: t)
            to_save = save.get('to_save')
            to_save = presave_parser(results[to_save])

            if save_dtype is None:
                save_sink = cmp_io.as_source(save['path'])
            else:
                save_sink = cmp_io.as_source(save['path'], dtype=save_dtype)

            if save_dtype == 'bool':
                save_sink[data_ctx.base_slicing] = (to_save[data_ctx.valid_slicing] > 0)
            else:
                save_sink[data_ctx.base_slicing] = to_save[data_ctx.valid_slicing]

        if sink_spec is not None:
            result_to_sink = sink_spec.result

            if result_to_sink:
                result = results.get(result_to_sink) # extract main result
                if result is not None:
                    if data_ctx.debug_sink:
                        debug_sink = data_ctx.debug_sink # WARNING replaces cmp_io.as_source(data_ctx.parameter["binary_status"])
                        debug_sink[result[data_ctx.valid_slicing]] += sink_spec.debug_label
                    else:
                        operator = sink_spec.operator
                        data_ctx.sink[data_ctx.valid_slicing] = operator(result[data_ctx.valid_slicing], data_ctx.sink[data_ctx.valid_slicing])
        return results

    return wrapper