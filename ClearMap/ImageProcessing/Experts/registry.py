"""
This module contains the Registry class that is used to define a series of StepFunctions
"""
from __future__ import annotations

from dataclasses import field, dataclass
from typing import List, Union, Optional, Tuple, Iterable

from ClearMap.ImageProcessing.Experts.image_processing_func_library import *
from ClearMap.ImageProcessing.Experts.utils import print_params
from ClearMap.ImageProcessing.Experts.block_function import BlockFunction

StepResource = Optional[List[Union[str, Tuple[str, str]]]]


class DataCtx:
    """
    Context for block-based image processing operations.

    Attributes
    ----------
    source : Any
        Source block containing input data.
    sink : Any or None
        Sink for output data.
    logger_verbose : bool
        Enable verbose output.
    logger_prefix : str
        Prefix for log messages. Default is "".
    base_slicing : tuple of slice
        See ClearMap.IO.Slice
    valid_slicing : tuple of slice
        See ClearMap.IO.Slice
    debug_sink : Any or None, optional
        Optional sink for debugging output.
    """

    def __init__(self, source: Any, sink: Optional[Any], parameter: Dict[str, Any]):
        self.source = source
        self.sink = sink

        # TODO move to Logger object
        self.logger_verbose = parameter.get("verbose", False)
        self.logger_prefix = f"Block {source.info()}: " if self.logger_verbose else ""

        self.base_slicing = sink.valid.base_slicing if sink is not None else source.valid.base_slicing
        self.valid_slicing = source.valid.slicing

        if parameter.get('binary_status') is not None:
            self.debug_sink = cmp_io.as_source(parameter['binary_status'])
        else:
            self.debug_sink = None


@dataclass
class SinkSpec:
    """
    result : name of the result to write to sink. Extracted from the dictionary returned by an ImageProcessingFunction
    operator : writing method. Callable that will take as args 'result' array and the sink.
        e.g. : np.add => will execute np.add(result, sink)
        np.add is the default choice for binary arrays combination
    debug_label : Binary power label (2^n) used to identify the pipeline step on the debug_sink.
                  Each step is represented  by a unique power of 2 (1, 2, 4, 8, 16, ...), determined by its order
                  in the pipeline.
    """
    result: str = 'result'
    operator: Union[Callable[..., Any], np.ufunc] = field(default=np.logical_or)
    debug_label: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Optional[Dict]) -> Optional['SinkSpec']:
        """Construct SinkSpec from a dictionary, or return None if data is None."""
        if data is None:
            return None

        operator = data.get('operator', cls.operator)  # Defaults to np.logical_or
        if isinstance(operator, str):
            if operator.startswith('np.'):
                operator = getattr(np, operator[3:])  # Extract function from numpy
            else:
                raise ValueError(f"Unsupported operator: {operator}")

        debug_label = data.get('debug_label')  # None if not provided

        return cls(
            result=data.get('result', cls.result),  # Defaults to 'result'
            operator=operator,
            debug_label=debug_label
        )


class StepFunction:
    """
    Wrap a BlockFunction, making it part of a pipeline.
    Feeds it with required results from previous steps and writes to sink if specified.
    Filters its output to only keep what 'produces' declares.

    Parameters
    ----------
    fn : BlockFunction
        The block-level function to execute.
    requires : StepResource, optional
        Names of previous steps' results needed by fn.
    produces : StepResource, optional
        Names of the results that fn is expected to produce.
    sink_spec : SinkSpec, optional
        Specification for writing results to sink/debug_sink.
    step_name : str, optional
        Name of the step (used for logging).
    dtypes : dict, optional
        Mapping of result names to their expected dtypes.
    """
    def __init__(self, fn: BlockFunction, requires: StepResource | None = None, produces: StepResource | None = None,
                 sink_spec: SinkSpec | None = None, step_name: str = None, dtypes: Dict[str, Any] | None = None):
        self.fn = fn
        self.requires = requires or []
        self.produces = produces or []
        self.sink_spec = sink_spec
        self.step_name = step_name  # necessary for StepFunction to report progress. in the future, move to Logger object
        self.dtypes = dtypes or {}

    @property
    def has_sink(self):
        return self.sink_spec is not None

    def __call__(self, data_ctx: DataCtx, tmp: Dict[str, Any], algo_params: Dict[str, Any]) -> Dict[str, Any]:
        return step_function(self.fn, self.requires, self.produces, self.sink_spec, self.step_name, data_ctx, tmp,
                             algo_params, self.dtypes)


def cast_to_dtypes(results, dtypes):
    """
    Cast each value in results to the dtype specified in dtypes.
    Skips copying if the value already has the correct dtype.
    Keys in results not found in dtypes are kept as-is.

    Parameters
    ----------
    results : dict
        Mapping of result names to arrays/values.
    dtypes : dict
        Mapping of result names to target dtypes (numpy dtype or Python type).

    Returns
    -------
    dict
        Results with values cast to the specified dtypes.
    """
    if dtypes:  # TODO maybe it should be mandatory to fill dtypes dict
        out = {}
        for key, val in results.items():
            if key not in dtypes:
                out[key] = val
                print(f"Warning: no dtype was specified for {key}. No datatype cast will be performed.")  # TODO move to Logger object
                continue
            dt = dtypes[key]
            if dt is str or isinstance(dt, type) and not issubclass(dt, np.generic):
                out[key] = dt(val) if not isinstance(val, dt) else val
            else:
                out[key] = np.asarray(val, dtype=dt)
        return out
    else:
        return results


def step_function(fn: BlockFunction, requires: StepResource, produces: StepResource, sink_spec: SinkSpec | None, step_name: str,
                  data_ctx: DataCtx, tmp: Dict[str, Any], algo_params: Dict[str, Any], dtypes: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a single pipeline step.

    Extracts required results from tmp (the pipeline's intermediate storage),
    calls fn, filters the output to only keep declared products, and casts
    results to specified dtypes.

    Requirements can contain tuples: the first element found in tmp is used,
    others are ignored.

    Parameters
    ----------
    fn : BlockFunction
        The block-level function to execute.
    requires : StepResource
        Names of required results from previous steps.
    produces : StepResource
        Names of the results to keep from fn's output.
    sink_spec : SinkSpec or None
        Specification for writing results to sink.
    step_name : str
        Name of the step (used for logging).
    data_ctx : DataCtx
        Block-level context (source, sink, slicing, etc.).
    tmp : dict
        Intermediate results from previous steps.
    algo_params : dict
        Algorithm parameters passed to fn.
    dtypes : dict
        Mapping of result names to target dtypes.

    Returns
    -------
    dict
        Filtered and dtype-cast results.
    """
    req = {}
    for r in requires:
        if isinstance(r, tuple):
            for i in range(0, len(r)):
                if tmp.get(r[i]) is not None:
                    req[r[i]] = tmp[r[i]]
                    break
        else:
            req[r] = tmp[r]
    step_param, timer = print_params(algo_params, step_name, prefix='', verbose=False)  # TODO move to Logger object
    results = fn(data_ctx, sink_spec, req, algo_params) or {}
    timer.print_elapsed_time(step_name.title())  # TODO move to Logger object
    results = {k: v for k, v in results.items() if k in produces}
    return cast_to_dtypes(results, dtypes)


"""
The default step_functions dictionary, corresponding to the version of TubeMap as described in [1].

References
----------
[1] C. Kirst et al., "Mapping the Fine-Scale Organization and Plasticity of the Brain Vasculature", Cell 180, 780 (2020)
"""
DEFAULT_STEP_FUNCTIONS: Dict[str, StepFunction] = {
        'clip': StepFunction(fn=clip,
                             produces=['clipped', 'mask', 'high_mask', 'not_low_mask'],
                             sink_spec=SinkSpec("high_mask", np.logical_or),
                             dtypes={'clipped': np.uint16, 'mask': np.bool_, 'high_mask': np.bool_, 'not_low_mask': np.bool_}),
        'lightsheet': StepFunction(fn=lightsheet_correction,
                                   requires=['clipped', 'mask'],
                                   produces=['lc_corrected'],
                                   dtypes={'lc_corrected': np.uint16}),
        'median': StepFunction(fn=median,
                               requires=['lc_corrected', 'not_low_mask'],
                               produces=['median'],
                               dtypes={'median': np.uint16}),
        'deconvolve': StepFunction(fn=deconvolve,
                                   requires=['median', 'high_mask'],
                                   produces=['deconvolved'],
                                   dtypes={'deconvolved': np.float64}),
        'threshold_bin': StepFunction(fn=threshold_bin,
                                      requires=['deconvolved'],
                                      sink_spec=SinkSpec('binary', np.logical_or)),
        'adaptive': StepFunction(fn=threshold_adaptive,
                                 requires=[('deconvolved', 'median')],
                                 sink_spec=SinkSpec('bin_adaptive', np.logical_or)),
        'equalize': StepFunction(fn=equalize,
                                 requires=['median', 'mask'],
                                 produces=['equalized'],
                                 dtypes={'equalized': np.float64}),
        'bin_equalized': StepFunction(fn=bin_equalized,
                                      requires=['equalized'],
                                      produces=['bg_scaled_equalized', 'bin_equalized'],
                                      sink_spec=SinkSpec('bin_equalized', np.logical_or)),
        'vesselize': StepFunction(fn=vesselize,
                                  requires=['bg_scaled_equalized', 'mask'],
                                  sink_spec=SinkSpec('bin_vesselized', np.logical_or)),
    }


class Registry:
    """
    This class is used as an interface between TubeMap pipeline and step functions.
    """
    def __init__(self, steps: Dict[str, Any] = None):
        if steps is None:
            steps = {}
        self.step_functions = {}
        self.produced_by: dict[str, str] = {}
        self.debug_binary_steps_labels: dict[str, str] = {}
        self.products = []
        self.submit_steps(steps)

    def submit_steps(self, steps: Dict[str, Any]):
        """
        Add new steps or override existing ones.

        Parameters
        ----------
        steps : dict
            Each entry must be either:

            - A ``StepFunction`` instance, or
            - A dict with at least the following keys:

              - ``'fn'``: a BlockFunction, or a string referring to a
                BlockFunction in image_processing_func_library.
              - ``'requires'``: list of requirement names (previous steps'
                products). Empty list if the source is the only input.
              - ``'produces'``: list of product names. Must be a subset of
                the keys returned by the wrapped ImageProcessingFunction.
              - ``'sink_spec'`` (optional): a SinkSpec instance describing
                which result to write to sink and how.
              - ``'dtypes'`` (optional): dict mapping product names to
                target dtypes.

        Notes
        -----
        - If ``produces`` is empty and ``sink_spec`` is None, the step
          will most likely have no effect.
        - During ``_setup``, each SinkSpec receives a ``debug_label``
          (a unique power of 2) for debug sink identification.
        """
        if not steps: return

        def _as_list(x):
            return list(x) if isinstance(x, (list, tuple)) else [x]

        for name, obj in steps.items():
            if isinstance(obj, dict) and all(k in obj for k in ('fn', 'requires', 'produces')):
                reqs = _as_list(obj['requires'])
                prods = _as_list(obj['produces'])

                fn = obj['fn']
                fn = self._search_function_in_library(fn)

                sink_spec = SinkSpec.from_dict(obj.get('sink_spec'))

                self.step_functions[name] = StepFunction(fn=fn, requires=reqs, produces=prods,
                                                         sink_spec=sink_spec,
                                                         dtypes=obj.get('dtypes', {}))

            elif isinstance(obj, StepFunction):
                self.step_functions[name] = obj

            else:
                raise TypeError(
                    f'steps[{name!r}] must be a dict with keys '
                    f'{{"fn","requires","produces"}} or a StepFunction, not {type(obj)!r}.')
        self._setup()

    def _search_function_in_library(self, fn: str | BlockFunction):
        if isinstance(fn, str):
            fn = globals()[fn]  # image_processing_func_library is imported so the function should be in globals
        if not isinstance(fn, BlockFunction):
            raise TypeError(f"{fn} is not an BlockFunction. Make sure to wrap an ImageProcessingFunction"
                            f" with the block_function decorator")
        return fn

    def _check_duplicates_products(self):
        prod, counts = np.unique(self.products, return_counts=True)
        duplicates = prod[counts > 1]
        if duplicates.size > 0:
            dup = duplicates.tolist()
            raise AssertionError(f'The following results are produced by two or more steps: {dup}.'
                                 f'Please make sure that one product is produced by one and only one step.')

    def get_requirements(self, steps: Iterable[str], as_set=False):
        """Get requirements for specified steps.

        Parameters
        ----------
            steps: Step names to get requirements for
            as_set: If True, return a flat set of all requirements (step-reqs mapping is then lost).
                    If False, return a dictionary mapping step names to requirements.

        Returns
        ----------
            dict or set: Step requirements as dict {step: reqs} with reqs being list[str | tuple] or flattened set
        """
        reqs = {step_name: step_fn.requires
                for step_name, step_fn in self.step_functions.items()
                if step_name in steps}

        if as_set:
            flat_reqs = []
            for req_list in reqs.values():
                for req in req_list:
                    if isinstance(req, tuple):
                        flat_reqs.extend(req)
                    else:
                        flat_reqs.append(req)
            reqs = set(flat_reqs)

        return reqs

    def _propagate_step_name(self, step_name: str, step_fn: StepFunction):
        """Set step_name on a single step function."""
        step_fn.step_name = step_name

    def _extract_products_from_step(self, step_name: str, step_fn: StepFunction):
        """Register products from a step and map each to its producer step name."""
        self.products.extend(step_fn.produces)
        for p in step_fn.produces:
            self.produced_by[p] = step_name

    def _create_debug_label_for_step(self, step_name: str, step_fn: StepFunction, label_index: int) -> int:
        """Create debug label for a single step function. Returns next label_index."""
        if step_fn.has_sink:
            debug_label = 2 ** label_index
            step_fn.sink_spec.debug_label = debug_label
            self.debug_binary_steps_labels[step_name] = debug_label
            return label_index + 1
        return label_index

    def _setup(self):
        """
        Iterates over step_functions and:

        1. Sets ``step_name`` on each StepFunction (for logging).
        2. Extracts products and builds ``produced_by`` mapping
           (``{product_name: producer_step_name}``).
        3. Assigns debug labels (unique powers of 2) to steps that
           have a sink_spec.
        4. Checks that no product is produced by more than one step.
        """
        self.products = []
        label_index = 0
        for step_name, step_fn in self.step_functions.items():
            self._propagate_step_name(step_name, step_fn)
            self._extract_products_from_step(step_name, step_fn)
            label_index = self._create_debug_label_for_step(step_name, step_fn, label_index)
        self._check_duplicates_products()

