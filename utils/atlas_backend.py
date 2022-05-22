from foundations import load_parameters, log_params
from foundations import set_tensorboard_logdir
from foundations import save_artifact
from foundations import submit

metrics = {}


def log_metric(key, value):
    if isinstance(value, float):
        import numpy as np
        value = np.clip(value, 1e-10, 1e10)
        if np.isnan(value):
            value = "nan"
        else:
            value = float(value)
    elif isinstance(value, bool):
        value = 1 if value else 0
    elif isinstance(value, (str, int)):
        value = value
    else:
        raise TypeError("value must be float, int, bool, or str")

    metrics[key] = value
    import pickle
    with open('metric.pkl', 'wb') as f:
        pickle.dump(metrics, f)
    save_artifact('metric.pkl', key='metric')

    from foundations import log_metric as flog_metric
    flog_metric(key, value)


def log(*args, **kwargs):
    print(*args, **kwargs)


log("using atlas framework")
name = 'atlas_backend'

"""
get hparams from atlas webpage
-------
all = ""
$($(".input-metric-column-container")[3]).find(".job-table-row").each(
    function(){
        ret = "["
        $(this).find(".job-cell").each(
        function(){
            ret += "'" + $($(this).find("p")[0]).text() + "'" + ", ";
        });
    all += ret + '],\n';
});
console.log(all)
-------
get results from atlas webpage
-------
all = ""
$($(".input-metric-column-container")[5]).find(".job-table-row").each(
    function(){
        ret = "["
        $(this).find(".job-cell").each(
        function(){
            ret += $($(this).find("p")[0]).text() + ", ";
        });
    all += ret + '],\n';
});
console.log(all)
-------
summarise the results
-------
seed_pos = 12
assert len(params) == len(metrics)
ret = {}
for p, m in zip(params, metrics):
  key = tuple([v for idx, v in enumerate(p) if idx != seed_pos])
  if key not in ret:
    ret[key] = [m]
  else:
    ret[key].append(m)

import numpy as np
for key in sorted(ret.keys()):
  t = np.mean(ret[key], axis=0)
  tt = ("|".join(key)) 
  print(f'|{tt}|%.2f|%.2f|%.2f|%.2f|'% (t[0], t[1], t[3], t[4]))
------- 
"""