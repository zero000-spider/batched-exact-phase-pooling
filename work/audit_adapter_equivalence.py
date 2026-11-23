"""Check the permitted MultiRocket adaptation against the pinned source."""
from pathlib import Path
import ast, hashlib, importlib.util, json, sys, types
import numpy as np
from numba import njit, set_num_threads
from sklearn.linear_model import RidgeClassifierCV
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
VENDOR=HERE/'modern_baselines'

@njit
def seed_compiled(seed): np.random.seed(seed)

def load_source(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module

def same_parameters(a,b):
    assert len(a)==len(b)
    for x,y in zip(a,b):np.testing.assert_array_equal(x,y)

def main(destination):
    set_num_threads(2)
    source=VENDOR/'MultiRocket/multirocket/multirocket.py'
    adapter=VENDOR/'multirocket_ridge_compat.py'
    original_text=source.read_text(encoding='utf-8')
    adapted_text=adapter.read_text(encoding='utf-8')
    obsolete_import='from multirocket.logistic_regression import LogisticRegression'
    assert original_text.count(obsolete_import)==1
    assert original_text.count('normalize=False')==1
    expected=original_text.replace(obsolete_import,'').replace('normalize=False','')
    assert ast.dump(ast.parse(expected))==ast.dump(ast.parse(adapted_text))
    assert '2026-11-23' in adapted_text.splitlines()[0]
    # Import the original source without installing its unused torch dependency.
    # This sentinel raises if the logistic option is invoked during the audit.
    placeholder=types.ModuleType('multirocket.logistic_regression')
    def unused_logistic(*args,**kwargs):
        raise AssertionError('The ridge audit invoked the logistic classifier')
    placeholder.LogisticRegression=unused_logistic
    saved=sys.modules.get(placeholder.__name__)
    sys.modules[placeholder.__name__]=placeholder
    try:
        original=load_source('_ei_upstream_multirocket',source)
    finally:
        if saved is None:sys.modules.pop(placeholder.__name__,None)
        else:sys.modules[placeholder.__name__]=saved
    adapted=load_source('multirocket_ridge_compat',adapter)
    shim_calls=[]
    def current_ridge(**kwargs):
        # The original API call cannot run on modern sklearn without this shim.
        assert kwargs.pop('normalize') is False
        shim_calls.append(True)
        return RidgeClassifierCV(**kwargs)
    original.RidgeClassifierCV=current_ridge
    rows=[]
    with threadpool_limits(limits=1):
        for n in (32,64):
            rng=np.random.default_rng(20261123+n)
            train=np.ascontiguousarray(rng.normal(size=(18,n)),dtype=np.float64)
            test=np.ascontiguousarray(rng.normal(size=(7,n)),dtype=np.float64)
            labels=np.tile(np.arange(3),6)
            for budget in (9408,49728):
                models=[]
                for module in (original,adapted):
                    np.random.seed(23);seed_compiled(23)
                    model=module.MultiRocket(num_features=budget,classifier='ridge')
                    model.fit(train,labels,predict_on_train=False);models.append(model)
                a,b=models
                same_parameters(a.base_parameters,b.base_parameters)
                same_parameters(a.diff1_parameters,b.diff1_parameters)
                values=0;dimension=None
                for matrix in (train,test):
                    diff=np.ascontiguousarray(np.diff(matrix,axis=1))
                    fa=original.transform(matrix,diff,a.base_parameters,a.diff1_parameters,4)
                    fb=adapted.transform(matrix,diff,b.base_parameters,b.diff1_parameters,4)
                    np.testing.assert_array_equal(fa,fb)
                    values+=fa.size;dimension=fa.shape[1]
                assert dimension==budget
                pa=a.predict(test);pb=b.predict(test)
                np.testing.assert_array_equal(pa,pb)
                np.testing.assert_array_equal(a.classifier.decision_function(fa),b.classifier.decision_function(fb))
                assert float(a.classifier[-1].alpha_)==float(b.classifier[-1].alpha_)
                rows.append(dict(length=n,feature_dim=dimension,feature_values_compared=values,
                    fitted_parameters_identical=True,features_bitwise_equal=True,
                    ridge_scores_identical=True,predictions_identical=True,alpha=float(a.classifier[-1].alpha_)))
    assert len(shim_calls)==4
    result=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        adapter_sha256=hashlib.sha256(adapter.read_bytes()).hexdigest(),
        source_edits_verified=True,configurations=rows,
        method='Pinned upstream source loaded unchanged. Unused logistic import is supplied by a sentinel that raises if called. '
               'Only the upstream RidgeClassifierCV binding uses a shim accepting normalize=False for the current sklearn API. '
               'Synthetic fixtures exercise both lengths and both feature budgets; upstream and adapter share seeds and inputs.')
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('ADAPTER_EQUIVALENCE_VERIFIED '+str(sum(r['feature_values_compared'] for r in rows))+' feature values',flush=True)

if __name__=='__main__':main(HERE.parent/'reproduced/adapter_equivalence.json')
