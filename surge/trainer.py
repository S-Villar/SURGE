from .models import MLPModel, GPRModel, RFRModel
from .preprocessing import train_test_split_data, make_cv_splits
from .metrics import evaluate, summarize
from .utils import get_device, save_model, save_json


class SurrogateTrainer:
    def __init__(self, model_type="rfr", **kwargs):
        self.device = get_device()
        model_map = {"mlp": MLPModel, "gpr": GPRModel, "rfr": RFRModel}
        if model_type not in model_map:
            raise ValueError(f"Unknown model type '{model_type}'")
        self.model = model_map[model_type](**kwargs)

    def fit(self, X, y, test_size=0.2):
        self.X_tv, self.X_test, self.y_tv, self.y_test = train_test_split_data(
            X, y, test_size
        )
        return self

    def cross_validate(self, n_splits=5):
        results = {"mse_list": [], "r2_list": [], "time_list": []}
        for tr_idx, vl_idx in make_cv_splits(self.X_tv, self.y_tv, n_splits):
            X_tr, X_vl = self.X_tv[tr_idx], self.X_tv[vl_idx]
            y_tr, y_vl = self.y_tv[tr_idx], self.y_tv[vl_idx]
            self.model.fit(X_tr, y_tr)
            mse, r2, t = evaluate(self.model, X_vl, y_vl)
            results["mse_list"].append(mse)
            results["r2_list"].append(r2)
            results["time_list"].append(t)
        self.results = summarize(results)
        return self.results

    def save(self, out_dir="models"):
        save_model(self.model, f"{out_dir}/model.joblib")
        save_json(self.results, f"{out_dir}/results.json")
