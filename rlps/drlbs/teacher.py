import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression

from .reward import normalize


class HybridTeacher:
    def __init__(self, spectra, labels, reward, seed: int = 40, estimators: int = 120):
        self.spectra = np.asarray(spectra, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.reward = reward
        self.seed = seed
        self.filter_score = self._filter_prior()
        self.embedding_score = self._embedding_prior(estimators)
        self.wrapper_score = self._wrapper_prior()
        self.hybrid_score = normalize(
            0.30 * self.filter_score + 0.35 * self.embedding_score + 0.35 * self.wrapper_score
        )
        self.progressive_priors = (
            self.filter_score,
            self.embedding_score,
            self.wrapper_score,
            self.hybrid_score,
        )

    def _filter_prior(self):
        mutual_information = mutual_info_classif(self.spectra, self.labels, random_state=self.seed)
        f_value, _ = f_classif(self.spectra, self.labels)
        return normalize(0.55 * normalize(mutual_information) + 0.45 * normalize(f_value))

    def _embedding_prior(self, estimators):
        forest = RandomForestClassifier(
            n_estimators=estimators,
            max_depth=8,
            random_state=self.seed,
            n_jobs=1,
            class_weight="balanced_subsample",
        )
        forest.fit(self.spectra, self.labels)
        logistic = LogisticRegression(max_iter=300, solver="liblinear", multi_class="ovr")
        logistic.fit(self.spectra, self.labels)
        linear_score = np.mean(np.abs(logistic.coef_), axis=0)
        return normalize(0.5 * normalize(forest.feature_importances_) + 0.5 * normalize(linear_score))

    def _wrapper_prior(self):
        separation = normalize(self.reward.diversity_distance.mean(axis=1))
        return normalize(
            0.35 * normalize(self.reward.quality_score)
            + 0.35 * normalize(self.reward.pointset_score)
            + 0.30 * separation
        )

    def suggest(self, state, candidate_mask, episode: int, switch_span: int = 120):
        available = np.flatnonzero((np.asarray(state) < 0.5) & candidate_mask)
        if available.size == 0:
            raise RuntimeError("No candidate band remains available.")
        prior_index = min(len(self.progressive_priors) - 1, episode // max(1, switch_span))
        score = self.progressive_priors[prior_index]
        return int(available[np.argmax(score[available])])
