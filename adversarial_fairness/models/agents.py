"""
Model-level agents (paper Section 3.2).

The classifier and adversary are the two reactive optimization agents of the
minimax game:

  ClassifierAgent / ImageClassifierAgent
      Utility-based agent f_θf : balances task accuracy against the adversarial
      fairness penalty. Outputs the scalar prediction Ŷ = f_θf(x).
  AdversaryAgent
      Goal-based agent r_θr : receives only Ŷ and tries to recover the sensitive
      attributes (one logit per attribute).

These are thin role-labelled subclasses of the underlying predictors
(Classifier / ImageClassifier / Adversary). They add no parameters and do NOT
override forward(), so the computation, .parameters() and .state_dict() are
byte-for-byte identical to the base modules — this layer is structural only and
does not change any training result.
"""

from .classifier import Classifier
from .image_classifier import ImageClassifier
from .adversary import Adversary


class ClassifierAgent(Classifier):
    """Utility-based agent — tabular MLP predictor f_θf (paper §3.2)."""

    role = "classifier"

    def act(self, x):
        """Produce the prediction probability Ŷ = f_θf(x)."""
        return self.forward(x)


class ImageClassifierAgent(ImageClassifier):
    """Utility-based agent — ResNet-18 predictor f_θf for image data (paper §3.2)."""

    role = "classifier"

    def act(self, x):
        """Produce the prediction probability Ŷ = f_θf(x)."""
        return self.forward(x)


class AdversaryAgent(Adversary):
    """Goal-based agent — recovers sensitive attributes from Ŷ, r_θr (paper §3.2)."""

    role = "adversary"

    def act(self, y_hat):
        """Return one logit per sensitive attribute from the prediction Ŷ."""
        return self.forward(y_hat)
