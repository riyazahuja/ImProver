@[inherit_doc] notation "ℚ" => Rat


/-- Nonnegative rational numbers. -/
def NNRat := {q : ℚ // 0 ≤ q}


@[inherit_doc] notation "ℚ≥0" => NNRat


/-- Typeclass for the canonical homomorphism `ℚ≥0 → K`.

This should be considered as a notation typeclass. The sole purpose of this typeclass is to be
extended by `DivisionSemiring`. -/
class NNRatCast (K : Type*) where
  /-- The canonical homomorphism `ℚ≥0 → K`.

  Do not use directly. Use the coercion instead. -/
  protected nnratCast : ℚ≥0 → K


instance NNRat.instNNRatCast : NNRatCast ℚ≥0 where nnratCast q := q


/-- Canonical homomorphism from `ℚ≥0` to a division semiring `K`.

This is just the bare function in order to aid in creating instances of `DivisionSemiring`. -/
@[coe, reducible, match_pattern] protected def NNRat.cast : ℚ≥0 → K := NNRatCast.nnratCast

-- See note [coercion into rings]

instance NNRatCast.toCoeTail [NNRatCast K] : CoeTail ℚ≥0 K where coe := NNRat.cast

-- See note [coercion into rings]

instance NNRatCast.toCoeHTCT [NNRatCast K] : CoeHTCT ℚ≥0 K where coe := NNRat.cast


instance Rat.instNNRatCast : NNRatCast ℚ := ⟨Subtype.val⟩


/-- The numerator of a nonnegative rational. -/
def num (q : ℚ≥0) : ℕ := (q : ℚ).num.natAbs


/-- The denominator of a nonnegative rational. -/
def den (q : ℚ≥0) : ℕ := (q : ℚ).den


@[simp] lemma num_mk (q : ℚ) (hq : 0 ≤ q) : num ⟨q, hq⟩ = q.num.natAbs := rfl

@[simp] lemma den_mk (q : ℚ) (hq : 0 ≤ q) : den ⟨q, hq⟩ = q.den := rfl


@[norm_cast] lemma cast_id (n : ℚ≥0) : NNRat.cast n = n := rfl

@[simp] lemma cast_eq_id : NNRat.cast = id := rfl


@[norm_cast] lemma cast_id (n : ℚ) : Rat.cast n = n := rfl

@[simp] lemma cast_eq_id : Rat.cast = id := rfl


