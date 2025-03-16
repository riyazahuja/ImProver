/-- Auxiliary class, endowing a type `E` with a function `norm : E → ℝ` with notation `‖x‖`. This
class is designed to be extended in more interesting classes specifying the properties of the norm.
-/
@[notation_class]
class Norm (E : Type*) where
  /-- the `ℝ`-valued norm function. -/
  norm : E → ℝ


/-- Auxiliary class, endowing a type `α` with a function `nnnorm : α → ℝ≥0` with notation `‖x‖₊`. -/
@[notation_class]
class NNNorm (E : Type*) where
  /-- the `ℝ≥0`-valued norm function. -/
  nnnorm : E → ℝ≥0


/-- Auxiliary class, endowing a type `α` with a function `enorm : α → ℝ≥0∞` with notation `‖x‖ₑ`. -/
@[notation_class]
class ENorm (E : Type*) where
  /-- the `ℝ≥0∞`-valued norm function. -/
  enorm : E → ℝ≥0∞


@[inherit_doc] notation "‖" e "‖" => norm e

@[inherit_doc] notation "‖" e "‖₊" => nnnorm e

@[inherit_doc] notation "‖" e "‖ₑ" => enorm e


/-- A seminormed group is an additive group endowed with a norm for which `dist x y = ‖x - y‖`
defines a pseudometric space structure. -/
class SeminormedAddGroup (E : Type*) extends Norm E, AddGroup E, PseudoMetricSpace E where
  dist := fun x y => ‖x - y‖
  /-- The distance function is induced by the norm. -/
  dist_eq : ∀ x y, dist x y = ‖x - y‖ := by aesop


/-- A seminormed group is a group endowed with a norm for which `dist x y = ‖x / y‖` defines a
pseudometric space structure. -/
@[to_additive]
class SeminormedGroup (E : Type*) extends Norm E, Group E, PseudoMetricSpace E where
  dist := fun x y => ‖x / y‖
  /-- The distance function is induced by the norm. -/
  dist_eq : ∀ x y, dist x y = ‖x / y‖ := by aesop


/-- A normed group is an additive group endowed with a norm for which `dist x y = ‖x - y‖` defines a
metric space structure. -/
class NormedAddGroup (E : Type*) extends Norm E, AddGroup E, MetricSpace E where
  dist := fun x y => ‖x - y‖
  /-- The distance function is induced by the norm. -/
  dist_eq : ∀ x y, dist x y = ‖x - y‖ := by aesop


/-- A normed group is a group endowed with a norm for which `dist x y = ‖x / y‖` defines a metric
space structure. -/
@[to_additive]
class NormedGroup (E : Type*) extends Norm E, Group E, MetricSpace E where
  dist := fun x y => ‖x / y‖
  /-- The distance function is induced by the norm. -/
  dist_eq : ∀ x y, dist x y = ‖x / y‖ := by aesop


/-- A seminormed group is an additive group endowed with a norm for which `dist x y = ‖x - y‖`
defines a pseudometric space structure. -/
class SeminormedAddCommGroup (E : Type*) extends Norm E, AddCommGroup E,
  PseudoMetricSpace E where
  dist := fun x y => ‖x - y‖
  /-- The distance function is induced by the norm. -/
  dist_eq : ∀ x y, dist x y = ‖x - y‖ := by aesop


/-- A seminormed group is a group endowed with a norm for which `dist x y = ‖x / y‖`
defines a pseudometric space structure. -/
@[to_additive]
class SeminormedCommGroup (E : Type*) extends Norm E, CommGroup E, PseudoMetricSpace E where
  dist := fun x y => ‖x / y‖
  /-- The distance function is induced by the norm. -/
  dist_eq : ∀ x y, dist x y = ‖x / y‖ := by aesop


/-- A normed group is an additive group endowed with a norm for which `dist x y = ‖x - y‖` defines a
metric space structure. -/
class NormedAddCommGroup (E : Type*) extends Norm E, AddCommGroup E, MetricSpace E where
  dist := fun x y => ‖x - y‖
  /-- The distance function is induced by the norm. -/
  dist_eq : ∀ x y, dist x y = ‖x - y‖ := by aesop


/-- A normed group is a group endowed with a norm for which `dist x y = ‖x / y‖` defines a metric
space structure. -/
@[to_additive]
class NormedCommGroup (E : Type*) extends Norm E, CommGroup E, MetricSpace E where
  dist := fun x y => ‖x / y‖
  /-- The distance function is induced by the norm. -/
  dist_eq : ∀ x y, dist x y = ‖x / y‖ := by aesop

-- See note [lower instance priority]

@[to_additive]
instance (priority := 100) NormedGroup.toSeminormedGroup [NormedGroup E] : SeminormedGroup E :=
  { ‹NormedGroup E› with }

-- See note [lower instance priority]

@[to_additive]
instance (priority := 100) NormedCommGroup.toSeminormedCommGroup [NormedCommGroup E] :
    SeminormedCommGroup E :=
  { ‹NormedCommGroup E› with }

-- See note [lower instance priority]

@[to_additive]
instance (priority := 100) SeminormedCommGroup.toSeminormedGroup [SeminormedCommGroup E] :
    SeminormedGroup E :=
  { ‹SeminormedCommGroup E› with }

-- See note [lower instance priority]

@[to_additive]
instance (priority := 100) NormedCommGroup.toNormedGroup [NormedCommGroup E] : NormedGroup E :=
  { ‹NormedCommGroup E› with }

-- See note [reducible non-instances]

/-- Construct a `NormedGroup` from a `SeminormedGroup` satisfying `∀ x, ‖x‖ = 0 → x = 1`. This
avoids having to go back to the `(Pseudo)MetricSpace` level when declaring a `NormedGroup`
instance as a special case of a more general `SeminormedGroup` instance. -/
@[to_additive "Construct a `NormedAddGroup` from a `SeminormedAddGroup`
satisfying `∀ x, ‖x‖ = 0 → x = 0`. This avoids having to go back to the `(Pseudo)MetricSpace`
level when declaring a `NormedAddGroup` instance as a special case of a more general
`SeminormedAddGroup` instance."]
abbrev NormedGroup.ofSeparation [SeminormedGroup E] (h : ∀ x : E, ‖x‖ = 0 → x = 1) :
    NormedGroup E where
  dist_eq := ‹SeminormedGroup E›.dist_eq
  toMetricSpace :=
    { eq_of_dist_eq_zero := fun hxy =>
                                  /-
                                    𝓕 : Type u_1
                                    α : Type u_2
                                    ι : Type u_3
                                    κ : Type u_4
                                    E : Type u_5
                                    F : Type u_6
                                    G : Type u_7
                                    inst✝ : SeminormedGroup E
                                    h : ∀ (x : E), Eq (Norm.norm x) 0 → Eq x 1
                                    x✝ y✝ : E
                                    hxy : Eq (Dist.dist x✝ y✝) 0
                                    ⊢ Eq (Norm.norm (HDiv.hDiv x✝ y✝)) 0
                                  -/
        div_eq_one.1 <| h _ <| by exact (‹SeminormedGroup E›.dist_eq _ _).symm.trans hxy }
                                  /-
                                    🎉 no goals
                                  -/
      -- Porting note: the `rwa` no longer worked, but it was easy enough to provide the term.
      -- however, notice that if you make `x` and `y` accessible, then the following does work:
      -- `have := ‹SeminormedGroup E›.dist_eq x y; rwa [← this]`, so I'm not sure why the `rwa`
      -- was broken.

-- See note [reducible non-instances]

/-- Construct a `NormedCommGroup` from a `SeminormedCommGroup` satisfying
`∀ x, ‖x‖ = 0 → x = 1`. This avoids having to go back to the `(Pseudo)MetricSpace` level when
declaring a `NormedCommGroup` instance as a special case of a more general `SeminormedCommGroup`
instance. -/
@[to_additive "Construct a `NormedAddCommGroup` from a
`SeminormedAddCommGroup` satisfying `∀ x, ‖x‖ = 0 → x = 0`. This avoids having to go back to the
`(Pseudo)MetricSpace` level when declaring a `NormedAddCommGroup` instance as a special case
of a more general `SeminormedAddCommGroup` instance."]
abbrev NormedCommGroup.ofSeparation [SeminormedCommGroup E] (h : ∀ x : E, ‖x‖ = 0 → x = 1) :
    NormedCommGroup E :=
  { ‹SeminormedCommGroup E›, NormedGroup.ofSeparation h with }

-- See note [reducible non-instances]

/-- Construct a seminormed group from a multiplication-invariant distance. -/
@[to_additive
  "Construct a seminormed group from a translation-invariant distance."]
abbrev SeminormedGroup.ofMulDist [Norm E] [Group E] [PseudoMetricSpace E]
    (h₁ : ∀ x : E, ‖x‖ = dist x 1) (h₂ : ∀ x y z : E, dist x y ≤ dist (x * z) (y * z)) :
    SeminormedGroup E where
  dist_eq x y := by
    /-
      𝓕 : Type u_1
      α : Type u_2
      ι : Type u_3
      κ : Type u_4
      E : Type u_5
      F : Type u_6
      G : Type u_7
      inst✝² : Norm E
      inst✝¹ : Group E
      inst✝ : PseudoMetricSpace E
      h₁ : ∀ (x : E), Eq (Norm.norm x) (Dist.dist x 1)
      h₂ : ∀ (x y z : E), LE.le (Dist.dist x y) (Dist.dist (HMul.hMul x z) (HMul.hMu …
      x y : E
      ⊢ Eq (Dist.dist x y) (Norm.norm (HDiv.hDiv x y))
    -/
    rw [h₁]; apply le_antisymm
      /-
        case a
        𝓕 : Type u_1
        α : Type u_2
        ι : Type u_3
        κ : Type u_4
        E : Type u_5
        F : Type u_6
        G : Type u_7
        inst✝² : Norm E
        inst✝¹ : Group E
        inst✝ : PseudoMetricSpace E
        h₁ : ∀ (x : E), Eq (Norm.norm x) (Dist.dist x 1)
        h₂ : ∀ (x y z : E), LE.le (Dist.dist x y) (Dist.dist (HMul.hMul x z) (HMul.hMu …
        x y : E
        ⊢ LE.le (Dist.dist x y) (Dist.dist (HDiv.hDiv x y) 1)
      -/
    · simpa only [div_eq_mul_inv, ← mul_inv_cancel y] using h₂ _ _ _
      /-
        🎉 no goals
      -/
      /-
        case a
        𝓕 : Type u_1
        α : Type u_2
        ι : Type u_3
        κ : Type u_4
        E : Type u_5
        F : Type u_6
        G : Type u_7
        inst✝² : Norm E
        inst✝¹ : Group E
        inst✝ : PseudoMetricSpace E
        h₁ : ∀ (x : E), Eq (Norm.norm x) (Dist.dist x 1)
        h₂ : ∀ (x y z : E), LE.le (Dist.dist x y) (Dist.dist (HMul.hMul x z) (HMul.hMu …
        x y : E
        ⊢ LE.le (Dist.dist (HDiv.hDiv x y) 1) (Dist.dist x y)
      -/
    · simpa only [div_mul_cancel, one_mul] using h₂ (x / y) 1 y
      /-
        🎉 no goals
      -/

-- See note [reducible non-instances]

/-- Construct a seminormed group from a multiplication-invariant pseudodistance. -/
@[to_additive
  "Construct a seminormed group from a translation-invariant pseudodistance."]
abbrev SeminormedGroup.ofMulDist' [Norm E] [Group E] [PseudoMetricSpace E]
    (h₁ : ∀ x : E, ‖x‖ = dist x 1) (h₂ : ∀ x y z : E, dist (x * z) (y * z) ≤ dist x y) :
    SeminormedGroup E where
  dist_eq x y := by
    /-
      𝓕 : Type u_1
      α : Type u_2
      ι : Type u_3
      κ : Type u_4
      E : Type u_5
      F : Type u_6
      G : Type u_7
      inst✝² : Norm E
      inst✝¹ : Group E
      inst✝ : PseudoMetricSpace E
      h₁ : ∀ (x : E), Eq (Norm.norm x) (Dist.dist x 1)
      h₂ : ∀ (x y z : E), LE.le (Dist.dist (HMul.hMul x z) (HMul.hMul y z)) (Dist.di …
      x y : E
      ⊢ Eq (Dist.dist x y) (Norm.norm (HDiv.hDiv x y))
    -/
    rw [h₁]; apply le_antisymm
      /-
        case a
        𝓕 : Type u_1
        α : Type u_2
        ι : Type u_3
        κ : Type u_4
        E : Type u_5
        F : Type u_6
        G : Type u_7
        inst✝² : Norm E
        inst✝¹ : Group E
        inst✝ : PseudoMetricSpace E
        h₁ : ∀ (x : E), Eq (Norm.norm x) (Dist.dist x 1)
        h₂ : ∀ (x y z : E), LE.le (Dist.dist (HMul.hMul x z) (HMul.hMul y z)) (Dist.di …
        x y : E
        ⊢ LE.le (Dist.dist x y) (Dist.dist (HDiv.hDiv x y) 1)
      -/
    · simpa only [div_mul_cancel, one_mul] using h₂ (x / y) 1 y
      /-
        🎉 no goals
      -/
      /-
        case a
        𝓕 : Type u_1
        α : Type u_2
        ι : Type u_3
        κ : Type u_4
        E : Type u_5
        F : Type u_6
        G : Type u_7
        inst✝² : Norm E
        inst✝¹ : Group E
        inst✝ : PseudoMetricSpace E
        h₁ : ∀ (x : E), Eq (Norm.norm x) (Dist.dist x 1)
        h₂ : ∀ (x y z : E), LE.le (Dist.dist (HMul.hMul x z) (HMul.hMul y z)) (Dist.di …
        x y : E
        ⊢ LE.le (Dist.dist (HDiv.hDiv x y) 1) (Dist.dist x y)
      -/
    · simpa only [div_eq_mul_inv, ← mul_inv_cancel y] using h₂ _ _ _
      /-
        🎉 no goals
      -/

-- See note [reducible non-instances]

/-- Construct a seminormed group from a multiplication-invariant pseudodistance. -/
@[to_additive
  "Construct a seminormed group from a translation-invariant pseudodistance."]
abbrev SeminormedCommGroup.ofMulDist [Norm E] [CommGroup E] [PseudoMetricSpace E]
    (h₁ : ∀ x : E, ‖x‖ = dist x 1) (h₂ : ∀ x y z : E, dist x y ≤ dist (x * z) (y * z)) :
    SeminormedCommGroup E :=
  { SeminormedGroup.ofMulDist h₁ h₂ with
    mul_comm := mul_comm }

-- See note [reducible non-instances]

/-- Construct a seminormed group from a multiplication-invariant pseudodistance. -/
@[to_additive
  "Construct a seminormed group from a translation-invariant pseudodistance."]
abbrev SeminormedCommGroup.ofMulDist' [Norm E] [CommGroup E] [PseudoMetricSpace E]
    (h₁ : ∀ x : E, ‖x‖ = dist x 1) (h₂ : ∀ x y z : E, dist (x * z) (y * z) ≤ dist x y) :
    SeminormedCommGroup E :=
  { SeminormedGroup.ofMulDist' h₁ h₂ with
    mul_comm := mul_comm }

-- See note [reducible non-instances]

/-- Construct a normed group from a multiplication-invariant distance. -/
@[to_additive
  "Construct a normed group from a translation-invariant distance."]
abbrev NormedGroup.ofMulDist [Norm E] [Group E] [MetricSpace E] (h₁ : ∀ x : E, ‖x‖ = dist x 1)
    (h₂ : ∀ x y z : E, dist x y ≤ dist (x * z) (y * z)) : NormedGroup E :=
  { SeminormedGroup.ofMulDist h₁ h₂ with
    eq_of_dist_eq_zero := eq_of_dist_eq_zero }

-- See note [reducible non-instances]

/-- Construct a normed group from a multiplication-invariant pseudodistance. -/
@[to_additive
  "Construct a normed group from a translation-invariant pseudodistance."]
abbrev NormedGroup.ofMulDist' [Norm E] [Group E] [MetricSpace E] (h₁ : ∀ x : E, ‖x‖ = dist x 1)
    (h₂ : ∀ x y z : E, dist (x * z) (y * z) ≤ dist x y) : NormedGroup E :=
  { SeminormedGroup.ofMulDist' h₁ h₂ with
    eq_of_dist_eq_zero := eq_of_dist_eq_zero }

-- See note [reducible non-instances]

/-- Construct a normed group from a multiplication-invariant pseudodistance. -/
@[to_additive
"Construct a normed group from a translation-invariant pseudodistance."]
abbrev NormedCommGroup.ofMulDist [Norm E] [CommGroup E] [MetricSpace E]
    (h₁ : ∀ x : E, ‖x‖ = dist x 1) (h₂ : ∀ x y z : E, dist x y ≤ dist (x * z) (y * z)) :
    NormedCommGroup E :=
  { NormedGroup.ofMulDist h₁ h₂ with
    mul_comm := mul_comm }

-- See note [reducible non-instances]

/-- Construct a normed group from a multiplication-invariant pseudodistance. -/
@[to_additive
  "Construct a normed group from a translation-invariant pseudodistance."]
abbrev NormedCommGroup.ofMulDist' [Norm E] [CommGroup E] [MetricSpace E]
    (h₁ : ∀ x : E, ‖x‖ = dist x 1) (h₂ : ∀ x y z : E, dist (x * z) (y * z) ≤ dist x y) :
    NormedCommGroup E :=
  { NormedGroup.ofMulDist' h₁ h₂ with
    mul_comm := mul_comm }

-- See note [reducible non-instances]

/-- Construct a seminormed group from a seminorm, i.e., registering the pseudodistance and the
pseudometric space structure from the seminorm properties. Note that in most cases this instance
creates bad definitional equalities (e.g., it does not take into account a possibly existing
`UniformSpace` instance on `E`). -/
@[to_additive
  "Construct a seminormed group from a seminorm, i.e., registering the pseudodistance
and the pseudometric space structure from the seminorm properties. Note that in most cases this
instance creates bad definitional equalities (e.g., it does not take into account a possibly
existing `UniformSpace` instance on `E`)."]
abbrev GroupSeminorm.toSeminormedGroup [Group E] (f : GroupSeminorm E) : SeminormedGroup E where
  dist x y := f (x / y)
  norm := f
  dist_eq _ _ := rfl
                    /-
                      𝓕 : Type u_1
                      α : Type u_2
                      ι : Type u_3
                      κ : Type u_4
                      E : Type u_5
                      F : Type u_6
                      G : Type u_7
                      inst✝ : Group E
                      f : GroupSeminorm E
                      x : E
                      ⊢ Eq (Dist.dist x x) 0
                    -/
  dist_self x := by simp only [div_self', map_one_eq_zero]
                    /-
                      🎉 no goals
                    -/
  dist_triangle := le_map_div_add_map_div f
  dist_comm := map_div_rev f
                       /-
                         𝓕 : Type u_1
                         α : Type u_2
                         ι : Type u_3
                         κ : Type u_4
                         E : Type u_5
                         F : Type u_6
                         G : Type u_7
                         inst✝ : Group E
                         f : GroupSeminorm E
                         x y : E
                         ⊢ Eq ((fun x y => ↑⟨f (HDiv.hDiv x y), ⋯⟩) x y) (ENNReal.ofReal (Dist.dist x y))
                       -/
  edist_dist x y := by exact ENNReal.coe_nnreal_eq _
                       /-
                         🎉 no goals
                       -/
  -- Porting note: how did `mathlib3` solve this automatically?

-- See note [reducible non-instances]

/-- Construct a seminormed group from a seminorm, i.e., registering the pseudodistance and the
pseudometric space structure from the seminorm properties. Note that in most cases this instance
creates bad definitional equalities (e.g., it does not take into account a possibly existing
`UniformSpace` instance on `E`). -/
@[to_additive
  "Construct a seminormed group from a seminorm, i.e., registering the pseudodistance
and the pseudometric space structure from the seminorm properties. Note that in most cases this
instance creates bad definitional equalities (e.g., it does not take into account a possibly
existing `UniformSpace` instance on `E`)."]
abbrev GroupSeminorm.toSeminormedCommGroup [CommGroup E] (f : GroupSeminorm E) :
    SeminormedCommGroup E :=
  { f.toSeminormedGroup with
    mul_comm := mul_comm }

-- See note [reducible non-instances]

/-- Construct a normed group from a norm, i.e., registering the distance and the metric space
structure from the norm properties. Note that in most cases this instance creates bad definitional
equalities (e.g., it does not take into account a possibly existing `UniformSpace` instance on
`E`). -/
@[to_additive
  "Construct a normed group from a norm, i.e., registering the distance and the metric
space structure from the norm properties. Note that in most cases this instance creates bad
definitional equalities (e.g., it does not take into account a possibly existing `UniformSpace`
instance on `E`)."]
abbrev GroupNorm.toNormedGroup [Group E] (f : GroupNorm E) : NormedGroup E :=
  { f.toGroupSeminorm.toSeminormedGroup with
    eq_of_dist_eq_zero := fun h => div_eq_one.1 <| eq_one_of_map_eq_zero f h }

-- See note [reducible non-instances]

/-- Construct a normed group from a norm, i.e., registering the distance and the metric space
structure from the norm properties. Note that in most cases this instance creates bad definitional
equalities (e.g., it does not take into account a possibly existing `UniformSpace` instance on
`E`). -/
@[to_additive
  "Construct a normed group from a norm, i.e., registering the distance and the metric
space structure from the norm properties. Note that in most cases this instance creates bad
definitional equalities (e.g., it does not take into account a possibly existing `UniformSpace`
instance on `E`)."]
abbrev GroupNorm.toNormedCommGroup [CommGroup E] (f : GroupNorm E) : NormedCommGroup E :=
  { f.toNormedGroup with
    mul_comm := mul_comm }


@[to_additive]
theorem dist_eq_norm_div (a b : E) : dist a b = ‖a / b‖ :=
  SeminormedGroup.dist_eq _ _


@[to_additive]
                                                               /-
                                                                 E : Type u_5
                                                                 inst✝ : SeminormedGroup E
                                                                 a b : E
                                                                 ⊢ Eq (Dist.dist a b) (Norm.norm (HDiv.hDiv b a))
                                                               -/
theorem dist_eq_norm_div' (a b : E) : dist a b = ‖b / a‖ := by rw [dist_comm, dist_eq_norm_div]
                                                               /-
                                                                 🎉 no goals
                                                               -/


alias dist_eq_norm := dist_eq_norm_sub


alias dist_eq_norm' := dist_eq_norm_sub'


@[to_additive of_forall_le_norm]
lemma DiscreteTopology.of_forall_le_norm' (hpos : 0 < r) (hr : ∀ x : E, x ≠ 1 → r ≤ ‖x‖) :
    DiscreteTopology E :=
  .of_forall_le_dist hpos fun x y hne ↦ by
    /-
      E : Type u_5
      inst✝ : SeminormedGroup E
      r : Real
      hpos : LT.lt 0 r
      hr : ∀ (x : E), Ne x 1 → LE.le r (Norm.norm x)
      x y : E
      hne : Ne x y
      ⊢ (fun x1 x2 => LE.le r (Dist.dist x1 x2)) x y
    -/
    simp only [dist_eq_norm_div]
    /-
      E : Type u_5
      inst✝ : SeminormedGroup E
      r : Real
      hpos : LT.lt 0 r
      hr : ∀ (x : E), Ne x 1 → LE.le r (Norm.norm x)
      x y : E
      hne : Ne x y
      ⊢ LE.le r (Norm.norm (HDiv.hDiv x y))
    -/
    exact hr _ (div_ne_one.2 hne)
    /-
      🎉 no goals
    -/


@[to_additive (attr := simp)]
                                                      /-
                                                        E : Type u_5
                                                        inst✝ : SeminormedGroup E
                                                        a : E
                                                        ⊢ Eq (Dist.dist a 1) (Norm.norm a)
                                                      -/
theorem dist_one_right (a : E) : dist a 1 = ‖a‖ := by rw [dist_eq_norm_div, div_one]
                                                      /-
                                                        🎉 no goals
                                                      -/


@[to_additive]
theorem inseparable_one_iff_norm {a : E} : Inseparable a 1 ↔ ‖a‖ = 0 := by
  /-
    E : Type u_5
    inst✝ : SeminormedGroup E
    a : E
    ⊢ Iff (Inseparable a 1) (Eq (Norm.norm a) 0)
  -/
  rw [Metric.inseparable_iff, dist_one_right]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem dist_one_left : dist (1 : E) = norm :=
                     /-
                       E : Type u_5
                       inst✝ : SeminormedGroup E
                       a : E
                       ⊢ Eq (Dist.dist 1 a) (Norm.norm a)
                     -/
  funext fun a => by rw [dist_comm, dist_one_right]
                     /-
                       🎉 no goals
                     -/


@[to_additive]
theorem norm_div_rev (a b : E) : ‖a / b‖ = ‖b / a‖ := by
  /-
    E : Type u_5
    inst✝ : SeminormedGroup E
    a b : E
    ⊢ Eq (Norm.norm (HDiv.hDiv a b)) (Norm.norm (HDiv.hDiv b a))
  -/
  simpa only [dist_eq_norm_div] using dist_comm a b
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp) norm_neg]
                                              /-
                                                E : Type u_5
                                                inst✝ : SeminormedGroup E
                                                a : E
                                                ⊢ Eq (Norm.norm (Inv.inv a)) (Norm.norm a)
                                              -/
theorem norm_inv' (a : E) : ‖a⁻¹‖ = ‖a‖ := by simpa using norm_div_rev 1 a
                                              /-
                                                🎉 no goals
                                              -/


@[to_additive (attr := simp) norm_abs_zsmul]
theorem norm_zpow_abs (a : E) (n : ℤ) : ‖a ^ |n|‖ = ‖a ^ n‖ := by
  /-
    E : Type u_5
    inst✝ : SeminormedGroup E
    a : E
    n : Int
    ⊢ Eq (Norm.norm (HPow.hPow a (abs n))) (Norm.norm (HPow.hPow a n))
  -/
                                       /-
                                         🎉 no goals
                                       -/
  rcases le_total 0 n with hn | hn <;> simp [hn, abs_of_nonneg, abs_of_nonpos]
                                       /-
                                         🎉 no goals
                                       -/


@[to_additive (attr := simp) norm_natAbs_smul]
theorem norm_pow_natAbs (a : E) (n : ℤ) : ‖a ^ n.natAbs‖ = ‖a ^ n‖ := by
  /-
    E : Type u_5
    inst✝ : SeminormedGroup E
    a : E
    n : Int
    ⊢ Eq (Norm.norm (HPow.hPow a n.natAbs)) (Norm.norm (HPow.hPow a n))
  -/
  rw [← zpow_natCast, ← Int.abs_eq_natAbs, norm_zpow_abs]
  /-
    🎉 no goals
  -/


@[to_additive norm_isUnit_zsmul]
theorem norm_zpow_isUnit (a : E) {n : ℤ} (hn : IsUnit n) : ‖a ^ n‖ = ‖a‖ := by
  /-
    E : Type u_5
    inst✝ : SeminormedGroup E
    a : E
    n : Int
    hn : IsUnit n
    ⊢ Eq (Norm.norm (HPow.hPow a n)) (Norm.norm a)
  -/
  rw [← norm_pow_natAbs, Int.isUnit_iff_natAbs_eq.mp hn, pow_one]
  /-
    🎉 no goals
  -/


@[simp]
theorem norm_units_zsmul {E : Type*} [SeminormedAddGroup E] (n : ℤˣ) (a : E) : ‖n • a‖ = ‖a‖ :=
  norm_isUnit_zsmul a n.isUnit


open scoped symmDiff in
@[to_additive]
theorem dist_mulIndicator (s t : Set α) (f : α → E) (x : α) :
    dist (s.mulIndicator f x) (t.mulIndicator f x) = ‖(s ∆ t).mulIndicator f x‖ := by
  /-
    α : Type u_2
    E : Type u_5
    inst✝ : SeminormedGroup E
    s t : Set α
    f : α → E
    x : α
    ⊢ Eq (Dist.dist (s.mulIndicator f x) (t.mulIndicator f x)) (Norm.norm ((symmDi …
  -/
  rw [dist_eq_norm_div, Set.apply_mulIndicator_symmDiff norm_inv']
  /-
    🎉 no goals
  -/


/-- **Triangle inequality** for the norm. -/
@[to_additive norm_add_le "**Triangle inequality** for the norm."]
theorem norm_mul_le' (a b : E) : ‖a * b‖ ≤ ‖a‖ + ‖b‖ := by
  /-
    E : Type u_5
    inst✝ : SeminormedGroup E
    a b : E
    ⊢ LE.le (Norm.norm (HMul.hMul a b)) (HAdd.hAdd (Norm.norm a) (Norm.norm b))
  -/
  simpa [dist_eq_norm_div] using dist_triangle a 1 b⁻¹
  /-
    🎉 no goals
  -/


/-- **Triangle inequality** for the norm. -/
@[to_additive norm_add_le_of_le "**Triangle inequality** for the norm."]
theorem norm_mul_le_of_le' (h₁ : ‖a₁‖ ≤ r₁) (h₂ : ‖a₂‖ ≤ r₂) : ‖a₁ * a₂‖ ≤ r₁ + r₂ :=
  (norm_mul_le' a₁ a₂).trans <| add_le_add h₁ h₂


/-- **Triangle inequality** for the norm. -/
@[to_additive norm_add₃_le "**Triangle inequality** for the norm."]
lemma norm_mul₃_le' : ‖a * b * c‖ ≤ ‖a‖ + ‖b‖ + ‖c‖ := norm_mul_le_of_le' (norm_mul_le' _ _) le_rfl


@[to_additive]
lemma norm_div_le_norm_div_add_norm_div (a b c : E) : ‖a / c‖ ≤ ‖a / b‖ + ‖b / c‖ := by
  /-
    E : Type u_5
    inst✝ : SeminormedGroup E
    a b c : E
    ⊢ LE.le (Norm.norm (HDiv.hDiv a c)) (HAdd.hAdd (Norm.norm (HDiv.hDiv a b)) (No …
  -/
  simpa only [dist_eq_norm_div] using dist_triangle a b c
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp) norm_nonneg]
theorem norm_nonneg' (a : E) : 0 ≤ ‖a‖ := by
  /-
    E : Type u_5
    inst✝ : SeminormedGroup E
    a : E
    ⊢ LE.le 0 (Norm.norm a)
  -/
  rw [← dist_one_right]
  /-
    E : Type u_5
    inst✝ : SeminormedGroup E
    a : E
    ⊢ LE.le 0 (Dist.dist a 1)
  -/
  exact dist_nonneg
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp) abs_norm]
theorem abs_norm' (z : E) : |‖z‖| = ‖z‖ := abs_of_nonneg <| norm_nonneg' _


@[to_additive (attr := simp) norm_zero]
                                        /-
                                          E : Type u_5
                                          inst✝ : SeminormedGroup E
                                          ⊢ Eq (Norm.norm 1) 0
                                        -/
theorem norm_one' : ‖(1 : E)‖ = 0 := by rw [← dist_one_right, dist_self]
                                        /-
                                          🎉 no goals
                                        -/


@[to_additive]
theorem ne_one_of_norm_ne_zero : ‖a‖ ≠ 0 → a ≠ 1 :=
  mt <| by
    /-
      E : Type u_5
      inst✝ : SeminormedGroup E
      a : E
      ⊢ Eq a 1 → Eq (Norm.norm a) 0
    -/
    rintro rfl
    /-
      E : Type u_5
      inst✝ : SeminormedGroup E
      ⊢ Eq (Norm.norm 1) 0
    -/
    exact norm_one'
    /-
      🎉 no goals
    -/


@[to_additive (attr := nontriviality) norm_of_subsingleton]
theorem norm_of_subsingleton' [Subsingleton E] (a : E) : ‖a‖ = 0 := by
  /-
    E : Type u_5
    inst✝¹ : SeminormedGroup E
    inst✝ : Subsingleton E
    a : E
    ⊢ Eq (Norm.norm a) 0
  -/
  rw [Subsingleton.elim a 1, norm_one']
  /-
    🎉 no goals
  -/


@[to_additive zero_lt_one_add_norm_sq]
theorem zero_lt_one_add_norm_sq' (x : E) : 0 < 1 + ‖x‖ ^ 2 := by
  /-
    E : Type u_5
    inst✝ : SeminormedGroup E
    x : E
    ⊢ LT.lt 0 (HAdd.hAdd 1 (HPow.hPow (Norm.norm x) 2))
  -/
  positivity
  /-
    🎉 no goals
  -/


@[to_additive]
theorem norm_div_le (a b : E) : ‖a / b‖ ≤ ‖a‖ + ‖b‖ := by
  /-
    E : Type u_5
    inst✝ : SeminormedGroup E
    a b : E
    ⊢ LE.le (Norm.norm (HDiv.hDiv a b)) (HAdd.hAdd (Norm.norm a) (Norm.norm b))
  -/
  simpa [dist_eq_norm_div] using dist_triangle a 1 b
  /-
    🎉 no goals
  -/


@[to_additive]
theorem norm_div_le_of_le {r₁ r₂ : ℝ} (H₁ : ‖a₁‖ ≤ r₁) (H₂ : ‖a₂‖ ≤ r₂) : ‖a₁ / a₂‖ ≤ r₁ + r₂ :=
  (norm_div_le a₁ a₂).trans <| add_le_add H₁ H₂


@[to_additive dist_le_norm_add_norm]
theorem dist_le_norm_add_norm' (a b : E) : dist a b ≤ ‖a‖ + ‖b‖ := by
  /-
    E : Type u_5
    inst✝ : SeminormedGroup E
    a b : E
    ⊢ LE.le (Dist.dist a b) (HAdd.hAdd (Norm.norm a) (Norm.norm b))
  -/
  rw [dist_eq_norm_div]
  /-
    E : Type u_5
    inst✝ : SeminormedGroup E
    a b : E
    ⊢ LE.le (Norm.norm (HDiv.hDiv a b)) (HAdd.hAdd (Norm.norm a) (Norm.norm b))
  -/
  apply norm_div_le
  /-
    🎉 no goals
  -/


@[to_additive abs_norm_sub_norm_le]
theorem abs_norm_sub_norm_le' (a b : E) : |‖a‖ - ‖b‖| ≤ ‖a / b‖ := by
  /-
    E : Type u_5
    inst✝ : SeminormedGroup E
    a b : E
    ⊢ LE.le (abs (HSub.hSub (Norm.norm a) (Norm.norm b))) (Norm.norm (HDiv.hDiv a  …
  -/
  simpa [dist_eq_norm_div] using abs_dist_sub_le a b 1
  /-
    🎉 no goals
  -/


@[to_additive norm_sub_norm_le]
theorem norm_sub_norm_le' (a b : E) : ‖a‖ - ‖b‖ ≤ ‖a / b‖ :=
  (le_abs_self _).trans (abs_norm_sub_norm_le' a b)


@[to_additive dist_norm_norm_le]
theorem dist_norm_norm_le' (a b : E) : dist ‖a‖ ‖b‖ ≤ ‖a / b‖ :=
  abs_norm_sub_norm_le' a b


@[to_additive]
theorem norm_le_norm_add_norm_div' (u v : E) : ‖u‖ ≤ ‖v‖ + ‖u / v‖ := by
  /-
    E : Type u_5
    inst✝ : SeminormedGroup E
    u v : E
    ⊢ LE.le (Norm.norm u) (HAdd.hAdd (Norm.norm v) (Norm.norm (HDiv.hDiv u v)))
  -/
  rw [add_comm]
  /-
    E : Type u_5
    inst✝ : SeminormedGroup E
    u v : E
    ⊢ LE.le (Norm.norm u) (HAdd.hAdd (Norm.norm (HDiv.hDiv u v)) (Norm.norm v))
  -/
  refine (norm_mul_le' _ _).trans_eq' ?_
  /-
    E : Type u_5
    inst✝ : SeminormedGroup E
    u v : E
    ⊢ Eq (Norm.norm u) (Norm.norm (HMul.hMul (HDiv.hDiv u v) v))
  -/
  rw [div_mul_cancel]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem norm_le_norm_add_norm_div (u v : E) : ‖v‖ ≤ ‖u‖ + ‖u / v‖ := by
  /-
    E : Type u_5
    inst✝ : SeminormedGroup E
    u v : E
    ⊢ LE.le (Norm.norm v) (HAdd.hAdd (Norm.norm u) (Norm.norm (HDiv.hDiv u v)))
  -/
  rw [norm_div_rev]
  /-
    E : Type u_5
    inst✝ : SeminormedGroup E
    u v : E
    ⊢ LE.le (Norm.norm v) (HAdd.hAdd (Norm.norm u) (Norm.norm (HDiv.hDiv v u)))
  -/
  exact norm_le_norm_add_norm_div' v u
  /-
    🎉 no goals
  -/


alias norm_le_insert' := norm_le_norm_add_norm_sub'

alias norm_le_insert := norm_le_norm_add_norm_sub


@[to_additive]
theorem norm_le_mul_norm_add (u v : E) : ‖u‖ ≤ ‖u * v‖ + ‖v‖ :=
  calc
                            /-
                              E : Type u_5
                              inst✝ : SeminormedGroup E
                              u v : E
                              ⊢ Eq (Norm.norm u) (Norm.norm (HDiv.hDiv (HMul.hMul u v) v))
                            -/
    ‖u‖ = ‖u * v / v‖ := by rw [mul_div_cancel_right]
                            /-
                              🎉 no goals
                            -/
    _ ≤ ‖u * v‖ + ‖v‖ := norm_div_le _ _


/-- An analogue of `norm_le_mul_norm_add` for the multiplication from the left. -/
@[to_additive "An analogue of `norm_le_add_norm_add` for the addition from the left."]
theorem norm_le_mul_norm_add' (u v : E) : ‖v‖ ≤ ‖u * v‖ + ‖u‖ :=
  calc
                                /-
                                  E : Type u_5
                                  inst✝ : SeminormedGroup E
                                  u v : E
                                  ⊢ Eq (Norm.norm v) (Norm.norm (HMul.hMul (Inv.inv u) (HMul.hMul u v)))
                                -/
    ‖v‖ = ‖u⁻¹ * (u * v)‖ := by rw [← mul_assoc, inv_mul_cancel, one_mul]
                                /-
                                  🎉 no goals
                                -/
    _ ≤ ‖u⁻¹‖ + ‖u * v‖ := norm_mul_le' u⁻¹ (u * v)
                            /-
                              E : Type u_5
                              inst✝ : SeminormedGroup E
                              u v : E
                              ⊢ Eq (HAdd.hAdd (Norm.norm (Inv.inv u)) (Norm.norm (HMul.hMul u v))) (HAdd.hAd …
                            -/
    _ = ‖u * v‖ + ‖u‖ := by rw [norm_inv', add_comm]
                            /-
                              🎉 no goals
                            -/


@[to_additive]
lemma norm_mul_eq_norm_right {x : E} (y : E) (h : ‖x‖ = 0) : ‖x * y‖ = ‖y‖ := by
  /-
    E : Type u_5
    inst✝ : SeminormedGroup E
    x y : E
    h : Eq (Norm.norm x) 0
    ⊢ Eq (Norm.norm (HMul.hMul x y)) (Norm.norm y)
  -/
  apply le_antisymm ?_ ?_
    /-
      E : Type u_5
      inst✝ : SeminormedGroup E
      x y : E
      h : Eq (Norm.norm x) 0
      ⊢ LE.le (Norm.norm (HMul.hMul x y)) (Norm.norm y)
    -/
  · simpa [h] using norm_mul_le' x y
    /-
      🎉 no goals
    -/
    /-
      E : Type u_5
      inst✝ : SeminormedGroup E
      x y : E
      h : Eq (Norm.norm x) 0
      ⊢ LE.le (Norm.norm y) (Norm.norm (HMul.hMul x y))
    -/
  · simpa [h] using norm_le_mul_norm_add' x y
    /-
      🎉 no goals
    -/


@[to_additive]
lemma norm_mul_eq_norm_left (x : E) {y : E} (h : ‖y‖ = 0) : ‖x * y‖ = ‖x‖ := by
  /-
    E : Type u_5
    inst✝ : SeminormedGroup E
    x y : E
    h : Eq (Norm.norm y) 0
    ⊢ Eq (Norm.norm (HMul.hMul x y)) (Norm.norm x)
  -/
  apply le_antisymm ?_ ?_
    /-
      E : Type u_5
      inst✝ : SeminormedGroup E
      x y : E
      h : Eq (Norm.norm y) 0
      ⊢ LE.le (Norm.norm (HMul.hMul x y)) (Norm.norm x)
    -/
  · simpa [h] using norm_mul_le' x y
    /-
      🎉 no goals
    -/
    /-
      E : Type u_5
      inst✝ : SeminormedGroup E
      x y : E
      h : Eq (Norm.norm y) 0
      ⊢ LE.le (Norm.norm x) (Norm.norm (HMul.hMul x y))
    -/
  · simpa [h] using norm_le_mul_norm_add x y
    /-
      🎉 no goals
    -/


@[to_additive]
lemma norm_div_eq_norm_right {x : E} (y : E) (h : ‖x‖ = 0) : ‖x / y‖ = ‖y‖ := by
  /-
    E : Type u_5
    inst✝ : SeminormedGroup E
    x y : E
    h : Eq (Norm.norm x) 0
    ⊢ Eq (Norm.norm (HDiv.hDiv x y)) (Norm.norm y)
  -/
  apply le_antisymm ?_ ?_
    /-
      E : Type u_5
      inst✝ : SeminormedGroup E
      x y : E
      h : Eq (Norm.norm x) 0
      ⊢ LE.le (Norm.norm (HDiv.hDiv x y)) (Norm.norm y)
    -/
  · simpa [h] using norm_div_le x y
    /-
      🎉 no goals
    -/
    /-
      E : Type u_5
      inst✝ : SeminormedGroup E
      x y : E
      h : Eq (Norm.norm x) 0
      ⊢ LE.le (Norm.norm y) (Norm.norm (HDiv.hDiv x y))
    -/
  · simpa [h, norm_div_rev x y] using norm_sub_norm_le' y x
    /-
      🎉 no goals
    -/


@[to_additive]
lemma norm_div_eq_norm_left (x : E) {y : E} (h : ‖y‖ = 0) : ‖x / y‖ = ‖x‖ := by
  /-
    E : Type u_5
    inst✝ : SeminormedGroup E
    x y : E
    h : Eq (Norm.norm y) 0
    ⊢ Eq (Norm.norm (HDiv.hDiv x y)) (Norm.norm x)
  -/
  apply le_antisymm ?_ ?_
    /-
      E : Type u_5
      inst✝ : SeminormedGroup E
      x y : E
      h : Eq (Norm.norm y) 0
      ⊢ LE.le (Norm.norm (HDiv.hDiv x y)) (Norm.norm x)
    -/
  · simpa [h] using norm_div_le x y
    /-
      🎉 no goals
    -/
    /-
      E : Type u_5
      inst✝ : SeminormedGroup E
      x y : E
      h : Eq (Norm.norm y) 0
      ⊢ LE.le (Norm.norm x) (Norm.norm (HDiv.hDiv x y))
    -/
  · simpa [h] using norm_sub_norm_le' x y
    /-
      🎉 no goals
    -/


@[to_additive ball_eq]
theorem ball_eq' (y : E) (ε : ℝ) : ball y ε = { x | ‖x / y‖ < ε } :=
                      /-
                        E : Type u_5
                        inst✝ : SeminormedGroup E
                        y : E
                        ε : Real
                        a : E
                        ⊢ Iff (Membership.mem (Metric.ball y ε) a) (Membership.mem (setOf fun x => LT. …
                      -/
  Set.ext fun a => by simp [dist_eq_norm_div]
                      /-
                        🎉 no goals
                      -/


@[to_additive]
theorem ball_one_eq (r : ℝ) : ball (1 : E) r = { x | ‖x‖ < r } :=
                      /-
                        E : Type u_5
                        inst✝ : SeminormedGroup E
                        r : Real
                        a : E
                        ⊢ Iff (Membership.mem (Metric.ball 1 r) a) (Membership.mem (setOf fun x => LT. …
                      -/
  Set.ext fun a => by simp
                      /-
                        🎉 no goals
                      -/


@[to_additive mem_ball_iff_norm]
                                                               /-
                                                                 E : Type u_5
                                                                 inst✝ : SeminormedGroup E
                                                                 a b : E
                                                                 r : Real
                                                                 ⊢ Iff (Membership.mem (Metric.ball a r) b) (LT.lt (Norm.norm (HDiv.hDiv b a)) r)
                                                               -/
theorem mem_ball_iff_norm'' : b ∈ ball a r ↔ ‖b / a‖ < r := by rw [mem_ball, dist_eq_norm_div]
                                                               /-
                                                                 🎉 no goals
                                                               -/


@[to_additive mem_ball_iff_norm']
                                                                /-
                                                                  E : Type u_5
                                                                  inst✝ : SeminormedGroup E
                                                                  a b : E
                                                                  r : Real
                                                                  ⊢ Iff (Membership.mem (Metric.ball a r) b) (LT.lt (Norm.norm (HDiv.hDiv a b)) r)
                                                                -/
theorem mem_ball_iff_norm''' : b ∈ ball a r ↔ ‖a / b‖ < r := by rw [mem_ball', dist_eq_norm_div]
                                                                /-
                                                                  🎉 no goals
                                                                -/


@[to_additive]
                                                              /-
                                                                E : Type u_5
                                                                inst✝ : SeminormedGroup E
                                                                a : E
                                                                r : Real
                                                                ⊢ Iff (Membership.mem (Metric.ball 1 r) a) (LT.lt (Norm.norm a) r)
                                                              -/
theorem mem_ball_one_iff : a ∈ ball (1 : E) r ↔ ‖a‖ < r := by rw [mem_ball, dist_one_right]
                                                              /-
                                                                🎉 no goals
                                                              -/


@[to_additive mem_closedBall_iff_norm]
theorem mem_closedBall_iff_norm'' : b ∈ closedBall a r ↔ ‖b / a‖ ≤ r := by
  /-
    E : Type u_5
    inst✝ : SeminormedGroup E
    a b : E
    r : Real
    ⊢ Iff (Membership.mem (Metric.closedBall a r) b) (LE.le (Norm.norm (HDiv.hDiv  …
  -/
  rw [mem_closedBall, dist_eq_norm_div]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mem_closedBall_one_iff : a ∈ closedBall (1 : E) r ↔ ‖a‖ ≤ r := by
  /-
    E : Type u_5
    inst✝ : SeminormedGroup E
    a : E
    r : Real
    ⊢ Iff (Membership.mem (Metric.closedBall 1 r) a) (LE.le (Norm.norm a) r)
  -/
  rw [mem_closedBall, dist_one_right]
  /-
    🎉 no goals
  -/


@[to_additive mem_closedBall_iff_norm']
theorem mem_closedBall_iff_norm''' : b ∈ closedBall a r ↔ ‖a / b‖ ≤ r := by
  /-
    E : Type u_5
    inst✝ : SeminormedGroup E
    a b : E
    r : Real
    ⊢ Iff (Membership.mem (Metric.closedBall a r) b) (LE.le (Norm.norm (HDiv.hDiv  …
  -/
  rw [mem_closedBall', dist_eq_norm_div]
  /-
    🎉 no goals
  -/


@[to_additive norm_le_of_mem_closedBall]
theorem norm_le_of_mem_closedBall' (h : b ∈ closedBall a r) : ‖b‖ ≤ ‖a‖ + r :=
                                                                /-
                                                                  E : Type u_5
                                                                  inst✝ : SeminormedGroup E
                                                                  a b : E
                                                                  r : Real
                                                                  h : Membership.mem (Metric.closedBall a r) b
                                                                  ⊢ LE.le (Norm.norm (HDiv.hDiv b a)) r
                                                                -/
  (norm_le_norm_add_norm_div' _ _).trans <| add_le_add_left (by rwa [← dist_eq_norm_div]) _
                                                                /-
                                                                  🎉 no goals
                                                                -/


@[to_additive norm_le_norm_add_const_of_dist_le]
theorem norm_le_norm_add_const_of_dist_le' : dist a b ≤ r → ‖a‖ ≤ ‖b‖ + r :=
  norm_le_of_mem_closedBall'


@[to_additive norm_lt_of_mem_ball]
theorem norm_lt_of_mem_ball' (h : b ∈ ball a r) : ‖b‖ < ‖a‖ + r :=
                                                                   /-
                                                                     E : Type u_5
                                                                     inst✝ : SeminormedGroup E
                                                                     a b : E
                                                                     r : Real
                                                                     h : Membership.mem (Metric.ball a r) b
                                                                     ⊢ LT.lt (Norm.norm (HDiv.hDiv b a)) r
                                                                   -/
  (norm_le_norm_add_norm_div' _ _).trans_lt <| add_lt_add_left (by rwa [← dist_eq_norm_div]) _
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[to_additive]
theorem norm_div_sub_norm_div_le_norm_div (u v w : E) : ‖u / w‖ - ‖v / w‖ ≤ ‖u / v‖ := by
  /-
    E : Type u_5
    inst✝ : SeminormedGroup E
    u v w : E
    ⊢ LE.le (HSub.hSub (Norm.norm (HDiv.hDiv u w)) (Norm.norm (HDiv.hDiv v w))) (N …
  -/
  simpa only [div_div_div_cancel_right] using norm_sub_norm_le' (u / w) (v / w)
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp 1001) mem_sphere_iff_norm]
-- Porting note: increase priority so the left-hand side doesn't reduce
                                                                  /-
                                                                    E : Type u_5
                                                                    inst✝ : SeminormedGroup E
                                                                    a b : E
                                                                    r : Real
                                                                    ⊢ Iff (Membership.mem (Metric.sphere a r) b) (Eq (Norm.norm (HDiv.hDiv b a)) r)
                                                                  -/
theorem mem_sphere_iff_norm' : b ∈ sphere a r ↔ ‖b / a‖ = r := by simp [dist_eq_norm_div]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


@[to_additive] -- `simp` can prove this
                                                                       /-
                                                                         E : Type u_5
                                                                         inst✝ : SeminormedGroup E
                                                                         a : E
                                                                         r : Real
                                                                         ⊢ Iff (Membership.mem (Metric.sphere 1 r) a) (Eq (Norm.norm a) r)
                                                                       -/
theorem mem_sphere_one_iff_norm : a ∈ sphere (1 : E) r ↔ ‖a‖ = r := by simp [dist_eq_norm_div]
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


@[to_additive (attr := simp) norm_eq_of_mem_sphere]
theorem norm_eq_of_mem_sphere' (x : sphere (1 : E) r) : ‖(x : E)‖ = r :=
  mem_sphere_one_iff_norm.mp x.2


@[to_additive]
theorem ne_one_of_mem_sphere (hr : r ≠ 0) (x : sphere (1 : E) r) : (x : E) ≠ 1 :=
                               /-
                                 E : Type u_5
                                 inst✝ : SeminormedGroup E
                                 r : Real
                                 hr : Ne r 0
                                 x : ↑(Metric.sphere 1 r)
                                 ⊢ Ne (Norm.norm ↑x) 0
                               -/
  ne_one_of_norm_ne_zero <| by rwa [norm_eq_of_mem_sphere' x]
                               /-
                                 🎉 no goals
                               -/


@[to_additive ne_zero_of_mem_unit_sphere]
theorem ne_one_of_mem_unit_sphere (x : sphere (1 : E) 1) : (x : E) ≠ 1 :=
  ne_one_of_mem_sphere one_ne_zero _


/-- The norm of a seminormed group as a group seminorm. -/
@[to_additive "The norm of a seminormed group as an additive group seminorm."]
def normGroupSeminorm : GroupSeminorm E :=
  ⟨norm, norm_one', norm_mul_le', norm_inv'⟩


@[to_additive (attr := simp)]
theorem coe_normGroupSeminorm : ⇑(normGroupSeminorm E) = norm :=
  rfl


@[to_additive]
theorem NormedCommGroup.tendsto_nhds_one {f : α → E} {l : Filter α} :
    Tendsto f l (𝓝 1) ↔ ∀ ε > 0, ∀ᶠ x in l, ‖f x‖ < ε :=
                                  /-
                                    α : Type u_2
                                    E : Type u_5
                                    inst✝ : SeminormedGroup E
                                    f : α → E
                                    l : Filter α
                                    ⊢ Iff (∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun x => LT.lt (Dist.dist  …
                                  -/
  Metric.tendsto_nhds.trans <| by simp only [dist_one_right]
                                  /-
                                    🎉 no goals
                                  -/


@[to_additive]
theorem NormedCommGroup.tendsto_nhds_nhds {f : E → F} {x : E} {y : F} :
    Tendsto f (𝓝 x) (𝓝 y) ↔ ∀ ε > 0, ∃ δ > 0, ∀ x', ‖x' / x‖ < δ → ‖f x' / y‖ < ε := by
  /-
    E : Type u_5
    F : Type u_6
    inst✝¹ : SeminormedGroup E
    inst✝ : SeminormedGroup F
    f : E → F
    x : E
    y : F
    ⊢ Iff (Filter.Tendsto f (nhds x) (nhds y)) (∀ (ε : Real), GT.gt ε 0 → Exists f …
  -/
  simp_rw [Metric.tendsto_nhds_nhds, dist_eq_norm_div]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem NormedCommGroup.nhds_basis_norm_lt (x : E) :
    (𝓝 x).HasBasis (fun ε : ℝ => 0 < ε) fun ε => { y | ‖y / x‖ < ε } := by
  /-
    E : Type u_5
    inst✝ : SeminormedGroup E
    x : E
    ⊢ (nhds x).HasBasis (fun ε => LT.lt 0 ε) fun ε => setOf fun y => LT.lt (Norm.n …
  -/
  simp_rw [← ball_eq']
  /-
    E : Type u_5
    inst✝ : SeminormedGroup E
    x : E
    ⊢ (nhds x).HasBasis (fun ε => LT.lt 0 ε) fun ε => Metric.ball x ε
  -/
  exact Metric.nhds_basis_ball
  /-
    🎉 no goals
  -/


@[to_additive]
theorem NormedCommGroup.nhds_one_basis_norm_lt :
    (𝓝 (1 : E)).HasBasis (fun ε : ℝ => 0 < ε) fun ε => { y | ‖y‖ < ε } := by
  /-
    E : Type u_5
    inst✝ : SeminormedGroup E
    ⊢ (nhds 1).HasBasis (fun ε => LT.lt 0 ε) fun ε => setOf fun y => LT.lt (Norm.n …
  -/
  convert NormedCommGroup.nhds_basis_norm_lt (1 : E)
  /-
    case h.e'_5.h.h.e'_2.h.h.e'_3.h.e'_3
    E : Type u_5
    inst✝ : SeminormedGroup E
    x✝¹ : Real
    x✝ : E
    ⊢ Eq x✝ (HDiv.hDiv x✝ 1)
  -/
  simp
  /-
    🎉 no goals
  -/


@[to_additive]
theorem NormedCommGroup.uniformity_basis_dist :
    (𝓤 E).HasBasis (fun ε : ℝ => 0 < ε) fun ε => { p : E × E | ‖p.fst / p.snd‖ < ε } := by
  /-
    E : Type u_5
    inst✝ : SeminormedGroup E
    ⊢ (uniformity E).HasBasis (fun ε => LT.lt 0 ε) fun ε => setOf fun p => LT.lt ( …
  -/
  convert Metric.uniformity_basis_dist (α := E) using 1
  /-
    case h.e'_5
    E : Type u_5
    inst✝ : SeminormedGroup E
    ⊢ Eq (fun ε => setOf fun p => LT.lt (Norm.norm (HDiv.hDiv p.1 p.2)) ε) fun ε = …
  -/
  simp [dist_eq_norm_div]
  /-
    🎉 no goals
  -/


@[to_additive]
instance (priority := 100) SeminormedGroup.toNNNorm : NNNorm E :=
  ⟨fun a => ⟨‖a‖, norm_nonneg' a⟩⟩


@[to_additive (attr := simp, norm_cast) coe_nnnorm]
theorem coe_nnnorm' (a : E) : (‖a‖₊ : ℝ) = ‖a‖ := rfl


@[to_additive (attr := simp) coe_comp_nnnorm]
theorem coe_comp_nnnorm' : (toReal : ℝ≥0 → ℝ) ∘ (nnnorm : E → ℝ≥0) = norm :=
  rfl


@[to_additive norm_toNNReal]
theorem norm_toNNReal' : ‖a‖.toNNReal = ‖a‖₊ :=
  @Real.toNNReal_coe ‖a‖₊


@[to_additive]
theorem nndist_eq_nnnorm_div (a b : E) : nndist a b = ‖a / b‖₊ :=
  NNReal.eq <| dist_eq_norm_div _ _


alias nndist_eq_nnnorm := nndist_eq_nnnorm_sub


@[to_additive (attr := simp)]
                                                           /-
                                                             E : Type u_5
                                                             inst✝ : SeminormedGroup E
                                                             a : E
                                                             ⊢ Eq (NNDist.nndist a 1) (NNNorm.nnnorm a)
                                                           -/
theorem nndist_one_right (a : E) : nndist a 1 = ‖a‖₊ := by simp [nndist_eq_nnnorm_div]
                                                           /-
                                                             🎉 no goals
                                                           -/


@[to_additive (attr := simp)]
theorem edist_one_right (a : E) : edist a 1 = ‖a‖₊ := by
  /-
    E : Type u_5
    inst✝ : SeminormedGroup E
    a : E
    ⊢ Eq (EDist.edist a 1) ↑(NNNorm.nnnorm a)
  -/
  rw [edist_nndist, nndist_one_right]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp) nnnorm_zero]
theorem nnnorm_one' : ‖(1 : E)‖₊ = 0 := NNReal.eq norm_one'


@[to_additive]
theorem ne_one_of_nnnorm_ne_zero {a : E} : ‖a‖₊ ≠ 0 → a ≠ 1 :=
  mt <| by
    /-
      E : Type u_5
      inst✝ : SeminormedGroup E
      a : E
      ⊢ Eq a 1 → Eq (NNNorm.nnnorm a) 0
    -/
    rintro rfl
    /-
      E : Type u_5
      inst✝ : SeminormedGroup E
      ⊢ Eq (NNNorm.nnnorm 1) 0
    -/
    exact nnnorm_one'
    /-
      🎉 no goals
    -/


@[to_additive nnnorm_add_le]
theorem nnnorm_mul_le' (a b : E) : ‖a * b‖₊ ≤ ‖a‖₊ + ‖b‖₊ :=
  NNReal.coe_le_coe.1 <| norm_mul_le' a b


@[to_additive norm_nsmul_le]
lemma norm_pow_le_mul_norm : ∀ {n : ℕ}, ‖a ^ n‖ ≤ n * ‖a‖
            /-
              E : Type u_5
              inst✝ : SeminormedGroup E
              a : E
              ⊢ LE.le (Norm.norm (HPow.hPow a 0)) (HMul.hMul (↑0) (Norm.norm a))
            -/
  | 0 => by simp
            /-
              🎉 no goals
            -/
                /-
                  E : Type u_5
                  inst✝ : SeminormedGroup E
                  a : E
                  n : Nat
                  ⊢ LE.le (Norm.norm (HPow.hPow a (HAdd.hAdd n 1))) (HMul.hMul (↑(HAdd.hAdd n 1) …
                -/
  | n + 1 => by simpa [pow_succ, add_mul] using norm_mul_le_of_le' norm_pow_le_mul_norm le_rfl
                /-
                  🎉 no goals
                -/


@[to_additive nnnorm_nsmul_le]
lemma nnnorm_pow_le_mul_norm {n : ℕ} : ‖a ^ n‖₊ ≤ n * ‖a‖₊ := by
  /-
    E : Type u_5
    inst✝ : SeminormedGroup E
    a : E
    n : Nat
    ⊢ LE.le (NNNorm.nnnorm (HPow.hPow a n)) (HMul.hMul (↑n) (NNNorm.nnnorm a))
  -/
  simpa only [← NNReal.coe_le_coe, NNReal.coe_mul, NNReal.coe_natCast] using norm_pow_le_mul_norm
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp) nnnorm_neg]
theorem nnnorm_inv' (a : E) : ‖a⁻¹‖₊ = ‖a‖₊ :=
  NNReal.eq <| norm_inv' a


@[to_additive (attr := simp) nnnorm_abs_zsmul]
theorem nnnorm_zpow_abs (a : E) (n : ℤ) : ‖a ^ |n|‖₊ = ‖a ^ n‖₊ :=
  NNReal.eq <| norm_zpow_abs a n


@[to_additive (attr := simp) nnnorm_natAbs_smul]
theorem nnnorm_pow_natAbs (a : E) (n : ℤ) : ‖a ^ n.natAbs‖₊ = ‖a ^ n‖₊ :=
  NNReal.eq <| norm_pow_natAbs a n


@[to_additive nnnorm_isUnit_zsmul]
theorem nnnorm_zpow_isUnit (a : E) {n : ℤ} (hn : IsUnit n) : ‖a ^ n‖₊ = ‖a‖₊ :=
  NNReal.eq <| norm_zpow_isUnit a hn


@[simp]
theorem nnnorm_units_zsmul {E : Type*} [SeminormedAddGroup E] (n : ℤˣ) (a : E) : ‖n • a‖₊ = ‖a‖₊ :=
  NNReal.eq <| norm_isUnit_zsmul a n.isUnit


@[to_additive (attr := simp)]
                                                          /-
                                                            E : Type u_5
                                                            inst✝ : SeminormedGroup E
                                                            a : E
                                                            ⊢ Eq (NNDist.nndist 1 a) (NNNorm.nnnorm a)
                                                          -/
theorem nndist_one_left (a : E) : nndist 1 a = ‖a‖₊ := by simp [nndist_eq_nnnorm_div]
                                                          /-
                                                            🎉 no goals
                                                          -/


@[to_additive (attr := simp)]
theorem edist_one_left (a : E) : edist 1 a = ‖a‖₊ := by
  /-
    E : Type u_5
    inst✝ : SeminormedGroup E
    a : E
    ⊢ Eq (EDist.edist 1 a) ↑(NNNorm.nnnorm a)
  -/
  rw [edist_nndist, nndist_one_left]
  /-
    🎉 no goals
  -/


open scoped symmDiff in
@[to_additive]
theorem nndist_mulIndicator (s t : Set α) (f : α → E) (x : α) :
    nndist (s.mulIndicator f x) (t.mulIndicator f x) = ‖(s ∆ t).mulIndicator f x‖₊ :=
  NNReal.eq <| dist_mulIndicator s t f x


@[to_additive]
theorem nnnorm_div_le (a b : E) : ‖a / b‖₊ ≤ ‖a‖₊ + ‖b‖₊ :=
  NNReal.coe_le_coe.1 <| norm_div_le _ _


@[to_additive nndist_nnnorm_nnnorm_le]
theorem nndist_nnnorm_nnnorm_le' (a b : E) : nndist ‖a‖₊ ‖b‖₊ ≤ ‖a / b‖₊ :=
  NNReal.coe_le_coe.1 <| dist_norm_norm_le' a b


@[to_additive]
theorem nnnorm_le_nnnorm_add_nnnorm_div (a b : E) : ‖b‖₊ ≤ ‖a‖₊ + ‖a / b‖₊ :=
  norm_le_norm_add_norm_div _ _


@[to_additive]
theorem nnnorm_le_nnnorm_add_nnnorm_div' (a b : E) : ‖a‖₊ ≤ ‖b‖₊ + ‖a / b‖₊ :=
  norm_le_norm_add_norm_div' _ _


alias nnnorm_le_insert' := nnnorm_le_nnnorm_add_nnnorm_sub'


alias nnnorm_le_insert := nnnorm_le_nnnorm_add_nnnorm_sub


@[to_additive]
theorem nnnorm_le_mul_nnnorm_add (a b : E) : ‖a‖₊ ≤ ‖a * b‖₊ + ‖b‖₊ :=
  norm_le_mul_norm_add _ _


/-- An analogue of `nnnorm_le_mul_nnnorm_add` for the multiplication from the left. -/
@[to_additive "An analogue of `nnnorm_le_add_nnnorm_add` for the addition from the left."]
theorem nnnorm_le_mul_nnnorm_add' (a b : E) : ‖b‖₊ ≤ ‖a * b‖₊ + ‖a‖₊ :=
  norm_le_mul_norm_add' _ _


@[to_additive]
lemma nnnorm_mul_eq_nnnorm_right {x : E} (y : E) (h : ‖x‖₊ = 0) : ‖x * y‖₊ = ‖y‖₊ :=
  NNReal.eq <| norm_mul_eq_norm_right _ <| congr_arg NNReal.toReal h


@[to_additive]
lemma nnnorm_mul_eq_nnnorm_left (x : E) {y : E} (h : ‖y‖₊ = 0) : ‖x * y‖₊ = ‖x‖₊ :=
  NNReal.eq <| norm_mul_eq_norm_left _ <| congr_arg NNReal.toReal h


@[to_additive]
lemma nnnorm_div_eq_nnnorm_right {x : E} (y : E) (h : ‖x‖₊ = 0) : ‖x / y‖₊ = ‖y‖₊ :=
  NNReal.eq <| norm_div_eq_norm_right _ <| congr_arg NNReal.toReal h


@[to_additive]
lemma nnnorm_div_eq_nnnorm_left (x : E) {y : E} (h : ‖y‖₊ = 0) : ‖x / y‖₊ = ‖x‖₊ :=
  NNReal.eq <| norm_div_eq_norm_left _ <| congr_arg NNReal.toReal h


@[to_additive ofReal_norm_eq_coe_nnnorm]
theorem ofReal_norm_eq_coe_nnnorm' (a : E) : ENNReal.ofReal ‖a‖ = ‖a‖₊ :=
  ENNReal.ofReal_eq_coe_nnreal _


/-- The non negative norm seen as an `ENNReal` and then as a `Real` is equal to the norm. -/
@[to_additive toReal_coe_nnnorm "The non negative norm seen as an `ENNReal` and
then as a `Real` is equal to the norm."]
theorem toReal_coe_nnnorm' (a : E) : (‖a‖₊ : ℝ≥0∞).toReal = ‖a‖ := rfl


@[to_additive]
theorem edist_eq_coe_nnnorm_div (a b : E) : edist a b = ‖a / b‖₊ := by
  /-
    E : Type u_5
    inst✝ : SeminormedGroup E
    a b : E
    ⊢ Eq (EDist.edist a b) ↑(NNNorm.nnnorm (HDiv.hDiv a b))
  -/
  rw [edist_dist, dist_eq_norm_div, ofReal_norm_eq_coe_nnnorm']
  /-
    🎉 no goals
  -/


@[to_additive edist_eq_coe_nnnorm]
theorem edist_eq_coe_nnnorm' (x : E) : edist x 1 = (‖x‖₊ : ℝ≥0∞) := by
  /-
    E : Type u_5
    inst✝ : SeminormedGroup E
    x : E
    ⊢ Eq (EDist.edist x 1) ↑(NNNorm.nnnorm x)
  -/
  rw [edist_eq_coe_nnnorm_div, div_one]
  /-
    🎉 no goals
  -/


open scoped symmDiff in
@[to_additive]
theorem edist_mulIndicator (s t : Set α) (f : α → E) (x : α) :
    edist (s.mulIndicator f x) (t.mulIndicator f x) = ‖(s ∆ t).mulIndicator f x‖₊ := by
  /-
    α : Type u_2
    E : Type u_5
    inst✝ : SeminormedGroup E
    s t : Set α
    f : α → E
    x : α
    ⊢ Eq (EDist.edist (s.mulIndicator f x) (t.mulIndicator f x)) ↑(NNNorm.nnnorm ( …
  -/
  rw [edist_nndist, nndist_mulIndicator]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mem_emetric_ball_one_iff {r : ℝ≥0∞} : a ∈ EMetric.ball (1 : E) r ↔ ↑‖a‖₊ < r := by
  /-
    E : Type u_5
    inst✝ : SeminormedGroup E
    a : E
    r : ENNReal
    ⊢ Iff (Membership.mem (EMetric.ball 1 r) a) (LT.lt (↑(NNNorm.nnnorm a)) r)
  -/
  rw [EMetric.mem_ball, edist_eq_coe_nnnorm']
  /-
    🎉 no goals
  -/


instance {E : Type*} [NNNorm E] : ENorm E where
  enorm := (‖·‖₊ : E → ℝ≥0∞)


lemma enorm_eq_nnnorm {E : Type*} [NNNorm E] {x : E} : ‖x‖ₑ = ‖x‖₊ := rfl


instance : ENorm ℝ≥0∞ where
  enorm x := x


@[simp] lemma enorm_eq_self (x : ℝ≥0∞) : ‖x‖ₑ = x := rfl


@[to_additive]
theorem tendsto_iff_norm_div_tendsto_zero {f : α → E} {a : Filter α} {b : E} :
    Tendsto f a (𝓝 b) ↔ Tendsto (fun e => ‖f e / b‖) a (𝓝 0) := by
  /-
    α : Type u_2
    E : Type u_5
    inst✝ : SeminormedGroup E
    f : α → E
    a : Filter α
    b : E
    ⊢ Iff (Filter.Tendsto f a (nhds b)) (Filter.Tendsto (fun e => Norm.norm (HDiv. …
  -/
  simp only [← dist_eq_norm_div, ← tendsto_iff_dist_tendsto_zero]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem tendsto_one_iff_norm_tendsto_zero {f : α → E} {a : Filter α} :
    Tendsto f a (𝓝 1) ↔ Tendsto (‖f ·‖) a (𝓝 0) :=
                                                /-
                                                  α : Type u_2
                                                  E : Type u_5
                                                  inst✝ : SeminormedGroup E
                                                  f : α → E
                                                  a : Filter α
                                                  ⊢ Iff (Filter.Tendsto (fun e => Norm.norm (HDiv.hDiv (f e) 1)) a (nhds 0)) (Fi …
                                                -/
  tendsto_iff_norm_div_tendsto_zero.trans <| by simp only [div_one]
                                                /-
                                                  🎉 no goals
                                                -/


@[to_additive]
theorem comap_norm_nhds_one : comap norm (𝓝 0) = 𝓝 (1 : E) := by
  /-
    E : Type u_5
    inst✝ : SeminormedGroup E
    ⊢ Eq (Filter.comap Norm.norm (nhds 0)) (nhds 1)
  -/
  simpa only [dist_one_right] using nhds_comap_dist (1 : E)
  /-
    🎉 no goals
  -/


/-- Special case of the sandwich theorem: if the norm of `f` is eventually bounded by a real
function `a` which tends to `0`, then `f` tends to `1` (neutral element of `SeminormedGroup`).
In this pair of lemmas (`squeeze_one_norm'` and `squeeze_one_norm`), following a convention of
similar lemmas in `Topology.MetricSpace.Basic` and `Topology.Algebra.Order`, the `'` version is
phrased using "eventually" and the non-`'` version is phrased absolutely. -/
@[to_additive "Special case of the sandwich theorem: if the norm of `f` is eventually bounded by a
real function `a` which tends to `0`, then `f` tends to `0`. In this pair of lemmas
(`squeeze_zero_norm'` and `squeeze_zero_norm`), following a convention of similar lemmas in
`Topology.MetricSpace.Pseudo.Defs` and `Topology.Algebra.Order`, the `'` version is phrased using
\"eventually\" and the non-`'` version is phrased absolutely."]
theorem squeeze_one_norm' {f : α → E} {a : α → ℝ} {t₀ : Filter α} (h : ∀ᶠ n in t₀, ‖f n‖ ≤ a n)
    (h' : Tendsto a t₀ (𝓝 0)) : Tendsto f t₀ (𝓝 1) :=
  tendsto_one_iff_norm_tendsto_zero.2 <|
    squeeze_zero' (Eventually.of_forall fun _n => norm_nonneg' _) h h'


/-- Special case of the sandwich theorem: if the norm of `f` is bounded by a real function `a` which
tends to `0`, then `f` tends to `1`. -/
@[to_additive "Special case of the sandwich theorem: if the norm of `f` is bounded by a real
function `a` which tends to `0`, then `f` tends to `0`."]
theorem squeeze_one_norm {f : α → E} {a : α → ℝ} {t₀ : Filter α} (h : ∀ n, ‖f n‖ ≤ a n) :
    Tendsto a t₀ (𝓝 0) → Tendsto f t₀ (𝓝 1) :=
  squeeze_one_norm' <| Eventually.of_forall h


@[to_additive]
theorem tendsto_norm_div_self (x : E) : Tendsto (fun a => ‖a / x‖) (𝓝 x) (𝓝 0) := by
  simpa [dist_eq_norm_div] using
    tendsto_id.dist (tendsto_const_nhds : Tendsto (fun _a => (x : E)) (𝓝 x) _)


@[to_additive]
theorem tendsto_norm_div_self_nhdsGE (x : E) : Tendsto (fun a ↦ ‖a / x‖) (𝓝 x) (𝓝[≥] 0) :=
                                                          /-
                                                            E : Type u_5
                                                            inst✝ : SeminormedGroup E
                                                            x : E
                                                            ⊢ Filter.Eventually (fun n => Membership.mem (Set.Ici 0) (Norm.norm (HDiv.hDiv …
                                                          -/
  tendsto_nhdsWithin_iff.mpr ⟨tendsto_norm_div_self x, by simp⟩
                                                          /-
                                                            🎉 no goals
                                                          -/


@[to_additive tendsto_norm]
theorem tendsto_norm' {x : E} : Tendsto (fun a => ‖a‖) (𝓝 x) (𝓝 ‖x‖) := by
  /-
    E : Type u_5
    inst✝ : SeminormedGroup E
    x : E
    ⊢ Filter.Tendsto (fun a => Norm.norm a) (nhds x) (nhds (Norm.norm x))
  -/
  simpa using tendsto_id.dist (tendsto_const_nhds : Tendsto (fun _a => (1 : E)) _ _)
  /-
    🎉 no goals
  -/


/-- See `tendsto_norm_one` for a version with pointed neighborhoods. -/
@[to_additive "See `tendsto_norm_zero` for a version with pointed neighborhoods."]
theorem tendsto_norm_one : Tendsto (fun a : E => ‖a‖) (𝓝 1) (𝓝 0) := by
  /-
    E : Type u_5
    inst✝ : SeminormedGroup E
    ⊢ Filter.Tendsto (fun a => Norm.norm a) (nhds 1) (nhds 0)
  -/
  simpa using tendsto_norm_div_self (1 : E)
  /-
    🎉 no goals
  -/


@[to_additive (attr := continuity) continuous_norm]
theorem continuous_norm' : Continuous fun a : E => ‖a‖ := by
  /-
    E : Type u_5
    inst✝ : SeminormedGroup E
    ⊢ Continuous fun a => Norm.norm a
  -/
  simpa using continuous_id.dist (continuous_const : Continuous fun _a => (1 : E))
  /-
    🎉 no goals
  -/


@[to_additive (attr := continuity) continuous_nnnorm]
theorem continuous_nnnorm' : Continuous fun a : E => ‖a‖₊ :=
  continuous_norm'.subtype_mk _


set_option linter.docPrime false in
@[to_additive Inseparable.norm_eq_norm]
theorem Inseparable.norm_eq_norm' {u v : E} (h : Inseparable u v) : ‖u‖ = ‖v‖ :=
  h.map continuous_norm' |>.eq


set_option linter.docPrime false in
@[to_additive Inseparable.nnnorm_eq_nnnorm]
theorem Inseparable.nnnorm_eq_nnnorm' {u v : E} (h : Inseparable u v) : ‖u‖₊ = ‖v‖₊ :=
  h.map continuous_nnnorm' |>.eq


@[to_additive]
theorem mem_closure_one_iff_norm {x : E} : x ∈ closure ({1} : Set E) ↔ ‖x‖ = 0 := by
  /-
    E : Type u_5
    inst✝ : SeminormedGroup E
    x : E
    ⊢ Iff (Membership.mem (closure (Singleton.singleton 1)) x) (Eq (Norm.norm x) 0)
  -/
  rw [← closedBall_zero', mem_closedBall_one_iff, (norm_nonneg' x).le_iff_eq]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem closure_one_eq : closure ({1} : Set E) = { x | ‖x‖ = 0 } :=
  Set.ext fun _x => mem_closure_one_iff_norm


@[to_additive Filter.Tendsto.norm]
theorem Filter.Tendsto.norm' (h : Tendsto f l (𝓝 a)) : Tendsto (fun x => ‖f x‖) l (𝓝 ‖a‖) :=
  tendsto_norm'.comp h


@[to_additive Filter.Tendsto.nnnorm]
theorem Filter.Tendsto.nnnorm' (h : Tendsto f l (𝓝 a)) : Tendsto (fun x => ‖f x‖₊) l (𝓝 ‖a‖₊) :=
  Tendsto.comp continuous_nnnorm'.continuousAt h


@[to_additive (attr := fun_prop) Continuous.norm]
theorem Continuous.norm' : Continuous f → Continuous fun x => ‖f x‖ :=
  continuous_norm'.comp


@[to_additive (attr := fun_prop) Continuous.nnnorm]
theorem Continuous.nnnorm' : Continuous f → Continuous fun x => ‖f x‖₊ :=
  continuous_nnnorm'.comp


@[to_additive (attr := fun_prop) ContinuousAt.norm]
theorem ContinuousAt.norm' {a : α} (h : ContinuousAt f a) : ContinuousAt (fun x => ‖f x‖) a :=
  Tendsto.norm' h


@[to_additive (attr := fun_prop) ContinuousAt.nnnorm]
theorem ContinuousAt.nnnorm' {a : α} (h : ContinuousAt f a) : ContinuousAt (fun x => ‖f x‖₊) a :=
  Tendsto.nnnorm' h


@[to_additive ContinuousWithinAt.norm]
theorem ContinuousWithinAt.norm' {s : Set α} {a : α} (h : ContinuousWithinAt f s a) :
    ContinuousWithinAt (fun x => ‖f x‖) s a :=
  Tendsto.norm' h


@[to_additive ContinuousWithinAt.nnnorm]
theorem ContinuousWithinAt.nnnorm' {s : Set α} {a : α} (h : ContinuousWithinAt f s a) :
    ContinuousWithinAt (fun x => ‖f x‖₊) s a :=
  Tendsto.nnnorm' h


@[to_additive (attr := fun_prop) ContinuousOn.norm]
theorem ContinuousOn.norm' {s : Set α} (h : ContinuousOn f s) : ContinuousOn (fun x => ‖f x‖) s :=
  fun x hx => (h x hx).norm'


@[to_additive (attr := fun_prop) ContinuousOn.nnnorm]
theorem ContinuousOn.nnnorm' {s : Set α} (h : ContinuousOn f s) :
    ContinuousOn (fun x => ‖f x‖₊) s := fun x hx => (h x hx).nnnorm'


/-- If `‖y‖ → ∞`, then we can assume `y ≠ x` for any fixed `x`. -/
@[to_additive eventually_ne_of_tendsto_norm_atTop "If `‖y‖→∞`, then we can assume `y≠x` for any
fixed `x`"]
theorem eventually_ne_of_tendsto_norm_atTop' {l : Filter α} {f : α → E}
    (h : Tendsto (fun y => ‖f y‖) l atTop) (x : E) : ∀ᶠ y in l, f y ≠ x :=
  (h.eventually_ne_atTop _).mono fun _x => ne_of_apply_ne norm


@[to_additive]
theorem SeminormedCommGroup.mem_closure_iff :
    a ∈ closure s ↔ ∀ ε, 0 < ε → ∃ b ∈ s, ‖a / b‖ < ε := by
  /-
    E : Type u_5
    inst✝ : SeminormedGroup E
    s : Set E
    a : E
    ⊢ Iff (Membership.mem (closure s) a) (∀ (ε : Real), LT.lt 0 ε → Exists fun b = …
  -/
  simp [Metric.mem_closure_iff, dist_eq_norm_div]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem SeminormedGroup.tendstoUniformlyOn_one {f : ι → κ → G} {s : Set κ} {l : Filter ι} :
    TendstoUniformlyOn f 1 l s ↔ ∀ ε > 0, ∀ᶠ i in l, ∀ x ∈ s, ‖f i x‖ < ε := by
  /-
    ι : Type u_3
    κ : Type u_4
    G : Type u_7
    inst✝ : SeminormedGroup G
    f : ι → κ → G
    s : Set κ
    l : Filter ι
    ⊢ Iff (TendstoUniformlyOn f 1 l s) (∀ (ε : Real), GT.gt ε 0 → Filter.Eventuall …
  -/
  simp only [tendstoUniformlyOn_iff, Pi.one_apply, dist_one_left]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem SeminormedGroup.uniformCauchySeqOnFilter_iff_tendstoUniformlyOnFilter_one {f : ι → κ → G}
    {l : Filter ι} {l' : Filter κ} :
    UniformCauchySeqOnFilter f l l' ↔
      TendstoUniformlyOnFilter (fun n : ι × ι => fun z => f n.fst z / f n.snd z) 1 (l ×ˢ l) l' := by
  /-
    ι : Type u_3
    κ : Type u_4
    G : Type u_7
    inst✝ : SeminormedGroup G
    f : ι → κ → G
    l : Filter ι
    l' : Filter κ
    ⊢ Iff (UniformCauchySeqOnFilter f l l') (TendstoUniformlyOnFilter (fun n z =>  …
  -/
  refine ⟨fun hf u hu => ?_, fun hf u hu => ?_⟩
    /-
      case refine_1
      ι : Type u_3
      κ : Type u_4
      G : Type u_7
      inst✝ : SeminormedGroup G
      f : ι → κ → G
      l : Filter ι
      l' : Filter κ
      hf : UniformCauchySeqOnFilter f l l'
      u : Set (Prod G G)
      hu : Membership.mem (uniformity G) u
      ⊢ Filter.Eventually (fun n => Membership.mem u { fst := 1 n.2, snd := (fun n z …
    -/
  · obtain ⟨ε, hε, H⟩ := uniformity_basis_dist.mem_uniformity_iff.mp hu
    refine
      (hf { p : G × G | dist p.fst p.snd < ε } <| dist_mem_uniformity hε).mono fun x hx =>
        H 1 (f x.fst.fst x.snd / f x.fst.snd x.snd) ?_
    /-
      case refine_1.intro.intro
      ι : Type u_3
      κ : Type u_4
      G : Type u_7
      inst✝ : SeminormedGroup G
      f : ι → κ → G
      l : Filter ι
      l' : Filter κ
      hf : UniformCauchySeqOnFilter f l l'
      u : Set (Prod G G)
      hu : Membership.mem (uniformity G) u
      ε : Real
      hε : LT.lt 0 ε
      H : ∀ (a b : G), Membership.mem (setOf fun p => LT.lt (Dist.dist p.1 p.2) ε) { …
      x : Prod (Prod ι ι) κ
      hx : Membership.mem (setOf fun p => LT.lt (Dist.dist p.1 p.2) ε) { fst := f x. …
      ⊢ Membership.mem (setOf fun p => LT.lt (Dist.dist p.1 p.2) ε) { fst := 1, snd  …
    -/
    simpa [dist_eq_norm_div, norm_div_rev] using hx
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      ι : Type u_3
      κ : Type u_4
      G : Type u_7
      inst✝ : SeminormedGroup G
      f : ι → κ → G
      l : Filter ι
      l' : Filter κ
      hf : TendstoUniformlyOnFilter (fun n z => HDiv.hDiv (f n.1 z) (f n.2 z)) 1 (SP …
      u : Set (Prod G G)
      hu : Membership.mem (uniformity G) u
      ⊢ Filter.Eventually (fun m => Membership.mem u { fst := f m.1.1 m.2, snd := f  …
    -/
  · obtain ⟨ε, hε, H⟩ := uniformity_basis_dist.mem_uniformity_iff.mp hu
    refine
      (hf { p : G × G | dist p.fst p.snd < ε } <| dist_mem_uniformity hε).mono fun x hx =>
        H (f x.fst.fst x.snd) (f x.fst.snd x.snd) ?_
    /-
      case refine_2.intro.intro
      ι : Type u_3
      κ : Type u_4
      G : Type u_7
      inst✝ : SeminormedGroup G
      f : ι → κ → G
      l : Filter ι
      l' : Filter κ
      hf : TendstoUniformlyOnFilter (fun n z => HDiv.hDiv (f n.1 z) (f n.2 z)) 1 (SP …
      u : Set (Prod G G)
      hu : Membership.mem (uniformity G) u
      ε : Real
      hε : LT.lt 0 ε
      H : ∀ (a b : G), Membership.mem (setOf fun p => LT.lt (Dist.dist p.1 p.2) ε) { …
      x : Prod (Prod ι ι) κ
      hx : Membership.mem (setOf fun p => LT.lt (Dist.dist p.1 p.2) ε) { fst := 1 x. …
      ⊢ Membership.mem (setOf fun p => LT.lt (Dist.dist p.1 p.2) ε) { fst := f x.1.1 …
    -/
    simpa [dist_eq_norm_div, norm_div_rev] using hx
    /-
      🎉 no goals
    -/


@[to_additive]
theorem SeminormedGroup.uniformCauchySeqOn_iff_tendstoUniformlyOn_one {f : ι → κ → G} {s : Set κ}
    {l : Filter ι} :
    UniformCauchySeqOn f l s ↔
      TendstoUniformlyOn (fun n : ι × ι => fun z => f n.fst z / f n.snd z) 1 (l ×ˢ l) s := by
  rw [tendstoUniformlyOn_iff_tendstoUniformlyOnFilter,
    uniformCauchySeqOn_iff_uniformCauchySeqOnFilter,
    SeminormedGroup.uniformCauchySeqOnFilter_iff_tendstoUniformlyOnFilter_one]


/-- A group homomorphism from a `Group` to a `SeminormedGroup` induces a `SeminormedGroup`
structure on the domain. -/
@[to_additive "A group homomorphism from an `AddGroup` to a
`SeminormedAddGroup` induces a `SeminormedAddGroup` structure on the domain."]
abbrev SeminormedGroup.induced [Group E] [SeminormedGroup F] [MonoidHomClass 𝓕 E F] (f : 𝓕) :
    SeminormedGroup E :=
  { PseudoMetricSpace.induced f toPseudoMetricSpace with
    -- Porting note: needed to add the instance explicitly, and `‹PseudoMetricSpace F›` failed
    norm := fun x => ‖f x‖
                             /-
                               𝓕 : Type u_1
                               α : Type u_2
                               ι : Type u_3
                               κ : Type u_4
                               E : Type u_5
                               F : Type u_6
                               G : Type u_7
                               inst✝³ : FunLike 𝓕 E F
                               inst✝² : Group E
                               inst✝¹ : SeminormedGroup F
                               inst✝ : MonoidHomClass 𝓕 E F
                               f : 𝓕
                               x y : E
                               ⊢ Eq (Dist.dist x y) (Norm.norm (HDiv.hDiv x y))
                             -/
    dist_eq := fun x y => by simp only [map_div, ← dist_eq_norm_div]; rfl }
                                                                      /-
                                                                        🎉 no goals
                                                                      -/

-- See note [reducible non-instances]

/-- A group homomorphism from a `CommGroup` to a `SeminormedGroup` induces a
`SeminormedCommGroup` structure on the domain. -/
@[to_additive "A group homomorphism from an `AddCommGroup` to a
`SeminormedAddGroup` induces a `SeminormedAddCommGroup` structure on the domain."]
abbrev SeminormedCommGroup.induced
    [CommGroup E] [SeminormedGroup F] [MonoidHomClass 𝓕 E F] (f : 𝓕) :
    SeminormedCommGroup E :=
  { SeminormedGroup.induced E F f with
    mul_comm := mul_comm }

-- See note [reducible non-instances].

/-- An injective group homomorphism from a `Group` to a `NormedGroup` induces a `NormedGroup`
structure on the domain. -/
@[to_additive "An injective group homomorphism from an `AddGroup` to a
`NormedAddGroup` induces a `NormedAddGroup` structure on the domain."]
abbrev NormedGroup.induced
    [Group E] [NormedGroup F] [MonoidHomClass 𝓕 E F] (f : 𝓕) (h : Injective f) :
    NormedGroup E :=
  { SeminormedGroup.induced E F f, MetricSpace.induced f h _ with }

-- See note [reducible non-instances].

/-- An injective group homomorphism from a `CommGroup` to a `NormedGroup` induces a
`NormedCommGroup` structure on the domain. -/
@[to_additive "An injective group homomorphism from a `CommGroup` to a
`NormedCommGroup` induces a `NormedCommGroup` structure on the domain."]
abbrev NormedCommGroup.induced [CommGroup E] [NormedGroup F] [MonoidHomClass 𝓕 E F] (f : 𝓕)
    (h : Injective f) : NormedCommGroup E :=
  { SeminormedGroup.induced E F f, MetricSpace.induced f h _ with
    mul_comm := mul_comm }


@[to_additive]
theorem dist_inv (x y : E) : dist x⁻¹ y = dist x y⁻¹ := by
  /-
    E : Type u_5
    inst✝ : SeminormedCommGroup E
    x y : E
    ⊢ Eq (Dist.dist (Inv.inv x) y) (Dist.dist x (Inv.inv y))
  -/
  simp_rw [dist_eq_norm_div, ← norm_inv' (x⁻¹ / y), inv_div, div_inv_eq_mul, mul_comm]
  /-
    🎉 no goals
  -/


theorem norm_multiset_sum_le {E} [SeminormedAddCommGroup E] (m : Multiset E) :
    ‖m.sum‖ ≤ (m.map fun x => ‖x‖).sum :=
  m.le_sum_of_subadditive norm norm_zero norm_add_le


@[to_additive existing]
theorem norm_multiset_prod_le (m : Multiset E) : ‖m.prod‖ ≤ (m.map fun x => ‖x‖).sum := by
  /-
    E : Type u_5
    inst✝ : SeminormedCommGroup E
    m : Multiset E
    ⊢ LE.le (Norm.norm m.prod) (Multiset.map (fun x => Norm.norm x) m).sum
  -/
  rw [← Multiplicative.ofAdd_le, ofAdd_multiset_prod, Multiset.map_map]
  /-
    E : Type u_5
    inst✝ : SeminormedCommGroup E
    m : Multiset E
    ⊢ LE.le (Multiplicative.ofAdd (Norm.norm m.prod)) (Multiset.map (Function.comp …
  -/
  refine Multiset.le_prod_of_submultiplicative (Multiplicative.ofAdd ∘ norm) ?_ (fun x y => ?_) _
    /-
      case refine_1
      E : Type u_5
      inst✝ : SeminormedCommGroup E
      m : Multiset E
      ⊢ Eq (Function.comp (⇑Multiplicative.ofAdd) Norm.norm 1) 1
    -/
  · simp only [comp_apply, norm_one', ofAdd_zero]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      E : Type u_5
      inst✝ : SeminormedCommGroup E
      m : Multiset E
      x y : E
      ⊢ LE.le (Function.comp (⇑Multiplicative.ofAdd) Norm.norm (HMul.hMul x y)) (HMu …
    -/
  · exact norm_mul_le' x y
    /-
      🎉 no goals
    -/

-- Porting note: had to add `ι` here because otherwise the universe order gets switched compared to
-- `norm_prod_le` below

@[bound]
theorem norm_sum_le {ι E} [SeminormedAddCommGroup E] (s : Finset ι) (f : ι → E) :
    ‖∑ i ∈ s, f i‖ ≤ ∑ i ∈ s, ‖f i‖ :=
  s.le_sum_of_subadditive norm norm_zero norm_add_le f


@[to_additive existing]
theorem norm_prod_le (s : Finset ι) (f : ι → E) : ‖∏ i ∈ s, f i‖ ≤ ∑ i ∈ s, ‖f i‖ := by
  /-
    ι : Type u_3
    E : Type u_5
    inst✝ : SeminormedCommGroup E
    s : Finset ι
    f : ι → E
    ⊢ LE.le (Norm.norm (s.prod fun i => f i)) (s.sum fun i => Norm.norm (f i))
  -/
  rw [← Multiplicative.ofAdd_le, ofAdd_sum]
  /-
    ι : Type u_3
    E : Type u_5
    inst✝ : SeminormedCommGroup E
    s : Finset ι
    f : ι → E
    ⊢ LE.le (Multiplicative.ofAdd (Norm.norm (s.prod fun i => f i))) (s.prod fun i …
  -/
  refine Finset.le_prod_of_submultiplicative (Multiplicative.ofAdd ∘ norm) ?_ (fun x y => ?_) _ _
    /-
      case refine_1
      ι : Type u_3
      E : Type u_5
      inst✝ : SeminormedCommGroup E
      s : Finset ι
      f : ι → E
      ⊢ Eq (Function.comp (⇑Multiplicative.ofAdd) Norm.norm 1) 1
    -/
  · simp only [comp_apply, norm_one', ofAdd_zero]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      ι : Type u_3
      E : Type u_5
      inst✝ : SeminormedCommGroup E
      s : Finset ι
      f : ι → E
      x y : E
      ⊢ LE.le (Function.comp (⇑Multiplicative.ofAdd) Norm.norm (HMul.hMul x y)) (HMu …
    -/
  · exact norm_mul_le' x y
    /-
      🎉 no goals
    -/


@[to_additive]
theorem norm_prod_le_of_le (s : Finset ι) {f : ι → E} {n : ι → ℝ} (h : ∀ b ∈ s, ‖f b‖ ≤ n b) :
    ‖∏ b ∈ s, f b‖ ≤ ∑ b ∈ s, n b :=
  (norm_prod_le s f).trans <| Finset.sum_le_sum h


@[to_additive]
theorem dist_prod_prod_le_of_le (s : Finset ι) {f a : ι → E} {d : ι → ℝ}
    (h : ∀ b ∈ s, dist (f b) (a b) ≤ d b) :
    dist (∏ b ∈ s, f b) (∏ b ∈ s, a b) ≤ ∑ b ∈ s, d b := by
  /-
    ι : Type u_3
    E : Type u_5
    inst✝ : SeminormedCommGroup E
    s : Finset ι
    f a : ι → E
    d : ι → Real
    h : ∀ (b : ι), Membership.mem s b → LE.le (Dist.dist (f b) (a b)) (d b)
    ⊢ LE.le (Dist.dist (s.prod fun b => f b) (s.prod fun b => a b)) (s.sum fun b = …
  -/
  simp only [dist_eq_norm_div, ← Finset.prod_div_distrib] at *
  /-
    ι : Type u_3
    E : Type u_5
    inst✝ : SeminormedCommGroup E
    s : Finset ι
    f a : ι → E
    d : ι → Real
    h : ∀ (b : ι), Membership.mem s b → LE.le (Norm.norm (HDiv.hDiv (f b) (a b)))  …
    ⊢ LE.le (Norm.norm (s.prod fun x => HDiv.hDiv (f x) (a x))) (s.sum fun b => d b)
  -/
  exact norm_prod_le_of_le s h
  /-
    🎉 no goals
  -/


@[to_additive]
theorem dist_prod_prod_le (s : Finset ι) (f a : ι → E) :
    dist (∏ b ∈ s, f b) (∏ b ∈ s, a b) ≤ ∑ b ∈ s, dist (f b) (a b) :=
  dist_prod_prod_le_of_le s fun _ _ => le_rfl


@[to_additive]
theorem mul_mem_ball_iff_norm : a * b ∈ ball a r ↔ ‖b‖ < r := by
  /-
    E : Type u_5
    inst✝ : SeminormedCommGroup E
    a b : E
    r : Real
    ⊢ Iff (Membership.mem (Metric.ball a r) (HMul.hMul a b)) (LT.lt (Norm.norm b) r)
  -/
  rw [mem_ball_iff_norm'', mul_div_cancel_left]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mul_mem_closedBall_iff_norm : a * b ∈ closedBall a r ↔ ‖b‖ ≤ r := by
  /-
    E : Type u_5
    inst✝ : SeminormedCommGroup E
    a b : E
    r : Real
    ⊢ Iff (Membership.mem (Metric.closedBall a r) (HMul.hMul a b)) (LE.le (Norm.no …
  -/
  rw [mem_closedBall_iff_norm'', mul_div_cancel_left]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp 1001)]
-- Porting note: increase priority so that the left-hand side doesn't simplify
theorem preimage_mul_ball (a b : E) (r : ℝ) : (b * ·) ⁻¹' ball a r = ball (a / b) r := by
  /-
    E : Type u_5
    inst✝ : SeminormedCommGroup E
    a b : E
    r : Real
    ⊢ Eq (Set.preimage (fun x => HMul.hMul b x) (Metric.ball a r)) (Metric.ball (H …
  -/
  ext c
  /-
    case h
    E : Type u_5
    inst✝ : SeminormedCommGroup E
    a b : E
    r : Real
    c : E
    ⊢ Iff (Membership.mem (Set.preimage (fun x => HMul.hMul b x) (Metric.ball a r) …
  -/
  simp only [dist_eq_norm_div, Set.mem_preimage, mem_ball, div_div_eq_mul_div, mul_comm]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp 1001)]
-- Porting note: increase priority so that the left-hand side doesn't simplify
theorem preimage_mul_closedBall (a b : E) (r : ℝ) :
    (b * ·) ⁻¹' closedBall a r = closedBall (a / b) r := by
  /-
    E : Type u_5
    inst✝ : SeminormedCommGroup E
    a b : E
    r : Real
    ⊢ Eq (Set.preimage (fun x => HMul.hMul b x) (Metric.closedBall a r)) (Metric.c …
  -/
  ext c
  /-
    case h
    E : Type u_5
    inst✝ : SeminormedCommGroup E
    a b : E
    r : Real
    c : E
    ⊢ Iff (Membership.mem (Set.preimage (fun x => HMul.hMul b x) (Metric.closedBal …
  -/
  simp only [dist_eq_norm_div, Set.mem_preimage, mem_closedBall, div_div_eq_mul_div, mul_comm]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem preimage_mul_sphere (a b : E) (r : ℝ) : (b * ·) ⁻¹' sphere a r = sphere (a / b) r := by
  /-
    E : Type u_5
    inst✝ : SeminormedCommGroup E
    a b : E
    r : Real
    ⊢ Eq (Set.preimage (fun x => HMul.hMul b x) (Metric.sphere a r)) (Metric.spher …
  -/
  ext c
  /-
    case h
    E : Type u_5
    inst✝ : SeminormedCommGroup E
    a b : E
    r : Real
    c : E
    ⊢ Iff (Membership.mem (Set.preimage (fun x => HMul.hMul b x) (Metric.sphere a  …
  -/
  simp only [Set.mem_preimage, mem_sphere_iff_norm', div_div_eq_mul_div, mul_comm]
  /-
    🎉 no goals
  -/



@[to_additive]
theorem pow_mem_closedBall {n : ℕ} (h : a ∈ closedBall b r) :
    a ^ n ∈ closedBall (b ^ n) (n • r) := by
  /-
    E : Type u_5
    inst✝ : SeminormedCommGroup E
    a b : E
    r : Real
    n : Nat
    h : Membership.mem (Metric.closedBall b r) a
    ⊢ Membership.mem (Metric.closedBall (HPow.hPow b n) (HSMul.hSMul n r)) (HPow.h …
  -/
  simp only [mem_closedBall, dist_eq_norm_div, ← div_pow] at h ⊢
  /-
    E : Type u_5
    inst✝ : SeminormedCommGroup E
    a b : E
    r : Real
    n : Nat
    h : LE.le (Norm.norm (HDiv.hDiv a b)) r
    ⊢ LE.le (Norm.norm (HPow.hPow (HDiv.hDiv a b) n)) (HSMul.hSMul n r)
  -/
  refine norm_pow_le_mul_norm.trans ?_
  /-
    E : Type u_5
    inst✝ : SeminormedCommGroup E
    a b : E
    r : Real
    n : Nat
    h : LE.le (Norm.norm (HDiv.hDiv a b)) r
    ⊢ LE.le (HMul.hMul (↑n) (Norm.norm (HDiv.hDiv a b))) (HSMul.hSMul n r)
  -/
  simpa only [nsmul_eq_mul] using mul_le_mul_of_nonneg_left h n.cast_nonneg
  /-
    🎉 no goals
  -/


@[to_additive]
theorem pow_mem_ball {n : ℕ} (hn : 0 < n) (h : a ∈ ball b r) : a ^ n ∈ ball (b ^ n) (n • r) := by
  /-
    E : Type u_5
    inst✝ : SeminormedCommGroup E
    a b : E
    r : Real
    n : Nat
    hn : LT.lt 0 n
    h : Membership.mem (Metric.ball b r) a
    ⊢ Membership.mem (Metric.ball (HPow.hPow b n) (HSMul.hSMul n r)) (HPow.hPow a n)
  -/
  simp only [mem_ball, dist_eq_norm_div, ← div_pow] at h ⊢
  /-
    E : Type u_5
    inst✝ : SeminormedCommGroup E
    a b : E
    r : Real
    n : Nat
    hn : LT.lt 0 n
    h : LT.lt (Norm.norm (HDiv.hDiv a b)) r
    ⊢ LT.lt (Norm.norm (HPow.hPow (HDiv.hDiv a b) n)) (HSMul.hSMul n r)
  -/
  refine lt_of_le_of_lt norm_pow_le_mul_norm ?_
  /-
    E : Type u_5
    inst✝ : SeminormedCommGroup E
    a b : E
    r : Real
    n : Nat
    hn : LT.lt 0 n
    h : LT.lt (Norm.norm (HDiv.hDiv a b)) r
    ⊢ LT.lt (HMul.hMul (↑n) (Norm.norm (HDiv.hDiv a b))) (HSMul.hSMul n r)
  -/
  replace hn : 0 < (n : ℝ) := by norm_cast
  /-
    E : Type u_5
    inst✝ : SeminormedCommGroup E
    a b : E
    r : Real
    n : Nat
    h : LT.lt (Norm.norm (HDiv.hDiv a b)) r
    hn : LT.lt 0 ↑n
    ⊢ LT.lt (HMul.hMul (↑n) (Norm.norm (HDiv.hDiv a b))) (HSMul.hSMul n r)
  -/
  rw [nsmul_eq_mul]
  /-
    E : Type u_5
    inst✝ : SeminormedCommGroup E
    a b : E
    r : Real
    n : Nat
    h : LT.lt (Norm.norm (HDiv.hDiv a b)) r
    hn : LT.lt 0 ↑n
    ⊢ LT.lt (HMul.hMul (↑n) (Norm.norm (HDiv.hDiv a b))) (HMul.hMul (↑n) r)
  -/
  nlinarith
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mul_mem_closedBall_mul_iff {c : E} : a * c ∈ closedBall (b * c) r ↔ a ∈ closedBall b r := by
  /-
    E : Type u_5
    inst✝ : SeminormedCommGroup E
    a b : E
    r : Real
    c : E
    ⊢ Iff (Membership.mem (Metric.closedBall (HMul.hMul b c) r) (HMul.hMul a c)) ( …
  -/
  simp only [mem_closedBall, dist_eq_norm_div, mul_div_mul_right_eq_div]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mul_mem_ball_mul_iff {c : E} : a * c ∈ ball (b * c) r ↔ a ∈ ball b r := by
  /-
    E : Type u_5
    inst✝ : SeminormedCommGroup E
    a b : E
    r : Real
    c : E
    ⊢ Iff (Membership.mem (Metric.ball (HMul.hMul b c) r) (HMul.hMul a c)) (Member …
  -/
  simp only [mem_ball, dist_eq_norm_div, mul_div_mul_right_eq_div]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem smul_closedBall'' : a • closedBall b r = closedBall (a • b) r := by
  /-
    E : Type u_5
    inst✝ : SeminormedCommGroup E
    a b : E
    r : Real
    ⊢ Eq (HSMul.hSMul a (Metric.closedBall b r)) (Metric.closedBall (HSMul.hSMul a …
  -/
  ext
  simp [mem_closedBall, Set.mem_smul_set, dist_eq_norm_div, div_eq_inv_mul, ←
    eq_inv_mul_iff_mul_eq, mul_assoc]


@[to_additive]
theorem smul_ball'' : a • ball b r = ball (a • b) r := by
  /-
    E : Type u_5
    inst✝ : SeminormedCommGroup E
    a b : E
    r : Real
    ⊢ Eq (HSMul.hSMul a (Metric.ball b r)) (Metric.ball (HSMul.hSMul a b) r)
  -/
  ext
  simp [mem_ball, Set.mem_smul_set, dist_eq_norm_div, _root_.div_eq_inv_mul,
    ← eq_inv_mul_iff_mul_eq, mul_assoc]


@[to_additive]
theorem controlled_prod_of_mem_closure {s : Subgroup E} (hg : a ∈ closure (s : Set E)) {b : ℕ → ℝ}
    (b_pos : ∀ n, 0 < b n) :
    ∃ v : ℕ → E,
      Tendsto (fun n => ∏ i ∈ range (n + 1), v i) atTop (𝓝 a) ∧
        (∀ n, v n ∈ s) ∧ ‖v 0 / a‖ < b 0 ∧ ∀ n, 0 < n → ‖v n‖ < b n := by
  obtain ⟨u : ℕ → E, u_in : ∀ n, u n ∈ s, lim_u : Tendsto u atTop (𝓝 a)⟩ :=
    mem_closure_iff_seq_limit.mp hg
  obtain ⟨n₀, hn₀⟩ : ∃ n₀, ∀ n ≥ n₀, ‖u n / a‖ < b 0 :=
    haveI : { x | ‖x / a‖ < b 0 } ∈ 𝓝 a := by
      simp_rw [← dist_eq_norm_div]
      exact Metric.ball_mem_nhds _ (b_pos _)
    Filter.tendsto_atTop'.mp lim_u _ this
  /-
    case intro.intro.intro
    E : Type u_5
    inst✝ : SeminormedCommGroup E
    a : E
    s : Subgroup E
    hg : Membership.mem (closure ↑s) a
    b : Nat → Real
    b_pos : ∀ (n : Nat), LT.lt 0 (b n)
    u : Nat → E
    u_in : ∀ (n : Nat), Membership.mem s (u n)
    lim_u : Filter.Tendsto u Filter.atTop (nhds a)
    n₀ : Nat
    hn₀ : ∀ (n : Nat), GE.ge n n₀ → LT.lt (Norm.norm (HDiv.hDiv (u n) a)) (b 0)
    ⊢ Exists fun v => And (Filter.Tendsto (fun n => (Finset.range (HAdd.hAdd n 1)) …
  -/
  set z : ℕ → E := fun n => u (n + n₀)
  /-
    case intro.intro.intro
    E : Type u_5
    inst✝ : SeminormedCommGroup E
    a : E
    s : Subgroup E
    hg : Membership.mem (closure ↑s) a
    b : Nat → Real
    b_pos : ∀ (n : Nat), LT.lt 0 (b n)
    u : Nat → E
    u_in : ∀ (n : Nat), Membership.mem s (u n)
    lim_u : Filter.Tendsto u Filter.atTop (nhds a)
    n₀ : Nat
    hn₀ : ∀ (n : Nat), GE.ge n n₀ → LT.lt (Norm.norm (HDiv.hDiv (u n) a)) (b 0)
    z : Nat → E := fun n => u (HAdd.hAdd n n₀)
    ⊢ Exists fun v => And (Filter.Tendsto (fun n => (Finset.range (HAdd.hAdd n 1)) …
  -/
  have lim_z : Tendsto z atTop (𝓝 a) := lim_u.comp (tendsto_add_atTop_nat n₀)
  have mem_𝓤 : ∀ n, { p : E × E | ‖p.1 / p.2‖ < b (n + 1) } ∈ 𝓤 E := fun n => by
    simpa [← dist_eq_norm_div] using Metric.dist_mem_uniformity (b_pos <| n + 1)
  obtain ⟨φ : ℕ → ℕ, φ_extr : StrictMono φ, hφ : ∀ n, ‖z (φ <| n + 1) / z (φ n)‖ < b (n + 1)⟩ :=
    lim_z.cauchySeq.subseq_mem mem_𝓤
  /-
    case intro.intro.intro.intro.intro
    E : Type u_5
    inst✝ : SeminormedCommGroup E
    a : E
    s : Subgroup E
    hg : Membership.mem (closure ↑s) a
    b : Nat → Real
    b_pos : ∀ (n : Nat), LT.lt 0 (b n)
    u : Nat → E
    u_in : ∀ (n : Nat), Membership.mem s (u n)
    lim_u : Filter.Tendsto u Filter.atTop (nhds a)
    n₀ : Nat
    hn₀ : ∀ (n : Nat), GE.ge n n₀ → LT.lt (Norm.norm (HDiv.hDiv (u n) a)) (b 0)
    z : Nat → E := fun n => u (HAdd.hAdd n n₀)
    lim_z : Filter.Tendsto z Filter.atTop (nhds a)
    mem_𝓤 : ∀ (n : Nat), Membership.mem (uniformity E) (setOf fun p => LT.lt (Norm …
    φ : Nat → Nat
    φ_extr : StrictMono φ
    hφ : ∀ (n : Nat), LT.lt (Norm.norm (HDiv.hDiv (z (φ (HAdd.hAdd n 1))) (z (φ n) …
    ⊢ Exists fun v => And (Filter.Tendsto (fun n => (Finset.range (HAdd.hAdd n 1)) …
  -/
  set w : ℕ → E := z ∘ φ
  /-
    case intro.intro.intro.intro.intro
    E : Type u_5
    inst✝ : SeminormedCommGroup E
    a : E
    s : Subgroup E
    hg : Membership.mem (closure ↑s) a
    b : Nat → Real
    b_pos : ∀ (n : Nat), LT.lt 0 (b n)
    u : Nat → E
    u_in : ∀ (n : Nat), Membership.mem s (u n)
    lim_u : Filter.Tendsto u Filter.atTop (nhds a)
    n₀ : Nat
    hn₀ : ∀ (n : Nat), GE.ge n n₀ → LT.lt (Norm.norm (HDiv.hDiv (u n) a)) (b 0)
    z : Nat → E := fun n => u (HAdd.hAdd n n₀)
    lim_z : Filter.Tendsto z Filter.atTop (nhds a)
    mem_𝓤 : ∀ (n : Nat), Membership.mem (uniformity E) (setOf fun p => LT.lt (Norm …
    φ : Nat → Nat
    φ_extr : StrictMono φ
    hφ : ∀ (n : Nat), LT.lt (Norm.norm (HDiv.hDiv (z (φ (HAdd.hAdd n 1))) (z (φ n) …
    w : Nat → E := Function.comp z φ
    ⊢ Exists fun v => And (Filter.Tendsto (fun n => (Finset.range (HAdd.hAdd n 1)) …
  -/
  have hw : Tendsto w atTop (𝓝 a) := lim_z.comp φ_extr.tendsto_atTop
  /-
    case intro.intro.intro.intro.intro
    E : Type u_5
    inst✝ : SeminormedCommGroup E
    a : E
    s : Subgroup E
    hg : Membership.mem (closure ↑s) a
    b : Nat → Real
    b_pos : ∀ (n : Nat), LT.lt 0 (b n)
    u : Nat → E
    u_in : ∀ (n : Nat), Membership.mem s (u n)
    lim_u : Filter.Tendsto u Filter.atTop (nhds a)
    n₀ : Nat
    hn₀ : ∀ (n : Nat), GE.ge n n₀ → LT.lt (Norm.norm (HDiv.hDiv (u n) a)) (b 0)
    z : Nat → E := fun n => u (HAdd.hAdd n n₀)
    lim_z : Filter.Tendsto z Filter.atTop (nhds a)
    mem_𝓤 : ∀ (n : Nat), Membership.mem (uniformity E) (setOf fun p => LT.lt (Norm …
    φ : Nat → Nat
    φ_extr : StrictMono φ
    hφ : ∀ (n : Nat), LT.lt (Norm.norm (HDiv.hDiv (z (φ (HAdd.hAdd n 1))) (z (φ n) …
    w : Nat → E := Function.comp z φ
    hw : Filter.Tendsto w Filter.atTop (nhds a)
    ⊢ Exists fun v => And (Filter.Tendsto (fun n => (Finset.range (HAdd.hAdd n 1)) …
  -/
  set v : ℕ → E := fun i => if i = 0 then w 0 else w i / w (i - 1)
  /-
    case intro.intro.intro.intro.intro
    E : Type u_5
    inst✝ : SeminormedCommGroup E
    a : E
    s : Subgroup E
    hg : Membership.mem (closure ↑s) a
    b : Nat → Real
    b_pos : ∀ (n : Nat), LT.lt 0 (b n)
    u : Nat → E
    u_in : ∀ (n : Nat), Membership.mem s (u n)
    lim_u : Filter.Tendsto u Filter.atTop (nhds a)
    n₀ : Nat
    hn₀ : ∀ (n : Nat), GE.ge n n₀ → LT.lt (Norm.norm (HDiv.hDiv (u n) a)) (b 0)
    z : Nat → E := fun n => u (HAdd.hAdd n n₀)
    lim_z : Filter.Tendsto z Filter.atTop (nhds a)
    mem_𝓤 : ∀ (n : Nat), Membership.mem (uniformity E) (setOf fun p => LT.lt (Norm …
    φ : Nat → Nat
    φ_extr : StrictMono φ
    hφ : ∀ (n : Nat), LT.lt (Norm.norm (HDiv.hDiv (z (φ (HAdd.hAdd n 1))) (z (φ n) …
    w : Nat → E := Function.comp z φ
    hw : Filter.Tendsto w Filter.atTop (nhds a)
    v : Nat → E := fun i => ite (Eq i 0) (w 0) (HDiv.hDiv (w i) (w (HSub.hSub i 1)))
    ⊢ Exists fun v => And (Filter.Tendsto (fun n => (Finset.range (HAdd.hAdd n 1)) …
  -/
  refine ⟨v, Tendsto.congr (Finset.eq_prod_range_div' w) hw, ?_, hn₀ _ (n₀.le_add_left _), ?_⟩
    /-
      case intro.intro.intro.intro.intro.refine_1
      E : Type u_5
      inst✝ : SeminormedCommGroup E
      a : E
      s : Subgroup E
      hg : Membership.mem (closure ↑s) a
      b : Nat → Real
      b_pos : ∀ (n : Nat), LT.lt 0 (b n)
      u : Nat → E
      u_in : ∀ (n : Nat), Membership.mem s (u n)
      lim_u : Filter.Tendsto u Filter.atTop (nhds a)
      n₀ : Nat
      hn₀ : ∀ (n : Nat), GE.ge n n₀ → LT.lt (Norm.norm (HDiv.hDiv (u n) a)) (b 0)
      z : Nat → E := fun n => u (HAdd.hAdd n n₀)
      lim_z : Filter.Tendsto z Filter.atTop (nhds a)
      mem_𝓤 : ∀ (n : Nat), Membership.mem (uniformity E) (setOf fun p => LT.lt (Norm …
      φ : Nat → Nat
      φ_extr : StrictMono φ
      hφ : ∀ (n : Nat), LT.lt (Norm.norm (HDiv.hDiv (z (φ (HAdd.hAdd n 1))) (z (φ n) …
      w : Nat → E := Function.comp z φ
      hw : Filter.Tendsto w Filter.atTop (nhds a)
      v : Nat → E := fun i => ite (Eq i 0) (w 0) (HDiv.hDiv (w i) (w (HSub.hSub i 1)))
      ⊢ ∀ (n : Nat), Membership.mem s (v n)
    -/
  · rintro ⟨⟩
      /-
        case intro.intro.intro.intro.intro.refine_1.zero
        E : Type u_5
        inst✝ : SeminormedCommGroup E
        a : E
        s : Subgroup E
        hg : Membership.mem (closure ↑s) a
        b : Nat → Real
        b_pos : ∀ (n : Nat), LT.lt 0 (b n)
        u : Nat → E
        u_in : ∀ (n : Nat), Membership.mem s (u n)
        lim_u : Filter.Tendsto u Filter.atTop (nhds a)
        n₀ : Nat
        hn₀ : ∀ (n : Nat), GE.ge n n₀ → LT.lt (Norm.norm (HDiv.hDiv (u n) a)) (b 0)
        z : Nat → E := fun n => u (HAdd.hAdd n n₀)
        lim_z : Filter.Tendsto z Filter.atTop (nhds a)
        mem_𝓤 : ∀ (n : Nat), Membership.mem (uniformity E) (setOf fun p => LT.lt (Norm …
        φ : Nat → Nat
        φ_extr : StrictMono φ
        hφ : ∀ (n : Nat), LT.lt (Norm.norm (HDiv.hDiv (z (φ (HAdd.hAdd n 1))) (z (φ n) …
        w : Nat → E := Function.comp z φ
        hw : Filter.Tendsto w Filter.atTop (nhds a)
        v : Nat → E := fun i => ite (Eq i 0) (w 0) (HDiv.hDiv (w i) (w (HSub.hSub i 1)))
        ⊢ Membership.mem s (v 0)
      -/
    · change w 0 ∈ s
      /-
        case intro.intro.intro.intro.intro.refine_1.zero
        E : Type u_5
        inst✝ : SeminormedCommGroup E
        a : E
        s : Subgroup E
        hg : Membership.mem (closure ↑s) a
        b : Nat → Real
        b_pos : ∀ (n : Nat), LT.lt 0 (b n)
        u : Nat → E
        u_in : ∀ (n : Nat), Membership.mem s (u n)
        lim_u : Filter.Tendsto u Filter.atTop (nhds a)
        n₀ : Nat
        hn₀ : ∀ (n : Nat), GE.ge n n₀ → LT.lt (Norm.norm (HDiv.hDiv (u n) a)) (b 0)
        z : Nat → E := fun n => u (HAdd.hAdd n n₀)
        lim_z : Filter.Tendsto z Filter.atTop (nhds a)
        mem_𝓤 : ∀ (n : Nat), Membership.mem (uniformity E) (setOf fun p => LT.lt (Norm …
        φ : Nat → Nat
        φ_extr : StrictMono φ
        hφ : ∀ (n : Nat), LT.lt (Norm.norm (HDiv.hDiv (z (φ (HAdd.hAdd n 1))) (z (φ n) …
        w : Nat → E := Function.comp z φ
        hw : Filter.Tendsto w Filter.atTop (nhds a)
        v : Nat → E := fun i => ite (Eq i 0) (w 0) (HDiv.hDiv (w i) (w (HSub.hSub i 1)))
        ⊢ Membership.mem s (w 0)
      -/
      apply u_in
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.intro.intro.refine_1.succ
        E : Type u_5
        inst✝ : SeminormedCommGroup E
        a : E
        s : Subgroup E
        hg : Membership.mem (closure ↑s) a
        b : Nat → Real
        b_pos : ∀ (n : Nat), LT.lt 0 (b n)
        u : Nat → E
        u_in : ∀ (n : Nat), Membership.mem s (u n)
        lim_u : Filter.Tendsto u Filter.atTop (nhds a)
        n₀ : Nat
        hn₀ : ∀ (n : Nat), GE.ge n n₀ → LT.lt (Norm.norm (HDiv.hDiv (u n) a)) (b 0)
        z : Nat → E := fun n => u (HAdd.hAdd n n₀)
        lim_z : Filter.Tendsto z Filter.atTop (nhds a)
        mem_𝓤 : ∀ (n : Nat), Membership.mem (uniformity E) (setOf fun p => LT.lt (Norm …
        φ : Nat → Nat
        φ_extr : StrictMono φ
        hφ : ∀ (n : Nat), LT.lt (Norm.norm (HDiv.hDiv (z (φ (HAdd.hAdd n 1))) (z (φ n) …
        w : Nat → E := Function.comp z φ
        hw : Filter.Tendsto w Filter.atTop (nhds a)
        v : Nat → E := fun i => ite (Eq i 0) (w 0) (HDiv.hDiv (w i) (w (HSub.hSub i 1)))
        n✝ : Nat
        ⊢ Membership.mem s (v (HAdd.hAdd n✝ 1))
      -/
                          /-
                            🎉 no goals
                          -/
    · apply s.div_mem <;> apply u_in
                          /-
                            🎉 no goals
                          -/
    /-
      case intro.intro.intro.intro.intro.refine_2
      E : Type u_5
      inst✝ : SeminormedCommGroup E
      a : E
      s : Subgroup E
      hg : Membership.mem (closure ↑s) a
      b : Nat → Real
      b_pos : ∀ (n : Nat), LT.lt 0 (b n)
      u : Nat → E
      u_in : ∀ (n : Nat), Membership.mem s (u n)
      lim_u : Filter.Tendsto u Filter.atTop (nhds a)
      n₀ : Nat
      hn₀ : ∀ (n : Nat), GE.ge n n₀ → LT.lt (Norm.norm (HDiv.hDiv (u n) a)) (b 0)
      z : Nat → E := fun n => u (HAdd.hAdd n n₀)
      lim_z : Filter.Tendsto z Filter.atTop (nhds a)
      mem_𝓤 : ∀ (n : Nat), Membership.mem (uniformity E) (setOf fun p => LT.lt (Norm …
      φ : Nat → Nat
      φ_extr : StrictMono φ
      hφ : ∀ (n : Nat), LT.lt (Norm.norm (HDiv.hDiv (z (φ (HAdd.hAdd n 1))) (z (φ n) …
      w : Nat → E := Function.comp z φ
      hw : Filter.Tendsto w Filter.atTop (nhds a)
      v : Nat → E := fun i => ite (Eq i 0) (w 0) (HDiv.hDiv (w i) (w (HSub.hSub i 1)))
      ⊢ ∀ (n : Nat), LT.lt 0 n → LT.lt (Norm.norm (v n)) (b n)
    -/
  · intro l hl
    /-
      case intro.intro.intro.intro.intro.refine_2
      E : Type u_5
      inst✝ : SeminormedCommGroup E
      a : E
      s : Subgroup E
      hg : Membership.mem (closure ↑s) a
      b : Nat → Real
      b_pos : ∀ (n : Nat), LT.lt 0 (b n)
      u : Nat → E
      u_in : ∀ (n : Nat), Membership.mem s (u n)
      lim_u : Filter.Tendsto u Filter.atTop (nhds a)
      n₀ : Nat
      hn₀ : ∀ (n : Nat), GE.ge n n₀ → LT.lt (Norm.norm (HDiv.hDiv (u n) a)) (b 0)
      z : Nat → E := fun n => u (HAdd.hAdd n n₀)
      lim_z : Filter.Tendsto z Filter.atTop (nhds a)
      mem_𝓤 : ∀ (n : Nat), Membership.mem (uniformity E) (setOf fun p => LT.lt (Norm …
      φ : Nat → Nat
      φ_extr : StrictMono φ
      hφ : ∀ (n : Nat), LT.lt (Norm.norm (HDiv.hDiv (z (φ (HAdd.hAdd n 1))) (z (φ n) …
      w : Nat → E := Function.comp z φ
      hw : Filter.Tendsto w Filter.atTop (nhds a)
      v : Nat → E := fun i => ite (Eq i 0) (w 0) (HDiv.hDiv (w i) (w (HSub.hSub i 1)))
      l : Nat
      hl : LT.lt 0 l
      ⊢ LT.lt (Norm.norm (v l)) (b l)
    -/
    obtain ⟨k, rfl⟩ : ∃ k, l = k + 1 := Nat.exists_eq_succ_of_ne_zero hl.ne'
    /-
      case intro.intro.intro.intro.intro.refine_2.intro
      E : Type u_5
      inst✝ : SeminormedCommGroup E
      a : E
      s : Subgroup E
      hg : Membership.mem (closure ↑s) a
      b : Nat → Real
      b_pos : ∀ (n : Nat), LT.lt 0 (b n)
      u : Nat → E
      u_in : ∀ (n : Nat), Membership.mem s (u n)
      lim_u : Filter.Tendsto u Filter.atTop (nhds a)
      n₀ : Nat
      hn₀ : ∀ (n : Nat), GE.ge n n₀ → LT.lt (Norm.norm (HDiv.hDiv (u n) a)) (b 0)
      z : Nat → E := fun n => u (HAdd.hAdd n n₀)
      lim_z : Filter.Tendsto z Filter.atTop (nhds a)
      mem_𝓤 : ∀ (n : Nat), Membership.mem (uniformity E) (setOf fun p => LT.lt (Norm …
      φ : Nat → Nat
      φ_extr : StrictMono φ
      hφ : ∀ (n : Nat), LT.lt (Norm.norm (HDiv.hDiv (z (φ (HAdd.hAdd n 1))) (z (φ n) …
      w : Nat → E := Function.comp z φ
      hw : Filter.Tendsto w Filter.atTop (nhds a)
      v : Nat → E := fun i => ite (Eq i 0) (w 0) (HDiv.hDiv (w i) (w (HSub.hSub i 1)))
      k : Nat
      hl : LT.lt 0 (HAdd.hAdd k 1)
      ⊢ LT.lt (Norm.norm (v (HAdd.hAdd k 1))) (b (HAdd.hAdd k 1))
    -/
    apply hφ
    /-
      🎉 no goals
    -/


@[to_additive]
theorem controlled_prod_of_mem_closure_range {j : E →* F} {b : F}
    (hb : b ∈ closure (j.range : Set F)) {f : ℕ → ℝ} (b_pos : ∀ n, 0 < f n) :
    ∃ a : ℕ → E,
      Tendsto (fun n => ∏ i ∈ range (n + 1), j (a i)) atTop (𝓝 b) ∧
        ‖j (a 0) / b‖ < f 0 ∧ ∀ n, 0 < n → ‖j (a n)‖ < f n := by
  /-
    E : Type u_5
    F : Type u_6
    inst✝¹ : SeminormedCommGroup E
    inst✝ : SeminormedCommGroup F
    j : MonoidHom E F
    b : F
    hb : Membership.mem (closure ↑j.range) b
    f : Nat → Real
    b_pos : ∀ (n : Nat), LT.lt 0 (f n)
    ⊢ Exists fun a => And (Filter.Tendsto (fun n => (Finset.range (HAdd.hAdd n 1)) …
  -/
  obtain ⟨v, sum_v, v_in, hv₀, hv_pos⟩ := controlled_prod_of_mem_closure hb b_pos
  /-
    case intro.intro.intro.intro
    E : Type u_5
    F : Type u_6
    inst✝¹ : SeminormedCommGroup E
    inst✝ : SeminormedCommGroup F
    j : MonoidHom E F
    b : F
    hb : Membership.mem (closure ↑j.range) b
    f : Nat → Real
    b_pos : ∀ (n : Nat), LT.lt 0 (f n)
    v : Nat → F
    sum_v : Filter.Tendsto (fun n => (Finset.range (HAdd.hAdd n 1)).prod fun i =>  …
    v_in : ∀ (n : Nat), Membership.mem j.range (v n)
    hv₀ : LT.lt (Norm.norm (HDiv.hDiv (v 0) b)) (f 0)
    hv_pos : ∀ (n : Nat), LT.lt 0 n → LT.lt (Norm.norm (v n)) (f n)
    ⊢ Exists fun a => And (Filter.Tendsto (fun n => (Finset.range (HAdd.hAdd n 1)) …
  -/
  choose g hg using v_in
  exact
    ⟨g, by simpa [← hg] using sum_v, by simpa [hg 0] using hv₀,
      fun n hn => by simpa [hg] using hv_pos n hn⟩


@[to_additive]
theorem nnnorm_multiset_prod_le (m : Multiset E) : ‖m.prod‖₊ ≤ (m.map fun x => ‖x‖₊).sum :=
  NNReal.coe_le_coe.1 <| by
    /-
      E : Type u_5
      inst✝ : SeminormedCommGroup E
      m : Multiset E
      ⊢ LE.le ↑(NNNorm.nnnorm m.prod) ↑(Multiset.map (fun x => NNNorm.nnnorm x) m).sum
    -/
    push_cast
    /-
      E : Type u_5
      inst✝ : SeminormedCommGroup E
      m : Multiset E
      ⊢ LE.le (Norm.norm m.prod) (Multiset.map NNReal.toReal (Multiset.map (fun x => …
    -/
    rw [Multiset.map_map]
    /-
      E : Type u_5
      inst✝ : SeminormedCommGroup E
      m : Multiset E
      ⊢ LE.le (Norm.norm m.prod) (Multiset.map (Function.comp NNReal.toReal fun x => …
    -/
    exact norm_multiset_prod_le _
    /-
      🎉 no goals
    -/


@[to_additive]
theorem nnnorm_prod_le (s : Finset ι) (f : ι → E) : ‖∏ a ∈ s, f a‖₊ ≤ ∑ a ∈ s, ‖f a‖₊ :=
  NNReal.coe_le_coe.1 <| by
    /-
      ι : Type u_3
      E : Type u_5
      inst✝ : SeminormedCommGroup E
      s : Finset ι
      f : ι → E
      ⊢ LE.le ↑(NNNorm.nnnorm (s.prod fun a => f a)) ↑(s.sum fun a => NNNorm.nnnorm  …
    -/
    push_cast
    /-
      ι : Type u_3
      E : Type u_5
      inst✝ : SeminormedCommGroup E
      s : Finset ι
      f : ι → E
      ⊢ LE.le (Norm.norm (s.prod fun a => f a)) (s.sum fun x => Norm.norm (f x))
    -/
    exact norm_prod_le _ _
    /-
      🎉 no goals
    -/


@[to_additive]
theorem nnnorm_prod_le_of_le (s : Finset ι) {f : ι → E} {n : ι → ℝ≥0} (h : ∀ b ∈ s, ‖f b‖₊ ≤ n b) :
    ‖∏ b ∈ s, f b‖₊ ≤ ∑ b ∈ s, n b :=
  (norm_prod_le_of_le s h).trans_eq (NNReal.coe_sum ..).symm


instance norm : Norm ℝ where
  norm r := |r|


@[simp]
theorem norm_eq_abs (r : ℝ) : ‖r‖ = |r| :=
  rfl


instance normedAddCommGroup : NormedAddCommGroup ℝ :=
  ⟨fun _r _y => rfl⟩


theorem norm_of_nonneg (hr : 0 ≤ r) : ‖r‖ = r :=
  abs_of_nonneg hr


theorem norm_of_nonpos (hr : r ≤ 0) : ‖r‖ = -r :=
  abs_of_nonpos hr


theorem le_norm_self (r : ℝ) : r ≤ ‖r‖ :=
  le_abs_self r


@[simp 1100] lemma norm_natCast (n : ℕ) : ‖(n : ℝ)‖ = n := abs_of_nonneg n.cast_nonneg

@[simp 1100] lemma nnnorm_natCast (n : ℕ) : ‖(n : ℝ)‖₊ = n := NNReal.eq <| norm_natCast _


@[deprecated (since := "2024-04-05")] alias norm_coe_nat := norm_natCast

@[deprecated (since := "2024-04-05")] alias nnnorm_coe_nat := nnnorm_natCast


@[simp 1100] lemma norm_ofNat (n : ℕ) [n.AtLeastTwo] :
    ‖(no_index (OfNat.ofNat n) : ℝ)‖ = OfNat.ofNat n := norm_natCast n


@[simp 1100] lemma nnnorm_ofNat (n : ℕ) [n.AtLeastTwo] :
    ‖(no_index (OfNat.ofNat n) : ℝ)‖₊ = OfNat.ofNat n := nnnorm_natCast n


lemma norm_two : ‖(2 : ℝ)‖ = 2 := abs_of_pos zero_lt_two

                                                     /-
                                                       ⊢ Eq ↑(NNNorm.nnnorm 2) ↑2
                                                     -/
lemma nnnorm_two : ‖(2 : ℝ)‖₊ = 2 := NNReal.eq <| by simp
                                                     /-
                                                       🎉 no goals
                                                     -/


@[simp 1100, norm_cast]
lemma norm_nnratCast (q : ℚ≥0) : ‖(q : ℝ)‖ = q := norm_of_nonneg q.cast_nonneg


@[simp 1100, norm_cast]
                                                        /-
                                                          q : NNRat
                                                          ⊢ Eq (NNNorm.nnnorm ↑q) ↑q
                                                        -/
lemma nnnorm_nnratCast (q : ℚ≥0) : ‖(q : ℝ)‖₊ = q := by simp [nnnorm, -norm_eq_abs]
                                                        /-
                                                          🎉 no goals
                                                        -/


theorem nnnorm_of_nonneg (hr : 0 ≤ r) : ‖r‖₊ = ⟨r, hr⟩ :=
  NNReal.eq <| norm_of_nonneg hr


@[simp]
                                                 /-
                                                   r : Real
                                                   ⊢ Eq (NNNorm.nnnorm (abs r)) (NNNorm.nnnorm r)
                                                 -/
theorem nnnorm_abs (r : ℝ) : ‖|r|‖₊ = ‖r‖₊ := by simp [nnnorm]
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem ennnorm_eq_ofReal (hr : 0 ≤ r) : (‖r‖₊ : ℝ≥0∞) = ENNReal.ofReal r := by
  /-
    r : Real
    hr : LE.le 0 r
    ⊢ Eq (↑(NNNorm.nnnorm r)) (ENNReal.ofReal r)
  -/
  rw [← ofReal_norm_eq_coe_nnnorm, norm_of_nonneg hr]
  /-
    🎉 no goals
  -/


theorem ennnorm_eq_ofReal_abs (r : ℝ) : (‖r‖₊ : ℝ≥0∞) = ENNReal.ofReal |r| := by
  /-
    r : Real
    ⊢ Eq (↑(NNNorm.nnnorm r)) (ENNReal.ofReal (abs r))
  -/
  rw [← Real.nnnorm_abs r, Real.ennnorm_eq_ofReal (abs_nonneg _)]
  /-
    🎉 no goals
  -/


theorem toNNReal_eq_nnnorm_of_nonneg (hr : 0 ≤ r) : r.toNNReal = ‖r‖₊ := by
  /-
    r : Real
    hr : LE.le 0 r
    ⊢ Eq r.toNNReal (NNNorm.nnnorm r)
  -/
  rw [Real.toNNReal_of_nonneg hr]
  /-
    r : Real
    hr : LE.le 0 r
    ⊢ Eq ⟨r, hr⟩ (NNNorm.nnnorm r)
  -/
  ext
  /-
    case a
    r : Real
    hr : LE.le 0 r
    ⊢ Eq ↑⟨r, hr⟩ ↑(NNNorm.nnnorm r)
  -/
  rw [coe_mk, coe_nnnorm r, Real.norm_eq_abs r, abs_of_nonneg hr]
  /-
    🎉 no goals
  -/
  -- Porting note: this is due to the change from `Subtype.val` to `NNReal.toReal` for the coercion


theorem ofReal_le_ennnorm (r : ℝ) : ENNReal.ofReal r ≤ ‖r‖₊ := by
  /-
    r : Real
    ⊢ LE.le (ENNReal.ofReal r) ↑(NNNorm.nnnorm r)
  -/
  obtain hr | hr := le_total 0 r
    /-
      case inl
      r : Real
      hr : LE.le 0 r
      ⊢ LE.le (ENNReal.ofReal r) ↑(NNNorm.nnnorm r)
    -/
  · exact (Real.ennnorm_eq_ofReal hr).ge
    /-
      🎉 no goals
    -/
    /-
      case inr
      r : Real
      hr : LE.le r 0
      ⊢ LE.le (ENNReal.ofReal r) ↑(NNNorm.nnnorm r)
    -/
  · rw [ENNReal.ofReal_eq_zero.2 hr]
    /-
      case inr
      r : Real
      hr : LE.le r 0
      ⊢ LE.le 0 ↑(NNNorm.nnnorm r)
    -/
    exact bot_le
    /-
      🎉 no goals
    -/
-- Porting note: should this be renamed to `Real.ofReal_le_nnnorm`?


instance : NNNorm ℝ≥0 where
  nnnorm x := x


@[simp] lemma nnnorm_eq_self (x : ℝ≥0) : ‖x‖₊ = x := rfl


@[to_additive (attr := simp) norm_le_zero_iff]
                                                /-
                                                  E : Type u_5
                                                  inst✝ : NormedGroup E
                                                  a : E
                                                  ⊢ Iff (LE.le (Norm.norm a) 0) (Eq a 1)
                                                -/
lemma norm_le_zero_iff' : ‖a‖ ≤ 0 ↔ a = 1 := by rw [← dist_one_right, dist_le_zero]
                                                /-
                                                  🎉 no goals
                                                -/


@[to_additive (attr := simp) norm_pos_iff]
                                            /-
                                              E : Type u_5
                                              inst✝ : NormedGroup E
                                              a : E
                                              ⊢ Iff (LT.lt 0 (Norm.norm a)) (Ne a 1)
                                            -/
lemma norm_pos_iff' : 0 < ‖a‖ ↔ a ≠ 1 := by rw [← not_le, norm_le_zero_iff']
                                            /-
                                              🎉 no goals
                                            -/


@[to_additive (attr := simp) norm_eq_zero]
lemma norm_eq_zero' : ‖a‖ = 0 ↔ a = 1 := (norm_nonneg' a).le_iff_eq.symm.trans norm_le_zero_iff'


@[to_additive norm_ne_zero_iff]
lemma norm_ne_zero_iff' : ‖a‖ ≠ 0 ↔ a ≠ 1 := norm_eq_zero'.not


@[deprecated (since := "2024-11-24")] alias norm_le_zero_iff'' := norm_le_zero_iff'

@[deprecated (since := "2024-11-24")] alias norm_le_zero_iff''' := norm_le_zero_iff'

@[deprecated (since := "2024-11-24")] alias norm_pos_iff'' := norm_pos_iff'

@[deprecated (since := "2024-11-24")] alias norm_eq_zero'' := norm_eq_zero'

@[deprecated (since := "2024-11-24")] alias norm_eq_zero''' := norm_eq_zero'


@[to_additive]
                                                         /-
                                                           E : Type u_5
                                                           inst✝ : NormedGroup E
                                                           a b : E
                                                           ⊢ Iff (Eq (Norm.norm (HDiv.hDiv a b)) 0) (Eq a b)
                                                         -/
theorem norm_div_eq_zero_iff : ‖a / b‖ = 0 ↔ a = b := by rw [norm_eq_zero', div_eq_one]
                                                         /-
                                                           🎉 no goals
                                                         -/


@[to_additive]
theorem norm_div_pos_iff : 0 < ‖a / b‖ ↔ a ≠ b := by
  /-
    E : Type u_5
    inst✝ : NormedGroup E
    a b : E
    ⊢ Iff (LT.lt 0 (Norm.norm (HDiv.hDiv a b))) (Ne a b)
  -/
  rw [(norm_nonneg' _).lt_iff_ne, ne_comm]
  /-
    E : Type u_5
    inst✝ : NormedGroup E
    a b : E
    ⊢ Iff (Ne (Norm.norm (HDiv.hDiv a b)) 0) (Ne a b)
  -/
  exact norm_div_eq_zero_iff.not
  /-
    🎉 no goals
  -/


@[to_additive eq_of_norm_sub_le_zero]
theorem eq_of_norm_div_le_zero (h : ‖a / b‖ ≤ 0) : a = b := by
  /-
    E : Type u_5
    inst✝ : NormedGroup E
    a b : E
    h : LE.le (Norm.norm (HDiv.hDiv a b)) 0
    ⊢ Eq a b
  -/
  rwa [← div_eq_one, ← norm_le_zero_iff']
  /-
    🎉 no goals
  -/


alias ⟨eq_of_norm_div_eq_zero, _⟩ := norm_div_eq_zero_iff


attribute [to_additive] eq_of_norm_div_eq_zero


@[to_additive]
theorem eq_one_or_norm_pos (a : E) : a = 1 ∨ 0 < ‖a‖ := by
  /-
    E : Type u_5
    inst✝ : NormedGroup E
    a : E
    ⊢ Or (Eq a 1) (LT.lt 0 (Norm.norm a))
  -/
  simpa [eq_comm] using (norm_nonneg' a).eq_or_lt
  /-
    🎉 no goals
  -/


@[to_additive]
theorem eq_one_or_nnnorm_pos (a : E) : a = 1 ∨ 0 < ‖a‖₊ :=
  eq_one_or_norm_pos a


@[to_additive (attr := simp) nnnorm_eq_zero]
theorem nnnorm_eq_zero' : ‖a‖₊ = 0 ↔ a = 1 := by
  /-
    E : Type u_5
    inst✝ : NormedGroup E
    a : E
    ⊢ Iff (Eq (NNNorm.nnnorm a) 0) (Eq a 1)
  -/
  rw [← NNReal.coe_eq_zero, coe_nnnorm', norm_eq_zero']
  /-
    🎉 no goals
  -/


@[to_additive nnnorm_ne_zero_iff]
theorem nnnorm_ne_zero_iff' : ‖a‖₊ ≠ 0 ↔ a ≠ 1 :=
  nnnorm_eq_zero'.not


@[to_additive (attr := simp) nnnorm_pos]
lemma nnnorm_pos' : 0 < ‖a‖₊ ↔ a ≠ 1 := pos_iff_ne_zero.trans nnnorm_ne_zero_iff'


/-- See `tendsto_norm_one` for a version with full neighborhoods. -/
@[to_additive "See `tendsto_norm_zero` for a version with full neighborhoods."]
lemma tendsto_norm_nhdsNE_one : Tendsto (norm : E → ℝ) (𝓝[≠] 1) (𝓝[>] 0) :=
  tendsto_norm_one.inf <| tendsto_principal_principal.2 fun _ hx ↦ norm_pos_iff'.2 hx


@[deprecated (since := "2024-12-22")]
alias tendsto_norm_zero' := tendsto_norm_nhdsNE_zero

@[to_additive existing, deprecated (since := "2024-12-22")]
alias tendsto_norm_one' := tendsto_norm_nhdsNE_one


@[deprecated (since := "2024-12-22")]
alias tendsto_norm_nhdsWithin_zero := tendsto_norm_nhdsNE_zero

@[to_additive existing, deprecated (since := "2024-12-22")]
alias tendsto_norm_nhdsWithin_one := tendsto_norm_nhdsNE_one


@[to_additive]
theorem tendsto_norm_div_self_nhdsNE (a : E) : Tendsto (fun x => ‖x / a‖) (𝓝[≠] a) (𝓝[>] 0) :=
  (tendsto_norm_div_self a).inf <|
    tendsto_principal_principal.2 fun _x hx => norm_pos_iff'.2 <| div_ne_one.2 hx


@[deprecated (since := "2024-12-22")]
alias tendsto_norm_sub_self_punctured_nhds := tendsto_norm_sub_self_nhdsNE

@[to_additive existing, deprecated (since := "2024-12-22")]
alias tendsto_norm_div_self_punctured_nhds := tendsto_norm_div_self_nhdsNE


/-- The norm of a normed group as a group norm. -/
@[to_additive "The norm of a normed group as an additive group norm."]
def normGroupNorm : GroupNorm E :=
  { normGroupSeminorm _ with eq_one_of_map_eq_zero' := fun _ => norm_eq_zero'.1 }


@[simp]
theorem coe_normGroupNorm : ⇑(normGroupNorm E) = norm :=
  rfl


/-- A version of `comap_norm_nhdsGT_zero` for a multiplicative normed group. -/
@[to_additive comap_norm_nhdsGT_zero]
lemma comap_norm_nhdsGT_zero' : comap norm (𝓝[>] 0) = 𝓝[≠] (1 : E) := by
  /-
    E : Type u_5
    inst✝ : NormedGroup E
    ⊢ Eq (Filter.comap Norm.norm (nhdsWithin 0 (Set.Ioi 0))) (nhdsWithin 1 (HasCom …
  -/
  simp [nhdsWithin, comap_norm_nhds_one, Set.preimage, Set.compl_def]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-22")]
alias comap_norm_nhdsWithin_Ioi_zero := comap_norm_nhdsGT_zero

@[to_additive existing comap_norm_nhdsWithin_Ioi_zero, deprecated (since := "2024-12-22")]
alias comap_norm_nhdsWithin_Ioi_zero' := comap_norm_nhdsGT_zero'


theorem hasCompactSupport_norm_iff : (HasCompactSupport fun x => ‖f x‖) ↔ HasCompactSupport f :=
  hasCompactSupport_comp_left norm_eq_zero


alias ⟨_, HasCompactSupport.norm⟩ := hasCompactSupport_norm_iff


/-- Extension for the `positivity` tactic: multiplicative norms are always nonnegative, and positive
on non-one inputs. -/
@[positivity ‖_‖]
def evalMulNorm : PositivityExt where eval {u α} _ _ e := do
  match u, α, e with
  | 0, ~q(ℝ), ~q(@Norm.norm $E $_n $a) =>
    let _seminormedGroup_E ← synthInstanceQ q(SeminormedGroup $E)
    assertInstancesCommute
    -- Check whether we are in a normed group and whether the context contains a `a ≠ 1` assumption
    let o : Option (Q(NormedGroup $E) × Q($a ≠ 1)) := ← do
      let .some normedGroup_E ← trySynthInstanceQ q(NormedGroup $E) | return none
      let some pa ← findLocalDeclWithTypeQ? q($a ≠ 1) | return none
      return some (normedGroup_E, pa)
    match o with
    -- If so, return a proof of `0 < ‖a‖`
    | some (_normedGroup_E, pa) =>
      assertInstancesCommute
      return .positive q(norm_pos_iff'.2 $pa)
    -- Else, return a proof of `0 ≤ ‖a‖`
    | none => return .nonnegative q(norm_nonneg' $a)
  | _, _, _ => throwError "not `‖·‖`"


/-- Extension for the `positivity` tactic: additive norms are always nonnegative, and positive
on non-zero inputs. -/
@[positivity ‖_‖]
def evalAddNorm : PositivityExt where eval {u α} _ _ e := do
  match u, α, e with
  | 0, ~q(ℝ), ~q(@Norm.norm $E $_n $a) =>
    let _seminormedAddGroup_E ← synthInstanceQ q(SeminormedAddGroup $E)
    assertInstancesCommute
    -- Check whether we are in a normed group and whether the context contains a `a ≠ 0` assumption
    let o : Option (Q(NormedAddGroup $E) × Q($a ≠ 0)) := ← do
      let .some normedAddGroup_E ← trySynthInstanceQ q(NormedAddGroup $E) | return none
      let some pa ← findLocalDeclWithTypeQ? q($a ≠ 0) | return none
      return some (normedAddGroup_E, pa)
    match o with
    -- If so, return a proof of `0 < ‖a‖`
    | some (_normedAddGroup_E, pa) =>
      assertInstancesCommute
      return .positive q(norm_pos_iff.2 $pa)
    -- Else, return a proof of `0 ≤ ‖a‖`
    | none => return .nonnegative q(norm_nonneg $a)
  | _, _, _ => throwError "not `‖·‖`"


/-- A subgroup of a seminormed group is also a seminormed group,
with the restriction of the norm. -/
@[to_additive "A subgroup of a seminormed group is also a seminormed group, with the restriction of
the norm."]
instance seminormedGroup : SeminormedGroup s :=
  SeminormedGroup.induced _ _ s.subtype


/-- If `x` is an element of a subgroup `s` of a seminormed group `E`, its norm in `s` is equal to
its norm in `E`. -/
@[to_additive (attr := simp) "If `x` is an element of a subgroup `s` of a seminormed group `E`, its
norm in `s` is equal to its norm in `E`."]
theorem coe_norm (x : s) : ‖x‖ = ‖(x : E)‖ :=
  rfl


/-- If `x` is an element of a subgroup `s` of a seminormed group `E`, its norm in `s` is equal to
its norm in `E`.

This is a reversed version of the `simp` lemma `Subgroup.coe_norm` for use by `norm_cast`. -/
@[to_additive (attr := norm_cast) "If `x` is an element of a subgroup `s` of a seminormed group `E`,
its norm in `s` is equal to its norm in `E`.

This is a reversed version of the `simp` lemma `AddSubgroup.coe_norm` for use by `norm_cast`."]
theorem norm_coe {s : Subgroup E} (x : s) : ‖(x : E)‖ = ‖x‖ :=
  rfl


@[to_additive]
instance seminormedCommGroup [SeminormedCommGroup E] {s : Subgroup E} : SeminormedCommGroup s :=
  SeminormedCommGroup.induced _ _ s.subtype


@[to_additive]
instance normedGroup [NormedGroup E] {s : Subgroup E} : NormedGroup s :=
  NormedGroup.induced _ _ s.subtype Subtype.coe_injective


@[to_additive]
instance normedCommGroup [NormedCommGroup E] {s : Subgroup E} : NormedCommGroup s :=
  NormedCommGroup.induced _ _ s.subtype Subtype.coe_injective


/-- A subgroup of a seminormed group is also a seminormed group,
with the restriction of the norm. -/
@[to_additive "A subgroup of a seminormed additive group is also a seminormed additive group, with
the restriction of the norm."]
instance (priority := 75) seminormedGroup : SeminormedGroup s :=
  SeminormedGroup.induced _ _ (SubgroupClass.subtype s)


/-- If `x` is an element of a subgroup `s` of a seminormed group `E`, its norm in `s` is equal to
its norm in `E`. -/
@[to_additive (attr := simp) "If `x` is an element of an additive subgroup `s` of a seminormed
additive group `E`, its norm in `s` is equal to its norm in `E`."]
theorem coe_norm (x : s) : ‖x‖ = ‖(x : E)‖ :=
  rfl


@[to_additive]
instance (priority := 75) seminormedCommGroup [SeminormedCommGroup E] {S : Type*} [SetLike S E]
    [SubgroupClass S E] (s : S) : SeminormedCommGroup s :=
  SeminormedCommGroup.induced _ _ (SubgroupClass.subtype s)


@[to_additive]
instance (priority := 75) normedGroup [NormedGroup E] {S : Type*} [SetLike S E] [SubgroupClass S E]
    (s : S) : NormedGroup s :=
  NormedGroup.induced _ _ (SubgroupClass.subtype s) Subtype.coe_injective


@[to_additive]
instance (priority := 75) normedCommGroup [NormedCommGroup E] {S : Type*} [SetLike S E]
    [SubgroupClass S E] (s : S) : NormedCommGroup s :=
  NormedCommGroup.induced _ _ (SubgroupClass.subtype s) Subtype.coe_injective


lemma tendsto_norm_atTop_atTop : Tendsto (norm : ℝ → ℝ) atTop atTop := tendsto_abs_atTop_atTop


