/-- We now define `MetricSpace`, extending `PseudoMetricSpace`. -/
class MetricSpace (α : Type u) extends PseudoMetricSpace α : Type u where
  eq_of_dist_eq_zero : ∀ {x y : α}, dist x y = 0 → x = y


/-- Two metric space structures with the same distance coincide. -/
@[ext]
theorem MetricSpace.ext {α : Type*} {m m' : MetricSpace α} (h : m.toDist = m'.toDist) :
    m = m' := by
  /-
    α : Type u_3
    m m' : MetricSpace α
    h : Eq PseudoMetricSpace.toDist PseudoMetricSpace.toDist
    ⊢ Eq m m'
  -/
  cases m; cases m'; congr; ext1; assumption
                                  /-
                                    🎉 no goals
                                  -/


/-- Construct a metric space structure whose underlying topological space structure
(definitionally) agrees which a pre-existing topology which is compatible with a given distance
function. -/
def MetricSpace.ofDistTopology {α : Type u} [TopologicalSpace α] (dist : α → α → ℝ)
    (dist_self : ∀ x : α, dist x x = 0) (dist_comm : ∀ x y : α, dist x y = dist y x)
    (dist_triangle : ∀ x y z : α, dist x z ≤ dist x y + dist y z)
    (H : ∀ s : Set α, IsOpen s ↔ ∀ x ∈ s, ∃ ε > 0, ∀ y, dist x y < ε → y ∈ s)
    (eq_of_dist_eq_zero : ∀ x y : α, dist x y = 0 → x = y) : MetricSpace α :=
  { PseudoMetricSpace.ofDistTopology dist dist_self dist_comm dist_triangle H with
    eq_of_dist_eq_zero := eq_of_dist_eq_zero _ _ }


theorem eq_of_dist_eq_zero {x y : γ} : dist x y = 0 → x = y :=
  MetricSpace.eq_of_dist_eq_zero


@[simp]
theorem dist_eq_zero {x y : γ} : dist x y = 0 ↔ x = y :=
  Iff.intro eq_of_dist_eq_zero fun this => this ▸ dist_self _


@[simp]
                                                            /-
                                                              γ : Type w
                                                              inst✝ : MetricSpace γ
                                                              x y : γ
                                                              ⊢ Iff (Eq 0 (Dist.dist x y)) (Eq x y)
                                                            -/
theorem zero_eq_dist {x y : γ} : 0 = dist x y ↔ x = y := by rw [eq_comm, dist_eq_zero]
                                                            /-
                                                              🎉 no goals
                                                            -/


theorem dist_ne_zero {x y : γ} : dist x y ≠ 0 ↔ x ≠ y := by
  /-
    γ : Type w
    inst✝ : MetricSpace γ
    x y : γ
    ⊢ Iff (Ne (Dist.dist x y) 0) (Ne x y)
  -/
  simpa only [not_iff_not] using dist_eq_zero
  /-
    🎉 no goals
  -/


@[simp]
theorem dist_le_zero {x y : γ} : dist x y ≤ 0 ↔ x = y := by
  /-
    γ : Type w
    inst✝ : MetricSpace γ
    x y : γ
    ⊢ Iff (LE.le (Dist.dist x y) 0) (Eq x y)
  -/
  simpa [le_antisymm_iff, dist_nonneg] using @dist_eq_zero _ _ x y
  /-
    🎉 no goals
  -/


@[simp]
theorem dist_pos {x y : γ} : 0 < dist x y ↔ x ≠ y := by
  /-
    γ : Type w
    inst✝ : MetricSpace γ
    x y : γ
    ⊢ Iff (LT.lt 0 (Dist.dist x y)) (Ne x y)
  -/
  simpa only [not_le] using not_congr dist_le_zero
  /-
    🎉 no goals
  -/


theorem eq_of_forall_dist_le {x y : γ} (h : ∀ ε > 0, dist x y ≤ ε) : x = y :=
  eq_of_dist_eq_zero (eq_of_le_of_forall_le_of_dense dist_nonneg h)


/-- Deduce the equality of points from the vanishing of the nonnegative distance -/
theorem eq_of_nndist_eq_zero {x y : γ} : nndist x y = 0 → x = y := by
  /-
    γ : Type w
    inst✝ : MetricSpace γ
    x y : γ
    ⊢ Eq (NNDist.nndist x y) 0 → Eq x y
  -/
  simp only [NNReal.eq_iff, ← dist_nndist, imp_self, NNReal.coe_zero, dist_eq_zero]
  /-
    🎉 no goals
  -/


/-- Characterize the equality of points as the vanishing of the nonnegative distance -/
@[simp]
theorem nndist_eq_zero {x y : γ} : nndist x y = 0 ↔ x = y := by
  /-
    γ : Type w
    inst✝ : MetricSpace γ
    x y : γ
    ⊢ Iff (Eq (NNDist.nndist x y) 0) (Eq x y)
  -/
  simp only [NNReal.eq_iff, ← dist_nndist, imp_self, NNReal.coe_zero, dist_eq_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem zero_eq_nndist {x y : γ} : 0 = nndist x y ↔ x = y := by
  /-
    γ : Type w
    inst✝ : MetricSpace γ
    x y : γ
    ⊢ Iff (Eq 0 (NNDist.nndist x y)) (Eq x y)
  -/
  simp only [NNReal.eq_iff, ← dist_nndist, imp_self, NNReal.coe_zero, zero_eq_dist]
  /-
    🎉 no goals
  -/


@[simp] theorem closedBall_zero : closedBall x 0 = {x} := Set.ext fun _ => dist_le_zero


@[simp] theorem sphere_zero : sphere x 0 = {x} := Set.ext fun _ => dist_eq_zero


theorem subsingleton_closedBall (x : γ) {r : ℝ} (hr : r ≤ 0) : (closedBall x r).Subsingleton := by
  /-
    γ : Type w
    inst✝ : MetricSpace γ
    x : γ
    r : Real
    hr : LE.le r 0
    ⊢ (Metric.closedBall x r).Subsingleton
  -/
  rcases hr.lt_or_eq with (hr | rfl)
    /-
      case inl
      γ : Type w
      inst✝ : MetricSpace γ
      x : γ
      r : Real
      hr✝ : LE.le r 0
      hr : LT.lt r 0
      ⊢ (Metric.closedBall x r).Subsingleton
    -/
  · rw [closedBall_eq_empty.2 hr]
    /-
      case inl
      γ : Type w
      inst✝ : MetricSpace γ
      x : γ
      r : Real
      hr✝ : LE.le r 0
      hr : LT.lt r 0
      ⊢ EmptyCollection.emptyCollection.Subsingleton
    -/
    exact subsingleton_empty
    /-
      🎉 no goals
    -/
    /-
      case inr
      γ : Type w
      inst✝ : MetricSpace γ
      x : γ
      hr : LE.le 0 0
      ⊢ (Metric.closedBall x 0).Subsingleton
    -/
  · rw [closedBall_zero]
    /-
      case inr
      γ : Type w
      inst✝ : MetricSpace γ
      x : γ
      hr : LE.le 0 0
      ⊢ (Singleton.singleton x).Subsingleton
    -/
    exact subsingleton_singleton
    /-
      🎉 no goals
    -/


theorem subsingleton_sphere (x : γ) {r : ℝ} (hr : r ≤ 0) : (sphere x r).Subsingleton :=
  (subsingleton_closedBall x hr).anti sphere_subset_closedBall


/-- Build a new metric space from an old one where the bundled uniform structure is provably
(but typically non-definitionaly) equal to some given uniform structure.
See Note [forgetful inheritance].
See Note [reducible non-instances].
-/
abbrev MetricSpace.replaceUniformity {γ} [U : UniformSpace γ] (m : MetricSpace γ)
    (H : 𝓤[U] = 𝓤[PseudoEMetricSpace.toUniformSpace]) : MetricSpace γ where
  toPseudoMetricSpace := PseudoMetricSpace.replaceUniformity m.toPseudoMetricSpace H
  eq_of_dist_eq_zero := @eq_of_dist_eq_zero _ _


theorem MetricSpace.replaceUniformity_eq {γ} [U : UniformSpace γ] (m : MetricSpace γ)
    (H : 𝓤[U] = 𝓤[PseudoEMetricSpace.toUniformSpace]) : m.replaceUniformity H = m := by
  /-
    γ : Type u_3
    U : UniformSpace γ
    m : MetricSpace γ
    H : Eq (uniformity γ) (uniformity γ)
    ⊢ Eq (m.replaceUniformity H) m
  -/
  ext; rfl
       /-
         🎉 no goals
       -/


/-- Build a new metric space from an old one where the bundled topological structure is provably
(but typically non-definitionaly) equal to some given topological structure.
See Note [forgetful inheritance].
See Note [reducible non-instances].
-/
abbrev MetricSpace.replaceTopology {γ} [U : TopologicalSpace γ] (m : MetricSpace γ)
    (H : U = m.toPseudoMetricSpace.toUniformSpace.toTopologicalSpace) : MetricSpace γ :=
  @MetricSpace.replaceUniformity γ (m.toUniformSpace.replaceTopology H) m rfl


theorem MetricSpace.replaceTopology_eq {γ} [U : TopologicalSpace γ] (m : MetricSpace γ)
    (H : U = m.toPseudoMetricSpace.toUniformSpace.toTopologicalSpace) :
    m.replaceTopology H = m := by
  /-
    γ : Type u_3
    U : TopologicalSpace γ
    m : MetricSpace γ
    H : Eq U UniformSpace.toTopologicalSpace
    ⊢ Eq (m.replaceTopology H) m
  -/
  ext; rfl
       /-
         🎉 no goals
       -/


/-- Build a new metric space from an old one where the bundled bornology structure is provably
(but typically non-definitionaly) equal to some given bornology structure.
See Note [forgetful inheritance].
See Note [reducible non-instances].
-/
abbrev MetricSpace.replaceBornology {α} [B : Bornology α] (m : MetricSpace α)
    (H : ∀ s, @IsBounded _ B s ↔ @IsBounded _ PseudoMetricSpace.toBornology s) : MetricSpace α :=
  { PseudoMetricSpace.replaceBornology _ H, m with toBornology := B }


theorem MetricSpace.replaceBornology_eq {α} [m : MetricSpace α] [B : Bornology α]
    (H : ∀ s, @IsBounded _ B s ↔ @IsBounded _ PseudoMetricSpace.toBornology s) :
    MetricSpace.replaceBornology _ H = m := by
  /-
    α : Type u_3
    m : MetricSpace α
    B : Bornology α
    H : ∀ (s : Set α), Iff (Bornology.IsBounded s) (Bornology.IsBounded s)
    ⊢ Eq (m.replaceBornology H) m
  -/
  ext
  /-
    case h.dist.h.h
    α : Type u_3
    m : MetricSpace α
    B : Bornology α
    H : ∀ (s : Set α), Iff (Bornology.IsBounded s) (Bornology.IsBounded s)
    x✝¹ x✝ : α
    ⊢ Eq (Dist.dist x✝¹ x✝) (Dist.dist x✝¹ x✝)
  -/
  rfl
  /-
    🎉 no goals
  -/


instance : MetricSpace Empty where
  dist _ _ := 0
  dist_self _ := rfl
  dist_comm _ _ := rfl
  edist _ _ := 0
  eq_of_dist_eq_zero _ := Subsingleton.elim _ _
                                                 /-
                                                   α : Type u
                                                   β : Type v
                                                   X : Type u_1
                                                   ι : Type u_2
                                                   inst✝¹ : PseudoMetricSpace α
                                                   γ : Type w
                                                   inst✝ : MetricSpace γ
                                                   x✝² x✝¹ x✝ : Empty
                                                   ⊢ LE.le 0 (HAdd.hAdd 0 0)
                                                 -/
  dist_triangle _ _ _ := show (0 : ℝ) ≤ 0 + 0 by rw [add_zero]
                                                 /-
                                                   🎉 no goals
                                                 -/
  toUniformSpace := inferInstance
  uniformity_dist := Subsingleton.elim _ _


instance : MetricSpace PUnit.{u + 1} where
  dist _ _ := 0
  dist_self _ := rfl
  dist_comm _ _ := rfl
  edist _ _ := 0
  eq_of_dist_eq_zero _ := Subsingleton.elim _ _
                                                 /-
                                                   α : Type u
                                                   β : Type v
                                                   X : Type u_1
                                                   ι : Type u_2
                                                   inst✝¹ : PseudoMetricSpace α
                                                   γ : Type w
                                                   inst✝ : MetricSpace γ
                                                   x✝² x✝¹ x✝ : PUnit.{u + 1}
                                                   ⊢ LE.le 0 (HAdd.hAdd 0 0)
                                                 -/
  dist_triangle _ _ _ := show (0 : ℝ) ≤ 0 + 0 by rw [add_zero]
                                                 /-
                                                   🎉 no goals
                                                 -/
  toUniformSpace := inferInstance
  uniformity_dist := by
    /-
      α : Type u
      β : Type v
      X : Type u_1
      ι : Type u_2
      inst✝¹ : PseudoMetricSpace α
      γ : Type w
      inst✝ : MetricSpace γ
      ⊢ Eq (uniformity PUnit.{u + 1}) (iInf fun ε => iInf fun h => Filter.principal  …
    -/
    simp +contextual [principal_univ, eq_top_of_neBot (𝓤 PUnit)]
    /-
      🎉 no goals
    -/


instance : Dist (Additive X) := ‹Dist X›

instance : Dist (Multiplicative X) := ‹Dist X›


@[simp] theorem dist_ofMul (a b : X) : dist (ofMul a) (ofMul b) = dist a b := rfl


@[simp] theorem dist_ofAdd (a b : X) : dist (ofAdd a) (ofAdd b) = dist a b := rfl


@[simp] theorem dist_toMul (a b : Additive X) : dist a.toMul b.toMul = dist a b := rfl


@[simp] theorem dist_toAdd (a b : Multiplicative X) : dist a.toAdd b.toAdd = dist a b := rfl


@[simp] theorem nndist_ofMul (a b : X) : nndist (ofMul a) (ofMul b) = nndist a b := rfl


@[simp] theorem nndist_ofAdd (a b : X) : nndist (ofAdd a) (ofAdd b) = nndist a b := rfl


@[simp] theorem nndist_toMul (a b : Additive X) : nndist a.toMul b.toMul = nndist a b := rfl


@[simp]
theorem nndist_toAdd (a b : Multiplicative X) : nndist a.toAdd b.toAdd = nndist a b := rfl


instance [MetricSpace X] : MetricSpace (Additive X) := ‹MetricSpace X›

instance [MetricSpace X] : MetricSpace (Multiplicative X) := ‹MetricSpace X›


instance : Dist Xᵒᵈ := ‹Dist X›


@[simp] theorem dist_toDual (a b : X) : dist (toDual a) (toDual b) = dist a b := rfl


@[simp] theorem dist_ofDual (a b : Xᵒᵈ) : dist (ofDual a) (ofDual b) = dist a b := rfl


instance : PseudoMetricSpace Xᵒᵈ := ‹PseudoMetricSpace X›


@[simp] theorem nndist_toDual (a b : X) : nndist (toDual a) (toDual b) = nndist a b := rfl


@[simp] theorem nndist_ofDual (a b : Xᵒᵈ) : nndist (ofDual a) (ofDual b) = nndist a b := rfl


instance [MetricSpace X] : MetricSpace Xᵒᵈ := ‹MetricSpace X›

