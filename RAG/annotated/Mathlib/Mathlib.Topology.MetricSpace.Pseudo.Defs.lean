theorem UniformSpace.ofDist_aux (ε : ℝ) (hε : 0 < ε) : ∃ δ > (0 : ℝ), ∀ x < δ, ∀ y < δ, x + y < ε :=
  ⟨ε / 2, half_pos hε, fun _x hx _y hy => add_halves ε ▸ add_lt_add hx hy⟩


/-- Construct a uniform structure from a distance function and metric space axioms -/
def UniformSpace.ofDist (dist : α → α → ℝ) (dist_self : ∀ x : α, dist x x = 0)
    (dist_comm : ∀ x y : α, dist x y = dist y x)
    (dist_triangle : ∀ x y z : α, dist x z ≤ dist x y + dist y z) : UniformSpace α :=
  .ofFun dist dist_self dist_comm dist_triangle ofDist_aux

-- Porting note: dropped the `dist_self` argument

/-- Construct a bornology from a distance function and metric space axioms. -/
abbrev Bornology.ofDist {α : Type*} (dist : α → α → ℝ) (dist_comm : ∀ x y, dist x y = dist y x)
    (dist_triangle : ∀ x y z, dist x z ≤ dist x y + dist y z) : Bornology α :=
  Bornology.ofBounded { s : Set α | ∃ C, ∀ ⦃x⦄, x ∈ s → ∀ ⦃y⦄, y ∈ s → dist x y ≤ C }
    ⟨0, fun _ hx _ => hx.elim⟩ (fun _ ⟨c, hc⟩ _ h => ⟨c, fun _ hx _ hy => hc (h hx) (h hy)⟩)
    (fun s hs t ht => by
      /-
        α✝ : Type u
        β : Type v
        X : Type u_1
        ι : Type u_2
        α : Type u_3
        dist : α → α → Real
        dist_comm : ∀ (x y : α), Eq (dist x y) (dist y x)
        dist_triangle : ∀ (x y z : α), LE.le (dist x z) (HAdd.hAdd (dist x y) (dist y  …
        s : Set α
        hs : Membership.mem (setOf fun s => Exists fun C => ∀ ⦃x : α⦄, Membership.mem  …
        t : Set α
        ht : Membership.mem (setOf fun s => Exists fun C => ∀ ⦃x : α⦄, Membership.mem  …
        ⊢ Membership.mem (setOf fun s => Exists fun C => ∀ ⦃x : α⦄, Membership.mem s x …
      -/
      rcases s.eq_empty_or_nonempty with rfl | ⟨x, hx⟩
        /-
          case inl
          α✝ : Type u
          β : Type v
          X : Type u_1
          ι : Type u_2
          α : Type u_3
          dist : α → α → Real
          dist_comm : ∀ (x y : α), Eq (dist x y) (dist y x)
          dist_triangle : ∀ (x y z : α), LE.le (dist x z) (HAdd.hAdd (dist x y) (dist y  …
          t : Set α
          ht : Membership.mem (setOf fun s => Exists fun C => ∀ ⦃x : α⦄, Membership.mem  …
          hs : Membership.mem (setOf fun s => Exists fun C => ∀ ⦃x : α⦄, Membership.mem  …
          ⊢ Membership.mem (setOf fun s => Exists fun C => ∀ ⦃x : α⦄, Membership.mem s x …
        -/
      · rwa [empty_union]
        /-
          🎉 no goals
        -/
      /-
        case inr.intro
        α✝ : Type u
        β : Type v
        X : Type u_1
        ι : Type u_2
        α : Type u_3
        dist : α → α → Real
        dist_comm : ∀ (x y : α), Eq (dist x y) (dist y x)
        dist_triangle : ∀ (x y z : α), LE.le (dist x z) (HAdd.hAdd (dist x y) (dist y  …
        s : Set α
        hs : Membership.mem (setOf fun s => Exists fun C => ∀ ⦃x : α⦄, Membership.mem  …
        t : Set α
        ht : Membership.mem (setOf fun s => Exists fun C => ∀ ⦃x : α⦄, Membership.mem  …
        x : α
        hx : Membership.mem s x
        ⊢ Membership.mem (setOf fun s => Exists fun C => ∀ ⦃x : α⦄, Membership.mem s x …
      -/
      rcases t.eq_empty_or_nonempty with rfl | ⟨y, hy⟩
        /-
          case inr.intro.inl
          α✝ : Type u
          β : Type v
          X : Type u_1
          ι : Type u_2
          α : Type u_3
          dist : α → α → Real
          dist_comm : ∀ (x y : α), Eq (dist x y) (dist y x)
          dist_triangle : ∀ (x y z : α), LE.le (dist x z) (HAdd.hAdd (dist x y) (dist y  …
          s : Set α
          hs : Membership.mem (setOf fun s => Exists fun C => ∀ ⦃x : α⦄, Membership.mem  …
          x : α
          hx : Membership.mem s x
          ht : Membership.mem (setOf fun s => Exists fun C => ∀ ⦃x : α⦄, Membership.mem  …
          ⊢ Membership.mem (setOf fun s => Exists fun C => ∀ ⦃x : α⦄, Membership.mem s x …
        -/
      · rwa [union_empty]
        /-
          🎉 no goals
        -/
      /-
        case inr.intro.inr.intro
        α✝ : Type u
        β : Type v
        X : Type u_1
        ι : Type u_2
        α : Type u_3
        dist : α → α → Real
        dist_comm : ∀ (x y : α), Eq (dist x y) (dist y x)
        dist_triangle : ∀ (x y z : α), LE.le (dist x z) (HAdd.hAdd (dist x y) (dist y  …
        s : Set α
        hs : Membership.mem (setOf fun s => Exists fun C => ∀ ⦃x : α⦄, Membership.mem  …
        t : Set α
        ht : Membership.mem (setOf fun s => Exists fun C => ∀ ⦃x : α⦄, Membership.mem  …
        x : α
        hx : Membership.mem s x
        y : α
        hy : Membership.mem t y
        ⊢ Membership.mem (setOf fun s => Exists fun C => ∀ ⦃x : α⦄, Membership.mem s x …
      -/
      rsuffices ⟨C, hC⟩ : ∃ C, ∀ z ∈ s ∪ t, dist x z ≤ C
        /-
          case inr.intro.inr.intro.intro
          α✝ : Type u
          β : Type v
          X : Type u_1
          ι : Type u_2
          α : Type u_3
          dist : α → α → Real
          dist_comm : ∀ (x y : α), Eq (dist x y) (dist y x)
          dist_triangle : ∀ (x y z : α), LE.le (dist x z) (HAdd.hAdd (dist x y) (dist y  …
          s : Set α
          hs : Membership.mem (setOf fun s => Exists fun C => ∀ ⦃x : α⦄, Membership.mem  …
          t : Set α
          ht : Membership.mem (setOf fun s => Exists fun C => ∀ ⦃x : α⦄, Membership.mem  …
          x : α
          hx : Membership.mem s x
          y : α
          hy : Membership.mem t y
          C : Real
          hC : ∀ (z : α), Membership.mem (Union.union s t) z → LE.le (dist x z) C
          ⊢ Membership.mem (setOf fun s => Exists fun C => ∀ ⦃x : α⦄, Membership.mem s x …
        -/
      · refine ⟨C + C, fun a ha b hb => (dist_triangle a x b).trans ?_⟩
        /-
          case inr.intro.inr.intro.intro
          α✝ : Type u
          β : Type v
          X : Type u_1
          ι : Type u_2
          α : Type u_3
          dist : α → α → Real
          dist_comm : ∀ (x y : α), Eq (dist x y) (dist y x)
          dist_triangle : ∀ (x y z : α), LE.le (dist x z) (HAdd.hAdd (dist x y) (dist y  …
          s : Set α
          hs : Membership.mem (setOf fun s => Exists fun C => ∀ ⦃x : α⦄, Membership.mem  …
          t : Set α
          ht : Membership.mem (setOf fun s => Exists fun C => ∀ ⦃x : α⦄, Membership.mem  …
          x : α
          hx : Membership.mem s x
          y : α
          hy : Membership.mem t y
          C : Real
          hC : ∀ (z : α), Membership.mem (Union.union s t) z → LE.le (dist x z) C
          a : α
          ha : Membership.mem (Union.union s t) a
          b : α
          hb : Membership.mem (Union.union s t) b
          ⊢ LE.le (HAdd.hAdd (dist a x) (dist x b)) (HAdd.hAdd C C)
        -/
        simpa only [dist_comm] using add_le_add (hC _ ha) (hC _ hb)
        /-
          🎉 no goals
        -/
      /-
        α✝ : Type u
        β : Type v
        X : Type u_1
        ι : Type u_2
        α : Type u_3
        dist : α → α → Real
        dist_comm : ∀ (x y : α), Eq (dist x y) (dist y x)
        dist_triangle : ∀ (x y z : α), LE.le (dist x z) (HAdd.hAdd (dist x y) (dist y  …
        s : Set α
        hs : Membership.mem (setOf fun s => Exists fun C => ∀ ⦃x : α⦄, Membership.mem  …
        t : Set α
        ht : Membership.mem (setOf fun s => Exists fun C => ∀ ⦃x : α⦄, Membership.mem  …
        x : α
        hx : Membership.mem s x
        y : α
        hy : Membership.mem t y
        ⊢ Exists fun C => ∀ (z : α), Membership.mem (Union.union s t) z → LE.le (dist  …
      -/
      rcases hs with ⟨Cs, hs⟩; rcases ht with ⟨Ct, ht⟩
      refine ⟨max Cs (dist x y + Ct), fun z hz => hz.elim
        (fun hz => (hs hx hz).trans (le_max_left _ _))
        (fun hz => (dist_triangle x y z).trans <|
          (add_le_add le_rfl (ht hy hz)).trans (le_max_right _ _))⟩)
    fun z => ⟨dist z z, forall_eq.2 <| forall_eq.2 le_rfl⟩


/-- The distance function (given an ambient metric space on `α`), which returns
  a nonnegative real number `dist x y` given `x y : α`. -/
@[ext]
class Dist (α : Type*) where
  dist : α → α → ℝ


/-- This is an internal lemma used inside the default of `PseudoMetricSpace.edist`. -/
private theorem dist_nonneg' {α} {x y : α} (dist : α → α → ℝ)
    (dist_self : ∀ x : α, dist x x = 0) (dist_comm : ∀ x y : α, dist x y = dist y x)
    (dist_triangle : ∀ x y z : α, dist x z ≤ dist x y + dist y z) : 0 ≤ dist x y :=
  have : 0 ≤ 2 * dist x y :=
    calc 0 = dist x x := (dist_self _).symm
    _ ≤ dist x y + dist y x := dist_triangle _ _ _
                           /-
                             α : Sort u_3
                             x y : α
                             dist : α → α → Real
                             dist_self : ∀ (x : α), Eq (dist x x) 0
                             dist_comm : ∀ (x y : α), Eq (dist x y) (dist y x)
                             dist_triangle : ∀ (x y z : α), LE.le (dist x z) (HAdd.hAdd (dist x y) (dist y  …
                             ⊢ Eq (HAdd.hAdd (dist x y) (dist y x)) (HMul.hMul 2 (dist x y))
                           -/
    _ = 2 * dist x y := by rw [two_mul, dist_comm]
                           /-
                             🎉 no goals
                           -/
  nonneg_of_mul_nonneg_right this two_pos


/-- Pseudo metric and Metric spaces

A pseudo metric space is endowed with a distance for which the requirement `d(x,y)=0 → x = y` might
not hold. A metric space is a pseudo metric space such that `d(x,y)=0 → x = y`.
Each pseudo metric space induces a canonical `UniformSpace` and hence a canonical
`TopologicalSpace` This is enforced in the type class definition, by extending the `UniformSpace`
structure. When instantiating a `PseudoMetricSpace` structure, the uniformity fields are not
necessary, they will be filled in by default. In the same way, each (pseudo) metric space induces a
(pseudo) emetric space structure. It is included in the structure, but filled in by default.
-/
class PseudoMetricSpace (α : Type u) extends Dist α : Type u where
  dist_self : ∀ x : α, dist x x = 0
  dist_comm : ∀ x y : α, dist x y = dist y x
  dist_triangle : ∀ x y z : α, dist x z ≤ dist x y + dist y z
  edist : α → α → ℝ≥0∞ := fun x y => ENNReal.ofNNReal ⟨dist x y, dist_nonneg' _ ‹_› ‹_› ‹_›⟩
  edist_dist : ∀ x y : α, edist x y = ENNReal.ofReal (dist x y) := by
    intros x y; exact ENNReal.coe_nnreal_eq _
  toUniformSpace : UniformSpace α := .ofDist dist dist_self dist_comm dist_triangle
  uniformity_dist : 𝓤 α = ⨅ ε > 0, 𝓟 { p : α × α | dist p.1 p.2 < ε } := by intros; rfl
  toBornology : Bornology α := Bornology.ofDist dist dist_comm dist_triangle
  cobounded_sets : (Bornology.cobounded α).sets =
    { s | ∃ C : ℝ, ∀ x ∈ sᶜ, ∀ y ∈ sᶜ, dist x y ≤ C } := by intros; rfl


/-- Two pseudo metric space structures with the same distance function coincide. -/
@[ext]
theorem PseudoMetricSpace.ext {α : Type*} {m m' : PseudoMetricSpace α}
    (h : m.toDist = m'.toDist) : m = m' := by
  /-
    α : Type u_3
    m m' : PseudoMetricSpace α
    h : Eq PseudoMetricSpace.toDist PseudoMetricSpace.toDist
    ⊢ Eq m m'
  -/
  cases' m with d _ _ _ ed hed U hU B hB
  /-
    case mk
    α : Type u_3
    m' : PseudoMetricSpace α
    d : Dist α
    dist_self✝ : ∀ (x : α), Eq (Dist.dist x x) 0
    dist_comm✝ : ∀ (x y : α), Eq (Dist.dist x y) (Dist.dist y x)
    dist_triangle✝ : ∀ (x y z : α), LE.le (Dist.dist x z) (HAdd.hAdd (Dist.dist x  …
    ed : α → α → ENNReal
    hed : ∀ (x y : α), Eq (ed x y) (ENNReal.ofReal (Dist.dist x y))
    U : UniformSpace α
    hU : Eq (uniformity α) (iInf fun ε => iInf fun h => Filter.principal (setOf fu …
    B : Bornology α
    hB : Eq (Bornology.cobounded α).sets (setOf fun s => Exists fun C => ∀ (x : α) …
    h : Eq PseudoMetricSpace.toDist PseudoMetricSpace.toDist
    ⊢ Eq (PseudoMetricSpace.mk dist_self✝ dist_comm✝ dist_triangle✝ ed hed U hU B  …
  -/
  cases' m' with d' _ _ _ ed' hed' U' hU' B' hB'
  /-
    case mk.mk
    α : Type u_3
    d : Dist α
    dist_self✝¹ : ∀ (x : α), Eq (Dist.dist x x) 0
    dist_comm✝¹ : ∀ (x y : α), Eq (Dist.dist x y) (Dist.dist y x)
    dist_triangle✝¹ : ∀ (x y z : α), LE.le (Dist.dist x z) (HAdd.hAdd (Dist.dist x …
    ed : α → α → ENNReal
    hed : ∀ (x y : α), Eq (ed x y) (ENNReal.ofReal (Dist.dist x y))
    U : UniformSpace α
    hU : Eq (uniformity α) (iInf fun ε => iInf fun h => Filter.principal (setOf fu …
    B : Bornology α
    hB : Eq (Bornology.cobounded α).sets (setOf fun s => Exists fun C => ∀ (x : α) …
    d' : Dist α
    dist_self✝ : ∀ (x : α), Eq (Dist.dist x x) 0
    dist_comm✝ : ∀ (x y : α), Eq (Dist.dist x y) (Dist.dist y x)
    dist_triangle✝ : ∀ (x y z : α), LE.le (Dist.dist x z) (HAdd.hAdd (Dist.dist x  …
    ed' : α → α → ENNReal
    hed' : ∀ (x y : α), Eq (ed' x y) (ENNReal.ofReal (Dist.dist x y))
    U' : UniformSpace α
    hU' : Eq (uniformity α) (iInf fun ε => iInf fun h => Filter.principal (setOf f …
    B' : Bornology α
    hB' : Eq (Bornology.cobounded α).sets (setOf fun s => Exists fun C => ∀ (x : α …
    h : Eq PseudoMetricSpace.toDist PseudoMetricSpace.toDist
    ⊢ Eq (PseudoMetricSpace.mk dist_self✝¹ dist_comm✝¹ dist_triangle✝¹ ed hed U hU …
  -/
  obtain rfl : d = d' := h
  /-
    case mk.mk
    α : Type u_3
    d : Dist α
    dist_self✝¹ : ∀ (x : α), Eq (Dist.dist x x) 0
    dist_comm✝¹ : ∀ (x y : α), Eq (Dist.dist x y) (Dist.dist y x)
    dist_triangle✝¹ : ∀ (x y z : α), LE.le (Dist.dist x z) (HAdd.hAdd (Dist.dist x …
    ed : α → α → ENNReal
    hed : ∀ (x y : α), Eq (ed x y) (ENNReal.ofReal (Dist.dist x y))
    U : UniformSpace α
    hU : Eq (uniformity α) (iInf fun ε => iInf fun h => Filter.principal (setOf fu …
    B : Bornology α
    hB : Eq (Bornology.cobounded α).sets (setOf fun s => Exists fun C => ∀ (x : α) …
    ed' : α → α → ENNReal
    U' : UniformSpace α
    B' : Bornology α
    dist_self✝ : ∀ (x : α), Eq (Dist.dist x x) 0
    dist_comm✝ : ∀ (x y : α), Eq (Dist.dist x y) (Dist.dist y x)
    dist_triangle✝ : ∀ (x y z : α), LE.le (Dist.dist x z) (HAdd.hAdd (Dist.dist x  …
    hed' : ∀ (x y : α), Eq (ed' x y) (ENNReal.ofReal (Dist.dist x y))
    hU' : Eq (uniformity α) (iInf fun ε => iInf fun h => Filter.principal (setOf f …
    hB' : Eq (Bornology.cobounded α).sets (setOf fun s => Exists fun C => ∀ (x : α …
    ⊢ Eq (PseudoMetricSpace.mk dist_self✝¹ dist_comm✝¹ dist_triangle✝¹ ed hed U hU …
  -/
  congr
    /-
      case mk.mk.e_edist
      α : Type u_3
      d : Dist α
      dist_self✝¹ : ∀ (x : α), Eq (Dist.dist x x) 0
      dist_comm✝¹ : ∀ (x y : α), Eq (Dist.dist x y) (Dist.dist y x)
      dist_triangle✝¹ : ∀ (x y z : α), LE.le (Dist.dist x z) (HAdd.hAdd (Dist.dist x …
      ed : α → α → ENNReal
      hed : ∀ (x y : α), Eq (ed x y) (ENNReal.ofReal (Dist.dist x y))
      U : UniformSpace α
      hU : Eq (uniformity α) (iInf fun ε => iInf fun h => Filter.principal (setOf fu …
      B : Bornology α
      hB : Eq (Bornology.cobounded α).sets (setOf fun s => Exists fun C => ∀ (x : α) …
      ed' : α → α → ENNReal
      U' : UniformSpace α
      B' : Bornology α
      dist_self✝ : ∀ (x : α), Eq (Dist.dist x x) 0
      dist_comm✝ : ∀ (x y : α), Eq (Dist.dist x y) (Dist.dist y x)
      dist_triangle✝ : ∀ (x y z : α), LE.le (Dist.dist x z) (HAdd.hAdd (Dist.dist x  …
      hed' : ∀ (x y : α), Eq (ed' x y) (ENNReal.ofReal (Dist.dist x y))
      hU' : Eq (uniformity α) (iInf fun ε => iInf fun h => Filter.principal (setOf f …
      hB' : Eq (Bornology.cobounded α).sets (setOf fun s => Exists fun C => ∀ (x : α …
      ⊢ Eq ed ed'
    -/
  · ext x y : 2
    /-
      case mk.mk.e_edist.h.h
      α : Type u_3
      d : Dist α
      dist_self✝¹ : ∀ (x : α), Eq (Dist.dist x x) 0
      dist_comm✝¹ : ∀ (x y : α), Eq (Dist.dist x y) (Dist.dist y x)
      dist_triangle✝¹ : ∀ (x y z : α), LE.le (Dist.dist x z) (HAdd.hAdd (Dist.dist x …
      ed : α → α → ENNReal
      hed : ∀ (x y : α), Eq (ed x y) (ENNReal.ofReal (Dist.dist x y))
      U : UniformSpace α
      hU : Eq (uniformity α) (iInf fun ε => iInf fun h => Filter.principal (setOf fu …
      B : Bornology α
      hB : Eq (Bornology.cobounded α).sets (setOf fun s => Exists fun C => ∀ (x : α) …
      ed' : α → α → ENNReal
      U' : UniformSpace α
      B' : Bornology α
      dist_self✝ : ∀ (x : α), Eq (Dist.dist x x) 0
      dist_comm✝ : ∀ (x y : α), Eq (Dist.dist x y) (Dist.dist y x)
      dist_triangle✝ : ∀ (x y z : α), LE.le (Dist.dist x z) (HAdd.hAdd (Dist.dist x  …
      hed' : ∀ (x y : α), Eq (ed' x y) (ENNReal.ofReal (Dist.dist x y))
      hU' : Eq (uniformity α) (iInf fun ε => iInf fun h => Filter.principal (setOf f …
      hB' : Eq (Bornology.cobounded α).sets (setOf fun s => Exists fun C => ∀ (x : α …
      x y : α
      ⊢ Eq (ed x y) (ed' x y)
    -/
    rw [hed, hed']
    /-
      🎉 no goals
    -/
    /-
      case mk.mk.e_toUniformSpace
      α : Type u_3
      d : Dist α
      dist_self✝¹ : ∀ (x : α), Eq (Dist.dist x x) 0
      dist_comm✝¹ : ∀ (x y : α), Eq (Dist.dist x y) (Dist.dist y x)
      dist_triangle✝¹ : ∀ (x y z : α), LE.le (Dist.dist x z) (HAdd.hAdd (Dist.dist x …
      ed : α → α → ENNReal
      hed : ∀ (x y : α), Eq (ed x y) (ENNReal.ofReal (Dist.dist x y))
      U : UniformSpace α
      hU : Eq (uniformity α) (iInf fun ε => iInf fun h => Filter.principal (setOf fu …
      B : Bornology α
      hB : Eq (Bornology.cobounded α).sets (setOf fun s => Exists fun C => ∀ (x : α) …
      ed' : α → α → ENNReal
      U' : UniformSpace α
      B' : Bornology α
      dist_self✝ : ∀ (x : α), Eq (Dist.dist x x) 0
      dist_comm✝ : ∀ (x y : α), Eq (Dist.dist x y) (Dist.dist y x)
      dist_triangle✝ : ∀ (x y z : α), LE.le (Dist.dist x z) (HAdd.hAdd (Dist.dist x  …
      hed' : ∀ (x y : α), Eq (ed' x y) (ENNReal.ofReal (Dist.dist x y))
      hU' : Eq (uniformity α) (iInf fun ε => iInf fun h => Filter.principal (setOf f …
      hB' : Eq (Bornology.cobounded α).sets (setOf fun s => Exists fun C => ∀ (x : α …
      ⊢ Eq U U'
    -/
  · exact UniformSpace.ext (hU.trans hU'.symm)
    /-
      🎉 no goals
    -/
    /-
      case mk.mk.e_toBornology
      α : Type u_3
      d : Dist α
      dist_self✝¹ : ∀ (x : α), Eq (Dist.dist x x) 0
      dist_comm✝¹ : ∀ (x y : α), Eq (Dist.dist x y) (Dist.dist y x)
      dist_triangle✝¹ : ∀ (x y z : α), LE.le (Dist.dist x z) (HAdd.hAdd (Dist.dist x …
      ed : α → α → ENNReal
      hed : ∀ (x y : α), Eq (ed x y) (ENNReal.ofReal (Dist.dist x y))
      U : UniformSpace α
      hU : Eq (uniformity α) (iInf fun ε => iInf fun h => Filter.principal (setOf fu …
      B : Bornology α
      hB : Eq (Bornology.cobounded α).sets (setOf fun s => Exists fun C => ∀ (x : α) …
      ed' : α → α → ENNReal
      U' : UniformSpace α
      B' : Bornology α
      dist_self✝ : ∀ (x : α), Eq (Dist.dist x x) 0
      dist_comm✝ : ∀ (x y : α), Eq (Dist.dist x y) (Dist.dist y x)
      dist_triangle✝ : ∀ (x y z : α), LE.le (Dist.dist x z) (HAdd.hAdd (Dist.dist x  …
      hed' : ∀ (x y : α), Eq (ed' x y) (ENNReal.ofReal (Dist.dist x y))
      hU' : Eq (uniformity α) (iInf fun ε => iInf fun h => Filter.principal (setOf f …
      hB' : Eq (Bornology.cobounded α).sets (setOf fun s => Exists fun C => ∀ (x : α …
      ⊢ Eq B B'
    -/
  · ext : 2
    /-
      case mk.mk.e_toBornology.h_cobounded.h
      α : Type u_3
      d : Dist α
      dist_self✝¹ : ∀ (x : α), Eq (Dist.dist x x) 0
      dist_comm✝¹ : ∀ (x y : α), Eq (Dist.dist x y) (Dist.dist y x)
      dist_triangle✝¹ : ∀ (x y z : α), LE.le (Dist.dist x z) (HAdd.hAdd (Dist.dist x …
      ed : α → α → ENNReal
      hed : ∀ (x y : α), Eq (ed x y) (ENNReal.ofReal (Dist.dist x y))
      U : UniformSpace α
      hU : Eq (uniformity α) (iInf fun ε => iInf fun h => Filter.principal (setOf fu …
      B : Bornology α
      hB : Eq (Bornology.cobounded α).sets (setOf fun s => Exists fun C => ∀ (x : α) …
      ed' : α → α → ENNReal
      U' : UniformSpace α
      B' : Bornology α
      dist_self✝ : ∀ (x : α), Eq (Dist.dist x x) 0
      dist_comm✝ : ∀ (x y : α), Eq (Dist.dist x y) (Dist.dist y x)
      dist_triangle✝ : ∀ (x y z : α), LE.le (Dist.dist x z) (HAdd.hAdd (Dist.dist x  …
      hed' : ∀ (x y : α), Eq (ed' x y) (ENNReal.ofReal (Dist.dist x y))
      hU' : Eq (uniformity α) (iInf fun ε => iInf fun h => Filter.principal (setOf f …
      hB' : Eq (Bornology.cobounded α).sets (setOf fun s => Exists fun C => ∀ (x : α …
      s✝ : Set α
      ⊢ Iff (Membership.mem (Bornology.cobounded α) s✝) (Membership.mem (Bornology.c …
    -/
    rw [← Filter.mem_sets, ← Filter.mem_sets, hB, hB']
    /-
      🎉 no goals
    -/


instance (priority := 200) PseudoMetricSpace.toEDist : EDist α :=
  ⟨PseudoMetricSpace.edist⟩


/-- Construct a pseudo-metric space structure whose underlying topological space structure
(definitionally) agrees which a pre-existing topology which is compatible with a given distance
function. -/
def PseudoMetricSpace.ofDistTopology {α : Type u} [TopologicalSpace α] (dist : α → α → ℝ)
    (dist_self : ∀ x : α, dist x x = 0) (dist_comm : ∀ x y : α, dist x y = dist y x)
    (dist_triangle : ∀ x y z : α, dist x z ≤ dist x y + dist y z)
    (H : ∀ s : Set α, IsOpen s ↔ ∀ x ∈ s, ∃ ε > 0, ∀ y, dist x y < ε → y ∈ s) :
    PseudoMetricSpace α :=
  { dist := dist
    dist_self := dist_self
    dist_comm := dist_comm
    dist_triangle := dist_triangle
    toUniformSpace :=
      (UniformSpace.ofDist dist dist_self dist_comm dist_triangle).replaceTopology <|
        TopologicalSpace.ext_iff.2 fun s ↦ (H s).trans <| forall₂_congr fun x _ ↦
          ((UniformSpace.hasBasis_ofFun (exists_gt (0 : ℝ)) dist dist_self dist_comm dist_triangle
            UniformSpace.ofDist_aux).comap (Prod.mk x)).mem_iff.symm
    uniformity_dist := rfl
    toBornology := Bornology.ofDist dist dist_comm dist_triangle
    cobounded_sets := rfl }


@[simp]
theorem dist_self (x : α) : dist x x = 0 :=
  PseudoMetricSpace.dist_self x


theorem dist_comm (x y : α) : dist x y = dist y x :=
  PseudoMetricSpace.dist_comm x y


theorem edist_dist (x y : α) : edist x y = ENNReal.ofReal (dist x y) :=
  PseudoMetricSpace.edist_dist x y


@[bound]
theorem dist_triangle (x y z : α) : dist x z ≤ dist x y + dist y z :=
  PseudoMetricSpace.dist_triangle x y z


theorem dist_triangle_left (x y z : α) : dist x y ≤ dist z x + dist z y := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    x y z : α
    ⊢ LE.le (Dist.dist x y) (HAdd.hAdd (Dist.dist z x) (Dist.dist z y))
  -/
  rw [dist_comm z]; apply dist_triangle
                    /-
                      🎉 no goals
                    -/


theorem dist_triangle_right (x y z : α) : dist x y ≤ dist x z + dist y z := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    x y z : α
    ⊢ LE.le (Dist.dist x y) (HAdd.hAdd (Dist.dist x z) (Dist.dist y z))
  -/
  rw [dist_comm y]; apply dist_triangle
                    /-
                      🎉 no goals
                    -/


theorem dist_triangle4 (x y z w : α) : dist x w ≤ dist x y + dist y z + dist z w :=
  calc
    dist x w ≤ dist x z + dist z w := dist_triangle x z w
    _ ≤ dist x y + dist y z + dist z w := add_le_add_right (dist_triangle x y z) _


theorem dist_triangle4_left (x₁ y₁ x₂ y₂ : α) :
    dist x₂ y₂ ≤ dist x₁ y₁ + (dist x₁ x₂ + dist y₁ y₂) := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    x₁ y₁ x₂ y₂ : α
    ⊢ LE.le (Dist.dist x₂ y₂) (HAdd.hAdd (Dist.dist x₁ y₁) (HAdd.hAdd (Dist.dist x …
  -/
  rw [add_left_comm, dist_comm x₁, ← add_assoc]
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    x₁ y₁ x₂ y₂ : α
    ⊢ LE.le (Dist.dist x₂ y₂) (HAdd.hAdd (HAdd.hAdd (Dist.dist x₂ x₁) (Dist.dist x …
  -/
  apply dist_triangle4
  /-
    🎉 no goals
  -/


theorem dist_triangle4_right (x₁ y₁ x₂ y₂ : α) :
    dist x₁ y₁ ≤ dist x₁ x₂ + dist y₁ y₂ + dist x₂ y₂ := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    x₁ y₁ x₂ y₂ : α
    ⊢ LE.le (Dist.dist x₁ y₁) (HAdd.hAdd (HAdd.hAdd (Dist.dist x₁ x₂) (Dist.dist y …
  -/
  rw [add_right_comm, dist_comm y₁]
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    x₁ y₁ x₂ y₂ : α
    ⊢ LE.le (Dist.dist x₁ y₁) (HAdd.hAdd (HAdd.hAdd (Dist.dist x₁ x₂) (Dist.dist x …
  -/
  apply dist_triangle4
  /-
    🎉 no goals
  -/


                                                           /-
                                                             α : Type u
                                                             inst✝ : PseudoMetricSpace α
                                                             ⊢ Eq (Function.swap Dist.dist) Dist.dist
                                                           -/
theorem swap_dist : Function.swap (@dist α _) = dist := by funext x y; exact dist_comm _ _
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


theorem abs_dist_sub_le (x y z : α) : |dist x z - dist y z| ≤ dist x y :=
  abs_sub_le_iff.2
    ⟨sub_le_iff_le_add.2 (dist_triangle _ _ _), sub_le_iff_le_add.2 (dist_triangle_left _ _ _)⟩


@[bound]
theorem dist_nonneg {x y : α} : 0 ≤ dist x y :=
  dist_nonneg' dist dist_self dist_comm dist_triangle


/-- Extension for the `positivity` tactic: distances are nonnegative. -/
@[positivity Dist.dist _ _]
def evalDist : PositivityExt where eval {u α} _zα _pα e := do
  match u, α, e with
  | 0, ~q(ℝ), ~q(@Dist.dist $β $inst $a $b) =>
    let _inst ← synthInstanceQ q(PseudoMetricSpace $β)
    assertInstancesCommute
    pure (.nonnegative q(dist_nonneg))
  | _, _, _ => throwError "not dist"


@[simp] theorem abs_dist {a b : α} : |dist a b| = dist a b := abs_of_nonneg dist_nonneg


/-- A version of `Dist` that takes value in `ℝ≥0`. -/
class NNDist (α : Type*) where
  nndist : α → α → ℝ≥0


/-- Distance as a nonnegative real number. -/
instance (priority := 100) PseudoMetricSpace.toNNDist : NNDist α :=
  ⟨fun a b => ⟨dist a b, dist_nonneg⟩⟩


/-- Express `dist` in terms of `nndist`-/
theorem dist_nndist (x y : α) : dist x y = nndist x y := rfl


@[simp, norm_cast]
theorem coe_nndist (x y : α) : ↑(nndist x y) = dist x y := rfl


/-- Express `edist` in terms of `nndist`-/
theorem edist_nndist (x y : α) : edist x y = nndist x y := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    x y : α
    ⊢ Eq (EDist.edist x y) ↑(NNDist.nndist x y)
  -/
  rw [edist_dist, dist_nndist, ENNReal.ofReal_coe_nnreal]
  /-
    🎉 no goals
  -/


/-- Express `nndist` in terms of `edist`-/
theorem nndist_edist (x y : α) : nndist x y = (edist x y).toNNReal := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    x y : α
    ⊢ Eq (NNDist.nndist x y) (EDist.edist x y).toNNReal
  -/
  simp [edist_nndist]
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem coe_nnreal_ennreal_nndist (x y : α) : ↑(nndist x y) = edist x y :=
  (edist_nndist x y).symm


@[simp, norm_cast]
theorem edist_lt_coe {x y : α} {c : ℝ≥0} : edist x y < c ↔ nndist x y < c := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    x y : α
    c : NNReal
    ⊢ Iff (LT.lt (EDist.edist x y) ↑c) (LT.lt (NNDist.nndist x y) c)
  -/
  rw [edist_nndist, ENNReal.coe_lt_coe]
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem edist_le_coe {x y : α} {c : ℝ≥0} : edist x y ≤ c ↔ nndist x y ≤ c := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    x y : α
    c : NNReal
    ⊢ Iff (LE.le (EDist.edist x y) ↑c) (LE.le (NNDist.nndist x y) c)
  -/
  rw [edist_nndist, ENNReal.coe_le_coe]
  /-
    🎉 no goals
  -/


/-- In a pseudometric space, the extended distance is always finite -/
theorem edist_lt_top {α : Type*} [PseudoMetricSpace α] (x y : α) : edist x y < ⊤ :=
  (edist_dist x y).symm ▸ ENNReal.ofReal_lt_top


/-- In a pseudometric space, the extended distance is always finite -/
theorem edist_ne_top (x y : α) : edist x y ≠ ⊤ :=
  (edist_lt_top x y).ne


/-- `nndist x x` vanishes -/
@[simp] theorem nndist_self (a : α) : nndist a a = 0 := NNReal.coe_eq_zero.1 (dist_self a)

-- Porting note: `dist_nndist` and `coe_nndist` moved up


@[simp, norm_cast]
theorem dist_lt_coe {x y : α} {c : ℝ≥0} : dist x y < c ↔ nndist x y < c :=
  Iff.rfl


@[simp, norm_cast]
theorem dist_le_coe {x y : α} {c : ℝ≥0} : dist x y ≤ c ↔ nndist x y ≤ c :=
  Iff.rfl


@[simp]
theorem edist_lt_ofReal {x y : α} {r : ℝ} : edist x y < ENNReal.ofReal r ↔ dist x y < r := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    x y : α
    r : Real
    ⊢ Iff (LT.lt (EDist.edist x y) (ENNReal.ofReal r)) (LT.lt (Dist.dist x y) r)
  -/
  rw [edist_dist, ENNReal.ofReal_lt_ofReal_iff_of_nonneg dist_nonneg]
  /-
    🎉 no goals
  -/


@[simp]
theorem edist_le_ofReal {x y : α} {r : ℝ} (hr : 0 ≤ r) :
    edist x y ≤ ENNReal.ofReal r ↔ dist x y ≤ r := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    x y : α
    r : Real
    hr : LE.le 0 r
    ⊢ Iff (LE.le (EDist.edist x y) (ENNReal.ofReal r)) (LE.le (Dist.dist x y) r)
  -/
  rw [edist_dist, ENNReal.ofReal_le_ofReal_iff hr]
  /-
    🎉 no goals
  -/


/-- Express `nndist` in terms of `dist`-/
theorem nndist_dist (x y : α) : nndist x y = Real.toNNReal (dist x y) := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    x y : α
    ⊢ Eq (NNDist.nndist x y) (Dist.dist x y).toNNReal
  -/
  rw [dist_nndist, Real.toNNReal_coe]
  /-
    🎉 no goals
  -/


theorem nndist_comm (x y : α) : nndist x y = nndist y x := NNReal.eq <| dist_comm x y


/-- Triangle inequality for the nonnegative distance -/
theorem nndist_triangle (x y z : α) : nndist x z ≤ nndist x y + nndist y z :=
  dist_triangle _ _ _


theorem nndist_triangle_left (x y z : α) : nndist x y ≤ nndist z x + nndist z y :=
  dist_triangle_left _ _ _


theorem nndist_triangle_right (x y z : α) : nndist x y ≤ nndist x z + nndist y z :=
  dist_triangle_right _ _ _


/-- Express `dist` in terms of `edist`-/
theorem dist_edist (x y : α) : dist x y = (edist x y).toReal := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    x y : α
    ⊢ Eq (Dist.dist x y) (EDist.edist x y).toReal
  -/
  rw [edist_dist, ENNReal.toReal_ofReal dist_nonneg]
  /-
    🎉 no goals
  -/


/-- `ball x ε` is the set of all points `y` with `dist y x < ε` -/
def ball (x : α) (ε : ℝ) : Set α :=
  { y | dist y x < ε }


@[simp]
theorem mem_ball : y ∈ ball x ε ↔ dist y x < ε :=
  Iff.rfl


                                                      /-
                                                        α : Type u
                                                        inst✝ : PseudoMetricSpace α
                                                        x y : α
                                                        ε : Real
                                                        ⊢ Iff (Membership.mem (Metric.ball x ε) y) (LT.lt (Dist.dist x y) ε)
                                                      -/
theorem mem_ball' : y ∈ ball x ε ↔ dist x y < ε := by rw [dist_comm, mem_ball]
                                                      /-
                                                        🎉 no goals
                                                      -/


theorem pos_of_mem_ball (hy : y ∈ ball x ε) : 0 < ε :=
  dist_nonneg.trans_lt hy


theorem mem_ball_self (h : 0 < ε) : x ∈ ball x ε := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    x : α
    ε : Real
    h : LT.lt 0 ε
    ⊢ Membership.mem (Metric.ball x ε) x
  -/
  rwa [mem_ball, dist_self]
  /-
    🎉 no goals
  -/


@[simp]
theorem nonempty_ball : (ball x ε).Nonempty ↔ 0 < ε :=
  ⟨fun ⟨_x, hx⟩ => pos_of_mem_ball hx, fun h => ⟨x, mem_ball_self h⟩⟩


@[simp]
theorem ball_eq_empty : ball x ε = ∅ ↔ ε ≤ 0 := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    x : α
    ε : Real
    ⊢ Iff (Eq (Metric.ball x ε) EmptyCollection.emptyCollection) (LE.le ε 0)
  -/
  rw [← not_nonempty_iff_eq_empty, nonempty_ball, not_lt]
  /-
    🎉 no goals
  -/


@[simp]
                                       /-
                                         α : Type u
                                         inst✝ : PseudoMetricSpace α
                                         x : α
                                         ⊢ Eq (Metric.ball x 0) EmptyCollection.emptyCollection
                                       -/
theorem ball_zero : ball x 0 = ∅ := by rw [ball_eq_empty]
                                       /-
                                         🎉 no goals
                                       -/


/-- If a point belongs to an open ball, then there is a strictly smaller radius whose ball also
contains it.

See also `exists_lt_subset_ball`. -/
theorem exists_lt_mem_ball_of_mem_ball (h : x ∈ ball y ε) : ∃ ε' < ε, x ∈ ball y ε' := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    x y : α
    ε : Real
    h : Membership.mem (Metric.ball y ε) x
    ⊢ Exists fun ε' => And (LT.lt ε' ε) (Membership.mem (Metric.ball y ε') x)
  -/
  simp only [mem_ball] at h ⊢
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    x y : α
    ε : Real
    h : LT.lt (Dist.dist x y) ε
    ⊢ Exists fun ε' => And (LT.lt ε' ε) (LT.lt (Dist.dist x y) ε')
  -/
  exact ⟨(dist x y + ε) / 2, by linarith, by linarith⟩
  /-
    🎉 no goals
  -/


theorem ball_eq_ball (ε : ℝ) (x : α) :
    UniformSpace.ball x { p | dist p.2 p.1 < ε } = Metric.ball x ε :=
  rfl


theorem ball_eq_ball' (ε : ℝ) (x : α) :
    UniformSpace.ball x { p | dist p.1 p.2 < ε } = Metric.ball x ε := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    ε : Real
    x : α
    ⊢ Eq (UniformSpace.ball x (setOf fun p => LT.lt (Dist.dist p.1 p.2) ε)) (Metri …
  -/
  ext
  /-
    case h
    α : Type u
    inst✝ : PseudoMetricSpace α
    ε : Real
    x x✝ : α
    ⊢ Iff (Membership.mem (UniformSpace.ball x (setOf fun p => LT.lt (Dist.dist p. …
  -/
  simp [dist_comm, UniformSpace.ball]
  /-
    🎉 no goals
  -/


@[simp]
theorem iUnion_ball_nat (x : α) : ⋃ n : ℕ, ball x n = univ :=
  iUnion_eq_univ_iff.2 fun y => exists_nat_gt (dist y x)


@[simp]
theorem iUnion_ball_nat_succ (x : α) : ⋃ n : ℕ, ball x (n + 1) = univ :=
  iUnion_eq_univ_iff.2 fun y => (exists_nat_gt (dist y x)).imp fun _ h => h.trans (lt_add_one _)


/-- `closedBall x ε` is the set of all points `y` with `dist y x ≤ ε` -/
def closedBall (x : α) (ε : ℝ) :=
  { y | dist y x ≤ ε }


@[simp] theorem mem_closedBall : y ∈ closedBall x ε ↔ dist y x ≤ ε := Iff.rfl


                                                                  /-
                                                                    α : Type u
                                                                    inst✝ : PseudoMetricSpace α
                                                                    x y : α
                                                                    ε : Real
                                                                    ⊢ Iff (Membership.mem (Metric.closedBall x ε) y) (LE.le (Dist.dist x y) ε)
                                                                  -/
theorem mem_closedBall' : y ∈ closedBall x ε ↔ dist x y ≤ ε := by rw [dist_comm, mem_closedBall]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


/-- `sphere x ε` is the set of all points `y` with `dist y x = ε` -/
def sphere (x : α) (ε : ℝ) := { y | dist y x = ε }


@[simp] theorem mem_sphere : y ∈ sphere x ε ↔ dist y x = ε := Iff.rfl


                                                          /-
                                                            α : Type u
                                                            inst✝ : PseudoMetricSpace α
                                                            x y : α
                                                            ε : Real
                                                            ⊢ Iff (Membership.mem (Metric.sphere x ε) y) (Eq (Dist.dist x y) ε)
                                                          -/
theorem mem_sphere' : y ∈ sphere x ε ↔ dist x y = ε := by rw [dist_comm, mem_sphere]
                                                          /-
                                                            🎉 no goals
                                                          -/


theorem ne_of_mem_sphere (h : y ∈ sphere x ε) (hε : ε ≠ 0) : y ≠ x :=
                               /-
                                 α : Type u
                                 inst✝ : PseudoMetricSpace α
                                 x y : α
                                 ε : Real
                                 h : Membership.mem (Metric.sphere x ε) y
                                 hε : Ne ε 0
                                 ⊢ Not (Membership.mem (Metric.sphere x ε) x)
                               -/
  ne_of_mem_of_not_mem h <| by simpa using hε.symm
                               /-
                                 🎉 no goals
                               -/


theorem nonneg_of_mem_sphere (hy : y ∈ sphere x ε) : 0 ≤ ε :=
  dist_nonneg.trans_eq hy


@[simp]
theorem sphere_eq_empty_of_neg (hε : ε < 0) : sphere x ε = ∅ :=
  Set.eq_empty_iff_forall_not_mem.mpr fun _y hy => (nonneg_of_mem_sphere hy).not_lt hε


theorem sphere_eq_empty_of_subsingleton [Subsingleton α] (hε : ε ≠ 0) : sphere x ε = ∅ :=
  Set.eq_empty_iff_forall_not_mem.mpr fun _ h => ne_of_mem_sphere h hε (Subsingleton.elim _ _)


instance sphere_isEmpty_of_subsingleton [Subsingleton α] [NeZero ε] : IsEmpty (sphere x ε) := by
  /-
    α : Type u
    β : Type v
    X : Type u_1
    ι : Type u_2
    inst✝² : PseudoMetricSpace α
    x y z : α
    δ ε ε₁ ε₂ : Real
    s : Set α
    inst✝¹ : Subsingleton α
    inst✝ : NeZero ε
    ⊢ IsEmpty ↑(Metric.sphere x ε)
  -/
  rw [sphere_eq_empty_of_subsingleton (NeZero.ne ε)]; infer_instance
                                                      /-
                                                        🎉 no goals
                                                      -/


theorem closedBall_eq_singleton_of_subsingleton [Subsingleton α] (h : 0 ≤ ε) :
    closedBall x ε = {x} := by
  /-
    α : Type u
    inst✝¹ : PseudoMetricSpace α
    x : α
    ε : Real
    inst✝ : Subsingleton α
    h : LE.le 0 ε
    ⊢ Eq (Metric.closedBall x ε) (Singleton.singleton x)
  -/
  ext x'
  /-
    case h
    α : Type u
    inst✝¹ : PseudoMetricSpace α
    x : α
    ε : Real
    inst✝ : Subsingleton α
    h : LE.le 0 ε
    x' : α
    ⊢ Iff (Membership.mem (Metric.closedBall x ε) x') (Membership.mem (Singleton.s …
  -/
  simpa [Subsingleton.allEq x x']
  /-
    🎉 no goals
  -/


theorem ball_eq_singleton_of_subsingleton [Subsingleton α] (h : 0 < ε) : ball x ε = {x} := by
  /-
    α : Type u
    inst✝¹ : PseudoMetricSpace α
    x : α
    ε : Real
    inst✝ : Subsingleton α
    h : LT.lt 0 ε
    ⊢ Eq (Metric.ball x ε) (Singleton.singleton x)
  -/
  ext x'
  /-
    case h
    α : Type u
    inst✝¹ : PseudoMetricSpace α
    x : α
    ε : Real
    inst✝ : Subsingleton α
    h : LT.lt 0 ε
    x' : α
    ⊢ Iff (Membership.mem (Metric.ball x ε) x') (Membership.mem (Singleton.singlet …
  -/
  simpa [Subsingleton.allEq x x']
  /-
    🎉 no goals
  -/


theorem mem_closedBall_self (h : 0 ≤ ε) : x ∈ closedBall x ε := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    x : α
    ε : Real
    h : LE.le 0 ε
    ⊢ Membership.mem (Metric.closedBall x ε) x
  -/
  rwa [mem_closedBall, dist_self]
  /-
    🎉 no goals
  -/


@[simp]
theorem nonempty_closedBall : (closedBall x ε).Nonempty ↔ 0 ≤ ε :=
  ⟨fun ⟨_x, hx⟩ => dist_nonneg.trans hx, fun h => ⟨x, mem_closedBall_self h⟩⟩


@[simp]
theorem closedBall_eq_empty : closedBall x ε = ∅ ↔ ε < 0 := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    x : α
    ε : Real
    ⊢ Iff (Eq (Metric.closedBall x ε) EmptyCollection.emptyCollection) (LT.lt ε 0)
  -/
  rw [← not_nonempty_iff_eq_empty, nonempty_closedBall, not_le]
  /-
    🎉 no goals
  -/


/-- Closed balls and spheres coincide when the radius is non-positive -/
theorem closedBall_eq_sphere_of_nonpos (hε : ε ≤ 0) : closedBall x ε = sphere x ε :=
  Set.ext fun _ => (hε.trans dist_nonneg).le_iff_eq


theorem ball_subset_closedBall : ball x ε ⊆ closedBall x ε := fun _y hy =>
  mem_closedBall.2 (le_of_lt hy)


theorem sphere_subset_closedBall : sphere x ε ⊆ closedBall x ε := fun _ => le_of_eq


lemma sphere_subset_ball {r R : ℝ} (h : r < R) : sphere x r ⊆ ball x R := fun _x hx ↦
  (mem_sphere.1 hx).trans_lt h


theorem closedBall_disjoint_ball (h : δ + ε ≤ dist x y) : Disjoint (closedBall x δ) (ball y ε) :=
  Set.disjoint_left.mpr fun _a ha1 ha2 =>
    (h.trans <| dist_triangle_left _ _ _).not_lt <| add_lt_add_of_le_of_lt ha1 ha2


theorem ball_disjoint_closedBall (h : δ + ε ≤ dist x y) : Disjoint (ball x δ) (closedBall y ε) :=
                                  /-
                                    α : Type u
                                    inst✝ : PseudoMetricSpace α
                                    x y : α
                                    δ ε : Real
                                    h : LE.le (HAdd.hAdd δ ε) (Dist.dist x y)
                                    ⊢ LE.le (HAdd.hAdd ε δ) (Dist.dist y x)
                                  -/
  (closedBall_disjoint_ball <| by rwa [add_comm, dist_comm]).symm
                                  /-
                                    🎉 no goals
                                  -/


theorem ball_disjoint_ball (h : δ + ε ≤ dist x y) : Disjoint (ball x δ) (ball y ε) :=
  (closedBall_disjoint_ball h).mono_left ball_subset_closedBall


theorem closedBall_disjoint_closedBall (h : δ + ε < dist x y) :
    Disjoint (closedBall x δ) (closedBall y ε) :=
  Set.disjoint_left.mpr fun _a ha1 ha2 =>
    h.not_le <| (dist_triangle_left _ _ _).trans <| add_le_add ha1 ha2


theorem sphere_disjoint_ball : Disjoint (sphere x ε) (ball x ε) :=
  Set.disjoint_left.mpr fun _y hy₁ hy₂ => absurd hy₁ <| ne_of_lt hy₂


@[simp]
theorem ball_union_sphere : ball x ε ∪ sphere x ε = closedBall x ε :=
  Set.ext fun _y => (@le_iff_lt_or_eq ℝ _ _ _).symm


@[simp]
theorem sphere_union_ball : sphere x ε ∪ ball x ε = closedBall x ε := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    x : α
    ε : Real
    ⊢ Eq (Union.union (Metric.sphere x ε) (Metric.ball x ε)) (Metric.closedBall x ε)
  -/
  rw [union_comm, ball_union_sphere]
  /-
    🎉 no goals
  -/


@[simp]
theorem closedBall_diff_sphere : closedBall x ε \ sphere x ε = ball x ε := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    x : α
    ε : Real
    ⊢ Eq (SDiff.sdiff (Metric.closedBall x ε) (Metric.sphere x ε)) (Metric.ball x ε)
  -/
  rw [← ball_union_sphere, Set.union_diff_cancel_right sphere_disjoint_ball.symm.le_bot]
  /-
    🎉 no goals
  -/


@[simp]
theorem closedBall_diff_ball : closedBall x ε \ ball x ε = sphere x ε := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    x : α
    ε : Real
    ⊢ Eq (SDiff.sdiff (Metric.closedBall x ε) (Metric.ball x ε)) (Metric.sphere x ε)
  -/
  rw [← ball_union_sphere, Set.union_diff_cancel_left sphere_disjoint_ball.symm.le_bot]
  /-
    🎉 no goals
  -/


                                                          /-
                                                            α : Type u
                                                            inst✝ : PseudoMetricSpace α
                                                            x y : α
                                                            ε : Real
                                                            ⊢ Iff (Membership.mem (Metric.ball y ε) x) (Membership.mem (Metric.ball x ε) y)
                                                          -/
theorem mem_ball_comm : x ∈ ball y ε ↔ y ∈ ball x ε := by rw [mem_ball', mem_ball]
                                                          /-
                                                            🎉 no goals
                                                          -/


theorem mem_closedBall_comm : x ∈ closedBall y ε ↔ y ∈ closedBall x ε := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    x y : α
    ε : Real
    ⊢ Iff (Membership.mem (Metric.closedBall y ε) x) (Membership.mem (Metric.close …
  -/
  rw [mem_closedBall', mem_closedBall]
  /-
    🎉 no goals
  -/


                                                                /-
                                                                  α : Type u
                                                                  inst✝ : PseudoMetricSpace α
                                                                  x y : α
                                                                  ε : Real
                                                                  ⊢ Iff (Membership.mem (Metric.sphere y ε) x) (Membership.mem (Metric.sphere x  …
                                                                -/
theorem mem_sphere_comm : x ∈ sphere y ε ↔ y ∈ sphere x ε := by rw [mem_sphere', mem_sphere]
                                                                /-
                                                                  🎉 no goals
                                                                -/


@[gcongr]
theorem ball_subset_ball (h : ε₁ ≤ ε₂) : ball x ε₁ ⊆ ball x ε₂ := fun _y yx =>
  lt_of_lt_of_le (mem_ball.1 yx) h


theorem closedBall_eq_bInter_ball : closedBall x ε = ⋂ δ > ε, ball x δ := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    x : α
    ε : Real
    ⊢ Eq (Metric.closedBall x ε) (Set.iInter fun δ => Set.iInter fun h => Metric.b …
  -/
  ext y; rw [mem_closedBall, ← forall_lt_iff_le', mem_iInter₂]; rfl
                                                                /-
                                                                  🎉 no goals
                                                                -/


theorem ball_subset_ball' (h : ε₁ + dist x y ≤ ε₂) : ball x ε₁ ⊆ ball y ε₂ := fun z hz =>
  calc
    dist z y ≤ dist z x + dist x y := dist_triangle _ _ _
    _ < ε₁ + dist x y := add_lt_add_right (mem_ball.1 hz) _
    _ ≤ ε₂ := h


@[gcongr]
theorem closedBall_subset_closedBall (h : ε₁ ≤ ε₂) : closedBall x ε₁ ⊆ closedBall x ε₂ :=
  fun _y (yx : _ ≤ ε₁) => le_trans yx h


theorem closedBall_subset_closedBall' (h : ε₁ + dist x y ≤ ε₂) :
    closedBall x ε₁ ⊆ closedBall y ε₂ := fun z hz =>
  calc
    dist z y ≤ dist z x + dist x y := dist_triangle _ _ _
    _ ≤ ε₁ + dist x y := add_le_add_right (mem_closedBall.1 hz) _
    _ ≤ ε₂ := h


theorem closedBall_subset_ball (h : ε₁ < ε₂) : closedBall x ε₁ ⊆ ball x ε₂ :=
  fun y (yh : dist y x ≤ ε₁) => lt_of_le_of_lt yh h


theorem closedBall_subset_ball' (h : ε₁ + dist x y < ε₂) :
    closedBall x ε₁ ⊆ ball y ε₂ := fun z hz =>
  calc
    dist z y ≤ dist z x + dist x y := dist_triangle _ _ _
    _ ≤ ε₁ + dist x y := add_le_add_right (mem_closedBall.1 hz) _
    _ < ε₂ := h


theorem dist_le_add_of_nonempty_closedBall_inter_closedBall
    (h : (closedBall x ε₁ ∩ closedBall y ε₂).Nonempty) : dist x y ≤ ε₁ + ε₂ :=
  let ⟨z, hz⟩ := h
  calc
    dist x y ≤ dist z x + dist z y := dist_triangle_left _ _ _
    _ ≤ ε₁ + ε₂ := add_le_add hz.1 hz.2


theorem dist_lt_add_of_nonempty_closedBall_inter_ball (h : (closedBall x ε₁ ∩ ball y ε₂).Nonempty) :
    dist x y < ε₁ + ε₂ :=
  let ⟨z, hz⟩ := h
  calc
    dist x y ≤ dist z x + dist z y := dist_triangle_left _ _ _
    _ < ε₁ + ε₂ := add_lt_add_of_le_of_lt hz.1 hz.2


theorem dist_lt_add_of_nonempty_ball_inter_closedBall (h : (ball x ε₁ ∩ closedBall y ε₂).Nonempty) :
    dist x y < ε₁ + ε₂ := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    x y : α
    ε₁ ε₂ : Real
    h : (Inter.inter (Metric.ball x ε₁) (Metric.closedBall y ε₂)).Nonempty
    ⊢ LT.lt (Dist.dist x y) (HAdd.hAdd ε₁ ε₂)
  -/
  rw [inter_comm] at h
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    x y : α
    ε₁ ε₂ : Real
    h : (Inter.inter (Metric.closedBall y ε₂) (Metric.ball x ε₁)).Nonempty
    ⊢ LT.lt (Dist.dist x y) (HAdd.hAdd ε₁ ε₂)
  -/
  rw [add_comm, dist_comm]
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    x y : α
    ε₁ ε₂ : Real
    h : (Inter.inter (Metric.closedBall y ε₂) (Metric.ball x ε₁)).Nonempty
    ⊢ LT.lt (Dist.dist y x) (HAdd.hAdd ε₂ ε₁)
  -/
  exact dist_lt_add_of_nonempty_closedBall_inter_ball h
  /-
    🎉 no goals
  -/


theorem dist_lt_add_of_nonempty_ball_inter_ball (h : (ball x ε₁ ∩ ball y ε₂).Nonempty) :
    dist x y < ε₁ + ε₂ :=
  dist_lt_add_of_nonempty_closedBall_inter_ball <|
    h.mono (inter_subset_inter ball_subset_closedBall Subset.rfl)


@[simp]
theorem iUnion_closedBall_nat (x : α) : ⋃ n : ℕ, closedBall x n = univ :=
  iUnion_eq_univ_iff.2 fun y => exists_nat_ge (dist y x)


theorem iUnion_inter_closedBall_nat (s : Set α) (x : α) : ⋃ n : ℕ, s ∩ closedBall x n = s := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Set α
    x : α
    ⊢ Eq (Set.iUnion fun n => Inter.inter s (Metric.closedBall x ↑n)) s
  -/
  rw [← inter_iUnion, iUnion_closedBall_nat, inter_univ]
  /-
    🎉 no goals
  -/


theorem ball_subset (h : dist x y ≤ ε₂ - ε₁) : ball x ε₁ ⊆ ball y ε₂ := fun z zx => by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    x y : α
    ε₁ ε₂ : Real
    h : LE.le (Dist.dist x y) (HSub.hSub ε₂ ε₁)
    z : α
    zx : Membership.mem (Metric.ball x ε₁) z
    ⊢ Membership.mem (Metric.ball y ε₂) z
  -/
  rw [← add_sub_cancel ε₁ ε₂]
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    x y : α
    ε₁ ε₂ : Real
    h : LE.le (Dist.dist x y) (HSub.hSub ε₂ ε₁)
    z : α
    zx : Membership.mem (Metric.ball x ε₁) z
    ⊢ Membership.mem (Metric.ball y (HAdd.hAdd ε₁ (HSub.hSub ε₂ ε₁))) z
  -/
  exact lt_of_le_of_lt (dist_triangle z x y) (add_lt_add_of_lt_of_le zx h)
  /-
    🎉 no goals
  -/


theorem ball_half_subset (y) (h : y ∈ ball x (ε / 2)) : ball y (ε / 2) ⊆ ball x ε :=
                    /-
                      α : Type u
                      inst✝ : PseudoMetricSpace α
                      x : α
                      ε : Real
                      y : α
                      h : Membership.mem (Metric.ball x (HDiv.hDiv ε 2)) y
                      ⊢ LE.le (Dist.dist y x) (HSub.hSub ε (HDiv.hDiv ε 2))
                    -/
  ball_subset <| by rw [sub_self_div_two]; exact le_of_lt h
                                           /-
                                             🎉 no goals
                                           -/


theorem exists_ball_subset_ball (h : y ∈ ball x ε) : ∃ ε' > 0, ball y ε' ⊆ ball x ε :=
                                     /-
                                       α : Type u
                                       inst✝ : PseudoMetricSpace α
                                       x y : α
                                       ε : Real
                                       h : Membership.mem (Metric.ball x ε) y
                                       ⊢ LE.le (Dist.dist y x) (HSub.hSub ε (HSub.hSub ε (Dist.dist y x)))
                                     -/
  ⟨_, sub_pos.2 h, ball_subset <| by rw [sub_sub_self]⟩
                                     /-
                                       🎉 no goals
                                     -/


/-- If a property holds for all points in closed balls of arbitrarily large radii, then it holds for
all points. -/
theorem forall_of_forall_mem_closedBall (p : α → Prop) (x : α)
    (H : ∃ᶠ R : ℝ in atTop, ∀ y ∈ closedBall x R, p y) (y : α) : p y := by
  obtain ⟨R, hR, h⟩ : ∃ R ≥ dist y x, ∀ z : α, z ∈ closedBall x R → p z :=
    frequently_iff.1 H (Ici_mem_atTop (dist y x))
  /-
    case intro.intro
    α : Type u
    inst✝ : PseudoMetricSpace α
    p : α → Prop
    x : α
    H : Filter.Frequently (fun R => ∀ (y : α), Membership.mem (Metric.closedBall x …
    y : α
    R : Real
    hR : GE.ge R (Dist.dist y x)
    h : ∀ (z : α), Membership.mem (Metric.closedBall x R) z → p z
    ⊢ p y
  -/
  exact h _ hR
  /-
    🎉 no goals
  -/


/-- If a property holds for all points in balls of arbitrarily large radii, then it holds for all
points. -/
theorem forall_of_forall_mem_ball (p : α → Prop) (x : α)
    (H : ∃ᶠ R : ℝ in atTop, ∀ y ∈ ball x R, p y) (y : α) : p y := by
  obtain ⟨R, hR, h⟩ : ∃ R > dist y x, ∀ z : α, z ∈ ball x R → p z :=
    frequently_iff.1 H (Ioi_mem_atTop (dist y x))
  /-
    case intro.intro
    α : Type u
    inst✝ : PseudoMetricSpace α
    p : α → Prop
    x : α
    H : Filter.Frequently (fun R => ∀ (y : α), Membership.mem (Metric.ball x R) y  …
    y : α
    R : Real
    hR : GT.gt R (Dist.dist y x)
    h : ∀ (z : α), Membership.mem (Metric.ball x R) z → p z
    ⊢ p y
  -/
  exact h _ hR
  /-
    🎉 no goals
  -/


theorem isBounded_iff {s : Set α} :
    IsBounded s ↔ ∃ C : ℝ, ∀ ⦃x⦄, x ∈ s → ∀ ⦃y⦄, y ∈ s → dist x y ≤ C := by
  rw [isBounded_def, ← Filter.mem_sets, @PseudoMetricSpace.cobounded_sets α, mem_setOf_eq,
    compl_compl]


theorem isBounded_iff_eventually {s : Set α} :
    IsBounded s ↔ ∀ᶠ C in atTop, ∀ ⦃x⦄, x ∈ s → ∀ ⦃y⦄, y ∈ s → dist x y ≤ C :=
  isBounded_iff.trans
    ⟨fun ⟨C, h⟩ => eventually_atTop.2 ⟨C, fun _C' hC' _x hx _y hy => (h hx hy).trans hC'⟩,
      Eventually.exists⟩


theorem isBounded_iff_exists_ge {s : Set α} (c : ℝ) :
    IsBounded s ↔ ∃ C, c ≤ C ∧ ∀ ⦃x⦄, x ∈ s → ∀ ⦃y⦄, y ∈ s → dist x y ≤ C :=
  ⟨fun h => ((eventually_ge_atTop c).and (isBounded_iff_eventually.1 h)).exists, fun h =>
    isBounded_iff.2 <| h.imp fun _ => And.right⟩


theorem isBounded_iff_nndist {s : Set α} :
    IsBounded s ↔ ∃ C : ℝ≥0, ∀ ⦃x⦄, x ∈ s → ∀ ⦃y⦄, y ∈ s → nndist x y ≤ C := by
  simp only [isBounded_iff_exists_ge 0, NNReal.exists, ← NNReal.coe_le_coe, ← dist_nndist,
    NNReal.coe_mk, exists_prop]


theorem toUniformSpace_eq :
    ‹PseudoMetricSpace α›.toUniformSpace = .ofDist dist dist_self dist_comm dist_triangle :=
  UniformSpace.ext PseudoMetricSpace.uniformity_dist


theorem uniformity_basis_dist :
    (𝓤 α).HasBasis (fun ε : ℝ => 0 < ε) fun ε => { p : α × α | dist p.1 p.2 < ε } := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    ⊢ (uniformity α).HasBasis (fun ε => LT.lt 0 ε) fun ε => setOf fun p => LT.lt ( …
  -/
  rw [toUniformSpace_eq]
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    ⊢ (uniformity α).HasBasis (fun ε => LT.lt 0 ε) fun ε => setOf fun p => LT.lt ( …
  -/
  exact UniformSpace.hasBasis_ofFun (exists_gt _) _ _ _ _ _
  /-
    🎉 no goals
  -/


/-- Given `f : β → ℝ`, if `f` sends `{i | p i}` to a set of positive numbers
accumulating to zero, then `f i`-neighborhoods of the diagonal form a basis of `𝓤 α`.

For specific bases see `uniformity_basis_dist`, `uniformity_basis_dist_inv_nat_succ`,
and `uniformity_basis_dist_inv_nat_pos`. -/
protected theorem mk_uniformity_basis {β : Type*} {p : β → Prop} {f : β → ℝ}
    (hf₀ : ∀ i, p i → 0 < f i) (hf : ∀ ⦃ε⦄, 0 < ε → ∃ i, p i ∧ f i ≤ ε) :
    (𝓤 α).HasBasis p fun i => { p : α × α | dist p.1 p.2 < f i } := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    β : Type u_3
    p : β → Prop
    f : β → Real
    hf₀ : ∀ (i : β), p i → LT.lt 0 (f i)
    hf : ∀ ⦃ε : Real⦄, LT.lt 0 ε → Exists fun i => And (p i) (LE.le (f i) ε)
    ⊢ (uniformity α).HasBasis p fun i => setOf fun p => LT.lt (Dist.dist p.1 p.2)  …
  -/
  refine ⟨fun s => uniformity_basis_dist.mem_iff.trans ?_⟩
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    β : Type u_3
    p : β → Prop
    f : β → Real
    hf₀ : ∀ (i : β), p i → LT.lt 0 (f i)
    hf : ∀ ⦃ε : Real⦄, LT.lt 0 ε → Exists fun i => And (p i) (LE.le (f i) ε)
    s : Set (Prod α α)
    ⊢ Iff (Exists fun i => And (LT.lt 0 i) (HasSubset.Subset (setOf fun p => LT.lt …
  -/
  constructor
    /-
      case mp
      α : Type u
      inst✝ : PseudoMetricSpace α
      β : Type u_3
      p : β → Prop
      f : β → Real
      hf₀ : ∀ (i : β), p i → LT.lt 0 (f i)
      hf : ∀ ⦃ε : Real⦄, LT.lt 0 ε → Exists fun i => And (p i) (LE.le (f i) ε)
      s : Set (Prod α α)
      ⊢ (Exists fun i => And (LT.lt 0 i) (HasSubset.Subset (setOf fun p => LT.lt (Di …
    -/
  · rintro ⟨ε, ε₀, hε⟩
    /-
      case mp.intro.intro
      α : Type u
      inst✝ : PseudoMetricSpace α
      β : Type u_3
      p : β → Prop
      f : β → Real
      hf₀ : ∀ (i : β), p i → LT.lt 0 (f i)
      hf : ∀ ⦃ε : Real⦄, LT.lt 0 ε → Exists fun i => And (p i) (LE.le (f i) ε)
      s : Set (Prod α α)
      ε : Real
      ε₀ : LT.lt 0 ε
      hε : HasSubset.Subset (setOf fun p => LT.lt (Dist.dist p.1 p.2) ε) s
      ⊢ Exists fun i => And (p i) (HasSubset.Subset (setOf fun p => LT.lt (Dist.dist …
    -/
    rcases hf ε₀ with ⟨i, hi, H⟩
    /-
      case mp.intro.intro.intro.intro
      α : Type u
      inst✝ : PseudoMetricSpace α
      β : Type u_3
      p : β → Prop
      f : β → Real
      hf₀ : ∀ (i : β), p i → LT.lt 0 (f i)
      hf : ∀ ⦃ε : Real⦄, LT.lt 0 ε → Exists fun i => And (p i) (LE.le (f i) ε)
      s : Set (Prod α α)
      ε : Real
      ε₀ : LT.lt 0 ε
      hε : HasSubset.Subset (setOf fun p => LT.lt (Dist.dist p.1 p.2) ε) s
      i : β
      hi : p i
      H : LE.le (f i) ε
      ⊢ Exists fun i => And (p i) (HasSubset.Subset (setOf fun p => LT.lt (Dist.dist …
    -/
    exact ⟨i, hi, fun x (hx : _ < _) => hε <| lt_of_lt_of_le hx H⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u
      inst✝ : PseudoMetricSpace α
      β : Type u_3
      p : β → Prop
      f : β → Real
      hf₀ : ∀ (i : β), p i → LT.lt 0 (f i)
      hf : ∀ ⦃ε : Real⦄, LT.lt 0 ε → Exists fun i => And (p i) (LE.le (f i) ε)
      s : Set (Prod α α)
      ⊢ (Exists fun i => And (p i) (HasSubset.Subset (setOf fun p => LT.lt (Dist.dis …
    -/
  · exact fun ⟨i, hi, H⟩ => ⟨f i, hf₀ i hi, H⟩
    /-
      🎉 no goals
    -/


theorem uniformity_basis_dist_rat :
    (𝓤 α).HasBasis (fun r : ℚ => 0 < r) fun r => { p : α × α | dist p.1 p.2 < r } :=
  Metric.mk_uniformity_basis (fun _ => Rat.cast_pos.2) fun _ε hε =>
    let ⟨r, hr0, hrε⟩ := exists_rat_btwn hε
    ⟨r, Rat.cast_pos.1 hr0, hrε.le⟩


theorem uniformity_basis_dist_inv_nat_succ :
    (𝓤 α).HasBasis (fun _ => True) fun n : ℕ => { p : α × α | dist p.1 p.2 < 1 / (↑n + 1) } :=
  Metric.mk_uniformity_basis (fun n _ => div_pos zero_lt_one <| Nat.cast_add_one_pos n) fun _ε ε0 =>
    (exists_nat_one_div_lt ε0).imp fun _n hn => ⟨trivial, le_of_lt hn⟩


theorem uniformity_basis_dist_inv_nat_pos :
    (𝓤 α).HasBasis (fun n : ℕ => 0 < n) fun n : ℕ => { p : α × α | dist p.1 p.2 < 1 / ↑n } :=
  Metric.mk_uniformity_basis (fun _ hn => div_pos zero_lt_one <| Nat.cast_pos.2 hn) fun _ ε0 =>
    let ⟨n, hn⟩ := exists_nat_one_div_lt ε0
    ⟨n + 1, Nat.succ_pos n, mod_cast hn.le⟩


theorem uniformity_basis_dist_pow {r : ℝ} (h0 : 0 < r) (h1 : r < 1) :
    (𝓤 α).HasBasis (fun _ : ℕ => True) fun n : ℕ => { p : α × α | dist p.1 p.2 < r ^ n } :=
  Metric.mk_uniformity_basis (fun _ _ => pow_pos h0 _) fun _ε ε0 =>
    let ⟨n, hn⟩ := exists_pow_lt_of_lt_one ε0 h1
    ⟨n, trivial, hn.le⟩


theorem uniformity_basis_dist_lt {R : ℝ} (hR : 0 < R) :
    (𝓤 α).HasBasis (fun r : ℝ => 0 < r ∧ r < R) fun r => { p : α × α | dist p.1 p.2 < r } :=
  Metric.mk_uniformity_basis (fun _ => And.left) fun r hr =>
    ⟨min r (R / 2), ⟨lt_min hr (half_pos hR), min_lt_iff.2 <| Or.inr (half_lt_self hR)⟩,
      min_le_left _ _⟩


/-- Given `f : β → ℝ`, if `f` sends `{i | p i}` to a set of positive numbers
accumulating to zero, then closed neighborhoods of the diagonal of sizes `{f i | p i}`
form a basis of `𝓤 α`.

Currently we have only one specific basis `uniformity_basis_dist_le` based on this constructor.
More can be easily added if needed in the future. -/
protected theorem mk_uniformity_basis_le {β : Type*} {p : β → Prop} {f : β → ℝ}
    (hf₀ : ∀ x, p x → 0 < f x) (hf : ∀ ε, 0 < ε → ∃ x, p x ∧ f x ≤ ε) :
    (𝓤 α).HasBasis p fun x => { p : α × α | dist p.1 p.2 ≤ f x } := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    β : Type u_3
    p : β → Prop
    f : β → Real
    hf₀ : ∀ (x : β), p x → LT.lt 0 (f x)
    hf : ∀ (ε : Real), LT.lt 0 ε → Exists fun x => And (p x) (LE.le (f x) ε)
    ⊢ (uniformity α).HasBasis p fun x => setOf fun p => LE.le (Dist.dist p.1 p.2)  …
  -/
  refine ⟨fun s => uniformity_basis_dist.mem_iff.trans ?_⟩
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    β : Type u_3
    p : β → Prop
    f : β → Real
    hf₀ : ∀ (x : β), p x → LT.lt 0 (f x)
    hf : ∀ (ε : Real), LT.lt 0 ε → Exists fun x => And (p x) (LE.le (f x) ε)
    s : Set (Prod α α)
    ⊢ Iff (Exists fun i => And (LT.lt 0 i) (HasSubset.Subset (setOf fun p => LT.lt …
  -/
  constructor
    /-
      case mp
      α : Type u
      inst✝ : PseudoMetricSpace α
      β : Type u_3
      p : β → Prop
      f : β → Real
      hf₀ : ∀ (x : β), p x → LT.lt 0 (f x)
      hf : ∀ (ε : Real), LT.lt 0 ε → Exists fun x => And (p x) (LE.le (f x) ε)
      s : Set (Prod α α)
      ⊢ (Exists fun i => And (LT.lt 0 i) (HasSubset.Subset (setOf fun p => LT.lt (Di …
    -/
  · rintro ⟨ε, ε₀, hε⟩
    /-
      case mp.intro.intro
      α : Type u
      inst✝ : PseudoMetricSpace α
      β : Type u_3
      p : β → Prop
      f : β → Real
      hf₀ : ∀ (x : β), p x → LT.lt 0 (f x)
      hf : ∀ (ε : Real), LT.lt 0 ε → Exists fun x => And (p x) (LE.le (f x) ε)
      s : Set (Prod α α)
      ε : Real
      ε₀ : LT.lt 0 ε
      hε : HasSubset.Subset (setOf fun p => LT.lt (Dist.dist p.1 p.2) ε) s
      ⊢ Exists fun i => And (p i) (HasSubset.Subset (setOf fun p => LE.le (Dist.dist …
    -/
    rcases exists_between ε₀ with ⟨ε', hε'⟩
    /-
      case mp.intro.intro.intro
      α : Type u
      inst✝ : PseudoMetricSpace α
      β : Type u_3
      p : β → Prop
      f : β → Real
      hf₀ : ∀ (x : β), p x → LT.lt 0 (f x)
      hf : ∀ (ε : Real), LT.lt 0 ε → Exists fun x => And (p x) (LE.le (f x) ε)
      s : Set (Prod α α)
      ε : Real
      ε₀ : LT.lt 0 ε
      hε : HasSubset.Subset (setOf fun p => LT.lt (Dist.dist p.1 p.2) ε) s
      ε' : Real
      hε' : And (LT.lt 0 ε') (LT.lt ε' ε)
      ⊢ Exists fun i => And (p i) (HasSubset.Subset (setOf fun p => LE.le (Dist.dist …
    -/
    rcases hf ε' hε'.1 with ⟨i, hi, H⟩
    /-
      case mp.intro.intro.intro.intro.intro
      α : Type u
      inst✝ : PseudoMetricSpace α
      β : Type u_3
      p : β → Prop
      f : β → Real
      hf₀ : ∀ (x : β), p x → LT.lt 0 (f x)
      hf : ∀ (ε : Real), LT.lt 0 ε → Exists fun x => And (p x) (LE.le (f x) ε)
      s : Set (Prod α α)
      ε : Real
      ε₀ : LT.lt 0 ε
      hε : HasSubset.Subset (setOf fun p => LT.lt (Dist.dist p.1 p.2) ε) s
      ε' : Real
      hε' : And (LT.lt 0 ε') (LT.lt ε' ε)
      i : β
      hi : p i
      H : LE.le (f i) ε'
      ⊢ Exists fun i => And (p i) (HasSubset.Subset (setOf fun p => LE.le (Dist.dist …
    -/
    exact ⟨i, hi, fun x (hx : _ ≤ _) => hε <| lt_of_le_of_lt (le_trans hx H) hε'.2⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u
      inst✝ : PseudoMetricSpace α
      β : Type u_3
      p : β → Prop
      f : β → Real
      hf₀ : ∀ (x : β), p x → LT.lt 0 (f x)
      hf : ∀ (ε : Real), LT.lt 0 ε → Exists fun x => And (p x) (LE.le (f x) ε)
      s : Set (Prod α α)
      ⊢ (Exists fun i => And (p i) (HasSubset.Subset (setOf fun p => LE.le (Dist.dis …
    -/
  · exact fun ⟨i, hi, H⟩ => ⟨f i, hf₀ i hi, fun x (hx : _ < _) => H (mem_setOf.2 hx.le)⟩
    /-
      🎉 no goals
    -/


/-- Constant size closed neighborhoods of the diagonal form a basis
of the uniformity filter. -/
theorem uniformity_basis_dist_le :
    (𝓤 α).HasBasis ((0 : ℝ) < ·) fun ε => { p : α × α | dist p.1 p.2 ≤ ε } :=
  Metric.mk_uniformity_basis_le (fun _ => id) fun ε ε₀ => ⟨ε, ε₀, le_refl ε⟩


theorem uniformity_basis_dist_le_pow {r : ℝ} (h0 : 0 < r) (h1 : r < 1) :
    (𝓤 α).HasBasis (fun _ : ℕ => True) fun n : ℕ => { p : α × α | dist p.1 p.2 ≤ r ^ n } :=
  Metric.mk_uniformity_basis_le (fun _ _ => pow_pos h0 _) fun _ε ε0 =>
    let ⟨n, hn⟩ := exists_pow_lt_of_lt_one ε0 h1
    ⟨n, trivial, hn.le⟩


theorem mem_uniformity_dist {s : Set (α × α)} :
    s ∈ 𝓤 α ↔ ∃ ε > 0, ∀ ⦃a b : α⦄, dist a b < ε → (a, b) ∈ s :=
  uniformity_basis_dist.mem_uniformity_iff


/-- A constant size neighborhood of the diagonal is an entourage. -/
theorem dist_mem_uniformity {ε : ℝ} (ε0 : 0 < ε) : { p : α × α | dist p.1 p.2 < ε } ∈ 𝓤 α :=
  mem_uniformity_dist.2 ⟨ε, ε0, fun _ _ ↦ id⟩


theorem uniformContinuous_iff [PseudoMetricSpace β] {f : α → β} :
    UniformContinuous f ↔ ∀ ε > 0, ∃ δ > 0, ∀ ⦃a b : α⦄, dist a b < δ → dist (f a) (f b) < ε :=
  uniformity_basis_dist.uniformContinuous_iff uniformity_basis_dist


theorem uniformContinuousOn_iff [PseudoMetricSpace β] {f : α → β} {s : Set α} :
    UniformContinuousOn f s ↔
      ∀ ε > 0, ∃ δ > 0, ∀ x ∈ s, ∀ y ∈ s, dist x y < δ → dist (f x) (f y) < ε :=
  Metric.uniformity_basis_dist.uniformContinuousOn_iff Metric.uniformity_basis_dist


theorem uniformContinuousOn_iff_le [PseudoMetricSpace β] {f : α → β} {s : Set α} :
    UniformContinuousOn f s ↔
      ∀ ε > 0, ∃ δ > 0, ∀ x ∈ s, ∀ y ∈ s, dist x y ≤ δ → dist (f x) (f y) ≤ ε :=
  Metric.uniformity_basis_dist_le.uniformContinuousOn_iff Metric.uniformity_basis_dist_le


theorem nhds_basis_ball : (𝓝 x).HasBasis (0 < ·) (ball x) :=
  nhds_basis_uniformity uniformity_basis_dist


theorem mem_nhds_iff : s ∈ 𝓝 x ↔ ∃ ε > 0, ball x ε ⊆ s :=
  nhds_basis_ball.mem_iff


theorem eventually_nhds_iff {p : α → Prop} :
    (∀ᶠ y in 𝓝 x, p y) ↔ ∃ ε > 0, ∀ ⦃y⦄, dist y x < ε → p y :=
  mem_nhds_iff


theorem eventually_nhds_iff_ball {p : α → Prop} :
    (∀ᶠ y in 𝓝 x, p y) ↔ ∃ ε > 0, ∀ y ∈ ball x ε, p y :=
  mem_nhds_iff


/-- A version of `Filter.eventually_prod_iff` where the first filter consists of neighborhoods
in a pseudo-metric space. -/
theorem eventually_nhds_prod_iff {f : Filter ι} {x₀ : α} {p : α × ι → Prop} :
    (∀ᶠ x in 𝓝 x₀ ×ˢ f, p x) ↔ ∃ ε > (0 : ℝ), ∃ pa : ι → Prop, (∀ᶠ i in f, pa i) ∧
      ∀ ⦃x⦄, dist x x₀ < ε → ∀ ⦃i⦄, pa i → p (x, i) := by
  /-
    α : Type u
    ι : Type u_2
    inst✝ : PseudoMetricSpace α
    f : Filter ι
    x₀ : α
    p : Prod α ι → Prop
    ⊢ Iff (Filter.Eventually (fun x => p x) (SProd.sprod (nhds x₀) f)) (Exists fun …
  -/
  refine (nhds_basis_ball.prod f.basis_sets).eventually_iff.trans ?_
  /-
    α : Type u
    ι : Type u_2
    inst✝ : PseudoMetricSpace α
    f : Filter ι
    x₀ : α
    p : Prod α ι → Prop
    ⊢ Iff (Exists fun i => And (And (LT.lt 0 i.1) (Membership.mem f i.2)) (∀ ⦃x :  …
  -/
  simp only [Prod.exists, forall_prod_set, id, mem_ball, and_assoc, exists_and_left, and_imp]
  /-
    α : Type u
    ι : Type u_2
    inst✝ : PseudoMetricSpace α
    f : Filter ι
    x₀ : α
    p : Prod α ι → Prop
    ⊢ Iff (Exists fun a => And (LT.lt 0 a) (Exists fun x => And (Membership.mem f  …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- A version of `Filter.eventually_prod_iff` where the second filter consists of neighborhoods
in a pseudo-metric space. -/
theorem eventually_prod_nhds_iff {f : Filter ι} {x₀ : α} {p : ι × α → Prop} :
    (∀ᶠ x in f ×ˢ 𝓝 x₀, p x) ↔ ∃ pa : ι → Prop, (∀ᶠ i in f, pa i) ∧
      ∃ ε > 0, ∀ ⦃i⦄, pa i → ∀ ⦃x⦄, dist x x₀ < ε → p (i, x) := by
  /-
    α : Type u
    ι : Type u_2
    inst✝ : PseudoMetricSpace α
    f : Filter ι
    x₀ : α
    p : Prod ι α → Prop
    ⊢ Iff (Filter.Eventually (fun x => p x) (SProd.sprod f (nhds x₀))) (Exists fun …
  -/
  rw [eventually_swap_iff, Metric.eventually_nhds_prod_iff]
  /-
    α : Type u
    ι : Type u_2
    inst✝ : PseudoMetricSpace α
    f : Filter ι
    x₀ : α
    p : Prod ι α → Prop
    ⊢ Iff (Exists fun ε => And (GT.gt ε 0) (Exists fun pa => And (Filter.Eventuall …
  -/
  constructor <;>
      /-
        case mp
        α : Type u
        ι : Type u_2
        inst✝ : PseudoMetricSpace α
        f : Filter ι
        x₀ : α
        p : Prod ι α → Prop
        ⊢ (Exists fun ε => And (GT.gt ε 0) (Exists fun pa => And (Filter.Eventually (f …
      -/
      /-
        case mp.intro.intro.intro.intro
        α : Type u
        ι : Type u_2
        inst✝ : PseudoMetricSpace α
        f : Filter ι
        x₀ : α
        p : Prod ι α → Prop
        a1 : Real
        a2 : GT.gt a1 0
        a3 : ι → Prop
        a4 : Filter.Eventually (fun i => a3 i) f
        a5 : ∀ ⦃x : α⦄, LT.lt (Dist.dist x x₀) a1 → ∀ ⦃i : ι⦄, a3 i → p { fst := x, sn …
        ⊢ Exists fun pa => And (Filter.Eventually (fun i => pa i) f) (Exists fun ε =>  …
      -/
      /-
        🎉 no goals
      -/
      /-
        case mpr.intro.intro.intro.intro
        α : Type u
        ι : Type u_2
        inst✝ : PseudoMetricSpace α
        f : Filter ι
        x₀ : α
        p : Prod ι α → Prop
        a1 : ι → Prop
        a2 : Filter.Eventually (fun i => a1 i) f
        a3 : Real
        a4 : GT.gt a3 0
        a5 : ∀ ⦃i : ι⦄, a1 i → ∀ ⦃x : α⦄, LT.lt (Dist.dist x x₀) a3 → p { fst := i, sn …
        ⊢ Exists fun ε => And (GT.gt ε 0) (Exists fun pa => And (Filter.Eventually (fu …
      -/
      exact ⟨a3, a4, a1, a2, fun _ b1 b2 b3 => a5 b3 b1⟩
      /-
        🎉 no goals
      -/


theorem nhds_basis_closedBall : (𝓝 x).HasBasis (fun ε : ℝ => 0 < ε) (closedBall x) :=
  nhds_basis_uniformity uniformity_basis_dist_le


theorem nhds_basis_ball_inv_nat_succ :
    (𝓝 x).HasBasis (fun _ => True) fun n : ℕ => ball x (1 / (↑n + 1)) :=
  nhds_basis_uniformity uniformity_basis_dist_inv_nat_succ


theorem nhds_basis_ball_inv_nat_pos :
    (𝓝 x).HasBasis (fun n => 0 < n) fun n : ℕ => ball x (1 / ↑n) :=
  nhds_basis_uniformity uniformity_basis_dist_inv_nat_pos


theorem nhds_basis_ball_pow {r : ℝ} (h0 : 0 < r) (h1 : r < 1) :
    (𝓝 x).HasBasis (fun _ => True) fun n : ℕ => ball x (r ^ n) :=
  nhds_basis_uniformity (uniformity_basis_dist_pow h0 h1)


theorem nhds_basis_closedBall_pow {r : ℝ} (h0 : 0 < r) (h1 : r < 1) :
    (𝓝 x).HasBasis (fun _ => True) fun n : ℕ => closedBall x (r ^ n) :=
  nhds_basis_uniformity (uniformity_basis_dist_le_pow h0 h1)


theorem isOpen_iff : IsOpen s ↔ ∀ x ∈ s, ∃ ε > 0, ball x ε ⊆ s := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Set α
    ⊢ Iff (IsOpen s) (∀ (x : α), Membership.mem s x → Exists fun ε => And (GT.gt ε …
  -/
  simp only [isOpen_iff_mem_nhds, mem_nhds_iff]
  /-
    🎉 no goals
  -/


theorem isOpen_ball : IsOpen (ball x ε) :=
  isOpen_iff.2 fun _ => exists_ball_subset_ball


theorem ball_mem_nhds (x : α) {ε : ℝ} (ε0 : 0 < ε) : ball x ε ∈ 𝓝 x :=
  isOpen_ball.mem_nhds (mem_ball_self ε0)


theorem closedBall_mem_nhds (x : α) {ε : ℝ} (ε0 : 0 < ε) : closedBall x ε ∈ 𝓝 x :=
  mem_of_superset (ball_mem_nhds x ε0) ball_subset_closedBall


theorem closedBall_mem_nhds_of_mem {x c : α} {ε : ℝ} (h : x ∈ ball c ε) : closedBall c ε ∈ 𝓝 x :=
  mem_of_superset (isOpen_ball.mem_nhds h) ball_subset_closedBall


theorem nhdsWithin_basis_ball {s : Set α} :
    (𝓝[s] x).HasBasis (fun ε : ℝ => 0 < ε) fun ε => ball x ε ∩ s :=
  nhdsWithin_hasBasis nhds_basis_ball s


theorem mem_nhdsWithin_iff {t : Set α} : s ∈ 𝓝[t] x ↔ ∃ ε > 0, ball x ε ∩ t ⊆ s :=
  nhdsWithin_basis_ball.mem_iff


theorem tendsto_nhdsWithin_nhdsWithin [PseudoMetricSpace β] {t : Set β} {f : α → β} {a b} :
    Tendsto f (𝓝[s] a) (𝓝[t] b) ↔
      ∀ ε > 0, ∃ δ > 0, ∀ ⦃x : α⦄, x ∈ s → dist x a < δ → f x ∈ t ∧ dist (f x) b < ε :=
  (nhdsWithin_basis_ball.tendsto_iff nhdsWithin_basis_ball).trans <| by
    /-
      α : Type u
      β : Type v
      inst✝¹ : PseudoMetricSpace α
      s : Set α
      inst✝ : PseudoMetricSpace β
      t : Set β
      f : α → β
      a : α
      b : β
      ⊢ Iff (∀ (ib : Real), LT.lt 0 ib → Exists fun ia => And (LT.lt 0 ia) (∀ (x : α …
    -/
    simp only [inter_comm _ s, inter_comm _ t, mem_inter_iff, and_imp, gt_iff_lt, mem_ball]
    /-
      🎉 no goals
    -/


theorem tendsto_nhdsWithin_nhds [PseudoMetricSpace β] {f : α → β} {a b} :
    Tendsto f (𝓝[s] a) (𝓝 b) ↔
      ∀ ε > 0, ∃ δ > 0, ∀ ⦃x : α⦄, x ∈ s → dist x a < δ → dist (f x) b < ε := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : PseudoMetricSpace α
    s : Set α
    inst✝ : PseudoMetricSpace β
    f : α → β
    a : α
    b : β
    ⊢ Iff (Filter.Tendsto f (nhdsWithin a s) (nhds b)) (∀ (ε : Real), GT.gt ε 0 →  …
  -/
  rw [← nhdsWithin_univ b, tendsto_nhdsWithin_nhdsWithin]
  /-
    α : Type u
    β : Type v
    inst✝¹ : PseudoMetricSpace α
    s : Set α
    inst✝ : PseudoMetricSpace β
    f : α → β
    a : α
    b : β
    ⊢ Iff (∀ (ε : Real), GT.gt ε 0 → Exists fun δ => And (GT.gt δ 0) (∀ ⦃x : α⦄, M …
  -/
  simp only [mem_univ, true_and]
  /-
    🎉 no goals
  -/


theorem tendsto_nhds_nhds [PseudoMetricSpace β] {f : α → β} {a b} :
    Tendsto f (𝓝 a) (𝓝 b) ↔ ∀ ε > 0, ∃ δ > 0, ∀ ⦃x : α⦄, dist x a < δ → dist (f x) b < ε :=
  nhds_basis_ball.tendsto_iff nhds_basis_ball


theorem continuousAt_iff [PseudoMetricSpace β] {f : α → β} {a : α} :
    ContinuousAt f a ↔ ∀ ε > 0, ∃ δ > 0, ∀ ⦃x : α⦄, dist x a < δ → dist (f x) (f a) < ε := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : PseudoMetricSpace α
    inst✝ : PseudoMetricSpace β
    f : α → β
    a : α
    ⊢ Iff (ContinuousAt f a) (∀ (ε : Real), GT.gt ε 0 → Exists fun δ => And (GT.gt …
  -/
  rw [ContinuousAt, tendsto_nhds_nhds]
  /-
    🎉 no goals
  -/


theorem continuousWithinAt_iff [PseudoMetricSpace β] {f : α → β} {a : α} {s : Set α} :
    ContinuousWithinAt f s a ↔
      ∀ ε > 0, ∃ δ > 0, ∀ ⦃x : α⦄, x ∈ s → dist x a < δ → dist (f x) (f a) < ε := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : PseudoMetricSpace α
    inst✝ : PseudoMetricSpace β
    f : α → β
    a : α
    s : Set α
    ⊢ Iff (ContinuousWithinAt f s a) (∀ (ε : Real), GT.gt ε 0 → Exists fun δ => An …
  -/
  rw [ContinuousWithinAt, tendsto_nhdsWithin_nhds]
  /-
    🎉 no goals
  -/


theorem continuousOn_iff [PseudoMetricSpace β] {f : α → β} {s : Set α} :
    ContinuousOn f s ↔ ∀ b ∈ s, ∀ ε > 0, ∃ δ > 0, ∀ a ∈ s, dist a b < δ → dist (f a) (f b) < ε := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : PseudoMetricSpace α
    inst✝ : PseudoMetricSpace β
    f : α → β
    s : Set α
    ⊢ Iff (ContinuousOn f s) (∀ (b : α), Membership.mem s b → ∀ (ε : Real), GT.gt  …
  -/
  simp [ContinuousOn, continuousWithinAt_iff]
  /-
    🎉 no goals
  -/


theorem continuous_iff [PseudoMetricSpace β] {f : α → β} :
    Continuous f ↔ ∀ b, ∀ ε > 0, ∃ δ > 0, ∀ a, dist a b < δ → dist (f a) (f b) < ε :=
  continuous_iff_continuousAt.trans <| forall_congr' fun _ => tendsto_nhds_nhds


theorem tendsto_nhds {f : Filter β} {u : β → α} {a : α} :
    Tendsto u f (𝓝 a) ↔ ∀ ε > 0, ∀ᶠ x in f, dist (u x) a < ε :=
  nhds_basis_ball.tendsto_right_iff


theorem continuousAt_iff' [TopologicalSpace β] {f : β → α} {b : β} :
    ContinuousAt f b ↔ ∀ ε > 0, ∀ᶠ x in 𝓝 b, dist (f x) (f b) < ε := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : PseudoMetricSpace α
    inst✝ : TopologicalSpace β
    f : β → α
    b : β
    ⊢ Iff (ContinuousAt f b) (∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun x = …
  -/
  rw [ContinuousAt, tendsto_nhds]
  /-
    🎉 no goals
  -/


theorem continuousWithinAt_iff' [TopologicalSpace β] {f : β → α} {b : β} {s : Set β} :
    ContinuousWithinAt f s b ↔ ∀ ε > 0, ∀ᶠ x in 𝓝[s] b, dist (f x) (f b) < ε := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : PseudoMetricSpace α
    inst✝ : TopologicalSpace β
    f : β → α
    b : β
    s : Set β
    ⊢ Iff (ContinuousWithinAt f s b) (∀ (ε : Real), GT.gt ε 0 → Filter.Eventually  …
  -/
  rw [ContinuousWithinAt, tendsto_nhds]
  /-
    🎉 no goals
  -/


theorem continuousOn_iff' [TopologicalSpace β] {f : β → α} {s : Set β} :
    ContinuousOn f s ↔ ∀ b ∈ s, ∀ ε > 0, ∀ᶠ x in 𝓝[s] b, dist (f x) (f b) < ε := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : PseudoMetricSpace α
    inst✝ : TopologicalSpace β
    f : β → α
    s : Set β
    ⊢ Iff (ContinuousOn f s) (∀ (b : β), Membership.mem s b → ∀ (ε : Real), GT.gt  …
  -/
  simp [ContinuousOn, continuousWithinAt_iff']
  /-
    🎉 no goals
  -/


theorem continuous_iff' [TopologicalSpace β] {f : β → α} :
    Continuous f ↔ ∀ (a), ∀ ε > 0, ∀ᶠ x in 𝓝 a, dist (f x) (f a) < ε :=
  continuous_iff_continuousAt.trans <| forall_congr' fun _ => tendsto_nhds


theorem tendsto_atTop [Nonempty β] [SemilatticeSup β] {u : β → α} {a : α} :
    Tendsto u atTop (𝓝 a) ↔ ∀ ε > 0, ∃ N, ∀ n ≥ N, dist (u n) a < ε :=
  (atTop_basis.tendsto_iff nhds_basis_ball).trans <| by
    /-
      α : Type u
      β : Type v
      inst✝² : PseudoMetricSpace α
      inst✝¹ : Nonempty β
      inst✝ : SemilatticeSup β
      u : β → α
      a : α
      ⊢ Iff (∀ (ib : Real), LT.lt 0 ib → Exists fun ia => And True (∀ (x : β), Membe …
    -/
    simp only [true_and, mem_ball, mem_Ici]
    /-
      🎉 no goals
    -/


/-- A variant of `tendsto_atTop` that
uses `∃ N, ∀ n > N, ...` rather than `∃ N, ∀ n ≥ N, ...`
-/
theorem tendsto_atTop' [Nonempty β] [SemilatticeSup β] [NoMaxOrder β] {u : β → α} {a : α} :
    Tendsto u atTop (𝓝 a) ↔ ∀ ε > 0, ∃ N, ∀ n > N, dist (u n) a < ε :=
  (atTop_basis_Ioi.tendsto_iff nhds_basis_ball).trans <| by
    /-
      α : Type u
      β : Type v
      inst✝³ : PseudoMetricSpace α
      inst✝² : Nonempty β
      inst✝¹ : SemilatticeSup β
      inst✝ : NoMaxOrder β
      u : β → α
      a : α
      ⊢ Iff (∀ (ib : Real), LT.lt 0 ib → Exists fun ia => And True (∀ (x : β), Membe …
    -/
    simp only [true_and, gt_iff_lt, mem_Ioi, mem_ball]
    /-
      🎉 no goals
    -/


theorem isOpen_singleton_iff {α : Type*} [PseudoMetricSpace α] {x : α} :
    IsOpen ({x} : Set α) ↔ ∃ ε > 0, ∀ y, dist y x < ε → y = x := by
  /-
    α : Type u_3
    inst✝ : PseudoMetricSpace α
    x : α
    ⊢ Iff (IsOpen (Singleton.singleton x)) (Exists fun ε => And (GT.gt ε 0) (∀ (y  …
  -/
  simp [isOpen_iff, subset_singleton_iff, mem_ball]
  /-
    🎉 no goals
  -/


theorem _root_.Dense.exists_dist_lt {s : Set α} (hs : Dense s) (x : α) {ε : ℝ} (hε : 0 < ε) :
    ∃ y ∈ s, dist x y < ε := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Set α
    hs : Dense s
    x : α
    ε : Real
    hε : LT.lt 0 ε
    ⊢ Exists fun y => And (Membership.mem s y) (LT.lt (Dist.dist x y) ε)
  -/
  have : (ball x ε).Nonempty := by simp [hε]
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Set α
    hs : Dense s
    x : α
    ε : Real
    hε : LT.lt 0 ε
    this : (Metric.ball x ε).Nonempty
    ⊢ Exists fun y => And (Membership.mem s y) (LT.lt (Dist.dist x y) ε)
  -/
  simpa only [mem_ball'] using hs.exists_mem_open isOpen_ball this
  /-
    🎉 no goals
  -/


nonrec theorem _root_.DenseRange.exists_dist_lt {β : Type*} {f : β → α} (hf : DenseRange f) (x : α)
    {ε : ℝ} (hε : 0 < ε) : ∃ y, dist x (f y) < ε :=
  exists_range_iff.1 (hf.exists_dist_lt x hε)


/-- (Pseudo) metric space has discrete `UniformSpace` structure
iff the distances between distinct points are uniformly bounded away from zero. -/
protected lemma uniformSpace_eq_bot :
    ‹PseudoMetricSpace α›.toUniformSpace = ⊥ ↔
      ∃ r : ℝ, 0 < r ∧ Pairwise (r ≤ dist · · : α → α → Prop) := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    ⊢ Iff (Eq PseudoMetricSpace.toUniformSpace Bot.bot) (Exists fun r => And (LT.l …
  -/
  simp only [uniformity_basis_dist.uniformSpace_eq_bot, mem_setOf_eq, not_lt]
  /-
    🎉 no goals
  -/


/-- If the distances between distinct points in a (pseudo) metric space
are uniformly bounded away from zero, then the space has discrete topology. -/
lemma DiscreteTopology.of_forall_le_dist {α} [PseudoMetricSpace α] {r : ℝ} (hpos : 0 < r)
    (hr : Pairwise (r ≤ dist · · : α → α → Prop)) : DiscreteTopology α :=
      /-
        α : Type u_3
        inst✝ : PseudoMetricSpace α
        r : Real
        hpos : LT.lt 0 r
        hr : Pairwise fun x1 x2 => LE.le r (Dist.dist x1 x2)
        ⊢ Eq UniformSpace.toTopologicalSpace Bot.bot
      -/
  ⟨by rw [Metric.uniformSpace_eq_bot.2 ⟨r, hpos, hr⟩, UniformSpace.toTopologicalSpace_bot]⟩
      /-
        🎉 no goals
      -/

/- Instantiate a pseudometric space as a pseudoemetric space. Before we can state the instance,
we need to show that the uniform structure coming from the edistance and the
distance coincide. -/


theorem Metric.uniformity_edist_aux {α} (d : α → α → ℝ≥0) :
    ⨅ ε > (0 : ℝ), 𝓟 { p : α × α | ↑(d p.1 p.2) < ε } =
      ⨅ ε > (0 : ℝ≥0∞), 𝓟 { p : α × α | ↑(d p.1 p.2) < ε } := by
  /-
    α : Type u_3
    d : α → α → NNReal
    ⊢ Eq (iInf fun ε => iInf fun h => Filter.principal (setOf fun p => LT.lt (↑(d  …
  -/
  simp only [le_antisymm_iff, le_iInf_iff, le_principal_iff]
  /-
    α : Type u_3
    d : α → α → NNReal
    ⊢ And (∀ (i : ENNReal), GT.gt i 0 → Membership.mem (iInf fun ε => iInf fun h = …
  -/
  refine ⟨fun ε hε => ?_, fun ε hε => ?_⟩
    /-
      case refine_1
      α : Type u_3
      d : α → α → NNReal
      ε : ENNReal
      hε : GT.gt ε 0
      ⊢ Membership.mem (iInf fun ε => iInf fun h => Filter.principal (setOf fun p => …
    -/
  · rcases ENNReal.lt_iff_exists_nnreal_btwn.1 hε with ⟨ε', ε'0, ε'ε⟩
    /-
      case refine_1.intro.intro
      α : Type u_3
      d : α → α → NNReal
      ε : ENNReal
      hε : GT.gt ε 0
      ε' : NNReal
      ε'0 : LT.lt 0 ↑ε'
      ε'ε : LT.lt (↑ε') ε
      ⊢ Membership.mem (iInf fun ε => iInf fun h => Filter.principal (setOf fun p => …
    -/
    refine mem_iInf_of_mem (ε' : ℝ) (mem_iInf_of_mem (ENNReal.coe_pos.1 ε'0) ?_)
    /-
      case refine_1.intro.intro
      α : Type u_3
      d : α → α → NNReal
      ε : ENNReal
      hε : GT.gt ε 0
      ε' : NNReal
      ε'0 : LT.lt 0 ↑ε'
      ε'ε : LT.lt (↑ε') ε
      ⊢ Membership.mem (Filter.principal (setOf fun p => LT.lt ↑(d p.1 p.2) ↑ε')) (s …
    -/
    exact fun x hx => lt_trans (ENNReal.coe_lt_coe.2 hx) ε'ε
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_3
      d : α → α → NNReal
      ε : Real
      hε : GT.gt ε 0
      ⊢ Membership.mem (iInf fun ε => iInf fun h => Filter.principal (setOf fun p => …
    -/
  · lift ε to ℝ≥0 using le_of_lt hε
    /-
      case refine_2.intro
      α : Type u_3
      d : α → α → NNReal
      ε : NNReal
      hε : GT.gt (↑ε) 0
      ⊢ Membership.mem (iInf fun ε => iInf fun h => Filter.principal (setOf fun p => …
    -/
    refine mem_iInf_of_mem (ε : ℝ≥0∞) (mem_iInf_of_mem (ENNReal.coe_pos.2 hε) ?_)
    /-
      case refine_2.intro
      α : Type u_3
      d : α → α → NNReal
      ε : NNReal
      hε : GT.gt (↑ε) 0
      ⊢ Membership.mem (Filter.principal (setOf fun p => LT.lt ↑(d p.1 p.2) ↑ε)) (se …
    -/
    exact fun _ => ENNReal.coe_lt_coe.1
    /-
      🎉 no goals
    -/


theorem Metric.uniformity_edist : 𝓤 α = ⨅ ε > 0, 𝓟 { p : α × α | edist p.1 p.2 < ε } := by
  simp only [PseudoMetricSpace.uniformity_dist, dist_nndist, edist_nndist,
    Metric.uniformity_edist_aux]

-- see Note [lower instance priority]

/-- A pseudometric space induces a pseudoemetric space -/
instance (priority := 100) PseudoMetricSpace.toPseudoEMetricSpace : PseudoEMetricSpace α :=
  { ‹PseudoMetricSpace α› with
                     /-
                       α : Type u
                       β : Type v
                       X : Type u_1
                       ι : Type u_2
                       inst✝ : PseudoMetricSpace α
                       ⊢ ∀ (x : α), Eq (EDist.edist x x) 0
                     -/
    edist_self := by simp [edist_dist]
                     /-
                       🎉 no goals
                     -/
                                /-
                                  α : Type u
                                  β : Type v
                                  X : Type u_1
                                  ι : Type u_2
                                  inst✝ : PseudoMetricSpace α
                                  x✝¹ x✝ : α
                                  ⊢ Eq (EDist.edist x✝¹ x✝) (EDist.edist x✝ x✝¹)
                                -/
    edist_comm := fun _ _ => by simp only [edist_dist, dist_comm]
                                /-
                                  🎉 no goals
                                -/
    edist_triangle := fun x y z => by
      /-
        α : Type u
        β : Type v
        X : Type u_1
        ι : Type u_2
        inst✝ : PseudoMetricSpace α
        x y z : α
        ⊢ LE.le (EDist.edist x z) (HAdd.hAdd (EDist.edist x y) (EDist.edist y z))
      -/
      simp only [edist_dist, ← ENNReal.ofReal_add, dist_nonneg]
      /-
        α : Type u
        β : Type v
        X : Type u_1
        ι : Type u_2
        inst✝ : PseudoMetricSpace α
        x y z : α
        ⊢ LE.le (ENNReal.ofReal (Dist.dist x z)) (ENNReal.ofReal (HAdd.hAdd (Dist.dist …
      -/
      rw [ENNReal.ofReal_le_ofReal_iff _]
        /-
          α : Type u
          β : Type v
          X : Type u_1
          ι : Type u_2
          inst✝ : PseudoMetricSpace α
          x y z : α
          ⊢ LE.le (Dist.dist x z) (HAdd.hAdd (Dist.dist x y) (Dist.dist y z))
        -/
      · exact dist_triangle _ _ _
        /-
          🎉 no goals
        -/
        /-
          α : Type u
          β : Type v
          X : Type u_1
          ι : Type u_2
          inst✝ : PseudoMetricSpace α
          x y z : α
          ⊢ LE.le 0 (HAdd.hAdd (Dist.dist x y) (Dist.dist y z))
        -/
      · simpa using add_le_add (dist_nonneg : 0 ≤ dist x y) dist_nonneg
        /-
          🎉 no goals
        -/
    uniformity_edist := Metric.uniformity_edist }


/-- In a pseudometric space, an open ball of infinite radius is the whole space -/
theorem Metric.eball_top_eq_univ (x : α) : EMetric.ball x ∞ = Set.univ :=
  Set.eq_univ_iff_forall.mpr fun y => edist_lt_top y x


/-- Balls defined using the distance or the edistance coincide -/
@[simp]
theorem Metric.emetric_ball {x : α} {ε : ℝ} : EMetric.ball x (ENNReal.ofReal ε) = ball x ε := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    x : α
    ε : Real
    ⊢ Eq (EMetric.ball x (ENNReal.ofReal ε)) (Metric.ball x ε)
  -/
  ext y
  /-
    case h
    α : Type u
    inst✝ : PseudoMetricSpace α
    x : α
    ε : Real
    y : α
    ⊢ Iff (Membership.mem (EMetric.ball x (ENNReal.ofReal ε)) y) (Membership.mem ( …
  -/
  simp only [EMetric.mem_ball, mem_ball, edist_dist]
  /-
    case h
    α : Type u
    inst✝ : PseudoMetricSpace α
    x : α
    ε : Real
    y : α
    ⊢ Iff (LT.lt (ENNReal.ofReal (Dist.dist y x)) (ENNReal.ofReal ε)) (LT.lt (Dist …
  -/
  exact ENNReal.ofReal_lt_ofReal_iff_of_nonneg dist_nonneg
  /-
    🎉 no goals
  -/


/-- Balls defined using the distance or the edistance coincide -/
@[simp]
theorem Metric.emetric_ball_nnreal {x : α} {ε : ℝ≥0} : EMetric.ball x ε = ball x ε := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    x : α
    ε : NNReal
    ⊢ Eq (EMetric.ball x ↑ε) (Metric.ball x ↑ε)
  -/
  rw [← Metric.emetric_ball]
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    x : α
    ε : NNReal
    ⊢ Eq (EMetric.ball x ↑ε) (EMetric.ball x (ENNReal.ofReal ↑ε))
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Closed balls defined using the distance or the edistance coincide -/
theorem Metric.emetric_closedBall {x : α} {ε : ℝ} (h : 0 ≤ ε) :
    EMetric.closedBall x (ENNReal.ofReal ε) = closedBall x ε := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    x : α
    ε : Real
    h : LE.le 0 ε
    ⊢ Eq (EMetric.closedBall x (ENNReal.ofReal ε)) (Metric.closedBall x ε)
  -/
  ext y; simp [edist_le_ofReal h]
         /-
           🎉 no goals
         -/


/-- Closed balls defined using the distance or the edistance coincide -/
@[simp]
theorem Metric.emetric_closedBall_nnreal {x : α} {ε : ℝ≥0} :
    EMetric.closedBall x ε = closedBall x ε := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    x : α
    ε : NNReal
    ⊢ Eq (EMetric.closedBall x ↑ε) (Metric.closedBall x ↑ε)
  -/
  rw [← Metric.emetric_closedBall ε.coe_nonneg, ENNReal.ofReal_coe_nnreal]
  /-
    🎉 no goals
  -/


@[simp]
theorem Metric.emetric_ball_top (x : α) : EMetric.ball x ⊤ = univ :=
  eq_univ_of_forall fun _ => edist_lt_top _ _


/-- Build a new pseudometric space from an old one where the bundled uniform structure is provably
(but typically non-definitionaly) equal to some given uniform structure.
See Note [forgetful inheritance].
See Note [reducible non-instances].
-/
abbrev PseudoMetricSpace.replaceUniformity {α} [U : UniformSpace α] (m : PseudoMetricSpace α)
    (H : 𝓤[U] = 𝓤[PseudoEMetricSpace.toUniformSpace]) : PseudoMetricSpace α :=
  { m with
    toUniformSpace := U
    uniformity_dist := H.trans PseudoMetricSpace.uniformity_dist }


theorem PseudoMetricSpace.replaceUniformity_eq {α} [U : UniformSpace α] (m : PseudoMetricSpace α)
    (H : 𝓤[U] = 𝓤[PseudoEMetricSpace.toUniformSpace]) : m.replaceUniformity H = m := by
  /-
    α : Type u_3
    U : UniformSpace α
    m : PseudoMetricSpace α
    H : Eq (uniformity α) (uniformity α)
    ⊢ Eq (m.replaceUniformity H) m
  -/
  ext
  /-
    case h.dist.h.h
    α : Type u_3
    U : UniformSpace α
    m : PseudoMetricSpace α
    H : Eq (uniformity α) (uniformity α)
    x✝¹ x✝ : α
    ⊢ Eq (Dist.dist x✝¹ x✝) (Dist.dist x✝¹ x✝)
  -/
  rfl
  /-
    🎉 no goals
  -/

-- ensure that the bornology is unchanged when replacing the uniformity.

/-- Build a new pseudo metric space from an old one where the bundled topological structure is
provably (but typically non-definitionaly) equal to some given topological structure.
See Note [forgetful inheritance].
See Note [reducible non-instances].
-/
abbrev PseudoMetricSpace.replaceTopology {γ} [U : TopologicalSpace γ] (m : PseudoMetricSpace γ)
    (H : U = m.toUniformSpace.toTopologicalSpace) : PseudoMetricSpace γ :=
  @PseudoMetricSpace.replaceUniformity γ (m.toUniformSpace.replaceTopology H) m rfl


theorem PseudoMetricSpace.replaceTopology_eq {γ} [U : TopologicalSpace γ] (m : PseudoMetricSpace γ)
    (H : U = m.toUniformSpace.toTopologicalSpace) : m.replaceTopology H = m := by
  /-
    γ : Type u_3
    U : TopologicalSpace γ
    m : PseudoMetricSpace γ
    H : Eq U UniformSpace.toTopologicalSpace
    ⊢ Eq (m.replaceTopology H) m
  -/
  ext
  /-
    case h.dist.h.h
    γ : Type u_3
    U : TopologicalSpace γ
    m : PseudoMetricSpace γ
    H : Eq U UniformSpace.toTopologicalSpace
    x✝¹ x✝ : γ
    ⊢ Eq (Dist.dist x✝¹ x✝) (Dist.dist x✝¹ x✝)
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- One gets a pseudometric space from an emetric space if the edistance
is everywhere finite, by pushing the edistance to reals. We set it up so that the edist and the
uniformity are defeq in the pseudometric space and the pseudoemetric space. In this definition, the
distance is given separately, to be able to prescribe some expression which is not defeq to the
push-forward of the edistance to reals. See note [reducible non-instances]. -/
abbrev PseudoEMetricSpace.toPseudoMetricSpaceOfDist {α : Type u} [e : PseudoEMetricSpace α]
    (dist : α → α → ℝ) (edist_ne_top : ∀ x y : α, edist x y ≠ ⊤)
    (h : ∀ x y, dist x y = ENNReal.toReal (edist x y)) : PseudoMetricSpace α where
  dist := dist
                    /-
                      α✝ : Type u
                      β : Type v
                      X : Type u_1
                      ι : Type u_2
                      inst✝ : PseudoMetricSpace α✝
                      α : Type u
                      e : PseudoEMetricSpace α
                      dist : α → α → Real
                      edist_ne_top : ∀ (x y : α), Ne (EDist.edist x y) Top.top
                      h : ∀ (x y : α), Eq (dist x y) (EDist.edist x y).toReal
                      x : α
                      ⊢ Eq (Dist.dist x x) 0
                    -/
  dist_self x := by simp [h]
                    /-
                      🎉 no goals
                    -/
                      /-
                        α✝ : Type u
                        β : Type v
                        X : Type u_1
                        ι : Type u_2
                        inst✝ : PseudoMetricSpace α✝
                        α : Type u
                        e : PseudoEMetricSpace α
                        dist : α → α → Real
                        edist_ne_top : ∀ (x y : α), Ne (EDist.edist x y) Top.top
                        h : ∀ (x y : α), Eq (dist x y) (EDist.edist x y).toReal
                        x y : α
                        ⊢ Eq (Dist.dist x y) (Dist.dist y x)
                      -/
  dist_comm x y := by simp [h, edist_comm]
                      /-
                        🎉 no goals
                      -/
  dist_triangle x y z := by
    /-
      α✝ : Type u
      β : Type v
      X : Type u_1
      ι : Type u_2
      inst✝ : PseudoMetricSpace α✝
      α : Type u
      e : PseudoEMetricSpace α
      dist : α → α → Real
      edist_ne_top : ∀ (x y : α), Ne (EDist.edist x y) Top.top
      h : ∀ (x y : α), Eq (dist x y) (EDist.edist x y).toReal
      x y z : α
      ⊢ LE.le (Dist.dist x z) (HAdd.hAdd (Dist.dist x y) (Dist.dist y z))
    -/
    simp only [h]
    /-
      α✝ : Type u
      β : Type v
      X : Type u_1
      ι : Type u_2
      inst✝ : PseudoMetricSpace α✝
      α : Type u
      e : PseudoEMetricSpace α
      dist : α → α → Real
      edist_ne_top : ∀ (x y : α), Ne (EDist.edist x y) Top.top
      h : ∀ (x y : α), Eq (dist x y) (EDist.edist x y).toReal
      x y z : α
      ⊢ LE.le (EDist.edist x z).toReal (HAdd.hAdd (EDist.edist x y).toReal (EDist.ed …
    -/
    exact ENNReal.toReal_le_add (edist_triangle _ _ _) (edist_ne_top _ _) (edist_ne_top _ _)
    /-
      🎉 no goals
    -/
  edist := edist
                       /-
                         α✝ : Type u
                         β : Type v
                         X : Type u_1
                         ι : Type u_2
                         inst✝ : PseudoMetricSpace α✝
                         α : Type u
                         e : PseudoEMetricSpace α
                         dist : α → α → Real
                         edist_ne_top : ∀ (x y : α), Ne (EDist.edist x y) Top.top
                         h : ∀ (x y : α), Eq (dist x y) (EDist.edist x y).toReal
                         x✝¹ x✝ : α
                         ⊢ Eq (EDist.edist x✝¹ x✝) (ENNReal.ofReal (Dist.dist x✝¹ x✝))
                       -/
  edist_dist _ _ := by simp only [h, ENNReal.ofReal_toReal (edist_ne_top _ _)]
                       /-
                         🎉 no goals
                       -/
  toUniformSpace := e.toUniformSpace
  uniformity_dist := e.uniformity_edist.trans <| by
    simpa only [ENNReal.coe_toNNReal (edist_ne_top _ _), h]
      using (Metric.uniformity_edist_aux fun x y : α => (edist x y).toNNReal).symm


/-- One gets a pseudometric space from an emetric space if the edistance
is everywhere finite, by pushing the edistance to reals. We set it up so that the edist and the
uniformity are defeq in the pseudometric space and the emetric space. -/
abbrev PseudoEMetricSpace.toPseudoMetricSpace {α : Type u} [PseudoEMetricSpace α]
    (h : ∀ x y : α, edist x y ≠ ⊤) : PseudoMetricSpace α :=
  PseudoEMetricSpace.toPseudoMetricSpaceOfDist (fun x y => ENNReal.toReal (edist x y)) h fun _ _ =>
    rfl


/-- Build a new pseudometric space from an old one where the bundled bornology structure is provably
(but typically non-definitionaly) equal to some given bornology structure.
See Note [forgetful inheritance].
See Note [reducible non-instances].
-/
abbrev PseudoMetricSpace.replaceBornology {α} [B : Bornology α] (m : PseudoMetricSpace α)
    (H : ∀ s, @IsBounded _ B s ↔ @IsBounded _ PseudoMetricSpace.toBornology s) :
    PseudoMetricSpace α :=
  { m with
    toBornology := B
    cobounded_sets := Set.ext <| compl_surjective.forall.2 fun s =>
                          /-
                            α✝ : Type u
                            β : Type v
                            X : Type u_1
                            ι : Type u_2
                            inst✝ : PseudoMetricSpace α✝
                            α : Type ?u.149089
                            B : Bornology α
                            m : PseudoMetricSpace α
                            H : ∀ (s : Set α), Iff (Bornology.IsBounded s) (Bornology.IsBounded s)
                            s : Set α
                            ⊢ Iff (Bornology.IsBounded s) (Membership.mem (setOf fun s => Exists fun C =>  …
                          -/
        (H s).trans <| by rw [isBounded_iff, mem_setOf_eq, compl_compl] }
                          /-
                            🎉 no goals
                          -/


theorem PseudoMetricSpace.replaceBornology_eq {α} [m : PseudoMetricSpace α] [B : Bornology α]
    (H : ∀ s, @IsBounded _ B s ↔ @IsBounded _ PseudoMetricSpace.toBornology s) :
    PseudoMetricSpace.replaceBornology _ H = m := by
  /-
    α : Type u_3
    m : PseudoMetricSpace α
    B : Bornology α
    H : ∀ (s : Set α), Iff (Bornology.IsBounded s) (Bornology.IsBounded s)
    ⊢ Eq (m.replaceBornology H) m
  -/
  ext
  /-
    case h.dist.h.h
    α : Type u_3
    m : PseudoMetricSpace α
    B : Bornology α
    H : ∀ (s : Set α), Iff (Bornology.IsBounded s) (Bornology.IsBounded s)
    x✝¹ x✝ : α
    ⊢ Eq (Dist.dist x✝¹ x✝) (Dist.dist x✝¹ x✝)
  -/
  rfl
  /-
    🎉 no goals
  -/

-- ensure that the uniformity is unchanged when replacing the bornology.

/-- Instantiate the reals as a pseudometric space. -/
instance Real.pseudoMetricSpace : PseudoMetricSpace ℝ where
  dist x y := |x - y|
                  /-
                    α : Type u
                    β : Type v
                    X : Type u_1
                    ι : Type u_2
                    inst✝ : PseudoMetricSpace α
                    ⊢ ∀ (x : Real), Eq (Dist.dist x x) 0
                  -/
  dist_self := by simp [abs_zero]
                  /-
                    🎉 no goals
                  -/
  dist_comm _ _ := abs_sub_comm _ _
  dist_triangle _ _ _ := abs_sub_le _ _ _


theorem Real.dist_eq (x y : ℝ) : dist x y = |x - y| := rfl


theorem Real.nndist_eq (x y : ℝ) : nndist x y = Real.nnabs (x - y) := rfl


theorem Real.nndist_eq' (x y : ℝ) : nndist x y = Real.nnabs (y - x) :=
  nndist_comm _ _


                                                          /-
                                                            x : Real
                                                            ⊢ Eq (Dist.dist x 0) (abs x)
                                                          -/
theorem Real.dist_0_eq_abs (x : ℝ) : dist x 0 = |x| := by simp [Real.dist_eq]
                                                          /-
                                                            🎉 no goals
                                                          -/


theorem Real.sub_le_dist (x y : ℝ) : x - y ≤ dist x y := by
  /-
    x y : Real
    ⊢ LE.le (HSub.hSub x y) (Dist.dist x y)
  -/
  rw [Real.dist_eq, le_abs]
  /-
    x y : Real
    ⊢ Or (LE.le (HSub.hSub x y) (HSub.hSub x y)) (LE.le (HSub.hSub x y) (Neg.neg ( …
  -/
  exact Or.inl (le_refl _)
  /-
    🎉 no goals
  -/


theorem Real.ball_eq_Ioo (x r : ℝ) : ball x r = Ioo (x - r) (x + r) :=
  Set.ext fun y => by
    rw [mem_ball, dist_comm, Real.dist_eq, abs_sub_lt_iff, mem_Ioo, ← sub_lt_iff_lt_add',
      sub_lt_comm]


theorem Real.closedBall_eq_Icc {x r : ℝ} : closedBall x r = Icc (x - r) (x + r) := by
  /-
    x r : Real
    ⊢ Eq (Metric.closedBall x r) (Set.Icc (HSub.hSub x r) (HAdd.hAdd x r))
  -/
  ext y
  rw [mem_closedBall, dist_comm, Real.dist_eq, abs_sub_le_iff, mem_Icc, ← sub_le_iff_le_add',
    sub_le_comm]


theorem Real.Ioo_eq_ball (x y : ℝ) : Ioo x y = ball ((x + y) / 2) ((y - x) / 2) := by
  rw [Real.ball_eq_Ioo, ← sub_div, add_comm, ← sub_add, add_sub_cancel_left, add_self_div_two,
    ← add_div, add_assoc, add_sub_cancel, add_self_div_two]


theorem Real.Icc_eq_closedBall (x y : ℝ) : Icc x y = closedBall ((x + y) / 2) ((y - x) / 2) := by
  rw [Real.closedBall_eq_Icc, ← sub_div, add_comm, ← sub_add, add_sub_cancel_left, add_self_div_two,
    ← add_div, add_assoc, add_sub_cancel, add_self_div_two]


theorem Metric.uniformity_eq_comap_nhds_zero :
    𝓤 α = comap (fun p : α × α => dist p.1 p.2) (𝓝 (0 : ℝ)) := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    ⊢ Eq (uniformity α) (Filter.comap (fun p => Dist.dist p.1 p.2) (nhds 0))
  -/
  ext s
  /-
    case h
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Set (Prod α α)
    ⊢ Iff (Membership.mem (uniformity α) s) (Membership.mem (Filter.comap (fun p = …
  -/
  simp only [mem_uniformity_dist, (nhds_basis_ball.comap _).mem_iff]
  /-
    case h
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Set (Prod α α)
    ⊢ Iff (Exists fun ε => And (GT.gt ε 0) (∀ ⦃a b : α⦄, LT.lt (Dist.dist a b) ε → …
  -/
  simp [subset_def, Real.dist_0_eq_abs]
  /-
    🎉 no goals
  -/


theorem tendsto_uniformity_iff_dist_tendsto_zero {f : ι → α × α} {p : Filter ι} :
    Tendsto f p (𝓤 α) ↔ Tendsto (fun x => dist (f x).1 (f x).2) p (𝓝 0) := by
  /-
    α : Type u
    ι : Type u_2
    inst✝ : PseudoMetricSpace α
    f : ι → Prod α α
    p : Filter ι
    ⊢ Iff (Filter.Tendsto f p (uniformity α)) (Filter.Tendsto (fun x => Dist.dist  …
  -/
  rw [Metric.uniformity_eq_comap_nhds_zero, tendsto_comap_iff, Function.comp_def]
  /-
    🎉 no goals
  -/


theorem Filter.Tendsto.congr_dist {f₁ f₂ : ι → α} {p : Filter ι} {a : α}
    (h₁ : Tendsto f₁ p (𝓝 a)) (h : Tendsto (fun x => dist (f₁ x) (f₂ x)) p (𝓝 0)) :
    Tendsto f₂ p (𝓝 a) :=
  h₁.congr_uniformity <| tendsto_uniformity_iff_dist_tendsto_zero.2 h


alias tendsto_of_tendsto_of_dist := Filter.Tendsto.congr_dist


theorem tendsto_iff_of_dist {f₁ f₂ : ι → α} {p : Filter ι} {a : α}
    (h : Tendsto (fun x => dist (f₁ x) (f₂ x)) p (𝓝 0)) : Tendsto f₁ p (𝓝 a) ↔ Tendsto f₂ p (𝓝 a) :=
  Uniform.tendsto_congr <| tendsto_uniformity_iff_dist_tendsto_zero.2 h


theorem PseudoMetricSpace.dist_eq_of_dist_zero (x : α) {y z : α} (h : dist y z = 0) :
    dist x y = dist x z :=
  dist_comm y x ▸ dist_comm z x ▸ sub_eq_zero.1 (abs_nonpos_iff.1 (h ▸ abs_dist_sub_le y z x))

-- Porting note: 3 new lemmas

theorem dist_dist_dist_le_left (x y z : α) : dist (dist x z) (dist y z) ≤ dist x y :=
  abs_dist_sub_le ..


theorem dist_dist_dist_le_right (x y z : α) : dist (dist x y) (dist x z) ≤ dist y z := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    x y z : α
    ⊢ LE.le (Dist.dist (Dist.dist x y) (Dist.dist x z)) (Dist.dist y z)
  -/
  simpa only [dist_comm x] using dist_dist_dist_le_left y z x
  /-
    🎉 no goals
  -/


theorem dist_dist_dist_le (x y x' y' : α) : dist (dist x y) (dist x' y') ≤ dist x x' + dist y y' :=
  (dist_triangle _ _ _).trans <|
    add_le_add (dist_dist_dist_le_left _ _ _) (dist_dist_dist_le_right _ _ _)


theorem nhds_comap_dist (a : α) : ((𝓝 (0 : ℝ)).comap (dist · a)) = 𝓝 a := by
  simp only [@nhds_eq_comap_uniformity α, Metric.uniformity_eq_comap_nhds_zero, comap_comap,
    Function.comp_def, dist_comm]


theorem tendsto_iff_dist_tendsto_zero {f : β → α} {x : Filter β} {a : α} :
    Tendsto f x (𝓝 a) ↔ Tendsto (fun b => dist (f b) a) x (𝓝 0) := by
  /-
    α : Type u
    β : Type v
    inst✝ : PseudoMetricSpace α
    f : β → α
    x : Filter β
    a : α
    ⊢ Iff (Filter.Tendsto f x (nhds a)) (Filter.Tendsto (fun b => Dist.dist (f b)  …
  -/
  rw [← nhds_comap_dist a, tendsto_comap_iff, Function.comp_def]
  /-
    🎉 no goals
  -/


theorem ball_subset_interior_closedBall : ball x ε ⊆ interior (closedBall x ε) :=
  interior_maximal ball_subset_closedBall isOpen_ball


/-- ε-characterization of the closure in pseudometric spaces -/
theorem mem_closure_iff {s : Set α} {a : α} : a ∈ closure s ↔ ∀ ε > 0, ∃ b ∈ s, dist a b < ε :=
                                                           /-
                                                             α : Type u
                                                             inst✝ : PseudoMetricSpace α
                                                             s : Set α
                                                             a : α
                                                             ⊢ Iff (∀ (i : Real), LT.lt 0 i → Exists fun y => And (Membership.mem s y) (Mem …
                                                           -/
  (mem_closure_iff_nhds_basis nhds_basis_ball).trans <| by simp only [mem_ball, dist_comm]
                                                           /-
                                                             🎉 no goals
                                                           -/


theorem mem_closure_range_iff {e : β → α} {a : α} :
    a ∈ closure (range e) ↔ ∀ ε > 0, ∃ k : β, dist a (e k) < ε := by
  /-
    α : Type u
    β : Type v
    inst✝ : PseudoMetricSpace α
    e : β → α
    a : α
    ⊢ Iff (Membership.mem (closure (Set.range e)) a) (∀ (ε : Real), GT.gt ε 0 → Ex …
  -/
  simp only [mem_closure_iff, exists_range_iff]
  /-
    🎉 no goals
  -/


theorem mem_closure_range_iff_nat {e : β → α} {a : α} :
    a ∈ closure (range e) ↔ ∀ n : ℕ, ∃ k : β, dist a (e k) < 1 / ((n : ℝ) + 1) :=
  (mem_closure_iff_nhds_basis nhds_basis_ball_inv_nat_succ).trans <| by
    /-
      α : Type u
      β : Type v
      inst✝ : PseudoMetricSpace α
      e : β → α
      a : α
      ⊢ Iff (∀ (i : Nat), True → Exists fun y => And (Membership.mem (Set.range e) y …
    -/
    simp only [mem_ball, dist_comm, exists_range_iff, forall_const]
    /-
      🎉 no goals
    -/


theorem mem_of_closed' {s : Set α} (hs : IsClosed s) {a : α} :
    a ∈ s ↔ ∀ ε > 0, ∃ b ∈ s, dist a b < ε := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Set α
    hs : IsClosed s
    a : α
    ⊢ Iff (Membership.mem s a) (∀ (ε : Real), GT.gt ε 0 → Exists fun b => And (Mem …
  -/
  simpa only [hs.closure_eq] using @mem_closure_iff _ _ s a
  /-
    🎉 no goals
  -/


theorem dense_iff {s : Set α} : Dense s ↔ ∀ x, ∀ r > 0, (ball x r ∩ s).Nonempty :=
  forall_congr' fun x => by
    /-
      α : Type u
      inst✝ : PseudoMetricSpace α
      s : Set α
      x : α
      ⊢ Iff (Membership.mem (closure s) x) (∀ (r : Real), GT.gt r 0 → (Inter.inter ( …
    -/
    simp only [mem_closure_iff, Set.Nonempty, exists_prop, mem_inter_iff, mem_ball', and_comm]
    /-
      🎉 no goals
    -/


theorem dense_iff_iUnion_ball (s : Set α) : Dense s ↔ ∀ r > 0, ⋃ c ∈ s, ball c r = univ := by
  simp_rw [eq_univ_iff_forall, mem_iUnion, exists_prop, mem_ball, Dense, mem_closure_iff,
    forall_comm (α := α)]


theorem denseRange_iff {f : β → α} : DenseRange f ↔ ∀ x, ∀ r > 0, ∃ y, dist x (f y) < r :=
                            /-
                              α : Type u
                              β : Type v
                              inst✝ : PseudoMetricSpace α
                              f : β → α
                              x : α
                              ⊢ Iff (Membership.mem (closure (Set.range f)) x) (∀ (r : Real), GT.gt r 0 → Ex …
                            -/
  forall_congr' fun x => by simp only [mem_closure_iff, exists_range_iff]
                            /-
                              🎉 no goals
                            -/


/-- If a map is continuous on a separable set `s`, then the image of `s` is also separable. -/
theorem _root_.ContinuousOn.isSeparable_image [TopologicalSpace β] {f : α → β} {s : Set α}
    (hf : ContinuousOn f s) (hs : IsSeparable s) : IsSeparable (f '' s) := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : PseudoMetricSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    s : Set α
    hf : ContinuousOn f s
    hs : TopologicalSpace.IsSeparable s
    ⊢ TopologicalSpace.IsSeparable (Set.image f s)
  -/
  rw [image_eq_range, ← image_univ]
  /-
    α : Type u
    β : Type v
    inst✝¹ : PseudoMetricSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    s : Set α
    hf : ContinuousOn f s
    hs : TopologicalSpace.IsSeparable s
    ⊢ TopologicalSpace.IsSeparable (Set.image (fun x => f ↑x) Set.univ)
  -/
  exact (isSeparable_univ_iff.2 hs.separableSpace).image hf.restrict
  /-
    🎉 no goals
  -/


/-- Any compact set in a pseudometric space can be covered by finitely many balls of a given
positive radius -/
theorem finite_cover_balls_of_compact {α : Type u} [PseudoMetricSpace α] {s : Set α}
    (hs : IsCompact s) {e : ℝ} (he : 0 < e) :
    ∃ t, t ⊆ s ∧ Set.Finite t ∧ s ⊆ ⋃ x ∈ t, ball x e :=
  let ⟨t, hts, ht⟩ := hs.elim_nhds_subcover _ (fun x _ => ball_mem_nhds x he)
  ⟨t, hts, t.finite_toSet, ht⟩


alias IsCompact.finite_cover_balls := finite_cover_balls_of_compact


theorem lebesgue_number_lemma_of_metric {s : Set α} {ι : Sort*} {c : ι → Set α} (hs : IsCompact s)
    (hc₁ : ∀ i, IsOpen (c i)) (hc₂ : s ⊆ ⋃ i, c i) : ∃ δ > 0, ∀ x ∈ s, ∃ i, ball x δ ⊆ c i := by
  simpa only [ball, UniformSpace.ball, preimage_setOf_eq, dist_comm]
    using uniformity_basis_dist.lebesgue_number_lemma hs hc₁ hc₂


theorem lebesgue_number_lemma_of_metric_sUnion {s : Set α} {c : Set (Set α)} (hs : IsCompact s)
    (hc₁ : ∀ t ∈ c, IsOpen t) (hc₂ : s ⊆ ⋃₀ c) : ∃ δ > 0, ∀ x ∈ s, ∃ t ∈ c, ball x δ ⊆ t := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Set α
    c : Set (Set α)
    hs : IsCompact s
    hc₁ : ∀ (t : Set α), Membership.mem c t → IsOpen t
    hc₂ : HasSubset.Subset s c.sUnion
    ⊢ Exists fun δ => And (GT.gt δ 0) (∀ (x : α), Membership.mem s x → Exists fun  …
  -/
  rw [sUnion_eq_iUnion] at hc₂; simpa using lebesgue_number_lemma_of_metric hs (by simpa) hc₂
                                /-
                                  🎉 no goals
                                -/


instance : PseudoMetricSpace (Additive α) := ‹_›

instance : PseudoMetricSpace (Multiplicative α) := ‹_›

instance : PseudoMetricSpace αᵒᵈ := ‹_›

