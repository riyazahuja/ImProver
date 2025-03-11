/-- A real function `f` is lower semicontinuous at `x` within a set `s` if, for any `ε > 0`, for all
`x'` close enough to `x` in `s`, then `f x'` is at least `f x - ε`. We formulate this in a general
preordered space, using an arbitrary `y < f x` instead of `f x - ε`. -/
def LowerSemicontinuousWithinAt (f : α → β) (s : Set α) (x : α) :=
  ∀ y < f x, ∀ᶠ x' in 𝓝[s] x, y < f x'


/-- A real function `f` is lower semicontinuous on a set `s` if, for any `ε > 0`, for any `x ∈ s`,
for all `x'` close enough to `x` in `s`, then `f x'` is at least `f x - ε`. We formulate this in
a general preordered space, using an arbitrary `y < f x` instead of `f x - ε`. -/
def LowerSemicontinuousOn (f : α → β) (s : Set α) :=
  ∀ x ∈ s, LowerSemicontinuousWithinAt f s x


/-- A real function `f` is lower semicontinuous at `x` if, for any `ε > 0`, for all `x'` close
enough to `x`, then `f x'` is at least `f x - ε`. We formulate this in a general preordered space,
using an arbitrary `y < f x` instead of `f x - ε`. -/
def LowerSemicontinuousAt (f : α → β) (x : α) :=
  ∀ y < f x, ∀ᶠ x' in 𝓝 x, y < f x'


/-- A real function `f` is lower semicontinuous if, for any `ε > 0`, for any `x`, for all `x'` close
enough to `x`, then `f x'` is at least `f x - ε`. We formulate this in a general preordered space,
using an arbitrary `y < f x` instead of `f x - ε`. -/
def LowerSemicontinuous (f : α → β) :=
  ∀ x, LowerSemicontinuousAt f x


/-- A real function `f` is upper semicontinuous at `x` within a set `s` if, for any `ε > 0`, for all
`x'` close enough to `x` in `s`, then `f x'` is at most `f x + ε`. We formulate this in a general
preordered space, using an arbitrary `y > f x` instead of `f x + ε`. -/
def UpperSemicontinuousWithinAt (f : α → β) (s : Set α) (x : α) :=
  ∀ y, f x < y → ∀ᶠ x' in 𝓝[s] x, f x' < y


/-- A real function `f` is upper semicontinuous on a set `s` if, for any `ε > 0`, for any `x ∈ s`,
for all `x'` close enough to `x` in `s`, then `f x'` is at most `f x + ε`. We formulate this in a
general preordered space, using an arbitrary `y > f x` instead of `f x + ε`. -/
def UpperSemicontinuousOn (f : α → β) (s : Set α) :=
  ∀ x ∈ s, UpperSemicontinuousWithinAt f s x


/-- A real function `f` is upper semicontinuous at `x` if, for any `ε > 0`, for all `x'` close
enough to `x`, then `f x'` is at most `f x + ε`. We formulate this in a general preordered space,
using an arbitrary `y > f x` instead of `f x + ε`. -/
def UpperSemicontinuousAt (f : α → β) (x : α) :=
  ∀ y, f x < y → ∀ᶠ x' in 𝓝 x, f x' < y


/-- A real function `f` is upper semicontinuous if, for any `ε > 0`, for any `x`, for all `x'`
close enough to `x`, then `f x'` is at most `f x + ε`. We formulate this in a general preordered
space, using an arbitrary `y > f x` instead of `f x + ε`. -/
def UpperSemicontinuous (f : α → β) :=
  ∀ x, UpperSemicontinuousAt f x


theorem LowerSemicontinuousWithinAt.mono (h : LowerSemicontinuousWithinAt f s x) (hst : t ⊆ s) :
    LowerSemicontinuousWithinAt f t x := fun y hy =>
  Filter.Eventually.filter_mono (nhdsWithin_mono _ hst) (h y hy)


theorem lowerSemicontinuousWithinAt_univ_iff :
    LowerSemicontinuousWithinAt f univ x ↔ LowerSemicontinuousAt f x := by
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    β : Type u_2
    inst✝ : Preorder β
    f : α → β
    x : α
    ⊢ Iff (LowerSemicontinuousWithinAt f Set.univ x) (LowerSemicontinuousAt f x)
  -/
  simp [LowerSemicontinuousWithinAt, LowerSemicontinuousAt, nhdsWithin_univ]
  /-
    🎉 no goals
  -/


theorem LowerSemicontinuousAt.lowerSemicontinuousWithinAt (s : Set α)
    (h : LowerSemicontinuousAt f x) : LowerSemicontinuousWithinAt f s x := fun y hy =>
  Filter.Eventually.filter_mono nhdsWithin_le_nhds (h y hy)


theorem LowerSemicontinuousOn.lowerSemicontinuousWithinAt (h : LowerSemicontinuousOn f s)
    (hx : x ∈ s) : LowerSemicontinuousWithinAt f s x :=
  h x hx


theorem LowerSemicontinuousOn.mono (h : LowerSemicontinuousOn f s) (hst : t ⊆ s) :
    LowerSemicontinuousOn f t := fun x hx => (h x (hst hx)).mono hst


theorem lowerSemicontinuousOn_univ_iff : LowerSemicontinuousOn f univ ↔ LowerSemicontinuous f := by
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    β : Type u_2
    inst✝ : Preorder β
    f : α → β
    ⊢ Iff (LowerSemicontinuousOn f Set.univ) (LowerSemicontinuous f)
  -/
  simp [LowerSemicontinuousOn, LowerSemicontinuous, lowerSemicontinuousWithinAt_univ_iff]
  /-
    🎉 no goals
  -/


theorem LowerSemicontinuous.lowerSemicontinuousAt (h : LowerSemicontinuous f) (x : α) :
    LowerSemicontinuousAt f x :=
  h x


theorem LowerSemicontinuous.lowerSemicontinuousWithinAt (h : LowerSemicontinuous f) (s : Set α)
    (x : α) : LowerSemicontinuousWithinAt f s x :=
  (h x).lowerSemicontinuousWithinAt s


theorem LowerSemicontinuous.lowerSemicontinuousOn (h : LowerSemicontinuous f) (s : Set α) :
    LowerSemicontinuousOn f s := fun x _hx => h.lowerSemicontinuousWithinAt s x


theorem lowerSemicontinuousWithinAt_const : LowerSemicontinuousWithinAt (fun _x => z) s x :=
  fun _y hy => Filter.Eventually.of_forall fun _x => hy


theorem lowerSemicontinuousAt_const : LowerSemicontinuousAt (fun _x => z) x := fun _y hy =>
  Filter.Eventually.of_forall fun _x => hy


theorem lowerSemicontinuousOn_const : LowerSemicontinuousOn (fun _x => z) s := fun _x _hx =>
  lowerSemicontinuousWithinAt_const


theorem lowerSemicontinuous_const : LowerSemicontinuous fun _x : α => z := fun _x =>
  lowerSemicontinuousAt_const


theorem IsOpen.lowerSemicontinuous_indicator (hs : IsOpen s) (hy : 0 ≤ y) :
    LowerSemicontinuous (indicator s fun _x => y) := by
  /-
    α : Type u_1
    inst✝² : TopologicalSpace α
    β : Type u_2
    inst✝¹ : Preorder β
    s : Set α
    y : β
    inst✝ : Zero β
    hs : IsOpen s
    hy : LE.le 0 y
    ⊢ LowerSemicontinuous (s.indicator fun _x => y)
  -/
  intro x z hz
  /-
    α : Type u_1
    inst✝² : TopologicalSpace α
    β : Type u_2
    inst✝¹ : Preorder β
    s : Set α
    y : β
    inst✝ : Zero β
    hs : IsOpen s
    hy : LE.le 0 y
    x : α
    z : β
    hz : LT.lt z (s.indicator (fun _x => y) x)
    ⊢ Filter.Eventually (fun x' => LT.lt z (s.indicator (fun _x => y) x')) (nhds x)
  -/
  by_cases h : x ∈ s <;> simp [h] at hz
    /-
      case pos
      α : Type u_1
      inst✝² : TopologicalSpace α
      β : Type u_2
      inst✝¹ : Preorder β
      s : Set α
      y : β
      inst✝ : Zero β
      hs : IsOpen s
      hy : LE.le 0 y
      x : α
      z : β
      h : Membership.mem s x
      hz : LT.lt z y
      ⊢ Filter.Eventually (fun x' => LT.lt z (s.indicator (fun _x => y) x')) (nhds x)
    -/
  · filter_upwards [hs.mem_nhds h]
    /-
      case h
      α : Type u_1
      inst✝² : TopologicalSpace α
      β : Type u_2
      inst✝¹ : Preorder β
      s : Set α
      y : β
      inst✝ : Zero β
      hs : IsOpen s
      hy : LE.le 0 y
      x : α
      z : β
      h : Membership.mem s x
      hz : LT.lt z y
      ⊢ ∀ (a : α), Membership.mem s a → LT.lt z (s.indicator (fun _x => y) a)
    -/
    simp +contextual [hz]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝² : TopologicalSpace α
      β : Type u_2
      inst✝¹ : Preorder β
      s : Set α
      y : β
      inst✝ : Zero β
      hs : IsOpen s
      hy : LE.le 0 y
      x : α
      z : β
      h : Not (Membership.mem s x)
      hz : LT.lt z 0
      ⊢ Filter.Eventually (fun x' => LT.lt z (s.indicator (fun _x => y) x')) (nhds x)
    -/
  · refine Filter.Eventually.of_forall fun x' => ?_
    /-
      case neg
      α : Type u_1
      inst✝² : TopologicalSpace α
      β : Type u_2
      inst✝¹ : Preorder β
      s : Set α
      y : β
      inst✝ : Zero β
      hs : IsOpen s
      hy : LE.le 0 y
      x : α
      z : β
      h : Not (Membership.mem s x)
      hz : LT.lt z 0
      x' : α
      ⊢ LT.lt z (s.indicator (fun _x => y) x')
    -/
                             /-
                               🎉 no goals
                             -/
    by_cases h' : x' ∈ s <;> simp [h', hz.trans_le hy, hz]
                             /-
                               🎉 no goals
                             -/


theorem IsOpen.lowerSemicontinuousOn_indicator (hs : IsOpen s) (hy : 0 ≤ y) :
    LowerSemicontinuousOn (indicator s fun _x => y) t :=
  (hs.lowerSemicontinuous_indicator hy).lowerSemicontinuousOn t


theorem IsOpen.lowerSemicontinuousAt_indicator (hs : IsOpen s) (hy : 0 ≤ y) :
    LowerSemicontinuousAt (indicator s fun _x => y) x :=
  (hs.lowerSemicontinuous_indicator hy).lowerSemicontinuousAt x


theorem IsOpen.lowerSemicontinuousWithinAt_indicator (hs : IsOpen s) (hy : 0 ≤ y) :
    LowerSemicontinuousWithinAt (indicator s fun _x => y) t x :=
  (hs.lowerSemicontinuous_indicator hy).lowerSemicontinuousWithinAt t x


theorem IsClosed.lowerSemicontinuous_indicator (hs : IsClosed s) (hy : y ≤ 0) :
    LowerSemicontinuous (indicator s fun _x => y) := by
  /-
    α : Type u_1
    inst✝² : TopologicalSpace α
    β : Type u_2
    inst✝¹ : Preorder β
    s : Set α
    y : β
    inst✝ : Zero β
    hs : IsClosed s
    hy : LE.le y 0
    ⊢ LowerSemicontinuous (s.indicator fun _x => y)
  -/
  intro x z hz
  /-
    α : Type u_1
    inst✝² : TopologicalSpace α
    β : Type u_2
    inst✝¹ : Preorder β
    s : Set α
    y : β
    inst✝ : Zero β
    hs : IsClosed s
    hy : LE.le y 0
    x : α
    z : β
    hz : LT.lt z (s.indicator (fun _x => y) x)
    ⊢ Filter.Eventually (fun x' => LT.lt z (s.indicator (fun _x => y) x')) (nhds x)
  -/
  by_cases h : x ∈ s <;> simp [h] at hz
    /-
      case pos
      α : Type u_1
      inst✝² : TopologicalSpace α
      β : Type u_2
      inst✝¹ : Preorder β
      s : Set α
      y : β
      inst✝ : Zero β
      hs : IsClosed s
      hy : LE.le y 0
      x : α
      z : β
      h : Membership.mem s x
      hz : LT.lt z y
      ⊢ Filter.Eventually (fun x' => LT.lt z (s.indicator (fun _x => y) x')) (nhds x)
    -/
  · refine Filter.Eventually.of_forall fun x' => ?_
    /-
      case pos
      α : Type u_1
      inst✝² : TopologicalSpace α
      β : Type u_2
      inst✝¹ : Preorder β
      s : Set α
      y : β
      inst✝ : Zero β
      hs : IsClosed s
      hy : LE.le y 0
      x : α
      z : β
      h : Membership.mem s x
      hz : LT.lt z y
      x' : α
      ⊢ LT.lt z (s.indicator (fun _x => y) x')
    -/
                             /-
                               🎉 no goals
                             -/
    by_cases h' : x' ∈ s <;> simp [h', hz, hz.trans_le hy]
                             /-
                               🎉 no goals
                             -/
    /-
      case neg
      α : Type u_1
      inst✝² : TopologicalSpace α
      β : Type u_2
      inst✝¹ : Preorder β
      s : Set α
      y : β
      inst✝ : Zero β
      hs : IsClosed s
      hy : LE.le y 0
      x : α
      z : β
      h : Not (Membership.mem s x)
      hz : LT.lt z 0
      ⊢ Filter.Eventually (fun x' => LT.lt z (s.indicator (fun _x => y) x')) (nhds x)
    -/
  · filter_upwards [hs.isOpen_compl.mem_nhds h]
    /-
      case h
      α : Type u_1
      inst✝² : TopologicalSpace α
      β : Type u_2
      inst✝¹ : Preorder β
      s : Set α
      y : β
      inst✝ : Zero β
      hs : IsClosed s
      hy : LE.le y 0
      x : α
      z : β
      h : Not (Membership.mem s x)
      hz : LT.lt z 0
      ⊢ ∀ (a : α), Membership.mem (HasCompl.compl s) a → LT.lt z (s.indicator (fun _ …
    -/
    simp +contextual [hz]
    /-
      🎉 no goals
    -/


theorem IsClosed.lowerSemicontinuousOn_indicator (hs : IsClosed s) (hy : y ≤ 0) :
    LowerSemicontinuousOn (indicator s fun _x => y) t :=
  (hs.lowerSemicontinuous_indicator hy).lowerSemicontinuousOn t


theorem IsClosed.lowerSemicontinuousAt_indicator (hs : IsClosed s) (hy : y ≤ 0) :
    LowerSemicontinuousAt (indicator s fun _x => y) x :=
  (hs.lowerSemicontinuous_indicator hy).lowerSemicontinuousAt x


theorem IsClosed.lowerSemicontinuousWithinAt_indicator (hs : IsClosed s) (hy : y ≤ 0) :
    LowerSemicontinuousWithinAt (indicator s fun _x => y) t x :=
  (hs.lowerSemicontinuous_indicator hy).lowerSemicontinuousWithinAt t x


theorem lowerSemicontinuous_iff_isOpen_preimage :
    LowerSemicontinuous f ↔ ∀ y, IsOpen (f ⁻¹' Ioi y) :=
  ⟨fun H y => isOpen_iff_mem_nhds.2 fun x hx => H x y hx, fun H _x y y_lt =>
    IsOpen.mem_nhds (H y) y_lt⟩


theorem LowerSemicontinuous.isOpen_preimage (hf : LowerSemicontinuous f) (y : β) :
    IsOpen (f ⁻¹' Ioi y) :=
  lowerSemicontinuous_iff_isOpen_preimage.1 hf y


theorem lowerSemicontinuous_iff_isClosed_preimage {f : α → γ} :
    LowerSemicontinuous f ↔ ∀ y, IsClosed (f ⁻¹' Iic y) := by
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    γ : Type u_3
    inst✝ : LinearOrder γ
    f : α → γ
    ⊢ Iff (LowerSemicontinuous f) (∀ (y : γ), IsClosed (Set.preimage f (Set.Iic y)))
  -/
  rw [lowerSemicontinuous_iff_isOpen_preimage]
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    γ : Type u_3
    inst✝ : LinearOrder γ
    f : α → γ
    ⊢ Iff (∀ (y : γ), IsOpen (Set.preimage f (Set.Ioi y))) (∀ (y : γ), IsClosed (S …
  -/
  simp only [← isOpen_compl_iff, ← preimage_compl, compl_Iic]
  /-
    🎉 no goals
  -/


theorem LowerSemicontinuous.isClosed_preimage {f : α → γ} (hf : LowerSemicontinuous f) (y : γ) :
    IsClosed (f ⁻¹' Iic y) :=
  lowerSemicontinuous_iff_isClosed_preimage.1 hf y


theorem ContinuousWithinAt.lowerSemicontinuousWithinAt {f : α → γ} (h : ContinuousWithinAt f s x) :
    LowerSemicontinuousWithinAt f s x := fun _y hy => h (Ioi_mem_nhds hy)


theorem ContinuousAt.lowerSemicontinuousAt {f : α → γ} (h : ContinuousAt f x) :
    LowerSemicontinuousAt f x := fun _y hy => h (Ioi_mem_nhds hy)


theorem ContinuousOn.lowerSemicontinuousOn {f : α → γ} (h : ContinuousOn f s) :
    LowerSemicontinuousOn f s := fun x hx => (h x hx).lowerSemicontinuousWithinAt


theorem Continuous.lowerSemicontinuous {f : α → γ} (h : Continuous f) : LowerSemicontinuous f :=
  fun _x => h.continuousAt.lowerSemicontinuousAt


theorem lowerSemicontinuousWithinAt_iff_le_liminf {f : α → γ} :
    LowerSemicontinuousWithinAt f s x ↔ f x ≤ liminf f (𝓝[s] x) := by
  /-
    α : Type u_1
    inst✝² : TopologicalSpace α
    x : α
    s : Set α
    γ : Type u_3
    inst✝¹ : CompleteLinearOrder γ
    inst✝ : DenselyOrdered γ
    f : α → γ
    ⊢ Iff (LowerSemicontinuousWithinAt f s x) (LE.le (f x) (Filter.liminf f (nhdsW …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      inst✝² : TopologicalSpace α
      x : α
      s : Set α
      γ : Type u_3
      inst✝¹ : CompleteLinearOrder γ
      inst✝ : DenselyOrdered γ
      f : α → γ
      ⊢ LowerSemicontinuousWithinAt f s x → LE.le (f x) (Filter.liminf f (nhdsWithin …
    -/
  · intro hf; unfold LowerSemicontinuousWithinAt at hf
    /-
      case mp
      α : Type u_1
      inst✝² : TopologicalSpace α
      x : α
      s : Set α
      γ : Type u_3
      inst✝¹ : CompleteLinearOrder γ
      inst✝ : DenselyOrdered γ
      f : α → γ
      hf : ∀ (y : γ), LT.lt y (f x) → Filter.Eventually (fun x' => LT.lt y (f x')) ( …
      ⊢ LE.le (f x) (Filter.liminf f (nhdsWithin x s))
    -/
    contrapose! hf
    /-
      case mp
      α : Type u_1
      inst✝² : TopologicalSpace α
      x : α
      s : Set α
      γ : Type u_3
      inst✝¹ : CompleteLinearOrder γ
      inst✝ : DenselyOrdered γ
      f : α → γ
      hf : LT.lt (Filter.liminf f (nhdsWithin x s)) (f x)
      ⊢ Exists fun y => And (LT.lt y (f x)) (Not (Filter.Eventually (fun x' => LT.lt …
    -/
    obtain ⟨y, lty, ylt⟩ := exists_between hf; use y
    exact ⟨ylt, fun h => lty.not_le
      (le_liminf_of_le (by isBoundedDefault) (h.mono fun _ hx => le_of_lt hx))⟩
  /-
    case mpr
    α : Type u_1
    inst✝² : TopologicalSpace α
    x : α
    s : Set α
    γ : Type u_3
    inst✝¹ : CompleteLinearOrder γ
    inst✝ : DenselyOrdered γ
    f : α → γ
    ⊢ LE.le (f x) (Filter.liminf f (nhdsWithin x s)) → LowerSemicontinuousWithinAt …
  -/
  exact fun hf y ylt => eventually_lt_of_lt_liminf (ylt.trans_le hf)
  /-
    🎉 no goals
  -/


alias ⟨LowerSemicontinuousWithinAt.le_liminf, _⟩ := lowerSemicontinuousWithinAt_iff_le_liminf


theorem lowerSemicontinuousAt_iff_le_liminf {f : α → γ} :
    LowerSemicontinuousAt f x ↔ f x ≤ liminf f (𝓝 x) := by
  rw [← lowerSemicontinuousWithinAt_univ_iff, lowerSemicontinuousWithinAt_iff_le_liminf,
    ← nhdsWithin_univ]


alias ⟨LowerSemicontinuousAt.le_liminf, _⟩ := lowerSemicontinuousAt_iff_le_liminf


theorem lowerSemicontinuous_iff_le_liminf {f : α → γ} :
    LowerSemicontinuous f ↔ ∀ x, f x ≤ liminf f (𝓝 x) := by
  /-
    α : Type u_1
    inst✝² : TopologicalSpace α
    γ : Type u_3
    inst✝¹ : CompleteLinearOrder γ
    inst✝ : DenselyOrdered γ
    f : α → γ
    ⊢ Iff (LowerSemicontinuous f) (∀ (x : α), LE.le (f x) (Filter.liminf f (nhds x …
  -/
  simp only [← lowerSemicontinuousAt_iff_le_liminf, LowerSemicontinuous]
  /-
    🎉 no goals
  -/


alias ⟨LowerSemicontinuous.le_liminf, _⟩ := lowerSemicontinuous_iff_le_liminf


theorem lowerSemicontinuousOn_iff_le_liminf {f : α → γ} :
    LowerSemicontinuousOn f s ↔ ∀ x ∈ s, f x ≤ liminf f (𝓝[s] x) := by
  /-
    α : Type u_1
    inst✝² : TopologicalSpace α
    s : Set α
    γ : Type u_3
    inst✝¹ : CompleteLinearOrder γ
    inst✝ : DenselyOrdered γ
    f : α → γ
    ⊢ Iff (LowerSemicontinuousOn f s) (∀ (x : α), Membership.mem s x → LE.le (f x) …
  -/
  simp only [← lowerSemicontinuousWithinAt_iff_le_liminf, LowerSemicontinuousOn]
  /-
    🎉 no goals
  -/


alias ⟨LowerSemicontinuousOn.le_liminf, _⟩ := lowerSemicontinuousOn_iff_le_liminf


theorem lowerSemicontinuous_iff_isClosed_epigraph {f : α → γ} :
    LowerSemicontinuous f ↔ IsClosed {p : α × γ | f p.1 ≤ p.2} := by
  /-
    α : Type u_1
    inst✝⁴ : TopologicalSpace α
    γ : Type u_3
    inst✝³ : CompleteLinearOrder γ
    inst✝² : DenselyOrdered γ
    inst✝¹ : TopologicalSpace γ
    inst✝ : OrderTopology γ
    f : α → γ
    ⊢ Iff (LowerSemicontinuous f) (IsClosed (setOf fun p => LE.le (f p.1) p.2))
  -/
  constructor
    /-
      case mp
      α : Type u_1
      inst✝⁴ : TopologicalSpace α
      γ : Type u_3
      inst✝³ : CompleteLinearOrder γ
      inst✝² : DenselyOrdered γ
      inst✝¹ : TopologicalSpace γ
      inst✝ : OrderTopology γ
      f : α → γ
      ⊢ LowerSemicontinuous f → IsClosed (setOf fun p => LE.le (f p.1) p.2)
    -/
  · rw [lowerSemicontinuous_iff_le_liminf, isClosed_iff_forall_filter]
    /-
      case mp
      α : Type u_1
      inst✝⁴ : TopologicalSpace α
      γ : Type u_3
      inst✝³ : CompleteLinearOrder γ
      inst✝² : DenselyOrdered γ
      inst✝¹ : TopologicalSpace γ
      inst✝ : OrderTopology γ
      f : α → γ
      ⊢ (∀ (x : α), LE.le (f x) (Filter.liminf f (nhds x))) → ∀ (x : Prod α γ) (F :  …
    -/
    rintro hf ⟨x, y⟩ F F_ne h h'
    /-
      case mp.mk
      α : Type u_1
      inst✝⁴ : TopologicalSpace α
      γ : Type u_3
      inst✝³ : CompleteLinearOrder γ
      inst✝² : DenselyOrdered γ
      inst✝¹ : TopologicalSpace γ
      inst✝ : OrderTopology γ
      f : α → γ
      hf : ∀ (x : α), LE.le (f x) (Filter.liminf f (nhds x))
      x : α
      y : γ
      F : Filter (Prod α γ)
      F_ne : F.NeBot
      h : LE.le F (Filter.principal (setOf fun p => LE.le (f p.1) p.2))
      h' : LE.le F (nhds { fst := x, snd := y })
      ⊢ Membership.mem (setOf fun p => LE.le (f p.1) p.2) { fst := x, snd := y }
    -/
    rw [nhds_prod_eq, le_prod] at h'
    calc f x ≤ liminf f (𝓝 x) := hf x
    _ ≤ liminf f (map Prod.fst F) := liminf_le_liminf_of_le h'.1
    _ = liminf (f ∘ Prod.fst) F := (Filter.liminf_comp _ _ _).symm
    _ ≤ liminf Prod.snd F := liminf_le_liminf <| by
          simpa using (eventually_principal.2 fun (_ : α × γ) ↦ id).filter_mono h
    _ = y := h'.2.liminf_eq
    /-
      case mpr
      α : Type u_1
      inst✝⁴ : TopologicalSpace α
      γ : Type u_3
      inst✝³ : CompleteLinearOrder γ
      inst✝² : DenselyOrdered γ
      inst✝¹ : TopologicalSpace γ
      inst✝ : OrderTopology γ
      f : α → γ
      ⊢ IsClosed (setOf fun p => LE.le (f p.1) p.2) → LowerSemicontinuous f
    -/
  · rw [lowerSemicontinuous_iff_isClosed_preimage]
    /-
      case mpr
      α : Type u_1
      inst✝⁴ : TopologicalSpace α
      γ : Type u_3
      inst✝³ : CompleteLinearOrder γ
      inst✝² : DenselyOrdered γ
      inst✝¹ : TopologicalSpace γ
      inst✝ : OrderTopology γ
      f : α → γ
      ⊢ IsClosed (setOf fun p => LE.le (f p.1) p.2) → ∀ (y : γ), IsClosed (Set.preim …
    -/
    exact fun hf y ↦ hf.preimage (Continuous.Prod.mk_left y)
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-03-02")]
alias lowerSemicontinuous_iff_IsClosed_epigraph := lowerSemicontinuous_iff_isClosed_epigraph


alias ⟨LowerSemicontinuous.isClosed_epigraph, _⟩ := lowerSemicontinuous_iff_isClosed_epigraph


@[deprecated (since := "2024-03-02")]
alias LowerSemicontinuous.IsClosed_epigraph := LowerSemicontinuous.isClosed_epigraph


theorem ContinuousAt.comp_lowerSemicontinuousWithinAt {g : γ → δ} {f : α → γ}
    (hg : ContinuousAt g (f x)) (hf : LowerSemicontinuousWithinAt f s x) (gmon : Monotone g) :
    LowerSemicontinuousWithinAt (g ∘ f) s x := by
  /-
    α : Type u_1
    inst✝⁶ : TopologicalSpace α
    x : α
    s : Set α
    γ : Type u_3
    inst✝⁵ : LinearOrder γ
    inst✝⁴ : TopologicalSpace γ
    inst✝³ : OrderTopology γ
    δ : Type u_4
    inst✝² : LinearOrder δ
    inst✝¹ : TopologicalSpace δ
    inst✝ : OrderTopology δ
    g : γ → δ
    f : α → γ
    hg : ContinuousAt g (f x)
    hf : LowerSemicontinuousWithinAt f s x
    gmon : Monotone g
    ⊢ LowerSemicontinuousWithinAt (Function.comp g f) s x
  -/
  intro y hy
  /-
    α : Type u_1
    inst✝⁶ : TopologicalSpace α
    x : α
    s : Set α
    γ : Type u_3
    inst✝⁵ : LinearOrder γ
    inst✝⁴ : TopologicalSpace γ
    inst✝³ : OrderTopology γ
    δ : Type u_4
    inst✝² : LinearOrder δ
    inst✝¹ : TopologicalSpace δ
    inst✝ : OrderTopology δ
    g : γ → δ
    f : α → γ
    hg : ContinuousAt g (f x)
    hf : LowerSemicontinuousWithinAt f s x
    gmon : Monotone g
    y : δ
    hy : LT.lt y (Function.comp g f x)
    ⊢ Filter.Eventually (fun x' => LT.lt y (Function.comp g f x')) (nhdsWithin x s)
  -/
  by_cases h : ∃ l, l < f x
  · obtain ⟨z, zlt, hz⟩ : ∃ z < f x, Ioc z (f x) ⊆ g ⁻¹' Ioi y :=
      exists_Ioc_subset_of_mem_nhds (hg (Ioi_mem_nhds hy)) h
    /-
      case pos.intro.intro
      α : Type u_1
      inst✝⁶ : TopologicalSpace α
      x : α
      s : Set α
      γ : Type u_3
      inst✝⁵ : LinearOrder γ
      inst✝⁴ : TopologicalSpace γ
      inst✝³ : OrderTopology γ
      δ : Type u_4
      inst✝² : LinearOrder δ
      inst✝¹ : TopologicalSpace δ
      inst✝ : OrderTopology δ
      g : γ → δ
      f : α → γ
      hg : ContinuousAt g (f x)
      hf : LowerSemicontinuousWithinAt f s x
      gmon : Monotone g
      y : δ
      hy : LT.lt y (Function.comp g f x)
      h : Exists fun l => LT.lt l (f x)
      z : γ
      zlt : LT.lt z (f x)
      hz : HasSubset.Subset (Set.Ioc z (f x)) (Set.preimage g (Set.Ioi y))
      ⊢ Filter.Eventually (fun x' => LT.lt y (Function.comp g f x')) (nhdsWithin x s)
    -/
    filter_upwards [hf z zlt] with a ha
    calc
      y < g (min (f x) (f a)) := hz (by simp [zlt, ha, le_refl])
      _ ≤ g (f a) := gmon (min_le_right _ _)

    /-
      case neg
      α : Type u_1
      inst✝⁶ : TopologicalSpace α
      x : α
      s : Set α
      γ : Type u_3
      inst✝⁵ : LinearOrder γ
      inst✝⁴ : TopologicalSpace γ
      inst✝³ : OrderTopology γ
      δ : Type u_4
      inst✝² : LinearOrder δ
      inst✝¹ : TopologicalSpace δ
      inst✝ : OrderTopology δ
      g : γ → δ
      f : α → γ
      hg : ContinuousAt g (f x)
      hf : LowerSemicontinuousWithinAt f s x
      gmon : Monotone g
      y : δ
      hy : LT.lt y (Function.comp g f x)
      h : Not (Exists fun l => LT.lt l (f x))
      ⊢ Filter.Eventually (fun x' => LT.lt y (Function.comp g f x')) (nhdsWithin x s)
    -/
  · simp only [not_exists, not_lt] at h
    /-
      case neg
      α : Type u_1
      inst✝⁶ : TopologicalSpace α
      x : α
      s : Set α
      γ : Type u_3
      inst✝⁵ : LinearOrder γ
      inst✝⁴ : TopologicalSpace γ
      inst✝³ : OrderTopology γ
      δ : Type u_4
      inst✝² : LinearOrder δ
      inst✝¹ : TopologicalSpace δ
      inst✝ : OrderTopology δ
      g : γ → δ
      f : α → γ
      hg : ContinuousAt g (f x)
      hf : LowerSemicontinuousWithinAt f s x
      gmon : Monotone g
      y : δ
      hy : LT.lt y (Function.comp g f x)
      h : ∀ (x_1 : γ), LE.le (f x) x_1
      ⊢ Filter.Eventually (fun x' => LT.lt y (Function.comp g f x')) (nhdsWithin x s)
    -/
    exact Filter.Eventually.of_forall fun a => hy.trans_le (gmon (h (f a)))
    /-
      🎉 no goals
    -/


theorem ContinuousAt.comp_lowerSemicontinuousAt {g : γ → δ} {f : α → γ} (hg : ContinuousAt g (f x))
    (hf : LowerSemicontinuousAt f x) (gmon : Monotone g) : LowerSemicontinuousAt (g ∘ f) x := by
  /-
    α : Type u_1
    inst✝⁶ : TopologicalSpace α
    x : α
    γ : Type u_3
    inst✝⁵ : LinearOrder γ
    inst✝⁴ : TopologicalSpace γ
    inst✝³ : OrderTopology γ
    δ : Type u_4
    inst✝² : LinearOrder δ
    inst✝¹ : TopologicalSpace δ
    inst✝ : OrderTopology δ
    g : γ → δ
    f : α → γ
    hg : ContinuousAt g (f x)
    hf : LowerSemicontinuousAt f x
    gmon : Monotone g
    ⊢ LowerSemicontinuousAt (Function.comp g f) x
  -/
  simp only [← lowerSemicontinuousWithinAt_univ_iff] at hf ⊢
  /-
    α : Type u_1
    inst✝⁶ : TopologicalSpace α
    x : α
    γ : Type u_3
    inst✝⁵ : LinearOrder γ
    inst✝⁴ : TopologicalSpace γ
    inst✝³ : OrderTopology γ
    δ : Type u_4
    inst✝² : LinearOrder δ
    inst✝¹ : TopologicalSpace δ
    inst✝ : OrderTopology δ
    g : γ → δ
    f : α → γ
    hg : ContinuousAt g (f x)
    gmon : Monotone g
    hf : LowerSemicontinuousWithinAt f Set.univ x
    ⊢ LowerSemicontinuousWithinAt (Function.comp g f) Set.univ x
  -/
  exact hg.comp_lowerSemicontinuousWithinAt hf gmon
  /-
    🎉 no goals
  -/


theorem Continuous.comp_lowerSemicontinuousOn {g : γ → δ} {f : α → γ} (hg : Continuous g)
    (hf : LowerSemicontinuousOn f s) (gmon : Monotone g) : LowerSemicontinuousOn (g ∘ f) s :=
  fun x hx => hg.continuousAt.comp_lowerSemicontinuousWithinAt (hf x hx) gmon


theorem Continuous.comp_lowerSemicontinuous {g : γ → δ} {f : α → γ} (hg : Continuous g)
    (hf : LowerSemicontinuous f) (gmon : Monotone g) : LowerSemicontinuous (g ∘ f) := fun x =>
  hg.continuousAt.comp_lowerSemicontinuousAt (hf x) gmon


theorem ContinuousAt.comp_lowerSemicontinuousWithinAt_antitone {g : γ → δ} {f : α → γ}
    (hg : ContinuousAt g (f x)) (hf : LowerSemicontinuousWithinAt f s x) (gmon : Antitone g) :
    UpperSemicontinuousWithinAt (g ∘ f) s x :=
  @ContinuousAt.comp_lowerSemicontinuousWithinAt α _ x s γ _ _ _ δᵒᵈ _ _ _ g f hg hf gmon


theorem ContinuousAt.comp_lowerSemicontinuousAt_antitone {g : γ → δ} {f : α → γ}
    (hg : ContinuousAt g (f x)) (hf : LowerSemicontinuousAt f x) (gmon : Antitone g) :
    UpperSemicontinuousAt (g ∘ f) x :=
  @ContinuousAt.comp_lowerSemicontinuousAt α _ x γ _ _ _ δᵒᵈ _ _ _ g f hg hf gmon


theorem Continuous.comp_lowerSemicontinuousOn_antitone {g : γ → δ} {f : α → γ} (hg : Continuous g)
    (hf : LowerSemicontinuousOn f s) (gmon : Antitone g) : UpperSemicontinuousOn (g ∘ f) s :=
  fun x hx => hg.continuousAt.comp_lowerSemicontinuousWithinAt_antitone (hf x hx) gmon


theorem Continuous.comp_lowerSemicontinuous_antitone {g : γ → δ} {f : α → γ} (hg : Continuous g)
    (hf : LowerSemicontinuous f) (gmon : Antitone g) : UpperSemicontinuous (g ∘ f) := fun x =>
  hg.continuousAt.comp_lowerSemicontinuousAt_antitone (hf x) gmon


theorem LowerSemicontinuousAt.comp_continuousAt {f : α → β} {g : ι → α} {x : ι}
    (hf : LowerSemicontinuousAt f (g x)) (hg : ContinuousAt g x) :
    LowerSemicontinuousAt (fun x ↦ f (g x)) x :=
  fun _ lt ↦ hg.eventually (hf _ lt)


theorem LowerSemicontinuousAt.comp_continuousAt_of_eq {f : α → β} {g : ι → α} {y : α} {x : ι}
    (hf : LowerSemicontinuousAt f y) (hg : ContinuousAt g x) (hy : g x = y) :
    LowerSemicontinuousAt (fun x ↦ f (g x)) x := by
  /-
    α : Type u_1
    inst✝² : TopologicalSpace α
    β : Type u_2
    inst✝¹ : Preorder β
    ι : Type u_5
    inst✝ : TopologicalSpace ι
    f : α → β
    g : ι → α
    y : α
    x : ι
    hf : LowerSemicontinuousAt f y
    hg : ContinuousAt g x
    hy : Eq (g x) y
    ⊢ LowerSemicontinuousAt (fun x => f (g x)) x
  -/
  rw [← hy] at hf
  /-
    α : Type u_1
    inst✝² : TopologicalSpace α
    β : Type u_2
    inst✝¹ : Preorder β
    ι : Type u_5
    inst✝ : TopologicalSpace ι
    f : α → β
    g : ι → α
    y : α
    x : ι
    hf : LowerSemicontinuousAt f (g x)
    hg : ContinuousAt g x
    hy : Eq (g x) y
    ⊢ LowerSemicontinuousAt (fun x => f (g x)) x
  -/
  exact comp_continuousAt hf hg
  /-
    🎉 no goals
  -/


theorem LowerSemicontinuous.comp_continuous {f : α → β} {g : ι → α}
    (hf : LowerSemicontinuous f) (hg : Continuous g) : LowerSemicontinuous fun x ↦ f (g x) :=
  fun x ↦ (hf (g x)).comp_continuousAt hg.continuousAt


/-- The sum of two lower semicontinuous functions is lower semicontinuous. Formulated with an
explicit continuity assumption on addition, for application to `EReal`. The unprimed version of
the lemma uses `[ContinuousAdd]`. -/
theorem LowerSemicontinuousWithinAt.add' {f g : α → γ} (hf : LowerSemicontinuousWithinAt f s x)
    (hg : LowerSemicontinuousWithinAt g s x)
    (hcont : ContinuousAt (fun p : γ × γ => p.1 + p.2) (f x, g x)) :
    LowerSemicontinuousWithinAt (fun z => f z + g z) s x := by
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    x : α
    s : Set α
    γ : Type u_4
    inst✝² : LinearOrderedAddCommMonoid γ
    inst✝¹ : TopologicalSpace γ
    inst✝ : OrderTopology γ
    f g : α → γ
    hf : LowerSemicontinuousWithinAt f s x
    hg : LowerSemicontinuousWithinAt g s x
    hcont : ContinuousAt (fun p => HAdd.hAdd p.1 p.2) { fst := f x, snd := g x }
    ⊢ LowerSemicontinuousWithinAt (fun z => HAdd.hAdd (f z) (g z)) s x
  -/
  intro y hy
  obtain ⟨u, v, u_open, xu, v_open, xv, h⟩ :
    ∃ u v : Set γ,
      IsOpen u ∧ f x ∈ u ∧ IsOpen v ∧ g x ∈ v ∧ u ×ˢ v ⊆ { p : γ × γ | y < p.fst + p.snd } :=
    mem_nhds_prod_iff'.1 (hcont (isOpen_Ioi.mem_nhds hy))
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝³ : TopologicalSpace α
    x : α
    s : Set α
    γ : Type u_4
    inst✝² : LinearOrderedAddCommMonoid γ
    inst✝¹ : TopologicalSpace γ
    inst✝ : OrderTopology γ
    f g : α → γ
    hf : LowerSemicontinuousWithinAt f s x
    hg : LowerSemicontinuousWithinAt g s x
    hcont : ContinuousAt (fun p => HAdd.hAdd p.1 p.2) { fst := f x, snd := g x }
    y : γ
    hy : LT.lt y ((fun z => HAdd.hAdd (f z) (g z)) x)
    u v : Set γ
    u_open : IsOpen u
    xu : Membership.mem u (f x)
    v_open : IsOpen v
    xv : Membership.mem v (g x)
    h : HasSubset.Subset (SProd.sprod u v) (setOf fun p => LT.lt y (HAdd.hAdd p.1  …
    ⊢ Filter.Eventually (fun x' => LT.lt y ((fun z => HAdd.hAdd (f z) (g z)) x'))  …
  -/
  by_cases hx₁ : ∃ l, l < f x
  · obtain ⟨z₁, z₁lt, h₁⟩ : ∃ z₁ < f x, Ioc z₁ (f x) ⊆ u :=
      exists_Ioc_subset_of_mem_nhds (u_open.mem_nhds xu) hx₁
    /-
      case pos.intro.intro
      α : Type u_1
      inst✝³ : TopologicalSpace α
      x : α
      s : Set α
      γ : Type u_4
      inst✝² : LinearOrderedAddCommMonoid γ
      inst✝¹ : TopologicalSpace γ
      inst✝ : OrderTopology γ
      f g : α → γ
      hf : LowerSemicontinuousWithinAt f s x
      hg : LowerSemicontinuousWithinAt g s x
      hcont : ContinuousAt (fun p => HAdd.hAdd p.1 p.2) { fst := f x, snd := g x }
      y : γ
      hy : LT.lt y ((fun z => HAdd.hAdd (f z) (g z)) x)
      u v : Set γ
      u_open : IsOpen u
      xu : Membership.mem u (f x)
      v_open : IsOpen v
      xv : Membership.mem v (g x)
      h : HasSubset.Subset (SProd.sprod u v) (setOf fun p => LT.lt y (HAdd.hAdd p.1  …
      hx₁ : Exists fun l => LT.lt l (f x)
      z₁ : γ
      z₁lt : LT.lt z₁ (f x)
      h₁ : HasSubset.Subset (Set.Ioc z₁ (f x)) u
      ⊢ Filter.Eventually (fun x' => LT.lt y ((fun z => HAdd.hAdd (f z) (g z)) x'))  …
    -/
    by_cases hx₂ : ∃ l, l < g x
    · obtain ⟨z₂, z₂lt, h₂⟩ : ∃ z₂ < g x, Ioc z₂ (g x) ⊆ v :=
        exists_Ioc_subset_of_mem_nhds (v_open.mem_nhds xv) hx₂
      /-
        case pos.intro.intro
        α : Type u_1
        inst✝³ : TopologicalSpace α
        x : α
        s : Set α
        γ : Type u_4
        inst✝² : LinearOrderedAddCommMonoid γ
        inst✝¹ : TopologicalSpace γ
        inst✝ : OrderTopology γ
        f g : α → γ
        hf : LowerSemicontinuousWithinAt f s x
        hg : LowerSemicontinuousWithinAt g s x
        hcont : ContinuousAt (fun p => HAdd.hAdd p.1 p.2) { fst := f x, snd := g x }
        y : γ
        hy : LT.lt y ((fun z => HAdd.hAdd (f z) (g z)) x)
        u v : Set γ
        u_open : IsOpen u
        xu : Membership.mem u (f x)
        v_open : IsOpen v
        xv : Membership.mem v (g x)
        h : HasSubset.Subset (SProd.sprod u v) (setOf fun p => LT.lt y (HAdd.hAdd p.1  …
        hx₁ : Exists fun l => LT.lt l (f x)
        z₁ : γ
        z₁lt : LT.lt z₁ (f x)
        h₁ : HasSubset.Subset (Set.Ioc z₁ (f x)) u
        hx₂ : Exists fun l => LT.lt l (g x)
        z₂ : γ
        z₂lt : LT.lt z₂ (g x)
        h₂ : HasSubset.Subset (Set.Ioc z₂ (g x)) v
        ⊢ Filter.Eventually (fun x' => LT.lt y ((fun z => HAdd.hAdd (f z) (g z)) x'))  …
      -/
      filter_upwards [hf z₁ z₁lt, hg z₂ z₂lt] with z h₁z h₂z
      have A1 : min (f z) (f x) ∈ u := by
        by_cases H : f z ≤ f x
        · simpa [H] using h₁ ⟨h₁z, H⟩
        · simpa [le_of_not_le H]
      have A2 : min (g z) (g x) ∈ v := by
        by_cases H : g z ≤ g x
        · simpa [H] using h₂ ⟨h₂z, H⟩
        · simpa [le_of_not_le H]
      /-
        case h
        α : Type u_1
        inst✝³ : TopologicalSpace α
        x : α
        s : Set α
        γ : Type u_4
        inst✝² : LinearOrderedAddCommMonoid γ
        inst✝¹ : TopologicalSpace γ
        inst✝ : OrderTopology γ
        f g : α → γ
        hf : LowerSemicontinuousWithinAt f s x
        hg : LowerSemicontinuousWithinAt g s x
        hcont : ContinuousAt (fun p => HAdd.hAdd p.1 p.2) { fst := f x, snd := g x }
        y : γ
        hy : LT.lt y ((fun z => HAdd.hAdd (f z) (g z)) x)
        u v : Set γ
        u_open : IsOpen u
        xu : Membership.mem u (f x)
        v_open : IsOpen v
        xv : Membership.mem v (g x)
        h : HasSubset.Subset (SProd.sprod u v) (setOf fun p => LT.lt y (HAdd.hAdd p.1  …
        hx₁ : Exists fun l => LT.lt l (f x)
        z₁ : γ
        z₁lt : LT.lt z₁ (f x)
        h₁ : HasSubset.Subset (Set.Ioc z₁ (f x)) u
        hx₂ : Exists fun l => LT.lt l (g x)
        z₂ : γ
        z₂lt : LT.lt z₂ (g x)
        h₂ : HasSubset.Subset (Set.Ioc z₂ (g x)) v
        z : α
        h₁z : LT.lt z₁ (f z)
        h₂z : LT.lt z₂ (g z)
        A1 : Membership.mem u (Min.min (f z) (f x))
        A2 : Membership.mem v (Min.min (g z) (g x))
        ⊢ LT.lt y (HAdd.hAdd (f z) (g z))
      -/
      have : (min (f z) (f x), min (g z) (g x)) ∈ u ×ˢ v := ⟨A1, A2⟩
      calc
        y < min (f z) (f x) + min (g z) (g x) := h this
        _ ≤ f z + g z := add_le_add (min_le_left _ _) (min_le_left _ _)

      /-
        case neg
        α : Type u_1
        inst✝³ : TopologicalSpace α
        x : α
        s : Set α
        γ : Type u_4
        inst✝² : LinearOrderedAddCommMonoid γ
        inst✝¹ : TopologicalSpace γ
        inst✝ : OrderTopology γ
        f g : α → γ
        hf : LowerSemicontinuousWithinAt f s x
        hg : LowerSemicontinuousWithinAt g s x
        hcont : ContinuousAt (fun p => HAdd.hAdd p.1 p.2) { fst := f x, snd := g x }
        y : γ
        hy : LT.lt y ((fun z => HAdd.hAdd (f z) (g z)) x)
        u v : Set γ
        u_open : IsOpen u
        xu : Membership.mem u (f x)
        v_open : IsOpen v
        xv : Membership.mem v (g x)
        h : HasSubset.Subset (SProd.sprod u v) (setOf fun p => LT.lt y (HAdd.hAdd p.1  …
        hx₁ : Exists fun l => LT.lt l (f x)
        z₁ : γ
        z₁lt : LT.lt z₁ (f x)
        h₁ : HasSubset.Subset (Set.Ioc z₁ (f x)) u
        hx₂ : Not (Exists fun l => LT.lt l (g x))
        ⊢ Filter.Eventually (fun x' => LT.lt y ((fun z => HAdd.hAdd (f z) (g z)) x'))  …
      -/
    · simp only [not_exists, not_lt] at hx₂
      /-
        case neg
        α : Type u_1
        inst✝³ : TopologicalSpace α
        x : α
        s : Set α
        γ : Type u_4
        inst✝² : LinearOrderedAddCommMonoid γ
        inst✝¹ : TopologicalSpace γ
        inst✝ : OrderTopology γ
        f g : α → γ
        hf : LowerSemicontinuousWithinAt f s x
        hg : LowerSemicontinuousWithinAt g s x
        hcont : ContinuousAt (fun p => HAdd.hAdd p.1 p.2) { fst := f x, snd := g x }
        y : γ
        hy : LT.lt y ((fun z => HAdd.hAdd (f z) (g z)) x)
        u v : Set γ
        u_open : IsOpen u
        xu : Membership.mem u (f x)
        v_open : IsOpen v
        xv : Membership.mem v (g x)
        h : HasSubset.Subset (SProd.sprod u v) (setOf fun p => LT.lt y (HAdd.hAdd p.1  …
        hx₁ : Exists fun l => LT.lt l (f x)
        z₁ : γ
        z₁lt : LT.lt z₁ (f x)
        h₁ : HasSubset.Subset (Set.Ioc z₁ (f x)) u
        hx₂ : ∀ (x_1 : γ), LE.le (g x) x_1
        ⊢ Filter.Eventually (fun x' => LT.lt y ((fun z => HAdd.hAdd (f z) (g z)) x'))  …
      -/
      filter_upwards [hf z₁ z₁lt] with z h₁z
      have A1 : min (f z) (f x) ∈ u := by
        by_cases H : f z ≤ f x
        · simpa [H] using h₁ ⟨h₁z, H⟩
        · simpa [le_of_not_le H]
      /-
        case h
        α : Type u_1
        inst✝³ : TopologicalSpace α
        x : α
        s : Set α
        γ : Type u_4
        inst✝² : LinearOrderedAddCommMonoid γ
        inst✝¹ : TopologicalSpace γ
        inst✝ : OrderTopology γ
        f g : α → γ
        hf : LowerSemicontinuousWithinAt f s x
        hg : LowerSemicontinuousWithinAt g s x
        hcont : ContinuousAt (fun p => HAdd.hAdd p.1 p.2) { fst := f x, snd := g x }
        y : γ
        hy : LT.lt y ((fun z => HAdd.hAdd (f z) (g z)) x)
        u v : Set γ
        u_open : IsOpen u
        xu : Membership.mem u (f x)
        v_open : IsOpen v
        xv : Membership.mem v (g x)
        h : HasSubset.Subset (SProd.sprod u v) (setOf fun p => LT.lt y (HAdd.hAdd p.1  …
        hx₁ : Exists fun l => LT.lt l (f x)
        z₁ : γ
        z₁lt : LT.lt z₁ (f x)
        h₁ : HasSubset.Subset (Set.Ioc z₁ (f x)) u
        hx₂ : ∀ (x_1 : γ), LE.le (g x) x_1
        z : α
        h₁z : LT.lt z₁ (f z)
        A1 : Membership.mem u (Min.min (f z) (f x))
        ⊢ LT.lt y (HAdd.hAdd (f z) (g z))
      -/
      have : (min (f z) (f x), g x) ∈ u ×ˢ v := ⟨A1, xv⟩
      calc
        y < min (f z) (f x) + g x := h this
        _ ≤ f z + g z := add_le_add (min_le_left _ _) (hx₂ (g z))

    /-
      case neg
      α : Type u_1
      inst✝³ : TopologicalSpace α
      x : α
      s : Set α
      γ : Type u_4
      inst✝² : LinearOrderedAddCommMonoid γ
      inst✝¹ : TopologicalSpace γ
      inst✝ : OrderTopology γ
      f g : α → γ
      hf : LowerSemicontinuousWithinAt f s x
      hg : LowerSemicontinuousWithinAt g s x
      hcont : ContinuousAt (fun p => HAdd.hAdd p.1 p.2) { fst := f x, snd := g x }
      y : γ
      hy : LT.lt y ((fun z => HAdd.hAdd (f z) (g z)) x)
      u v : Set γ
      u_open : IsOpen u
      xu : Membership.mem u (f x)
      v_open : IsOpen v
      xv : Membership.mem v (g x)
      h : HasSubset.Subset (SProd.sprod u v) (setOf fun p => LT.lt y (HAdd.hAdd p.1  …
      hx₁ : Not (Exists fun l => LT.lt l (f x))
      ⊢ Filter.Eventually (fun x' => LT.lt y ((fun z => HAdd.hAdd (f z) (g z)) x'))  …
    -/
  · simp only [not_exists, not_lt] at hx₁
    /-
      case neg
      α : Type u_1
      inst✝³ : TopologicalSpace α
      x : α
      s : Set α
      γ : Type u_4
      inst✝² : LinearOrderedAddCommMonoid γ
      inst✝¹ : TopologicalSpace γ
      inst✝ : OrderTopology γ
      f g : α → γ
      hf : LowerSemicontinuousWithinAt f s x
      hg : LowerSemicontinuousWithinAt g s x
      hcont : ContinuousAt (fun p => HAdd.hAdd p.1 p.2) { fst := f x, snd := g x }
      y : γ
      hy : LT.lt y ((fun z => HAdd.hAdd (f z) (g z)) x)
      u v : Set γ
      u_open : IsOpen u
      xu : Membership.mem u (f x)
      v_open : IsOpen v
      xv : Membership.mem v (g x)
      h : HasSubset.Subset (SProd.sprod u v) (setOf fun p => LT.lt y (HAdd.hAdd p.1  …
      hx₁ : ∀ (x_1 : γ), LE.le (f x) x_1
      ⊢ Filter.Eventually (fun x' => LT.lt y ((fun z => HAdd.hAdd (f z) (g z)) x'))  …
    -/
    by_cases hx₂ : ∃ l, l < g x
    · obtain ⟨z₂, z₂lt, h₂⟩ : ∃ z₂ < g x, Ioc z₂ (g x) ⊆ v :=
        exists_Ioc_subset_of_mem_nhds (v_open.mem_nhds xv) hx₂
      /-
        case pos.intro.intro
        α : Type u_1
        inst✝³ : TopologicalSpace α
        x : α
        s : Set α
        γ : Type u_4
        inst✝² : LinearOrderedAddCommMonoid γ
        inst✝¹ : TopologicalSpace γ
        inst✝ : OrderTopology γ
        f g : α → γ
        hf : LowerSemicontinuousWithinAt f s x
        hg : LowerSemicontinuousWithinAt g s x
        hcont : ContinuousAt (fun p => HAdd.hAdd p.1 p.2) { fst := f x, snd := g x }
        y : γ
        hy : LT.lt y ((fun z => HAdd.hAdd (f z) (g z)) x)
        u v : Set γ
        u_open : IsOpen u
        xu : Membership.mem u (f x)
        v_open : IsOpen v
        xv : Membership.mem v (g x)
        h : HasSubset.Subset (SProd.sprod u v) (setOf fun p => LT.lt y (HAdd.hAdd p.1  …
        hx₁ : ∀ (x_1 : γ), LE.le (f x) x_1
        hx₂ : Exists fun l => LT.lt l (g x)
        z₂ : γ
        z₂lt : LT.lt z₂ (g x)
        h₂ : HasSubset.Subset (Set.Ioc z₂ (g x)) v
        ⊢ Filter.Eventually (fun x' => LT.lt y ((fun z => HAdd.hAdd (f z) (g z)) x'))  …
      -/
      filter_upwards [hg z₂ z₂lt] with z h₂z
      have A2 : min (g z) (g x) ∈ v := by
        by_cases H : g z ≤ g x
        · simpa [H] using h₂ ⟨h₂z, H⟩
        · simpa [le_of_not_le H] using h₂ ⟨z₂lt, le_rfl⟩
      /-
        case h
        α : Type u_1
        inst✝³ : TopologicalSpace α
        x : α
        s : Set α
        γ : Type u_4
        inst✝² : LinearOrderedAddCommMonoid γ
        inst✝¹ : TopologicalSpace γ
        inst✝ : OrderTopology γ
        f g : α → γ
        hf : LowerSemicontinuousWithinAt f s x
        hg : LowerSemicontinuousWithinAt g s x
        hcont : ContinuousAt (fun p => HAdd.hAdd p.1 p.2) { fst := f x, snd := g x }
        y : γ
        hy : LT.lt y ((fun z => HAdd.hAdd (f z) (g z)) x)
        u v : Set γ
        u_open : IsOpen u
        xu : Membership.mem u (f x)
        v_open : IsOpen v
        xv : Membership.mem v (g x)
        h : HasSubset.Subset (SProd.sprod u v) (setOf fun p => LT.lt y (HAdd.hAdd p.1  …
        hx₁ : ∀ (x_1 : γ), LE.le (f x) x_1
        hx₂ : Exists fun l => LT.lt l (g x)
        z₂ : γ
        z₂lt : LT.lt z₂ (g x)
        h₂ : HasSubset.Subset (Set.Ioc z₂ (g x)) v
        z : α
        h₂z : LT.lt z₂ (g z)
        A2 : Membership.mem v (Min.min (g z) (g x))
        ⊢ LT.lt y (HAdd.hAdd (f z) (g z))
      -/
      have : (f x, min (g z) (g x)) ∈ u ×ˢ v := ⟨xu, A2⟩
      calc
        y < f x + min (g z) (g x) := h this
        _ ≤ f z + g z := add_le_add (hx₁ (f z)) (min_le_left _ _)
      /-
        case neg
        α : Type u_1
        inst✝³ : TopologicalSpace α
        x : α
        s : Set α
        γ : Type u_4
        inst✝² : LinearOrderedAddCommMonoid γ
        inst✝¹ : TopologicalSpace γ
        inst✝ : OrderTopology γ
        f g : α → γ
        hf : LowerSemicontinuousWithinAt f s x
        hg : LowerSemicontinuousWithinAt g s x
        hcont : ContinuousAt (fun p => HAdd.hAdd p.1 p.2) { fst := f x, snd := g x }
        y : γ
        hy : LT.lt y ((fun z => HAdd.hAdd (f z) (g z)) x)
        u v : Set γ
        u_open : IsOpen u
        xu : Membership.mem u (f x)
        v_open : IsOpen v
        xv : Membership.mem v (g x)
        h : HasSubset.Subset (SProd.sprod u v) (setOf fun p => LT.lt y (HAdd.hAdd p.1  …
        hx₁ : ∀ (x_1 : γ), LE.le (f x) x_1
        hx₂ : Not (Exists fun l => LT.lt l (g x))
        ⊢ Filter.Eventually (fun x' => LT.lt y ((fun z => HAdd.hAdd (f z) (g z)) x'))  …
      -/
    · simp only [not_exists, not_lt] at hx₁ hx₂
      /-
        case neg
        α : Type u_1
        inst✝³ : TopologicalSpace α
        x : α
        s : Set α
        γ : Type u_4
        inst✝² : LinearOrderedAddCommMonoid γ
        inst✝¹ : TopologicalSpace γ
        inst✝ : OrderTopology γ
        f g : α → γ
        hf : LowerSemicontinuousWithinAt f s x
        hg : LowerSemicontinuousWithinAt g s x
        hcont : ContinuousAt (fun p => HAdd.hAdd p.1 p.2) { fst := f x, snd := g x }
        y : γ
        hy : LT.lt y ((fun z => HAdd.hAdd (f z) (g z)) x)
        u v : Set γ
        u_open : IsOpen u
        xu : Membership.mem u (f x)
        v_open : IsOpen v
        xv : Membership.mem v (g x)
        h : HasSubset.Subset (SProd.sprod u v) (setOf fun p => LT.lt y (HAdd.hAdd p.1  …
        hx₁ : ∀ (x_1 : γ), LE.le (f x) x_1
        hx₂ : ∀ (x_1 : γ), LE.le (g x) x_1
        ⊢ Filter.Eventually (fun x' => LT.lt y ((fun z => HAdd.hAdd (f z) (g z)) x'))  …
      -/
      apply Filter.Eventually.of_forall
      /-
        case neg.hp
        α : Type u_1
        inst✝³ : TopologicalSpace α
        x : α
        s : Set α
        γ : Type u_4
        inst✝² : LinearOrderedAddCommMonoid γ
        inst✝¹ : TopologicalSpace γ
        inst✝ : OrderTopology γ
        f g : α → γ
        hf : LowerSemicontinuousWithinAt f s x
        hg : LowerSemicontinuousWithinAt g s x
        hcont : ContinuousAt (fun p => HAdd.hAdd p.1 p.2) { fst := f x, snd := g x }
        y : γ
        hy : LT.lt y ((fun z => HAdd.hAdd (f z) (g z)) x)
        u v : Set γ
        u_open : IsOpen u
        xu : Membership.mem u (f x)
        v_open : IsOpen v
        xv : Membership.mem v (g x)
        h : HasSubset.Subset (SProd.sprod u v) (setOf fun p => LT.lt y (HAdd.hAdd p.1  …
        hx₁ : ∀ (x_1 : γ), LE.le (f x) x_1
        hx₂ : ∀ (x_1 : γ), LE.le (g x) x_1
        ⊢ ∀ (x : α), LT.lt y ((fun z => HAdd.hAdd (f z) (g z)) x)
      -/
      intro z
      /-
        case neg.hp
        α : Type u_1
        inst✝³ : TopologicalSpace α
        x : α
        s : Set α
        γ : Type u_4
        inst✝² : LinearOrderedAddCommMonoid γ
        inst✝¹ : TopologicalSpace γ
        inst✝ : OrderTopology γ
        f g : α → γ
        hf : LowerSemicontinuousWithinAt f s x
        hg : LowerSemicontinuousWithinAt g s x
        hcont : ContinuousAt (fun p => HAdd.hAdd p.1 p.2) { fst := f x, snd := g x }
        y : γ
        hy : LT.lt y ((fun z => HAdd.hAdd (f z) (g z)) x)
        u v : Set γ
        u_open : IsOpen u
        xu : Membership.mem u (f x)
        v_open : IsOpen v
        xv : Membership.mem v (g x)
        h : HasSubset.Subset (SProd.sprod u v) (setOf fun p => LT.lt y (HAdd.hAdd p.1  …
        hx₁ : ∀ (x_1 : γ), LE.le (f x) x_1
        hx₂ : ∀ (x_1 : γ), LE.le (g x) x_1
        z : α
        ⊢ LT.lt y ((fun z => HAdd.hAdd (f z) (g z)) z)
      -/
      have : (f x, g x) ∈ u ×ˢ v := ⟨xu, xv⟩
      calc
        y < f x + g x := h this
        _ ≤ f z + g z := add_le_add (hx₁ (f z)) (hx₂ (g z))


/-- The sum of two lower semicontinuous functions is lower semicontinuous. Formulated with an
explicit continuity assumption on addition, for application to `EReal`. The unprimed version of
the lemma uses `[ContinuousAdd]`. -/
theorem LowerSemicontinuousAt.add' {f g : α → γ} (hf : LowerSemicontinuousAt f x)
    (hg : LowerSemicontinuousAt g x)
    (hcont : ContinuousAt (fun p : γ × γ => p.1 + p.2) (f x, g x)) :
    LowerSemicontinuousAt (fun z => f z + g z) x := by
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    x : α
    γ : Type u_4
    inst✝² : LinearOrderedAddCommMonoid γ
    inst✝¹ : TopologicalSpace γ
    inst✝ : OrderTopology γ
    f g : α → γ
    hf : LowerSemicontinuousAt f x
    hg : LowerSemicontinuousAt g x
    hcont : ContinuousAt (fun p => HAdd.hAdd p.1 p.2) { fst := f x, snd := g x }
    ⊢ LowerSemicontinuousAt (fun z => HAdd.hAdd (f z) (g z)) x
  -/
  simp_rw [← lowerSemicontinuousWithinAt_univ_iff] at *
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    x : α
    γ : Type u_4
    inst✝² : LinearOrderedAddCommMonoid γ
    inst✝¹ : TopologicalSpace γ
    inst✝ : OrderTopology γ
    f g : α → γ
    hcont : ContinuousAt (fun p => HAdd.hAdd p.1 p.2) { fst := f x, snd := g x }
    hf : LowerSemicontinuousWithinAt f Set.univ x
    hg : LowerSemicontinuousWithinAt g Set.univ x
    ⊢ LowerSemicontinuousWithinAt (fun z => HAdd.hAdd (f z) (g z)) Set.univ x
  -/
  exact hf.add' hg hcont
  /-
    🎉 no goals
  -/


/-- The sum of two lower semicontinuous functions is lower semicontinuous. Formulated with an
explicit continuity assumption on addition, for application to `EReal`. The unprimed version of
the lemma uses `[ContinuousAdd]`. -/
theorem LowerSemicontinuousOn.add' {f g : α → γ} (hf : LowerSemicontinuousOn f s)
    (hg : LowerSemicontinuousOn g s)
    (hcont : ∀ x ∈ s, ContinuousAt (fun p : γ × γ => p.1 + p.2) (f x, g x)) :
    LowerSemicontinuousOn (fun z => f z + g z) s := fun x hx =>
  (hf x hx).add' (hg x hx) (hcont x hx)


/-- The sum of two lower semicontinuous functions is lower semicontinuous. Formulated with an
explicit continuity assumption on addition, for application to `EReal`. The unprimed version of
the lemma uses `[ContinuousAdd]`. -/
theorem LowerSemicontinuous.add' {f g : α → γ} (hf : LowerSemicontinuous f)
    (hg : LowerSemicontinuous g)
    (hcont : ∀ x, ContinuousAt (fun p : γ × γ => p.1 + p.2) (f x, g x)) :
    LowerSemicontinuous fun z => f z + g z := fun x => (hf x).add' (hg x) (hcont x)


/-- The sum of two lower semicontinuous functions is lower semicontinuous. Formulated with
`[ContinuousAdd]`. The primed version of the lemma uses an explicit continuity assumption on
addition, for application to `EReal`. -/
theorem LowerSemicontinuousWithinAt.add {f g : α → γ} (hf : LowerSemicontinuousWithinAt f s x)
    (hg : LowerSemicontinuousWithinAt g s x) :
    LowerSemicontinuousWithinAt (fun z => f z + g z) s x :=
  hf.add' hg continuous_add.continuousAt


/-- The sum of two lower semicontinuous functions is lower semicontinuous. Formulated with
`[ContinuousAdd]`. The primed version of the lemma uses an explicit continuity assumption on
addition, for application to `EReal`. -/
theorem LowerSemicontinuousAt.add {f g : α → γ} (hf : LowerSemicontinuousAt f x)
    (hg : LowerSemicontinuousAt g x) : LowerSemicontinuousAt (fun z => f z + g z) x :=
  hf.add' hg continuous_add.continuousAt


/-- The sum of two lower semicontinuous functions is lower semicontinuous. Formulated with
`[ContinuousAdd]`. The primed version of the lemma uses an explicit continuity assumption on
addition, for application to `EReal`. -/
theorem LowerSemicontinuousOn.add {f g : α → γ} (hf : LowerSemicontinuousOn f s)
    (hg : LowerSemicontinuousOn g s) : LowerSemicontinuousOn (fun z => f z + g z) s :=
  hf.add' hg fun _x _hx => continuous_add.continuousAt


/-- The sum of two lower semicontinuous functions is lower semicontinuous. Formulated with
`[ContinuousAdd]`. The primed version of the lemma uses an explicit continuity assumption on
addition, for application to `EReal`. -/
theorem LowerSemicontinuous.add {f g : α → γ} (hf : LowerSemicontinuous f)
    (hg : LowerSemicontinuous g) : LowerSemicontinuous fun z => f z + g z :=
  hf.add' hg fun _x => continuous_add.continuousAt


theorem lowerSemicontinuousWithinAt_sum {f : ι → α → γ} {a : Finset ι}
    (ha : ∀ i ∈ a, LowerSemicontinuousWithinAt (f i) s x) :
    LowerSemicontinuousWithinAt (fun z => ∑ i ∈ a, f i z) s x := by
  classical
    induction' a using Finset.induction_on with i a ia IH
    · exact lowerSemicontinuousWithinAt_const
    · simp only [ia, Finset.sum_insert, not_false_iff]
      exact
        LowerSemicontinuousWithinAt.add (ha _ (Finset.mem_insert_self i a))
          (IH fun j ja => ha j (Finset.mem_insert_of_mem ja))


theorem lowerSemicontinuousAt_sum {f : ι → α → γ} {a : Finset ι}
    (ha : ∀ i ∈ a, LowerSemicontinuousAt (f i) x) :
    LowerSemicontinuousAt (fun z => ∑ i ∈ a, f i z) x := by
  /-
    α : Type u_1
    inst✝⁴ : TopologicalSpace α
    x : α
    ι : Type u_3
    γ : Type u_4
    inst✝³ : LinearOrderedAddCommMonoid γ
    inst✝² : TopologicalSpace γ
    inst✝¹ : OrderTopology γ
    inst✝ : ContinuousAdd γ
    f : ι → α → γ
    a : Finset ι
    ha : ∀ (i : ι), Membership.mem a i → LowerSemicontinuousAt (f i) x
    ⊢ LowerSemicontinuousAt (fun z => a.sum fun i => f i z) x
  -/
  simp_rw [← lowerSemicontinuousWithinAt_univ_iff] at *
  /-
    α : Type u_1
    inst✝⁴ : TopologicalSpace α
    x : α
    ι : Type u_3
    γ : Type u_4
    inst✝³ : LinearOrderedAddCommMonoid γ
    inst✝² : TopologicalSpace γ
    inst✝¹ : OrderTopology γ
    inst✝ : ContinuousAdd γ
    f : ι → α → γ
    a : Finset ι
    ha : ∀ (i : ι), Membership.mem a i → LowerSemicontinuousWithinAt (f i) Set.uni …
    ⊢ LowerSemicontinuousWithinAt (fun z => a.sum fun i => f i z) Set.univ x
  -/
  exact lowerSemicontinuousWithinAt_sum ha
  /-
    🎉 no goals
  -/


theorem lowerSemicontinuousOn_sum {f : ι → α → γ} {a : Finset ι}
    (ha : ∀ i ∈ a, LowerSemicontinuousOn (f i) s) :
    LowerSemicontinuousOn (fun z => ∑ i ∈ a, f i z) s := fun x hx =>
  lowerSemicontinuousWithinAt_sum fun i hi => ha i hi x hx


theorem lowerSemicontinuous_sum {f : ι → α → γ} {a : Finset ι}
    (ha : ∀ i ∈ a, LowerSemicontinuous (f i)) : LowerSemicontinuous fun z => ∑ i ∈ a, f i z :=
  fun x => lowerSemicontinuousAt_sum fun i hi => ha i hi x


theorem lowerSemicontinuousWithinAt_ciSup {f : ι → α → δ'}
    (bdd : ∀ᶠ y in 𝓝[s] x, BddAbove (range fun i => f i y))
    (h : ∀ i, LowerSemicontinuousWithinAt (f i) s x) :
    LowerSemicontinuousWithinAt (fun x' => ⨆ i, f i x') s x := by
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    x : α
    s : Set α
    ι : Sort u_3
    δ' : Type u_5
    inst✝ : ConditionallyCompleteLinearOrder δ'
    f : ι → α → δ'
    bdd : Filter.Eventually (fun y => BddAbove (Set.range fun i => f i y)) (nhdsWi …
    h : ∀ (i : ι), LowerSemicontinuousWithinAt (f i) s x
    ⊢ LowerSemicontinuousWithinAt (fun x' => iSup fun i => f i x') s x
  -/
  cases isEmpty_or_nonempty ι
    /-
      case inl
      α : Type u_1
      inst✝¹ : TopologicalSpace α
      x : α
      s : Set α
      ι : Sort u_3
      δ' : Type u_5
      inst✝ : ConditionallyCompleteLinearOrder δ'
      f : ι → α → δ'
      bdd : Filter.Eventually (fun y => BddAbove (Set.range fun i => f i y)) (nhdsWi …
      h : ∀ (i : ι), LowerSemicontinuousWithinAt (f i) s x
      h✝ : IsEmpty ι
      ⊢ LowerSemicontinuousWithinAt (fun x' => iSup fun i => f i x') s x
    -/
  · simpa only [iSup_of_empty'] using lowerSemicontinuousWithinAt_const
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      inst✝¹ : TopologicalSpace α
      x : α
      s : Set α
      ι : Sort u_3
      δ' : Type u_5
      inst✝ : ConditionallyCompleteLinearOrder δ'
      f : ι → α → δ'
      bdd : Filter.Eventually (fun y => BddAbove (Set.range fun i => f i y)) (nhdsWi …
      h : ∀ (i : ι), LowerSemicontinuousWithinAt (f i) s x
      h✝ : Nonempty ι
      ⊢ LowerSemicontinuousWithinAt (fun x' => iSup fun i => f i x') s x
    -/
  · intro y hy
    /-
      case inr
      α : Type u_1
      inst✝¹ : TopologicalSpace α
      x : α
      s : Set α
      ι : Sort u_3
      δ' : Type u_5
      inst✝ : ConditionallyCompleteLinearOrder δ'
      f : ι → α → δ'
      bdd : Filter.Eventually (fun y => BddAbove (Set.range fun i => f i y)) (nhdsWi …
      h : ∀ (i : ι), LowerSemicontinuousWithinAt (f i) s x
      h✝ : Nonempty ι
      y : δ'
      hy : LT.lt y ((fun x' => iSup fun i => f i x') x)
      ⊢ Filter.Eventually (fun x' => LT.lt y ((fun x' => iSup fun i => f i x') x'))  …
    -/
    rcases exists_lt_of_lt_ciSup hy with ⟨i, hi⟩
    /-
      case inr.intro
      α : Type u_1
      inst✝¹ : TopologicalSpace α
      x : α
      s : Set α
      ι : Sort u_3
      δ' : Type u_5
      inst✝ : ConditionallyCompleteLinearOrder δ'
      f : ι → α → δ'
      bdd : Filter.Eventually (fun y => BddAbove (Set.range fun i => f i y)) (nhdsWi …
      h : ∀ (i : ι), LowerSemicontinuousWithinAt (f i) s x
      h✝ : Nonempty ι
      y : δ'
      hy : LT.lt y ((fun x' => iSup fun i => f i x') x)
      i : ι
      hi : LT.lt y (f i x)
      ⊢ Filter.Eventually (fun x' => LT.lt y ((fun x' => iSup fun i => f i x') x'))  …
    -/
    filter_upwards [h i y hi, bdd] with y hy hy' using hy.trans_le (le_ciSup hy' i)
    /-
      🎉 no goals
    -/


theorem lowerSemicontinuousWithinAt_iSup {f : ι → α → δ}
    (h : ∀ i, LowerSemicontinuousWithinAt (f i) s x) :
    LowerSemicontinuousWithinAt (fun x' => ⨆ i, f i x') s x :=
                                        /-
                                          α : Type u_1
                                          inst✝¹ : TopologicalSpace α
                                          x : α
                                          s : Set α
                                          ι : Sort u_3
                                          δ : Type u_4
                                          inst✝ : CompleteLinearOrder δ
                                          f : ι → α → δ
                                          h : ∀ (i : ι), LowerSemicontinuousWithinAt (f i) s x
                                          ⊢ Filter.Eventually (fun y => BddAbove (Set.range fun i => f i y)) (nhdsWithin …
                                        -/
  lowerSemicontinuousWithinAt_ciSup (by simp) h
                                        /-
                                          🎉 no goals
                                        -/


theorem lowerSemicontinuousWithinAt_biSup {p : ι → Prop} {f : ∀ i, p i → α → δ}
    (h : ∀ i hi, LowerSemicontinuousWithinAt (f i hi) s x) :
    LowerSemicontinuousWithinAt (fun x' => ⨆ (i) (hi), f i hi x') s x :=
  lowerSemicontinuousWithinAt_iSup fun i => lowerSemicontinuousWithinAt_iSup fun hi => h i hi


theorem lowerSemicontinuousAt_ciSup {f : ι → α → δ'}
    (bdd : ∀ᶠ y in 𝓝 x, BddAbove (range fun i => f i y)) (h : ∀ i, LowerSemicontinuousAt (f i) x) :
    LowerSemicontinuousAt (fun x' => ⨆ i, f i x') x := by
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    x : α
    ι : Sort u_3
    δ' : Type u_5
    inst✝ : ConditionallyCompleteLinearOrder δ'
    f : ι → α → δ'
    bdd : Filter.Eventually (fun y => BddAbove (Set.range fun i => f i y)) (nhds x)
    h : ∀ (i : ι), LowerSemicontinuousAt (f i) x
    ⊢ LowerSemicontinuousAt (fun x' => iSup fun i => f i x') x
  -/
  simp_rw [← lowerSemicontinuousWithinAt_univ_iff] at *
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    x : α
    ι : Sort u_3
    δ' : Type u_5
    inst✝ : ConditionallyCompleteLinearOrder δ'
    f : ι → α → δ'
    bdd : Filter.Eventually (fun y => BddAbove (Set.range fun i => f i y)) (nhds x)
    h : ∀ (i : ι), LowerSemicontinuousWithinAt (f i) Set.univ x
    ⊢ LowerSemicontinuousWithinAt (fun x' => iSup fun i => f i x') Set.univ x
  -/
  rw [← nhdsWithin_univ] at bdd
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    x : α
    ι : Sort u_3
    δ' : Type u_5
    inst✝ : ConditionallyCompleteLinearOrder δ'
    f : ι → α → δ'
    bdd : Filter.Eventually (fun y => BddAbove (Set.range fun i => f i y)) (nhdsWi …
    h : ∀ (i : ι), LowerSemicontinuousWithinAt (f i) Set.univ x
    ⊢ LowerSemicontinuousWithinAt (fun x' => iSup fun i => f i x') Set.univ x
  -/
  exact lowerSemicontinuousWithinAt_ciSup bdd h
  /-
    🎉 no goals
  -/


theorem lowerSemicontinuousAt_iSup {f : ι → α → δ} (h : ∀ i, LowerSemicontinuousAt (f i) x) :
    LowerSemicontinuousAt (fun x' => ⨆ i, f i x') x :=
                                  /-
                                    α : Type u_1
                                    inst✝¹ : TopologicalSpace α
                                    x : α
                                    ι : Sort u_3
                                    δ : Type u_4
                                    inst✝ : CompleteLinearOrder δ
                                    f : ι → α → δ
                                    h : ∀ (i : ι), LowerSemicontinuousAt (f i) x
                                    ⊢ Filter.Eventually (fun y => BddAbove (Set.range fun i => f i y)) (nhds x)
                                  -/
  lowerSemicontinuousAt_ciSup (by simp) h
                                  /-
                                    🎉 no goals
                                  -/


theorem lowerSemicontinuousAt_biSup {p : ι → Prop} {f : ∀ i, p i → α → δ}
    (h : ∀ i hi, LowerSemicontinuousAt (f i hi) x) :
    LowerSemicontinuousAt (fun x' => ⨆ (i) (hi), f i hi x') x :=
  lowerSemicontinuousAt_iSup fun i => lowerSemicontinuousAt_iSup fun hi => h i hi


theorem lowerSemicontinuousOn_ciSup {f : ι → α → δ'}
    (bdd : ∀ x ∈ s, BddAbove (range fun i => f i x)) (h : ∀ i, LowerSemicontinuousOn (f i) s) :
    LowerSemicontinuousOn (fun x' => ⨆ i, f i x') s := fun x hx =>
  lowerSemicontinuousWithinAt_ciSup (eventually_nhdsWithin_of_forall bdd) fun i => h i x hx


theorem lowerSemicontinuousOn_iSup {f : ι → α → δ} (h : ∀ i, LowerSemicontinuousOn (f i) s) :
    LowerSemicontinuousOn (fun x' => ⨆ i, f i x') s :=
                                  /-
                                    α : Type u_1
                                    inst✝¹ : TopologicalSpace α
                                    s : Set α
                                    ι : Sort u_3
                                    δ : Type u_4
                                    inst✝ : CompleteLinearOrder δ
                                    f : ι → α → δ
                                    h : ∀ (i : ι), LowerSemicontinuousOn (f i) s
                                    ⊢ ∀ (x : α), Membership.mem s x → BddAbove (Set.range fun i => f i x)
                                  -/
  lowerSemicontinuousOn_ciSup (by simp) h
                                  /-
                                    🎉 no goals
                                  -/


theorem lowerSemicontinuousOn_biSup {p : ι → Prop} {f : ∀ i, p i → α → δ}
    (h : ∀ i hi, LowerSemicontinuousOn (f i hi) s) :
    LowerSemicontinuousOn (fun x' => ⨆ (i) (hi), f i hi x') s :=
  lowerSemicontinuousOn_iSup fun i => lowerSemicontinuousOn_iSup fun hi => h i hi


theorem lowerSemicontinuous_ciSup {f : ι → α → δ'} (bdd : ∀ x, BddAbove (range fun i => f i x))
    (h : ∀ i, LowerSemicontinuous (f i)) : LowerSemicontinuous fun x' => ⨆ i, f i x' := fun x =>
  lowerSemicontinuousAt_ciSup (Eventually.of_forall bdd) fun i => h i x


theorem lowerSemicontinuous_iSup {f : ι → α → δ} (h : ∀ i, LowerSemicontinuous (f i)) :
    LowerSemicontinuous fun x' => ⨆ i, f i x' :=
                                /-
                                  α : Type u_1
                                  inst✝¹ : TopologicalSpace α
                                  ι : Sort u_3
                                  δ : Type u_4
                                  inst✝ : CompleteLinearOrder δ
                                  f : ι → α → δ
                                  h : ∀ (i : ι), LowerSemicontinuous (f i)
                                  ⊢ ∀ (x : α), BddAbove (Set.range fun i => f i x)
                                -/
  lowerSemicontinuous_ciSup (by simp) h
                                /-
                                  🎉 no goals
                                -/


theorem lowerSemicontinuous_biSup {p : ι → Prop} {f : ∀ i, p i → α → δ}
    (h : ∀ i hi, LowerSemicontinuous (f i hi)) :
    LowerSemicontinuous fun x' => ⨆ (i) (hi), f i hi x' :=
  lowerSemicontinuous_iSup fun i => lowerSemicontinuous_iSup fun hi => h i hi


theorem lowerSemicontinuousWithinAt_tsum {f : ι → α → ℝ≥0∞}
    (h : ∀ i, LowerSemicontinuousWithinAt (f i) s x) :
    LowerSemicontinuousWithinAt (fun x' => ∑' i, f i x') s x := by
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    x : α
    s : Set α
    ι : Type u_3
    f : ι → α → ENNReal
    h : ∀ (i : ι), LowerSemicontinuousWithinAt (f i) s x
    ⊢ LowerSemicontinuousWithinAt (fun x' => tsum fun i => f i x') s x
  -/
  simp_rw [ENNReal.tsum_eq_iSup_sum]
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    x : α
    s : Set α
    ι : Type u_3
    f : ι → α → ENNReal
    h : ∀ (i : ι), LowerSemicontinuousWithinAt (f i) s x
    ⊢ LowerSemicontinuousWithinAt (fun x' => iSup fun s => s.sum fun i => f i x')  …
  -/
  refine lowerSemicontinuousWithinAt_iSup fun b => ?_
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    x : α
    s : Set α
    ι : Type u_3
    f : ι → α → ENNReal
    h : ∀ (i : ι), LowerSemicontinuousWithinAt (f i) s x
    b : Finset ι
    ⊢ LowerSemicontinuousWithinAt (fun x' => b.sum fun i => f i x') s x
  -/
  exact lowerSemicontinuousWithinAt_sum fun i _hi => h i
  /-
    🎉 no goals
  -/


theorem lowerSemicontinuousAt_tsum {f : ι → α → ℝ≥0∞} (h : ∀ i, LowerSemicontinuousAt (f i) x) :
    LowerSemicontinuousAt (fun x' => ∑' i, f i x') x := by
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    x : α
    ι : Type u_3
    f : ι → α → ENNReal
    h : ∀ (i : ι), LowerSemicontinuousAt (f i) x
    ⊢ LowerSemicontinuousAt (fun x' => tsum fun i => f i x') x
  -/
  simp_rw [← lowerSemicontinuousWithinAt_univ_iff] at *
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    x : α
    ι : Type u_3
    f : ι → α → ENNReal
    h : ∀ (i : ι), LowerSemicontinuousWithinAt (f i) Set.univ x
    ⊢ LowerSemicontinuousWithinAt (fun x' => tsum fun i => f i x') Set.univ x
  -/
  exact lowerSemicontinuousWithinAt_tsum h
  /-
    🎉 no goals
  -/


theorem lowerSemicontinuousOn_tsum {f : ι → α → ℝ≥0∞} (h : ∀ i, LowerSemicontinuousOn (f i) s) :
    LowerSemicontinuousOn (fun x' => ∑' i, f i x') s := fun x hx =>
  lowerSemicontinuousWithinAt_tsum fun i => h i x hx


theorem lowerSemicontinuous_tsum {f : ι → α → ℝ≥0∞} (h : ∀ i, LowerSemicontinuous (f i)) :
    LowerSemicontinuous fun x' => ∑' i, f i x' := fun x => lowerSemicontinuousAt_tsum fun i => h i x


theorem UpperSemicontinuousWithinAt.mono (h : UpperSemicontinuousWithinAt f s x) (hst : t ⊆ s) :
    UpperSemicontinuousWithinAt f t x := fun y hy =>
  Filter.Eventually.filter_mono (nhdsWithin_mono _ hst) (h y hy)


theorem upperSemicontinuousWithinAt_univ_iff :
    UpperSemicontinuousWithinAt f univ x ↔ UpperSemicontinuousAt f x := by
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    β : Type u_2
    inst✝ : Preorder β
    f : α → β
    x : α
    ⊢ Iff (UpperSemicontinuousWithinAt f Set.univ x) (UpperSemicontinuousAt f x)
  -/
  simp [UpperSemicontinuousWithinAt, UpperSemicontinuousAt, nhdsWithin_univ]
  /-
    🎉 no goals
  -/


theorem UpperSemicontinuousAt.upperSemicontinuousWithinAt (s : Set α)
    (h : UpperSemicontinuousAt f x) : UpperSemicontinuousWithinAt f s x := fun y hy =>
  Filter.Eventually.filter_mono nhdsWithin_le_nhds (h y hy)


theorem UpperSemicontinuousOn.upperSemicontinuousWithinAt (h : UpperSemicontinuousOn f s)
    (hx : x ∈ s) : UpperSemicontinuousWithinAt f s x :=
  h x hx


theorem UpperSemicontinuousOn.mono (h : UpperSemicontinuousOn f s) (hst : t ⊆ s) :
    UpperSemicontinuousOn f t := fun x hx => (h x (hst hx)).mono hst


theorem upperSemicontinuousOn_univ_iff : UpperSemicontinuousOn f univ ↔ UpperSemicontinuous f := by
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    β : Type u_2
    inst✝ : Preorder β
    f : α → β
    ⊢ Iff (UpperSemicontinuousOn f Set.univ) (UpperSemicontinuous f)
  -/
  simp [UpperSemicontinuousOn, UpperSemicontinuous, upperSemicontinuousWithinAt_univ_iff]
  /-
    🎉 no goals
  -/


theorem UpperSemicontinuous.upperSemicontinuousAt (h : UpperSemicontinuous f) (x : α) :
    UpperSemicontinuousAt f x :=
  h x


theorem UpperSemicontinuous.upperSemicontinuousWithinAt (h : UpperSemicontinuous f) (s : Set α)
    (x : α) : UpperSemicontinuousWithinAt f s x :=
  (h x).upperSemicontinuousWithinAt s


theorem UpperSemicontinuous.upperSemicontinuousOn (h : UpperSemicontinuous f) (s : Set α) :
    UpperSemicontinuousOn f s := fun x _hx => h.upperSemicontinuousWithinAt s x


theorem upperSemicontinuousWithinAt_const : UpperSemicontinuousWithinAt (fun _x => z) s x :=
  fun _y hy => Filter.Eventually.of_forall fun _x => hy


theorem upperSemicontinuousAt_const : UpperSemicontinuousAt (fun _x => z) x := fun _y hy =>
  Filter.Eventually.of_forall fun _x => hy


theorem upperSemicontinuousOn_const : UpperSemicontinuousOn (fun _x => z) s := fun _x _hx =>
  upperSemicontinuousWithinAt_const


theorem upperSemicontinuous_const : UpperSemicontinuous fun _x : α => z := fun _x =>
  upperSemicontinuousAt_const


theorem IsOpen.upperSemicontinuous_indicator (hs : IsOpen s) (hy : y ≤ 0) :
    UpperSemicontinuous (indicator s fun _x => y) :=
  @IsOpen.lowerSemicontinuous_indicator α _ βᵒᵈ _ s y _ hs hy


theorem IsOpen.upperSemicontinuousOn_indicator (hs : IsOpen s) (hy : y ≤ 0) :
    UpperSemicontinuousOn (indicator s fun _x => y) t :=
  (hs.upperSemicontinuous_indicator hy).upperSemicontinuousOn t


theorem IsOpen.upperSemicontinuousAt_indicator (hs : IsOpen s) (hy : y ≤ 0) :
    UpperSemicontinuousAt (indicator s fun _x => y) x :=
  (hs.upperSemicontinuous_indicator hy).upperSemicontinuousAt x


theorem IsOpen.upperSemicontinuousWithinAt_indicator (hs : IsOpen s) (hy : y ≤ 0) :
    UpperSemicontinuousWithinAt (indicator s fun _x => y) t x :=
  (hs.upperSemicontinuous_indicator hy).upperSemicontinuousWithinAt t x


theorem IsClosed.upperSemicontinuous_indicator (hs : IsClosed s) (hy : 0 ≤ y) :
    UpperSemicontinuous (indicator s fun _x => y) :=
  @IsClosed.lowerSemicontinuous_indicator α _ βᵒᵈ _ s y _ hs hy


theorem IsClosed.upperSemicontinuousOn_indicator (hs : IsClosed s) (hy : 0 ≤ y) :
    UpperSemicontinuousOn (indicator s fun _x => y) t :=
  (hs.upperSemicontinuous_indicator hy).upperSemicontinuousOn t


theorem IsClosed.upperSemicontinuousAt_indicator (hs : IsClosed s) (hy : 0 ≤ y) :
    UpperSemicontinuousAt (indicator s fun _x => y) x :=
  (hs.upperSemicontinuous_indicator hy).upperSemicontinuousAt x


theorem IsClosed.upperSemicontinuousWithinAt_indicator (hs : IsClosed s) (hy : 0 ≤ y) :
    UpperSemicontinuousWithinAt (indicator s fun _x => y) t x :=
  (hs.upperSemicontinuous_indicator hy).upperSemicontinuousWithinAt t x


theorem upperSemicontinuous_iff_isOpen_preimage :
    UpperSemicontinuous f ↔ ∀ y, IsOpen (f ⁻¹' Iio y) :=
  ⟨fun H y => isOpen_iff_mem_nhds.2 fun x hx => H x y hx, fun H _x y y_lt =>
    IsOpen.mem_nhds (H y) y_lt⟩


theorem UpperSemicontinuous.isOpen_preimage (hf : UpperSemicontinuous f) (y : β) :
    IsOpen (f ⁻¹' Iio y) :=
  upperSemicontinuous_iff_isOpen_preimage.1 hf y


theorem upperSemicontinuous_iff_isClosed_preimage {f : α → γ} :
    UpperSemicontinuous f ↔ ∀ y, IsClosed (f ⁻¹' Ici y) := by
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    γ : Type u_3
    inst✝ : LinearOrder γ
    f : α → γ
    ⊢ Iff (UpperSemicontinuous f) (∀ (y : γ), IsClosed (Set.preimage f (Set.Ici y)))
  -/
  rw [upperSemicontinuous_iff_isOpen_preimage]
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    γ : Type u_3
    inst✝ : LinearOrder γ
    f : α → γ
    ⊢ Iff (∀ (y : γ), IsOpen (Set.preimage f (Set.Iio y))) (∀ (y : γ), IsClosed (S …
  -/
  simp only [← isOpen_compl_iff, ← preimage_compl, compl_Ici]
  /-
    🎉 no goals
  -/


theorem UpperSemicontinuous.isClosed_preimage {f : α → γ} (hf : UpperSemicontinuous f) (y : γ) :
    IsClosed (f ⁻¹' Ici y) :=
  upperSemicontinuous_iff_isClosed_preimage.1 hf y


theorem ContinuousWithinAt.upperSemicontinuousWithinAt {f : α → γ} (h : ContinuousWithinAt f s x) :
    UpperSemicontinuousWithinAt f s x := fun _y hy => h (Iio_mem_nhds hy)


theorem ContinuousAt.upperSemicontinuousAt {f : α → γ} (h : ContinuousAt f x) :
    UpperSemicontinuousAt f x := fun _y hy => h (Iio_mem_nhds hy)


theorem ContinuousOn.upperSemicontinuousOn {f : α → γ} (h : ContinuousOn f s) :
    UpperSemicontinuousOn f s := fun x hx => (h x hx).upperSemicontinuousWithinAt


theorem Continuous.upperSemicontinuous {f : α → γ} (h : Continuous f) : UpperSemicontinuous f :=
  fun _x => h.continuousAt.upperSemicontinuousAt


theorem upperSemicontinuousWithinAt_iff_limsup_le {f : α → γ} :
    UpperSemicontinuousWithinAt f s x ↔ limsup f (𝓝[s] x) ≤ f x :=
  lowerSemicontinuousWithinAt_iff_le_liminf (γ := γᵒᵈ)


alias ⟨UpperSemicontinuousWithinAt.limsup_le, _⟩ := upperSemicontinuousWithinAt_iff_limsup_le


theorem upperSemicontinuousAt_iff_limsup_le {f : α → γ} :
    UpperSemicontinuousAt f x ↔ limsup f (𝓝 x) ≤ f x :=
  lowerSemicontinuousAt_iff_le_liminf (γ := γᵒᵈ)


alias ⟨UpperSemicontinuousAt.limsup_le, _⟩ := upperSemicontinuousAt_iff_limsup_le


theorem upperSemicontinuous_iff_limsup_le {f : α → γ} :
    UpperSemicontinuous f ↔ ∀ x, limsup f (𝓝 x) ≤ f x :=
  lowerSemicontinuous_iff_le_liminf (γ := γᵒᵈ)


alias ⟨UpperSemicontinuous.limsup_le, _⟩ := upperSemicontinuous_iff_limsup_le


theorem upperSemicontinuousOn_iff_limsup_le {f : α → γ} :
    UpperSemicontinuousOn f s ↔ ∀ x ∈ s, limsup f (𝓝[s] x) ≤ f x :=
  lowerSemicontinuousOn_iff_le_liminf (γ := γᵒᵈ)


alias ⟨UpperSemicontinuousOn.limsup_le, _⟩ := upperSemicontinuousOn_iff_limsup_le


theorem upperSemicontinuous_iff_IsClosed_hypograph {f : α → γ} :
    UpperSemicontinuous f ↔ IsClosed {p : α × γ | p.2 ≤ f p.1} :=
  lowerSemicontinuous_iff_isClosed_epigraph (γ := γᵒᵈ)


alias ⟨UpperSemicontinuous.IsClosed_hypograph, _⟩ := upperSemicontinuous_iff_IsClosed_hypograph


theorem ContinuousAt.comp_upperSemicontinuousWithinAt {g : γ → δ} {f : α → γ}
    (hg : ContinuousAt g (f x)) (hf : UpperSemicontinuousWithinAt f s x) (gmon : Monotone g) :
    UpperSemicontinuousWithinAt (g ∘ f) s x :=
  @ContinuousAt.comp_lowerSemicontinuousWithinAt α _ x s γᵒᵈ _ _ _ δᵒᵈ _ _ _ g f hg hf gmon.dual


theorem ContinuousAt.comp_upperSemicontinuousAt {g : γ → δ} {f : α → γ} (hg : ContinuousAt g (f x))
    (hf : UpperSemicontinuousAt f x) (gmon : Monotone g) : UpperSemicontinuousAt (g ∘ f) x :=
  @ContinuousAt.comp_lowerSemicontinuousAt α _ x γᵒᵈ _ _ _ δᵒᵈ _ _ _ g f hg hf gmon.dual


theorem Continuous.comp_upperSemicontinuousOn {g : γ → δ} {f : α → γ} (hg : Continuous g)
    (hf : UpperSemicontinuousOn f s) (gmon : Monotone g) : UpperSemicontinuousOn (g ∘ f) s :=
  fun x hx => hg.continuousAt.comp_upperSemicontinuousWithinAt (hf x hx) gmon


theorem Continuous.comp_upperSemicontinuous {g : γ → δ} {f : α → γ} (hg : Continuous g)
    (hf : UpperSemicontinuous f) (gmon : Monotone g) : UpperSemicontinuous (g ∘ f) := fun x =>
  hg.continuousAt.comp_upperSemicontinuousAt (hf x) gmon


theorem ContinuousAt.comp_upperSemicontinuousWithinAt_antitone {g : γ → δ} {f : α → γ}
    (hg : ContinuousAt g (f x)) (hf : UpperSemicontinuousWithinAt f s x) (gmon : Antitone g) :
    LowerSemicontinuousWithinAt (g ∘ f) s x :=
  @ContinuousAt.comp_upperSemicontinuousWithinAt α _ x s γ _ _ _ δᵒᵈ _ _ _ g f hg hf gmon


theorem ContinuousAt.comp_upperSemicontinuousAt_antitone {g : γ → δ} {f : α → γ}
    (hg : ContinuousAt g (f x)) (hf : UpperSemicontinuousAt f x) (gmon : Antitone g) :
    LowerSemicontinuousAt (g ∘ f) x :=
  @ContinuousAt.comp_upperSemicontinuousAt α _ x γ _ _ _ δᵒᵈ _ _ _ g f hg hf gmon


theorem Continuous.comp_upperSemicontinuousOn_antitone {g : γ → δ} {f : α → γ} (hg : Continuous g)
    (hf : UpperSemicontinuousOn f s) (gmon : Antitone g) : LowerSemicontinuousOn (g ∘ f) s :=
  fun x hx => hg.continuousAt.comp_upperSemicontinuousWithinAt_antitone (hf x hx) gmon


theorem Continuous.comp_upperSemicontinuous_antitone {g : γ → δ} {f : α → γ} (hg : Continuous g)
    (hf : UpperSemicontinuous f) (gmon : Antitone g) : LowerSemicontinuous (g ∘ f) := fun x =>
  hg.continuousAt.comp_upperSemicontinuousAt_antitone (hf x) gmon


theorem UpperSemicontinuousAt.comp_continuousAt {f : α → β} {g : ι → α} {x : ι}
    (hf : UpperSemicontinuousAt f (g x)) (hg : ContinuousAt g x) :
    UpperSemicontinuousAt (fun x ↦ f (g x)) x :=
  fun _ lt ↦ hg.eventually (hf _ lt)


theorem UpperSemicontinuousAt.comp_continuousAt_of_eq {f : α → β} {g : ι → α} {y : α} {x : ι}
    (hf : UpperSemicontinuousAt f y) (hg : ContinuousAt g x) (hy : g x = y) :
    UpperSemicontinuousAt (fun x ↦ f (g x)) x := by
  /-
    α : Type u_1
    inst✝² : TopologicalSpace α
    β : Type u_2
    inst✝¹ : Preorder β
    ι : Type u_5
    inst✝ : TopologicalSpace ι
    f : α → β
    g : ι → α
    y : α
    x : ι
    hf : UpperSemicontinuousAt f y
    hg : ContinuousAt g x
    hy : Eq (g x) y
    ⊢ UpperSemicontinuousAt (fun x => f (g x)) x
  -/
  rw [← hy] at hf
  /-
    α : Type u_1
    inst✝² : TopologicalSpace α
    β : Type u_2
    inst✝¹ : Preorder β
    ι : Type u_5
    inst✝ : TopologicalSpace ι
    f : α → β
    g : ι → α
    y : α
    x : ι
    hf : UpperSemicontinuousAt f (g x)
    hg : ContinuousAt g x
    hy : Eq (g x) y
    ⊢ UpperSemicontinuousAt (fun x => f (g x)) x
  -/
  exact comp_continuousAt hf hg
  /-
    🎉 no goals
  -/


theorem UpperSemicontinuous.comp_continuous {f : α → β} {g : ι → α}
    (hf : UpperSemicontinuous f) (hg : Continuous g) : UpperSemicontinuous fun x ↦ f (g x) :=
  fun x ↦ (hf (g x)).comp_continuousAt hg.continuousAt


/-- The sum of two upper semicontinuous functions is upper semicontinuous. Formulated with an
explicit continuity assumption on addition, for application to `EReal`. The unprimed version of
the lemma uses `[ContinuousAdd]`. -/
theorem UpperSemicontinuousWithinAt.add' {f g : α → γ} (hf : UpperSemicontinuousWithinAt f s x)
    (hg : UpperSemicontinuousWithinAt g s x)
    (hcont : ContinuousAt (fun p : γ × γ => p.1 + p.2) (f x, g x)) :
    UpperSemicontinuousWithinAt (fun z => f z + g z) s x :=
  @LowerSemicontinuousWithinAt.add' α _ x s γᵒᵈ _ _ _ _ _ hf hg hcont


/-- The sum of two upper semicontinuous functions is upper semicontinuous. Formulated with an
explicit continuity assumption on addition, for application to `EReal`. The unprimed version of
the lemma uses `[ContinuousAdd]`. -/
theorem UpperSemicontinuousAt.add' {f g : α → γ} (hf : UpperSemicontinuousAt f x)
    (hg : UpperSemicontinuousAt g x)
    (hcont : ContinuousAt (fun p : γ × γ => p.1 + p.2) (f x, g x)) :
    UpperSemicontinuousAt (fun z => f z + g z) x := by
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    x : α
    γ : Type u_4
    inst✝² : LinearOrderedAddCommMonoid γ
    inst✝¹ : TopologicalSpace γ
    inst✝ : OrderTopology γ
    f g : α → γ
    hf : UpperSemicontinuousAt f x
    hg : UpperSemicontinuousAt g x
    hcont : ContinuousAt (fun p => HAdd.hAdd p.1 p.2) { fst := f x, snd := g x }
    ⊢ UpperSemicontinuousAt (fun z => HAdd.hAdd (f z) (g z)) x
  -/
  simp_rw [← upperSemicontinuousWithinAt_univ_iff] at *
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    x : α
    γ : Type u_4
    inst✝² : LinearOrderedAddCommMonoid γ
    inst✝¹ : TopologicalSpace γ
    inst✝ : OrderTopology γ
    f g : α → γ
    hcont : ContinuousAt (fun p => HAdd.hAdd p.1 p.2) { fst := f x, snd := g x }
    hf : UpperSemicontinuousWithinAt f Set.univ x
    hg : UpperSemicontinuousWithinAt g Set.univ x
    ⊢ UpperSemicontinuousWithinAt (fun z => HAdd.hAdd (f z) (g z)) Set.univ x
  -/
  exact hf.add' hg hcont
  /-
    🎉 no goals
  -/


/-- The sum of two upper semicontinuous functions is upper semicontinuous. Formulated with an
explicit continuity assumption on addition, for application to `EReal`. The unprimed version of
the lemma uses `[ContinuousAdd]`. -/
theorem UpperSemicontinuousOn.add' {f g : α → γ} (hf : UpperSemicontinuousOn f s)
    (hg : UpperSemicontinuousOn g s)
    (hcont : ∀ x ∈ s, ContinuousAt (fun p : γ × γ => p.1 + p.2) (f x, g x)) :
    UpperSemicontinuousOn (fun z => f z + g z) s := fun x hx =>
  (hf x hx).add' (hg x hx) (hcont x hx)


/-- The sum of two upper semicontinuous functions is upper semicontinuous. Formulated with an
explicit continuity assumption on addition, for application to `EReal`. The unprimed version of
the lemma uses `[ContinuousAdd]`. -/
theorem UpperSemicontinuous.add' {f g : α → γ} (hf : UpperSemicontinuous f)
    (hg : UpperSemicontinuous g)
    (hcont : ∀ x, ContinuousAt (fun p : γ × γ => p.1 + p.2) (f x, g x)) :
    UpperSemicontinuous fun z => f z + g z := fun x => (hf x).add' (hg x) (hcont x)


/-- The sum of two upper semicontinuous functions is upper semicontinuous. Formulated with
`[ContinuousAdd]`. The primed version of the lemma uses an explicit continuity assumption on
addition, for application to `EReal`. -/
theorem UpperSemicontinuousWithinAt.add {f g : α → γ} (hf : UpperSemicontinuousWithinAt f s x)
    (hg : UpperSemicontinuousWithinAt g s x) :
    UpperSemicontinuousWithinAt (fun z => f z + g z) s x :=
  hf.add' hg continuous_add.continuousAt


/-- The sum of two upper semicontinuous functions is upper semicontinuous. Formulated with
`[ContinuousAdd]`. The primed version of the lemma uses an explicit continuity assumption on
addition, for application to `EReal`. -/
theorem UpperSemicontinuousAt.add {f g : α → γ} (hf : UpperSemicontinuousAt f x)
    (hg : UpperSemicontinuousAt g x) : UpperSemicontinuousAt (fun z => f z + g z) x :=
  hf.add' hg continuous_add.continuousAt


/-- The sum of two upper semicontinuous functions is upper semicontinuous. Formulated with
`[ContinuousAdd]`. The primed version of the lemma uses an explicit continuity assumption on
addition, for application to `EReal`. -/
theorem UpperSemicontinuousOn.add {f g : α → γ} (hf : UpperSemicontinuousOn f s)
    (hg : UpperSemicontinuousOn g s) : UpperSemicontinuousOn (fun z => f z + g z) s :=
  hf.add' hg fun _x _hx => continuous_add.continuousAt


/-- The sum of two upper semicontinuous functions is upper semicontinuous. Formulated with
`[ContinuousAdd]`. The primed version of the lemma uses an explicit continuity assumption on
addition, for application to `EReal`. -/
theorem UpperSemicontinuous.add {f g : α → γ} (hf : UpperSemicontinuous f)
    (hg : UpperSemicontinuous g) : UpperSemicontinuous fun z => f z + g z :=
  hf.add' hg fun _x => continuous_add.continuousAt


theorem upperSemicontinuousWithinAt_sum {f : ι → α → γ} {a : Finset ι}
    (ha : ∀ i ∈ a, UpperSemicontinuousWithinAt (f i) s x) :
    UpperSemicontinuousWithinAt (fun z => ∑ i ∈ a, f i z) s x :=
  @lowerSemicontinuousWithinAt_sum α _ x s ι γᵒᵈ _ _ _ _ f a ha


theorem upperSemicontinuousAt_sum {f : ι → α → γ} {a : Finset ι}
    (ha : ∀ i ∈ a, UpperSemicontinuousAt (f i) x) :
    UpperSemicontinuousAt (fun z => ∑ i ∈ a, f i z) x := by
  /-
    α : Type u_1
    inst✝⁴ : TopologicalSpace α
    x : α
    ι : Type u_3
    γ : Type u_4
    inst✝³ : LinearOrderedAddCommMonoid γ
    inst✝² : TopologicalSpace γ
    inst✝¹ : OrderTopology γ
    inst✝ : ContinuousAdd γ
    f : ι → α → γ
    a : Finset ι
    ha : ∀ (i : ι), Membership.mem a i → UpperSemicontinuousAt (f i) x
    ⊢ UpperSemicontinuousAt (fun z => a.sum fun i => f i z) x
  -/
  simp_rw [← upperSemicontinuousWithinAt_univ_iff] at *
  /-
    α : Type u_1
    inst✝⁴ : TopologicalSpace α
    x : α
    ι : Type u_3
    γ : Type u_4
    inst✝³ : LinearOrderedAddCommMonoid γ
    inst✝² : TopologicalSpace γ
    inst✝¹ : OrderTopology γ
    inst✝ : ContinuousAdd γ
    f : ι → α → γ
    a : Finset ι
    ha : ∀ (i : ι), Membership.mem a i → UpperSemicontinuousWithinAt (f i) Set.uni …
    ⊢ UpperSemicontinuousWithinAt (fun z => a.sum fun i => f i z) Set.univ x
  -/
  exact upperSemicontinuousWithinAt_sum ha
  /-
    🎉 no goals
  -/


theorem upperSemicontinuousOn_sum {f : ι → α → γ} {a : Finset ι}
    (ha : ∀ i ∈ a, UpperSemicontinuousOn (f i) s) :
    UpperSemicontinuousOn (fun z => ∑ i ∈ a, f i z) s := fun x hx =>
  upperSemicontinuousWithinAt_sum fun i hi => ha i hi x hx


theorem upperSemicontinuous_sum {f : ι → α → γ} {a : Finset ι}
    (ha : ∀ i ∈ a, UpperSemicontinuous (f i)) : UpperSemicontinuous fun z => ∑ i ∈ a, f i z :=
  fun x => upperSemicontinuousAt_sum fun i hi => ha i hi x


theorem upperSemicontinuousWithinAt_ciInf {f : ι → α → δ'}
    (bdd : ∀ᶠ y in 𝓝[s] x, BddBelow (range fun i => f i y))
    (h : ∀ i, UpperSemicontinuousWithinAt (f i) s x) :
    UpperSemicontinuousWithinAt (fun x' => ⨅ i, f i x') s x :=
  @lowerSemicontinuousWithinAt_ciSup α _ x s ι δ'ᵒᵈ _ f bdd h


theorem upperSemicontinuousWithinAt_iInf {f : ι → α → δ}
    (h : ∀ i, UpperSemicontinuousWithinAt (f i) s x) :
    UpperSemicontinuousWithinAt (fun x' => ⨅ i, f i x') s x :=
  @lowerSemicontinuousWithinAt_iSup α _ x s ι δᵒᵈ _ f h


theorem upperSemicontinuousWithinAt_biInf {p : ι → Prop} {f : ∀ i, p i → α → δ}
    (h : ∀ i hi, UpperSemicontinuousWithinAt (f i hi) s x) :
    UpperSemicontinuousWithinAt (fun x' => ⨅ (i) (hi), f i hi x') s x :=
  upperSemicontinuousWithinAt_iInf fun i => upperSemicontinuousWithinAt_iInf fun hi => h i hi


theorem upperSemicontinuousAt_ciInf {f : ι → α → δ'}
    (bdd : ∀ᶠ y in 𝓝 x, BddBelow (range fun i => f i y)) (h : ∀ i, UpperSemicontinuousAt (f i) x) :
    UpperSemicontinuousAt (fun x' => ⨅ i, f i x') x :=
  @lowerSemicontinuousAt_ciSup α _ x ι δ'ᵒᵈ _ f bdd h


theorem upperSemicontinuousAt_iInf {f : ι → α → δ} (h : ∀ i, UpperSemicontinuousAt (f i) x) :
    UpperSemicontinuousAt (fun x' => ⨅ i, f i x') x :=
  @lowerSemicontinuousAt_iSup α _ x ι δᵒᵈ _ f h


theorem upperSemicontinuousAt_biInf {p : ι → Prop} {f : ∀ i, p i → α → δ}
    (h : ∀ i hi, UpperSemicontinuousAt (f i hi) x) :
    UpperSemicontinuousAt (fun x' => ⨅ (i) (hi), f i hi x') x :=
  upperSemicontinuousAt_iInf fun i => upperSemicontinuousAt_iInf fun hi => h i hi


theorem upperSemicontinuousOn_ciInf {f : ι → α → δ'}
    (bdd : ∀ x ∈ s, BddBelow (range fun i => f i x)) (h : ∀ i, UpperSemicontinuousOn (f i) s) :
    UpperSemicontinuousOn (fun x' => ⨅ i, f i x') s := fun x hx =>
  upperSemicontinuousWithinAt_ciInf (eventually_nhdsWithin_of_forall bdd) fun i => h i x hx


theorem upperSemicontinuousOn_iInf {f : ι → α → δ} (h : ∀ i, UpperSemicontinuousOn (f i) s) :
    UpperSemicontinuousOn (fun x' => ⨅ i, f i x') s := fun x hx =>
  upperSemicontinuousWithinAt_iInf fun i => h i x hx


theorem upperSemicontinuousOn_biInf {p : ι → Prop} {f : ∀ i, p i → α → δ}
    (h : ∀ i hi, UpperSemicontinuousOn (f i hi) s) :
    UpperSemicontinuousOn (fun x' => ⨅ (i) (hi), f i hi x') s :=
  upperSemicontinuousOn_iInf fun i => upperSemicontinuousOn_iInf fun hi => h i hi


theorem upperSemicontinuous_ciInf {f : ι → α → δ'} (bdd : ∀ x, BddBelow (range fun i => f i x))
    (h : ∀ i, UpperSemicontinuous (f i)) : UpperSemicontinuous fun x' => ⨅ i, f i x' := fun x =>
  upperSemicontinuousAt_ciInf (Eventually.of_forall bdd) fun i => h i x


theorem upperSemicontinuous_iInf {f : ι → α → δ} (h : ∀ i, UpperSemicontinuous (f i)) :
    UpperSemicontinuous fun x' => ⨅ i, f i x' := fun x => upperSemicontinuousAt_iInf fun i => h i x


theorem upperSemicontinuous_biInf {p : ι → Prop} {f : ∀ i, p i → α → δ}
    (h : ∀ i hi, UpperSemicontinuous (f i hi)) :
    UpperSemicontinuous fun x' => ⨅ (i) (hi), f i hi x' :=
  upperSemicontinuous_iInf fun i => upperSemicontinuous_iInf fun hi => h i hi


theorem continuousWithinAt_iff_lower_upperSemicontinuousWithinAt {f : α → γ} :
    ContinuousWithinAt f s x ↔
      LowerSemicontinuousWithinAt f s x ∧ UpperSemicontinuousWithinAt f s x := by
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    x : α
    s : Set α
    γ : Type u_3
    inst✝² : LinearOrder γ
    inst✝¹ : TopologicalSpace γ
    inst✝ : OrderTopology γ
    f : α → γ
    ⊢ Iff (ContinuousWithinAt f s x) (And (LowerSemicontinuousWithinAt f s x) (Upp …
  -/
  refine ⟨fun h => ⟨h.lowerSemicontinuousWithinAt, h.upperSemicontinuousWithinAt⟩, ?_⟩
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    x : α
    s : Set α
    γ : Type u_3
    inst✝² : LinearOrder γ
    inst✝¹ : TopologicalSpace γ
    inst✝ : OrderTopology γ
    f : α → γ
    ⊢ And (LowerSemicontinuousWithinAt f s x) (UpperSemicontinuousWithinAt f s x)  …
  -/
  rintro ⟨h₁, h₂⟩
  /-
    case intro
    α : Type u_1
    inst✝³ : TopologicalSpace α
    x : α
    s : Set α
    γ : Type u_3
    inst✝² : LinearOrder γ
    inst✝¹ : TopologicalSpace γ
    inst✝ : OrderTopology γ
    f : α → γ
    h₁ : LowerSemicontinuousWithinAt f s x
    h₂ : UpperSemicontinuousWithinAt f s x
    ⊢ ContinuousWithinAt f s x
  -/
  intro v hv
  /-
    case intro
    α : Type u_1
    inst✝³ : TopologicalSpace α
    x : α
    s : Set α
    γ : Type u_3
    inst✝² : LinearOrder γ
    inst✝¹ : TopologicalSpace γ
    inst✝ : OrderTopology γ
    f : α → γ
    h₁ : LowerSemicontinuousWithinAt f s x
    h₂ : UpperSemicontinuousWithinAt f s x
    v : Set γ
    hv : Membership.mem (nhds (f x)) v
    ⊢ Membership.mem (Filter.map f (nhdsWithin x s)) v
  -/
  simp only [Filter.mem_map]
  /-
    case intro
    α : Type u_1
    inst✝³ : TopologicalSpace α
    x : α
    s : Set α
    γ : Type u_3
    inst✝² : LinearOrder γ
    inst✝¹ : TopologicalSpace γ
    inst✝ : OrderTopology γ
    f : α → γ
    h₁ : LowerSemicontinuousWithinAt f s x
    h₂ : UpperSemicontinuousWithinAt f s x
    v : Set γ
    hv : Membership.mem (nhds (f x)) v
    ⊢ Membership.mem (nhdsWithin x s) (Set.preimage f v)
  -/
  by_cases Hl : ∃ l, l < f x
    /-
      case pos
      α : Type u_1
      inst✝³ : TopologicalSpace α
      x : α
      s : Set α
      γ : Type u_3
      inst✝² : LinearOrder γ
      inst✝¹ : TopologicalSpace γ
      inst✝ : OrderTopology γ
      f : α → γ
      h₁ : LowerSemicontinuousWithinAt f s x
      h₂ : UpperSemicontinuousWithinAt f s x
      v : Set γ
      hv : Membership.mem (nhds (f x)) v
      Hl : Exists fun l => LT.lt l (f x)
      ⊢ Membership.mem (nhdsWithin x s) (Set.preimage f v)
    -/
  · rcases exists_Ioc_subset_of_mem_nhds hv Hl with ⟨l, lfx, hl⟩
    /-
      case pos.intro.intro
      α : Type u_1
      inst✝³ : TopologicalSpace α
      x : α
      s : Set α
      γ : Type u_3
      inst✝² : LinearOrder γ
      inst✝¹ : TopologicalSpace γ
      inst✝ : OrderTopology γ
      f : α → γ
      h₁ : LowerSemicontinuousWithinAt f s x
      h₂ : UpperSemicontinuousWithinAt f s x
      v : Set γ
      hv : Membership.mem (nhds (f x)) v
      Hl : Exists fun l => LT.lt l (f x)
      l : γ
      lfx : LT.lt l (f x)
      hl : HasSubset.Subset (Set.Ioc l (f x)) v
      ⊢ Membership.mem (nhdsWithin x s) (Set.preimage f v)
    -/
    by_cases Hu : ∃ u, f x < u
      /-
        case pos
        α : Type u_1
        inst✝³ : TopologicalSpace α
        x : α
        s : Set α
        γ : Type u_3
        inst✝² : LinearOrder γ
        inst✝¹ : TopologicalSpace γ
        inst✝ : OrderTopology γ
        f : α → γ
        h₁ : LowerSemicontinuousWithinAt f s x
        h₂ : UpperSemicontinuousWithinAt f s x
        v : Set γ
        hv : Membership.mem (nhds (f x)) v
        Hl : Exists fun l => LT.lt l (f x)
        l : γ
        lfx : LT.lt l (f x)
        hl : HasSubset.Subset (Set.Ioc l (f x)) v
        Hu : Exists fun u => LT.lt (f x) u
        ⊢ Membership.mem (nhdsWithin x s) (Set.preimage f v)
      -/
    · rcases exists_Ico_subset_of_mem_nhds hv Hu with ⟨u, fxu, hu⟩
      /-
        case pos.intro.intro
        α : Type u_1
        inst✝³ : TopologicalSpace α
        x : α
        s : Set α
        γ : Type u_3
        inst✝² : LinearOrder γ
        inst✝¹ : TopologicalSpace γ
        inst✝ : OrderTopology γ
        f : α → γ
        h₁ : LowerSemicontinuousWithinAt f s x
        h₂ : UpperSemicontinuousWithinAt f s x
        v : Set γ
        hv : Membership.mem (nhds (f x)) v
        Hl : Exists fun l => LT.lt l (f x)
        l : γ
        lfx : LT.lt l (f x)
        hl : HasSubset.Subset (Set.Ioc l (f x)) v
        Hu : Exists fun u => LT.lt (f x) u
        u : γ
        fxu : LT.lt (f x) u
        hu : HasSubset.Subset (Set.Ico (f x) u) v
        ⊢ Membership.mem (nhdsWithin x s) (Set.preimage f v)
      -/
      filter_upwards [h₁ l lfx, h₂ u fxu] with a lfa fau
      /-
        case h
        α : Type u_1
        inst✝³ : TopologicalSpace α
        x : α
        s : Set α
        γ : Type u_3
        inst✝² : LinearOrder γ
        inst✝¹ : TopologicalSpace γ
        inst✝ : OrderTopology γ
        f : α → γ
        h₁ : LowerSemicontinuousWithinAt f s x
        h₂ : UpperSemicontinuousWithinAt f s x
        v : Set γ
        hv : Membership.mem (nhds (f x)) v
        Hl : Exists fun l => LT.lt l (f x)
        l : γ
        lfx : LT.lt l (f x)
        hl : HasSubset.Subset (Set.Ioc l (f x)) v
        Hu : Exists fun u => LT.lt (f x) u
        u : γ
        fxu : LT.lt (f x) u
        hu : HasSubset.Subset (Set.Ico (f x) u) v
        a : α
        lfa : LT.lt l (f a)
        fau : LT.lt (f a) u
        ⊢ Membership.mem (Set.preimage f v) a
      -/
      cases' le_or_gt (f a) (f x) with h h
        /-
          case h.inl
          α : Type u_1
          inst✝³ : TopologicalSpace α
          x : α
          s : Set α
          γ : Type u_3
          inst✝² : LinearOrder γ
          inst✝¹ : TopologicalSpace γ
          inst✝ : OrderTopology γ
          f : α → γ
          h₁ : LowerSemicontinuousWithinAt f s x
          h₂ : UpperSemicontinuousWithinAt f s x
          v : Set γ
          hv : Membership.mem (nhds (f x)) v
          Hl : Exists fun l => LT.lt l (f x)
          l : γ
          lfx : LT.lt l (f x)
          hl : HasSubset.Subset (Set.Ioc l (f x)) v
          Hu : Exists fun u => LT.lt (f x) u
          u : γ
          fxu : LT.lt (f x) u
          hu : HasSubset.Subset (Set.Ico (f x) u) v
          a : α
          lfa : LT.lt l (f a)
          fau : LT.lt (f a) u
          h : LE.le (f a) (f x)
          ⊢ Membership.mem (Set.preimage f v) a
        -/
      · exact hl ⟨lfa, h⟩
        /-
          🎉 no goals
        -/
        /-
          case h.inr
          α : Type u_1
          inst✝³ : TopologicalSpace α
          x : α
          s : Set α
          γ : Type u_3
          inst✝² : LinearOrder γ
          inst✝¹ : TopologicalSpace γ
          inst✝ : OrderTopology γ
          f : α → γ
          h₁ : LowerSemicontinuousWithinAt f s x
          h₂ : UpperSemicontinuousWithinAt f s x
          v : Set γ
          hv : Membership.mem (nhds (f x)) v
          Hl : Exists fun l => LT.lt l (f x)
          l : γ
          lfx : LT.lt l (f x)
          hl : HasSubset.Subset (Set.Ioc l (f x)) v
          Hu : Exists fun u => LT.lt (f x) u
          u : γ
          fxu : LT.lt (f x) u
          hu : HasSubset.Subset (Set.Ico (f x) u) v
          a : α
          lfa : LT.lt l (f a)
          fau : LT.lt (f a) u
          h : GT.gt (f a) (f x)
          ⊢ Membership.mem (Set.preimage f v) a
        -/
      · exact hu ⟨le_of_lt h, fau⟩
        /-
          🎉 no goals
        -/
      /-
        case neg
        α : Type u_1
        inst✝³ : TopologicalSpace α
        x : α
        s : Set α
        γ : Type u_3
        inst✝² : LinearOrder γ
        inst✝¹ : TopologicalSpace γ
        inst✝ : OrderTopology γ
        f : α → γ
        h₁ : LowerSemicontinuousWithinAt f s x
        h₂ : UpperSemicontinuousWithinAt f s x
        v : Set γ
        hv : Membership.mem (nhds (f x)) v
        Hl : Exists fun l => LT.lt l (f x)
        l : γ
        lfx : LT.lt l (f x)
        hl : HasSubset.Subset (Set.Ioc l (f x)) v
        Hu : Not (Exists fun u => LT.lt (f x) u)
        ⊢ Membership.mem (nhdsWithin x s) (Set.preimage f v)
      -/
    · simp only [not_exists, not_lt] at Hu
      /-
        case neg
        α : Type u_1
        inst✝³ : TopologicalSpace α
        x : α
        s : Set α
        γ : Type u_3
        inst✝² : LinearOrder γ
        inst✝¹ : TopologicalSpace γ
        inst✝ : OrderTopology γ
        f : α → γ
        h₁ : LowerSemicontinuousWithinAt f s x
        h₂ : UpperSemicontinuousWithinAt f s x
        v : Set γ
        hv : Membership.mem (nhds (f x)) v
        Hl : Exists fun l => LT.lt l (f x)
        l : γ
        lfx : LT.lt l (f x)
        hl : HasSubset.Subset (Set.Ioc l (f x)) v
        Hu : ∀ (x_1 : γ), LE.le x_1 (f x)
        ⊢ Membership.mem (nhdsWithin x s) (Set.preimage f v)
      -/
      filter_upwards [h₁ l lfx] with a lfa using hl ⟨lfa, Hu (f a)⟩
      /-
        🎉 no goals
      -/
    /-
      case neg
      α : Type u_1
      inst✝³ : TopologicalSpace α
      x : α
      s : Set α
      γ : Type u_3
      inst✝² : LinearOrder γ
      inst✝¹ : TopologicalSpace γ
      inst✝ : OrderTopology γ
      f : α → γ
      h₁ : LowerSemicontinuousWithinAt f s x
      h₂ : UpperSemicontinuousWithinAt f s x
      v : Set γ
      hv : Membership.mem (nhds (f x)) v
      Hl : Not (Exists fun l => LT.lt l (f x))
      ⊢ Membership.mem (nhdsWithin x s) (Set.preimage f v)
    -/
  · simp only [not_exists, not_lt] at Hl
    /-
      case neg
      α : Type u_1
      inst✝³ : TopologicalSpace α
      x : α
      s : Set α
      γ : Type u_3
      inst✝² : LinearOrder γ
      inst✝¹ : TopologicalSpace γ
      inst✝ : OrderTopology γ
      f : α → γ
      h₁ : LowerSemicontinuousWithinAt f s x
      h₂ : UpperSemicontinuousWithinAt f s x
      v : Set γ
      hv : Membership.mem (nhds (f x)) v
      Hl : ∀ (x_1 : γ), LE.le (f x) x_1
      ⊢ Membership.mem (nhdsWithin x s) (Set.preimage f v)
    -/
    by_cases Hu : ∃ u, f x < u
      /-
        case pos
        α : Type u_1
        inst✝³ : TopologicalSpace α
        x : α
        s : Set α
        γ : Type u_3
        inst✝² : LinearOrder γ
        inst✝¹ : TopologicalSpace γ
        inst✝ : OrderTopology γ
        f : α → γ
        h₁ : LowerSemicontinuousWithinAt f s x
        h₂ : UpperSemicontinuousWithinAt f s x
        v : Set γ
        hv : Membership.mem (nhds (f x)) v
        Hl : ∀ (x_1 : γ), LE.le (f x) x_1
        Hu : Exists fun u => LT.lt (f x) u
        ⊢ Membership.mem (nhdsWithin x s) (Set.preimage f v)
      -/
    · rcases exists_Ico_subset_of_mem_nhds hv Hu with ⟨u, fxu, hu⟩
      /-
        case pos.intro.intro
        α : Type u_1
        inst✝³ : TopologicalSpace α
        x : α
        s : Set α
        γ : Type u_3
        inst✝² : LinearOrder γ
        inst✝¹ : TopologicalSpace γ
        inst✝ : OrderTopology γ
        f : α → γ
        h₁ : LowerSemicontinuousWithinAt f s x
        h₂ : UpperSemicontinuousWithinAt f s x
        v : Set γ
        hv : Membership.mem (nhds (f x)) v
        Hl : ∀ (x_1 : γ), LE.le (f x) x_1
        Hu : Exists fun u => LT.lt (f x) u
        u : γ
        fxu : LT.lt (f x) u
        hu : HasSubset.Subset (Set.Ico (f x) u) v
        ⊢ Membership.mem (nhdsWithin x s) (Set.preimage f v)
      -/
      filter_upwards [h₂ u fxu] with a lfa
      /-
        case h
        α : Type u_1
        inst✝³ : TopologicalSpace α
        x : α
        s : Set α
        γ : Type u_3
        inst✝² : LinearOrder γ
        inst✝¹ : TopologicalSpace γ
        inst✝ : OrderTopology γ
        f : α → γ
        h₁ : LowerSemicontinuousWithinAt f s x
        h₂ : UpperSemicontinuousWithinAt f s x
        v : Set γ
        hv : Membership.mem (nhds (f x)) v
        Hl : ∀ (x_1 : γ), LE.le (f x) x_1
        Hu : Exists fun u => LT.lt (f x) u
        u : γ
        fxu : LT.lt (f x) u
        hu : HasSubset.Subset (Set.Ico (f x) u) v
        a : α
        lfa : LT.lt (f a) u
        ⊢ Membership.mem (Set.preimage f v) a
      -/
      apply hu
      /-
        case h.a
        α : Type u_1
        inst✝³ : TopologicalSpace α
        x : α
        s : Set α
        γ : Type u_3
        inst✝² : LinearOrder γ
        inst✝¹ : TopologicalSpace γ
        inst✝ : OrderTopology γ
        f : α → γ
        h₁ : LowerSemicontinuousWithinAt f s x
        h₂ : UpperSemicontinuousWithinAt f s x
        v : Set γ
        hv : Membership.mem (nhds (f x)) v
        Hl : ∀ (x_1 : γ), LE.le (f x) x_1
        Hu : Exists fun u => LT.lt (f x) u
        u : γ
        fxu : LT.lt (f x) u
        hu : HasSubset.Subset (Set.Ico (f x) u) v
        a : α
        lfa : LT.lt (f a) u
        ⊢ Membership.mem (Set.Ico (f x) u) (f a)
      -/
      exact ⟨Hl (f a), lfa⟩
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        inst✝³ : TopologicalSpace α
        x : α
        s : Set α
        γ : Type u_3
        inst✝² : LinearOrder γ
        inst✝¹ : TopologicalSpace γ
        inst✝ : OrderTopology γ
        f : α → γ
        h₁ : LowerSemicontinuousWithinAt f s x
        h₂ : UpperSemicontinuousWithinAt f s x
        v : Set γ
        hv : Membership.mem (nhds (f x)) v
        Hl : ∀ (x_1 : γ), LE.le (f x) x_1
        Hu : Not (Exists fun u => LT.lt (f x) u)
        ⊢ Membership.mem (nhdsWithin x s) (Set.preimage f v)
      -/
    · simp only [not_exists, not_lt] at Hu
      /-
        case neg
        α : Type u_1
        inst✝³ : TopologicalSpace α
        x : α
        s : Set α
        γ : Type u_3
        inst✝² : LinearOrder γ
        inst✝¹ : TopologicalSpace γ
        inst✝ : OrderTopology γ
        f : α → γ
        h₁ : LowerSemicontinuousWithinAt f s x
        h₂ : UpperSemicontinuousWithinAt f s x
        v : Set γ
        hv : Membership.mem (nhds (f x)) v
        Hl : ∀ (x_1 : γ), LE.le (f x) x_1
        Hu : ∀ (x_1 : γ), LE.le x_1 (f x)
        ⊢ Membership.mem (nhdsWithin x s) (Set.preimage f v)
      -/
      apply Filter.Eventually.of_forall
      /-
        case neg.hp
        α : Type u_1
        inst✝³ : TopologicalSpace α
        x : α
        s : Set α
        γ : Type u_3
        inst✝² : LinearOrder γ
        inst✝¹ : TopologicalSpace γ
        inst✝ : OrderTopology γ
        f : α → γ
        h₁ : LowerSemicontinuousWithinAt f s x
        h₂ : UpperSemicontinuousWithinAt f s x
        v : Set γ
        hv : Membership.mem (nhds (f x)) v
        Hl : ∀ (x_1 : γ), LE.le (f x) x_1
        Hu : ∀ (x_1 : γ), LE.le x_1 (f x)
        ⊢ ∀ (x : α), Membership.mem v (f x)
      -/
      intro a
      /-
        case neg.hp
        α : Type u_1
        inst✝³ : TopologicalSpace α
        x : α
        s : Set α
        γ : Type u_3
        inst✝² : LinearOrder γ
        inst✝¹ : TopologicalSpace γ
        inst✝ : OrderTopology γ
        f : α → γ
        h₁ : LowerSemicontinuousWithinAt f s x
        h₂ : UpperSemicontinuousWithinAt f s x
        v : Set γ
        hv : Membership.mem (nhds (f x)) v
        Hl : ∀ (x_1 : γ), LE.le (f x) x_1
        Hu : ∀ (x_1 : γ), LE.le x_1 (f x)
        a : α
        ⊢ Membership.mem v (f a)
      -/
      have : f a = f x := le_antisymm (Hu _) (Hl _)
      /-
        case neg.hp
        α : Type u_1
        inst✝³ : TopologicalSpace α
        x : α
        s : Set α
        γ : Type u_3
        inst✝² : LinearOrder γ
        inst✝¹ : TopologicalSpace γ
        inst✝ : OrderTopology γ
        f : α → γ
        h₁ : LowerSemicontinuousWithinAt f s x
        h₂ : UpperSemicontinuousWithinAt f s x
        v : Set γ
        hv : Membership.mem (nhds (f x)) v
        Hl : ∀ (x_1 : γ), LE.le (f x) x_1
        Hu : ∀ (x_1 : γ), LE.le x_1 (f x)
        a : α
        this : Eq (f a) (f x)
        ⊢ Membership.mem v (f a)
      -/
      rw [this]
      /-
        case neg.hp
        α : Type u_1
        inst✝³ : TopologicalSpace α
        x : α
        s : Set α
        γ : Type u_3
        inst✝² : LinearOrder γ
        inst✝¹ : TopologicalSpace γ
        inst✝ : OrderTopology γ
        f : α → γ
        h₁ : LowerSemicontinuousWithinAt f s x
        h₂ : UpperSemicontinuousWithinAt f s x
        v : Set γ
        hv : Membership.mem (nhds (f x)) v
        Hl : ∀ (x_1 : γ), LE.le (f x) x_1
        Hu : ∀ (x_1 : γ), LE.le x_1 (f x)
        a : α
        this : Eq (f a) (f x)
        ⊢ Membership.mem v (f x)
      -/
      exact mem_of_mem_nhds hv
      /-
        🎉 no goals
      -/


theorem continuousAt_iff_lower_upperSemicontinuousAt {f : α → γ} :
    ContinuousAt f x ↔ LowerSemicontinuousAt f x ∧ UpperSemicontinuousAt f x := by
  simp_rw [← continuousWithinAt_univ, ← lowerSemicontinuousWithinAt_univ_iff, ←
    upperSemicontinuousWithinAt_univ_iff, continuousWithinAt_iff_lower_upperSemicontinuousWithinAt]


theorem continuousOn_iff_lower_upperSemicontinuousOn {f : α → γ} :
    ContinuousOn f s ↔ LowerSemicontinuousOn f s ∧ UpperSemicontinuousOn f s := by
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    s : Set α
    γ : Type u_3
    inst✝² : LinearOrder γ
    inst✝¹ : TopologicalSpace γ
    inst✝ : OrderTopology γ
    f : α → γ
    ⊢ Iff (ContinuousOn f s) (And (LowerSemicontinuousOn f s) (UpperSemicontinuous …
  -/
  simp only [ContinuousOn, continuousWithinAt_iff_lower_upperSemicontinuousWithinAt]
  exact
    ⟨fun H => ⟨fun x hx => (H x hx).1, fun x hx => (H x hx).2⟩, fun H x hx => ⟨H.1 x hx, H.2 x hx⟩⟩


theorem continuous_iff_lower_upperSemicontinuous {f : α → γ} :
    Continuous f ↔ LowerSemicontinuous f ∧ UpperSemicontinuous f := by
  simp_rw [continuous_iff_continuousOn_univ, continuousOn_iff_lower_upperSemicontinuousOn,
    lowerSemicontinuousOn_univ_iff, upperSemicontinuousOn_univ_iff]


