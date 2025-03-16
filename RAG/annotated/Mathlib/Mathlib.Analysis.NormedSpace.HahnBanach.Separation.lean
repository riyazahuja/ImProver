/-- Given a set `s` which is a convex neighbourhood of `0` and a point `x₀` outside of it, there is
a continuous linear functional `f` separating `x₀` and `s`, in the sense that it sends `x₀` to 1 and
all of `s` to values strictly below `1`. -/
theorem separate_convex_open_set [TopologicalSpace E] [AddCommGroup E] [TopologicalAddGroup E]
    [Module ℝ E] [ContinuousSMul ℝ E] {s : Set E} (hs₀ : (0 : E) ∈ s) (hs₁ : Convex ℝ s)
    (hs₂ : IsOpen s) {x₀ : E} (hx₀ : x₀ ∉ s) : ∃ f : E →L[ℝ] ℝ, f x₀ = 1 ∧ ∀ x ∈ s, f x < 1 := by
  /-
    E : Type u_2
    inst✝⁴ : TopologicalSpace E
    inst✝³ : AddCommGroup E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : Module Real E
    inst✝ : ContinuousSMul Real E
    s : Set E
    hs₀ : Membership.mem s 0
    hs₁ : Convex Real s
    hs₂ : IsOpen s
    x₀ : E
    hx₀ : Not (Membership.mem s x₀)
    ⊢ Exists fun f => And (Eq (f x₀) 1) (∀ (x : E), Membership.mem s x → LT.lt (f  …
  -/
  let f : E →ₗ.[ℝ] ℝ := LinearPMap.mkSpanSingleton x₀ 1 (ne_of_mem_of_not_mem hs₀ hx₀).symm
  have := exists_extension_of_le_sublinear f (gauge s) (fun c hc => gauge_smul_of_nonneg hc.le)
    (gauge_add_le hs₁ <| absorbent_nhds_zero <| hs₂.mem_nhds hs₀) ?_
    /-
      case refine_2
      E : Type u_2
      inst✝⁴ : TopologicalSpace E
      inst✝³ : AddCommGroup E
      inst✝² : TopologicalAddGroup E
      inst✝¹ : Module Real E
      inst✝ : ContinuousSMul Real E
      s : Set E
      hs₀ : Membership.mem s 0
      hs₁ : Convex Real s
      hs₂ : IsOpen s
      x₀ : E
      hx₀ : Not (Membership.mem s x₀)
      f : LinearPMap Real E Real := LinearPMap.mkSpanSingleton x₀ 1 ⋯
      this : Exists fun g => And (∀ (x : Subtype fun x => Membership.mem f.domain x) …
      ⊢ Exists fun f => And (Eq (f x₀) 1) (∀ (x : E), Membership.mem s x → LT.lt (f  …
    -/
  · obtain ⟨φ, hφ₁, hφ₂⟩ := this
    have hφ₃ : φ x₀ = 1 := by
      rw [← f.domain.coe_mk x₀ (Submodule.mem_span_singleton_self _), hφ₁,
        LinearPMap.mkSpanSingleton'_apply_self]
    have hφ₄ : ∀ x ∈ s, φ x < 1 := fun x hx =>
      (hφ₂ x).trans_lt (gauge_lt_one_of_mem_of_isOpen hs₂ hx)
    /-
      case refine_2.intro.intro
      E : Type u_2
      inst✝⁴ : TopologicalSpace E
      inst✝³ : AddCommGroup E
      inst✝² : TopologicalAddGroup E
      inst✝¹ : Module Real E
      inst✝ : ContinuousSMul Real E
      s : Set E
      hs₀ : Membership.mem s 0
      hs₁ : Convex Real s
      hs₂ : IsOpen s
      x₀ : E
      hx₀ : Not (Membership.mem s x₀)
      f : LinearPMap Real E Real := LinearPMap.mkSpanSingleton x₀ 1 ⋯
      φ : LinearMap (RingHom.id Real) E Real
      hφ₁ : ∀ (x : Subtype fun x => Membership.mem f.domain x), Eq (φ ↑x) (↑f x)
      hφ₂ : ∀ (x : E), LE.le (φ x) (gauge s x)
      hφ₃ : Eq (φ x₀) 1
      hφ₄ : ∀ (x : E), Membership.mem s x → LT.lt (φ x) 1
      ⊢ Exists fun f => And (Eq (f x₀) 1) (∀ (x : E), Membership.mem s x → LT.lt (f  …
    -/
    refine ⟨⟨φ, ?_⟩, hφ₃, hφ₄⟩
    refine
      φ.continuous_of_nonzero_on_open _ (hs₂.vadd (-x₀)) (Nonempty.vadd_set ⟨0, hs₀⟩)
        (vadd_set_subset_iff.mpr fun x hx => ?_)
    /-
      case refine_2.intro.intro
      E : Type u_2
      inst✝⁴ : TopologicalSpace E
      inst✝³ : AddCommGroup E
      inst✝² : TopologicalAddGroup E
      inst✝¹ : Module Real E
      inst✝ : ContinuousSMul Real E
      s : Set E
      hs₀ : Membership.mem s 0
      hs₁ : Convex Real s
      hs₂ : IsOpen s
      x₀ : E
      hx₀ : Not (Membership.mem s x₀)
      f : LinearPMap Real E Real := LinearPMap.mkSpanSingleton x₀ 1 ⋯
      φ : LinearMap (RingHom.id Real) E Real
      hφ₁ : ∀ (x : Subtype fun x => Membership.mem f.domain x), Eq (φ ↑x) (↑f x)
      hφ₂ : ∀ (x : E), LE.le (φ x) (gauge s x)
      hφ₃ : Eq (φ x₀) 1
      hφ₄ : ∀ (x : E), Membership.mem s x → LT.lt (φ x) 1
      x : E
      hx : Membership.mem s x
      ⊢ Membership.mem (fun x => Eq (φ x) 0 → False) (HVAdd.hVAdd (Neg.neg x₀) x)
    -/
    change φ (-x₀ + x) ≠ 0
    /-
      case refine_2.intro.intro
      E : Type u_2
      inst✝⁴ : TopologicalSpace E
      inst✝³ : AddCommGroup E
      inst✝² : TopologicalAddGroup E
      inst✝¹ : Module Real E
      inst✝ : ContinuousSMul Real E
      s : Set E
      hs₀ : Membership.mem s 0
      hs₁ : Convex Real s
      hs₂ : IsOpen s
      x₀ : E
      hx₀ : Not (Membership.mem s x₀)
      f : LinearPMap Real E Real := LinearPMap.mkSpanSingleton x₀ 1 ⋯
      φ : LinearMap (RingHom.id Real) E Real
      hφ₁ : ∀ (x : Subtype fun x => Membership.mem f.domain x), Eq (φ ↑x) (↑f x)
      hφ₂ : ∀ (x : E), LE.le (φ x) (gauge s x)
      hφ₃ : Eq (φ x₀) 1
      hφ₄ : ∀ (x : E), Membership.mem s x → LT.lt (φ x) 1
      x : E
      hx : Membership.mem s x
      ⊢ Ne (φ (HAdd.hAdd (Neg.neg x₀) x)) 0
    -/
    rw [map_add, map_neg]
    /-
      case refine_2.intro.intro
      E : Type u_2
      inst✝⁴ : TopologicalSpace E
      inst✝³ : AddCommGroup E
      inst✝² : TopologicalAddGroup E
      inst✝¹ : Module Real E
      inst✝ : ContinuousSMul Real E
      s : Set E
      hs₀ : Membership.mem s 0
      hs₁ : Convex Real s
      hs₂ : IsOpen s
      x₀ : E
      hx₀ : Not (Membership.mem s x₀)
      f : LinearPMap Real E Real := LinearPMap.mkSpanSingleton x₀ 1 ⋯
      φ : LinearMap (RingHom.id Real) E Real
      hφ₁ : ∀ (x : Subtype fun x => Membership.mem f.domain x), Eq (φ ↑x) (↑f x)
      hφ₂ : ∀ (x : E), LE.le (φ x) (gauge s x)
      hφ₃ : Eq (φ x₀) 1
      hφ₄ : ∀ (x : E), Membership.mem s x → LT.lt (φ x) 1
      x : E
      hx : Membership.mem s x
      ⊢ Ne (HAdd.hAdd (Neg.neg (φ x₀)) (φ x)) 0
    -/
    specialize hφ₄ x hx
    /-
      case refine_2.intro.intro
      E : Type u_2
      inst✝⁴ : TopologicalSpace E
      inst✝³ : AddCommGroup E
      inst✝² : TopologicalAddGroup E
      inst✝¹ : Module Real E
      inst✝ : ContinuousSMul Real E
      s : Set E
      hs₀ : Membership.mem s 0
      hs₁ : Convex Real s
      hs₂ : IsOpen s
      x₀ : E
      hx₀ : Not (Membership.mem s x₀)
      f : LinearPMap Real E Real := LinearPMap.mkSpanSingleton x₀ 1 ⋯
      φ : LinearMap (RingHom.id Real) E Real
      hφ₁ : ∀ (x : Subtype fun x => Membership.mem f.domain x), Eq (φ ↑x) (↑f x)
      hφ₂ : ∀ (x : E), LE.le (φ x) (gauge s x)
      hφ₃ : Eq (φ x₀) 1
      x : E
      hx : Membership.mem s x
      hφ₄ : LT.lt (φ x) 1
      ⊢ Ne (HAdd.hAdd (Neg.neg (φ x₀)) (φ x)) 0
    -/
    linarith
    /-
      🎉 no goals
    -/
  /-
    case refine_1
    E : Type u_2
    inst✝⁴ : TopologicalSpace E
    inst✝³ : AddCommGroup E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : Module Real E
    inst✝ : ContinuousSMul Real E
    s : Set E
    hs₀ : Membership.mem s 0
    hs₁ : Convex Real s
    hs₂ : IsOpen s
    x₀ : E
    hx₀ : Not (Membership.mem s x₀)
    f : LinearPMap Real E Real := LinearPMap.mkSpanSingleton x₀ 1 ⋯
    ⊢ ∀ (x : Subtype fun x => Membership.mem f.domain x), LE.le (↑f x) (gauge s ↑x)
  -/
  rintro ⟨x, hx⟩
  /-
    case refine_1.mk
    E : Type u_2
    inst✝⁴ : TopologicalSpace E
    inst✝³ : AddCommGroup E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : Module Real E
    inst✝ : ContinuousSMul Real E
    s : Set E
    hs₀ : Membership.mem s 0
    hs₁ : Convex Real s
    hs₂ : IsOpen s
    x₀ : E
    hx₀ : Not (Membership.mem s x₀)
    f : LinearPMap Real E Real := LinearPMap.mkSpanSingleton x₀ 1 ⋯
    x : E
    hx : Membership.mem f.domain x
    ⊢ LE.le (↑f ⟨x, hx⟩) (gauge s ↑⟨x, hx⟩)
  -/
  obtain ⟨y, rfl⟩ := Submodule.mem_span_singleton.1 hx
  /-
    case refine_1.mk.intro
    E : Type u_2
    inst✝⁴ : TopologicalSpace E
    inst✝³ : AddCommGroup E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : Module Real E
    inst✝ : ContinuousSMul Real E
    s : Set E
    hs₀ : Membership.mem s 0
    hs₁ : Convex Real s
    hs₂ : IsOpen s
    x₀ : E
    hx₀ : Not (Membership.mem s x₀)
    f : LinearPMap Real E Real := LinearPMap.mkSpanSingleton x₀ 1 ⋯
    y : Real
    hx : Membership.mem f.domain (HSMul.hSMul y x₀)
    ⊢ LE.le (↑f ⟨HSMul.hSMul y x₀, hx⟩) (gauge s ↑⟨HSMul.hSMul y x₀, hx⟩)
  -/
  rw [LinearPMap.mkSpanSingleton'_apply]
  /-
    case refine_1.mk.intro
    E : Type u_2
    inst✝⁴ : TopologicalSpace E
    inst✝³ : AddCommGroup E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : Module Real E
    inst✝ : ContinuousSMul Real E
    s : Set E
    hs₀ : Membership.mem s 0
    hs₁ : Convex Real s
    hs₂ : IsOpen s
    x₀ : E
    hx₀ : Not (Membership.mem s x₀)
    f : LinearPMap Real E Real := LinearPMap.mkSpanSingleton x₀ 1 ⋯
    y : Real
    hx : Membership.mem f.domain (HSMul.hSMul y x₀)
    ⊢ LE.le (HSMul.hSMul y 1) (gauge s ↑⟨HSMul.hSMul y x₀, hx⟩)
  -/
  simp only [mul_one, Algebra.id.smul_eq_mul, Submodule.coe_mk]
  /-
    case refine_1.mk.intro
    E : Type u_2
    inst✝⁴ : TopologicalSpace E
    inst✝³ : AddCommGroup E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : Module Real E
    inst✝ : ContinuousSMul Real E
    s : Set E
    hs₀ : Membership.mem s 0
    hs₁ : Convex Real s
    hs₂ : IsOpen s
    x₀ : E
    hx₀ : Not (Membership.mem s x₀)
    f : LinearPMap Real E Real := LinearPMap.mkSpanSingleton x₀ 1 ⋯
    y : Real
    hx : Membership.mem f.domain (HSMul.hSMul y x₀)
    ⊢ LE.le y (gauge s (HSMul.hSMul y x₀))
  -/
  obtain h | h := le_or_lt y 0
    /-
      case refine_1.mk.intro.inl
      E : Type u_2
      inst✝⁴ : TopologicalSpace E
      inst✝³ : AddCommGroup E
      inst✝² : TopologicalAddGroup E
      inst✝¹ : Module Real E
      inst✝ : ContinuousSMul Real E
      s : Set E
      hs₀ : Membership.mem s 0
      hs₁ : Convex Real s
      hs₂ : IsOpen s
      x₀ : E
      hx₀ : Not (Membership.mem s x₀)
      f : LinearPMap Real E Real := LinearPMap.mkSpanSingleton x₀ 1 ⋯
      y : Real
      hx : Membership.mem f.domain (HSMul.hSMul y x₀)
      h : LE.le y 0
      ⊢ LE.le y (gauge s (HSMul.hSMul y x₀))
    -/
  · exact h.trans (gauge_nonneg _)
    /-
      🎉 no goals
    -/
    /-
      case refine_1.mk.intro.inr
      E : Type u_2
      inst✝⁴ : TopologicalSpace E
      inst✝³ : AddCommGroup E
      inst✝² : TopologicalAddGroup E
      inst✝¹ : Module Real E
      inst✝ : ContinuousSMul Real E
      s : Set E
      hs₀ : Membership.mem s 0
      hs₁ : Convex Real s
      hs₂ : IsOpen s
      x₀ : E
      hx₀ : Not (Membership.mem s x₀)
      f : LinearPMap Real E Real := LinearPMap.mkSpanSingleton x₀ 1 ⋯
      y : Real
      hx : Membership.mem f.domain (HSMul.hSMul y x₀)
      h : LT.lt 0 y
      ⊢ LE.le y (gauge s (HSMul.hSMul y x₀))
    -/
  · rw [gauge_smul_of_nonneg h.le, smul_eq_mul, le_mul_iff_one_le_right h]
    exact
      one_le_gauge_of_not_mem (hs₁.starConvex hs₀)
        (absorbent_nhds_zero <| hs₂.mem_nhds hs₀).absorbs hx₀


/-- A version of the **Hahn-Banach theorem**: given disjoint convex sets `s`, `t` where `s` is open,
there is a continuous linear functional which separates them. -/
theorem geometric_hahn_banach_open (hs₁ : Convex ℝ s) (hs₂ : IsOpen s) (ht : Convex ℝ t)
    (disj : Disjoint s t) : ∃ (f : E →L[ℝ] ℝ) (u : ℝ), (∀ a ∈ s, f a < u) ∧ ∀ b ∈ t, u ≤ f b := by
  /-
    E : Type u_2
    inst✝⁴ : TopologicalSpace E
    inst✝³ : AddCommGroup E
    inst✝² : Module Real E
    s t : Set E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul Real E
    hs₁ : Convex Real s
    hs₂ : IsOpen s
    ht : Convex Real t
    disj : Disjoint s t
    ⊢ Exists fun f => Exists fun u => And (∀ (a : E), Membership.mem s a → LT.lt ( …
  -/
  obtain rfl | ⟨a₀, ha₀⟩ := s.eq_empty_or_nonempty
    /-
      case inl
      E : Type u_2
      inst✝⁴ : TopologicalSpace E
      inst✝³ : AddCommGroup E
      inst✝² : Module Real E
      t : Set E
      inst✝¹ : TopologicalAddGroup E
      inst✝ : ContinuousSMul Real E
      ht : Convex Real t
      hs₁ : Convex Real EmptyCollection.emptyCollection
      hs₂ : IsOpen EmptyCollection.emptyCollection
      disj : Disjoint EmptyCollection.emptyCollection t
      ⊢ Exists fun f => Exists fun u => And (∀ (a : E), Membership.mem EmptyCollecti …
    -/
  · exact ⟨0, 0, by simp, fun b _hb => le_rfl⟩
    /-
      🎉 no goals
    -/
  /-
    case inr.intro
    E : Type u_2
    inst✝⁴ : TopologicalSpace E
    inst✝³ : AddCommGroup E
    inst✝² : Module Real E
    s t : Set E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul Real E
    hs₁ : Convex Real s
    hs₂ : IsOpen s
    ht : Convex Real t
    disj : Disjoint s t
    a₀ : E
    ha₀ : Membership.mem s a₀
    ⊢ Exists fun f => Exists fun u => And (∀ (a : E), Membership.mem s a → LT.lt ( …
  -/
  obtain rfl | ⟨b₀, hb₀⟩ := t.eq_empty_or_nonempty
    /-
      case inr.intro.inl
      E : Type u_2
      inst✝⁴ : TopologicalSpace E
      inst✝³ : AddCommGroup E
      inst✝² : Module Real E
      s : Set E
      inst✝¹ : TopologicalAddGroup E
      inst✝ : ContinuousSMul Real E
      hs₁ : Convex Real s
      hs₂ : IsOpen s
      a₀ : E
      ha₀ : Membership.mem s a₀
      ht : Convex Real EmptyCollection.emptyCollection
      disj : Disjoint s EmptyCollection.emptyCollection
      ⊢ Exists fun f => Exists fun u => And (∀ (a : E), Membership.mem s a → LT.lt ( …
    -/
  · exact ⟨0, 1, fun a _ha => zero_lt_one, by simp⟩
    /-
      🎉 no goals
    -/
  /-
    case inr.intro.inr.intro
    E : Type u_2
    inst✝⁴ : TopologicalSpace E
    inst✝³ : AddCommGroup E
    inst✝² : Module Real E
    s t : Set E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul Real E
    hs₁ : Convex Real s
    hs₂ : IsOpen s
    ht : Convex Real t
    disj : Disjoint s t
    a₀ : E
    ha₀ : Membership.mem s a₀
    b₀ : E
    hb₀ : Membership.mem t b₀
    ⊢ Exists fun f => Exists fun u => And (∀ (a : E), Membership.mem s a → LT.lt ( …
  -/
  let x₀ := b₀ - a₀
  /-
    case inr.intro.inr.intro
    E : Type u_2
    inst✝⁴ : TopologicalSpace E
    inst✝³ : AddCommGroup E
    inst✝² : Module Real E
    s t : Set E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul Real E
    hs₁ : Convex Real s
    hs₂ : IsOpen s
    ht : Convex Real t
    disj : Disjoint s t
    a₀ : E
    ha₀ : Membership.mem s a₀
    b₀ : E
    hb₀ : Membership.mem t b₀
    x₀ : E := HSub.hSub b₀ a₀
    ⊢ Exists fun f => Exists fun u => And (∀ (a : E), Membership.mem s a → LT.lt ( …
  -/
  let C := x₀ +ᵥ (s - t)
  have : (0 : E) ∈ C :=
    ⟨a₀ - b₀, sub_mem_sub ha₀ hb₀, by simp_rw [x₀, vadd_eq_add, sub_add_sub_cancel', sub_self]⟩
  /-
    case inr.intro.inr.intro
    E : Type u_2
    inst✝⁴ : TopologicalSpace E
    inst✝³ : AddCommGroup E
    inst✝² : Module Real E
    s t : Set E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul Real E
    hs₁ : Convex Real s
    hs₂ : IsOpen s
    ht : Convex Real t
    disj : Disjoint s t
    a₀ : E
    ha₀ : Membership.mem s a₀
    b₀ : E
    hb₀ : Membership.mem t b₀
    x₀ : E := HSub.hSub b₀ a₀
    C : Set E := HVAdd.hVAdd x₀ (HSub.hSub s t)
    this : Membership.mem C 0
    ⊢ Exists fun f => Exists fun u => And (∀ (a : E), Membership.mem s a → LT.lt ( …
  -/
  have : Convex ℝ C := (hs₁.sub ht).vadd _
  have : x₀ ∉ C := by
    intro hx₀
    rw [← add_zero x₀] at hx₀
    exact disj.zero_not_mem_sub_set (vadd_mem_vadd_set_iff.1 hx₀)
  /-
    case inr.intro.inr.intro
    E : Type u_2
    inst✝⁴ : TopologicalSpace E
    inst✝³ : AddCommGroup E
    inst✝² : Module Real E
    s t : Set E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul Real E
    hs₁ : Convex Real s
    hs₂ : IsOpen s
    ht : Convex Real t
    disj : Disjoint s t
    a₀ : E
    ha₀ : Membership.mem s a₀
    b₀ : E
    hb₀ : Membership.mem t b₀
    x₀ : E := HSub.hSub b₀ a₀
    C : Set E := HVAdd.hVAdd x₀ (HSub.hSub s t)
    this✝¹ : Membership.mem C 0
    this✝ : Convex Real C
    this : Not (Membership.mem C x₀)
    ⊢ Exists fun f => Exists fun u => And (∀ (a : E), Membership.mem s a → LT.lt ( …
  -/
  obtain ⟨f, hf₁, hf₂⟩ := separate_convex_open_set ‹0 ∈ C› ‹_› (hs₂.sub_right.vadd _) ‹x₀ ∉ C›
  /-
    case inr.intro.inr.intro.intro.intro
    E : Type u_2
    inst✝⁴ : TopologicalSpace E
    inst✝³ : AddCommGroup E
    inst✝² : Module Real E
    s t : Set E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul Real E
    hs₁ : Convex Real s
    hs₂ : IsOpen s
    ht : Convex Real t
    disj : Disjoint s t
    a₀ : E
    ha₀ : Membership.mem s a₀
    b₀ : E
    hb₀ : Membership.mem t b₀
    x₀ : E := HSub.hSub b₀ a₀
    C : Set E := HVAdd.hVAdd x₀ (HSub.hSub s t)
    this✝¹ : Membership.mem C 0
    this✝ : Convex Real C
    this : Not (Membership.mem C x₀)
    f : ContinuousLinearMap (RingHom.id Real) E Real
    hf₁ : Eq (f x₀) 1
    hf₂ : ∀ (x : E), Membership.mem C x → LT.lt (f x) 1
    ⊢ Exists fun f => Exists fun u => And (∀ (a : E), Membership.mem s a → LT.lt ( …
  -/
  have : f b₀ = f a₀ + 1 := by simp [x₀, ← hf₁]
  have forall_le : ∀ a ∈ s, ∀ b ∈ t, f a ≤ f b := by
    intro a ha b hb
    have := hf₂ (x₀ + (a - b)) (vadd_mem_vadd_set <| sub_mem_sub ha hb)
    simp only [f.map_add, f.map_sub, hf₁] at this
    linarith
  /-
    case inr.intro.inr.intro.intro.intro
    E : Type u_2
    inst✝⁴ : TopologicalSpace E
    inst✝³ : AddCommGroup E
    inst✝² : Module Real E
    s t : Set E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul Real E
    hs₁ : Convex Real s
    hs₂ : IsOpen s
    ht : Convex Real t
    disj : Disjoint s t
    a₀ : E
    ha₀ : Membership.mem s a₀
    b₀ : E
    hb₀ : Membership.mem t b₀
    x₀ : E := HSub.hSub b₀ a₀
    C : Set E := HVAdd.hVAdd x₀ (HSub.hSub s t)
    this✝² : Membership.mem C 0
    this✝¹ : Convex Real C
    this✝ : Not (Membership.mem C x₀)
    f : ContinuousLinearMap (RingHom.id Real) E Real
    hf₁ : Eq (f x₀) 1
    hf₂ : ∀ (x : E), Membership.mem C x → LT.lt (f x) 1
    this : Eq (f b₀) (HAdd.hAdd (f a₀) 1)
    forall_le : ∀ (a : E), Membership.mem s a → ∀ (b : E), Membership.mem t b → LE …
    ⊢ Exists fun f => Exists fun u => And (∀ (a : E), Membership.mem s a → LT.lt ( …
  -/
  refine ⟨f, sInf (f '' t), image_subset_iff.1 (?_ : f '' s ⊆ Iio (sInf (f '' t))), fun b hb => ?_⟩
    /-
      case inr.intro.inr.intro.intro.intro.refine_1
      E : Type u_2
      inst✝⁴ : TopologicalSpace E
      inst✝³ : AddCommGroup E
      inst✝² : Module Real E
      s t : Set E
      inst✝¹ : TopologicalAddGroup E
      inst✝ : ContinuousSMul Real E
      hs₁ : Convex Real s
      hs₂ : IsOpen s
      ht : Convex Real t
      disj : Disjoint s t
      a₀ : E
      ha₀ : Membership.mem s a₀
      b₀ : E
      hb₀ : Membership.mem t b₀
      x₀ : E := HSub.hSub b₀ a₀
      C : Set E := HVAdd.hVAdd x₀ (HSub.hSub s t)
      this✝² : Membership.mem C 0
      this✝¹ : Convex Real C
      this✝ : Not (Membership.mem C x₀)
      f : ContinuousLinearMap (RingHom.id Real) E Real
      hf₁ : Eq (f x₀) 1
      hf₂ : ∀ (x : E), Membership.mem C x → LT.lt (f x) 1
      this : Eq (f b₀) (HAdd.hAdd (f a₀) 1)
      forall_le : ∀ (a : E), Membership.mem s a → ∀ (b : E), Membership.mem t b → LE …
      ⊢ HasSubset.Subset (Set.image (⇑f) s) (Set.Iio (InfSet.sInf (Set.image (⇑f) t)))
    -/
  · rw [← interior_Iic]
    /-
      case inr.intro.inr.intro.intro.intro.refine_1
      E : Type u_2
      inst✝⁴ : TopologicalSpace E
      inst✝³ : AddCommGroup E
      inst✝² : Module Real E
      s t : Set E
      inst✝¹ : TopologicalAddGroup E
      inst✝ : ContinuousSMul Real E
      hs₁ : Convex Real s
      hs₂ : IsOpen s
      ht : Convex Real t
      disj : Disjoint s t
      a₀ : E
      ha₀ : Membership.mem s a₀
      b₀ : E
      hb₀ : Membership.mem t b₀
      x₀ : E := HSub.hSub b₀ a₀
      C : Set E := HVAdd.hVAdd x₀ (HSub.hSub s t)
      this✝² : Membership.mem C 0
      this✝¹ : Convex Real C
      this✝ : Not (Membership.mem C x₀)
      f : ContinuousLinearMap (RingHom.id Real) E Real
      hf₁ : Eq (f x₀) 1
      hf₂ : ∀ (x : E), Membership.mem C x → LT.lt (f x) 1
      this : Eq (f b₀) (HAdd.hAdd (f a₀) 1)
      forall_le : ∀ (a : E), Membership.mem s a → ∀ (b : E), Membership.mem t b → LE …
      ⊢ HasSubset.Subset (Set.image (⇑f) s) (interior (Set.Iic (InfSet.sInf (Set.ima …
    -/
    refine interior_maximal (image_subset_iff.2 fun a ha => ?_) (f.isOpenMap_of_ne_zero ?_ _ hs₂)
      /-
        case inr.intro.inr.intro.intro.intro.refine_1.refine_1
        E : Type u_2
        inst✝⁴ : TopologicalSpace E
        inst✝³ : AddCommGroup E
        inst✝² : Module Real E
        s t : Set E
        inst✝¹ : TopologicalAddGroup E
        inst✝ : ContinuousSMul Real E
        hs₁ : Convex Real s
        hs₂ : IsOpen s
        ht : Convex Real t
        disj : Disjoint s t
        a₀ : E
        ha₀ : Membership.mem s a₀
        b₀ : E
        hb₀ : Membership.mem t b₀
        x₀ : E := HSub.hSub b₀ a₀
        C : Set E := HVAdd.hVAdd x₀ (HSub.hSub s t)
        this✝² : Membership.mem C 0
        this✝¹ : Convex Real C
        this✝ : Not (Membership.mem C x₀)
        f : ContinuousLinearMap (RingHom.id Real) E Real
        hf₁ : Eq (f x₀) 1
        hf₂ : ∀ (x : E), Membership.mem C x → LT.lt (f x) 1
        this : Eq (f b₀) (HAdd.hAdd (f a₀) 1)
        forall_le : ∀ (a : E), Membership.mem s a → ∀ (b : E), Membership.mem t b → LE …
        a : E
        ha : Membership.mem s a
        ⊢ Membership.mem (Set.preimage (⇑f) (Set.Iic (InfSet.sInf (Set.image (⇑f) t))) …
      -/
    · exact le_csInf (Nonempty.image _ ⟨_, hb₀⟩) (forall_mem_image.2 <| forall_le _ ha)
      /-
        🎉 no goals
      -/
      /-
        case inr.intro.inr.intro.intro.intro.refine_1.refine_2
        E : Type u_2
        inst✝⁴ : TopologicalSpace E
        inst✝³ : AddCommGroup E
        inst✝² : Module Real E
        s t : Set E
        inst✝¹ : TopologicalAddGroup E
        inst✝ : ContinuousSMul Real E
        hs₁ : Convex Real s
        hs₂ : IsOpen s
        ht : Convex Real t
        disj : Disjoint s t
        a₀ : E
        ha₀ : Membership.mem s a₀
        b₀ : E
        hb₀ : Membership.mem t b₀
        x₀ : E := HSub.hSub b₀ a₀
        C : Set E := HVAdd.hVAdd x₀ (HSub.hSub s t)
        this✝² : Membership.mem C 0
        this✝¹ : Convex Real C
        this✝ : Not (Membership.mem C x₀)
        f : ContinuousLinearMap (RingHom.id Real) E Real
        hf₁ : Eq (f x₀) 1
        hf₂ : ∀ (x : E), Membership.mem C x → LT.lt (f x) 1
        this : Eq (f b₀) (HAdd.hAdd (f a₀) 1)
        forall_le : ∀ (a : E), Membership.mem s a → ∀ (b : E), Membership.mem t b → LE …
        ⊢ Ne f 0
      -/
    · rintro rfl
      /-
        case inr.intro.inr.intro.intro.intro.refine_1.refine_2
        E : Type u_2
        inst✝⁴ : TopologicalSpace E
        inst✝³ : AddCommGroup E
        inst✝² : Module Real E
        s t : Set E
        inst✝¹ : TopologicalAddGroup E
        inst✝ : ContinuousSMul Real E
        hs₁ : Convex Real s
        hs₂ : IsOpen s
        ht : Convex Real t
        disj : Disjoint s t
        a₀ : E
        ha₀ : Membership.mem s a₀
        b₀ : E
        hb₀ : Membership.mem t b₀
        x₀ : E := HSub.hSub b₀ a₀
        C : Set E := HVAdd.hVAdd x₀ (HSub.hSub s t)
        this✝² : Membership.mem C 0
        this✝¹ : Convex Real C
        this✝ : Not (Membership.mem C x₀)
        hf₁ : Eq (0 x₀) 1
        hf₂ : ∀ (x : E), Membership.mem C x → LT.lt (0 x) 1
        this : Eq (0 b₀) (HAdd.hAdd (0 a₀) 1)
        forall_le : ∀ (a : E), Membership.mem s a → ∀ (b : E), Membership.mem t b → LE …
        ⊢ False
      -/
      simp at hf₁
      /-
        🎉 no goals
      -/
    /-
      case inr.intro.inr.intro.intro.intro.refine_2
      E : Type u_2
      inst✝⁴ : TopologicalSpace E
      inst✝³ : AddCommGroup E
      inst✝² : Module Real E
      s t : Set E
      inst✝¹ : TopologicalAddGroup E
      inst✝ : ContinuousSMul Real E
      hs₁ : Convex Real s
      hs₂ : IsOpen s
      ht : Convex Real t
      disj : Disjoint s t
      a₀ : E
      ha₀ : Membership.mem s a₀
      b₀ : E
      hb₀ : Membership.mem t b₀
      x₀ : E := HSub.hSub b₀ a₀
      C : Set E := HVAdd.hVAdd x₀ (HSub.hSub s t)
      this✝² : Membership.mem C 0
      this✝¹ : Convex Real C
      this✝ : Not (Membership.mem C x₀)
      f : ContinuousLinearMap (RingHom.id Real) E Real
      hf₁ : Eq (f x₀) 1
      hf₂ : ∀ (x : E), Membership.mem C x → LT.lt (f x) 1
      this : Eq (f b₀) (HAdd.hAdd (f a₀) 1)
      forall_le : ∀ (a : E), Membership.mem s a → ∀ (b : E), Membership.mem t b → LE …
      b : E
      hb : Membership.mem t b
      ⊢ LE.le (InfSet.sInf (Set.image (⇑f) t)) (f b)
    -/
  · exact csInf_le ⟨f a₀, forall_mem_image.2 <| forall_le _ ha₀⟩ (mem_image_of_mem _ hb)
    /-
      🎉 no goals
    -/


theorem geometric_hahn_banach_open_point (hs₁ : Convex ℝ s) (hs₂ : IsOpen s) (disj : x ∉ s) :
    ∃ f : E →L[ℝ] ℝ, ∀ a ∈ s, f a < f x :=
  let ⟨f, _s, hs, hx⟩ :=
    geometric_hahn_banach_open hs₁ hs₂ (convex_singleton x) (disjoint_singleton_right.2 disj)
  ⟨f, fun a ha => lt_of_lt_of_le (hs a ha) (hx x (mem_singleton _))⟩


theorem geometric_hahn_banach_point_open (ht₁ : Convex ℝ t) (ht₂ : IsOpen t) (disj : x ∉ t) :
    ∃ f : E →L[ℝ] ℝ, ∀ b ∈ t, f x < f b :=
  let ⟨f, hf⟩ := geometric_hahn_banach_open_point ht₁ ht₂ disj
          /-
            E : Type u_2
            inst✝⁴ : TopologicalSpace E
            inst✝³ : AddCommGroup E
            inst✝² : Module Real E
            t : Set E
            x : E
            inst✝¹ : TopologicalAddGroup E
            inst✝ : ContinuousSMul Real E
            ht₁ : Convex Real t
            ht₂ : IsOpen t
            disj : Not (Membership.mem t x)
            f : ContinuousLinearMap (RingHom.id Real) E Real
            hf : ∀ (a : E), Membership.mem t a → LT.lt (f a) (f x)
            ⊢ ∀ (b : E), Membership.mem t b → LT.lt ((Neg.neg f) x) ((Neg.neg f) b)
          -/
  ⟨-f, by simpa⟩
          /-
            🎉 no goals
          -/


theorem geometric_hahn_banach_open_open (hs₁ : Convex ℝ s) (hs₂ : IsOpen s) (ht₁ : Convex ℝ t)
    (ht₃ : IsOpen t) (disj : Disjoint s t) :
    ∃ (f : E →L[ℝ] ℝ) (u : ℝ), (∀ a ∈ s, f a < u) ∧ ∀ b ∈ t, u < f b := by
  /-
    E : Type u_2
    inst✝⁴ : TopologicalSpace E
    inst✝³ : AddCommGroup E
    inst✝² : Module Real E
    s t : Set E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul Real E
    hs₁ : Convex Real s
    hs₂ : IsOpen s
    ht₁ : Convex Real t
    ht₃ : IsOpen t
    disj : Disjoint s t
    ⊢ Exists fun f => Exists fun u => And (∀ (a : E), Membership.mem s a → LT.lt ( …
  -/
  obtain rfl | ⟨a₀, ha₀⟩ := s.eq_empty_or_nonempty
    /-
      case inl
      E : Type u_2
      inst✝⁴ : TopologicalSpace E
      inst✝³ : AddCommGroup E
      inst✝² : Module Real E
      t : Set E
      inst✝¹ : TopologicalAddGroup E
      inst✝ : ContinuousSMul Real E
      ht₁ : Convex Real t
      ht₃ : IsOpen t
      hs₁ : Convex Real EmptyCollection.emptyCollection
      hs₂ : IsOpen EmptyCollection.emptyCollection
      disj : Disjoint EmptyCollection.emptyCollection t
      ⊢ Exists fun f => Exists fun u => And (∀ (a : E), Membership.mem EmptyCollecti …
    -/
  · exact ⟨0, -1, by simp, fun b _hb => by norm_num⟩
    /-
      🎉 no goals
    -/
  /-
    case inr.intro
    E : Type u_2
    inst✝⁴ : TopologicalSpace E
    inst✝³ : AddCommGroup E
    inst✝² : Module Real E
    s t : Set E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul Real E
    hs₁ : Convex Real s
    hs₂ : IsOpen s
    ht₁ : Convex Real t
    ht₃ : IsOpen t
    disj : Disjoint s t
    a₀ : E
    ha₀ : Membership.mem s a₀
    ⊢ Exists fun f => Exists fun u => And (∀ (a : E), Membership.mem s a → LT.lt ( …
  -/
  obtain rfl | ⟨b₀, hb₀⟩ := t.eq_empty_or_nonempty
    /-
      case inr.intro.inl
      E : Type u_2
      inst✝⁴ : TopologicalSpace E
      inst✝³ : AddCommGroup E
      inst✝² : Module Real E
      s : Set E
      inst✝¹ : TopologicalAddGroup E
      inst✝ : ContinuousSMul Real E
      hs₁ : Convex Real s
      hs₂ : IsOpen s
      a₀ : E
      ha₀ : Membership.mem s a₀
      ht₁ : Convex Real EmptyCollection.emptyCollection
      ht₃ : IsOpen EmptyCollection.emptyCollection
      disj : Disjoint s EmptyCollection.emptyCollection
      ⊢ Exists fun f => Exists fun u => And (∀ (a : E), Membership.mem s a → LT.lt ( …
    -/
  · exact ⟨0, 1, fun a _ha => by norm_num, by simp⟩
    /-
      🎉 no goals
    -/
  /-
    case inr.intro.inr.intro
    E : Type u_2
    inst✝⁴ : TopologicalSpace E
    inst✝³ : AddCommGroup E
    inst✝² : Module Real E
    s t : Set E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul Real E
    hs₁ : Convex Real s
    hs₂ : IsOpen s
    ht₁ : Convex Real t
    ht₃ : IsOpen t
    disj : Disjoint s t
    a₀ : E
    ha₀ : Membership.mem s a₀
    b₀ : E
    hb₀ : Membership.mem t b₀
    ⊢ Exists fun f => Exists fun u => And (∀ (a : E), Membership.mem s a → LT.lt ( …
  -/
  obtain ⟨f, s, hf₁, hf₂⟩ := geometric_hahn_banach_open hs₁ hs₂ ht₁ disj
  have hf : IsOpenMap f := by
    refine f.isOpenMap_of_ne_zero ?_
    rintro rfl
    simp_rw [ContinuousLinearMap.zero_apply] at hf₁ hf₂
    exact (hf₁ _ ha₀).not_le (hf₂ _ hb₀)
  /-
    case inr.intro.inr.intro.intro.intro.intro
    E : Type u_2
    inst✝⁴ : TopologicalSpace E
    inst✝³ : AddCommGroup E
    inst✝² : Module Real E
    s✝ t : Set E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul Real E
    hs₁ : Convex Real s✝
    hs₂ : IsOpen s✝
    ht₁ : Convex Real t
    ht₃ : IsOpen t
    disj : Disjoint s✝ t
    a₀ : E
    ha₀ : Membership.mem s✝ a₀
    b₀ : E
    hb₀ : Membership.mem t b₀
    f : ContinuousLinearMap (RingHom.id Real) E Real
    s : Real
    hf₁ : ∀ (a : E), Membership.mem s✝ a → LT.lt (f a) s
    hf₂ : ∀ (b : E), Membership.mem t b → LE.le s (f b)
    hf : IsOpenMap ⇑f
    ⊢ Exists fun f => Exists fun u => And (∀ (a : E), Membership.mem s✝ a → LT.lt  …
  -/
  refine ⟨f, s, hf₁, image_subset_iff.1 (?_ : f '' t ⊆ Ioi s)⟩
  /-
    case inr.intro.inr.intro.intro.intro.intro
    E : Type u_2
    inst✝⁴ : TopologicalSpace E
    inst✝³ : AddCommGroup E
    inst✝² : Module Real E
    s✝ t : Set E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul Real E
    hs₁ : Convex Real s✝
    hs₂ : IsOpen s✝
    ht₁ : Convex Real t
    ht₃ : IsOpen t
    disj : Disjoint s✝ t
    a₀ : E
    ha₀ : Membership.mem s✝ a₀
    b₀ : E
    hb₀ : Membership.mem t b₀
    f : ContinuousLinearMap (RingHom.id Real) E Real
    s : Real
    hf₁ : ∀ (a : E), Membership.mem s✝ a → LT.lt (f a) s
    hf₂ : ∀ (b : E), Membership.mem t b → LE.le s (f b)
    hf : IsOpenMap ⇑f
    ⊢ HasSubset.Subset (Set.image (⇑f) t) (Set.Ioi s)
  -/
  rw [← interior_Ici]
  /-
    case inr.intro.inr.intro.intro.intro.intro
    E : Type u_2
    inst✝⁴ : TopologicalSpace E
    inst✝³ : AddCommGroup E
    inst✝² : Module Real E
    s✝ t : Set E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul Real E
    hs₁ : Convex Real s✝
    hs₂ : IsOpen s✝
    ht₁ : Convex Real t
    ht₃ : IsOpen t
    disj : Disjoint s✝ t
    a₀ : E
    ha₀ : Membership.mem s✝ a₀
    b₀ : E
    hb₀ : Membership.mem t b₀
    f : ContinuousLinearMap (RingHom.id Real) E Real
    s : Real
    hf₁ : ∀ (a : E), Membership.mem s✝ a → LT.lt (f a) s
    hf₂ : ∀ (b : E), Membership.mem t b → LE.le s (f b)
    hf : IsOpenMap ⇑f
    ⊢ HasSubset.Subset (Set.image (⇑f) t) (interior (Set.Ici s))
  -/
  refine interior_maximal (image_subset_iff.2 hf₂) (f.isOpenMap_of_ne_zero ?_ _ ht₃)
  /-
    case inr.intro.inr.intro.intro.intro.intro
    E : Type u_2
    inst✝⁴ : TopologicalSpace E
    inst✝³ : AddCommGroup E
    inst✝² : Module Real E
    s✝ t : Set E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul Real E
    hs₁ : Convex Real s✝
    hs₂ : IsOpen s✝
    ht₁ : Convex Real t
    ht₃ : IsOpen t
    disj : Disjoint s✝ t
    a₀ : E
    ha₀ : Membership.mem s✝ a₀
    b₀ : E
    hb₀ : Membership.mem t b₀
    f : ContinuousLinearMap (RingHom.id Real) E Real
    s : Real
    hf₁ : ∀ (a : E), Membership.mem s✝ a → LT.lt (f a) s
    hf₂ : ∀ (b : E), Membership.mem t b → LE.le s (f b)
    hf : IsOpenMap ⇑f
    ⊢ Ne f 0
  -/
  rintro rfl
  /-
    case inr.intro.inr.intro.intro.intro.intro
    E : Type u_2
    inst✝⁴ : TopologicalSpace E
    inst✝³ : AddCommGroup E
    inst✝² : Module Real E
    s✝ t : Set E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul Real E
    hs₁ : Convex Real s✝
    hs₂ : IsOpen s✝
    ht₁ : Convex Real t
    ht₃ : IsOpen t
    disj : Disjoint s✝ t
    a₀ : E
    ha₀ : Membership.mem s✝ a₀
    b₀ : E
    hb₀ : Membership.mem t b₀
    s : Real
    hf₁ : ∀ (a : E), Membership.mem s✝ a → LT.lt (0 a) s
    hf₂ : ∀ (b : E), Membership.mem t b → LE.le s (0 b)
    hf : IsOpenMap ⇑0
    ⊢ False
  -/
  simp_rw [ContinuousLinearMap.zero_apply] at hf₁ hf₂
  /-
    case inr.intro.inr.intro.intro.intro.intro
    E : Type u_2
    inst✝⁴ : TopologicalSpace E
    inst✝³ : AddCommGroup E
    inst✝² : Module Real E
    s✝ t : Set E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul Real E
    hs₁ : Convex Real s✝
    hs₂ : IsOpen s✝
    ht₁ : Convex Real t
    ht₃ : IsOpen t
    disj : Disjoint s✝ t
    a₀ : E
    ha₀ : Membership.mem s✝ a₀
    b₀ : E
    hb₀ : Membership.mem t b₀
    s : Real
    hf₁ : ∀ (a : E), Membership.mem s✝ a → LT.lt 0 s
    hf₂ : ∀ (b : E), Membership.mem t b → LE.le s 0
    hf : IsOpenMap ⇑0
    ⊢ False
  -/
  exact (hf₁ _ ha₀).not_le (hf₂ _ hb₀)
  /-
    🎉 no goals
  -/


/-- A version of the **Hahn-Banach theorem**: given disjoint convex sets `s`, `t` where `s` is
compact and `t` is closed, there is a continuous linear functional which strongly separates them. -/
theorem geometric_hahn_banach_compact_closed (hs₁ : Convex ℝ s) (hs₂ : IsCompact s)
    (ht₁ : Convex ℝ t) (ht₂ : IsClosed t) (disj : Disjoint s t) :
    ∃ (f : E →L[ℝ] ℝ) (u v : ℝ), (∀ a ∈ s, f a < u) ∧ u < v ∧ ∀ b ∈ t, v < f b := by
  /-
    E : Type u_2
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    s t : Set E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : ContinuousSMul Real E
    inst✝ : LocallyConvexSpace Real E
    hs₁ : Convex Real s
    hs₂ : IsCompact s
    ht₁ : Convex Real t
    ht₂ : IsClosed t
    disj : Disjoint s t
    ⊢ Exists fun f => Exists fun u => Exists fun v => And (∀ (a : E), Membership.m …
  -/
  obtain rfl | hs := s.eq_empty_or_nonempty
    /-
      case inl
      E : Type u_2
      inst✝⁵ : TopologicalSpace E
      inst✝⁴ : AddCommGroup E
      inst✝³ : Module Real E
      t : Set E
      inst✝² : TopologicalAddGroup E
      inst✝¹ : ContinuousSMul Real E
      inst✝ : LocallyConvexSpace Real E
      ht₁ : Convex Real t
      ht₂ : IsClosed t
      hs₁ : Convex Real EmptyCollection.emptyCollection
      hs₂ : IsCompact EmptyCollection.emptyCollection
      disj : Disjoint EmptyCollection.emptyCollection t
      ⊢ Exists fun f => Exists fun u => Exists fun v => And (∀ (a : E), Membership.m …
    -/
  · exact ⟨0, -2, -1, by simp, by norm_num, fun b _hb => by norm_num⟩
    /-
      🎉 no goals
    -/
  /-
    case inr
    E : Type u_2
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    s t : Set E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : ContinuousSMul Real E
    inst✝ : LocallyConvexSpace Real E
    hs₁ : Convex Real s
    hs₂ : IsCompact s
    ht₁ : Convex Real t
    ht₂ : IsClosed t
    disj : Disjoint s t
    hs : s.Nonempty
    ⊢ Exists fun f => Exists fun u => Exists fun v => And (∀ (a : E), Membership.m …
  -/
  obtain rfl | _ht := t.eq_empty_or_nonempty
    /-
      case inr.inl
      E : Type u_2
      inst✝⁵ : TopologicalSpace E
      inst✝⁴ : AddCommGroup E
      inst✝³ : Module Real E
      s : Set E
      inst✝² : TopologicalAddGroup E
      inst✝¹ : ContinuousSMul Real E
      inst✝ : LocallyConvexSpace Real E
      hs₁ : Convex Real s
      hs₂ : IsCompact s
      hs : s.Nonempty
      ht₁ : Convex Real EmptyCollection.emptyCollection
      ht₂ : IsClosed EmptyCollection.emptyCollection
      disj : Disjoint s EmptyCollection.emptyCollection
      ⊢ Exists fun f => Exists fun u => Exists fun v => And (∀ (a : E), Membership.m …
    -/
  · exact ⟨0, 1, 2, fun a _ha => by norm_num, by norm_num, by simp⟩
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    E : Type u_2
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    s t : Set E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : ContinuousSMul Real E
    inst✝ : LocallyConvexSpace Real E
    hs₁ : Convex Real s
    hs₂ : IsCompact s
    ht₁ : Convex Real t
    ht₂ : IsClosed t
    disj : Disjoint s t
    hs : s.Nonempty
    _ht : t.Nonempty
    ⊢ Exists fun f => Exists fun u => Exists fun v => And (∀ (a : E), Membership.m …
  -/
  obtain ⟨U, V, hU, hV, hU₁, hV₁, sU, tV, disj'⟩ := disj.exists_open_convexes hs₁ hs₂ ht₁ ht₂
  /-
    case inr.inr.intro.intro.intro.intro.intro.intro.intro.intro
    E : Type u_2
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    s t : Set E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : ContinuousSMul Real E
    inst✝ : LocallyConvexSpace Real E
    hs₁ : Convex Real s
    hs₂ : IsCompact s
    ht₁ : Convex Real t
    ht₂ : IsClosed t
    disj : Disjoint s t
    hs : s.Nonempty
    _ht : t.Nonempty
    U V : Set E
    hU : IsOpen U
    hV : IsOpen V
    hU₁ : Convex Real U
    hV₁ : Convex Real V
    sU : HasSubset.Subset s U
    tV : HasSubset.Subset t V
    disj' : Disjoint U V
    ⊢ Exists fun f => Exists fun u => Exists fun v => And (∀ (a : E), Membership.m …
  -/
  obtain ⟨f, u, hf₁, hf₂⟩ := geometric_hahn_banach_open_open hU₁ hU hV₁ hV disj'
  /-
    case inr.inr.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    E : Type u_2
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    s t : Set E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : ContinuousSMul Real E
    inst✝ : LocallyConvexSpace Real E
    hs₁ : Convex Real s
    hs₂ : IsCompact s
    ht₁ : Convex Real t
    ht₂ : IsClosed t
    disj : Disjoint s t
    hs : s.Nonempty
    _ht : t.Nonempty
    U V : Set E
    hU : IsOpen U
    hV : IsOpen V
    hU₁ : Convex Real U
    hV₁ : Convex Real V
    sU : HasSubset.Subset s U
    tV : HasSubset.Subset t V
    disj' : Disjoint U V
    f : ContinuousLinearMap (RingHom.id Real) E Real
    u : Real
    hf₁ : ∀ (a : E), Membership.mem U a → LT.lt (f a) u
    hf₂ : ∀ (b : E), Membership.mem V b → LT.lt u (f b)
    ⊢ Exists fun f => Exists fun u => Exists fun v => And (∀ (a : E), Membership.m …
  -/
  obtain ⟨x, hx₁, hx₂⟩ := hs₂.exists_isMaxOn hs f.continuous.continuousOn
  /-
    case inr.inr.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro …
    E : Type u_2
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    s t : Set E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : ContinuousSMul Real E
    inst✝ : LocallyConvexSpace Real E
    hs₁ : Convex Real s
    hs₂ : IsCompact s
    ht₁ : Convex Real t
    ht₂ : IsClosed t
    disj : Disjoint s t
    hs : s.Nonempty
    _ht : t.Nonempty
    U V : Set E
    hU : IsOpen U
    hV : IsOpen V
    hU₁ : Convex Real U
    hV₁ : Convex Real V
    sU : HasSubset.Subset s U
    tV : HasSubset.Subset t V
    disj' : Disjoint U V
    f : ContinuousLinearMap (RingHom.id Real) E Real
    u : Real
    hf₁ : ∀ (a : E), Membership.mem U a → LT.lt (f a) u
    hf₂ : ∀ (b : E), Membership.mem V b → LT.lt u (f b)
    x : E
    hx₁ : Membership.mem s x
    hx₂ : IsMaxOn (⇑f) s x
    ⊢ Exists fun f => Exists fun u => Exists fun v => And (∀ (a : E), Membership.m …
  -/
  have : f x < u := hf₁ x (sU hx₁)
  exact
    ⟨f, (f x + u) / 2, u,
      fun a ha => by have := hx₂ ha; dsimp at this; linarith,
      by linarith,
      fun b hb => hf₂ b (tV hb)⟩


/-- A version of the **Hahn-Banach theorem**: given disjoint convex sets `s`, `t` where `s` is
closed, and `t` is compact, there is a continuous linear functional which strongly separates them.
-/
theorem geometric_hahn_banach_closed_compact (hs₁ : Convex ℝ s) (hs₂ : IsClosed s)
    (ht₁ : Convex ℝ t) (ht₂ : IsCompact t) (disj : Disjoint s t) :
    ∃ (f : E →L[ℝ] ℝ) (u v : ℝ), (∀ a ∈ s, f a < u) ∧ u < v ∧ ∀ b ∈ t, v < f b :=
  let ⟨f, s, t, hs, st, ht⟩ := geometric_hahn_banach_compact_closed ht₁ ht₂ hs₁ hs₂ disj.symm
                  /-
                    E : Type u_2
                    inst✝⁵ : TopologicalSpace E
                    inst✝⁴ : AddCommGroup E
                    inst✝³ : Module Real E
                    s✝ t✝ : Set E
                    inst✝² : TopologicalAddGroup E
                    inst✝¹ : ContinuousSMul Real E
                    inst✝ : LocallyConvexSpace Real E
                    hs₁ : Convex Real s✝
                    hs₂ : IsClosed s✝
                    ht₁ : Convex Real t✝
                    ht₂ : IsCompact t✝
                    disj : Disjoint s✝ t✝
                    f : ContinuousLinearMap (RingHom.id Real) E Real
                    s t : Real
                    hs : ∀ (a : E), Membership.mem t✝ a → LT.lt (f a) s
                    st : LT.lt s t
                    ht : ∀ (b : E), Membership.mem s✝ b → LT.lt t (f b)
                    ⊢ ∀ (a : E), Membership.mem s✝ a → LT.lt ((Neg.neg f) a) (Neg.neg t)
                  -/
                  /-
                    🎉 no goals
                  -/
                                     /-
                                       🎉 no goals
                                     -/
  ⟨-f, -t, -s, by simpa using ht, by simpa using st, by simpa using hs⟩
                                                        /-
                                                          🎉 no goals
                                                        -/


theorem geometric_hahn_banach_point_closed (ht₁ : Convex ℝ t) (ht₂ : IsClosed t) (disj : x ∉ t) :
    ∃ (f : E →L[ℝ] ℝ) (u : ℝ), f x < u ∧ ∀ b ∈ t, u < f b :=
  let ⟨f, _u, v, ha, hst, hb⟩ :=
    geometric_hahn_banach_compact_closed (convex_singleton x) isCompact_singleton ht₁ ht₂
      (disjoint_singleton_left.2 disj)
  ⟨f, v, hst.trans' <| ha x <| mem_singleton _, hb⟩


theorem geometric_hahn_banach_closed_point (hs₁ : Convex ℝ s) (hs₂ : IsClosed s) (disj : x ∉ s) :
    ∃ (f : E →L[ℝ] ℝ) (u : ℝ), (∀ a ∈ s, f a < u) ∧ u < f x :=
  let ⟨f, s, _t, ha, hst, hb⟩ :=
    geometric_hahn_banach_closed_compact hs₁ hs₂ (convex_singleton x) isCompact_singleton
      (disjoint_singleton_right.2 disj)
  ⟨f, s, ha, hst.trans <| hb x <| mem_singleton _⟩


/-- See also `NormedSpace.eq_iff_forall_dual_eq`. -/
theorem geometric_hahn_banach_point_point [T1Space E] (hxy : x ≠ y) :
    ∃ f : E →L[ℝ] ℝ, f x < f y := by
  obtain ⟨f, s, t, hs, st, ht⟩ :=
    geometric_hahn_banach_compact_closed (convex_singleton x) isCompact_singleton
      (convex_singleton y) isClosed_singleton (disjoint_singleton.2 hxy)
  /-
    case intro.intro.intro.intro.intro
    E : Type u_2
    inst✝⁶ : TopologicalSpace E
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : Module Real E
    x y : E
    inst✝³ : TopologicalAddGroup E
    inst✝² : ContinuousSMul Real E
    inst✝¹ : LocallyConvexSpace Real E
    inst✝ : T1Space E
    hxy : Ne x y
    f : ContinuousLinearMap (RingHom.id Real) E Real
    s t : Real
    hs : ∀ (a : E), Membership.mem (Singleton.singleton x) a → LT.lt (f a) s
    st : LT.lt s t
    ht : ∀ (b : E), Membership.mem (Singleton.singleton y) b → LT.lt t (f b)
    ⊢ Exists fun f => LT.lt (f x) (f y)
  -/
  exact ⟨f, by linarith [hs x rfl, ht y rfl]⟩
  /-
    🎉 no goals
  -/


/-- A closed convex set is the intersection of the half-spaces containing it. -/
theorem iInter_halfSpaces_eq (hs₁ : Convex ℝ s) (hs₂ : IsClosed s) :
    ⋂ l : E →L[ℝ] ℝ, { x | ∃ y ∈ s, l x ≤ l y } = s := by
  /-
    E : Type u_2
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    s : Set E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : ContinuousSMul Real E
    inst✝ : LocallyConvexSpace Real E
    hs₁ : Convex Real s
    hs₂ : IsClosed s
    ⊢ Eq (Set.iInter fun l => setOf fun x => Exists fun y => And (Membership.mem s …
  -/
  rw [Set.iInter_setOf]
  /-
    E : Type u_2
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    s : Set E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : ContinuousSMul Real E
    inst✝ : LocallyConvexSpace Real E
    hs₁ : Convex Real s
    hs₂ : IsClosed s
    ⊢ Eq (setOf fun x => ∀ (i : ContinuousLinearMap (RingHom.id Real) E Real), Exi …
  -/
  refine Set.Subset.antisymm (fun x hx => ?_) fun x hx l => ⟨x, hx, le_rfl⟩
  /-
    E : Type u_2
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    s : Set E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : ContinuousSMul Real E
    inst✝ : LocallyConvexSpace Real E
    hs₁ : Convex Real s
    hs₂ : IsClosed s
    x : E
    hx : Membership.mem (setOf fun x => ∀ (i : ContinuousLinearMap (RingHom.id Rea …
    ⊢ Membership.mem s x
  -/
  by_contra h
  /-
    E : Type u_2
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    s : Set E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : ContinuousSMul Real E
    inst✝ : LocallyConvexSpace Real E
    hs₁ : Convex Real s
    hs₂ : IsClosed s
    x : E
    hx : Membership.mem (setOf fun x => ∀ (i : ContinuousLinearMap (RingHom.id Rea …
    h : Not (Membership.mem s x)
    ⊢ False
  -/
  obtain ⟨l, s, hlA, hl⟩ := geometric_hahn_banach_closed_point hs₁ hs₂ h
  /-
    case intro.intro.intro
    E : Type u_2
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    s✝ : Set E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : ContinuousSMul Real E
    inst✝ : LocallyConvexSpace Real E
    hs₁ : Convex Real s✝
    hs₂ : IsClosed s✝
    x : E
    hx : Membership.mem (setOf fun x => ∀ (i : ContinuousLinearMap (RingHom.id Rea …
    h : Not (Membership.mem s✝ x)
    l : ContinuousLinearMap (RingHom.id Real) E Real
    s : Real
    hlA : ∀ (a : E), Membership.mem s✝ a → LT.lt (l a) s
    hl : LT.lt s (l x)
    ⊢ False
  -/
  obtain ⟨y, hy, hxy⟩ := hx l
  /-
    case intro.intro.intro.intro.intro
    E : Type u_2
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    s✝ : Set E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : ContinuousSMul Real E
    inst✝ : LocallyConvexSpace Real E
    hs₁ : Convex Real s✝
    hs₂ : IsClosed s✝
    x : E
    hx : Membership.mem (setOf fun x => ∀ (i : ContinuousLinearMap (RingHom.id Rea …
    h : Not (Membership.mem s✝ x)
    l : ContinuousLinearMap (RingHom.id Real) E Real
    s : Real
    hlA : ∀ (a : E), Membership.mem s✝ a → LT.lt (l a) s
    hl : LT.lt s (l x)
    y : E
    hy : Membership.mem s✝ y
    hxy : LE.le (l x) (l y)
    ⊢ False
  -/
  exact ((hxy.trans_lt (hlA y hy)).trans hl).not_le le_rfl
  /-
    🎉 no goals
  -/

@[deprecated (since := "2024-11-12")] alias iInter_halfspaces_eq := iInter_halfSpaces_eq


/--Real linear extension of continuous extension of `LinearMap.extendTo𝕜'` -/
noncomputable def extendTo𝕜'ₗ [ContinuousConstSMul 𝕜 E]: (E →L[ℝ] ℝ) →ₗ[ℝ] (E →L[𝕜] 𝕜) :=
  letI to𝕜 (fr : (E →L[ℝ] ℝ)) : (E →L[𝕜] 𝕜) :=
    { toLinearMap := LinearMap.extendTo𝕜' fr
                                                                                       /-
                                                                                         𝕜 : Type u_1
                                                                                         E : Type u_2
                                                                                         inst✝⁶ : TopologicalSpace E
                                                                                         inst✝⁵ : AddCommGroup E
                                                                                         inst✝⁴ : Module Real E
                                                                                         s t : Set E
                                                                                         x y : E
                                                                                         inst✝³ : RCLike 𝕜
                                                                                         inst✝² : Module 𝕜 E
                                                                                         inst✝¹ : IsScalarTower Real 𝕜 E
                                                                                         inst✝ : ContinuousConstSMul 𝕜 E
                                                                                         fr : ContinuousLinearMap (RingHom.id Real) E Real
                                                                                         ⊢ Continuous fun x => HSub.hSub (↑(fr x)) (HMul.hMul RCLike.I ↑(fr (HSMul.hSMu …
                                                                                       -/
      cont := show Continuous fun x ↦ (fr x : 𝕜) - (I : 𝕜) * (fr ((I : 𝕜) • x) : 𝕜) by fun_prop }
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/
  have h fr x : to𝕜 fr x = ((fr x : 𝕜) - (I : 𝕜) * (fr ((I : 𝕜) • x) : 𝕜)) := rfl
  { toFun := to𝕜
                   /-
                     𝕜 : Type u_1
                     E : Type u_2
                     inst✝⁶ : TopologicalSpace E
                     inst✝⁵ : AddCommGroup E
                     inst✝⁴ : Module Real E
                     s t : Set E
                     x y : E
                     inst✝³ : RCLike 𝕜
                     inst✝² : Module 𝕜 E
                     inst✝¹ : IsScalarTower Real 𝕜 E
                     inst✝ : ContinuousConstSMul 𝕜 E
                     to𝕜 : ContinuousLinearMap (RingHom.id Real) E Real → ContinuousLinearMap (Ring …
                     h : ∀ (fr : ContinuousLinearMap (RingHom.id Real) E Real) (x : E), Eq ((to𝕜 fr …
                     ⊢ ∀ (x y : ContinuousLinearMap (RingHom.id Real) E Real), Eq (to𝕜 (HAdd.hAdd x …
                   -/
    map_add' := by intros; ext; simp [h]; ring
                                          /-
                                            🎉 no goals
                                          -/
                    /-
                      𝕜 : Type u_1
                      E : Type u_2
                      inst✝⁶ : TopologicalSpace E
                      inst✝⁵ : AddCommGroup E
                      inst✝⁴ : Module Real E
                      s t : Set E
                      x y : E
                      inst✝³ : RCLike 𝕜
                      inst✝² : Module 𝕜 E
                      inst✝¹ : IsScalarTower Real 𝕜 E
                      inst✝ : ContinuousConstSMul 𝕜 E
                      to𝕜 : ContinuousLinearMap (RingHom.id Real) E Real → ContinuousLinearMap (Ring …
                      h : ∀ (fr : ContinuousLinearMap (RingHom.id Real) E Real) (x : E), Eq ((to𝕜 fr …
                      ⊢ ∀ (m : Real) (x : ContinuousLinearMap (RingHom.id Real) E Real), Eq ({ toFun …
                    -/
    map_smul' := by intros; ext; simp [h, real_smul_eq_coe_mul]; ring }
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[simp]
lemma re_extendTo𝕜'ₗ [ContinuousConstSMul 𝕜 E] (g : E →L[ℝ] ℝ) (x : E) : re ((extendTo𝕜'ₗ g) x : 𝕜)
    = g x := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁶ : TopologicalSpace E
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : Module Real E
    inst✝³ : RCLike 𝕜
    inst✝² : Module 𝕜 E
    inst✝¹ : IsScalarTower Real 𝕜 E
    inst✝ : ContinuousConstSMul 𝕜 E
    g : ContinuousLinearMap (RingHom.id Real) E Real
    x : E
    ⊢ Eq (RCLike.re ((RCLike.extendTo𝕜'ₗ g) x)) (g x)
  -/
  have h g (x : E) : extendTo𝕜'ₗ g x = ((g x : 𝕜) - (I : 𝕜) * (g ((I : 𝕜) • x) : 𝕜)) := rfl
  simp only [h , map_sub, ofReal_re, mul_re, I_re, zero_mul, ofReal_im, mul_zero,
    sub_self, sub_zero]


theorem separate_convex_open_set {s : Set E}
    (hs₀ : (0 : E) ∈ s) (hs₁ : Convex ℝ s) (hs₂ : IsOpen s) {x₀ : E} (hx₀ : x₀ ∉ s) :
    ∃ f : E →L[𝕜] 𝕜, re (f x₀) = 1 ∧ ∀ x ∈ s, re (f x) < 1 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁷ : TopologicalSpace E
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module Real E
    inst✝⁴ : RCLike 𝕜
    inst✝³ : Module 𝕜 E
    inst✝² : IsScalarTower Real 𝕜 E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul 𝕜 E
    s : Set E
    hs₀ : Membership.mem s 0
    hs₁ : Convex Real s
    hs₂ : IsOpen s
    x₀ : E
    hx₀ : Not (Membership.mem s x₀)
    ⊢ Exists fun f => And (Eq (RCLike.re (f x₀)) 1) (∀ (x : E), Membership.mem s x …
  -/
  have := IsScalarTower.continuousSMul (M := ℝ) (α := E) 𝕜
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁷ : TopologicalSpace E
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module Real E
    inst✝⁴ : RCLike 𝕜
    inst✝³ : Module 𝕜 E
    inst✝² : IsScalarTower Real 𝕜 E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul 𝕜 E
    s : Set E
    hs₀ : Membership.mem s 0
    hs₁ : Convex Real s
    hs₂ : IsOpen s
    x₀ : E
    hx₀ : Not (Membership.mem s x₀)
    this : ContinuousSMul Real E
    ⊢ Exists fun f => And (Eq (RCLike.re (f x₀)) 1) (∀ (x : E), Membership.mem s x …
  -/
  obtain ⟨g, hg⟩ := _root_.separate_convex_open_set hs₀ hs₁ hs₂ hx₀
  /-
    case intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁷ : TopologicalSpace E
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module Real E
    inst✝⁴ : RCLike 𝕜
    inst✝³ : Module 𝕜 E
    inst✝² : IsScalarTower Real 𝕜 E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul 𝕜 E
    s : Set E
    hs₀ : Membership.mem s 0
    hs₁ : Convex Real s
    hs₂ : IsOpen s
    x₀ : E
    hx₀ : Not (Membership.mem s x₀)
    this : ContinuousSMul Real E
    g : ContinuousLinearMap (RingHom.id Real) E Real
    hg : And (Eq (g x₀) 1) (∀ (x : E), Membership.mem s x → LT.lt (g x) 1)
    ⊢ Exists fun f => And (Eq (RCLike.re (f x₀)) 1) (∀ (x : E), Membership.mem s x …
  -/
  use extendTo𝕜'ₗ g
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁷ : TopologicalSpace E
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module Real E
    inst✝⁴ : RCLike 𝕜
    inst✝³ : Module 𝕜 E
    inst✝² : IsScalarTower Real 𝕜 E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul 𝕜 E
    s : Set E
    hs₀ : Membership.mem s 0
    hs₁ : Convex Real s
    hs₂ : IsOpen s
    x₀ : E
    hx₀ : Not (Membership.mem s x₀)
    this : ContinuousSMul Real E
    g : ContinuousLinearMap (RingHom.id Real) E Real
    hg : And (Eq (g x₀) 1) (∀ (x : E), Membership.mem s x → LT.lt (g x) 1)
    ⊢ And (Eq (RCLike.re ((RCLike.extendTo𝕜'ₗ g) x₀)) 1) (∀ (x : E), Membership.me …
  -/
  simp only [re_extendTo𝕜'ₗ]
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁷ : TopologicalSpace E
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module Real E
    inst✝⁴ : RCLike 𝕜
    inst✝³ : Module 𝕜 E
    inst✝² : IsScalarTower Real 𝕜 E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul 𝕜 E
    s : Set E
    hs₀ : Membership.mem s 0
    hs₁ : Convex Real s
    hs₂ : IsOpen s
    x₀ : E
    hx₀ : Not (Membership.mem s x₀)
    this : ContinuousSMul Real E
    g : ContinuousLinearMap (RingHom.id Real) E Real
    hg : And (Eq (g x₀) 1) (∀ (x : E), Membership.mem s x → LT.lt (g x) 1)
    ⊢ And (Eq (g x₀) 1) (∀ (x : E), Membership.mem s x → LT.lt (g x) 1)
  -/
  exact hg
  /-
    🎉 no goals
  -/


theorem geometric_hahn_banach_open (hs₁ : Convex ℝ s) (hs₂ : IsOpen s) (ht : Convex ℝ t)
    (disj : Disjoint s t) : ∃ (f : E →L[𝕜] 𝕜) (u : ℝ), (∀ a ∈ s, re (f a) < u) ∧
    ∀ b ∈ t, u ≤ re (f b) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁷ : TopologicalSpace E
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module Real E
    s t : Set E
    inst✝⁴ : RCLike 𝕜
    inst✝³ : Module 𝕜 E
    inst✝² : IsScalarTower Real 𝕜 E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul 𝕜 E
    hs₁ : Convex Real s
    hs₂ : IsOpen s
    ht : Convex Real t
    disj : Disjoint s t
    ⊢ Exists fun f => Exists fun u => And (∀ (a : E), Membership.mem s a → LT.lt ( …
  -/
  have := IsScalarTower.continuousSMul (M := ℝ) (α := E) 𝕜
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁷ : TopologicalSpace E
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module Real E
    s t : Set E
    inst✝⁴ : RCLike 𝕜
    inst✝³ : Module 𝕜 E
    inst✝² : IsScalarTower Real 𝕜 E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul 𝕜 E
    hs₁ : Convex Real s
    hs₂ : IsOpen s
    ht : Convex Real t
    disj : Disjoint s t
    this : ContinuousSMul Real E
    ⊢ Exists fun f => Exists fun u => And (∀ (a : E), Membership.mem s a → LT.lt ( …
  -/
  obtain ⟨f, u, h⟩ := _root_.geometric_hahn_banach_open hs₁ hs₂ ht disj
  /-
    case intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁷ : TopologicalSpace E
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module Real E
    s t : Set E
    inst✝⁴ : RCLike 𝕜
    inst✝³ : Module 𝕜 E
    inst✝² : IsScalarTower Real 𝕜 E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul 𝕜 E
    hs₁ : Convex Real s
    hs₂ : IsOpen s
    ht : Convex Real t
    disj : Disjoint s t
    this : ContinuousSMul Real E
    f : ContinuousLinearMap (RingHom.id Real) E Real
    u : Real
    h : And (∀ (a : E), Membership.mem s a → LT.lt (f a) u) (∀ (b : E), Membership …
    ⊢ Exists fun f => Exists fun u => And (∀ (a : E), Membership.mem s a → LT.lt ( …
  -/
  use extendTo𝕜'ₗ f
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁷ : TopologicalSpace E
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module Real E
    s t : Set E
    inst✝⁴ : RCLike 𝕜
    inst✝³ : Module 𝕜 E
    inst✝² : IsScalarTower Real 𝕜 E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul 𝕜 E
    hs₁ : Convex Real s
    hs₂ : IsOpen s
    ht : Convex Real t
    disj : Disjoint s t
    this : ContinuousSMul Real E
    f : ContinuousLinearMap (RingHom.id Real) E Real
    u : Real
    h : And (∀ (a : E), Membership.mem s a → LT.lt (f a) u) (∀ (b : E), Membership …
    ⊢ Exists fun u => And (∀ (a : E), Membership.mem s a → LT.lt (RCLike.re ((RCLi …
  -/
  simp only [re_extendTo𝕜'ₗ]
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁷ : TopologicalSpace E
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module Real E
    s t : Set E
    inst✝⁴ : RCLike 𝕜
    inst✝³ : Module 𝕜 E
    inst✝² : IsScalarTower Real 𝕜 E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul 𝕜 E
    hs₁ : Convex Real s
    hs₂ : IsOpen s
    ht : Convex Real t
    disj : Disjoint s t
    this : ContinuousSMul Real E
    f : ContinuousLinearMap (RingHom.id Real) E Real
    u : Real
    h : And (∀ (a : E), Membership.mem s a → LT.lt (f a) u) (∀ (b : E), Membership …
    ⊢ Exists fun u => And (∀ (a : E), Membership.mem s a → LT.lt (f a) u) (∀ (b :  …
  -/
  exact Exists.intro u h
  /-
    🎉 no goals
  -/


theorem geometric_hahn_banach_open_point (hs₁ : Convex ℝ s) (hs₂ : IsOpen s) (disj : x ∉ s) :
    ∃ f : E →L[𝕜] 𝕜, ∀ a ∈ s, re (f a) < re (f x) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁷ : TopologicalSpace E
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module Real E
    s : Set E
    x : E
    inst✝⁴ : RCLike 𝕜
    inst✝³ : Module 𝕜 E
    inst✝² : IsScalarTower Real 𝕜 E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul 𝕜 E
    hs₁ : Convex Real s
    hs₂ : IsOpen s
    disj : Not (Membership.mem s x)
    ⊢ Exists fun f => ∀ (a : E), Membership.mem s a → LT.lt (RCLike.re (f a)) (RCL …
  -/
  have := IsScalarTower.continuousSMul (M := ℝ) (α := E) 𝕜
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁷ : TopologicalSpace E
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module Real E
    s : Set E
    x : E
    inst✝⁴ : RCLike 𝕜
    inst✝³ : Module 𝕜 E
    inst✝² : IsScalarTower Real 𝕜 E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul 𝕜 E
    hs₁ : Convex Real s
    hs₂ : IsOpen s
    disj : Not (Membership.mem s x)
    this : ContinuousSMul Real E
    ⊢ Exists fun f => ∀ (a : E), Membership.mem s a → LT.lt (RCLike.re (f a)) (RCL …
  -/
  obtain ⟨f, h⟩ := _root_.geometric_hahn_banach_open_point hs₁ hs₂ disj
  /-
    case intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁷ : TopologicalSpace E
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module Real E
    s : Set E
    x : E
    inst✝⁴ : RCLike 𝕜
    inst✝³ : Module 𝕜 E
    inst✝² : IsScalarTower Real 𝕜 E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul 𝕜 E
    hs₁ : Convex Real s
    hs₂ : IsOpen s
    disj : Not (Membership.mem s x)
    this : ContinuousSMul Real E
    f : ContinuousLinearMap (RingHom.id Real) E Real
    h : ∀ (a : E), Membership.mem s a → LT.lt (f a) (f x)
    ⊢ Exists fun f => ∀ (a : E), Membership.mem s a → LT.lt (RCLike.re (f a)) (RCL …
  -/
  use extendTo𝕜'ₗ f
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁷ : TopologicalSpace E
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module Real E
    s : Set E
    x : E
    inst✝⁴ : RCLike 𝕜
    inst✝³ : Module 𝕜 E
    inst✝² : IsScalarTower Real 𝕜 E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul 𝕜 E
    hs₁ : Convex Real s
    hs₂ : IsOpen s
    disj : Not (Membership.mem s x)
    this : ContinuousSMul Real E
    f : ContinuousLinearMap (RingHom.id Real) E Real
    h : ∀ (a : E), Membership.mem s a → LT.lt (f a) (f x)
    ⊢ ∀ (a : E), Membership.mem s a → LT.lt (RCLike.re ((RCLike.extendTo𝕜'ₗ f) a)) …
  -/
  simp only [re_extendTo𝕜'ₗ]
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁷ : TopologicalSpace E
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module Real E
    s : Set E
    x : E
    inst✝⁴ : RCLike 𝕜
    inst✝³ : Module 𝕜 E
    inst✝² : IsScalarTower Real 𝕜 E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul 𝕜 E
    hs₁ : Convex Real s
    hs₂ : IsOpen s
    disj : Not (Membership.mem s x)
    this : ContinuousSMul Real E
    f : ContinuousLinearMap (RingHom.id Real) E Real
    h : ∀ (a : E), Membership.mem s a → LT.lt (f a) (f x)
    ⊢ ∀ (a : E), Membership.mem s a → LT.lt (f a) (f x)
  -/
  exact fun a a_1 ↦ h a a_1
  /-
    🎉 no goals
  -/


theorem geometric_hahn_banach_point_open (ht₁ : Convex ℝ t) (ht₂ : IsOpen t) (disj : x ∉ t) :
    ∃ f : E →L[𝕜] 𝕜, ∀ b ∈ t, re (f x) < re (f b) :=
  let ⟨f, hf⟩ := geometric_hahn_banach_open_point ht₁ ht₂ disj
          /-
            𝕜 : Type u_1
            E : Type u_2
            inst✝⁷ : TopologicalSpace E
            inst✝⁶ : AddCommGroup E
            inst✝⁵ : Module Real E
            t : Set E
            x : E
            inst✝⁴ : RCLike 𝕜
            inst✝³ : Module 𝕜 E
            inst✝² : IsScalarTower Real 𝕜 E
            inst✝¹ : TopologicalAddGroup E
            inst✝ : ContinuousSMul 𝕜 E
            ht₁ : Convex Real t
            ht₂ : IsOpen t
            disj : Not (Membership.mem t x)
            f : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
            hf : ∀ (a : E), Membership.mem t a → LT.lt (RCLike.re (f a)) (RCLike.re (f x))
            ⊢ ∀ (b : E), Membership.mem t b → LT.lt (RCLike.re ((Neg.neg f) x)) (RCLike.re …
          -/
  ⟨-f, by simpa⟩
          /-
            🎉 no goals
          -/


theorem geometric_hahn_banach_open_open (hs₁ : Convex ℝ s) (hs₂ : IsOpen s)
    (ht₁ : Convex ℝ t) (ht₃ : IsOpen t) (disj : Disjoint s t) :
    ∃ (f : E →L[𝕜] 𝕜) (u : ℝ), (∀ a ∈ s, re (f a) < u) ∧ ∀ b ∈ t, u < re (f b) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁷ : TopologicalSpace E
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module Real E
    s t : Set E
    inst✝⁴ : RCLike 𝕜
    inst✝³ : Module 𝕜 E
    inst✝² : IsScalarTower Real 𝕜 E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul 𝕜 E
    hs₁ : Convex Real s
    hs₂ : IsOpen s
    ht₁ : Convex Real t
    ht₃ : IsOpen t
    disj : Disjoint s t
    ⊢ Exists fun f => Exists fun u => And (∀ (a : E), Membership.mem s a → LT.lt ( …
  -/
  have := IsScalarTower.continuousSMul (M := ℝ) (α := E) 𝕜
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁷ : TopologicalSpace E
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module Real E
    s t : Set E
    inst✝⁴ : RCLike 𝕜
    inst✝³ : Module 𝕜 E
    inst✝² : IsScalarTower Real 𝕜 E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul 𝕜 E
    hs₁ : Convex Real s
    hs₂ : IsOpen s
    ht₁ : Convex Real t
    ht₃ : IsOpen t
    disj : Disjoint s t
    this : ContinuousSMul Real E
    ⊢ Exists fun f => Exists fun u => And (∀ (a : E), Membership.mem s a → LT.lt ( …
  -/
  obtain ⟨f, u, h⟩ := _root_.geometric_hahn_banach_open_open hs₁ hs₂ ht₁ ht₃ disj
  /-
    case intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁷ : TopologicalSpace E
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module Real E
    s t : Set E
    inst✝⁴ : RCLike 𝕜
    inst✝³ : Module 𝕜 E
    inst✝² : IsScalarTower Real 𝕜 E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul 𝕜 E
    hs₁ : Convex Real s
    hs₂ : IsOpen s
    ht₁ : Convex Real t
    ht₃ : IsOpen t
    disj : Disjoint s t
    this : ContinuousSMul Real E
    f : ContinuousLinearMap (RingHom.id Real) E Real
    u : Real
    h : And (∀ (a : E), Membership.mem s a → LT.lt (f a) u) (∀ (b : E), Membership …
    ⊢ Exists fun f => Exists fun u => And (∀ (a : E), Membership.mem s a → LT.lt ( …
  -/
  use extendTo𝕜'ₗ f
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁷ : TopologicalSpace E
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module Real E
    s t : Set E
    inst✝⁴ : RCLike 𝕜
    inst✝³ : Module 𝕜 E
    inst✝² : IsScalarTower Real 𝕜 E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul 𝕜 E
    hs₁ : Convex Real s
    hs₂ : IsOpen s
    ht₁ : Convex Real t
    ht₃ : IsOpen t
    disj : Disjoint s t
    this : ContinuousSMul Real E
    f : ContinuousLinearMap (RingHom.id Real) E Real
    u : Real
    h : And (∀ (a : E), Membership.mem s a → LT.lt (f a) u) (∀ (b : E), Membership …
    ⊢ Exists fun u => And (∀ (a : E), Membership.mem s a → LT.lt (RCLike.re ((RCLi …
  -/
  simp only [re_extendTo𝕜'ₗ]
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁷ : TopologicalSpace E
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module Real E
    s t : Set E
    inst✝⁴ : RCLike 𝕜
    inst✝³ : Module 𝕜 E
    inst✝² : IsScalarTower Real 𝕜 E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul 𝕜 E
    hs₁ : Convex Real s
    hs₂ : IsOpen s
    ht₁ : Convex Real t
    ht₃ : IsOpen t
    disj : Disjoint s t
    this : ContinuousSMul Real E
    f : ContinuousLinearMap (RingHom.id Real) E Real
    u : Real
    h : And (∀ (a : E), Membership.mem s a → LT.lt (f a) u) (∀ (b : E), Membership …
    ⊢ Exists fun u => And (∀ (a : E), Membership.mem s a → LT.lt (f a) u) (∀ (b :  …
  -/
  exact Exists.intro u h
  /-
    🎉 no goals
  -/


theorem geometric_hahn_banach_compact_closed (hs₁ : Convex ℝ s) (hs₂ : IsCompact s)
    (ht₁ : Convex ℝ t) (ht₂ : IsClosed t) (disj : Disjoint s t) :
    ∃ (f : E →L[𝕜] 𝕜) (u v : ℝ), (∀ a ∈ s, re (f a) < u) ∧ u < v ∧ ∀ b ∈ t, v < re (f b) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁸ : TopologicalSpace E
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : Module Real E
    s t : Set E
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : Module 𝕜 E
    inst✝³ : IsScalarTower Real 𝕜 E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : ContinuousSMul 𝕜 E
    inst✝ : LocallyConvexSpace Real E
    hs₁ : Convex Real s
    hs₂ : IsCompact s
    ht₁ : Convex Real t
    ht₂ : IsClosed t
    disj : Disjoint s t
    ⊢ Exists fun f => Exists fun u => Exists fun v => And (∀ (a : E), Membership.m …
  -/
  have := IsScalarTower.continuousSMul (M := ℝ) (α := E) 𝕜
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁸ : TopologicalSpace E
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : Module Real E
    s t : Set E
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : Module 𝕜 E
    inst✝³ : IsScalarTower Real 𝕜 E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : ContinuousSMul 𝕜 E
    inst✝ : LocallyConvexSpace Real E
    hs₁ : Convex Real s
    hs₂ : IsCompact s
    ht₁ : Convex Real t
    ht₂ : IsClosed t
    disj : Disjoint s t
    this : ContinuousSMul Real E
    ⊢ Exists fun f => Exists fun u => Exists fun v => And (∀ (a : E), Membership.m …
  -/
  obtain ⟨g, u, v, h1⟩ := _root_.geometric_hahn_banach_compact_closed hs₁ hs₂ ht₁ ht₂ disj
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁸ : TopologicalSpace E
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : Module Real E
    s t : Set E
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : Module 𝕜 E
    inst✝³ : IsScalarTower Real 𝕜 E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : ContinuousSMul 𝕜 E
    inst✝ : LocallyConvexSpace Real E
    hs₁ : Convex Real s
    hs₂ : IsCompact s
    ht₁ : Convex Real t
    ht₂ : IsClosed t
    disj : Disjoint s t
    this : ContinuousSMul Real E
    g : ContinuousLinearMap (RingHom.id Real) E Real
    u v : Real
    h1 : And (∀ (a : E), Membership.mem s a → LT.lt (g a) u) (And (LT.lt u v) (∀ ( …
    ⊢ Exists fun f => Exists fun u => Exists fun v => And (∀ (a : E), Membership.m …
  -/
  use extendTo𝕜'ₗ g
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁸ : TopologicalSpace E
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : Module Real E
    s t : Set E
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : Module 𝕜 E
    inst✝³ : IsScalarTower Real 𝕜 E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : ContinuousSMul 𝕜 E
    inst✝ : LocallyConvexSpace Real E
    hs₁ : Convex Real s
    hs₂ : IsCompact s
    ht₁ : Convex Real t
    ht₂ : IsClosed t
    disj : Disjoint s t
    this : ContinuousSMul Real E
    g : ContinuousLinearMap (RingHom.id Real) E Real
    u v : Real
    h1 : And (∀ (a : E), Membership.mem s a → LT.lt (g a) u) (And (LT.lt u v) (∀ ( …
    ⊢ Exists fun u => Exists fun v => And (∀ (a : E), Membership.mem s a → LT.lt ( …
  -/
  simp only [re_extendTo𝕜'ₗ, exists_and_left]
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁸ : TopologicalSpace E
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : Module Real E
    s t : Set E
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : Module 𝕜 E
    inst✝³ : IsScalarTower Real 𝕜 E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : ContinuousSMul 𝕜 E
    inst✝ : LocallyConvexSpace Real E
    hs₁ : Convex Real s
    hs₂ : IsCompact s
    ht₁ : Convex Real t
    ht₂ : IsClosed t
    disj : Disjoint s t
    this : ContinuousSMul Real E
    g : ContinuousLinearMap (RingHom.id Real) E Real
    u v : Real
    h1 : And (∀ (a : E), Membership.mem s a → LT.lt (g a) u) (And (LT.lt u v) (∀ ( …
    ⊢ Exists fun u => And (∀ (a : E), Membership.mem s a → LT.lt (g a) u) (Exists  …
  -/
  exact ⟨u, h1.1, v, h1.2⟩
  /-
    🎉 no goals
  -/


theorem geometric_hahn_banach_closed_compact (hs₁ : Convex ℝ s) (hs₂ : IsClosed s)
    (ht₁ : Convex ℝ t) (ht₂ : IsCompact t) (disj : Disjoint s t) :
    ∃ (f : E →L[𝕜] 𝕜) (u v : ℝ), (∀ a ∈ s, re (f a) < u) ∧ u < v ∧ ∀ b ∈ t, v < re (f b) :=
  let ⟨f, s, t, hs, st, ht⟩ := geometric_hahn_banach_compact_closed ht₁ ht₂ hs₁ hs₂ disj.symm
                  /-
                    𝕜 : Type u_1
                    E : Type u_2
                    inst✝⁸ : TopologicalSpace E
                    inst✝⁷ : AddCommGroup E
                    inst✝⁶ : Module Real E
                    s✝ t✝ : Set E
                    inst✝⁵ : RCLike 𝕜
                    inst✝⁴ : Module 𝕜 E
                    inst✝³ : IsScalarTower Real 𝕜 E
                    inst✝² : TopologicalAddGroup E
                    inst✝¹ : ContinuousSMul 𝕜 E
                    inst✝ : LocallyConvexSpace Real E
                    hs₁ : Convex Real s✝
                    hs₂ : IsClosed s✝
                    ht₁ : Convex Real t✝
                    ht₂ : IsCompact t✝
                    disj : Disjoint s✝ t✝
                    f : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
                    s t : Real
                    hs : ∀ (a : E), Membership.mem t✝ a → LT.lt (RCLike.re (f a)) s
                    st : LT.lt s t
                    ht : ∀ (b : E), Membership.mem s✝ b → LT.lt t (RCLike.re (f b))
                    ⊢ ∀ (a : E), Membership.mem s✝ a → LT.lt (RCLike.re ((Neg.neg f) a)) (Neg.neg t)
                  -/
                  /-
                    🎉 no goals
                  -/
                                     /-
                                       🎉 no goals
                                     -/
  ⟨-f, -t, -s, by simpa using ht, by simpa using st, by simpa using hs⟩
                                                        /-
                                                          🎉 no goals
                                                        -/


theorem geometric_hahn_banach_point_closed (ht₁ : Convex ℝ t) (ht₂ : IsClosed t)
    (disj : x ∉ t) : ∃ (f : E →L[𝕜] 𝕜) (u : ℝ), re (f x) < u ∧ ∀ b ∈ t, u < re (f b) :=
  let ⟨f, _u, v, ha, hst, hb⟩ :=
    geometric_hahn_banach_compact_closed (convex_singleton x) isCompact_singleton ht₁ ht₂
      (disjoint_singleton_left.2 disj)
  ⟨f, v, hst.trans' <| ha x <| mem_singleton _, hb⟩


theorem geometric_hahn_banach_closed_point (hs₁ : Convex ℝ s) (hs₂ : IsClosed s)
    (disj : x ∉ s) : ∃ (f : E →L[𝕜] 𝕜) (u : ℝ), (∀ a ∈ s, re (f a) < u) ∧ u < re (f x) :=
  let ⟨f, s, _t, ha, hst, hb⟩ :=
    geometric_hahn_banach_closed_compact hs₁ hs₂ (convex_singleton x) isCompact_singleton
      (disjoint_singleton_right.2 disj)
  ⟨f, s, ha, hst.trans <| hb x <| mem_singleton _⟩


theorem geometric_hahn_banach_point_point [T1Space E] (hxy : x ≠ y) :
    ∃ f : E →L[𝕜] 𝕜, re (f x) < re (f y) := by
  obtain ⟨f, s, t, hs, st, ht⟩ :=
    geometric_hahn_banach_compact_closed (𝕜 := 𝕜) (convex_singleton x) isCompact_singleton
      (convex_singleton y) isClosed_singleton (disjoint_singleton.2 hxy)
  /-
    case intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁹ : TopologicalSpace E
    inst✝⁸ : AddCommGroup E
    inst✝⁷ : Module Real E
    x y : E
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : Module 𝕜 E
    inst✝⁴ : IsScalarTower Real 𝕜 E
    inst✝³ : TopologicalAddGroup E
    inst✝² : ContinuousSMul 𝕜 E
    inst✝¹ : LocallyConvexSpace Real E
    inst✝ : T1Space E
    hxy : Ne x y
    f : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
    s t : Real
    hs : ∀ (a : E), Membership.mem (Singleton.singleton x) a → LT.lt (RCLike.re (f …
    st : LT.lt s t
    ht : ∀ (b : E), Membership.mem (Singleton.singleton y) b → LT.lt t (RCLike.re  …
    ⊢ Exists fun f => LT.lt (RCLike.re (f x)) (RCLike.re (f y))
  -/
  exact ⟨f, by linarith [hs x rfl, ht y rfl]⟩
  /-
    🎉 no goals
  -/


theorem iInter_halfSpaces_eq (hs₁ : Convex ℝ s) (hs₂ : IsClosed s) :
    ⋂ l : E →L[𝕜] 𝕜, { x | ∃ y ∈ s, re (l x) ≤ re (l y) } = s := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁸ : TopologicalSpace E
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : Module Real E
    s : Set E
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : Module 𝕜 E
    inst✝³ : IsScalarTower Real 𝕜 E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : ContinuousSMul 𝕜 E
    inst✝ : LocallyConvexSpace Real E
    hs₁ : Convex Real s
    hs₂ : IsClosed s
    ⊢ Eq (Set.iInter fun l => setOf fun x => Exists fun y => And (Membership.mem s …
  -/
  rw [Set.iInter_setOf]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁸ : TopologicalSpace E
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : Module Real E
    s : Set E
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : Module 𝕜 E
    inst✝³ : IsScalarTower Real 𝕜 E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : ContinuousSMul 𝕜 E
    inst✝ : LocallyConvexSpace Real E
    hs₁ : Convex Real s
    hs₂ : IsClosed s
    ⊢ Eq (setOf fun x => ∀ (i : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜), Exists fu …
  -/
  refine Set.Subset.antisymm (fun x hx => ?_) fun x hx l => ⟨x, hx, le_rfl⟩
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁸ : TopologicalSpace E
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : Module Real E
    s : Set E
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : Module 𝕜 E
    inst✝³ : IsScalarTower Real 𝕜 E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : ContinuousSMul 𝕜 E
    inst✝ : LocallyConvexSpace Real E
    hs₁ : Convex Real s
    hs₂ : IsClosed s
    x : E
    hx : Membership.mem (setOf fun x => ∀ (i : ContinuousLinearMap (RingHom.id 𝕜)  …
    ⊢ Membership.mem s x
  -/
  by_contra h
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁸ : TopologicalSpace E
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : Module Real E
    s : Set E
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : Module 𝕜 E
    inst✝³ : IsScalarTower Real 𝕜 E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : ContinuousSMul 𝕜 E
    inst✝ : LocallyConvexSpace Real E
    hs₁ : Convex Real s
    hs₂ : IsClosed s
    x : E
    hx : Membership.mem (setOf fun x => ∀ (i : ContinuousLinearMap (RingHom.id 𝕜)  …
    h : Not (Membership.mem s x)
    ⊢ False
  -/
  obtain ⟨l, s, hlA, hl⟩ := geometric_hahn_banach_closed_point (𝕜 := 𝕜) hs₁ hs₂ h
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁸ : TopologicalSpace E
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : Module Real E
    s✝ : Set E
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : Module 𝕜 E
    inst✝³ : IsScalarTower Real 𝕜 E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : ContinuousSMul 𝕜 E
    inst✝ : LocallyConvexSpace Real E
    hs₁ : Convex Real s✝
    hs₂ : IsClosed s✝
    x : E
    hx : Membership.mem (setOf fun x => ∀ (i : ContinuousLinearMap (RingHom.id 𝕜)  …
    h : Not (Membership.mem s✝ x)
    l : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
    s : Real
    hlA : ∀ (a : E), Membership.mem s✝ a → LT.lt (RCLike.re (l a)) s
    hl : LT.lt s (RCLike.re (l x))
    ⊢ False
  -/
  obtain ⟨y, hy, hxy⟩ := hx l
  /-
    case intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁸ : TopologicalSpace E
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : Module Real E
    s✝ : Set E
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : Module 𝕜 E
    inst✝³ : IsScalarTower Real 𝕜 E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : ContinuousSMul 𝕜 E
    inst✝ : LocallyConvexSpace Real E
    hs₁ : Convex Real s✝
    hs₂ : IsClosed s✝
    x : E
    hx : Membership.mem (setOf fun x => ∀ (i : ContinuousLinearMap (RingHom.id 𝕜)  …
    h : Not (Membership.mem s✝ x)
    l : ContinuousLinearMap (RingHom.id 𝕜) E 𝕜
    s : Real
    hlA : ∀ (a : E), Membership.mem s✝ a → LT.lt (RCLike.re (l a)) s
    hl : LT.lt s (RCLike.re (l x))
    y : E
    hy : Membership.mem s✝ y
    hxy : LE.le (RCLike.re (l x)) (RCLike.re (l y))
    ⊢ False
  -/
  exact ((hxy.trans_lt (hlA y hy)).trans hl).false
  /-
    🎉 no goals
  -/

