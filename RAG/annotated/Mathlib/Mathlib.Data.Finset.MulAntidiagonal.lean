@[to_additive]
theorem IsPWO.mul [OrderedCancelCommMonoid α] (hs : s.IsPWO) (ht : t.IsPWO) : IsPWO (s * t) := by
  /-
    α : Type u_1
    s t : Set α
    inst✝ : OrderedCancelCommMonoid α
    hs : s.IsPWO
    ht : t.IsPWO
    ⊢ (HMul.hMul s t).IsPWO
  -/
  rw [← image_mul_prod]
  /-
    α : Type u_1
    s t : Set α
    inst✝ : OrderedCancelCommMonoid α
    hs : s.IsPWO
    ht : t.IsPWO
    ⊢ (Set.image (fun x => HMul.hMul x.1 x.2) (SProd.sprod s t)).IsPWO
  -/
  exact (hs.prod ht).image_of_monotone (monotone_fst.mul' monotone_snd)
  /-
    🎉 no goals
  -/


@[to_additive]
theorem IsWF.mul (hs : s.IsWF) (ht : t.IsWF) : IsWF (s * t) :=
  (hs.isPWO.mul ht.isPWO).isWF


@[to_additive]
theorem IsWF.min_mul (hs : s.IsWF) (ht : t.IsWF) (hsn : s.Nonempty) (htn : t.Nonempty) :
    (hs.mul ht).min (hsn.mul htn) = hs.min hsn * ht.min htn := by
  /-
    α : Type u_1
    s t : Set α
    inst✝ : LinearOrderedCancelCommMonoid α
    hs : s.IsWF
    ht : t.IsWF
    hsn : s.Nonempty
    htn : t.Nonempty
    ⊢ Eq (⋯.min ⋯) (HMul.hMul (hs.min hsn) (ht.min htn))
  -/
  refine le_antisymm (IsWF.min_le _ _ (mem_mul.2 ⟨_, hs.min_mem _, _, ht.min_mem _, rfl⟩)) ?_
  /-
    α : Type u_1
    s t : Set α
    inst✝ : LinearOrderedCancelCommMonoid α
    hs : s.IsWF
    ht : t.IsWF
    hsn : s.Nonempty
    htn : t.Nonempty
    ⊢ LE.le (HMul.hMul (hs.min hsn) (ht.min htn)) (⋯.min ⋯)
  -/
  rw [IsWF.le_min_iff]
  /-
    α : Type u_1
    s t : Set α
    inst✝ : LinearOrderedCancelCommMonoid α
    hs : s.IsWF
    ht : t.IsWF
    hsn : s.Nonempty
    htn : t.Nonempty
    ⊢ ∀ (b : α), Membership.mem (HMul.hMul s t) b → LE.le (HMul.hMul (hs.min hsn)  …
  -/
  rintro _ ⟨x, hx, y, hy, rfl⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    s t : Set α
    inst✝ : LinearOrderedCancelCommMonoid α
    hs : s.IsWF
    ht : t.IsWF
    hsn : s.Nonempty
    htn : t.Nonempty
    x : α
    hx : Membership.mem s x
    y : α
    hy : Membership.mem t y
    ⊢ LE.le (HMul.hMul (hs.min hsn) (ht.min htn)) ((fun x1 x2 => HMul.hMul x1 x2)  …
  -/
  exact mul_le_mul' (hs.min_le _ hx) (ht.min_le _ hy)
  /-
    🎉 no goals
  -/


/-- `Finset.mulAntidiagonal hs ht a` is the set of all pairs of an element in `s` and an
element in `t` that multiply to `a`, but its construction requires proofs that `s` and `t` are
well-ordered. -/
@[to_additive "`Finset.addAntidiagonal hs ht a` is the set of all pairs of an element in
`s` and an element in `t` that add to `a`, but its construction requires proofs that `s` and `t` are
well-ordered."]
noncomputable def mulAntidiagonal : Finset (α × α) :=
  (Set.MulAntidiagonal.finite_of_isPWO hs ht a).toFinset


@[to_additive (attr := simp)]
theorem mem_mulAntidiagonal : x ∈ mulAntidiagonal hs ht a ↔ x.1 ∈ s ∧ x.2 ∈ t ∧ x.1 * x.2 = a := by
  /-
    α : Type u_1
    inst✝ : OrderedCancelCommMonoid α
    s t : Set α
    hs : s.IsPWO
    ht : t.IsPWO
    a : α
    x : Prod α α
    ⊢ Iff (Membership.mem (Finset.mulAntidiagonal hs ht a) x) (And (Membership.mem …
  -/
  simp only [mulAntidiagonal, Set.Finite.mem_toFinset, Set.mem_mulAntidiagonal]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mulAntidiagonal_mono_left (h : u ⊆ s) : mulAntidiagonal hu ht a ⊆ mulAntidiagonal hs ht a :=
  Set.Finite.toFinset_mono <| Set.mulAntidiagonal_mono_left h


@[to_additive]
theorem mulAntidiagonal_mono_right (h : u ⊆ t) :
    mulAntidiagonal hs hu a ⊆ mulAntidiagonal hs ht a :=
  Set.Finite.toFinset_mono <| Set.mulAntidiagonal_mono_right h

-- Porting note: removed `(attr := simp)`. simp can prove this.

@[to_additive]
theorem swap_mem_mulAntidiagonal :
    x.swap ∈ Finset.mulAntidiagonal hs ht a ↔ x ∈ Finset.mulAntidiagonal ht hs a := by
  simp only [mem_mulAntidiagonal, Prod.fst_swap, Prod.snd_swap, Set.swap_mem_mulAntidiagonal_aux,
             Set.mem_mulAntidiagonal]


@[to_additive]
theorem support_mulAntidiagonal_subset_mul : { a | (mulAntidiagonal hs ht a).Nonempty } ⊆ s * t :=
  fun a ⟨b, hb⟩ => by
  /-
    α : Type u_1
    inst✝ : OrderedCancelCommMonoid α
    s t : Set α
    hs : s.IsPWO
    ht : t.IsPWO
    a : α
    x✝ : Membership.mem (setOf fun a => (Finset.mulAntidiagonal hs ht a).Nonempty) a
    b : Prod α α
    hb : Membership.mem (Finset.mulAntidiagonal hs ht a) b
    ⊢ Membership.mem (HMul.hMul s t) a
  -/
  rw [mem_mulAntidiagonal] at hb
  /-
    α : Type u_1
    inst✝ : OrderedCancelCommMonoid α
    s t : Set α
    hs : s.IsPWO
    ht : t.IsPWO
    a : α
    x✝ : Membership.mem (setOf fun a => (Finset.mulAntidiagonal hs ht a).Nonempty) a
    b : Prod α α
    hb : And (Membership.mem s b.1) (And (Membership.mem t b.2) (Eq (HMul.hMul b.1 …
    ⊢ Membership.mem (HMul.hMul s t) a
  -/
  exact ⟨b.1, hb.1, b.2, hb.2⟩
  /-
    🎉 no goals
  -/


@[to_additive]
theorem isPWO_support_mulAntidiagonal : { a | (mulAntidiagonal hs ht a).Nonempty }.IsPWO :=
  (hs.mul ht).mono support_mulAntidiagonal_subset_mul


@[to_additive]
theorem mulAntidiagonal_min_mul_min {α} [LinearOrderedCancelCommMonoid α] {s t : Set α}
    (hs : s.IsWF) (ht : t.IsWF) (hns : s.Nonempty) (hnt : t.Nonempty) :
    mulAntidiagonal hs.isPWO ht.isPWO (hs.min hns * ht.min hnt) = {(hs.min hns, ht.min hnt)} := by
  /-
    α : Type u_2
    inst✝ : LinearOrderedCancelCommMonoid α
    s t : Set α
    hs : s.IsWF
    ht : t.IsWF
    hns : s.Nonempty
    hnt : t.Nonempty
    ⊢ Eq (Finset.mulAntidiagonal ⋯ ⋯ (HMul.hMul (hs.min hns) (ht.min hnt))) (Singl …
  -/
  ext ⟨a, b⟩
  /-
    case h.mk
    α : Type u_2
    inst✝ : LinearOrderedCancelCommMonoid α
    s t : Set α
    hs : s.IsWF
    ht : t.IsWF
    hns : s.Nonempty
    hnt : t.Nonempty
    a b : α
    ⊢ Iff (Membership.mem (Finset.mulAntidiagonal ⋯ ⋯ (HMul.hMul (hs.min hns) (ht. …
  -/
  simp only [mem_mulAntidiagonal, mem_singleton, Prod.ext_iff]
  /-
    case h.mk
    α : Type u_2
    inst✝ : LinearOrderedCancelCommMonoid α
    s t : Set α
    hs : s.IsWF
    ht : t.IsWF
    hns : s.Nonempty
    hnt : t.Nonempty
    a b : α
    ⊢ Iff (And (Membership.mem s a) (And (Membership.mem t b) (Eq (HMul.hMul a b)  …
  -/
  constructor
    /-
      case h.mk.mp
      α : Type u_2
      inst✝ : LinearOrderedCancelCommMonoid α
      s t : Set α
      hs : s.IsWF
      ht : t.IsWF
      hns : s.Nonempty
      hnt : t.Nonempty
      a b : α
      ⊢ And (Membership.mem s a) (And (Membership.mem t b) (Eq (HMul.hMul a b) (HMul …
    -/
  · rintro ⟨has, hat, hst⟩
    obtain rfl :=
      (hs.min_le hns has).eq_of_not_lt fun hlt =>
        (mul_lt_mul_of_lt_of_le hlt <| ht.min_le hnt hat).ne' hst
    /-
      case h.mk.mp.intro.intro
      α : Type u_2
      inst✝ : LinearOrderedCancelCommMonoid α
      s t : Set α
      hs : s.IsWF
      ht : t.IsWF
      hns : s.Nonempty
      hnt : t.Nonempty
      b : α
      hat : Membership.mem t b
      has : Membership.mem s (hs.min hns)
      hst : Eq (HMul.hMul (hs.min hns) b) (HMul.hMul (hs.min hns) (ht.min hnt))
      ⊢ And (Eq (hs.min hns) (hs.min hns)) (Eq b (ht.min hnt))
    -/
    exact ⟨rfl, mul_left_cancel hst⟩
    /-
      🎉 no goals
    -/
    /-
      case h.mk.mpr
      α : Type u_2
      inst✝ : LinearOrderedCancelCommMonoid α
      s t : Set α
      hs : s.IsWF
      ht : t.IsWF
      hns : s.Nonempty
      hnt : t.Nonempty
      a b : α
      ⊢ And (Eq a (hs.min hns)) (Eq b (ht.min hnt)) → And (Membership.mem s a) (And  …
    -/
  · rintro ⟨rfl, rfl⟩
    /-
      case h.mk.mpr.intro
      α : Type u_2
      inst✝ : LinearOrderedCancelCommMonoid α
      s t : Set α
      hs : s.IsWF
      ht : t.IsWF
      hns : s.Nonempty
      hnt : t.Nonempty
      ⊢ And (Membership.mem s (hs.min hns)) (And (Membership.mem t (ht.min hnt)) (Eq …
    -/
    exact ⟨hs.min_mem _, ht.min_mem _, rfl⟩
    /-
      🎉 no goals
    -/


