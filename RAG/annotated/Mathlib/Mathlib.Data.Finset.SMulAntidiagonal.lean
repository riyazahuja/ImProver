@[to_additive]
theorem IsPWO.smul [PartialOrder G] [PartialOrder P] [SMul G P] [IsOrderedCancelSMul G P]
    {s : Set G} {t : Set P} (hs : s.IsPWO) (ht : t.IsPWO) : IsPWO (s • t) := by
  /-
    G : Type u_1
    P : Type u_2
    inst✝³ : PartialOrder G
    inst✝² : PartialOrder P
    inst✝¹ : SMul G P
    inst✝ : IsOrderedCancelSMul G P
    s : Set G
    t : Set P
    hs : s.IsPWO
    ht : t.IsPWO
    ⊢ (HSMul.hSMul s t).IsPWO
  -/
  rw [← @image_smul_prod]
  /-
    G : Type u_1
    P : Type u_2
    inst✝³ : PartialOrder G
    inst✝² : PartialOrder P
    inst✝¹ : SMul G P
    inst✝ : IsOrderedCancelSMul G P
    s : Set G
    t : Set P
    hs : s.IsPWO
    ht : t.IsPWO
    ⊢ (Set.image (fun x => HSMul.hSMul x.1 x.2) (SProd.sprod s t)).IsPWO
  -/
  exact (hs.prod ht).image_of_monotone (monotone_fst.smul monotone_snd)
  /-
    🎉 no goals
  -/


@[to_additive]
theorem IsWF.smul [LinearOrder G] [LinearOrder P] [SMul G P] [IsOrderedCancelSMul G P] {s : Set G}
    {t : Set P} (hs : s.IsWF) (ht : t.IsWF) : IsWF (s • t) :=
  (hs.isPWO.smul ht.isPWO).isWF


@[to_additive]
theorem IsWF.min_smul [LinearOrder G] [LinearOrder P] [SMul G P] [IsOrderedCancelSMul G P]
    {s : Set G} {t : Set P} (hs : s.IsWF) (ht : t.IsWF) (hsn : s.Nonempty) (htn : t.Nonempty) :
    (hs.smul ht).min (hsn.smul htn) = hs.min hsn • ht.min htn := by
  /-
    G : Type u_1
    P : Type u_2
    inst✝³ : LinearOrder G
    inst✝² : LinearOrder P
    inst✝¹ : SMul G P
    inst✝ : IsOrderedCancelSMul G P
    s : Set G
    t : Set P
    hs : s.IsWF
    ht : t.IsWF
    hsn : s.Nonempty
    htn : t.Nonempty
    ⊢ Eq (⋯.min ⋯) (HSMul.hSMul (hs.min hsn) (ht.min htn))
  -/
  refine le_antisymm (IsWF.min_le _ _ (mem_smul.2 ⟨_, hs.min_mem _, _, ht.min_mem _, rfl⟩)) ?_
  /-
    G : Type u_1
    P : Type u_2
    inst✝³ : LinearOrder G
    inst✝² : LinearOrder P
    inst✝¹ : SMul G P
    inst✝ : IsOrderedCancelSMul G P
    s : Set G
    t : Set P
    hs : s.IsWF
    ht : t.IsWF
    hsn : s.Nonempty
    htn : t.Nonempty
    ⊢ LE.le (HSMul.hSMul (hs.min hsn) (ht.min htn)) (⋯.min ⋯)
  -/
  rw [IsWF.le_min_iff]
  /-
    G : Type u_1
    P : Type u_2
    inst✝³ : LinearOrder G
    inst✝² : LinearOrder P
    inst✝¹ : SMul G P
    inst✝ : IsOrderedCancelSMul G P
    s : Set G
    t : Set P
    hs : s.IsWF
    ht : t.IsWF
    hsn : s.Nonempty
    htn : t.Nonempty
    ⊢ ∀ (b : P), Membership.mem (HSMul.hSMul s t) b → LE.le (HSMul.hSMul (hs.min h …
  -/
  rintro _ ⟨x, hx, y, hy, rfl⟩
  /-
    case intro.intro.intro.intro
    G : Type u_1
    P : Type u_2
    inst✝³ : LinearOrder G
    inst✝² : LinearOrder P
    inst✝¹ : SMul G P
    inst✝ : IsOrderedCancelSMul G P
    s : Set G
    t : Set P
    hs : s.IsWF
    ht : t.IsWF
    hsn : s.Nonempty
    htn : t.Nonempty
    x : G
    hx : Membership.mem s x
    y : P
    hy : Membership.mem t y
    ⊢ LE.le (HSMul.hSMul (hs.min hsn) (ht.min htn)) ((fun x1 x2 => HSMul.hSMul x1  …
  -/
  exact IsOrderedSMul.smul_le_smul (hs.min_le _ hx) (ht.min_le _ hy)
  /-
    🎉 no goals
  -/


/-- `Finset.SMulAntidiagonal hs ht a` is the set of all pairs of an element in `s` and an
element in `t` whose scalar multiplication yields `a`, but its construction requires proofs that `s`
and `t` are well-ordered. -/
@[to_additive "`Finset.VAddAntidiagonal hs ht a` is the set of all pairs of an element in `s` and an
element in `t` whose vector addition yields `a`, but its construction requires proofs that `s` and
`t` are well-ordered."]
noncomputable def SMulAntidiagonal [PartialOrder G] [PartialOrder P] [IsOrderedCancelSMul G P]
    {s : Set G} {t : Set P} (hs : s.IsPWO) (ht : t.IsPWO) (a : P) : Finset (G × P) :=
  (SMulAntidiagonal.finite_of_isPWO hs ht a).toFinset


@[to_additive (attr := simp)]
theorem mem_smulAntidiagonal :
    x ∈ SMulAntidiagonal hs ht a ↔ x.1 ∈ s ∧ x.2 ∈ t ∧ x.1 • x.2 = a := by
  /-
    G : Type u_1
    P : Type u_2
    inst✝³ : PartialOrder G
    inst✝² : PartialOrder P
    inst✝¹ : SMul G P
    inst✝ : IsOrderedCancelSMul G P
    s : Set G
    t : Set P
    hs : s.IsPWO
    ht : t.IsPWO
    a : P
    x : Prod G P
    ⊢ Iff (Membership.mem (Finset.SMulAntidiagonal hs ht a) x) (And (Membership.me …
  -/
  simp only [SMulAntidiagonal, Set.Finite.mem_toFinset]
  /-
    G : Type u_1
    P : Type u_2
    inst✝³ : PartialOrder G
    inst✝² : PartialOrder P
    inst✝¹ : SMul G P
    inst✝ : IsOrderedCancelSMul G P
    s : Set G
    t : Set P
    hs : s.IsPWO
    ht : t.IsPWO
    a : P
    x : Prod G P
    ⊢ Iff (Membership.mem (s.smulAntidiagonal t a) x) (And (Membership.mem s x.1)  …
  -/
  exact Set.mem_sep_iff
  /-
    🎉 no goals
  -/


@[to_additive]
theorem smulAntidiagonal_mono_left {a : P} {hs : s.IsPWO} {ht : t.IsPWO} (h : u ⊆ s) :
    SMulAntidiagonal hu ht a ⊆ SMulAntidiagonal hs ht a :=
  Set.Finite.toFinset_mono <| Set.smulAntidiagonal_mono_left h


@[to_additive]
theorem smulAntidiagonal_mono_right {a : P} {hs : s.IsPWO} {ht : t.IsPWO} (h : v ⊆ t) :
    SMulAntidiagonal hs hv a ⊆ SMulAntidiagonal hs ht a :=
  Set.Finite.toFinset_mono <| Set.smulAntidiagonal_mono_right h


@[to_additive]
theorem support_smulAntidiagonal_subset_smul {hs : s.IsPWO} {ht : t.IsPWO} :
    { a | (SMulAntidiagonal hs ht a).Nonempty } ⊆ (s • t) :=
  fun a ⟨b, hb⟩ => by
  /-
    G : Type u_1
    P : Type u_2
    inst✝³ : PartialOrder G
    inst✝² : PartialOrder P
    inst✝¹ : SMul G P
    inst✝ : IsOrderedCancelSMul G P
    s : Set G
    t : Set P
    hs : s.IsPWO
    ht : t.IsPWO
    a : P
    x✝ : Membership.mem (setOf fun a => (Finset.SMulAntidiagonal hs ht a).Nonempty …
    b : Prod G P
    hb : Membership.mem (Finset.SMulAntidiagonal hs ht a) b
    ⊢ Membership.mem (HSMul.hSMul s t) a
  -/
  rw [mem_smulAntidiagonal] at hb
  /-
    G : Type u_1
    P : Type u_2
    inst✝³ : PartialOrder G
    inst✝² : PartialOrder P
    inst✝¹ : SMul G P
    inst✝ : IsOrderedCancelSMul G P
    s : Set G
    t : Set P
    hs : s.IsPWO
    ht : t.IsPWO
    a : P
    x✝ : Membership.mem (setOf fun a => (Finset.SMulAntidiagonal hs ht a).Nonempty …
    b : Prod G P
    hb : And (Membership.mem s b.1) (And (Membership.mem t b.2) (Eq (HSMul.hSMul b …
    ⊢ Membership.mem (HSMul.hSMul s t) a
  -/
  rw [Set.mem_smul]
  /-
    G : Type u_1
    P : Type u_2
    inst✝³ : PartialOrder G
    inst✝² : PartialOrder P
    inst✝¹ : SMul G P
    inst✝ : IsOrderedCancelSMul G P
    s : Set G
    t : Set P
    hs : s.IsPWO
    ht : t.IsPWO
    a : P
    x✝ : Membership.mem (setOf fun a => (Finset.SMulAntidiagonal hs ht a).Nonempty …
    b : Prod G P
    hb : And (Membership.mem s b.1) (And (Membership.mem t b.2) (Eq (HSMul.hSMul b …
    ⊢ Exists fun x => And (Membership.mem s x) (Exists fun y => And (Membership.me …
  -/
  use b.1
  /-
    case h
    G : Type u_1
    P : Type u_2
    inst✝³ : PartialOrder G
    inst✝² : PartialOrder P
    inst✝¹ : SMul G P
    inst✝ : IsOrderedCancelSMul G P
    s : Set G
    t : Set P
    hs : s.IsPWO
    ht : t.IsPWO
    a : P
    x✝ : Membership.mem (setOf fun a => (Finset.SMulAntidiagonal hs ht a).Nonempty …
    b : Prod G P
    hb : And (Membership.mem s b.1) (And (Membership.mem t b.2) (Eq (HSMul.hSMul b …
    ⊢ And (Membership.mem s b.1) (Exists fun y => And (Membership.mem t y) (Eq (HS …
  -/
  refine { left := hb.1, right := ?_ }
  /-
    case h
    G : Type u_1
    P : Type u_2
    inst✝³ : PartialOrder G
    inst✝² : PartialOrder P
    inst✝¹ : SMul G P
    inst✝ : IsOrderedCancelSMul G P
    s : Set G
    t : Set P
    hs : s.IsPWO
    ht : t.IsPWO
    a : P
    x✝ : Membership.mem (setOf fun a => (Finset.SMulAntidiagonal hs ht a).Nonempty …
    b : Prod G P
    hb : And (Membership.mem s b.1) (And (Membership.mem t b.2) (Eq (HSMul.hSMul b …
    ⊢ Exists fun y => And (Membership.mem t y) (Eq (HSMul.hSMul b.1 y) a)
  -/
  use b.2
  /-
    case h
    G : Type u_1
    P : Type u_2
    inst✝³ : PartialOrder G
    inst✝² : PartialOrder P
    inst✝¹ : SMul G P
    inst✝ : IsOrderedCancelSMul G P
    s : Set G
    t : Set P
    hs : s.IsPWO
    ht : t.IsPWO
    a : P
    x✝ : Membership.mem (setOf fun a => (Finset.SMulAntidiagonal hs ht a).Nonempty …
    b : Prod G P
    hb : And (Membership.mem s b.1) (And (Membership.mem t b.2) (Eq (HSMul.hSMul b …
    ⊢ And (Membership.mem t b.2) (Eq (HSMul.hSMul b.1 b.2) a)
  -/
  exact { left := hb.2.1, right := hb.2.2 }
  /-
    🎉 no goals
  -/


@[to_additive]
theorem isPWO_support_smulAntidiagonal {hs : s.IsPWO} {ht : t.IsPWO} :
    { a | (SMulAntidiagonal hs ht a).Nonempty }.IsPWO :=
  (hs.smul ht).mono (support_smulAntidiagonal_subset_smul)


@[to_additive]
theorem smulAntidiagonal_min_smul_min [LinearOrder G] [LinearOrder P] [SMul G P]
    [IsOrderedCancelSMul G P] {s : Set G} {t : Set P} (hs : s.IsWF) (ht : t.IsWF) (hns : s.Nonempty)
    (hnt : t.Nonempty) :
    SMulAntidiagonal hs.isPWO ht.isPWO (hs.min hns • ht.min hnt) = {(hs.min hns, ht.min hnt)} := by
  /-
    G : Type u_1
    P : Type u_2
    inst✝³ : LinearOrder G
    inst✝² : LinearOrder P
    inst✝¹ : SMul G P
    inst✝ : IsOrderedCancelSMul G P
    s : Set G
    t : Set P
    hs : s.IsWF
    ht : t.IsWF
    hns : s.Nonempty
    hnt : t.Nonempty
    ⊢ Eq (Finset.SMulAntidiagonal ⋯ ⋯ (HSMul.hSMul (hs.min hns) (ht.min hnt))) (Si …
  -/
  ext ⟨a, b⟩
  /-
    case h.mk
    G : Type u_1
    P : Type u_2
    inst✝³ : LinearOrder G
    inst✝² : LinearOrder P
    inst✝¹ : SMul G P
    inst✝ : IsOrderedCancelSMul G P
    s : Set G
    t : Set P
    hs : s.IsWF
    ht : t.IsWF
    hns : s.Nonempty
    hnt : t.Nonempty
    a : G
    b : P
    ⊢ Iff (Membership.mem (Finset.SMulAntidiagonal ⋯ ⋯ (HSMul.hSMul (hs.min hns) ( …
  -/
  simp only [mem_smulAntidiagonal, mem_singleton, Prod.ext_iff]
  /-
    case h.mk
    G : Type u_1
    P : Type u_2
    inst✝³ : LinearOrder G
    inst✝² : LinearOrder P
    inst✝¹ : SMul G P
    inst✝ : IsOrderedCancelSMul G P
    s : Set G
    t : Set P
    hs : s.IsWF
    ht : t.IsWF
    hns : s.Nonempty
    hnt : t.Nonempty
    a : G
    b : P
    ⊢ Iff (And (Membership.mem s a) (And (Membership.mem t b) (Eq (HSMul.hSMul a b …
  -/
  constructor
    /-
      case h.mk.mp
      G : Type u_1
      P : Type u_2
      inst✝³ : LinearOrder G
      inst✝² : LinearOrder P
      inst✝¹ : SMul G P
      inst✝ : IsOrderedCancelSMul G P
      s : Set G
      t : Set P
      hs : s.IsWF
      ht : t.IsWF
      hns : s.Nonempty
      hnt : t.Nonempty
      a : G
      b : P
      ⊢ And (Membership.mem s a) (And (Membership.mem t b) (Eq (HSMul.hSMul a b) (HS …
    -/
  · rintro ⟨has, hat, hst⟩
    obtain rfl :=
      (hs.min_le hns has).eq_of_not_lt fun hlt =>
        (SMul.smul_lt_smul_of_lt_of_le hlt <| ht.min_le hnt hat).ne' hst
    /-
      case h.mk.mp.intro.intro
      G : Type u_1
      P : Type u_2
      inst✝³ : LinearOrder G
      inst✝² : LinearOrder P
      inst✝¹ : SMul G P
      inst✝ : IsOrderedCancelSMul G P
      s : Set G
      t : Set P
      hs : s.IsWF
      ht : t.IsWF
      hns : s.Nonempty
      hnt : t.Nonempty
      b : P
      hat : Membership.mem t b
      has : Membership.mem s (hs.min hns)
      hst : Eq (HSMul.hSMul (hs.min hns) b) (HSMul.hSMul (hs.min hns) (ht.min hnt))
      ⊢ And (Eq (hs.min hns) (hs.min hns)) (Eq b (ht.min hnt))
    -/
    exact ⟨rfl, IsCancelSMul.left_cancel _ _ _ hst⟩
    /-
      🎉 no goals
    -/
    /-
      case h.mk.mpr
      G : Type u_1
      P : Type u_2
      inst✝³ : LinearOrder G
      inst✝² : LinearOrder P
      inst✝¹ : SMul G P
      inst✝ : IsOrderedCancelSMul G P
      s : Set G
      t : Set P
      hs : s.IsWF
      ht : t.IsWF
      hns : s.Nonempty
      hnt : t.Nonempty
      a : G
      b : P
      ⊢ And (Eq a (hs.min hns)) (Eq b (ht.min hnt)) → And (Membership.mem s a) (And  …
    -/
  · rintro ⟨rfl, rfl⟩
    /-
      case h.mk.mpr.intro
      G : Type u_1
      P : Type u_2
      inst✝³ : LinearOrder G
      inst✝² : LinearOrder P
      inst✝¹ : SMul G P
      inst✝ : IsOrderedCancelSMul G P
      s : Set G
      t : Set P
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


