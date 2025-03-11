/-- A version of `Submodule.span_eq` for when the span is by a smaller ring. -/
@[simp]
theorem span_coe_eq_restrictScalars [Semiring S] [SMul S R] [Module S M] [IsScalarTower S R M] :
    span S (p : Set M) = p.restrictScalars S :=
  span_eq (p.restrictScalars S)


include σ₁₂ in
/-- A version of `Submodule.map_span_le` that does not require the `RingHomSurjective`
assumption. -/
theorem image_span_subset (f : F) (s : Set M) (N : Submodule R₂ M₂) :
    f '' span R s ⊆ N ↔ ∀ m ∈ s, f m ∈ N := image_subset_iff.trans <| span_le (p := N.comap f)


include σ₁₂ in
theorem image_span_subset_span (f : F) (s : Set M) : f '' span R s ⊆ span R₂ (f '' s) :=
  (image_span_subset f s _).2 fun x hx ↦ subset_span ⟨x, hx, rfl⟩


theorem map_span [RingHomSurjective σ₁₂] (f : F) (s : Set M) :
    (span R s).map f = span R₂ (f '' s) :=
  Eq.symm <| span_eq_of_le _ (Set.image_subset f subset_span) (image_span_subset_span f s)


alias _root_.LinearMap.map_span := Submodule.map_span


theorem map_span_le [RingHomSurjective σ₁₂] (f : F) (s : Set M) (N : Submodule R₂ M₂) :
    map f (span R s) ≤ N ↔ ∀ m ∈ s, f m ∈ N := image_span_subset f s N


alias _root_.LinearMap.map_span_le := Submodule.map_span_le

-- See also `span_preimage_eq` below.

theorem span_preimage_le (f : F) (s : Set M₂) :
    span R (f ⁻¹' s) ≤ (span R₂ s).comap f := by
  /-
    R : Type u_1
    R₂ : Type u_2
    M : Type u_4
    M₂ : Type u_5
    inst✝⁷ : Semiring R
    inst✝⁶ : AddCommMonoid M
    inst✝⁵ : Module R M
    inst✝⁴ : Semiring R₂
    σ₁₂ : RingHom R R₂
    inst✝³ : AddCommMonoid M₂
    inst✝² : Module R₂ M₂
    F : Type u_8
    inst✝¹ : FunLike F M M₂
    inst✝ : SemilinearMapClass F σ₁₂ M M₂
    f : F
    s : Set M₂
    ⊢ LE.le (Submodule.span R (Set.preimage (⇑f) s)) (Submodule.comap f (Submodule …
  -/
  rw [span_le, comap_coe]
  /-
    R : Type u_1
    R₂ : Type u_2
    M : Type u_4
    M₂ : Type u_5
    inst✝⁷ : Semiring R
    inst✝⁶ : AddCommMonoid M
    inst✝⁵ : Module R M
    inst✝⁴ : Semiring R₂
    σ₁₂ : RingHom R R₂
    inst✝³ : AddCommMonoid M₂
    inst✝² : Module R₂ M₂
    F : Type u_8
    inst✝¹ : FunLike F M M₂
    inst✝ : SemilinearMapClass F σ₁₂ M M₂
    f : F
    s : Set M₂
    ⊢ HasSubset.Subset (Set.preimage (⇑f) s) (Set.preimage ⇑f ↑(Submodule.span R₂  …
  -/
  exact preimage_mono subset_span
  /-
    🎉 no goals
  -/


alias _root_.LinearMap.span_preimage_le := Submodule.span_preimage_le


lemma linearMap_eq_iff_of_eq_span {V : Submodule R M} (f g : V →ₗ[R] N)
    {S : Set M} (hV : V = span R S) :
                                /-
                                  R : Type u_1
                                  R₂ : Type u_2
                                  K : Type u_3
                                  M : Type u_4
                                  M₂ : Type u_5
                                  V✝ : Type u_6
                                  S✝ : Type u_7
                                  inst✝⁹ : Semiring R
                                  inst✝⁸ : AddCommMonoid M
                                  inst✝⁷ : Module R M
                                  x : M
                                  p p' : Submodule R M
                                  inst✝⁶ : Semiring R₂
                                  σ₁₂ : RingHom R R₂
                                  inst✝⁵ : AddCommMonoid M₂
                                  inst✝⁴ : Module R₂ M₂
                                  F : Type u_8
                                  inst✝³ : FunLike F M M₂
                                  inst✝² : SemilinearMapClass F σ₁₂ M M₂
                                  s✝ t : Set M
                                  N : Type u_9
                                  inst✝¹ : AddCommMonoid N
                                  inst✝ : Module R N
                                  V : Submodule R M
                                  f g : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem V x) N
                                  S : Set M
                                  hV : Eq V (Submodule.span R S)
                                  s : ↑S
                                  ⊢ Membership.mem V ↑s
                                -/
    f = g ↔ ∀ (s : S), f ⟨s, by simpa only [hV] using subset_span (by simp)⟩ =
                                /-
                                  🎉 no goals
                                -/
               /-
                 R : Type u_1
                 R₂ : Type u_2
                 K : Type u_3
                 M : Type u_4
                 M₂ : Type u_5
                 V✝ : Type u_6
                 S✝ : Type u_7
                 inst✝⁹ : Semiring R
                 inst✝⁸ : AddCommMonoid M
                 inst✝⁷ : Module R M
                 x : M
                 p p' : Submodule R M
                 inst✝⁶ : Semiring R₂
                 σ₁₂ : RingHom R R₂
                 inst✝⁵ : AddCommMonoid M₂
                 inst✝⁴ : Module R₂ M₂
                 F : Type u_8
                 inst✝³ : FunLike F M M₂
                 inst✝² : SemilinearMapClass F σ₁₂ M M₂
                 s✝ t : Set M
                 N : Type u_9
                 inst✝¹ : AddCommMonoid N
                 inst✝ : Module R N
                 V : Submodule R M
                 f g : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem V x) N
                 S : Set M
                 hV : Eq V (Submodule.span R S)
                 s : ↑S
                 ⊢ Membership.mem V ↑s
               -/
      g ⟨s, by simpa only [hV] using subset_span (by simp)⟩ := by
               /-
                 🎉 no goals
               -/
  /-
    R : Type u_1
    M : Type u_4
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    N : Type u_9
    inst✝¹ : AddCommMonoid N
    inst✝ : Module R N
    V : Submodule R M
    f g : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem V x) N
    S : Set M
    hV : Eq V (Submodule.span R S)
    ⊢ Iff (Eq f g) (∀ (s : ↑S), Eq (f ⟨↑s, ⋯⟩) (g ⟨↑s, ⋯⟩))
  -/
  constructor
    /-
      case mp
      R : Type u_1
      M : Type u_4
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      N : Type u_9
      inst✝¹ : AddCommMonoid N
      inst✝ : Module R N
      V : Submodule R M
      f g : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem V x) N
      S : Set M
      hV : Eq V (Submodule.span R S)
      ⊢ Eq f g → ∀ (s : ↑S), Eq (f ⟨↑s, ⋯⟩) (g ⟨↑s, ⋯⟩)
    -/
  · rintro rfl _
    /-
      case mp
      R : Type u_1
      M : Type u_4
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      N : Type u_9
      inst✝¹ : AddCommMonoid N
      inst✝ : Module R N
      V : Submodule R M
      f : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem V x) N
      S : Set M
      hV : Eq V (Submodule.span R S)
      s✝ : ↑S
      ⊢ Eq (f ⟨↑s✝, ⋯⟩) (f ⟨↑s✝, ⋯⟩)
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      M : Type u_4
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      N : Type u_9
      inst✝¹ : AddCommMonoid N
      inst✝ : Module R N
      V : Submodule R M
      f g : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem V x) N
      S : Set M
      hV : Eq V (Submodule.span R S)
      ⊢ (∀ (s : ↑S), Eq (f ⟨↑s, ⋯⟩) (g ⟨↑s, ⋯⟩)) → Eq f g
    -/
  · intro h
    /-
      case mpr
      R : Type u_1
      M : Type u_4
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      N : Type u_9
      inst✝¹ : AddCommMonoid N
      inst✝ : Module R N
      V : Submodule R M
      f g : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem V x) N
      S : Set M
      hV : Eq V (Submodule.span R S)
      h : ∀ (s : ↑S), Eq (f ⟨↑s, ⋯⟩) (g ⟨↑s, ⋯⟩)
      ⊢ Eq f g
    -/
    subst hV
    suffices ∀ (x : M) (hx : x ∈ span R S), f ⟨x, hx⟩ = g ⟨x, hx⟩ by
      ext ⟨x, hx⟩
      exact this x hx
    /-
      case mpr
      R : Type u_1
      M : Type u_4
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      N : Type u_9
      inst✝¹ : AddCommMonoid N
      inst✝ : Module R N
      S : Set M
      f g : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem (Submodule.spa …
      h : ∀ (s : ↑S), Eq (f ⟨↑s, ⋯⟩) (g ⟨↑s, ⋯⟩)
      ⊢ ∀ (x : M) (hx : Membership.mem (Submodule.span R S) x), Eq (f ⟨x, hx⟩) (g ⟨x …
    -/
    intro x hx
    induction hx using span_induction with
    | mem x hx => exact h ⟨x, hx⟩
    | zero => erw [map_zero, map_zero]
    | add x y hx hy hx' hy' =>
        erw [f.map_add ⟨x, hx⟩ ⟨y, hy⟩, g.map_add ⟨x, hx⟩ ⟨y, hy⟩]
        rw [hx', hy']
    | smul a x hx hx' =>
        erw [f.map_smul a ⟨x, hx⟩, g.map_smul a ⟨x, hx⟩]
        rw [hx']


lemma linearMap_eq_iff_of_span_eq_top (f g : M →ₗ[R] N)
    {S : Set M} (hM : span R S = ⊤) :
    f = g ↔ ∀ (s : S), f s = g s := by
  convert linearMap_eq_iff_of_eq_span (f.comp (Submodule.subtype _))
    (g.comp (Submodule.subtype _)) hM.symm
  /-
    case h.e'_1.a
    R : Type u_1
    M : Type u_4
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    N : Type u_9
    inst✝¹ : AddCommMonoid N
    inst✝ : Module R N
    f g : LinearMap (RingHom.id R) M N
    S : Set M
    hM : Eq (Submodule.span R S) Top.top
    ⊢ Iff (Eq f g) (Eq (f.comp Top.top.subtype) (g.comp Top.top.subtype))
  -/
  constructor
    /-
      case h.e'_1.a.mp
      R : Type u_1
      M : Type u_4
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      N : Type u_9
      inst✝¹ : AddCommMonoid N
      inst✝ : Module R N
      f g : LinearMap (RingHom.id R) M N
      S : Set M
      hM : Eq (Submodule.span R S) Top.top
      ⊢ Eq f g → Eq (f.comp Top.top.subtype) (g.comp Top.top.subtype)
    -/
  · rintro rfl
    /-
      case h.e'_1.a.mp
      R : Type u_1
      M : Type u_4
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      N : Type u_9
      inst✝¹ : AddCommMonoid N
      inst✝ : Module R N
      f : LinearMap (RingHom.id R) M N
      S : Set M
      hM : Eq (Submodule.span R S) Top.top
      ⊢ Eq (f.comp Top.top.subtype) (f.comp Top.top.subtype)
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case h.e'_1.a.mpr
      R : Type u_1
      M : Type u_4
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      N : Type u_9
      inst✝¹ : AddCommMonoid N
      inst✝ : Module R N
      f g : LinearMap (RingHom.id R) M N
      S : Set M
      hM : Eq (Submodule.span R S) Top.top
      ⊢ Eq (f.comp Top.top.subtype) (g.comp Top.top.subtype) → Eq f g
    -/
  · intro h
    /-
      case h.e'_1.a.mpr
      R : Type u_1
      M : Type u_4
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      N : Type u_9
      inst✝¹ : AddCommMonoid N
      inst✝ : Module R N
      f g : LinearMap (RingHom.id R) M N
      S : Set M
      hM : Eq (Submodule.span R S) Top.top
      h : Eq (f.comp Top.top.subtype) (g.comp Top.top.subtype)
      ⊢ Eq f g
    -/
    ext x
    /-
      case h.e'_1.a.mpr.h
      R : Type u_1
      M : Type u_4
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      N : Type u_9
      inst✝¹ : AddCommMonoid N
      inst✝ : Module R N
      f g : LinearMap (RingHom.id R) M N
      S : Set M
      hM : Eq (Submodule.span R S) Top.top
      h : Eq (f.comp Top.top.subtype) (g.comp Top.top.subtype)
      x : M
      ⊢ Eq (f x) (g x)
    -/
    exact DFunLike.congr_fun h ⟨x, by simp⟩
    /-
      🎉 no goals
    -/


lemma linearMap_eq_zero_iff_of_span_eq_top (f : M →ₗ[R] N)
    {S : Set M} (hM : span R S = ⊤) :
    f = 0 ↔ ∀ (s : S), f s = 0 :=
  linearMap_eq_iff_of_span_eq_top f 0 hM


lemma linearMap_eq_zero_iff_of_eq_span {V : Submodule R M} (f : V →ₗ[R] N)
    {S : Set M} (hV : V = span R S) :
                                /-
                                  R : Type u_1
                                  R₂ : Type u_2
                                  K : Type u_3
                                  M : Type u_4
                                  M₂ : Type u_5
                                  V✝ : Type u_6
                                  S✝ : Type u_7
                                  inst✝⁹ : Semiring R
                                  inst✝⁸ : AddCommMonoid M
                                  inst✝⁷ : Module R M
                                  x : M
                                  p p' : Submodule R M
                                  inst✝⁶ : Semiring R₂
                                  σ₁₂ : RingHom R R₂
                                  inst✝⁵ : AddCommMonoid M₂
                                  inst✝⁴ : Module R₂ M₂
                                  F : Type u_8
                                  inst✝³ : FunLike F M M₂
                                  inst✝² : SemilinearMapClass F σ₁₂ M M₂
                                  s✝ t : Set M
                                  N : Type u_9
                                  inst✝¹ : AddCommMonoid N
                                  inst✝ : Module R N
                                  V : Submodule R M
                                  f : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem V x) N
                                  S : Set M
                                  hV : Eq V (Submodule.span R S)
                                  s : ↑S
                                  ⊢ Membership.mem V ↑s
                                -/
    f = 0 ↔ ∀ (s : S), f ⟨s, by simpa only [hV] using subset_span (by simp)⟩ = 0 :=
                                /-
                                  🎉 no goals
                                -/
  linearMap_eq_iff_of_eq_span f 0 hV


/-- See `Submodule.span_smul_eq` (in `RingTheory.Ideal.Operations`) for
`span R (r • s) = r • span R s` that holds for arbitrary `r` in a `CommSemiring`. -/
theorem span_smul_eq_of_isUnit (s : Set M) (r : R) (hr : IsUnit r) : span R (r • s) = span R s := by
  /-
    R : Type u_1
    M : Type u_4
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    s : Set M
    r : R
    hr : IsUnit r
    ⊢ Eq (Submodule.span R (HSMul.hSMul r s)) (Submodule.span R s)
  -/
  apply le_antisymm
    /-
      case a
      R : Type u_1
      M : Type u_4
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      s : Set M
      r : R
      hr : IsUnit r
      ⊢ LE.le (Submodule.span R (HSMul.hSMul r s)) (Submodule.span R s)
    -/
  · apply span_smul_le
    /-
      🎉 no goals
    -/
    /-
      case a
      R : Type u_1
      M : Type u_4
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      s : Set M
      r : R
      hr : IsUnit r
      ⊢ LE.le (Submodule.span R s) (Submodule.span R (HSMul.hSMul r s))
    -/
  · convert span_smul_le (r • s) ((hr.unit⁻¹ : _) : R)
    /-
      case h.e'_3.h.e'_6
      R : Type u_1
      M : Type u_4
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      s : Set M
      r : R
      hr : IsUnit r
      ⊢ Eq s (HSMul.hSMul (↑(Inv.inv hr.unit)) (HSMul.hSMul r s))
    -/
    rw [smul_smul]
    /-
      case h.e'_3.h.e'_6
      R : Type u_1
      M : Type u_4
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      s : Set M
      r : R
      hr : IsUnit r
      ⊢ Eq s (HSMul.hSMul (HMul.hMul (↑(Inv.inv hr.unit)) r) s)
    -/
    erw [hr.unit.inv_val]
    /-
      case h.e'_3.h.e'_6
      R : Type u_1
      M : Type u_4
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      s : Set M
      r : R
      hr : IsUnit r
      ⊢ Eq s (HSMul.hSMul 1 s)
    -/
    rw [one_smul]
    /-
      🎉 no goals
    -/


/-- We can regard `coe_iSup_of_chain` as the statement that `(↑) : (Submodule R M) → Set M` is
Scott continuous for the ω-complete partial order induced by the complete lattice structures. -/
theorem coe_scott_continuous :
    OmegaCompletePartialOrder.ωScottContinuous ((↑) : Submodule R M → Set M) :=
  OmegaCompletePartialOrder.ωScottContinuous.of_monotone_map_ωSup
    ⟨SetLike.coe_mono, coe_iSup_of_chain⟩


/-- If `R` is "smaller" ring than `S` then the span by `R` is smaller than the span by `S`. -/
theorem span_le_restrictScalars [Semiring S] [SMul R S] [Module S M] [IsScalarTower R S M] :
    span R s ≤ (span S s).restrictScalars R :=
  Submodule.span_le.2 Submodule.subset_span


/-- A version of `Submodule.span_le_restrictScalars` with coercions. -/
@[simp]
theorem span_subset_span [Semiring S] [SMul R S] [Module S M] [IsScalarTower R S M] :
    ↑(span R s) ⊆ (span S s : Set M) :=
  span_le_restrictScalars R S s


/-- Taking the span by a large ring of the span by the small ring is the same as taking the span
by just the large ring. -/
theorem span_span_of_tower [Semiring S] [SMul R S] [Module S M] [IsScalarTower R S M] :
    span S (span R s : Set M) = span S s :=
  le_antisymm (span_le.2 <| span_subset_span R S s) (span_mono subset_span)


theorem span_singleton_eq_span_singleton {R M : Type*} [Ring R] [AddCommGroup M] [Module R M]
    [NoZeroSMulDivisors R M] {x y : M} : ((R ∙ x) = R ∙ y) ↔ ∃ z : Rˣ, z • x = y := by
  /-
    R : Type u_9
    M : Type u_10
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroSMulDivisors R M
    x y : M
    ⊢ Iff (Eq (Submodule.span R (Singleton.singleton x)) (Submodule.span R (Single …
  -/
  constructor
    /-
      case mp
      R : Type u_9
      M : Type u_10
      inst✝³ : Ring R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : NoZeroSMulDivisors R M
      x y : M
      ⊢ Eq (Submodule.span R (Singleton.singleton x)) (Submodule.span R (Singleton.s …
    -/
  · simp only [le_antisymm_iff, span_singleton_le_iff_mem, mem_span_singleton]
    /-
      case mp
      R : Type u_9
      M : Type u_10
      inst✝³ : Ring R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : NoZeroSMulDivisors R M
      x y : M
      ⊢ And (Exists fun a => Eq (HSMul.hSMul a y) x) (Exists fun a => Eq (HSMul.hSMu …
    -/
    rintro ⟨⟨a, rfl⟩, b, hb⟩
    /-
      case mp.intro.intro.intro
      R : Type u_9
      M : Type u_10
      inst✝³ : Ring R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : NoZeroSMulDivisors R M
      y : M
      a b : R
      hb : Eq (HSMul.hSMul b (HSMul.hSMul a y)) y
      ⊢ Exists fun z => Eq (HSMul.hSMul z (HSMul.hSMul a y)) y
    -/
    rcases eq_or_ne y 0 with rfl | hy; · simp
                                         /-
                                           🎉 no goals
                                         -/
    /-
      case mp.intro.intro.intro.inr
      R : Type u_9
      M : Type u_10
      inst✝³ : Ring R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : NoZeroSMulDivisors R M
      y : M
      a b : R
      hb : Eq (HSMul.hSMul b (HSMul.hSMul a y)) y
      hy : Ne y 0
      ⊢ Exists fun z => Eq (HSMul.hSMul z (HSMul.hSMul a y)) y
    -/
    refine ⟨⟨b, a, ?_, ?_⟩, hb⟩
      /-
        case mp.intro.intro.intro.inr.refine_1
        R : Type u_9
        M : Type u_10
        inst✝³ : Ring R
        inst✝² : AddCommGroup M
        inst✝¹ : Module R M
        inst✝ : NoZeroSMulDivisors R M
        y : M
        a b : R
        hb : Eq (HSMul.hSMul b (HSMul.hSMul a y)) y
        hy : Ne y 0
        ⊢ Eq (HMul.hMul b a) 1
      -/
    · apply smul_left_injective R hy
      /-
        case mp.intro.intro.intro.inr.refine_1.a
        R : Type u_9
        M : Type u_10
        inst✝³ : Ring R
        inst✝² : AddCommGroup M
        inst✝¹ : Module R M
        inst✝ : NoZeroSMulDivisors R M
        y : M
        a b : R
        hb : Eq (HSMul.hSMul b (HSMul.hSMul a y)) y
        hy : Ne y 0
        ⊢ Eq ((fun c => HSMul.hSMul c y) (HMul.hMul b a)) ((fun c => HSMul.hSMul c y) 1)
      -/
      simpa only [mul_smul, one_smul]
      /-
        🎉 no goals
      -/
      /-
        case mp.intro.intro.intro.inr.refine_2
        R : Type u_9
        M : Type u_10
        inst✝³ : Ring R
        inst✝² : AddCommGroup M
        inst✝¹ : Module R M
        inst✝ : NoZeroSMulDivisors R M
        y : M
        a b : R
        hb : Eq (HSMul.hSMul b (HSMul.hSMul a y)) y
        hy : Ne y 0
        ⊢ Eq (HMul.hMul a b) 1
      -/
    · rw [← hb] at hy
      /-
        case mp.intro.intro.intro.inr.refine_2
        R : Type u_9
        M : Type u_10
        inst✝³ : Ring R
        inst✝² : AddCommGroup M
        inst✝¹ : Module R M
        inst✝ : NoZeroSMulDivisors R M
        y : M
        a b : R
        hb : Eq (HSMul.hSMul b (HSMul.hSMul a y)) y
        hy : Ne (HSMul.hSMul b (HSMul.hSMul a y)) 0
        ⊢ Eq (HMul.hMul a b) 1
      -/
      apply smul_left_injective R (smul_ne_zero_iff.1 hy).2
      /-
        case mp.intro.intro.intro.inr.refine_2.a
        R : Type u_9
        M : Type u_10
        inst✝³ : Ring R
        inst✝² : AddCommGroup M
        inst✝¹ : Module R M
        inst✝ : NoZeroSMulDivisors R M
        y : M
        a b : R
        hb : Eq (HSMul.hSMul b (HSMul.hSMul a y)) y
        hy : Ne (HSMul.hSMul b (HSMul.hSMul a y)) 0
        ⊢ Eq ((fun c => HSMul.hSMul c (HSMul.hSMul a y)) (HMul.hMul a b)) ((fun c => H …
      -/
      simp only [mul_smul, one_smul, hb]
      /-
        🎉 no goals
      -/
    /-
      case mpr
      R : Type u_9
      M : Type u_10
      inst✝³ : Ring R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : NoZeroSMulDivisors R M
      x y : M
      ⊢ (Exists fun z => Eq (HSMul.hSMul z x) y) → Eq (Submodule.span R (Singleton.s …
    -/
  · rintro ⟨u, rfl⟩
    /-
      case mpr.intro
      R : Type u_9
      M : Type u_10
      inst✝³ : Ring R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : NoZeroSMulDivisors R M
      x : M
      u : Units R
      ⊢ Eq (Submodule.span R (Singleton.singleton x)) (Submodule.span R (Singleton.s …
    -/
    exact (span_singleton_group_smul_eq _ _ _).symm
    /-
      🎉 no goals
    -/

-- Should be `@[simp]` but doesn't fire due to https://github.com/leanprover/lean4/pull/3701.

theorem span_image [RingHomSurjective σ₁₂] (f : F) :
    span R₂ (f '' s) = map f (span R s) :=
  (map_span f s).symm


@[simp] -- Should be replaced with `Submodule.span_image` when https://github.com/leanprover/lean4/pull/3701 is fixed.
theorem span_image' [RingHomSurjective σ₁₂] (f : M →ₛₗ[σ₁₂] M₂) :
    span R₂ (f '' s) = map f (span R s) :=
  span_image _


theorem apply_mem_span_image_of_mem_span [RingHomSurjective σ₁₂] (f : F) {x : M}
    {s : Set M} (h : x ∈ Submodule.span R s) : f x ∈ Submodule.span R₂ (f '' s) := by
  /-
    R : Type u_1
    R₂ : Type u_2
    M : Type u_4
    M₂ : Type u_5
    inst✝⁸ : Semiring R
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : Module R M
    inst✝⁵ : Semiring R₂
    σ₁₂ : RingHom R R₂
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : Module R₂ M₂
    F : Type u_8
    inst✝² : FunLike F M M₂
    inst✝¹ : SemilinearMapClass F σ₁₂ M M₂
    inst✝ : RingHomSurjective σ₁₂
    f : F
    x : M
    s : Set M
    h : Membership.mem (Submodule.span R s) x
    ⊢ Membership.mem (Submodule.span R₂ (Set.image (⇑f) s)) (f x)
  -/
  rw [Submodule.span_image]
  /-
    R : Type u_1
    R₂ : Type u_2
    M : Type u_4
    M₂ : Type u_5
    inst✝⁸ : Semiring R
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : Module R M
    inst✝⁵ : Semiring R₂
    σ₁₂ : RingHom R R₂
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : Module R₂ M₂
    F : Type u_8
    inst✝² : FunLike F M M₂
    inst✝¹ : SemilinearMapClass F σ₁₂ M M₂
    inst✝ : RingHomSurjective σ₁₂
    f : F
    x : M
    s : Set M
    h : Membership.mem (Submodule.span R s) x
    ⊢ Membership.mem (Submodule.map f (Submodule.span R s)) (f x)
  -/
  exact Submodule.mem_map_of_mem h
  /-
    🎉 no goals
  -/


theorem apply_mem_span_image_iff_mem_span [RingHomSurjective σ₁₂] {f : F} {x : M}
    {s : Set M} (hf : Function.Injective f) :
    f x ∈ Submodule.span R₂ (f '' s) ↔ x ∈ Submodule.span R s := by
  /-
    R : Type u_1
    R₂ : Type u_2
    M : Type u_4
    M₂ : Type u_5
    inst✝⁸ : Semiring R
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : Module R M
    inst✝⁵ : Semiring R₂
    σ₁₂ : RingHom R R₂
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : Module R₂ M₂
    F : Type u_8
    inst✝² : FunLike F M M₂
    inst✝¹ : SemilinearMapClass F σ₁₂ M M₂
    inst✝ : RingHomSurjective σ₁₂
    f : F
    x : M
    s : Set M
    hf : Function.Injective ⇑f
    ⊢ Iff (Membership.mem (Submodule.span R₂ (Set.image (⇑f) s)) (f x)) (Membershi …
  -/
  rw [← Submodule.mem_comap, ← Submodule.map_span, Submodule.comap_map_eq_of_injective hf]
  /-
    🎉 no goals
  -/


@[simp]
theorem map_subtype_span_singleton {p : Submodule R M} (x : p) :
                                              /-
                                                R : Type u_1
                                                M : Type u_4
                                                inst✝² : Semiring R
                                                inst✝¹ : AddCommMonoid M
                                                inst✝ : Module R M
                                                p : Submodule R M
                                                x : Subtype fun x => Membership.mem p x
                                                ⊢ Eq (Submodule.map p.subtype (Submodule.span R (Singleton.singleton x))) (Sub …
                                              -/
    map p.subtype (R ∙ x) = R ∙ (x : M) := by simp [← span_image]
                                              /-
                                                🎉 no goals
                                              -/


/-- `f` is an explicit argument so we can `apply` this theorem and obtain `h` as a new goal. -/
theorem not_mem_span_of_apply_not_mem_span_image [RingHomSurjective σ₁₂] (f : F) {x : M}
    {s : Set M} (h : f x ∉ Submodule.span R₂ (f '' s)) : x ∉ Submodule.span R s :=
  h.imp (apply_mem_span_image_of_mem_span f)


theorem iSup_toAddSubmonoid {ι : Sort*} (p : ι → Submodule R M) :
    (⨆ i, p i).toAddSubmonoid = ⨆ i, (p i).toAddSubmonoid := by
  /-
    R : Type u_1
    M : Type u_4
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    ι : Sort u_9
    p : ι → Submodule R M
    ⊢ Eq (iSup fun i => p i).toAddSubmonoid (iSup fun i => (p i).toAddSubmonoid)
  -/
  refine le_antisymm (fun x => ?_) (iSup_le fun i => toAddSubmonoid_mono <| le_iSup _ i)
  /-
    R : Type u_1
    M : Type u_4
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    ι : Sort u_9
    p : ι → Submodule R M
    x : M
    ⊢ Membership.mem (iSup fun i => p i).toAddSubmonoid x → Membership.mem (iSup f …
  -/
  simp_rw [iSup_eq_span, AddSubmonoid.iSup_eq_closure, mem_toAddSubmonoid, coe_toAddSubmonoid]
  /-
    R : Type u_1
    M : Type u_4
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    ι : Sort u_9
    p : ι → Submodule R M
    x : M
    ⊢ Membership.mem (Submodule.span R (Set.iUnion fun i => ↑(p i))) x → Membershi …
  -/
  intro hx
  refine Submodule.span_induction (fun x hx => ?_) ?_ (fun x y _ _ hx hy => ?_)
    (fun r x _ hx => ?_) hx
    /-
      case refine_1
      R : Type u_1
      M : Type u_4
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      ι : Sort u_9
      p : ι → Submodule R M
      x✝ : M
      hx✝ : Membership.mem (Submodule.span R (Set.iUnion fun i => ↑(p i))) x✝
      x : M
      hx : Membership.mem (Set.iUnion fun i => ↑(p i)) x
      ⊢ Membership.mem (AddSubmonoid.closure (Set.iUnion fun i => ↑(p i))) x
    -/
  · exact AddSubmonoid.subset_closure hx
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      M : Type u_4
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      ι : Sort u_9
      p : ι → Submodule R M
      x : M
      hx : Membership.mem (Submodule.span R (Set.iUnion fun i => ↑(p i))) x
      ⊢ Membership.mem (AddSubmonoid.closure (Set.iUnion fun i => ↑(p i))) 0
    -/
  · exact AddSubmonoid.zero_mem _
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      R : Type u_1
      M : Type u_4
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      ι : Sort u_9
      p : ι → Submodule R M
      x✝² : M
      hx✝ : Membership.mem (Submodule.span R (Set.iUnion fun i => ↑(p i))) x✝²
      x y : M
      x✝¹ : Membership.mem (Submodule.span R (Set.iUnion fun i => ↑(p i))) x
      x✝ : Membership.mem (Submodule.span R (Set.iUnion fun i => ↑(p i))) y
      hx : Membership.mem (AddSubmonoid.closure (Set.iUnion fun i => ↑(p i))) x
      hy : Membership.mem (AddSubmonoid.closure (Set.iUnion fun i => ↑(p i))) y
      ⊢ Membership.mem (AddSubmonoid.closure (Set.iUnion fun i => ↑(p i))) (HAdd.hAd …
    -/
  · exact AddSubmonoid.add_mem _ hx hy
    /-
      🎉 no goals
    -/
    /-
      case refine_4
      R : Type u_1
      M : Type u_4
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      ι : Sort u_9
      p : ι → Submodule R M
      x✝¹ : M
      hx✝ : Membership.mem (Submodule.span R (Set.iUnion fun i => ↑(p i))) x✝¹
      r : R
      x : M
      x✝ : Membership.mem (Submodule.span R (Set.iUnion fun i => ↑(p i))) x
      hx : Membership.mem (AddSubmonoid.closure (Set.iUnion fun i => ↑(p i))) x
      ⊢ Membership.mem (AddSubmonoid.closure (Set.iUnion fun i => ↑(p i))) (HSMul.hS …
    -/
  · refine AddSubmonoid.closure_induction ?_ ?_ ?_ hx
      /-
        case refine_4.refine_1
        R : Type u_1
        M : Type u_4
        inst✝² : Semiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        ι : Sort u_9
        p : ι → Submodule R M
        x✝¹ : M
        hx✝ : Membership.mem (Submodule.span R (Set.iUnion fun i => ↑(p i))) x✝¹
        r : R
        x : M
        x✝ : Membership.mem (Submodule.span R (Set.iUnion fun i => ↑(p i))) x
        hx : Membership.mem (AddSubmonoid.closure (Set.iUnion fun i => ↑(p i))) x
        ⊢ ∀ (x : M), Membership.mem (Set.iUnion fun i => ↑(p i)) x → Membership.mem (A …
      -/
    · rintro x ⟨_, ⟨i, rfl⟩, hix : x ∈ p i⟩
      /-
        case refine_4.refine_1.intro.intro.intro
        R : Type u_1
        M : Type u_4
        inst✝² : Semiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        ι : Sort u_9
        p : ι → Submodule R M
        x✝² : M
        hx✝ : Membership.mem (Submodule.span R (Set.iUnion fun i => ↑(p i))) x✝²
        r : R
        x✝¹ : M
        x✝ : Membership.mem (Submodule.span R (Set.iUnion fun i => ↑(p i))) x✝¹
        hx : Membership.mem (AddSubmonoid.closure (Set.iUnion fun i => ↑(p i))) x✝¹
        x : M
        i : ι
        hix : Membership.mem (p i) x
        ⊢ Membership.mem (AddSubmonoid.closure (Set.iUnion fun i => ↑(p i))) (HSMul.hS …
      -/
      apply AddSubmonoid.subset_closure (Set.mem_iUnion.mpr ⟨i, _⟩)
      /-
        R : Type u_1
        M : Type u_4
        inst✝² : Semiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        ι : Sort u_9
        p : ι → Submodule R M
        x✝² : M
        hx✝ : Membership.mem (Submodule.span R (Set.iUnion fun i => ↑(p i))) x✝²
        r : R
        x✝¹ : M
        x✝ : Membership.mem (Submodule.span R (Set.iUnion fun i => ↑(p i))) x✝¹
        hx : Membership.mem (AddSubmonoid.closure (Set.iUnion fun i => ↑(p i))) x✝¹
        x : M
        i : ι
        hix : Membership.mem (p i) x
        ⊢ Membership.mem (↑(p i)) (HSMul.hSMul r x)
      -/
      exact smul_mem _ r hix
      /-
        🎉 no goals
      -/
      /-
        case refine_4.refine_2
        R : Type u_1
        M : Type u_4
        inst✝² : Semiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        ι : Sort u_9
        p : ι → Submodule R M
        x✝¹ : M
        hx✝ : Membership.mem (Submodule.span R (Set.iUnion fun i => ↑(p i))) x✝¹
        r : R
        x : M
        x✝ : Membership.mem (Submodule.span R (Set.iUnion fun i => ↑(p i))) x
        hx : Membership.mem (AddSubmonoid.closure (Set.iUnion fun i => ↑(p i))) x
        ⊢ Membership.mem (AddSubmonoid.closure (Set.iUnion fun i => ↑(p i))) (HSMul.hS …
      -/
    · rw [smul_zero]
      /-
        case refine_4.refine_2
        R : Type u_1
        M : Type u_4
        inst✝² : Semiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        ι : Sort u_9
        p : ι → Submodule R M
        x✝¹ : M
        hx✝ : Membership.mem (Submodule.span R (Set.iUnion fun i => ↑(p i))) x✝¹
        r : R
        x : M
        x✝ : Membership.mem (Submodule.span R (Set.iUnion fun i => ↑(p i))) x
        hx : Membership.mem (AddSubmonoid.closure (Set.iUnion fun i => ↑(p i))) x
        ⊢ Membership.mem (AddSubmonoid.closure (Set.iUnion fun i => ↑(p i))) 0
      -/
      exact AddSubmonoid.zero_mem _
      /-
        🎉 no goals
      -/
      /-
        case refine_4.refine_3
        R : Type u_1
        M : Type u_4
        inst✝² : Semiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        ι : Sort u_9
        p : ι → Submodule R M
        x✝¹ : M
        hx✝ : Membership.mem (Submodule.span R (Set.iUnion fun i => ↑(p i))) x✝¹
        r : R
        x : M
        x✝ : Membership.mem (Submodule.span R (Set.iUnion fun i => ↑(p i))) x
        hx : Membership.mem (AddSubmonoid.closure (Set.iUnion fun i => ↑(p i))) x
        ⊢ ∀ (x y : M), Membership.mem (AddSubmonoid.closure (Set.iUnion fun i => ↑(p i …
      -/
    · intro x y _ _ hx hy
      /-
        case refine_4.refine_3
        R : Type u_1
        M : Type u_4
        inst✝² : Semiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        ι : Sort u_9
        p : ι → Submodule R M
        x✝² : M
        hx✝² : Membership.mem (Submodule.span R (Set.iUnion fun i => ↑(p i))) x✝²
        r : R
        x✝¹ : M
        x✝ : Membership.mem (Submodule.span R (Set.iUnion fun i => ↑(p i))) x✝¹
        hx✝¹ : Membership.mem (AddSubmonoid.closure (Set.iUnion fun i => ↑(p i))) x✝¹
        x y : M
        hx✝ : Membership.mem (AddSubmonoid.closure (Set.iUnion fun i => ↑(p i))) x
        hy✝ : Membership.mem (AddSubmonoid.closure (Set.iUnion fun i => ↑(p i))) y
        hx : Membership.mem (AddSubmonoid.closure (Set.iUnion fun i => ↑(p i))) (HSMul …
        hy : Membership.mem (AddSubmonoid.closure (Set.iUnion fun i => ↑(p i))) (HSMul …
        ⊢ Membership.mem (AddSubmonoid.closure (Set.iUnion fun i => ↑(p i))) (HSMul.hS …
      -/
      rw [smul_add]
      /-
        case refine_4.refine_3
        R : Type u_1
        M : Type u_4
        inst✝² : Semiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        ι : Sort u_9
        p : ι → Submodule R M
        x✝² : M
        hx✝² : Membership.mem (Submodule.span R (Set.iUnion fun i => ↑(p i))) x✝²
        r : R
        x✝¹ : M
        x✝ : Membership.mem (Submodule.span R (Set.iUnion fun i => ↑(p i))) x✝¹
        hx✝¹ : Membership.mem (AddSubmonoid.closure (Set.iUnion fun i => ↑(p i))) x✝¹
        x y : M
        hx✝ : Membership.mem (AddSubmonoid.closure (Set.iUnion fun i => ↑(p i))) x
        hy✝ : Membership.mem (AddSubmonoid.closure (Set.iUnion fun i => ↑(p i))) y
        hx : Membership.mem (AddSubmonoid.closure (Set.iUnion fun i => ↑(p i))) (HSMul …
        hy : Membership.mem (AddSubmonoid.closure (Set.iUnion fun i => ↑(p i))) (HSMul …
        ⊢ Membership.mem (AddSubmonoid.closure (Set.iUnion fun i => ↑(p i))) (HAdd.hAd …
      -/
      exact AddSubmonoid.add_mem _ hx hy
      /-
        🎉 no goals
      -/


/-- An induction principle for elements of `⨆ i, p i`.
If `C` holds for `0` and all elements of `p i` for all `i`, and is preserved under addition,
then it holds for all elements of the supremum of `p`. -/
@[elab_as_elim]
theorem iSup_induction {ι : Sort*} (p : ι → Submodule R M) {C : M → Prop} {x : M}
    (hx : x ∈ ⨆ i, p i) (hp : ∀ (i), ∀ x ∈ p i, C x) (h0 : C 0)
    (hadd : ∀ x y, C x → C y → C (x + y)) : C x := by
  /-
    R : Type u_1
    M : Type u_4
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    ι : Sort u_9
    p : ι → Submodule R M
    C : M → Prop
    x : M
    hx : Membership.mem (iSup fun i => p i) x
    hp : ∀ (i : ι) (x : M), Membership.mem (p i) x → C x
    h0 : C 0
    hadd : ∀ (x y : M), C x → C y → C (HAdd.hAdd x y)
    ⊢ C x
  -/
  rw [← mem_toAddSubmonoid, iSup_toAddSubmonoid] at hx
  /-
    R : Type u_1
    M : Type u_4
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    ι : Sort u_9
    p : ι → Submodule R M
    C : M → Prop
    x : M
    hx : Membership.mem (iSup fun i => (p i).toAddSubmonoid) x
    hp : ∀ (i : ι) (x : M), Membership.mem (p i) x → C x
    h0 : C 0
    hadd : ∀ (x y : M), C x → C y → C (HAdd.hAdd x y)
    ⊢ C x
  -/
  exact AddSubmonoid.iSup_induction (x := x) _ hx hp h0 hadd
  /-
    🎉 no goals
  -/


/-- A dependent version of `submodule.iSup_induction`. -/
@[elab_as_elim]
theorem iSup_induction' {ι : Sort*} (p : ι → Submodule R M) {C : ∀ x, (x ∈ ⨆ i, p i) → Prop}
    (mem : ∀ (i) (x) (hx : x ∈ p i), C x (mem_iSup_of_mem i hx)) (zero : C 0 (zero_mem _))
    (add : ∀ x y hx hy, C x hx → C y hy → C (x + y) (add_mem ‹_› ‹_›)) {x : M}
    (hx : x ∈ ⨆ i, p i) : C x hx := by
  /-
    R : Type u_1
    M : Type u_4
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    ι : Sort u_9
    p : ι → Submodule R M
    C : (x : M) → Membership.mem (iSup fun i => p i) x → Prop
    mem : ∀ (i : ι) (x : M) (hx : Membership.mem (p i) x), C x ⋯
    zero : C 0 ⋯
    add : ∀ (x y : M) (hx : Membership.mem (iSup fun i => p i) x) (hy : Membership …
    x : M
    hx : Membership.mem (iSup fun i => p i) x
    ⊢ C x hx
  -/
  refine Exists.elim ?_ fun (hx : x ∈ ⨆ i, p i) (hc : C x hx) => hc
  refine iSup_induction p (C := fun x : M ↦ ∃ (hx : x ∈ ⨆ i, p i), C x hx) hx
    (fun i x hx => ?_) ?_ fun x y => ?_
    /-
      case refine_1
      R : Type u_1
      M : Type u_4
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      ι : Sort u_9
      p : ι → Submodule R M
      C : (x : M) → Membership.mem (iSup fun i => p i) x → Prop
      mem : ∀ (i : ι) (x : M) (hx : Membership.mem (p i) x), C x ⋯
      zero : C 0 ⋯
      add : ∀ (x y : M) (hx : Membership.mem (iSup fun i => p i) x) (hy : Membership …
      x✝ : M
      hx✝ : Membership.mem (iSup fun i => p i) x✝
      i : ι
      x : M
      hx : Membership.mem (p i) x
      ⊢ (fun x => Exists fun hx => C x hx) x
    -/
  · exact ⟨_, mem _ _ hx⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      M : Type u_4
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      ι : Sort u_9
      p : ι → Submodule R M
      C : (x : M) → Membership.mem (iSup fun i => p i) x → Prop
      mem : ∀ (i : ι) (x : M) (hx : Membership.mem (p i) x), C x ⋯
      zero : C 0 ⋯
      add : ∀ (x y : M) (hx : Membership.mem (iSup fun i => p i) x) (hy : Membership …
      x : M
      hx : Membership.mem (iSup fun i => p i) x
      ⊢ (fun x => Exists fun hx => C x hx) 0
    -/
  · exact ⟨_, zero⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      R : Type u_1
      M : Type u_4
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      ι : Sort u_9
      p : ι → Submodule R M
      C : (x : M) → Membership.mem (iSup fun i => p i) x → Prop
      mem : ∀ (i : ι) (x : M) (hx : Membership.mem (p i) x), C x ⋯
      zero : C 0 ⋯
      add : ∀ (x y : M) (hx : Membership.mem (iSup fun i => p i) x) (hy : Membership …
      x✝ : M
      hx : Membership.mem (iSup fun i => p i) x✝
      x y : M
      ⊢ (fun x => Exists fun hx => C x hx) x → (fun x => Exists fun hx => C x hx) y  …
    -/
  · rintro ⟨_, Cx⟩ ⟨_, Cy⟩
    /-
      case refine_3.intro.intro
      R : Type u_1
      M : Type u_4
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      ι : Sort u_9
      p : ι → Submodule R M
      C : (x : M) → Membership.mem (iSup fun i => p i) x → Prop
      mem : ∀ (i : ι) (x : M) (hx : Membership.mem (p i) x), C x ⋯
      zero : C 0 ⋯
      add : ∀ (x y : M) (hx : Membership.mem (iSup fun i => p i) x) (hy : Membership …
      x✝ : M
      hx : Membership.mem (iSup fun i => p i) x✝
      x y : M
      w✝¹ : Membership.mem (iSup fun i => p i) x
      Cx : C x w✝¹
      w✝ : Membership.mem (iSup fun i => p i) y
      Cy : C y w✝
      ⊢ Exists fun hx => C (HAdd.hAdd x y) hx
    -/
    exact ⟨_, add _ _ _ _ Cx Cy⟩
    /-
      🎉 no goals
    -/


theorem singleton_span_isCompactElement (x : M) :
    CompleteLattice.IsCompactElement (span R {x} : Submodule R M) := by
  /-
    R : Type u_1
    M : Type u_4
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    x : M
    ⊢ CompleteLattice.IsCompactElement (Submodule.span R (Singleton.singleton x))
  -/
  rw [CompleteLattice.isCompactElement_iff_le_of_directed_sSup_le]
  /-
    R : Type u_1
    M : Type u_4
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    x : M
    ⊢ ∀ (s : Set (Submodule R M)), s.Nonempty → DirectedOn (fun x1 x2 => LE.le x1  …
  -/
  intro d hemp hdir hsup
  /-
    R : Type u_1
    M : Type u_4
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    x : M
    d : Set (Submodule R M)
    hemp : d.Nonempty
    hdir : DirectedOn (fun x1 x2 => LE.le x1 x2) d
    hsup : LE.le (Submodule.span R (Singleton.singleton x)) (SupSet.sSup d)
    ⊢ Exists fun x_1 => And (Membership.mem d x_1) (LE.le (Submodule.span R (Singl …
  -/
  have : x ∈ (sSup d) := (SetLike.le_def.mp hsup) (mem_span_singleton_self x)
  /-
    R : Type u_1
    M : Type u_4
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    x : M
    d : Set (Submodule R M)
    hemp : d.Nonempty
    hdir : DirectedOn (fun x1 x2 => LE.le x1 x2) d
    hsup : LE.le (Submodule.span R (Singleton.singleton x)) (SupSet.sSup d)
    this : Membership.mem (SupSet.sSup d) x
    ⊢ Exists fun x_1 => And (Membership.mem d x_1) (LE.le (Submodule.span R (Singl …
  -/
  obtain ⟨y, ⟨hyd, hxy⟩⟩ := (mem_sSup_of_directed hemp hdir).mp this
  /-
    case intro.intro
    R : Type u_1
    M : Type u_4
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    x : M
    d : Set (Submodule R M)
    hemp : d.Nonempty
    hdir : DirectedOn (fun x1 x2 => LE.le x1 x2) d
    hsup : LE.le (Submodule.span R (Singleton.singleton x)) (SupSet.sSup d)
    this : Membership.mem (SupSet.sSup d) x
    y : Submodule R M
    hyd : Membership.mem d y
    hxy : Membership.mem y x
    ⊢ Exists fun x_1 => And (Membership.mem d x_1) (LE.le (Submodule.span R (Singl …
  -/
  exact ⟨y, ⟨hyd, by simpa only [span_le, singleton_subset_iff] ⟩⟩
  /-
    🎉 no goals
  -/


/-- The span of a finite subset is compact in the lattice of submodules. -/
theorem finset_span_isCompactElement (S : Finset M) :
    CompleteLattice.IsCompactElement (span R S : Submodule R M) := by
  /-
    R : Type u_1
    M : Type u_4
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    S : Finset M
    ⊢ CompleteLattice.IsCompactElement (Submodule.span R ↑S)
  -/
  rw [span_eq_iSup_of_singleton_spans]
  /-
    R : Type u_1
    M : Type u_4
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    S : Finset M
    ⊢ CompleteLattice.IsCompactElement (iSup fun x => iSup fun h => Submodule.span …
  -/
  simp only [Finset.mem_coe]
  /-
    R : Type u_1
    M : Type u_4
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    S : Finset M
    ⊢ CompleteLattice.IsCompactElement (iSup fun x => iSup fun x_1 => Submodule.sp …
  -/
  rw [← Finset.sup_eq_iSup]
  exact
    CompleteLattice.isCompactElement_finsetSup S fun x _ => singleton_span_isCompactElement x


/-- The span of a finite subset is compact in the lattice of submodules. -/
theorem finite_span_isCompactElement (S : Set M) (h : S.Finite) :
    CompleteLattice.IsCompactElement (span R S : Submodule R M) :=
  Finite.coe_toFinset h ▸ finset_span_isCompactElement h.toFinset


instance : IsCompactlyGenerated (Submodule R M) :=
  ⟨fun s =>
    ⟨(fun x => span R {x}) '' s,
      ⟨fun t ht => by
        /-
          R : Type u_1
          R₂ : Type u_2
          K : Type u_3
          M : Type u_4
          M₂ : Type u_5
          V : Type u_6
          S : Type u_7
          inst✝⁷ : Semiring R
          inst✝⁶ : AddCommMonoid M
          inst✝⁵ : Module R M
          x : M
          p p' : Submodule R M
          inst✝⁴ : Semiring R₂
          σ₁₂ : RingHom R R₂
          inst✝³ : AddCommMonoid M₂
          inst✝² : Module R₂ M₂
          F : Type u_8
          inst✝¹ : FunLike F M M₂
          inst✝ : SemilinearMapClass F σ₁₂ M M₂
          s✝ t✝ : Set M
          s t : Submodule R M
          ht : Membership.mem (Set.image (fun x => Submodule.span R (Singleton.singleton …
          ⊢ CompleteLattice.IsCompactElement t
        -/
        rcases (Set.mem_image _ _ _).1 ht with ⟨x, _, rfl⟩
        /-
          case intro.intro
          R : Type u_1
          R₂ : Type u_2
          K : Type u_3
          M : Type u_4
          M₂ : Type u_5
          V : Type u_6
          S : Type u_7
          inst✝⁷ : Semiring R
          inst✝⁶ : AddCommMonoid M
          inst✝⁵ : Module R M
          x✝ : M
          p p' : Submodule R M
          inst✝⁴ : Semiring R₂
          σ₁₂ : RingHom R R₂
          inst✝³ : AddCommMonoid M₂
          inst✝² : Module R₂ M₂
          F : Type u_8
          inst✝¹ : FunLike F M M₂
          inst✝ : SemilinearMapClass F σ₁₂ M M₂
          s✝ t : Set M
          s : Submodule R M
          x : M
          left✝ : Membership.mem (↑s) x
          ht : Membership.mem (Set.image (fun x => Submodule.span R (Singleton.singleton …
          ⊢ CompleteLattice.IsCompactElement (Submodule.span R (Singleton.singleton x))
        -/
        apply singleton_span_isCompactElement, by
        /-
          🎉 no goals
        -/
        /-
          R : Type u_1
          R₂ : Type u_2
          K : Type u_3
          M : Type u_4
          M₂ : Type u_5
          V : Type u_6
          S : Type u_7
          inst✝⁷ : Semiring R
          inst✝⁶ : AddCommMonoid M
          inst✝⁵ : Module R M
          x : M
          p p' : Submodule R M
          inst✝⁴ : Semiring R₂
          σ₁₂ : RingHom R R₂
          inst✝³ : AddCommMonoid M₂
          inst✝² : Module R₂ M₂
          F : Type u_8
          inst✝¹ : FunLike F M M₂
          inst✝ : SemilinearMapClass F σ₁₂ M M₂
          s✝ t : Set M
          s : Submodule R M
          ⊢ Eq (SupSet.sSup (Set.image (fun x => Submodule.span R (Singleton.singleton x …
        -/
        rw [sSup_eq_iSup, iSup_image, ← span_eq_iSup_of_singleton_spans, span_eq]⟩⟩⟩
        /-
          🎉 no goals
        -/


/-- The product of two submodules is a submodule. -/
def prod : Submodule R (M × M') :=
  { p.toAddSubmonoid.prod q₁.toAddSubmonoid with
    carrier := p ×ˢ q₁
                    /-
                      R : Type u_1
                      R₂ : Type u_2
                      K : Type u_3
                      M : Type u_4
                      M₂ : Type u_5
                      V : Type u_6
                      S : Type u_7
                      inst✝⁹ : Semiring R
                      inst✝⁸ : AddCommMonoid M
                      inst✝⁷ : Module R M
                      x : M
                      p p' : Submodule R M
                      inst✝⁶ : Semiring R₂
                      σ₁₂ : RingHom R R₂
                      inst✝⁵ : AddCommMonoid M₂
                      inst✝⁴ : Module R₂ M₂
                      F : Type u_8
                      inst✝³ : FunLike F M M₂
                      inst✝² : SemilinearMapClass F σ₁₂ M M₂
                      s t : Set M
                      M' : Type u_9
                      inst✝¹ : AddCommMonoid M'
                      inst✝ : Module R M'
                      q₁ q₁' : Submodule R M'
                      ⊢ ∀ (c : R) {x : Prod M M'}, Membership.mem { carrier := SProd.sprod ↑p ↑q₁, a …
                    -/
    smul_mem' := by rintro a ⟨x, y⟩ ⟨hx, hy⟩; exact ⟨smul_mem _ a hx, smul_mem _ a hy⟩ }
                                              /-
                                                🎉 no goals
                                              -/


@[simp]
theorem prod_coe : (prod p q₁ : Set (M × M')) = (p : Set M) ×ˢ (q₁ : Set M') :=
  rfl


@[simp]
theorem mem_prod {p : Submodule R M} {q : Submodule R M'} {x : M × M'} :
    x ∈ prod p q ↔ x.1 ∈ p ∧ x.2 ∈ q :=
  Set.mem_prod


theorem span_prod_le (s : Set M) (t : Set M') : span R (s ×ˢ t) ≤ prod (span R s) (span R t) :=
  span_le.2 <| Set.prod_mono subset_span subset_span


@[simp]
                                                               /-
                                                                 R : Type u_1
                                                                 M : Type u_4
                                                                 inst✝⁴ : Semiring R
                                                                 inst✝³ : AddCommMonoid M
                                                                 inst✝² : Module R M
                                                                 M' : Type u_9
                                                                 inst✝¹ : AddCommMonoid M'
                                                                 inst✝ : Module R M'
                                                                 ⊢ Eq (Top.top.prod Top.top) Top.top
                                                               -/
theorem prod_top : (prod ⊤ ⊤ : Submodule R (M × M')) = ⊤ := by ext; simp
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


@[simp]
                                                               /-
                                                                 R : Type u_1
                                                                 M : Type u_4
                                                                 inst✝⁴ : Semiring R
                                                                 inst✝³ : AddCommMonoid M
                                                                 inst✝² : Module R M
                                                                 M' : Type u_9
                                                                 inst✝¹ : AddCommMonoid M'
                                                                 inst✝ : Module R M'
                                                                 ⊢ Eq (Bot.bot.prod Bot.bot) Bot.bot
                                                               -/
theorem prod_bot : (prod ⊥ ⊥ : Submodule R (M × M')) = ⊥ := by ext ⟨x, y⟩; simp
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


theorem prod_mono {p p' : Submodule R M} {q q' : Submodule R M'} :
    p ≤ p' → q ≤ q' → prod p q ≤ prod p' q' :=
  Set.prod_mono


@[simp]
theorem prod_inf_prod : prod p q₁ ⊓ prod p' q₁' = prod (p ⊓ p') (q₁ ⊓ q₁') :=
  SetLike.coe_injective Set.prod_inter_prod


@[simp]
theorem prod_sup_prod : prod p q₁ ⊔ prod p' q₁' = prod (p ⊔ p') (q₁ ⊔ q₁') := by
  refine le_antisymm
    (sup_le (prod_mono le_sup_left le_sup_left) (prod_mono le_sup_right le_sup_right)) ?_
  /-
    R : Type u_1
    M : Type u_4
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    p p' : Submodule R M
    M' : Type u_9
    inst✝¹ : AddCommMonoid M'
    inst✝ : Module R M'
    q₁ q₁' : Submodule R M'
    ⊢ LE.le ((Max.max p p').prod (Max.max q₁ q₁')) (Max.max (p.prod q₁) (p'.prod q …
  -/
  simp only [SetLike.le_def, mem_prod, and_imp, Prod.forall]; intro xx yy hxx hyy
  /-
    R : Type u_1
    M : Type u_4
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    p p' : Submodule R M
    M' : Type u_9
    inst✝¹ : AddCommMonoid M'
    inst✝ : Module R M'
    q₁ q₁' : Submodule R M'
    xx : M
    yy : M'
    hxx : Membership.mem (Max.max p p') xx
    hyy : Membership.mem (Max.max q₁ q₁') yy
    ⊢ Membership.mem (Max.max (p.prod q₁) (p'.prod q₁')) { fst := xx, snd := yy }
  -/
  rcases mem_sup.1 hxx with ⟨x, hx, x', hx', rfl⟩
  /-
    case intro.intro.intro.intro
    R : Type u_1
    M : Type u_4
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    p p' : Submodule R M
    M' : Type u_9
    inst✝¹ : AddCommMonoid M'
    inst✝ : Module R M'
    q₁ q₁' : Submodule R M'
    yy : M'
    hyy : Membership.mem (Max.max q₁ q₁') yy
    x : M
    hx : Membership.mem p x
    x' : M
    hx' : Membership.mem p' x'
    hxx : Membership.mem (Max.max p p') (HAdd.hAdd x x')
    ⊢ Membership.mem (Max.max (p.prod q₁) (p'.prod q₁')) { fst := HAdd.hAdd x x',  …
  -/
  rcases mem_sup.1 hyy with ⟨y, hy, y', hy', rfl⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    R : Type u_1
    M : Type u_4
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    p p' : Submodule R M
    M' : Type u_9
    inst✝¹ : AddCommMonoid M'
    inst✝ : Module R M'
    q₁ q₁' : Submodule R M'
    x : M
    hx : Membership.mem p x
    x' : M
    hx' : Membership.mem p' x'
    hxx : Membership.mem (Max.max p p') (HAdd.hAdd x x')
    y : M'
    hy : Membership.mem q₁ y
    y' : M'
    hy' : Membership.mem q₁' y'
    hyy : Membership.mem (Max.max q₁ q₁') (HAdd.hAdd y y')
    ⊢ Membership.mem (Max.max (p.prod q₁) (p'.prod q₁')) { fst := HAdd.hAdd x x',  …
  -/
  exact mem_sup.2 ⟨(x, y), ⟨hx, hy⟩, (x', y'), ⟨hx', hy'⟩, rfl⟩
  /-
    🎉 no goals
  -/


@[simp]
theorem span_neg (s : Set M) : span R (-s) = span R s :=
  calc
                                                                  /-
                                                                    R : Type u_1
                                                                    M : Type u_4
                                                                    inst✝² : Ring R
                                                                    inst✝¹ : AddCommGroup M
                                                                    inst✝ : Module R M
                                                                    s : Set M
                                                                    ⊢ Eq (Submodule.span R (Neg.neg s)) (Submodule.span R (Set.image (⇑(Neg.neg Li …
                                                                  -/
    span R (-s) = span R ((-LinearMap.id : M →ₗ[R] M) '' s) := by simp
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
    _ = map (-LinearMap.id) (span R s) := (map_span (-LinearMap.id) _).symm
                       /-
                         R : Type u_1
                         M : Type u_4
                         inst✝² : Ring R
                         inst✝¹ : AddCommGroup M
                         inst✝ : Module R M
                         s : Set M
                         ⊢ Eq (Submodule.map (Neg.neg LinearMap.id) (Submodule.span R s)) (Submodule.sp …
                       -/
    _ = span R s := by simp
                       /-
                         🎉 no goals
                       -/


instance : IsModularLattice (Submodule R M) :=
  ⟨fun y z xz a ha => by
    /-
      R : Type u_1
      R₂ : Type u_2
      K : Type u_3
      M : Type u_4
      M₂ : Type u_5
      V : Type u_6
      S : Type u_7
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      x✝ y z : Submodule R M
      xz : LE.le x✝ z
      a : M
      ha : Membership.mem (Min.min (Max.max x✝ y) z) a
      ⊢ Membership.mem (Max.max x✝ (Min.min y z)) a
    -/
    rw [mem_inf, mem_sup] at ha
    /-
      R : Type u_1
      R₂ : Type u_2
      K : Type u_3
      M : Type u_4
      M₂ : Type u_5
      V : Type u_6
      S : Type u_7
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      x✝ y z : Submodule R M
      xz : LE.le x✝ z
      a : M
      ha : And (Exists fun y_1 => And (Membership.mem x✝ y_1) (Exists fun z => And ( …
      ⊢ Membership.mem (Max.max x✝ (Min.min y z)) a
    -/
    rcases ha with ⟨⟨b, hb, c, hc, rfl⟩, haz⟩
    /-
      case intro.intro.intro.intro.intro
      R : Type u_1
      R₂ : Type u_2
      K : Type u_3
      M : Type u_4
      M₂ : Type u_5
      V : Type u_6
      S : Type u_7
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      x✝ y z : Submodule R M
      xz : LE.le x✝ z
      b : M
      hb : Membership.mem x✝ b
      c : M
      hc : Membership.mem y c
      haz : Membership.mem z (HAdd.hAdd b c)
      ⊢ Membership.mem (Max.max x✝ (Min.min y z)) (HAdd.hAdd b c)
    -/
    rw [mem_sup]
    /-
      case intro.intro.intro.intro.intro
      R : Type u_1
      R₂ : Type u_2
      K : Type u_3
      M : Type u_4
      M₂ : Type u_5
      V : Type u_6
      S : Type u_7
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      x✝ y z : Submodule R M
      xz : LE.le x✝ z
      b : M
      hb : Membership.mem x✝ b
      c : M
      hc : Membership.mem y c
      haz : Membership.mem z (HAdd.hAdd b c)
      ⊢ Exists fun y_1 => And (Membership.mem x✝ y_1) (Exists fun z_1 => And (Member …
    -/
    refine ⟨b, hb, c, mem_inf.2 ⟨hc, ?_⟩, rfl⟩
    /-
      case intro.intro.intro.intro.intro
      R : Type u_1
      R₂ : Type u_2
      K : Type u_3
      M : Type u_4
      M₂ : Type u_5
      V : Type u_6
      S : Type u_7
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      x✝ y z : Submodule R M
      xz : LE.le x✝ z
      b : M
      hb : Membership.mem x✝ b
      c : M
      hc : Membership.mem y c
      haz : Membership.mem z (HAdd.hAdd b c)
      ⊢ Membership.mem z c
    -/
    rw [← add_sub_cancel_right c b, add_comm]
    /-
      case intro.intro.intro.intro.intro
      R : Type u_1
      R₂ : Type u_2
      K : Type u_3
      M : Type u_4
      M₂ : Type u_5
      V : Type u_6
      S : Type u_7
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      x✝ y z : Submodule R M
      xz : LE.le x✝ z
      b : M
      hb : Membership.mem x✝ b
      c : M
      hc : Membership.mem y c
      haz : Membership.mem z (HAdd.hAdd b c)
      ⊢ Membership.mem z (HSub.hSub (HAdd.hAdd b c) b)
    -/
    apply z.sub_mem haz (xz hb)⟩
    /-
      🎉 no goals
    -/


lemma isCompl_comap_subtype_of_isCompl_of_le {p q r : Submodule R M}
    (h₁ : IsCompl q r) (h₂ : q ≤ p) :
    IsCompl (q.comap p.subtype) (r.comap p.subtype) := by
  /-
    R : Type u_1
    M : Type u_4
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    p q r : Submodule R M
    h₁ : IsCompl q r
    h₂ : LE.le q p
    ⊢ IsCompl (Submodule.comap p.subtype q) (Submodule.comap p.subtype r)
  -/
  simpa [p.mapIic.isCompl_iff, Iic.isCompl_iff] using Iic.isCompl_inf_inf_of_isCompl_of_le h₁ h₂
  /-
    🎉 no goals
  -/


theorem comap_map_eq (f : F) (p : Submodule R M) : comap f (map f p) = p ⊔ LinearMap.ker f := by
  /-
    R : Type u_1
    R₂ : Type u_2
    M : Type u_4
    M₂ : Type u_5
    inst✝⁸ : Semiring R
    inst✝⁷ : Semiring R₂
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : AddCommGroup M₂
    inst✝³ : Module R₂ M₂
    τ₁₂ : RingHom R R₂
    inst✝² : RingHomSurjective τ₁₂
    F : Type u_8
    inst✝¹ : FunLike F M M₂
    inst✝ : SemilinearMapClass F τ₁₂ M M₂
    f : F
    p : Submodule R M
    ⊢ Eq (Submodule.comap f (Submodule.map f p)) (Max.max p (LinearMap.ker f))
  -/
  refine le_antisymm ?_ (sup_le (le_comap_map _ _) (comap_mono bot_le))
  /-
    R : Type u_1
    R₂ : Type u_2
    M : Type u_4
    M₂ : Type u_5
    inst✝⁸ : Semiring R
    inst✝⁷ : Semiring R₂
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : AddCommGroup M₂
    inst✝³ : Module R₂ M₂
    τ₁₂ : RingHom R R₂
    inst✝² : RingHomSurjective τ₁₂
    F : Type u_8
    inst✝¹ : FunLike F M M₂
    inst✝ : SemilinearMapClass F τ₁₂ M M₂
    f : F
    p : Submodule R M
    ⊢ LE.le (Submodule.comap f (Submodule.map f p)) (Max.max p (LinearMap.ker f))
  -/
  rintro x ⟨y, hy, e⟩
  /-
    case intro.intro
    R : Type u_1
    R₂ : Type u_2
    M : Type u_4
    M₂ : Type u_5
    inst✝⁸ : Semiring R
    inst✝⁷ : Semiring R₂
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : AddCommGroup M₂
    inst✝³ : Module R₂ M₂
    τ₁₂ : RingHom R R₂
    inst✝² : RingHomSurjective τ₁₂
    F : Type u_8
    inst✝¹ : FunLike F M M₂
    inst✝ : SemilinearMapClass F τ₁₂ M M₂
    f : F
    p : Submodule R M
    x y : M
    hy : Membership.mem (↑p) y
    e : Eq (f y) (f x)
    ⊢ Membership.mem (Max.max p (LinearMap.ker f)) x
  -/
  exact mem_sup.2 ⟨y, hy, x - y, by simpa using sub_eq_zero.2 e.symm, by simp⟩
  /-
    🎉 no goals
  -/


theorem comap_map_eq_self {f : F} {p : Submodule R M} (h : LinearMap.ker f ≤ p) :
                                /-
                                  R : Type u_1
                                  R₂ : Type u_2
                                  M : Type u_4
                                  M₂ : Type u_5
                                  inst✝⁸ : Semiring R
                                  inst✝⁷ : Semiring R₂
                                  inst✝⁶ : AddCommGroup M
                                  inst✝⁵ : Module R M
                                  inst✝⁴ : AddCommGroup M₂
                                  inst✝³ : Module R₂ M₂
                                  τ₁₂ : RingHom R R₂
                                  inst✝² : RingHomSurjective τ₁₂
                                  F : Type u_8
                                  inst✝¹ : FunLike F M M₂
                                  inst✝ : SemilinearMapClass F τ₁₂ M M₂
                                  f : F
                                  p : Submodule R M
                                  h : LE.le (LinearMap.ker f) p
                                  ⊢ Eq (Submodule.comap f (Submodule.map f p)) p
                                -/
    comap f (map f p) = p := by rw [Submodule.comap_map_eq, sup_of_le_left h]
                                /-
                                  🎉 no goals
                                -/


lemma _root_.LinearMap.range_domRestrict_eq_range_iff {f : M →ₛₗ[τ₁₂] M₂} {S : Submodule R M} :
    LinearMap.range (f.domRestrict S) = LinearMap.range f ↔ S ⊔ (LinearMap.ker f) = ⊤ := by
  /-
    R : Type u_1
    R₂ : Type u_2
    M : Type u_4
    M₂ : Type u_5
    inst✝⁶ : Semiring R
    inst✝⁵ : Semiring R₂
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup M₂
    inst✝¹ : Module R₂ M₂
    τ₁₂ : RingHom R R₂
    inst✝ : RingHomSurjective τ₁₂
    f : LinearMap τ₁₂ M M₂
    S : Submodule R M
    ⊢ Iff (Eq (LinearMap.range (f.domRestrict S)) (LinearMap.range f)) (Eq (Max.ma …
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ ?_⟩
    /-
      case refine_1
      R : Type u_1
      R₂ : Type u_2
      M : Type u_4
      M₂ : Type u_5
      inst✝⁶ : Semiring R
      inst✝⁵ : Semiring R₂
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : AddCommGroup M₂
      inst✝¹ : Module R₂ M₂
      τ₁₂ : RingHom R R₂
      inst✝ : RingHomSurjective τ₁₂
      f : LinearMap τ₁₂ M M₂
      S : Submodule R M
      h : Eq (LinearMap.range (f.domRestrict S)) (LinearMap.range f)
      ⊢ Eq (Max.max S (LinearMap.ker f)) Top.top
    -/
  · rw [eq_top_iff]
    /-
      case refine_1
      R : Type u_1
      R₂ : Type u_2
      M : Type u_4
      M₂ : Type u_5
      inst✝⁶ : Semiring R
      inst✝⁵ : Semiring R₂
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : AddCommGroup M₂
      inst✝¹ : Module R₂ M₂
      τ₁₂ : RingHom R R₂
      inst✝ : RingHomSurjective τ₁₂
      f : LinearMap τ₁₂ M M₂
      S : Submodule R M
      h : Eq (LinearMap.range (f.domRestrict S)) (LinearMap.range f)
      ⊢ LE.le Top.top (Max.max S (LinearMap.ker f))
    -/
    intro x _
    /-
      case refine_1
      R : Type u_1
      R₂ : Type u_2
      M : Type u_4
      M₂ : Type u_5
      inst✝⁶ : Semiring R
      inst✝⁵ : Semiring R₂
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : AddCommGroup M₂
      inst✝¹ : Module R₂ M₂
      τ₁₂ : RingHom R R₂
      inst✝ : RingHomSurjective τ₁₂
      f : LinearMap τ₁₂ M M₂
      S : Submodule R M
      h : Eq (LinearMap.range (f.domRestrict S)) (LinearMap.range f)
      x : M
      a✝ : Membership.mem Top.top x
      ⊢ Membership.mem (Max.max S (LinearMap.ker f)) x
    -/
    have : f x ∈ LinearMap.range f := LinearMap.mem_range_self f x
    /-
      case refine_1
      R : Type u_1
      R₂ : Type u_2
      M : Type u_4
      M₂ : Type u_5
      inst✝⁶ : Semiring R
      inst✝⁵ : Semiring R₂
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : AddCommGroup M₂
      inst✝¹ : Module R₂ M₂
      τ₁₂ : RingHom R R₂
      inst✝ : RingHomSurjective τ₁₂
      f : LinearMap τ₁₂ M M₂
      S : Submodule R M
      h : Eq (LinearMap.range (f.domRestrict S)) (LinearMap.range f)
      x : M
      a✝ : Membership.mem Top.top x
      this : Membership.mem (LinearMap.range f) (f x)
      ⊢ Membership.mem (Max.max S (LinearMap.ker f)) x
    -/
    rw [← h] at this
    /-
      case refine_1
      R : Type u_1
      R₂ : Type u_2
      M : Type u_4
      M₂ : Type u_5
      inst✝⁶ : Semiring R
      inst✝⁵ : Semiring R₂
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : AddCommGroup M₂
      inst✝¹ : Module R₂ M₂
      τ₁₂ : RingHom R R₂
      inst✝ : RingHomSurjective τ₁₂
      f : LinearMap τ₁₂ M M₂
      S : Submodule R M
      h : Eq (LinearMap.range (f.domRestrict S)) (LinearMap.range f)
      x : M
      a✝ : Membership.mem Top.top x
      this : Membership.mem (LinearMap.range (f.domRestrict S)) (f x)
      ⊢ Membership.mem (Max.max S (LinearMap.ker f)) x
    -/
    obtain ⟨y, hy⟩ : ∃ y : S, f.domRestrict S y = f x := this
    /-
      case refine_1.intro
      R : Type u_1
      R₂ : Type u_2
      M : Type u_4
      M₂ : Type u_5
      inst✝⁶ : Semiring R
      inst✝⁵ : Semiring R₂
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : AddCommGroup M₂
      inst✝¹ : Module R₂ M₂
      τ₁₂ : RingHom R R₂
      inst✝ : RingHomSurjective τ₁₂
      f : LinearMap τ₁₂ M M₂
      S : Submodule R M
      h : Eq (LinearMap.range (f.domRestrict S)) (LinearMap.range f)
      x : M
      a✝ : Membership.mem Top.top x
      y : Subtype fun x => Membership.mem S x
      hy : Eq ((f.domRestrict S) y) (f x)
      ⊢ Membership.mem (Max.max S (LinearMap.ker f)) x
    -/
    have : (y : M) + (x - y) ∈ S ⊔ (LinearMap.ker f) := Submodule.add_mem_sup y.2 (by simp [← hy])
    /-
      case refine_1.intro
      R : Type u_1
      R₂ : Type u_2
      M : Type u_4
      M₂ : Type u_5
      inst✝⁶ : Semiring R
      inst✝⁵ : Semiring R₂
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : AddCommGroup M₂
      inst✝¹ : Module R₂ M₂
      τ₁₂ : RingHom R R₂
      inst✝ : RingHomSurjective τ₁₂
      f : LinearMap τ₁₂ M M₂
      S : Submodule R M
      h : Eq (LinearMap.range (f.domRestrict S)) (LinearMap.range f)
      x : M
      a✝ : Membership.mem Top.top x
      y : Subtype fun x => Membership.mem S x
      hy : Eq ((f.domRestrict S) y) (f x)
      this : Membership.mem (Max.max S (LinearMap.ker f)) (HAdd.hAdd (↑y) (HSub.hSub …
      ⊢ Membership.mem (Max.max S (LinearMap.ker f)) x
    -/
    simpa using this
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      R₂ : Type u_2
      M : Type u_4
      M₂ : Type u_5
      inst✝⁶ : Semiring R
      inst✝⁵ : Semiring R₂
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : AddCommGroup M₂
      inst✝¹ : Module R₂ M₂
      τ₁₂ : RingHom R R₂
      inst✝ : RingHomSurjective τ₁₂
      f : LinearMap τ₁₂ M M₂
      S : Submodule R M
      h : Eq (Max.max S (LinearMap.ker f)) Top.top
      ⊢ Eq (LinearMap.range (f.domRestrict S)) (LinearMap.range f)
    -/
  · refine le_antisymm (LinearMap.range_domRestrict_le_range f S) ?_
    /-
      case refine_2
      R : Type u_1
      R₂ : Type u_2
      M : Type u_4
      M₂ : Type u_5
      inst✝⁶ : Semiring R
      inst✝⁵ : Semiring R₂
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : AddCommGroup M₂
      inst✝¹ : Module R₂ M₂
      τ₁₂ : RingHom R R₂
      inst✝ : RingHomSurjective τ₁₂
      f : LinearMap τ₁₂ M M₂
      S : Submodule R M
      h : Eq (Max.max S (LinearMap.ker f)) Top.top
      ⊢ LE.le (LinearMap.range f) (LinearMap.range (f.domRestrict S))
    -/
    rintro x ⟨y, rfl⟩
    obtain ⟨s, hs, t, ht, rfl⟩ : ∃ s, s ∈ S ∧ ∃ t, t ∈ LinearMap.ker f ∧ s + t = y :=
      Submodule.mem_sup.1 (by simp [h])
    /-
      case refine_2.intro.intro.intro.intro.intro
      R : Type u_1
      R₂ : Type u_2
      M : Type u_4
      M₂ : Type u_5
      inst✝⁶ : Semiring R
      inst✝⁵ : Semiring R₂
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : AddCommGroup M₂
      inst✝¹ : Module R₂ M₂
      τ₁₂ : RingHom R R₂
      inst✝ : RingHomSurjective τ₁₂
      f : LinearMap τ₁₂ M M₂
      S : Submodule R M
      h : Eq (Max.max S (LinearMap.ker f)) Top.top
      s : M
      hs : Membership.mem S s
      t : M
      ht : Membership.mem (LinearMap.ker f) t
      ⊢ Membership.mem (LinearMap.range (f.domRestrict S)) (f (HAdd.hAdd s t))
    -/
    exact ⟨⟨s, hs⟩, by simp [LinearMap.mem_ker.1 ht]⟩
    /-
      🎉 no goals
    -/


@[simp] lemma _root_.LinearMap.surjective_domRestrict_iff
    {f : M →ₛₗ[τ₁₂] M₂} {S : Submodule R M} (hf : Surjective f) :
    Surjective (f.domRestrict S) ↔ S ⊔ LinearMap.ker f = ⊤ := by
  /-
    R : Type u_1
    R₂ : Type u_2
    M : Type u_4
    M₂ : Type u_5
    inst✝⁶ : Semiring R
    inst✝⁵ : Semiring R₂
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup M₂
    inst✝¹ : Module R₂ M₂
    τ₁₂ : RingHom R R₂
    inst✝ : RingHomSurjective τ₁₂
    f : LinearMap τ₁₂ M M₂
    S : Submodule R M
    hf : Function.Surjective ⇑f
    ⊢ Iff (Function.Surjective ⇑(f.domRestrict S)) (Eq (Max.max S (LinearMap.ker f …
  -/
  rw [← LinearMap.range_eq_top] at hf ⊢
  /-
    R : Type u_1
    R₂ : Type u_2
    M : Type u_4
    M₂ : Type u_5
    inst✝⁶ : Semiring R
    inst✝⁵ : Semiring R₂
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup M₂
    inst✝¹ : Module R₂ M₂
    τ₁₂ : RingHom R R₂
    inst✝ : RingHomSurjective τ₁₂
    f : LinearMap τ₁₂ M M₂
    S : Submodule R M
    hf : Eq (LinearMap.range f) Top.top
    ⊢ Iff (Eq (LinearMap.range (f.domRestrict S)) Top.top) (Eq (Max.max S (LinearM …
  -/
  rw [← hf]
  /-
    R : Type u_1
    R₂ : Type u_2
    M : Type u_4
    M₂ : Type u_5
    inst✝⁶ : Semiring R
    inst✝⁵ : Semiring R₂
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup M₂
    inst✝¹ : Module R₂ M₂
    τ₁₂ : RingHom R R₂
    inst✝ : RingHomSurjective τ₁₂
    f : LinearMap τ₁₂ M M₂
    S : Submodule R M
    hf : Eq (LinearMap.range f) Top.top
    ⊢ Iff (Eq (LinearMap.range (f.domRestrict S)) (LinearMap.range f)) (Eq (Max.ma …
  -/
  exact LinearMap.range_domRestrict_eq_range_iff
  /-
    🎉 no goals
  -/


@[simp]
lemma biSup_comap_subtype_eq_top {ι : Type*} (s : Set ι) (p : ι → Submodule R M) :
    ⨆ i ∈ s, (p i).comap (⨆ i ∈ s, p i).subtype = ⊤ := by
  /-
    R : Type u_1
    M : Type u_4
    inst✝² : Semiring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ι : Type u_9
    s : Set ι
    p : ι → Submodule R M
    ⊢ Eq (iSup fun i => iSup fun h => Submodule.comap (iSup fun i => iSup fun h => …
  -/
  refine eq_top_iff.mpr fun ⟨x, hx⟩ _ ↦ ?_
  suffices x ∈ (⨆ i ∈ s, (p i).comap (⨆ i ∈ s, p i).subtype).map (⨆ i ∈ s, (p i)).subtype by
    obtain ⟨y, hy, rfl⟩ := Submodule.mem_map.mp this
    exact hy
  suffices ∀ i ∈ s, (comap (⨆ i ∈ s, p i).subtype (p i)).map (⨆ i ∈ s, p i).subtype = p i by
    simpa only [map_iSup, biSup_congr this]
  /-
    R : Type u_1
    M : Type u_4
    inst✝² : Semiring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ι : Type u_9
    s : Set ι
    p : ι → Submodule R M
    x✝¹ : Subtype fun x => Membership.mem (iSup fun i => iSup fun h => p i) x
    x : M
    hx : Membership.mem (iSup fun i => iSup fun h => p i) x
    x✝ : Membership.mem Top.top ⟨x, hx⟩
    ⊢ ∀ (i : ι), Membership.mem s i → Eq (Submodule.map (iSup fun i => iSup fun h  …
  -/
  intro i hi
  /-
    R : Type u_1
    M : Type u_4
    inst✝² : Semiring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ι : Type u_9
    s : Set ι
    p : ι → Submodule R M
    x✝¹ : Subtype fun x => Membership.mem (iSup fun i => iSup fun h => p i) x
    x : M
    hx : Membership.mem (iSup fun i => iSup fun h => p i) x
    x✝ : Membership.mem Top.top ⟨x, hx⟩
    i : ι
    hi : Membership.mem s i
    ⊢ Eq (Submodule.map (iSup fun i => iSup fun h => p i).subtype (Submodule.comap …
  -/
  rw [map_comap_eq, range_subtype, inf_eq_right]
  /-
    R : Type u_1
    M : Type u_4
    inst✝² : Semiring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ι : Type u_9
    s : Set ι
    p : ι → Submodule R M
    x✝¹ : Subtype fun x => Membership.mem (iSup fun i => iSup fun h => p i) x
    x : M
    hx : Membership.mem (iSup fun i => iSup fun h => p i) x
    x✝ : Membership.mem Top.top ⟨x, hx⟩
    i : ι
    hi : Membership.mem s i
    ⊢ LE.le (p i) (iSup fun i => iSup fun h => p i)
  -/
  exact le_biSup p hi
  /-
    🎉 no goals
  -/


lemma biSup_comap_eq_top_of_surjective {ι : Type*} (s : Set ι) (hs : s.Nonempty)
    (p : ι → Submodule R₂ M₂) (hp : ⨆ i ∈ s, p i = ⊤)
    (f : M →ₛₗ[τ₁₂] M₂) (hf : Surjective f) :
    ⨆ i ∈ s, (p i).comap f = ⊤ := by
  /-
    R : Type u_1
    R₂ : Type u_2
    M : Type u_4
    M₂ : Type u_5
    inst✝⁶ : Semiring R
    inst✝⁵ : Semiring R₂
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup M₂
    inst✝¹ : Module R₂ M₂
    τ₁₂ : RingHom R R₂
    inst✝ : RingHomSurjective τ₁₂
    ι : Type u_9
    s : Set ι
    hs : s.Nonempty
    p : ι → Submodule R₂ M₂
    hp : Eq (iSup fun i => iSup fun h => p i) Top.top
    f : LinearMap τ₁₂ M M₂
    hf : Function.Surjective ⇑f
    ⊢ Eq (iSup fun i => iSup fun h => Submodule.comap f (p i)) Top.top
  -/
  obtain ⟨k, hk⟩ := hs
  suffices (⨆ i ∈ s, (p i).comap f) ⊔ LinearMap.ker f = ⊤ by
    rw [← this, left_eq_sup]; exact le_trans f.ker_le_comap (le_biSup (fun i ↦ (p i).comap f) hk)
  /-
    case intro
    R : Type u_1
    R₂ : Type u_2
    M : Type u_4
    M₂ : Type u_5
    inst✝⁶ : Semiring R
    inst✝⁵ : Semiring R₂
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup M₂
    inst✝¹ : Module R₂ M₂
    τ₁₂ : RingHom R R₂
    inst✝ : RingHomSurjective τ₁₂
    ι : Type u_9
    s : Set ι
    p : ι → Submodule R₂ M₂
    hp : Eq (iSup fun i => iSup fun h => p i) Top.top
    f : LinearMap τ₁₂ M M₂
    hf : Function.Surjective ⇑f
    k : ι
    hk : Membership.mem s k
    ⊢ Eq (Max.max (iSup fun i => iSup fun h => Submodule.comap f (p i)) (LinearMap …
  -/
  rw [iSup_subtype'] at hp ⊢
  /-
    case intro
    R : Type u_1
    R₂ : Type u_2
    M : Type u_4
    M₂ : Type u_5
    inst✝⁶ : Semiring R
    inst✝⁵ : Semiring R₂
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup M₂
    inst✝¹ : Module R₂ M₂
    τ₁₂ : RingHom R R₂
    inst✝ : RingHomSurjective τ₁₂
    ι : Type u_9
    s : Set ι
    p : ι → Submodule R₂ M₂
    hp : Eq (iSup fun x => p ↑x) Top.top
    f : LinearMap τ₁₂ M M₂
    hf : Function.Surjective ⇑f
    k : ι
    hk : Membership.mem s k
    ⊢ Eq (Max.max (iSup fun x => Submodule.comap f (p ↑x)) (LinearMap.ker f)) Top. …
  -/
  rw [← comap_map_eq, map_iSup_comap_of_sujective hf, hp, comap_top]
  /-
    🎉 no goals
  -/


lemma biSup_comap_eq_top_of_range_eq_biSup
    {R R₂ : Type*} [Ring R] [Ring R₂] {τ₁₂ : R →+* R₂} [RingHomSurjective τ₁₂]
    [Module R M] [Module R₂ M₂] {ι : Type*} (s : Set ι) (hs : s.Nonempty)
    (p : ι → Submodule R₂ M₂) (f : M →ₛₗ[τ₁₂] M₂) (hf : LinearMap.range f = ⨆ i ∈ s, p i) :
    ⨆ i ∈ s, (p i).comap f = ⊤ := by
  suffices ⨆ i ∈ s, (p i).comap (LinearMap.range f).subtype = ⊤ by
    rw [← biSup_comap_eq_top_of_surjective s hs _ this _ f.surjective_rangeRestrict]; rfl
  /-
    M : Type u_4
    M₂ : Type u_5
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : AddCommGroup M₂
    R : Type u_9
    R₂ : Type u_10
    inst✝⁴ : Ring R
    inst✝³ : Ring R₂
    τ₁₂ : RingHom R R₂
    inst✝² : RingHomSurjective τ₁₂
    inst✝¹ : Module R M
    inst✝ : Module R₂ M₂
    ι : Type u_11
    s : Set ι
    hs : s.Nonempty
    p : ι → Submodule R₂ M₂
    f : LinearMap τ₁₂ M M₂
    hf : Eq (LinearMap.range f) (iSup fun i => iSup fun h => p i)
    ⊢ Eq (iSup fun i => iSup fun h => Submodule.comap (LinearMap.range f).subtype  …
  -/
  exact hf ▸ biSup_comap_subtype_eq_top s p
  /-
    🎉 no goals
  -/


/-- There is no vector subspace between `s` and `(K ∙ x) ⊔ s`, `WCovBy` version. -/
theorem wcovBy_span_singleton_sup (x : V) (s : Submodule K V) : WCovBy s ((K ∙ x) ⊔ s) := by
  /-
    K : Type u_3
    V : Type u_6
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    x : V
    s : Submodule K V
    ⊢ WCovBy s (Max.max (Submodule.span K (Singleton.singleton x)) s)
  -/
  refine ⟨le_sup_right, fun q hpq hqp ↦ hqp.not_le ?_⟩
  /-
    K : Type u_3
    V : Type u_6
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    x : V
    s q : Submodule K V
    hpq : LT.lt s q
    hqp : LT.lt q (Max.max (Submodule.span K (Singleton.singleton x)) s)
    ⊢ LE.le (Max.max (Submodule.span K (Singleton.singleton x)) s) q
  -/
  rcases SetLike.exists_of_lt hpq with ⟨y, hyq, hyp⟩
  obtain ⟨c, z, hz, rfl⟩ : ∃ c : K, ∃ z ∈ s, c • x + z = y := by
    simpa [mem_sup, mem_span_singleton] using hqp.le hyq
  /-
    case intro.intro.intro.intro.intro
    K : Type u_3
    V : Type u_6
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    x : V
    s q : Submodule K V
    hpq : LT.lt s q
    hqp : LT.lt q (Max.max (Submodule.span K (Singleton.singleton x)) s)
    c : K
    z : V
    hz : Membership.mem s z
    hyq : Membership.mem q (HAdd.hAdd (HSMul.hSMul c x) z)
    hyp : Not (Membership.mem s (HAdd.hAdd (HSMul.hSMul c x) z))
    ⊢ LE.le (Max.max (Submodule.span K (Singleton.singleton x)) s) q
  -/
  rcases eq_or_ne c 0 with rfl | hc
    /-
      case intro.intro.intro.intro.intro.inl
      K : Type u_3
      V : Type u_6
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      x : V
      s q : Submodule K V
      hpq : LT.lt s q
      hqp : LT.lt q (Max.max (Submodule.span K (Singleton.singleton x)) s)
      z : V
      hz : Membership.mem s z
      hyq : Membership.mem q (HAdd.hAdd (HSMul.hSMul 0 x) z)
      hyp : Not (Membership.mem s (HAdd.hAdd (HSMul.hSMul 0 x) z))
      ⊢ LE.le (Max.max (Submodule.span K (Singleton.singleton x)) s) q
    -/
  · simp [hz] at hyp
    /-
      🎉 no goals
    -/
  · have : x ∈ q := by
      rwa [q.add_mem_iff_left (hpq.le hz), q.smul_mem_iff hc] at hyq
    /-
      case intro.intro.intro.intro.intro.inr
      K : Type u_3
      V : Type u_6
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      x : V
      s q : Submodule K V
      hpq : LT.lt s q
      hqp : LT.lt q (Max.max (Submodule.span K (Singleton.singleton x)) s)
      c : K
      z : V
      hz : Membership.mem s z
      hyq : Membership.mem q (HAdd.hAdd (HSMul.hSMul c x) z)
      hyp : Not (Membership.mem s (HAdd.hAdd (HSMul.hSMul c x) z))
      hc : Ne c 0
      this : Membership.mem q x
      ⊢ LE.le (Max.max (Submodule.span K (Singleton.singleton x)) s) q
    -/
    simp [hpq.le, this]
    /-
      🎉 no goals
    -/


/-- There is no vector subspace between `s` and `(K ∙ x) ⊔ s`, `CovBy` version. -/
theorem covBy_span_singleton_sup {x : V} {s : Submodule K V} (h : x ∉ s) : CovBy s ((K ∙ x) ⊔ s) :=
      /-
        K : Type u_3
        V : Type u_6
        inst✝² : DivisionRing K
        inst✝¹ : AddCommGroup V
        inst✝ : Module K V
        x : V
        s : Submodule K V
        h : Not (Membership.mem s x)
        ⊢ LT.lt s (Max.max (Submodule.span K (Singleton.singleton x)) s)
      -/
  ⟨by simpa, (wcovBy_span_singleton_sup _ _).2⟩
      /-
        🎉 no goals
      -/


theorem disjoint_span_singleton : Disjoint s (K ∙ x) ↔ x ∈ s → x = 0 := by
  /-
    K : Type u_3
    V : Type u_6
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    s : Submodule K V
    x : V
    ⊢ Iff (Disjoint s (Submodule.span K (Singleton.singleton x))) (Membership.mem  …
  -/
  refine disjoint_def.trans ⟨fun H hx => H x hx <| subset_span <| mem_singleton x, ?_⟩
  /-
    K : Type u_3
    V : Type u_6
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    s : Submodule K V
    x : V
    ⊢ (Membership.mem s x → Eq x 0) → ∀ (x_1 : V), Membership.mem s x_1 → Membersh …
  -/
  intro H y hy hyx
  /-
    K : Type u_3
    V : Type u_6
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    s : Submodule K V
    x : V
    H : Membership.mem s x → Eq x 0
    y : V
    hy : Membership.mem s y
    hyx : Membership.mem (Submodule.span K (Singleton.singleton x)) y
    ⊢ Eq y 0
  -/
  obtain ⟨c, rfl⟩ := mem_span_singleton.1 hyx
  /-
    case intro
    K : Type u_3
    V : Type u_6
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    s : Submodule K V
    x : V
    H : Membership.mem s x → Eq x 0
    c : K
    hy : Membership.mem s (HSMul.hSMul c x)
    hyx : Membership.mem (Submodule.span K (Singleton.singleton x)) (HSMul.hSMul c …
    ⊢ Eq (HSMul.hSMul c x) 0
  -/
  by_cases hc : c = 0
    /-
      case pos
      K : Type u_3
      V : Type u_6
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      s : Submodule K V
      x : V
      H : Membership.mem s x → Eq x 0
      c : K
      hy : Membership.mem s (HSMul.hSMul c x)
      hyx : Membership.mem (Submodule.span K (Singleton.singleton x)) (HSMul.hSMul c …
      hc : Eq c 0
      ⊢ Eq (HSMul.hSMul c x) 0
    -/
  · rw [hc, zero_smul]
    /-
      🎉 no goals
    -/
    /-
      case neg
      K : Type u_3
      V : Type u_6
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      s : Submodule K V
      x : V
      H : Membership.mem s x → Eq x 0
      c : K
      hy : Membership.mem s (HSMul.hSMul c x)
      hyx : Membership.mem (Submodule.span K (Singleton.singleton x)) (HSMul.hSMul c …
      hc : Not (Eq c 0)
      ⊢ Eq (HSMul.hSMul c x) 0
    -/
  · rw [s.smul_mem_iff hc] at hy
    /-
      case neg
      K : Type u_3
      V : Type u_6
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      s : Submodule K V
      x : V
      H : Membership.mem s x → Eq x 0
      c : K
      hy : Membership.mem s x
      hyx : Membership.mem (Submodule.span K (Singleton.singleton x)) (HSMul.hSMul c …
      hc : Not (Eq c 0)
      ⊢ Eq (HSMul.hSMul c x) 0
    -/
    rw [H hy, smul_zero]
    /-
      🎉 no goals
    -/


theorem disjoint_span_singleton' (x0 : x ≠ 0) : Disjoint s (K ∙ x) ↔ x ∉ s :=
  disjoint_span_singleton.trans ⟨fun h₁ h₂ => x0 (h₁ h₂), fun h₁ h₂ => (h₁ h₂).elim⟩


lemma disjoint_span_singleton_of_not_mem (hx : x ∉ s) : Disjoint s (K ∙ x) := by
  /-
    K : Type u_3
    V : Type u_6
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    s : Submodule K V
    x : V
    hx : Not (Membership.mem s x)
    ⊢ Disjoint s (Submodule.span K (Singleton.singleton x))
  -/
  rw [disjoint_span_singleton]
  /-
    K : Type u_3
    V : Type u_6
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    s : Submodule K V
    x : V
    hx : Not (Membership.mem s x)
    ⊢ Membership.mem s x → Eq x 0
  -/
  intro h
  /-
    K : Type u_3
    V : Type u_6
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    s : Submodule K V
    x : V
    hx : Not (Membership.mem s x)
    h : Membership.mem s x
    ⊢ Eq x 0
  -/
  contradiction
  /-
    🎉 no goals
  -/


lemma isCompl_span_singleton_of_isCoatom_of_not_mem (hs : IsCoatom s) (hx : x ∉ s) :
    IsCompl s (K ∙ x) := by
  /-
    K : Type u_3
    V : Type u_6
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    s : Submodule K V
    x : V
    hs : IsCoatom s
    hx : Not (Membership.mem s x)
    ⊢ IsCompl s (Submodule.span K (Singleton.singleton x))
  -/
  refine ⟨disjoint_span_singleton_of_not_mem hx, ?_⟩
  /-
    K : Type u_3
    V : Type u_6
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    s : Submodule K V
    x : V
    hs : IsCoatom s
    hx : Not (Membership.mem s x)
    ⊢ Codisjoint s (Submodule.span K (Singleton.singleton x))
  -/
  rw [← covBy_top_iff] at hs
  /-
    K : Type u_3
    V : Type u_6
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    s : Submodule K V
    x : V
    hs : CovBy s Top.top
    hx : Not (Membership.mem s x)
    ⊢ Codisjoint s (Submodule.span K (Singleton.singleton x))
  -/
  simpa only [codisjoint_iff, sup_comm, not_lt_top_iff] using hs.2 (covBy_span_singleton_sup hx).1
  /-
    🎉 no goals
  -/


protected theorem map_le_map_iff (f : F) {p p'} : map f p ≤ map f p' ↔ p ≤ p' ⊔ ker f := by
  /-
    R : Type u_1
    R₂ : Type u_2
    M : Type u_4
    M₂ : Type u_5
    inst✝⁸ : Semiring R
    inst✝⁷ : Semiring R₂
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : AddCommGroup M₂
    inst✝⁴ : Module R M
    inst✝³ : Module R₂ M₂
    τ₁₂ : RingHom R R₂
    inst✝² : RingHomSurjective τ₁₂
    F : Type u_8
    inst✝¹ : FunLike F M M₂
    inst✝ : SemilinearMapClass F τ₁₂ M M₂
    f : F
    p p' : Submodule R M
    ⊢ Iff (LE.le (Submodule.map f p) (Submodule.map f p')) (LE.le p (Max.max p' (L …
  -/
  rw [map_le_iff_le_comap, Submodule.comap_map_eq]
  /-
    🎉 no goals
  -/


theorem map_le_map_iff' {f : F} (hf : ker f = ⊥) {p p'} : map f p ≤ map f p' ↔ p ≤ p' := by
  /-
    R : Type u_1
    R₂ : Type u_2
    M : Type u_4
    M₂ : Type u_5
    inst✝⁸ : Semiring R
    inst✝⁷ : Semiring R₂
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : AddCommGroup M₂
    inst✝⁴ : Module R M
    inst✝³ : Module R₂ M₂
    τ₁₂ : RingHom R R₂
    inst✝² : RingHomSurjective τ₁₂
    F : Type u_8
    inst✝¹ : FunLike F M M₂
    inst✝ : SemilinearMapClass F τ₁₂ M M₂
    f : F
    hf : Eq (LinearMap.ker f) Bot.bot
    p p' : Submodule R M
    ⊢ Iff (LE.le (Submodule.map f p) (Submodule.map f p')) (LE.le p p')
  -/
  rw [LinearMap.map_le_map_iff, hf, sup_bot_eq]
  /-
    🎉 no goals
  -/


theorem map_injective {f : F} (hf : ker f = ⊥) : Injective (map f) := fun _ _ h =>
  le_antisymm ((map_le_map_iff' hf).1 (le_of_eq h)) ((map_le_map_iff' hf).1 (ge_of_eq h))


theorem map_eq_top_iff {f : F} (hf : range f = ⊤) {p : Submodule R M} :
    p.map f = ⊤ ↔ p ⊔ LinearMap.ker f = ⊤ := by
  /-
    R : Type u_1
    R₂ : Type u_2
    M : Type u_4
    M₂ : Type u_5
    inst✝⁸ : Semiring R
    inst✝⁷ : Semiring R₂
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : AddCommGroup M₂
    inst✝⁴ : Module R M
    inst✝³ : Module R₂ M₂
    τ₁₂ : RingHom R R₂
    inst✝² : RingHomSurjective τ₁₂
    F : Type u_8
    inst✝¹ : FunLike F M M₂
    inst✝ : SemilinearMapClass F τ₁₂ M M₂
    f : F
    hf : Eq (LinearMap.range f) Top.top
    p : Submodule R M
    ⊢ Iff (Eq (Submodule.map f p) Top.top) (Eq (Max.max p (LinearMap.ker f)) Top.t …
  -/
  simp_rw [← top_le_iff, ← hf, range_eq_map, LinearMap.map_le_map_iff]
  /-
    🎉 no goals
  -/


/-- Given an element `x` of a module `M` over `R`, the natural map from
    `R` to scalar multiples of `x`. See also `LinearMap.ringLmapEquivSelf`. -/
@[simps!]
def toSpanSingleton (x : M) : R →ₗ[R] M :=
  LinearMap.id.smulRight x


/-- The range of `toSpanSingleton x` is the span of `x`. -/
theorem span_singleton_eq_range (x : M) : (R ∙ x) = range (toSpanSingleton R M x) :=
  Submodule.ext fun y => by
    /-
      R : Type u_1
      M : Type u_4
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      x y : M
      ⊢ Iff (Membership.mem (Submodule.span R (Singleton.singleton x)) y) (Membershi …
    -/
    refine Iff.trans ?_ LinearMap.mem_range.symm
    /-
      R : Type u_1
      M : Type u_4
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      x y : M
      ⊢ Iff (Membership.mem (Submodule.span R (Singleton.singleton x)) y) (Exists fu …
    -/
    exact mem_span_singleton
    /-
      🎉 no goals
    -/


theorem toSpanSingleton_one (x : M) : toSpanSingleton R M x 1 = x :=
  one_smul _ _


@[simp]
theorem toSpanSingleton_zero : toSpanSingleton R M 0 = 0 := by
  /-
    R : Type u_1
    M : Type u_4
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    ⊢ Eq (LinearMap.toSpanSingleton R M 0) 0
  -/
  ext
  /-
    case h
    R : Type u_1
    M : Type u_4
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    ⊢ Eq ((LinearMap.toSpanSingleton R M 0) 1) (0 1)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem toSpanSingleton_isIdempotentElem_iff {e : R} :
    IsIdempotentElem (toSpanSingleton R R e) ↔ IsIdempotentElem e := by
  simp_rw [IsIdempotentElem, LinearMap.ext_iff, mul_apply, toSpanSingleton_apply, smul_eq_mul,
    mul_assoc]
  /-
    R : Type u_1
    inst✝ : Semiring R
    e : R
    ⊢ Iff (∀ (x : R), Eq (HMul.hMul x (HMul.hMul e e)) (HMul.hMul x e)) (Eq (HMul. …
  -/
  exact ⟨fun h ↦ by conv_rhs => rw [← one_mul e, ← h, one_mul], fun h _ ↦ by rw [h]⟩
  /-
    🎉 no goals
  -/


theorem isIdempotentElem_apply_one_iff {f : Module.End R R} :
    IsIdempotentElem (f 1) ↔ IsIdempotentElem f := by
  rw [IsIdempotentElem, ← smul_eq_mul, ← map_smul, smul_eq_mul, mul_one, IsIdempotentElem,
    LinearMap.ext_iff]
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : Module.End R R
    ⊢ Iff (Eq (f (f 1)) (f 1)) (∀ (x : R), Eq ((HMul.hMul f f) x) (f x))
  -/
  simp_rw [mul_apply]
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : Module.End R R
    ⊢ Iff (Eq (f (f 1)) (f 1)) (∀ (x : R), Eq (f (f x)) (f x))
  -/
  exact ⟨fun h r ↦ by rw [← mul_one r, ← smul_eq_mul, map_smul, map_smul, h], (· 1)⟩
  /-
    🎉 no goals
  -/


/-- Two linear maps are equal on `Submodule.span s` iff they are equal on `s`. -/
theorem eqOn_span_iff {s : Set M} {f g : F} : Set.EqOn f g (span R s) ↔ Set.EqOn f g s := by
  /-
    R : Type u_1
    R₂ : Type u_2
    M : Type u_4
    M₂ : Type u_5
    inst✝⁷ : Semiring R
    inst✝⁶ : AddCommMonoid M
    inst✝⁵ : Module R M
    inst✝⁴ : Semiring R₂
    inst✝³ : AddCommMonoid M₂
    inst✝² : Module R₂ M₂
    F : Type u_8
    σ₁₂ : RingHom R R₂
    inst✝¹ : FunLike F M M₂
    inst✝ : SemilinearMapClass F σ₁₂ M M₂
    s : Set M
    f g : F
    ⊢ Iff (Set.EqOn ⇑f ⇑g ↑(Submodule.span R s)) (Set.EqOn (⇑f) (⇑g) s)
  -/
  rw [← le_eqLocus, span_le]; rfl
                              /-
                                🎉 no goals
                              -/


/-- If two linear maps are equal on a set `s`, then they are equal on `Submodule.span s`.

This version uses `Set.EqOn`, and the hidden argument will expand to `h : x ∈ (span R s : Set M)`.
See `LinearMap.eqOn_span` for a version that takes `h : x ∈ span R s` as an argument. -/
theorem eqOn_span' {s : Set M} {f g : F} (H : Set.EqOn f g s) :
    Set.EqOn f g (span R s : Set M) :=
  eqOn_span_iff.2 H


/-- If two linear maps are equal on a set `s`, then they are equal on `Submodule.span s`.

See also `LinearMap.eqOn_span'` for a version using `Set.EqOn`. -/
theorem eqOn_span {s : Set M} {f g : F} (H : Set.EqOn f g s) ⦃x⦄ (h : x ∈ span R s) :
    f x = g x :=
  eqOn_span' H h


/-- If `s` generates the whole module and linear maps `f`, `g` are equal on `s`, then they are
equal. -/
theorem ext_on {s : Set M} {f g : F} (hv : span R s = ⊤) (h : Set.EqOn f g s) : f = g :=
  DFunLike.ext _ _ fun _ => eqOn_span h (eq_top_iff'.1 hv _)


/-- If the range of `v : ι → M` generates the whole module and linear maps `f`, `g` are equal at
each `v i`, then they are equal. -/
theorem ext_on_range {ι : Sort*} {v : ι → M} {f g : F} (hv : span R (Set.range v) = ⊤)
    (h : ∀ i, f (v i) = g (v i)) : f = g :=
  ext_on hv (Set.forall_mem_range.2 h)


theorem ker_toSpanSingleton {x : M} (h : x ≠ 0) : LinearMap.ker (toSpanSingleton R M x) = ⊥ :=
  SetLike.ext fun _ => smul_eq_zero.trans <| or_iff_left_of_imp fun h' => (h h').elim


theorem span_singleton_sup_ker_eq_top (f : V →ₗ[K] K) {x : V} (hx : f x ≠ 0) :
    (K ∙ x) ⊔ ker f = ⊤ :=
  top_unique fun y _ =>
    Submodule.mem_sup.2
      ⟨(f y * (f x)⁻¹) • x, Submodule.mem_span_singleton.2 ⟨f y * (f x)⁻¹, rfl⟩,
                                     /-
                                       K : Type u_3
                                       V : Type u_6
                                       inst✝² : Field K
                                       inst✝¹ : AddCommGroup V
                                       inst✝ : Module K V
                                       f : LinearMap (RingHom.id K) V K
                                       x : V
                                       hx : Ne (f x) 0
                                       y : V
                                       x✝ : Membership.mem Top.top y
                                       ⊢ And (Membership.mem (LinearMap.ker f) (HSub.hSub y (HSMul.hSMul (HMul.hMul ( …
                                     -/
        ⟨y - (f y * (f x)⁻¹) • x, by simp [hx]⟩⟩
                                     /-
                                       🎉 no goals
                                     -/


/-- Given a nonzero element `x` of a torsion-free module `M` over a ring `R`, the natural
isomorphism from `R` to the span of `x` given by $r \mapsto r \cdot x$. -/
noncomputable
def toSpanNonzeroSingleton : R ≃ₗ[R] R ∙ x :=
  LinearEquiv.trans
    (LinearEquiv.ofInjective (LinearMap.toSpanSingleton R M x)
      (ker_eq_bot.1 <| ker_toSpanSingleton R M h))
    (LinearEquiv.ofEq (range <| toSpanSingleton R M x) (R ∙ x) (span_singleton_eq_range R M x).symm)


@[simp] theorem toSpanNonzeroSingleton_apply (t : R) :
    toSpanNonzeroSingleton R M x h t =
      (⟨t • x, Submodule.smul_mem _ _ (Submodule.mem_span_singleton_self x)⟩ : R ∙ x) := by
  /-
    R : Type u_1
    M : Type u_4
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroSMulDivisors R M
    x : M
    h : Ne x 0
    t : R
    ⊢ Eq ((LinearEquiv.toSpanNonzeroSingleton R M x h) t) ⟨HSMul.hSMul t x, ⋯⟩
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
lemma toSpanNonzeroSingleton_symm_apply_smul (m : R ∙ x) :
    (toSpanNonzeroSingleton R M x h).symm m • x = m :=
  congrArg Subtype.val <| apply_symm_apply (toSpanNonzeroSingleton R M x h) m


theorem toSpanNonzeroSingleton_one :
    LinearEquiv.toSpanNonzeroSingleton R M x h 1 =
                                                               /-
                                                                 R : Type u_1
                                                                 M : Type u_4
                                                                 inst✝³ : Ring R
                                                                 inst✝² : AddCommGroup M
                                                                 inst✝¹ : Module R M
                                                                 inst✝ : NoZeroSMulDivisors R M
                                                                 x : M
                                                                 h : Ne x 0
                                                                 ⊢ Eq ((LinearEquiv.toSpanNonzeroSingleton R M x h) 1) ⟨x, ⋯⟩
                                                               -/
      (⟨x, Submodule.mem_span_singleton_self x⟩ : R ∙ x) := by simp
                                                               /-
                                                                 🎉 no goals
                                                               -/


/-- Given a nonzero element `x` of a torsion-free module `M` over a ring `R`, the natural
isomorphism from the span of `x` to `R` given by $r \cdot x \mapsto r$. -/
noncomputable
abbrev coord : (R ∙ x) ≃ₗ[R] R :=
  (toSpanNonzeroSingleton R M x h).symm


theorem coord_self : (coord R M x h) (⟨x, Submodule.mem_span_singleton_self x⟩ : R ∙ x) = 1 := by
  /-
    R : Type u_1
    M : Type u_4
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroSMulDivisors R M
    x : M
    h : Ne x 0
    ⊢ Eq ((LinearEquiv.coord R M x h) ⟨x, ⋯⟩) 1
  -/
  rw [← toSpanNonzeroSingleton_one R M x h, LinearEquiv.symm_apply_apply]
  /-
    🎉 no goals
  -/


theorem coord_apply_smul (y : Submodule.span R ({x} : Set M)) : coord R M x h y • x = y :=
  Subtype.ext_iff.1 <| (toSpanNonzeroSingleton R M x h).apply_symm_apply _


