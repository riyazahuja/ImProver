/-- Being finitely-presented is preserved by localizations. -/
theorem finitePresentation_localizationPreserves : LocalizationPreserves @FinitePresentation := by
  /-
    ⊢ RingHom.LocalizationPreserves @RingHom.FinitePresentation
  -/
  introv R hf
  /-
    R S : Type u_1
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    f : RingHom R S
    M : Submonoid R
    R' S' : Type u_1
    inst✝⁵ : CommRing R'
    inst✝⁴ : CommRing S'
    inst✝³ : Algebra R R'
    inst✝² : Algebra S S'
    inst✝¹ : IsLocalization M R'
    inst✝ : IsLocalization (Submonoid.map f M) S'
    hf : f.FinitePresentation
    ⊢ (IsLocalization.map S' f ⋯).FinitePresentation
  -/
  letI := f.toAlgebra
  /-
    R S : Type u_1
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    f : RingHom R S
    M : Submonoid R
    R' S' : Type u_1
    inst✝⁵ : CommRing R'
    inst✝⁴ : CommRing S'
    inst✝³ : Algebra R R'
    inst✝² : Algebra S S'
    inst✝¹ : IsLocalization M R'
    inst✝ : IsLocalization (Submonoid.map f M) S'
    hf : f.FinitePresentation
    this : Algebra R S := f.toAlgebra
    ⊢ (IsLocalization.map S' f ⋯).FinitePresentation
  -/
  letI := ((algebraMap S S').comp f).toAlgebra
  /-
    R S : Type u_1
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    f : RingHom R S
    M : Submonoid R
    R' S' : Type u_1
    inst✝⁵ : CommRing R'
    inst✝⁴ : CommRing S'
    inst✝³ : Algebra R R'
    inst✝² : Algebra S S'
    inst✝¹ : IsLocalization M R'
    inst✝ : IsLocalization (Submonoid.map f M) S'
    hf : f.FinitePresentation
    this✝ : Algebra R S := f.toAlgebra
    this : Algebra R S' := ((algebraMap S S').comp f).toAlgebra
    ⊢ (IsLocalization.map S' f ⋯).FinitePresentation
  -/
  let f' : R' →+* S' := IsLocalization.map S' f M.le_comap_map
  /-
    R S : Type u_1
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    f : RingHom R S
    M : Submonoid R
    R' S' : Type u_1
    inst✝⁵ : CommRing R'
    inst✝⁴ : CommRing S'
    inst✝³ : Algebra R R'
    inst✝² : Algebra S S'
    inst✝¹ : IsLocalization M R'
    inst✝ : IsLocalization (Submonoid.map f M) S'
    hf : f.FinitePresentation
    this✝ : Algebra R S := f.toAlgebra
    this : Algebra R S' := ((algebraMap S S').comp f).toAlgebra
    f' : RingHom R' S' := IsLocalization.map S' f ⋯
    ⊢ (IsLocalization.map S' f ⋯).FinitePresentation
  -/
  letI := f'.toAlgebra
  haveI : IsScalarTower R R' S' :=
    IsScalarTower.of_algebraMap_eq' (IsLocalization.map_comp M.le_comap_map).symm
  /-
    R S : Type u_1
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    f : RingHom R S
    M : Submonoid R
    R' S' : Type u_1
    inst✝⁵ : CommRing R'
    inst✝⁴ : CommRing S'
    inst✝³ : Algebra R R'
    inst✝² : Algebra S S'
    inst✝¹ : IsLocalization M R'
    inst✝ : IsLocalization (Submonoid.map f M) S'
    hf : f.FinitePresentation
    this✝² : Algebra R S := f.toAlgebra
    this✝¹ : Algebra R S' := ((algebraMap S S').comp f).toAlgebra
    f' : RingHom R' S' := IsLocalization.map S' f ⋯
    this✝ : Algebra R' S' := f'.toAlgebra
    this : IsScalarTower R R' S'
    ⊢ (IsLocalization.map S' f ⋯).FinitePresentation
  -/
  obtain ⟨n, g, hgsurj, hgker⟩ := hf
  let MX : Submonoid (MvPolynomial (Fin n) R) :=
    Algebra.algebraMapSubmonoid (MvPolynomial (Fin n) R) M
  haveI : IsLocalization MX (MvPolynomial (Fin n) R') :=
    inferInstanceAs <| IsLocalization (M.map MvPolynomial.C) (MvPolynomial (Fin n) R')
  /-
    case mk.intro.intro.intro
    R S : Type u_1
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    f : RingHom R S
    M : Submonoid R
    R' S' : Type u_1
    inst✝⁵ : CommRing R'
    inst✝⁴ : CommRing S'
    inst✝³ : Algebra R R'
    inst✝² : Algebra S S'
    inst✝¹ : IsLocalization M R'
    inst✝ : IsLocalization (Submonoid.map f M) S'
    this✝³ : Algebra R S := f.toAlgebra
    this✝² : Algebra R S' := ((algebraMap S S').comp f).toAlgebra
    f' : RingHom R' S' := IsLocalization.map S' f ⋯
    this✝¹ : Algebra R' S' := f'.toAlgebra
    this✝ : IsScalarTower R R' S'
    n : Nat
    g : AlgHom R (MvPolynomial (Fin n) R) S
    hgsurj : Function.Surjective ⇑g
    hgker : (RingHom.ker g.toRingHom).FG
    MX : Submonoid (MvPolynomial (Fin n) R) := Algebra.algebraMapSubmonoid (MvPoly …
    this : IsLocalization MX (MvPolynomial (Fin n) R')
    ⊢ (IsLocalization.map S' f ⋯).FinitePresentation
  -/
  haveI : IsScalarTower R S S' := IsScalarTower.of_algebraMap_eq' rfl
  haveI : IsLocalization (Algebra.algebraMapSubmonoid S M) S' :=
    inferInstanceAs <| IsLocalization (M.map f) S'
  /-
    case mk.intro.intro.intro
    R S : Type u_1
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    f : RingHom R S
    M : Submonoid R
    R' S' : Type u_1
    inst✝⁵ : CommRing R'
    inst✝⁴ : CommRing S'
    inst✝³ : Algebra R R'
    inst✝² : Algebra S S'
    inst✝¹ : IsLocalization M R'
    inst✝ : IsLocalization (Submonoid.map f M) S'
    this✝⁵ : Algebra R S := f.toAlgebra
    this✝⁴ : Algebra R S' := ((algebraMap S S').comp f).toAlgebra
    f' : RingHom R' S' := IsLocalization.map S' f ⋯
    this✝³ : Algebra R' S' := f'.toAlgebra
    this✝² : IsScalarTower R R' S'
    n : Nat
    g : AlgHom R (MvPolynomial (Fin n) R) S
    hgsurj : Function.Surjective ⇑g
    hgker : (RingHom.ker g.toRingHom).FG
    MX : Submonoid (MvPolynomial (Fin n) R) := Algebra.algebraMapSubmonoid (MvPoly …
    this✝¹ : IsLocalization MX (MvPolynomial (Fin n) R')
    this✝ : IsScalarTower R S S'
    this : IsLocalization (Algebra.algebraMapSubmonoid S M) S'
    ⊢ (IsLocalization.map S' f ⋯).FinitePresentation
  -/
  let g' : MvPolynomial (Fin n) R' →ₐ[R'] S' := IsLocalization.mapₐ M R' _ S' g
  let k : RingHom.ker g →ₗ[MvPolynomial (Fin n) R] RingHom.ker g' :=
    AlgHom.toKerIsLocalization M R' _ S' g
  /-
    case mk.intro.intro.intro
    R S : Type u_1
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    f : RingHom R S
    M : Submonoid R
    R' S' : Type u_1
    inst✝⁵ : CommRing R'
    inst✝⁴ : CommRing S'
    inst✝³ : Algebra R R'
    inst✝² : Algebra S S'
    inst✝¹ : IsLocalization M R'
    inst✝ : IsLocalization (Submonoid.map f M) S'
    this✝⁵ : Algebra R S := f.toAlgebra
    this✝⁴ : Algebra R S' := ((algebraMap S S').comp f).toAlgebra
    f' : RingHom R' S' := IsLocalization.map S' f ⋯
    this✝³ : Algebra R' S' := f'.toAlgebra
    this✝² : IsScalarTower R R' S'
    n : Nat
    g : AlgHom R (MvPolynomial (Fin n) R) S
    hgsurj : Function.Surjective ⇑g
    hgker : (RingHom.ker g.toRingHom).FG
    MX : Submonoid (MvPolynomial (Fin n) R) := Algebra.algebraMapSubmonoid (MvPoly …
    this✝¹ : IsLocalization MX (MvPolynomial (Fin n) R')
    this✝ : IsScalarTower R S S'
    this : IsLocalization (Algebra.algebraMapSubmonoid S M) S'
    g' : AlgHom R' (MvPolynomial (Fin n) R') S' := IsLocalization.mapₐ M R' (MvPol …
    k : LinearMap (RingHom.id (MvPolynomial (Fin n) R)) (Subtype fun x => Membersh …
    ⊢ (IsLocalization.map S' f ⋯).FinitePresentation
  -/
  have : IsLocalizedModule MX k := AlgHom.toKerIsLocalization_isLocalizedModule M _ _ _ g
  /-
    case mk.intro.intro.intro
    R S : Type u_1
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    f : RingHom R S
    M : Submonoid R
    R' S' : Type u_1
    inst✝⁵ : CommRing R'
    inst✝⁴ : CommRing S'
    inst✝³ : Algebra R R'
    inst✝² : Algebra S S'
    inst✝¹ : IsLocalization M R'
    inst✝ : IsLocalization (Submonoid.map f M) S'
    this✝⁶ : Algebra R S := f.toAlgebra
    this✝⁵ : Algebra R S' := ((algebraMap S S').comp f).toAlgebra
    f' : RingHom R' S' := IsLocalization.map S' f ⋯
    this✝⁴ : Algebra R' S' := f'.toAlgebra
    this✝³ : IsScalarTower R R' S'
    n : Nat
    g : AlgHom R (MvPolynomial (Fin n) R) S
    hgsurj : Function.Surjective ⇑g
    hgker : (RingHom.ker g.toRingHom).FG
    MX : Submonoid (MvPolynomial (Fin n) R) := Algebra.algebraMapSubmonoid (MvPoly …
    this✝² : IsLocalization MX (MvPolynomial (Fin n) R')
    this✝¹ : IsScalarTower R S S'
    this✝ : IsLocalization (Algebra.algebraMapSubmonoid S M) S'
    g' : AlgHom R' (MvPolynomial (Fin n) R') S' := IsLocalization.mapₐ M R' (MvPol …
    k : LinearMap (RingHom.id (MvPolynomial (Fin n) R)) (Subtype fun x => Membersh …
    this : IsLocalizedModule MX k
    ⊢ (IsLocalization.map S' f ⋯).FinitePresentation
  -/
  have : Module.Finite (MvPolynomial (Fin n) R) (ker g) := Module.Finite.iff_fg.mpr hgker
  exact ⟨n, g', IsLocalization.mapₐ_surjective_of_surjective M R' _ S' g hgsurj,
    Module.Finite.iff_fg.mp (Module.Finite.of_isLocalizedModule MX k)⟩


/-- Being finitely-presented is stable under composition. -/
theorem finitePresentation_stableUnderComposition : StableUnderComposition @FinitePresentation := by
  /-
    ⊢ RingHom.StableUnderComposition @RingHom.FinitePresentation
  -/
  introv R hf hg
  /-
    R S T : Type u_1
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : CommRing T
    f : RingHom R S
    g : RingHom S T
    hf : f.FinitePresentation
    hg : g.FinitePresentation
    ⊢ (g.comp f).FinitePresentation
  -/
  exact hg.comp hf
  /-
    🎉 no goals
  -/


/-- If `R` is a ring, then `Rᵣ` is `R`-finitely-presented for any `r : R`. -/
theorem finitePresentation_holdsForLocalizationAway :
    HoldsForLocalizationAway @FinitePresentation := by
  /-
    ⊢ RingHom.HoldsForLocalizationAway @RingHom.FinitePresentation
  -/
  introv R _
  suffices Algebra.FinitePresentation R S by
    rw [RingHom.FinitePresentation]
    convert this; ext;
    rw [Algebra.smul_def]; rfl
  /-
    R S : Type u_1
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    r : R
    inst✝ : IsLocalization.Away r S
    ⊢ Algebra.FinitePresentation R S
  -/
  exact IsLocalization.Away.finitePresentation r
  /-
    🎉 no goals
  -/


/--
If `S` is an `R`-algebra with a surjection from a finitely-presented `R`-algebra `A`, such that
localized at a spanning set `{ r }` of elements of `A`, `Sᵣ` is finitely-presented, then
`S` is finitely presented.
This is almost `finitePresentation_ofLocalizationSpanTarget`. The difference is,
that here the set `t` generates the unit ideal of `A`, while in the general version,
it only generates a quotient of `A`.
-/
lemma finitePresentation_ofLocalizationSpanTarget_aux
    {R S A : Type*} [CommRing R] [CommRing S] [CommRing A] [Algebra R S] [Algebra R A]
    [Algebra.FinitePresentation R A] (f : A →ₐ[R] S) (hf : Function.Surjective f)
    (t : Finset A) (ht : Ideal.span (t : Set A) = ⊤)
    (H : ∀ g : t, Algebra.FinitePresentation R (Localization.Away (f g))) :
    Algebra.FinitePresentation R S := by
  /-
    R : Type u_1
    S : Type u_2
    A : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing S
    inst✝³ : CommRing A
    inst✝² : Algebra R S
    inst✝¹ : Algebra R A
    inst✝ : Algebra.FinitePresentation R A
    f : AlgHom R A S
    hf : Function.Surjective ⇑f
    t : Finset A
    ht : Eq (Ideal.span ↑t) Top.top
    H : ∀ (g : Subtype fun x => Membership.mem t x), Algebra.FinitePresentation R  …
    ⊢ Algebra.FinitePresentation R S
  -/
  apply Algebra.FinitePresentation.of_surjective hf
  /-
    case hker
    R : Type u_1
    S : Type u_2
    A : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing S
    inst✝³ : CommRing A
    inst✝² : Algebra R S
    inst✝¹ : Algebra R A
    inst✝ : Algebra.FinitePresentation R A
    f : AlgHom R A S
    hf : Function.Surjective ⇑f
    t : Finset A
    ht : Eq (Ideal.span ↑t) Top.top
    H : ∀ (g : Subtype fun x => Membership.mem t x), Algebra.FinitePresentation R  …
    ⊢ (RingHom.ker f.toRingHom).FG
  -/
  apply ker_fg_of_localizationSpan t ht
  /-
    case hker
    R : Type u_1
    S : Type u_2
    A : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing S
    inst✝³ : CommRing A
    inst✝² : Algebra R S
    inst✝¹ : Algebra R A
    inst✝ : Algebra.FinitePresentation R A
    f : AlgHom R A S
    hf : Function.Surjective ⇑f
    t : Finset A
    ht : Eq (Ideal.span ↑t) Top.top
    H : ∀ (g : Subtype fun x => Membership.mem t x), Algebra.FinitePresentation R  …
    ⊢ ∀ (g : ↑↑t), (RingHom.ker (Localization.awayMap f.toRingHom ↑g)).FG
  -/
  intro g
  let f' : Localization.Away g.val →ₐ[R] Localization.Away (f g) :=
    Localization.awayMapₐ f g.val
  have (g : t) : Algebra.FinitePresentation R (Localization.Away g.val) :=
    haveI : Algebra.FinitePresentation A (Localization.Away g.val) :=
      IsLocalization.Away.finitePresentation g.val
    Algebra.FinitePresentation.trans R A (Localization.Away g.val)
  /-
    case hker
    R : Type u_1
    S : Type u_2
    A : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing S
    inst✝³ : CommRing A
    inst✝² : Algebra R S
    inst✝¹ : Algebra R A
    inst✝ : Algebra.FinitePresentation R A
    f : AlgHom R A S
    hf : Function.Surjective ⇑f
    t : Finset A
    ht : Eq (Ideal.span ↑t) Top.top
    H : ∀ (g : Subtype fun x => Membership.mem t x), Algebra.FinitePresentation R  …
    g : ↑↑t
    f' : AlgHom R (Localization.Away ↑g) (Localization.Away (f ↑g)) := Localizatio …
    this : ∀ (g : Subtype fun x => Membership.mem t x), Algebra.FinitePresentation …
    ⊢ (RingHom.ker (Localization.awayMap f.toRingHom ↑g)).FG
  -/
  apply Algebra.FinitePresentation.ker_fG_of_surjective f'
  /-
    case hker.hf
    R : Type u_1
    S : Type u_2
    A : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing S
    inst✝³ : CommRing A
    inst✝² : Algebra R S
    inst✝¹ : Algebra R A
    inst✝ : Algebra.FinitePresentation R A
    f : AlgHom R A S
    hf : Function.Surjective ⇑f
    t : Finset A
    ht : Eq (Ideal.span ↑t) Top.top
    H : ∀ (g : Subtype fun x => Membership.mem t x), Algebra.FinitePresentation R  …
    g : ↑↑t
    f' : AlgHom R (Localization.Away ↑g) (Localization.Away (f ↑g)) := Localizatio …
    this : ∀ (g : Subtype fun x => Membership.mem t x), Algebra.FinitePresentation …
    ⊢ Function.Surjective ⇑f'
  -/
  exact IsLocalization.Away.mapₐ_surjective_of_surjective _ hf
  /-
    🎉 no goals
  -/


/-- Finite-presentation can be checked on a standard covering of the target. -/
theorem finitePresentation_ofLocalizationSpanTarget :
    OfLocalizationSpanTarget @FinitePresentation := by
  /-
    ⊢ RingHom.OfLocalizationSpanTarget @RingHom.FinitePresentation
  -/
  rw [ofLocalizationSpanTarget_iff_finite]
  /-
    ⊢ RingHom.OfLocalizationFiniteSpanTarget @RingHom.FinitePresentation
  -/
  introv R hs H
  classical
  letI := f.toAlgebra
  replace H : ∀ r : s, Algebra.FinitePresentation R (Localization.Away (r : S)) := by
    intro r; simp_rw [RingHom.FinitePresentation] at H;
    convert H r; ext; simp_rw [Algebra.smul_def]; rfl
  /-
  We already know that `S` is of finite type over `R`, so we have a surjection
  `MvPolynomial (Fin n) R →ₐ[R] S`. To reason about the kernel, we want to check it on the stalks
  of preimages of `s`. But the preimages do not necessarily span `MvPolynomial (Fin n) R`, so
  we quotient out by an ideal and apply `finitePresentation_ofLocalizationSpanTarget_aux`.
  -/
  have hfintype : Algebra.FiniteType R S := by
    apply finiteType_ofLocalizationSpanTarget f s hs
    intro r
    convert_to Algebra.FiniteType R (Localization.Away r.val)
    · rw [RingHom.FiniteType]
      constructor <;> intro h <;> convert h <;> ext <;> simp_rw [Algebra.smul_def] <;> rfl
    · infer_instance
  rw [RingHom.FinitePresentation]
  obtain ⟨n, f, hf⟩ := Algebra.FiniteType.iff_quotient_mvPolynomial''.mp hfintype
  obtain ⟨l, hl⟩ := (Finsupp.mem_span_iff_linearCombination S (s : Set S) 1).mp
      (show (1 : S) ∈ Ideal.span (s : Set S) by rw [hs]; trivial)
  choose g' hg' using (fun g : s ↦ hf g)
  choose h' hh' using (fun g : s ↦ hf (l g))
  let I : Ideal (MvPolynomial (Fin n) R) := Ideal.span { ∑ g : s, g' g * h' g - 1 }
  let A := MvPolynomial (Fin n) R ⧸ I
  have hfI : ∀ a ∈ I, f a = 0 := by
    intro p hp
    simp only [Finset.univ_eq_attach, I, Ideal.mem_span_singleton] at hp
    obtain ⟨q, rfl⟩ := hp
    simp only [map_mul, map_sub, map_sum, map_one, hg', hh']
    erw [Finsupp.linearCombination_apply_of_mem_supported S (s := s.attach)] at hl
    · rw [← hl]
      simp only [Finset.coe_sort_coe, smul_eq_mul, mul_comm, sub_self, mul_zero, zero_mul]
    · rintro a -
      simp
  let f' : A →ₐ[R] S := Ideal.Quotient.liftₐ I f hfI
  have hf' : Function.Surjective f' :=
    Ideal.Quotient.lift_surjective_of_surjective I hfI hf
  let t : Finset A := Finset.image (fun g ↦ g' g) Finset.univ
  have ht : Ideal.span (t : Set A) = ⊤ := by
    rw [Ideal.eq_top_iff_one]
    have : ∑ g : { x // x ∈ s }, g' g * h' g = (1 : A) := by
      apply eq_of_sub_eq_zero
      rw [← map_one (Ideal.Quotient.mk I), ← map_sub, Ideal.Quotient.eq_zero_iff_mem]
      apply Ideal.subset_span
      simp
    simp_rw [← this, Finset.univ_eq_attach, map_sum, map_mul]
    refine Ideal.sum_mem _ (fun g _ ↦ Ideal.mul_mem_right _ _ <| Ideal.subset_span ?_)
    simp [t]
  have : Algebra.FinitePresentation R A := by
    apply Algebra.FinitePresentation.quotient
    simp only [Finset.univ_eq_attach, I]
    exact ⟨{∑ g ∈ s.attach, g' g * h' g - 1}, by simp⟩
  have Ht (g : t) : Algebra.FinitePresentation R (Localization.Away (f' g)) := by
    have : ∃ (a : S) (hb : a ∈ s), (Ideal.Quotient.mk I) (g' ⟨a, hb⟩) = g.val := by
      obtain ⟨g, hg⟩ := g
      convert hg
      simp [A, f', t]
    obtain ⟨r, hr, hrr⟩ := this
    simp only [f']
    rw [← hrr, Ideal.Quotient.liftₐ_apply, Ideal.Quotient.lift_mk]
    simp_rw [coe_coe]
    rw [hg']
    apply H
  exact finitePresentation_ofLocalizationSpanTarget_aux f' hf' t ht Ht


/-- Being finitely-presented is a local property of rings. -/
theorem finitePresentation_isLocal : PropertyIsLocal @FinitePresentation :=
  ⟨finitePresentation_localizationPreserves.away,
    finitePresentation_ofLocalizationSpanTarget,
    finitePresentation_ofLocalizationSpanTarget.ofLocalizationSpan
      (finitePresentation_stableUnderComposition.stableUnderCompositionWithLocalizationAway
        finitePresentation_holdsForLocalizationAway).left,
    (finitePresentation_stableUnderComposition.stableUnderCompositionWithLocalizationAway
      finitePresentation_holdsForLocalizationAway).right⟩


/-- Being finitely-presented respects isomorphisms. -/
theorem finitePresentation_respectsIso : RingHom.RespectsIso @RingHom.FinitePresentation :=
  RingHom.finitePresentation_isLocal.respectsIso


/-- Being finitely-presented is stable under base change. -/
theorem finitePresentation_isStableUnderBaseChange :
    IsStableUnderBaseChange @FinitePresentation := by
  /-
    ⊢ RingHom.IsStableUnderBaseChange @RingHom.FinitePresentation
  -/
  apply IsStableUnderBaseChange.mk
    /-
      case h₁
      ⊢ RingHom.RespectsIso @RingHom.FinitePresentation
    -/
  · exact finitePresentation_respectsIso
    /-
      🎉 no goals
    -/
    /-
      case h₂
      ⊢ ∀ ⦃R S T : Type u_1⦄ [inst : CommRing R] [inst_1 : CommRing S] [inst_2 : Com …
    -/
  · introv h
    replace h : Algebra.FinitePresentation R T := by
      rw [RingHom.FinitePresentation] at h; convert h; ext; simp_rw [Algebra.smul_def]; rfl
    suffices Algebra.FinitePresentation S (S ⊗[R] T) by
      rw [RingHom.FinitePresentation]; convert this; ext; simp_rw [Algebra.smul_def]; rfl
    /-
      case h₂
      R S T : Type u_1
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : CommRing T
      inst✝¹ : Algebra R S
      inst✝ : Algebra R T
      h : Algebra.FinitePresentation R T
      ⊢ Algebra.FinitePresentation S (TensorProduct R S T)
    -/
    infer_instance
    /-
      🎉 no goals
    -/


