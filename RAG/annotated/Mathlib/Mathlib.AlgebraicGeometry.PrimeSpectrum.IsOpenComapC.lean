/-- Given a polynomial `f ∈ R[x]`, `imageOfDf` is the subset of `Spec R` where at least one
of the coefficients of `f` does not vanish.  Lemma `imageOfDf_eq_comap_C_compl_zeroLocus`
proves that `imageOfDf` is the image of `(zeroLocus {f})ᶜ` under the morphism
`comap C : Spec R[x] → Spec R`. -/
def imageOfDf (f : R[X]) : Set (PrimeSpectrum R) :=
  { p : PrimeSpectrum R | ∃ i : ℕ, coeff f i ∉ p.asIdeal }


theorem isOpen_imageOfDf : IsOpen (imageOfDf f) := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    f : Polynomial R
    ⊢ IsOpen (AlgebraicGeometry.Polynomial.imageOfDf f)
  -/
  rw [imageOfDf, setOf_exists fun i (x : PrimeSpectrum R) => coeff f i ∉ x.asIdeal]
  /-
    R : Type u_1
    inst✝ : CommRing R
    f : Polynomial R
    ⊢ IsOpen (Set.iUnion fun i => setOf fun x => Not (Membership.mem x.asIdeal (f. …
  -/
  exact isOpen_iUnion fun i => isOpen_basicOpen
  /-
    🎉 no goals
  -/


/-- If a point of `Spec R[x]` is not contained in the vanishing set of `f`, then its image in
`Spec R` is contained in the open set where at least one of the coefficients of `f` is non-zero.
This lemma is a reformulation of `exists_C_coeff_not_mem`. -/
theorem comap_C_mem_imageOfDf {I : PrimeSpectrum R[X]}
    (H : I ∈ (zeroLocus {f} : Set (PrimeSpectrum R[X]))ᶜ) :
    PrimeSpectrum.comap (Polynomial.C : R →+* R[X]) I ∈ imageOfDf f :=
  exists_C_coeff_not_mem (mem_compl_zeroLocus_iff_not_mem.mp H)


/-- The open set `imageOfDf f` coincides with the image of `basicOpen f` under the
morphism `C⁺ : Spec R[x] → Spec R`. -/
theorem imageOfDf_eq_comap_C_compl_zeroLocus :
    imageOfDf f = PrimeSpectrum.comap (C : R →+* R[X]) '' (zeroLocus {f})ᶜ := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    f : Polynomial R
    ⊢ Eq (AlgebraicGeometry.Polynomial.imageOfDf f) (Set.image (⇑(PrimeSpectrum.co …
  -/
  ext x
  /-
    case h
    R : Type u_1
    inst✝ : CommRing R
    f : Polynomial R
    x : PrimeSpectrum R
    ⊢ Iff (Membership.mem (AlgebraicGeometry.Polynomial.imageOfDf f) x) (Membershi …
  -/
  refine ⟨fun hx => ⟨⟨map C x.asIdeal, isPrime_map_C_of_isPrime x.isPrime⟩, ⟨?_, ?_⟩⟩, ?_⟩
    /-
      case h.refine_1
      R : Type u_1
      inst✝ : CommRing R
      f : Polynomial R
      x : PrimeSpectrum R
      hx : Membership.mem (AlgebraicGeometry.Polynomial.imageOfDf f) x
      ⊢ Membership.mem (HasCompl.compl (PrimeSpectrum.zeroLocus (Singleton.singleton …
    -/
  · rw [mem_compl_iff, mem_zeroLocus, singleton_subset_iff]
    /-
      case h.refine_1
      R : Type u_1
      inst✝ : CommRing R
      f : Polynomial R
      x : PrimeSpectrum R
      hx : Membership.mem (AlgebraicGeometry.Polynomial.imageOfDf f) x
      ⊢ Not (Membership.mem (↑{ asIdeal := Ideal.map Polynomial.C x.asIdeal, isPrime …
    -/
    cases' hx with i hi
    /-
      case h.refine_1.intro
      R : Type u_1
      inst✝ : CommRing R
      f : Polynomial R
      x : PrimeSpectrum R
      i : Nat
      hi : Not (Membership.mem x.asIdeal (f.coeff i))
      ⊢ Not (Membership.mem (↑{ asIdeal := Ideal.map Polynomial.C x.asIdeal, isPrime …
    -/
    exact fun a => hi (mem_map_C_iff.mp a i)
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2
      R : Type u_1
      inst✝ : CommRing R
      f : Polynomial R
      x : PrimeSpectrum R
      hx : Membership.mem (AlgebraicGeometry.Polynomial.imageOfDf f) x
      ⊢ Eq ((PrimeSpectrum.comap Polynomial.C) { asIdeal := Ideal.map Polynomial.C x …
    -/
  · ext x
    /-
      case h.refine_2.asIdeal.h
      R : Type u_1
      inst✝ : CommRing R
      f : Polynomial R
      x✝ : PrimeSpectrum R
      hx : Membership.mem (AlgebraicGeometry.Polynomial.imageOfDf f) x✝
      x : R
      ⊢ Iff (Membership.mem ((PrimeSpectrum.comap Polynomial.C) { asIdeal := Ideal.m …
    -/
    refine ⟨fun h => ?_, fun h => subset_span (mem_image_of_mem C.1 h)⟩
    /-
      case h.refine_2.asIdeal.h
      R : Type u_1
      inst✝ : CommRing R
      f : Polynomial R
      x✝ : PrimeSpectrum R
      hx : Membership.mem (AlgebraicGeometry.Polynomial.imageOfDf f) x✝
      x : R
      h : Membership.mem ((PrimeSpectrum.comap Polynomial.C) { asIdeal := Ideal.map  …
      ⊢ Membership.mem x✝.asIdeal x
    -/
    rw [← @coeff_C_zero R x _]
    /-
      case h.refine_2.asIdeal.h
      R : Type u_1
      inst✝ : CommRing R
      f : Polynomial R
      x✝ : PrimeSpectrum R
      hx : Membership.mem (AlgebraicGeometry.Polynomial.imageOfDf f) x✝
      x : R
      h : Membership.mem ((PrimeSpectrum.comap Polynomial.C) { asIdeal := Ideal.map  …
      ⊢ Membership.mem x✝.asIdeal ((Polynomial.C x).coeff 0)
    -/
    exact mem_map_C_iff.mp h 0
    /-
      🎉 no goals
    -/
    /-
      case h.refine_3
      R : Type u_1
      inst✝ : CommRing R
      f : Polynomial R
      x : PrimeSpectrum R
      ⊢ Membership.mem (Set.image (⇑(PrimeSpectrum.comap Polynomial.C)) (HasCompl.co …
    -/
  · rintro ⟨xli, complement, rfl⟩
    /-
      case h.refine_3.intro.intro
      R : Type u_1
      inst✝ : CommRing R
      f : Polynomial R
      xli : PrimeSpectrum (Polynomial R)
      complement : Membership.mem (HasCompl.compl (PrimeSpectrum.zeroLocus (Singleto …
      ⊢ Membership.mem (AlgebraicGeometry.Polynomial.imageOfDf f) ((PrimeSpectrum.co …
    -/
    exact comap_C_mem_imageOfDf complement
    /-
      🎉 no goals
    -/


/-- The morphism `C⁺ : Spec R[x] → Spec R` is open.
Stacks Project "Lemma 00FB", first part.

https://stacks.math.columbia.edu/tag/00FB
-/
theorem isOpenMap_comap_C : IsOpenMap (PrimeSpectrum.comap (C : R →+* R[X])) := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    ⊢ IsOpenMap ⇑(PrimeSpectrum.comap Polynomial.C)
  -/
  rintro U ⟨s, z⟩
  rw [← compl_compl U, ← z, ← iUnion_of_singleton_coe s, zeroLocus_iUnion, compl_iInter,
    image_iUnion]
  /-
    case intro
    R : Type u_1
    inst✝ : CommRing R
    U : Set (PrimeSpectrum (Polynomial R))
    s : Set (Polynomial R)
    z : Eq (PrimeSpectrum.zeroLocus s) (HasCompl.compl U)
    ⊢ IsOpen (Set.iUnion fun i => Set.image (⇑(PrimeSpectrum.comap Polynomial.C))  …
  -/
  simp_rw [← imageOfDf_eq_comap_C_compl_zeroLocus]
  /-
    case intro
    R : Type u_1
    inst✝ : CommRing R
    U : Set (PrimeSpectrum (Polynomial R))
    s : Set (Polynomial R)
    z : Eq (PrimeSpectrum.zeroLocus s) (HasCompl.compl U)
    ⊢ IsOpen (Set.iUnion fun i => AlgebraicGeometry.Polynomial.imageOfDf ↑i)
  -/
  exact isOpen_iUnion fun f => isOpen_imageOfDf
  /-
    🎉 no goals
  -/


