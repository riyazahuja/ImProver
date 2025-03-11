/-- The free locus of a module, i.e. the set of primes `p` such that `Mₚ` is free over `Rₚ`. -/
def freeLocus : Set (PrimeSpectrum R) :=
  { p | Module.Free (Localization.AtPrime p.asIdeal) (LocalizedModule p.asIdeal.primeCompl M) }


lemma mem_freeLocus {p} : p ∈ freeLocus R M ↔
    Module.Free (Localization.AtPrime p.asIdeal) (LocalizedModule p.asIdeal.primeCompl M) :=
  Iff.rfl


attribute [local instance] RingHomInvPair.of_ringEquiv in
lemma mem_freeLocus_of_isLocalization (p : PrimeSpectrum R)
    (Rₚ Mₚ) [CommRing Rₚ] [Algebra R Rₚ] [IsLocalization.AtPrime Rₚ p.asIdeal]
    [AddCommGroup Mₚ] [Module R Mₚ] (f : M →ₗ[R] Mₚ) [IsLocalizedModule p.asIdeal.primeCompl f]
    [Module Rₚ Mₚ] [IsScalarTower R Rₚ Mₚ] :
    p ∈ freeLocus R M ↔ Module.Free Rₚ Mₚ := by
  apply Module.Free.iff_of_ringEquiv (IsLocalization.algEquiv p.asIdeal.primeCompl
      (Localization.AtPrime p.asIdeal) Rₚ).toRingEquiv
  /-
    R : Type uR
    M : Type uM
    inst✝¹⁰ : CommRing R
    inst✝⁹ : AddCommGroup M
    inst✝⁸ : Module R M
    p : PrimeSpectrum R
    Rₚ : Type u_1
    Mₚ : Type u_2
    inst✝⁷ : CommRing Rₚ
    inst✝⁶ : Algebra R Rₚ
    inst✝⁵ : IsLocalization.AtPrime Rₚ p.asIdeal
    inst✝⁴ : AddCommGroup Mₚ
    inst✝³ : Module R Mₚ
    f : LinearMap (RingHom.id R) M Mₚ
    inst✝² : IsLocalizedModule p.asIdeal.primeCompl f
    inst✝¹ : Module Rₚ Mₚ
    inst✝ : IsScalarTower R Rₚ Mₚ
    ⊢ LinearEquiv (↑(IsLocalization.algEquiv p.asIdeal.primeCompl (Localization.At …
  -/
  refine { __ := IsLocalizedModule.iso p.asIdeal.primeCompl f, map_smul' := ?_ }
  /-
    R : Type uR
    M : Type uM
    inst✝¹⁰ : CommRing R
    inst✝⁹ : AddCommGroup M
    inst✝⁸ : Module R M
    p : PrimeSpectrum R
    Rₚ : Type u_1
    Mₚ : Type u_2
    inst✝⁷ : CommRing Rₚ
    inst✝⁶ : Algebra R Rₚ
    inst✝⁵ : IsLocalization.AtPrime Rₚ p.asIdeal
    inst✝⁴ : AddCommGroup Mₚ
    inst✝³ : Module R Mₚ
    f : LinearMap (RingHom.id R) M Mₚ
    inst✝² : IsLocalizedModule p.asIdeal.primeCompl f
    inst✝¹ : Module Rₚ Mₚ
    inst✝ : IsScalarTower R Rₚ Mₚ
    ⊢ ∀ (m : Localization.AtPrime p.asIdeal) (x : LocalizedModule p.asIdeal.primeC …
  -/
  intro r x
  /-
    R : Type uR
    M : Type uM
    inst✝¹⁰ : CommRing R
    inst✝⁹ : AddCommGroup M
    inst✝⁸ : Module R M
    p : PrimeSpectrum R
    Rₚ : Type u_1
    Mₚ : Type u_2
    inst✝⁷ : CommRing Rₚ
    inst✝⁶ : Algebra R Rₚ
    inst✝⁵ : IsLocalization.AtPrime Rₚ p.asIdeal
    inst✝⁴ : AddCommGroup Mₚ
    inst✝³ : Module R Mₚ
    f : LinearMap (RingHom.id R) M Mₚ
    inst✝² : IsLocalizedModule p.asIdeal.primeCompl f
    inst✝¹ : Module Rₚ Mₚ
    inst✝ : IsScalarTower R Rₚ Mₚ
    r : Localization.AtPrime p.asIdeal
    x : LocalizedModule p.asIdeal.primeCompl M
    ⊢ Eq ((↑__spread✝⁻⁰).toFun (HSMul.hSMul r x)) (HSMul.hSMul (↑(IsLocalization.a …
  -/
  obtain ⟨r, s, rfl⟩ := IsLocalization.mk'_surjective p.asIdeal.primeCompl r
  /-
    case intro.intro
    R : Type uR
    M : Type uM
    inst✝¹⁰ : CommRing R
    inst✝⁹ : AddCommGroup M
    inst✝⁸ : Module R M
    p : PrimeSpectrum R
    Rₚ : Type u_1
    Mₚ : Type u_2
    inst✝⁷ : CommRing Rₚ
    inst✝⁶ : Algebra R Rₚ
    inst✝⁵ : IsLocalization.AtPrime Rₚ p.asIdeal
    inst✝⁴ : AddCommGroup Mₚ
    inst✝³ : Module R Mₚ
    f : LinearMap (RingHom.id R) M Mₚ
    inst✝² : IsLocalizedModule p.asIdeal.primeCompl f
    inst✝¹ : Module Rₚ Mₚ
    inst✝ : IsScalarTower R Rₚ Mₚ
    x : LocalizedModule p.asIdeal.primeCompl M
    r : R
    s : Subtype fun x => Membership.mem p.asIdeal.primeCompl x
    ⊢ Eq ((↑__spread✝⁻⁰).toFun (HSMul.hSMul (IsLocalization.mk' (Localization.AtPr …
  -/
  apply ((Module.End_isUnit_iff _).mp (IsLocalizedModule.map_units f s)).1
  simp only [AddHom.toFun_eq_coe, LinearMap.coe_toAddHom, LinearEquiv.coe_coe,
    algebraMap_end_apply, AlgEquiv.toRingEquiv_eq_coe,
    AlgEquiv.toRingEquiv_toRingHom, RingHom.coe_coe, IsLocalization.algEquiv_apply,
    IsLocalization.map_id_mk']
  simp only [← map_smul, ← smul_assoc, IsLocalization.smul_mk'_self, algebraMap_smul,
    IsLocalization.map_id_mk']


attribute [local instance] RingHomInvPair.of_ringEquiv in
lemma mem_freeLocus_iff_tensor (p : PrimeSpectrum R)
    (Rₚ) [CommRing Rₚ] [Algebra R Rₚ] [IsLocalization.AtPrime Rₚ p.asIdeal] :
    p ∈ freeLocus R M ↔ Module.Free Rₚ (Rₚ ⊗[R] M) := by
  have := (isLocalizedModule_iff_isBaseChange p.asIdeal.primeCompl _ _).mpr
    (TensorProduct.isBaseChange R M Rₚ)
  /-
    R : Type uR
    M : Type uM
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    p : PrimeSpectrum R
    Rₚ : Type u_1
    inst✝² : CommRing Rₚ
    inst✝¹ : Algebra R Rₚ
    inst✝ : IsLocalization.AtPrime Rₚ p.asIdeal
    this : IsLocalizedModule p.asIdeal.primeCompl ((TensorProduct.mk R Rₚ M) 1)
    ⊢ Iff (Membership.mem (Module.freeLocus R M) p) (Module.Free Rₚ (TensorProduct …
  -/
  exact mem_freeLocus_of_isLocalization p Rₚ (f := TensorProduct.mk R Rₚ M 1)
  /-
    🎉 no goals
  -/


lemma freeLocus_congr {M'} [AddCommGroup M'] [Module R M'] (e : M ≃ₗ[R] M') :
    freeLocus R M = freeLocus R M' := by
  /-
    R : Type uR
    M : Type uM
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    M' : Type u_1
    inst✝¹ : AddCommGroup M'
    inst✝ : Module R M'
    e : LinearEquiv (RingHom.id R) M M'
    ⊢ Eq (Module.freeLocus R M) (Module.freeLocus R M')
  -/
  ext p
  exact mem_freeLocus_of_isLocalization _ _ _
    (LocalizedModule.mkLinearMap p.asIdeal.primeCompl M' ∘ₗ e.toLinearMap)


open TensorProduct in
lemma comap_freeLocus_le {A} [CommRing A] [Algebra R A] :
    comap (algebraMap R A) ⁻¹' freeLocus R M ≤ freeLocus A (A ⊗[R] M) := by
  /-
    R : Type uR
    M : Type uM
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    A : Type u_1
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    ⊢ LE.le (Set.preimage (⇑(PrimeSpectrum.comap (algebraMap R A))) (Module.freeLo …
  -/
  intro p hp
  /-
    R : Type uR
    M : Type uM
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    A : Type u_1
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    p : PrimeSpectrum A
    hp : Membership.mem (Set.preimage (⇑(PrimeSpectrum.comap (algebraMap R A))) (M …
    ⊢ Membership.mem (Module.freeLocus A (TensorProduct R A M)) p
  -/
  let Rₚ := Localization.AtPrime (comap (algebraMap R A) p).asIdeal
  /-
    R : Type uR
    M : Type uM
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    A : Type u_1
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    p : PrimeSpectrum A
    hp : Membership.mem (Set.preimage (⇑(PrimeSpectrum.comap (algebraMap R A))) (M …
    Rₚ : Type uR := Localization.AtPrime ((PrimeSpectrum.comap (algebraMap R A)) p …
    ⊢ Membership.mem (Module.freeLocus A (TensorProduct R A M)) p
  -/
  let Aₚ := Localization.AtPrime p.asIdeal
  /-
    R : Type uR
    M : Type uM
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    A : Type u_1
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    p : PrimeSpectrum A
    hp : Membership.mem (Set.preimage (⇑(PrimeSpectrum.comap (algebraMap R A))) (M …
    Rₚ : Type uR := Localization.AtPrime ((PrimeSpectrum.comap (algebraMap R A)) p …
    Aₚ : Type u_1 := Localization.AtPrime p.asIdeal
    ⊢ Membership.mem (Module.freeLocus A (TensorProduct R A M)) p
  -/
  rw [Set.mem_preimage, mem_freeLocus_iff_tensor _ Rₚ] at hp
  /-
    R : Type uR
    M : Type uM
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    A : Type u_1
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    p : PrimeSpectrum A
    Rₚ : Type uR := Localization.AtPrime ((PrimeSpectrum.comap (algebraMap R A)) p …
    hp : Module.Free Rₚ (TensorProduct R Rₚ M)
    Aₚ : Type u_1 := Localization.AtPrime p.asIdeal
    ⊢ Membership.mem (Module.freeLocus A (TensorProduct R A M)) p
  -/
  rw [mem_freeLocus_iff_tensor _ Aₚ]
  letI algebra : Algebra Rₚ Aₚ := (Localization.localRingHom
    (comap (algebraMap R A) p).asIdeal p.asIdeal (algebraMap R A) rfl).toAlgebra
  have : IsScalarTower R Rₚ Aₚ := IsScalarTower.of_algebraMap_eq'
    (by simp [Rₚ, Aₚ, algebra, RingHom.algebraMap_toAlgebra, Localization.localRingHom,
        ← IsScalarTower.algebraMap_eq])
  let e := AlgebraTensorModule.cancelBaseChange R Rₚ Aₚ Aₚ M ≪≫ₗ
    (AlgebraTensorModule.cancelBaseChange R A Aₚ Aₚ M).symm
  /-
    R : Type uR
    M : Type uM
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    A : Type u_1
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    p : PrimeSpectrum A
    Rₚ : Type uR := Localization.AtPrime ((PrimeSpectrum.comap (algebraMap R A)) p …
    hp : Module.Free Rₚ (TensorProduct R Rₚ M)
    Aₚ : Type u_1 := Localization.AtPrime p.asIdeal
    algebra : Algebra Rₚ Aₚ := (Localization.localRingHom ((PrimeSpectrum.comap (a …
    this : IsScalarTower R Rₚ Aₚ
    e : LinearEquiv (RingHom.id Aₚ) (TensorProduct Rₚ Aₚ (TensorProduct R Rₚ M)) ( …
    ⊢ Module.Free Aₚ (TensorProduct A Aₚ (TensorProduct R A M))
  -/
  exact .of_equiv e
  /-
    🎉 no goals
  -/


lemma freeLocus_localization (S : Submonoid R) :
    freeLocus (Localization S) (LocalizedModule S M) =
      comap (algebraMap R _) ⁻¹' freeLocus R M := by
  /-
    R : Type uR
    M : Type uM
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    S : Submonoid R
    ⊢ Eq (Module.freeLocus (Localization S) (LocalizedModule S M)) (Set.preimage ( …
  -/
  ext p
  /-
    case h
    R : Type uR
    M : Type uM
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    S : Submonoid R
    p : PrimeSpectrum (Localization S)
    ⊢ Iff (Membership.mem (Module.freeLocus (Localization S) (LocalizedModule S M) …
  -/
  simp only [Set.mem_preimage]
  /-
    case h
    R : Type uR
    M : Type uM
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    S : Submonoid R
    p : PrimeSpectrum (Localization S)
    ⊢ Iff (Membership.mem (Module.freeLocus (Localization S) (LocalizedModule S M) …
  -/
  let p' := p.asIdeal.comap (algebraMap R _)
  have hp' : S ≤ p'.primeCompl := fun x hx H ↦
    p.isPrime.ne_top (Ideal.eq_top_of_isUnit_mem _ H (IsLocalization.map_units _ ⟨x, hx⟩))
  /-
    case h
    R : Type uR
    M : Type uM
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    S : Submonoid R
    p : PrimeSpectrum (Localization S)
    p' : Ideal R := Ideal.comap (algebraMap R (Localization S)) p.asIdeal
    hp' : LE.le S p'.primeCompl
    ⊢ Iff (Membership.mem (Module.freeLocus (Localization S) (LocalizedModule S M) …
  -/
  let Rₚ := Localization.AtPrime p'
  /-
    case h
    R : Type uR
    M : Type uM
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    S : Submonoid R
    p : PrimeSpectrum (Localization S)
    p' : Ideal R := Ideal.comap (algebraMap R (Localization S)) p.asIdeal
    hp' : LE.le S p'.primeCompl
    Rₚ : Type uR := Localization.AtPrime p'
    ⊢ Iff (Membership.mem (Module.freeLocus (Localization S) (LocalizedModule S M) …
  -/
  let Mₚ := LocalizedModule p'.primeCompl M
  letI : Algebra (Localization S) Rₚ :=
    IsLocalization.localizationAlgebraOfSubmonoidLe _ _ S p'.primeCompl hp'
  have : IsScalarTower R (Localization S) Rₚ :=
    IsLocalization.localization_isScalarTower_of_submonoid_le ..
  have : IsLocalization.AtPrime Rₚ p.asIdeal := by
    have := IsLocalization.isLocalization_of_submonoid_le (Localization S) Rₚ _ _ hp'
    apply IsLocalization.isLocalization_of_is_exists_mul_mem _
      (Submonoid.map (algebraMap R (Localization S)) p'.primeCompl)
    · rintro _ ⟨x, hx, rfl⟩; exact hx
    · rintro ⟨x, hx⟩
      obtain ⟨x, s, rfl⟩ := IsLocalization.mk'_surjective S x
      refine ⟨algebraMap _ _ s.1, x, fun H ↦ hx ?_, by simp⟩
      rw [IsLocalization.mk'_eq_mul_mk'_one]
      exact Ideal.mul_mem_right _ _ H
  /-
    case h
    R : Type uR
    M : Type uM
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    S : Submonoid R
    p : PrimeSpectrum (Localization S)
    p' : Ideal R := Ideal.comap (algebraMap R (Localization S)) p.asIdeal
    hp' : LE.le S p'.primeCompl
    Rₚ : Type uR := Localization.AtPrime p'
    Mₚ : Type (max uR uM) := LocalizedModule p'.primeCompl M
    this✝¹ : Algebra (Localization S) Rₚ := IsLocalization.localizationAlgebraOfSu …
    this✝ : IsScalarTower R (Localization S) Rₚ
    this : IsLocalization.AtPrime Rₚ p.asIdeal
    ⊢ Iff (Membership.mem (Module.freeLocus (Localization S) (LocalizedModule S M) …
  -/
  letI : Module (Localization S) Mₚ := Module.compHom Mₚ (algebraMap _ Rₚ)
  have : IsScalarTower R (Localization S) Mₚ :=
    ⟨fun r r' m ↦ show algebraMap _ Rₚ (r • r') • m = _ by
      simp [p', Rₚ, Mₚ, Algebra.smul_def, ← IsScalarTower.algebraMap_apply, mul_smul]; rfl⟩
  have : IsScalarTower (Localization S) Rₚ Mₚ :=
    ⟨fun r r' m ↦ show _ = algebraMap _ Rₚ r • r' • m by rw [← mul_smul, ← Algebra.smul_def]⟩
  let l := (IsLocalizedModule.liftOfLE _ _ hp' (LocalizedModule.mkLinearMap S M)
    (LocalizedModule.mkLinearMap p'.primeCompl M)).extendScalarsOfIsLocalization S
    (Localization S)
  have : IsLocalizedModule p.asIdeal.primeCompl l := by
    have : IsLocalizedModule p'.primeCompl (l.restrictScalars R) :=
      inferInstanceAs (IsLocalizedModule p'.primeCompl
        (IsLocalizedModule.liftOfLE _ _ hp' (LocalizedModule.mkLinearMap S M)
        (LocalizedModule.mkLinearMap p'.primeCompl M)))
    have : IsLocalizedModule (Algebra.algebraMapSubmonoid (Localization S) p'.primeCompl) l :=
      IsLocalizedModule.of_restrictScalars p'.primeCompl ..
    apply IsLocalizedModule.of_exists_mul_mem
      (Algebra.algebraMapSubmonoid (Localization S) p'.primeCompl)
    · rintro _ ⟨x, hx, rfl⟩; exact hx
    · rintro ⟨x, hx⟩
      obtain ⟨x, s, rfl⟩ := IsLocalization.mk'_surjective S x
      refine ⟨algebraMap _ _ s.1, x, fun H ↦ hx ?_, by simp⟩
      rw [IsLocalization.mk'_eq_mul_mk'_one]
      exact Ideal.mul_mem_right _ _ H
  /-
    case h
    R : Type uR
    M : Type uM
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    S : Submonoid R
    p : PrimeSpectrum (Localization S)
    p' : Ideal R := Ideal.comap (algebraMap R (Localization S)) p.asIdeal
    hp' : LE.le S p'.primeCompl
    Rₚ : Type uR := Localization.AtPrime p'
    Mₚ : Type (max uR uM) := LocalizedModule p'.primeCompl M
    this✝⁵ : Algebra (Localization S) Rₚ := IsLocalization.localizationAlgebraOfSu …
    this✝⁴ : IsScalarTower R (Localization S) Rₚ
    this✝³ : IsLocalization.AtPrime Rₚ p.asIdeal
    this✝² : Module (Localization S) Mₚ := Module.compHom Mₚ (algebraMap (Localiza …
    this✝¹ : IsScalarTower R (Localization S) Mₚ
    this✝ : IsScalarTower (Localization S) Rₚ Mₚ
    l : LinearMap (RingHom.id (Localization S)) (LocalizedModule S M) (LocalizedMo …
    this : IsLocalizedModule p.asIdeal.primeCompl l
    ⊢ Iff (Membership.mem (Module.freeLocus (Localization S) (LocalizedModule S M) …
  -/
  rw [mem_freeLocus_of_isLocalization (R := Localization S) p Rₚ Mₚ l]
  /-
    case h
    R : Type uR
    M : Type uM
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    S : Submonoid R
    p : PrimeSpectrum (Localization S)
    p' : Ideal R := Ideal.comap (algebraMap R (Localization S)) p.asIdeal
    hp' : LE.le S p'.primeCompl
    Rₚ : Type uR := Localization.AtPrime p'
    Mₚ : Type (max uR uM) := LocalizedModule p'.primeCompl M
    this✝⁵ : Algebra (Localization S) Rₚ := IsLocalization.localizationAlgebraOfSu …
    this✝⁴ : IsScalarTower R (Localization S) Rₚ
    this✝³ : IsLocalization.AtPrime Rₚ p.asIdeal
    this✝² : Module (Localization S) Mₚ := Module.compHom Mₚ (algebraMap (Localiza …
    this✝¹ : IsScalarTower R (Localization S) Mₚ
    this✝ : IsScalarTower (Localization S) Rₚ Mₚ
    l : LinearMap (RingHom.id (Localization S)) (LocalizedModule S M) (LocalizedMo …
    this : IsLocalizedModule p.asIdeal.primeCompl l
    ⊢ Iff (Module.Free Rₚ Mₚ) (Membership.mem (Module.freeLocus R M) ((PrimeSpectr …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma freeLocus_eq_univ_iff [Module.FinitePresentation R M] :
    freeLocus R M = Set.univ ↔ Module.Projective R M := by
  /-
    R : Type uR
    M : Type uM
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Module.FinitePresentation R M
    ⊢ Iff (Eq (Module.freeLocus R M) Set.univ) (Module.Projective R M)
  -/
  simp_rw [Set.eq_univ_iff_forall, mem_freeLocus]
  exact ⟨fun H ↦ Module.projective_of_localization_maximal fun I hI ↦
    have := H ⟨I, hI.isPrime⟩; .of_free, fun H x ↦ Module.free_of_flat_of_isLocalRing⟩


lemma freeLocus_eq_univ [Module.FinitePresentation R M] [Module.Flat R M] :
    freeLocus R M = Set.univ := by
  /-
    R : Type uR
    M : Type uM
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module.FinitePresentation R M
    inst✝ : Module.Flat R M
    ⊢ Eq (Module.freeLocus R M) Set.univ
  -/
  simp_rw [Set.eq_univ_iff_forall, mem_freeLocus]
  /-
    R : Type uR
    M : Type uM
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module.FinitePresentation R M
    inst✝ : Module.Flat R M
    ⊢ ∀ (x : PrimeSpectrum R), Module.Free (Localization.AtPrime x.asIdeal) (Local …
  -/
  exact fun x ↦ Module.free_of_flat_of_isLocalRing
  /-
    🎉 no goals
  -/


lemma basicOpen_subset_freeLocus_iff [Module.FinitePresentation R M] {f : R} :
    (basicOpen f : Set (PrimeSpectrum R)) ⊆ freeLocus R M ↔
      Module.Projective (Localization.Away f) (LocalizedModule (.powers f) M) := by
  rw [← freeLocus_eq_univ_iff, freeLocus_localization,
    Set.preimage_eq_univ_iff, localization_away_comap_range _ f]


lemma isOpen_freeLocus [Module.FinitePresentation R M] :
    IsOpen (freeLocus R M) := by
  /-
    R : Type uR
    M : Type uM
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Module.FinitePresentation R M
    ⊢ IsOpen (Module.freeLocus R M)
  -/
  refine isOpen_iff_forall_mem_open.mpr fun x hx ↦ ?_
  /-
    R : Type uR
    M : Type uM
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Module.FinitePresentation R M
    x : PrimeSpectrum R
    hx : Membership.mem (Module.freeLocus R M) x
    ⊢ Exists fun t => And (HasSubset.Subset t (Module.freeLocus R M)) (And (IsOpen …
  -/
  have : Module.Free _ _ := hx
  obtain ⟨r, hr, hr', _⟩ := Module.FinitePresentation.exists_free_localizedModule_powers
    x.asIdeal.primeCompl (LocalizedModule.mkLinearMap x.asIdeal.primeCompl M)
    (Localization.AtPrime x.asIdeal)
  /-
    case intro.intro.intro
    R : Type uR
    M : Type uM
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Module.FinitePresentation R M
    x : PrimeSpectrum R
    hx : Membership.mem (Module.freeLocus R M) x
    this : Module.Free (Localization.AtPrime x.asIdeal) (LocalizedModule x.asIdeal …
    r : R
    hr : Membership.mem x.asIdeal.primeCompl r
    hr' : Module.Free (Localization (Submonoid.powers r)) (LocalizedModule (Submon …
    right✝ : Eq (Module.finrank (Localization (Submonoid.powers r)) (LocalizedModu …
    ⊢ Exists fun t => And (HasSubset.Subset t (Module.freeLocus R M)) (And (IsOpen …
  -/
  exact ⟨basicOpen r, basicOpen_subset_freeLocus_iff.mpr inferInstance, (basicOpen r).2, hr⟩
  /-
    🎉 no goals
  -/


variable (M) in
/-- The rank of `M` at the stalk of `p` is the rank of `Mₚ` as a `Rₚ`-module. -/
noncomputable
def rankAtStalk (p : PrimeSpectrum R) : ℕ :=
  Module.finrank (Localization.AtPrime p.asIdeal) (LocalizedModule p.asIdeal.primeCompl M)


lemma isLocallyConstant_rankAtStalk_freeLocus [Module.FinitePresentation R M] :
    IsLocallyConstant (fun x : freeLocus R M ↦ rankAtStalk M x.1) := by
  /-
    R : Type uR
    M : Type uM
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Module.FinitePresentation R M
    ⊢ IsLocallyConstant fun x => Module.rankAtStalk M ↑x
  -/
  refine (IsLocallyConstant.iff_exists_open _).mpr fun ⟨x, hx⟩ ↦ ?_
  /-
    R : Type uR
    M : Type uM
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Module.FinitePresentation R M
    x✝ : ↑(Module.freeLocus R M)
    x : PrimeSpectrum R
    hx : Membership.mem (Module.freeLocus R M) x
    ⊢ Exists fun U => And (IsOpen U) (And (Membership.mem U ⟨x, hx⟩) (∀ (x' : ↑(Mo …
  -/
  have : Module.Free _ _ := hx
  obtain ⟨f, hf, hf', hf''⟩ := Module.FinitePresentation.exists_free_localizedModule_powers
    x.asIdeal.primeCompl (LocalizedModule.mkLinearMap x.asIdeal.primeCompl M)
    (Localization.AtPrime x.asIdeal)
  /-
    case intro.intro.intro
    R : Type uR
    M : Type uM
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Module.FinitePresentation R M
    x✝ : ↑(Module.freeLocus R M)
    x : PrimeSpectrum R
    hx : Membership.mem (Module.freeLocus R M) x
    this : Module.Free (Localization.AtPrime x.asIdeal) (LocalizedModule x.asIdeal …
    f : R
    hf : Membership.mem x.asIdeal.primeCompl f
    hf' : Module.Free (Localization (Submonoid.powers f)) (LocalizedModule (Submon …
    hf'' : Eq (Module.finrank (Localization (Submonoid.powers f)) (LocalizedModule …
    ⊢ Exists fun U => And (IsOpen U) (And (Membership.mem U ⟨x, hx⟩) (∀ (x' : ↑(Mo …
  -/
  refine ⟨Subtype.val ⁻¹' basicOpen f, (basicOpen f).2.preimage continuous_subtype_val, hf, ?_⟩
  /-
    case intro.intro.intro
    R : Type uR
    M : Type uM
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Module.FinitePresentation R M
    x✝ : ↑(Module.freeLocus R M)
    x : PrimeSpectrum R
    hx : Membership.mem (Module.freeLocus R M) x
    this : Module.Free (Localization.AtPrime x.asIdeal) (LocalizedModule x.asIdeal …
    f : R
    hf : Membership.mem x.asIdeal.primeCompl f
    hf' : Module.Free (Localization (Submonoid.powers f)) (LocalizedModule (Submon …
    hf'' : Eq (Module.finrank (Localization (Submonoid.powers f)) (LocalizedModule …
    ⊢ ∀ (x' : ↑(Module.freeLocus R M)), Membership.mem (Set.preimage Subtype.val ↑ …
  -/
  rintro ⟨p, hp''⟩ hp
  /-
    case intro.intro.intro.mk
    R : Type uR
    M : Type uM
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Module.FinitePresentation R M
    x✝ : ↑(Module.freeLocus R M)
    x : PrimeSpectrum R
    hx : Membership.mem (Module.freeLocus R M) x
    this : Module.Free (Localization.AtPrime x.asIdeal) (LocalizedModule x.asIdeal …
    f : R
    hf : Membership.mem x.asIdeal.primeCompl f
    hf' : Module.Free (Localization (Submonoid.powers f)) (LocalizedModule (Submon …
    hf'' : Eq (Module.finrank (Localization (Submonoid.powers f)) (LocalizedModule …
    p : PrimeSpectrum R
    hp'' : Membership.mem (Module.freeLocus R M) p
    hp : Membership.mem (Set.preimage Subtype.val ↑(PrimeSpectrum.basicOpen f)) ⟨p …
    ⊢ Eq (Module.rankAtStalk M ↑⟨p, hp''⟩) (Module.rankAtStalk M ↑⟨x, hx⟩)
  -/
  let p' := Algebra.algebraMapSubmonoid (Localization (.powers f)) p.asIdeal.primeCompl
  have hp' : Submonoid.powers f ≤ p.asIdeal.primeCompl := by
    simpa [Submonoid.powers_le, Ideal.primeCompl]
  /-
    case intro.intro.intro.mk
    R : Type uR
    M : Type uM
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Module.FinitePresentation R M
    x✝ : ↑(Module.freeLocus R M)
    x : PrimeSpectrum R
    hx : Membership.mem (Module.freeLocus R M) x
    this : Module.Free (Localization.AtPrime x.asIdeal) (LocalizedModule x.asIdeal …
    f : R
    hf : Membership.mem x.asIdeal.primeCompl f
    hf' : Module.Free (Localization (Submonoid.powers f)) (LocalizedModule (Submon …
    hf'' : Eq (Module.finrank (Localization (Submonoid.powers f)) (LocalizedModule …
    p : PrimeSpectrum R
    hp'' : Membership.mem (Module.freeLocus R M) p
    hp : Membership.mem (Set.preimage Subtype.val ↑(PrimeSpectrum.basicOpen f)) ⟨p …
    p' : Submonoid (Localization (Submonoid.powers f)) := Algebra.algebraMapSubmon …
    hp' : LE.le (Submonoid.powers f) p.asIdeal.primeCompl
    ⊢ Eq (Module.rankAtStalk M ↑⟨p, hp''⟩) (Module.rankAtStalk M ↑⟨x, hx⟩)
  -/
  let Rₚ := Localization.AtPrime p.asIdeal
  /-
    case intro.intro.intro.mk
    R : Type uR
    M : Type uM
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Module.FinitePresentation R M
    x✝ : ↑(Module.freeLocus R M)
    x : PrimeSpectrum R
    hx : Membership.mem (Module.freeLocus R M) x
    this : Module.Free (Localization.AtPrime x.asIdeal) (LocalizedModule x.asIdeal …
    f : R
    hf : Membership.mem x.asIdeal.primeCompl f
    hf' : Module.Free (Localization (Submonoid.powers f)) (LocalizedModule (Submon …
    hf'' : Eq (Module.finrank (Localization (Submonoid.powers f)) (LocalizedModule …
    p : PrimeSpectrum R
    hp'' : Membership.mem (Module.freeLocus R M) p
    hp : Membership.mem (Set.preimage Subtype.val ↑(PrimeSpectrum.basicOpen f)) ⟨p …
    p' : Submonoid (Localization (Submonoid.powers f)) := Algebra.algebraMapSubmon …
    hp' : LE.le (Submonoid.powers f) p.asIdeal.primeCompl
    Rₚ : Type uR := Localization.AtPrime p.asIdeal
    ⊢ Eq (Module.rankAtStalk M ↑⟨p, hp''⟩) (Module.rankAtStalk M ↑⟨x, hx⟩)
  -/
  let Mₚ := LocalizedModule p.asIdeal.primeCompl M
  letI : Algebra (Localization.Away f) Rₚ :=
    IsLocalization.localizationAlgebraOfSubmonoidLe _ _ (.powers f) p.asIdeal.primeCompl hp'
  have : IsScalarTower R (Localization.Away f) Rₚ :=
    IsLocalization.localization_isScalarTower_of_submonoid_le ..
  /-
    case intro.intro.intro.mk
    R : Type uR
    M : Type uM
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Module.FinitePresentation R M
    x✝ : ↑(Module.freeLocus R M)
    x : PrimeSpectrum R
    hx : Membership.mem (Module.freeLocus R M) x
    this✝¹ : Module.Free (Localization.AtPrime x.asIdeal) (LocalizedModule x.asIde …
    f : R
    hf : Membership.mem x.asIdeal.primeCompl f
    hf' : Module.Free (Localization (Submonoid.powers f)) (LocalizedModule (Submon …
    hf'' : Eq (Module.finrank (Localization (Submonoid.powers f)) (LocalizedModule …
    p : PrimeSpectrum R
    hp'' : Membership.mem (Module.freeLocus R M) p
    hp : Membership.mem (Set.preimage Subtype.val ↑(PrimeSpectrum.basicOpen f)) ⟨p …
    p' : Submonoid (Localization (Submonoid.powers f)) := Algebra.algebraMapSubmon …
    hp' : LE.le (Submonoid.powers f) p.asIdeal.primeCompl
    Rₚ : Type uR := Localization.AtPrime p.asIdeal
    Mₚ : Type (max uR uM) := LocalizedModule p.asIdeal.primeCompl M
    this✝ : Algebra (Localization.Away f) Rₚ := IsLocalization.localizationAlgebra …
    this : IsScalarTower R (Localization.Away f) Rₚ
    ⊢ Eq (Module.rankAtStalk M ↑⟨p, hp''⟩) (Module.rankAtStalk M ↑⟨x, hx⟩)
  -/
  letI : Module (Localization.Away f) Mₚ := Module.compHom Mₚ (algebraMap _ Rₚ)
  have : IsScalarTower R (Localization.Away f) Mₚ :=
    ⟨fun r r' m ↦ show algebraMap _ Rₚ (r • r') • m = _ by
      simp [Rₚ, Mₚ, Algebra.smul_def, ← IsScalarTower.algebraMap_apply, mul_smul]; rfl⟩
  have : IsScalarTower (Localization.Away f) Rₚ Mₚ :=
    ⟨fun r r' m ↦ show _ = algebraMap _ Rₚ r • r' • m by rw [← mul_smul, ← Algebra.smul_def]⟩
  let l := (IsLocalizedModule.liftOfLE _ _ hp' (LocalizedModule.mkLinearMap (.powers f) M)
    (LocalizedModule.mkLinearMap p.asIdeal.primeCompl M)).extendScalarsOfIsLocalization (.powers f)
    (Localization.Away f)
  have : IsLocalization p' Rₚ :=
    IsLocalization.isLocalization_of_submonoid_le (Localization.Away f) Rₚ _ _ hp'
  have : IsLocalizedModule p.asIdeal.primeCompl (l.restrictScalars R) :=
    inferInstanceAs (IsLocalizedModule p.asIdeal.primeCompl
    ((IsLocalizedModule.liftOfLE _ _ hp' (LocalizedModule.mkLinearMap (.powers f) M)
      (LocalizedModule.mkLinearMap p.asIdeal.primeCompl M))))
  have : IsLocalizedModule (Algebra.algebraMapSubmonoid _ p.asIdeal.primeCompl) l :=
      IsLocalizedModule.of_restrictScalars p.asIdeal.primeCompl ..
  /-
    case intro.intro.intro.mk
    R : Type uR
    M : Type uM
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Module.FinitePresentation R M
    x✝ : ↑(Module.freeLocus R M)
    x : PrimeSpectrum R
    hx : Membership.mem (Module.freeLocus R M) x
    this✝⁷ : Module.Free (Localization.AtPrime x.asIdeal) (LocalizedModule x.asIde …
    f : R
    hf : Membership.mem x.asIdeal.primeCompl f
    hf' : Module.Free (Localization (Submonoid.powers f)) (LocalizedModule (Submon …
    hf'' : Eq (Module.finrank (Localization (Submonoid.powers f)) (LocalizedModule …
    p : PrimeSpectrum R
    hp'' : Membership.mem (Module.freeLocus R M) p
    hp : Membership.mem (Set.preimage Subtype.val ↑(PrimeSpectrum.basicOpen f)) ⟨p …
    p' : Submonoid (Localization (Submonoid.powers f)) := Algebra.algebraMapSubmon …
    hp' : LE.le (Submonoid.powers f) p.asIdeal.primeCompl
    Rₚ : Type uR := Localization.AtPrime p.asIdeal
    Mₚ : Type (max uR uM) := LocalizedModule p.asIdeal.primeCompl M
    this✝⁶ : Algebra (Localization.Away f) Rₚ := IsLocalization.localizationAlgebr …
    this✝⁵ : IsScalarTower R (Localization.Away f) Rₚ
    this✝⁴ : Module (Localization.Away f) Mₚ := Module.compHom Mₚ (algebraMap (Loc …
    this✝³ : IsScalarTower R (Localization.Away f) Mₚ
    this✝² : IsScalarTower (Localization.Away f) Rₚ Mₚ
    l : LinearMap (RingHom.id (Localization.Away f)) (LocalizedModule (Submonoid.p …
    this✝¹ : IsLocalization p' Rₚ
    this✝ : IsLocalizedModule p.asIdeal.primeCompl (↑R l)
    this : IsLocalizedModule (Algebra.algebraMapSubmonoid (Localization.Away f) p. …
    ⊢ Eq (Module.rankAtStalk M ↑⟨p, hp''⟩) (Module.rankAtStalk M ↑⟨x, hx⟩)
  -/
  have := Module.finrank_of_isLocalizedModule_of_free Rₚ p' l
  /-
    case intro.intro.intro.mk
    R : Type uR
    M : Type uM
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Module.FinitePresentation R M
    x✝ : ↑(Module.freeLocus R M)
    x : PrimeSpectrum R
    hx : Membership.mem (Module.freeLocus R M) x
    this✝⁸ : Module.Free (Localization.AtPrime x.asIdeal) (LocalizedModule x.asIde …
    f : R
    hf : Membership.mem x.asIdeal.primeCompl f
    hf' : Module.Free (Localization (Submonoid.powers f)) (LocalizedModule (Submon …
    hf'' : Eq (Module.finrank (Localization (Submonoid.powers f)) (LocalizedModule …
    p : PrimeSpectrum R
    hp'' : Membership.mem (Module.freeLocus R M) p
    hp : Membership.mem (Set.preimage Subtype.val ↑(PrimeSpectrum.basicOpen f)) ⟨p …
    p' : Submonoid (Localization (Submonoid.powers f)) := Algebra.algebraMapSubmon …
    hp' : LE.le (Submonoid.powers f) p.asIdeal.primeCompl
    Rₚ : Type uR := Localization.AtPrime p.asIdeal
    Mₚ : Type (max uR uM) := LocalizedModule p.asIdeal.primeCompl M
    this✝⁷ : Algebra (Localization.Away f) Rₚ := IsLocalization.localizationAlgebr …
    this✝⁶ : IsScalarTower R (Localization.Away f) Rₚ
    this✝⁵ : Module (Localization.Away f) Mₚ := Module.compHom Mₚ (algebraMap (Loc …
    this✝⁴ : IsScalarTower R (Localization.Away f) Mₚ
    this✝³ : IsScalarTower (Localization.Away f) Rₚ Mₚ
    l : LinearMap (RingHom.id (Localization.Away f)) (LocalizedModule (Submonoid.p …
    this✝² : IsLocalization p' Rₚ
    this✝¹ : IsLocalizedModule p.asIdeal.primeCompl (↑R l)
    this✝ : IsLocalizedModule (Algebra.algebraMapSubmonoid (Localization.Away f) p …
    this : Eq (Module.finrank Rₚ (LocalizedModule p.asIdeal.primeCompl M)) (Module …
    ⊢ Eq (Module.rankAtStalk M ↑⟨p, hp''⟩) (Module.rankAtStalk M ↑⟨x, hx⟩)
  -/
  simp [Rₚ, rankAtStalk, this, hf'']
  /-
    🎉 no goals
  -/


lemma isLocallyConstant_rankAtStalk [Module.FinitePresentation R M] [Module.Flat R M] :
    IsLocallyConstant (rankAtStalk (R := R) M) := by
  let e : freeLocus R M ≃ₜ PrimeSpectrum R :=
    (Homeomorph.setCongr freeLocus_eq_univ).trans (Homeomorph.Set.univ (PrimeSpectrum R))
  /-
    R : Type uR
    M : Type uM
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module.FinitePresentation R M
    inst✝ : Module.Flat R M
    e : Homeomorph (↑(Module.freeLocus R M)) (PrimeSpectrum R) := (Homeomorph.setC …
    ⊢ IsLocallyConstant (Module.rankAtStalk M)
  -/
  convert isLocallyConstant_rankAtStalk_freeLocus.comp_continuous e.symm.continuous
  /-
    🎉 no goals
  -/


