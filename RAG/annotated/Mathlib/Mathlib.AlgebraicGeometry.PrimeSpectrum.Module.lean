/-- `M[1/f] = 0` if and only if `D(f) ∩ Supp M = 0`. -/
lemma LocalizedModule.subsingleton_iff_disjoint {f : R} :
    Subsingleton (LocalizedModule (.powers f) M) ↔
      Disjoint ↑(PrimeSpectrum.basicOpen f) (Module.support R M) := by
  rw [subsingleton_iff_support_subset, PrimeSpectrum.basicOpen_eq_zeroLocus_compl,
    disjoint_compl_left_iff]
  /-
    R : Type u_1
    M : Type u_3
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : R
    ⊢ Iff (HasSubset.Subset (Module.support R M) (PrimeSpectrum.zeroLocus (Singlet …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma Module.stableUnderSpecialization_support :
    StableUnderSpecialization (Module.support R M) := by
  /-
    R : Type u_1
    M : Type u_3
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ⊢ StableUnderSpecialization (Module.support R M)
  -/
  intros x y e H
  /-
    R : Type u_1
    M : Type u_3
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    x y : PrimeSpectrum R
    e : Specializes x y
    H : Membership.mem (Module.support R M) x
    ⊢ Membership.mem (Module.support R M) y
  -/
  rw [mem_support_iff_exists_annihilator] at H ⊢
  /-
    R : Type u_1
    M : Type u_3
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    x y : PrimeSpectrum R
    e : Specializes x y
    H : Exists fun m => LE.le (Submodule.span R (Singleton.singleton m)).annihilat …
    ⊢ Exists fun m => LE.le (Submodule.span R (Singleton.singleton m)).annihilator …
  -/
  obtain ⟨m, hm⟩ := H
  /-
    case intro
    R : Type u_1
    M : Type u_3
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    x y : PrimeSpectrum R
    e : Specializes x y
    m : M
    hm : LE.le (Submodule.span R (Singleton.singleton m)).annihilator x.asIdeal
    ⊢ Exists fun m => LE.le (Submodule.span R (Singleton.singleton m)).annihilator …
  -/
  exact ⟨m, hm.trans ((PrimeSpectrum.le_iff_specializes _ _).mpr e)⟩
  /-
    🎉 no goals
  -/


lemma Module.isClosed_support [Module.Finite R M] :
    IsClosed (Module.support R M) := by
  /-
    R : Type u_1
    M : Type u_3
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Module.Finite R M
    ⊢ IsClosed (Module.support R M)
  -/
  rw [support_eq_zeroLocus]
  /-
    R : Type u_1
    M : Type u_3
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Module.Finite R M
    ⊢ IsClosed (PrimeSpectrum.zeroLocus ↑(Module.annihilator R M))
  -/
  apply PrimeSpectrum.isClosed_zeroLocus
  /-
    🎉 no goals
  -/


lemma Module.support_subset_preimage_comap [IsScalarTower R A M] :
    Module.support A M ⊆ PrimeSpectrum.comap (algebraMap R A) ⁻¹' Module.support R M := by
  /-
    R : Type u_1
    A : Type u_2
    M : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : CommRing A
    inst✝² : Algebra R A
    inst✝¹ : Module A M
    inst✝ : IsScalarTower R A M
    ⊢ HasSubset.Subset (Module.support A M) (Set.preimage (⇑(PrimeSpectrum.comap ( …
  -/
  intro x hx
  simp only [Set.mem_preimage, mem_support_iff', PrimeSpectrum.comap_asIdeal, Ideal.mem_comap,
    ne_eq, not_imp_not] at hx ⊢
  /-
    R : Type u_1
    A : Type u_2
    M : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : CommRing A
    inst✝² : Algebra R A
    inst✝¹ : Module A M
    inst✝ : IsScalarTower R A M
    x : PrimeSpectrum A
    hx : Exists fun m => ∀ (r : A), Eq (HSMul.hSMul r m) 0 → Membership.mem x.asId …
    ⊢ Exists fun m => ∀ (r : R), Eq (HSMul.hSMul r m) 0 → Membership.mem x.asIdeal …
  -/
  obtain ⟨m, hm⟩ := hx
  /-
    case intro
    R : Type u_1
    A : Type u_2
    M : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : CommRing A
    inst✝² : Algebra R A
    inst✝¹ : Module A M
    inst✝ : IsScalarTower R A M
    x : PrimeSpectrum A
    m : M
    hm : ∀ (r : A), Eq (HSMul.hSMul r m) 0 → Membership.mem x.asIdeal r
    ⊢ Exists fun m => ∀ (r : R), Eq (HSMul.hSMul r m) 0 → Membership.mem x.asIdeal …
  -/
  exact ⟨m, fun r e ↦ hm _ (by simpa)⟩
  /-
    🎉 no goals
  -/


