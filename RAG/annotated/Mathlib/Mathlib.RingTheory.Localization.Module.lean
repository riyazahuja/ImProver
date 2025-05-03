theorem span_eq_top_of_isLocalizedModule {v : Set M} (hv : span R v = ⊤) :
    span Rₛ (f '' v) = ⊤ := top_unique fun x _ ↦ by
  /-
    R : Type u_1
    Rₛ : Type u_2
    inst✝⁹ : CommSemiring R
    S : Submonoid R
    inst✝⁸ : CommSemiring Rₛ
    inst✝⁷ : Algebra R Rₛ
    hT : IsLocalization S Rₛ
    M : Type u_3
    M' : Type u_4
    inst✝⁶ : AddCommMonoid M
    inst✝⁵ : Module R M
    inst✝⁴ : AddCommMonoid M'
    inst✝³ : Module R M'
    inst✝² : Module Rₛ M'
    inst✝¹ : IsScalarTower R Rₛ M'
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    v : Set M
    hv : Eq (Submodule.span R v) Top.top
    x : M'
    x✝ : Membership.mem Top.top x
    ⊢ Membership.mem (Submodule.span Rₛ (Set.image (⇑f) v)) x
  -/
  obtain ⟨⟨m, s⟩, h⟩ := IsLocalizedModule.surj S f x
  rw [Submonoid.smul_def, ← algebraMap_smul Rₛ, ← Units.smul_isUnit (IsLocalization.map_units Rₛ s),
    eq_comm, ← inv_smul_eq_iff] at h
  /-
    case intro.mk
    R : Type u_1
    Rₛ : Type u_2
    inst✝⁹ : CommSemiring R
    S : Submonoid R
    inst✝⁸ : CommSemiring Rₛ
    inst✝⁷ : Algebra R Rₛ
    hT : IsLocalization S Rₛ
    M : Type u_3
    M' : Type u_4
    inst✝⁶ : AddCommMonoid M
    inst✝⁵ : Module R M
    inst✝⁴ : AddCommMonoid M'
    inst✝³ : Module R M'
    inst✝² : Module Rₛ M'
    inst✝¹ : IsScalarTower R Rₛ M'
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    v : Set M
    hv : Eq (Submodule.span R v) Top.top
    x : M'
    x✝ : Membership.mem Top.top x
    m : M
    s : Subtype fun x => Membership.mem S x
    h : Eq (HSMul.hSMul (Inv.inv ⋯.unit) (f { fst := m, snd := s }.1)) x
    ⊢ Membership.mem (Submodule.span Rₛ (Set.image (⇑f) v)) x
  -/
  refine h ▸ smul_mem _ _  (span_subset_span R Rₛ _ ?_)
  /-
    case intro.mk
    R : Type u_1
    Rₛ : Type u_2
    inst✝⁹ : CommSemiring R
    S : Submonoid R
    inst✝⁸ : CommSemiring Rₛ
    inst✝⁷ : Algebra R Rₛ
    hT : IsLocalization S Rₛ
    M : Type u_3
    M' : Type u_4
    inst✝⁶ : AddCommMonoid M
    inst✝⁵ : Module R M
    inst✝⁴ : AddCommMonoid M'
    inst✝³ : Module R M'
    inst✝² : Module Rₛ M'
    inst✝¹ : IsScalarTower R Rₛ M'
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    v : Set M
    hv : Eq (Submodule.span R v) Top.top
    x : M'
    x✝ : Membership.mem Top.top x
    m : M
    s : Subtype fun x => Membership.mem S x
    h : Eq (HSMul.hSMul (Inv.inv ⋯.unit) (f { fst := m, snd := s }.1)) x
    ⊢ Membership.mem (↑(Submodule.span R (Set.image (⇑f) v))) (f { fst := m, snd : …
  -/
  rw [← LinearMap.coe_restrictScalars R, ← LinearMap.map_span, hv]
  /-
    case intro.mk
    R : Type u_1
    Rₛ : Type u_2
    inst✝⁹ : CommSemiring R
    S : Submonoid R
    inst✝⁸ : CommSemiring Rₛ
    inst✝⁷ : Algebra R Rₛ
    hT : IsLocalization S Rₛ
    M : Type u_3
    M' : Type u_4
    inst✝⁶ : AddCommMonoid M
    inst✝⁵ : Module R M
    inst✝⁴ : AddCommMonoid M'
    inst✝³ : Module R M'
    inst✝² : Module Rₛ M'
    inst✝¹ : IsScalarTower R Rₛ M'
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    v : Set M
    hv : Eq (Submodule.span R v) Top.top
    x : M'
    x✝ : Membership.mem Top.top x
    m : M
    s : Subtype fun x => Membership.mem S x
    h : Eq (HSMul.hSMul (Inv.inv ⋯.unit) (f { fst := m, snd := s }.1)) x
    ⊢ Membership.mem (↑(Submodule.map (↑R f) Top.top)) ((↑R f) { fst := m, snd :=  …
  -/
  exact mem_map_of_mem mem_top
  /-
    🎉 no goals
  -/


theorem LinearIndependent.of_isLocalizedModule {ι : Type*} {v : ι → M}
    (hv : LinearIndependent R v) : LinearIndependent Rₛ (f ∘ v) := by
  /-
    R : Type u_3
    Rₛ : Type u_4
    inst✝⁹ : CommRing R
    S : Submonoid R
    inst✝⁸ : CommRing Rₛ
    inst✝⁷ : Algebra R Rₛ
    hT : IsLocalization S Rₛ
    M : Type u_5
    M' : Type u_6
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : AddCommGroup M'
    inst✝³ : Module R M'
    inst✝² : Module Rₛ M'
    inst✝¹ : IsScalarTower R Rₛ M'
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    ι : Type u_7
    v : ι → M
    hv : LinearIndependent R v
    ⊢ LinearIndependent Rₛ (Function.comp (⇑f) v)
  -/
  rw [linearIndependent_iff'] at hv ⊢
  /-
    R : Type u_3
    Rₛ : Type u_4
    inst✝⁹ : CommRing R
    S : Submonoid R
    inst✝⁸ : CommRing Rₛ
    inst✝⁷ : Algebra R Rₛ
    hT : IsLocalization S Rₛ
    M : Type u_5
    M' : Type u_6
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : AddCommGroup M'
    inst✝³ : Module R M'
    inst✝² : Module Rₛ M'
    inst✝¹ : IsScalarTower R Rₛ M'
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    ι : Type u_7
    v : ι → M
    hv : ∀ (s : Finset ι) (g : ι → R), Eq (s.sum fun i => HSMul.hSMul (g i) (v i)) …
    ⊢ ∀ (s : Finset ι) (g : ι → Rₛ), Eq (s.sum fun i => HSMul.hSMul (g i) (Functio …
  -/
  intro t g hg i hi
  /-
    R : Type u_3
    Rₛ : Type u_4
    inst✝⁹ : CommRing R
    S : Submonoid R
    inst✝⁸ : CommRing Rₛ
    inst✝⁷ : Algebra R Rₛ
    hT : IsLocalization S Rₛ
    M : Type u_5
    M' : Type u_6
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : AddCommGroup M'
    inst✝³ : Module R M'
    inst✝² : Module Rₛ M'
    inst✝¹ : IsScalarTower R Rₛ M'
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    ι : Type u_7
    v : ι → M
    hv : ∀ (s : Finset ι) (g : ι → R), Eq (s.sum fun i => HSMul.hSMul (g i) (v i)) …
    t : Finset ι
    g : ι → Rₛ
    hg : Eq (t.sum fun i => HSMul.hSMul (g i) (Function.comp (⇑f) v i)) 0
    i : ι
    hi : Membership.mem t i
    ⊢ Eq (g i) 0
  -/
  choose! a g' hg' using IsLocalization.exist_integer_multiples S t g
  have h0 : f (∑ i ∈ t, g' i • v i) = 0 := by
    apply_fun ((a : R) • ·) at hg
    rw [smul_zero, Finset.smul_sum] at hg
    rw [map_sum, ← hg]
    refine Finset.sum_congr rfl fun i hi => ?_
    rw [← smul_assoc, ← hg' i hi, map_smul, Function.comp_apply, algebraMap_smul]
  /-
    R : Type u_3
    Rₛ : Type u_4
    inst✝⁹ : CommRing R
    S : Submonoid R
    inst✝⁸ : CommRing Rₛ
    inst✝⁷ : Algebra R Rₛ
    hT : IsLocalization S Rₛ
    M : Type u_5
    M' : Type u_6
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : AddCommGroup M'
    inst✝³ : Module R M'
    inst✝² : Module Rₛ M'
    inst✝¹ : IsScalarTower R Rₛ M'
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    ι : Type u_7
    v : ι → M
    hv : ∀ (s : Finset ι) (g : ι → R), Eq (s.sum fun i => HSMul.hSMul (g i) (v i)) …
    t : Finset ι
    g : ι → Rₛ
    hg : Eq (t.sum fun i => HSMul.hSMul (g i) (Function.comp (⇑f) v i)) 0
    i : ι
    hi : Membership.mem t i
    a : Subtype fun x => Membership.mem S x
    g' : ι → R
    hg' : ∀ (i : ι), Membership.mem t i → Eq ((algebraMap R Rₛ) (g' i)) (HSMul.hSM …
    h0 : Eq (f (t.sum fun i => HSMul.hSMul (g' i) (v i))) 0
    ⊢ Eq (g i) 0
  -/
  obtain ⟨s, hs⟩ := (IsLocalizedModule.eq_zero_iff S f).mp h0
  /-
    case intro
    R : Type u_3
    Rₛ : Type u_4
    inst✝⁹ : CommRing R
    S : Submonoid R
    inst✝⁸ : CommRing Rₛ
    inst✝⁷ : Algebra R Rₛ
    hT : IsLocalization S Rₛ
    M : Type u_5
    M' : Type u_6
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : AddCommGroup M'
    inst✝³ : Module R M'
    inst✝² : Module Rₛ M'
    inst✝¹ : IsScalarTower R Rₛ M'
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    ι : Type u_7
    v : ι → M
    hv : ∀ (s : Finset ι) (g : ι → R), Eq (s.sum fun i => HSMul.hSMul (g i) (v i)) …
    t : Finset ι
    g : ι → Rₛ
    hg : Eq (t.sum fun i => HSMul.hSMul (g i) (Function.comp (⇑f) v i)) 0
    i : ι
    hi : Membership.mem t i
    a : Subtype fun x => Membership.mem S x
    g' : ι → R
    hg' : ∀ (i : ι), Membership.mem t i → Eq ((algebraMap R Rₛ) (g' i)) (HSMul.hSM …
    h0 : Eq (f (t.sum fun i => HSMul.hSMul (g' i) (v i))) 0
    s : Subtype fun x => Membership.mem S x
    hs : Eq (HSMul.hSMul s (t.sum fun i => HSMul.hSMul (g' i) (v i))) 0
    ⊢ Eq (g i) 0
  -/
  simp_rw [Finset.smul_sum, Submonoid.smul_def, smul_smul] at hs
  /-
    case intro
    R : Type u_3
    Rₛ : Type u_4
    inst✝⁹ : CommRing R
    S : Submonoid R
    inst✝⁸ : CommRing Rₛ
    inst✝⁷ : Algebra R Rₛ
    hT : IsLocalization S Rₛ
    M : Type u_5
    M' : Type u_6
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : AddCommGroup M'
    inst✝³ : Module R M'
    inst✝² : Module Rₛ M'
    inst✝¹ : IsScalarTower R Rₛ M'
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    ι : Type u_7
    v : ι → M
    hv : ∀ (s : Finset ι) (g : ι → R), Eq (s.sum fun i => HSMul.hSMul (g i) (v i)) …
    t : Finset ι
    g : ι → Rₛ
    hg : Eq (t.sum fun i => HSMul.hSMul (g i) (Function.comp (⇑f) v i)) 0
    i : ι
    hi : Membership.mem t i
    a : Subtype fun x => Membership.mem S x
    g' : ι → R
    hg' : ∀ (i : ι), Membership.mem t i → Eq ((algebraMap R Rₛ) (g' i)) (HSMul.hSM …
    h0 : Eq (f (t.sum fun i => HSMul.hSMul (g' i) (v i))) 0
    s : Subtype fun x => Membership.mem S x
    hs : Eq (t.sum fun x => HSMul.hSMul (HMul.hMul (↑s) (g' x)) (v x)) 0
    ⊢ Eq (g i) 0
  -/
  specialize hv t _ hs i hi
  /-
    case intro
    R : Type u_3
    Rₛ : Type u_4
    inst✝⁹ : CommRing R
    S : Submonoid R
    inst✝⁸ : CommRing Rₛ
    inst✝⁷ : Algebra R Rₛ
    hT : IsLocalization S Rₛ
    M : Type u_5
    M' : Type u_6
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : AddCommGroup M'
    inst✝³ : Module R M'
    inst✝² : Module Rₛ M'
    inst✝¹ : IsScalarTower R Rₛ M'
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    ι : Type u_7
    v : ι → M
    t : Finset ι
    g : ι → Rₛ
    hg : Eq (t.sum fun i => HSMul.hSMul (g i) (Function.comp (⇑f) v i)) 0
    i : ι
    hi : Membership.mem t i
    a : Subtype fun x => Membership.mem S x
    g' : ι → R
    hg' : ∀ (i : ι), Membership.mem t i → Eq ((algebraMap R Rₛ) (g' i)) (HSMul.hSM …
    h0 : Eq (f (t.sum fun i => HSMul.hSMul (g' i) (v i))) 0
    s : Subtype fun x => Membership.mem S x
    hs : Eq (t.sum fun x => HSMul.hSMul (HMul.hMul (↑s) (g' x)) (v x)) 0
    hv : Eq (HMul.hMul (↑s) (g' i)) 0
    ⊢ Eq (g i) 0
  -/
  rw [← (IsLocalization.map_units Rₛ a).mul_right_eq_zero, ← Algebra.smul_def, ← hg' i hi]
  /-
    case intro
    R : Type u_3
    Rₛ : Type u_4
    inst✝⁹ : CommRing R
    S : Submonoid R
    inst✝⁸ : CommRing Rₛ
    inst✝⁷ : Algebra R Rₛ
    hT : IsLocalization S Rₛ
    M : Type u_5
    M' : Type u_6
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : AddCommGroup M'
    inst✝³ : Module R M'
    inst✝² : Module Rₛ M'
    inst✝¹ : IsScalarTower R Rₛ M'
    f : LinearMap (RingHom.id R) M M'
    inst✝ : IsLocalizedModule S f
    ι : Type u_7
    v : ι → M
    t : Finset ι
    g : ι → Rₛ
    hg : Eq (t.sum fun i => HSMul.hSMul (g i) (Function.comp (⇑f) v i)) 0
    i : ι
    hi : Membership.mem t i
    a : Subtype fun x => Membership.mem S x
    g' : ι → R
    hg' : ∀ (i : ι), Membership.mem t i → Eq ((algebraMap R Rₛ) (g' i)) (HSMul.hSM …
    h0 : Eq (f (t.sum fun i => HSMul.hSMul (g' i) (v i))) 0
    s : Subtype fun x => Membership.mem S x
    hs : Eq (t.sum fun x => HSMul.hSMul (HMul.hMul (↑s) (g' x)) (v x)) 0
    hv : Eq (HMul.hMul (↑s) (g' i)) 0
    ⊢ Eq ((algebraMap R Rₛ) (g' i)) 0
  -/
  exact (IsLocalization.map_eq_zero_iff S _ _).2 ⟨s, hv⟩
  /-
    🎉 no goals
  -/


theorem LinearIndependent.localization {ι : Type*} {b : ι → M} (hli : LinearIndependent R b) :
    LinearIndependent Rₛ b := by
  /-
    R : Type u_3
    Rₛ : Type u_4
    inst✝⁶ : CommRing R
    S : Submonoid R
    inst✝⁵ : CommRing Rₛ
    inst✝⁴ : Algebra R Rₛ
    hT : IsLocalization S Rₛ
    M : Type u_5
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module Rₛ M
    inst✝ : IsScalarTower R Rₛ M
    ι : Type u_7
    b : ι → M
    hli : LinearIndependent R b
    ⊢ LinearIndependent Rₛ b
  -/
  have := isLocalizedModule_id S M Rₛ
  /-
    R : Type u_3
    Rₛ : Type u_4
    inst✝⁶ : CommRing R
    S : Submonoid R
    inst✝⁵ : CommRing Rₛ
    inst✝⁴ : Algebra R Rₛ
    hT : IsLocalization S Rₛ
    M : Type u_5
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module Rₛ M
    inst✝ : IsScalarTower R Rₛ M
    ι : Type u_7
    b : ι → M
    hli : LinearIndependent R b
    this : IsLocalizedModule S LinearMap.id
    ⊢ LinearIndependent Rₛ b
  -/
  exact hli.of_isLocalizedModule Rₛ S .id
  /-
    🎉 no goals
  -/


/-- If `M` has an `R`-basis, then localizing `M` at `S` has a basis over `R` localized at `S`. -/
noncomputable def Basis.ofIsLocalizedModule : Basis ι Rₛ Mₛ :=
  .mk (b.linearIndependent.of_isLocalizedModule Rₛ S f) <| by
    /-
      R : Type u_1
      Rₛ : Type u_2
      inst✝⁹ : CommRing R
      S : Submonoid R
      inst✝⁸ : CommRing Rₛ
      inst✝⁷ : Algebra R Rₛ
      hT : IsLocalization S Rₛ
      M : Type u_3
      Mₛ : Type u_4
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : AddCommGroup Mₛ
      inst✝⁴ : Module R M
      inst✝³ : Module R Mₛ
      inst✝² : Module Rₛ Mₛ
      f : LinearMap (RingHom.id R) M Mₛ
      inst✝¹ : IsLocalizedModule S f
      inst✝ : IsScalarTower R Rₛ Mₛ
      ι : Type u_5
      b : Basis ι R M
      ⊢ LE.le Top.top (Submodule.span Rₛ (Set.range (Function.comp ⇑f ⇑b)))
    -/
    rw [Set.range_comp, span_eq_top_of_isLocalizedModule Rₛ S _ b.span_eq]
    /-
      🎉 no goals
    -/


@[simp]
theorem Basis.ofIsLocalizedModule_apply (i : ι) : b.ofIsLocalizedModule Rₛ S f i = f (b i) := by
  /-
    R : Type u_1
    Rₛ : Type u_2
    inst✝⁹ : CommRing R
    S : Submonoid R
    inst✝⁸ : CommRing Rₛ
    inst✝⁷ : Algebra R Rₛ
    hT : IsLocalization S Rₛ
    M : Type u_3
    Mₛ : Type u_4
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : AddCommGroup Mₛ
    inst✝⁴ : Module R M
    inst✝³ : Module R Mₛ
    inst✝² : Module Rₛ Mₛ
    f : LinearMap (RingHom.id R) M Mₛ
    inst✝¹ : IsLocalizedModule S f
    inst✝ : IsScalarTower R Rₛ Mₛ
    ι : Type u_5
    b : Basis ι R M
    i : ι
    ⊢ Eq ((Basis.ofIsLocalizedModule Rₛ S f b) i) (f (b i))
  -/
  rw [ofIsLocalizedModule, coe_mk, Function.comp_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem Basis.ofIsLocalizedModule_repr_apply (m : M) (i : ι) :
    ((b.ofIsLocalizedModule Rₛ S f).repr (f m)) i = algebraMap R Rₛ (b.repr m i) := by
  suffices ((b.ofIsLocalizedModule Rₛ S f).repr.toLinearMap.restrictScalars R) ∘ₗ f =
      Finsupp.mapRange.linearMap (Algebra.linearMap R Rₛ) ∘ₗ b.repr.toLinearMap by
    exact DFunLike.congr_fun (LinearMap.congr_fun this m) i
  /-
    R : Type u_1
    Rₛ : Type u_2
    inst✝⁹ : CommRing R
    S : Submonoid R
    inst✝⁸ : CommRing Rₛ
    inst✝⁷ : Algebra R Rₛ
    hT : IsLocalization S Rₛ
    M : Type u_3
    Mₛ : Type u_4
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : AddCommGroup Mₛ
    inst✝⁴ : Module R M
    inst✝³ : Module R Mₛ
    inst✝² : Module Rₛ Mₛ
    f : LinearMap (RingHom.id R) M Mₛ
    inst✝¹ : IsLocalizedModule S f
    inst✝ : IsScalarTower R Rₛ Mₛ
    ι : Type u_5
    b : Basis ι R M
    m : M
    i : ι
    ⊢ Eq ((↑R ↑(Basis.ofIsLocalizedModule Rₛ S f b).repr).comp f) ((Finsupp.mapRan …
  -/
  refine Basis.ext b fun i ↦ ?_
  rw [LinearMap.coe_comp, Function.comp_apply, LinearMap.coe_restrictScalars,
    LinearEquiv.coe_coe, ← b.ofIsLocalizedModule_apply Rₛ S f, repr_self, LinearMap.coe_comp,
    Function.comp_apply, LinearEquiv.coe_coe, repr_self, Finsupp.mapRange.linearMap_apply,
    Finsupp.mapRange_single, Algebra.linearMap_apply, map_one]


theorem Basis.ofIsLocalizedModule_span :
    span R (Set.range (b.ofIsLocalizedModule Rₛ S f)) = LinearMap.range f := by
  calc span R (Set.range (b.ofIsLocalizedModule Rₛ S f))
    _ = span R (f '' (Set.range b)) := by congr; ext; simp
    _ = map f (span R (Set.range b)) := by rw [Submodule.map_span]
    _ = LinearMap.range f := by rw [b.span_eq, Submodule.map_top]


theorem LinearIndependent.localization_localization {ι : Type*} {v : ι → A}
    (hv : LinearIndependent R v) : LinearIndependent Rₛ ((algebraMap A Aₛ) ∘ v) :=
  hv.of_isLocalizedModule Rₛ S (IsScalarTower.toAlgHom R A Aₛ).toLinearMap


theorem span_eq_top_localization_localization {v : Set A} (hv : span R v = ⊤) :
    span Rₛ (algebraMap A Aₛ '' v) = ⊤ :=
  span_eq_top_of_isLocalizedModule Rₛ S (IsScalarTower.toAlgHom R A Aₛ).toLinearMap hv


/-- If `A` has an `R`-basis, then localizing `A` at `S` has a basis over `R` localized at `S`.

A suitable instance for `[Algebra A Aₛ]` is `localizationAlgebra`.
-/
noncomputable def Basis.localizationLocalization {ι : Type*} (b : Basis ι R A) : Basis ι Rₛ Aₛ :=
  b.ofIsLocalizedModule Rₛ S (IsScalarTower.toAlgHom R A Aₛ).toLinearMap


@[simp]
theorem Basis.localizationLocalization_apply {ι : Type*} (b : Basis ι R A) (i) :
    b.localizationLocalization Rₛ S Aₛ i = algebraMap A Aₛ (b i) :=
  b.ofIsLocalizedModule_apply Rₛ S _ i


@[simp]
theorem Basis.localizationLocalization_repr_algebraMap {ι : Type*} (b : Basis ι R A) (x i) :
    (b.localizationLocalization Rₛ S Aₛ).repr (algebraMap A Aₛ x) i =
      algebraMap R Rₛ (b.repr x i) := b.ofIsLocalizedModule_repr_apply Rₛ S _ _ i


theorem Basis.localizationLocalization_span {ι : Type*} (b : Basis ι R A) :
    Submodule.span R (Set.range (b.localizationLocalization Rₛ S Aₛ)) =
      LinearMap.range (IsScalarTower.toAlgHom R A Aₛ) := b.ofIsLocalizedModule_span Rₛ S _


theorem LinearIndependent.iff_fractionRing {ι : Type*} {b : ι → V} :
    LinearIndependent R b ↔ LinearIndependent K b :=
  ⟨LinearIndependent.localization K R⁰,
    LinearIndependent.restrict_scalars (smul_left_injective R one_ne_zero)⟩


/-- An `R`-linear map between two `S⁻¹R`-modules is actually `S⁻¹R`-linear. -/
def LinearMap.extendScalarsOfIsLocalization (f : M →ₗ[R] N) : M →ₗ[A] N where
  toFun := f
  map_add' := f.map_add
  map_smul' := (IsLocalization.linearMap_compatibleSMul S A M N).map_smul _


@[simp] lemma LinearMap.restrictScalars_extendScalarsOfIsLocalization (f : M →ₗ[R] N) :
    (f.extendScalarsOfIsLocalization S A).restrictScalars R = f := rfl


@[simp] lemma LinearMap.extendScalarsOfIsLocalization_apply (f : M →ₗ[A] N) :
    f.extendScalarsOfIsLocalization S A = f := rfl


@[simp] lemma LinearMap.extendScalarsOfIsLocalization_apply' (f : M →ₗ[R] N) (x : M) :
    (f.extendScalarsOfIsLocalization S A) x = f x := rfl


/-- The `S⁻¹R`-linear maps between two `S⁻¹R`-modules are exactly the `R`-linear maps. -/
@[simps]
def LinearMap.extendScalarsOfIsLocalizationEquiv : (M →ₗ[R] N) ≃ₗ[A] (M →ₗ[A] N) where
  toFun := LinearMap.extendScalarsOfIsLocalization S A
  invFun := LinearMap.restrictScalars R
                 /-
                   R✝ : Type u_1
                   Rₛ : Type u_2
                   R : Type u_3
                   inst✝¹¹ : CommSemiring R
                   S : Submonoid R
                   A : Type u_4
                   inst✝¹⁰ : CommSemiring A
                   inst✝⁹ : Algebra R A
                   inst✝⁸ : IsLocalization S A
                   M : Type u_5
                   N : Type u_6
                   inst✝⁷ : AddCommMonoid M
                   inst✝⁶ : Module R M
                   inst✝⁵ : Module A M
                   inst✝⁴ : IsScalarTower R A M
                   inst✝³ : AddCommMonoid N
                   inst✝² : Module R N
                   inst✝¹ : Module A N
                   inst✝ : IsScalarTower R A N
                   ⊢ ∀ (x y : LinearMap (RingHom.id R) M N), Eq (LinearMap.extendScalarsOfIsLocal …
                 -/
  map_add' := by intros; ext; simp
                              /-
                                🎉 no goals
                              -/
                  /-
                    R✝ : Type u_1
                    Rₛ : Type u_2
                    R : Type u_3
                    inst✝¹¹ : CommSemiring R
                    S : Submonoid R
                    A : Type u_4
                    inst✝¹⁰ : CommSemiring A
                    inst✝⁹ : Algebra R A
                    inst✝⁸ : IsLocalization S A
                    M : Type u_5
                    N : Type u_6
                    inst✝⁷ : AddCommMonoid M
                    inst✝⁶ : Module R M
                    inst✝⁵ : Module A M
                    inst✝⁴ : IsScalarTower R A M
                    inst✝³ : AddCommMonoid N
                    inst✝² : Module R N
                    inst✝¹ : Module A N
                    inst✝ : IsScalarTower R A N
                    ⊢ ∀ (m : A) (x : LinearMap (RingHom.id R) M N), Eq ({ toFun := LinearMap.exten …
                  -/
  map_smul' := by intros; ext; simp
                               /-
                                 🎉 no goals
                               -/
                 /-
                   R✝ : Type u_1
                   Rₛ : Type u_2
                   R : Type u_3
                   inst✝¹¹ : CommSemiring R
                   S : Submonoid R
                   A : Type u_4
                   inst✝¹⁰ : CommSemiring A
                   inst✝⁹ : Algebra R A
                   inst✝⁸ : IsLocalization S A
                   M : Type u_5
                   N : Type u_6
                   inst✝⁷ : AddCommMonoid M
                   inst✝⁶ : Module R M
                   inst✝⁵ : Module A M
                   inst✝⁴ : IsScalarTower R A M
                   inst✝³ : AddCommMonoid N
                   inst✝² : Module R N
                   inst✝¹ : Module A N
                   inst✝ : IsScalarTower R A N
                   ⊢ Function.LeftInverse ↑R { toFun := LinearMap.extendScalarsOfIsLocalization S …
                 -/
  left_inv := by intros _; ext; simp
                                /-
                                  🎉 no goals
                                -/
                  /-
                    R✝ : Type u_1
                    Rₛ : Type u_2
                    R : Type u_3
                    inst✝¹¹ : CommSemiring R
                    S : Submonoid R
                    A : Type u_4
                    inst✝¹⁰ : CommSemiring A
                    inst✝⁹ : Algebra R A
                    inst✝⁸ : IsLocalization S A
                    M : Type u_5
                    N : Type u_6
                    inst✝⁷ : AddCommMonoid M
                    inst✝⁶ : Module R M
                    inst✝⁵ : Module A M
                    inst✝⁴ : IsScalarTower R A M
                    inst✝³ : AddCommMonoid N
                    inst✝² : Module R N
                    inst✝¹ : Module A N
                    inst✝ : IsScalarTower R A N
                    ⊢ Function.RightInverse ↑R { toFun := LinearMap.extendScalarsOfIsLocalization  …
                  -/
  right_inv := by intros _; ext; simp
                                 /-
                                   🎉 no goals
                                 -/


/-- A linear map `M →ₗ[R] N` gives a map between localized modules `Mₛ →ₗ[Rₛ] Nₛ`. -/
@[simps!]
noncomputable
def mapExtendScalars : (M →ₗ[R] N) →ₗ[R] (M' →ₗ[Rₛ] N') :=
  ((LinearMap.extendScalarsOfIsLocalizationEquiv S Rₛ).restrictScalars R).toLinearMap ∘ₗ map S f g


/-- A linear map `M →ₗ[R] N` gives a map between localized modules `Mₛ →ₗ[Rₛ] Nₛ`. -/
noncomputable
def LocalizedModule.map :
    (M →ₗ[R] N) →ₗ[R] (LocalizedModule S M →ₗ[Localization S] LocalizedModule S N) :=
  IsLocalizedModule.mapExtendScalars S (LocalizedModule.mkLinearMap S M)
    (LocalizedModule.mkLinearMap S N) (Localization S)


@[simp]
lemma LocalizedModule.map_mk (f : M →ₗ[R] N) (x y) :
    map S f (.mk x y) = LocalizedModule.mk (f x) y := by
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    N : Type u_3
    inst✝¹ : AddCommMonoid N
    inst✝ : Module R N
    f : LinearMap (RingHom.id R) M N
    x : M
    y : Subtype fun x => Membership.mem S x
    ⊢ Eq (((LocalizedModule.map S) f) (LocalizedModule.mk x y)) (LocalizedModule.m …
  -/
  rw [IsLocalizedModule.mk_eq_mk', IsLocalizedModule.mk_eq_mk']
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    N : Type u_3
    inst✝¹ : AddCommMonoid N
    inst✝ : Module R N
    f : LinearMap (RingHom.id R) M N
    x : M
    y : Subtype fun x => Membership.mem S x
    ⊢ Eq (((LocalizedModule.map S) f) (IsLocalizedModule.mk' (LocalizedModule.mkLi …
  -/
  exact IsLocalizedModule.map_mk' _ _ _ _ _ _
  /-
    🎉 no goals
  -/


@[simp]
lemma LocalizedModule.map_id :
    LocalizedModule.map S (.id (R := R) (M := M)) = LinearMap.id :=
  LinearMap.ext fun x ↦ LinearMap.congr_fun (IsLocalizedModule.map_id S (mkLinearMap S M)) x


lemma LocalizedModule.map_injective (l : M →ₗ[R] N) (hl : Function.Injective l) :
    Function.Injective (map S l) :=
  IsLocalizedModule.map_injective S (mkLinearMap S M) (mkLinearMap S N) l hl


lemma LocalizedModule.map_surjective (l : M →ₗ[R] N) (hl : Function.Surjective l) :
    Function.Surjective (map S l) :=
  IsLocalizedModule.map_surjective S (mkLinearMap S M) (mkLinearMap S N) l hl


lemma LocalizedModule.restrictScalars_map_eq {M' N' : Type*} [AddCommMonoid M'] [AddCommMonoid N']
    [Module R M'] [Module R N'] (g₁ : M →ₗ[R] M') (g₂ : N →ₗ[R] N')
    [IsLocalizedModule S g₁] [IsLocalizedModule S g₂]
    (l : M →ₗ[R] N) :
    (map S l).restrictScalars R = (IsLocalizedModule.iso S g₂).symm ∘ₗ
      IsLocalizedModule.map S g₁ g₂ l ∘ₗ IsLocalizedModule.iso S g₁ := by
  /-
    R : Type u_1
    inst✝¹⁰ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    inst✝⁹ : AddCommMonoid M
    inst✝⁸ : Module R M
    N : Type u_5
    inst✝⁷ : AddCommMonoid N
    inst✝⁶ : Module R N
    M' : Type u_3
    N' : Type u_4
    inst✝⁵ : AddCommMonoid M'
    inst✝⁴ : AddCommMonoid N'
    inst✝³ : Module R M'
    inst✝² : Module R N'
    g₁ : LinearMap (RingHom.id R) M M'
    g₂ : LinearMap (RingHom.id R) N N'
    inst✝¹ : IsLocalizedModule S g₁
    inst✝ : IsLocalizedModule S g₂
    l : LinearMap (RingHom.id R) M N
    ⊢ Eq (↑R ((LocalizedModule.map S) l)) ((↑(IsLocalizedModule.iso S g₂).symm).co …
  -/
  rw [LinearEquiv.eq_toLinearMap_symm_comp, ← LinearEquiv.comp_toLinearMap_symm_eq]
  /-
    R : Type u_1
    inst✝¹⁰ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    inst✝⁹ : AddCommMonoid M
    inst✝⁸ : Module R M
    N : Type u_5
    inst✝⁷ : AddCommMonoid N
    inst✝⁶ : Module R N
    M' : Type u_3
    N' : Type u_4
    inst✝⁵ : AddCommMonoid M'
    inst✝⁴ : AddCommMonoid N'
    inst✝³ : Module R M'
    inst✝² : Module R N'
    g₁ : LinearMap (RingHom.id R) M M'
    g₂ : LinearMap (RingHom.id R) N N'
    inst✝¹ : IsLocalizedModule S g₁
    inst✝ : IsLocalizedModule S g₂
    l : LinearMap (RingHom.id R) M N
    ⊢ Eq (((↑(IsLocalizedModule.iso S g₂)).comp (↑R ((LocalizedModule.map S) l))). …
  -/
  apply IsLocalizedModule.linearMap_ext S g₁ g₂
  /-
    case h
    R : Type u_1
    inst✝¹⁰ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    inst✝⁹ : AddCommMonoid M
    inst✝⁸ : Module R M
    N : Type u_5
    inst✝⁷ : AddCommMonoid N
    inst✝⁶ : Module R N
    M' : Type u_3
    N' : Type u_4
    inst✝⁵ : AddCommMonoid M'
    inst✝⁴ : AddCommMonoid N'
    inst✝³ : Module R M'
    inst✝² : Module R N'
    g₁ : LinearMap (RingHom.id R) M M'
    g₂ : LinearMap (RingHom.id R) N N'
    inst✝¹ : IsLocalizedModule S g₁
    inst✝ : IsLocalizedModule S g₂
    l : LinearMap (RingHom.id R) M N
    ⊢ Eq ((((↑(IsLocalizedModule.iso S g₂)).comp (↑R ((LocalizedModule.map S) l))) …
  -/
  rw [LinearMap.comp_assoc, IsLocalizedModule.iso_symm_comp]
  /-
    case h
    R : Type u_1
    inst✝¹⁰ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    inst✝⁹ : AddCommMonoid M
    inst✝⁸ : Module R M
    N : Type u_5
    inst✝⁷ : AddCommMonoid N
    inst✝⁶ : Module R N
    M' : Type u_3
    N' : Type u_4
    inst✝⁵ : AddCommMonoid M'
    inst✝⁴ : AddCommMonoid N'
    inst✝³ : Module R M'
    inst✝² : Module R N'
    g₁ : LinearMap (RingHom.id R) M M'
    g₂ : LinearMap (RingHom.id R) N N'
    inst✝¹ : IsLocalizedModule S g₁
    inst✝ : IsLocalizedModule S g₂
    l : LinearMap (RingHom.id R) M N
    ⊢ Eq (((↑(IsLocalizedModule.iso S g₂)).comp (↑R ((LocalizedModule.map S) l))). …
  -/
  ext
  /-
    case h.h
    R : Type u_1
    inst✝¹⁰ : CommSemiring R
    S : Submonoid R
    M : Type u_2
    inst✝⁹ : AddCommMonoid M
    inst✝⁸ : Module R M
    N : Type u_5
    inst✝⁷ : AddCommMonoid N
    inst✝⁶ : Module R N
    M' : Type u_3
    N' : Type u_4
    inst✝⁵ : AddCommMonoid M'
    inst✝⁴ : AddCommMonoid N'
    inst✝³ : Module R M'
    inst✝² : Module R N'
    g₁ : LinearMap (RingHom.id R) M M'
    g₂ : LinearMap (RingHom.id R) N N'
    inst✝¹ : IsLocalizedModule S g₁
    inst✝ : IsLocalizedModule S g₂
    l : LinearMap (RingHom.id R) M N
    x✝ : M
    ⊢ Eq ((((↑(IsLocalizedModule.iso S g₂)).comp (↑R ((LocalizedModule.map S) l))) …
  -/
  simp
  /-
    🎉 no goals
  -/


