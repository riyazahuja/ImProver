variable (R M) in
/-- The support of a module, defined as the set of primes `p` such that `Mₚ ≠ 0`. -/
def Module.support : Set (PrimeSpectrum R) :=
  { p | Nontrivial (LocalizedModule p.asIdeal.primeCompl M) }


lemma Module.mem_support_iff :
    p ∈ Module.support R M ↔ Nontrivial (LocalizedModule p.asIdeal.primeCompl M) := Iff.rfl


lemma Module.not_mem_support_iff :
    p ∉ Module.support R M ↔ Subsingleton (LocalizedModule p.asIdeal.primeCompl M) :=
  not_nontrivial_iff_subsingleton


lemma Module.not_mem_support_iff' :
    p ∉ Module.support R M ↔ ∀ m : M, ∃ r ∉ p.asIdeal, r • m = 0 := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    p : PrimeSpectrum R
    ⊢ Iff (Not (Membership.mem (Module.support R M) p)) (∀ (m : M), Exists fun r = …
  -/
  rw [not_mem_support_iff, LocalizedModule.subsingleton_iff]
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    p : PrimeSpectrum R
    ⊢ Iff (∀ (m : M), Exists fun r => And (Membership.mem p.asIdeal.primeCompl r)  …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma Module.mem_support_iff' :
    p ∈ Module.support R M ↔ ∃ m : M, ∀ r ∉ p.asIdeal, r • m ≠ 0 := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    p : PrimeSpectrum R
    ⊢ Iff (Membership.mem (Module.support R M) p) (Exists fun m => ∀ (r : R), Not  …
  -/
  rw [← @not_not (_ ∈ _), not_mem_support_iff']
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    p : PrimeSpectrum R
    ⊢ Iff (Not (∀ (m : M), Exists fun r => And (Not (Membership.mem p.asIdeal r))  …
  -/
  push_neg
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    p : PrimeSpectrum R
    ⊢ Iff (Exists fun m => ∀ (r : R), Not (Membership.mem p.asIdeal r) → Ne (HSMul …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma Module.mem_support_iff_exists_annihilator :
    p ∈ Module.support R M ↔ ∃ m : M, (R ∙ m).annihilator ≤ p.asIdeal := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    p : PrimeSpectrum R
    ⊢ Iff (Membership.mem (Module.support R M) p) (Exists fun m => LE.le (Submodul …
  -/
  rw [Module.mem_support_iff']
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    p : PrimeSpectrum R
    ⊢ Iff (Exists fun m => ∀ (r : R), Not (Membership.mem p.asIdeal r) → Ne (HSMul …
  -/
  simp_rw [not_imp_not, SetLike.le_def, Submodule.mem_annihilator_span_singleton]
  /-
    🎉 no goals
  -/


lemma Module.mem_support_iff_of_span_eq_top {s : Set M} (hs : Submodule.span R s = ⊤) :
    p ∈ Module.support R M ↔ ∃ m ∈ s, (R ∙ m).annihilator ≤ p.asIdeal := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    p : PrimeSpectrum R
    s : Set M
    hs : Eq (Submodule.span R s) Top.top
    ⊢ Iff (Membership.mem (Module.support R M) p) (Exists fun m => And (Membership …
  -/
  constructor
    /-
      case mp
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      p : PrimeSpectrum R
      s : Set M
      hs : Eq (Submodule.span R s) Top.top
      ⊢ Membership.mem (Module.support R M) p → Exists fun m => And (Membership.mem  …
    -/
  · contrapose
    rw [not_mem_support_iff, LocalizedModule.subsingleton_iff_ker_eq_top, ← top_le_iff,
      ← hs, Submodule.span_le, Set.subset_def]
    simp_rw [SetLike.le_def, Submodule.mem_annihilator_span_singleton, SetLike.mem_coe,
      LocalizedModule.mem_ker_mkLinearMap_iff]
    /-
      case mp
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      p : PrimeSpectrum R
      s : Set M
      hs : Eq (Submodule.span R s) Top.top
      ⊢ Not (Exists fun m => And (Membership.mem s m) (∀ ⦃x : R⦄, Eq (HSMul.hSMul x  …
    -/
    push_neg
    /-
      case mp
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      p : PrimeSpectrum R
      s : Set M
      hs : Eq (Submodule.span R s) Top.top
      ⊢ (∀ (m : M), Membership.mem s m → Exists fun ⦃x⦄ => And (Eq (HSMul.hSMul x m) …
    -/
    simp_rw [and_comm]
    /-
      case mp
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      p : PrimeSpectrum R
      s : Set M
      hs : Eq (Submodule.span R s) Top.top
      ⊢ (∀ (m : M), Membership.mem s m → Exists fun ⦃x⦄ => And (Not (Membership.mem  …
    -/
    exact id
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      p : PrimeSpectrum R
      s : Set M
      hs : Eq (Submodule.span R s) Top.top
      ⊢ (Exists fun m => And (Membership.mem s m) (LE.le (Submodule.span R (Singleto …
    -/
  · intro ⟨m, _, hm⟩
    /-
      case mpr
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      p : PrimeSpectrum R
      s : Set M
      hs : Eq (Submodule.span R s) Top.top
      m : M
      left✝ : Membership.mem s m
      hm : LE.le (Submodule.span R (Singleton.singleton m)).annihilator p.asIdeal
      ⊢ Membership.mem (Module.support R M) p
    -/
    exact mem_support_iff_exists_annihilator.mpr ⟨m, hm⟩
    /-
      🎉 no goals
    -/


lemma Module.annihilator_le_of_mem_support (hp : p ∈ Module.support R M) :
    Module.annihilator R M ≤ p.asIdeal := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    p : PrimeSpectrum R
    hp : Membership.mem (Module.support R M) p
    ⊢ LE.le (Module.annihilator R M) p.asIdeal
  -/
  obtain ⟨m, hm⟩ := mem_support_iff_exists_annihilator.mp hp
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    p : PrimeSpectrum R
    hp : Membership.mem (Module.support R M) p
    m : M
    hm : LE.le (Submodule.span R (Singleton.singleton m)).annihilator p.asIdeal
    ⊢ LE.le (Module.annihilator R M) p.asIdeal
  -/
  exact le_trans ((Submodule.subtype _).annihilator_le_of_injective Subtype.val_injective) hm
  /-
    🎉 no goals
  -/


lemma LocalizedModule.subsingleton_iff_support_subset {f : R} :
    Subsingleton (LocalizedModule (.powers f) M) ↔
      Module.support R M ⊆ PrimeSpectrum.zeroLocus {f} := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : R
    ⊢ Iff (Subsingleton (LocalizedModule (Submonoid.powers f) M)) (HasSubset.Subse …
  -/
  rw [LocalizedModule.subsingleton_iff]
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : R
    ⊢ Iff (∀ (m : M), Exists fun r => And (Membership.mem (Submonoid.powers f) r)  …
  -/
  constructor
    /-
      case mp
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      f : R
      ⊢ (∀ (m : M), Exists fun r => And (Membership.mem (Submonoid.powers f) r) (Eq  …
    -/
  · rintro H x hx' f rfl
    /-
      case mp
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      x : PrimeSpectrum R
      hx' : Membership.mem (Module.support R M) x
      f : R
      H : ∀ (m : M), Exists fun r => And (Membership.mem (Submonoid.powers f) r) (Eq …
      ⊢ Membership.mem (↑x.asIdeal) f
    -/
    obtain ⟨m, hm⟩ := Module.mem_support_iff_exists_annihilator.mp hx'
    /-
      case mp.intro
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      x : PrimeSpectrum R
      hx' : Membership.mem (Module.support R M) x
      f : R
      H : ∀ (m : M), Exists fun r => And (Membership.mem (Submonoid.powers f) r) (Eq …
      m : M
      hm : LE.le (Submodule.span R (Singleton.singleton m)).annihilator x.asIdeal
      ⊢ Membership.mem (↑x.asIdeal) f
    -/
    obtain ⟨_, ⟨n, rfl⟩, e⟩ := H m
    exact Ideal.IsPrime.mem_of_pow_mem inferInstance n
      (hm ((Submodule.mem_annihilator_span_singleton _ _).mpr e))
    /-
      case mpr
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      f : R
      ⊢ HasSubset.Subset (Module.support R M) (PrimeSpectrum.zeroLocus (Singleton.si …
    -/
  · intro H m
    /-
      case mpr
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      f : R
      H : HasSubset.Subset (Module.support R M) (PrimeSpectrum.zeroLocus (Singleton. …
      m : M
      ⊢ Exists fun r => And (Membership.mem (Submonoid.powers f) r) (Eq (HSMul.hSMul …
    -/
    by_cases h : (Submodule.span R {m}).annihilator = ⊤
      /-
        case pos
        R : Type u_1
        M : Type u_2
        inst✝² : CommRing R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        f : R
        H : HasSubset.Subset (Module.support R M) (PrimeSpectrum.zeroLocus (Singleton. …
        m : M
        h : Eq (Submodule.span R (Singleton.singleton m)).annihilator Top.top
        ⊢ Exists fun r => And (Membership.mem (Submonoid.powers f) r) (Eq (HSMul.hSMul …
      -/
    · rw [Submodule.annihilator_eq_top_iff, Submodule.span_singleton_eq_bot] at h
      /-
        case pos
        R : Type u_1
        M : Type u_2
        inst✝² : CommRing R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        f : R
        H : HasSubset.Subset (Module.support R M) (PrimeSpectrum.zeroLocus (Singleton. …
        m : M
        h : Eq m 0
        ⊢ Exists fun r => And (Membership.mem (Submonoid.powers f) r) (Eq (HSMul.hSMul …
      -/
      exact ⟨1, one_mem _, by simpa using h⟩
      /-
        🎉 no goals
      -/
    obtain ⟨n, hn⟩ : f ∈ (Submodule.span R {m}).annihilator.radical := by
      rw [Ideal.radical_eq_sInf, Ideal.mem_sInf]
      rintro p ⟨hp, hp'⟩
      simpa using H (Module.mem_support_iff_exists_annihilator (p := ⟨p, hp'⟩).mpr ⟨_, hp⟩)
    /-
      case neg.intro
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      f : R
      H : HasSubset.Subset (Module.support R M) (PrimeSpectrum.zeroLocus (Singleton. …
      m : M
      h : Not (Eq (Submodule.span R (Singleton.singleton m)).annihilator Top.top)
      n : Nat
      hn : Membership.mem (Submodule.span R (Singleton.singleton m)).annihilator (HP …
      ⊢ Exists fun r => And (Membership.mem (Submonoid.powers f) r) (Eq (HSMul.hSMul …
    -/
    exact ⟨_, ⟨n, rfl⟩, (Submodule.mem_annihilator_span_singleton _ _).mp hn⟩
    /-
      🎉 no goals
    -/


lemma Module.support_eq_empty_iff :
    Module.support R M = ∅ ↔ Subsingleton M := by
  rw [← Set.subset_empty_iff, ← PrimeSpectrum.zeroLocus_singleton_one,
    ← LocalizedModule.subsingleton_iff_support_subset, LocalizedModule.subsingleton_iff,
    subsingleton_iff_forall_eq 0]
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ⊢ Iff (∀ (m : M), Exists fun r => And (Membership.mem (Submonoid.powers 1) r)  …
  -/
  simp only [Submonoid.powers_one, Submonoid.mem_bot, exists_eq_left, one_smul]
  /-
    🎉 no goals
  -/


lemma Module.support_eq_empty [Subsingleton M] :
    Module.support R M = ∅ :=
  Module.support_eq_empty_iff.mpr ‹_›


lemma Module.support_of_algebra {A : Type*} [Ring A] [Algebra R A] :
    Module.support R A = PrimeSpectrum.zeroLocus (RingHom.ker (algebraMap R A)) := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    A : Type u_3
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    ⊢ Eq (Module.support R A) (PrimeSpectrum.zeroLocus ↑(RingHom.ker (algebraMap R …
  -/
  ext p
  /-
    case h
    R : Type u_1
    inst✝² : CommRing R
    A : Type u_3
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    p : PrimeSpectrum R
    ⊢ Iff (Membership.mem (Module.support R A) p) (Membership.mem (PrimeSpectrum.z …
  -/
  simp only [mem_support_iff', ne_eq, PrimeSpectrum.mem_zeroLocus, SetLike.coe_subset_coe]
  /-
    case h
    R : Type u_1
    inst✝² : CommRing R
    A : Type u_3
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    p : PrimeSpectrum R
    ⊢ Iff (Exists fun m => ∀ (r : R), Not (Membership.mem p.asIdeal r) → Not (Eq ( …
  -/
  refine ⟨fun ⟨m, hm⟩ x hx ↦ not_not.mp fun hx' ↦ ?_, fun H ↦ ⟨1, fun r hr e ↦ ?_⟩⟩
    /-
      case h.refine_1
      R : Type u_1
      inst✝² : CommRing R
      A : Type u_3
      inst✝¹ : Ring A
      inst✝ : Algebra R A
      p : PrimeSpectrum R
      x✝ : Exists fun m => ∀ (r : R), Not (Membership.mem p.asIdeal r) → Not (Eq (HS …
      x : R
      hx : Membership.mem (RingHom.ker (algebraMap R A)) x
      m : A
      hm : ∀ (r : R), Not (Membership.mem p.asIdeal r) → Not (Eq (HSMul.hSMul r m) 0)
      hx' : Not (Membership.mem p.asIdeal x)
      ⊢ False
    -/
  · simpa [Algebra.smul_def, (show _ = _ from hx)] using hm _ hx'
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2
      R : Type u_1
      inst✝² : CommRing R
      A : Type u_3
      inst✝¹ : Ring A
      inst✝ : Algebra R A
      p : PrimeSpectrum R
      H : LE.le (RingHom.ker (algebraMap R A)) p.asIdeal
      r : R
      hr : Not (Membership.mem p.asIdeal r)
      e : Eq (HSMul.hSMul r 1) 0
      ⊢ False
    -/
  · exact hr (H ((Algebra.algebraMap_eq_smul_one _).trans e))
    /-
      🎉 no goals
    -/


lemma Module.support_of_noZeroSMulDivisors [NoZeroSMulDivisors R M] [Nontrivial M] :
    Module.support R M = Set.univ := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : NoZeroSMulDivisors R M
    inst✝ : Nontrivial M
    ⊢ Eq (Module.support R M) Set.univ
  -/
  simp only [Set.eq_univ_iff_forall, mem_support_iff', ne_eq, smul_eq_zero, not_or]
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : NoZeroSMulDivisors R M
    inst✝ : Nontrivial M
    ⊢ ∀ (x : PrimeSpectrum R), Exists fun m => ∀ (r : R), Not (Membership.mem x.as …
  -/
  obtain ⟨x, hx⟩ := exists_ne (0 : M)
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : NoZeroSMulDivisors R M
    inst✝ : Nontrivial M
    x : M
    hx : Ne x 0
    ⊢ ∀ (x : PrimeSpectrum R), Exists fun m => ∀ (r : R), Not (Membership.mem x.as …
  -/
  exact fun p ↦ ⟨x, fun r hr ↦ ⟨fun e ↦ hr (e ▸ p.asIdeal.zero_mem), hx⟩⟩
  /-
    🎉 no goals
  -/


lemma Module.mem_support_iff_of_finite [Module.Finite R M] :
    p ∈ Module.support R M ↔ Module.annihilator R M ≤ p.asIdeal := by
  classical
  obtain ⟨s, hs⟩ := ‹Module.Finite R M›
  refine ⟨annihilator_le_of_mem_support, fun H ↦ (mem_support_iff_of_span_eq_top hs).mpr ?_⟩
  simp only [SetLike.le_def, Submodule.mem_annihilator_span_singleton] at H ⊢
  contrapose! H
  choose x hx hx' using Subtype.forall'.mp H
  refine ⟨s.attach.prod x, ?_, ?_⟩
  · rw [← Submodule.annihilator_top, ← hs, Submodule.mem_annihilator_span]
    intro m
    obtain ⟨k, hk⟩ := Finset.dvd_prod_of_mem x (Finset.mem_attach _ m)
    rw [hk, mul_comm, mul_smul, hx, smul_zero]
  · exact p.asIdeal.primeCompl.prod_mem (fun x _ ↦ hx' x)


lemma Module.support_subset_of_injective (hf : Function.Injective f) :
    Module.support R M ⊆ Module.support R N := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type u_3
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    f : LinearMap (RingHom.id R) M N
    hf : Function.Injective ⇑f
    ⊢ HasSubset.Subset (Module.support R M) (Module.support R N)
  -/
  simp_rw [Set.subset_def, mem_support_iff']
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type u_3
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    f : LinearMap (RingHom.id R) M N
    hf : Function.Injective ⇑f
    ⊢ ∀ (x : PrimeSpectrum R), (Exists fun m => ∀ (r : R), Not (Membership.mem x.a …
  -/
  rintro x ⟨m, hm⟩
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type u_3
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    f : LinearMap (RingHom.id R) M N
    hf : Function.Injective ⇑f
    x : PrimeSpectrum R
    m : M
    hm : ∀ (r : R), Not (Membership.mem x.asIdeal r) → Ne (HSMul.hSMul r m) 0
    ⊢ Exists fun m => ∀ (r : R), Not (Membership.mem x.asIdeal r) → Ne (HSMul.hSMu …
  -/
  exact ⟨f m, fun r hr ↦ by simpa using hf.ne (hm r hr)⟩
  /-
    🎉 no goals
  -/


lemma Module.support_subset_of_surjective (hf : Function.Surjective f) :
    Module.support R N ⊆ Module.support R M := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type u_3
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    f : LinearMap (RingHom.id R) M N
    hf : Function.Surjective ⇑f
    ⊢ HasSubset.Subset (Module.support R N) (Module.support R M)
  -/
  simp_rw [Set.subset_def, mem_support_iff']
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type u_3
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    f : LinearMap (RingHom.id R) M N
    hf : Function.Surjective ⇑f
    ⊢ ∀ (x : PrimeSpectrum R), (Exists fun m => ∀ (r : R), Not (Membership.mem x.a …
  -/
  rintro x ⟨m, hm⟩
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type u_3
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    f : LinearMap (RingHom.id R) M N
    hf : Function.Surjective ⇑f
    x : PrimeSpectrum R
    m : N
    hm : ∀ (r : R), Not (Membership.mem x.asIdeal r) → Ne (HSMul.hSMul r m) 0
    ⊢ Exists fun m => ∀ (r : R), Not (Membership.mem x.asIdeal r) → Ne (HSMul.hSMu …
  -/
  obtain ⟨m, rfl⟩ := hf m
  /-
    case intro.intro
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type u_3
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    f : LinearMap (RingHom.id R) M N
    hf : Function.Surjective ⇑f
    x : PrimeSpectrum R
    m : M
    hm : ∀ (r : R), Not (Membership.mem x.asIdeal r) → Ne (HSMul.hSMul r (f m)) 0
    ⊢ Exists fun m => ∀ (r : R), Not (Membership.mem x.asIdeal r) → Ne (HSMul.hSMu …
  -/
  exact ⟨m, fun r hr e ↦ hm r hr (by simpa using congr(f $e))⟩
  /-
    🎉 no goals
  -/


variable {f g} in
/-- Given an exact sequence `0 → M → N → P → 0` of `R`-modules, `Supp N = Supp M ∪ Supp P`. -/
lemma Module.support_of_exact (h : Function.Exact f g)
    (hf : Function.Injective f) (hg : Function.Surjective g) :
    Module.support R N = Module.support R M ∪ Module.support R P := by
  refine subset_antisymm ?_ (Set.union_subset (Module.support_subset_of_injective f hf)
    (Module.support_subset_of_surjective g hg))
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    N : Type u_3
    P : Type u_4
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    inst✝¹ : AddCommGroup P
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    h : Function.Exact ⇑f ⇑g
    hf : Function.Injective ⇑f
    hg : Function.Surjective ⇑g
    ⊢ HasSubset.Subset (Module.support R N) (Union.union (Module.support R M) (Mod …
  -/
  intro x
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    N : Type u_3
    P : Type u_4
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    inst✝¹ : AddCommGroup P
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    h : Function.Exact ⇑f ⇑g
    hf : Function.Injective ⇑f
    hg : Function.Surjective ⇑g
    x : PrimeSpectrum R
    ⊢ Membership.mem (Module.support R N) x → Membership.mem (Union.union (Module. …
  -/
  contrapose
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    N : Type u_3
    P : Type u_4
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    inst✝¹ : AddCommGroup P
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    h : Function.Exact ⇑f ⇑g
    hf : Function.Injective ⇑f
    hg : Function.Surjective ⇑g
    x : PrimeSpectrum R
    ⊢ Not (Membership.mem (Union.union (Module.support R M) (Module.support R P))  …
  -/
  simp only [Set.mem_union, not_or, and_imp, not_mem_support_iff']
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    N : Type u_3
    P : Type u_4
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    inst✝¹ : AddCommGroup P
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    h : Function.Exact ⇑f ⇑g
    hf : Function.Injective ⇑f
    hg : Function.Surjective ⇑g
    x : PrimeSpectrum R
    ⊢ (∀ (m : M), Exists fun r => And (Not (Membership.mem x.asIdeal r)) (Eq (HSMu …
  -/
  intro H₁ H₂ m
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    N : Type u_3
    P : Type u_4
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    inst✝¹ : AddCommGroup P
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    h : Function.Exact ⇑f ⇑g
    hf : Function.Injective ⇑f
    hg : Function.Surjective ⇑g
    x : PrimeSpectrum R
    H₁ : ∀ (m : M), Exists fun r => And (Not (Membership.mem x.asIdeal r)) (Eq (HS …
    H₂ : ∀ (m : P), Exists fun r => And (Not (Membership.mem x.asIdeal r)) (Eq (HS …
    m : N
    ⊢ Exists fun r => And (Not (Membership.mem x.asIdeal r)) (Eq (HSMul.hSMul r m) …
  -/
  obtain ⟨r, hr, e₁⟩ := H₂ (g m)
  /-
    case intro.intro
    R : Type u_1
    M : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    N : Type u_3
    P : Type u_4
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    inst✝¹ : AddCommGroup P
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    h : Function.Exact ⇑f ⇑g
    hf : Function.Injective ⇑f
    hg : Function.Surjective ⇑g
    x : PrimeSpectrum R
    H₁ : ∀ (m : M), Exists fun r => And (Not (Membership.mem x.asIdeal r)) (Eq (HS …
    H₂ : ∀ (m : P), Exists fun r => And (Not (Membership.mem x.asIdeal r)) (Eq (HS …
    m : N
    r : R
    hr : Not (Membership.mem x.asIdeal r)
    e₁ : Eq (HSMul.hSMul r (g m)) 0
    ⊢ Exists fun r => And (Not (Membership.mem x.asIdeal r)) (Eq (HSMul.hSMul r m) …
  -/
  rw [← map_smul, h] at e₁
  /-
    case intro.intro
    R : Type u_1
    M : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    N : Type u_3
    P : Type u_4
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    inst✝¹ : AddCommGroup P
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    h : Function.Exact ⇑f ⇑g
    hf : Function.Injective ⇑f
    hg : Function.Surjective ⇑g
    x : PrimeSpectrum R
    H₁ : ∀ (m : M), Exists fun r => And (Not (Membership.mem x.asIdeal r)) (Eq (HS …
    H₂ : ∀ (m : P), Exists fun r => And (Not (Membership.mem x.asIdeal r)) (Eq (HS …
    m : N
    r : R
    hr : Not (Membership.mem x.asIdeal r)
    e₁ : Membership.mem (Set.range ⇑f) (HSMul.hSMul r m)
    ⊢ Exists fun r => And (Not (Membership.mem x.asIdeal r)) (Eq (HSMul.hSMul r m) …
  -/
  obtain ⟨m', hm'⟩ := e₁
  /-
    case intro.intro.intro
    R : Type u_1
    M : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    N : Type u_3
    P : Type u_4
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    inst✝¹ : AddCommGroup P
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    h : Function.Exact ⇑f ⇑g
    hf : Function.Injective ⇑f
    hg : Function.Surjective ⇑g
    x : PrimeSpectrum R
    H₁ : ∀ (m : M), Exists fun r => And (Not (Membership.mem x.asIdeal r)) (Eq (HS …
    H₂ : ∀ (m : P), Exists fun r => And (Not (Membership.mem x.asIdeal r)) (Eq (HS …
    m : N
    r : R
    hr : Not (Membership.mem x.asIdeal r)
    m' : M
    hm' : Eq (f m') (HSMul.hSMul r m)
    ⊢ Exists fun r => And (Not (Membership.mem x.asIdeal r)) (Eq (HSMul.hSMul r m) …
  -/
  obtain ⟨s, hs, e₁⟩ := H₁ m'
  /-
    case intro.intro.intro.intro.intro
    R : Type u_1
    M : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    N : Type u_3
    P : Type u_4
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    inst✝¹ : AddCommGroup P
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    h : Function.Exact ⇑f ⇑g
    hf : Function.Injective ⇑f
    hg : Function.Surjective ⇑g
    x : PrimeSpectrum R
    H₁ : ∀ (m : M), Exists fun r => And (Not (Membership.mem x.asIdeal r)) (Eq (HS …
    H₂ : ∀ (m : P), Exists fun r => And (Not (Membership.mem x.asIdeal r)) (Eq (HS …
    m : N
    r : R
    hr : Not (Membership.mem x.asIdeal r)
    m' : M
    hm' : Eq (f m') (HSMul.hSMul r m)
    s : R
    hs : Not (Membership.mem x.asIdeal s)
    e₁ : Eq (HSMul.hSMul s m') 0
    ⊢ Exists fun r => And (Not (Membership.mem x.asIdeal r)) (Eq (HSMul.hSMul r m) …
  -/
  exact ⟨_, x.asIdeal.primeCompl.mul_mem hs hr, by rw [mul_smul, ← hm', ← map_smul, e₁, map_zero]⟩
  /-
    🎉 no goals
  -/


lemma LinearEquiv.support_eq (e : M ≃ₗ[R] N) :
    Module.support R M = Module.support R N :=
  (Module.support_subset_of_injective e.toLinearMap e.injective).antisymm
    (Module.support_subset_of_surjective e.toLinearMap e.surjective)


/-- If `M` is `R`-finite, then `Supp M = Z(Ann(M))`. -/
lemma Module.support_eq_zeroLocus [Module.Finite R M] :
    Module.support R M = PrimeSpectrum.zeroLocus (Module.annihilator R M) :=
  Set.ext fun _ ↦ mem_support_iff_of_finite


/-- If `M` is a finite module such that `Mₚ = 0` for some `p`,
then `M[1/f] = 0` for some `p ∈ D(f)`. -/
lemma LocalizedModule.exists_subsingleton_away [Module.Finite R M] (p : Ideal R) [p.IsPrime]
    [Subsingleton (LocalizedModule p.primeCompl M)] :
    ∃ f ∉ p, Subsingleton (LocalizedModule (.powers f) M) := by
  have : ⟨p, inferInstance⟩ ∈ (Module.support R M)ᶜ := by
    simpa [Module.not_mem_support_iff]
  rw [Module.support_eq_zeroLocus, ← Set.biUnion_of_singleton (Module.annihilator R M : Set R),
    PrimeSpectrum.zeroLocus_iUnion₂, Set.compl_iInter₂, Set.mem_iUnion₂] at this
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : Module.Finite R M
    p : Ideal R
    inst✝¹ : p.IsPrime
    inst✝ : Subsingleton (LocalizedModule p.primeCompl M)
    this : Exists fun i => Exists fun j => Membership.mem (HasCompl.compl (PrimeSp …
    ⊢ Exists fun f => And (Not (Membership.mem p f)) (Subsingleton (LocalizedModul …
  -/
  obtain ⟨f, hf, hf'⟩ := this
  exact ⟨f, by simpa using hf', subsingleton_iff.mpr
    fun m ↦ ⟨f, Submonoid.mem_powers f, Module.mem_annihilator.mp hf _⟩⟩


