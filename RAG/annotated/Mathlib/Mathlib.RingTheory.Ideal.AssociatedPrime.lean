/-- `IsAssociatedPrime I M` if the prime ideal `I` is the annihilator of some `x : M`. -/
def IsAssociatedPrime : Prop :=
  I.IsPrime ∧ ∃ x : M, I = (R ∙ x).annihilator


/-- The set of associated primes of a module. -/
def associatedPrimes : Set (Ideal R) :=
  { I | IsAssociatedPrime I M }


theorem AssociatePrimes.mem_iff : I ∈ associatedPrimes R M ↔ IsAssociatedPrime I M := Iff.rfl


theorem IsAssociatedPrime.isPrime (h : IsAssociatedPrime I M) : I.IsPrime := h.1

theorem IsAssociatedPrime.map_of_injective (h : IsAssociatedPrime I M) (hf : Function.Injective f) :
    IsAssociatedPrime I M' := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    I : Ideal R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    M' : Type u_3
    inst✝¹ : AddCommGroup M'
    inst✝ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    h : IsAssociatedPrime I M
    hf : Function.Injective ⇑f
    ⊢ IsAssociatedPrime I M'
  -/
  obtain ⟨x, rfl⟩ := h.2
  /-
    case intro
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    M' : Type u_3
    inst✝¹ : AddCommGroup M'
    inst✝ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    hf : Function.Injective ⇑f
    x : M
    h : IsAssociatedPrime (Submodule.span R (Singleton.singleton x)).annihilator M
    ⊢ IsAssociatedPrime (Submodule.span R (Singleton.singleton x)).annihilator M'
  -/
  refine ⟨h.1, ⟨f x, ?_⟩⟩
  /-
    case intro
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    M' : Type u_3
    inst✝¹ : AddCommGroup M'
    inst✝ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    hf : Function.Injective ⇑f
    x : M
    h : IsAssociatedPrime (Submodule.span R (Singleton.singleton x)).annihilator M
    ⊢ Eq (Submodule.span R (Singleton.singleton x)).annihilator (Submodule.span R  …
  -/
  ext r
  rw [Submodule.mem_annihilator_span_singleton, Submodule.mem_annihilator_span_singleton, ←
    map_smul, ← f.map_zero, hf.eq_iff]


theorem LinearEquiv.isAssociatedPrime_iff (l : M ≃ₗ[R] M') :
    IsAssociatedPrime I M ↔ IsAssociatedPrime I M' :=
  ⟨fun h => h.map_of_injective l l.injective,
    fun h => h.map_of_injective l.symm l.symm.injective⟩


theorem not_isAssociatedPrime_of_subsingleton [Subsingleton M] : ¬IsAssociatedPrime I M := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    I : Ideal R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Subsingleton M
    ⊢ Not (IsAssociatedPrime I M)
  -/
  rintro ⟨hI, x, hx⟩
  /-
    case intro.intro
    R : Type u_1
    inst✝³ : CommRing R
    I : Ideal R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Subsingleton M
    hI : I.IsPrime
    x : M
    hx : Eq I (Submodule.span R (Singleton.singleton x)).annihilator
    ⊢ False
  -/
  apply hI.ne_top
  rwa [Subsingleton.elim x 0, Submodule.span_singleton_eq_bot.mpr rfl, Submodule.annihilator_bot]
    at hx


theorem exists_le_isAssociatedPrime_of_isNoetherianRing [H : IsNoetherianRing R] (x : M)
    (hx : x ≠ 0) : ∃ P : Ideal R, IsAssociatedPrime P M ∧ (R ∙ x).annihilator ≤ P := by
  have : (R ∙ x).annihilator ≠ ⊤ := by
    rwa [Ne, Ideal.eq_top_iff_one, Submodule.mem_annihilator_span_singleton, one_smul]
  obtain ⟨P, ⟨l, h₁, y, rfl⟩, h₃⟩ :=
    set_has_maximal_iff_noetherian.mpr H
      { P | (R ∙ x).annihilator ≤ P ∧ P ≠ ⊤ ∧ ∃ y : M, P = (R ∙ y).annihilator }
      ⟨(R ∙ x).annihilator, rfl.le, this, x, rfl⟩
  /-
    case intro.intro.intro.intro.intro
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    H : IsNoetherianRing R
    x : M
    hx : Ne x 0
    this : Ne (Submodule.span R (Singleton.singleton x)).annihilator Top.top
    y : M
    h₃ : ∀ (I : Submodule R R), Membership.mem (setOf fun P => And (LE.le (Submodu …
    l : LE.le (Submodule.span R (Singleton.singleton x)).annihilator (Submodule.sp …
    h₁ : Ne (Submodule.span R (Singleton.singleton y)).annihilator Top.top
    ⊢ Exists fun P => And (IsAssociatedPrime P M) (LE.le (Submodule.span R (Single …
  -/
  refine ⟨_, ⟨⟨h₁, ?_⟩, y, rfl⟩, l⟩
  /-
    case intro.intro.intro.intro.intro
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    H : IsNoetherianRing R
    x : M
    hx : Ne x 0
    this : Ne (Submodule.span R (Singleton.singleton x)).annihilator Top.top
    y : M
    h₃ : ∀ (I : Submodule R R), Membership.mem (setOf fun P => And (LE.le (Submodu …
    l : LE.le (Submodule.span R (Singleton.singleton x)).annihilator (Submodule.sp …
    h₁ : Ne (Submodule.span R (Singleton.singleton y)).annihilator Top.top
    ⊢ ∀ {x y_1 : R}, Membership.mem (Submodule.span R (Singleton.singleton y)).ann …
  -/
  intro a b hab
  /-
    case intro.intro.intro.intro.intro
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    H : IsNoetherianRing R
    x : M
    hx : Ne x 0
    this : Ne (Submodule.span R (Singleton.singleton x)).annihilator Top.top
    y : M
    h₃ : ∀ (I : Submodule R R), Membership.mem (setOf fun P => And (LE.le (Submodu …
    l : LE.le (Submodule.span R (Singleton.singleton x)).annihilator (Submodule.sp …
    h₁ : Ne (Submodule.span R (Singleton.singleton y)).annihilator Top.top
    a b : R
    hab : Membership.mem (Submodule.span R (Singleton.singleton y)).annihilator (H …
    ⊢ Or (Membership.mem (Submodule.span R (Singleton.singleton y)).annihilator a) …
  -/
  rw [or_iff_not_imp_left]
  /-
    case intro.intro.intro.intro.intro
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    H : IsNoetherianRing R
    x : M
    hx : Ne x 0
    this : Ne (Submodule.span R (Singleton.singleton x)).annihilator Top.top
    y : M
    h₃ : ∀ (I : Submodule R R), Membership.mem (setOf fun P => And (LE.le (Submodu …
    l : LE.le (Submodule.span R (Singleton.singleton x)).annihilator (Submodule.sp …
    h₁ : Ne (Submodule.span R (Singleton.singleton y)).annihilator Top.top
    a b : R
    hab : Membership.mem (Submodule.span R (Singleton.singleton y)).annihilator (H …
    ⊢ Not (Membership.mem (Submodule.span R (Singleton.singleton y)).annihilator a …
  -/
  intro ha
  /-
    case intro.intro.intro.intro.intro
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    H : IsNoetherianRing R
    x : M
    hx : Ne x 0
    this : Ne (Submodule.span R (Singleton.singleton x)).annihilator Top.top
    y : M
    h₃ : ∀ (I : Submodule R R), Membership.mem (setOf fun P => And (LE.le (Submodu …
    l : LE.le (Submodule.span R (Singleton.singleton x)).annihilator (Submodule.sp …
    h₁ : Ne (Submodule.span R (Singleton.singleton y)).annihilator Top.top
    a b : R
    hab : Membership.mem (Submodule.span R (Singleton.singleton y)).annihilator (H …
    ha : Not (Membership.mem (Submodule.span R (Singleton.singleton y)).annihilato …
    ⊢ Membership.mem (Submodule.span R (Singleton.singleton y)).annihilator b
  -/
  rw [Submodule.mem_annihilator_span_singleton] at ha hab
  have H₁ : (R ∙ y).annihilator ≤ (R ∙ a • y).annihilator := by
    intro c hc
    rw [Submodule.mem_annihilator_span_singleton] at hc ⊢
    rw [smul_comm, hc, smul_zero]
  have H₂ : (Submodule.span R {a • y}).annihilator ≠ ⊤ := by
    rwa [Ne, Submodule.annihilator_eq_top_iff, Submodule.span_singleton_eq_bot]
  rwa [H₁.eq_of_not_lt (h₃ (R ∙ a • y).annihilator ⟨l.trans H₁, H₂, _, rfl⟩),
    Submodule.mem_annihilator_span_singleton, smul_comm, smul_smul]


theorem associatedPrimes.subset_of_injective (hf : Function.Injective f) :
    associatedPrimes R M ⊆ associatedPrimes R M' := fun _I h => h.map_of_injective f hf


theorem LinearEquiv.AssociatedPrimes.eq (l : M ≃ₗ[R] M') :
    associatedPrimes R M = associatedPrimes R M' :=
  le_antisymm (associatedPrimes.subset_of_injective l l.injective)
    (associatedPrimes.subset_of_injective l.symm l.symm.injective)


theorem associatedPrimes.eq_empty_of_subsingleton [Subsingleton M] : associatedPrimes R M = ∅ := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Subsingleton M
    ⊢ Eq (associatedPrimes R M) EmptyCollection.emptyCollection
  -/
  ext; simp only [Set.mem_empty_iff_false, iff_false]
  /-
    case h
    R : Type u_1
    inst✝³ : CommRing R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Subsingleton M
    x✝ : Ideal R
    ⊢ Not (Membership.mem (associatedPrimes R M) x✝)
  -/
  apply not_isAssociatedPrime_of_subsingleton
  /-
    🎉 no goals
  -/


theorem associatedPrimes.nonempty [IsNoetherianRing R] [Nontrivial M] :
    (associatedPrimes R M).Nonempty := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : IsNoetherianRing R
    inst✝ : Nontrivial M
    ⊢ (associatedPrimes R M).Nonempty
  -/
  obtain ⟨x, hx⟩ := exists_ne (0 : M)
  /-
    case intro
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : IsNoetherianRing R
    inst✝ : Nontrivial M
    x : M
    hx : Ne x 0
    ⊢ (associatedPrimes R M).Nonempty
  -/
  obtain ⟨P, hP, _⟩ := exists_le_isAssociatedPrime_of_isNoetherianRing R x hx
  /-
    case intro.intro.intro
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : IsNoetherianRing R
    inst✝ : Nontrivial M
    x : M
    hx : Ne x 0
    P : Ideal R
    hP : IsAssociatedPrime P M
    right✝ : LE.le (Submodule.span R (Singleton.singleton x)).annihilator P
    ⊢ (associatedPrimes R M).Nonempty
  -/
  exact ⟨P, hP⟩
  /-
    🎉 no goals
  -/


theorem biUnion_associatedPrimes_eq_zero_divisors [IsNoetherianRing R] :
    ⋃ p ∈ associatedPrimes R M, p = { r : R | ∃ x : M, x ≠ 0 ∧ r • x = 0 } := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsNoetherianRing R
    ⊢ Eq (Set.iUnion fun p => Set.iUnion fun h => ↑p) (setOf fun r => Exists fun x …
  -/
  simp_rw [← Submodule.mem_annihilator_span_singleton]
  /-
    R : Type u_1
    inst✝³ : CommRing R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsNoetherianRing R
    ⊢ Eq (Set.iUnion fun p => Set.iUnion fun h => ↑p) (setOf fun r => Exists fun x …
  -/
  refine subset_antisymm (Set.iUnion₂_subset ?_) ?_
    /-
      case refine_1
      R : Type u_1
      inst✝³ : CommRing R
      M : Type u_2
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : IsNoetherianRing R
      ⊢ ∀ (i : Ideal R), Membership.mem (associatedPrimes R M) i → HasSubset.Subset  …
    -/
  · rintro _ ⟨h, x, ⟨⟩⟩ r h'
    /-
      case refine_1.intro.intro.refl
      R : Type u_1
      inst✝³ : CommRing R
      M : Type u_2
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : IsNoetherianRing R
      x : M
      h : (Submodule.span R (Singleton.singleton x)).annihilator.IsPrime
      r : R
      h' : Membership.mem (↑(Submodule.span R (Singleton.singleton x)).annihilator) r
      ⊢ Membership.mem (setOf fun r => Exists fun x => And (Ne x 0) (Membership.mem  …
    -/
    refine ⟨x, ne_of_eq_of_ne (one_smul R x).symm ?_, h'⟩
    /-
      case refine_1.intro.intro.refl
      R : Type u_1
      inst✝³ : CommRing R
      M : Type u_2
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : IsNoetherianRing R
      x : M
      h : (Submodule.span R (Singleton.singleton x)).annihilator.IsPrime
      r : R
      h' : Membership.mem (↑(Submodule.span R (Singleton.singleton x)).annihilator) r
      ⊢ Ne (HSMul.hSMul 1 x) 0
    -/
    refine mt (Submodule.mem_annihilator_span_singleton _ _).mpr ?_
    /-
      case refine_1.intro.intro.refl
      R : Type u_1
      inst✝³ : CommRing R
      M : Type u_2
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : IsNoetherianRing R
      x : M
      h : (Submodule.span R (Singleton.singleton x)).annihilator.IsPrime
      r : R
      h' : Membership.mem (↑(Submodule.span R (Singleton.singleton x)).annihilator) r
      ⊢ Not (Membership.mem (Submodule.span R (Singleton.singleton x)).annihilator 1)
    -/
    exact (Ideal.ne_top_iff_one _).mp h.ne_top
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      inst✝³ : CommRing R
      M : Type u_2
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : IsNoetherianRing R
      ⊢ HasSubset.Subset (setOf fun r => Exists fun x => And (Ne x 0) (Membership.me …
    -/
  · intro r ⟨x, h, h'⟩
    /-
      case refine_2
      R : Type u_1
      inst✝³ : CommRing R
      M : Type u_2
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : IsNoetherianRing R
      r : R
      x : M
      h : Ne x 0
      h' : Membership.mem (Submodule.span R (Singleton.singleton x)).annihilator r
      ⊢ Membership.mem (Set.iUnion fun p => Set.iUnion fun h => ↑p) r
    -/
    obtain ⟨P, hP, hx⟩ := exists_le_isAssociatedPrime_of_isNoetherianRing R x h
    /-
      case refine_2.intro.intro
      R : Type u_1
      inst✝³ : CommRing R
      M : Type u_2
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : IsNoetherianRing R
      r : R
      x : M
      h : Ne x 0
      h' : Membership.mem (Submodule.span R (Singleton.singleton x)).annihilator r
      P : Ideal R
      hP : IsAssociatedPrime P M
      hx : LE.le (Submodule.span R (Singleton.singleton x)).annihilator P
      ⊢ Membership.mem (Set.iUnion fun p => Set.iUnion fun h => ↑p) r
    -/
    exact Set.mem_biUnion hP (hx h')
    /-
      🎉 no goals
    -/


theorem IsAssociatedPrime.annihilator_le (h : IsAssociatedPrime I M) :
    (⊤ : Submodule R M).annihilator ≤ I := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    I : Ideal R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    h : IsAssociatedPrime I M
    ⊢ LE.le Top.top.annihilator I
  -/
  obtain ⟨hI, x, rfl⟩ := h
  /-
    case intro.intro
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    x : M
    hI : (Submodule.span R (Singleton.singleton x)).annihilator.IsPrime
    ⊢ LE.le Top.top.annihilator (Submodule.span R (Singleton.singleton x)).annihil …
  -/
  exact Submodule.annihilator_mono le_top
  /-
    🎉 no goals
  -/


theorem IsAssociatedPrime.eq_radical (hI : I.IsPrimary) (h : IsAssociatedPrime J (R ⧸ I)) :
    J = I.radical := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    I J : Ideal R
    hI : I.IsPrimary
    h : IsAssociatedPrime J (HasQuotient.Quotient R I)
    ⊢ Eq J I.radical
  -/
  obtain ⟨hJ, x, e⟩ := h
  have : x ≠ 0 := by
    rintro rfl
    apply hJ.1
    rwa [Submodule.span_singleton_eq_bot.mpr rfl, Submodule.annihilator_bot] at e
  /-
    case intro.intro
    R : Type u_1
    inst✝ : CommRing R
    I J : Ideal R
    hI : I.IsPrimary
    hJ : J.IsPrime
    x : HasQuotient.Quotient R I
    e : Eq J (Submodule.span R (Singleton.singleton x)).annihilator
    this : Ne x 0
    ⊢ Eq J I.radical
  -/
  obtain ⟨x, rfl⟩ := Ideal.Quotient.mkₐ_surjective R _ x
  replace e : ∀ {y}, y ∈ J ↔ x * y ∈ I := by
    intro y
    rw [e, Submodule.mem_annihilator_span_singleton, ← map_smul, smul_eq_mul, mul_comm,
      Ideal.Quotient.mkₐ_eq_mk, ← Ideal.Quotient.mk_eq_mk, Submodule.Quotient.mk_eq_zero]
  /-
    case intro.intro.intro
    R : Type u_1
    inst✝ : CommRing R
    I J : Ideal R
    hI : I.IsPrimary
    hJ : J.IsPrime
    x : R
    this : Ne ((Ideal.Quotient.mkₐ R I) x) 0
    e : ∀ {y : R}, Iff (Membership.mem J y) (Membership.mem I (HMul.hMul x y))
    ⊢ Eq J I.radical
  -/
  apply le_antisymm
    /-
      case intro.intro.intro.a
      R : Type u_1
      inst✝ : CommRing R
      I J : Ideal R
      hI : I.IsPrimary
      hJ : J.IsPrime
      x : R
      this : Ne ((Ideal.Quotient.mkₐ R I) x) 0
      e : ∀ {y : R}, Iff (Membership.mem J y) (Membership.mem I (HMul.hMul x y))
      ⊢ LE.le J I.radical
    -/
  · intro y hy
    exact ((Ideal.isPrimary_iff.1 hI).2 <| e.mp hy).resolve_left
      ((Submodule.Quotient.mk_eq_zero I).not.mp this)
    /-
      case intro.intro.intro.a
      R : Type u_1
      inst✝ : CommRing R
      I J : Ideal R
      hI : I.IsPrimary
      hJ : J.IsPrime
      x : R
      this : Ne ((Ideal.Quotient.mkₐ R I) x) 0
      e : ∀ {y : R}, Iff (Membership.mem J y) (Membership.mem I (HMul.hMul x y))
      ⊢ LE.le I.radical J
    -/
  · rw [hJ.radical_le_iff]
    /-
      case intro.intro.intro.a
      R : Type u_1
      inst✝ : CommRing R
      I J : Ideal R
      hI : I.IsPrimary
      hJ : J.IsPrime
      x : R
      this : Ne ((Ideal.Quotient.mkₐ R I) x) 0
      e : ∀ {y : R}, Iff (Membership.mem J y) (Membership.mem I (HMul.hMul x y))
      ⊢ LE.le I J
    -/
    intro y hy
    /-
      case intro.intro.intro.a
      R : Type u_1
      inst✝ : CommRing R
      I J : Ideal R
      hI : I.IsPrimary
      hJ : J.IsPrime
      x : R
      this : Ne ((Ideal.Quotient.mkₐ R I) x) 0
      e : ∀ {y : R}, Iff (Membership.mem J y) (Membership.mem I (HMul.hMul x y))
      y : R
      hy : Membership.mem I y
      ⊢ Membership.mem J y
    -/
    exact e.mpr (I.mul_mem_left x hy)
    /-
      🎉 no goals
    -/


theorem associatedPrimes.eq_singleton_of_isPrimary [IsNoetherianRing R] (hI : I.IsPrimary) :
    associatedPrimes R (R ⧸ I) = {I.radical} := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    I : Ideal R
    inst✝ : IsNoetherianRing R
    hI : I.IsPrimary
    ⊢ Eq (associatedPrimes R (HasQuotient.Quotient R I)) (Singleton.singleton I.ra …
  -/
  ext J
  /-
    case h
    R : Type u_1
    inst✝¹ : CommRing R
    I : Ideal R
    inst✝ : IsNoetherianRing R
    hI : I.IsPrimary
    J : Ideal R
    ⊢ Iff (Membership.mem (associatedPrimes R (HasQuotient.Quotient R I)) J) (Memb …
  -/
  rw [Set.mem_singleton_iff]
  /-
    case h
    R : Type u_1
    inst✝¹ : CommRing R
    I : Ideal R
    inst✝ : IsNoetherianRing R
    hI : I.IsPrimary
    J : Ideal R
    ⊢ Iff (Membership.mem (associatedPrimes R (HasQuotient.Quotient R I)) J) (Eq J …
  -/
  refine ⟨IsAssociatedPrime.eq_radical hI, ?_⟩
  /-
    case h
    R : Type u_1
    inst✝¹ : CommRing R
    I : Ideal R
    inst✝ : IsNoetherianRing R
    hI : I.IsPrimary
    J : Ideal R
    ⊢ Eq J I.radical → Membership.mem (associatedPrimes R (HasQuotient.Quotient R  …
  -/
  rintro rfl
  haveI : Nontrivial (R ⧸ I) := by
    refine ⟨(Ideal.Quotient.mk I : _) 1, (Ideal.Quotient.mk I : _) 0, ?_⟩
    rw [Ne, Ideal.Quotient.eq, sub_zero, ← Ideal.eq_top_iff_one]
    exact hI.1
  /-
    case h
    R : Type u_1
    inst✝¹ : CommRing R
    I : Ideal R
    inst✝ : IsNoetherianRing R
    hI : I.IsPrimary
    this : Nontrivial (HasQuotient.Quotient R I)
    ⊢ Membership.mem (associatedPrimes R (HasQuotient.Quotient R I)) I.radical
  -/
  obtain ⟨a, ha⟩ := associatedPrimes.nonempty R (R ⧸ I)
  /-
    case h.intro
    R : Type u_1
    inst✝¹ : CommRing R
    I : Ideal R
    inst✝ : IsNoetherianRing R
    hI : I.IsPrimary
    this : Nontrivial (HasQuotient.Quotient R I)
    a : Ideal R
    ha : Membership.mem (associatedPrimes R (HasQuotient.Quotient R I)) a
    ⊢ Membership.mem (associatedPrimes R (HasQuotient.Quotient R I)) I.radical
  -/
  exact ha.eq_radical hI ▸ ha
  /-
    🎉 no goals
  -/

