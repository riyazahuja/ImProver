/-- The canonical map from `Spec(S ⊗[R] T)` to the cartesian product `Spec S × Spec T`. -/
noncomputable
def PrimeSpectrum.tensorProductTo (x : PrimeSpectrum (S ⊗[R] T)) :
    PrimeSpectrum S × PrimeSpectrum T :=
  ⟨comap (algebraMap _ _) x, comap Algebra.TensorProduct.includeRight.toRingHom x⟩


lemma PrimeSpectrum.continuous_tensorProductTo : Continuous (tensorProductTo R S T) :=
  (comap _).2.prod_mk (comap _).2


lemma PrimeSpectrum.isEmbedding_tensorProductTo_of_surjectiveOnStalks_aux
    (p₁ p₂ : PrimeSpectrum (S ⊗[R] T))
    (h : tensorProductTo R S T p₁ = tensorProductTo R S T p₂) :
    p₁ ≤ p₂ := by
  /-
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    inst✝¹ : CommRing T
    inst✝ : Algebra R T
    hRT : (algebraMap R T).SurjectiveOnStalks
    p₁ p₂ : PrimeSpectrum (TensorProduct R S T)
    h : Eq (PrimeSpectrum.tensorProductTo R S T p₁) (PrimeSpectrum.tensorProductTo …
    ⊢ LE.le p₁ p₂
  -/
  let g : T →+* S ⊗[R] T := Algebra.TensorProduct.includeRight.toRingHom
  /-
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    inst✝¹ : CommRing T
    inst✝ : Algebra R T
    hRT : (algebraMap R T).SurjectiveOnStalks
    p₁ p₂ : PrimeSpectrum (TensorProduct R S T)
    h : Eq (PrimeSpectrum.tensorProductTo R S T p₁) (PrimeSpectrum.tensorProductTo …
    g : RingHom T (TensorProduct R S T) := Algebra.TensorProduct.includeRight.toRi …
    ⊢ LE.le p₁ p₂
  -/
  intros x hxp₁
  /-
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    inst✝¹ : CommRing T
    inst✝ : Algebra R T
    hRT : (algebraMap R T).SurjectiveOnStalks
    p₁ p₂ : PrimeSpectrum (TensorProduct R S T)
    h : Eq (PrimeSpectrum.tensorProductTo R S T p₁) (PrimeSpectrum.tensorProductTo …
    g : RingHom T (TensorProduct R S T) := Algebra.TensorProduct.includeRight.toRi …
    x : TensorProduct R S T
    hxp₁ : Membership.mem p₁.asIdeal x
    ⊢ Membership.mem p₂.asIdeal x
  -/
  by_contra hxp₂
  obtain ⟨t, r, a, ht, e⟩ := hRT.exists_mul_eq_tmul x
    (p₂.asIdeal.comap g) inferInstance
  /-
    case intro.intro.intro.intro
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    inst✝¹ : CommRing T
    inst✝ : Algebra R T
    hRT : (algebraMap R T).SurjectiveOnStalks
    p₁ p₂ : PrimeSpectrum (TensorProduct R S T)
    h : Eq (PrimeSpectrum.tensorProductTo R S T p₁) (PrimeSpectrum.tensorProductTo …
    g : RingHom T (TensorProduct R S T) := Algebra.TensorProduct.includeRight.toRi …
    x : TensorProduct R S T
    hxp₁ : Membership.mem p₁.asIdeal x
    hxp₂ : Not (Membership.mem p₂.asIdeal x)
    t : T
    r : R
    a : S
    ht : Not (Membership.mem (Ideal.comap g p₂.asIdeal) (HSMul.hSMul r t))
    e : Eq (HMul.hMul (TensorProduct.tmul R 1 (HSMul.hSMul r t)) x) (TensorProduct …
    ⊢ False
  -/
  have h₁ : a ⊗ₜ[R] t ∈ p₁.asIdeal := e ▸ p₁.asIdeal.mul_mem_left (1 ⊗ₜ[R] (r • t)) hxp₁
  /-
    case intro.intro.intro.intro
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    inst✝¹ : CommRing T
    inst✝ : Algebra R T
    hRT : (algebraMap R T).SurjectiveOnStalks
    p₁ p₂ : PrimeSpectrum (TensorProduct R S T)
    h : Eq (PrimeSpectrum.tensorProductTo R S T p₁) (PrimeSpectrum.tensorProductTo …
    g : RingHom T (TensorProduct R S T) := Algebra.TensorProduct.includeRight.toRi …
    x : TensorProduct R S T
    hxp₁ : Membership.mem p₁.asIdeal x
    hxp₂ : Not (Membership.mem p₂.asIdeal x)
    t : T
    r : R
    a : S
    ht : Not (Membership.mem (Ideal.comap g p₂.asIdeal) (HSMul.hSMul r t))
    e : Eq (HMul.hMul (TensorProduct.tmul R 1 (HSMul.hSMul r t)) x) (TensorProduct …
    h₁ : Membership.mem p₁.asIdeal (TensorProduct.tmul R a t)
    ⊢ False
  -/
  have h₂ : a ⊗ₜ[R] t ∉ p₂.asIdeal := e ▸ p₂.asIdeal.primeCompl.mul_mem ht hxp₂
  /-
    case intro.intro.intro.intro
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    inst✝¹ : CommRing T
    inst✝ : Algebra R T
    hRT : (algebraMap R T).SurjectiveOnStalks
    p₁ p₂ : PrimeSpectrum (TensorProduct R S T)
    h : Eq (PrimeSpectrum.tensorProductTo R S T p₁) (PrimeSpectrum.tensorProductTo …
    g : RingHom T (TensorProduct R S T) := Algebra.TensorProduct.includeRight.toRi …
    x : TensorProduct R S T
    hxp₁ : Membership.mem p₁.asIdeal x
    hxp₂ : Not (Membership.mem p₂.asIdeal x)
    t : T
    r : R
    a : S
    ht : Not (Membership.mem (Ideal.comap g p₂.asIdeal) (HSMul.hSMul r t))
    e : Eq (HMul.hMul (TensorProduct.tmul R 1 (HSMul.hSMul r t)) x) (TensorProduct …
    h₁ : Membership.mem p₁.asIdeal (TensorProduct.tmul R a t)
    h₂ : Not (Membership.mem p₂.asIdeal (TensorProduct.tmul R a t))
    ⊢ False
  -/
  rw [← mul_one a, ← one_mul t, ← Algebra.TensorProduct.tmul_mul_tmul] at h₁ h₂
  /-
    case intro.intro.intro.intro
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    inst✝¹ : CommRing T
    inst✝ : Algebra R T
    hRT : (algebraMap R T).SurjectiveOnStalks
    p₁ p₂ : PrimeSpectrum (TensorProduct R S T)
    h : Eq (PrimeSpectrum.tensorProductTo R S T p₁) (PrimeSpectrum.tensorProductTo …
    g : RingHom T (TensorProduct R S T) := Algebra.TensorProduct.includeRight.toRi …
    x : TensorProduct R S T
    hxp₁ : Membership.mem p₁.asIdeal x
    hxp₂ : Not (Membership.mem p₂.asIdeal x)
    t : T
    r : R
    a : S
    ht : Not (Membership.mem (Ideal.comap g p₂.asIdeal) (HSMul.hSMul r t))
    e : Eq (HMul.hMul (TensorProduct.tmul R 1 (HSMul.hSMul r t)) x) (TensorProduct …
    h₁ : Membership.mem p₁.asIdeal (HMul.hMul (TensorProduct.tmul R a 1) (TensorPr …
    h₂ : Not (Membership.mem p₂.asIdeal (HMul.hMul (TensorProduct.tmul R a 1) (Ten …
    ⊢ False
  -/
  have h₃ : t ∉ p₂.asIdeal.comap g := fun h ↦ h₂ (Ideal.mul_mem_left _ _ h)
  have h₄ : a ∉ p₂.asIdeal.comap (algebraMap S (S ⊗[R] T)) :=
    fun h ↦ h₂ (Ideal.mul_mem_right _ _ h)
  replace h₃ : t ∉ p₁.asIdeal.comap g := by
    rwa [show p₁.asIdeal.comap g = p₂.asIdeal.comap g from congr($h.2.1)]
  replace h₄ : a ∉ p₁.asIdeal.comap (algebraMap S (S ⊗[R] T)) := by
    rwa [show p₁.asIdeal.comap (algebraMap S (S ⊗[R] T)) = p₂.asIdeal.comap _ from congr($h.1.1)]
  /-
    case intro.intro.intro.intro
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    inst✝¹ : CommRing T
    inst✝ : Algebra R T
    hRT : (algebraMap R T).SurjectiveOnStalks
    p₁ p₂ : PrimeSpectrum (TensorProduct R S T)
    h : Eq (PrimeSpectrum.tensorProductTo R S T p₁) (PrimeSpectrum.tensorProductTo …
    g : RingHom T (TensorProduct R S T) := Algebra.TensorProduct.includeRight.toRi …
    x : TensorProduct R S T
    hxp₁ : Membership.mem p₁.asIdeal x
    hxp₂ : Not (Membership.mem p₂.asIdeal x)
    t : T
    r : R
    a : S
    ht : Not (Membership.mem (Ideal.comap g p₂.asIdeal) (HSMul.hSMul r t))
    e : Eq (HMul.hMul (TensorProduct.tmul R 1 (HSMul.hSMul r t)) x) (TensorProduct …
    h₁ : Membership.mem p₁.asIdeal (HMul.hMul (TensorProduct.tmul R a 1) (TensorPr …
    h₂ : Not (Membership.mem p₂.asIdeal (HMul.hMul (TensorProduct.tmul R a 1) (Ten …
    h₃ : Not (Membership.mem (Ideal.comap g p₁.asIdeal) t)
    h₄ : Not (Membership.mem (Ideal.comap (algebraMap S (TensorProduct R S T)) p₁. …
    ⊢ False
  -/
  exact p₁.asIdeal.primeCompl.mul_mem h₄ h₃ h₁
  /-
    🎉 no goals
  -/


lemma PrimeSpectrum.isEmbedding_tensorProductTo_of_surjectiveOnStalks :
    IsEmbedding (tensorProductTo R S T) := by
  refine ⟨?_, fun p₁ p₂ e ↦
    (isEmbedding_tensorProductTo_of_surjectiveOnStalks_aux R S T hRT p₁ p₂ e).antisymm
      (isEmbedding_tensorProductTo_of_surjectiveOnStalks_aux R S T hRT p₂ p₁ e.symm)⟩
  /-
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    inst✝¹ : CommRing T
    inst✝ : Algebra R T
    hRT : (algebraMap R T).SurjectiveOnStalks
    ⊢ Topology.IsInducing (PrimeSpectrum.tensorProductTo R S T)
  -/
  let g : T →+* S ⊗[R] T := Algebra.TensorProduct.includeRight.toRingHom
  /-
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    inst✝¹ : CommRing T
    inst✝ : Algebra R T
    hRT : (algebraMap R T).SurjectiveOnStalks
    g : RingHom T (TensorProduct R S T) := Algebra.TensorProduct.includeRight.toRi …
    ⊢ Topology.IsInducing (PrimeSpectrum.tensorProductTo R S T)
  -/
  refine ⟨(continuous_tensorProductTo ..).le_induced.antisymm (isBasis_basic_opens.le_iff.mpr ?_)⟩
  /-
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    inst✝¹ : CommRing T
    inst✝ : Algebra R T
    hRT : (algebraMap R T).SurjectiveOnStalks
    g : RingHom T (TensorProduct R S T) := Algebra.TensorProduct.includeRight.toRi …
    ⊢ ∀ (U : TopologicalSpace.Opens (PrimeSpectrum (TensorProduct R S T))), Member …
  -/
  rintro _ ⟨f, rfl⟩
  /-
    case intro
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    inst✝¹ : CommRing T
    inst✝ : Algebra R T
    hRT : (algebraMap R T).SurjectiveOnStalks
    g : RingHom T (TensorProduct R S T) := Algebra.TensorProduct.includeRight.toRi …
    f : TensorProduct R S T
    ⊢ IsOpen ↑(PrimeSpectrum.basicOpen f)
  -/
  rw [@isOpen_iff_forall_mem_open]
  /-
    case intro
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    inst✝¹ : CommRing T
    inst✝ : Algebra R T
    hRT : (algebraMap R T).SurjectiveOnStalks
    g : RingHom T (TensorProduct R S T) := Algebra.TensorProduct.includeRight.toRi …
    f : TensorProduct R S T
    ⊢ ∀ (x : PrimeSpectrum (TensorProduct R S T)), Membership.mem (↑(PrimeSpectrum …
  -/
  rintro J (hJ : f ∉ J.asIdeal)
  obtain ⟨t, r, a, ht, e⟩ := hRT.exists_mul_eq_tmul f
    (J.asIdeal.comap g) inferInstance
  /-
    case intro.intro.intro.intro.intro
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    inst✝¹ : CommRing T
    inst✝ : Algebra R T
    hRT : (algebraMap R T).SurjectiveOnStalks
    g : RingHom T (TensorProduct R S T) := Algebra.TensorProduct.includeRight.toRi …
    f : TensorProduct R S T
    J : PrimeSpectrum (TensorProduct R S T)
    hJ : Not (Membership.mem J.asIdeal f)
    t : T
    r : R
    a : S
    ht : Not (Membership.mem (Ideal.comap g J.asIdeal) (HSMul.hSMul r t))
    e : Eq (HMul.hMul (TensorProduct.tmul R 1 (HSMul.hSMul r t)) f) (TensorProduct …
    ⊢ Exists fun t => And (HasSubset.Subset t ↑(PrimeSpectrum.basicOpen f)) (And ( …
  -/
  refine ⟨_, ?_, ⟨_, (basicOpen a).2.prod (basicOpen t).2, rfl⟩, ?_⟩
  · rintro x ⟨hx₁ : a ⊗ₜ[R] (1 : T) ∉ x.asIdeal, hx₂ : (1 : S) ⊗ₜ[R] t ∉ x.asIdeal⟩
      (hx₃ : f ∈ x.asIdeal)
    /-
      case intro.intro.intro.intro.intro.refine_1.intro
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      inst✝¹ : CommRing T
      inst✝ : Algebra R T
      hRT : (algebraMap R T).SurjectiveOnStalks
      g : RingHom T (TensorProduct R S T) := Algebra.TensorProduct.includeRight.toRi …
      f : TensorProduct R S T
      J : PrimeSpectrum (TensorProduct R S T)
      hJ : Not (Membership.mem J.asIdeal f)
      t : T
      r : R
      a : S
      ht : Not (Membership.mem (Ideal.comap g J.asIdeal) (HSMul.hSMul r t))
      e : Eq (HMul.hMul (TensorProduct.tmul R 1 (HSMul.hSMul r t)) f) (TensorProduct …
      x : PrimeSpectrum (TensorProduct R S T)
      hx₁ : Not (Membership.mem x.asIdeal (TensorProduct.tmul R a 1))
      hx₂ : Not (Membership.mem x.asIdeal (TensorProduct.tmul R 1 t))
      hx₃ : Membership.mem x.asIdeal f
      ⊢ False
    -/
    apply x.asIdeal.primeCompl.mul_mem hx₁ hx₂
    /-
      case intro.intro.intro.intro.intro.refine_1.intro
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      inst✝¹ : CommRing T
      inst✝ : Algebra R T
      hRT : (algebraMap R T).SurjectiveOnStalks
      g : RingHom T (TensorProduct R S T) := Algebra.TensorProduct.includeRight.toRi …
      f : TensorProduct R S T
      J : PrimeSpectrum (TensorProduct R S T)
      hJ : Not (Membership.mem J.asIdeal f)
      t : T
      r : R
      a : S
      ht : Not (Membership.mem (Ideal.comap g J.asIdeal) (HSMul.hSMul r t))
      e : Eq (HMul.hMul (TensorProduct.tmul R 1 (HSMul.hSMul r t)) f) (TensorProduct …
      x : PrimeSpectrum (TensorProduct R S T)
      hx₁ : Not (Membership.mem x.asIdeal (TensorProduct.tmul R a 1))
      hx₂ : Not (Membership.mem x.asIdeal (TensorProduct.tmul R 1 t))
      hx₃ : Membership.mem x.asIdeal f
      ⊢ Membership.mem (↑x.asIdeal) (HMul.hMul (TensorProduct.tmul R a 1) (TensorPro …
    -/
    rw [Algebra.TensorProduct.tmul_mul_tmul, mul_one, one_mul, ← e]
    /-
      case intro.intro.intro.intro.intro.refine_1.intro
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      inst✝¹ : CommRing T
      inst✝ : Algebra R T
      hRT : (algebraMap R T).SurjectiveOnStalks
      g : RingHom T (TensorProduct R S T) := Algebra.TensorProduct.includeRight.toRi …
      f : TensorProduct R S T
      J : PrimeSpectrum (TensorProduct R S T)
      hJ : Not (Membership.mem J.asIdeal f)
      t : T
      r : R
      a : S
      ht : Not (Membership.mem (Ideal.comap g J.asIdeal) (HSMul.hSMul r t))
      e : Eq (HMul.hMul (TensorProduct.tmul R 1 (HSMul.hSMul r t)) f) (TensorProduct …
      x : PrimeSpectrum (TensorProduct R S T)
      hx₁ : Not (Membership.mem x.asIdeal (TensorProduct.tmul R a 1))
      hx₂ : Not (Membership.mem x.asIdeal (TensorProduct.tmul R 1 t))
      hx₃ : Membership.mem x.asIdeal f
      ⊢ Membership.mem (↑x.asIdeal) (HMul.hMul (TensorProduct.tmul R 1 (HSMul.hSMul  …
    -/
    exact x.asIdeal.mul_mem_left _ hx₃
    /-
      🎉 no goals
    -/
  · have : a ⊗ₜ[R] (1 : T) * (1 : S) ⊗ₜ[R] t ∉ J.asIdeal := by
      rw [Algebra.TensorProduct.tmul_mul_tmul, mul_one, one_mul, ← e]
      exact J.asIdeal.primeCompl.mul_mem ht hJ
    /-
      case intro.intro.intro.intro.intro.refine_2
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      inst✝¹ : CommRing T
      inst✝ : Algebra R T
      hRT : (algebraMap R T).SurjectiveOnStalks
      g : RingHom T (TensorProduct R S T) := Algebra.TensorProduct.includeRight.toRi …
      f : TensorProduct R S T
      J : PrimeSpectrum (TensorProduct R S T)
      hJ : Not (Membership.mem J.asIdeal f)
      t : T
      r : R
      a : S
      ht : Not (Membership.mem (Ideal.comap g J.asIdeal) (HSMul.hSMul r t))
      e : Eq (HMul.hMul (TensorProduct.tmul R 1 (HSMul.hSMul r t)) f) (TensorProduct …
      this : Not (Membership.mem J.asIdeal (HMul.hMul (TensorProduct.tmul R a 1) (Te …
      ⊢ Membership.mem (Set.preimage (PrimeSpectrum.tensorProductTo R S T) (SProd.sp …
    -/
    rwa [J.isPrime.mul_mem_iff_mem_or_mem.not, not_or] at this
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-10-26")]
alias PrimeSpectrum.embedding_tensorProductTo_of_surjectiveOnStalks :=
  PrimeSpectrum.isEmbedding_tensorProductTo_of_surjectiveOnStalks

