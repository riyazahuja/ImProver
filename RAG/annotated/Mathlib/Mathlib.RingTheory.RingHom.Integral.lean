theorem isIntegral_stableUnderComposition : StableUnderComposition fun f => f.IsIntegral := by
  /-
    ⊢ RingHom.StableUnderComposition fun {R S} [CommRing R] [CommRing S] f => f.Is …
  -/
  introv R hf hg; exact hf.trans _ _ hg
                  /-
                    🎉 no goals
                  -/


theorem isIntegral_respectsIso : RespectsIso fun f => f.IsIntegral := by
  /-
    ⊢ RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] f => f.IsIntegral
  -/
  apply isIntegral_stableUnderComposition.respectsIso
  /-
    ⊢ ∀ {R S : Type u_1} [inst : CommRing R] [inst_1 : CommRing S] (e : RingEquiv  …
  -/
  introv x
  /-
    R S : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    e : RingEquiv R S
    x : S
    ⊢ e.toRingHom.IsIntegralElem x
  -/
  rw [← e.apply_symm_apply x]
  /-
    R S : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    e : RingEquiv R S
    x : S
    ⊢ e.toRingHom.IsIntegralElem (e (e.symm x))
  -/
  apply RingHom.isIntegralElem_map
  /-
    🎉 no goals
  -/


theorem isIntegral_isStableUnderBaseChange : IsStableUnderBaseChange fun f => f.IsIntegral := by
  /-
    ⊢ RingHom.IsStableUnderBaseChange fun {R S} [CommRing R] [CommRing S] f => f.I …
  -/
  refine IsStableUnderBaseChange.mk _ isIntegral_respectsIso ?_
  /-
    ⊢ ∀ ⦃R S T : Type u_1⦄ [inst : CommRing R] [inst_1 : CommRing S] [inst_2 : Com …
  -/
  introv h x
  /-
    R S T : Type u_1
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : CommRing T
    inst✝¹ : Algebra R S
    inst✝ : Algebra R T
    h : (algebraMap R T).IsIntegral
    x : TensorProduct R S T
    ⊢ Algebra.TensorProduct.includeLeftRingHom.IsIntegralElem x
  -/
  refine TensorProduct.induction_on x ?_ ?_ ?_
    /-
      case refine_1
      R S T : Type u_1
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : CommRing T
      inst✝¹ : Algebra R S
      inst✝ : Algebra R T
      h : (algebraMap R T).IsIntegral
      x : TensorProduct R S T
      ⊢ Algebra.TensorProduct.includeLeftRingHom.IsIntegralElem 0
    -/
  · apply isIntegral_zero
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R S T : Type u_1
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : CommRing T
      inst✝¹ : Algebra R S
      inst✝ : Algebra R T
      h : (algebraMap R T).IsIntegral
      x : TensorProduct R S T
      ⊢ ∀ (x : S) (y : T), Algebra.TensorProduct.includeLeftRingHom.IsIntegralElem ( …
    -/
  · intro x y; exact IsIntegral.tmul x (h y)
               /-
                 🎉 no goals
               -/
    /-
      case refine_3
      R S T : Type u_1
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : CommRing T
      inst✝¹ : Algebra R S
      inst✝ : Algebra R T
      h : (algebraMap R T).IsIntegral
      x : TensorProduct R S T
      ⊢ ∀ (x y : TensorProduct R S T), Algebra.TensorProduct.includeLeftRingHom.IsIn …
    -/
  · intro x y hx hy; exact IsIntegral.add hx hy
                     /-
                       🎉 no goals
                     -/


open Polynomial in
/-- `S` is an integral `R`-algebra if there exists a set `{ r }` that
  spans `R` such that each `Sᵣ` is an integral `Rᵣ`-algebra. -/
theorem isIntegral_ofLocalizationSpan :
    OfLocalizationSpan (RingHom.IsIntegral ·) := by
  /-
    ⊢ RingHom.OfLocalizationSpan fun {R S} [CommRing R] [CommRing S] x => x.IsInte …
  -/
  introv R hs H r
  /-
    R S : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    s : Set R
    hs : Eq (Ideal.span s) Top.top
    H : ∀ (r : ↑s), (fun {R S} [CommRing R] [CommRing S] x => x.IsIntegral) (Local …
    r : S
    ⊢ f.IsIntegralElem r
  -/
  letI := f.toAlgebra
  /-
    R S : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    s : Set R
    hs : Eq (Ideal.span s) Top.top
    H : ∀ (r : ↑s), (fun {R S} [CommRing R] [CommRing S] x => x.IsIntegral) (Local …
    r : S
    this : Algebra R S := f.toAlgebra
    ⊢ f.IsIntegralElem r
  -/
  show r ∈ (integralClosure R S).toSubmodule
  /-
    R S : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    s : Set R
    hs : Eq (Ideal.span s) Top.top
    H : ∀ (r : ↑s), (fun {R S} [CommRing R] [CommRing S] x => x.IsIntegral) (Local …
    r : S
    this : Algebra R S := f.toAlgebra
    ⊢ Membership.mem (Subalgebra.toSubmodule (integralClosure R S)) r
  -/
  apply Submodule.mem_of_span_eq_top_of_smul_pow_mem _ s hs
  /-
    case H
    R S : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    s : Set R
    hs : Eq (Ideal.span s) Top.top
    H : ∀ (r : ↑s), (fun {R S} [CommRing R] [CommRing S] x => x.IsIntegral) (Local …
    r : S
    this : Algebra R S := f.toAlgebra
    ⊢ ∀ (r_1 : ↑s), Exists fun n => Membership.mem (Subalgebra.toSubmodule (integr …
  -/
  rintro ⟨t, ht⟩
  /-
    case H.mk
    R S : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    s : Set R
    hs : Eq (Ideal.span s) Top.top
    H : ∀ (r : ↑s), (fun {R S} [CommRing R] [CommRing S] x => x.IsIntegral) (Local …
    r : S
    this : Algebra R S := f.toAlgebra
    t : R
    ht : Membership.mem s t
    ⊢ Exists fun n => Membership.mem (Subalgebra.toSubmodule (integralClosure R S) …
  -/
  letI := (Localization.awayMap f t).toAlgebra
  haveI : IsScalarTower R (Localization.Away t) (Localization.Away (f t)) := .of_algebraMap_eq'
    (IsLocalization.lift_comp _).symm
  have : _root_.IsIntegral (Localization.Away t) (algebraMap S (Localization.Away (f t)) r) :=
    H ⟨t, ht⟩ (algebraMap _ _ r)
  /-
    case H.mk
    R S : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    s : Set R
    hs : Eq (Ideal.span s) Top.top
    H : ∀ (r : ↑s), (fun {R S} [CommRing R] [CommRing S] x => x.IsIntegral) (Local …
    r : S
    this✝² : Algebra R S := f.toAlgebra
    t : R
    ht : Membership.mem s t
    this✝¹ : Algebra (Localization.Away t) (Localization.Away (f t)) := (Localizat …
    this✝ : IsScalarTower R (Localization.Away t) (Localization.Away (f t))
    this : _root_.IsIntegral (Localization.Away t) ((algebraMap S (Localization.Aw …
    ⊢ Exists fun n => Membership.mem (Subalgebra.toSubmodule (integralClosure R S) …
  -/
  obtain ⟨⟨_, n, rfl⟩, p, hp, hp'⟩ := this.exists_multiple_integral_of_isLocalization (.powers t)
  rw [IsScalarTower.algebraMap_eq R S, Submonoid.smul_def, Algebra.smul_def,
    IsScalarTower.algebraMap_apply R S, ← map_mul, ← hom_eval₂,
    IsLocalization.map_eq_zero_iff (.powers (f t))] at hp'
  /-
    case H.mk.intro.mk.intro.intro.intro
    R S : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    s : Set R
    hs : Eq (Ideal.span s) Top.top
    H : ∀ (r : ↑s), (fun {R S} [CommRing R] [CommRing S] x => x.IsIntegral) (Local …
    r : S
    this✝² : Algebra R S := f.toAlgebra
    t : R
    ht : Membership.mem s t
    this✝¹ : Algebra (Localization.Away t) (Localization.Away (f t)) := (Localizat …
    this✝ : IsScalarTower R (Localization.Away t) (Localization.Away (f t))
    this : _root_.IsIntegral (Localization.Away t) ((algebraMap S (Localization.Aw …
    n : Nat
    p : Polynomial R
    hp : p.Monic
    hp' : Exists fun m => Eq (HMul.hMul (↑m) (Polynomial.eval₂ (algebraMap R S) (H …
    ⊢ Exists fun n => Membership.mem (Subalgebra.toSubmodule (integralClosure R S) …
  -/
  obtain ⟨⟨x, m, (rfl : algebraMap R S t ^ m = x)⟩, e⟩ := hp'
  /-
    case H.mk.intro.mk.intro.intro.intro.intro.mk.intro
    R S : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    s : Set R
    hs : Eq (Ideal.span s) Top.top
    H : ∀ (r : ↑s), (fun {R S} [CommRing R] [CommRing S] x => x.IsIntegral) (Local …
    r : S
    this✝² : Algebra R S := f.toAlgebra
    t : R
    ht : Membership.mem s t
    this✝¹ : Algebra (Localization.Away t) (Localization.Away (f t)) := (Localizat …
    this✝ : IsScalarTower R (Localization.Away t) (Localization.Away (f t))
    this : _root_.IsIntegral (Localization.Away t) ((algebraMap S (Localization.Aw …
    n : Nat
    p : Polynomial R
    hp : p.Monic
    m : Nat
    e : Eq (HMul.hMul (↑⟨HPow.hPow ((algebraMap R S) t) m, ⋯⟩) (Polynomial.eval₂ ( …
    ⊢ Exists fun n => Membership.mem (Subalgebra.toSubmodule (integralClosure R S) …
  -/
  by_cases hp' : 1 ≤ p.natDegree; swap
    /-
      case neg
      R S : Type u_1
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      s : Set R
      hs : Eq (Ideal.span s) Top.top
      H : ∀ (r : ↑s), (fun {R S} [CommRing R] [CommRing S] x => x.IsIntegral) (Local …
      r : S
      this✝² : Algebra R S := f.toAlgebra
      t : R
      ht : Membership.mem s t
      this✝¹ : Algebra (Localization.Away t) (Localization.Away (f t)) := (Localizat …
      this✝ : IsScalarTower R (Localization.Away t) (Localization.Away (f t))
      this : _root_.IsIntegral (Localization.Away t) ((algebraMap S (Localization.Aw …
      n : Nat
      p : Polynomial R
      hp : p.Monic
      m : Nat
      e : Eq (HMul.hMul (↑⟨HPow.hPow ((algebraMap R S) t) m, ⋯⟩) (Polynomial.eval₂ ( …
      hp' : Not (LE.le 1 p.natDegree)
      ⊢ Exists fun n => Membership.mem (Subalgebra.toSubmodule (integralClosure R S) …
    -/
  · obtain rfl : p = 1 := eq_one_of_monic_natDegree_zero hp (by omega)
    /-
      case neg
      R S : Type u_1
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      s : Set R
      hs : Eq (Ideal.span s) Top.top
      H : ∀ (r : ↑s), (fun {R S} [CommRing R] [CommRing S] x => x.IsIntegral) (Local …
      r : S
      this✝² : Algebra R S := f.toAlgebra
      t : R
      ht : Membership.mem s t
      this✝¹ : Algebra (Localization.Away t) (Localization.Away (f t)) := (Localizat …
      this✝ : IsScalarTower R (Localization.Away t) (Localization.Away (f t))
      this : _root_.IsIntegral (Localization.Away t) ((algebraMap S (Localization.Aw …
      n m : Nat
      hp : Polynomial.Monic 1
      e : Eq (HMul.hMul (↑⟨HPow.hPow ((algebraMap R S) t) m, ⋯⟩) (Polynomial.eval₂ ( …
      hp' : Not (LE.le 1 (Polynomial.natDegree 1))
      ⊢ Exists fun n => Membership.mem (Subalgebra.toSubmodule (integralClosure R S) …
    -/
    exact ⟨m, by simp [Algebra.smul_def, show algebraMap R S t ^ m = 0 by simpa using e]⟩
    /-
      🎉 no goals
    -/
  /-
    case pos
    R S : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    s : Set R
    hs : Eq (Ideal.span s) Top.top
    H : ∀ (r : ↑s), (fun {R S} [CommRing R] [CommRing S] x => x.IsIntegral) (Local …
    r : S
    this✝² : Algebra R S := f.toAlgebra
    t : R
    ht : Membership.mem s t
    this✝¹ : Algebra (Localization.Away t) (Localization.Away (f t)) := (Localizat …
    this✝ : IsScalarTower R (Localization.Away t) (Localization.Away (f t))
    this : _root_.IsIntegral (Localization.Away t) ((algebraMap S (Localization.Aw …
    n : Nat
    p : Polynomial R
    hp : p.Monic
    m : Nat
    e : Eq (HMul.hMul (↑⟨HPow.hPow ((algebraMap R S) t) m, ⋯⟩) (Polynomial.eval₂ ( …
    hp' : LE.le 1 p.natDegree
    ⊢ Exists fun n => Membership.mem (Subalgebra.toSubmodule (integralClosure R S) …
  -/
  refine ⟨m + n, p.scaleRoots (t ^ m), (monic_scaleRoots_iff _).mpr hp, ?_⟩
  /-
    case pos
    R S : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    s : Set R
    hs : Eq (Ideal.span s) Top.top
    H : ∀ (r : ↑s), (fun {R S} [CommRing R] [CommRing S] x => x.IsIntegral) (Local …
    r : S
    this✝² : Algebra R S := f.toAlgebra
    t : R
    ht : Membership.mem s t
    this✝¹ : Algebra (Localization.Away t) (Localization.Away (f t)) := (Localizat …
    this✝ : IsScalarTower R (Localization.Away t) (Localization.Away (f t))
    this : _root_.IsIntegral (Localization.Away t) ((algebraMap S (Localization.Aw …
    n : Nat
    p : Polynomial R
    hp : p.Monic
    m : Nat
    e : Eq (HMul.hMul (↑⟨HPow.hPow ((algebraMap R S) t) m, ⋯⟩) (Polynomial.eval₂ ( …
    hp' : LE.le 1 p.natDegree
    ⊢ Eq (Polynomial.eval₂ (algebraMap R S) (HSMul.hSMul (HPow.hPow (↑⟨t, ht⟩) (HA …
  -/
  have := p.scaleRoots_eval₂_mul (algebraMap R S) (t ^ n • r) (t ^ m)
  /-
    case pos
    R S : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    s : Set R
    hs : Eq (Ideal.span s) Top.top
    H : ∀ (r : ↑s), (fun {R S} [CommRing R] [CommRing S] x => x.IsIntegral) (Local …
    r : S
    this✝³ : Algebra R S := f.toAlgebra
    t : R
    ht : Membership.mem s t
    this✝² : Algebra (Localization.Away t) (Localization.Away (f t)) := (Localizat …
    this✝¹ : IsScalarTower R (Localization.Away t) (Localization.Away (f t))
    this✝ : _root_.IsIntegral (Localization.Away t) ((algebraMap S (Localization.A …
    n : Nat
    p : Polynomial R
    hp : p.Monic
    m : Nat
    e : Eq (HMul.hMul (↑⟨HPow.hPow ((algebraMap R S) t) m, ⋯⟩) (Polynomial.eval₂ ( …
    hp' : LE.le 1 p.natDegree
    this : Eq (Polynomial.eval₂ (algebraMap R S) (HMul.hMul ((algebraMap R S) (HPo …
    ⊢ Eq (Polynomial.eval₂ (algebraMap R S) (HSMul.hSMul (HPow.hPow (↑⟨t, ht⟩) (HA …
  -/
  simp only [pow_add, ← Algebra.smul_def, mul_smul, ← map_pow] at e this ⊢
  /-
    case pos
    R S : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    s : Set R
    hs : Eq (Ideal.span s) Top.top
    H : ∀ (r : ↑s), (fun {R S} [CommRing R] [CommRing S] x => x.IsIntegral) (Local …
    r : S
    this✝³ : Algebra R S := f.toAlgebra
    t : R
    ht : Membership.mem s t
    this✝² : Algebra (Localization.Away t) (Localization.Away (f t)) := (Localizat …
    this✝¹ : IsScalarTower R (Localization.Away t) (Localization.Away (f t))
    this✝ : _root_.IsIntegral (Localization.Away t) ((algebraMap S (Localization.A …
    n : Nat
    p : Polynomial R
    hp : p.Monic
    m : Nat
    hp' : LE.le 1 p.natDegree
    e : Eq (HSMul.hSMul (HPow.hPow t m) (Polynomial.eval₂ (algebraMap R S) (HSMul. …
    this : Eq (Polynomial.eval₂ (algebraMap R S) (HSMul.hSMul (HPow.hPow t m) (HSM …
    ⊢ Eq (Polynomial.eval₂ (algebraMap R S) (HSMul.hSMul (HPow.hPow t m) (HSMul.hS …
  -/
  rw [this, ← tsub_add_cancel_of_le hp', pow_succ, mul_smul, e, smul_zero]
  /-
    🎉 no goals
  -/


