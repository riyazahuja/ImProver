/-- A nonzero element in a domain integral over a field is a unit. -/
theorem IsIntegral.isUnit [Field R] [Ring S] [IsDomain S] [Algebra R S] {x : S}
    (int : IsIntegral R x) (h0 : x ≠ 0) : IsUnit x :=
  have : FiniteDimensional R (adjoin R {x}) := ⟨(Submodule.fg_top _).mpr int.fg_adjoin_singleton⟩
  (FiniteDimensional.isUnit R (K := adjoin R {x})
    (x := ⟨x, subset_adjoin rfl⟩) <| mt Subtype.ext_iff.mp h0).map (adjoin R {x}).val


/-- A commutative domain that is an integral algebra over a field is a field. -/
theorem isField_of_isIntegral_of_isField' [CommRing R] [CommRing S] [IsDomain S]
    [Algebra R S] [Algebra.IsIntegral R S] (hR : IsField R) : IsField S where
  exists_pair_ne := ⟨0, 1, zero_ne_one⟩
  mul_comm := mul_comm
  mul_inv_cancel {x} hx := by
    /-
      R : Type u_1
      S : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : IsDomain S
      inst✝¹ : Algebra R S
      inst✝ : Algebra.IsIntegral R S
      hR : IsField R
      x : S
      hx : Ne x 0
      ⊢ Exists fun b => Eq (HMul.hMul x b) 1
    -/
    letI := hR.toField
    /-
      R : Type u_1
      S : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : IsDomain S
      inst✝¹ : Algebra R S
      inst✝ : Algebra.IsIntegral R S
      hR : IsField R
      x : S
      hx : Ne x 0
      this : Field R := hR.toField
      ⊢ Exists fun b => Eq (HMul.hMul x b) 1
    -/
    obtain ⟨y, rfl⟩ := (Algebra.IsIntegral.isIntegral (R := R) x).isUnit hx
    /-
      case intro
      R : Type u_1
      S : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : IsDomain S
      inst✝¹ : Algebra R S
      inst✝ : Algebra.IsIntegral R S
      hR : IsField R
      this : Field R := hR.toField
      y : Units S
      hx : Ne (↑y) 0
      ⊢ Exists fun b => Eq (HMul.hMul (↑y) b) 1
    -/
    exact ⟨y.inv, y.val_inv⟩
    /-
      🎉 no goals
    -/


theorem IsIntegral.inv_mem_adjoin (int : IsIntegral R x) : x⁻¹ ∈ adjoin R {x} := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : Field R
    inst✝¹ : DivisionRing S
    inst✝ : Algebra R S
    x : S
    int : IsIntegral R x
    ⊢ Membership.mem (Algebra.adjoin R (Singleton.singleton x)) (Inv.inv x)
  -/
  obtain rfl | h0 := eq_or_ne x 0
    /-
      case inl
      R : Type u_1
      S : Type u_2
      inst✝² : Field R
      inst✝¹ : DivisionRing S
      inst✝ : Algebra R S
      int : IsIntegral R 0
      ⊢ Membership.mem (Algebra.adjoin R (Singleton.singleton 0)) (Inv.inv 0)
    -/
  · rw [inv_zero]; exact Subalgebra.zero_mem _
                   /-
                     🎉 no goals
                   -/
  /-
    case inr
    R : Type u_1
    S : Type u_2
    inst✝² : Field R
    inst✝¹ : DivisionRing S
    inst✝ : Algebra R S
    x : S
    int : IsIntegral R x
    h0 : Ne x 0
    ⊢ Membership.mem (Algebra.adjoin R (Singleton.singleton x)) (Inv.inv x)
  -/
  have : FiniteDimensional R (adjoin R {x}) := ⟨(Submodule.fg_top _).mpr int.fg_adjoin_singleton⟩
  obtain ⟨⟨y, hy⟩, h1⟩ := FiniteDimensional.exists_mul_eq_one R
    (K := adjoin R {x}) (x := ⟨x, subset_adjoin rfl⟩) (mt Subtype.ext_iff.mp h0)
  /-
    case inr.intro.mk
    R : Type u_1
    S : Type u_2
    inst✝² : Field R
    inst✝¹ : DivisionRing S
    inst✝ : Algebra R S
    x : S
    int : IsIntegral R x
    h0 : Ne x 0
    this : FiniteDimensional R (Subtype fun x_1 => Membership.mem (Algebra.adjoin  …
    y : S
    hy : Membership.mem (Algebra.adjoin R (Singleton.singleton x)) y
    h1 : Eq (HMul.hMul ⟨x, ⋯⟩ ⟨y, hy⟩) 1
    ⊢ Membership.mem (Algebra.adjoin R (Singleton.singleton x)) (Inv.inv x)
  -/
  rwa [← mul_left_cancel₀ h0 ((Subtype.ext_iff.mp h1).trans (mul_inv_cancel₀ h0).symm)]
  /-
    🎉 no goals
  -/


/-- The inverse of an integral element in a subalgebra of a division ring over a field
  also lies in that subalgebra. -/
theorem IsIntegral.inv_mem (int : IsIntegral R x) (hx : x ∈ A) : x⁻¹ ∈ A :=
  adjoin_le_iff.mpr (Set.singleton_subset_iff.mpr hx) int.inv_mem_adjoin


/-- An integral subalgebra of a division ring over a field is closed under inverses. -/
theorem Algebra.IsIntegral.inv_mem [Algebra.IsIntegral R A] (hx : x ∈ A) : x⁻¹ ∈ A :=
  ((isIntegral_algHom_iff A.val Subtype.val_injective).mpr <|
    Algebra.IsIntegral.isIntegral (⟨x, hx⟩ : A)).inv_mem hx


/-- The inverse of an integral element in a division ring over a field is also integral. -/
theorem IsIntegral.inv (int : IsIntegral R x) : IsIntegral R x⁻¹ :=
  .of_mem_of_fg _ int.fg_adjoin_singleton _ int.inv_mem_adjoin


theorem IsIntegral.mem_of_inv_mem (int : IsIntegral R x) (inv_mem : x⁻¹ ∈ A) : x ∈ A := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : Field R
    inst✝¹ : DivisionRing S
    inst✝ : Algebra R S
    x : S
    A : Subalgebra R S
    int : IsIntegral R x
    inv_mem : Membership.mem A (Inv.inv x)
    ⊢ Membership.mem A x
  -/
  rw [← inv_inv x]; exact int.inv.inv_mem inv_mem
                    /-
                      🎉 no goals
                    -/


/-- The [Kurosh problem](https://en.wikipedia.org/wiki/Kurosh_problem) asks to show that
  this is still true when `A` is not necessarily commutative and `R` is a field, but it has
  been solved in the negative. See https://arxiv.org/pdf/1706.02383.pdf for criteria for a
  finitely generated algebraic (= integral) algebra over a field to be finite dimensional.

This could be an `instance`, but we tend to go from `Module.Finite` to `IsIntegral`/`IsAlgebraic`,
and making it an instance will cause the search to be complicated a lot.
-/
theorem Algebra.IsIntegral.finite [Algebra.IsIntegral R A] [h' : Algebra.FiniteType R A] :
    Module.Finite R A :=
  have ⟨s, hs⟩ := h'
      /-
        R : Type u_1
        A : Type u_2
        inst✝³ : CommRing R
        inst✝² : CommRing A
        inst✝¹ : Algebra R A
        inst✝ : Algebra.IsIntegral R A
        h' : Algebra.FiniteType R A
        s : Finset A
        hs : Eq (Algebra.adjoin R ↑s) Top.top
        ⊢ Top.top.FG
      -/
  ⟨by apply hs ▸ fg_adjoin_of_finite s.finite_toSet fun x _ ↦ Algebra.IsIntegral.isIntegral x⟩
      /-
        🎉 no goals
      -/


/-- finite = integral + finite type -/
theorem Algebra.finite_iff_isIntegral_and_finiteType :
    Module.Finite R A ↔ Algebra.IsIntegral R A ∧ Algebra.FiniteType R A :=
  ⟨fun _ ↦ ⟨⟨.of_finite R⟩, inferInstance⟩, fun ⟨h, _⟩ ↦ h.finite⟩


theorem RingHom.IsIntegral.to_finite (h : f.IsIntegral) (h' : f.FiniteType) : f.Finite :=
  let _ := f.toAlgebra
  let _ : Algebra.IsIntegral R S := ⟨h⟩
  Algebra.IsIntegral.finite (h' := h')


alias RingHom.Finite.of_isIntegral_of_finiteType := RingHom.IsIntegral.to_finite


/-- finite = integral + finite type -/
theorem RingHom.finite_iff_isIntegral_and_finiteType : f.Finite ↔ f.IsIntegral ∧ f.FiniteType :=
  ⟨fun h ↦ ⟨h.to_isIntegral, h.to_finiteType⟩, fun ⟨h, h'⟩ ↦ h.to_finite h'⟩


theorem mem_integralClosure_iff_mem_fg {r : A} :
    r ∈ integralClosure R A ↔ ∃ M : Subalgebra R A, M.toSubmodule.FG ∧ r ∈ M :=
  ⟨fun hr =>
    ⟨Algebra.adjoin R {r}, hr.fg_adjoin_singleton, Algebra.subset_adjoin rfl⟩,
    fun ⟨M, Hf, hrM⟩ => .of_mem_of_fg M Hf _ hrM⟩


theorem adjoin_le_integralClosure {x : A} (hx : IsIntegral R x) :
    Algebra.adjoin R {x} ≤ integralClosure R A := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    x : A
    hx : IsIntegral R x
    ⊢ LE.le (Algebra.adjoin R (Singleton.singleton x)) (integralClosure R A)
  -/
  rw [Algebra.adjoin_le_iff]
  /-
    R : Type u_1
    A : Type u_2
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    x : A
    hx : IsIntegral R x
    ⊢ HasSubset.Subset (Singleton.singleton x) ↑(integralClosure R A)
  -/
  simp only [SetLike.mem_coe, Set.singleton_subset_iff]
  /-
    R : Type u_1
    A : Type u_2
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    x : A
    hx : IsIntegral R x
    ⊢ Membership.mem (integralClosure R A) x
  -/
  exact hx
  /-
    🎉 no goals
  -/


theorem le_integralClosure_iff_isIntegral {S : Subalgebra R A} :
    S ≤ integralClosure R A ↔ Algebra.IsIntegral R S :=
  SetLike.forall.symm.trans <|
    (forall_congr' fun x =>
      show IsIntegral R (algebraMap S A x) ↔ IsIntegral R x from
        isIntegral_algebraMap_iff Subtype.coe_injective).trans
      Algebra.isIntegral_def.symm


theorem Algebra.IsIntegral.adjoin {S : Set A} (hS : ∀ x ∈ S, IsIntegral R x) :
    Algebra.IsIntegral R (Algebra.adjoin R S) :=
  le_integralClosure_iff_isIntegral.mp <| adjoin_le_iff.mpr hS


theorem integralClosure_eq_top_iff : integralClosure R A = ⊤ ↔ Algebra.IsIntegral R A := by
  rw [← top_le_iff, le_integralClosure_iff_isIntegral,
      (Subalgebra.topEquiv (R := R) (A := A)).isIntegral_iff] -- explicit arguments for speedup


theorem Algebra.isIntegral_sup {S T : Subalgebra R A} :
    Algebra.IsIntegral R (S ⊔ T : Subalgebra R A) ↔
      Algebra.IsIntegral R S ∧ Algebra.IsIntegral R T := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    S T : Subalgebra R A
    ⊢ Iff (Algebra.IsIntegral R (Subtype fun x => Membership.mem (Max.max S T) x)) …
  -/
  simp_rw [← le_integralClosure_iff_isIntegral, sup_le_iff]
  /-
    🎉 no goals
  -/


theorem Algebra.isIntegral_iSup {ι} (S : ι → Subalgebra R A) :
    Algebra.IsIntegral R ↑(iSup S) ↔ ∀ i, Algebra.IsIntegral R (S i) := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    ι : Sort u_5
    S : ι → Subalgebra R A
    ⊢ Iff (Algebra.IsIntegral R (Subtype fun x => Membership.mem (iSup S) x)) (∀ ( …
  -/
  simp_rw [← le_integralClosure_iff_isIntegral, iSup_le_iff]
  /-
    🎉 no goals
  -/


/-- Mapping an integral closure along an `AlgEquiv` gives the integral closure. -/
theorem integralClosure_map_algEquiv [Algebra R S] (f : A ≃ₐ[R] S) :
    (integralClosure R A).map (f : A →ₐ[R] S) = integralClosure R S := by
  /-
    R : Type u_1
    A : Type u_2
    S : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : CommRing A
    inst✝² : CommRing S
    inst✝¹ : Algebra R A
    inst✝ : Algebra R S
    f : AlgEquiv R A S
    ⊢ Eq (Subalgebra.map (↑f) (integralClosure R A)) (integralClosure R S)
  -/
  ext y
  /-
    case h
    R : Type u_1
    A : Type u_2
    S : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : CommRing A
    inst✝² : CommRing S
    inst✝¹ : Algebra R A
    inst✝ : Algebra R S
    f : AlgEquiv R A S
    y : S
    ⊢ Iff (Membership.mem (Subalgebra.map (↑f) (integralClosure R A)) y) (Membersh …
  -/
  rw [Subalgebra.mem_map]
  /-
    case h
    R : Type u_1
    A : Type u_2
    S : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : CommRing A
    inst✝² : CommRing S
    inst✝¹ : Algebra R A
    inst✝ : Algebra R S
    f : AlgEquiv R A S
    y : S
    ⊢ Iff (Exists fun x => And (Membership.mem (integralClosure R A) x) (Eq (↑f x) …
  -/
  constructor
    /-
      case h.mp
      R : Type u_1
      A : Type u_2
      S : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : CommRing A
      inst✝² : CommRing S
      inst✝¹ : Algebra R A
      inst✝ : Algebra R S
      f : AlgEquiv R A S
      y : S
      ⊢ (Exists fun x => And (Membership.mem (integralClosure R A) x) (Eq (↑f x) y)) …
    -/
  · rintro ⟨x, hx, rfl⟩
    /-
      case h.mp.intro.intro
      R : Type u_1
      A : Type u_2
      S : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : CommRing A
      inst✝² : CommRing S
      inst✝¹ : Algebra R A
      inst✝ : Algebra R S
      f : AlgEquiv R A S
      x : A
      hx : Membership.mem (integralClosure R A) x
      ⊢ Membership.mem (integralClosure R S) (↑f x)
    -/
    exact hx.map f
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      R : Type u_1
      A : Type u_2
      S : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : CommRing A
      inst✝² : CommRing S
      inst✝¹ : Algebra R A
      inst✝ : Algebra R S
      f : AlgEquiv R A S
      y : S
      ⊢ Membership.mem (integralClosure R S) y → Exists fun x => And (Membership.mem …
    -/
  · intro hy
    /-
      case h.mpr
      R : Type u_1
      A : Type u_2
      S : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : CommRing A
      inst✝² : CommRing S
      inst✝¹ : Algebra R A
      inst✝ : Algebra R S
      f : AlgEquiv R A S
      y : S
      hy : Membership.mem (integralClosure R S) y
      ⊢ Exists fun x => And (Membership.mem (integralClosure R A) x) (Eq (↑f x) y)
    -/
    use f.symm y, hy.map (f.symm : S →ₐ[R] A)
    /-
      case right
      R : Type u_1
      A : Type u_2
      S : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : CommRing A
      inst✝² : CommRing S
      inst✝¹ : Algebra R A
      inst✝ : Algebra R S
      f : AlgEquiv R A S
      y : S
      hy : Membership.mem (integralClosure R S) y
      ⊢ Eq (↑f (f.symm y)) y
    -/
    simp
    /-
      🎉 no goals
    -/


/-- An `AlgHom` between two rings restrict to an `AlgHom` between the integral closures inside
them. -/
def AlgHom.mapIntegralClosure [Algebra R S] (f : A →ₐ[R] S) :
    integralClosure R A →ₐ[R] integralClosure R S :=
  (f.restrictDomain (integralClosure R A)).codRestrict (integralClosure R S) (fun ⟨_, h⟩ => h.map f)


@[simp]
theorem AlgHom.coe_mapIntegralClosure [Algebra R S] (f : A →ₐ[R] S)
    (x : integralClosure R A) : (f.mapIntegralClosure x : S) = f (x : A) := rfl


/-- An `AlgEquiv` between two rings restrict to an `AlgEquiv` between the integral closures inside
them. -/
def AlgEquiv.mapIntegralClosure [Algebra R S] (f : A ≃ₐ[R] S) :
    integralClosure R A ≃ₐ[R] integralClosure R S :=
  AlgEquiv.ofAlgHom (f : A →ₐ[R] S).mapIntegralClosure (f.symm : S →ₐ[R] A).mapIntegralClosure
    (AlgHom.ext fun _ ↦ Subtype.ext (f.right_inv _))
    (AlgHom.ext fun _ ↦ Subtype.ext (f.left_inv _))


@[simp]
theorem AlgEquiv.coe_mapIntegralClosure [Algebra R S] (f : A ≃ₐ[R] S)
    (x : integralClosure R A) : (f.mapIntegralClosure x : S) = f (x : A) := rfl


theorem integralClosure.isIntegral (x : integralClosure R A) : IsIntegral R x :=
  let ⟨p, hpm, hpx⟩ := x.2
  ⟨p, hpm,
    Subtype.eq <| by
      /-
        R : Type u_1
        A : Type u_2
        inst✝² : CommRing R
        inst✝¹ : CommRing A
        inst✝ : Algebra R A
        x : Subtype fun x => Membership.mem (integralClosure R A) x
        p : Polynomial R
        hpm : p.Monic
        hpx : Eq (Polynomial.eval₂ (algebraMap R A) (↑x) p) 0
        ⊢ Eq ↑(Polynomial.eval₂ (algebraMap R (Subtype fun x => Membership.mem (integr …
      -/
      rwa [← aeval_def, ← Subalgebra.val_apply, aeval_algHom_apply] at hpx⟩
      /-
        🎉 no goals
      -/


instance integralClosure.AlgebraIsIntegral : Algebra.IsIntegral R (integralClosure R A) :=
  ⟨integralClosure.isIntegral⟩


theorem IsIntegral.of_mul_unit {x y : B} {r : R} (hr : algebraMap R B r * y = 1)
    (hx : IsIntegral R (x * y)) : IsIntegral R x := by
  /-
    R : Type u_1
    B : Type u_3
    inst✝² : CommRing R
    inst✝¹ : Ring B
    inst✝ : Algebra R B
    x y : B
    r : R
    hr : Eq (HMul.hMul ((algebraMap R B) r) y) 1
    hx : IsIntegral R (HMul.hMul x y)
    ⊢ IsIntegral R x
  -/
  obtain ⟨p, p_monic, hp⟩ := hx
  /-
    case intro.intro
    R : Type u_1
    B : Type u_3
    inst✝² : CommRing R
    inst✝¹ : Ring B
    inst✝ : Algebra R B
    x y : B
    r : R
    hr : Eq (HMul.hMul ((algebraMap R B) r) y) 1
    p : Polynomial R
    p_monic : p.Monic
    hp : Eq (Polynomial.eval₂ (algebraMap R B) (HMul.hMul x y) p) 0
    ⊢ IsIntegral R x
  -/
  refine ⟨scaleRoots p r, (monic_scaleRoots_iff r).2 p_monic, ?_⟩
  /-
    case intro.intro
    R : Type u_1
    B : Type u_3
    inst✝² : CommRing R
    inst✝¹ : Ring B
    inst✝ : Algebra R B
    x y : B
    r : R
    hr : Eq (HMul.hMul ((algebraMap R B) r) y) 1
    p : Polynomial R
    p_monic : p.Monic
    hp : Eq (Polynomial.eval₂ (algebraMap R B) (HMul.hMul x y) p) 0
    ⊢ Eq (Polynomial.eval₂ (algebraMap R B) x (p.scaleRoots r)) 0
  -/
  convert scaleRoots_aeval_eq_zero hp
  /-
    case h.e'_2.h.e
    R : Type u_1
    B : Type u_3
    inst✝² : CommRing R
    inst✝¹ : Ring B
    inst✝ : Algebra R B
    x y : B
    r : R
    hr : Eq (HMul.hMul ((algebraMap R B) r) y) 1
    p : Polynomial R
    p_monic : p.Monic
    hp : Eq (Polynomial.eval₂ (algebraMap R B) (HMul.hMul x y) p) 0
    ⊢ Eq (Polynomial.eval₂ (algebraMap R B) x) ⇑(Polynomial.aeval (HMul.hMul ((alg …
  -/
  rw [Algebra.commutes] at hr ⊢
  /-
    case h.e'_2.h.e
    R : Type u_1
    B : Type u_3
    inst✝² : CommRing R
    inst✝¹ : Ring B
    inst✝ : Algebra R B
    x y : B
    r : R
    hr : Eq (HMul.hMul y ((algebraMap R B) r)) 1
    p : Polynomial R
    p_monic : p.Monic
    hp : Eq (Polynomial.eval₂ (algebraMap R B) (HMul.hMul x y) p) 0
    ⊢ Eq (Polynomial.eval₂ (algebraMap R B) x) ⇑(Polynomial.aeval (HMul.hMul (HMul …
  -/
  rw [mul_assoc, hr, mul_one]; rfl
                               /-
                                 🎉 no goals
                               -/


theorem RingHom.IsIntegralElem.of_mul_unit (x y : S) (r : R) (hr : f r * y = 1)
    (hx : f.IsIntegralElem (x * y)) : f.IsIntegralElem x :=
  letI : Algebra R S := f.toAlgebra
  IsIntegral.of_mul_unit hr hx


/-- Generalization of `IsIntegral.of_mem_closure` bootstrapped up from that lemma -/
theorem IsIntegral.of_mem_closure' (G : Set A) (hG : ∀ x ∈ G, IsIntegral R x) :
    ∀ x ∈ Subring.closure G, IsIntegral R x := fun _ hx ↦
  Subring.closure_induction hG isIntegral_zero isIntegral_one (fun _ _ _ _ ↦ IsIntegral.add)
    (fun _ _ ↦ IsIntegral.neg) (fun _ _ _ _ ↦ IsIntegral.mul) hx


theorem IsIntegral.of_mem_closure'' {S : Type*} [CommRing S] {f : R →+* S} (G : Set S)
    (hG : ∀ x ∈ G, f.IsIntegralElem x) : ∀ x ∈ Subring.closure G, f.IsIntegralElem x := fun x hx =>
  @IsIntegral.of_mem_closure' R S _ _ f.toAlgebra G hG x hx


theorem IsIntegral.pow {x : B} (h : IsIntegral R x) (n : ℕ) : IsIntegral R (x ^ n) :=
  .of_mem_of_fg _ h.fg_adjoin_singleton _ <|
                             /-
                               R : Type u_1
                               B : Type u_3
                               inst✝² : CommRing R
                               inst✝¹ : Ring B
                               inst✝ : Algebra R B
                               x : B
                               h : IsIntegral R x
                               n : Nat
                               ⊢ Membership.mem (Algebra.adjoin R (Singleton.singleton x)) x
                             -/
    Subalgebra.pow_mem _ (by exact Algebra.subset_adjoin rfl) _
                             /-
                               🎉 no goals
                             -/


theorem IsIntegral.nsmul {x : B} (h : IsIntegral R x) (n : ℕ) : IsIntegral R (n • x) :=
  h.smul n


theorem IsIntegral.zsmul {x : B} (h : IsIntegral R x) (n : ℤ) : IsIntegral R (n • x) :=
  h.smul n


theorem IsIntegral.multiset_prod {s : Multiset A} (h : ∀ x ∈ s, IsIntegral R x) :
    IsIntegral R s.prod :=
  (integralClosure R A).multiset_prod_mem h


theorem IsIntegral.multiset_sum {s : Multiset A} (h : ∀ x ∈ s, IsIntegral R x) :
    IsIntegral R s.sum :=
  (integralClosure R A).multiset_sum_mem h


theorem IsIntegral.prod {α : Type*} {s : Finset α} (f : α → A) (h : ∀ x ∈ s, IsIntegral R (f x)) :
    IsIntegral R (∏ x ∈ s, f x) :=
  (integralClosure R A).prod_mem h


theorem IsIntegral.sum {α : Type*} {s : Finset α} (f : α → A) (h : ∀ x ∈ s, IsIntegral R (f x)) :
    IsIntegral R (∑ x ∈ s, f x) :=
  (integralClosure R A).sum_mem h


theorem IsIntegral.det {n : Type*} [Fintype n] [DecidableEq n] {M : Matrix n n A}
    (h : ∀ i j, IsIntegral R (M i j)) : IsIntegral R M.det := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : CommRing A
    inst✝² : Algebra R A
    n : Type u_5
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    M : Matrix n n A
    h : ∀ (i j : n), IsIntegral R (M i j)
    ⊢ IsIntegral R M.det
  -/
  rw [Matrix.det_apply]
  /-
    R : Type u_1
    A : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : CommRing A
    inst✝² : Algebra R A
    n : Type u_5
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    M : Matrix n n A
    h : ∀ (i j : n), IsIntegral R (M i j)
    ⊢ IsIntegral R (Finset.univ.sum fun σ => HSMul.hSMul (Equiv.Perm.sign σ) (Fins …
  -/
  exact IsIntegral.sum _ fun σ _hσ ↦ (IsIntegral.prod _ fun i _hi => h _ _).zsmul _
  /-
    🎉 no goals
  -/


@[simp]
theorem IsIntegral.pow_iff {x : A} {n : ℕ} (hn : 0 < n) : IsIntegral R (x ^ n) ↔ IsIntegral R x :=
  ⟨IsIntegral.of_pow hn, fun hx ↦ hx.pow n⟩


theorem IsIntegral.tmul (x : A) {y : B} (h : IsIntegral R y) : IsIntegral A (x ⊗ₜ[R] y) := by
  /-
    R : Type u_1
    A : Type u_2
    B : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : CommRing A
    inst✝² : Ring B
    inst✝¹ : Algebra R A
    inst✝ : Algebra R B
    x : A
    y : B
    h : IsIntegral R y
    ⊢ IsIntegral A (TensorProduct.tmul R x y)
  -/
  rw [← mul_one x, ← smul_eq_mul, ← smul_tmul']
  exact smul _ (h.map_of_comp_eq (algebraMap R A)
    (Algebra.TensorProduct.includeRight (R := R) (A := A) (B := B)).toRingHom
    Algebra.TensorProduct.includeLeftRingHom_comp_algebraMap)


/-- The monic polynomial whose roots are `p.leadingCoeff * x` for roots `x` of `p`. -/
@[deprecated (since := "2024-11-30")]
alias normalizeScaleRoots := integralNormalization


@[deprecated (since := "2024-11-30")]
alias normalizeScaleRoots_coeff_mul_leadingCoeff_pow :=
  integralNormalization_coeff_mul_leadingCoeff_pow


@[deprecated (since := "2024-11-30")]
alias leadingCoeff_smul_normalizeScaleRoots := leadingCoeff_smul_integralNormalization


@[deprecated (since := "2024-11-30")]
alias normalizeScaleRoots_support := support_integralNormalization_subset


@[deprecated (since := "2024-11-30")]
alias normalizeScaleRoots_degree := integralNormalization_degree


@[deprecated (since := "2024-11-30")]
alias normalizeScaleRoots_eval₂_leadingCoeff_mul := integralNormalization_eval₂_leadingCoeff_mul


@[deprecated (since := "2024-11-30")]
alias normalizeScaleRoots_monic := monic_integralNormalization


/-- Given a `p : R[X]` and a `x : S` such that `p.eval₂ f x = 0`,
`f p.leadingCoeff * x` is integral. -/
theorem RingHom.isIntegralElem_leadingCoeff_mul (h : p.eval₂ f x = 0) :
    f.IsIntegralElem (f p.leadingCoeff * x) := by
  /-
    R : Type u_1
    S : Type u_4
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    p : Polynomial R
    x : S
    h : Eq (Polynomial.eval₂ f x p) 0
    ⊢ f.IsIntegralElem (HMul.hMul (f p.leadingCoeff) x)
  -/
  by_cases h' : 1 ≤ p.natDegree
    /-
      case pos
      R : Type u_1
      S : Type u_4
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      p : Polynomial R
      x : S
      h : Eq (Polynomial.eval₂ f x p) 0
      h' : LE.le 1 p.natDegree
      ⊢ f.IsIntegralElem (HMul.hMul (f p.leadingCoeff) x)
    -/
  · use integralNormalization p
    have : p ≠ 0 := fun h'' => by
      rw [h'', natDegree_zero] at h'
      exact Nat.not_succ_le_zero 0 h'
    /-
      case h
      R : Type u_1
      S : Type u_4
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      p : Polynomial R
      x : S
      h : Eq (Polynomial.eval₂ f x p) 0
      h' : LE.le 1 p.natDegree
      this : Ne p 0
      ⊢ And p.integralNormalization.Monic (Eq (Polynomial.eval₂ f (HMul.hMul (f p.le …
    -/
    use monic_integralNormalization this
    /-
      case right
      R : Type u_1
      S : Type u_4
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      p : Polynomial R
      x : S
      h : Eq (Polynomial.eval₂ f x p) 0
      h' : LE.le 1 p.natDegree
      this : Ne p 0
      ⊢ Eq (Polynomial.eval₂ f (HMul.hMul (f p.leadingCoeff) x) p.integralNormalizat …
    -/
    rw [integralNormalization_eval₂_leadingCoeff_mul h' f x, h, mul_zero]
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      S : Type u_4
      inst✝¹ : CommRing R
      inst✝ : CommRing S
      f : RingHom R S
      p : Polynomial R
      x : S
      h : Eq (Polynomial.eval₂ f x p) 0
      h' : Not (LE.le 1 p.natDegree)
      ⊢ f.IsIntegralElem (HMul.hMul (f p.leadingCoeff) x)
    -/
  · by_cases hp : p.map f = 0
      /-
        case pos
        R : Type u_1
        S : Type u_4
        inst✝¹ : CommRing R
        inst✝ : CommRing S
        f : RingHom R S
        p : Polynomial R
        x : S
        h : Eq (Polynomial.eval₂ f x p) 0
        h' : Not (LE.le 1 p.natDegree)
        hp : Eq (Polynomial.map f p) 0
        ⊢ f.IsIntegralElem (HMul.hMul (f p.leadingCoeff) x)
      -/
    · apply_fun fun q => coeff q p.natDegree at hp
      /-
        case pos
        R : Type u_1
        S : Type u_4
        inst✝¹ : CommRing R
        inst✝ : CommRing S
        f : RingHom R S
        p : Polynomial R
        x : S
        h : Eq (Polynomial.eval₂ f x p) 0
        h' : Not (LE.le 1 p.natDegree)
        hp : Eq ((Polynomial.map f p).coeff p.natDegree) (Polynomial.coeff 0 p.natDegr …
        ⊢ f.IsIntegralElem (HMul.hMul (f p.leadingCoeff) x)
      -/
      rw [coeff_map, coeff_zero, coeff_natDegree] at hp
      /-
        case pos
        R : Type u_1
        S : Type u_4
        inst✝¹ : CommRing R
        inst✝ : CommRing S
        f : RingHom R S
        p : Polynomial R
        x : S
        h : Eq (Polynomial.eval₂ f x p) 0
        h' : Not (LE.le 1 p.natDegree)
        hp : Eq (f p.leadingCoeff) 0
        ⊢ f.IsIntegralElem (HMul.hMul (f p.leadingCoeff) x)
      -/
      rw [hp, zero_mul]
      /-
        case pos
        R : Type u_1
        S : Type u_4
        inst✝¹ : CommRing R
        inst✝ : CommRing S
        f : RingHom R S
        p : Polynomial R
        x : S
        h : Eq (Polynomial.eval₂ f x p) 0
        h' : Not (LE.le 1 p.natDegree)
        hp : Eq (f p.leadingCoeff) 0
        ⊢ f.IsIntegralElem 0
      -/
      exact f.isIntegralElem_zero
      /-
        🎉 no goals
      -/
      /-
        case neg
        R : Type u_1
        S : Type u_4
        inst✝¹ : CommRing R
        inst✝ : CommRing S
        f : RingHom R S
        p : Polynomial R
        x : S
        h : Eq (Polynomial.eval₂ f x p) 0
        h' : Not (LE.le 1 p.natDegree)
        hp : Not (Eq (Polynomial.map f p) 0)
        ⊢ f.IsIntegralElem (HMul.hMul (f p.leadingCoeff) x)
      -/
    · rw [Nat.one_le_iff_ne_zero, Classical.not_not] at h'
      /-
        case neg
        R : Type u_1
        S : Type u_4
        inst✝¹ : CommRing R
        inst✝ : CommRing S
        f : RingHom R S
        p : Polynomial R
        x : S
        h : Eq (Polynomial.eval₂ f x p) 0
        h' : Eq p.natDegree 0
        hp : Not (Eq (Polynomial.map f p) 0)
        ⊢ f.IsIntegralElem (HMul.hMul (f p.leadingCoeff) x)
      -/
      rw [eq_C_of_natDegree_eq_zero h', eval₂_C] at h
      /-
        case neg
        R : Type u_1
        S : Type u_4
        inst✝¹ : CommRing R
        inst✝ : CommRing S
        f : RingHom R S
        p : Polynomial R
        x : S
        h : Eq (f (p.coeff 0)) 0
        h' : Eq p.natDegree 0
        hp : Not (Eq (Polynomial.map f p) 0)
        ⊢ f.IsIntegralElem (HMul.hMul (f p.leadingCoeff) x)
      -/
      suffices p.map f = 0 by exact (hp this).elim
      /-
        case neg
        R : Type u_1
        S : Type u_4
        inst✝¹ : CommRing R
        inst✝ : CommRing S
        f : RingHom R S
        p : Polynomial R
        x : S
        h : Eq (f (p.coeff 0)) 0
        h' : Eq p.natDegree 0
        hp : Not (Eq (Polynomial.map f p) 0)
        ⊢ Eq (Polynomial.map f p) 0
      -/
      rw [eq_C_of_natDegree_eq_zero h', map_C, h, C_eq_zero]
      /-
        🎉 no goals
      -/


/-- Given a `p : R[X]` and a root `x : S`,
then `p.leadingCoeff • x : S` is integral over `R`. -/
theorem isIntegral_leadingCoeff_smul [Algebra R S] (h : aeval x p = 0) :
    IsIntegral R (p.leadingCoeff • x) := by
  /-
    R : Type u_1
    S : Type u_4
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    p : Polynomial R
    x : S
    inst✝ : Algebra R S
    h : Eq ((Polynomial.aeval x) p) 0
    ⊢ IsIntegral R (HSMul.hSMul p.leadingCoeff x)
  -/
  rw [aeval_def] at h
  /-
    R : Type u_1
    S : Type u_4
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    p : Polynomial R
    x : S
    inst✝ : Algebra R S
    h : Eq (Polynomial.eval₂ (algebraMap R S) x p) 0
    ⊢ IsIntegral R (HSMul.hSMul p.leadingCoeff x)
  -/
  rw [Algebra.smul_def]
  /-
    R : Type u_1
    S : Type u_4
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    p : Polynomial R
    x : S
    inst✝ : Algebra R S
    h : Eq (Polynomial.eval₂ (algebraMap R S) x p) 0
    ⊢ IsIntegral R (HMul.hMul ((algebraMap R S) p.leadingCoeff) x)
  -/
  exact (algebraMap R S).isIntegralElem_leadingCoeff_mul p x h
  /-
    🎉 no goals
  -/


lemma Polynomial.Monic.quotient_isIntegralElem {g : S[X]} (mon : g.Monic) {I : Ideal S[X]}
    (h : g ∈ I) :
    ((Ideal.Quotient.mk I).comp (algebraMap S S[X])).IsIntegralElem (Ideal.Quotient.mk I X) := by
  exact ⟨g, mon, by
  rw [← (Ideal.Quotient.eq_zero_iff_mem.mpr h), eval₂_eq_sum_range]
  nth_rw 3 [(as_sum_range_C_mul_X_pow g)]
  simp only [map_sum, algebraMap_eq, RingHom.coe_comp, Function.comp_apply, map_mul, map_pow]⟩

/- If `I` is an ideal of the polynomial ring `S[X]` and contains a monic polynomial `f`,
then `S[X]/I` is integral over `S`. -/

lemma Polynomial.Monic.quotient_isIntegral {g : S[X]} (mon : g.Monic) {I : Ideal S[X]} (h : g ∈ I) :
    ((Ideal.Quotient.mkₐ S I).comp (Algebra.ofId S S[X])).IsIntegral := by
  have eq_top : Algebra.adjoin S {(Ideal.Quotient.mkₐ S I) X} = ⊤ := by
    ext g
    constructor
    · simp only [Algebra.mem_top, implies_true]
    · intro _
      obtain ⟨g', hg⟩ := Ideal.Quotient.mkₐ_surjective S I g
      have : g = (Polynomial.aeval ((Ideal.Quotient.mkₐ S I) X)) g' := by
        nth_rw 1 [← hg, aeval_eq_sum_range' (lt_add_one _),
          as_sum_range_C_mul_X_pow g', map_sum]
        simp only [Polynomial.C_mul', ← map_pow, map_smul]
      exact this ▸ (aeval_mem_adjoin_singleton S ((Ideal.Quotient.mk I) Polynomial.X))
  exact fun a ↦ (eq_top ▸ (adjoin_le_integralClosure (mon.quotient_isIntegralElem h)))
    Algebra.mem_top


instance integralClosure.isIntegralClosure (R A : Type*) [CommRing R] [CommRing A] [Algebra R A] :
    IsIntegralClosure (integralClosure R A) R A where
  algebraMap_injective' := Subtype.coe_injective
                                                    /-
                                                      R : Type u_1
                                                      A : Type u_2
                                                      inst✝² : CommRing R
                                                      inst✝¹ : CommRing A
                                                      inst✝ : Algebra R A
                                                      x : A
                                                      ⊢ (Exists fun y => Eq ((algebraMap (Subtype fun x => Membership.mem (integralC …
                                                    -/
  isIntegral_iff {x} := ⟨fun h => ⟨⟨x, h⟩, rfl⟩, by rintro ⟨⟨_, h⟩, rfl⟩; exact h⟩
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


theorem algebraMap_injective (A R B : Type*) [CommRing R] [CommSemiring A] [CommRing B]
    [Algebra R B] [Algebra A B] [IsIntegralClosure A R B] : Function.Injective (algebraMap A B) :=
  algebraMap_injective' R


protected theorem isIntegral [Algebra R A] [IsScalarTower R A B] (x : A) : IsIntegral R x :=
  (isIntegral_algebraMap_iff (algebraMap_injective A R B)).mp <|
    show IsIntegral R (algebraMap A B x) from isIntegral_iff.mpr ⟨x, rfl⟩


theorem isIntegral_algebra [Algebra R A] [IsScalarTower R A B] : Algebra.IsIntegral R A :=
  ⟨fun x => IsIntegralClosure.isIntegral R B x⟩


theorem noZeroSMulDivisors [Algebra R A] [IsScalarTower R A B] [NoZeroSMulDivisors R B] :
    NoZeroSMulDivisors R A := by
  refine
    Function.Injective.noZeroSMulDivisors _ (IsIntegralClosure.algebraMap_injective A R B)
      (map_zero _) fun _ _ => ?_
  /-
    R : Type u_1
    A : Type u_2
    B : Type u_3
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing A
    inst✝⁶ : CommRing B
    inst✝⁵ : Algebra R B
    inst✝⁴ : Algebra A B
    inst✝³ : IsIntegralClosure A R B
    inst✝² : Algebra R A
    inst✝¹ : IsScalarTower R A B
    inst✝ : NoZeroSMulDivisors R B
    x✝¹ : R
    x✝ : A
    ⊢ Eq ((algebraMap A B) (HSMul.hSMul x✝¹ x✝)) (HSMul.hSMul x✝¹ ((algebraMap A B …
  -/
  simp only [Algebra.algebraMap_eq_smul_one, IsScalarTower.smul_assoc]
  /-
    🎉 no goals
  -/


/-- If `x : B` is integral over `R`, then it is an element of the integral closure of `R` in `B`. -/
noncomputable def mk' (x : B) (hx : IsIntegral R x) : A :=
  Classical.choose (isIntegral_iff.mp hx)


@[simp]
theorem algebraMap_mk' (x : B) (hx : IsIntegral R x) : algebraMap A B (mk' A x hx) = x :=
  Classical.choose_spec (isIntegral_iff.mp hx)


@[simp]
theorem mk'_one (h : IsIntegral R (1 : B) := isIntegral_one) : mk' A 1 h = 1 :=
                                   /-
                                     R : Type u_1
                                     A : Type u_2
                                     B : Type u_3
                                     inst✝⁵ : CommRing R
                                     inst✝⁴ : CommRing A
                                     inst✝³ : CommRing B
                                     inst✝² : Algebra R B
                                     inst✝¹ : Algebra A B
                                     inst✝ : IsIntegralClosure A R B
                                     h : optParam (IsIntegral R 1) ⋯
                                     ⊢ Eq ((algebraMap A B) (IsIntegralClosure.mk' A 1 h)) ((algebraMap A B) 1)
                                   -/
  algebraMap_injective A R B <| by rw [algebraMap_mk', RingHom.map_one]
                                   /-
                                     🎉 no goals
                                   -/


@[simp]
theorem mk'_zero (h : IsIntegral R (0 : B) := isIntegral_zero) : mk' A 0 h = 0 :=
                                   /-
                                     R : Type u_1
                                     A : Type u_2
                                     B : Type u_3
                                     inst✝⁵ : CommRing R
                                     inst✝⁴ : CommRing A
                                     inst✝³ : CommRing B
                                     inst✝² : Algebra R B
                                     inst✝¹ : Algebra A B
                                     inst✝ : IsIntegralClosure A R B
                                     h : optParam (IsIntegral R 0) ⋯
                                     ⊢ Eq ((algebraMap A B) (IsIntegralClosure.mk' A 0 h)) ((algebraMap A B) 0)
                                   -/
  algebraMap_injective A R B <| by rw [algebraMap_mk', RingHom.map_zero]
                                   /-
                                     🎉 no goals
                                   -/

-- Porting note: Left-hand side does not simplify @[simp]

theorem mk'_add (x y : B) (hx : IsIntegral R x) (hy : IsIntegral R y) :
    mk' A (x + y) (hx.add hy) = mk' A x hx + mk' A y hy :=
                                   /-
                                     R : Type u_1
                                     A : Type u_2
                                     B : Type u_3
                                     inst✝⁵ : CommRing R
                                     inst✝⁴ : CommRing A
                                     inst✝³ : CommRing B
                                     inst✝² : Algebra R B
                                     inst✝¹ : Algebra A B
                                     inst✝ : IsIntegralClosure A R B
                                     x y : B
                                     hx : IsIntegral R x
                                     hy : IsIntegral R y
                                     ⊢ Eq ((algebraMap A B) (IsIntegralClosure.mk' A (HAdd.hAdd x y) ⋯)) ((algebraM …
                                   -/
  algebraMap_injective A R B <| by simp only [algebraMap_mk', RingHom.map_add]
                                   /-
                                     🎉 no goals
                                   -/

-- Porting note: Left-hand side does not simplify @[simp]

theorem mk'_mul (x y : B) (hx : IsIntegral R x) (hy : IsIntegral R y) :
    mk' A (x * y) (hx.mul hy) = mk' A x hx * mk' A y hy :=
                                   /-
                                     R : Type u_1
                                     A : Type u_2
                                     B : Type u_3
                                     inst✝⁵ : CommRing R
                                     inst✝⁴ : CommRing A
                                     inst✝³ : CommRing B
                                     inst✝² : Algebra R B
                                     inst✝¹ : Algebra A B
                                     inst✝ : IsIntegralClosure A R B
                                     x y : B
                                     hx : IsIntegral R x
                                     hy : IsIntegral R y
                                     ⊢ Eq ((algebraMap A B) (IsIntegralClosure.mk' A (HMul.hMul x y) ⋯)) ((algebraM …
                                   -/
  algebraMap_injective A R B <| by simp only [algebraMap_mk', RingHom.map_mul]
                                   /-
                                     🎉 no goals
                                   -/


@[simp]
theorem mk'_algebraMap [Algebra R A] [IsScalarTower R A B] (x : R)
    (h : IsIntegral R (algebraMap R B x) := isIntegral_algebraMap) :
    IsIntegralClosure.mk' A (algebraMap R B x) h = algebraMap R A x :=
                                   /-
                                     R : Type u_1
                                     A : Type u_2
                                     B : Type u_3
                                     inst✝⁷ : CommRing R
                                     inst✝⁶ : CommRing A
                                     inst✝⁵ : CommRing B
                                     inst✝⁴ : Algebra R B
                                     inst✝³ : Algebra A B
                                     inst✝² : IsIntegralClosure A R B
                                     inst✝¹ : Algebra R A
                                     inst✝ : IsScalarTower R A B
                                     x : R
                                     h : optParam (IsIntegral R ((algebraMap R B) x)) ⋯
                                     ⊢ Eq ((algebraMap A B) (IsIntegralClosure.mk' A ((algebraMap R B) x) h)) ((alg …
                                   -/
  algebraMap_injective A R B <| by rw [algebraMap_mk', ← IsScalarTower.algebraMap_apply]
                                   /-
                                     🎉 no goals
                                   -/


/-- The integral closure of a field in a commutative domain is always a field. -/
theorem isField [Algebra R A] [IsScalarTower R A B] [IsDomain A] (hR : IsField R) :
    IsField A :=
  have := IsIntegralClosure.isIntegral_algebra R (A := A) B
  isField_of_isIntegral_of_isField' hR


/-- If `B / S / R` is a tower of ring extensions where `S` is integral over `R`,
then `S` maps (uniquely) into an integral closure `B / A / R`. -/
noncomputable def lift : S →ₐ[R] A where
  toFun x := mk' A (algebraMap S B x) (IsIntegral.algebraMap
    (Algebra.IsIntegral.isIntegral (R := R) x))
                 /-
                   R : Type u_1
                   A : Type u_2
                   B : Type u_3
                   inst✝¹¹ : CommRing R
                   inst✝¹⁰ : CommRing A
                   inst✝⁹ : CommRing B
                   inst✝⁸ : Algebra R B
                   inst✝⁷ : Algebra A B
                   inst✝⁶ : IsIntegralClosure A R B
                   S : Type u_4
                   inst✝⁵ : CommRing S
                   inst✝⁴ : Algebra R S
                   inst✝³ : Algebra S B
                   inst✝² : IsScalarTower R S B
                   inst✝¹ : Algebra R A
                   inst✝ : IsScalarTower R A B
                   isIntegral : Algebra.IsIntegral R S
                   ⊢ Eq ((fun x => IsIntegralClosure.mk' A ((algebraMap S B) x) ⋯) 1) 1
                 -/
  map_one' := by simp only [RingHom.map_one, mk'_one]
                 /-
                   🎉 no goals
                 -/
                  /-
                    R : Type u_1
                    A : Type u_2
                    B : Type u_3
                    inst✝¹¹ : CommRing R
                    inst✝¹⁰ : CommRing A
                    inst✝⁹ : CommRing B
                    inst✝⁸ : Algebra R B
                    inst✝⁷ : Algebra A B
                    inst✝⁶ : IsIntegralClosure A R B
                    S : Type u_4
                    inst✝⁵ : CommRing S
                    inst✝⁴ : Algebra R S
                    inst✝³ : Algebra S B
                    inst✝² : IsScalarTower R S B
                    inst✝¹ : Algebra R A
                    inst✝ : IsScalarTower R A B
                    isIntegral : Algebra.IsIntegral R S
                    ⊢ Eq ((↑{ toFun := fun x => IsIntegralClosure.mk' A ((algebraMap S B) x) ⋯, ma …
                  -/
  map_zero' := by simp only [RingHom.map_zero, mk'_zero]
                     /-
                       R : Type u_1
                       A : Type u_2
                       B : Type u_3
                       inst✝¹¹ : CommRing R
                       inst✝¹⁰ : CommRing A
                       inst✝⁹ : CommRing B
                       inst✝⁸ : Algebra R B
                       inst✝⁷ : Algebra A B
                       inst✝⁶ : IsIntegralClosure A R B
                       S : Type u_4
                       inst✝⁵ : CommRing S
                       inst✝⁴ : Algebra R S
                       inst✝³ : Algebra S B
                       inst✝² : IsScalarTower R S B
                       inst✝¹ : Algebra R A
                       inst✝ : IsScalarTower R A B
                       isIntegral : Algebra.IsIntegral R S
                       x y : S
                       ⊢ Eq ({ toFun := fun x => IsIntegralClosure.mk' A ((algebraMap S B) x) ⋯, map_ …
                     -/
                  /-
                    🎉 no goals
                  -/
                     /-
                       🎉 no goals
                     -/
                     /-
                       R : Type u_1
                       A : Type u_2
                       B : Type u_3
                       inst✝¹¹ : CommRing R
                       inst✝¹⁰ : CommRing A
                       inst✝⁹ : CommRing B
                       inst✝⁸ : Algebra R B
                       inst✝⁷ : Algebra A B
                       inst✝⁶ : IsIntegralClosure A R B
                       S : Type u_4
                       inst✝⁵ : CommRing S
                       inst✝⁴ : Algebra R S
                       inst✝³ : Algebra S B
                       inst✝² : IsScalarTower R S B
                       inst✝¹ : Algebra R A
                       inst✝ : IsScalarTower R A B
                       isIntegral : Algebra.IsIntegral R S
                       x y : S
                       ⊢ Eq ((↑{ toFun := fun x => IsIntegralClosure.mk' A ((algebraMap S B) x) ⋯, ma …
                     -/
  map_add' x y := by simp_rw [← mk'_add, map_add]
                     /-
                       🎉 no goals
                     -/
  map_mul' x y := by simp_rw [← mk'_mul, RingHom.map_mul]
                    /-
                      R : Type u_1
                      A : Type u_2
                      B : Type u_3
                      inst✝¹¹ : CommRing R
                      inst✝¹⁰ : CommRing A
                      inst✝⁹ : CommRing B
                      inst✝⁸ : Algebra R B
                      inst✝⁷ : Algebra A B
                      inst✝⁶ : IsIntegralClosure A R B
                      S : Type u_4
                      inst✝⁵ : CommRing S
                      inst✝⁴ : Algebra R S
                      inst✝³ : Algebra S B
                      inst✝² : IsScalarTower R S B
                      inst✝¹ : Algebra R A
                      inst✝ : IsScalarTower R A B
                      isIntegral : Algebra.IsIntegral R S
                      x : R
                      ⊢ Eq ((↑↑{ toFun := fun x => IsIntegralClosure.mk' A ((algebraMap S B) x) ⋯, m …
                    -/
  commutes' x := by simp_rw [← IsScalarTower.algebraMap_apply, mk'_algebraMap]
                    /-
                      🎉 no goals
                    -/


@[simp]
theorem algebraMap_lift (x : S) : algebraMap A B (lift R A B x) = algebraMap S B x :=
  algebraMap_mk' A (algebraMap S B x) (IsIntegral.algebraMap
    (Algebra.IsIntegral.isIntegral (R := R) x))


/-- Integral closures are all isomorphic to each other. -/
noncomputable def equiv : A ≃ₐ[R] A' :=
  AlgEquiv.ofAlgHom
    (lift R A' B (isIntegral := isIntegral_algebra R B))
    (lift R A B (isIntegral := isIntegral_algebra R B))
        /-
          R : Type u_1
          A : Type u_2
          B : Type u_3
          inst✝¹² : CommRing R
          inst✝¹¹ : CommRing A
          inst✝¹⁰ : CommRing B
          inst✝⁹ : Algebra R B
          inst✝⁸ : Algebra A B
          inst✝⁷ : IsIntegralClosure A R B
          A' : Type u_4
          inst✝⁶ : CommRing A'
          inst✝⁵ : Algebra A' B
          inst✝⁴ : IsIntegralClosure A' R B
          inst✝³ : Algebra R A
          inst✝² : Algebra R A'
          inst✝¹ : IsScalarTower R A B
          inst✝ : IsScalarTower R A' B
          ⊢ Eq ((IsIntegralClosure.lift R A' B).comp (IsIntegralClosure.lift R A B)) (Al …
        -/
    (by ext x; apply algebraMap_injective A' R B; simp)
                                                  /-
                                                    🎉 no goals
                                                  -/
        /-
          R : Type u_1
          A : Type u_2
          B : Type u_3
          inst✝¹² : CommRing R
          inst✝¹¹ : CommRing A
          inst✝¹⁰ : CommRing B
          inst✝⁹ : Algebra R B
          inst✝⁸ : Algebra A B
          inst✝⁷ : IsIntegralClosure A R B
          A' : Type u_4
          inst✝⁶ : CommRing A'
          inst✝⁵ : Algebra A' B
          inst✝⁴ : IsIntegralClosure A' R B
          inst✝³ : Algebra R A
          inst✝² : Algebra R A'
          inst✝¹ : IsScalarTower R A B
          inst✝ : IsScalarTower R A' B
          ⊢ Eq ((IsIntegralClosure.lift R A B).comp (IsIntegralClosure.lift R A' B)) (Al …
        -/
    (by ext x; apply algebraMap_injective A R B; simp)
                                                 /-
                                                   🎉 no goals
                                                 -/


@[simp]
theorem algebraMap_equiv (x : A) : algebraMap A' B (equiv R A B A' x) = algebraMap A B x :=
  algebraMap_lift R A' B (isIntegral := isIntegral_algebra R B) x


/-- If A is an R-algebra all of whose elements are integral over R,
and x is an element of an A-algebra that is integral over A, then x is integral over R. -/
theorem isIntegral_trans [Algebra.IsIntegral R A] (x : B) (hx : IsIntegral A x) :
    IsIntegral R x := by
  /-
    R : Type u_1
    A : Type u_2
    B : Type u_3
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing A
    inst✝⁵ : Ring B
    inst✝⁴ : Algebra A B
    inst✝³ : Algebra R B
    inst✝² : Algebra R A
    inst✝¹ : IsScalarTower R A B
    inst✝ : Algebra.IsIntegral R A
    x : B
    hx : IsIntegral A x
    ⊢ IsIntegral R x
  -/
  rcases hx with ⟨p, pmonic, hp⟩
  /-
    case intro.intro
    R : Type u_1
    A : Type u_2
    B : Type u_3
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing A
    inst✝⁵ : Ring B
    inst✝⁴ : Algebra A B
    inst✝³ : Algebra R B
    inst✝² : Algebra R A
    inst✝¹ : IsScalarTower R A B
    inst✝ : Algebra.IsIntegral R A
    x : B
    p : Polynomial A
    pmonic : p.Monic
    hp : Eq (Polynomial.eval₂ (algebraMap A B) x p) 0
    ⊢ IsIntegral R x
  -/
  let S := adjoin R (p.coeffs : Set A)
  have : Module.Finite R S := ⟨(Subalgebra.toSubmodule S).fg_top.mpr <|
    fg_adjoin_of_finite p.coeffs.finite_toSet fun a _ ↦ Algebra.IsIntegral.isIntegral a⟩
  /-
    case intro.intro
    R : Type u_1
    A : Type u_2
    B : Type u_3
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing A
    inst✝⁵ : Ring B
    inst✝⁴ : Algebra A B
    inst✝³ : Algebra R B
    inst✝² : Algebra R A
    inst✝¹ : IsScalarTower R A B
    inst✝ : Algebra.IsIntegral R A
    x : B
    p : Polynomial A
    pmonic : p.Monic
    hp : Eq (Polynomial.eval₂ (algebraMap A B) x p) 0
    S : Subalgebra R A := Algebra.adjoin R ↑p.coeffs
    this : Module.Finite R (Subtype fun x => Membership.mem S x)
    ⊢ IsIntegral R x
  -/
  let p' : S[X] := p.toSubring S.toSubring subset_adjoin
  have hSx : IsIntegral S x := ⟨p', (p.monic_toSubring _ _).mpr pmonic, by
    rw [IsScalarTower.algebraMap_eq S A B, ← eval₂_map]
    convert hp; apply p.map_toSubring S.toSubring⟩
  /-
    case intro.intro
    R : Type u_1
    A : Type u_2
    B : Type u_3
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing A
    inst✝⁵ : Ring B
    inst✝⁴ : Algebra A B
    inst✝³ : Algebra R B
    inst✝² : Algebra R A
    inst✝¹ : IsScalarTower R A B
    inst✝ : Algebra.IsIntegral R A
    x : B
    p : Polynomial A
    pmonic : p.Monic
    hp : Eq (Polynomial.eval₂ (algebraMap A B) x p) 0
    S : Subalgebra R A := Algebra.adjoin R ↑p.coeffs
    this : Module.Finite R (Subtype fun x => Membership.mem S x)
    p' : Polynomial (Subtype fun x => Membership.mem S x) := p.toSubring S.toSubri …
    hSx : IsIntegral (Subtype fun x => Membership.mem S x) x
    ⊢ IsIntegral R x
  -/
  let Sx := Subalgebra.toSubmodule (adjoin S {x})
  /-
    case intro.intro
    R : Type u_1
    A : Type u_2
    B : Type u_3
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing A
    inst✝⁵ : Ring B
    inst✝⁴ : Algebra A B
    inst✝³ : Algebra R B
    inst✝² : Algebra R A
    inst✝¹ : IsScalarTower R A B
    inst✝ : Algebra.IsIntegral R A
    x : B
    p : Polynomial A
    pmonic : p.Monic
    hp : Eq (Polynomial.eval₂ (algebraMap A B) x p) 0
    S : Subalgebra R A := Algebra.adjoin R ↑p.coeffs
    this : Module.Finite R (Subtype fun x => Membership.mem S x)
    p' : Polynomial (Subtype fun x => Membership.mem S x) := p.toSubring S.toSubri …
    hSx : IsIntegral (Subtype fun x => Membership.mem S x) x
    Sx : Submodule (Subtype fun x => Membership.mem S x) B := Subalgebra.toSubmodu …
    ⊢ IsIntegral R x
  -/
  let MSx : Module S Sx := SMulMemClass.toModule _ -- the next line times out without this
  /-
    case intro.intro
    R : Type u_1
    A : Type u_2
    B : Type u_3
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing A
    inst✝⁵ : Ring B
    inst✝⁴ : Algebra A B
    inst✝³ : Algebra R B
    inst✝² : Algebra R A
    inst✝¹ : IsScalarTower R A B
    inst✝ : Algebra.IsIntegral R A
    x : B
    p : Polynomial A
    pmonic : p.Monic
    hp : Eq (Polynomial.eval₂ (algebraMap A B) x p) 0
    S : Subalgebra R A := Algebra.adjoin R ↑p.coeffs
    this : Module.Finite R (Subtype fun x => Membership.mem S x)
    p' : Polynomial (Subtype fun x => Membership.mem S x) := p.toSubring S.toSubri …
    hSx : IsIntegral (Subtype fun x => Membership.mem S x) x
    Sx : Submodule (Subtype fun x => Membership.mem S x) B := Subalgebra.toSubmodu …
    MSx : Module (Subtype fun x => Membership.mem S x) (Subtype fun x => Membershi …
    ⊢ IsIntegral R x
  -/
  have : Module.Finite S Sx := ⟨(Submodule.fg_top _).mpr hSx.fg_adjoin_singleton⟩
  refine .of_mem_of_fg ((adjoin S {x}).restrictScalars R) ?_ _
    ((Subalgebra.mem_restrictScalars R).mpr <| subset_adjoin rfl)
  /-
    case intro.intro
    R : Type u_1
    A : Type u_2
    B : Type u_3
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing A
    inst✝⁵ : Ring B
    inst✝⁴ : Algebra A B
    inst✝³ : Algebra R B
    inst✝² : Algebra R A
    inst✝¹ : IsScalarTower R A B
    inst✝ : Algebra.IsIntegral R A
    x : B
    p : Polynomial A
    pmonic : p.Monic
    hp : Eq (Polynomial.eval₂ (algebraMap A B) x p) 0
    S : Subalgebra R A := Algebra.adjoin R ↑p.coeffs
    this✝ : Module.Finite R (Subtype fun x => Membership.mem S x)
    p' : Polynomial (Subtype fun x => Membership.mem S x) := p.toSubring S.toSubri …
    hSx : IsIntegral (Subtype fun x => Membership.mem S x) x
    Sx : Submodule (Subtype fun x => Membership.mem S x) B := Subalgebra.toSubmodu …
    MSx : Module (Subtype fun x => Membership.mem S x) (Subtype fun x => Membershi …
    this : Module.Finite (Subtype fun x => Membership.mem S x) (Subtype fun x => M …
    ⊢ (Subalgebra.toSubmodule (Subalgebra.restrictScalars R (Algebra.adjoin (Subty …
  -/
  rw [← Submodule.fg_top, ← Module.finite_def]
  /-
    case intro.intro
    R : Type u_1
    A : Type u_2
    B : Type u_3
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing A
    inst✝⁵ : Ring B
    inst✝⁴ : Algebra A B
    inst✝³ : Algebra R B
    inst✝² : Algebra R A
    inst✝¹ : IsScalarTower R A B
    inst✝ : Algebra.IsIntegral R A
    x : B
    p : Polynomial A
    pmonic : p.Monic
    hp : Eq (Polynomial.eval₂ (algebraMap A B) x p) 0
    S : Subalgebra R A := Algebra.adjoin R ↑p.coeffs
    this✝ : Module.Finite R (Subtype fun x => Membership.mem S x)
    p' : Polynomial (Subtype fun x => Membership.mem S x) := p.toSubring S.toSubri …
    hSx : IsIntegral (Subtype fun x => Membership.mem S x) x
    Sx : Submodule (Subtype fun x => Membership.mem S x) B := Subalgebra.toSubmodu …
    MSx : Module (Subtype fun x => Membership.mem S x) (Subtype fun x => Membershi …
    this : Module.Finite (Subtype fun x => Membership.mem S x) (Subtype fun x => M …
    ⊢ Module.Finite R (Subtype fun x_1 => Membership.mem (Subalgebra.toSubmodule ( …
  -/
  letI : SMul S Sx := { MSx with } -- need this even though MSx is there
  have : IsScalarTower R S Sx :=
    Submodule.isScalarTower Sx -- Lean looks for `Module A Sx` without this
  /-
    case intro.intro
    R : Type u_1
    A : Type u_2
    B : Type u_3
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing A
    inst✝⁵ : Ring B
    inst✝⁴ : Algebra A B
    inst✝³ : Algebra R B
    inst✝² : Algebra R A
    inst✝¹ : IsScalarTower R A B
    inst✝ : Algebra.IsIntegral R A
    x : B
    p : Polynomial A
    pmonic : p.Monic
    hp : Eq (Polynomial.eval₂ (algebraMap A B) x p) 0
    S : Subalgebra R A := Algebra.adjoin R ↑p.coeffs
    this✝² : Module.Finite R (Subtype fun x => Membership.mem S x)
    p' : Polynomial (Subtype fun x => Membership.mem S x) := p.toSubring S.toSubri …
    hSx : IsIntegral (Subtype fun x => Membership.mem S x) x
    Sx : Submodule (Subtype fun x => Membership.mem S x) B := Subalgebra.toSubmodu …
    MSx : Module (Subtype fun x => Membership.mem S x) (Subtype fun x => Membershi …
    this✝¹ : Module.Finite (Subtype fun x => Membership.mem S x) (Subtype fun x => …
    this✝ : SMul (Subtype fun x => Membership.mem S x) (Subtype fun x => Membershi …
    this : IsScalarTower R (Subtype fun x => Membership.mem S x) (Subtype fun x => …
    ⊢ Module.Finite R (Subtype fun x_1 => Membership.mem (Subalgebra.toSubmodule ( …
  -/
  exact Module.Finite.trans S Sx
  /-
    🎉 no goals
  -/


variable (A) in
/-- If A is an R-algebra all of whose elements are integral over R,
and B is an A-algebra all of whose elements are integral over A,
then all elements of B are integral over R. -/
protected theorem Algebra.IsIntegral.trans
    [Algebra.IsIntegral R A] [Algebra.IsIntegral A B] : Algebra.IsIntegral R B :=
  ⟨fun x ↦ isIntegral_trans x (Algebra.IsIntegral.isIntegral (R := A) x)⟩


protected theorem RingHom.IsIntegral.trans
    (hf : f.IsIntegral) (hg : g.IsIntegral) : (g.comp f).IsIntegral :=
  let _ := f.toAlgebra; let _ := g.toAlgebra; let _ := (g.comp f).toAlgebra
  have : IsScalarTower R S T := IsScalarTower.of_algebraMap_eq fun _ ↦ rfl
  have : Algebra.IsIntegral R S := ⟨hf⟩
  have : Algebra.IsIntegral S T := ⟨hg⟩
  have : Algebra.IsIntegral R T := Algebra.IsIntegral.trans S
  Algebra.IsIntegral.isIntegral


/-- If `R → A → B` is an algebra tower, `C` is the integral closure of `R` in `B`
and `A` is integral over `R`, then `C` is the integral closure of `A` in `B`. -/
lemma IsIntegralClosure.tower_top {B C : Type*} [CommRing C] [CommRing B]
    [Algebra R B] [Algebra A B] [Algebra C B] [IsScalarTower R A B]
    [IsIntegralClosure C R B] [Algebra.IsIntegral R A] :
    IsIntegralClosure C A B :=
  ⟨IsIntegralClosure.algebraMap_injective _ R _,
   fun hx => (IsIntegralClosure.isIntegral_iff).mp (isIntegral_trans (R := R) _ hx),
   fun hx => ((IsIntegralClosure.isIntegral_iff (R := R)).mpr hx).tower_top⟩


theorem RingHom.isIntegral_of_surjective (hf : Function.Surjective f) : f.IsIntegral :=
  fun x ↦ (hf x).recOn fun _y hy ↦ hy ▸ f.isIntegralElem_map


theorem Algebra.isIntegral_of_surjective (h : Function.Surjective (algebraMap R A)) :
    Algebra.IsIntegral R A :=
  ⟨(algebraMap R A).isIntegral_of_surjective h⟩


/-- If `R → A → B` is an algebra tower with `A → B` injective,
then if the entire tower is an integral extension so is `R → A` -/
theorem IsIntegral.tower_bot (H : Function.Injective (algebraMap A B)) {x : A}
    (h : IsIntegral R (algebraMap A B x)) : IsIntegral R x :=
  (isIntegral_algHom_iff (IsScalarTower.toAlgHom R A B) H).mp h


nonrec theorem RingHom.IsIntegral.tower_bot (hg : Function.Injective g)
    (hfg : (g.comp f).IsIntegral) : f.IsIntegral :=
  letI := f.toAlgebra; letI := g.toAlgebra; letI := (g.comp f).toAlgebra
  haveI : IsScalarTower R S T := IsScalarTower.of_algebraMap_eq fun _ ↦ rfl
  fun x ↦ IsIntegral.tower_bot hg (hfg (g x))


variable (T) in
/-- Let `T / S / R` be a tower of algebras, `T` is non-trivial and is a torsion free `S`-module,
  then if `T` is an integral `R`-algebra, then `S` is an integral `R`-algebra. -/
theorem Algebra.IsIntegral.tower_bot [Algebra R S] [Algebra R T] [Algebra S T]
    [NoZeroSMulDivisors S T] [Nontrivial T] [IsScalarTower R S T]
    [h : Algebra.IsIntegral R T] : Algebra.IsIntegral R S where
  isIntegral := by
    apply RingHom.IsIntegral.tower_bot (algebraMap R S) (algebraMap S T)
      (NoZeroSMulDivisors.algebraMap_injective S T)
    /-
      R : Type u_1
      S : Type u_4
      T : Type u_5
      inst✝⁸ : CommRing R
      inst✝⁷ : CommRing S
      inst✝⁶ : CommRing T
      inst✝⁵ : Algebra R S
      inst✝⁴ : Algebra R T
      inst✝³ : Algebra S T
      inst✝² : NoZeroSMulDivisors S T
      inst✝¹ : Nontrivial T
      inst✝ : IsScalarTower R S T
      h : Algebra.IsIntegral R T
      ⊢ ((algebraMap S T).comp (algebraMap R S)).IsIntegral
    -/
    rw [← IsScalarTower.algebraMap_eq R S T]
    /-
      R : Type u_1
      S : Type u_4
      T : Type u_5
      inst✝⁸ : CommRing R
      inst✝⁷ : CommRing S
      inst✝⁶ : CommRing T
      inst✝⁵ : Algebra R S
      inst✝⁴ : Algebra R T
      inst✝³ : Algebra S T
      inst✝² : NoZeroSMulDivisors S T
      inst✝¹ : Nontrivial T
      inst✝ : IsScalarTower R S T
      h : Algebra.IsIntegral R T
      ⊢ (algebraMap R T).IsIntegral
    -/
    exact h.isIntegral
    /-
      🎉 no goals
    -/


theorem IsIntegral.tower_bot_of_field {R A B : Type*} [CommRing R] [Field A]
    [CommRing B] [Nontrivial B] [Algebra R A] [Algebra A B] [Algebra R B] [IsScalarTower R A B]
    {x : A} (h : IsIntegral R (algebraMap A B x)) : IsIntegral R x :=
  h.tower_bot (algebraMap A B).injective


theorem RingHom.isIntegralElem.of_comp {x : T} (h : (g.comp f).IsIntegralElem x) :
    g.IsIntegralElem x :=
  let ⟨p, hp, hp'⟩ := h
                         /-
                           R : Type u_1
                           S : Type u_4
                           T : Type u_5
                           inst✝² : CommRing R
                           inst✝¹ : CommRing S
                           inst✝ : CommRing T
                           f : RingHom R S
                           g : RingHom S T
                           x : T
                           h : (g.comp f).IsIntegralElem x
                           p : Polynomial R
                           hp : p.Monic
                           hp' : Eq (Polynomial.eval₂ (g.comp f) x p) 0
                           ⊢ Eq (Polynomial.eval₂ g x (Polynomial.map f p)) 0
                         -/
  ⟨p.map f, hp.map f, by rwa [← eval₂_map] at hp'⟩
                         /-
                           🎉 no goals
                         -/


theorem RingHom.IsIntegral.tower_top (h : (g.comp f).IsIntegral) : g.IsIntegral :=
  fun x ↦ RingHom.isIntegralElem.of_comp f g (h x)


variable (R) in
/-- Let `T / S / R` be a tower of algebras, `T` is an integral `R`-algebra, then it is integral
  as an `S`-algebra. -/
theorem Algebra.IsIntegral.tower_top [Algebra R S] [Algebra R T] [Algebra S T] [IsScalarTower R S T]
    [h : Algebra.IsIntegral R T] : Algebra.IsIntegral S T where
  isIntegral := by
    /-
      R : Type u_1
      S : Type u_4
      T : Type u_5
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : CommRing T
      inst✝³ : Algebra R S
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      h : Algebra.IsIntegral R T
      ⊢ ∀ (x : T), IsIntegral S x
    -/
    apply RingHom.IsIntegral.tower_top (algebraMap R S) (algebraMap S T)
    /-
      R : Type u_1
      S : Type u_4
      T : Type u_5
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : CommRing T
      inst✝³ : Algebra R S
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      h : Algebra.IsIntegral R T
      ⊢ ((algebraMap S T).comp (algebraMap R S)).IsIntegral
    -/
    rw [← IsScalarTower.algebraMap_eq R S T]
    /-
      R : Type u_1
      S : Type u_4
      T : Type u_5
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : CommRing T
      inst✝³ : Algebra R S
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      h : Algebra.IsIntegral R T
      ⊢ (algebraMap R T).IsIntegral
    -/
    exact h.isIntegral
    /-
      🎉 no goals
    -/


theorem RingHom.IsIntegral.quotient {I : Ideal S} (hf : f.IsIntegral) :
    (Ideal.quotientMap I f le_rfl).IsIntegral := by
  /-
    R : Type u_1
    S : Type u_4
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    I : Ideal S
    hf : f.IsIntegral
    ⊢ (Ideal.quotientMap I f ⋯).IsIntegral
  -/
  rintro ⟨x⟩
  /-
    case mk
    R : Type u_1
    S : Type u_4
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    I : Ideal S
    hf : f.IsIntegral
    x✝ : HasQuotient.Quotient S I
    x : S
    ⊢ (Ideal.quotientMap I f ⋯).IsIntegralElem (Quot.mk (⇑(Submodule.quotientRel I …
  -/
  obtain ⟨p, p_monic, hpx⟩ := hf x
  /-
    case mk.intro.intro
    R : Type u_1
    S : Type u_4
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    I : Ideal S
    hf : f.IsIntegral
    x✝ : HasQuotient.Quotient S I
    x : S
    p : Polynomial R
    p_monic : p.Monic
    hpx : Eq (Polynomial.eval₂ f x p) 0
    ⊢ (Ideal.quotientMap I f ⋯).IsIntegralElem (Quot.mk (⇑(Submodule.quotientRel I …
  -/
  refine ⟨p.map (Ideal.Quotient.mk _), p_monic.map _, ?_⟩
  /-
    case mk.intro.intro
    R : Type u_1
    S : Type u_4
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    I : Ideal S
    hf : f.IsIntegral
    x✝ : HasQuotient.Quotient S I
    x : S
    p : Polynomial R
    p_monic : p.Monic
    hpx : Eq (Polynomial.eval₂ f x p) 0
    ⊢ Eq (Polynomial.eval₂ (Ideal.quotientMap I f ⋯) (Quot.mk (⇑(Submodule.quotien …
  -/
  simpa only [hom_eval₂, eval₂_map] using congr_arg (Ideal.Quotient.mk I) hpx
  /-
    🎉 no goals
  -/


instance {I : Ideal A} [Algebra.IsIntegral R A] : Algebra.IsIntegral R (A ⧸ I) :=
  Algebra.IsIntegral.trans A


instance Algebra.IsIntegral.quotient {I : Ideal A} [Algebra.IsIntegral R A] :
    Algebra.IsIntegral (R ⧸ I.comap (algebraMap R A)) (A ⧸ I) :=
  ⟨RingHom.IsIntegral.quotient (algebraMap R A) Algebra.IsIntegral.isIntegral⟩


theorem isIntegral_quotientMap_iff {I : Ideal S} :
    (Ideal.quotientMap I f le_rfl).IsIntegral ↔
      ((Ideal.Quotient.mk I).comp f : R →+* S ⧸ I).IsIntegral := by
  /-
    R : Type u_1
    S : Type u_4
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    I : Ideal S
    ⊢ Iff (Ideal.quotientMap I f ⋯).IsIntegral ((Ideal.Quotient.mk I).comp f).IsIn …
  -/
  let g := Ideal.Quotient.mk (I.comap f)
  -- Porting note: added type ascription
  have : (Ideal.quotientMap I f le_rfl).comp g = (Ideal.Quotient.mk I).comp f :=
    Ideal.quotientMap_comp_mk le_rfl
  /-
    R : Type u_1
    S : Type u_4
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    I : Ideal S
    g : RingHom R (HasQuotient.Quotient R (Ideal.comap f I)) := Ideal.Quotient.mk  …
    this : Eq ((Ideal.quotientMap I f ⋯).comp g) ((Ideal.Quotient.mk I).comp f)
    ⊢ Iff (Ideal.quotientMap I f ⋯).IsIntegral ((Ideal.Quotient.mk I).comp f).IsIn …
  -/
  refine ⟨fun h => ?_, fun h => RingHom.IsIntegral.tower_top g _ (this ▸ h)⟩
  /-
    R : Type u_1
    S : Type u_4
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    I : Ideal S
    g : RingHom R (HasQuotient.Quotient R (Ideal.comap f I)) := Ideal.Quotient.mk  …
    this : Eq ((Ideal.quotientMap I f ⋯).comp g) ((Ideal.Quotient.mk I).comp f)
    h : (Ideal.quotientMap I f ⋯).IsIntegral
    ⊢ ((Ideal.Quotient.mk I).comp f).IsIntegral
  -/
  refine this ▸ RingHom.IsIntegral.trans g (Ideal.quotientMap I f le_rfl) ?_ h
  /-
    R : Type u_1
    S : Type u_4
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    I : Ideal S
    g : RingHom R (HasQuotient.Quotient R (Ideal.comap f I)) := Ideal.Quotient.mk  …
    this : Eq ((Ideal.quotientMap I f ⋯).comp g) ((Ideal.Quotient.mk I).comp f)
    h : (Ideal.quotientMap I f ⋯).IsIntegral
    ⊢ g.IsIntegral
  -/
  exact g.isIntegral_of_surjective Ideal.Quotient.mk_surjective
  /-
    🎉 no goals
  -/


/-- If the integral extension `R → S` is injective, and `S` is a field, then `R` is also a field. -/
theorem isField_of_isIntegral_of_isField {R S : Type*} [CommRing R] [CommRing S]
    [Algebra R S] [Algebra.IsIntegral R S]
    (hRS : Function.Injective (algebraMap R S)) (hS : IsField S) : IsField R := by
  /-
    R : Type u_6
    S : Type u_7
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : Algebra.IsIntegral R S
    hRS : Function.Injective ⇑(algebraMap R S)
    hS : IsField S
    ⊢ IsField R
  -/
  have := hS.nontrivial; have := Module.nontrivial R S
  /-
    R : Type u_6
    S : Type u_7
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : Algebra.IsIntegral R S
    hRS : Function.Injective ⇑(algebraMap R S)
    hS : IsField S
    this✝ : Nontrivial S
    this : Nontrivial R
    ⊢ IsField R
  -/
  refine ⟨⟨0, 1, zero_ne_one⟩, mul_comm, fun {a} ha ↦ ?_⟩
  -- Let `a_inv` be the inverse of `algebraMap R S a`,
  -- then we need to show that `a_inv` is of the form `algebraMap R S b`.
  /-
    R : Type u_6
    S : Type u_7
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : Algebra.IsIntegral R S
    hRS : Function.Injective ⇑(algebraMap R S)
    hS : IsField S
    this✝ : Nontrivial S
    this : Nontrivial R
    a : R
    ha : Ne a 0
    ⊢ Exists fun b => Eq (HMul.hMul a b) 1
  -/
  obtain ⟨a_inv, ha_inv⟩ := hS.mul_inv_cancel fun h ↦ ha (hRS (h.trans (RingHom.map_zero _).symm))
  /-
    case intro
    R : Type u_6
    S : Type u_7
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : Algebra.IsIntegral R S
    hRS : Function.Injective ⇑(algebraMap R S)
    hS : IsField S
    this✝ : Nontrivial S
    this : Nontrivial R
    a : R
    ha : Ne a 0
    a_inv : S
    ha_inv : Eq (HMul.hMul ((algebraMap R S) a) a_inv) 1
    ⊢ Exists fun b => Eq (HMul.hMul a b) 1
  -/
  letI : Invertible a_inv := (Units.mkOfMulEqOne a_inv _ <| mul_comm _ a_inv ▸ ha_inv).invertible
  -- Let `p : R[X]` be monic with root `a_inv`,
  /-
    case intro
    R : Type u_6
    S : Type u_7
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : Algebra.IsIntegral R S
    hRS : Function.Injective ⇑(algebraMap R S)
    hS : IsField S
    this✝¹ : Nontrivial S
    this✝ : Nontrivial R
    a : R
    ha : Ne a 0
    a_inv : S
    ha_inv : Eq (HMul.hMul ((algebraMap R S) a) a_inv) 1
    this : Invertible a_inv := (Units.mkOfMulEqOne a_inv ((algebraMap R S) a) ⋯).i …
    ⊢ Exists fun b => Eq (HMul.hMul a b) 1
  -/
  obtain ⟨p, p_monic, hp⟩ := Algebra.IsIntegral.isIntegral (R := R) a_inv
  -- and `q` be `p` with coefficients reversed (so `q(a) = q'(a) * a + 1`).
  -- We have `q(a) = 0`, so `-q'(a)` is the inverse of `a`.
  /-
    case intro.intro.intro
    R : Type u_6
    S : Type u_7
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : Algebra.IsIntegral R S
    hRS : Function.Injective ⇑(algebraMap R S)
    hS : IsField S
    this✝¹ : Nontrivial S
    this✝ : Nontrivial R
    a : R
    ha : Ne a 0
    a_inv : S
    ha_inv : Eq (HMul.hMul ((algebraMap R S) a) a_inv) 1
    this : Invertible a_inv := (Units.mkOfMulEqOne a_inv ((algebraMap R S) a) ⋯).i …
    p : Polynomial R
    p_monic : p.Monic
    hp : Eq (Polynomial.eval₂ (algebraMap R S) a_inv p) 0
    ⊢ Exists fun b => Eq (HMul.hMul a b) 1
  -/
  use -p.reverse.divX.eval a -- -q'(a)
  nth_rewrite 1 [mul_neg, ← eval_X (x := a), ← eval_mul, ← p_monic, ← coeff_zero_reverse,
    ← add_eq_zero_iff_neg_eq, ← eval_C (a := p.reverse.coeff 0), ← eval_add, X_mul_divX_add,
    ← (injective_iff_map_eq_zero' _).mp hRS, ← aeval_algebraMap_apply_eq_algebraMap_eval]
  /-
    case h
    R : Type u_6
    S : Type u_7
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : Algebra.IsIntegral R S
    hRS : Function.Injective ⇑(algebraMap R S)
    hS : IsField S
    this✝¹ : Nontrivial S
    this✝ : Nontrivial R
    a : R
    ha : Ne a 0
    a_inv : S
    ha_inv : Eq (HMul.hMul ((algebraMap R S) a) a_inv) 1
    this : Invertible a_inv := (Units.mkOfMulEqOne a_inv ((algebraMap R S) a) ⋯).i …
    p : Polynomial R
    p_monic : p.Monic
    hp : Eq (Polynomial.eval₂ (algebraMap R S) a_inv p) 0
    ⊢ Eq ((Polynomial.aeval ((algebraMap R S) a)) p.reverse) 0
  -/
  rwa [← eval₂_reverse_eq_zero_iff] at hp
  /-
    🎉 no goals
  -/


theorem Algebra.IsIntegral.isField_iff_isField {R S : Type*} [CommRing R]
    [CommRing S] [IsDomain S] [Algebra R S] [Algebra.IsIntegral R S]
    (hRS : Function.Injective (algebraMap R S)) : IsField R ↔ IsField S :=
  ⟨isField_of_isIntegral_of_isField', isField_of_isIntegral_of_isField hRS⟩


theorem integralClosure_idem {R A : Type*} [CommRing R] [CommRing A] [Algebra R A] :
    integralClosure (integralClosure R A) A = ⊥ :=
  letI := (integralClosure R A).algebra
  eq_bot_iff.2 fun x hx ↦ Algebra.mem_bot.2
    ⟨⟨x, isIntegral_trans (A := integralClosure R A) x hx⟩, rfl⟩


instance : IsDomain (integralClosure R S) :=
  inferInstance


theorem roots_mem_integralClosure {f : R[X]} (hf : f.Monic) {a : S}
    (ha : a ∈ f.aroots S) : a ∈ integralClosure R S :=
  ⟨f, hf, (eval₂_eq_eval_map _).trans <| (mem_roots <| (hf.map _).ne_zero).1 ha⟩


