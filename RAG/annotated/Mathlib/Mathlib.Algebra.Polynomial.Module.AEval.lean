/--
Suppose `a` is an element of an `R`-algebra `A` and `M` is an `A`-module.
Loosely speaking, `Module.AEval R M a` is the `R[X]`-module with elements `m : M`,
where the action of a polynomial $f$ is given by $f • m = f(a) • m$.

More precisely, `Module.AEval R M a` has elements `Module.AEval.of R M a m` for `m : M`,
and the action of `f` is `f • (of R M a m) = of R M a ((aeval a f) • m)`.
-/
@[nolint unusedArguments]
def AEval (R M : Type*) {A : Type*} [CommSemiring R] [Semiring A] [Algebra R A]
    [AddCommMonoid M] [Module A M] [Module R M] [IsScalarTower R A M] (_ : A) := M


instance AEval.instAddCommGroup {R A M} [CommSemiring R] [Semiring A] (a : A) [Algebra R A]
    [AddCommGroup M] [Module A M] [Module R M] [IsScalarTower R A M] :
    AddCommGroup <| AEval R M a := inferInstanceAs (AddCommGroup M)


instance instAddCommMonoid : AddCommMonoid <| AEval R M a := inferInstanceAs (AddCommMonoid M)


instance instModuleOrig : Module R <| AEval R M a := inferInstanceAs (Module R M)


instance instFiniteOrig [Module.Finite R M] : Module.Finite R <| AEval R M a :=
  ‹Module.Finite R M›


instance instModulePolynomial : Module R[X] <| AEval R M a := compHom M (aeval a).toRingHom


/--
The canonical linear equivalence between `M` and `Module.AEval R M a` as an `R`-module.
-/
def of : M ≃ₗ[R] AEval R M a :=
  LinearEquiv.refl _ _


lemma of_aeval_smul (f : R[X]) (m : M) : of R M a (aeval a f • m) = f • of R M a m := rfl


@[simp] lemma of_symm_smul (f : R[X]) (m : AEval R M a) :
    (of R M a).symm (f • m) = aeval a f • (of R M a).symm m := rfl


@[simp] lemma C_smul (t : R) (m : AEval R M a) : C t • m = t • m :=
                                  /-
                                    R : Type u_1
                                    A : Type u_3
                                    M : Type u_2
                                    inst✝⁶ : CommSemiring R
                                    inst✝⁵ : Semiring A
                                    a : A
                                    inst✝⁴ : Algebra R A
                                    inst✝³ : AddCommMonoid M
                                    inst✝² : Module A M
                                    inst✝¹ : Module R M
                                    inst✝ : IsScalarTower R A M
                                    t : R
                                    m : Module.AEval R M a
                                    ⊢ Eq ((Module.AEval.of R M a).symm (HSMul.hSMul (Polynomial.C t) m)) ((Module. …
                                  -/
  (of R M a).symm.injective <| by simp
                                  /-
                                    🎉 no goals
                                  -/


lemma X_smul_of (m : M) : (X : R[X]) • (of R M a m) = of R M a (a • m) := by
  /-
    R : Type u_2
    A : Type u_3
    M : Type u_1
    inst✝⁶ : CommSemiring R
    inst✝⁵ : Semiring A
    a : A
    inst✝⁴ : Algebra R A
    inst✝³ : AddCommMonoid M
    inst✝² : Module A M
    inst✝¹ : Module R M
    inst✝ : IsScalarTower R A M
    m : M
    ⊢ Eq (HSMul.hSMul Polynomial.X ((Module.AEval.of R M a) m)) ((Module.AEval.of  …
  -/
  rw [← of_aeval_smul, aeval_X]
  /-
    🎉 no goals
  -/


lemma of_symm_X_smul (m : AEval R M a) :
    (of R M a).symm ((X : R[X]) • m) = a • (of R M a).symm m := by
  /-
    R : Type u_1
    A : Type u_3
    M : Type u_2
    inst✝⁶ : CommSemiring R
    inst✝⁵ : Semiring A
    a : A
    inst✝⁴ : Algebra R A
    inst✝³ : AddCommMonoid M
    inst✝² : Module A M
    inst✝¹ : Module R M
    inst✝ : IsScalarTower R A M
    m : Module.AEval R M a
    ⊢ Eq ((Module.AEval.of R M a).symm (HSMul.hSMul Polynomial.X m)) (HSMul.hSMul  …
  -/
  rw [of_symm_smul, aeval_X]
  /-
    🎉 no goals
  -/


instance instIsScalarTowerOrigPolynomial : IsScalarTower R R[X] <| AEval R M a where
  smul_assoc r f m := by
    /-
      R : Type u_1
      A : Type u_3
      M : Type u_2
      inst✝⁶ : CommSemiring R
      inst✝⁵ : Semiring A
      a : A
      inst✝⁴ : Algebra R A
      inst✝³ : AddCommMonoid M
      inst✝² : Module A M
      inst✝¹ : Module R M
      inst✝ : IsScalarTower R A M
      r : R
      f : Polynomial R
      m : Module.AEval R M a
      ⊢ Eq (HSMul.hSMul (HSMul.hSMul r f) m) (HSMul.hSMul r (HSMul.hSMul f m))
    -/
    apply (of R M a).symm.injective
    /-
      case a
      R : Type u_1
      A : Type u_3
      M : Type u_2
      inst✝⁶ : CommSemiring R
      inst✝⁵ : Semiring A
      a : A
      inst✝⁴ : Algebra R A
      inst✝³ : AddCommMonoid M
      inst✝² : Module A M
      inst✝¹ : Module R M
      inst✝ : IsScalarTower R A M
      r : R
      f : Polynomial R
      m : Module.AEval R M a
      ⊢ Eq ((Module.AEval.of R M a).symm (HSMul.hSMul (HSMul.hSMul r f) m)) ((Module …
    -/
    rw [of_symm_smul, map_smul, smul_assoc, map_smul, of_symm_smul]
    /-
      🎉 no goals
    -/


instance instFinitePolynomial [Module.Finite R M] : Module.Finite R[X] <| AEval R M a :=
  Finite.of_restrictScalars_finite R _ _


/-- Construct an `R[X]`-linear map out of `AEval R M a` from a `R`-linear map out of `M`. -/
def _root_.LinearMap.ofAEval {N} [AddCommMonoid N] [Module R N] [Module R[X] N]
    [IsScalarTower R R[X] N] (f : M →ₗ[R] N) (hf : ∀ m : M, f (a • m) = (X : R[X]) • f m) :
    AEval R M a →ₗ[R[X]] N where
  __ := f ∘ₗ (of R M a).symm
                                              /-
                                                R : Type ?u.31409
                                                A : Type ?u.31412
                                                M : Type ?u.31440
                                                inst✝¹⁰ : CommSemiring R
                                                inst✝⁹ : Semiring A
                                                a : A
                                                inst✝⁸ : Algebra R A
                                                inst✝⁷ : AddCommMonoid M
                                                inst✝⁶ : Module A M
                                                inst✝⁵ : Module R M
                                                inst✝⁴ : IsScalarTower R A M
                                                N : Type ?u.31889
                                                inst✝³ : AddCommMonoid N
                                                inst✝² : Module R N
                                                inst✝¹ : Module (Polynomial R) N
                                                inst✝ : IsScalarTower R (Polynomial R) N
                                                f : LinearMap (RingHom.id R) M N
                                                hf : ∀ (m : M), Eq (f (HSMul.hSMul a m)) (HSMul.hSMul Polynomial.X (f m))
                                                p : Polynomial R
                                                k : R
                                                m : Module.AEval R M a
                                                ⊢ Eq (__spread✝⁻⁰.toFun (HSMul.hSMul (Polynomial.C k) m)) (HSMul.hSMul ((RingH …
                                              -/
  map_smul' p := p.induction_on (fun k m ↦ by simp [C_eq_algebraMap])
                                              /-
                                                🎉 no goals
                                              -/
                          /-
                            R : Type ?u.31409
                            A : Type ?u.31412
                            M : Type ?u.31440
                            inst✝¹⁰ : CommSemiring R
                            inst✝⁹ : Semiring A
                            a : A
                            inst✝⁸ : Algebra R A
                            inst✝⁷ : AddCommMonoid M
                            inst✝⁶ : Module A M
                            inst✝⁵ : Module R M
                            inst✝⁴ : IsScalarTower R A M
                            N : Type ?u.31889
                            inst✝³ : AddCommMonoid N
                            inst✝² : Module R N
                            inst✝¹ : Module (Polynomial R) N
                            inst✝ : IsScalarTower R (Polynomial R) N
                            f : LinearMap (RingHom.id R) M N
                            hf : ∀ (m : M), Eq (f (HSMul.hSMul a m)) (HSMul.hSMul Polynomial.X (f m))
                            p✝ p q : Polynomial R
                            hp : ∀ (x : Module.AEval R M a), Eq (__spread✝⁻⁰.toFun (HSMul.hSMul p x)) (HSM …
                            hq : ∀ (x : Module.AEval R M a), Eq (__spread✝⁻⁰.toFun (HSMul.hSMul q x)) (HSM …
                            m : Module.AEval R M a
                            ⊢ Eq (__spread✝⁻⁰.toFun (HSMul.hSMul (HAdd.hAdd p q) m)) (HSMul.hSMul ((RingHo …
                          -/
    (fun p q hp hq m ↦ by simp_all [add_smul]) fun n k h m ↦ by
                          /-
                            🎉 no goals
                          -/
      simp_rw [RingHom.id_apply, AddHom.toFun_eq_coe, LinearMap.coe_toAddHom,
        LinearMap.comp_apply, LinearEquiv.coe_toLinearMap] at h ⊢
      /-
        R : Type ?u.31409
        A : Type ?u.31412
        M : Type ?u.31440
        inst✝¹⁰ : CommSemiring R
        inst✝⁹ : Semiring A
        a : A
        inst✝⁸ : Algebra R A
        inst✝⁷ : AddCommMonoid M
        inst✝⁶ : Module A M
        inst✝⁵ : Module R M
        inst✝⁴ : IsScalarTower R A M
        N : Type ?u.31889
        inst✝³ : AddCommMonoid N
        inst✝² : Module R N
        inst✝¹ : Module (Polynomial R) N
        inst✝ : IsScalarTower R (Polynomial R) N
        f : LinearMap (RingHom.id R) M N
        hf : ∀ (m : M), Eq (f (HSMul.hSMul a m)) (HSMul.hSMul Polynomial.X (f m))
        p : Polynomial R
        n : Nat
        k : R
        h : ∀ (x : Module.AEval R M a), Eq (f ((Module.AEval.of R M a).symm (HSMul.hSM …
        m : Module.AEval R M a
        ⊢ Eq (f ((Module.AEval.of R M a).symm (HSMul.hSMul (HMul.hMul (Polynomial.C k) …
      -/
      simp_rw [pow_succ, ← mul_assoc, mul_smul _ X, ← hf, ← of_symm_X_smul, ← h]
      /-
        🎉 no goals
      -/


/-- Construct an `R[X]`-linear equivalence out of `AEval R M a` from a `R`-linear map out of `M`. -/
def _root_.LinearEquiv.ofAEval {N} [AddCommMonoid N] [Module R N] [Module R[X] N]
    [IsScalarTower R R[X] N] (f : M ≃ₗ[R] N) (hf : ∀ m : M, f (a • m) = (X : R[X]) • f m) :
    AEval R M a ≃ₗ[R[X]] N where
  __ := LinearMap.ofAEval a f hf
  invFun := (of R M a) ∘ f.symm
                   /-
                     R : Type ?u.74517
                     A : Type ?u.74520
                     M : Type ?u.74548
                     inst✝¹⁰ : CommSemiring R
                     inst✝⁹ : Semiring A
                     a : A
                     inst✝⁸ : Algebra R A
                     inst✝⁷ : AddCommMonoid M
                     inst✝⁶ : Module A M
                     inst✝⁵ : Module R M
                     inst✝⁴ : IsScalarTower R A M
                     N : Type ?u.74997
                     inst✝³ : AddCommMonoid N
                     inst✝² : Module R N
                     inst✝¹ : Module (Polynomial R) N
                     inst✝ : IsScalarTower R (Polynomial R) N
                     f : LinearEquiv (RingHom.id R) M N
                     hf : ∀ (m : M), Eq (f (HSMul.hSMul a m)) (HSMul.hSMul Polynomial.X (f m))
                     x : Module.AEval R M a
                     ⊢ Eq (Function.comp (⇑(Module.AEval.of R M a)) (⇑f.symm) (__spread✝⁻⁰.toFun x) …
                   -/
  left_inv x := by simp [LinearMap.ofAEval]
                   /-
                     🎉 no goals
                   -/
                    /-
                      R : Type ?u.74517
                      A : Type ?u.74520
                      M : Type ?u.74548
                      inst✝¹⁰ : CommSemiring R
                      inst✝⁹ : Semiring A
                      a : A
                      inst✝⁸ : Algebra R A
                      inst✝⁷ : AddCommMonoid M
                      inst✝⁶ : Module A M
                      inst✝⁵ : Module R M
                      inst✝⁴ : IsScalarTower R A M
                      N : Type ?u.74997
                      inst✝³ : AddCommMonoid N
                      inst✝² : Module R N
                      inst✝¹ : Module (Polynomial R) N
                      inst✝ : IsScalarTower R (Polynomial R) N
                      f : LinearEquiv (RingHom.id R) M N
                      hf : ∀ (m : M), Eq (f (HSMul.hSMul a m)) (HSMul.hSMul Polynomial.X (f m))
                      x : N
                      ⊢ Eq (__spread✝⁻⁰.toFun (Function.comp (⇑(Module.AEval.of R M a)) (⇑f.symm) x) …
                    -/
  right_inv x := by simp [LinearMap.ofAEval]
                    /-
                      🎉 no goals
                    -/


lemma annihilator_eq_ker_aeval [FaithfulSMul A M] :
    annihilator R[X] (AEval R M a) = RingHom.ker (aeval a) := by
  /-
    R : Type u_3
    A : Type u_1
    M : Type u_2
    inst✝⁷ : CommSemiring R
    inst✝⁶ : Semiring A
    a : A
    inst✝⁵ : Algebra R A
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module A M
    inst✝² : Module R M
    inst✝¹ : IsScalarTower R A M
    inst✝ : FaithfulSMul A M
    ⊢ Eq (Module.annihilator (Polynomial R) (Module.AEval R M a)) (RingHom.ker (Po …
  -/
  ext p
  /-
    case h
    R : Type u_3
    A : Type u_1
    M : Type u_2
    inst✝⁷ : CommSemiring R
    inst✝⁶ : Semiring A
    a : A
    inst✝⁵ : Algebra R A
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module A M
    inst✝² : Module R M
    inst✝¹ : IsScalarTower R A M
    inst✝ : FaithfulSMul A M
    p : Polynomial R
    ⊢ Iff (Membership.mem (Module.annihilator (Polynomial R) (Module.AEval R M a)) …
  -/
  simp_rw [mem_annihilator, RingHom.mem_ker]
  /-
    case h
    R : Type u_3
    A : Type u_1
    M : Type u_2
    inst✝⁷ : CommSemiring R
    inst✝⁶ : Semiring A
    a : A
    inst✝⁵ : Algebra R A
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module A M
    inst✝² : Module R M
    inst✝¹ : IsScalarTower R A M
    inst✝ : FaithfulSMul A M
    p : Polynomial R
    ⊢ Iff (∀ (m : Module.AEval R M a), Eq (HSMul.hSMul p m) 0) (Eq ((Polynomial.ae …
  -/
  change (∀ m : M, aeval a p • m = 0) ↔ _
  /-
    case h
    R : Type u_3
    A : Type u_1
    M : Type u_2
    inst✝⁷ : CommSemiring R
    inst✝⁶ : Semiring A
    a : A
    inst✝⁵ : Algebra R A
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module A M
    inst✝² : Module R M
    inst✝¹ : IsScalarTower R A M
    inst✝ : FaithfulSMul A M
    p : Polynomial R
    ⊢ Iff (∀ (m : M), Eq (HSMul.hSMul ((Polynomial.aeval a) p) m) 0) (Eq ((Polynom …
  -/
  exact ⟨fun h ↦ eq_of_smul_eq_smul (α := M) <| by simp [h], fun h ↦ by simp [h]⟩
  /-
    🎉 no goals
  -/


@[simp]
lemma annihilator_top_eq_ker_aeval [FaithfulSMul A M] :
    (⊤ : Submodule R[X] <| AEval R M a).annihilator = RingHom.ker (aeval a) := by
  /-
    R : Type u_3
    A : Type u_1
    M : Type u_2
    inst✝⁷ : CommSemiring R
    inst✝⁶ : Semiring A
    a : A
    inst✝⁵ : Algebra R A
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module A M
    inst✝² : Module R M
    inst✝¹ : IsScalarTower R A M
    inst✝ : FaithfulSMul A M
    ⊢ Eq Top.top.annihilator (RingHom.ker (Polynomial.aeval a))
  -/
  ext p
  /-
    case h
    R : Type u_3
    A : Type u_1
    M : Type u_2
    inst✝⁷ : CommSemiring R
    inst✝⁶ : Semiring A
    a : A
    inst✝⁵ : Algebra R A
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module A M
    inst✝² : Module R M
    inst✝¹ : IsScalarTower R A M
    inst✝ : FaithfulSMul A M
    p : Polynomial R
    ⊢ Iff (Membership.mem Top.top.annihilator p) (Membership.mem (RingHom.ker (Pol …
  -/
  simp only [Submodule.mem_annihilator, Submodule.mem_top, forall_true_left, RingHom.mem_ker]
  /-
    case h
    R : Type u_3
    A : Type u_1
    M : Type u_2
    inst✝⁷ : CommSemiring R
    inst✝⁶ : Semiring A
    a : A
    inst✝⁵ : Algebra R A
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module A M
    inst✝² : Module R M
    inst✝¹ : IsScalarTower R A M
    inst✝ : FaithfulSMul A M
    p : Polynomial R
    ⊢ Iff (∀ (n : Module.AEval R M a), Eq (HSMul.hSMul p n) 0) (Eq ((Polynomial.ae …
  -/
  change (∀ m : M, aeval a p • m = 0) ↔ _
  /-
    case h
    R : Type u_3
    A : Type u_1
    M : Type u_2
    inst✝⁷ : CommSemiring R
    inst✝⁶ : Semiring A
    a : A
    inst✝⁵ : Algebra R A
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module A M
    inst✝² : Module R M
    inst✝¹ : IsScalarTower R A M
    inst✝ : FaithfulSMul A M
    p : Polynomial R
    ⊢ Iff (∀ (m : M), Eq (HSMul.hSMul ((Polynomial.aeval a) p) m) 0) (Eq ((Polynom …
  -/
  exact ⟨fun h ↦ eq_of_smul_eq_smul (α := M) <| by simp [h], fun h ↦ by simp [h]⟩
  /-
    🎉 no goals
  -/


/-- The natural order isomorphism between the two ways to represent invariant submodules. -/
def mapSubmodule :
    (Algebra.lsmul R R M a).invtSubmodule ≃o Submodule R[X] (AEval R M a) where
  toFun p :=
    { toAddSubmonoid := (p : Submodule R M).toAddSubmonoid.map (of R M a)
      smul_mem' := by
        /-
          R : Type ?u.91560
          A : Type ?u.91563
          M : Type ?u.91591
          inst✝⁶ : CommSemiring R
          inst✝⁵ : Semiring A
          a : A
          inst✝⁴ : Algebra R A
          inst✝³ : AddCommMonoid M
          inst✝² : Module A M
          inst✝¹ : Module R M
          inst✝ : IsScalarTower R A M
          p : Subtype fun x => Membership.mem ((Algebra.lsmul R R M) a).invtSubmodule x
          ⊢ ∀ (c : Polynomial R) {x : Module.AEval R M a}, Membership.mem (AddSubmonoid. …
        -/
        rintro f - ⟨m : M, h : m ∈ (p : Submodule R M), rfl⟩
        simp only [AddSubsemigroup.mem_carrier, AddSubmonoid.mem_toSubsemigroup,
          AddSubmonoid.mem_map, Submodule.mem_toAddSubmonoid]
        /-
          case intro.intro
          R : Type ?u.91560
          A : Type ?u.91563
          M : Type ?u.91591
          inst✝⁶ : CommSemiring R
          inst✝⁵ : Semiring A
          a : A
          inst✝⁴ : Algebra R A
          inst✝³ : AddCommMonoid M
          inst✝² : Module A M
          inst✝¹ : Module R M
          inst✝ : IsScalarTower R A M
          p : Subtype fun x => Membership.mem ((Algebra.lsmul R R M) a).invtSubmodule x
          f : Polynomial R
          m : M
          h : Membership.mem (↑p) m
          ⊢ Exists fun x => And (Membership.mem (↑p) x) (Eq ((Module.AEval.of R M a) x)  …
        -/
        exact ⟨aeval a f • m, aeval_apply_smul_mem_of_le_comap' h f a p.2, of_aeval_smul a f m⟩ }
        /-
          🎉 no goals
        -/
  invFun q := ⟨(Submodule.orderIsoMapComap (of R M a)).symm (q.restrictScalars R), fun m hm ↦ by
    /-
      R : Type ?u.91560
      A : Type ?u.91563
      M : Type ?u.91591
      inst✝⁶ : CommSemiring R
      inst✝⁵ : Semiring A
      a : A
      inst✝⁴ : Algebra R A
      inst✝³ : AddCommMonoid M
      inst✝² : Module A M
      inst✝¹ : Module R M
      inst✝ : IsScalarTower R A M
      q : Submodule (Polynomial R) (Module.AEval R M a)
      m : M
      hm : Membership.mem ((Submodule.orderIsoMapComap (Module.AEval.of R M a)).symm …
      ⊢ Membership.mem (Submodule.comap ((Algebra.lsmul R R M) a) ((Submodule.orderI …
    -/
    simpa [← X_smul_of] using q.smul_mem (X : R[X]) hm⟩
    /-
      🎉 no goals
    -/
                   /-
                     R : Type ?u.91560
                     A : Type ?u.91563
                     M : Type ?u.91591
                     inst✝⁶ : CommSemiring R
                     inst✝⁵ : Semiring A
                     a : A
                     inst✝⁴ : Algebra R A
                     inst✝³ : AddCommMonoid M
                     inst✝² : Module A M
                     inst✝¹ : Module R M
                     inst✝ : IsScalarTower R A M
                     p : Subtype fun x => Membership.mem ((Algebra.lsmul R R M) a).invtSubmodule x
                     ⊢ Eq ((fun q => ⟨(Submodule.orderIsoMapComap (Module.AEval.of R M a)).symm (Su …
                   -/
  left_inv p := by ext; simp
                        /-
                          🎉 no goals
                        -/
                    /-
                      R : Type ?u.91560
                      A : Type ?u.91563
                      M : Type ?u.91591
                      inst✝⁶ : CommSemiring R
                      inst✝⁵ : Semiring A
                      a : A
                      inst✝⁴ : Algebra R A
                      inst✝³ : AddCommMonoid M
                      inst✝² : Module A M
                      inst✝¹ : Module R M
                      inst✝ : IsScalarTower R A M
                      q : Submodule (Polynomial R) (Module.AEval R M a)
                      ⊢ Eq ((fun p => { toAddSubmonoid := AddSubmonoid.map (Module.AEval.of R M a) ( …
                    -/
  right_inv q := by ext; aesop
                         /-
                           🎉 no goals
                         -/
                                          /-
                                            R : Type ?u.91560
                                            A : Type ?u.91563
                                            M : Type ?u.91591
                                            inst✝⁶ : CommSemiring R
                                            inst✝⁵ : Semiring A
                                            a : A
                                            inst✝⁴ : Algebra R A
                                            inst✝³ : AddCommMonoid M
                                            inst✝² : Module A M
                                            inst✝¹ : Module R M
                                            inst✝ : IsScalarTower R A M
                                            p p' : Subtype fun x => Membership.mem ((Algebra.lsmul R R M) a).invtSubmodule x
                                            h : LE.le ({ toFun := fun p => { toAddSubmonoid := AddSubmonoid.map (Module.AE …
                                            x : M
                                            hx : Membership.mem (↑p) x
                                            ⊢ Membership.mem (↑p') x
                                          -/
                                          /-
                                            🎉 no goals
                                          -/
  map_rel_iff' {p p'} := ⟨fun h x hx ↦ by aesop, fun h x hx ↦ by aesop⟩
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[simp] lemma mem_mapSubmodule_apply {p : (Algebra.lsmul R R M a).invtSubmodule} {m : AEval R M a} :
    m ∈ mapSubmodule R M a p ↔ (of R M a).symm m ∈ (p : Submodule R M) :=
  ⟨fun ⟨_, hm, hm'⟩ ↦ hm'.symm ▸ hm, fun hm ↦ ⟨(of R M a).symm m, hm, rfl⟩⟩


@[simp] lemma mem_mapSubmodule_symm_apply {q : Submodule R[X] (AEval R M a)} {m : M} :
    m ∈ ((mapSubmodule R M a).symm q : Submodule R M) ↔ of R M a m ∈ q :=
  Iff.rfl


/-- The natural `R`-linear equivalence between the two ways to represent an invariant submodule. -/
def equiv_mapSubmodule :
    p ≃ₗ[R] mapSubmodule R M a ⟨p, hp⟩ where
                             /-
                               R : Type ?u.157357
                               A : Type ?u.157360
                               M : Type ?u.157388
                               inst✝⁶ : CommSemiring R
                               inst✝⁵ : Semiring A
                               a : A
                               inst✝⁴ : Algebra R A
                               inst✝³ : AddCommMonoid M
                               inst✝² : Module A M
                               inst✝¹ : Module R M
                               inst✝ : IsScalarTower R A M
                               p : Submodule R M
                               hp : Membership.mem ((Algebra.lsmul R R M) a).invtSubmodule p
                               x : Subtype fun x => Membership.mem p x
                               ⊢ Membership.mem ((Module.AEval.mapSubmodule R M a) ⟨p, hp⟩) ((Module.AEval.of …
                             -/
  toFun x := ⟨of R M a x, by simp⟩
                             /-
                               🎉 no goals
                             -/
                                                       /-
                                                         R : Type ?u.157357
                                                         A : Type ?u.157360
                                                         M : Type ?u.157388
                                                         inst✝⁶ : CommSemiring R
                                                         inst✝⁵ : Semiring A
                                                         a : A
                                                         inst✝⁴ : Algebra R A
                                                         inst✝³ : AddCommMonoid M
                                                         inst✝² : Module A M
                                                         inst✝¹ : Module R M
                                                         inst✝ : IsScalarTower R A M
                                                         p : Submodule R M
                                                         hp : Membership.mem ((Algebra.lsmul R R M) a).invtSubmodule p
                                                         x : Subtype fun x => Membership.mem ((Module.AEval.mapSubmodule R M a) ⟨p, hp⟩ …
                                                         ⊢ Membership.mem p ((Module.AEval.of R M a).symm ↑x)
                                                       -/
  invFun x := ⟨((of R M _).symm (x : AEval R M a)), by obtain ⟨x, hx⟩ := x; simpa using hx⟩
                                                                            /-
                                                                              🎉 no goals
                                                                            -/
  left_inv x := rfl
  right_inv x := rfl
  map_add' x y := rfl
  map_smul' t x := rfl


/-- The natural `R[X]`-linear equivalence between the two ways to represent an invariant submodule.
-/
noncomputable def restrict_equiv_mapSubmodule :
    (AEval R p <| (Algebra.lsmul R R M a).restrict hp) ≃ₗ[R[X]] mapSubmodule R M a ⟨p, hp⟩ :=
  LinearEquiv.ofAEval ((Algebra.lsmul R R M a).restrict hp) (equiv_mapSubmodule a p hp)
                /-
                  R : Type ?u.163684
                  A : Type ?u.163687
                  M : Type ?u.163715
                  inst✝⁶ : CommSemiring R
                  inst✝⁵ : Semiring A
                  a : A
                  inst✝⁴ : Algebra R A
                  inst✝³ : AddCommMonoid M
                  inst✝² : Module A M
                  inst✝¹ : Module R M
                  inst✝ : IsScalarTower R A M
                  p : Submodule R M
                  hp : Membership.mem ((Algebra.lsmul R R M) a).invtSubmodule p
                  x : Subtype fun x => Membership.mem p x
                  ⊢ Eq ((Module.AEval.equiv_mapSubmodule a p hp) (HSMul.hSMul (LinearMap.restric …
                -/
    (fun x ↦ by simp [equiv_mapSubmodule, X_smul_of])
                /-
                  🎉 no goals
                -/


/--
Given and `R`-module `M` and a linear map `φ : M →ₗ[R] M`, `Module.AEval' φ` is loosely speaking
the `R[X]`-module with elements `m : M`, where the action of a polynomial $f$ is given by
$f • m = f(a) • m$.

More precisely, `Module.AEval' φ` has elements `Module.AEval'.of φ m` for `m : M`,
and the action of `f` is `f • (of φ m) = of φ ((aeval φ f) • m)`.

`Module.AEval'` is defined as a special case of `Module.AEval` in which the `R`-algebra is
`M →ₗ[R] M`. Lemmas involving `Module.AEval` may be applied to `Module.AEval'`.
-/
abbrev AEval' := AEval R M φ

/--
The canonical linear equivalence between `M` and `Module.AEval' φ` as an `R`-module,
where `φ : M →ₗ[R] M`.
-/
abbrev AEval'.of : M ≃ₗ[R] AEval' φ := AEval.of R M φ

lemma AEval'_def : AEval' φ = AEval R M φ := rfl

lemma AEval'.X_smul_of (m : M) : (X : R[X]) • AEval'.of φ m = AEval'.of φ (φ m) :=
  AEval.X_smul_of _ _

lemma AEval'.of_symm_X_smul (m : AEval' φ) :
    (AEval'.of φ).symm ((X : R[X]) • m) = φ ((AEval'.of φ).symm m) := AEval.of_symm_X_smul _ _


instance [Module.Finite R M] : Module.Finite R[X] <| AEval' φ := inferInstance


