/-- `D : Derivation R A M` is an `R`-linear map from `A` to `M` that satisfies the `leibniz`
equality. We also require that `D 1 = 0`. See `Derivation.mk'` for a constructor that deduces this
assumption from the Leibniz rule when `M` is cancellative.

TODO: update this when bimodules are defined. -/
structure Derivation (R : Type*) (A : Type*) (M : Type*)
    [CommSemiring R] [CommSemiring A] [AddCommMonoid M] [Algebra R A] [Module A M] [Module R M]
    extends A →ₗ[R] M where
  protected map_one_eq_zero' : toLinearMap 1 = 0
  protected leibniz' (a b : A) : toLinearMap (a * b) = a • toLinearMap b + b • toLinearMap a


instance : FunLike (Derivation R A M) A M where
  coe D := D.toFun
                               /-
                                 R : Type u_1
                                 A : Type u_2
                                 B : Type u_3
                                 M : Type u_4
                                 inst✝⁸ : CommSemiring R
                                 inst✝⁷ : CommSemiring A
                                 inst✝⁶ : CommSemiring B
                                 inst✝⁵ : AddCommMonoid M
                                 inst✝⁴ : Algebra R A
                                 inst✝³ : Algebra R B
                                 inst✝² : Module A M
                                 inst✝¹ : Module B M
                                 inst✝ : Module R M
                                 D D1✝ D2✝ : Derivation R A M
                                 r : R
                                 a b : A
                                 D1 D2 : Derivation R A M
                                 h : Eq ((fun D => D.toFun) D1) ((fun D => D.toFun) D2)
                                 ⊢ Eq D1 D2
                               -/
  coe_injective' D1 D2 h := by cases D1; cases D2; congr; exact DFunLike.coe_injective h
                                                          /-
                                                            🎉 no goals
                                                          -/


instance : AddMonoidHomClass (Derivation R A M) A M where
  map_add D := D.toLinearMap.map_add'
  map_zero D := D.toLinearMap.map_zero

-- Not a simp lemma because it can be proved via `coeFn_coe` + `toLinearMap_eq_coe`

theorem toFun_eq_coe : D.toFun = ⇑D :=
  rfl


/-- See Note [custom simps projection] -/
def Simps.apply (D : Derivation R A M) : A → M := D


instance hasCoeToLinearMap : Coe (Derivation R A M) (A →ₗ[R] M) :=
  ⟨fun D => D.toLinearMap⟩


@[simp]
theorem mk_coe (f : A →ₗ[R] M) (h₁ h₂) : ((⟨f, h₁, h₂⟩ : Derivation R A M) : A → M) = f :=
  rfl


@[simp, norm_cast]
theorem coeFn_coe (f : Derivation R A M) : ⇑(f : A →ₗ[R] M) = f :=
  rfl


theorem coe_injective : @Function.Injective (Derivation R A M) (A → M) DFunLike.coe :=
  DFunLike.coe_injective


@[ext]
theorem ext (H : ∀ a, D1 a = D2 a) : D1 = D2 :=
  DFunLike.ext _ _ H


theorem congr_fun (h : D1 = D2) (a : A) : D1 a = D2 a :=
  DFunLike.congr_fun h a


protected theorem map_add : D (a + b) = D a + D b :=
  map_add D a b


protected theorem map_zero : D 0 = 0 :=
  map_zero D


@[simp]
theorem map_smul : D (r • a) = r • D a :=
  D.toLinearMap.map_smul r a


@[simp]
theorem leibniz : D (a * b) = a • D b + b • D a :=
  D.leibniz' _ _


@[simp]
theorem map_smul_of_tower {S : Type*} [SMul S A] [SMul S M] [LinearMap.CompatibleSMul A M S R]
    (D : Derivation R A M) (r : S) (a : A) : D (r • a) = r • D a :=
  D.toLinearMap.map_smul_of_tower r a


@[simp]
theorem map_one_eq_zero : D 1 = 0 :=
  D.map_one_eq_zero'


@[simp]
theorem map_algebraMap : D (algebraMap R A r) = 0 := by
  rw [← mul_one r, RingHom.map_mul, RingHom.map_one, ← smul_def, map_smul, map_one_eq_zero,
    smul_zero]


@[simp]
theorem map_natCast (n : ℕ) : D (n : A) = 0 := by
  /-
    R : Type u_1
    A : Type u_2
    M : Type u_4
    inst✝⁵ : CommSemiring R
    inst✝⁴ : CommSemiring A
    inst✝³ : AddCommMonoid M
    inst✝² : Algebra R A
    inst✝¹ : Module A M
    inst✝ : Module R M
    D : Derivation R A M
    n : Nat
    ⊢ Eq (D ↑n) 0
  -/
  rw [← nsmul_one, D.map_smul_of_tower n, map_one_eq_zero, smul_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem leibniz_pow (n : ℕ) : D (a ^ n) = n • a ^ (n - 1) • D a := by
  /-
    R : Type u_1
    A : Type u_2
    M : Type u_4
    inst✝⁵ : CommSemiring R
    inst✝⁴ : CommSemiring A
    inst✝³ : AddCommMonoid M
    inst✝² : Algebra R A
    inst✝¹ : Module A M
    inst✝ : Module R M
    D : Derivation R A M
    a : A
    n : Nat
    ⊢ Eq (D (HPow.hPow a n)) (HSMul.hSMul n (HSMul.hSMul (HPow.hPow a (HSub.hSub n …
  -/
  induction' n with n ihn
    /-
      case zero
      R : Type u_1
      A : Type u_2
      M : Type u_4
      inst✝⁵ : CommSemiring R
      inst✝⁴ : CommSemiring A
      inst✝³ : AddCommMonoid M
      inst✝² : Algebra R A
      inst✝¹ : Module A M
      inst✝ : Module R M
      D : Derivation R A M
      a : A
      ⊢ Eq (D (HPow.hPow a 0)) (HSMul.hSMul 0 (HSMul.hSMul (HPow.hPow a (HSub.hSub 0 …
    -/
  · rw [pow_zero, map_one_eq_zero, zero_smul]
    /-
      🎉 no goals
    -/
    /-
      case succ
      R : Type u_1
      A : Type u_2
      M : Type u_4
      inst✝⁵ : CommSemiring R
      inst✝⁴ : CommSemiring A
      inst✝³ : AddCommMonoid M
      inst✝² : Algebra R A
      inst✝¹ : Module A M
      inst✝ : Module R M
      D : Derivation R A M
      a : A
      n : Nat
      ihn : Eq (D (HPow.hPow a n)) (HSMul.hSMul n (HSMul.hSMul (HPow.hPow a (HSub.hS …
      ⊢ Eq (D (HPow.hPow a (HAdd.hAdd n 1))) (HSMul.hSMul (HAdd.hAdd n 1) (HSMul.hSM …
    -/
  · rcases (zero_le n).eq_or_lt with (rfl | hpos)
      /-
        case succ.inl
        R : Type u_1
        A : Type u_2
        M : Type u_4
        inst✝⁵ : CommSemiring R
        inst✝⁴ : CommSemiring A
        inst✝³ : AddCommMonoid M
        inst✝² : Algebra R A
        inst✝¹ : Module A M
        inst✝ : Module R M
        D : Derivation R A M
        a : A
        ihn : Eq (D (HPow.hPow a 0)) (HSMul.hSMul 0 (HSMul.hSMul (HPow.hPow a (HSub.hS …
        ⊢ Eq (D (HPow.hPow a (HAdd.hAdd 0 1))) (HSMul.hSMul (HAdd.hAdd 0 1) (HSMul.hSM …
      -/
    · erw [pow_one, one_smul, pow_zero, one_smul]
      /-
        🎉 no goals
      -/
      /-
        case succ.inr
        R : Type u_1
        A : Type u_2
        M : Type u_4
        inst✝⁵ : CommSemiring R
        inst✝⁴ : CommSemiring A
        inst✝³ : AddCommMonoid M
        inst✝² : Algebra R A
        inst✝¹ : Module A M
        inst✝ : Module R M
        D : Derivation R A M
        a : A
        n : Nat
        ihn : Eq (D (HPow.hPow a n)) (HSMul.hSMul n (HSMul.hSMul (HPow.hPow a (HSub.hS …
        hpos : LT.lt 0 n
        ⊢ Eq (D (HPow.hPow a (HAdd.hAdd n 1))) (HSMul.hSMul (HAdd.hAdd n 1) (HSMul.hSM …
      -/
    · have : a * a ^ (n - 1) = a ^ n := by rw [← pow_succ', Nat.sub_add_cancel hpos]
      simp only [pow_succ', leibniz, ihn, smul_comm a n (_ : M), smul_smul a, add_smul, this,
        Nat.succ_eq_add_one, Nat.add_succ_sub_one, add_zero, one_nsmul]


open Polynomial in
@[simp]
theorem map_aeval (P : R[X]) (x : A) :
    D (aeval x P) = aeval x (derivative P) • D x := by
  /-
    R : Type u_1
    A : Type u_2
    M : Type u_4
    inst✝⁵ : CommSemiring R
    inst✝⁴ : CommSemiring A
    inst✝³ : AddCommMonoid M
    inst✝² : Algebra R A
    inst✝¹ : Module A M
    inst✝ : Module R M
    D : Derivation R A M
    P : Polynomial R
    x : A
    ⊢ Eq (D ((Polynomial.aeval x) P)) (HSMul.hSMul ((Polynomial.aeval x) (Polynomi …
  -/
  induction P using Polynomial.induction_on
    /-
      case h_C
      R : Type u_1
      A : Type u_2
      M : Type u_4
      inst✝⁵ : CommSemiring R
      inst✝⁴ : CommSemiring A
      inst✝³ : AddCommMonoid M
      inst✝² : Algebra R A
      inst✝¹ : Module A M
      inst✝ : Module R M
      D : Derivation R A M
      x : A
      a✝ : R
      ⊢ Eq (D ((Polynomial.aeval x) (Polynomial.C a✝))) (HSMul.hSMul ((Polynomial.ae …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case h_add
      R : Type u_1
      A : Type u_2
      M : Type u_4
      inst✝⁵ : CommSemiring R
      inst✝⁴ : CommSemiring A
      inst✝³ : AddCommMonoid M
      inst✝² : Algebra R A
      inst✝¹ : Module A M
      inst✝ : Module R M
      D : Derivation R A M
      x : A
      p✝ q✝ : Polynomial R
      a✝¹ : Eq (D ((Polynomial.aeval x) p✝)) (HSMul.hSMul ((Polynomial.aeval x) (Pol …
      a✝ : Eq (D ((Polynomial.aeval x) q✝)) (HSMul.hSMul ((Polynomial.aeval x) (Poly …
      ⊢ Eq (D ((Polynomial.aeval x) (HAdd.hAdd p✝ q✝))) (HSMul.hSMul ((Polynomial.ae …
    -/
  · simp [add_smul, *]
    /-
      🎉 no goals
    -/
    /-
      case h_monomial
      R : Type u_1
      A : Type u_2
      M : Type u_4
      inst✝⁵ : CommSemiring R
      inst✝⁴ : CommSemiring A
      inst✝³ : AddCommMonoid M
      inst✝² : Algebra R A
      inst✝¹ : Module A M
      inst✝ : Module R M
      D : Derivation R A M
      x : A
      n✝ : Nat
      a✝¹ : R
      a✝ : Eq (D ((Polynomial.aeval x) (HMul.hMul (Polynomial.C a✝¹) (HPow.hPow Poly …
      ⊢ Eq (D ((Polynomial.aeval x) (HMul.hMul (Polynomial.C a✝¹) (HPow.hPow Polynom …
    -/
  · simp [mul_smul, ← Nat.cast_smul_eq_nsmul A]
    /-
      🎉 no goals
    -/


theorem eqOn_adjoin {s : Set A} (h : Set.EqOn D1 D2 s) : Set.EqOn D1 D2 (adjoin R s) := fun _ hx =>
  Algebra.adjoin_induction (hx := hx) h
    (fun r => (D1.map_algebraMap r).trans (D2.map_algebraMap r).symm)
                             /-
                               R : Type u_1
                               A : Type u_2
                               M : Type u_4
                               inst✝⁵ : CommSemiring R
                               inst✝⁴ : CommSemiring A
                               inst✝³ : AddCommMonoid M
                               inst✝² : Algebra R A
                               inst✝¹ : Module A M
                               inst✝ : Module R M
                               D1 D2 : Derivation R A M
                               s : Set A
                               h : Set.EqOn (⇑D1) (⇑D2) s
                               x✝² : A
                               hx✝ : Membership.mem (↑(Algebra.adjoin R s)) x✝²
                               x y : A
                               x✝¹ : Membership.mem (Algebra.adjoin R s) x
                               x✝ : Membership.mem (Algebra.adjoin R s) y
                               hx : Eq (D1 x) (D2 x)
                               hy : Eq (D1 y) (D2 y)
                               ⊢ Eq (D1 (HAdd.hAdd x y)) (D2 (HAdd.hAdd x y))
                             -/
                             /-
                               🎉 no goals
                             -/
    (fun x y _ _ hx hy => by simp only [map_add, *]) fun x y _ _ hx hy => by simp only [leibniz, *]
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


/-- If adjoin of a set is the whole algebra, then any two derivations equal on this set are equal
on the whole algebra. -/
theorem ext_of_adjoin_eq_top (s : Set A) (hs : adjoin R s = ⊤) (h : Set.EqOn D1 D2 s) : D1 = D2 :=
  ext fun _ => eqOn_adjoin h <| hs.symm ▸ trivial

-- Data typeclasses

instance : Zero (Derivation R A M) :=
  ⟨{  toLinearMap := 0
      map_one_eq_zero' := rfl
                                /-
                                  R : Type u_1
                                  A : Type u_2
                                  B : Type u_3
                                  M : Type u_4
                                  inst✝⁸ : CommSemiring R
                                  inst✝⁷ : CommSemiring A
                                  inst✝⁶ : CommSemiring B
                                  inst✝⁵ : AddCommMonoid M
                                  inst✝⁴ : Algebra R A
                                  inst✝³ : Algebra R B
                                  inst✝² : Module A M
                                  inst✝¹ : Module B M
                                  inst✝ : Module R M
                                  D D1 D2 : Derivation R A M
                                  r : R
                                  a✝ b✝ a b : A
                                  ⊢ Eq (0 (HMul.hMul a b)) (HAdd.hAdd (HSMul.hSMul a (0 b)) (HSMul.hSMul b (0 a)))
                                -/
      leibniz' := fun a b => by simp only [add_zero, LinearMap.zero_apply, smul_zero] }⟩
                                /-
                                  🎉 no goals
                                -/


@[simp]
theorem coe_zero : ⇑(0 : Derivation R A M) = 0 :=
  rfl


@[simp]
theorem coe_zero_linearMap : ↑(0 : Derivation R A M) = (0 : A →ₗ[R] M) :=
  rfl


theorem zero_apply (a : A) : (0 : Derivation R A M) a = 0 :=
  rfl


instance : Add (Derivation R A M) :=
  ⟨fun D1 D2 =>
    { toLinearMap := D1 + D2
                             /-
                               R : Type u_1
                               A : Type u_2
                               B : Type u_3
                               M : Type u_4
                               inst✝⁸ : CommSemiring R
                               inst✝⁷ : CommSemiring A
                               inst✝⁶ : CommSemiring B
                               inst✝⁵ : AddCommMonoid M
                               inst✝⁴ : Algebra R A
                               inst✝³ : Algebra R B
                               inst✝² : Module A M
                               inst✝¹ : Module B M
                               inst✝ : Module R M
                               D D1✝ D2✝ : Derivation R A M
                               r : R
                               a b : A
                               D1 D2 : Derivation R A M
                               ⊢ Eq ((HAdd.hAdd ↑D1 ↑D2) 1) 0
                             -/
      map_one_eq_zero' := by simp
                             /-
                               🎉 no goals
                             -/
      leibniz' := fun a b => by
        /-
          R : Type u_1
          A : Type u_2
          B : Type u_3
          M : Type u_4
          inst✝⁸ : CommSemiring R
          inst✝⁷ : CommSemiring A
          inst✝⁶ : CommSemiring B
          inst✝⁵ : AddCommMonoid M
          inst✝⁴ : Algebra R A
          inst✝³ : Algebra R B
          inst✝² : Module A M
          inst✝¹ : Module B M
          inst✝ : Module R M
          D D1✝ D2✝ : Derivation R A M
          r : R
          a✝ b✝ : A
          D1 D2 : Derivation R A M
          a b : A
          ⊢ Eq ((HAdd.hAdd ↑D1 ↑D2) (HMul.hMul a b)) (HAdd.hAdd (HSMul.hSMul a ((HAdd.hA …
        -/
        simp only [leibniz, LinearMap.add_apply, coeFn_coe, smul_add, add_add_add_comm] }⟩
        /-
          🎉 no goals
        -/


@[simp]
theorem coe_add (D1 D2 : Derivation R A M) : ⇑(D1 + D2) = D1 + D2 :=
  rfl


@[simp]
theorem coe_add_linearMap (D1 D2 : Derivation R A M) : ↑(D1 + D2) = (D1 + D2 : A →ₗ[R] M) :=
  rfl


theorem add_apply : (D1 + D2) a = D1 a + D2 a :=
  rfl


instance : Inhabited (Derivation R A M) :=
  ⟨0⟩


instance : SMul S (Derivation R A M) :=
  ⟨fun r D =>
    { toLinearMap := r • D.1
                             /-
                               R : Type u_1
                               A : Type u_2
                               B : Type u_3
                               M : Type u_4
                               inst✝¹⁶ : CommSemiring R
                               inst✝¹⁵ : CommSemiring A
                               inst✝¹⁴ : CommSemiring B
                               inst✝¹³ : AddCommMonoid M
                               inst✝¹² : Algebra R A
                               inst✝¹¹ : Algebra R B
                               inst✝¹⁰ : Module A M
                               inst✝⁹ : Module B M
                               inst✝⁸ : Module R M
                               D✝ D1 D2 : Derivation R A M
                               r✝ : R
                               a b : A
                               S : Type u_5
                               T : Type u_6
                               inst✝⁷ : Monoid S
                               inst✝⁶ : DistribMulAction S M
                               inst✝⁵ : SMulCommClass R S M
                               inst✝⁴ : SMulCommClass S A M
                               inst✝³ : Monoid T
                               inst✝² : DistribMulAction T M
                               inst✝¹ : SMulCommClass R T M
                               inst✝ : SMulCommClass T A M
                               r : S
                               D : Derivation R A M
                               ⊢ Eq ((HSMul.hSMul r ↑D) 1) 0
                             -/
      map_one_eq_zero' := by rw [LinearMap.smul_apply, coeFn_coe, D.map_one_eq_zero, smul_zero]
                             /-
                               🎉 no goals
                             -/
      leibniz' := fun a b => by simp only [LinearMap.smul_apply, coeFn_coe, leibniz, smul_add,
        smul_comm r (_ : A) (_ : M)] }⟩


@[simp]
theorem coe_smul (r : S) (D : Derivation R A M) : ⇑(r • D) = r • ⇑D :=
  rfl


@[simp]
theorem coe_smul_linearMap (r : S) (D : Derivation R A M) : ↑(r • D) = r • (D : A →ₗ[R] M) :=
  rfl


theorem smul_apply (r : S) (D : Derivation R A M) : (r • D) a = r • D a :=
  rfl


instance : AddCommMonoid (Derivation R A M) :=
  coe_injective.addCommMonoid _ coe_zero coe_add fun _ _ => rfl


/-- `coe_fn` as an `AddMonoidHom`. -/
def coeFnAddMonoidHom : Derivation R A M →+ A → M where
  toFun := (↑)
  map_zero' := coe_zero
  map_add' := coe_add


instance : DistribMulAction S (Derivation R A M) :=
  Function.Injective.distribMulAction coeFnAddMonoidHom coe_injective coe_smul


instance [DistribMulAction Sᵐᵒᵖ M] [IsCentralScalar S M] :
    IsCentralScalar S (Derivation R A M) where
  op_smul_eq_smul _ _ := ext fun _ => op_smul_eq_smul _ _


instance [SMul S T] [IsScalarTower S T M] : IsScalarTower S T (Derivation R A M) :=
  ⟨fun _ _ _ => ext fun _ => smul_assoc _ _ _⟩


instance [SMulCommClass S T M] : SMulCommClass S T (Derivation R A M) :=
  ⟨fun _ _ _ => ext fun _ => smul_comm _ _ _⟩


instance instModule {S : Type*} [Semiring S] [Module S M] [SMulCommClass R S M]
    [SMulCommClass S A M] : Module S (Derivation R A M) :=
  Function.Injective.module S coeFnAddMonoidHom coe_injective coe_smul


/-- We can push forward derivations using linear maps, i.e., the composition of a derivation with a
linear map is a derivation. Furthermore, this operation is linear on the spaces of derivations. -/
def _root_.LinearMap.compDer : Derivation R A M →ₗ[R] Derivation R A N where
  toFun D :=
    { toLinearMap := (f : M →ₗ[R] N).comp (D : A →ₗ[R] M)
                             /-
                               R : Type u_1
                               A : Type u_2
                               B : Type u_3
                               M : Type u_4
                               inst✝¹³ : CommSemiring R
                               inst✝¹² : CommSemiring A
                               inst✝¹¹ : CommSemiring B
                               inst✝¹⁰ : AddCommMonoid M
                               inst✝⁹ : Algebra R A
                               inst✝⁸ : Algebra R B
                               inst✝⁷ : Module A M
                               inst✝⁶ : Module B M
                               inst✝⁵ : Module R M
                               D✝ D1 D2 : Derivation R A M
                               r : R
                               a b : A
                               N : Type u_5
                               inst✝⁴ : AddCommMonoid N
                               inst✝³ : Module A N
                               inst✝² : Module R N
                               inst✝¹ : IsScalarTower R A M
                               inst✝ : IsScalarTower R A N
                               f : LinearMap (RingHom.id A) M N
                               e : LinearEquiv (RingHom.id A) M N
                               D : Derivation R A M
                               ⊢ Eq (((↑R f).comp ↑D) 1) 0
                             -/
      map_one_eq_zero' := by simp only [LinearMap.comp_apply, coeFn_coe, map_one_eq_zero, map_zero]
                             /-
                               🎉 no goals
                             -/
      leibniz' := fun a b => by
        simp only [coeFn_coe, LinearMap.comp_apply, LinearMap.map_add, leibniz,
          LinearMap.coe_restrictScalars, LinearMap.map_smul] }
                       /-
                         R : Type u_1
                         A : Type u_2
                         B : Type u_3
                         M : Type u_4
                         inst✝¹³ : CommSemiring R
                         inst✝¹² : CommSemiring A
                         inst✝¹¹ : CommSemiring B
                         inst✝¹⁰ : AddCommMonoid M
                         inst✝⁹ : Algebra R A
                         inst✝⁸ : Algebra R B
                         inst✝⁷ : Module A M
                         inst✝⁶ : Module B M
                         inst✝⁵ : Module R M
                         D D1 D2 : Derivation R A M
                         r : R
                         a b : A
                         N : Type u_5
                         inst✝⁴ : AddCommMonoid N
                         inst✝³ : Module A N
                         inst✝² : Module R N
                         inst✝¹ : IsScalarTower R A M
                         inst✝ : IsScalarTower R A N
                         f : LinearMap (RingHom.id A) M N
                         e : LinearEquiv (RingHom.id A) M N
                         D₁ D₂ : Derivation R A M
                         ⊢ Eq ((fun D => { toLinearMap := (↑R f).comp ↑D, map_one_eq_zero' := ⋯, leibni …
                       -/
  map_add' D₁ D₂ := by ext; exact LinearMap.map_add _ _ _
                            /-
                              🎉 no goals
                            -/
                      /-
                        R : Type u_1
                        A : Type u_2
                        B : Type u_3
                        M : Type u_4
                        inst✝¹³ : CommSemiring R
                        inst✝¹² : CommSemiring A
                        inst✝¹¹ : CommSemiring B
                        inst✝¹⁰ : AddCommMonoid M
                        inst✝⁹ : Algebra R A
                        inst✝⁸ : Algebra R B
                        inst✝⁷ : Module A M
                        inst✝⁶ : Module B M
                        inst✝⁵ : Module R M
                        D✝ D1 D2 : Derivation R A M
                        r✝ : R
                        a b : A
                        N : Type u_5
                        inst✝⁴ : AddCommMonoid N
                        inst✝³ : Module A N
                        inst✝² : Module R N
                        inst✝¹ : IsScalarTower R A M
                        inst✝ : IsScalarTower R A N
                        f : LinearMap (RingHom.id A) M N
                        e : LinearEquiv (RingHom.id A) M N
                        r : R
                        D : Derivation R A M
                        ⊢ Eq ({ toFun := fun D => { toLinearMap := (↑R f).comp ↑D, map_one_eq_zero' := …
                      -/
  map_smul' r D := by dsimp; ext; exact LinearMap.map_smul (f : M →ₗ[R] N) _ _
                                  /-
                                    🎉 no goals
                                  -/


@[simp]
theorem coe_to_linearMap_comp : (f.compDer D : A →ₗ[R] N) = (f : M →ₗ[R] N).comp (D : A →ₗ[R] M) :=
  rfl


@[simp]
theorem coe_comp : (f.compDer D : A → N) = (f : M →ₗ[R] N).comp (D : A →ₗ[R] M) :=
  rfl


/-- The composition of a derivation with a linear map as a bilinear map -/
@[simps]
def llcomp : (M →ₗ[A] N) →ₗ[A] Derivation R A M →ₗ[R] Derivation R A N where
  toFun f := f.compDer
                       /-
                         R : Type u_1
                         A : Type u_2
                         B : Type u_3
                         M : Type u_4
                         inst✝¹³ : CommSemiring R
                         inst✝¹² : CommSemiring A
                         inst✝¹¹ : CommSemiring B
                         inst✝¹⁰ : AddCommMonoid M
                         inst✝⁹ : Algebra R A
                         inst✝⁸ : Algebra R B
                         inst✝⁷ : Module A M
                         inst✝⁶ : Module B M
                         inst✝⁵ : Module R M
                         D D1 D2 : Derivation R A M
                         r : R
                         a b : A
                         N : Type u_5
                         inst✝⁴ : AddCommMonoid N
                         inst✝³ : Module A N
                         inst✝² : Module R N
                         inst✝¹ : IsScalarTower R A M
                         inst✝ : IsScalarTower R A N
                         f : LinearMap (RingHom.id A) M N
                         e : LinearEquiv (RingHom.id A) M N
                         f₁ f₂ : LinearMap (RingHom.id A) M N
                         ⊢ Eq ((fun f => f.compDer) (HAdd.hAdd f₁ f₂)) (HAdd.hAdd ((fun f => f.compDer) …
                       -/
  map_add' f₁ f₂ := by ext; rfl
                            /-
                              🎉 no goals
                            -/
                      /-
                        R : Type u_1
                        A : Type u_2
                        B : Type u_3
                        M : Type u_4
                        inst✝¹³ : CommSemiring R
                        inst✝¹² : CommSemiring A
                        inst✝¹¹ : CommSemiring B
                        inst✝¹⁰ : AddCommMonoid M
                        inst✝⁹ : Algebra R A
                        inst✝⁸ : Algebra R B
                        inst✝⁷ : Module A M
                        inst✝⁶ : Module B M
                        inst✝⁵ : Module R M
                        D✝ D1 D2 : Derivation R A M
                        r✝ : R
                        a b : A
                        N : Type u_5
                        inst✝⁴ : AddCommMonoid N
                        inst✝³ : Module A N
                        inst✝² : Module R N
                        inst✝¹ : IsScalarTower R A M
                        inst✝ : IsScalarTower R A N
                        f : LinearMap (RingHom.id A) M N
                        e : LinearEquiv (RingHom.id A) M N
                        r : A
                        D : LinearMap (RingHom.id A) M N
                        ⊢ Eq ({ toFun := fun f => f.compDer, map_add' := ⋯ }.toFun (HSMul.hSMul r D))  …
                      -/
  map_smul' r D := by ext; rfl
                           /-
                             🎉 no goals
                           -/


/-- Pushing a derivation forward through a linear equivalence is an equivalence. -/
def _root_.LinearEquiv.compDer : Derivation R A M ≃ₗ[R] Derivation R A N :=
  { e.toLinearMap.compDer with
    invFun := e.symm.toLinearMap.compDer
                            /-
                              R : Type u_1
                              A : Type u_2
                              B : Type u_3
                              M : Type u_4
                              inst✝¹³ : CommSemiring R
                              inst✝¹² : CommSemiring A
                              inst✝¹¹ : CommSemiring B
                              inst✝¹⁰ : AddCommMonoid M
                              inst✝⁹ : Algebra R A
                              inst✝⁸ : Algebra R B
                              inst✝⁷ : Module A M
                              inst✝⁶ : Module B M
                              inst✝⁵ : Module R M
                              D✝ D1 D2 : Derivation R A M
                              r : R
                              a b : A
                              N : Type u_5
                              inst✝⁴ : AddCommMonoid N
                              inst✝³ : Module A N
                              inst✝² : Module R N
                              inst✝¹ : IsScalarTower R A M
                              inst✝ : IsScalarTower R A N
                              f : LinearMap (RingHom.id A) M N
                              e : LinearEquiv (RingHom.id A) M N
                              D : Derivation R A M
                              ⊢ Eq ((↑e.symm).compDer (__src✝.toFun D)) D
                            -/
    left_inv := fun D => by ext a; exact e.symm_apply_apply (D a)
                                   /-
                                     🎉 no goals
                                   -/
                             /-
                               R : Type u_1
                               A : Type u_2
                               B : Type u_3
                               M : Type u_4
                               inst✝¹³ : CommSemiring R
                               inst✝¹² : CommSemiring A
                               inst✝¹¹ : CommSemiring B
                               inst✝¹⁰ : AddCommMonoid M
                               inst✝⁹ : Algebra R A
                               inst✝⁸ : Algebra R B
                               inst✝⁷ : Module A M
                               inst✝⁶ : Module B M
                               inst✝⁵ : Module R M
                               D✝ D1 D2 : Derivation R A M
                               r : R
                               a b : A
                               N : Type u_5
                               inst✝⁴ : AddCommMonoid N
                               inst✝³ : Module A N
                               inst✝² : Module R N
                               inst✝¹ : IsScalarTower R A M
                               inst✝ : IsScalarTower R A N
                               f : LinearMap (RingHom.id A) M N
                               e : LinearEquiv (RingHom.id A) M N
                               D : Derivation R A N
                               ⊢ Eq (__src✝.toFun ((↑e.symm).compDer D)) D
                             -/
    right_inv := fun D => by ext a; exact e.apply_symm_apply (D a) }
                                    /-
                                      🎉 no goals
                                    -/


@[simp]
theorem linearEquiv_coe_to_linearMap_comp :
    (e.compDer D : A →ₗ[R] N) = (e.toLinearMap : M →ₗ[R] N).comp (D : A →ₗ[R] M) :=
  rfl


@[simp]
theorem linearEquiv_coe_comp :
    (e.compDer D : A → N) = (e.toLinearMap : M →ₗ[R] N).comp (D : A →ₗ[R] M) :=
  rfl


variable (A) in
/-- For a tower `R → A → B` and an `R`-derivation `B → M`, we may compose with `A → B` to obtain an
`R`-derivation `A → M`. -/
@[simps!]
def compAlgebraMap [Algebra A B] [IsScalarTower R A B] [IsScalarTower A B M]
    (d : Derivation R B M) : Derivation R A M where
                         /-
                           R : Type u_1
                           A : Type u_2
                           B : Type u_3
                           M : Type u_4
                           inst✝¹¹ : CommSemiring R
                           inst✝¹⁰ : CommSemiring A
                           inst✝⁹ : CommSemiring B
                           inst✝⁸ : AddCommMonoid M
                           inst✝⁷ : Algebra R A
                           inst✝⁶ : Algebra R B
                           inst✝⁵ : Module A M
                           inst✝⁴ : Module B M
                           inst✝³ : Module R M
                           D D1 D2 : Derivation R A M
                           r : R
                           a b : A
                           inst✝² : Algebra A B
                           inst✝¹ : IsScalarTower R A B
                           inst✝ : IsScalarTower A B M
                           d : Derivation R B M
                           ⊢ Eq (((↑d).comp (IsScalarTower.toAlgHom R A B).toLinearMap) 1) 0
                         -/
  map_one_eq_zero' := by simp
                         /-
                           🎉 no goals
                         -/
                     /-
                       R : Type u_1
                       A : Type u_2
                       B : Type u_3
                       M : Type u_4
                       inst✝¹¹ : CommSemiring R
                       inst✝¹⁰ : CommSemiring A
                       inst✝⁹ : CommSemiring B
                       inst✝⁸ : AddCommMonoid M
                       inst✝⁷ : Algebra R A
                       inst✝⁶ : Algebra R B
                       inst✝⁵ : Module A M
                       inst✝⁴ : Module B M
                       inst✝³ : Module R M
                       D D1 D2 : Derivation R A M
                       r : R
                       a✝ b✝ : A
                       inst✝² : Algebra A B
                       inst✝¹ : IsScalarTower R A B
                       inst✝ : IsScalarTower A B M
                       d : Derivation R B M
                       a b : A
                       ⊢ Eq (((↑d).comp (IsScalarTower.toAlgHom R A B).toLinearMap) (HMul.hMul a b))  …
                     -/
  leibniz' a b := by simp
                     /-
                       🎉 no goals
                     -/
  toLinearMap := d.toLinearMap.comp (IsScalarTower.toAlgHom R A B).toLinearMap


/-- If `A` is both an `R`-algebra and an `S`-algebra; `M` is both an `R`-module and an `S`-module,
then an `S`-derivation `A → M` is also an `R`-derivation if it is also `R`-linear. -/
protected def restrictScalars (d : Derivation S A M) : Derivation R A M where
  map_one_eq_zero' := d.map_one_eq_zero
  leibniz' := d.leibniz
  toLinearMap := d.toLinearMap.restrictScalars R


lemma coe_restrictScalars (d : Derivation S A M) : ⇑(d.restrictScalars R) = ⇑d := rfl


@[simp]
lemma restrictScalars_apply (d : Derivation S A M) (x : A) : d.restrictScalars R x = d x := rfl


/--
Lift a derivation via an algebra homomorphism `f` with a right inverse such that
`f(x) = 0 → f(d(x)) = 0`. This gives the derivation `f ∘ d ∘ f⁻¹`.
This is needed for an argument in [Rosenlicht, M. Integration in finite terms][Rosenlicht_1972].
-/
def liftOfRightInverse {f : F} {f_inv : M → A} (hf : Function.RightInverse f_inv f)
    ⦃d : Derivation R A A⦄ (hd : ∀ x, f x = 0 → f (d x) = 0) : Derivation R M M where
  toFun x := f (d (f_inv x))
  map_add' x y := by
    /-
      R : Type u_1
      A : Type u_2
      M : Type u_3
      inst✝⁶ : CommSemiring R
      inst✝⁵ : CommRing A
      inst✝⁴ : CommRing M
      inst✝³ : Algebra R A
      inst✝² : Algebra R M
      F : Type u_4
      inst✝¹ : FunLike F A M
      inst✝ : AlgHomClass F R A M
      f : F
      f_inv : M → A
      hf : Function.RightInverse f_inv ⇑f
      d : Derivation R A A
      hd : ∀ (x : A), Eq (f x) 0 → Eq (f (d x)) 0
      x y : M
      ⊢ Eq ((fun x => f (d (f_inv x))) (HAdd.hAdd x y)) (HAdd.hAdd ((fun x => f (d ( …
    -/
    suffices f (d (f_inv (x + y) - (f_inv x + f_inv y))) = 0 by simpa [sub_eq_zero]
    /-
      R : Type u_1
      A : Type u_2
      M : Type u_3
      inst✝⁶ : CommSemiring R
      inst✝⁵ : CommRing A
      inst✝⁴ : CommRing M
      inst✝³ : Algebra R A
      inst✝² : Algebra R M
      F : Type u_4
      inst✝¹ : FunLike F A M
      inst✝ : AlgHomClass F R A M
      f : F
      f_inv : M → A
      hf : Function.RightInverse f_inv ⇑f
      d : Derivation R A A
      hd : ∀ (x : A), Eq (f x) 0 → Eq (f (d x)) 0
      x y : M
      ⊢ Eq (f (d (HSub.hSub (f_inv (HAdd.hAdd x y)) (HAdd.hAdd (f_inv x) (f_inv y))) …
    -/
    apply hd
    /-
      case a
      R : Type u_1
      A : Type u_2
      M : Type u_3
      inst✝⁶ : CommSemiring R
      inst✝⁵ : CommRing A
      inst✝⁴ : CommRing M
      inst✝³ : Algebra R A
      inst✝² : Algebra R M
      F : Type u_4
      inst✝¹ : FunLike F A M
      inst✝ : AlgHomClass F R A M
      f : F
      f_inv : M → A
      hf : Function.RightInverse f_inv ⇑f
      d : Derivation R A A
      hd : ∀ (x : A), Eq (f x) 0 → Eq (f (d x)) 0
      x y : M
      ⊢ Eq (f (HSub.hSub (f_inv (HAdd.hAdd x y)) (HAdd.hAdd (f_inv x) (f_inv y)))) 0
    -/
    simp [hf _]
    /-
      🎉 no goals
    -/
  map_smul' x y := by
    /-
      R : Type u_1
      A : Type u_2
      M : Type u_3
      inst✝⁶ : CommSemiring R
      inst✝⁵ : CommRing A
      inst✝⁴ : CommRing M
      inst✝³ : Algebra R A
      inst✝² : Algebra R M
      F : Type u_4
      inst✝¹ : FunLike F A M
      inst✝ : AlgHomClass F R A M
      f : F
      f_inv : M → A
      hf : Function.RightInverse f_inv ⇑f
      d : Derivation R A A
      hd : ∀ (x : A), Eq (f x) 0 → Eq (f (d x)) 0
      x : R
      y : M
      ⊢ Eq ({ toFun := fun x => f (d (f_inv x)), map_add' := ⋯ }.toFun (HSMul.hSMul  …
    -/
    suffices f (d (f_inv (x • y) - x • f_inv y)) = 0 by simpa [sub_eq_zero]
    /-
      R : Type u_1
      A : Type u_2
      M : Type u_3
      inst✝⁶ : CommSemiring R
      inst✝⁵ : CommRing A
      inst✝⁴ : CommRing M
      inst✝³ : Algebra R A
      inst✝² : Algebra R M
      F : Type u_4
      inst✝¹ : FunLike F A M
      inst✝ : AlgHomClass F R A M
      f : F
      f_inv : M → A
      hf : Function.RightInverse f_inv ⇑f
      d : Derivation R A A
      hd : ∀ (x : A), Eq (f x) 0 → Eq (f (d x)) 0
      x : R
      y : M
      ⊢ Eq (f (d (HSub.hSub (f_inv (HSMul.hSMul x y)) (HSMul.hSMul x (f_inv y))))) 0
    -/
    apply hd
    /-
      case a
      R : Type u_1
      A : Type u_2
      M : Type u_3
      inst✝⁶ : CommSemiring R
      inst✝⁵ : CommRing A
      inst✝⁴ : CommRing M
      inst✝³ : Algebra R A
      inst✝² : Algebra R M
      F : Type u_4
      inst✝¹ : FunLike F A M
      inst✝ : AlgHomClass F R A M
      f : F
      f_inv : M → A
      hf : Function.RightInverse f_inv ⇑f
      d : Derivation R A A
      hd : ∀ (x : A), Eq (f x) 0 → Eq (f (d x)) 0
      x : R
      y : M
      ⊢ Eq (f (HSub.hSub (f_inv (HSMul.hSMul x y)) (HSMul.hSMul x (f_inv y)))) 0
    -/
    simp [hf _]
    /-
      🎉 no goals
    -/
  map_one_eq_zero' := by
    /-
      R : Type u_1
      A : Type u_2
      M : Type u_3
      inst✝⁶ : CommSemiring R
      inst✝⁵ : CommRing A
      inst✝⁴ : CommRing M
      inst✝³ : Algebra R A
      inst✝² : Algebra R M
      F : Type u_4
      inst✝¹ : FunLike F A M
      inst✝ : AlgHomClass F R A M
      f : F
      f_inv : M → A
      hf : Function.RightInverse f_inv ⇑f
      d : Derivation R A A
      hd : ∀ (x : A), Eq (f x) 0 → Eq (f (d x)) 0
      ⊢ Eq ({ toFun := fun x => f (d (f_inv x)), map_add' := ⋯, map_smul' := ⋯ } 1) 0
    -/
    suffices f (d (f_inv 1 - 1)) = 0 by simpa [sub_eq_zero]
    /-
      R : Type u_1
      A : Type u_2
      M : Type u_3
      inst✝⁶ : CommSemiring R
      inst✝⁵ : CommRing A
      inst✝⁴ : CommRing M
      inst✝³ : Algebra R A
      inst✝² : Algebra R M
      F : Type u_4
      inst✝¹ : FunLike F A M
      inst✝ : AlgHomClass F R A M
      f : F
      f_inv : M → A
      hf : Function.RightInverse f_inv ⇑f
      d : Derivation R A A
      hd : ∀ (x : A), Eq (f x) 0 → Eq (f (d x)) 0
      ⊢ Eq (f (d (HSub.hSub (f_inv 1) 1))) 0
    -/
    apply hd
    /-
      case a
      R : Type u_1
      A : Type u_2
      M : Type u_3
      inst✝⁶ : CommSemiring R
      inst✝⁵ : CommRing A
      inst✝⁴ : CommRing M
      inst✝³ : Algebra R A
      inst✝² : Algebra R M
      F : Type u_4
      inst✝¹ : FunLike F A M
      inst✝ : AlgHomClass F R A M
      f : F
      f_inv : M → A
      hf : Function.RightInverse f_inv ⇑f
      d : Derivation R A A
      hd : ∀ (x : A), Eq (f x) 0 → Eq (f (d x)) 0
      ⊢ Eq (f (HSub.hSub (f_inv 1) 1)) 0
    -/
    simp [hf _]
    /-
      🎉 no goals
    -/
  leibniz' x y := by
    /-
      R : Type u_1
      A : Type u_2
      M : Type u_3
      inst✝⁶ : CommSemiring R
      inst✝⁵ : CommRing A
      inst✝⁴ : CommRing M
      inst✝³ : Algebra R A
      inst✝² : Algebra R M
      F : Type u_4
      inst✝¹ : FunLike F A M
      inst✝ : AlgHomClass F R A M
      f : F
      f_inv : M → A
      hf : Function.RightInverse f_inv ⇑f
      d : Derivation R A A
      hd : ∀ (x : A), Eq (f x) 0 → Eq (f (d x)) 0
      x y : M
      ⊢ Eq ({ toFun := fun x => f (d (f_inv x)), map_add' := ⋯, map_smul' := ⋯ } (HM …
    -/
    suffices f (d (f_inv (x * y) - f_inv x * f_inv y)) = 0 by simpa [sub_eq_zero, hf _]
    /-
      R : Type u_1
      A : Type u_2
      M : Type u_3
      inst✝⁶ : CommSemiring R
      inst✝⁵ : CommRing A
      inst✝⁴ : CommRing M
      inst✝³ : Algebra R A
      inst✝² : Algebra R M
      F : Type u_4
      inst✝¹ : FunLike F A M
      inst✝ : AlgHomClass F R A M
      f : F
      f_inv : M → A
      hf : Function.RightInverse f_inv ⇑f
      d : Derivation R A A
      hd : ∀ (x : A), Eq (f x) 0 → Eq (f (d x)) 0
      x y : M
      ⊢ Eq (f (d (HSub.hSub (f_inv (HMul.hMul x y)) (HMul.hMul (f_inv x) (f_inv y))) …
    -/
    apply hd
    /-
      case a
      R : Type u_1
      A : Type u_2
      M : Type u_3
      inst✝⁶ : CommSemiring R
      inst✝⁵ : CommRing A
      inst✝⁴ : CommRing M
      inst✝³ : Algebra R A
      inst✝² : Algebra R M
      F : Type u_4
      inst✝¹ : FunLike F A M
      inst✝ : AlgHomClass F R A M
      f : F
      f_inv : M → A
      hf : Function.RightInverse f_inv ⇑f
      d : Derivation R A A
      hd : ∀ (x : A), Eq (f x) 0 → Eq (f (d x)) 0
      x y : M
      ⊢ Eq (f (HSub.hSub (f_inv (HMul.hMul x y)) (HMul.hMul (f_inv x) (f_inv y)))) 0
    -/
    simp [hf _]
    /-
      🎉 no goals
    -/


@[simp]
lemma liftOfRightInverse_apply {f : F} {f_inv : M → A} (hf : Function.RightInverse f_inv f)
    {d : Derivation R A A} (hd : ∀ x, f x = 0 → f (d x) = 0) (x : A) :
    Derivation.liftOfRightInverse hf hd (f x) = f (d x) := by
  /-
    R : Type u_1
    A : Type u_2
    M : Type u_3
    inst✝⁶ : CommSemiring R
    inst✝⁵ : CommRing A
    inst✝⁴ : CommRing M
    inst✝³ : Algebra R A
    inst✝² : Algebra R M
    F : Type u_4
    inst✝¹ : FunLike F A M
    inst✝ : AlgHomClass F R A M
    f : F
    f_inv : M → A
    hf : Function.RightInverse f_inv ⇑f
    d : Derivation R A A
    hd : ∀ (x : A), Eq (f x) 0 → Eq (f (d x)) 0
    x : A
    ⊢ Eq ((Derivation.liftOfRightInverse hf hd) (f x)) (f (d x))
  -/
  suffices f (d (f_inv (f x) - x)) = 0 by simpa [sub_eq_zero]
  /-
    R : Type u_1
    A : Type u_2
    M : Type u_3
    inst✝⁶ : CommSemiring R
    inst✝⁵ : CommRing A
    inst✝⁴ : CommRing M
    inst✝³ : Algebra R A
    inst✝² : Algebra R M
    F : Type u_4
    inst✝¹ : FunLike F A M
    inst✝ : AlgHomClass F R A M
    f : F
    f_inv : M → A
    hf : Function.RightInverse f_inv ⇑f
    d : Derivation R A A
    hd : ∀ (x : A), Eq (f x) 0 → Eq (f (d x)) 0
    x : A
    ⊢ Eq (f (d (HSub.hSub (f_inv (f x)) x))) 0
  -/
  apply hd
  /-
    case a
    R : Type u_1
    A : Type u_2
    M : Type u_3
    inst✝⁶ : CommSemiring R
    inst✝⁵ : CommRing A
    inst✝⁴ : CommRing M
    inst✝³ : Algebra R A
    inst✝² : Algebra R M
    F : Type u_4
    inst✝¹ : FunLike F A M
    inst✝ : AlgHomClass F R A M
    f : F
    f_inv : M → A
    hf : Function.RightInverse f_inv ⇑f
    d : Derivation R A A
    hd : ∀ (x : A), Eq (f x) 0 → Eq (f (d x)) 0
    x : A
    ⊢ Eq (f (HSub.hSub (f_inv (f x)) x)) 0
  -/
  simp [hf _]
  /-
    🎉 no goals
  -/


lemma liftOfRightInverse_eq {f : F} {f_inv₁ f_inv₂ : M → A} (hf₁ : Function.RightInverse f_inv₁ f)
    (hf₂ : Function.RightInverse f_inv₂ f) :
    liftOfRightInverse hf₁ = liftOfRightInverse hf₂ := by
  /-
    R : Type u_1
    A : Type u_2
    M : Type u_3
    inst✝⁶ : CommSemiring R
    inst✝⁵ : CommRing A
    inst✝⁴ : CommRing M
    inst✝³ : Algebra R A
    inst✝² : Algebra R M
    F : Type u_4
    inst✝¹ : FunLike F A M
    inst✝ : AlgHomClass F R A M
    f : F
    f_inv₁ f_inv₂ : M → A
    hf₁ : Function.RightInverse f_inv₁ ⇑f
    hf₂ : Function.RightInverse f_inv₂ ⇑f
    ⊢ Eq (Derivation.liftOfRightInverse hf₁) (Derivation.liftOfRightInverse hf₂)
  -/
  ext _ _ x
  /-
    case h.h.H
    R : Type u_1
    A : Type u_2
    M : Type u_3
    inst✝⁶ : CommSemiring R
    inst✝⁵ : CommRing A
    inst✝⁴ : CommRing M
    inst✝³ : Algebra R A
    inst✝² : Algebra R M
    F : Type u_4
    inst✝¹ : FunLike F A M
    inst✝ : AlgHomClass F R A M
    f : F
    f_inv₁ f_inv₂ : M → A
    hf₁ : Function.RightInverse f_inv₁ ⇑f
    hf₂ : Function.RightInverse f_inv₂ ⇑f
    x✝¹ : Derivation R A A
    x✝ : ∀ (x : A), Eq (f x) 0 → Eq (f (x✝¹ x)) 0
    x : M
    ⊢ Eq ((Derivation.liftOfRightInverse hf₁ x✝) x) ((Derivation.liftOfRightInvers …
  -/
  obtain ⟨x, rfl⟩ := hf₁.surjective x
  /-
    case h.h.H.intro
    R : Type u_1
    A : Type u_2
    M : Type u_3
    inst✝⁶ : CommSemiring R
    inst✝⁵ : CommRing A
    inst✝⁴ : CommRing M
    inst✝³ : Algebra R A
    inst✝² : Algebra R M
    F : Type u_4
    inst✝¹ : FunLike F A M
    inst✝ : AlgHomClass F R A M
    f : F
    f_inv₁ f_inv₂ : M → A
    hf₁ : Function.RightInverse f_inv₁ ⇑f
    hf₂ : Function.RightInverse f_inv₂ ⇑f
    x✝¹ : Derivation R A A
    x✝ : ∀ (x : A), Eq (f x) 0 → Eq (f (x✝¹ x)) 0
    x : A
    ⊢ Eq ((Derivation.liftOfRightInverse hf₁ x✝) (f x)) ((Derivation.liftOfRightIn …
  -/
  simp
  /-
    🎉 no goals
  -/


/--
A noncomputable version of `liftOfRightInverse` for surjective homomorphisms.
-/
noncomputable abbrev liftOfSurjective {f : F} (hf : Function.Surjective f)
    ⦃d : Derivation R A A⦄ (hd : ∀ x, f x = 0 → f (d x) = 0) : Derivation R M M :=
  d.liftOfRightInverse (Function.rightInverse_surjInv hf) hd


lemma liftOfSurjective_apply {f : F} (hf : Function.Surjective f)
    {d : Derivation R A A} (hd : ∀ x, f x = 0 → f (d x) = 0) (x : A) :
                                                            /-
                                                              R : Type u_1
                                                              A : Type u_2
                                                              M : Type u_3
                                                              inst✝⁶ : CommSemiring R
                                                              inst✝⁵ : CommRing A
                                                              inst✝⁴ : CommRing M
                                                              inst✝³ : Algebra R A
                                                              inst✝² : Algebra R M
                                                              F : Type u_4
                                                              inst✝¹ : FunLike F A M
                                                              inst✝ : AlgHomClass F R A M
                                                              f : F
                                                              hf : Function.Surjective ⇑f
                                                              d : Derivation R A A
                                                              hd : ∀ (x : A), Eq (f x) 0 → Eq (f (d x)) 0
                                                              x : A
                                                              ⊢ Eq ((Derivation.liftOfSurjective hf hd) (f x)) (f (d x))
                                                            -/
    Derivation.liftOfSurjective hf hd (f x) = f (d x) := by simp
                                                            /-
                                                              🎉 no goals
                                                            -/


/-- Define `Derivation R A M` from a linear map when `M` is cancellative by verifying the Leibniz
rule. -/
def mk' (D : A →ₗ[R] M) (h : ∀ a b, D (a * b) = a • D b + b • D a) : Derivation R A M where
  toLinearMap := D
  map_one_eq_zero' := (add_right_eq_self (a := D 1)).1 <| by
    /-
      R : Type u_1
      inst✝⁵ : CommSemiring R
      A : Type u_2
      inst✝⁴ : CommSemiring A
      inst✝³ : Algebra R A
      M : Type u_3
      inst✝² : AddCancelCommMonoid M
      inst✝¹ : Module R M
      inst✝ : Module A M
      D : LinearMap (RingHom.id R) A M
      h : ∀ (a b : A), Eq (D (HMul.hMul a b)) (HAdd.hAdd (HSMul.hSMul a (D b)) (HSMu …
      ⊢ Eq (HAdd.hAdd (D 1) (D 1)) (D 1)
    -/
    simpa only [one_smul, one_mul] using (h 1 1).symm
    /-
      🎉 no goals
    -/
  leibniz' := h


@[simp]
theorem coe_mk' (D : A →ₗ[R] M) (h) : ⇑(mk' D h) = D :=
  rfl


@[simp]
theorem coe_mk'_linearMap (D : A →ₗ[R] M) (h) : (mk' D h : A →ₗ[R] M) = D :=
  rfl


protected theorem map_neg : D (-a) = -D a :=
  map_neg D a


protected theorem map_sub : D (a - b) = D a - D b :=
  map_sub D a b


@[simp]
theorem map_intCast (n : ℤ) : D (n : A) = 0 := by
  /-
    R : Type u_1
    inst✝⁵ : CommRing R
    A : Type u_2
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    M : Type u_3
    inst✝² : AddCommGroup M
    inst✝¹ : Module A M
    inst✝ : Module R M
    D : Derivation R A M
    n : Int
    ⊢ Eq (D ↑n) 0
  -/
  rw [← zsmul_one, D.map_smul_of_tower n, map_one_eq_zero, smul_zero]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-05")] alias map_coe_nat := map_natCast

@[deprecated (since := "2024-04-05")] alias map_coe_int := map_intCast


theorem leibniz_of_mul_eq_one {a b : A} (h : a * b = 1) : D a = -a ^ 2 • D b := by
  /-
    R : Type u_1
    inst✝⁵ : CommRing R
    A : Type u_2
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    M : Type u_3
    inst✝² : AddCommGroup M
    inst✝¹ : Module A M
    inst✝ : Module R M
    D : Derivation R A M
    a b : A
    h : Eq (HMul.hMul a b) 1
    ⊢ Eq (D a) (HSMul.hSMul (Neg.neg (HPow.hPow a 2)) (D b))
  -/
  rw [neg_smul]
  /-
    R : Type u_1
    inst✝⁵ : CommRing R
    A : Type u_2
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    M : Type u_3
    inst✝² : AddCommGroup M
    inst✝¹ : Module A M
    inst✝ : Module R M
    D : Derivation R A M
    a b : A
    h : Eq (HMul.hMul a b) 1
    ⊢ Eq (D a) (Neg.neg (HSMul.hSMul (HPow.hPow a 2) (D b)))
  -/
  refine eq_neg_of_add_eq_zero_left ?_
  calc
    D a + a ^ 2 • D b = a • b • D a + a • a • D b := by simp only [smul_smul, h, one_smul, sq]
    _ = a • D (a * b) := by rw [leibniz, smul_add, add_comm]
    _ = 0 := by rw [h, map_one_eq_zero, smul_zero]


theorem leibniz_invOf [Invertible a] : D (⅟ a) = -⅟ a ^ 2 • D a :=
  D.leibniz_of_mul_eq_one <| invOf_mul_self a


theorem leibniz_inv (a : K) : D a⁻¹ = -a⁻¹ ^ 2 • D a := by
  /-
    R : Type u_1
    inst✝⁵ : CommRing R
    M : Type u_3
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    K : Type u_4
    inst✝² : Field K
    inst✝¹ : Module K M
    inst✝ : Algebra R K
    D : Derivation R K M
    a : K
    ⊢ Eq (D (Inv.inv a)) (HSMul.hSMul (Neg.neg (HPow.hPow (Inv.inv a) 2)) (D a))
  -/
  rcases eq_or_ne a 0 with (rfl | ha)
    /-
      case inl
      R : Type u_1
      inst✝⁵ : CommRing R
      M : Type u_3
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      K : Type u_4
      inst✝² : Field K
      inst✝¹ : Module K M
      inst✝ : Algebra R K
      D : Derivation R K M
      ⊢ Eq (D (Inv.inv 0)) (HSMul.hSMul (Neg.neg (HPow.hPow (Inv.inv 0) 2)) (D 0))
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      R : Type u_1
      inst✝⁵ : CommRing R
      M : Type u_3
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      K : Type u_4
      inst✝² : Field K
      inst✝¹ : Module K M
      inst✝ : Algebra R K
      D : Derivation R K M
      a : K
      ha : Ne a 0
      ⊢ Eq (D (Inv.inv a)) (HSMul.hSMul (Neg.neg (HPow.hPow (Inv.inv a) 2)) (D a))
    -/
  · exact D.leibniz_of_mul_eq_one (inv_mul_cancel₀ ha)
    /-
      🎉 no goals
    -/


theorem leibniz_div (a b : K) : D (a / b) = b⁻¹ ^ 2 • (b • D a - a • D b) := by
  simp only [div_eq_mul_inv, leibniz, leibniz_inv, inv_pow, neg_smul, smul_neg, smul_smul, add_comm,
    sub_eq_add_neg, smul_add]
  /-
    R : Type u_1
    inst✝⁵ : CommRing R
    M : Type u_3
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    K : Type u_4
    inst✝² : Field K
    inst✝¹ : Module K M
    inst✝ : Algebra R K
    D : Derivation R K M
    a b : K
    ⊢ Eq (HAdd.hAdd (HSMul.hSMul (Inv.inv b) (D a)) (Neg.neg (HSMul.hSMul (HMul.hM …
  -/
  rw [← inv_mul_mul_self b⁻¹, inv_inv]
  /-
    R : Type u_1
    inst✝⁵ : CommRing R
    M : Type u_3
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    K : Type u_4
    inst✝² : Field K
    inst✝¹ : Module K M
    inst✝ : Algebra R K
    D : Derivation R K M
    a b : K
    ⊢ Eq (HAdd.hAdd (HSMul.hSMul (HMul.hMul (HMul.hMul b (Inv.inv b)) (Inv.inv b)) …
  -/
  ring_nf
  /-
    🎉 no goals
  -/


theorem leibniz_div_const (a b : K) (h : D b = 0) : D (a / b) = b⁻¹ • D a := by
  /-
    R : Type u_1
    inst✝⁵ : CommRing R
    M : Type u_3
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    K : Type u_4
    inst✝² : Field K
    inst✝¹ : Module K M
    inst✝ : Algebra R K
    D : Derivation R K M
    a b : K
    h : Eq (D b) 0
    ⊢ Eq (D (HDiv.hDiv a b)) (HSMul.hSMul (Inv.inv b) (D a))
  -/
  simp only [leibniz_div, inv_pow, h, smul_zero, sub_zero, smul_smul]
  /-
    R : Type u_1
    inst✝⁵ : CommRing R
    M : Type u_3
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    K : Type u_4
    inst✝² : Field K
    inst✝¹ : Module K M
    inst✝ : Algebra R K
    D : Derivation R K M
    a b : K
    h : Eq (D b) 0
    ⊢ Eq (HSMul.hSMul (HMul.hMul (Inv.inv (HPow.hPow b 2)) b) (D a)) (HSMul.hSMul  …
  -/
  rw [← mul_self_mul_inv b⁻¹, inv_inv]
  /-
    R : Type u_1
    inst✝⁵ : CommRing R
    M : Type u_3
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    K : Type u_4
    inst✝² : Field K
    inst✝¹ : Module K M
    inst✝ : Algebra R K
    D : Derivation R K M
    a b : K
    h : Eq (D b) 0
    ⊢ Eq (HSMul.hSMul (HMul.hMul (Inv.inv (HPow.hPow b 2)) b) (D a)) (HSMul.hSMul  …
  -/
  ring_nf
  /-
    🎉 no goals
  -/


lemma leibniz_zpow (a : K) (n : ℤ) : D (a ^ n) = n • a ^ (n - 1) • D a := by
  /-
    R : Type u_1
    inst✝⁵ : CommRing R
    M : Type u_3
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    K : Type u_4
    inst✝² : Field K
    inst✝¹ : Module K M
    inst✝ : Algebra R K
    D : Derivation R K M
    a : K
    n : Int
    ⊢ Eq (D (HPow.hPow a n)) (HSMul.hSMul n (HSMul.hSMul (HPow.hPow a (HSub.hSub n …
  -/
  by_cases hn : n = 0
    /-
      case pos
      R : Type u_1
      inst✝⁵ : CommRing R
      M : Type u_3
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      K : Type u_4
      inst✝² : Field K
      inst✝¹ : Module K M
      inst✝ : Algebra R K
      D : Derivation R K M
      a : K
      n : Int
      hn : Eq n 0
      ⊢ Eq (D (HPow.hPow a n)) (HSMul.hSMul n (HSMul.hSMul (HPow.hPow a (HSub.hSub n …
    -/
  · simp [hn]
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u_1
    inst✝⁵ : CommRing R
    M : Type u_3
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    K : Type u_4
    inst✝² : Field K
    inst✝¹ : Module K M
    inst✝ : Algebra R K
    D : Derivation R K M
    a : K
    n : Int
    hn : Not (Eq n 0)
    ⊢ Eq (D (HPow.hPow a n)) (HSMul.hSMul n (HSMul.hSMul (HPow.hPow a (HSub.hSub n …
  -/
  by_cases ha : a = 0
    /-
      case pos
      R : Type u_1
      inst✝⁵ : CommRing R
      M : Type u_3
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      K : Type u_4
      inst✝² : Field K
      inst✝¹ : Module K M
      inst✝ : Algebra R K
      D : Derivation R K M
      a : K
      n : Int
      hn : Not (Eq n 0)
      ha : Eq a 0
      ⊢ Eq (D (HPow.hPow a n)) (HSMul.hSMul n (HSMul.hSMul (HPow.hPow a (HSub.hSub n …
    -/
  · simp [ha, zero_zpow n hn]
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u_1
    inst✝⁵ : CommRing R
    M : Type u_3
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    K : Type u_4
    inst✝² : Field K
    inst✝¹ : Module K M
    inst✝ : Algebra R K
    D : Derivation R K M
    a : K
    n : Int
    hn : Not (Eq n 0)
    ha : Not (Eq a 0)
    ⊢ Eq (D (HPow.hPow a n)) (HSMul.hSMul n (HSMul.hSMul (HPow.hPow a (HSub.hSub n …
  -/
  rcases Int.natAbs_eq n with h | h
    /-
      case neg.inl
      R : Type u_1
      inst✝⁵ : CommRing R
      M : Type u_3
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      K : Type u_4
      inst✝² : Field K
      inst✝¹ : Module K M
      inst✝ : Algebra R K
      D : Derivation R K M
      a : K
      n : Int
      hn : Not (Eq n 0)
      ha : Not (Eq a 0)
      h : Eq n ↑n.natAbs
      ⊢ Eq (D (HPow.hPow a n)) (HSMul.hSMul n (HSMul.hSMul (HPow.hPow a (HSub.hSub n …
    -/
  · rw [h]
    /-
      case neg.inl
      R : Type u_1
      inst✝⁵ : CommRing R
      M : Type u_3
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      K : Type u_4
      inst✝² : Field K
      inst✝¹ : Module K M
      inst✝ : Algebra R K
      D : Derivation R K M
      a : K
      n : Int
      hn : Not (Eq n 0)
      ha : Not (Eq a 0)
      h : Eq n ↑n.natAbs
      ⊢ Eq (D (HPow.hPow a ↑n.natAbs)) (HSMul.hSMul (↑n.natAbs) (HSMul.hSMul (HPow.h …
    -/
    simp only [zpow_natCast, leibniz_pow, natCast_zsmul]
    /-
      case neg.inl
      R : Type u_1
      inst✝⁵ : CommRing R
      M : Type u_3
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      K : Type u_4
      inst✝² : Field K
      inst✝¹ : Module K M
      inst✝ : Algebra R K
      D : Derivation R K M
      a : K
      n : Int
      hn : Not (Eq n 0)
      ha : Not (Eq a 0)
      h : Eq n ↑n.natAbs
      ⊢ Eq (HSMul.hSMul n.natAbs (HSMul.hSMul (HPow.hPow a (HSub.hSub n.natAbs 1)) ( …
    -/
    rw [← zpow_natCast]
    /-
      case neg.inl
      R : Type u_1
      inst✝⁵ : CommRing R
      M : Type u_3
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      K : Type u_4
      inst✝² : Field K
      inst✝¹ : Module K M
      inst✝ : Algebra R K
      D : Derivation R K M
      a : K
      n : Int
      hn : Not (Eq n 0)
      ha : Not (Eq a 0)
      h : Eq n ↑n.natAbs
      ⊢ Eq (HSMul.hSMul n.natAbs (HSMul.hSMul (HPow.hPow a ↑(HSub.hSub n.natAbs 1))  …
    -/
    congr
    /-
      case neg.inl.e_a.e_a.e_a
      R : Type u_1
      inst✝⁵ : CommRing R
      M : Type u_3
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      K : Type u_4
      inst✝² : Field K
      inst✝¹ : Module K M
      inst✝ : Algebra R K
      D : Derivation R K M
      a : K
      n : Int
      hn : Not (Eq n 0)
      ha : Not (Eq a 0)
      h : Eq n ↑n.natAbs
      ⊢ Eq (↑(HSub.hSub n.natAbs 1)) (HSub.hSub (↑n.natAbs) 1)
    -/
    omega
    /-
      🎉 no goals
    -/
  · rw [h, zpow_neg, zpow_natCast, leibniz_inv, leibniz_pow, inv_pow, ← pow_mul, ← zpow_natCast,
      ← zpow_natCast, ← Nat.cast_smul_eq_nsmul K, ← Int.cast_smul_eq_zsmul K, smul_smul, smul_smul,
      smul_smul]
    /-
      case neg.inr
      R : Type u_1
      inst✝⁵ : CommRing R
      M : Type u_3
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      K : Type u_4
      inst✝² : Field K
      inst✝¹ : Module K M
      inst✝ : Algebra R K
      D : Derivation R K M
      a : K
      n : Int
      hn : Not (Eq n 0)
      ha : Not (Eq a 0)
      h : Eq n (Neg.neg ↑n.natAbs)
      ⊢ Eq (HSMul.hSMul (HMul.hMul (HMul.hMul (Neg.neg (Inv.inv (HPow.hPow a ↑(HMul. …
    -/
    trans (-n.natAbs * (a ^ ((n.natAbs - 1 : ℕ) : ℤ) / (a ^ ((n.natAbs * 2 : ℕ) : ℤ)))) • D a
      /-
        R : Type u_1
        inst✝⁵ : CommRing R
        M : Type u_3
        inst✝⁴ : AddCommGroup M
        inst✝³ : Module R M
        K : Type u_4
        inst✝² : Field K
        inst✝¹ : Module K M
        inst✝ : Algebra R K
        D : Derivation R K M
        a : K
        n : Int
        hn : Not (Eq n 0)
        ha : Not (Eq a 0)
        h : Eq n (Neg.neg ↑n.natAbs)
        ⊢ Eq (HSMul.hSMul (HMul.hMul (HMul.hMul (Neg.neg (Inv.inv (HPow.hPow a ↑(HMul. …
      -/
    · ring_nf
      /-
        🎉 no goals
      -/
    /-
      R : Type u_1
      inst✝⁵ : CommRing R
      M : Type u_3
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      K : Type u_4
      inst✝² : Field K
      inst✝¹ : Module K M
      inst✝ : Algebra R K
      D : Derivation R K M
      a : K
      n : Int
      hn : Not (Eq n 0)
      ha : Not (Eq a 0)
      h : Eq n (Neg.neg ↑n.natAbs)
      ⊢ Eq (HSMul.hSMul (HMul.hMul (Neg.neg ↑n.natAbs) (HDiv.hDiv (HPow.hPow a ↑(HSu …
    -/
    rw [← zpow_sub₀ ha]
    /-
      R : Type u_1
      inst✝⁵ : CommRing R
      M : Type u_3
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      K : Type u_4
      inst✝² : Field K
      inst✝¹ : Module K M
      inst✝ : Algebra R K
      D : Derivation R K M
      a : K
      n : Int
      hn : Not (Eq n 0)
      ha : Not (Eq a 0)
      h : Eq n (Neg.neg ↑n.natAbs)
      ⊢ Eq (HSMul.hSMul (HMul.hMul (Neg.neg ↑n.natAbs) (HPow.hPow a (HSub.hSub ↑(HSu …
    -/
    congr 3
      /-
        case e_a.e_a
        R : Type u_1
        inst✝⁵ : CommRing R
        M : Type u_3
        inst✝⁴ : AddCommGroup M
        inst✝³ : Module R M
        K : Type u_4
        inst✝² : Field K
        inst✝¹ : Module K M
        inst✝ : Algebra R K
        D : Derivation R K M
        a : K
        n : Int
        hn : Not (Eq n 0)
        ha : Not (Eq a 0)
        h : Eq n (Neg.neg ↑n.natAbs)
        ⊢ Eq (Neg.neg ↑n.natAbs) ↑(Neg.neg ↑n.natAbs)
      -/
    · norm_cast
      /-
        🎉 no goals
      -/
    /-
      case e_a.e_a.e_a
      R : Type u_1
      inst✝⁵ : CommRing R
      M : Type u_3
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      K : Type u_4
      inst✝² : Field K
      inst✝¹ : Module K M
      inst✝ : Algebra R K
      D : Derivation R K M
      a : K
      n : Int
      hn : Not (Eq n 0)
      ha : Not (Eq a 0)
      h : Eq n (Neg.neg ↑n.natAbs)
      ⊢ Eq (HSub.hSub ↑(HSub.hSub n.natAbs 1) ↑(HMul.hMul n.natAbs 2)) (HSub.hSub (N …
    -/
    omega
    /-
      🎉 no goals
    -/


instance : Neg (Derivation R A M) :=
  ⟨fun D =>
    mk' (-D) fun a b => by
      /-
        R : Type u_1
        inst✝⁵ : CommRing R
        A : Type u_2
        inst✝⁴ : CommRing A
        inst✝³ : Algebra R A
        M : Type u_3
        inst✝² : AddCommGroup M
        inst✝¹ : Module A M
        inst✝ : Module R M
        D✝ D1 D2 : Derivation R A M
        r : R
        a✝ b✝ : A
        D : Derivation R A M
        a b : A
        ⊢ Eq ((Neg.neg ↑D) (HMul.hMul a b)) (HAdd.hAdd (HSMul.hSMul a ((Neg.neg ↑D) b) …
      -/
      simp only [LinearMap.neg_apply, smul_neg, neg_add_rev, leibniz, coeFn_coe, add_comm]⟩
      /-
        🎉 no goals
      -/


@[simp]
theorem coe_neg (D : Derivation R A M) : ⇑(-D) = -D :=
  rfl


@[simp]
theorem coe_neg_linearMap (D : Derivation R A M) : ↑(-D) = (-D : A →ₗ[R] M) :=
  rfl


theorem neg_apply : (-D) a = -D a :=
  rfl


instance : Sub (Derivation R A M) :=
  ⟨fun D1 D2 =>
    mk' (D1 - D2 : A →ₗ[R] M) fun a b => by
      /-
        R : Type u_1
        inst✝⁵ : CommRing R
        A : Type u_2
        inst✝⁴ : CommRing A
        inst✝³ : Algebra R A
        M : Type u_3
        inst✝² : AddCommGroup M
        inst✝¹ : Module A M
        inst✝ : Module R M
        D D1✝ D2✝ : Derivation R A M
        r : R
        a✝ b✝ : A
        D1 D2 : Derivation R A M
        a b : A
        ⊢ Eq ((HSub.hSub ↑D1 ↑D2) (HMul.hMul a b)) (HAdd.hAdd (HSMul.hSMul a ((HSub.hS …
      -/
      simp only [LinearMap.sub_apply, leibniz, coeFn_coe, smul_sub, add_sub_add_comm]⟩
      /-
        🎉 no goals
      -/


@[simp]
theorem coe_sub (D1 D2 : Derivation R A M) : ⇑(D1 - D2) = D1 - D2 :=
  rfl


@[simp]
theorem coe_sub_linearMap (D1 D2 : Derivation R A M) : ↑(D1 - D2) = (D1 - D2 : A →ₗ[R] M) :=
  rfl


theorem sub_apply : (D1 - D2) a = D1 a - D2 a :=
  rfl


instance : AddCommGroup (Derivation R A M) :=
  coe_injective.addCommGroup _ coe_zero coe_add coe_neg coe_sub (fun _ _ => rfl) fun _ _ => rfl


