/-- `Polynomial R` is the type of univariate polynomials over `R`.

Polynomials should be seen as (semi-)rings with the additional constructor `X`.
The embedding from `R` is called `C`. -/
structure Polynomial (R : Type*) [Semiring R] where ofFinsupp ::
  toFinsupp : AddMonoidAlgebra R ℕ


@[inherit_doc] scoped[Polynomial] notation:9000 R "[X]" => Polynomial R


theorem forall_iff_forall_finsupp (P : R[X] → Prop) :
    (∀ p, P p) ↔ ∀ q : R[ℕ], P ⟨q⟩ :=
  ⟨fun h q => h ⟨q⟩, fun h ⟨p⟩ => h p⟩


theorem exists_iff_exists_finsupp (P : R[X] → Prop) :
    (∃ p, P p) ↔ ∃ q : R[ℕ], P ⟨q⟩ :=
  ⟨fun ⟨⟨p⟩, hp⟩ => ⟨p, hp⟩, fun ⟨q, hq⟩ => ⟨⟨q⟩, hq⟩⟩


@[simp]
                                                                    /-
                                                                      R : Type u
                                                                      inst✝ : Semiring R
                                                                      f : Polynomial R
                                                                      ⊢ Eq { toFinsupp := f.toFinsupp } f
                                                                    -/
theorem eta (f : R[X]) : Polynomial.ofFinsupp f.toFinsupp = f := by cases f; rfl
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


private irreducible_def add : R[X] → R[X] → R[X]
  | ⟨a⟩, ⟨b⟩ => ⟨a + b⟩


private irreducible_def neg {R : Type u} [Ring R] : R[X] → R[X]
  | ⟨a⟩ => ⟨-a⟩


private irreducible_def mul : R[X] → R[X] → R[X]
  | ⟨a⟩, ⟨b⟩ => ⟨a * b⟩


instance zero : Zero R[X] :=
  ⟨⟨0⟩⟩


instance one : One R[X] :=
  ⟨⟨1⟩⟩


instance add' : Add R[X] :=
  ⟨add⟩


instance neg' {R : Type u} [Ring R] : Neg R[X] :=
  ⟨neg⟩


instance sub {R : Type u} [Ring R] : Sub R[X] :=
  ⟨fun a b => a + -b⟩


instance mul' : Mul R[X] :=
  ⟨mul⟩

-- If the private definitions are accidentally exposed, simplify them away.

@[simp] theorem add_eq_add : add p q = p + q := rfl

@[simp] theorem mul_eq_mul : mul p q = p * q := rfl


instance instNSMul : SMul ℕ R[X] where
  smul r p := ⟨r • p.toFinsupp⟩


instance smulZeroClass {S : Type*} [SMulZeroClass S R] : SMulZeroClass S R[X] where
  smul r p := ⟨r • p.toFinsupp⟩
  smul_zero a := congr_arg ofFinsupp (smul_zero a)


instance {S : Type*} [Zero S] [SMulZeroClass S R] [NoZeroSMulDivisors S R] :
    NoZeroSMulDivisors S R[X] where
  eq_zero_or_eq_zero_of_smul_eq_zero eq :=
    (eq_zero_or_eq_zero_of_smul_eq_zero <| congr_arg toFinsupp eq).imp id (congr_arg ofFinsupp)

-- to avoid a bug in the `ring` tactic

instance (priority := 1) pow : Pow R[X] ℕ where pow p n := npowRec n p


@[simp]
theorem ofFinsupp_zero : (⟨0⟩ : R[X]) = 0 :=
  rfl


@[simp]
theorem ofFinsupp_one : (⟨1⟩ : R[X]) = 1 :=
  rfl


@[simp]
theorem ofFinsupp_add {a b} : (⟨a + b⟩ : R[X]) = ⟨a⟩ + ⟨b⟩ :=
                      /-
                        R : Type u
                        inst✝ : Semiring R
                        a b : AddMonoidAlgebra R Nat
                        ⊢ Eq { toFinsupp := HAdd.hAdd a b } (Polynomial.add { toFinsupp := a } { toFin …
                      -/
  show _ = add _ _ by rw [add_def]
                      /-
                        🎉 no goals
                      -/


@[simp]
theorem ofFinsupp_neg {R : Type u} [Ring R] {a} : (⟨-a⟩ : R[X]) = -⟨a⟩ :=
                    /-
                      R : Type u
                      inst✝ : Ring R
                      a : AddMonoidAlgebra R Nat
                      ⊢ Eq { toFinsupp := Neg.neg a } (Polynomial.neg { toFinsupp := a })
                    -/
  show _ = neg _ by rw [neg_def]
                    /-
                      🎉 no goals
                    -/


@[simp]
theorem ofFinsupp_sub {R : Type u} [Ring R] {a b} : (⟨a - b⟩ : R[X]) = ⟨a⟩ - ⟨b⟩ := by
  /-
    R : Type u
    inst✝ : Ring R
    a b : AddMonoidAlgebra R Nat
    ⊢ Eq { toFinsupp := HSub.hSub a b } (HSub.hSub { toFinsupp := a } { toFinsupp  …
  -/
  rw [sub_eq_add_neg, ofFinsupp_add, ofFinsupp_neg]
  /-
    R : Type u
    inst✝ : Ring R
    a b : AddMonoidAlgebra R Nat
    ⊢ Eq (HAdd.hAdd { toFinsupp := a } (Neg.neg { toFinsupp := b })) (HSub.hSub {  …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem ofFinsupp_mul (a b) : (⟨a * b⟩ : R[X]) = ⟨a⟩ * ⟨b⟩ :=
                      /-
                        R : Type u
                        inst✝ : Semiring R
                        a b : AddMonoidAlgebra R Nat
                        ⊢ Eq { toFinsupp := HMul.hMul a b } (Polynomial.mul { toFinsupp := a } { toFin …
                      -/
  show _ = mul _ _ by rw [mul_def]
                      /-
                        🎉 no goals
                      -/


@[simp]
theorem ofFinsupp_nsmul (a : ℕ) (b) :
    (⟨a • b⟩ : R[X]) = (a • ⟨b⟩ : R[X]) :=
  rfl


@[simp]
theorem ofFinsupp_smul {S : Type*} [SMulZeroClass S R] (a : S) (b) :
    (⟨a • b⟩ : R[X]) = (a • ⟨b⟩ : R[X]) :=
  rfl


@[simp]
theorem ofFinsupp_pow (a) (n : ℕ) : (⟨a ^ n⟩ : R[X]) = ⟨a⟩ ^ n := by
  /-
    R : Type u
    inst✝ : Semiring R
    a : AddMonoidAlgebra R Nat
    n : Nat
    ⊢ Eq { toFinsupp := HPow.hPow a n } (HPow.hPow { toFinsupp := a } n)
  -/
  change _ = npowRec n _
  induction n with
  | zero        => simp [npowRec]
  | succ n n_ih => simp [npowRec, n_ih, pow_succ]


@[simp]
theorem toFinsupp_zero : (0 : R[X]).toFinsupp = 0 :=
  rfl


@[simp]
theorem toFinsupp_one : (1 : R[X]).toFinsupp = 1 :=
  rfl


@[simp]
theorem toFinsupp_add (a b : R[X]) : (a + b).toFinsupp = a.toFinsupp + b.toFinsupp := by
  /-
    R : Type u
    inst✝ : Semiring R
    a b : Polynomial R
    ⊢ Eq (HAdd.hAdd a b).toFinsupp (HAdd.hAdd a.toFinsupp b.toFinsupp)
  -/
  cases a
  /-
    case ofFinsupp
    R : Type u
    inst✝ : Semiring R
    b : Polynomial R
    toFinsupp✝ : AddMonoidAlgebra R Nat
    ⊢ Eq (HAdd.hAdd { toFinsupp := toFinsupp✝ } b).toFinsupp (HAdd.hAdd { toFinsup …
  -/
  cases b
  /-
    case ofFinsupp.ofFinsupp
    R : Type u
    inst✝ : Semiring R
    toFinsupp✝¹ toFinsupp✝ : AddMonoidAlgebra R Nat
    ⊢ Eq (HAdd.hAdd { toFinsupp := toFinsupp✝¹ } { toFinsupp := toFinsupp✝ }).toFi …
  -/
  rw [← ofFinsupp_add]
  /-
    🎉 no goals
  -/


@[simp]
theorem toFinsupp_neg {R : Type u} [Ring R] (a : R[X]) : (-a).toFinsupp = -a.toFinsupp := by
  /-
    R : Type u
    inst✝ : Ring R
    a : Polynomial R
    ⊢ Eq (Neg.neg a).toFinsupp (Neg.neg a.toFinsupp)
  -/
  cases a
  /-
    case ofFinsupp
    R : Type u
    inst✝ : Ring R
    toFinsupp✝ : AddMonoidAlgebra R Nat
    ⊢ Eq (Neg.neg { toFinsupp := toFinsupp✝ }).toFinsupp (Neg.neg { toFinsupp := t …
  -/
  rw [← ofFinsupp_neg]
  /-
    🎉 no goals
  -/


@[simp]
theorem toFinsupp_sub {R : Type u} [Ring R] (a b : R[X]) :
    (a - b).toFinsupp = a.toFinsupp - b.toFinsupp := by
  /-
    R : Type u
    inst✝ : Ring R
    a b : Polynomial R
    ⊢ Eq (HSub.hSub a b).toFinsupp (HSub.hSub a.toFinsupp b.toFinsupp)
  -/
  rw [sub_eq_add_neg, ← toFinsupp_neg, ← toFinsupp_add]
  /-
    R : Type u
    inst✝ : Ring R
    a b : Polynomial R
    ⊢ Eq (HSub.hSub a b).toFinsupp (HAdd.hAdd a (Neg.neg b)).toFinsupp
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem toFinsupp_mul (a b : R[X]) : (a * b).toFinsupp = a.toFinsupp * b.toFinsupp := by
  /-
    R : Type u
    inst✝ : Semiring R
    a b : Polynomial R
    ⊢ Eq (HMul.hMul a b).toFinsupp (HMul.hMul a.toFinsupp b.toFinsupp)
  -/
  cases a
  /-
    case ofFinsupp
    R : Type u
    inst✝ : Semiring R
    b : Polynomial R
    toFinsupp✝ : AddMonoidAlgebra R Nat
    ⊢ Eq (HMul.hMul { toFinsupp := toFinsupp✝ } b).toFinsupp (HMul.hMul { toFinsup …
  -/
  cases b
  /-
    case ofFinsupp.ofFinsupp
    R : Type u
    inst✝ : Semiring R
    toFinsupp✝¹ toFinsupp✝ : AddMonoidAlgebra R Nat
    ⊢ Eq (HMul.hMul { toFinsupp := toFinsupp✝¹ } { toFinsupp := toFinsupp✝ }).toFi …
  -/
  rw [← ofFinsupp_mul]
  /-
    🎉 no goals
  -/


@[simp]
theorem toFinsupp_nsmul (a : ℕ) (b : R[X]) :
    (a • b).toFinsupp = a • b.toFinsupp :=
  rfl


@[simp]
theorem toFinsupp_smul {S : Type*} [SMulZeroClass S R] (a : S) (b : R[X]) :
    (a • b).toFinsupp = a • b.toFinsupp :=
  rfl


@[simp]
theorem toFinsupp_pow (a : R[X]) (n : ℕ) : (a ^ n).toFinsupp = a.toFinsupp ^ n := by
  /-
    R : Type u
    inst✝ : Semiring R
    a : Polynomial R
    n : Nat
    ⊢ Eq (HPow.hPow a n).toFinsupp (HPow.hPow a.toFinsupp n)
  -/
  cases a
  /-
    case ofFinsupp
    R : Type u
    inst✝ : Semiring R
    n : Nat
    toFinsupp✝ : AddMonoidAlgebra R Nat
    ⊢ Eq (HPow.hPow { toFinsupp := toFinsupp✝ } n).toFinsupp (HPow.hPow { toFinsup …
  -/
  rw [← ofFinsupp_pow]
  /-
    🎉 no goals
  -/


theorem _root_.IsSMulRegular.polynomial {S : Type*} [Monoid S] [DistribMulAction S R] {a : S}
    (ha : IsSMulRegular R a) : IsSMulRegular R[X] a
  | ⟨_x⟩, ⟨_y⟩, h => congr_arg _ <| ha.finsupp (Polynomial.ofFinsupp.inj h)


theorem toFinsupp_injective : Function.Injective (toFinsupp : R[X] → AddMonoidAlgebra _ _) :=
  fun ⟨_x⟩ ⟨_y⟩ => congr_arg _


@[simp]
theorem toFinsupp_inj {a b : R[X]} : a.toFinsupp = b.toFinsupp ↔ a = b :=
  toFinsupp_injective.eq_iff


@[simp]
theorem toFinsupp_eq_zero {a : R[X]} : a.toFinsupp = 0 ↔ a = 0 := by
  /-
    R : Type u
    inst✝ : Semiring R
    a : Polynomial R
    ⊢ Iff (Eq a.toFinsupp 0) (Eq a 0)
  -/
  rw [← toFinsupp_zero, toFinsupp_inj]
  /-
    🎉 no goals
  -/


@[simp]
theorem toFinsupp_eq_one {a : R[X]} : a.toFinsupp = 1 ↔ a = 1 := by
  /-
    R : Type u
    inst✝ : Semiring R
    a : Polynomial R
    ⊢ Iff (Eq a.toFinsupp 1) (Eq a 1)
  -/
  rw [← toFinsupp_one, toFinsupp_inj]
  /-
    🎉 no goals
  -/


/-- A more convenient spelling of `Polynomial.ofFinsupp.injEq` in terms of `Iff`. -/
theorem ofFinsupp_inj {a b} : (⟨a⟩ : R[X]) = ⟨b⟩ ↔ a = b :=
  iff_of_eq (ofFinsupp.injEq _ _)


@[simp]
theorem ofFinsupp_eq_zero {a} : (⟨a⟩ : R[X]) = 0 ↔ a = 0 := by
  /-
    R : Type u
    inst✝ : Semiring R
    a : AddMonoidAlgebra R Nat
    ⊢ Iff (Eq { toFinsupp := a } 0) (Eq a 0)
  -/
  rw [← ofFinsupp_zero, ofFinsupp_inj]
  /-
    🎉 no goals
  -/


@[simp]
                                                              /-
                                                                R : Type u
                                                                inst✝ : Semiring R
                                                                a : AddMonoidAlgebra R Nat
                                                                ⊢ Iff (Eq { toFinsupp := a } 1) (Eq a 1)
                                                              -/
theorem ofFinsupp_eq_one {a} : (⟨a⟩ : R[X]) = 1 ↔ a = 1 := by rw [← ofFinsupp_one, ofFinsupp_inj]
                                                              /-
                                                                🎉 no goals
                                                              -/


instance inhabited : Inhabited R[X] :=
  ⟨0⟩


instance instNatCast : NatCast R[X] where natCast n := ofFinsupp n


instance semiring : Semiring R[X] :=
  --TODO: add reference to library note in PR https://github.com/leanprover-community/mathlib4/pull/7432
  { Function.Injective.semiring toFinsupp toFinsupp_injective toFinsupp_zero toFinsupp_one
      toFinsupp_add toFinsupp_mul (fun _ _ => toFinsupp_nsmul _ _) toFinsupp_pow fun _ => rfl with
    toAdd := Polynomial.add'
    toMul := Polynomial.mul'
    toZero := Polynomial.zero
    toOne := Polynomial.one
    nsmul := (· • ·)
    npow := fun n x => (x ^ n) }


instance distribSMul {S} [DistribSMul S R] : DistribSMul S R[X] :=
  --TODO: add reference to library note in PR https://github.com/leanprover-community/mathlib4/pull/7432
  { Function.Injective.distribSMul ⟨⟨toFinsupp, toFinsupp_zero⟩, toFinsupp_add⟩ toFinsupp_injective
      toFinsupp_smul with
    toSMulZeroClass := Polynomial.smulZeroClass }


instance distribMulAction {S} [Monoid S] [DistribMulAction S R] : DistribMulAction S R[X] :=
  --TODO: add reference to library note in PR https://github.com/leanprover-community/mathlib4/pull/7432
  { Function.Injective.distribMulAction ⟨⟨toFinsupp, toFinsupp_zero (R := R)⟩, toFinsupp_add⟩
      toFinsupp_injective toFinsupp_smul with
    toSMul := Polynomial.smulZeroClass.toSMul }


instance faithfulSMul {S} [SMulZeroClass S R] [FaithfulSMul S R] : FaithfulSMul S R[X] where
  eq_of_smul_eq_smul {_s₁ _s₂} h :=
    eq_of_smul_eq_smul fun a : ℕ →₀ R => congr_arg toFinsupp (h ⟨a⟩)


instance module {S} [Semiring S] [Module S R] : Module S R[X] :=
  --TODO: add reference to library note in PR https://github.com/leanprover-community/mathlib4/pull/7432
  { Function.Injective.module _ ⟨⟨toFinsupp, toFinsupp_zero⟩, toFinsupp_add⟩ toFinsupp_injective
      toFinsupp_smul with
    toDistribMulAction := Polynomial.distribMulAction }


instance smulCommClass {S₁ S₂} [SMulZeroClass S₁ R] [SMulZeroClass S₂ R] [SMulCommClass S₁ S₂ R] :
  SMulCommClass S₁ S₂ R[X] :=
  ⟨by
    /-
      R : Type u
      a b : R
      m n : Nat
      inst✝³ : Semiring R
      p q : Polynomial R
      S₁ : Type u_1
      S₂ : Type u_2
      inst✝² : SMulZeroClass S₁ R
      inst✝¹ : SMulZeroClass S₂ R
      inst✝ : SMulCommClass S₁ S₂ R
      ⊢ ∀ (m : S₁) (n : S₂) (a : Polynomial R), Eq (HSMul.hSMul m (HSMul.hSMul n a)) …
    -/
    rintro m n ⟨f⟩
    /-
      case ofFinsupp
      R : Type u
      a b : R
      m✝ n✝ : Nat
      inst✝³ : Semiring R
      p q : Polynomial R
      S₁ : Type u_1
      S₂ : Type u_2
      inst✝² : SMulZeroClass S₁ R
      inst✝¹ : SMulZeroClass S₂ R
      inst✝ : SMulCommClass S₁ S₂ R
      m : S₁
      n : S₂
      f : AddMonoidAlgebra R Nat
      ⊢ Eq (HSMul.hSMul m (HSMul.hSMul n { toFinsupp := f })) (HSMul.hSMul n (HSMul. …
    -/
    simp_rw [← ofFinsupp_smul, smul_comm m n f]⟩
    /-
      🎉 no goals
    -/


instance isScalarTower {S₁ S₂} [SMul S₁ S₂] [SMulZeroClass S₁ R] [SMulZeroClass S₂ R]
  [IsScalarTower S₁ S₂ R] : IsScalarTower S₁ S₂ R[X] :=
  ⟨by
    /-
      R : Type u
      a b : R
      m n : Nat
      inst✝⁴ : Semiring R
      p q : Polynomial R
      S₁ : Type u_1
      S₂ : Type u_2
      inst✝³ : SMul S₁ S₂
      inst✝² : SMulZeroClass S₁ R
      inst✝¹ : SMulZeroClass S₂ R
      inst✝ : IsScalarTower S₁ S₂ R
      ⊢ ∀ (x : S₁) (y : S₂) (z : Polynomial R), Eq (HSMul.hSMul (HSMul.hSMul x y) z) …
    -/
    rintro _ _ ⟨⟩
    /-
      case ofFinsupp
      R : Type u
      a b : R
      m n : Nat
      inst✝⁴ : Semiring R
      p q : Polynomial R
      S₁ : Type u_1
      S₂ : Type u_2
      inst✝³ : SMul S₁ S₂
      inst✝² : SMulZeroClass S₁ R
      inst✝¹ : SMulZeroClass S₂ R
      inst✝ : IsScalarTower S₁ S₂ R
      x✝ : S₁
      y✝ : S₂
      toFinsupp✝ : AddMonoidAlgebra R Nat
      ⊢ Eq (HSMul.hSMul (HSMul.hSMul x✝ y✝) { toFinsupp := toFinsupp✝ }) (HSMul.hSMu …
    -/
    simp_rw [← ofFinsupp_smul, smul_assoc]⟩
    /-
      🎉 no goals
    -/


instance isScalarTower_right {α K : Type*} [Semiring K] [DistribSMul α K] [IsScalarTower α K K] :
    IsScalarTower α K[X] K[X] :=
  ⟨by
    /-
      R : Type u
      a b : R
      m n : Nat
      inst✝³ : Semiring R
      p q : Polynomial R
      α : Type u_1
      K : Type u_2
      inst✝² : Semiring K
      inst✝¹ : DistribSMul α K
      inst✝ : IsScalarTower α K K
      ⊢ ∀ (x : α) (y z : Polynomial K), Eq (HSMul.hSMul (HSMul.hSMul x y) z) (HSMul. …
    -/
    rintro _ ⟨⟩ ⟨⟩
    /-
      case ofFinsupp.ofFinsupp
      R : Type u
      a b : R
      m n : Nat
      inst✝³ : Semiring R
      p q : Polynomial R
      α : Type u_1
      K : Type u_2
      inst✝² : Semiring K
      inst✝¹ : DistribSMul α K
      inst✝ : IsScalarTower α K K
      x✝ : α
      toFinsupp✝¹ toFinsupp✝ : AddMonoidAlgebra K Nat
      ⊢ Eq (HSMul.hSMul (HSMul.hSMul x✝ { toFinsupp := toFinsupp✝¹ }) { toFinsupp := …
    -/
    simp_rw [smul_eq_mul, ← ofFinsupp_smul, ← ofFinsupp_mul, ← ofFinsupp_smul, smul_mul_assoc]⟩
    /-
      🎉 no goals
    -/


instance isCentralScalar {S} [SMulZeroClass S R] [SMulZeroClass Sᵐᵒᵖ R] [IsCentralScalar S R] :
  IsCentralScalar S R[X] :=
  ⟨by
    /-
      R : Type u
      a b : R
      m n : Nat
      inst✝³ : Semiring R
      p q : Polynomial R
      S : Type u_1
      inst✝² : SMulZeroClass S R
      inst✝¹ : SMulZeroClass (MulOpposite S) R
      inst✝ : IsCentralScalar S R
      ⊢ ∀ (m : S) (a : Polynomial R), Eq (HSMul.hSMul (MulOpposite.op m) a) (HSMul.h …
    -/
    rintro _ ⟨⟩
    /-
      case ofFinsupp
      R : Type u
      a b : R
      m n : Nat
      inst✝³ : Semiring R
      p q : Polynomial R
      S : Type u_1
      inst✝² : SMulZeroClass S R
      inst✝¹ : SMulZeroClass (MulOpposite S) R
      inst✝ : IsCentralScalar S R
      m✝ : S
      toFinsupp✝ : AddMonoidAlgebra R Nat
      ⊢ Eq (HSMul.hSMul (MulOpposite.op m✝) { toFinsupp := toFinsupp✝ }) (HSMul.hSMu …
    -/
    simp_rw [← ofFinsupp_smul, op_smul_eq_smul]⟩
    /-
      🎉 no goals
    -/


instance unique [Subsingleton R] : Unique R[X] :=
  { Polynomial.inhabited with
    uniq := by
      /-
        R : Type u
        a b : R
        m n : Nat
        inst✝¹ : Semiring R
        p q : Polynomial R
        inst✝ : Subsingleton R
        ⊢ ∀ (a : Polynomial R), Eq a Inhabited.default
      -/
      rintro ⟨x⟩
      /-
        case ofFinsupp
        R : Type u
        a b : R
        m n : Nat
        inst✝¹ : Semiring R
        p q : Polynomial R
        inst✝ : Subsingleton R
        x : AddMonoidAlgebra R Nat
        ⊢ Eq { toFinsupp := x } Inhabited.default
      -/
      apply congr_arg ofFinsupp
      /-
        case ofFinsupp
        R : Type u
        a b : R
        m n : Nat
        inst✝¹ : Semiring R
        p q : Polynomial R
        inst✝ : Subsingleton R
        x : AddMonoidAlgebra R Nat
        ⊢ Eq x 0
      -/
      simp [eq_iff_true_of_subsingleton] }
      /-
        🎉 no goals
      -/


/-- Ring isomorphism between `R[X]` and `R[ℕ]`. This is just an
implementation detail, but it can be useful to transfer results from `Finsupp` to polynomials. -/
@[simps apply symm_apply]
def toFinsuppIso : R[X] ≃+* R[ℕ] where
  toFun := toFinsupp
  invFun := ofFinsupp
  left_inv := fun ⟨_p⟩ => rfl
  right_inv _p := rfl
  map_mul' := toFinsupp_mul
  map_add' := toFinsupp_add


instance [DecidableEq R] : DecidableEq R[X] :=
  @Equiv.decidableEq R[X] _ (toFinsuppIso R).toEquiv (Finsupp.instDecidableEq)


/-- Linear isomorphism between `R[X]` and `R[ℕ]`. This is just an
implementation detail, but it can be useful to transfer results from `Finsupp` to polynomials. -/
@[simps!]
def toFinsuppIsoLinear : R[X] ≃ₗ[R] R[ℕ] where
  __ := toFinsuppIso R
  map_smul' _ _ := rfl


theorem ofFinsupp_sum {ι : Type*} (s : Finset ι) (f : ι → R[ℕ]) :
    (⟨∑ i ∈ s, f i⟩ : R[X]) = ∑ i ∈ s, ⟨f i⟩ :=
  map_sum (toFinsuppIso R).symm f s


theorem toFinsupp_sum {ι : Type*} (s : Finset ι) (f : ι → R[X]) :
    (∑ i ∈ s, f i : R[X]).toFinsupp = ∑ i ∈ s, (f i).toFinsupp :=
  map_sum (toFinsuppIso R) f s


/-- The set of all `n` such that `X^n` has a non-zero coefficient.
-/
-- @[simp] -- Porting note: The original generated theorem is same to `support_ofFinsupp` and
           --               the new generated theorem is different, so this attribute should be
           --               removed.
def support : R[X] → Finset ℕ
  | ⟨p⟩ => p.support


@[simp]
                                                                       /-
                                                                         R : Type u
                                                                         inst✝ : Semiring R
                                                                         p : AddMonoidAlgebra R Nat
                                                                         ⊢ Eq { toFinsupp := p }.support p.support
                                                                       -/
theorem support_ofFinsupp (p) : support (⟨p⟩ : R[X]) = p.support := by rw [support]
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


                                                                             /-
                                                                               R : Type u
                                                                               inst✝ : Semiring R
                                                                               p : Polynomial R
                                                                               ⊢ Eq p.toFinsupp.support p.support
                                                                             -/
theorem support_toFinsupp (p : R[X]) : p.toFinsupp.support = p.support := by rw [support]
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


@[simp]
theorem support_zero : (0 : R[X]).support = ∅ :=
  rfl


@[simp]
theorem support_eq_empty : p.support = ∅ ↔ p = 0 := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    ⊢ Iff (Eq p.support EmptyCollection.emptyCollection) (Eq p 0)
  -/
  rcases p with ⟨⟩
  /-
    case ofFinsupp
    R : Type u
    inst✝ : Semiring R
    toFinsupp✝ : AddMonoidAlgebra R Nat
    ⊢ Iff (Eq { toFinsupp := toFinsupp✝ }.support EmptyCollection.emptyCollection) …
  -/
  simp [support]
  /-
    🎉 no goals
  -/


@[simp] lemma support_nonempty : p.support.Nonempty ↔ p ≠ 0 :=
  Finset.nonempty_iff_ne_empty.trans support_eq_empty.not


                                                            /-
                                                              R : Type u
                                                              inst✝ : Semiring R
                                                              p : Polynomial R
                                                              ⊢ Iff (Eq p.support.card 0) (Eq p 0)
                                                            -/
theorem card_support_eq_zero : #p.support = 0 ↔ p = 0 := by simp
                                                            /-
                                                              🎉 no goals
                                                            -/


/-- `monomial s a` is the monomial `a * X^s` -/
def monomial (n : ℕ) : R →ₗ[R] R[X] where
  toFun t := ⟨Finsupp.single n t⟩
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10745): was `simp`.
                     /-
                       R : Type u
                       a b : R
                       m n✝ : Nat
                       inst✝ : Semiring R
                       p q : Polynomial R
                       n : Nat
                       x y : R
                       ⊢ Eq ((fun t => { toFinsupp := Finsupp.single n t }) (HAdd.hAdd x y)) (HAdd.hA …
                     -/
  map_add' x y := by simp; rw [ofFinsupp_add]
                           /-
                             🎉 no goals
                           -/
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10745): was `simp [← ofFinsupp_smul]`.
                      /-
                        R : Type u
                        a b : R
                        m n✝ : Nat
                        inst✝ : Semiring R
                        p q : Polynomial R
                        n : Nat
                        r x : R
                        ⊢ Eq ({ toFun := fun t => { toFinsupp := Finsupp.single n t }, map_add' := ⋯ } …
                      -/
  map_smul' r x := by simp; rw [← ofFinsupp_smul, smul_single']
                            /-
                              🎉 no goals
                            -/


@[simp]
theorem toFinsupp_monomial (n : ℕ) (r : R) : (monomial n r).toFinsupp = Finsupp.single n r := by
  /-
    R : Type u
    inst✝ : Semiring R
    n : Nat
    r : R
    ⊢ Eq ((Polynomial.monomial n) r).toFinsupp (Finsupp.single n r)
  -/
  simp [monomial]
  /-
    🎉 no goals
  -/


@[simp]
theorem ofFinsupp_single (n : ℕ) (r : R) : (⟨Finsupp.single n r⟩ : R[X]) = monomial n r := by
  /-
    R : Type u
    inst✝ : Semiring R
    n : Nat
    r : R
    ⊢ Eq { toFinsupp := Finsupp.single n r } ((Polynomial.monomial n) r)
  -/
  simp [monomial]
  /-
    🎉 no goals
  -/


@[simp]
theorem monomial_zero_right (n : ℕ) : monomial n (0 : R) = 0 :=
  (monomial n).map_zero

-- This is not a `simp` lemma as `monomial_zero_left` is more general.

theorem monomial_zero_one : monomial 0 (1 : R) = 1 :=
  rfl

-- TODO: can't we just delete this one?

theorem monomial_add (n : ℕ) (r s : R) : monomial n (r + s) = monomial n r + monomial n s :=
  (monomial n).map_add _ _


theorem monomial_mul_monomial (n m : ℕ) (r s : R) :
    monomial n r * monomial m s = monomial (n + m) (r * s) :=
  toFinsupp_injective <| by
    /-
      R : Type u
      inst✝ : Semiring R
      n m : Nat
      r s : R
      ⊢ Eq (HMul.hMul ((Polynomial.monomial n) r) ((Polynomial.monomial m) s)).toFin …
    -/
    simp only [toFinsupp_monomial, toFinsupp_mul, AddMonoidAlgebra.single_mul_single]
    /-
      🎉 no goals
    -/


@[simp]
theorem monomial_pow (n : ℕ) (r : R) (k : ℕ) : monomial n r ^ k = monomial (n * k) (r ^ k) := by
  induction k with
  | zero => simp [pow_zero, monomial_zero_one]
  | succ k ih => simp [pow_succ, ih, monomial_mul_monomial, mul_add, add_comm]


theorem smul_monomial {S} [SMulZeroClass S R] (a : S) (n : ℕ) (b : R) :
    a • monomial n b = monomial n (a • b) :=
                            /-
                              R : Type u
                              inst✝¹ : Semiring R
                              S : Type u_1
                              inst✝ : SMulZeroClass S R
                              a : S
                              n : Nat
                              b : R
                              ⊢ Eq (HSMul.hSMul a ((Polynomial.monomial n) b)).toFinsupp ((Polynomial.monomi …
                            -/
  toFinsupp_injective <| by simp; rw [smul_single]
                                  /-
                                    🎉 no goals
                                  -/


theorem monomial_injective (n : ℕ) : Function.Injective (monomial n : R → R[X]) :=
  (toFinsuppIso R).symm.injective.comp (single_injective n)


@[simp]
theorem monomial_eq_zero_iff (t : R) (n : ℕ) : monomial n t = 0 ↔ t = 0 :=
  LinearMap.map_eq_zero_iff _ (Polynomial.monomial_injective n)


theorem monomial_eq_monomial_iff {m n : ℕ} {a b : R} :
    monomial m a = monomial n b ↔ m = n ∧ a = b ∨ a = 0 ∧ b = 0 := by
  /-
    R : Type u
    inst✝ : Semiring R
    m n : Nat
    a b : R
    ⊢ Iff (Eq ((Polynomial.monomial m) a) ((Polynomial.monomial n) b)) (Or (And (E …
  -/
  rw [← toFinsupp_inj, toFinsupp_monomial, toFinsupp_monomial, Finsupp.single_eq_single_iff]
  /-
    🎉 no goals
  -/


theorem support_add : (p + q).support ⊆ p.support ∪ q.support := by
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    ⊢ HasSubset.Subset (HAdd.hAdd p q).support (Union.union p.support q.support)
  -/
  simpa [support] using Finsupp.support_add
  /-
    🎉 no goals
  -/


/-- `C a` is the constant polynomial `a`.
`C` is provided as a ring homomorphism.
-/
def C : R →+* R[X] :=
  { monomial 0 with
                   /-
                     R : Type u
                     a b : R
                     m n : Nat
                     inst✝ : Semiring R
                     p q : Polynomial R
                     ⊢ Eq (__src✝.toFun 1) 1
                   -/
    map_one' := by simp [monomial_zero_one]
                   /-
                     🎉 no goals
                   -/
                   /-
                     R : Type u
                     a b : R
                     m n : Nat
                     inst✝ : Semiring R
                     p q : Polynomial R
                     ⊢ ∀ (x y : R), Eq ({ toFun := __src✝.toFun, map_one' := ⋯ }.toFun (HMul.hMul x …
                   -/
    map_mul' := by simp [monomial_mul_monomial]
                   /-
                     🎉 no goals
                   -/
                    /-
                      R : Type u
                      a b : R
                      m n : Nat
                      inst✝ : Semiring R
                      p q : Polynomial R
                      ⊢ Eq ((↑{ toFun := __src✝.toFun, map_one' := ⋯, map_mul' := ⋯ }).toFun 0) 0
                    -/
    map_zero' := by simp }
                    /-
                      🎉 no goals
                    -/


@[simp]
theorem monomial_zero_left (a : R) : monomial 0 a = C a :=
  rfl


@[simp]
theorem toFinsupp_C (a : R) : (C a).toFinsupp = single 0 a :=
  rfl


                                  /-
                                    R : Type u
                                    inst✝ : Semiring R
                                    ⊢ Eq (Polynomial.C 0) 0
                                  -/
theorem C_0 : C (0 : R) = 0 := by simp
                                  /-
                                    🎉 no goals
                                  -/


theorem C_1 : C (1 : R) = 1 :=
  rfl


theorem C_mul : C (a * b) = C a * C b :=
  C.map_mul a b


theorem C_add : C (a + b) = C a + C b :=
  C.map_add a b


@[simp]
theorem smul_C {S} [SMulZeroClass S R] (s : S) (r : R) : s • C r = C (s • r) :=
  smul_monomial _ _ r


theorem C_pow : C (a ^ n) = C a ^ n :=
  C.map_pow a n


theorem C_eq_natCast (n : ℕ) : C (n : R) = (n : R[X]) :=
  map_natCast C n


@[deprecated (since := "2024-04-17")]
alias C_eq_nat_cast := C_eq_natCast


@[simp]
theorem C_mul_monomial : C a * monomial n b = monomial n (a * b) := by
  /-
    R : Type u
    a b : R
    n : Nat
    inst✝ : Semiring R
    ⊢ Eq (HMul.hMul (Polynomial.C a) ((Polynomial.monomial n) b)) ((Polynomial.mon …
  -/
  simp only [← monomial_zero_left, monomial_mul_monomial, zero_add]
  /-
    🎉 no goals
  -/


@[simp]
theorem monomial_mul_C : monomial n a * C b = monomial n (a * b) := by
  /-
    R : Type u
    a b : R
    n : Nat
    inst✝ : Semiring R
    ⊢ Eq (HMul.hMul ((Polynomial.monomial n) a) (Polynomial.C b)) ((Polynomial.mon …
  -/
  simp only [← monomial_zero_left, monomial_mul_monomial, add_zero]
  /-
    🎉 no goals
  -/


/-- `X` is the polynomial variable (aka indeterminate). -/
def X : R[X] :=
  monomial 1 1


theorem monomial_one_one_eq_X : monomial 1 (1 : R) = X :=
  rfl


theorem monomial_one_right_eq_X_pow (n : ℕ) : monomial n (1 : R) = X ^ n := by
  induction n with
  | zero => simp [monomial_zero_one]
  | succ n ih => rw [pow_succ, ← ih, ← monomial_one_one_eq_X, monomial_mul_monomial, mul_one]


@[simp]
theorem toFinsupp_X : X.toFinsupp = Finsupp.single 1 (1 : R) :=
  rfl


theorem X_ne_C [Nontrivial R] (a : R) : X ≠ C a := by
  /-
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : Nontrivial R
    a : R
    ⊢ Ne Polynomial.X (Polynomial.C a)
  -/
  intro he
  /-
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : Nontrivial R
    a : R
    he : Eq Polynomial.X (Polynomial.C a)
    ⊢ False
  -/
  simpa using monomial_eq_monomial_iff.1 he
  /-
    🎉 no goals
  -/


/-- `X` commutes with everything, even when the coefficients are noncommutative. -/
theorem X_mul : X * p = p * X := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    ⊢ Eq (HMul.hMul Polynomial.X p) (HMul.hMul p Polynomial.X)
  -/
  rcases p with ⟨⟩
  -- Porting note: `ofFinsupp.injEq` is required.
  /-
    case ofFinsupp
    R : Type u
    inst✝ : Semiring R
    toFinsupp✝ : AddMonoidAlgebra R Nat
    ⊢ Eq (HMul.hMul Polynomial.X { toFinsupp := toFinsupp✝ }) (HMul.hMul { toFinsu …
  -/
  simp only [X, ← ofFinsupp_single, ← ofFinsupp_mul, LinearMap.coe_mk, ofFinsupp.injEq]
  -- Porting note: Was `ext`.
  /-
    case ofFinsupp
    R : Type u
    inst✝ : Semiring R
    toFinsupp✝ : AddMonoidAlgebra R Nat
    ⊢ Eq (HMul.hMul (Finsupp.single 1 1) toFinsupp✝) (HMul.hMul toFinsupp✝ (Finsup …
  -/
  refine Finsupp.ext fun _ => ?_
  /-
    case ofFinsupp
    R : Type u
    inst✝ : Semiring R
    toFinsupp✝ : AddMonoidAlgebra R Nat
    x✝ : Nat
    ⊢ Eq ((HMul.hMul (Finsupp.single 1 1) toFinsupp✝) x✝) ((HMul.hMul toFinsupp✝ ( …
  -/
  simp [AddMonoidAlgebra.mul_apply, AddMonoidAlgebra.sum_single_index, add_comm]
  /-
    🎉 no goals
  -/


theorem X_pow_mul {n : ℕ} : X ^ n * p = p * X ^ n := by
  induction n with
  | zero => simp
  | succ n ih =>
    conv_lhs => rw [pow_succ]
    rw [mul_assoc, X_mul, ← mul_assoc, ih, mul_assoc, ← pow_succ]


/-- Prefer putting constants to the left of `X`.

This lemma is the loop-avoiding `simp` version of `Polynomial.X_mul`. -/
@[simp]
theorem X_mul_C (r : R) : X * C r = C r * X :=
  X_mul


/-- Prefer putting constants to the left of `X ^ n`.

This lemma is the loop-avoiding `simp` version of `X_pow_mul`. -/
@[simp]
theorem X_pow_mul_C (r : R) (n : ℕ) : X ^ n * C r = C r * X ^ n :=
  X_pow_mul


theorem X_pow_mul_assoc {n : ℕ} : p * X ^ n * q = p * q * X ^ n := by
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    n : Nat
    ⊢ Eq (HMul.hMul (HMul.hMul p (HPow.hPow Polynomial.X n)) q) (HMul.hMul (HMul.h …
  -/
  rw [mul_assoc, X_pow_mul, ← mul_assoc]
  /-
    🎉 no goals
  -/


/-- Prefer putting constants to the left of `X ^ n`.

This lemma is the loop-avoiding `simp` version of `X_pow_mul_assoc`. -/
@[simp]
theorem X_pow_mul_assoc_C {n : ℕ} (r : R) : p * X ^ n * C r = p * C r * X ^ n :=
  X_pow_mul_assoc


theorem commute_X (p : R[X]) : Commute X p :=
  X_mul


theorem commute_X_pow (p : R[X]) (n : ℕ) : Commute (X ^ n) p :=
  X_pow_mul


@[simp]
theorem monomial_mul_X (n : ℕ) (r : R) : monomial n r * X = monomial (n + 1) r := by
  /-
    R : Type u
    inst✝ : Semiring R
    n : Nat
    r : R
    ⊢ Eq (HMul.hMul ((Polynomial.monomial n) r) Polynomial.X) ((Polynomial.monomia …
  -/
  erw [monomial_mul_monomial, mul_one]
  /-
    🎉 no goals
  -/


@[simp]
theorem monomial_mul_X_pow (n : ℕ) (r : R) (k : ℕ) :
    monomial n r * X ^ k = monomial (n + k) r := by
  induction k with
  | zero => simp
  | succ k ih => simp [ih, pow_succ, ← mul_assoc, add_assoc]


@[simp]
theorem X_mul_monomial (n : ℕ) (r : R) : X * monomial n r = monomial (n + 1) r := by
  /-
    R : Type u
    inst✝ : Semiring R
    n : Nat
    r : R
    ⊢ Eq (HMul.hMul Polynomial.X ((Polynomial.monomial n) r)) ((Polynomial.monomia …
  -/
  rw [X_mul, monomial_mul_X]
  /-
    🎉 no goals
  -/


@[simp]
theorem X_pow_mul_monomial (k n : ℕ) (r : R) : X ^ k * monomial n r = monomial (n + k) r := by
  /-
    R : Type u
    inst✝ : Semiring R
    k n : Nat
    r : R
    ⊢ Eq (HMul.hMul (HPow.hPow Polynomial.X k) ((Polynomial.monomial n) r)) ((Poly …
  -/
  rw [X_pow_mul, monomial_mul_X_pow]
  /-
    🎉 no goals
  -/


/-- `coeff p n` (often denoted `p.coeff n`) is the coefficient of `X^n` in `p`. -/
-- @[simp] -- Porting note: The original generated theorem is same to `coeff_ofFinsupp` and
           --               the new generated theorem is different, so this attribute should be
           --               removed.
def coeff : R[X] → ℕ → R
  | ⟨p⟩ => p


@[simp]
                                                           /-
                                                             R : Type u
                                                             inst✝ : Semiring R
                                                             p : AddMonoidAlgebra R Nat
                                                             ⊢ Eq { toFinsupp := p }.coeff ⇑p
                                                           -/
theorem coeff_ofFinsupp (p) : coeff (⟨p⟩ : R[X]) = p := by rw [coeff]
                                                           /-
                                                             🎉 no goals
                                                           -/


theorem coeff_injective : Injective (coeff : R[X] → ℕ → R) := by
  /-
    R : Type u
    inst✝ : Semiring R
    ⊢ Function.Injective Polynomial.coeff
  -/
  rintro ⟨p⟩ ⟨q⟩
  -- Porting note: `ofFinsupp.injEq` is required.
  /-
    case ofFinsupp.ofFinsupp
    R : Type u
    inst✝ : Semiring R
    p q : AddMonoidAlgebra R Nat
    ⊢ Eq { toFinsupp := p }.coeff { toFinsupp := q }.coeff → Eq { toFinsupp := p } …
  -/
  simp only [coeff, DFunLike.coe_fn_eq, imp_self, ofFinsupp.injEq]
  /-
    🎉 no goals
  -/


@[simp]
theorem coeff_inj : p.coeff = q.coeff ↔ p = q :=
  coeff_injective.eq_iff


                                                                         /-
                                                                           R : Type u
                                                                           inst✝ : Semiring R
                                                                           f : Polynomial R
                                                                           i : Nat
                                                                           ⊢ Eq (f.toFinsupp i) (f.coeff i)
                                                                         -/
theorem toFinsupp_apply (f : R[X]) (i) : f.toFinsupp i = f.coeff i := by cases f; rfl
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


theorem coeff_monomial : coeff (monomial n a) m = if n = m then a else 0 := by
  /-
    R : Type u
    a : R
    m n : Nat
    inst✝ : Semiring R
    ⊢ Eq (((Polynomial.monomial n) a).coeff m) (ite (Eq n m) a 0)
  -/
  simp [coeff, Finsupp.single_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem coeff_monomial_same (n : ℕ) (c : R) : (monomial n c).coeff n = c :=
  Finsupp.single_eq_same


theorem coeff_monomial_of_ne {m n : ℕ} (c : R) (h : n ≠ m) : (monomial n c).coeff m = 0 :=
  Finsupp.single_eq_of_ne h


@[simp]
theorem coeff_zero (n : ℕ) : coeff (0 : R[X]) n = 0 :=
  rfl


theorem coeff_one {n : ℕ} : coeff (1 : R[X]) n = if n = 0 then 1 else 0 := by
  /-
    R : Type u
    inst✝ : Semiring R
    n : Nat
    ⊢ Eq (Polynomial.coeff 1 n) (ite (Eq n 0) 1 0)
  -/
  simp_rw [eq_comm (a := n) (b := 0)]
  /-
    R : Type u
    inst✝ : Semiring R
    n : Nat
    ⊢ Eq (Polynomial.coeff 1 n) (ite (Eq 0 n) 1 0)
  -/
  exact coeff_monomial
  /-
    🎉 no goals
  -/


@[simp]
theorem coeff_one_zero : coeff (1 : R[X]) 0 = 1 := by
  /-
    R : Type u
    inst✝ : Semiring R
    ⊢ Eq (Polynomial.coeff 1 0) 1
  -/
  simp [coeff_one]
  /-
    🎉 no goals
  -/


@[simp]
theorem coeff_X_one : coeff (X : R[X]) 1 = 1 :=
  coeff_monomial


@[simp]
theorem coeff_X_zero : coeff (X : R[X]) 0 = 0 :=
  coeff_monomial


@[simp]
                                                                     /-
                                                                       R : Type u
                                                                       a : R
                                                                       n : Nat
                                                                       inst✝ : Semiring R
                                                                       ⊢ Eq (((Polynomial.monomial (HAdd.hAdd n 1)) a).coeff 0) 0
                                                                     -/
theorem coeff_monomial_succ : coeff (monomial (n + 1) a) 0 = 0 := by simp [coeff_monomial]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


theorem coeff_X : coeff (X : R[X]) n = if 1 = n then 1 else 0 :=
  coeff_monomial


theorem coeff_X_of_ne_one {n : ℕ} (hn : n ≠ 1) : coeff (X : R[X]) n = 0 := by
  /-
    R : Type u
    inst✝ : Semiring R
    n : Nat
    hn : Ne n 1
    ⊢ Eq (Polynomial.X.coeff n) 0
  -/
  rw [coeff_X, if_neg hn.symm]
  /-
    🎉 no goals
  -/


@[simp]
theorem mem_support_iff : n ∈ p.support ↔ p.coeff n ≠ 0 := by
  /-
    R : Type u
    n : Nat
    inst✝ : Semiring R
    p : Polynomial R
    ⊢ Iff (Membership.mem p.support n) (Ne (p.coeff n) 0)
  -/
  rcases p with ⟨⟩
  /-
    case ofFinsupp
    R : Type u
    n : Nat
    inst✝ : Semiring R
    toFinsupp✝ : AddMonoidAlgebra R Nat
    ⊢ Iff (Membership.mem { toFinsupp := toFinsupp✝ }.support n) (Ne ({ toFinsupp  …
  -/
  simp
  /-
    🎉 no goals
  -/


                                                                  /-
                                                                    R : Type u
                                                                    n : Nat
                                                                    inst✝ : Semiring R
                                                                    p : Polynomial R
                                                                    ⊢ Iff (Not (Membership.mem p.support n)) (Eq (p.coeff n) 0)
                                                                  -/
theorem not_mem_support_iff : n ∉ p.support ↔ p.coeff n = 0 := by simp
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


theorem coeff_C : coeff (C a) n = ite (n = 0) a 0 := by
  /-
    R : Type u
    a : R
    n : Nat
    inst✝ : Semiring R
    ⊢ Eq ((Polynomial.C a).coeff n) (ite (Eq n 0) a 0)
  -/
  convert coeff_monomial (a := a) (m := n) (n := 0) using 2
  /-
    case h.e'_3.h₁.a
    R : Type u
    a : R
    n : Nat
    inst✝ : Semiring R
    ⊢ Iff (Eq n 0) (Eq 0 n)
  -/
  simp [eq_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem coeff_C_zero : coeff (C a) 0 = a :=
  coeff_monomial


                                                              /-
                                                                R : Type u
                                                                a : R
                                                                n : Nat
                                                                inst✝ : Semiring R
                                                                h : Ne n 0
                                                                ⊢ Eq ((Polynomial.C a).coeff n) 0
                                                              -/
theorem coeff_C_ne_zero (h : n ≠ 0) : (C a).coeff n = 0 := by rw [coeff_C, if_neg h]
                                                              /-
                                                                🎉 no goals
                                                              -/


@[simp]
                                                                   /-
                                                                     R : Type u
                                                                     inst✝ : Semiring R
                                                                     r : R
                                                                     n : Nat
                                                                     ⊢ Eq ((Polynomial.C r).coeff (HAdd.hAdd n 1)) 0
                                                                   -/
lemma coeff_C_succ {r : R} {n : ℕ} : coeff (C r) (n + 1) = 0 := by simp [coeff_C]
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[simp]
theorem coeff_natCast_ite : (Nat.cast m : R[X]).coeff n = ite (n = 0) m 0 := by
  /-
    R : Type u
    m n : Nat
    inst✝ : Semiring R
    ⊢ Eq ((↑m).coeff n) ↑(ite (Eq n 0) m 0)
  -/
  simp only [← C_eq_natCast, coeff_C, Nat.cast_ite, Nat.cast_zero]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias coeff_nat_cast_ite := coeff_natCast_ite


@[simp]
theorem coeff_ofNat_zero (a : ℕ) [a.AtLeastTwo] :
    coeff (ofNat(a) : R[X]) 0 = ofNat(a) :=
  coeff_monomial


@[simp]
theorem coeff_ofNat_succ (a n : ℕ) [h : a.AtLeastTwo] :
    coeff (ofNat(a) : R[X]) (n + 1) = 0 := by
  /-
    R : Type u
    inst✝ : Semiring R
    a n : Nat
    h : a.AtLeastTwo
    ⊢ Eq ((OfNat.ofNat a).coeff (HAdd.hAdd n 1)) 0
  -/
  rw [← Nat.cast_ofNat]
  /-
    R : Type u
    inst✝ : Semiring R
    a n : Nat
    h : a.AtLeastTwo
    ⊢ Eq ((↑(OfNat.ofNat a)).coeff (HAdd.hAdd n 1)) 0
  -/
  simp [-Nat.cast_ofNat]
  /-
    🎉 no goals
  -/


theorem C_mul_X_pow_eq_monomial : ∀ {n : ℕ}, C a * X ^ n = monomial n a
  | 0 => mul_one _
  | n + 1 => by
    /-
      R : Type u
      a : R
      inst✝ : Semiring R
      n : Nat
      ⊢ Eq (HMul.hMul (Polynomial.C a) (HPow.hPow Polynomial.X (HAdd.hAdd n 1))) ((P …
    -/
    rw [pow_succ, ← mul_assoc, C_mul_X_pow_eq_monomial, X, monomial_mul_monomial, mul_one]
    /-
      🎉 no goals
    -/


@[simp high]
theorem toFinsupp_C_mul_X_pow (a : R) (n : ℕ) :
    Polynomial.toFinsupp (C a * X ^ n) = Finsupp.single n a := by
  /-
    R : Type u
    inst✝ : Semiring R
    a : R
    n : Nat
    ⊢ Eq (HMul.hMul (Polynomial.C a) (HPow.hPow Polynomial.X n)).toFinsupp (Finsup …
  -/
  rw [C_mul_X_pow_eq_monomial, toFinsupp_monomial]
  /-
    🎉 no goals
  -/


                                                           /-
                                                             R : Type u
                                                             a : R
                                                             inst✝ : Semiring R
                                                             ⊢ Eq (HMul.hMul (Polynomial.C a) Polynomial.X) ((Polynomial.monomial 1) a)
                                                           -/
theorem C_mul_X_eq_monomial : C a * X = monomial 1 a := by rw [← C_mul_X_pow_eq_monomial, pow_one]
                                                           /-
                                                             🎉 no goals
                                                           -/


@[simp high]
theorem toFinsupp_C_mul_X (a : R) : Polynomial.toFinsupp (C a * X) = Finsupp.single 1 a := by
  /-
    R : Type u
    inst✝ : Semiring R
    a : R
    ⊢ Eq (HMul.hMul (Polynomial.C a) Polynomial.X).toFinsupp (Finsupp.single 1 a)
  -/
  rw [C_mul_X_eq_monomial, toFinsupp_monomial]
  /-
    🎉 no goals
  -/


theorem C_injective : Injective (C : R → R[X]) :=
  monomial_injective 0


@[simp]
theorem C_inj : C a = C b ↔ a = b :=
  C_injective.eq_iff


@[simp]
theorem C_eq_zero : C a = 0 ↔ a = 0 :=
  C_injective.eq_iff' (map_zero C)


theorem C_ne_zero : C a ≠ 0 ↔ a ≠ 0 :=
  C_eq_zero.not


theorem subsingleton_iff_subsingleton : Subsingleton R[X] ↔ Subsingleton R :=
  ⟨@Injective.subsingleton _ _ _ C_injective, by
    /-
      R : Type u
      inst✝ : Semiring R
      ⊢ Subsingleton R → Subsingleton (Polynomial R)
    -/
    intro
    /-
      R : Type u
      inst✝ : Semiring R
      a✝ : Subsingleton R
      ⊢ Subsingleton (Polynomial R)
    -/
    infer_instance⟩
    /-
      🎉 no goals
    -/


theorem Nontrivial.of_polynomial_ne (h : p ≠ q) : Nontrivial R :=
  (subsingleton_or_nontrivial R).resolve_left fun _hI => h <| Subsingleton.elim _ _


theorem forall_eq_iff_forall_eq : (∀ f g : R[X], f = g) ↔ ∀ a b : R, a = b := by
  /-
    R : Type u
    inst✝ : Semiring R
    ⊢ Iff (∀ (f g : Polynomial R), Eq f g) (∀ (a b : R), Eq a b)
  -/
  simpa only [← subsingleton_iff] using subsingleton_iff_subsingleton
  /-
    🎉 no goals
  -/


theorem ext_iff {p q : R[X]} : p = q ↔ ∀ n, coeff p n = coeff q n := by
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    ⊢ Iff (Eq p q) (∀ (n : Nat), Eq (p.coeff n) (q.coeff n))
  -/
  rcases p with ⟨f : ℕ →₀ R⟩
  /-
    case ofFinsupp
    R : Type u
    inst✝ : Semiring R
    q : Polynomial R
    f : Finsupp Nat R
    ⊢ Iff (Eq { toFinsupp := f } q) (∀ (n : Nat), Eq ({ toFinsupp := f }.coeff n)  …
  -/
  rcases q with ⟨g : ℕ →₀ R⟩
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10745): was `simp [coeff, DFunLike.ext_iff]`
  /-
    case ofFinsupp.ofFinsupp
    R : Type u
    inst✝ : Semiring R
    f g : Finsupp Nat R
    ⊢ Iff (Eq { toFinsupp := f } { toFinsupp := g }) (∀ (n : Nat), Eq ({ toFinsupp …
  -/
  simpa [coeff] using DFunLike.ext_iff (f := f) (g := g)
  /-
    🎉 no goals
  -/


@[ext]
theorem ext {p q : R[X]} : (∀ n, coeff p n = coeff q n) → p = q :=
  ext_iff.2


/-- Monomials generate the additive monoid of polynomials. -/
theorem addSubmonoid_closure_setOf_eq_monomial :
    AddSubmonoid.closure { p : R[X] | ∃ n a, p = monomial n a } = ⊤ := by
  /-
    R : Type u
    inst✝ : Semiring R
    ⊢ Eq (AddSubmonoid.closure (setOf fun p => Exists fun n => Exists fun a => Eq  …
  -/
  apply top_unique
  rw [← AddSubmonoid.map_equiv_top (toFinsuppIso R).symm.toAddEquiv, ←
    Finsupp.add_closure_setOf_eq_single, AddMonoidHom.map_mclosure]
  /-
    case h
    R : Type u
    inst✝ : Semiring R
    ⊢ LE.le (AddSubmonoid.closure (Set.image (⇑(Polynomial.toFinsuppIso R).symm.to …
  -/
  refine AddSubmonoid.closure_mono (Set.image_subset_iff.2 ?_)
  /-
    case h
    R : Type u
    inst✝ : Semiring R
    ⊢ HasSubset.Subset (setOf fun f => Exists fun a => Exists fun b => Eq f (Finsu …
  -/
  rintro _ ⟨n, a, rfl⟩
  /-
    case h.intro.intro
    R : Type u
    inst✝ : Semiring R
    n : Nat
    a : R
    ⊢ Membership.mem (Set.preimage (⇑(Polynomial.toFinsuppIso R).symm.toAddEquiv)  …
  -/
  exact ⟨n, a, Polynomial.ofFinsupp_single _ _⟩
  /-
    🎉 no goals
  -/


theorem addHom_ext {M : Type*} [AddMonoid M] {f g : R[X] →+ M}
    (h : ∀ n a, f (monomial n a) = g (monomial n a)) : f = g :=
  AddMonoidHom.eq_of_eqOn_denseM addSubmonoid_closure_setOf_eq_monomial <| by
    /-
      R : Type u
      inst✝¹ : Semiring R
      M : Type u_1
      inst✝ : AddMonoid M
      f g : AddMonoidHom (Polynomial R) M
      h : ∀ (n : Nat) (a : R), Eq (f ((Polynomial.monomial n) a)) (g ((Polynomial.mo …
      ⊢ Set.EqOn (⇑f) (⇑g) (setOf fun p => Exists fun n => Exists fun a => Eq p ((Po …
    -/
    rintro p ⟨n, a, rfl⟩
    /-
      case intro.intro
      R : Type u
      inst✝¹ : Semiring R
      M : Type u_1
      inst✝ : AddMonoid M
      f g : AddMonoidHom (Polynomial R) M
      h : ∀ (n : Nat) (a : R), Eq (f ((Polynomial.monomial n) a)) (g ((Polynomial.mo …
      n : Nat
      a : R
      ⊢ Eq (f ((Polynomial.monomial n) a)) (g ((Polynomial.monomial n) a))
    -/
    exact h n a
    /-
      🎉 no goals
    -/


@[ext high]
theorem addHom_ext' {M : Type*} [AddMonoid M] {f g : R[X] →+ M}
    (h : ∀ n, f.comp (monomial n).toAddMonoidHom = g.comp (monomial n).toAddMonoidHom) : f = g :=
  addHom_ext fun n => DFunLike.congr_fun (h n)


@[ext high]
theorem lhom_ext' {M : Type*} [AddCommMonoid M] [Module R M] {f g : R[X] →ₗ[R] M}
    (h : ∀ n, f.comp (monomial n) = g.comp (monomial n)) : f = g :=
  LinearMap.toAddMonoidHom_injective <| addHom_ext fun n => LinearMap.congr_fun (h n)

-- this has the same content as the subsingleton

theorem eq_zero_of_eq_zero (h : (0 : R) = (1 : R)) (p : R[X]) : p = 0 := by
  /-
    R : Type u
    inst✝ : Semiring R
    h : Eq 0 1
    p : Polynomial R
    ⊢ Eq p 0
  -/
  rw [← one_smul R p, ← h, zero_smul]
  /-
    🎉 no goals
  -/


theorem support_monomial (n) {a : R} (H : a ≠ 0) : (monomial n a).support = singleton n := by
  /-
    R : Type u
    inst✝ : Semiring R
    n : Nat
    a : R
    H : Ne a 0
    ⊢ Eq ((Polynomial.monomial n) a).support (Singleton.singleton n)
  -/
  rw [← ofFinsupp_single, support]; exact Finsupp.support_single_ne_zero _ H
                                    /-
                                      🎉 no goals
                                    -/


theorem support_monomial' (n) (a : R) : (monomial n a).support ⊆ singleton n := by
  /-
    R : Type u
    inst✝ : Semiring R
    n : Nat
    a : R
    ⊢ HasSubset.Subset ((Polynomial.monomial n) a).support (Singleton.singleton n)
  -/
  rw [← ofFinsupp_single, support]
  /-
    R : Type u
    inst✝ : Semiring R
    n : Nat
    a : R
    ⊢ HasSubset.Subset (Finsupp.single n a).support (Singleton.singleton n)
  -/
  exact Finsupp.support_single_subset
  /-
    🎉 no goals
  -/


theorem support_C {a : R} (h : a ≠ 0) : (C a).support = singleton 0 :=
  support_monomial 0 h


theorem support_C_subset (a : R) : (C a).support ⊆ singleton 0 :=
  support_monomial' 0 a


theorem support_C_mul_X {c : R} (h : c ≠ 0) : Polynomial.support (C c * X) = singleton 1 := by
  /-
    R : Type u
    inst✝ : Semiring R
    c : R
    h : Ne c 0
    ⊢ Eq (HMul.hMul (Polynomial.C c) Polynomial.X).support (Singleton.singleton 1)
  -/
  rw [C_mul_X_eq_monomial, support_monomial 1 h]
  /-
    🎉 no goals
  -/


theorem support_C_mul_X' (c : R) : Polynomial.support (C c * X) ⊆ singleton 1 := by
  /-
    R : Type u
    inst✝ : Semiring R
    c : R
    ⊢ HasSubset.Subset (HMul.hMul (Polynomial.C c) Polynomial.X).support (Singleto …
  -/
  simpa only [C_mul_X_eq_monomial] using support_monomial' 1 c
  /-
    🎉 no goals
  -/


theorem support_C_mul_X_pow (n : ℕ) {c : R} (h : c ≠ 0) :
    Polynomial.support (C c * X ^ n) = singleton n := by
  /-
    R : Type u
    inst✝ : Semiring R
    n : Nat
    c : R
    h : Ne c 0
    ⊢ Eq (HMul.hMul (Polynomial.C c) (HPow.hPow Polynomial.X n)).support (Singleto …
  -/
  rw [C_mul_X_pow_eq_monomial, support_monomial n h]
  /-
    🎉 no goals
  -/


theorem support_C_mul_X_pow' (n : ℕ) (c : R) : Polynomial.support (C c * X ^ n) ⊆ singleton n := by
  /-
    R : Type u
    inst✝ : Semiring R
    n : Nat
    c : R
    ⊢ HasSubset.Subset (HMul.hMul (Polynomial.C c) (HPow.hPow Polynomial.X n)).sup …
  -/
  simpa only [C_mul_X_pow_eq_monomial] using support_monomial' n c
  /-
    🎉 no goals
  -/


theorem support_binomial' (k m : ℕ) (x y : R) :
    Polynomial.support (C x * X ^ k + C y * X ^ m) ⊆ {k, m} :=
  support_add.trans
    (union_subset
      ((support_C_mul_X_pow' k x).trans (singleton_subset_iff.mpr (mem_insert_self k {m})))
      ((support_C_mul_X_pow' m y).trans
        (singleton_subset_iff.mpr (mem_insert_of_mem (mem_singleton_self m)))))


theorem support_trinomial' (k m n : ℕ) (x y z : R) :
    Polynomial.support (C x * X ^ k + C y * X ^ m + C z * X ^ n) ⊆ {k, m, n} :=
  support_add.trans
    (union_subset
      (support_add.trans
        (union_subset
          ((support_C_mul_X_pow' k x).trans (singleton_subset_iff.mpr (mem_insert_self k {m, n})))
          ((support_C_mul_X_pow' m y).trans
            (singleton_subset_iff.mpr (mem_insert_of_mem (mem_insert_self m {n}))))))
      ((support_C_mul_X_pow' n z).trans
        (singleton_subset_iff.mpr (mem_insert_of_mem (mem_insert_of_mem (mem_singleton_self n))))))


theorem X_pow_eq_monomial (n) : X ^ n = monomial n (1 : R) := by
  induction n with
  | zero => rw [pow_zero, monomial_zero_one]
  | succ n hn => rw [pow_succ, hn, X, monomial_mul_monomial, one_mul]


@[simp high]
theorem toFinsupp_X_pow (n : ℕ) : (X ^ n).toFinsupp = Finsupp.single n (1 : R) := by
  /-
    R : Type u
    inst✝ : Semiring R
    n : Nat
    ⊢ Eq (HPow.hPow Polynomial.X n).toFinsupp (Finsupp.single n 1)
  -/
  rw [X_pow_eq_monomial, toFinsupp_monomial]
  /-
    🎉 no goals
  -/


theorem smul_X_eq_monomial {n} : a • X ^ n = monomial n (a : R) := by
  /-
    R : Type u
    a : R
    inst✝ : Semiring R
    n : Nat
    ⊢ Eq (HSMul.hSMul a (HPow.hPow Polynomial.X n)) ((Polynomial.monomial n) a)
  -/
  rw [X_pow_eq_monomial, smul_monomial, smul_eq_mul, mul_one]
  /-
    🎉 no goals
  -/


theorem support_X_pow (H : ¬(1 : R) = 0) (n : ℕ) : (X ^ n : R[X]).support = singleton n := by
  /-
    R : Type u
    inst✝ : Semiring R
    H : Not (Eq 1 0)
    n : Nat
    ⊢ Eq (HPow.hPow Polynomial.X n).support (Singleton.singleton n)
  -/
  convert support_monomial n H
  /-
    case h.e'_2.h.e'_3
    R : Type u
    inst✝ : Semiring R
    H : Not (Eq 1 0)
    n : Nat
    ⊢ Eq (HPow.hPow Polynomial.X n) ((Polynomial.monomial n) 1)
  -/
  exact X_pow_eq_monomial n
  /-
    🎉 no goals
  -/


theorem support_X_empty (H : (1 : R) = 0) : (X : R[X]).support = ∅ := by
  /-
    R : Type u
    inst✝ : Semiring R
    H : Eq 1 0
    ⊢ Eq Polynomial.X.support EmptyCollection.emptyCollection
  -/
  rw [X, H, monomial_zero_right, support_zero]
  /-
    🎉 no goals
  -/


theorem support_X (H : ¬(1 : R) = 0) : (X : R[X]).support = singleton 1 := by
  /-
    R : Type u
    inst✝ : Semiring R
    H : Not (Eq 1 0)
    ⊢ Eq Polynomial.X.support (Singleton.singleton 1)
  -/
  rw [← pow_one X, support_X_pow H 1]
  /-
    🎉 no goals
  -/


theorem monomial_left_inj {a : R} (ha : a ≠ 0) {i j : ℕ} :
    monomial i a = monomial j a ↔ i = j := by
  /-
    R : Type u
    inst✝ : Semiring R
    a : R
    ha : Ne a 0
    i j : Nat
    ⊢ Iff (Eq ((Polynomial.monomial i) a) ((Polynomial.monomial j) a)) (Eq i j)
  -/
  simp only [← ofFinsupp_single, ofFinsupp.injEq, Finsupp.single_left_inj ha]
  /-
    🎉 no goals
  -/


theorem binomial_eq_binomial {k l m n : ℕ} {u v : R} (hu : u ≠ 0) (hv : v ≠ 0) :
    C u * X ^ k + C v * X ^ l = C u * X ^ m + C v * X ^ n ↔
      k = m ∧ l = n ∨ u = v ∧ k = n ∧ l = m ∨ u + v = 0 ∧ k = l ∧ m = n := by
  /-
    R : Type u
    inst✝ : Semiring R
    k l m n : Nat
    u v : R
    hu : Ne u 0
    hv : Ne v 0
    ⊢ Iff (Eq (HAdd.hAdd (HMul.hMul (Polynomial.C u) (HPow.hPow Polynomial.X k)) ( …
  -/
  simp_rw [C_mul_X_pow_eq_monomial, ← toFinsupp_inj, toFinsupp_add, toFinsupp_monomial]
  /-
    R : Type u
    inst✝ : Semiring R
    k l m n : Nat
    u v : R
    hu : Ne u 0
    hv : Ne v 0
    ⊢ Iff (Eq (HAdd.hAdd (Finsupp.single k u) (Finsupp.single l v)) (HAdd.hAdd (Fi …
  -/
  exact Finsupp.single_add_single_eq_single_add_single hu hv
  /-
    🎉 no goals
  -/


theorem natCast_mul (n : ℕ) (p : R[X]) : (n : R[X]) * p = n • p :=
  (nsmul_eq_mul _ _).symm


@[deprecated (since := "2024-04-17")]
alias nat_cast_mul := natCast_mul


/-- Summing the values of a function applied to the coefficients of a polynomial -/
def sum {S : Type*} [AddCommMonoid S] (p : R[X]) (f : ℕ → R → S) : S :=
  ∑ n ∈ p.support, f n (p.coeff n)


theorem sum_def {S : Type*} [AddCommMonoid S] (p : R[X]) (f : ℕ → R → S) :
    p.sum f = ∑ n ∈ p.support, f n (p.coeff n) :=
  rfl


theorem sum_eq_of_subset {S : Type*} [AddCommMonoid S] {p : R[X]} (f : ℕ → R → S)
    (hf : ∀ i, f i 0 = 0) {s : Finset ℕ} (hs : p.support ⊆ s) :
    p.sum f = ∑ n ∈ s, f n (p.coeff n) :=
  Finsupp.sum_of_support_subset _ hs f (fun i _ ↦ hf i)


/-- Expressing the product of two polynomials as a double sum. -/
theorem mul_eq_sum_sum :
    p * q = ∑ i ∈ p.support, q.sum fun j a => (monomial (i + j)) (p.coeff i * a) := by
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    ⊢ Eq (HMul.hMul p q) (p.support.sum fun i => q.sum fun j a => (Polynomial.mono …
  -/
  apply toFinsupp_injective
  /-
    case a
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    ⊢ Eq (HMul.hMul p q).toFinsupp (p.support.sum fun i => q.sum fun j a => (Polyn …
  -/
  rcases p with ⟨⟩; rcases q with ⟨⟩
  simp_rw [sum, coeff, toFinsupp_sum, support, toFinsupp_mul, toFinsupp_monomial,
    AddMonoidAlgebra.mul_def, Finsupp.sum]


@[simp]
theorem sum_zero_index {S : Type*} [AddCommMonoid S] (f : ℕ → R → S) : (0 : R[X]).sum f = 0 := by
  /-
    R : Type u
    inst✝¹ : Semiring R
    S : Type u_1
    inst✝ : AddCommMonoid S
    f : Nat → R → S
    ⊢ Eq (Polynomial.sum 0 f) 0
  -/
  simp [sum]
  /-
    🎉 no goals
  -/


@[simp]
theorem sum_monomial_index {S : Type*} [AddCommMonoid S] {n : ℕ} (a : R) (f : ℕ → R → S)
    (hf : f n 0 = 0) : (monomial n a : R[X]).sum f = f n a :=
  Finsupp.sum_single_index hf


@[simp]
theorem sum_C_index {a} {β} [AddCommMonoid β] {f : ℕ → R → β} (h : f 0 0 = 0) :
    (C a).sum f = f 0 a :=
  sum_monomial_index a f h

-- the assumption `hf` is only necessary when the ring is trivial

@[simp]
theorem sum_X_index {S : Type*} [AddCommMonoid S] {f : ℕ → R → S} (hf : f 1 0 = 0) :
    (X : R[X]).sum f = f 1 1 :=
  sum_monomial_index 1 f hf


theorem sum_add_index {S : Type*} [AddCommMonoid S] (p q : R[X]) (f : ℕ → R → S)
    (hf : ∀ i, f i 0 = 0) (h_add : ∀ a b₁ b₂, f a (b₁ + b₂) = f a b₁ + f a b₂) :
    (p + q).sum f = p.sum f + q.sum f := by
  /-
    R : Type u
    inst✝¹ : Semiring R
    S : Type u_1
    inst✝ : AddCommMonoid S
    p q : Polynomial R
    f : Nat → R → S
    hf : ∀ (i : Nat), Eq (f i 0) 0
    h_add : ∀ (a : Nat) (b₁ b₂ : R), Eq (f a (HAdd.hAdd b₁ b₂)) (HAdd.hAdd (f a b₁ …
    ⊢ Eq ((HAdd.hAdd p q).sum f) (HAdd.hAdd (p.sum f) (q.sum f))
  -/
  rw [show p + q = ⟨p.toFinsupp + q.toFinsupp⟩ from add_def p q]
  /-
    R : Type u
    inst✝¹ : Semiring R
    S : Type u_1
    inst✝ : AddCommMonoid S
    p q : Polynomial R
    f : Nat → R → S
    hf : ∀ (i : Nat), Eq (f i 0) 0
    h_add : ∀ (a : Nat) (b₁ b₂ : R), Eq (f a (HAdd.hAdd b₁ b₂)) (HAdd.hAdd (f a b₁ …
    ⊢ Eq ({ toFinsupp := HAdd.hAdd p.toFinsupp q.toFinsupp }.sum f) (HAdd.hAdd (p. …
  -/
  exact Finsupp.sum_add_index (fun i _ ↦ hf i) (fun a _ b₁ b₂ ↦ h_add a b₁ b₂)
  /-
    🎉 no goals
  -/


theorem sum_add' {S : Type*} [AddCommMonoid S] (p : R[X]) (f g : ℕ → R → S) :
                                            /-
                                              R : Type u
                                              inst✝¹ : Semiring R
                                              S : Type u_1
                                              inst✝ : AddCommMonoid S
                                              p : Polynomial R
                                              f g : Nat → R → S
                                              ⊢ Eq (p.sum (HAdd.hAdd f g)) (HAdd.hAdd (p.sum f) (p.sum g))
                                            -/
    p.sum (f + g) = p.sum f + p.sum g := by simp [sum_def, Finset.sum_add_distrib]
                                            /-
                                              🎉 no goals
                                            -/


theorem sum_add {S : Type*} [AddCommMonoid S] (p : R[X]) (f g : ℕ → R → S) :
    (p.sum fun n x => f n x + g n x) = p.sum f + p.sum g :=
  sum_add' _ _ _


theorem sum_smul_index {S : Type*} [AddCommMonoid S] (p : R[X]) (b : R) (f : ℕ → R → S)
    (hf : ∀ i, f i 0 = 0) : (b • p).sum f = p.sum fun n a => f n (b * a) :=
  Finsupp.sum_smul_index hf


theorem sum_smul_index' {S T : Type*} [DistribSMul T R] [AddCommMonoid S] (p : R[X]) (b : T)
    (f : ℕ → R → S) (hf : ∀ i, f i 0 = 0) : (b • p).sum f = p.sum fun n a => f n (b • a) :=
  Finsupp.sum_smul_index' hf


protected theorem smul_sum {S T : Type*} [AddCommMonoid S] [DistribSMul T S] (p : R[X]) (b : T)
    (f : ℕ → R → S) : b • p.sum f = p.sum fun n a => b • f n a :=
  Finsupp.smul_sum


@[simp]
theorem sum_monomial_eq : ∀ p : R[X], (p.sum fun n a => monomial n a) = p
  | ⟨_p⟩ => (ofFinsupp_sum _ _).symm.trans (congr_arg _ <| Finsupp.sum_single _)


theorem sum_C_mul_X_pow_eq (p : R[X]) : (p.sum fun n a => C a * X ^ n) = p := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    ⊢ Eq (p.sum fun n a => HMul.hMul (Polynomial.C a) (HPow.hPow Polynomial.X n)) p
  -/
  simp_rw [C_mul_X_pow_eq_monomial, sum_monomial_eq]
  /-
    🎉 no goals
  -/


@[elab_as_elim]
protected theorem induction_on {M : R[X] → Prop} (p : R[X]) (h_C : ∀ a, M (C a))
    (h_add : ∀ p q, M p → M q → M (p + q))
    (h_monomial : ∀ (n : ℕ) (a : R), M (C a * X ^ n) → M (C a * X ^ (n + 1))) : M p := by
  have A : ∀ {n : ℕ} {a}, M (C a * X ^ n) := by
    intro n a
    induction n with
    | zero => rw [pow_zero, mul_one]; exact h_C a
    | succ n ih => exact h_monomial _ _ ih
  have B : ∀ s : Finset ℕ, M (s.sum fun n : ℕ => C (p.coeff n) * X ^ n) := by
    apply Finset.induction
    · convert h_C 0
      exact C_0.symm
    · intro n s ns ih
      rw [sum_insert ns]
      exact h_add _ _ A ih
  /-
    R : Type u
    inst✝ : Semiring R
    M : Polynomial R → Prop
    p : Polynomial R
    h_C : ∀ (a : R), M (Polynomial.C a)
    h_add : ∀ (p q : Polynomial R), M p → M q → M (HAdd.hAdd p q)
    h_monomial : ∀ (n : Nat) (a : R), M (HMul.hMul (Polynomial.C a) (HPow.hPow Pol …
    A : ∀ {n : Nat} {a : R}, M (HMul.hMul (Polynomial.C a) (HPow.hPow Polynomial.X …
    B : ∀ (s : Finset Nat), M (s.sum fun n => HMul.hMul (Polynomial.C (p.coeff n)) …
    ⊢ M p
  -/
  rw [← sum_C_mul_X_pow_eq p, Polynomial.sum]
  /-
    R : Type u
    inst✝ : Semiring R
    M : Polynomial R → Prop
    p : Polynomial R
    h_C : ∀ (a : R), M (Polynomial.C a)
    h_add : ∀ (p q : Polynomial R), M p → M q → M (HAdd.hAdd p q)
    h_monomial : ∀ (n : Nat) (a : R), M (HMul.hMul (Polynomial.C a) (HPow.hPow Pol …
    A : ∀ {n : Nat} {a : R}, M (HMul.hMul (Polynomial.C a) (HPow.hPow Polynomial.X …
    B : ∀ (s : Finset Nat), M (s.sum fun n => HMul.hMul (Polynomial.C (p.coeff n)) …
    ⊢ M (p.support.sum fun n => HMul.hMul (Polynomial.C (p.coeff n)) (HPow.hPow Po …
  -/
  exact B (support p)
  /-
    🎉 no goals
  -/


/-- To prove something about polynomials,
it suffices to show the condition is closed under taking sums,
and it holds for monomials.
-/
@[elab_as_elim]
protected theorem induction_on' {M : R[X] → Prop} (p : R[X]) (h_add : ∀ p q, M p → M q → M (p + q))
    (h_monomial : ∀ (n : ℕ) (a : R), M (monomial n a)) : M p :=
  Polynomial.induction_on p (h_monomial 0) h_add fun n a _h =>
       /-
         R : Type u
         inst✝ : Semiring R
         M : Polynomial R → Prop
         p : Polynomial R
         h_add : ∀ (p q : Polynomial R), M p → M q → M (HAdd.hAdd p q)
         h_monomial : ∀ (n : Nat) (a : R), M ((Polynomial.monomial n) a)
         n : Nat
         a : R
         _h : M (HMul.hMul (Polynomial.C a) (HPow.hPow Polynomial.X n))
         ⊢ M (HMul.hMul (Polynomial.C a) (HPow.hPow Polynomial.X (HAdd.hAdd n 1)))
       -/
    by rw [C_mul_X_pow_eq_monomial]; exact h_monomial _ _
                                     /-
                                       🎉 no goals
                                     -/


/-- `erase p n` is the polynomial `p` in which the `X^n` term has been erased. -/
irreducible_def erase (n : ℕ) : R[X] → R[X]
  | ⟨p⟩ => ⟨p.erase n⟩


@[simp]
theorem toFinsupp_erase (p : R[X]) (n : ℕ) : toFinsupp (p.erase n) = p.toFinsupp.erase n := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    n : Nat
    ⊢ Eq (Polynomial.erase n p).toFinsupp (Finsupp.erase n p.toFinsupp)
  -/
  rcases p with ⟨⟩
  /-
    case ofFinsupp
    R : Type u
    inst✝ : Semiring R
    n : Nat
    toFinsupp✝ : AddMonoidAlgebra R Nat
    ⊢ Eq (Polynomial.erase n { toFinsupp := toFinsupp✝ }).toFinsupp (Finsupp.erase …
  -/
  simp only [erase_def]
  /-
    🎉 no goals
  -/


@[simp]
theorem ofFinsupp_erase (p : R[ℕ]) (n : ℕ) :
    (⟨p.erase n⟩ : R[X]) = (⟨p⟩ : R[X]).erase n := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : AddMonoidAlgebra R Nat
    n : Nat
    ⊢ Eq { toFinsupp := Finsupp.erase n p } (Polynomial.erase n { toFinsupp := p })
  -/
  rcases p with ⟨⟩
  /-
    case mk
    R : Type u
    inst✝ : Semiring R
    n : Nat
    support✝ : Finset Nat
    toFun✝ : Nat → R
    mem_support_toFun✝ : ∀ (a : Nat), Iff (Membership.mem support✝ a) (Ne (toFun✝  …
    ⊢ Eq { toFinsupp := Finsupp.erase n { support := support✝, toFun := toFun✝, me …
  -/
  simp only [erase_def]
  /-
    🎉 no goals
  -/


@[simp]
theorem support_erase (p : R[X]) (n : ℕ) : support (p.erase n) = (support p).erase n := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    n : Nat
    ⊢ Eq (Polynomial.erase n p).support (p.support.erase n)
  -/
  rcases p with ⟨⟩
  /-
    case ofFinsupp
    R : Type u
    inst✝ : Semiring R
    n : Nat
    toFinsupp✝ : AddMonoidAlgebra R Nat
    ⊢ Eq (Polynomial.erase n { toFinsupp := toFinsupp✝ }).support ({ toFinsupp :=  …
  -/
  simp only [support, erase_def, Finsupp.support_erase]
  /-
    🎉 no goals
  -/


theorem monomial_add_erase (p : R[X]) (n : ℕ) : monomial n (coeff p n) + p.erase n = p :=
  toFinsupp_injective <| by
    /-
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      n : Nat
      ⊢ Eq (HAdd.hAdd ((Polynomial.monomial n) (p.coeff n)) (Polynomial.erase n p)). …
    -/
    rcases p with ⟨⟩
    /-
      case ofFinsupp
      R : Type u
      inst✝ : Semiring R
      n : Nat
      toFinsupp✝ : AddMonoidAlgebra R Nat
      ⊢ Eq (HAdd.hAdd ((Polynomial.monomial n) ({ toFinsupp := toFinsupp✝ }.coeff n) …
    -/
    rw [toFinsupp_add, toFinsupp_monomial, toFinsupp_erase, coeff]
    /-
      case ofFinsupp
      R : Type u
      inst✝ : Semiring R
      n : Nat
      toFinsupp✝ : AddMonoidAlgebra R Nat
      ⊢ Eq (HAdd.hAdd (Finsupp.single n (toFinsupp✝ n)) (Finsupp.erase n { toFinsupp …
    -/
    exact Finsupp.single_add_erase _ _
    /-
      🎉 no goals
    -/


theorem coeff_erase (p : R[X]) (n i : ℕ) :
    (p.erase n).coeff i = if i = n then 0 else p.coeff i := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    n i : Nat
    ⊢ Eq ((Polynomial.erase n p).coeff i) (ite (Eq i n) 0 (p.coeff i))
  -/
  rcases p with ⟨⟩
  /-
    case ofFinsupp
    R : Type u
    inst✝ : Semiring R
    n i : Nat
    toFinsupp✝ : AddMonoidAlgebra R Nat
    ⊢ Eq ((Polynomial.erase n { toFinsupp := toFinsupp✝ }).coeff i) (ite (Eq i n)  …
  -/
  simp only [erase_def, coeff]
  -- Porting note: Was `convert rfl`.
  /-
    case ofFinsupp
    R : Type u
    inst✝ : Semiring R
    n i : Nat
    toFinsupp✝ : AddMonoidAlgebra R Nat
    ⊢ Eq ((Finsupp.erase n toFinsupp✝) i) (ite (Eq i n) 0 (toFinsupp✝ i))
  -/
  exact ite_congr rfl (fun _ => rfl) (fun _ => rfl)
  /-
    🎉 no goals
  -/


@[simp]
theorem erase_zero (n : ℕ) : (0 : R[X]).erase n = 0 :=
                            /-
                              R : Type u
                              inst✝ : Semiring R
                              n : Nat
                              ⊢ Eq (Polynomial.erase n 0).toFinsupp (Polynomial.toFinsupp 0)
                            -/
  toFinsupp_injective <| by simp
                            /-
                              🎉 no goals
                            -/


@[simp]
theorem erase_monomial {n : ℕ} {a : R} : erase n (monomial n a) = 0 :=
                            /-
                              R : Type u
                              inst✝ : Semiring R
                              n : Nat
                              a : R
                              ⊢ Eq (Polynomial.erase n ((Polynomial.monomial n) a)).toFinsupp (Polynomial.to …
                            -/
  toFinsupp_injective <| by simp
                            /-
                              🎉 no goals
                            -/


@[simp]
                                                                      /-
                                                                        R : Type u
                                                                        inst✝ : Semiring R
                                                                        p : Polynomial R
                                                                        n : Nat
                                                                        ⊢ Eq ((Polynomial.erase n p).coeff n) 0
                                                                      -/
theorem erase_same (p : R[X]) (n : ℕ) : coeff (p.erase n) n = 0 := by simp [coeff_erase]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


@[simp]
theorem erase_ne (p : R[X]) (n i : ℕ) (h : i ≠ n) : coeff (p.erase n) i = coeff p i := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    n i : Nat
    h : Ne i n
    ⊢ Eq ((Polynomial.erase n p).coeff i) (p.coeff i)
  -/
  simp [coeff_erase, h]
  /-
    🎉 no goals
  -/


/-- Replace the coefficient of a `p : R[X]` at a given degree `n : ℕ`
by a given value `a : R`. If `a = 0`, this is equal to `p.erase n`
If `p.natDegree < n` and `a ≠ 0`, this increases the degree to `n`. -/
def update (p : R[X]) (n : ℕ) (a : R) : R[X] :=
  Polynomial.ofFinsupp (p.toFinsupp.update n a)


theorem coeff_update (p : R[X]) (n : ℕ) (a : R) :
    (p.update n a).coeff = Function.update p.coeff n a := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    n : Nat
    a : R
    ⊢ Eq (p.update n a).coeff (Function.update p.coeff n a)
  -/
  ext
  /-
    case h
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    n : Nat
    a : R
    x✝ : Nat
    ⊢ Eq ((p.update n a).coeff x✝) (Function.update p.coeff n a x✝)
  -/
  cases p
  /-
    case h.ofFinsupp
    R : Type u
    inst✝ : Semiring R
    n : Nat
    a : R
    x✝ : Nat
    toFinsupp✝ : AddMonoidAlgebra R Nat
    ⊢ Eq (({ toFinsupp := toFinsupp✝ }.update n a).coeff x✝) (Function.update { to …
  -/
  simp only [coeff, update, Function.update_apply, coe_update]
  /-
    🎉 no goals
  -/


theorem coeff_update_apply (p : R[X]) (n : ℕ) (a : R) (i : ℕ) :
    (p.update n a).coeff i = if i = n then a else p.coeff i := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    n : Nat
    a : R
    i : Nat
    ⊢ Eq ((p.update n a).coeff i) (ite (Eq i n) a (p.coeff i))
  -/
  rw [coeff_update, Function.update_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem coeff_update_same (p : R[X]) (n : ℕ) (a : R) : (p.update n a).coeff n = a := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    n : Nat
    a : R
    ⊢ Eq ((p.update n a).coeff n) a
  -/
  rw [p.coeff_update_apply, if_pos rfl]
  /-
    🎉 no goals
  -/


theorem coeff_update_ne (p : R[X]) {n : ℕ} (a : R) {i : ℕ} (h : i ≠ n) :
                                             /-
                                               R : Type u
                                               inst✝ : Semiring R
                                               p : Polynomial R
                                               n : Nat
                                               a : R
                                               i : Nat
                                               h : Ne i n
                                               ⊢ Eq ((p.update n a).coeff i) (p.coeff i)
                                             -/
    (p.update n a).coeff i = p.coeff i := by rw [p.coeff_update_apply, if_neg h]
                                             /-
                                               🎉 no goals
                                             -/


@[simp]
theorem update_zero_eq_erase (p : R[X]) (n : ℕ) : p.update n 0 = p.erase n := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    n : Nat
    ⊢ Eq (p.update n 0) (Polynomial.erase n p)
  -/
  ext
  /-
    case a
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    n n✝ : Nat
    ⊢ Eq ((p.update n 0).coeff n✝) ((Polynomial.erase n p).coeff n✝)
  -/
  rw [coeff_update_apply, coeff_erase]
  /-
    🎉 no goals
  -/


theorem support_update (p : R[X]) (n : ℕ) (a : R) [Decidable (a = 0)] :
    support (p.update n a) = if a = 0 then p.support.erase n else insert n p.support := by
  classical
    cases p
    simp only [support, update, Finsupp.support_update]
    congr


theorem support_update_zero (p : R[X]) (n : ℕ) : support (p.update n 0) = p.support.erase n := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    n : Nat
    ⊢ Eq (p.update n 0).support (p.support.erase n)
  -/
  rw [update_zero_eq_erase, support_erase]
  /-
    🎉 no goals
  -/


theorem support_update_ne_zero (p : R[X]) (n : ℕ) {a : R} (ha : a ≠ 0) :
                                                      /-
                                                        R : Type u
                                                        inst✝ : Semiring R
                                                        p : Polynomial R
                                                        n : Nat
                                                        a : R
                                                        ha : Ne a 0
                                                        ⊢ Eq (p.update n a).support (Insert.insert n p.support)
                                                      -/
    support (p.update n a) = insert n p.support := by classical rw [support_update, if_neg ha]
                                                      /-
                                                        🎉 no goals
                                                      -/


instance commSemiring : CommSemiring R[X] :=
  { Function.Injective.commSemigroup toFinsupp toFinsupp_injective toFinsupp_mul with
    toSemiring := Polynomial.semiring }


instance instZSMul : SMul ℤ R[X] where
  smul r p := ⟨r • p.toFinsupp⟩


@[simp]
theorem ofFinsupp_zsmul (a : ℤ) (b) :
    (⟨a • b⟩ : R[X]) = (a • ⟨b⟩ : R[X]) :=
  rfl


@[simp]
theorem toFinsupp_zsmul (a : ℤ) (b : R[X]) :
    (a • b).toFinsupp = a • b.toFinsupp :=
  rfl


instance instIntCast : IntCast R[X] where intCast n := ofFinsupp n


instance ring : Ring R[X] :=
  --TODO: add reference to library note in PR https://github.com/leanprover-community/mathlib4/pull/7432
  { Function.Injective.ring toFinsupp toFinsupp_injective (toFinsupp_zero (R := R))
      toFinsupp_one toFinsupp_add
      toFinsupp_mul toFinsupp_neg toFinsupp_sub (fun _ _ => toFinsupp_nsmul _ _)
      (fun _ _ => toFinsupp_zsmul _ _) toFinsupp_pow (fun _ => rfl) fun _ => rfl with
    toSemiring := Polynomial.semiring,
    toNeg := Polynomial.neg'
    toSub := Polynomial.sub
    zsmul := ((· • ·) : ℤ → R[X] → R[X]) }


@[simp]
theorem coeff_neg (p : R[X]) (n : ℕ) : coeff (-p) n = -coeff p n := by
  /-
    R : Type u
    inst✝ : Ring R
    p : Polynomial R
    n : Nat
    ⊢ Eq ((Neg.neg p).coeff n) (Neg.neg (p.coeff n))
  -/
  rcases p with ⟨⟩
  /-
    case ofFinsupp
    R : Type u
    inst✝ : Ring R
    n : Nat
    toFinsupp✝ : AddMonoidAlgebra R Nat
    ⊢ Eq ((Neg.neg { toFinsupp := toFinsupp✝ }).coeff n) (Neg.neg ({ toFinsupp :=  …
  -/
  rw [← ofFinsupp_neg, coeff, coeff, Finsupp.neg_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem coeff_sub (p q : R[X]) (n : ℕ) : coeff (p - q) n = coeff p n - coeff q n := by
  /-
    R : Type u
    inst✝ : Ring R
    p q : Polynomial R
    n : Nat
    ⊢ Eq ((HSub.hSub p q).coeff n) (HSub.hSub (p.coeff n) (q.coeff n))
  -/
  rcases p with ⟨⟩
  /-
    case ofFinsupp
    R : Type u
    inst✝ : Ring R
    q : Polynomial R
    n : Nat
    toFinsupp✝ : AddMonoidAlgebra R Nat
    ⊢ Eq ((HSub.hSub { toFinsupp := toFinsupp✝ } q).coeff n) (HSub.hSub ({ toFinsu …
  -/
  rcases q with ⟨⟩
  /-
    case ofFinsupp.ofFinsupp
    R : Type u
    inst✝ : Ring R
    n : Nat
    toFinsupp✝¹ toFinsupp✝ : AddMonoidAlgebra R Nat
    ⊢ Eq ((HSub.hSub { toFinsupp := toFinsupp✝¹ } { toFinsupp := toFinsupp✝ }).coe …
  -/
  rw [← ofFinsupp_sub, coeff, coeff, coeff, Finsupp.sub_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem monomial_neg (n : ℕ) (a : R) : monomial n (-a) = -monomial n a := by
  /-
    R : Type u
    inst✝ : Ring R
    n : Nat
    a : R
    ⊢ Eq ((Polynomial.monomial n) (Neg.neg a)) (Neg.neg ((Polynomial.monomial n) a))
  -/
  rw [eq_neg_iff_add_eq_zero, ← monomial_add, neg_add_cancel, monomial_zero_right]
  /-
    🎉 no goals
  -/


theorem monomial_sub (n : ℕ) : monomial n (a - b) = monomial n a - monomial n b := by
  /-
    R : Type u
    a b : R
    inst✝ : Ring R
    n : Nat
    ⊢ Eq ((Polynomial.monomial n) (HSub.hSub a b)) (HSub.hSub ((Polynomial.monomia …
  -/
  rw [sub_eq_add_neg, monomial_add, monomial_neg, sub_eq_add_neg]
  /-
    🎉 no goals
  -/


@[simp]
theorem support_neg {p : R[X]} : (-p).support = p.support := by
  /-
    R : Type u
    inst✝ : Ring R
    p : Polynomial R
    ⊢ Eq (Neg.neg p).support p.support
  -/
  rcases p with ⟨⟩
  /-
    case ofFinsupp
    R : Type u
    inst✝ : Ring R
    toFinsupp✝ : AddMonoidAlgebra R Nat
    ⊢ Eq (Neg.neg { toFinsupp := toFinsupp✝ }).support { toFinsupp := toFinsupp✝ } …
  -/
  rw [← ofFinsupp_neg, support, support, Finsupp.support_neg]
  /-
    🎉 no goals
  -/


                                                   /-
                                                     R : Type u
                                                     inst✝ : Ring R
                                                     n : Int
                                                     ⊢ Eq (Polynomial.C ↑n) ↑n
                                                   -/
theorem C_eq_intCast (n : ℤ) : C (n : R) = n := by simp
                                                   /-
                                                     🎉 no goals
                                                   -/


@[deprecated (since := "2024-04-17")]
alias C_eq_int_cast := C_eq_intCast


theorem C_neg : C (-a) = -C a :=
  RingHom.map_neg C a


theorem C_sub : C (a - b) = C a - C b :=
  RingHom.map_sub C a b


instance commRing [CommRing R] : CommRing R[X] :=
  --TODO: add reference to library note in PR https://github.com/leanprover-community/mathlib4/pull/7432
  { toRing := Polynomial.ring
    mul_comm := mul_comm }


instance nontrivial [Nontrivial R] : Nontrivial R[X] := by
  /-
    R : Type u
    a b : R
    m n : Nat
    inst✝¹ : Semiring R
    inst✝ : Nontrivial R
    ⊢ Nontrivial (Polynomial R)
  -/
  have h : Nontrivial R[ℕ] := by infer_instance
  /-
    R : Type u
    a b : R
    m n : Nat
    inst✝¹ : Semiring R
    inst✝ : Nontrivial R
    h : Nontrivial (AddMonoidAlgebra R Nat)
    ⊢ Nontrivial (Polynomial R)
  -/
  rcases h.exists_pair_ne with ⟨x, y, hxy⟩
  /-
    case intro.intro
    R : Type u
    a b : R
    m n : Nat
    inst✝¹ : Semiring R
    inst✝ : Nontrivial R
    h : Nontrivial (AddMonoidAlgebra R Nat)
    x y : AddMonoidAlgebra R Nat
    hxy : Ne x y
    ⊢ Nontrivial (Polynomial R)
  -/
  refine ⟨⟨⟨x⟩, ⟨y⟩, ?_⟩⟩
  /-
    case intro.intro
    R : Type u
    a b : R
    m n : Nat
    inst✝¹ : Semiring R
    inst✝ : Nontrivial R
    h : Nontrivial (AddMonoidAlgebra R Nat)
    x y : AddMonoidAlgebra R Nat
    hxy : Ne x y
    ⊢ Ne { toFinsupp := x } { toFinsupp := y }
  -/
  simp [hxy]
  /-
    🎉 no goals
  -/


@[simp]
theorem X_ne_zero [Nontrivial R] : (X : R[X]) ≠ 0 :=
                                        /-
                                          R : Type u
                                          inst✝¹ : Semiring R
                                          inst✝ : Nontrivial R
                                          ⊢ Not (Eq (Polynomial.X.coeff 1) (Polynomial.coeff 0 1))
                                        -/
  mt (congr_arg fun p => coeff p 1) (by simp)
                                        /-
                                          🎉 no goals
                                        -/


lemma nnqsmul_eq_C_mul (q : ℚ≥0) (f : R[X]) : q • f = Polynomial.C (q : R) * f := by
  /-
    R : Type u
    inst✝ : DivisionSemiring R
    q : NNRat
    f : Polynomial R
    ⊢ Eq (HSMul.hSMul q f) (HMul.hMul (Polynomial.C ↑q) f)
  -/
  rw [← NNRat.smul_one_eq_cast, ← Polynomial.smul_C, C_1, smul_one_mul]
  /-
    🎉 no goals
  -/


theorem qsmul_eq_C_mul (a : ℚ) (f : R[X]) : a • f = Polynomial.C (a : R) * f := by
  /-
    R : Type u
    inst✝ : DivisionRing R
    a : Rat
    f : Polynomial R
    ⊢ Eq (HSMul.hSMul a f) (HMul.hMul (Polynomial.C ↑a) f)
  -/
  rw [← Rat.smul_one_eq_cast, ← Polynomial.smul_C, C_1, smul_one_mul]
  /-
    🎉 no goals
  -/


@[simp]
theorem nontrivial_iff [Semiring R] : Nontrivial R[X] ↔ Nontrivial R :=
  ⟨fun h =>
    let ⟨_r, _s, hrs⟩ := @exists_pair_ne _ h
    Nontrivial.of_polynomial_ne hrs,
    fun h => @Polynomial.nontrivial _ _ h⟩


protected instance repr [Repr R] [DecidableEq R] : Repr R[X] :=
  ⟨fun p prec =>
    let termPrecAndReprs : List (WithTop ℕ × Lean.Format) :=
      List.map (fun
        | 0 => (max_prec, "C " ++ reprArg (coeff p 0))
        | 1 => if coeff p 1 = 1
          then (⊤, "X")
          else (70, "C " ++ reprArg (coeff p 1) ++ " * X")
        | n =>
          if coeff p n = 1
          then (80, "X ^ " ++ Nat.repr n)
          else (70, "C " ++ reprArg (coeff p n) ++ " * X ^ " ++ Nat.repr n))
      (p.support.sort (· ≤ ·))
    match termPrecAndReprs with
    | [] => "0"
    | [(tprec, t)] => if prec ≥ tprec then Lean.Format.paren t else t
    | ts =>
      -- multiple terms, use `+` precedence
      (if prec ≥ 65 then Lean.Format.paren else id)
      (Lean.Format.fill
        (Lean.Format.joinSep (ts.map Prod.snd) (" +" ++ Lean.Format.line)))⟩


