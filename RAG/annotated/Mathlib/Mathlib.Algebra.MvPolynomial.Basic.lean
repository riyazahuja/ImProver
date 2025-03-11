/-- Multivariate polynomial, where `σ` is the index set of the variables and
  `R` is the coefficient ring -/
def MvPolynomial (σ : Type*) (R : Type*) [CommSemiring R] :=
  AddMonoidAlgebra R (σ →₀ ℕ)


instance decidableEqMvPolynomial [CommSemiring R] [DecidableEq σ] [DecidableEq R] :
    DecidableEq (MvPolynomial σ R) :=
  Finsupp.instDecidableEq


instance commSemiring [CommSemiring R] : CommSemiring (MvPolynomial σ R) :=
  AddMonoidAlgebra.commSemiring


instance inhabited [CommSemiring R] : Inhabited (MvPolynomial σ R) :=
  ⟨0⟩


instance distribuMulAction [Monoid R] [CommSemiring S₁] [DistribMulAction R S₁] :
    DistribMulAction R (MvPolynomial σ S₁) :=
  AddMonoidAlgebra.distribMulAction


instance smulZeroClass [CommSemiring S₁] [SMulZeroClass R S₁] :
    SMulZeroClass R (MvPolynomial σ S₁) :=
  AddMonoidAlgebra.smulZeroClass


instance faithfulSMul [CommSemiring S₁] [SMulZeroClass R S₁] [FaithfulSMul R S₁] :
    FaithfulSMul R (MvPolynomial σ S₁) :=
  AddMonoidAlgebra.faithfulSMul


instance module [Semiring R] [CommSemiring S₁] [Module R S₁] : Module R (MvPolynomial σ S₁) :=
  AddMonoidAlgebra.module


instance isScalarTower [CommSemiring S₂] [SMul R S₁] [SMulZeroClass R S₂] [SMulZeroClass S₁ S₂]
    [IsScalarTower R S₁ S₂] : IsScalarTower R S₁ (MvPolynomial σ S₂) :=
  AddMonoidAlgebra.isScalarTower


instance smulCommClass [CommSemiring S₂] [SMulZeroClass R S₂] [SMulZeroClass S₁ S₂]
    [SMulCommClass R S₁ S₂] : SMulCommClass R S₁ (MvPolynomial σ S₂) :=
  AddMonoidAlgebra.smulCommClass


instance isCentralScalar [CommSemiring S₁] [SMulZeroClass R S₁] [SMulZeroClass Rᵐᵒᵖ S₁]
    [IsCentralScalar R S₁] : IsCentralScalar R (MvPolynomial σ S₁) :=
  AddMonoidAlgebra.isCentralScalar


instance algebra [CommSemiring R] [CommSemiring S₁] [Algebra R S₁] :
    Algebra R (MvPolynomial σ S₁) :=
  AddMonoidAlgebra.algebra


instance isScalarTower_right [CommSemiring S₁] [DistribSMul R S₁] [IsScalarTower R S₁ S₁] :
    IsScalarTower R (MvPolynomial σ S₁) (MvPolynomial σ S₁) :=
  AddMonoidAlgebra.isScalarTower_self _


instance smulCommClass_right [CommSemiring S₁] [DistribSMul R S₁] [SMulCommClass R S₁ S₁] :
    SMulCommClass R (MvPolynomial σ S₁) (MvPolynomial σ S₁) :=
  AddMonoidAlgebra.smulCommClass_self _


/-- If `R` is a subsingleton, then `MvPolynomial σ R` has a unique element -/
instance unique [CommSemiring R] [Subsingleton R] : Unique (MvPolynomial σ R) :=
  AddMonoidAlgebra.unique


/-- `monomial s a` is the monomial with coefficient `a` and exponents given by `s`  -/
def monomial (s : σ →₀ ℕ) : R →ₗ[R] MvPolynomial σ R :=
  AddMonoidAlgebra.lsingle s


theorem single_eq_monomial (s : σ →₀ ℕ) (a : R) : Finsupp.single s a = monomial s a :=
  rfl


theorem mul_def : p * q = p.sum fun m a => q.sum fun n b => monomial (m + n) (a * b) :=
  AddMonoidAlgebra.mul_def


/-- `C a` is the constant polynomial with value `a` -/
def C : R →+* MvPolynomial σ R :=
  { singleZeroRingHom with toFun := monomial 0 }


@[simp]
theorem algebraMap_eq : algebraMap R (MvPolynomial σ R) = C :=
  rfl


/-- `X n` is the degree `1` monomial $X_n$. -/
def X (n : σ) : MvPolynomial σ R :=
  monomial (Finsupp.single n 1) 1


theorem monomial_left_injective {r : R} (hr : r ≠ 0) :
    Function.Injective fun s : σ →₀ ℕ => monomial s r :=
  Finsupp.single_left_injective hr


@[simp]
theorem monomial_left_inj {s t : σ →₀ ℕ} {r : R} (hr : r ≠ 0) :
    monomial s r = monomial t r ↔ s = t :=
  Finsupp.single_left_inj hr


theorem C_apply : (C a : MvPolynomial σ R) = monomial 0 a :=
  rfl


@[simp]
theorem C_0 : C 0 = (0 : MvPolynomial σ R) := map_zero _


@[simp]
theorem C_1 : C 1 = (1 : MvPolynomial σ R) :=
  rfl


theorem C_mul_monomial : C a * monomial s a' = monomial s (a * a') := by
  -- Porting note: this `show` feels like defeq abuse, but I can't find the appropriate lemmas
  /-
    R : Type u
    σ : Type u_1
    a a' : R
    s : Finsupp σ Nat
    inst✝ : CommSemiring R
    ⊢ Eq (HMul.hMul (MvPolynomial.C a) ((MvPolynomial.monomial s) a')) ((MvPolynom …
  -/
  show AddMonoidAlgebra.single _ _ * AddMonoidAlgebra.single _ _ = AddMonoidAlgebra.single _ _
  /-
    R : Type u
    σ : Type u_1
    a a' : R
    s : Finsupp σ Nat
    inst✝ : CommSemiring R
    ⊢ Eq (HMul.hMul (AddMonoidAlgebra.single 0 a) (AddMonoidAlgebra.single s a'))  …
  -/
  simp [C_apply, single_mul_single]
  /-
    🎉 no goals
  -/


@[simp]
theorem C_add : (C (a + a') : MvPolynomial σ R) = C a + C a' :=
  Finsupp.single_add _ _ _


@[simp]
theorem C_mul : (C (a * a') : MvPolynomial σ R) = C a * C a' :=
  C_mul_monomial.symm


@[simp]
theorem C_pow (a : R) (n : ℕ) : (C (a ^ n) : MvPolynomial σ R) = C a ^ n :=
  map_pow _ _ _


theorem C_injective (σ : Type*) (R : Type*) [CommSemiring R] :
    Function.Injective (C : R → MvPolynomial σ R) :=
  Finsupp.single_injective _


theorem C_surjective {R : Type*} [CommSemiring R] (σ : Type*) [IsEmpty σ] :
    Function.Surjective (C : R → MvPolynomial σ R) := by
  /-
    R : Type u_2
    inst✝¹ : CommSemiring R
    σ : Type u_3
    inst✝ : IsEmpty σ
    ⊢ Function.Surjective ⇑MvPolynomial.C
  -/
  refine fun p => ⟨p.toFun 0, Finsupp.ext fun a => ?_⟩
  simp only [C_apply, ← single_eq_monomial, (Finsupp.ext isEmptyElim (α := σ) : a = 0),
    single_eq_same]
  /-
    R : Type u_2
    inst✝¹ : CommSemiring R
    σ : Type u_3
    inst✝ : IsEmpty σ
    p : MvPolynomial σ R
    a : Finsupp σ Nat
    ⊢ Eq (p.toFun 0) (p 0)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem C_inj {σ : Type*} (R : Type*) [CommSemiring R] (r s : R) :
    (C r : MvPolynomial σ R) = C s ↔ r = s :=
  (C_injective σ R).eq_iff


                                                                     /-
                                                                       R : Type u
                                                                       σ : Type u_1
                                                                       a : R
                                                                       inst✝ : CommSemiring R
                                                                       ⊢ Iff (Eq (MvPolynomial.C a) 0) (Eq a 0)
                                                                     -/
@[simp] lemma C_eq_zero : (C a : MvPolynomial σ R) = 0 ↔ a = 0 := by rw [← map_zero C, C_inj]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


lemma C_ne_zero : (C a : MvPolynomial σ R) ≠ 0 ↔ a ≠ 0 :=
  C_eq_zero.ne


instance nontrivial_of_nontrivial (σ : Type*) (R : Type*) [CommSemiring R] [Nontrivial R] :
    Nontrivial (MvPolynomial σ R) :=
  inferInstanceAs (Nontrivial <| AddMonoidAlgebra R (σ →₀ ℕ))


instance infinite_of_infinite (σ : Type*) (R : Type*) [CommSemiring R] [Infinite R] :
    Infinite (MvPolynomial σ R) :=
  Infinite.of_injective C (C_injective _ _)


instance infinite_of_nonempty (σ : Type*) (R : Type*) [Nonempty σ] [CommSemiring R]
    [Nontrivial R] : Infinite (MvPolynomial σ R) :=
  Infinite.of_injective ((fun s : σ →₀ ℕ => monomial s 1) ∘ Finsupp.single (Classical.arbitrary σ))
    <| (monomial_left_injective one_ne_zero).comp (Finsupp.single_injective _)


theorem C_eq_coe_nat (n : ℕ) : (C ↑n : MvPolynomial σ R) = n := by
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    n : Nat
    ⊢ Eq (MvPolynomial.C ↑n) ↑n
  -/
                  /-
                    🎉 no goals
                  -/
  induction n <;> simp [*]
                  /-
                    🎉 no goals
                  -/


theorem C_mul' : MvPolynomial.C a * p = a • p :=
  (Algebra.smul_def a p).symm


theorem smul_eq_C_mul (p : MvPolynomial σ R) (a : R) : a • p = C a * p :=
  C_mul'.symm


theorem C_eq_smul_one : (C a : MvPolynomial σ R) = a • (1 : MvPolynomial σ R) := by
  /-
    R : Type u
    σ : Type u_1
    a : R
    inst✝ : CommSemiring R
    ⊢ Eq (MvPolynomial.C a) (HSMul.hSMul a 1)
  -/
  rw [← C_mul', mul_one]
  /-
    🎉 no goals
  -/


theorem smul_monomial {S₁ : Type*} [SMulZeroClass S₁ R] (r : S₁) :
    r • monomial s a = monomial s (r • a) :=
  Finsupp.smul_single _ _ _


theorem X_injective [Nontrivial R] : Function.Injective (X : σ → MvPolynomial σ R) :=
  (monomial_left_injective one_ne_zero).comp (Finsupp.single_left_injective one_ne_zero)


@[simp]
theorem X_inj [Nontrivial R] (m n : σ) : X m = (X n : MvPolynomial σ R) ↔ m = n :=
  X_injective.eq_iff


theorem monomial_pow : monomial s a ^ e = monomial (e • s) (a ^ e) :=
  AddMonoidAlgebra.single_pow e


@[simp]
theorem monomial_mul {s s' : σ →₀ ℕ} {a b : R} :
    monomial s a * monomial s' b = monomial (s + s') (a * b) :=
  AddMonoidAlgebra.single_mul_single


/-- `fun s ↦ monomial s 1` as a homomorphism. -/
def monomialOneHom : Multiplicative (σ →₀ ℕ) →* MvPolynomial σ R :=
  AddMonoidAlgebra.of _ _


@[simp]
theorem monomialOneHom_apply : monomialOneHom R σ s = (monomial s 1 : MvPolynomial σ R) :=
  rfl


theorem X_pow_eq_monomial : X n ^ e = monomial (Finsupp.single n e) (1 : R) := by
  /-
    R : Type u
    σ : Type u_1
    e : Nat
    n : σ
    inst✝ : CommSemiring R
    ⊢ Eq (HPow.hPow (MvPolynomial.X n) e) ((MvPolynomial.monomial (Finsupp.single  …
  -/
  simp [X, monomial_pow]
  /-
    🎉 no goals
  -/


theorem monomial_add_single : monomial (s + Finsupp.single n e) a = monomial s a * X n ^ e := by
  /-
    R : Type u
    σ : Type u_1
    a : R
    e : Nat
    n : σ
    s : Finsupp σ Nat
    inst✝ : CommSemiring R
    ⊢ Eq ((MvPolynomial.monomial (HAdd.hAdd s (Finsupp.single n e))) a) (HMul.hMul …
  -/
  rw [X_pow_eq_monomial, monomial_mul, mul_one]
  /-
    🎉 no goals
  -/


theorem monomial_single_add : monomial (Finsupp.single n e + s) a = X n ^ e * monomial s a := by
  /-
    R : Type u
    σ : Type u_1
    a : R
    e : Nat
    n : σ
    s : Finsupp σ Nat
    inst✝ : CommSemiring R
    ⊢ Eq ((MvPolynomial.monomial (HAdd.hAdd (Finsupp.single n e) s)) a) (HMul.hMul …
  -/
  rw [X_pow_eq_monomial, monomial_mul, one_mul]
  /-
    🎉 no goals
  -/


theorem C_mul_X_pow_eq_monomial {s : σ} {a : R} {n : ℕ} :
    C a * X s ^ n = monomial (Finsupp.single s n) a := by
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    s : σ
    a : R
    n : Nat
    ⊢ Eq (HMul.hMul (MvPolynomial.C a) (HPow.hPow (MvPolynomial.X s) n)) ((MvPolyn …
  -/
  rw [← zero_add (Finsupp.single s n), monomial_add_single, C_apply]
  /-
    🎉 no goals
  -/


theorem C_mul_X_eq_monomial {s : σ} {a : R} : C a * X s = monomial (Finsupp.single s 1) a := by
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    s : σ
    a : R
    ⊢ Eq (HMul.hMul (MvPolynomial.C a) (MvPolynomial.X s)) ((MvPolynomial.monomial …
  -/
  rw [← C_mul_X_pow_eq_monomial, pow_one]
  /-
    🎉 no goals
  -/


@[simp]
theorem monomial_zero {s : σ →₀ ℕ} : monomial s (0 : R) = 0 :=
  Finsupp.single_zero _


@[simp]
theorem monomial_zero' : (monomial (0 : σ →₀ ℕ) : R → MvPolynomial σ R) = C :=
  rfl


@[simp]
theorem monomial_eq_zero {s : σ →₀ ℕ} {b : R} : monomial s b = 0 ↔ b = 0 :=
  Finsupp.single_eq_zero


@[simp]
theorem sum_monomial_eq {A : Type*} [AddCommMonoid A] {u : σ →₀ ℕ} {r : R} {b : (σ →₀ ℕ) → R → A}
    (w : b u 0 = 0) : sum (monomial u r) b = b u r :=
  Finsupp.sum_single_index w


@[simp]
theorem sum_C {A : Type*} [AddCommMonoid A] {b : (σ →₀ ℕ) → R → A} (w : b 0 0 = 0) :
    sum (C a) b = b 0 a :=
  sum_monomial_eq w


theorem monomial_sum_one {α : Type*} (s : Finset α) (f : α → σ →₀ ℕ) :
    (monomial (∑ i ∈ s, f i) 1 : MvPolynomial σ R) = ∏ i ∈ s, monomial (f i) 1 :=
  map_prod (monomialOneHom R σ) (fun i => Multiplicative.ofAdd (f i)) s


theorem monomial_sum_index {α : Type*} (s : Finset α) (f : α → σ →₀ ℕ) (a : R) :
    monomial (∑ i ∈ s, f i) a = C a * ∏ i ∈ s, monomial (f i) 1 := by
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    α : Type u_2
    s : Finset α
    f : α → Finsupp σ Nat
    a : R
    ⊢ Eq ((MvPolynomial.monomial (s.sum fun i => f i)) a) (HMul.hMul (MvPolynomial …
  -/
  rw [← monomial_sum_one, C_mul', ← (monomial _).map_smul, smul_eq_mul, mul_one]
  /-
    🎉 no goals
  -/


theorem monomial_finsupp_sum_index {α β : Type*} [Zero β] (f : α →₀ β) (g : α → β → σ →₀ ℕ)
    (a : R) : monomial (f.sum g) a = C a * f.prod fun a b => monomial (g a b) 1 :=
  monomial_sum_index _ _ _


theorem monomial_eq_monomial_iff {α : Type*} (a₁ a₂ : α →₀ ℕ) (b₁ b₂ : R) :
    monomial a₁ b₁ = monomial a₂ b₂ ↔ a₁ = a₂ ∧ b₁ = b₂ ∨ b₁ = 0 ∧ b₂ = 0 :=
  Finsupp.single_eq_single_iff _ _ _ _


theorem monomial_eq : monomial s a = C a * (s.prod fun n e => X n ^ e : MvPolynomial σ R) := by
  /-
    R : Type u
    σ : Type u_1
    a : R
    s : Finsupp σ Nat
    inst✝ : CommSemiring R
    ⊢ Eq ((MvPolynomial.monomial s) a) (HMul.hMul (MvPolynomial.C a) (s.prod fun n …
  -/
  simp only [X_pow_eq_monomial, ← monomial_finsupp_sum_index, Finsupp.sum_single]
  /-
    🎉 no goals
  -/


@[simp]
lemma prod_X_pow_eq_monomial : ∏ x ∈ s.support, X x ^ s x = monomial s (1 : R) := by
  /-
    R : Type u
    σ : Type u_1
    s : Finsupp σ Nat
    inst✝ : CommSemiring R
    ⊢ Eq (s.support.prod fun x => HPow.hPow (MvPolynomial.X x) (s x)) ((MvPolynomi …
  -/
  simp only [monomial_eq, map_one, one_mul, Finsupp.prod]
  /-
    🎉 no goals
  -/


theorem induction_on_monomial {M : MvPolynomial σ R → Prop} (h_C : ∀ a, M (C a))
    (h_X : ∀ p n, M p → M (p * X n)) : ∀ s a, M (monomial s a) := by
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    M : MvPolynomial σ R → Prop
    h_C : ∀ (a : R), M (MvPolynomial.C a)
    h_X : ∀ (p : MvPolynomial σ R) (n : σ), M p → M (HMul.hMul p (MvPolynomial.X n))
    ⊢ ∀ (s : Finsupp σ Nat) (a : R), M ((MvPolynomial.monomial s) a)
  -/
  intro s a
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    M : MvPolynomial σ R → Prop
    h_C : ∀ (a : R), M (MvPolynomial.C a)
    h_X : ∀ (p : MvPolynomial σ R) (n : σ), M p → M (HMul.hMul p (MvPolynomial.X n))
    s : Finsupp σ Nat
    a : R
    ⊢ M ((MvPolynomial.monomial s) a)
  -/
  apply @Finsupp.induction σ ℕ _ _ s
    /-
      case h0
      R : Type u
      σ : Type u_1
      inst✝ : CommSemiring R
      M : MvPolynomial σ R → Prop
      h_C : ∀ (a : R), M (MvPolynomial.C a)
      h_X : ∀ (p : MvPolynomial σ R) (n : σ), M p → M (HMul.hMul p (MvPolynomial.X n))
      s : Finsupp σ Nat
      a : R
      ⊢ M ((MvPolynomial.monomial 0) a)
    -/
  · show M (monomial 0 a)
    /-
      case h0
      R : Type u
      σ : Type u_1
      inst✝ : CommSemiring R
      M : MvPolynomial σ R → Prop
      h_C : ∀ (a : R), M (MvPolynomial.C a)
      h_X : ∀ (p : MvPolynomial σ R) (n : σ), M p → M (HMul.hMul p (MvPolynomial.X n))
      s : Finsupp σ Nat
      a : R
      ⊢ M ((MvPolynomial.monomial 0) a)
    -/
    exact h_C a
    /-
      🎉 no goals
    -/
    /-
      case ha
      R : Type u
      σ : Type u_1
      inst✝ : CommSemiring R
      M : MvPolynomial σ R → Prop
      h_C : ∀ (a : R), M (MvPolynomial.C a)
      h_X : ∀ (p : MvPolynomial σ R) (n : σ), M p → M (HMul.hMul p (MvPolynomial.X n))
      s : Finsupp σ Nat
      a : R
      ⊢ ∀ (a_1 : σ) (b : Nat) (f : Finsupp σ Nat), Not (Membership.mem f.support a_1 …
    -/
  · intro n e p _hpn _he ih
    have : ∀ e : ℕ, M (monomial p a * X n ^ e) := by
      intro e
      induction e with
      | zero => simp [ih]
      | succ e e_ih => simp [ih, pow_succ, (mul_assoc _ _ _).symm, h_X, e_ih]
    /-
      case ha
      R : Type u
      σ : Type u_1
      inst✝ : CommSemiring R
      M : MvPolynomial σ R → Prop
      h_C : ∀ (a : R), M (MvPolynomial.C a)
      h_X : ∀ (p : MvPolynomial σ R) (n : σ), M p → M (HMul.hMul p (MvPolynomial.X n))
      s : Finsupp σ Nat
      a : R
      n : σ
      e : Nat
      p : Finsupp σ Nat
      _hpn : Not (Membership.mem p.support n)
      _he : Ne e 0
      ih : M ((MvPolynomial.monomial p) a)
      this : ∀ (e : Nat), M (HMul.hMul ((MvPolynomial.monomial p) a) (HPow.hPow (MvP …
      ⊢ M ((MvPolynomial.monomial (HAdd.hAdd (Finsupp.single n e) p)) a)
    -/
    simp [add_comm, monomial_add_single, this]
    /-
      🎉 no goals
    -/


/-- Analog of `Polynomial.induction_on'`.
To prove something about mv_polynomials,
it suffices to show the condition is closed under taking sums,
and it holds for monomials. -/
@[elab_as_elim]
theorem induction_on' {P : MvPolynomial σ R → Prop} (p : MvPolynomial σ R)
    (h1 : ∀ (u : σ →₀ ℕ) (a : R), P (monomial u a))
    (h2 : ∀ p q : MvPolynomial σ R, P p → P q → P (p + q)) : P p :=
  Finsupp.induction p
                                  /-
                                    R : Type u
                                    σ : Type u_1
                                    inst✝ : CommSemiring R
                                    P : MvPolynomial σ R → Prop
                                    p : MvPolynomial σ R
                                    h1 : ∀ (u : Finsupp σ Nat) (a : R), P ((MvPolynomial.monomial u) a)
                                    h2 : ∀ (p q : MvPolynomial σ R), P p → P q → P (HAdd.hAdd p q)
                                    this : P ((MvPolynomial.monomial 0) 0)
                                    ⊢ P 0
                                  -/
    (suffices P (monomial 0 0) by rwa [monomial_zero] at this
                                  /-
                                    🎉 no goals
                                  -/
    show P (monomial 0 0) from h1 0 0)
    fun _ _ _ _ha _hb hPf => h2 _ _ (h1 _ _) hPf


/-- Similar to `MvPolynomial.induction_on` but only a weak form of `h_add` is required. -/
theorem induction_on''' {M : MvPolynomial σ R → Prop} (p : MvPolynomial σ R) (h_C : ∀ a, M (C a))
    (h_add_weak :
      ∀ (a : σ →₀ ℕ) (b : R) (f : (σ →₀ ℕ) →₀ R),
        a ∉ f.support → b ≠ 0 → M f → M ((show (σ →₀ ℕ) →₀ R from monomial a b) + f)) :
    M p :=
    -- Porting note: I had to add the `show ... from ...` above, a type ascription was insufficient.
  Finsupp.induction p (C_0.rec <| h_C 0) h_add_weak


/-- Similar to `MvPolynomial.induction_on` but only a yet weaker form of `h_add` is required. -/
theorem induction_on'' {M : MvPolynomial σ R → Prop} (p : MvPolynomial σ R) (h_C : ∀ a, M (C a))
    (h_add_weak :
      ∀ (a : σ →₀ ℕ) (b : R) (f : (σ →₀ ℕ) →₀ R),
        a ∉ f.support → b ≠ 0 → M f → M (monomial a b) →
          M ((show (σ →₀ ℕ) →₀ R from monomial a b) + f))
    (h_X : ∀ (p : MvPolynomial σ R) (n : σ), M p → M (p * MvPolynomial.X n)) : M p :=
    -- Porting note: I had to add the `show ... from ...` above, a type ascription was insufficient.
  induction_on''' p h_C fun a b f ha hb hf =>
    h_add_weak a b f ha hb hf <| induction_on_monomial h_C h_X a b


/-- Analog of `Polynomial.induction_on`. -/
@[recursor 5]
theorem induction_on {M : MvPolynomial σ R → Prop} (p : MvPolynomial σ R) (h_C : ∀ a, M (C a))
    (h_add : ∀ p q, M p → M q → M (p + q)) (h_X : ∀ p n, M p → M (p * X n)) : M p :=
  induction_on'' p h_C (fun a b f _ha _hb hf hm => h_add (monomial a b) f hm hf) h_X


theorem ringHom_ext {A : Type*} [Semiring A] {f g : MvPolynomial σ R →+* A}
    (hC : ∀ r, f (C r) = g (C r)) (hX : ∀ i, f (X i) = g (X i)) : f = g := by
  /-
    R : Type u
    σ : Type u_1
    inst✝¹ : CommSemiring R
    A : Type u_2
    inst✝ : Semiring A
    f g : RingHom (MvPolynomial σ R) A
    hC : ∀ (r : R), Eq (f (MvPolynomial.C r)) (g (MvPolynomial.C r))
    hX : ∀ (i : σ), Eq (f (MvPolynomial.X i)) (g (MvPolynomial.X i))
    ⊢ Eq f g
  -/
  refine AddMonoidAlgebra.ringHom_ext' ?_ ?_
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): this has high priority, but Lean still chooses `RingHom.ext`, why?
  -- probably because of the type synonym
    /-
      case refine_1
      R : Type u
      σ : Type u_1
      inst✝¹ : CommSemiring R
      A : Type u_2
      inst✝ : Semiring A
      f g : RingHom (MvPolynomial σ R) A
      hC : ∀ (r : R), Eq (f (MvPolynomial.C r)) (g (MvPolynomial.C r))
      hX : ∀ (i : σ), Eq (f (MvPolynomial.X i)) (g (MvPolynomial.X i))
      ⊢ Eq (f.comp AddMonoidAlgebra.singleZeroRingHom) (g.comp AddMonoidAlgebra.sing …
    -/
  · ext x
    /-
      case refine_1.a
      R : Type u
      σ : Type u_1
      inst✝¹ : CommSemiring R
      A : Type u_2
      inst✝ : Semiring A
      f g : RingHom (MvPolynomial σ R) A
      hC : ∀ (r : R), Eq (f (MvPolynomial.C r)) (g (MvPolynomial.C r))
      hX : ∀ (i : σ), Eq (f (MvPolynomial.X i)) (g (MvPolynomial.X i))
      x : R
      ⊢ Eq ((f.comp AddMonoidAlgebra.singleZeroRingHom) x) ((g.comp AddMonoidAlgebra …
    -/
    exact hC _
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u
      σ : Type u_1
      inst✝¹ : CommSemiring R
      A : Type u_2
      inst✝ : Semiring A
      f g : RingHom (MvPolynomial σ R) A
      hC : ∀ (r : R), Eq (f (MvPolynomial.C r)) (g (MvPolynomial.C r))
      hX : ∀ (i : σ), Eq (f (MvPolynomial.X i)) (g (MvPolynomial.X i))
      ⊢ Eq ((↑f).comp (AddMonoidAlgebra.of R (Finsupp σ Nat))) ((↑g).comp (AddMonoid …
    -/
  · apply Finsupp.mulHom_ext'; intros x
    -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): `Finsupp.mulHom_ext'` needs to have increased priority
    /-
      case refine_2.H
      R : Type u
      σ : Type u_1
      inst✝¹ : CommSemiring R
      A : Type u_2
      inst✝ : Semiring A
      f g : RingHom (MvPolynomial σ R) A
      hC : ∀ (r : R), Eq (f (MvPolynomial.C r)) (g (MvPolynomial.C r))
      hX : ∀ (i : σ), Eq (f (MvPolynomial.X i)) (g (MvPolynomial.X i))
      x : σ
      ⊢ Eq (((↑f).comp (AddMonoidAlgebra.of R (Finsupp σ Nat))).comp (AddMonoidHom.t …
    -/
    apply MonoidHom.ext_mnat
    /-
      case refine_2.H.h
      R : Type u
      σ : Type u_1
      inst✝¹ : CommSemiring R
      A : Type u_2
      inst✝ : Semiring A
      f g : RingHom (MvPolynomial σ R) A
      hC : ∀ (r : R), Eq (f (MvPolynomial.C r)) (g (MvPolynomial.C r))
      hX : ∀ (i : σ), Eq (f (MvPolynomial.X i)) (g (MvPolynomial.X i))
      x : σ
      ⊢ Eq ((((↑f).comp (AddMonoidAlgebra.of R (Finsupp σ Nat))).comp (AddMonoidHom. …
    -/
    exact hX _
    /-
      🎉 no goals
    -/


/-- See note [partially-applied ext lemmas]. -/
@[ext 1100]
theorem ringHom_ext' {A : Type*} [Semiring A] {f g : MvPolynomial σ R →+* A}
    (hC : f.comp C = g.comp C) (hX : ∀ i, f (X i) = g (X i)) : f = g :=
  ringHom_ext (RingHom.ext_iff.1 hC) hX


theorem hom_eq_hom [Semiring S₂] (f g : MvPolynomial σ R →+* S₂) (hC : f.comp C = g.comp C)
    (hX : ∀ n : σ, f (X n) = g (X n)) (p : MvPolynomial σ R) : f p = g p :=
  RingHom.congr_fun (ringHom_ext' hC hX) p


theorem is_id (f : MvPolynomial σ R →+* MvPolynomial σ R) (hC : f.comp C = C)
    (hX : ∀ n : σ, f (X n) = X n) (p : MvPolynomial σ R) : f p = p :=
  hom_eq_hom f (RingHom.id _) hC hX p


@[ext 1100]
theorem algHom_ext' {A B : Type*} [CommSemiring A] [CommSemiring B] [Algebra R A] [Algebra R B]
    {f g : MvPolynomial σ A →ₐ[R] B}
    (h₁ :
      f.comp (IsScalarTower.toAlgHom R A (MvPolynomial σ A)) =
        g.comp (IsScalarTower.toAlgHom R A (MvPolynomial σ A)))
    (h₂ : ∀ i, f (X i) = g (X i)) : f = g :=
  AlgHom.coe_ringHom_injective (MvPolynomial.ringHom_ext' (congr_arg AlgHom.toRingHom h₁) h₂)


@[ext 1200]
theorem algHom_ext {A : Type*} [Semiring A] [Algebra R A] {f g : MvPolynomial σ R →ₐ[R] A}
    (hf : ∀ i : σ, f (X i) = g (X i)) : f = g :=
  AddMonoidAlgebra.algHom_ext' (mulHom_ext' fun X : σ => MonoidHom.ext_mnat (hf X))


@[simp]
theorem algHom_C {A : Type*} [Semiring A] [Algebra R A] (f : MvPolynomial σ R →ₐ[R] A) (r : R) :
    f (C r) = algebraMap R A r :=
  f.commutes r


@[simp]
theorem adjoin_range_X : Algebra.adjoin R (range (X : σ → MvPolynomial σ R)) = ⊤ := by
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    ⊢ Eq (Algebra.adjoin R (Set.range MvPolynomial.X)) Top.top
  -/
  set S := Algebra.adjoin R (range (X : σ → MvPolynomial σ R))
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    S : Subalgebra R (MvPolynomial σ R) := Algebra.adjoin R (Set.range MvPolynomia …
    ⊢ Eq S Top.top
  -/
  refine top_unique fun p hp => ?_; clear hp
  induction p using MvPolynomial.induction_on with
  | h_C => exact S.algebraMap_mem _
  | h_add p q hp hq => exact S.add_mem hp hq
  | h_X p i hp => exact S.mul_mem hp (Algebra.subset_adjoin <| mem_range_self _)


@[ext]
theorem linearMap_ext {M : Type*} [AddCommMonoid M] [Module R M] {f g : MvPolynomial σ R →ₗ[R] M}
    (h : ∀ s, f ∘ₗ monomial s = g ∘ₗ monomial s) : f = g :=
  Finsupp.lhom_ext' h


/-- The finite set of all `m : σ →₀ ℕ` such that `X^m` has a non-zero coefficient. -/
def support (p : MvPolynomial σ R) : Finset (σ →₀ ℕ) :=
  Finsupp.support p


theorem finsupp_support_eq_support (p : MvPolynomial σ R) : Finsupp.support p = p.support :=
  rfl


theorem support_monomial [h : Decidable (a = 0)] :
    (monomial s a).support = if a = 0 then ∅ else {s} := by
  /-
    R : Type u
    σ : Type u_1
    a : R
    s : Finsupp σ Nat
    inst✝ : CommSemiring R
    h : Decidable (Eq a 0)
    ⊢ Eq ((MvPolynomial.monomial s) a).support (ite (Eq a 0) EmptyCollection.empty …
  -/
  rw [← Subsingleton.elim (Classical.decEq R a 0) h]
  /-
    R : Type u
    σ : Type u_1
    a : R
    s : Finsupp σ Nat
    inst✝ : CommSemiring R
    h : Decidable (Eq a 0)
    ⊢ Eq ((MvPolynomial.monomial s) a).support (ite (Eq a 0) EmptyCollection.empty …
  -/
  rfl
  /-
    🎉 no goals
  -/
  -- Porting note: the proof in Lean 3 wasn't fundamentally better and needed `by convert rfl`
  -- the issue is the different decidability instances in the `ite` expressions


theorem support_monomial_subset : (monomial s a).support ⊆ {s} :=
  support_single_subset


theorem support_add [DecidableEq σ] : (p + q).support ⊆ p.support ∪ q.support :=
  Finsupp.support_add


theorem support_X [Nontrivial R] : (X n : MvPolynomial σ R).support = {Finsupp.single n 1} := by
  /-
    R : Type u
    σ : Type u_1
    n : σ
    inst✝¹ : CommSemiring R
    inst✝ : Nontrivial R
    ⊢ Eq (MvPolynomial.X n).support (Singleton.singleton (Finsupp.single n 1))
  -/
  classical rw [X, support_monomial, if_neg]; exact one_ne_zero
  /-
    🎉 no goals
  -/


theorem support_X_pow [Nontrivial R] (s : σ) (n : ℕ) :
    (X s ^ n : MvPolynomial σ R).support = {Finsupp.single s n} := by
  classical
    rw [X_pow_eq_monomial, support_monomial, if_neg (one_ne_zero' R)]


@[simp]
theorem support_zero : (0 : MvPolynomial σ R).support = ∅ :=
  rfl


theorem support_smul {S₁ : Type*} [SMulZeroClass S₁ R] {a : S₁} {f : MvPolynomial σ R} :
    (a • f).support ⊆ f.support :=
  Finsupp.support_smul


theorem support_sum {α : Type*} [DecidableEq σ] {s : Finset α} {f : α → MvPolynomial σ R} :
    (∑ x ∈ s, f x).support ⊆ s.biUnion fun x => (f x).support :=
  Finsupp.support_finset_sum


/-- The coefficient of the monomial `m` in the multi-variable polynomial `p`. -/
def coeff (m : σ →₀ ℕ) (p : MvPolynomial σ R) : R :=
  @DFunLike.coe ((σ →₀ ℕ) →₀ R) _ _ _ p m


@[simp]
theorem mem_support_iff {p : MvPolynomial σ R} {m : σ →₀ ℕ} : m ∈ p.support ↔ p.coeff m ≠ 0 := by
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    p : MvPolynomial σ R
    m : Finsupp σ Nat
    ⊢ Iff (Membership.mem p.support m) (Ne (MvPolynomial.coeff m p) 0)
  -/
  simp [support, coeff]
  /-
    🎉 no goals
  -/


theorem not_mem_support_iff {p : MvPolynomial σ R} {m : σ →₀ ℕ} : m ∉ p.support ↔ p.coeff m = 0 :=
     /-
       R : Type u
       σ : Type u_1
       inst✝ : CommSemiring R
       p : MvPolynomial σ R
       m : Finsupp σ Nat
       ⊢ Iff (Not (Membership.mem p.support m)) (Eq (MvPolynomial.coeff m p) 0)
     -/
  by simp
     /-
       🎉 no goals
     -/


theorem sum_def {A} [AddCommMonoid A] {p : MvPolynomial σ R} {b : (σ →₀ ℕ) → R → A} :
                                                     /-
                                                       R : Type u
                                                       σ : Type u_1
                                                       inst✝¹ : CommSemiring R
                                                       A : Type u_2
                                                       inst✝ : AddCommMonoid A
                                                       p : MvPolynomial σ R
                                                       b : Finsupp σ Nat → R → A
                                                       ⊢ Eq (Finsupp.sum p b) (p.support.sum fun m => b m (MvPolynomial.coeff m p))
                                                     -/
    p.sum b = ∑ m ∈ p.support, b m (p.coeff m) := by simp [support, Finsupp.sum, coeff]
                                                     /-
                                                       🎉 no goals
                                                     -/


theorem support_mul [DecidableEq σ] (p q : MvPolynomial σ R) :
    (p * q).support ⊆ p.support + q.support :=
  AddMonoidAlgebra.support_mul p q


@[ext]
theorem ext (p q : MvPolynomial σ R) : (∀ m, coeff m p = coeff m q) → p = q :=
  Finsupp.ext


@[simp]
theorem coeff_add (m : σ →₀ ℕ) (p q : MvPolynomial σ R) : coeff m (p + q) = coeff m p + coeff m q :=
  add_apply p q m


@[simp]
theorem coeff_smul {S₁ : Type*} [SMulZeroClass S₁ R] (m : σ →₀ ℕ) (C : S₁) (p : MvPolynomial σ R) :
    coeff m (C • p) = C • coeff m p :=
  smul_apply C p m


@[simp]
theorem coeff_zero (m : σ →₀ ℕ) : coeff m (0 : MvPolynomial σ R) = 0 :=
  rfl


@[simp]
theorem coeff_zero_X (i : σ) : coeff 0 (X i : MvPolynomial σ R) = 0 :=
                              /-
                                R : Type u
                                σ : Type u_1
                                inst✝ : CommSemiring R
                                i : σ
                                h : Eq (Finsupp.single i 1) 0
                                ⊢ False
                              -/
  single_eq_of_ne fun h => by cases Finsupp.single_eq_zero.1 h
                              /-
                                🎉 no goals
                              -/


/-- `MvPolynomial.coeff m` but promoted to an `AddMonoidHom`. -/
@[simps]
def coeffAddMonoidHom (m : σ →₀ ℕ) : MvPolynomial σ R →+ R where
  toFun := coeff m
  map_zero' := coeff_zero m
  map_add' := coeff_add m


variable (R) in
/-- `MvPolynomial.coeff m` but promoted to a `LinearMap`. -/
@[simps]
def lcoeff (m : σ →₀ ℕ) : MvPolynomial σ R →ₗ[R] R where
  toFun := coeff m
  map_add' := coeff_add m
  map_smul' := coeff_smul m


theorem coeff_sum {X : Type*} (s : Finset X) (f : X → MvPolynomial σ R) (m : σ →₀ ℕ) :
    coeff m (∑ x ∈ s, f x) = ∑ x ∈ s, coeff m (f x) :=
  map_sum (@coeffAddMonoidHom R σ _ _) _ s


theorem monic_monomial_eq (m) :
                                                                              /-
                                                                                R : Type u
                                                                                σ : Type u_1
                                                                                inst✝ : CommSemiring R
                                                                                m : Finsupp σ Nat
                                                                                ⊢ Eq ((MvPolynomial.monomial m) 1) (m.prod fun n e => HPow.hPow (MvPolynomial. …
                                                                              -/
    monomial m (1 : R) = (m.prod fun n e => X n ^ e : MvPolynomial σ R) := by simp [monomial_eq]
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


@[simp]
theorem coeff_monomial [DecidableEq σ] (m n) (a) :
    coeff m (monomial n a : MvPolynomial σ R) = if n = m then a else 0 :=
  Finsupp.single_apply


@[simp]
theorem coeff_C [DecidableEq σ] (m) (a) :
    coeff m (C a : MvPolynomial σ R) = if 0 = m then a else 0 :=
  Finsupp.single_apply


lemma eq_C_of_isEmpty [IsEmpty σ] (p : MvPolynomial σ R) :
    p = C (p.coeff 0) := by
  /-
    R : Type u
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : IsEmpty σ
    p : MvPolynomial σ R
    ⊢ Eq p (MvPolynomial.C (MvPolynomial.coeff 0 p))
  -/
  obtain ⟨x, rfl⟩ := C_surjective σ p
  /-
    case intro
    R : Type u
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : IsEmpty σ
    x : R
    ⊢ Eq (MvPolynomial.C x) (MvPolynomial.C (MvPolynomial.coeff 0 (MvPolynomial.C  …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem coeff_one [DecidableEq σ] (m) : coeff m (1 : MvPolynomial σ R) = if 0 = m then 1 else 0 :=
  coeff_C m 1


@[simp]
theorem coeff_zero_C (a) : coeff 0 (C a : MvPolynomial σ R) = a :=
  single_eq_same


@[simp]
theorem coeff_zero_one : coeff 0 (1 : MvPolynomial σ R) = 1 :=
  coeff_zero_C 1


theorem coeff_X_pow [DecidableEq σ] (i : σ) (m) (k : ℕ) :
    coeff m (X i ^ k : MvPolynomial σ R) = if Finsupp.single i k = m then 1 else 0 := by
  /-
    R : Type u
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : DecidableEq σ
    i : σ
    m : Finsupp σ Nat
    k : Nat
    ⊢ Eq (MvPolynomial.coeff m (HPow.hPow (MvPolynomial.X i) k)) (ite (Eq (Finsupp …
  -/
  have := coeff_monomial m (Finsupp.single i k) (1 : R)
  rwa [@monomial_eq _ _ (1 : R) (Finsupp.single i k) _, C_1, one_mul, Finsupp.prod_single_index]
    at this
  /-
    R : Type u
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : DecidableEq σ
    i : σ
    m : Finsupp σ Nat
    k : Nat
    this : Eq (MvPolynomial.coeff m ((Finsupp.single i k).prod fun n e => HPow.hPo …
    ⊢ Eq (HPow.hPow (MvPolynomial.X i) 0) 1
  -/
  exact pow_zero _
  /-
    🎉 no goals
  -/


theorem coeff_X' [DecidableEq σ] (i : σ) (m) :
    coeff m (X i : MvPolynomial σ R) = if Finsupp.single i 1 = m then 1 else 0 := by
  /-
    R : Type u
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : DecidableEq σ
    i : σ
    m : Finsupp σ Nat
    ⊢ Eq (MvPolynomial.coeff m (MvPolynomial.X i)) (ite (Eq (Finsupp.single i 1) m …
  -/
  rw [← coeff_X_pow, pow_one]
  /-
    🎉 no goals
  -/


@[simp]
theorem coeff_X (i : σ) : coeff (Finsupp.single i 1) (X i : MvPolynomial σ R) = 1 := by
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    i : σ
    ⊢ Eq (MvPolynomial.coeff (Finsupp.single i 1) (MvPolynomial.X i)) 1
  -/
  classical rw [coeff_X', if_pos rfl]
  /-
    🎉 no goals
  -/


@[simp]
theorem coeff_C_mul (m) (a : R) (p : MvPolynomial σ R) : coeff m (C a * p) = a * coeff m p := by
  classical
  rw [mul_def, sum_C]
  · simp +contextual [sum_def, coeff_sum]
  simp


theorem coeff_mul [DecidableEq σ] (p q : MvPolynomial σ R) (n : σ →₀ ℕ) :
    coeff n (p * q) = ∑ x ∈ Finset.antidiagonal n, coeff x.1 p * coeff x.2 q :=
  AddMonoidAlgebra.mul_apply_antidiagonal p q _ _ Finset.mem_antidiagonal


@[simp]
theorem coeff_mul_monomial (m) (s : σ →₀ ℕ) (r : R) (p : MvPolynomial σ R) :
    coeff (m + s) (p * monomial s r) = coeff m p * r :=
  AddMonoidAlgebra.mul_single_apply_aux p _ _ _ _ fun _a => add_left_inj _


@[simp]
theorem coeff_monomial_mul (m) (s : σ →₀ ℕ) (r : R) (p : MvPolynomial σ R) :
    coeff (s + m) (monomial s r * p) = r * coeff m p :=
  AddMonoidAlgebra.single_mul_apply_aux p _ _ _ _ fun _a => add_right_inj _


@[simp]
theorem coeff_mul_X (m) (s : σ) (p : MvPolynomial σ R) :
    coeff (m + Finsupp.single s 1) (p * X s) = coeff m p :=
  (coeff_mul_monomial _ _ _ _).trans (mul_one _)


@[simp]
theorem coeff_X_mul (m) (s : σ) (p : MvPolynomial σ R) :
    coeff (Finsupp.single s 1 + m) (X s * p) = coeff m p :=
  (coeff_monomial_mul _ _ _ _).trans (one_mul _)


lemma coeff_single_X_pow [DecidableEq σ] (s s' : σ) (n n' : ℕ) :
    (X (R := R) s ^ n).coeff (Finsupp.single s' n')
    = if s = s' ∧ n = n' ∨ n = 0 ∧ n' = 0 then 1 else 0 := by
  /-
    R : Type u
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : DecidableEq σ
    s s' : σ
    n n' : Nat
    ⊢ Eq (MvPolynomial.coeff (Finsupp.single s' n') (HPow.hPow (MvPolynomial.X s)  …
  -/
  simp only [coeff_X_pow, single_eq_single_iff]
  /-
    🎉 no goals
  -/


@[simp]
lemma coeff_single_X [DecidableEq σ] (s s' : σ) (n : ℕ) :
    (X s).coeff (R := R) (Finsupp.single s' n) = if n = 1 ∧ s = s' then 1 else 0 := by
  /-
    R : Type u
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : DecidableEq σ
    s s' : σ
    n : Nat
    ⊢ Eq (MvPolynomial.coeff (Finsupp.single s' n) (MvPolynomial.X s)) (ite (And ( …
  -/
  simpa [eq_comm, and_comm] using coeff_single_X_pow s s' 1 n
  /-
    🎉 no goals
  -/


@[simp]
theorem support_mul_X (s : σ) (p : MvPolynomial σ R) :
    (p * X s).support = p.support.map (addRightEmbedding (Finsupp.single s 1)) :=
                                              /-
                                                R : Type u
                                                σ : Type u_1
                                                inst✝ : CommSemiring R
                                                s : σ
                                                p : MvPolynomial σ R
                                                ⊢ ∀ (y : R), Iff (Eq (HMul.hMul y 1) 0) (Eq y 0)
                                              -/
  AddMonoidAlgebra.support_mul_single p _ (by simp) _
                                              /-
                                                🎉 no goals
                                              -/


@[simp]
theorem support_X_mul (s : σ) (p : MvPolynomial σ R) :
    (X s * p).support = p.support.map (addLeftEmbedding (Finsupp.single s 1)) :=
                                              /-
                                                R : Type u
                                                σ : Type u_1
                                                inst✝ : CommSemiring R
                                                s : σ
                                                p : MvPolynomial σ R
                                                ⊢ ∀ (y : R), Iff (Eq (HMul.hMul 1 y) 0) (Eq y 0)
                                              -/
  AddMonoidAlgebra.support_single_mul p _ (by simp) _
                                              /-
                                                🎉 no goals
                                              -/


@[simp]
theorem support_smul_eq {S₁ : Type*} [Semiring S₁] [Module S₁ R] [NoZeroSMulDivisors S₁ R] {a : S₁}
    (h : a ≠ 0) (p : MvPolynomial σ R) : (a • p).support = p.support :=
  Finsupp.support_smul_eq h


theorem support_sdiff_support_subset_support_add [DecidableEq σ] (p q : MvPolynomial σ R) :
    p.support \ q.support ⊆ (p + q).support := by
  /-
    R : Type u
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : DecidableEq σ
    p q : MvPolynomial σ R
    ⊢ HasSubset.Subset (SDiff.sdiff p.support q.support) (HAdd.hAdd p q).support
  -/
  intro m hm
  /-
    R : Type u
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : DecidableEq σ
    p q : MvPolynomial σ R
    m : Finsupp σ Nat
    hm : Membership.mem (SDiff.sdiff p.support q.support) m
    ⊢ Membership.mem (HAdd.hAdd p q).support m
  -/
  simp only [Classical.not_not, mem_support_iff, Finset.mem_sdiff, Ne] at hm
  /-
    R : Type u
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : DecidableEq σ
    p q : MvPolynomial σ R
    m : Finsupp σ Nat
    hm : And (Not (Eq (MvPolynomial.coeff m p) 0)) (Eq (MvPolynomial.coeff m q) 0)
    ⊢ Membership.mem (HAdd.hAdd p q).support m
  -/
  simp [hm.2, hm.1]
  /-
    🎉 no goals
  -/


open scoped symmDiff in
theorem support_symmDiff_support_subset_support_add [DecidableEq σ] (p q : MvPolynomial σ R) :
    p.support ∆ q.support ⊆ (p + q).support := by
  /-
    R : Type u
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : DecidableEq σ
    p q : MvPolynomial σ R
    ⊢ HasSubset.Subset (symmDiff p.support q.support) (HAdd.hAdd p q).support
  -/
  rw [symmDiff_def, Finset.sup_eq_union]
  /-
    R : Type u
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : DecidableEq σ
    p q : MvPolynomial σ R
    ⊢ HasSubset.Subset (Union.union (SDiff.sdiff p.support q.support) (SDiff.sdiff …
  -/
  apply Finset.union_subset
    /-
      case hs
      R : Type u
      σ : Type u_1
      inst✝¹ : CommSemiring R
      inst✝ : DecidableEq σ
      p q : MvPolynomial σ R
      ⊢ HasSubset.Subset (SDiff.sdiff p.support q.support) (HAdd.hAdd p q).support
    -/
  · exact support_sdiff_support_subset_support_add p q
    /-
      🎉 no goals
    -/
    /-
      case a
      R : Type u
      σ : Type u_1
      inst✝¹ : CommSemiring R
      inst✝ : DecidableEq σ
      p q : MvPolynomial σ R
      ⊢ HasSubset.Subset (SDiff.sdiff q.support p.support) (HAdd.hAdd p q).support
    -/
  · rw [add_comm]
    /-
      case a
      R : Type u
      σ : Type u_1
      inst✝¹ : CommSemiring R
      inst✝ : DecidableEq σ
      p q : MvPolynomial σ R
      ⊢ HasSubset.Subset (SDiff.sdiff q.support p.support) (HAdd.hAdd q p).support
    -/
    exact support_sdiff_support_subset_support_add q p
    /-
      🎉 no goals
    -/


theorem coeff_mul_monomial' (m) (s : σ →₀ ℕ) (r : R) (p : MvPolynomial σ R) :
    coeff m (p * monomial s r) = if s ≤ m then coeff (m - s) p * r else 0 := by
  classical
  split_ifs with h
  · conv_rhs => rw [← coeff_mul_monomial _ s]
    congr with t
    rw [tsub_add_cancel_of_le h]
  · contrapose! h
    rw [← mem_support_iff] at h
    obtain ⟨j, -, rfl⟩ : ∃ j ∈ support p, j + s = m := by
      simpa [Finset.mem_add]
        using Finset.add_subset_add_left support_monomial_subset <| support_mul _ _ h
    exact le_add_left le_rfl


theorem coeff_monomial_mul' (m) (s : σ →₀ ℕ) (r : R) (p : MvPolynomial σ R) :
    coeff m (monomial s r * p) = if s ≤ m then r * coeff (m - s) p else 0 := by
  -- note that if we allow `R` to be non-commutative we will have to duplicate the proof above.
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    m s : Finsupp σ Nat
    r : R
    p : MvPolynomial σ R
    ⊢ Eq (MvPolynomial.coeff m (HMul.hMul ((MvPolynomial.monomial s) r) p)) (ite ( …
  -/
  rw [mul_comm, mul_comm r]
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    m s : Finsupp σ Nat
    r : R
    p : MvPolynomial σ R
    ⊢ Eq (MvPolynomial.coeff m (HMul.hMul p ((MvPolynomial.monomial s) r))) (ite ( …
  -/
  exact coeff_mul_monomial' _ _ _ _
  /-
    🎉 no goals
  -/


theorem coeff_mul_X' [DecidableEq σ] (m) (s : σ) (p : MvPolynomial σ R) :
    coeff m (p * X s) = if s ∈ m.support then coeff (m - Finsupp.single s 1) p else 0 := by
  /-
    R : Type u
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : DecidableEq σ
    m : Finsupp σ Nat
    s : σ
    p : MvPolynomial σ R
    ⊢ Eq (MvPolynomial.coeff m (HMul.hMul p (MvPolynomial.X s))) (ite (Membership. …
  -/
  refine (coeff_mul_monomial' _ _ _ _).trans ?_
  simp_rw [Finsupp.single_le_iff, Finsupp.mem_support_iff, Nat.succ_le_iff, pos_iff_ne_zero,
    mul_one]


theorem coeff_X_mul' [DecidableEq σ] (m) (s : σ) (p : MvPolynomial σ R) :
    coeff m (X s * p) = if s ∈ m.support then coeff (m - Finsupp.single s 1) p else 0 := by
  /-
    R : Type u
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : DecidableEq σ
    m : Finsupp σ Nat
    s : σ
    p : MvPolynomial σ R
    ⊢ Eq (MvPolynomial.coeff m (HMul.hMul (MvPolynomial.X s) p)) (ite (Membership. …
  -/
  refine (coeff_monomial_mul' _ _ _ _).trans ?_
  simp_rw [Finsupp.single_le_iff, Finsupp.mem_support_iff, Nat.succ_le_iff, pos_iff_ne_zero,
    one_mul]


theorem eq_zero_iff {p : MvPolynomial σ R} : p = 0 ↔ ∀ d, coeff d p = 0 := by
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    p : MvPolynomial σ R
    ⊢ Iff (Eq p 0) (∀ (d : Finsupp σ Nat), Eq (MvPolynomial.coeff d p) 0)
  -/
  rw [MvPolynomial.ext_iff]
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    p : MvPolynomial σ R
    ⊢ Iff (∀ (m : Finsupp σ Nat), Eq (MvPolynomial.coeff m p) (MvPolynomial.coeff  …
  -/
  simp only [coeff_zero]
  /-
    🎉 no goals
  -/


theorem ne_zero_iff {p : MvPolynomial σ R} : p ≠ 0 ↔ ∃ d, coeff d p ≠ 0 := by
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    p : MvPolynomial σ R
    ⊢ Iff (Ne p 0) (Exists fun d => Ne (MvPolynomial.coeff d p) 0)
  -/
  rw [Ne, eq_zero_iff]
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    p : MvPolynomial σ R
    ⊢ Iff (Not (∀ (d : Finsupp σ Nat), Eq (MvPolynomial.coeff d p) 0)) (Exists fun …
  -/
  push_neg
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    p : MvPolynomial σ R
    ⊢ Iff (Exists fun d => Ne (MvPolynomial.coeff d p) 0) (Exists fun d => Ne (MvP …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem X_ne_zero [Nontrivial R] (s : σ) :
    X (R := R) s ≠ 0 := by
  /-
    R : Type u
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : Nontrivial R
    s : σ
    ⊢ Ne (MvPolynomial.X s) 0
  -/
  rw [ne_zero_iff]
  /-
    R : Type u
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : Nontrivial R
    s : σ
    ⊢ Exists fun d => Ne (MvPolynomial.coeff d (MvPolynomial.X s)) 0
  -/
  use Finsupp.single s 1
  /-
    case h
    R : Type u
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : Nontrivial R
    s : σ
    ⊢ Ne (MvPolynomial.coeff (Finsupp.single s 1) (MvPolynomial.X s)) 0
  -/
  simp only [coeff_X, ne_eq, one_ne_zero, not_false_eq_true]
  /-
    🎉 no goals
  -/


@[simp]
theorem support_eq_empty {p : MvPolynomial σ R} : p.support = ∅ ↔ p = 0 :=
  Finsupp.support_eq_empty


@[simp]
lemma support_nonempty {p : MvPolynomial σ R} : p.support.Nonempty ↔ p ≠ 0 := by
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    p : MvPolynomial σ R
    ⊢ Iff p.support.Nonempty (Ne p 0)
  -/
  rw [Finset.nonempty_iff_ne_empty, ne_eq, support_eq_empty]
  /-
    🎉 no goals
  -/


theorem exists_coeff_ne_zero {p : MvPolynomial σ R} (h : p ≠ 0) : ∃ d, coeff d p ≠ 0 :=
  ne_zero_iff.mp h


theorem C_dvd_iff_dvd_coeff (r : R) (φ : MvPolynomial σ R) : C r ∣ φ ↔ ∀ i, r ∣ φ.coeff i := by
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    r : R
    φ : MvPolynomial σ R
    ⊢ Iff (Dvd.dvd (MvPolynomial.C r) φ) (∀ (i : Finsupp σ Nat), Dvd.dvd r (MvPoly …
  -/
  constructor
    /-
      case mp
      R : Type u
      σ : Type u_1
      inst✝ : CommSemiring R
      r : R
      φ : MvPolynomial σ R
      ⊢ Dvd.dvd (MvPolynomial.C r) φ → ∀ (i : Finsupp σ Nat), Dvd.dvd r (MvPolynomia …
    -/
  · rintro ⟨φ, rfl⟩ c
    /-
      case mp.intro
      R : Type u
      σ : Type u_1
      inst✝ : CommSemiring R
      r : R
      φ : MvPolynomial σ R
      c : Finsupp σ Nat
      ⊢ Dvd.dvd r (MvPolynomial.coeff c (HMul.hMul (MvPolynomial.C r) φ))
    -/
    rw [coeff_C_mul]
    /-
      case mp.intro
      R : Type u
      σ : Type u_1
      inst✝ : CommSemiring R
      r : R
      φ : MvPolynomial σ R
      c : Finsupp σ Nat
      ⊢ Dvd.dvd r (HMul.hMul r (MvPolynomial.coeff c φ))
    -/
    apply dvd_mul_right
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u
      σ : Type u_1
      inst✝ : CommSemiring R
      r : R
      φ : MvPolynomial σ R
      ⊢ (∀ (i : Finsupp σ Nat), Dvd.dvd r (MvPolynomial.coeff i φ)) → Dvd.dvd (MvPol …
    -/
  · intro h
    /-
      case mpr
      R : Type u
      σ : Type u_1
      inst✝ : CommSemiring R
      r : R
      φ : MvPolynomial σ R
      h : ∀ (i : Finsupp σ Nat), Dvd.dvd r (MvPolynomial.coeff i φ)
      ⊢ Dvd.dvd (MvPolynomial.C r) φ
    -/
    choose C hc using h
    classical
      let c' : (σ →₀ ℕ) → R := fun i => if i ∈ φ.support then C i else 0
      let ψ : MvPolynomial σ R := ∑ i ∈ φ.support, monomial i (c' i)
      use ψ
      apply MvPolynomial.ext
      intro i
      simp only [ψ, c', coeff_C_mul, coeff_sum, coeff_monomial, Finset.sum_ite_eq']
      split_ifs with hi
      · rw [hc]
      · rw [not_mem_support_iff] at hi
        rwa [mul_zero]


@[simp] lemma isRegular_X : IsRegular (X n : MvPolynomial σ R) := by
  suffices IsLeftRegular (X n : MvPolynomial σ R) from
    ⟨this, this.right_of_commute <| Commute.all _⟩
  /-
    R : Type u
    σ : Type u_1
    n : σ
    inst✝ : CommSemiring R
    ⊢ IsLeftRegular (MvPolynomial.X n)
  -/
  intro P Q (hPQ : (X n) * P = (X n) * Q)
  /-
    R : Type u
    σ : Type u_1
    n : σ
    inst✝ : CommSemiring R
    P Q : MvPolynomial σ R
    hPQ : Eq (HMul.hMul (MvPolynomial.X n) P) (HMul.hMul (MvPolynomial.X n) Q)
    ⊢ Eq P Q
  -/
  ext i
  /-
    case a
    R : Type u
    σ : Type u_1
    n : σ
    inst✝ : CommSemiring R
    P Q : MvPolynomial σ R
    hPQ : Eq (HMul.hMul (MvPolynomial.X n) P) (HMul.hMul (MvPolynomial.X n) Q)
    i : Finsupp σ Nat
    ⊢ Eq (MvPolynomial.coeff i P) (MvPolynomial.coeff i Q)
  -/
  rw [← coeff_X_mul i n P, hPQ, coeff_X_mul i n Q]
  /-
    🎉 no goals
  -/


@[simp] lemma isRegular_X_pow (k : ℕ) : IsRegular (X n ^ k : MvPolynomial σ R) := isRegular_X.pow k


@[simp] lemma isRegular_prod_X (s : Finset σ) :
    IsRegular (∏ n ∈ s, X n : MvPolynomial σ R) :=
  IsRegular.prod fun _ _ ↦ isRegular_X


/-- The finset of nonzero coefficients of a multivariate polynomial. -/
def coeffs (p : MvPolynomial σ R) : Finset R :=
  letI := Classical.decEq R
  Finset.image p.coeff p.support


@[simp]
lemma coeffs_zero : coeffs (0 : MvPolynomial σ R) = ∅ :=
  rfl


lemma coeffs_one : coeffs (1 : MvPolynomial σ R) ⊆ {1} := by
  classical
    rw [coeffs, Finset.image_subset_iff]
    simp_all [coeff_one]


@[nontriviality]
lemma coeffs_eq_empty_of_subsingleton [Subsingleton R] (p : MvPolynomial σ R) : p.coeffs = ∅ := by
  /-
    R : Type u
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : Subsingleton R
    p : MvPolynomial σ R
    ⊢ Eq p.coeffs EmptyCollection.emptyCollection
  -/
  simpa [coeffs] using Subsingleton.eq_zero p
  /-
    🎉 no goals
  -/


@[simp]
lemma coeffs_one_of_nontrivial [Nontrivial R] : coeffs (1 : MvPolynomial σ R) = {1} := by
  /-
    R : Type u
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : Nontrivial R
    ⊢ Eq (MvPolynomial.coeffs 1) (Singleton.singleton 1)
  -/
  apply Finset.Subset.antisymm coeffs_one
  /-
    R : Type u
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : Nontrivial R
    ⊢ HasSubset.Subset (Singleton.singleton 1) (MvPolynomial.coeffs 1)
  -/
  simp only [coeffs, Finset.singleton_subset_iff, Finset.mem_image]
  /-
    R : Type u
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : Nontrivial R
    ⊢ Exists fun a => And (Membership.mem (MvPolynomial.support 1) a) (Eq (MvPolyn …
  -/
  exact ⟨0, by simp⟩
  /-
    🎉 no goals
  -/


lemma mem_coeffs_iff {p : MvPolynomial σ R} {c : R} :
    c ∈ p.coeffs ↔ ∃ n ∈ p.support, c = p.coeff n := by
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    p : MvPolynomial σ R
    c : R
    ⊢ Iff (Membership.mem p.coeffs c) (Exists fun n => And (Membership.mem p.suppo …
  -/
  simp [coeffs, eq_comm, (Finset.mem_image)]
  /-
    🎉 no goals
  -/


lemma coeff_mem_coeffs {p : MvPolynomial σ R} (m : σ →₀ ℕ)
    (h : p.coeff m ≠ 0) : p.coeff m ∈ p.coeffs :=
  letI := Classical.decEq R
  Finset.mem_image_of_mem p.coeff (mem_support_iff.mpr h)


lemma zero_not_mem_coeffs (p : MvPolynomial σ R) : 0 ∉ p.coeffs := by
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    p : MvPolynomial σ R
    ⊢ Not (Membership.mem p.coeffs 0)
  -/
  intro hz
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    p : MvPolynomial σ R
    hz : Membership.mem p.coeffs 0
    ⊢ False
  -/
  obtain ⟨n, hnsupp, hn⟩ := mem_coeffs_iff.mp hz
  /-
    case intro.intro
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    p : MvPolynomial σ R
    hz : Membership.mem p.coeffs 0
    n : Finsupp σ Nat
    hnsupp : Membership.mem p.support n
    hn : Eq 0 (MvPolynomial.coeff n p)
    ⊢ False
  -/
  exact (mem_support_iff.mp hnsupp) hn.symm
  /-
    🎉 no goals
  -/


/-- `constantCoeff p` returns the constant term of the polynomial `p`, defined as `coeff 0 p`.
This is a ring homomorphism.
-/
def constantCoeff : MvPolynomial σ R →+* R where
  toFun := coeff 0
                 /-
                   R : Type u
                   S₁ : Type v
                   S₂ : Type w
                   S₃ : Type x
                   σ : Type u_1
                   a a' a₁ a₂ : R
                   e : Nat
                   n m : σ
                   s : Finsupp σ Nat
                   inst✝¹ : CommSemiring R
                   inst✝ : CommSemiring S₁
                   p q : MvPolynomial σ R
                   ⊢ Eq (MvPolynomial.coeff 0 1) 1
                 -/
  map_one' := by simp [AddMonoidAlgebra.one_def]
                 /-
                   🎉 no goals
                 -/
                 /-
                   R : Type u
                   S₁ : Type v
                   S₂ : Type w
                   S₃ : Type x
                   σ : Type u_1
                   a a' a₁ a₂ : R
                   e : Nat
                   n m : σ
                   s : Finsupp σ Nat
                   inst✝¹ : CommSemiring R
                   inst✝ : CommSemiring S₁
                   p q : MvPolynomial σ R
                   ⊢ ∀ (x y : MvPolynomial σ R), Eq ({ toFun := MvPolynomial.coeff 0, map_one' := …
                 -/
  map_mul' := by classical simp [coeff_mul, Finsupp.support_single_ne_zero]
                 /-
                   🎉 no goals
                 -/
  map_zero' := coeff_zero _
  map_add' := coeff_add _


theorem constantCoeff_eq : (constantCoeff : MvPolynomial σ R → R) = coeff 0 :=
  rfl


@[simp]
theorem constantCoeff_C (r : R) : constantCoeff (C r : MvPolynomial σ R) = r := by
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    r : R
    ⊢ Eq (MvPolynomial.constantCoeff (MvPolynomial.C r)) r
  -/
  classical simp [constantCoeff_eq]
  /-
    🎉 no goals
  -/


@[simp]
theorem constantCoeff_X (i : σ) : constantCoeff (X i : MvPolynomial σ R) = 0 := by
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    i : σ
    ⊢ Eq (MvPolynomial.constantCoeff (MvPolynomial.X i)) 0
  -/
  simp [constantCoeff_eq]
  /-
    🎉 no goals
  -/


@[simp]
theorem constantCoeff_smul {R : Type*} [SMulZeroClass R S₁] (a : R) (f : MvPolynomial σ S₁) :
    constantCoeff (a • f) = a • constantCoeff f :=
  rfl


theorem constantCoeff_monomial [DecidableEq σ] (d : σ →₀ ℕ) (r : R) :
    constantCoeff (monomial d r) = if d = 0 then r else 0 := by
  /-
    R : Type u
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : DecidableEq σ
    d : Finsupp σ Nat
    r : R
    ⊢ Eq (MvPolynomial.constantCoeff ((MvPolynomial.monomial d) r)) (ite (Eq d 0)  …
  -/
  rw [constantCoeff_eq, coeff_monomial]
  /-
    🎉 no goals
  -/


@[simp]
theorem constantCoeff_comp_C : constantCoeff.comp (C : R →+* MvPolynomial σ R) = RingHom.id R := by
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    ⊢ Eq (MvPolynomial.constantCoeff.comp MvPolynomial.C) (RingHom.id R)
  -/
  ext x
  /-
    case a
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    x : R
    ⊢ Eq ((MvPolynomial.constantCoeff.comp MvPolynomial.C) x) ((RingHom.id R) x)
  -/
  exact constantCoeff_C σ x
  /-
    🎉 no goals
  -/


theorem constantCoeff_comp_algebraMap :
    constantCoeff.comp (algebraMap R (MvPolynomial σ R)) = RingHom.id R :=
  constantCoeff_comp_C _ _


@[simp]
theorem support_sum_monomial_coeff (p : MvPolynomial σ R) :
    (∑ v ∈ p.support, monomial v (coeff v p)) = p :=
  Finsupp.sum_single p


theorem as_sum (p : MvPolynomial σ R) : p = ∑ v ∈ p.support, monomial v (coeff v p) :=
  (support_sum_monomial_coeff p).symm


/-- Evaluate a polynomial `p` given a valuation `g` of all the variables
  and a ring hom `f` from the scalar ring to the target -/
def eval₂ (p : MvPolynomial σ R) : S₁ :=
  p.sum fun s a => f a * s.prod fun n e => g n ^ e


theorem eval₂_eq (g : R →+* S₁) (X : σ → S₁) (f : MvPolynomial σ R) :
    f.eval₂ g X = ∑ d ∈ f.support, g (f.coeff d) * ∏ i ∈ d.support, X i ^ d i :=
  rfl


theorem eval₂_eq' [Fintype σ] (g : R →+* S₁) (X : σ → S₁) (f : MvPolynomial σ R) :
    f.eval₂ g X = ∑ d ∈ f.support, g (f.coeff d) * ∏ i, X i ^ d i := by
  /-
    R : Type u
    S₁ : Type v
    σ : Type u_1
    inst✝² : CommSemiring R
    inst✝¹ : CommSemiring S₁
    inst✝ : Fintype σ
    g : RingHom R S₁
    X : σ → S₁
    f : MvPolynomial σ R
    ⊢ Eq (MvPolynomial.eval₂ g X f) (f.support.sum fun d => HMul.hMul (g (MvPolyno …
  -/
  simp only [eval₂_eq, ← Finsupp.prod_pow]
  /-
    R : Type u
    S₁ : Type v
    σ : Type u_1
    inst✝² : CommSemiring R
    inst✝¹ : CommSemiring S₁
    inst✝ : Fintype σ
    g : RingHom R S₁
    X : σ → S₁
    f : MvPolynomial σ R
    ⊢ Eq (f.support.sum fun d => HMul.hMul (g (MvPolynomial.coeff d f)) (d.support …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem eval₂_zero : (0 : MvPolynomial σ R).eval₂ f g = 0 :=
  Finsupp.sum_zero_index


@[simp]
theorem eval₂_add : (p + q).eval₂ f g = p.eval₂ f g + q.eval₂ f g := by
  /-
    R : Type u
    S₁ : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S₁
    p q : MvPolynomial σ R
    f : RingHom R S₁
    g : σ → S₁
    ⊢ Eq (MvPolynomial.eval₂ f g (HAdd.hAdd p q)) (HAdd.hAdd (MvPolynomial.eval₂ f …
  -/
  classical exact Finsupp.sum_add_index (by simp [f.map_zero]) (by simp [add_mul, f.map_add])
  /-
    🎉 no goals
  -/


@[simp]
theorem eval₂_monomial : (monomial s a).eval₂ f g = f a * s.prod fun n e => g n ^ e :=
                               /-
                                 R : Type u
                                 S₁ : Type v
                                 σ : Type u_1
                                 a : R
                                 s : Finsupp σ Nat
                                 inst✝¹ : CommSemiring R
                                 inst✝ : CommSemiring S₁
                                 f : RingHom R S₁
                                 g : σ → S₁
                                 ⊢ Eq (HMul.hMul (f 0) (s.prod fun n e => HPow.hPow (g n) e)) 0
                               -/
  Finsupp.sum_single_index (by simp [f.map_zero])
                               /-
                                 🎉 no goals
                               -/


@[simp]
theorem eval₂_C (a) : (C a).eval₂ f g = f a := by
  /-
    R : Type u
    S₁ : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S₁
    f : RingHom R S₁
    g : σ → S₁
    a : R
    ⊢ Eq (MvPolynomial.eval₂ f g (MvPolynomial.C a)) (f a)
  -/
  rw [C_apply, eval₂_monomial, prod_zero_index, mul_one]
  /-
    🎉 no goals
  -/


@[simp]
theorem eval₂_one : (1 : MvPolynomial σ R).eval₂ f g = 1 :=
  (eval₂_C _ _ _).trans f.map_one


@[simp] theorem eval₂_natCast (n : Nat) : (n : MvPolynomial σ R).eval₂ f g = n :=
  (eval₂_C _ _ _).trans (map_natCast f n)


@[simp] theorem eval₂_ofNat (n : Nat) [n.AtLeastTwo] :
    (ofNat(n) : MvPolynomial σ R).eval₂ f g = ofNat(n) :=
  eval₂_natCast f g n


@[simp]
theorem eval₂_X (n) : (X n).eval₂ f g = g n := by
  /-
    R : Type u
    S₁ : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S₁
    f : RingHom R S₁
    g : σ → S₁
    n : σ
    ⊢ Eq (MvPolynomial.eval₂ f g (MvPolynomial.X n)) (g n)
  -/
  simp [eval₂_monomial, f.map_one, X, prod_single_index, pow_one]
  /-
    🎉 no goals
  -/


theorem eval₂_mul_monomial :
    ∀ {s a}, (p * monomial s a).eval₂ f g = p.eval₂ f g * f a * s.prod fun n e => g n ^ e := by
  classical
  apply MvPolynomial.induction_on p
  · intro a' s a
    simp [C_mul_monomial, eval₂_monomial, f.map_mul]
  · intro p q ih_p ih_q
    simp [add_mul, eval₂_add, ih_p, ih_q]
  · intro p n ih s a
    exact
      calc (p * X n * monomial s a).eval₂ f g
        _ = (p * monomial (Finsupp.single n 1 + s) a).eval₂ f g := by
          rw [monomial_single_add, pow_one, mul_assoc]
        _ = (p * monomial (Finsupp.single n 1) 1).eval₂ f g * f a * s.prod fun n e => g n ^ e := by
          simp [ih, prod_single_index, prod_add_index, pow_one, pow_add, mul_assoc, mul_left_comm,
            f.map_one]


theorem eval₂_mul_C : (p * C a).eval₂ f g = p.eval₂ f g * f a :=
                                       /-
                                         R : Type u
                                         S₁ : Type v
                                         σ : Type u_1
                                         a : R
                                         inst✝¹ : CommSemiring R
                                         inst✝ : CommSemiring S₁
                                         p : MvPolynomial σ R
                                         f : RingHom R S₁
                                         g : σ → S₁
                                         ⊢ Eq (HMul.hMul (HMul.hMul (MvPolynomial.eval₂ f g p) (f a)) (Finsupp.prod 0 f …
                                       -/
  (eval₂_mul_monomial _ _).trans <| by simp
                                       /-
                                         🎉 no goals
                                       -/


@[simp]
theorem eval₂_mul : ∀ {p}, (p * q).eval₂ f g = p.eval₂ f g * q.eval₂ f g := by
  /-
    R : Type u
    S₁ : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S₁
    q : MvPolynomial σ R
    f : RingHom R S₁
    g : σ → S₁
    ⊢ ∀ {p : MvPolynomial σ R}, Eq (MvPolynomial.eval₂ f g (HMul.hMul p q)) (HMul. …
  -/
  apply MvPolynomial.induction_on q
    /-
      case h_C
      R : Type u
      S₁ : Type v
      σ : Type u_1
      inst✝¹ : CommSemiring R
      inst✝ : CommSemiring S₁
      q : MvPolynomial σ R
      f : RingHom R S₁
      g : σ → S₁
      ⊢ ∀ (a : R) {p : MvPolynomial σ R}, Eq (MvPolynomial.eval₂ f g (HMul.hMul p (M …
    -/
  · simp [eval₂_C, eval₂_mul_C]
    /-
      🎉 no goals
    -/
    /-
      case h_add
      R : Type u
      S₁ : Type v
      σ : Type u_1
      inst✝¹ : CommSemiring R
      inst✝ : CommSemiring S₁
      q : MvPolynomial σ R
      f : RingHom R S₁
      g : σ → S₁
      ⊢ ∀ (p q : MvPolynomial σ R), (∀ {p_1 : MvPolynomial σ R}, Eq (MvPolynomial.ev …
    -/
  · simp +contextual [mul_add, eval₂_add]
    /-
      🎉 no goals
    -/
    /-
      case h_X
      R : Type u
      S₁ : Type v
      σ : Type u_1
      inst✝¹ : CommSemiring R
      inst✝ : CommSemiring S₁
      q : MvPolynomial σ R
      f : RingHom R S₁
      g : σ → S₁
      ⊢ ∀ (p : MvPolynomial σ R) (n : σ), (∀ {p_1 : MvPolynomial σ R}, Eq (MvPolynom …
    -/
  · simp +contextual [X, eval₂_monomial, eval₂_mul_monomial, ← mul_assoc]
    /-
      🎉 no goals
    -/


@[simp]
theorem eval₂_pow {p : MvPolynomial σ R} : ∀ {n : ℕ}, (p ^ n).eval₂ f g = p.eval₂ f g ^ n
  | 0 => by
    /-
      R : Type u
      S₁ : Type v
      σ : Type u_1
      inst✝¹ : CommSemiring R
      inst✝ : CommSemiring S₁
      f : RingHom R S₁
      g : σ → S₁
      p : MvPolynomial σ R
      ⊢ Eq (MvPolynomial.eval₂ f g (HPow.hPow p 0)) (HPow.hPow (MvPolynomial.eval₂ f …
    -/
    rw [pow_zero, pow_zero]
    /-
      R : Type u
      S₁ : Type v
      σ : Type u_1
      inst✝¹ : CommSemiring R
      inst✝ : CommSemiring S₁
      f : RingHom R S₁
      g : σ → S₁
      p : MvPolynomial σ R
      ⊢ Eq (MvPolynomial.eval₂ f g 1) 1
    -/
    exact eval₂_one _ _
    /-
      🎉 no goals
    -/
                /-
                  R : Type u
                  S₁ : Type v
                  σ : Type u_1
                  inst✝¹ : CommSemiring R
                  inst✝ : CommSemiring S₁
                  f : RingHom R S₁
                  g : σ → S₁
                  p : MvPolynomial σ R
                  n : Nat
                  ⊢ Eq (MvPolynomial.eval₂ f g (HPow.hPow p (HAdd.hAdd n 1))) (HPow.hPow (MvPoly …
                -/
  | n + 1 => by rw [pow_add, pow_one, pow_add, pow_one, eval₂_mul, eval₂_pow]
                /-
                  🎉 no goals
                -/


/-- `MvPolynomial.eval₂` as a `RingHom`. -/
def eval₂Hom (f : R →+* S₁) (g : σ → S₁) : MvPolynomial σ R →+* S₁ where
  toFun := eval₂ f g
  map_one' := eval₂_one _ _
  map_mul' _ _ := eval₂_mul _ _
  map_zero' := eval₂_zero f g
  map_add' _ _ := eval₂_add _ _


@[simp]
theorem coe_eval₂Hom (f : R →+* S₁) (g : σ → S₁) : ⇑(eval₂Hom f g) = eval₂ f g :=
  rfl


theorem eval₂Hom_congr {f₁ f₂ : R →+* S₁} {g₁ g₂ : σ → S₁} {p₁ p₂ : MvPolynomial σ R} :
    f₁ = f₂ → g₁ = g₂ → p₁ = p₂ → eval₂Hom f₁ g₁ p₁ = eval₂Hom f₂ g₂ p₂ := by
  /-
    R : Type u
    S₁ : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S₁
    f₁ f₂ : RingHom R S₁
    g₁ g₂ : σ → S₁
    p₁ p₂ : MvPolynomial σ R
    ⊢ Eq f₁ f₂ → Eq g₁ g₂ → Eq p₁ p₂ → Eq ((MvPolynomial.eval₂Hom f₁ g₁) p₁) ((MvP …
  -/
  rintro rfl rfl rfl; rfl
                      /-
                        🎉 no goals
                      -/


@[simp]
theorem eval₂Hom_C (f : R →+* S₁) (g : σ → S₁) (r : R) : eval₂Hom f g (C r) = f r :=
  eval₂_C f g r


@[simp]
theorem eval₂Hom_X' (f : R →+* S₁) (g : σ → S₁) (i : σ) : eval₂Hom f g (X i) = g i :=
  eval₂_X f g i


@[simp]
theorem comp_eval₂Hom [CommSemiring S₂] (f : R →+* S₁) (g : σ → S₁) (φ : S₁ →+* S₂) :
    φ.comp (eval₂Hom f g) = eval₂Hom (φ.comp f) fun i => φ (g i) := by
  /-
    R : Type u
    S₁ : Type v
    S₂ : Type w
    σ : Type u_1
    inst✝² : CommSemiring R
    inst✝¹ : CommSemiring S₁
    inst✝ : CommSemiring S₂
    f : RingHom R S₁
    g : σ → S₁
    φ : RingHom S₁ S₂
    ⊢ Eq (φ.comp (MvPolynomial.eval₂Hom f g)) (MvPolynomial.eval₂Hom (φ.comp f) fu …
  -/
  apply MvPolynomial.ringHom_ext
    /-
      case hC
      R : Type u
      S₁ : Type v
      S₂ : Type w
      σ : Type u_1
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring S₁
      inst✝ : CommSemiring S₂
      f : RingHom R S₁
      g : σ → S₁
      φ : RingHom S₁ S₂
      ⊢ ∀ (r : R), Eq ((φ.comp (MvPolynomial.eval₂Hom f g)) (MvPolynomial.C r)) ((Mv …
    -/
  · intro r
    /-
      case hC
      R : Type u
      S₁ : Type v
      S₂ : Type w
      σ : Type u_1
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring S₁
      inst✝ : CommSemiring S₂
      f : RingHom R S₁
      g : σ → S₁
      φ : RingHom S₁ S₂
      r : R
      ⊢ Eq ((φ.comp (MvPolynomial.eval₂Hom f g)) (MvPolynomial.C r)) ((MvPolynomial. …
    -/
    rw [RingHom.comp_apply, eval₂Hom_C, eval₂Hom_C, RingHom.comp_apply]
    /-
      🎉 no goals
    -/
    /-
      case hX
      R : Type u
      S₁ : Type v
      S₂ : Type w
      σ : Type u_1
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring S₁
      inst✝ : CommSemiring S₂
      f : RingHom R S₁
      g : σ → S₁
      φ : RingHom S₁ S₂
      ⊢ ∀ (i : σ), Eq ((φ.comp (MvPolynomial.eval₂Hom f g)) (MvPolynomial.X i)) ((Mv …
    -/
  · intro i
    /-
      case hX
      R : Type u
      S₁ : Type v
      S₂ : Type w
      σ : Type u_1
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring S₁
      inst✝ : CommSemiring S₂
      f : RingHom R S₁
      g : σ → S₁
      φ : RingHom S₁ S₂
      i : σ
      ⊢ Eq ((φ.comp (MvPolynomial.eval₂Hom f g)) (MvPolynomial.X i)) ((MvPolynomial. …
    -/
    rw [RingHom.comp_apply, eval₂Hom_X', eval₂Hom_X']
    /-
      🎉 no goals
    -/


theorem map_eval₂Hom [CommSemiring S₂] (f : R →+* S₁) (g : σ → S₁) (φ : S₁ →+* S₂)
    (p : MvPolynomial σ R) : φ (eval₂Hom f g p) = eval₂Hom (φ.comp f) (fun i => φ (g i)) p := by
  /-
    R : Type u
    S₁ : Type v
    S₂ : Type w
    σ : Type u_1
    inst✝² : CommSemiring R
    inst✝¹ : CommSemiring S₁
    inst✝ : CommSemiring S₂
    f : RingHom R S₁
    g : σ → S₁
    φ : RingHom S₁ S₂
    p : MvPolynomial σ R
    ⊢ Eq (φ ((MvPolynomial.eval₂Hom f g) p)) ((MvPolynomial.eval₂Hom (φ.comp f) fu …
  -/
  rw [← comp_eval₂Hom]
  /-
    R : Type u
    S₁ : Type v
    S₂ : Type w
    σ : Type u_1
    inst✝² : CommSemiring R
    inst✝¹ : CommSemiring S₁
    inst✝ : CommSemiring S₂
    f : RingHom R S₁
    g : σ → S₁
    φ : RingHom S₁ S₂
    p : MvPolynomial σ R
    ⊢ Eq (φ ((MvPolynomial.eval₂Hom f g) p)) ((φ.comp (MvPolynomial.eval₂Hom f g)) …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem eval₂Hom_monomial (f : R →+* S₁) (g : σ → S₁) (d : σ →₀ ℕ) (r : R) :
    eval₂Hom f g (monomial d r) = f r * d.prod fun i k => g i ^ k := by
  simp only [monomial_eq, RingHom.map_mul, eval₂Hom_C, Finsupp.prod, map_prod,
    RingHom.map_pow, eval₂Hom_X']


theorem eval₂_comp_left {S₂} [CommSemiring S₂] (k : S₁ →+* S₂) (f : R →+* S₁) (g : σ → S₁) (p) :
    k (eval₂ f g p) = eval₂ (k.comp f) (k ∘ g) p := by
  /-
    R : Type u
    S₁ : Type v
    σ : Type u_1
    inst✝² : CommSemiring R
    inst✝¹ : CommSemiring S₁
    S₂ : Type u_2
    inst✝ : CommSemiring S₂
    k : RingHom S₁ S₂
    f : RingHom R S₁
    g : σ → S₁
    p : MvPolynomial σ R
    ⊢ Eq (k (MvPolynomial.eval₂ f g p)) (MvPolynomial.eval₂ (k.comp f) (Function.c …
  -/
  apply MvPolynomial.induction_on p <;>
    /-
      case h_C
      R : Type u
      S₁ : Type v
      σ : Type u_1
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring S₁
      S₂ : Type u_2
      inst✝ : CommSemiring S₂
      k : RingHom S₁ S₂
      f : RingHom R S₁
      g : σ → S₁
      p : MvPolynomial σ R
      ⊢ ∀ (a : R), Eq (k (MvPolynomial.eval₂ f g (MvPolynomial.C a))) (MvPolynomial. …
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    simp +contextual [eval₂_add, k.map_add, eval₂_mul, k.map_mul]
    /-
      🎉 no goals
    -/


@[simp]
theorem eval₂_eta (p : MvPolynomial σ R) : eval₂ C X p = p := by
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    p : MvPolynomial σ R
    ⊢ Eq (MvPolynomial.eval₂ MvPolynomial.C MvPolynomial.X p) p
  -/
  apply MvPolynomial.induction_on p <;>
    /-
      case h_C
      R : Type u
      σ : Type u_1
      inst✝ : CommSemiring R
      p : MvPolynomial σ R
      ⊢ ∀ (a : R), Eq (MvPolynomial.eval₂ MvPolynomial.C MvPolynomial.X (MvPolynomia …
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    simp +contextual [eval₂_add, eval₂_mul]
    /-
      🎉 no goals
    -/


theorem eval₂_congr (g₁ g₂ : σ → S₁)
    (h : ∀ {i : σ} {c : σ →₀ ℕ}, i ∈ c.support → coeff c p ≠ 0 → g₁ i = g₂ i) :
    p.eval₂ f g₁ = p.eval₂ f g₂ := by
  /-
    R : Type u
    S₁ : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S₁
    p : MvPolynomial σ R
    f : RingHom R S₁
    g₁ g₂ : σ → S₁
    h : ∀ {i : σ} {c : Finsupp σ Nat}, Membership.mem c.support i → Ne (MvPolynomi …
    ⊢ Eq (MvPolynomial.eval₂ f g₁ p) (MvPolynomial.eval₂ f g₂ p)
  -/
  apply Finset.sum_congr rfl
  /-
    R : Type u
    S₁ : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S₁
    p : MvPolynomial σ R
    f : RingHom R S₁
    g₁ g₂ : σ → S₁
    h : ∀ {i : σ} {c : Finsupp σ Nat}, Membership.mem c.support i → Ne (MvPolynomi …
    ⊢ ∀ (x : Finsupp σ Nat), Membership.mem p.support x → Eq ((fun s a => HMul.hMu …
  -/
  intro C hc; dsimp; congr 1
  /-
    case e_a
    R : Type u
    S₁ : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S₁
    p : MvPolynomial σ R
    f : RingHom R S₁
    g₁ g₂ : σ → S₁
    h : ∀ {i : σ} {c : Finsupp σ Nat}, Membership.mem c.support i → Ne (MvPolynomi …
    C : Finsupp σ Nat
    hc : Membership.mem p.support C
    ⊢ Eq (C.prod fun n e => HPow.hPow (g₁ n) e) (C.prod fun n e => HPow.hPow (g₂ n …
  -/
  apply Finset.prod_congr rfl
  /-
    case e_a
    R : Type u
    S₁ : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S₁
    p : MvPolynomial σ R
    f : RingHom R S₁
    g₁ g₂ : σ → S₁
    h : ∀ {i : σ} {c : Finsupp σ Nat}, Membership.mem c.support i → Ne (MvPolynomi …
    C : Finsupp σ Nat
    hc : Membership.mem p.support C
    ⊢ ∀ (x : σ), Membership.mem C.support x → Eq ((fun n e => HPow.hPow (g₁ n) e)  …
  -/
  intro i hi; dsimp; congr 1
  /-
    case e_a.e_a
    R : Type u
    S₁ : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S₁
    p : MvPolynomial σ R
    f : RingHom R S₁
    g₁ g₂ : σ → S₁
    h : ∀ {i : σ} {c : Finsupp σ Nat}, Membership.mem c.support i → Ne (MvPolynomi …
    C : Finsupp σ Nat
    hc : Membership.mem p.support C
    i : σ
    hi : Membership.mem C.support i
    ⊢ Eq (g₁ i) (g₂ i)
  -/
  apply h hi
  /-
    case e_a.e_a
    R : Type u
    S₁ : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S₁
    p : MvPolynomial σ R
    f : RingHom R S₁
    g₁ g₂ : σ → S₁
    h : ∀ {i : σ} {c : Finsupp σ Nat}, Membership.mem c.support i → Ne (MvPolynomi …
    C : Finsupp σ Nat
    hc : Membership.mem p.support C
    i : σ
    hi : Membership.mem C.support i
    ⊢ Ne (MvPolynomial.coeff C p) 0
  -/
  rwa [Finsupp.mem_support_iff] at hc
  /-
    🎉 no goals
  -/


theorem eval₂_sum (s : Finset S₂) (p : S₂ → MvPolynomial σ R) :
    eval₂ f g (∑ x ∈ s, p x) = ∑ x ∈ s, eval₂ f g (p x) :=
  map_sum (eval₂Hom f g) _ s


@[to_additive existing (attr := simp)]
theorem eval₂_prod (s : Finset S₂) (p : S₂ → MvPolynomial σ R) :
    eval₂ f g (∏ x ∈ s, p x) = ∏ x ∈ s, eval₂ f g (p x) :=
  map_prod (eval₂Hom f g) _ s


theorem eval₂_assoc (q : S₂ → MvPolynomial σ R) (p : MvPolynomial S₂ R) :
    eval₂ f (fun t => eval₂ f g (q t)) p = eval₂ f g (eval₂ C q p) := by
  /-
    R : Type u
    S₁ : Type v
    S₂ : Type w
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S₁
    f : RingHom R S₁
    g : σ → S₁
    q : S₂ → MvPolynomial σ R
    p : MvPolynomial S₂ R
    ⊢ Eq (MvPolynomial.eval₂ f (fun t => MvPolynomial.eval₂ f g (q t)) p) (MvPolyn …
  -/
  show _ = eval₂Hom f g (eval₂ C q p)
  /-
    R : Type u
    S₁ : Type v
    S₂ : Type w
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S₁
    f : RingHom R S₁
    g : σ → S₁
    q : S₂ → MvPolynomial σ R
    p : MvPolynomial S₂ R
    ⊢ Eq (MvPolynomial.eval₂ f (fun t => MvPolynomial.eval₂ f g (q t)) p) ((MvPoly …
  -/
  rw [eval₂_comp_left (eval₂Hom f g)]; congr with a; simp
                                                     /-
                                                       🎉 no goals
                                                     -/


/-- Evaluate a polynomial `p` given a valuation `f` of all the variables -/
def eval (f : σ → R) : MvPolynomial σ R →+* R :=
  eval₂Hom (RingHom.id _) f


theorem eval_eq (X : σ → R) (f : MvPolynomial σ R) :
    eval X f = ∑ d ∈ f.support, f.coeff d * ∏ i ∈ d.support, X i ^ d i :=
  rfl


theorem eval_eq' [Fintype σ] (X : σ → R) (f : MvPolynomial σ R) :
    eval X f = ∑ d ∈ f.support, f.coeff d * ∏ i, X i ^ d i :=
  eval₂_eq' (RingHom.id R) X f


theorem eval_monomial : eval f (monomial s a) = a * s.prod fun n e => f n ^ e :=
  eval₂_monomial _ _


@[simp]
theorem eval_C : ∀ a, eval f (C a) = a :=
  eval₂_C _ _


@[simp]
theorem eval_X : ∀ n, eval f (X n) = f n :=
  eval₂_X _ _


@[simp] theorem eval_ofNat (n : Nat) [n.AtLeastTwo] :
    (ofNat(n) : MvPolynomial σ R).eval f = ofNat(n) :=
  map_ofNat _ n


@[simp]
theorem smul_eval (x) (p : MvPolynomial σ R) (s) : eval x (s • p) = s * eval x p := by
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    x : σ → R
    p : MvPolynomial σ R
    s : R
    ⊢ Eq ((MvPolynomial.eval x) (HSMul.hSMul s p)) (HMul.hMul s ((MvPolynomial.eva …
  -/
  rw [smul_eq_C_mul, (eval x).map_mul, eval_C]
  /-
    🎉 no goals
  -/


theorem eval_add : eval f (p + q) = eval f p + eval f q :=
  eval₂_add _ _


theorem eval_mul : eval f (p * q) = eval f p * eval f q :=
  eval₂_mul _ _


theorem eval_pow : ∀ n, eval f (p ^ n) = eval f p ^ n :=
  fun _ => eval₂_pow _ _


theorem eval_sum {ι : Type*} (s : Finset ι) (f : ι → MvPolynomial σ R) (g : σ → R) :
    eval g (∑ i ∈ s, f i) = ∑ i ∈ s, eval g (f i) :=
  map_sum (eval g) _ _


@[to_additive existing]
theorem eval_prod {ι : Type*} (s : Finset ι) (f : ι → MvPolynomial σ R) (g : σ → R) :
    eval g (∏ i ∈ s, f i) = ∏ i ∈ s, eval g (f i) :=
  map_prod (eval g) _ _


theorem eval_assoc {τ} (f : σ → MvPolynomial τ R) (g : τ → R) (p : MvPolynomial σ R) :
    eval (eval g ∘ f) p = eval g (eval₂ C f p) := by
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    τ : Type u_2
    f : σ → MvPolynomial τ R
    g : τ → R
    p : MvPolynomial σ R
    ⊢ Eq ((MvPolynomial.eval (Function.comp (⇑(MvPolynomial.eval g)) f)) p) ((MvPo …
  -/
  rw [eval₂_comp_left (eval g)]
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    τ : Type u_2
    f : σ → MvPolynomial τ R
    g : τ → R
    p : MvPolynomial σ R
    ⊢ Eq ((MvPolynomial.eval (Function.comp (⇑(MvPolynomial.eval g)) f)) p) (MvPol …
  -/
  unfold eval; simp only [coe_eval₂Hom]
  /-
    R : Type u
    σ : Type u_1
    inst✝ : CommSemiring R
    τ : Type u_2
    f : σ → MvPolynomial τ R
    g : τ → R
    p : MvPolynomial σ R
    ⊢ Eq (MvPolynomial.eval₂ (RingHom.id R) (Function.comp (MvPolynomial.eval₂ (Ri …
  -/
  congr with a; simp
                /-
                  🎉 no goals
                -/


@[simp]
theorem eval₂_id {g : σ → R} (p : MvPolynomial σ R) : eval₂ (RingHom.id _) g p = eval g p :=
  rfl


theorem eval_eval₂ {S τ : Type*} {x : τ → S} [CommSemiring S]
    (f : R →+* MvPolynomial τ S) (g : σ → MvPolynomial τ S) (p : MvPolynomial σ R) :
    eval x (eval₂ f g p) = eval₂ ((eval x).comp f) (fun s => eval x (g s)) p := by
  /-
    R : Type u
    σ : Type u_1
    inst✝¹ : CommSemiring R
    S : Type u_2
    τ : Type u_3
    x : τ → S
    inst✝ : CommSemiring S
    f : RingHom R (MvPolynomial τ S)
    g : σ → MvPolynomial τ S
    p : MvPolynomial σ R
    ⊢ Eq ((MvPolynomial.eval x) (MvPolynomial.eval₂ f g p)) (MvPolynomial.eval₂ (( …
  -/
  apply induction_on p
    /-
      case h_C
      R : Type u
      σ : Type u_1
      inst✝¹ : CommSemiring R
      S : Type u_2
      τ : Type u_3
      x : τ → S
      inst✝ : CommSemiring S
      f : RingHom R (MvPolynomial τ S)
      g : σ → MvPolynomial τ S
      p : MvPolynomial σ R
      ⊢ ∀ (a : R), Eq ((MvPolynomial.eval x) (MvPolynomial.eval₂ f g (MvPolynomial.C …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case h_add
      R : Type u
      σ : Type u_1
      inst✝¹ : CommSemiring R
      S : Type u_2
      τ : Type u_3
      x : τ → S
      inst✝ : CommSemiring S
      f : RingHom R (MvPolynomial τ S)
      g : σ → MvPolynomial τ S
      p : MvPolynomial σ R
      ⊢ ∀ (p q : MvPolynomial σ R), Eq ((MvPolynomial.eval x) (MvPolynomial.eval₂ f  …
    -/
  · intro p q hp hq
    /-
      case h_add
      R : Type u
      σ : Type u_1
      inst✝¹ : CommSemiring R
      S : Type u_2
      τ : Type u_3
      x : τ → S
      inst✝ : CommSemiring S
      f : RingHom R (MvPolynomial τ S)
      g : σ → MvPolynomial τ S
      p✝ p q : MvPolynomial σ R
      hp : Eq ((MvPolynomial.eval x) (MvPolynomial.eval₂ f g p)) (MvPolynomial.eval₂ …
      hq : Eq ((MvPolynomial.eval x) (MvPolynomial.eval₂ f g q)) (MvPolynomial.eval₂ …
      ⊢ Eq ((MvPolynomial.eval x) (MvPolynomial.eval₂ f g (HAdd.hAdd p q))) (MvPolyn …
    -/
    simp [hp, hq]
    /-
      🎉 no goals
    -/
    /-
      case h_X
      R : Type u
      σ : Type u_1
      inst✝¹ : CommSemiring R
      S : Type u_2
      τ : Type u_3
      x : τ → S
      inst✝ : CommSemiring S
      f : RingHom R (MvPolynomial τ S)
      g : σ → MvPolynomial τ S
      p : MvPolynomial σ R
      ⊢ ∀ (p : MvPolynomial σ R) (n : σ), Eq ((MvPolynomial.eval x) (MvPolynomial.ev …
    -/
  · intro p n hp
    /-
      case h_X
      R : Type u
      σ : Type u_1
      inst✝¹ : CommSemiring R
      S : Type u_2
      τ : Type u_3
      x : τ → S
      inst✝ : CommSemiring S
      f : RingHom R (MvPolynomial τ S)
      g : σ → MvPolynomial τ S
      p✝ p : MvPolynomial σ R
      n : σ
      hp : Eq ((MvPolynomial.eval x) (MvPolynomial.eval₂ f g p)) (MvPolynomial.eval₂ …
      ⊢ Eq ((MvPolynomial.eval x) (MvPolynomial.eval₂ f g (HMul.hMul p (MvPolynomial …
    -/
    simp [hp]
    /-
      🎉 no goals
    -/


/-- `map f p` maps a polynomial `p` across a ring hom `f` -/
def map : MvPolynomial σ R →+* MvPolynomial σ S₁ :=
  eval₂Hom (C.comp f) X


@[simp]
theorem map_monomial (s : σ →₀ ℕ) (a : R) : map f (monomial s a) = monomial s (f a) :=
  (eval₂_monomial _ _).trans monomial_eq.symm


@[simp]
theorem map_C : ∀ a : R, map f (C a : MvPolynomial σ R) = C (f a) :=
  map_monomial _ _


@[simp] protected theorem map_ofNat (n : Nat) [n.AtLeastTwo] :
    (ofNat(n) : MvPolynomial σ R).map f = ofNat(n) :=
  _root_.map_ofNat _ _


@[simp]
theorem map_X : ∀ n : σ, map f (X n : MvPolynomial σ R) = X n :=
  eval₂_X _ _


theorem map_id : ∀ p : MvPolynomial σ R, map (RingHom.id R) p = p :=
  eval₂_eta


theorem map_map [CommSemiring S₂] (g : S₁ →+* S₂) (p : MvPolynomial σ R) :
    map g (map f p) = map (g.comp f) p :=
  (eval₂_comp_left (map g) (C.comp f) X p).trans <| by
    /-
      R : Type u
      S₁ : Type v
      S₂ : Type w
      σ : Type u_1
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring S₁
      f : RingHom R S₁
      inst✝ : CommSemiring S₂
      g : RingHom S₁ S₂
      p : MvPolynomial σ R
      ⊢ Eq (MvPolynomial.eval₂ ((MvPolynomial.map g).comp (MvPolynomial.C.comp f)) ( …
    -/
    congr
      /-
        case e_f
        R : Type u
        S₁ : Type v
        S₂ : Type w
        σ : Type u_1
        inst✝² : CommSemiring R
        inst✝¹ : CommSemiring S₁
        f : RingHom R S₁
        inst✝ : CommSemiring S₂
        g : RingHom S₁ S₂
        p : MvPolynomial σ R
        ⊢ Eq ((MvPolynomial.map g).comp (MvPolynomial.C.comp f)) (MvPolynomial.C.comp  …
      -/
    · ext1 a
      /-
        case e_f.a
        R : Type u
        S₁ : Type v
        S₂ : Type w
        σ : Type u_1
        inst✝² : CommSemiring R
        inst✝¹ : CommSemiring S₁
        f : RingHom R S₁
        inst✝ : CommSemiring S₂
        g : RingHom S₁ S₂
        p : MvPolynomial σ R
        a : R
        ⊢ Eq (((MvPolynomial.map g).comp (MvPolynomial.C.comp f)) a) ((MvPolynomial.C. …
      -/
      simp only [map_C, comp_apply, RingHom.coe_comp]
      /-
        🎉 no goals
      -/
      /-
        case e_g
        R : Type u
        S₁ : Type v
        S₂ : Type w
        σ : Type u_1
        inst✝² : CommSemiring R
        inst✝¹ : CommSemiring S₁
        f : RingHom R S₁
        inst✝ : CommSemiring S₂
        g : RingHom S₁ S₂
        p : MvPolynomial σ R
        ⊢ Eq (Function.comp (⇑(MvPolynomial.map g)) MvPolynomial.X) MvPolynomial.X
      -/
    · ext1 n
      /-
        case e_g.h
        R : Type u
        S₁ : Type v
        S₂ : Type w
        σ : Type u_1
        inst✝² : CommSemiring R
        inst✝¹ : CommSemiring S₁
        f : RingHom R S₁
        inst✝ : CommSemiring S₂
        g : RingHom S₁ S₂
        p : MvPolynomial σ R
        n : σ
        ⊢ Eq (Function.comp (⇑(MvPolynomial.map g)) MvPolynomial.X n) (MvPolynomial.X n)
      -/
      simp only [map_X, comp_apply]
      /-
        🎉 no goals
      -/


theorem eval₂_eq_eval_map (g : σ → S₁) (p : MvPolynomial σ R) : p.eval₂ f g = eval g (map f p) := by
  /-
    R : Type u
    S₁ : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S₁
    f : RingHom R S₁
    g : σ → S₁
    p : MvPolynomial σ R
    ⊢ Eq (MvPolynomial.eval₂ f g p) ((MvPolynomial.eval g) ((MvPolynomial.map f) p))
  -/
  unfold map eval; simp only [coe_eval₂Hom]

  /-
    R : Type u
    S₁ : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S₁
    f : RingHom R S₁
    g : σ → S₁
    p : MvPolynomial σ R
    ⊢ Eq (MvPolynomial.eval₂ f g p) (MvPolynomial.eval₂ (RingHom.id S₁) g (MvPolyn …
  -/
  have h := eval₂_comp_left (eval₂Hom (RingHom.id S₁) g) (C.comp f) X p
  -- Porting note: the Lean 3 version of `h` was full of metavariables which
  -- were later unified during `rw [h]`. Also needed to add `-eval₂_id`.
  /-
    R : Type u
    S₁ : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S₁
    f : RingHom R S₁
    g : σ → S₁
    p : MvPolynomial σ R
    h : Eq ((MvPolynomial.eval₂Hom (RingHom.id S₁) g) (MvPolynomial.eval₂ (MvPolyn …
    ⊢ Eq (MvPolynomial.eval₂ f g p) (MvPolynomial.eval₂ (RingHom.id S₁) g (MvPolyn …
  -/
  dsimp [-eval₂_id] at h
  /-
    R : Type u
    S₁ : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S₁
    f : RingHom R S₁
    g : σ → S₁
    p : MvPolynomial σ R
    h : Eq (MvPolynomial.eval₂ (RingHom.id S₁) g (MvPolynomial.eval₂ (MvPolynomial …
    ⊢ Eq (MvPolynomial.eval₂ f g p) (MvPolynomial.eval₂ (RingHom.id S₁) g (MvPolyn …
  -/
  rw [h]
  /-
    R : Type u
    S₁ : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S₁
    f : RingHom R S₁
    g : σ → S₁
    p : MvPolynomial σ R
    h : Eq (MvPolynomial.eval₂ (RingHom.id S₁) g (MvPolynomial.eval₂ (MvPolynomial …
    ⊢ Eq (MvPolynomial.eval₂ f g p) (MvPolynomial.eval₂ ((MvPolynomial.eval₂Hom (R …
  -/
  congr
    /-
      case e_f
      R : Type u
      S₁ : Type v
      σ : Type u_1
      inst✝¹ : CommSemiring R
      inst✝ : CommSemiring S₁
      f : RingHom R S₁
      g : σ → S₁
      p : MvPolynomial σ R
      h : Eq (MvPolynomial.eval₂ (RingHom.id S₁) g (MvPolynomial.eval₂ (MvPolynomial …
      ⊢ Eq f ((MvPolynomial.eval₂Hom (RingHom.id S₁) g).comp (MvPolynomial.C.comp f))
    -/
  · ext1 a
    /-
      case e_f.a
      R : Type u
      S₁ : Type v
      σ : Type u_1
      inst✝¹ : CommSemiring R
      inst✝ : CommSemiring S₁
      f : RingHom R S₁
      g : σ → S₁
      p : MvPolynomial σ R
      h : Eq (MvPolynomial.eval₂ (RingHom.id S₁) g (MvPolynomial.eval₂ (MvPolynomial …
      a : R
      ⊢ Eq (f a) (((MvPolynomial.eval₂Hom (RingHom.id S₁) g).comp (MvPolynomial.C.co …
    -/
    simp only [coe_eval₂Hom, RingHom.id_apply, comp_apply, eval₂_C, RingHom.coe_comp]
    /-
      🎉 no goals
    -/
    /-
      case e_g
      R : Type u
      S₁ : Type v
      σ : Type u_1
      inst✝¹ : CommSemiring R
      inst✝ : CommSemiring S₁
      f : RingHom R S₁
      g : σ → S₁
      p : MvPolynomial σ R
      h : Eq (MvPolynomial.eval₂ (RingHom.id S₁) g (MvPolynomial.eval₂ (MvPolynomial …
      ⊢ Eq g (Function.comp (MvPolynomial.eval₂ (RingHom.id S₁) g) MvPolynomial.X)
    -/
  · ext1 n
    /-
      case e_g.h
      R : Type u
      S₁ : Type v
      σ : Type u_1
      inst✝¹ : CommSemiring R
      inst✝ : CommSemiring S₁
      f : RingHom R S₁
      g : σ → S₁
      p : MvPolynomial σ R
      h : Eq (MvPolynomial.eval₂ (RingHom.id S₁) g (MvPolynomial.eval₂ (MvPolynomial …
      n : σ
      ⊢ Eq (g n) (Function.comp (MvPolynomial.eval₂ (RingHom.id S₁) g) MvPolynomial. …
    -/
    simp only [comp_apply, eval₂_X]
    /-
      🎉 no goals
    -/


theorem eval₂_comp_right {S₂} [CommSemiring S₂] (k : S₁ →+* S₂) (f : R →+* S₁) (g : σ → S₁) (p) :
    k (eval₂ f g p) = eval₂ k (k ∘ g) (map f p) := by
  /-
    R : Type u
    S₁ : Type v
    σ : Type u_1
    inst✝² : CommSemiring R
    inst✝¹ : CommSemiring S₁
    S₂ : Type u_2
    inst✝ : CommSemiring S₂
    k : RingHom S₁ S₂
    f : RingHom R S₁
    g : σ → S₁
    p : MvPolynomial σ R
    ⊢ Eq (k (MvPolynomial.eval₂ f g p)) (MvPolynomial.eval₂ k (Function.comp (⇑k)  …
  -/
  apply MvPolynomial.induction_on p
    /-
      case h_C
      R : Type u
      S₁ : Type v
      σ : Type u_1
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring S₁
      S₂ : Type u_2
      inst✝ : CommSemiring S₂
      k : RingHom S₁ S₂
      f : RingHom R S₁
      g : σ → S₁
      p : MvPolynomial σ R
      ⊢ ∀ (a : R), Eq (k (MvPolynomial.eval₂ f g (MvPolynomial.C a))) (MvPolynomial. …
    -/
  · intro r
    /-
      case h_C
      R : Type u
      S₁ : Type v
      σ : Type u_1
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring S₁
      S₂ : Type u_2
      inst✝ : CommSemiring S₂
      k : RingHom S₁ S₂
      f : RingHom R S₁
      g : σ → S₁
      p : MvPolynomial σ R
      r : R
      ⊢ Eq (k (MvPolynomial.eval₂ f g (MvPolynomial.C r))) (MvPolynomial.eval₂ k (Fu …
    -/
    rw [eval₂_C, map_C, eval₂_C]
    /-
      🎉 no goals
    -/
    /-
      case h_add
      R : Type u
      S₁ : Type v
      σ : Type u_1
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring S₁
      S₂ : Type u_2
      inst✝ : CommSemiring S₂
      k : RingHom S₁ S₂
      f : RingHom R S₁
      g : σ → S₁
      p : MvPolynomial σ R
      ⊢ ∀ (p q : MvPolynomial σ R), Eq (k (MvPolynomial.eval₂ f g p)) (MvPolynomial. …
    -/
  · intro p q hp hq
    /-
      case h_add
      R : Type u
      S₁ : Type v
      σ : Type u_1
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring S₁
      S₂ : Type u_2
      inst✝ : CommSemiring S₂
      k : RingHom S₁ S₂
      f : RingHom R S₁
      g : σ → S₁
      p✝ p q : MvPolynomial σ R
      hp : Eq (k (MvPolynomial.eval₂ f g p)) (MvPolynomial.eval₂ k (Function.comp (⇑ …
      hq : Eq (k (MvPolynomial.eval₂ f g q)) (MvPolynomial.eval₂ k (Function.comp (⇑ …
      ⊢ Eq (k (MvPolynomial.eval₂ f g (HAdd.hAdd p q))) (MvPolynomial.eval₂ k (Funct …
    -/
    rw [eval₂_add, k.map_add, (map f).map_add, eval₂_add, hp, hq]
    /-
      🎉 no goals
    -/
    /-
      case h_X
      R : Type u
      S₁ : Type v
      σ : Type u_1
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring S₁
      S₂ : Type u_2
      inst✝ : CommSemiring S₂
      k : RingHom S₁ S₂
      f : RingHom R S₁
      g : σ → S₁
      p : MvPolynomial σ R
      ⊢ ∀ (p : MvPolynomial σ R) (n : σ), Eq (k (MvPolynomial.eval₂ f g p)) (MvPolyn …
    -/
  · intro p s hp
    /-
      case h_X
      R : Type u
      S₁ : Type v
      σ : Type u_1
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring S₁
      S₂ : Type u_2
      inst✝ : CommSemiring S₂
      k : RingHom S₁ S₂
      f : RingHom R S₁
      g : σ → S₁
      p✝ p : MvPolynomial σ R
      s : σ
      hp : Eq (k (MvPolynomial.eval₂ f g p)) (MvPolynomial.eval₂ k (Function.comp (⇑ …
      ⊢ Eq (k (MvPolynomial.eval₂ f g (HMul.hMul p (MvPolynomial.X s)))) (MvPolynomi …
    -/
    rw [eval₂_mul, k.map_mul, (map f).map_mul, eval₂_mul, map_X, hp, eval₂_X, eval₂_X]
    /-
      case h_X
      R : Type u
      S₁ : Type v
      σ : Type u_1
      inst✝² : CommSemiring R
      inst✝¹ : CommSemiring S₁
      S₂ : Type u_2
      inst✝ : CommSemiring S₂
      k : RingHom S₁ S₂
      f : RingHom R S₁
      g : σ → S₁
      p✝ p : MvPolynomial σ R
      s : σ
      hp : Eq (k (MvPolynomial.eval₂ f g p)) (MvPolynomial.eval₂ k (Function.comp (⇑ …
      ⊢ Eq (HMul.hMul (MvPolynomial.eval₂ k (Function.comp (⇑k) g) ((MvPolynomial.ma …
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem map_eval₂ (f : R →+* S₁) (g : S₂ → MvPolynomial S₃ R) (p : MvPolynomial S₂ R) :
    map f (eval₂ C g p) = eval₂ C (map f ∘ g) (map f p) := by
  /-
    R : Type u
    S₁ : Type v
    S₂ : Type w
    S₃ : Type x
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S₁
    f : RingHom R S₁
    g : S₂ → MvPolynomial S₃ R
    p : MvPolynomial S₂ R
    ⊢ Eq ((MvPolynomial.map f) (MvPolynomial.eval₂ MvPolynomial.C g p)) (MvPolynom …
  -/
  apply MvPolynomial.induction_on p
    /-
      case h_C
      R : Type u
      S₁ : Type v
      S₂ : Type w
      S₃ : Type x
      inst✝¹ : CommSemiring R
      inst✝ : CommSemiring S₁
      f : RingHom R S₁
      g : S₂ → MvPolynomial S₃ R
      p : MvPolynomial S₂ R
      ⊢ ∀ (a : R), Eq ((MvPolynomial.map f) (MvPolynomial.eval₂ MvPolynomial.C g (Mv …
    -/
  · intro r
    /-
      case h_C
      R : Type u
      S₁ : Type v
      S₂ : Type w
      S₃ : Type x
      inst✝¹ : CommSemiring R
      inst✝ : CommSemiring S₁
      f : RingHom R S₁
      g : S₂ → MvPolynomial S₃ R
      p : MvPolynomial S₂ R
      r : R
      ⊢ Eq ((MvPolynomial.map f) (MvPolynomial.eval₂ MvPolynomial.C g (MvPolynomial. …
    -/
    rw [eval₂_C, map_C, map_C, eval₂_C]
    /-
      🎉 no goals
    -/
    /-
      case h_add
      R : Type u
      S₁ : Type v
      S₂ : Type w
      S₃ : Type x
      inst✝¹ : CommSemiring R
      inst✝ : CommSemiring S₁
      f : RingHom R S₁
      g : S₂ → MvPolynomial S₃ R
      p : MvPolynomial S₂ R
      ⊢ ∀ (p q : MvPolynomial S₂ R), Eq ((MvPolynomial.map f) (MvPolynomial.eval₂ Mv …
    -/
  · intro p q hp hq
    /-
      case h_add
      R : Type u
      S₁ : Type v
      S₂ : Type w
      S₃ : Type x
      inst✝¹ : CommSemiring R
      inst✝ : CommSemiring S₁
      f : RingHom R S₁
      g : S₂ → MvPolynomial S₃ R
      p✝ p q : MvPolynomial S₂ R
      hp : Eq ((MvPolynomial.map f) (MvPolynomial.eval₂ MvPolynomial.C g p)) (MvPoly …
      hq : Eq ((MvPolynomial.map f) (MvPolynomial.eval₂ MvPolynomial.C g q)) (MvPoly …
      ⊢ Eq ((MvPolynomial.map f) (MvPolynomial.eval₂ MvPolynomial.C g (HAdd.hAdd p q …
    -/
    rw [eval₂_add, (map f).map_add, hp, hq, (map f).map_add, eval₂_add]
    /-
      🎉 no goals
    -/
    /-
      case h_X
      R : Type u
      S₁ : Type v
      S₂ : Type w
      S₃ : Type x
      inst✝¹ : CommSemiring R
      inst✝ : CommSemiring S₁
      f : RingHom R S₁
      g : S₂ → MvPolynomial S₃ R
      p : MvPolynomial S₂ R
      ⊢ ∀ (p : MvPolynomial S₂ R) (n : S₂), Eq ((MvPolynomial.map f) (MvPolynomial.e …
    -/
  · intro p s hp
    /-
      case h_X
      R : Type u
      S₁ : Type v
      S₂ : Type w
      S₃ : Type x
      inst✝¹ : CommSemiring R
      inst✝ : CommSemiring S₁
      f : RingHom R S₁
      g : S₂ → MvPolynomial S₃ R
      p✝ p : MvPolynomial S₂ R
      s : S₂
      hp : Eq ((MvPolynomial.map f) (MvPolynomial.eval₂ MvPolynomial.C g p)) (MvPoly …
      ⊢ Eq ((MvPolynomial.map f) (MvPolynomial.eval₂ MvPolynomial.C g (HMul.hMul p ( …
    -/
    rw [eval₂_mul, (map f).map_mul, hp, (map f).map_mul, map_X, eval₂_mul, eval₂_X, eval₂_X]
    /-
      case h_X
      R : Type u
      S₁ : Type v
      S₂ : Type w
      S₃ : Type x
      inst✝¹ : CommSemiring R
      inst✝ : CommSemiring S₁
      f : RingHom R S₁
      g : S₂ → MvPolynomial S₃ R
      p✝ p : MvPolynomial S₂ R
      s : S₂
      hp : Eq ((MvPolynomial.map f) (MvPolynomial.eval₂ MvPolynomial.C g p)) (MvPoly …
      ⊢ Eq (HMul.hMul (MvPolynomial.eval₂ MvPolynomial.C (Function.comp (⇑(MvPolynom …
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem coeff_map (p : MvPolynomial σ R) : ∀ m : σ →₀ ℕ, coeff m (map f p) = f (coeff m p) := by
  classical
  apply MvPolynomial.induction_on p <;> clear p
  · intro r m
    rw [map_C]
    simp only [coeff_C]
    split_ifs
    · rfl
    rw [f.map_zero]
  · intro p q hp hq m
    simp only [hp, hq, (map f).map_add, coeff_add]
    rw [f.map_add]
  · intro p i hp m
    simp only [hp, (map f).map_mul, map_X]
    simp only [hp, mem_support_iff, coeff_mul_X']
    split_ifs
    · rfl
    rw [f.map_zero]


theorem map_injective (hf : Function.Injective f) :
    Function.Injective (map f : MvPolynomial σ R → MvPolynomial σ S₁) := by
  /-
    R : Type u
    S₁ : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S₁
    f : RingHom R S₁
    hf : Function.Injective ⇑f
    ⊢ Function.Injective ⇑(MvPolynomial.map f)
  -/
  intro p q h
  /-
    R : Type u
    S₁ : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S₁
    f : RingHom R S₁
    hf : Function.Injective ⇑f
    p q : MvPolynomial σ R
    h : Eq ((MvPolynomial.map f) p) ((MvPolynomial.map f) q)
    ⊢ Eq p q
  -/
  simp only [MvPolynomial.ext_iff, coeff_map] at h ⊢
  /-
    R : Type u
    S₁ : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S₁
    f : RingHom R S₁
    hf : Function.Injective ⇑f
    p q : MvPolynomial σ R
    h : ∀ (m : Finsupp σ Nat), Eq (f (MvPolynomial.coeff m p)) (f (MvPolynomial.co …
    ⊢ ∀ (m : Finsupp σ Nat), Eq (MvPolynomial.coeff m p) (MvPolynomial.coeff m q)
  -/
  intro m
  /-
    R : Type u
    S₁ : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S₁
    f : RingHom R S₁
    hf : Function.Injective ⇑f
    p q : MvPolynomial σ R
    h : ∀ (m : Finsupp σ Nat), Eq (f (MvPolynomial.coeff m p)) (f (MvPolynomial.co …
    m : Finsupp σ Nat
    ⊢ Eq (MvPolynomial.coeff m p) (MvPolynomial.coeff m q)
  -/
  exact hf (h m)
  /-
    🎉 no goals
  -/


theorem map_surjective (hf : Function.Surjective f) :
    Function.Surjective (map f : MvPolynomial σ R → MvPolynomial σ S₁) := fun p => by
  /-
    R : Type u
    S₁ : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S₁
    f : RingHom R S₁
    hf : Function.Surjective ⇑f
    p : MvPolynomial σ S₁
    ⊢ Exists fun a => Eq ((MvPolynomial.map f) a) p
  -/
  induction' p using MvPolynomial.induction_on' with i fr a b ha hb
    /-
      case h1
      R : Type u
      S₁ : Type v
      σ : Type u_1
      inst✝¹ : CommSemiring R
      inst✝ : CommSemiring S₁
      f : RingHom R S₁
      hf : Function.Surjective ⇑f
      i : Finsupp σ Nat
      fr : S₁
      ⊢ Exists fun a => Eq ((MvPolynomial.map f) a) ((MvPolynomial.monomial i) fr)
    -/
  · obtain ⟨r, rfl⟩ := hf fr
    /-
      case h1.intro
      R : Type u
      S₁ : Type v
      σ : Type u_1
      inst✝¹ : CommSemiring R
      inst✝ : CommSemiring S₁
      f : RingHom R S₁
      hf : Function.Surjective ⇑f
      i : Finsupp σ Nat
      r : R
      ⊢ Exists fun a => Eq ((MvPolynomial.map f) a) ((MvPolynomial.monomial i) (f r))
    -/
    exact ⟨monomial i r, map_monomial _ _ _⟩
    /-
      🎉 no goals
    -/
    /-
      case h2
      R : Type u
      S₁ : Type v
      σ : Type u_1
      inst✝¹ : CommSemiring R
      inst✝ : CommSemiring S₁
      f : RingHom R S₁
      hf : Function.Surjective ⇑f
      a b : MvPolynomial σ S₁
      ha : Exists fun a_1 => Eq ((MvPolynomial.map f) a_1) a
      hb : Exists fun a => Eq ((MvPolynomial.map f) a) b
      ⊢ Exists fun a_1 => Eq ((MvPolynomial.map f) a_1) (HAdd.hAdd a b)
    -/
  · obtain ⟨a, rfl⟩ := ha
    /-
      case h2.intro
      R : Type u
      S₁ : Type v
      σ : Type u_1
      inst✝¹ : CommSemiring R
      inst✝ : CommSemiring S₁
      f : RingHom R S₁
      hf : Function.Surjective ⇑f
      b : MvPolynomial σ S₁
      hb : Exists fun a => Eq ((MvPolynomial.map f) a) b
      a : MvPolynomial σ R
      ⊢ Exists fun a_1 => Eq ((MvPolynomial.map f) a_1) (HAdd.hAdd ((MvPolynomial.ma …
    -/
    obtain ⟨b, rfl⟩ := hb
    /-
      case h2.intro.intro
      R : Type u
      S₁ : Type v
      σ : Type u_1
      inst✝¹ : CommSemiring R
      inst✝ : CommSemiring S₁
      f : RingHom R S₁
      hf : Function.Surjective ⇑f
      a b : MvPolynomial σ R
      ⊢ Exists fun a_1 => Eq ((MvPolynomial.map f) a_1) (HAdd.hAdd ((MvPolynomial.ma …
    -/
    exact ⟨a + b, RingHom.map_add _ _ _⟩
    /-
      🎉 no goals
    -/


/-- If `f` is a left-inverse of `g` then `map f` is a left-inverse of `map g`. -/
theorem map_leftInverse {f : R →+* S₁} {g : S₁ →+* R} (hf : Function.LeftInverse f g) :
    Function.LeftInverse (map f : MvPolynomial σ R → MvPolynomial σ S₁) (map g) := fun X => by
  /-
    R : Type u
    S₁ : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S₁
    f : RingHom R S₁
    g : RingHom S₁ R
    hf : Function.LeftInverse ⇑f ⇑g
    X : MvPolynomial σ S₁
    ⊢ Eq ((MvPolynomial.map f) ((MvPolynomial.map g) X)) X
  -/
  rw [map_map, (RingHom.ext hf : f.comp g = RingHom.id _), map_id]
  /-
    🎉 no goals
  -/


/-- If `f` is a right-inverse of `g` then `map f` is a right-inverse of `map g`. -/
theorem map_rightInverse {f : R →+* S₁} {g : S₁ →+* R} (hf : Function.RightInverse f g) :
    Function.RightInverse (map f : MvPolynomial σ R → MvPolynomial σ S₁) (map g) :=
  (map_leftInverse hf.leftInverse).rightInverse


@[simp]
theorem eval_map (f : R →+* S₁) (g : σ → S₁) (p : MvPolynomial σ R) :
    eval g (map f p) = eval₂ f g p := by
  /-
    R : Type u
    S₁ : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S₁
    f : RingHom R S₁
    g : σ → S₁
    p : MvPolynomial σ R
    ⊢ Eq ((MvPolynomial.eval g) ((MvPolynomial.map f) p)) (MvPolynomial.eval₂ f g p)
  -/
                                          /-
                                            🎉 no goals
                                          -/
                                          /-
                                            🎉 no goals
                                          -/
  apply MvPolynomial.induction_on p <;> · simp +contextual
                                          /-
                                            🎉 no goals
                                          -/


theorem eval₂_comp (f : R →+* S₁) (g : σ → R) (p : MvPolynomial σ R) :
    f (eval g p) = eval₂ f (f ∘ g) p := by
  /-
    R : Type u
    S₁ : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S₁
    f : RingHom R S₁
    g : σ → R
    p : MvPolynomial σ R
    ⊢ Eq (f ((MvPolynomial.eval g) p)) (MvPolynomial.eval₂ f (Function.comp (⇑f) g …
  -/
  rw [← p.map_id, eval_map, eval₂_comp_right]
  /-
    🎉 no goals
  -/


@[simp]
theorem eval₂_map [CommSemiring S₂] (f : R →+* S₁) (g : σ → S₂) (φ : S₁ →+* S₂)
    (p : MvPolynomial σ R) : eval₂ φ g (map f p) = eval₂ (φ.comp f) g p := by
  /-
    R : Type u
    S₁ : Type v
    S₂ : Type w
    σ : Type u_1
    inst✝² : CommSemiring R
    inst✝¹ : CommSemiring S₁
    inst✝ : CommSemiring S₂
    f : RingHom R S₁
    g : σ → S₂
    φ : RingHom S₁ S₂
    p : MvPolynomial σ R
    ⊢ Eq (MvPolynomial.eval₂ φ g ((MvPolynomial.map f) p)) (MvPolynomial.eval₂ (φ. …
  -/
  rw [← eval_map, ← eval_map, map_map]
  /-
    🎉 no goals
  -/


@[simp]
theorem eval₂Hom_map_hom [CommSemiring S₂] (f : R →+* S₁) (g : σ → S₂) (φ : S₁ →+* S₂)
    (p : MvPolynomial σ R) : eval₂Hom φ g (map f p) = eval₂Hom (φ.comp f) g p :=
  eval₂_map f g φ p


@[simp]
theorem constantCoeff_map (f : R →+* S₁) (φ : MvPolynomial σ R) :
    constantCoeff (MvPolynomial.map f φ) = f (constantCoeff φ) :=
  coeff_map f φ 0


theorem constantCoeff_comp_map (f : R →+* S₁) :
    (constantCoeff : MvPolynomial σ S₁ →+* S₁).comp (MvPolynomial.map f) = f.comp constantCoeff :=
     /-
       R : Type u
       S₁ : Type v
       σ : Type u_1
       inst✝¹ : CommSemiring R
       inst✝ : CommSemiring S₁
       f : RingHom R S₁
       ⊢ Eq (MvPolynomial.constantCoeff.comp (MvPolynomial.map f)) (f.comp MvPolynomi …
     -/
             /-
               🎉 no goals
             -/
  by ext <;> simp
             /-
               🎉 no goals
             -/


theorem support_map_subset (p : MvPolynomial σ R) : (map f p).support ⊆ p.support := by
  /-
    R : Type u
    S₁ : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S₁
    f : RingHom R S₁
    p : MvPolynomial σ R
    ⊢ HasSubset.Subset ((MvPolynomial.map f) p).support p.support
  -/
  intro x
  /-
    R : Type u
    S₁ : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S₁
    f : RingHom R S₁
    p : MvPolynomial σ R
    x : Finsupp σ Nat
    ⊢ Membership.mem ((MvPolynomial.map f) p).support x → Membership.mem p.support x
  -/
  simp only [mem_support_iff]
  /-
    R : Type u
    S₁ : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S₁
    f : RingHom R S₁
    p : MvPolynomial σ R
    x : Finsupp σ Nat
    ⊢ Ne (MvPolynomial.coeff x ((MvPolynomial.map f) p)) 0 → Ne (MvPolynomial.coef …
  -/
  contrapose!
  /-
    R : Type u
    S₁ : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S₁
    f : RingHom R S₁
    p : MvPolynomial σ R
    x : Finsupp σ Nat
    ⊢ Eq (MvPolynomial.coeff x p) 0 → Eq (MvPolynomial.coeff x ((MvPolynomial.map  …
  -/
  change p.coeff x = 0 → (map f p).coeff x = 0
  /-
    R : Type u
    S₁ : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S₁
    f : RingHom R S₁
    p : MvPolynomial σ R
    x : Finsupp σ Nat
    ⊢ Eq (MvPolynomial.coeff x p) 0 → Eq (MvPolynomial.coeff x ((MvPolynomial.map  …
  -/
  rw [coeff_map]
  /-
    R : Type u
    S₁ : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S₁
    f : RingHom R S₁
    p : MvPolynomial σ R
    x : Finsupp σ Nat
    ⊢ Eq (MvPolynomial.coeff x p) 0 → Eq (f (MvPolynomial.coeff x p)) 0
  -/
  intro hx
  /-
    R : Type u
    S₁ : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S₁
    f : RingHom R S₁
    p : MvPolynomial σ R
    x : Finsupp σ Nat
    hx : Eq (MvPolynomial.coeff x p) 0
    ⊢ Eq (f (MvPolynomial.coeff x p)) 0
  -/
  rw [hx]
  /-
    R : Type u
    S₁ : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S₁
    f : RingHom R S₁
    p : MvPolynomial σ R
    x : Finsupp σ Nat
    hx : Eq (MvPolynomial.coeff x p) 0
    ⊢ Eq (f 0) 0
  -/
  exact RingHom.map_zero f
  /-
    🎉 no goals
  -/


theorem support_map_of_injective (p : MvPolynomial σ R) {f : R →+* S₁} (hf : Injective f) :
    (map f p).support = p.support := by
  /-
    R : Type u
    S₁ : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S₁
    p : MvPolynomial σ R
    f : RingHom R S₁
    hf : Function.Injective ⇑f
    ⊢ Eq ((MvPolynomial.map f) p).support p.support
  -/
  apply Finset.Subset.antisymm
    /-
      case H₁
      R : Type u
      S₁ : Type v
      σ : Type u_1
      inst✝¹ : CommSemiring R
      inst✝ : CommSemiring S₁
      p : MvPolynomial σ R
      f : RingHom R S₁
      hf : Function.Injective ⇑f
      ⊢ HasSubset.Subset ((MvPolynomial.map f) p).support p.support
    -/
  · exact MvPolynomial.support_map_subset _ _
    /-
      🎉 no goals
    -/
  /-
    case H₂
    R : Type u
    S₁ : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S₁
    p : MvPolynomial σ R
    f : RingHom R S₁
    hf : Function.Injective ⇑f
    ⊢ HasSubset.Subset p.support ((MvPolynomial.map f) p).support
  -/
  intro x hx
  /-
    case H₂
    R : Type u
    S₁ : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S₁
    p : MvPolynomial σ R
    f : RingHom R S₁
    hf : Function.Injective ⇑f
    x : Finsupp σ Nat
    hx : Membership.mem p.support x
    ⊢ Membership.mem ((MvPolynomial.map f) p).support x
  -/
  rw [mem_support_iff]
  /-
    case H₂
    R : Type u
    S₁ : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S₁
    p : MvPolynomial σ R
    f : RingHom R S₁
    hf : Function.Injective ⇑f
    x : Finsupp σ Nat
    hx : Membership.mem p.support x
    ⊢ Ne (MvPolynomial.coeff x ((MvPolynomial.map f) p)) 0
  -/
  contrapose! hx
  /-
    case H₂
    R : Type u
    S₁ : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S₁
    p : MvPolynomial σ R
    f : RingHom R S₁
    hf : Function.Injective ⇑f
    x : Finsupp σ Nat
    hx : Eq (MvPolynomial.coeff x ((MvPolynomial.map f) p)) 0
    ⊢ Not (Membership.mem p.support x)
  -/
  simp only [Classical.not_not, mem_support_iff]
  /-
    case H₂
    R : Type u
    S₁ : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S₁
    p : MvPolynomial σ R
    f : RingHom R S₁
    hf : Function.Injective ⇑f
    x : Finsupp σ Nat
    hx : Eq (MvPolynomial.coeff x ((MvPolynomial.map f) p)) 0
    ⊢ Eq (MvPolynomial.coeff x p) 0
  -/
  replace hx : (map f p).coeff x = 0 := hx
  /-
    case H₂
    R : Type u
    S₁ : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S₁
    p : MvPolynomial σ R
    f : RingHom R S₁
    hf : Function.Injective ⇑f
    x : Finsupp σ Nat
    hx : Eq (MvPolynomial.coeff x ((MvPolynomial.map f) p)) 0
    ⊢ Eq (MvPolynomial.coeff x p) 0
  -/
  rw [coeff_map, ← f.map_zero] at hx
  /-
    case H₂
    R : Type u
    S₁ : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S₁
    p : MvPolynomial σ R
    f : RingHom R S₁
    hf : Function.Injective ⇑f
    x : Finsupp σ Nat
    hx : Eq (f (MvPolynomial.coeff x p)) (f 0)
    ⊢ Eq (MvPolynomial.coeff x p) 0
  -/
  exact hf hx
  /-
    🎉 no goals
  -/


theorem C_dvd_iff_map_hom_eq_zero (q : R →+* S₁) (r : R) (hr : ∀ r' : R, q r' = 0 ↔ r ∣ r')
    (φ : MvPolynomial σ R) : C r ∣ φ ↔ map q φ = 0 := by
  /-
    R : Type u
    S₁ : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S₁
    q : RingHom R S₁
    r : R
    hr : ∀ (r' : R), Iff (Eq (q r') 0) (Dvd.dvd r r')
    φ : MvPolynomial σ R
    ⊢ Iff (Dvd.dvd (MvPolynomial.C r) φ) (Eq ((MvPolynomial.map q) φ) 0)
  -/
  rw [C_dvd_iff_dvd_coeff, MvPolynomial.ext_iff]
  /-
    R : Type u
    S₁ : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S₁
    q : RingHom R S₁
    r : R
    hr : ∀ (r' : R), Iff (Eq (q r') 0) (Dvd.dvd r r')
    φ : MvPolynomial σ R
    ⊢ Iff (∀ (i : Finsupp σ Nat), Dvd.dvd r (MvPolynomial.coeff i φ)) (∀ (m : Fins …
  -/
  simp only [coeff_map, coeff_zero, hr]
  /-
    🎉 no goals
  -/


theorem map_mapRange_eq_iff (f : R →+* S₁) (g : S₁ → R) (hg : g 0 = 0) (φ : MvPolynomial σ S₁) :
    map f (Finsupp.mapRange g hg φ) = φ ↔ ∀ d, f (g (coeff d φ)) = coeff d φ := by
  /-
    R : Type u
    S₁ : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S₁
    f : RingHom R S₁
    g : S₁ → R
    hg : Eq (g 0) 0
    φ : MvPolynomial σ S₁
    ⊢ Iff (Eq ((MvPolynomial.map f) (Finsupp.mapRange g hg φ)) φ) (∀ (d : Finsupp  …
  -/
  rw [MvPolynomial.ext_iff]
  /-
    R : Type u
    S₁ : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S₁
    f : RingHom R S₁
    g : S₁ → R
    hg : Eq (g 0) 0
    φ : MvPolynomial σ S₁
    ⊢ Iff (∀ (m : Finsupp σ Nat), Eq (MvPolynomial.coeff m ((MvPolynomial.map f) ( …
  -/
  apply forall_congr'; intro m
  /-
    case h
    R : Type u
    S₁ : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S₁
    f : RingHom R S₁
    g : S₁ → R
    hg : Eq (g 0) 0
    φ : MvPolynomial σ S₁
    m : Finsupp σ Nat
    ⊢ Iff (Eq (MvPolynomial.coeff m ((MvPolynomial.map f) (Finsupp.mapRange g hg φ …
  -/
  rw [coeff_map]
  /-
    case h
    R : Type u
    S₁ : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S₁
    f : RingHom R S₁
    g : S₁ → R
    hg : Eq (g 0) 0
    φ : MvPolynomial σ S₁
    m : Finsupp σ Nat
    ⊢ Iff (Eq (f (MvPolynomial.coeff m (Finsupp.mapRange g hg φ))) (MvPolynomial.c …
  -/
  apply eq_iff_eq_cancel_right.mpr
  /-
    case h.a
    R : Type u
    S₁ : Type v
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S₁
    f : RingHom R S₁
    g : S₁ → R
    hg : Eq (g 0) 0
    φ : MvPolynomial σ S₁
    m : Finsupp σ Nat
    ⊢ Eq (f (MvPolynomial.coeff m (Finsupp.mapRange g hg φ))) (f (g (MvPolynomial. …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- If `f : S₁ →ₐ[R] S₂` is a morphism of `R`-algebras, then so is `MvPolynomial.map f`. -/
@[simps!]
def mapAlgHom [CommSemiring S₂] [Algebra R S₁] [Algebra R S₂] (f : S₁ →ₐ[R] S₂) :
    MvPolynomial σ S₁ →ₐ[R] MvPolynomial σ S₂ :=
  { map (↑f : S₁ →+* S₂) with
    commutes' := fun r => by
      /-
        R : Type u
        S₁ : Type v
        S₂ : Type w
        S₃ : Type x
        σ : Type u_1
        a a' a₁ a₂ : R
        e : Nat
        n m : σ
        s : Finsupp σ Nat
        inst✝⁴ : CommSemiring R
        inst✝³ : CommSemiring S₁
        p q : MvPolynomial σ R
        f✝ : RingHom R S₁
        inst✝² : CommSemiring S₂
        inst✝¹ : Algebra R S₁
        inst✝ : Algebra R S₂
        f : AlgHom R S₁ S₂
        r : R
        ⊢ Eq ((↑↑__src✝).toFun ((algebraMap R (MvPolynomial σ S₁)) r)) ((algebraMap R  …
      -/
      have h₁ : algebraMap R (MvPolynomial σ S₁) r = C (algebraMap R S₁ r) := rfl
      /-
        R : Type u
        S₁ : Type v
        S₂ : Type w
        S₃ : Type x
        σ : Type u_1
        a a' a₁ a₂ : R
        e : Nat
        n m : σ
        s : Finsupp σ Nat
        inst✝⁴ : CommSemiring R
        inst✝³ : CommSemiring S₁
        p q : MvPolynomial σ R
        f✝ : RingHom R S₁
        inst✝² : CommSemiring S₂
        inst✝¹ : Algebra R S₁
        inst✝ : Algebra R S₂
        f : AlgHom R S₁ S₂
        r : R
        h₁ : Eq ((algebraMap R (MvPolynomial σ S₁)) r) (MvPolynomial.C ((algebraMap R  …
        ⊢ Eq ((↑↑__src✝).toFun ((algebraMap R (MvPolynomial σ S₁)) r)) ((algebraMap R  …
      -/
      have h₂ : algebraMap R (MvPolynomial σ S₂) r = C (algebraMap R S₂ r) := rfl
      /-
        R : Type u
        S₁ : Type v
        S₂ : Type w
        S₃ : Type x
        σ : Type u_1
        a a' a₁ a₂ : R
        e : Nat
        n m : σ
        s : Finsupp σ Nat
        inst✝⁴ : CommSemiring R
        inst✝³ : CommSemiring S₁
        p q : MvPolynomial σ R
        f✝ : RingHom R S₁
        inst✝² : CommSemiring S₂
        inst✝¹ : Algebra R S₁
        inst✝ : Algebra R S₂
        f : AlgHom R S₁ S₂
        r : R
        h₁ : Eq ((algebraMap R (MvPolynomial σ S₁)) r) (MvPolynomial.C ((algebraMap R  …
        h₂ : Eq ((algebraMap R (MvPolynomial σ S₂)) r) (MvPolynomial.C ((algebraMap R  …
        ⊢ Eq ((↑↑__src✝).toFun ((algebraMap R (MvPolynomial σ S₁)) r)) ((algebraMap R  …
      -/
      simp_rw [OneHom.toFun_eq_coe]
      -- Porting note: we're missing some `simp` lemmas like `MonoidHom.coe_toOneHom`
      /-
        R : Type u
        S₁ : Type v
        S₂ : Type w
        S₃ : Type x
        σ : Type u_1
        a a' a₁ a₂ : R
        e : Nat
        n m : σ
        s : Finsupp σ Nat
        inst✝⁴ : CommSemiring R
        inst✝³ : CommSemiring S₁
        p q : MvPolynomial σ R
        f✝ : RingHom R S₁
        inst✝² : CommSemiring S₂
        inst✝¹ : Algebra R S₁
        inst✝ : Algebra R S₂
        f : AlgHom R S₁ S₂
        r : R
        h₁ : Eq ((algebraMap R (MvPolynomial σ S₁)) r) (MvPolynomial.C ((algebraMap R  …
        h₂ : Eq ((algebraMap R (MvPolynomial σ S₂)) r) (MvPolynomial.C ((algebraMap R  …
        ⊢ Eq (↑↑(MvPolynomial.map ↑f) ((algebraMap R (MvPolynomial σ S₁)) r)) ((algebr …
      -/
      change @DFunLike.coe (_ →+* _) _ _ _ _ _ = _
      /-
        R : Type u
        S₁ : Type v
        S₂ : Type w
        S₃ : Type x
        σ : Type u_1
        a a' a₁ a₂ : R
        e : Nat
        n m : σ
        s : Finsupp σ Nat
        inst✝⁴ : CommSemiring R
        inst✝³ : CommSemiring S₁
        p q : MvPolynomial σ R
        f✝ : RingHom R S₁
        inst✝² : CommSemiring S₂
        inst✝¹ : Algebra R S₁
        inst✝ : Algebra R S₂
        f : AlgHom R S₁ S₂
        r : R
        h₁ : Eq ((algebraMap R (MvPolynomial σ S₁)) r) (MvPolynomial.C ((algebraMap R  …
        h₂ : Eq ((algebraMap R (MvPolynomial σ S₂)) r) (MvPolynomial.C ((algebraMap R  …
        ⊢ Eq ((MvPolynomial.map ↑f) ((algebraMap R (MvPolynomial σ S₁)) r)) ((algebraM …
      -/
      rw [h₁, h₂, map, eval₂Hom_C, RingHom.comp_apply, AlgHom.coe_toRingHom, AlgHom.commutes] }
      /-
        🎉 no goals
      -/


@[simp]
theorem mapAlgHom_id [Algebra R S₁] :
    mapAlgHom (AlgHom.id R S₁) = AlgHom.id R (MvPolynomial σ S₁) :=
  AlgHom.ext map_id


@[simp]
theorem mapAlgHom_coe_ringHom [CommSemiring S₂] [Algebra R S₁] [Algebra R S₂] (f : S₁ →ₐ[R] S₂) :
    ↑(mapAlgHom f : _ →ₐ[R] MvPolynomial σ S₂) =
      (map ↑f : MvPolynomial σ S₁ →+* MvPolynomial σ S₂) :=
  RingHom.mk_coe _ _ _ _ _


@[simp]
theorem algebraMap_apply (r : R) : algebraMap R (MvPolynomial σ S₁) r = C (algebraMap R S₁ r) := rfl


/-- A map `σ → S₁` where `S₁` is an algebra over `R` generates an `R`-algebra homomorphism
from multivariate polynomials over `σ` to `S₁`. -/
def aeval : MvPolynomial σ R →ₐ[R] S₁ :=
  { eval₂Hom (algebraMap R S₁) f with commutes' := fun _r => eval₂_C _ _ _ }


theorem aeval_def (p : MvPolynomial σ R) : aeval f p = eval₂ (algebraMap R S₁) f p :=
  rfl


theorem aeval_eq_eval₂Hom (p : MvPolynomial σ R) : aeval f p = eval₂Hom (algebraMap R S₁) f p :=
  rfl


@[simp]
lemma coe_aeval_eq_eval : RingHomClass.toRingHom (MvPolynomial.aeval f) = MvPolynomial.eval f :=
  rfl


@[simp]
theorem aeval_X (s : σ) : aeval f (X s : MvPolynomial _ R) = f s :=
  eval₂_X _ _ _


theorem aeval_C (r : R) : aeval f (C r) = algebraMap R S₁ r :=
  eval₂_C _ _ _


@[simp] theorem aeval_ofNat (n : Nat) [n.AtLeastTwo] :
    aeval f (ofNat(n) : MvPolynomial σ R) = ofNat(n) :=
  map_ofNat _ _


theorem aeval_unique (φ : MvPolynomial σ R →ₐ[R] S₁) : φ = aeval (φ ∘ X) := by
  /-
    R : Type u
    S₁ : Type v
    σ : Type u_1
    inst✝² : CommSemiring R
    inst✝¹ : CommSemiring S₁
    inst✝ : Algebra R S₁
    φ : AlgHom R (MvPolynomial σ R) S₁
    ⊢ Eq φ (MvPolynomial.aeval (Function.comp (⇑φ) MvPolynomial.X))
  -/
  ext i
  /-
    case hf
    R : Type u
    S₁ : Type v
    σ : Type u_1
    inst✝² : CommSemiring R
    inst✝¹ : CommSemiring S₁
    inst✝ : Algebra R S₁
    φ : AlgHom R (MvPolynomial σ R) S₁
    i : σ
    ⊢ Eq (φ (MvPolynomial.X i)) ((MvPolynomial.aeval (Function.comp (⇑φ) MvPolynom …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem aeval_X_left : aeval X = AlgHom.id R (MvPolynomial σ R) :=
  (aeval_unique (AlgHom.id R _)).symm


theorem aeval_X_left_apply (p : MvPolynomial σ R) : aeval X p = p :=
  AlgHom.congr_fun aeval_X_left p


theorem comp_aeval {B : Type*} [CommSemiring B] [Algebra R B] (φ : S₁ →ₐ[R] B) :
    φ.comp (aeval f) = aeval fun i => φ (f i) := by
  /-
    R : Type u
    S₁ : Type v
    σ : Type u_1
    inst✝⁴ : CommSemiring R
    inst✝³ : CommSemiring S₁
    inst✝² : Algebra R S₁
    f : σ → S₁
    B : Type u_2
    inst✝¹ : CommSemiring B
    inst✝ : Algebra R B
    φ : AlgHom R S₁ B
    ⊢ Eq (φ.comp (MvPolynomial.aeval f)) (MvPolynomial.aeval fun i => φ (f i))
  -/
  ext i
  /-
    case hf
    R : Type u
    S₁ : Type v
    σ : Type u_1
    inst✝⁴ : CommSemiring R
    inst✝³ : CommSemiring S₁
    inst✝² : Algebra R S₁
    f : σ → S₁
    B : Type u_2
    inst✝¹ : CommSemiring B
    inst✝ : Algebra R B
    φ : AlgHom R S₁ B
    i : σ
    ⊢ Eq ((φ.comp (MvPolynomial.aeval f)) (MvPolynomial.X i)) ((MvPolynomial.aeval …
  -/
  simp
  /-
    🎉 no goals
  -/


lemma comp_aeval_apply {B : Type*} [CommSemiring B] [Algebra R B] (φ : S₁ →ₐ[R] B)
    (p : MvPolynomial σ R) :
    φ (aeval f p) = aeval (fun i ↦ φ (f i)) p := by
  /-
    R : Type u
    S₁ : Type v
    σ : Type u_1
    inst✝⁴ : CommSemiring R
    inst✝³ : CommSemiring S₁
    inst✝² : Algebra R S₁
    f : σ → S₁
    B : Type u_2
    inst✝¹ : CommSemiring B
    inst✝ : Algebra R B
    φ : AlgHom R S₁ B
    p : MvPolynomial σ R
    ⊢ Eq (φ ((MvPolynomial.aeval f) p)) ((MvPolynomial.aeval fun i => φ (f i)) p)
  -/
  rw [← comp_aeval, AlgHom.coe_comp, comp_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem map_aeval {B : Type*} [CommSemiring B] (g : σ → S₁) (φ : S₁ →+* B) (p : MvPolynomial σ R) :
    φ (aeval g p) = eval₂Hom (φ.comp (algebraMap R S₁)) (fun i => φ (g i)) p := by
  /-
    R : Type u
    S₁ : Type v
    σ : Type u_1
    inst✝³ : CommSemiring R
    inst✝² : CommSemiring S₁
    inst✝¹ : Algebra R S₁
    B : Type u_2
    inst✝ : CommSemiring B
    g : σ → S₁
    φ : RingHom S₁ B
    p : MvPolynomial σ R
    ⊢ Eq (φ ((MvPolynomial.aeval g) p)) ((MvPolynomial.eval₂Hom (φ.comp (algebraMa …
  -/
  rw [← comp_eval₂Hom]
  /-
    R : Type u
    S₁ : Type v
    σ : Type u_1
    inst✝³ : CommSemiring R
    inst✝² : CommSemiring S₁
    inst✝¹ : Algebra R S₁
    B : Type u_2
    inst✝ : CommSemiring B
    g : σ → S₁
    φ : RingHom S₁ B
    p : MvPolynomial σ R
    ⊢ Eq (φ ((MvPolynomial.aeval g) p)) ((φ.comp (MvPolynomial.eval₂Hom (algebraMa …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem eval₂Hom_zero (f : R →+* S₂) : eval₂Hom f (0 : σ → S₂) = f.comp constantCoeff := by
  /-
    R : Type u
    S₂ : Type w
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S₂
    f : RingHom R S₂
    ⊢ Eq (MvPolynomial.eval₂Hom f 0) (f.comp MvPolynomial.constantCoeff)
  -/
          /-
            🎉 no goals
          -/
  ext <;> simp
          /-
            🎉 no goals
          -/


@[simp]
theorem eval₂Hom_zero' (f : R →+* S₂) : eval₂Hom f (fun _ => 0 : σ → S₂) = f.comp constantCoeff :=
  eval₂Hom_zero f


theorem eval₂Hom_zero_apply (f : R →+* S₂) (p : MvPolynomial σ R) :
    eval₂Hom f (0 : σ → S₂) p = f (constantCoeff p) :=
  RingHom.congr_fun (eval₂Hom_zero f) p


theorem eval₂Hom_zero'_apply (f : R →+* S₂) (p : MvPolynomial σ R) :
    eval₂Hom f (fun _ => 0 : σ → S₂) p = f (constantCoeff p) :=
  eval₂Hom_zero_apply f p


@[simp]
theorem eval₂_zero_apply (f : R →+* S₂) (p : MvPolynomial σ R) :
    eval₂ f (0 : σ → S₂) p = f (constantCoeff p) :=
  eval₂Hom_zero_apply _ _


@[simp]
theorem eval₂_zero'_apply (f : R →+* S₂) (p : MvPolynomial σ R) :
    eval₂ f (fun _ => 0 : σ → S₂) p = f (constantCoeff p) :=
  eval₂_zero_apply f p


@[simp]
theorem aeval_zero (p : MvPolynomial σ R) :
    aeval (0 : σ → S₁) p = algebraMap _ _ (constantCoeff p) :=
  eval₂Hom_zero_apply (algebraMap R S₁) p


@[simp]
theorem aeval_zero' (p : MvPolynomial σ R) :
    aeval (fun _ => 0 : σ → S₁) p = algebraMap _ _ (constantCoeff p) :=
  aeval_zero p


@[simp]
theorem eval_zero : eval (0 : σ → R) = constantCoeff :=
  eval₂Hom_zero _


@[simp]
theorem eval_zero' : eval (fun _ => 0 : σ → R) = constantCoeff :=
  eval₂Hom_zero _


theorem aeval_monomial (g : σ → S₁) (d : σ →₀ ℕ) (r : R) :
    aeval g (monomial d r) = algebraMap _ _ r * d.prod fun i k => g i ^ k :=
  eval₂Hom_monomial _ _ _ _


theorem eval₂Hom_eq_zero (f : R →+* S₂) (g : σ → S₂) (φ : MvPolynomial σ R)
    (h : ∀ d, φ.coeff d ≠ 0 → ∃ i ∈ d.support, g i = 0) : eval₂Hom f g φ = 0 := by
  /-
    R : Type u
    S₂ : Type w
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S₂
    f : RingHom R S₂
    g : σ → S₂
    φ : MvPolynomial σ R
    h : ∀ (d : Finsupp σ Nat), Ne (MvPolynomial.coeff d φ) 0 → Exists fun i => And …
    ⊢ Eq ((MvPolynomial.eval₂Hom f g) φ) 0
  -/
  rw [φ.as_sum, map_sum]
  /-
    R : Type u
    S₂ : Type w
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S₂
    f : RingHom R S₂
    g : σ → S₂
    φ : MvPolynomial σ R
    h : ∀ (d : Finsupp σ Nat), Ne (MvPolynomial.coeff d φ) 0 → Exists fun i => And …
    ⊢ Eq (φ.support.sum fun x => (MvPolynomial.eval₂Hom f g) ((MvPolynomial.monomi …
  -/
  refine Finset.sum_eq_zero fun d hd => ?_
  /-
    R : Type u
    S₂ : Type w
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S₂
    f : RingHom R S₂
    g : σ → S₂
    φ : MvPolynomial σ R
    h : ∀ (d : Finsupp σ Nat), Ne (MvPolynomial.coeff d φ) 0 → Exists fun i => And …
    d : Finsupp σ Nat
    hd : Membership.mem φ.support d
    ⊢ Eq ((MvPolynomial.eval₂Hom f g) ((MvPolynomial.monomial d) (MvPolynomial.coe …
  -/
  obtain ⟨i, hi, hgi⟩ : ∃ i ∈ d.support, g i = 0 := h d (Finsupp.mem_support_iff.mp hd)
  /-
    case intro.intro
    R : Type u
    S₂ : Type w
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S₂
    f : RingHom R S₂
    g : σ → S₂
    φ : MvPolynomial σ R
    h : ∀ (d : Finsupp σ Nat), Ne (MvPolynomial.coeff d φ) 0 → Exists fun i => And …
    d : Finsupp σ Nat
    hd : Membership.mem φ.support d
    i : σ
    hi : Membership.mem d.support i
    hgi : Eq (g i) 0
    ⊢ Eq ((MvPolynomial.eval₂Hom f g) ((MvPolynomial.monomial d) (MvPolynomial.coe …
  -/
  rw [eval₂Hom_monomial, Finsupp.prod, Finset.prod_eq_zero hi, mul_zero]
  /-
    case intro.intro
    R : Type u
    S₂ : Type w
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S₂
    f : RingHom R S₂
    g : σ → S₂
    φ : MvPolynomial σ R
    h : ∀ (d : Finsupp σ Nat), Ne (MvPolynomial.coeff d φ) 0 → Exists fun i => And …
    d : Finsupp σ Nat
    hd : Membership.mem φ.support d
    i : σ
    hi : Membership.mem d.support i
    hgi : Eq (g i) 0
    ⊢ Eq (HPow.hPow (g i) (d i)) 0
  -/
  rw [hgi, zero_pow]
  /-
    case intro.intro
    R : Type u
    S₂ : Type w
    σ : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S₂
    f : RingHom R S₂
    g : σ → S₂
    φ : MvPolynomial σ R
    h : ∀ (d : Finsupp σ Nat), Ne (MvPolynomial.coeff d φ) 0 → Exists fun i => And …
    d : Finsupp σ Nat
    hd : Membership.mem φ.support d
    i : σ
    hi : Membership.mem d.support i
    hgi : Eq (g i) 0
    ⊢ Ne (d i) 0
  -/
  rwa [← Finsupp.mem_support_iff]
  /-
    🎉 no goals
  -/


theorem aeval_eq_zero [Algebra R S₂] (f : σ → S₂) (φ : MvPolynomial σ R)
    (h : ∀ d, φ.coeff d ≠ 0 → ∃ i ∈ d.support, f i = 0) : aeval f φ = 0 :=
  eval₂Hom_eq_zero _ _ _ h


theorem aeval_sum {ι : Type*} (s : Finset ι) (φ : ι → MvPolynomial σ R) :
    aeval f (∑ i ∈ s, φ i) = ∑ i ∈ s, aeval f (φ i) :=
  map_sum (MvPolynomial.aeval f) _ _


@[to_additive existing]
theorem aeval_prod {ι : Type*} (s : Finset ι) (φ : ι → MvPolynomial σ R) :
    aeval f (∏ i ∈ s, φ i) = ∏ i ∈ s, aeval f (φ i) :=
  map_prod (MvPolynomial.aeval f) _ _


theorem _root_.Algebra.adjoin_range_eq_range_aeval :
    Algebra.adjoin R (Set.range f) = (MvPolynomial.aeval f).range := by
  simp only [← Algebra.map_top, ← MvPolynomial.adjoin_range_X, AlgHom.map_adjoin, ← Set.range_comp,
    Function.comp_def, MvPolynomial.aeval_X]


theorem _root_.Algebra.adjoin_eq_range (s : Set S₁) :
    Algebra.adjoin R s = (MvPolynomial.aeval ((↑) : s → S₁)).range := by
  /-
    R : Type u
    S₁ : Type v
    inst✝² : CommSemiring R
    inst✝¹ : CommSemiring S₁
    inst✝ : Algebra R S₁
    s : Set S₁
    ⊢ Eq (Algebra.adjoin R s) (MvPolynomial.aeval Subtype.val).range
  -/
  rw [← Algebra.adjoin_range_eq_range_aeval, Subtype.range_coe]
  /-
    🎉 no goals
  -/


/-- Version of `aeval` for defining algebra homs out of `MvPolynomial σ R` over a smaller base ring
  than `R`. -/
def aevalTower (f : R →ₐ[S] A) (X : σ → A) : MvPolynomial σ R →ₐ[S] A :=
  { eval₂Hom (↑f) X with
    commutes' := fun r => by
      /-
        R : Type u
        S₁ : Type v
        S₂ : Type w
        S₃ : Type x
        σ : Type u_1
        a a' a₁ a₂ : R
        e : Nat
        n m : σ
        s : Finsupp σ Nat
        inst✝⁷ : CommSemiring R
        inst✝⁶ : CommSemiring S₁
        p q : MvPolynomial σ R
        S : Type u_2
        A : Type u_3
        B : Type u_4
        inst✝⁵ : CommSemiring S
        inst✝⁴ : CommSemiring A
        inst✝³ : CommSemiring B
        inst✝² : Algebra S R
        inst✝¹ : Algebra S A
        inst✝ : Algebra S B
        f : AlgHom S R A
        X : σ → A
        r : S
        ⊢ Eq ((↑↑__src✝).toFun ((algebraMap S (MvPolynomial σ R)) r)) ((algebraMap S A …
      -/
      simp [IsScalarTower.algebraMap_eq S R (MvPolynomial σ R), algebraMap_eq] }
      /-
        🎉 no goals
      -/


@[simp]
theorem aevalTower_X (i : σ) : aevalTower g y (X i) = y i :=
  eval₂_X _ _ _


@[simp]
theorem aevalTower_C (x : R) : aevalTower g y (C x) = g x :=
  eval₂_C _ _ _


@[simp]
theorem aevalTower_ofNat (n : Nat) [n.AtLeastTwo] :
    aevalTower g y (ofNat(n) : MvPolynomial σ R) = ofNat(n) :=
  _root_.map_ofNat _ _


@[simp]
theorem aevalTower_comp_C : (aevalTower g y : MvPolynomial σ R →+* A).comp C = g :=
  RingHom.ext <| aevalTower_C _ _


theorem aevalTower_algebraMap (x : R) : aevalTower g y (algebraMap R (MvPolynomial σ R) x) = g x :=
  eval₂_C _ _ _


theorem aevalTower_comp_algebraMap :
    (aevalTower g y : MvPolynomial σ R →+* A).comp (algebraMap R (MvPolynomial σ R)) = g :=
  aevalTower_comp_C _ _


theorem aevalTower_toAlgHom (x : R) :
    aevalTower g y (IsScalarTower.toAlgHom S R (MvPolynomial σ R) x) = g x :=
  aevalTower_algebraMap _ _ _


@[simp]
theorem aevalTower_comp_toAlgHom :
    (aevalTower g y).comp (IsScalarTower.toAlgHom S R (MvPolynomial σ R)) = g :=
  AlgHom.coe_ringHom_injective <| aevalTower_comp_algebraMap _ _


@[simp]
theorem aevalTower_id :
    aevalTower (AlgHom.id S S) = (aeval : (σ → S) → MvPolynomial σ S →ₐ[S] S) := by
  /-
    σ : Type u_1
    S : Type u_2
    inst✝ : CommSemiring S
    ⊢ Eq (MvPolynomial.aevalTower (AlgHom.id S S)) MvPolynomial.aeval
  -/
  ext
  /-
    case h.hf
    σ : Type u_1
    S : Type u_2
    inst✝ : CommSemiring S
    x✝ : σ → S
    i✝ : σ
    ⊢ Eq ((MvPolynomial.aevalTower (AlgHom.id S S) x✝) (MvPolynomial.X i✝)) ((MvPo …
  -/
  simp only [aevalTower_X, aeval_X]
  /-
    🎉 no goals
  -/


@[simp]
theorem aevalTower_ofId :
    aevalTower (Algebra.ofId S A) = (aeval : (σ → A) → MvPolynomial σ S →ₐ[S] A) := by
  /-
    σ : Type u_1
    S : Type u_2
    A : Type u_3
    inst✝² : CommSemiring S
    inst✝¹ : CommSemiring A
    inst✝ : Algebra S A
    ⊢ Eq (MvPolynomial.aevalTower (Algebra.ofId S A)) MvPolynomial.aeval
  -/
  ext
  /-
    case h.hf
    σ : Type u_1
    S : Type u_2
    A : Type u_3
    inst✝² : CommSemiring S
    inst✝¹ : CommSemiring A
    inst✝ : Algebra S A
    x✝ : σ → A
    i✝ : σ
    ⊢ Eq ((MvPolynomial.aevalTower (Algebra.ofId S A) x✝) (MvPolynomial.X i✝)) ((M …
  -/
  simp only [aeval_X, aevalTower_X]
  /-
    🎉 no goals
  -/


theorem eval₂_mem {f : R →+* S} {p : MvPolynomial σ R} {s : subS}
    (hs : ∀ i ∈ p.support, f (p.coeff i) ∈ s) {v : σ → S} (hv : ∀ i, v i ∈ s) :
    MvPolynomial.eval₂ f v p ∈ s := by
  classical
  replace hs : ∀ i, f (p.coeff i) ∈ s := by
    intro i
    by_cases hi : i ∈ p.support
    · exact hs i hi
    · rw [MvPolynomial.not_mem_support_iff.1 hi, f.map_zero]
      exact zero_mem s
  induction' p using MvPolynomial.induction_on''' with a a b f ha _ ih
  · simpa using hs 0
  rw [eval₂_add, eval₂_monomial]
  refine add_mem (mul_mem ?_ <| prod_mem fun i _ => pow_mem (hv _) _) (ih fun i => ?_)
  · have := hs a -- Porting note: was `simpa only [...]`
    rwa [coeff_add, MvPolynomial.not_mem_support_iff.1 ha, add_zero, coeff_monomial,
      if_pos rfl] at this
  have := hs i
  rw [coeff_add, coeff_monomial] at this
  split_ifs at this with h
  · subst h
    rw [MvPolynomial.not_mem_support_iff.1 ha, map_zero]
    exact zero_mem _
  · rwa [zero_add] at this


theorem eval_mem {p : MvPolynomial σ S} {s : subS} (hs : ∀ i ∈ p.support, p.coeff i ∈ s) {v : σ → S}
    (hv : ∀ i, v i ∈ s) : MvPolynomial.eval v p ∈ s :=
  eval₂_mem hs hv


lemma aeval_sum_elim {σ τ : Type*} (p : MvPolynomial (σ ⊕ τ) R) (f : τ → S) (g : σ → T) :
    (aeval (Sum.elim g (algebraMap S T ∘ f))) p =
      (aeval g) ((aeval (Sum.elim X (C ∘ f))) p) := by
  /-
    R : Type u
    inst✝⁶ : CommSemiring R
    S : Type u_2
    T : Type u_3
    inst✝⁵ : CommSemiring S
    inst✝⁴ : Algebra R S
    inst✝³ : CommSemiring T
    inst✝² : Algebra R T
    inst✝¹ : Algebra S T
    inst✝ : IsScalarTower R S T
    σ : Type u_4
    τ : Type u_5
    p : MvPolynomial (Sum σ τ) R
    f : τ → S
    g : σ → T
    ⊢ Eq ((MvPolynomial.aeval (Sum.elim g (Function.comp (⇑(algebraMap S T)) f)))  …
  -/
  induction' p using MvPolynomial.induction_on with r p q hp hq p i h
    /-
      case h_C
      R : Type u
      inst✝⁶ : CommSemiring R
      S : Type u_2
      T : Type u_3
      inst✝⁵ : CommSemiring S
      inst✝⁴ : Algebra R S
      inst✝³ : CommSemiring T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      σ : Type u_4
      τ : Type u_5
      f : τ → S
      g : σ → T
      r : R
      ⊢ Eq ((MvPolynomial.aeval (Sum.elim g (Function.comp (⇑(algebraMap S T)) f)))  …
    -/
  · simp [← IsScalarTower.algebraMap_apply]
    /-
      🎉 no goals
    -/
    /-
      case h_add
      R : Type u
      inst✝⁶ : CommSemiring R
      S : Type u_2
      T : Type u_3
      inst✝⁵ : CommSemiring S
      inst✝⁴ : Algebra R S
      inst✝³ : CommSemiring T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      σ : Type u_4
      τ : Type u_5
      f : τ → S
      g : σ → T
      p q : MvPolynomial (Sum σ τ) R
      hp : Eq ((MvPolynomial.aeval (Sum.elim g (Function.comp (⇑(algebraMap S T)) f) …
      hq : Eq ((MvPolynomial.aeval (Sum.elim g (Function.comp (⇑(algebraMap S T)) f) …
      ⊢ Eq ((MvPolynomial.aeval (Sum.elim g (Function.comp (⇑(algebraMap S T)) f)))  …
    -/
  · simp [hp, hq]
    /-
      🎉 no goals
    -/
    /-
      case h_X
      R : Type u
      inst✝⁶ : CommSemiring R
      S : Type u_2
      T : Type u_3
      inst✝⁵ : CommSemiring S
      inst✝⁴ : Algebra R S
      inst✝³ : CommSemiring T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      σ : Type u_4
      τ : Type u_5
      f : τ → S
      g : σ → T
      p : MvPolynomial (Sum σ τ) R
      i : Sum σ τ
      h : Eq ((MvPolynomial.aeval (Sum.elim g (Function.comp (⇑(algebraMap S T)) f)) …
      ⊢ Eq ((MvPolynomial.aeval (Sum.elim g (Function.comp (⇑(algebraMap S T)) f)))  …
    -/
                /-
                  🎉 no goals
                -/
  · cases i <;> simp [h]
                /-
                  🎉 no goals
                -/


