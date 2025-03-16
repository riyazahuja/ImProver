local notation "𝕎" => WittVector p


/-- The root of this polynomial determines the `n+1`st coefficient of our solution. -/
def succNthDefiningPoly (n : ℕ) (a₁ a₂ : 𝕎 k) (bs : Fin (n + 1) → k) : Polynomial k :=
  X ^ p * C (a₁.coeff 0 ^ p ^ (n + 1)) - X * C (a₂.coeff 0 ^ p ^ (n + 1)) +
    C
      (a₁.coeff (n + 1) * (bs 0 ^ p) ^ p ^ (n + 1) +
            nthRemainder p n (fun v => bs v ^ p) (truncateFun (n + 1) a₁) -
          a₂.coeff (n + 1) * bs 0 ^ p ^ (n + 1) -
        nthRemainder p n bs (truncateFun (n + 1) a₂))


theorem succNthDefiningPoly_degree [IsDomain k] (n : ℕ) (a₁ a₂ : 𝕎 k) (bs : Fin (n + 1) → k)
    (ha₁ : a₁.coeff 0 ≠ 0) (ha₂ : a₂.coeff 0 ≠ 0) :
    (succNthDefiningPoly p n a₁ a₂ bs).degree = p := by
  have : (X ^ p * C (a₁.coeff 0 ^ p ^ (n + 1))).degree = (p : WithBot ℕ) := by
    rw [degree_mul, degree_C]
    · simp only [Nat.cast_withBot, add_zero, degree_X, degree_pow, Nat.smul_one_eq_cast]
    · exact pow_ne_zero _ ha₁
  have : (X ^ p * C (a₁.coeff 0 ^ p ^ (n + 1)) - X * C (a₂.coeff 0 ^ p ^ (n + 1))).degree =
      (p : WithBot ℕ) := by
    rw [degree_sub_eq_left_of_degree_lt, this]
    rw [this, degree_mul, degree_C, degree_X, add_zero]
    · exact mod_cast hp.out.one_lt
    · exact pow_ne_zero _ ha₂
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝² : CommRing k
    inst✝¹ : CharP k p
    inst✝ : IsDomain k
    n : Nat
    a₁ a₂ : WittVector p k
    bs : Fin (HAdd.hAdd n 1) → k
    ha₁ : Ne (a₁.coeff 0) 0
    ha₂ : Ne (a₂.coeff 0) 0
    this✝ : Eq (HMul.hMul (HPow.hPow Polynomial.X p) (Polynomial.C (HPow.hPow (a₁. …
    this : Eq (HSub.hSub (HMul.hMul (HPow.hPow Polynomial.X p) (Polynomial.C (HPow …
    ⊢ Eq (WittVector.RecursionMain.succNthDefiningPoly p n a₁ a₂ bs).degree ↑p
  -/
  rw [succNthDefiningPoly, degree_add_eq_left_of_degree_lt, this]
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝² : CommRing k
    inst✝¹ : CharP k p
    inst✝ : IsDomain k
    n : Nat
    a₁ a₂ : WittVector p k
    bs : Fin (HAdd.hAdd n 1) → k
    ha₁ : Ne (a₁.coeff 0) 0
    ha₂ : Ne (a₂.coeff 0) 0
    this✝ : Eq (HMul.hMul (HPow.hPow Polynomial.X p) (Polynomial.C (HPow.hPow (a₁. …
    this : Eq (HSub.hSub (HMul.hMul (HPow.hPow Polynomial.X p) (Polynomial.C (HPow …
    ⊢ LT.lt (Polynomial.C (HSub.hSub (HSub.hSub (HAdd.hAdd (HMul.hMul (a₁.coeff (H …
  -/
  apply lt_of_le_of_lt degree_C_le
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝² : CommRing k
    inst✝¹ : CharP k p
    inst✝ : IsDomain k
    n : Nat
    a₁ a₂ : WittVector p k
    bs : Fin (HAdd.hAdd n 1) → k
    ha₁ : Ne (a₁.coeff 0) 0
    ha₂ : Ne (a₂.coeff 0) 0
    this✝ : Eq (HMul.hMul (HPow.hPow Polynomial.X p) (Polynomial.C (HPow.hPow (a₁. …
    this : Eq (HSub.hSub (HMul.hMul (HPow.hPow Polynomial.X p) (Polynomial.C (HPow …
    ⊢ LT.lt 0 (HSub.hSub (HMul.hMul (HPow.hPow Polynomial.X p) (Polynomial.C (HPow …
  -/
  rw [this]
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝² : CommRing k
    inst✝¹ : CharP k p
    inst✝ : IsDomain k
    n : Nat
    a₁ a₂ : WittVector p k
    bs : Fin (HAdd.hAdd n 1) → k
    ha₁ : Ne (a₁.coeff 0) 0
    ha₂ : Ne (a₂.coeff 0) 0
    this✝ : Eq (HMul.hMul (HPow.hPow Polynomial.X p) (Polynomial.C (HPow.hPow (a₁. …
    this : Eq (HSub.hSub (HMul.hMul (HPow.hPow Polynomial.X p) (Polynomial.C (HPow …
    ⊢ LT.lt 0 ↑p
  -/
  exact mod_cast hp.out.pos
  /-
    🎉 no goals
  -/


theorem root_exists (n : ℕ) (a₁ a₂ : 𝕎 k) (bs : Fin (n + 1) → k) (ha₁ : a₁.coeff 0 ≠ 0)
    (ha₂ : a₂.coeff 0 ≠ 0) : ∃ b : k, (succNthDefiningPoly p n a₁ a₂ bs).IsRoot b :=
  IsAlgClosed.exists_root _ <| by
    simp only [succNthDefiningPoly_degree p n a₁ a₂ bs ha₁ ha₂, ne_eq, Nat.cast_eq_zero,
      hp.out.ne_zero, not_false_eq_true]


/-- This is the `n+1`st coefficient of our solution, projected from `root_exists`. -/
def succNthVal (n : ℕ) (a₁ a₂ : 𝕎 k) (bs : Fin (n + 1) → k) (ha₁ : a₁.coeff 0 ≠ 0)
    (ha₂ : a₂.coeff 0 ≠ 0) : k :=
  Classical.choose (root_exists p n a₁ a₂ bs ha₁ ha₂)


theorem succNthVal_spec (n : ℕ) (a₁ a₂ : 𝕎 k) (bs : Fin (n + 1) → k) (ha₁ : a₁.coeff 0 ≠ 0)
    (ha₂ : a₂.coeff 0 ≠ 0) :
    (succNthDefiningPoly p n a₁ a₂ bs).IsRoot (succNthVal p n a₁ a₂ bs ha₁ ha₂) :=
  Classical.choose_spec (root_exists p n a₁ a₂ bs ha₁ ha₂)


theorem succNthVal_spec' (n : ℕ) (a₁ a₂ : 𝕎 k) (bs : Fin (n + 1) → k) (ha₁ : a₁.coeff 0 ≠ 0)
    (ha₂ : a₂.coeff 0 ≠ 0) :
    succNthVal p n a₁ a₂ bs ha₁ ha₂ ^ p * a₁.coeff 0 ^ p ^ (n + 1) +
          a₁.coeff (n + 1) * (bs 0 ^ p) ^ p ^ (n + 1) +
        nthRemainder p n (fun v => bs v ^ p) (truncateFun (n + 1) a₁) =
      succNthVal p n a₁ a₂ bs ha₁ ha₂ * a₂.coeff 0 ^ p ^ (n + 1) +
          a₂.coeff (n + 1) * bs 0 ^ p ^ (n + 1) +
        nthRemainder p n bs (truncateFun (n + 1) a₂) := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝² : Field k
    inst✝¹ : CharP k p
    inst✝ : IsAlgClosed k
    n : Nat
    a₁ a₂ : WittVector p k
    bs : Fin (HAdd.hAdd n 1) → k
    ha₁ : Ne (a₁.coeff 0) 0
    ha₂ : Ne (a₂.coeff 0) 0
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HPow.hPow (WittVector.RecursionMain.suc …
  -/
  rw [← sub_eq_zero]
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝² : Field k
    inst✝¹ : CharP k p
    inst✝ : IsAlgClosed k
    n : Nat
    a₁ a₂ : WittVector p k
    bs : Fin (HAdd.hAdd n 1) → k
    ha₁ : Ne (a₁.coeff 0) 0
    ha₂ : Ne (a₂.coeff 0) 0
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HPow.hPow (WittVector.Recurs …
  -/
  have := succNthVal_spec p n a₁ a₂ bs ha₁ ha₂
  simp only [Polynomial.map_add, Polynomial.eval_X, Polynomial.map_pow, Polynomial.eval_C,
    Polynomial.eval_pow, succNthDefiningPoly, Polynomial.eval_mul, Polynomial.eval_add,
    Polynomial.eval_sub, Polynomial.map_mul, Polynomial.map_sub, Polynomial.IsRoot.def]
    at this
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝² : Field k
    inst✝¹ : CharP k p
    inst✝ : IsAlgClosed k
    n : Nat
    a₁ a₂ : WittVector p k
    bs : Fin (HAdd.hAdd n 1) → k
    ha₁ : Ne (a₁.coeff 0) 0
    ha₂ : Ne (a₂.coeff 0) 0
    this : Eq (HAdd.hAdd (HSub.hSub (HMul.hMul (HPow.hPow (WittVector.RecursionMai …
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HPow.hPow (WittVector.Recurs …
  -/
  convert this using 1
  /-
    case h.e'_2
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝² : Field k
    inst✝¹ : CharP k p
    inst✝ : IsAlgClosed k
    n : Nat
    a₁ a₂ : WittVector p k
    bs : Fin (HAdd.hAdd n 1) → k
    ha₁ : Ne (a₁.coeff 0) 0
    ha₂ : Ne (a₂.coeff 0) 0
    this : Eq (HAdd.hAdd (HSub.hSub (HMul.hMul (HPow.hPow (WittVector.RecursionMai …
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HPow.hPow (WittVector.Recurs …
  -/
  ring
  /-
    🎉 no goals
  -/


theorem solution_pow (a₁ a₂ : 𝕎 k) : ∃ x : k, x ^ (p - 1) = a₂.coeff 0 / a₁.coeff 0 :=
  IsAlgClosed.exists_pow_nat_eq _ <| tsub_pos_of_lt hp.out.one_lt


/-- The base case (0th coefficient) of our solution vector. -/
def solution (a₁ a₂ : 𝕎 k) : k :=
  Classical.choose <| solution_pow p a₁ a₂


theorem solution_spec (a₁ a₂ : 𝕎 k) : solution p a₁ a₂ ^ (p - 1) = a₂.coeff 0 / a₁.coeff 0 :=
  Classical.choose_spec <| solution_pow p a₁ a₂


theorem solution_nonzero {a₁ a₂ : 𝕎 k} (ha₁ : a₁.coeff 0 ≠ 0) (ha₂ : a₂.coeff 0 ≠ 0) :
    solution p a₁ a₂ ≠ 0 := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝¹ : Field k
    inst✝ : IsAlgClosed k
    a₁ a₂ : WittVector p k
    ha₁ : Ne (a₁.coeff 0) 0
    ha₂ : Ne (a₂.coeff 0) 0
    ⊢ Ne (WittVector.RecursionBase.solution p a₁ a₂) 0
  -/
  intro h
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝¹ : Field k
    inst✝ : IsAlgClosed k
    a₁ a₂ : WittVector p k
    ha₁ : Ne (a₁.coeff 0) 0
    ha₂ : Ne (a₂.coeff 0) 0
    h : Eq (WittVector.RecursionBase.solution p a₁ a₂) 0
    ⊢ False
  -/
  have := solution_spec p a₁ a₂
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝¹ : Field k
    inst✝ : IsAlgClosed k
    a₁ a₂ : WittVector p k
    ha₁ : Ne (a₁.coeff 0) 0
    ha₂ : Ne (a₂.coeff 0) 0
    h : Eq (WittVector.RecursionBase.solution p a₁ a₂) 0
    this : Eq (HPow.hPow (WittVector.RecursionBase.solution p a₁ a₂) (HSub.hSub p  …
    ⊢ False
  -/
  rw [h, zero_pow] at this
    /-
      p : Nat
      hp : Fact (Nat.Prime p)
      k : Type u_1
      inst✝¹ : Field k
      inst✝ : IsAlgClosed k
      a₁ a₂ : WittVector p k
      ha₁ : Ne (a₁.coeff 0) 0
      ha₂ : Ne (a₂.coeff 0) 0
      h : Eq (WittVector.RecursionBase.solution p a₁ a₂) 0
      this : Eq 0 (HDiv.hDiv (a₂.coeff 0) (a₁.coeff 0))
      ⊢ False
    -/
  · simpa [ha₁, ha₂] using _root_.div_eq_zero_iff.mp this.symm
    /-
      🎉 no goals
    -/
    /-
      p : Nat
      hp : Fact (Nat.Prime p)
      k : Type u_1
      inst✝¹ : Field k
      inst✝ : IsAlgClosed k
      a₁ a₂ : WittVector p k
      ha₁ : Ne (a₁.coeff 0) 0
      ha₂ : Ne (a₂.coeff 0) 0
      h : Eq (WittVector.RecursionBase.solution p a₁ a₂) 0
      this : Eq (HPow.hPow 0 (HSub.hSub p 1)) (HDiv.hDiv (a₂.coeff 0) (a₁.coeff 0))
      ⊢ Ne (HSub.hSub p 1) 0
    -/
  · exact Nat.sub_ne_zero_of_lt hp.out.one_lt
    /-
      🎉 no goals
    -/


theorem solution_spec' {a₁ : 𝕎 k} (ha₁ : a₁.coeff 0 ≠ 0) (a₂ : 𝕎 k) :
    solution p a₁ a₂ ^ p * a₁.coeff 0 = solution p a₁ a₂ * a₂.coeff 0 := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝¹ : Field k
    inst✝ : IsAlgClosed k
    a₁ : WittVector p k
    ha₁ : Ne (a₁.coeff 0) 0
    a₂ : WittVector p k
    ⊢ Eq (HMul.hMul (HPow.hPow (WittVector.RecursionBase.solution p a₁ a₂) p) (a₁. …
  -/
  have := solution_spec p a₁ a₂
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝¹ : Field k
    inst✝ : IsAlgClosed k
    a₁ : WittVector p k
    ha₁ : Ne (a₁.coeff 0) 0
    a₂ : WittVector p k
    this : Eq (HPow.hPow (WittVector.RecursionBase.solution p a₁ a₂) (HSub.hSub p  …
    ⊢ Eq (HMul.hMul (HPow.hPow (WittVector.RecursionBase.solution p a₁ a₂) p) (a₁. …
  -/
  cases' Nat.exists_eq_succ_of_ne_zero hp.out.ne_zero with q hq
  /-
    case intro
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝¹ : Field k
    inst✝ : IsAlgClosed k
    a₁ : WittVector p k
    ha₁ : Ne (a₁.coeff 0) 0
    a₂ : WittVector p k
    this : Eq (HPow.hPow (WittVector.RecursionBase.solution p a₁ a₂) (HSub.hSub p  …
    q : Nat
    hq : Eq p q.succ
    ⊢ Eq (HMul.hMul (HPow.hPow (WittVector.RecursionBase.solution p a₁ a₂) p) (a₁. …
  -/
  have hq' : q = p - 1 := by simp only [hq, tsub_zero, Nat.succ_sub_succ_eq_sub]
  conv_lhs =>
    congr
    congr
    · skip
    · rw [hq]
  /-
    case intro
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝¹ : Field k
    inst✝ : IsAlgClosed k
    a₁ : WittVector p k
    ha₁ : Ne (a₁.coeff 0) 0
    a₂ : WittVector p k
    this : Eq (HPow.hPow (WittVector.RecursionBase.solution p a₁ a₂) (HSub.hSub p  …
    q : Nat
    hq : Eq p q.succ
    hq' : Eq q (HSub.hSub p 1)
    ⊢ Eq (HMul.hMul (HPow.hPow (WittVector.RecursionBase.solution p a₁ a₂) q.succ) …
  -/
  rw [pow_succ', hq', this]
  /-
    case intro
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝¹ : Field k
    inst✝ : IsAlgClosed k
    a₁ : WittVector p k
    ha₁ : Ne (a₁.coeff 0) 0
    a₂ : WittVector p k
    this : Eq (HPow.hPow (WittVector.RecursionBase.solution p a₁ a₂) (HSub.hSub p  …
    q : Nat
    hq : Eq p q.succ
    hq' : Eq q (HSub.hSub p 1)
    ⊢ Eq (HMul.hMul (HMul.hMul (WittVector.RecursionBase.solution p a₁ a₂) (HDiv.h …
  -/
  field_simp [ha₁, mul_comm]
  /-
    🎉 no goals
  -/


/-- Recursively defines the sequence of coefficients for `WittVector.frobeniusRotation`.
-/
-- Constructions by well-founded recursion are by default irreducible.
-- As we rely on definitional properties below, we mark this `@[semireducible]`.
@[semireducible] noncomputable def frobeniusRotationCoeff {a₁ a₂ : 𝕎 k} (ha₁ : a₁.coeff 0 ≠ 0)
    (ha₂ : a₂.coeff 0 ≠ 0) : ℕ → k
  | 0 => solution p a₁ a₂
  | n + 1 => succNthVal p n a₁ a₂ (fun i => frobeniusRotationCoeff ha₁ ha₂ i.val) ha₁ ha₂


/-- For nonzero `a₁` and `a₂`, `frobeniusRotation a₁ a₂` is a Witt vector that satisfies the
equation `frobenius (frobeniusRotation a₁ a₂) * a₁ = (frobeniusRotation a₁ a₂) * a₂`.
-/
def frobeniusRotation {a₁ a₂ : 𝕎 k} (ha₁ : a₁.coeff 0 ≠ 0) (ha₂ : a₂.coeff 0 ≠ 0) : 𝕎 k :=
  WittVector.mk p (frobeniusRotationCoeff p ha₁ ha₂)


theorem frobeniusRotation_nonzero {a₁ a₂ : 𝕎 k} (ha₁ : a₁.coeff 0 ≠ 0) (ha₂ : a₂.coeff 0 ≠ 0) :
    frobeniusRotation p ha₁ ha₂ ≠ 0 := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝² : Field k
    inst✝¹ : CharP k p
    inst✝ : IsAlgClosed k
    a₁ a₂ : WittVector p k
    ha₁ : Ne (a₁.coeff 0) 0
    ha₂ : Ne (a₂.coeff 0) 0
    ⊢ Ne (WittVector.frobeniusRotation p ha₁ ha₂) 0
  -/
  intro h
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝² : Field k
    inst✝¹ : CharP k p
    inst✝ : IsAlgClosed k
    a₁ a₂ : WittVector p k
    ha₁ : Ne (a₁.coeff 0) 0
    ha₂ : Ne (a₂.coeff 0) 0
    h : Eq (WittVector.frobeniusRotation p ha₁ ha₂) 0
    ⊢ False
  -/
  apply solution_nonzero p ha₁ ha₂
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝² : Field k
    inst✝¹ : CharP k p
    inst✝ : IsAlgClosed k
    a₁ a₂ : WittVector p k
    ha₁ : Ne (a₁.coeff 0) 0
    ha₂ : Ne (a₂.coeff 0) 0
    h : Eq (WittVector.frobeniusRotation p ha₁ ha₂) 0
    ⊢ Eq (WittVector.RecursionBase.solution p a₁ a₂) 0
  -/
  simpa [← h, frobeniusRotation, frobeniusRotationCoeff] using WittVector.zero_coeff p k 0
  /-
    🎉 no goals
  -/


theorem frobenius_frobeniusRotation {a₁ a₂ : 𝕎 k} (ha₁ : a₁.coeff 0 ≠ 0) (ha₂ : a₂.coeff 0 ≠ 0) :
    frobenius (frobeniusRotation p ha₁ ha₂) * a₁ = frobeniusRotation p ha₁ ha₂ * a₂ := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝² : Field k
    inst✝¹ : CharP k p
    inst✝ : IsAlgClosed k
    a₁ a₂ : WittVector p k
    ha₁ : Ne (a₁.coeff 0) 0
    ha₂ : Ne (a₂.coeff 0) 0
    ⊢ Eq (HMul.hMul (WittVector.frobenius (WittVector.frobeniusRotation p ha₁ ha₂) …
  -/
  ext n
  /-
    case h
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝² : Field k
    inst✝¹ : CharP k p
    inst✝ : IsAlgClosed k
    a₁ a₂ : WittVector p k
    ha₁ : Ne (a₁.coeff 0) 0
    ha₂ : Ne (a₂.coeff 0) 0
    n : Nat
    ⊢ Eq ((HMul.hMul (WittVector.frobenius (WittVector.frobeniusRotation p ha₁ ha₂ …
  -/
  cases' n with n
  · simp only [WittVector.mul_coeff_zero, WittVector.coeff_frobenius_charP, frobeniusRotation,
      frobeniusRotationCoeff]
    /-
      case h.zero
      p : Nat
      hp : Fact (Nat.Prime p)
      k : Type u_1
      inst✝² : Field k
      inst✝¹ : CharP k p
      inst✝ : IsAlgClosed k
      a₁ a₂ : WittVector p k
      ha₁ : Ne (a₁.coeff 0) 0
      ha₂ : Ne (a₂.coeff 0) 0
      ⊢ Eq (HMul.hMul (HPow.hPow ((WittVector.mk p (WittVector.frobeniusRotationCoef …
    -/
    apply solution_spec' _ ha₁
    /-
      🎉 no goals
    -/
  · simp only [nthRemainder_spec, WittVector.coeff_frobenius_charP, frobeniusRotationCoeff,
      frobeniusRotation]
    have :=
      succNthVal_spec' p n a₁ a₂ (fun i : Fin (n + 1) => frobeniusRotationCoeff p ha₁ ha₂ i.val)
        ha₁ ha₂
    /-
      case h.succ
      p : Nat
      hp : Fact (Nat.Prime p)
      k : Type u_1
      inst✝² : Field k
      inst✝¹ : CharP k p
      inst✝ : IsAlgClosed k
      a₁ a₂ : WittVector p k
      ha₁ : Ne (a₁.coeff 0) 0
      ha₂ : Ne (a₂.coeff 0) 0
      n : Nat
      this : Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HPow.hPow (WittVector.RecursionMai …
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HPow.hPow ((WittVector.mk p (WittVector …
    -/
    simp only [frobeniusRotationCoeff, Fin.val_zero] at this
    /-
      case h.succ
      p : Nat
      hp : Fact (Nat.Prime p)
      k : Type u_1
      inst✝² : Field k
      inst✝¹ : CharP k p
      inst✝ : IsAlgClosed k
      a₁ a₂ : WittVector p k
      ha₁ : Ne (a₁.coeff 0) 0
      ha₂ : Ne (a₂.coeff 0) 0
      n : Nat
      this : Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HPow.hPow (WittVector.RecursionMai …
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HPow.hPow ((WittVector.mk p (WittVector …
    -/
    convert this using 3
    /-
      case h.e'_2.h.e'_6.h.e'_7
      p : Nat
      hp : Fact (Nat.Prime p)
      k : Type u_1
      inst✝² : Field k
      inst✝¹ : CharP k p
      inst✝ : IsAlgClosed k
      a₁ a₂ : WittVector p k
      ha₁ : Ne (a₁.coeff 0) 0
      ha₂ : Ne (a₂.coeff 0) 0
      n : Nat
      this : Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HPow.hPow (WittVector.RecursionMai …
      ⊢ Eq (WittVector.truncateFun (HAdd.hAdd n 1) (WittVector.frobenius (WittVector …
    -/
    apply TruncatedWittVector.ext
    /-
      case h.e'_2.h.e'_6.h.e'_7.h
      p : Nat
      hp : Fact (Nat.Prime p)
      k : Type u_1
      inst✝² : Field k
      inst✝¹ : CharP k p
      inst✝ : IsAlgClosed k
      a₁ a₂ : WittVector p k
      ha₁ : Ne (a₁.coeff 0) 0
      ha₂ : Ne (a₂.coeff 0) 0
      n : Nat
      this : Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HPow.hPow (WittVector.RecursionMai …
      ⊢ ∀ (i : Fin (HAdd.hAdd n 1)), Eq (TruncatedWittVector.coeff i (WittVector.tru …
    -/
    intro i
    /-
      case h.e'_2.h.e'_6.h.e'_7.h
      p : Nat
      hp : Fact (Nat.Prime p)
      k : Type u_1
      inst✝² : Field k
      inst✝¹ : CharP k p
      inst✝ : IsAlgClosed k
      a₁ a₂ : WittVector p k
      ha₁ : Ne (a₁.coeff 0) 0
      ha₂ : Ne (a₂.coeff 0) 0
      n : Nat
      this : Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HPow.hPow (WittVector.RecursionMai …
      i : Fin (HAdd.hAdd n 1)
      ⊢ Eq (TruncatedWittVector.coeff i (WittVector.truncateFun (HAdd.hAdd n 1) (Wit …
    -/
    simp only [WittVector.coeff_truncateFun, WittVector.coeff_frobenius_charP]
    /-
      case h.e'_2.h.e'_6.h.e'_7.h
      p : Nat
      hp : Fact (Nat.Prime p)
      k : Type u_1
      inst✝² : Field k
      inst✝¹ : CharP k p
      inst✝ : IsAlgClosed k
      a₁ a₂ : WittVector p k
      ha₁ : Ne (a₁.coeff 0) 0
      ha₂ : Ne (a₂.coeff 0) 0
      n : Nat
      this : Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HPow.hPow (WittVector.RecursionMai …
      i : Fin (HAdd.hAdd n 1)
      ⊢ Eq (HPow.hPow ((WittVector.mk p (WittVector.frobeniusRotationCoeff p ha₁ ha₂ …
    -/
    rfl
    /-
      🎉 no goals
    -/


local notation "φ" => IsFractionRing.ringEquivOfRingEquiv (frobeniusEquiv p k)


theorem exists_frobenius_solution_fractionRing_aux (m n : ℕ) (r' q' : 𝕎 k) (hr' : r'.coeff 0 ≠ 0)
    (hq' : q'.coeff 0 ≠ 0) (hq : (p : 𝕎 k) ^ n * q' ∈ nonZeroDivisors (𝕎 k)) :
    let b : 𝕎 k := frobeniusRotation p hr' hq'
    IsFractionRing.ringEquivOfRingEquiv (frobeniusEquiv p k)
          (algebraMap (𝕎 k) (FractionRing (𝕎 k)) b) *
        Localization.mk ((p : 𝕎 k) ^ m * r') ⟨(p : 𝕎 k) ^ n * q', hq⟩ =
      (p : Localization (nonZeroDivisors (𝕎 k))) ^ (m - n : ℤ) *
        algebraMap (𝕎 k) (FractionRing (𝕎 k)) b := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝² : Field k
    inst✝¹ : CharP k p
    inst✝ : IsAlgClosed k
    m n : Nat
    r' q' : WittVector p k
    hr' : Ne (r'.coeff 0) 0
    hq' : Ne (q'.coeff 0) 0
    hq : Membership.mem (nonZeroDivisors (WittVector p k)) (HMul.hMul (HPow.hPow ( …
    ⊢ let b := WittVector.frobeniusRotation p hr' hq';
      Eq (HMul.hMul ((IsFractionRing.ringEquivOfRingEquiv (WittVector.frobeniusEqu …
  -/
  intro b
  have key : WittVector.frobenius b * (p : 𝕎 k) ^ m * r' * (p : 𝕎 k) ^ n =
      (p : 𝕎 k) ^ m * b * ((p : 𝕎 k) ^ n * q') := by
    have H := congr_arg (fun x : 𝕎 k => x * (p : 𝕎 k) ^ m * (p : 𝕎 k) ^ n)
      (frobenius_frobeniusRotation p hr' hq')
    dsimp at H
    refine (Eq.trans ?_ H).trans ?_ <;> ring
  have hq'' : algebraMap (𝕎 k) (FractionRing (𝕎 k)) q' ≠ 0 := by
    have hq''' : q' ≠ 0 := fun h => hq' (by simp [h])
    simpa only [Ne, map_zero] using
      (IsFractionRing.injective (𝕎 k) (FractionRing (𝕎 k))).ne hq'''
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝² : Field k
    inst✝¹ : CharP k p
    inst✝ : IsAlgClosed k
    m n : Nat
    r' q' : WittVector p k
    hr' : Ne (r'.coeff 0) 0
    hq' : Ne (q'.coeff 0) 0
    hq : Membership.mem (nonZeroDivisors (WittVector p k)) (HMul.hMul (HPow.hPow ( …
    b : WittVector p k := WittVector.frobeniusRotation p hr' hq'
    key : Eq (HMul.hMul (HMul.hMul (HMul.hMul (WittVector.frobenius b) (HPow.hPow  …
    hq'' : Ne ((algebraMap (WittVector p k) (FractionRing (WittVector p k))) q') 0
    ⊢ Eq (HMul.hMul ((IsFractionRing.ringEquivOfRingEquiv (WittVector.frobeniusEqu …
  -/
  rw [zpow_sub₀ (FractionRing.p_nonzero p k)]
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝² : Field k
    inst✝¹ : CharP k p
    inst✝ : IsAlgClosed k
    m n : Nat
    r' q' : WittVector p k
    hr' : Ne (r'.coeff 0) 0
    hq' : Ne (q'.coeff 0) 0
    hq : Membership.mem (nonZeroDivisors (WittVector p k)) (HMul.hMul (HPow.hPow ( …
    b : WittVector p k := WittVector.frobeniusRotation p hr' hq'
    key : Eq (HMul.hMul (HMul.hMul (HMul.hMul (WittVector.frobenius b) (HPow.hPow  …
    hq'' : Ne ((algebraMap (WittVector p k) (FractionRing (WittVector p k))) q') 0
    ⊢ Eq (HMul.hMul ((IsFractionRing.ringEquivOfRingEquiv (WittVector.frobeniusEqu …
  -/
  field_simp [FractionRing.p_nonzero p k]
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝² : Field k
    inst✝¹ : CharP k p
    inst✝ : IsAlgClosed k
    m n : Nat
    r' q' : WittVector p k
    hr' : Ne (r'.coeff 0) 0
    hq' : Ne (q'.coeff 0) 0
    hq : Membership.mem (nonZeroDivisors (WittVector p k)) (HMul.hMul (HPow.hPow ( …
    b : WittVector p k := WittVector.frobeniusRotation p hr' hq'
    key : Eq (HMul.hMul (HMul.hMul (HMul.hMul (WittVector.frobenius b) (HPow.hPow  …
    hq'' : Ne ((algebraMap (WittVector p k) (FractionRing (WittVector p k))) q') 0
    ⊢ Eq (HMul.hMul (HMul.hMul ((algebraMap (WittVector p k) (Localization (nonZer …
  -/
  convert congr_arg (fun x => algebraMap (𝕎 k) (FractionRing (𝕎 k)) x) key using 1
    /-
      case h.e'_2
      p : Nat
      hp : Fact (Nat.Prime p)
      k : Type u_1
      inst✝² : Field k
      inst✝¹ : CharP k p
      inst✝ : IsAlgClosed k
      m n : Nat
      r' q' : WittVector p k
      hr' : Ne (r'.coeff 0) 0
      hq' : Ne (q'.coeff 0) 0
      hq : Membership.mem (nonZeroDivisors (WittVector p k)) (HMul.hMul (HPow.hPow ( …
      b : WittVector p k := WittVector.frobeniusRotation p hr' hq'
      key : Eq (HMul.hMul (HMul.hMul (HMul.hMul (WittVector.frobenius b) (HPow.hPow  …
      hq'' : Ne ((algebraMap (WittVector p k) (FractionRing (WittVector p k))) q') 0
      ⊢ Eq (HMul.hMul (HMul.hMul ((algebraMap (WittVector p k) (Localization (nonZer …
    -/
  · simp only [RingHom.map_mul, RingHom.map_pow, map_natCast, frobeniusEquiv_apply]
    /-
      case h.e'_2
      p : Nat
      hp : Fact (Nat.Prime p)
      k : Type u_1
      inst✝² : Field k
      inst✝¹ : CharP k p
      inst✝ : IsAlgClosed k
      m n : Nat
      r' q' : WittVector p k
      hr' : Ne (r'.coeff 0) 0
      hq' : Ne (q'.coeff 0) 0
      hq : Membership.mem (nonZeroDivisors (WittVector p k)) (HMul.hMul (HPow.hPow ( …
      b : WittVector p k := WittVector.frobeniusRotation p hr' hq'
      key : Eq (HMul.hMul (HMul.hMul (HMul.hMul (WittVector.frobenius b) (HPow.hPow  …
      hq'' : Ne ((algebraMap (WittVector p k) (FractionRing (WittVector p k))) q') 0
      ⊢ Eq (HMul.hMul (HMul.hMul ((algebraMap (WittVector p k) (Localization (nonZer …
    -/
    ring
    /-
      🎉 no goals
    -/
    /-
      case h.e'_3
      p : Nat
      hp : Fact (Nat.Prime p)
      k : Type u_1
      inst✝² : Field k
      inst✝¹ : CharP k p
      inst✝ : IsAlgClosed k
      m n : Nat
      r' q' : WittVector p k
      hr' : Ne (r'.coeff 0) 0
      hq' : Ne (q'.coeff 0) 0
      hq : Membership.mem (nonZeroDivisors (WittVector p k)) (HMul.hMul (HPow.hPow ( …
      b : WittVector p k := WittVector.frobeniusRotation p hr' hq'
      key : Eq (HMul.hMul (HMul.hMul (HMul.hMul (WittVector.frobenius b) (HPow.hPow  …
      hq'' : Ne ((algebraMap (WittVector p k) (FractionRing (WittVector p k))) q') 0
      ⊢ Eq (HMul.hMul (HMul.hMul (HPow.hPow (↑p) m) ((algebraMap (WittVector p k) (F …
    -/
  · simp only [RingHom.map_mul, RingHom.map_pow, map_natCast]
    /-
      🎉 no goals
    -/


theorem exists_frobenius_solution_fractionRing {a : FractionRing (𝕎 k)} (ha : a ≠ 0) :
    ∃ᵉ (b ≠ 0) (m : ℤ), φ b * a = (p : FractionRing (𝕎 k)) ^ m * b := by
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝² : Field k
    inst✝¹ : CharP k p
    inst✝ : IsAlgClosed k
    a : FractionRing (WittVector p k)
    ha : Ne a 0
    ⊢ Exists fun b => And (Ne b 0) (Exists fun m => Eq (HMul.hMul ((IsFractionRing …
  -/
  revert ha
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝² : Field k
    inst✝¹ : CharP k p
    inst✝ : IsAlgClosed k
    a : FractionRing (WittVector p k)
    ⊢ Ne a 0 → Exists fun b => And (Ne b 0) (Exists fun m => Eq (HMul.hMul ((IsFra …
  -/
  refine Localization.induction_on a ?_
  /-
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝² : Field k
    inst✝¹ : CharP k p
    inst✝ : IsAlgClosed k
    a : FractionRing (WittVector p k)
    ⊢ ∀ (y : Prod (WittVector p k) (Subtype fun x => Membership.mem (nonZeroDiviso …
  -/
  rintro ⟨r, q, hq⟩ hrq
  /-
    case mk.mk
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝² : Field k
    inst✝¹ : CharP k p
    inst✝ : IsAlgClosed k
    a : FractionRing (WittVector p k)
    r q : WittVector p k
    hq : Membership.mem (nonZeroDivisors (WittVector p k)) q
    hrq : Ne (Localization.mk { fst := r, snd := ⟨q, hq⟩ }.1 { fst := r, snd := ⟨q …
    ⊢ Exists fun b => And (Ne b 0) (Exists fun m => Eq (HMul.hMul ((IsFractionRing …
  -/
  have hq0 : q ≠ 0 := mem_nonZeroDivisors_iff_ne_zero.1 hq
  /-
    case mk.mk
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝² : Field k
    inst✝¹ : CharP k p
    inst✝ : IsAlgClosed k
    a : FractionRing (WittVector p k)
    r q : WittVector p k
    hq : Membership.mem (nonZeroDivisors (WittVector p k)) q
    hrq : Ne (Localization.mk { fst := r, snd := ⟨q, hq⟩ }.1 { fst := r, snd := ⟨q …
    hq0 : Ne q 0
    ⊢ Exists fun b => And (Ne b 0) (Exists fun m => Eq (HMul.hMul ((IsFractionRing …
  -/
  have hr0 : r ≠ 0 := fun h => hrq (by simp [h])
  /-
    case mk.mk
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝² : Field k
    inst✝¹ : CharP k p
    inst✝ : IsAlgClosed k
    a : FractionRing (WittVector p k)
    r q : WittVector p k
    hq : Membership.mem (nonZeroDivisors (WittVector p k)) q
    hrq : Ne (Localization.mk { fst := r, snd := ⟨q, hq⟩ }.1 { fst := r, snd := ⟨q …
    hq0 : Ne q 0
    hr0 : Ne r 0
    ⊢ Exists fun b => And (Ne b 0) (Exists fun m => Eq (HMul.hMul ((IsFractionRing …
  -/
  obtain ⟨m, r', hr', rfl⟩ := exists_eq_pow_p_mul r hr0
  /-
    case mk.mk.intro.intro.intro
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝² : Field k
    inst✝¹ : CharP k p
    inst✝ : IsAlgClosed k
    a : FractionRing (WittVector p k)
    q : WittVector p k
    hq : Membership.mem (nonZeroDivisors (WittVector p k)) q
    hq0 : Ne q 0
    m : Nat
    r' : WittVector p k
    hr' : Ne (r'.coeff 0) 0
    hrq : Ne (Localization.mk { fst := HMul.hMul (HPow.hPow (↑p) m) r', snd := ⟨q, …
    hr0 : Ne (HMul.hMul (HPow.hPow (↑p) m) r') 0
    ⊢ Exists fun b => And (Ne b 0) (Exists fun m_1 => Eq (HMul.hMul ((IsFractionRi …
  -/
  obtain ⟨n, q', hq', rfl⟩ := exists_eq_pow_p_mul q hq0
  /-
    case mk.mk.intro.intro.intro.intro.intro.intro
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝² : Field k
    inst✝¹ : CharP k p
    inst✝ : IsAlgClosed k
    a : FractionRing (WittVector p k)
    m : Nat
    r' : WittVector p k
    hr' : Ne (r'.coeff 0) 0
    hr0 : Ne (HMul.hMul (HPow.hPow (↑p) m) r') 0
    n : Nat
    q' : WittVector p k
    hq' : Ne (q'.coeff 0) 0
    hq : Membership.mem (nonZeroDivisors (WittVector p k)) (HMul.hMul (HPow.hPow ( …
    hq0 : Ne (HMul.hMul (HPow.hPow (↑p) n) q') 0
    hrq : Ne (Localization.mk { fst := HMul.hMul (HPow.hPow (↑p) m) r', snd := ⟨HM …
    ⊢ Exists fun b => And (Ne b 0) (Exists fun m_1 => Eq (HMul.hMul ((IsFractionRi …
  -/
  let b := frobeniusRotation p hr' hq'
  /-
    case mk.mk.intro.intro.intro.intro.intro.intro
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝² : Field k
    inst✝¹ : CharP k p
    inst✝ : IsAlgClosed k
    a : FractionRing (WittVector p k)
    m : Nat
    r' : WittVector p k
    hr' : Ne (r'.coeff 0) 0
    hr0 : Ne (HMul.hMul (HPow.hPow (↑p) m) r') 0
    n : Nat
    q' : WittVector p k
    hq' : Ne (q'.coeff 0) 0
    hq : Membership.mem (nonZeroDivisors (WittVector p k)) (HMul.hMul (HPow.hPow ( …
    hq0 : Ne (HMul.hMul (HPow.hPow (↑p) n) q') 0
    hrq : Ne (Localization.mk { fst := HMul.hMul (HPow.hPow (↑p) m) r', snd := ⟨HM …
    b : WittVector p k := WittVector.frobeniusRotation p hr' hq'
    ⊢ Exists fun b => And (Ne b 0) (Exists fun m_1 => Eq (HMul.hMul ((IsFractionRi …
  -/
  refine ⟨algebraMap (𝕎 k) (FractionRing (𝕎 k)) b, ?_, m - n, ?_⟩
  · simpa only [map_zero] using
      (IsFractionRing.injective (WittVector p k) (FractionRing (WittVector p k))).ne
        (frobeniusRotation_nonzero p hr' hq')
  /-
    case mk.mk.intro.intro.intro.intro.intro.intro.refine_2
    p : Nat
    hp : Fact (Nat.Prime p)
    k : Type u_1
    inst✝² : Field k
    inst✝¹ : CharP k p
    inst✝ : IsAlgClosed k
    a : FractionRing (WittVector p k)
    m : Nat
    r' : WittVector p k
    hr' : Ne (r'.coeff 0) 0
    hr0 : Ne (HMul.hMul (HPow.hPow (↑p) m) r') 0
    n : Nat
    q' : WittVector p k
    hq' : Ne (q'.coeff 0) 0
    hq : Membership.mem (nonZeroDivisors (WittVector p k)) (HMul.hMul (HPow.hPow ( …
    hq0 : Ne (HMul.hMul (HPow.hPow (↑p) n) q') 0
    hrq : Ne (Localization.mk { fst := HMul.hMul (HPow.hPow (↑p) m) r', snd := ⟨HM …
    b : WittVector p k := WittVector.frobeniusRotation p hr' hq'
    ⊢ Eq (HMul.hMul ((IsFractionRing.ringEquivOfRingEquiv (WittVector.frobeniusEqu …
  -/
  exact exists_frobenius_solution_fractionRing_aux p m n r' q' hr' hq' hq
  /-
    🎉 no goals
  -/


