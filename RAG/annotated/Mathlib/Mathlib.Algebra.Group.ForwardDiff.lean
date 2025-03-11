/--
Forward difference operator, `fwdDiff h f n = f (n + h) - f n`. The notation `Δ_[h]` for this
operator is available in the `fwdDiff` namespace.
-/
def fwdDiff (h : M) (f : M → G) : M → G := fun n ↦ f (n + h) - f n


@[inherit_doc] scoped[fwdDiff] notation "Δ_[" h "]" => fwdDiff h


@[simp] lemma fwdDiff_add (h : M) (f g : M → G) :
    Δ_[h] (f + g) = Δ_[h] f + Δ_[h] g :=
  add_sub_add_comm ..


@[simp] lemma fwdDiff_const (g : G) : Δ_[h] (fun _ ↦ g : M → G) = fun _ ↦ 0 :=
  funext fun _ ↦ sub_self g


lemma fwdDiff_smul {R : Type} [Ring R] [Module R G] (f : M → R) (g : M → G) :
    Δ_[h] (f • g) = Δ_[h] f • g + f • Δ_[h] g + Δ_[h] f • Δ_[h] g := by
  /-
    M : Type u_1
    G : Type u_2
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommGroup G
    h : M
    R : Type
    inst✝¹ : Ring R
    inst✝ : Module R G
    f : M → R
    g : M → G
    ⊢ Eq (fwdDiff h (HSMul.hSMul f g)) (HAdd.hAdd (HAdd.hAdd (HSMul.hSMul (fwdDiff …
  -/
  ext y
  /-
    case h
    M : Type u_1
    G : Type u_2
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommGroup G
    h : M
    R : Type
    inst✝¹ : Ring R
    inst✝ : Module R G
    f : M → R
    g : M → G
    y : M
    ⊢ Eq (fwdDiff h (HSMul.hSMul f g) y) (HAdd.hAdd (HAdd.hAdd (HSMul.hSMul (fwdDi …
  -/
  simp only [fwdDiff, Pi.smul_apply', Pi.add_apply, smul_sub, sub_smul]
  /-
    case h
    M : Type u_1
    G : Type u_2
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommGroup G
    h : M
    R : Type
    inst✝¹ : Ring R
    inst✝ : Module R G
    f : M → R
    g : M → G
    y : M
    ⊢ Eq (HSub.hSub (HSMul.hSMul (f (HAdd.hAdd y h)) (g (HAdd.hAdd y h))) (HSMul.h …
  -/
  /-
    🎉 no goals
  -/
  abel
  /-
    🎉 no goals
  -/

-- Note `fwdDiff_const_smul` is more general than `fwdDiff_smul` since it allows `R` to be a
-- semiring, rather than a ring; in particular `R = ℕ` is allowed.

@[simp] lemma fwdDiff_const_smul {R : Type*} [Semiring R] [Module R G] (r : R) (f : M → G) :
    Δ_[h] (r • f) = r • Δ_[h] f :=
  funext fun _ ↦ (smul_sub ..).symm


@[simp] lemma fwdDiff_smul_const {R : Type} [Ring R] [Module R G] (f : M → R) (g : G) :
    Δ_[h] (fun y ↦ f y • g) = Δ_[h] f • fun _ ↦ g := by
  /-
    M : Type u_1
    G : Type u_2
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommGroup G
    h : M
    R : Type
    inst✝¹ : Ring R
    inst✝ : Module R G
    f : M → R
    g : G
    ⊢ Eq (fwdDiff h fun y => HSMul.hSMul (f y) g) (HSMul.hSMul (fwdDiff h f) fun x …
  -/
  ext y
  /-
    case h
    M : Type u_1
    G : Type u_2
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommGroup G
    h : M
    R : Type
    inst✝¹ : Ring R
    inst✝ : Module R G
    f : M → R
    g : G
    y : M
    ⊢ Eq (fwdDiff h (fun y => HSMul.hSMul (f y) g) y) (HSMul.hSMul (fwdDiff h f) ( …
  -/
  simp only [fwdDiff, Pi.smul_apply', sub_smul]
  /-
    🎉 no goals
  -/


variable (M G) in
/-- Linear-endomorphism version of the forward difference operator. -/
@[simps]
def fwdDiffₗ  : Module.End ℤ (M → G) where
  toFun := fwdDiff h
  map_add' := fwdDiff_add h
  map_smul' := fwdDiff_const_smul h


lemma coe_fwdDiffₗ : ↑(fwdDiffₗ M G h) = fwdDiff h := rfl


lemma coe_fwdDiffₗ_pow (n : ℕ) : ↑(fwdDiffₗ M G h ^ n) = (fwdDiff h)^[n] := by
  /-
    M : Type u_1
    G : Type u_2
    inst✝¹ : AddCommMonoid M
    inst✝ : AddCommGroup G
    h : M
    n : Nat
    ⊢ Eq (⇑(HPow.hPow (fwdDiff_aux.fwdDiffₗ M G h) n)) (Nat.iterate (fwdDiff h) n)
  -/
  ext; rw [LinearMap.pow_apply, coe_fwdDiffₗ]
       /-
         🎉 no goals
       -/


variable (M G) in
/-- Linear-endomorphism version of the shift-by-1 operator. -/
def shiftₗ : Module.End ℤ (M → G) := fwdDiffₗ M G h + 1


lemma shiftₗ_apply (f : M → G) (y : M) : shiftₗ M G h f y = f (y + h) := by
  rw [shiftₗ, LinearMap.add_apply, Pi.add_apply, LinearMap.one_apply, fwdDiffₗ_apply, fwdDiff,
    sub_add_cancel]


lemma shiftₗ_pow_apply (f : M → G) (k : ℕ) (y : M) : (shiftₗ M G h ^ k) f y = f (y + k • h) := by
  /-
    M : Type u_1
    G : Type u_2
    inst✝¹ : AddCommMonoid M
    inst✝ : AddCommGroup G
    h : M
    f : M → G
    k : Nat
    y : M
    ⊢ Eq ((HPow.hPow (fwdDiff_aux.shiftₗ M G h) k) f y) (f (HAdd.hAdd y (HSMul.hSM …
  -/
  induction' k with k IH generalizing f
    /-
      case zero
      M : Type u_1
      G : Type u_2
      inst✝¹ : AddCommMonoid M
      inst✝ : AddCommGroup G
      h y : M
      f : M → G
      ⊢ Eq ((HPow.hPow (fwdDiff_aux.shiftₗ M G h) 0) f y) (f (HAdd.hAdd y (HSMul.hSM …
    -/
  · simp only [pow_zero, LinearMap.one_apply, cast_zero, add_zero, zero_smul]
    /-
      🎉 no goals
    -/
  · simp only [pow_add, pow_one, LinearMap.mul_apply, IH (shiftₗ M G h f), shiftₗ_apply, add_assoc,
      add_nsmul, one_smul]


@[simp] lemma fwdDiff_finset_sum {α : Type*} (s : Finset α) (f : α → M → G) :
    Δ_[h] (∑ k ∈ s, f k) = ∑ k ∈ s, Δ_[h] (f k) :=
  map_sum (fwdDiffₗ M G h) f s


@[simp] lemma fwdDiff_iter_add (f g : M → G) (n : ℕ) :
    Δ_[h]^[n] (f + g) = Δ_[h]^[n] f + Δ_[h]^[n] g := by
  /-
    M : Type u_1
    G : Type u_2
    inst✝¹ : AddCommMonoid M
    inst✝ : AddCommGroup G
    h : M
    f g : M → G
    n : Nat
    ⊢ Eq (Nat.iterate (fwdDiff h) n (HAdd.hAdd f g)) (HAdd.hAdd (Nat.iterate (fwdD …
  -/
  simpa only [coe_fwdDiffₗ_pow] using map_add (fwdDiffₗ M G h ^ n) f g
  /-
    🎉 no goals
  -/


@[simp] lemma fwdDiff_iter_const_smul {R : Type*} [Semiring R] [Module R G]
    (r : R) (f : M → G) (n : ℕ) : Δ_[h]^[n] (r • f) = r • Δ_[h]^[n] f := by
  /-
    M : Type u_1
    G : Type u_2
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommGroup G
    h : M
    R : Type u_3
    inst✝¹ : Semiring R
    inst✝ : Module R G
    r : R
    f : M → G
    n : Nat
    ⊢ Eq (Nat.iterate (fwdDiff h) n (HSMul.hSMul r f)) (HSMul.hSMul r (Nat.iterate …
  -/
  induction' n with n IH generalizing f
    /-
      case zero
      M : Type u_1
      G : Type u_2
      inst✝³ : AddCommMonoid M
      inst✝² : AddCommGroup G
      h : M
      R : Type u_3
      inst✝¹ : Semiring R
      inst✝ : Module R G
      r : R
      f : M → G
      ⊢ Eq (Nat.iterate (fwdDiff h) 0 (HSMul.hSMul r f)) (HSMul.hSMul r (Nat.iterate …
    -/
  · simp only [iterate_zero, id_eq]
    /-
      🎉 no goals
    -/
    /-
      case succ
      M : Type u_1
      G : Type u_2
      inst✝³ : AddCommMonoid M
      inst✝² : AddCommGroup G
      h : M
      R : Type u_3
      inst✝¹ : Semiring R
      inst✝ : Module R G
      r : R
      n : Nat
      IH : ∀ (f : M → G), Eq (Nat.iterate (fwdDiff h) n (HSMul.hSMul r f)) (HSMul.hS …
      f : M → G
      ⊢ Eq (Nat.iterate (fwdDiff h) (HAdd.hAdd n 1) (HSMul.hSMul r f)) (HSMul.hSMul  …
    -/
  · simp only [iterate_succ_apply, fwdDiff_const_smul, IH]
    /-
      🎉 no goals
    -/


@[simp] lemma fwdDiff_iter_finset_sum {α : Type*} (s : Finset α) (f : α → M → G) (n : ℕ) :
    Δ_[h]^[n] (∑ k ∈ s, f k) = ∑ k ∈ s, Δ_[h]^[n] (f k) := by
  /-
    M : Type u_1
    G : Type u_2
    inst✝¹ : AddCommMonoid M
    inst✝ : AddCommGroup G
    h : M
    α : Type u_3
    s : Finset α
    f : α → M → G
    n : Nat
    ⊢ Eq (Nat.iterate (fwdDiff h) n (s.sum fun k => f k)) (s.sum fun k => Nat.iter …
  -/
  simpa only [coe_fwdDiffₗ_pow] using map_sum (fwdDiffₗ M G h ^ n) f s
  /-
    🎉 no goals
  -/


/--
Express the `n`-th forward difference of `f` at `y` in terms of the values `f (y + k)`, for
`0 ≤ k ≤ n`.
-/
theorem fwdDiff_iter_eq_sum_shift (f : M → G) (n : ℕ) (y : M) :
    Δ_[h]^[n] f y = ∑ k in range (n + 1), ((-1 : ℤ) ^ (n - k) * n.choose k) • f (y + k • h) := by
  -- rewrite in terms of `(shiftₗ - 1) ^ n`
  /-
    M : Type u_1
    G : Type u_2
    inst✝¹ : AddCommMonoid M
    inst✝ : AddCommGroup G
    h : M
    f : M → G
    n : Nat
    y : M
    ⊢ Eq (Nat.iterate (fwdDiff h) n f y) ((Finset.range (HAdd.hAdd n 1)).sum fun k …
  -/
  have : fwdDiffₗ M G h = shiftₗ M G h - 1 := by simp only [shiftₗ, add_sub_cancel_right]
  /-
    M : Type u_1
    G : Type u_2
    inst✝¹ : AddCommMonoid M
    inst✝ : AddCommGroup G
    h : M
    f : M → G
    n : Nat
    y : M
    this : Eq (fwdDiff_aux.fwdDiffₗ M G h) (HSub.hSub (fwdDiff_aux.shiftₗ M G h) 1)
    ⊢ Eq (Nat.iterate (fwdDiff h) n f y) ((Finset.range (HAdd.hAdd n 1)).sum fun k …
  -/
  rw [← coe_fwdDiffₗ, this, ← LinearMap.pow_apply]
  -- use binomial theorem `Commute.add_pow` to expand this
  /-
    M : Type u_1
    G : Type u_2
    inst✝¹ : AddCommMonoid M
    inst✝ : AddCommGroup G
    h : M
    f : M → G
    n : Nat
    y : M
    this : Eq (fwdDiff_aux.fwdDiffₗ M G h) (HSub.hSub (fwdDiff_aux.shiftₗ M G h) 1)
    ⊢ Eq ((HPow.hPow (HSub.hSub (fwdDiff_aux.shiftₗ M G h) 1) n) f y) ((Finset.ran …
  -/
  have : Commute (shiftₗ M G h) (-1) := (Commute.one_right _).neg_right
  /-
    M : Type u_1
    G : Type u_2
    inst✝¹ : AddCommMonoid M
    inst✝ : AddCommGroup G
    h : M
    f : M → G
    n : Nat
    y : M
    this✝ : Eq (fwdDiff_aux.fwdDiffₗ M G h) (HSub.hSub (fwdDiff_aux.shiftₗ M G h) 1)
    this : Commute (fwdDiff_aux.shiftₗ M G h) (-1)
    ⊢ Eq ((HPow.hPow (HSub.hSub (fwdDiff_aux.shiftₗ M G h) 1) n) f y) ((Finset.ran …
  -/
  convert congr_fun (LinearMap.congr_fun (this.add_pow n) f) y using 3
    /-
      case h.e'_2.h.e'_2.h.e'_5
      M : Type u_1
      G : Type u_2
      inst✝¹ : AddCommMonoid M
      inst✝ : AddCommGroup G
      h : M
      f : M → G
      n : Nat
      y : M
      this✝ : Eq (fwdDiff_aux.fwdDiffₗ M G h) (HSub.hSub (fwdDiff_aux.shiftₗ M G h) 1)
      this : Commute (fwdDiff_aux.shiftₗ M G h) (-1)
      ⊢ Eq (HSub.hSub (fwdDiff_aux.shiftₗ M G h) 1) (HAdd.hAdd (fwdDiff_aux.shiftₗ M …
    -/
  · simp only [sub_eq_add_neg]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_3
      M : Type u_1
      G : Type u_2
      inst✝¹ : AddCommMonoid M
      inst✝ : AddCommGroup G
      h : M
      f : M → G
      n : Nat
      y : M
      this✝ : Eq (fwdDiff_aux.fwdDiffₗ M G h) (HSub.hSub (fwdDiff_aux.shiftₗ M G h) 1)
      this : Commute (fwdDiff_aux.shiftₗ M G h) (-1)
      ⊢ Eq ((Finset.range (HAdd.hAdd n 1)).sum fun k => HSMul.hSMul (HMul.hMul (HPow …
    -/
  · rw [LinearMap.sum_apply, sum_apply]
    /-
      case h.e'_3
      M : Type u_1
      G : Type u_2
      inst✝¹ : AddCommMonoid M
      inst✝ : AddCommGroup G
      h : M
      f : M → G
      n : Nat
      y : M
      this✝ : Eq (fwdDiff_aux.fwdDiffₗ M G h) (HSub.hSub (fwdDiff_aux.shiftₗ M G h) 1)
      this : Commute (fwdDiff_aux.shiftₗ M G h) (-1)
      ⊢ Eq ((Finset.range (HAdd.hAdd n 1)).sum fun k => HSMul.hSMul (HMul.hMul (HPow …
    -/
    congr 1 with k
    have : ((-1) ^ (n - k) * n.choose k : Module.End ℤ (M → G))
              = ↑((-1) ^ (n - k) * n.choose k : ℤ) := by norm_cast
    rw [mul_assoc, LinearMap.mul_apply, this, Module.End.intCast_apply, LinearMap.map_smul,
      Pi.smul_apply, shiftₗ_pow_apply]


/--
**Gregory-Newton formula** expressing `f (y + n • h)` in terms of the iterated forward differences
of `f` at `y`.
-/
theorem shift_eq_sum_fwdDiff_iter (f : M → G) (n : ℕ) (y : M) :
    f (y + n • h) = ∑ k in range (n + 1), n.choose k • Δ_[h]^[k] f y := by
  convert congr_fun (LinearMap.congr_fun
      ((Commute.one_right (fwdDiffₗ M G h)).add_pow n) f) y using 1
    /-
      case h.e'_2
      M : Type u_1
      G : Type u_2
      inst✝¹ : AddCommMonoid M
      inst✝ : AddCommGroup G
      h : M
      f : M → G
      n : Nat
      y : M
      ⊢ Eq (f (HAdd.hAdd y (HSMul.hSMul n h))) ((HPow.hPow (HAdd.hAdd (fwdDiff_aux.f …
    -/
  · rw [← shiftₗ_pow_apply h f, shiftₗ]
    /-
      🎉 no goals
    -/
  · simp only [LinearMap.sum_apply, sum_apply, one_pow, mul_one, LinearMap.mul_apply,
      Module.End.natCast_apply, map_nsmul, Pi.smul_apply, LinearMap.pow_apply, coe_fwdDiffₗ]


lemma fwdDiff_choose (j : ℕ) : Δ_[1] (fun x ↦ x.choose (j + 1) : ℕ → ℤ) = fun x ↦ x.choose j := by
  /-
    j : Nat
    ⊢ Eq (fwdDiff 1 fun x => ↑(x.choose (HAdd.hAdd j 1))) fun x => ↑(x.choose j)
  -/
  ext n
  /-
    case h
    j n : Nat
    ⊢ Eq (fwdDiff 1 (fun x => ↑(x.choose (HAdd.hAdd j 1))) n) ↑(n.choose j)
  -/
  simp only [fwdDiff, choose_succ_succ' n j, cast_add, add_sub_cancel_right]
  /-
    🎉 no goals
  -/


lemma fwdDiff_iter_choose (j k : ℕ) :
    Δ_[1]^[k] (fun x ↦ x.choose (k + j) : ℕ → ℤ) = fun x ↦ x.choose j := by
  /-
    j k : Nat
    ⊢ Eq (Nat.iterate (fwdDiff 1) k fun x => ↑(x.choose (HAdd.hAdd k j))) fun x => …
  -/
  induction' k with k IH generalizing j
    /-
      case zero
      j : Nat
      ⊢ Eq (Nat.iterate (fwdDiff 1) 0 fun x => ↑(x.choose (HAdd.hAdd 0 j))) fun x => …
    -/
  · simp only [zero_add, iterate_zero, id_eq]
    /-
      🎉 no goals
    -/
    /-
      case succ
      k : Nat
      IH : ∀ (j : Nat), Eq (Nat.iterate (fwdDiff 1) k fun x => ↑(x.choose (HAdd.hAdd …
      j : Nat
      ⊢ Eq (Nat.iterate (fwdDiff 1) (HAdd.hAdd k 1) fun x => ↑(x.choose (HAdd.hAdd ( …
    -/
  · simp only [Function.iterate_succ_apply', add_assoc, add_comm 1 j, IH, fwdDiff_choose]
    /-
      🎉 no goals
    -/


lemma fwdDiff_iter_choose_zero (m n : ℕ) :
    Δ_[1]^[n] (fun x ↦ x.choose m : ℕ → ℤ) 0 = if n = m then 1 else 0 := by
  /-
    m n : Nat
    ⊢ Eq (Nat.iterate (fwdDiff 1) n (fun x => ↑(x.choose m)) 0) (ite (Eq n m) 1 0)
  -/
  rcases lt_trichotomy m n with hmn | rfl | hnm
    /-
      case inl
      m n : Nat
      hmn : LT.lt m n
      ⊢ Eq (Nat.iterate (fwdDiff 1) n (fun x => ↑(x.choose m)) 0) (ite (Eq n m) 1 0)
    -/
  · rcases Nat.exists_eq_add_of_lt hmn with ⟨k, rfl⟩
    simp_rw [hmn.ne', if_false, (by ring : m + k + 1 = k + 1 + m), iterate_add_apply,
      add_zero m ▸ fwdDiff_iter_choose 0 m, choose_zero_right, iterate_one, cast_one, fwdDiff_const,
      fwdDiff_iter_eq_sum_shift, smul_zero, sum_const_zero]
    /-
      case inr.inl
      m : Nat
      ⊢ Eq (Nat.iterate (fwdDiff 1) m (fun x => ↑(x.choose m)) 0) (ite (Eq m m) 1 0)
    -/
  · simp only [if_true, add_zero m ▸ fwdDiff_iter_choose 0 m, choose_zero_right, cast_one]
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      m n : Nat
      hnm : LT.lt n m
      ⊢ Eq (Nat.iterate (fwdDiff 1) n (fun x => ↑(x.choose m)) 0) (ite (Eq n m) 1 0)
    -/
  · rcases Nat.exists_eq_add_of_lt hnm with ⟨k, rfl⟩
    /-
      case inr.inr.intro
      n k : Nat
      hnm : LT.lt n (HAdd.hAdd (HAdd.hAdd n k) 1)
      ⊢ Eq (Nat.iterate (fwdDiff 1) n (fun x => ↑(x.choose (HAdd.hAdd (HAdd.hAdd n k …
    -/
    simp_rw [hnm.ne, if_false, add_assoc n k 1, fwdDiff_iter_choose, choose_zero_succ, cast_zero]
    /-
      🎉 no goals
    -/


