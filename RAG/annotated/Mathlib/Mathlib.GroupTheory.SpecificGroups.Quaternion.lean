/-- The (generalised) quaternion group `QuaternionGroup n` of order `4n`. It can be defined by the
presentation $\langle a, x | a^{2n} = 1, x^2 = a^n, x^{-1}ax=a^{-1}\rangle$. We write `a i` for
$a^i$ and `xa i` for $x * a^i$.
-/
inductive QuaternionGroup (n : ℕ) : Type
  | a : ZMod (2 * n) → QuaternionGroup n
  | xa : ZMod (2 * n) → QuaternionGroup n
  deriving DecidableEq


/-- Multiplication of the dihedral group.
-/
private def mul : QuaternionGroup n → QuaternionGroup n → QuaternionGroup n
  | a i, a j => a (i + j)
  | a i, xa j => xa (j - i)
  | xa i, a j => xa (i + j)
  | xa i, xa j => a (n + j - i)


/-- The identity `1` is given by `aⁱ`.
-/
private def one : QuaternionGroup n :=
  a 0


instance : Inhabited (QuaternionGroup n) :=
  ⟨one⟩


/-- The inverse of an element of the quaternion group.
-/
private def inv : QuaternionGroup n → QuaternionGroup n
  | a i => a (-i)
  | xa i => xa (n + i)


/-- The group structure on `QuaternionGroup n`.
-/
instance : Group (QuaternionGroup n) where
  mul := mul
  mul_assoc := by
    /-
      n : Nat
      ⊢ ∀ (a b c : QuaternionGroup n), Eq (HMul.hMul (HMul.hMul a b) c) (HMul.hMul a …
    -/
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
    rintro (i | i) (j | j) (k | k) <;> simp only [(· * ·), mul] <;> ring_nf
    /-
      case xa.xa.xa
      n : Nat
      i j k : ZMod (HMul.hMul 2 n)
      ⊢ Eq (QuaternionGroup.xa (HAdd.hAdd (HAdd.hAdd k (HSub.hSub (Neg.neg ↑n) j)) i …
    -/
    congr
    calc
      -(n : ZMod (2 * n)) = 0 - n := by rw [zero_sub]
      _ = 2 * n - n := by norm_cast; simp
      _ = n := by ring
  one := one
  one_mul := by
    /-
      n : Nat
      ⊢ ∀ (a : QuaternionGroup n), Eq (HMul.hMul 1 a) a
    -/
    rintro (i | i)
      /-
        case a
        n : Nat
        i : ZMod (HMul.hMul 2 n)
        ⊢ Eq (HMul.hMul 1 (QuaternionGroup.a i)) (QuaternionGroup.a i)
      -/
    · exact congr_arg a (zero_add i)
      /-
        🎉 no goals
      -/
      /-
        case xa
        n : Nat
        i : ZMod (HMul.hMul 2 n)
        ⊢ Eq (HMul.hMul 1 (QuaternionGroup.xa i)) (QuaternionGroup.xa i)
      -/
    · exact congr_arg xa (sub_zero i)
      /-
        🎉 no goals
      -/
  mul_one := by
    /-
      n : Nat
      ⊢ ∀ (a : QuaternionGroup n), Eq (HMul.hMul a 1) a
    -/
    rintro (i | i)
      /-
        case a
        n : Nat
        i : ZMod (HMul.hMul 2 n)
        ⊢ Eq (HMul.hMul (QuaternionGroup.a i) 1) (QuaternionGroup.a i)
      -/
    · exact congr_arg a (add_zero i)
      /-
        🎉 no goals
      -/
      /-
        case xa
        n : Nat
        i : ZMod (HMul.hMul 2 n)
        ⊢ Eq (HMul.hMul (QuaternionGroup.xa i) 1) (QuaternionGroup.xa i)
      -/
    · exact congr_arg xa (add_zero i)
      /-
        🎉 no goals
      -/
  inv := inv
  inv_mul_cancel := by
    /-
      n : Nat
      ⊢ ∀ (a : QuaternionGroup n), Eq (HMul.hMul (Inv.inv a) a) 1
    -/
    rintro (i | i)
      /-
        case a
        n : Nat
        i : ZMod (HMul.hMul 2 n)
        ⊢ Eq (HMul.hMul (Inv.inv (QuaternionGroup.a i)) (QuaternionGroup.a i)) 1
      -/
    · exact congr_arg a (neg_add_cancel i)
      /-
        🎉 no goals
      -/
      /-
        case xa
        n : Nat
        i : ZMod (HMul.hMul 2 n)
        ⊢ Eq (HMul.hMul (Inv.inv (QuaternionGroup.xa i)) (QuaternionGroup.xa i)) 1
      -/
    · exact congr_arg a (sub_self (n + i))
      /-
        🎉 no goals
      -/


@[simp]
theorem a_mul_a (i j : ZMod (2 * n)) : a i * a j = a (i + j) :=
  rfl


@[simp]
theorem a_mul_xa (i j : ZMod (2 * n)) : a i * xa j = xa (j - i) :=
  rfl


@[simp]
theorem xa_mul_a (i j : ZMod (2 * n)) : xa i * a j = xa (i + j) :=
  rfl


@[simp]
theorem xa_mul_xa (i j : ZMod (2 * n)) : xa i * xa j = a ((n : ZMod (2 * n)) + j - i) :=
  rfl


theorem one_def : (1 : QuaternionGroup n) = a 0 :=
  rfl


private def fintypeHelper : ZMod (2 * n) ⊕ ZMod (2 * n) ≃ QuaternionGroup n where
  invFun i :=
    match i with
    | a j => Sum.inl j
    | xa j => Sum.inr j
  toFun i :=
    match i with
    | Sum.inl j => a j
    | Sum.inr j => xa j
                 /-
                   n : Nat
                   ⊢ Function.LeftInverse (fun i => QuaternionGroup.inv.match_1 (fun i => Sum (ZM …
                 -/
                                    /-
                                      🎉 no goals
                                    -/
  left_inv := by rintro (x | x) <;> rfl
                                    /-
                                      🎉 no goals
                                    -/
                  /-
                    n : Nat
                    ⊢ Function.RightInverse (fun i => QuaternionGroup.inv.match_1 (fun i => Sum (Z …
                  -/
                                     /-
                                       🎉 no goals
                                     -/
  right_inv := by rintro (x | x) <;> rfl
                                     /-
                                       🎉 no goals
                                     -/


/-- The special case that more or less by definition `QuaternionGroup 0` is isomorphic to the
infinite dihedral group. -/
def quaternionGroupZeroEquivDihedralGroupZero : QuaternionGroup 0 ≃* DihedralGroup 0 where
  toFun i :=
    -- Porting note: Originally `QuaternionGroup.recOn i DihedralGroup.r DihedralGroup.sr`
    match i with
    | a j => DihedralGroup.r j
    | xa j => DihedralGroup.sr j
  invFun i :=
    match i with
    | DihedralGroup.r j => a j
    | DihedralGroup.sr j => xa j
                 /-
                   n : Nat
                   ⊢ Function.LeftInverse (fun i => QuaternionGroup.quaternionGroupZeroEquivDihed …
                 -/
                                    /-
                                      🎉 no goals
                                    -/
  left_inv := by rintro (k | k) <;> rfl
                                    /-
                                      🎉 no goals
                                    -/
                  /-
                    n : Nat
                    ⊢ Function.RightInverse (fun i => QuaternionGroup.quaternionGroupZeroEquivDihe …
                  -/
                                     /-
                                       🎉 no goals
                                     -/
  right_inv := by rintro (k | k) <;> rfl
                                     /-
                                       🎉 no goals
                                     -/
                 /-
                   n : Nat
                   ⊢ ∀ (x y : QuaternionGroup 0), Eq ({ toFun := fun i => QuaternionGroup.quatern …
                 -/
                                            /-
                                              🎉 no goals
                                            -/
                                            /-
                                              🎉 no goals
                                            -/
                                            /-
                                              🎉 no goals
                                            -/
  map_mul' := by rintro (k | k) (l | l) <;> simp
                                            /-
                                              🎉 no goals
                                            -/


/-- If `0 < n`, then `QuaternionGroup n` is a finite group.
-/
instance [NeZero n] : Fintype (QuaternionGroup n) :=
  Fintype.ofEquiv _ fintypeHelper


instance : Nontrivial (QuaternionGroup n) :=
                  /-
                    n : Nat
                    ⊢ Ne (QuaternionGroup.a 0) (QuaternionGroup.xa 0)
                  -/
  ⟨⟨a 0, xa 0, by revert n; simp⟩⟩ -- Porting note: `revert n; simp` was `decide`
                            /-
                              🎉 no goals
                            -/


/-- If `0 < n`, then `QuaternionGroup n` has `4n` elements.
-/
theorem card [NeZero n] : Fintype.card (QuaternionGroup n) = 4 * n := by
  /-
    n : Nat
    inst✝ : NeZero n
    ⊢ Eq (Fintype.card (QuaternionGroup n)) (HMul.hMul 4 n)
  -/
  rw [← Fintype.card_eq.mpr ⟨fintypeHelper⟩, Fintype.card_sum, ZMod.card, two_mul]
  /-
    n : Nat
    inst✝ : NeZero n
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd n n) (HAdd.hAdd n n)) (HMul.hMul 4 n)
  -/
  ring
  /-
    🎉 no goals
  -/


@[simp]
theorem a_one_pow (k : ℕ) : (a 1 : QuaternionGroup n) ^ k = a k := by
  /-
    n k : Nat
    ⊢ Eq (HPow.hPow (QuaternionGroup.a 1) k) (QuaternionGroup.a ↑k)
  -/
  induction' k with k IH
    /-
      case zero
      n : Nat
      ⊢ Eq (HPow.hPow (QuaternionGroup.a 1) 0) (QuaternionGroup.a ↑0)
    -/
  · rw [Nat.cast_zero]; rfl
                        /-
                          🎉 no goals
                        -/
    /-
      case succ
      n k : Nat
      IH : Eq (HPow.hPow (QuaternionGroup.a 1) k) (QuaternionGroup.a ↑k)
      ⊢ Eq (HPow.hPow (QuaternionGroup.a 1) (HAdd.hAdd k 1)) (QuaternionGroup.a ↑(HA …
    -/
  · rw [pow_succ, IH, a_mul_a]
    /-
      case succ
      n k : Nat
      IH : Eq (HPow.hPow (QuaternionGroup.a 1) k) (QuaternionGroup.a ↑k)
      ⊢ Eq (QuaternionGroup.a (HAdd.hAdd (↑k) 1)) (QuaternionGroup.a ↑(HAdd.hAdd k 1))
    -/
    congr 1
    /-
      case succ.e_a
      n k : Nat
      IH : Eq (HPow.hPow (QuaternionGroup.a 1) k) (QuaternionGroup.a ↑k)
      ⊢ Eq (HAdd.hAdd (↑k) 1) ↑(HAdd.hAdd k 1)
    -/
    norm_cast
    /-
      🎉 no goals
    -/

-- @[simp] -- Porting note: simp changes this to `a 0 = 1`, so this is no longer a good simp lemma.

theorem a_one_pow_n : (a 1 : QuaternionGroup n) ^ (2 * n) = 1 := by
  /-
    n : Nat
    ⊢ Eq (HPow.hPow (QuaternionGroup.a 1) (HMul.hMul 2 n)) 1
  -/
  rw [a_one_pow, one_def]
  /-
    n : Nat
    ⊢ Eq (QuaternionGroup.a ↑(HMul.hMul 2 n)) (QuaternionGroup.a 0)
  -/
  congr 1
  /-
    case e_a
    n : Nat
    ⊢ Eq (↑(HMul.hMul 2 n)) 0
  -/
  exact ZMod.natCast_self _
  /-
    🎉 no goals
  -/


@[simp]
                                                        /-
                                                          n : Nat
                                                          i : ZMod (HMul.hMul 2 n)
                                                          ⊢ Eq (HPow.hPow (QuaternionGroup.xa i) 2) (QuaternionGroup.a ↑n)
                                                        -/
theorem xa_sq (i : ZMod (2 * n)) : xa i ^ 2 = a n := by simp [sq]
                                                        /-
                                                          🎉 no goals
                                                        -/


@[simp]
theorem xa_pow_four (i : ZMod (2 * n)) : xa i ^ 4 = 1 := by
  rw [pow_succ, pow_succ, sq, xa_mul_xa, a_mul_xa, xa_mul_xa,
    add_sub_cancel_right, add_sub_assoc, sub_sub_cancel]
  /-
    n : Nat
    i : ZMod (HMul.hMul 2 n)
    ⊢ Eq (QuaternionGroup.a (HAdd.hAdd ↑n ↑n)) 1
  -/
  norm_cast
  /-
    n : Nat
    i : ZMod (HMul.hMul 2 n)
    ⊢ Eq (QuaternionGroup.a ↑(HAdd.hAdd n n)) 1
  -/
  rw [← two_mul]
  /-
    n : Nat
    i : ZMod (HMul.hMul 2 n)
    ⊢ Eq (QuaternionGroup.a ↑(HMul.hMul 2 n)) 1
  -/
  simp [one_def]
  /-
    🎉 no goals
  -/


/-- If `0 < n`, then `xa i` has order 4.
-/
@[simp]
theorem orderOf_xa [NeZero n] (i : ZMod (2 * n)) : orderOf (xa i) = 4 := by
  /-
    n : Nat
    inst✝ : NeZero n
    i : ZMod (HMul.hMul 2 n)
    ⊢ Eq (orderOf (QuaternionGroup.xa i)) 4
  -/
  change _ = 2 ^ 2
  /-
    n : Nat
    inst✝ : NeZero n
    i : ZMod (HMul.hMul 2 n)
    ⊢ Eq (orderOf (QuaternionGroup.xa i)) (HPow.hPow 2 2)
  -/
  haveI : Fact (Nat.Prime 2) := Fact.mk Nat.prime_two
  /-
    n : Nat
    inst✝ : NeZero n
    i : ZMod (HMul.hMul 2 n)
    this : Fact (Nat.Prime 2)
    ⊢ Eq (orderOf (QuaternionGroup.xa i)) (HPow.hPow 2 2)
  -/
  apply orderOf_eq_prime_pow
    /-
      case hnot
      n : Nat
      inst✝ : NeZero n
      i : ZMod (HMul.hMul 2 n)
      this : Fact (Nat.Prime 2)
      ⊢ Not (Eq (HPow.hPow (QuaternionGroup.xa i) (HPow.hPow 2 1)) 1)
    -/
  · intro h
    /-
      case hnot
      n : Nat
      inst✝ : NeZero n
      i : ZMod (HMul.hMul 2 n)
      this : Fact (Nat.Prime 2)
      h : Eq (HPow.hPow (QuaternionGroup.xa i) (HPow.hPow 2 1)) 1
      ⊢ False
    -/
    simp only [pow_one, xa_sq] at h
    /-
      case hnot
      n : Nat
      inst✝ : NeZero n
      i : ZMod (HMul.hMul 2 n)
      this : Fact (Nat.Prime 2)
      h : Eq (QuaternionGroup.a ↑n) 1
      ⊢ False
    -/
    injection h with h'
    /-
      case hnot
      n : Nat
      inst✝ : NeZero n
      i : ZMod (HMul.hMul 2 n)
      this : Fact (Nat.Prime 2)
      h' : Eq (↑n) 0
      ⊢ False
    -/
    apply_fun ZMod.val at h'
    /-
      case hnot
      n : Nat
      inst✝ : NeZero n
      i : ZMod (HMul.hMul 2 n)
      this : Fact (Nat.Prime 2)
      h' : Eq (↑n).val (ZMod.val 0)
      ⊢ False
    -/
    apply_fun (· / n) at h'
    simp only [ZMod.val_natCast, ZMod.val_zero, Nat.zero_div, Nat.mod_mul_left_div_self,
      Nat.div_self (NeZero.pos n), reduceCtorEq] at h'
    /-
      case hfin
      n : Nat
      inst✝ : NeZero n
      i : ZMod (HMul.hMul 2 n)
      this : Fact (Nat.Prime 2)
      ⊢ Eq (HPow.hPow (QuaternionGroup.xa i) (HPow.hPow 2 (HAdd.hAdd 1 1))) 1
    -/
  · norm_num
    /-
      🎉 no goals
    -/


/-- In the special case `n = 1`, `Quaternion 1` is a cyclic group (of order `4`). -/
theorem quaternionGroup_one_isCyclic : IsCyclic (QuaternionGroup 1) := by
  /-
    ⊢ IsCyclic (QuaternionGroup 1)
  -/
  apply isCyclic_of_orderOf_eq_card
    /-
      case hx
      ⊢ Eq (orderOf ?x) (Nat.card (QuaternionGroup 1))
    -/
  · rw [Nat.card_eq_fintype_card, card, mul_one]
    /-
      case hx
      ⊢ Eq (orderOf ?x) 4
    -/
    exact orderOf_xa 0
    /-
      🎉 no goals
    -/


/-- If `0 < n`, then `a 1` has order `2 * n`.
-/
@[simp]
theorem orderOf_a_one : orderOf (a 1 : QuaternionGroup n) = 2 * n := by
  /-
    n : Nat
    ⊢ Eq (orderOf (QuaternionGroup.a 1)) (HMul.hMul 2 n)
  -/
  cases' eq_zero_or_neZero n with hn hn
    /-
      case inl
      n : Nat
      hn : Eq n 0
      ⊢ Eq (orderOf (QuaternionGroup.a 1)) (HMul.hMul 2 n)
    -/
  · subst hn
    /-
      case inl
      ⊢ Eq (orderOf (QuaternionGroup.a 1)) (HMul.hMul 2 0)
    -/
    simp_rw [mul_zero, orderOf_eq_zero_iff']
    /-
      case inl
      ⊢ ∀ (n : Nat), LT.lt 0 n → Ne (HPow.hPow (QuaternionGroup.a 1) n) 1
    -/
    intro n h
    /-
      case inl
      n : Nat
      h : LT.lt 0 n
      ⊢ Ne (HPow.hPow (QuaternionGroup.a 1) n) 1
    -/
    rw [one_def, a_one_pow]
    /-
      case inl
      n : Nat
      h : LT.lt 0 n
      ⊢ Ne (QuaternionGroup.a ↑n) (QuaternionGroup.a 0)
    -/
    apply mt a.inj
    /-
      case inl
      n : Nat
      h : LT.lt 0 n
      ⊢ Not (Eq (↑n) 0)
    -/
    haveI : CharZero (ZMod (2 * 0)) := ZMod.charZero
    /-
      case inl
      n : Nat
      h : LT.lt 0 n
      this : CharZero (ZMod (HMul.hMul 2 0))
      ⊢ Not (Eq (↑n) 0)
    -/
    simpa using h.ne'
    /-
      🎉 no goals
    -/
  apply (Nat.le_of_dvd
    (NeZero.pos _) (orderOf_dvd_of_pow_eq_one (@a_one_pow_n n))).lt_or_eq.resolve_left
  /-
    case inr
    n : Nat
    hn : NeZero n
    ⊢ Not (LT.lt (orderOf (QuaternionGroup.a 1)) (HMul.hMul 2 n))
  -/
  intro h
  /-
    case inr
    n : Nat
    hn : NeZero n
    h : LT.lt (orderOf (QuaternionGroup.a 1)) (HMul.hMul 2 n)
    ⊢ False
  -/
  have h1 : (a 1 : QuaternionGroup n) ^ orderOf (a 1) = 1 := pow_orderOf_eq_one _
  /-
    case inr
    n : Nat
    hn : NeZero n
    h : LT.lt (orderOf (QuaternionGroup.a 1)) (HMul.hMul 2 n)
    h1 : Eq (HPow.hPow (QuaternionGroup.a 1) (orderOf (QuaternionGroup.a 1))) 1
    ⊢ False
  -/
  rw [a_one_pow] at h1
  /-
    case inr
    n : Nat
    hn : NeZero n
    h : LT.lt (orderOf (QuaternionGroup.a 1)) (HMul.hMul 2 n)
    h1 : Eq (QuaternionGroup.a ↑(orderOf (QuaternionGroup.a 1))) 1
    ⊢ False
  -/
  injection h1 with h2
  /-
    case inr
    n : Nat
    hn : NeZero n
    h : LT.lt (orderOf (QuaternionGroup.a 1)) (HMul.hMul 2 n)
    h2 : Eq (↑(orderOf (QuaternionGroup.a 1))) 0
    ⊢ False
  -/
  rw [← ZMod.val_eq_zero, ZMod.val_natCast, Nat.mod_eq_of_lt h] at h2
  /-
    case inr
    n : Nat
    hn : NeZero n
    h : LT.lt (orderOf (QuaternionGroup.a 1)) (HMul.hMul 2 n)
    h2 : Eq (orderOf (QuaternionGroup.a 1)) 0
    ⊢ False
  -/
  exact absurd h2.symm (orderOf_pos _).ne
  /-
    🎉 no goals
  -/


/-- If `0 < n`, then `a i` has order `(2 * n) / gcd (2 * n) i`.
-/
theorem orderOf_a [NeZero n] (i : ZMod (2 * n)) :
    orderOf (a i) = 2 * n / Nat.gcd (2 * n) i.val := by
  /-
    n : Nat
    inst✝ : NeZero n
    i : ZMod (HMul.hMul 2 n)
    ⊢ Eq (orderOf (QuaternionGroup.a i)) (HDiv.hDiv (HMul.hMul 2 n) ((HMul.hMul 2  …
  -/
  conv_lhs => rw [← ZMod.natCast_zmod_val i]
  /-
    n : Nat
    inst✝ : NeZero n
    i : ZMod (HMul.hMul 2 n)
    ⊢ Eq (orderOf (QuaternionGroup.a ↑i.val)) (HDiv.hDiv (HMul.hMul 2 n) ((HMul.hM …
  -/
  rw [← a_one_pow, orderOf_pow, orderOf_a_one]
  /-
    🎉 no goals
  -/


theorem exponent : Monoid.exponent (QuaternionGroup n) = 2 * lcm n 2 := by
  /-
    n : Nat
    ⊢ Eq (Monoid.exponent (QuaternionGroup n)) (HMul.hMul 2 (GCDMonoid.lcm n 2))
  -/
  rw [← normalize_eq 2, ← lcm_mul_left, normalize_eq]
  /-
    n : Nat
    ⊢ Eq (Monoid.exponent (QuaternionGroup n)) (GCDMonoid.lcm (HMul.hMul 2 n) (HMu …
  -/
  norm_num
  /-
    n : Nat
    ⊢ Eq (Monoid.exponent (QuaternionGroup n)) (GCDMonoid.lcm (HMul.hMul 2 n) 4)
  -/
  cases' eq_zero_or_neZero n with hn hn
    /-
      case inl
      n : Nat
      hn : Eq n 0
      ⊢ Eq (Monoid.exponent (QuaternionGroup n)) (GCDMonoid.lcm (HMul.hMul 2 n) 4)
    -/
  · subst hn
    /-
      case inl
      ⊢ Eq (Monoid.exponent (QuaternionGroup 0)) (GCDMonoid.lcm (HMul.hMul 2 0) 4)
    -/
    simp only [lcm_zero_left, mul_zero]
    /-
      case inl
      ⊢ Eq (Monoid.exponent (QuaternionGroup 0)) 0
    -/
    exact Monoid.exponent_eq_zero_of_order_zero orderOf_a_one
    /-
      🎉 no goals
    -/
  /-
    case inr
    n : Nat
    hn : NeZero n
    ⊢ Eq (Monoid.exponent (QuaternionGroup n)) (GCDMonoid.lcm (HMul.hMul 2 n) 4)
  -/
  apply Nat.dvd_antisymm
    /-
      case inr.a
      n : Nat
      hn : NeZero n
      ⊢ Dvd.dvd (Monoid.exponent (QuaternionGroup n)) (GCDMonoid.lcm (HMul.hMul 2 n) …
    -/
  · apply Monoid.exponent_dvd_of_forall_pow_eq_one
    /-
      case inr.a.a
      n : Nat
      hn : NeZero n
      ⊢ ∀ (g : QuaternionGroup n), Eq (HPow.hPow g (GCDMonoid.lcm (HMul.hMul 2 n) 4) …
    -/
    rintro (m | m)
      /-
        case inr.a.a.a
        n : Nat
        hn : NeZero n
        m : ZMod (HMul.hMul 2 n)
        ⊢ Eq (HPow.hPow (QuaternionGroup.a m) (GCDMonoid.lcm (HMul.hMul 2 n) 4)) 1
      -/
    · rw [← orderOf_dvd_iff_pow_eq_one, orderOf_a]
      /-
        case inr.a.a.a
        n : Nat
        hn : NeZero n
        m : ZMod (HMul.hMul 2 n)
        ⊢ Dvd.dvd (HDiv.hDiv (HMul.hMul 2 n) ((HMul.hMul 2 n).gcd m.val)) (GCDMonoid.l …
      -/
      refine Nat.dvd_trans ⟨gcd (2 * n) m.val, ?_⟩ (dvd_lcm_left (2 * n) 4)
      /-
        case inr.a.a.a
        n : Nat
        hn : NeZero n
        m : ZMod (HMul.hMul 2 n)
        ⊢ Eq (HMul.hMul 2 n) (HMul.hMul (HDiv.hDiv (HMul.hMul 2 n) ((HMul.hMul 2 n).gc …
      -/
      exact (Nat.div_mul_cancel (Nat.gcd_dvd_left (2 * n) m.val)).symm
      /-
        🎉 no goals
      -/
      /-
        case inr.a.a.xa
        n : Nat
        hn : NeZero n
        m : ZMod (HMul.hMul 2 n)
        ⊢ Eq (HPow.hPow (QuaternionGroup.xa m) (GCDMonoid.lcm (HMul.hMul 2 n) 4)) 1
      -/
    · rw [← orderOf_dvd_iff_pow_eq_one, orderOf_xa]
      /-
        case inr.a.a.xa
        n : Nat
        hn : NeZero n
        m : ZMod (HMul.hMul 2 n)
        ⊢ Dvd.dvd 4 (GCDMonoid.lcm (HMul.hMul 2 n) 4)
      -/
      exact dvd_lcm_right (2 * n) 4
      /-
        🎉 no goals
      -/
    /-
      case inr.a
      n : Nat
      hn : NeZero n
      ⊢ Dvd.dvd (GCDMonoid.lcm (HMul.hMul 2 n) 4) (Monoid.exponent (QuaternionGroup  …
    -/
  · apply lcm_dvd
      /-
        case inr.a.hab
        n : Nat
        hn : NeZero n
        ⊢ Dvd.dvd (HMul.hMul 2 n) (Monoid.exponent (QuaternionGroup n))
      -/
    · convert Monoid.order_dvd_exponent (a 1)
      /-
        case h.e'_3
        n : Nat
        hn : NeZero n
        ⊢ Eq (HMul.hMul 2 n) (orderOf (QuaternionGroup.a 1))
      -/
      exact orderOf_a_one.symm
      /-
        🎉 no goals
      -/
      /-
        case inr.a.hcb
        n : Nat
        hn : NeZero n
        ⊢ Dvd.dvd 4 (Monoid.exponent (QuaternionGroup n))
      -/
    · convert Monoid.order_dvd_exponent (xa (0 : ZMod (2 * n)))
      /-
        case h.e'_3
        n : Nat
        hn : NeZero n
        ⊢ Eq 4 (orderOf (QuaternionGroup.xa 0))
      -/
      exact (orderOf_xa 0).symm
      /-
        🎉 no goals
      -/


