/-- For non-zero `n : ℕ`, the ring `Fin n` is equivalent to `ZMod n`. -/
def finEquiv : ∀ (n : ℕ) [NeZero n], Fin n ≃+* ZMod n
  | 0, h => (h.ne _ rfl).elim
  | _ + 1, _ => .refl _


instance charZero : CharZero (ZMod 0) := inferInstanceAs (CharZero ℤ)


/-- `val a` is a natural number defined as:
  - for `a : ZMod 0` it is the absolute value of `a`
  - for `a : ZMod n` with `0 < n` it is the least natural number in the equivalence class

See `ZMod.valMinAbs` for a variant that takes values in the integers.
-/
def val : ∀ {n : ℕ}, ZMod n → ℕ
  | 0 => Int.natAbs
  | n + 1 => ((↑) : Fin (n + 1) → ℕ)


theorem val_lt {n : ℕ} [NeZero n] (a : ZMod n) : a.val < n := by
  /-
    n : Nat
    inst✝ : NeZero n
    a : ZMod n
    ⊢ LT.lt a.val n
  -/
  cases n
    /-
      case zero
      inst✝ : NeZero 0
      a : ZMod 0
      ⊢ LT.lt a.val 0
    -/
  · cases NeZero.ne 0 rfl
    /-
      🎉 no goals
    -/
  /-
    case succ
    n✝ : Nat
    inst✝ : NeZero (HAdd.hAdd n✝ 1)
    a : ZMod (HAdd.hAdd n✝ 1)
    ⊢ LT.lt a.val (HAdd.hAdd n✝ 1)
  -/
  exact Fin.is_lt a
  /-
    🎉 no goals
  -/


theorem val_le {n : ℕ} [NeZero n] (a : ZMod n) : a.val ≤ n :=
  a.val_lt.le


@[simp]
theorem val_zero : ∀ {n}, (0 : ZMod n).val = 0
  | 0 => rfl
  | _ + 1 => rfl


@[simp]
theorem val_one' : (1 : ZMod 0).val = 1 :=
  rfl


@[simp]
theorem val_neg' {n : ZMod 0} : (-n).val = n.val :=
  Int.natAbs_neg n


@[simp]
theorem val_mul' {m n : ZMod 0} : (m * n).val = m.val * n.val :=
  Int.natAbs_mul m n


@[simp]
theorem val_natCast {n : ℕ} (a : ℕ) : (a : ZMod n).val = a % n := by
  /-
    n a : Nat
    ⊢ Eq (↑a).val (HMod.hMod a n)
  -/
  cases n
    /-
      case zero
      a : Nat
      ⊢ Eq (↑a).val (HMod.hMod a 0)
    -/
  · rw [Nat.mod_zero]
    /-
      case zero
      a : Nat
      ⊢ Eq (↑a).val a
    -/
    exact Int.natAbs_ofNat a
    /-
      🎉 no goals
    -/
    /-
      case succ
      a n✝ : Nat
      ⊢ Eq (↑a).val (HMod.hMod a (HAdd.hAdd n✝ 1))
    -/
  · apply Fin.val_natCast
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-04-17")]
alias val_nat_cast := val_natCast


theorem val_unit' {n : ZMod 0} : IsUnit n ↔ n.val = 1 := by
  /-
    n : ZMod 0
    ⊢ Iff (IsUnit n) (Eq n.val 1)
  -/
  simp only [val]
  /-
    n : ZMod 0
    ⊢ Iff (IsUnit n) (Eq (Int.natAbs n) 1)
  -/
  rw [Int.isUnit_iff, Int.natAbs_eq_iff, Nat.cast_one]
  /-
    🎉 no goals
  -/


lemma eq_one_of_isUnit_natCast {n : ℕ} (h : IsUnit (n : ZMod 0)) : n = 1 := by
  /-
    n : Nat
    h : IsUnit ↑n
    ⊢ Eq n 1
  -/
  rw [← Nat.mod_zero n, ← val_natCast, val_unit'.mp h]
  /-
    🎉 no goals
  -/


theorem val_natCast_of_lt {n a : ℕ} (h : a < n) : (a : ZMod n).val = a := by
  /-
    n a : Nat
    h : LT.lt a n
    ⊢ Eq (↑a).val a
  -/
  rwa [val_natCast, Nat.mod_eq_of_lt]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias val_nat_cast_of_lt := val_natCast_of_lt


instance charP (n : ℕ) : CharP (ZMod n) n where
  cast_eq_zero_iff' := by
    /-
      n : Nat
      ⊢ ∀ (x : Nat), Iff (Eq (↑x) 0) (Dvd.dvd n x)
    -/
    intro k
    /-
      n k : Nat
      ⊢ Iff (Eq (↑k) 0) (Dvd.dvd n k)
    -/
    cases' n with n
      /-
        case zero
        k : Nat
        ⊢ Iff (Eq (↑k) 0) (Dvd.dvd 0 k)
      -/
    · simp [zero_dvd_iff, Int.natCast_eq_zero]
      /-
        🎉 no goals
      -/
      /-
        case succ
        k n : Nat
        ⊢ Iff (Eq (↑k) 0) (Dvd.dvd (HAdd.hAdd n 1) k)
      -/
    · exact Fin.natCast_eq_zero
      /-
        🎉 no goals
      -/


@[simp]
theorem addOrderOf_one (n : ℕ) : addOrderOf (1 : ZMod n) = n :=
  CharP.eq _ (CharP.addOrderOf_one _) (ZMod.charP n)


/-- This lemma works in the case in which `ZMod n` is not infinite, i.e. `n ≠ 0`.  The version
where `a ≠ 0` is `addOrderOf_coe'`. -/
@[simp]
theorem addOrderOf_coe (a : ℕ) {n : ℕ} (n0 : n ≠ 0) : addOrderOf (a : ZMod n) = n / n.gcd a := by
  /-
    a n : Nat
    n0 : Ne n 0
    ⊢ Eq (addOrderOf ↑a) (HDiv.hDiv n (n.gcd a))
  -/
  cases' a with a
  · simp only [Nat.cast_zero, addOrderOf_zero, Nat.gcd_zero_right,
      Nat.pos_of_ne_zero n0, Nat.div_self]
  /-
    case succ
    n : Nat
    n0 : Ne n 0
    a : Nat
    ⊢ Eq (addOrderOf ↑(HAdd.hAdd a 1)) (HDiv.hDiv n (n.gcd (HAdd.hAdd a 1)))
  -/
  rw [← Nat.smul_one_eq_cast, addOrderOf_nsmul' _ a.succ_ne_zero, ZMod.addOrderOf_one]
  /-
    🎉 no goals
  -/


/-- This lemma works in the case in which `a ≠ 0`.  The version where
 `ZMod n` is not infinite, i.e. `n ≠ 0`, is `addOrderOf_coe`. -/
@[simp]
theorem addOrderOf_coe' {a : ℕ} (n : ℕ) (a0 : a ≠ 0) : addOrderOf (a : ZMod n) = n / n.gcd a := by
  /-
    a n : Nat
    a0 : Ne a 0
    ⊢ Eq (addOrderOf ↑a) (HDiv.hDiv n (n.gcd a))
  -/
  rw [← Nat.smul_one_eq_cast, addOrderOf_nsmul' _ a0, ZMod.addOrderOf_one]
  /-
    🎉 no goals
  -/


/-- We have that `ringChar (ZMod n) = n`. -/
theorem ringChar_zmod_n (n : ℕ) : ringChar (ZMod n) = n := by
  /-
    n : Nat
    ⊢ Eq (ringChar (ZMod n)) n
  -/
  rw [ringChar.eq_iff]
  /-
    n : Nat
    ⊢ CharP (ZMod n) n
  -/
  exact ZMod.charP n
  /-
    🎉 no goals
  -/


theorem natCast_self (n : ℕ) : (n : ZMod n) = 0 :=
  CharP.cast_eq_zero (ZMod n) n


@[deprecated (since := "2024-04-17")]
alias nat_cast_self := natCast_self


@[simp]
theorem natCast_self' (n : ℕ) : (n + 1 : ZMod (n + 1)) = 0 := by
  /-
    n : Nat
    ⊢ Eq (HAdd.hAdd (↑n) 1) 0
  -/
  rw [← Nat.cast_add_one, natCast_self (n + 1)]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias nat_cast_self' := natCast_self'


/-- Cast an integer modulo `n` to another semiring.
This function is a morphism if the characteristic of `R` divides `n`.
See `ZMod.castHom` for a bundled version. -/
def cast : ∀ {n : ℕ}, ZMod n → R
  | 0 => Int.cast
  | _ + 1 => fun i => i.val



@[simp]
theorem cast_zero : (cast (0 : ZMod n) : R) = 0 := by
  /-
    n : Nat
    R : Type u_1
    inst✝ : AddGroupWithOne R
    ⊢ Eq (ZMod.cast 0) 0
  -/
  delta ZMod.cast
  /-
    n : Nat
    R : Type u_1
    inst✝ : AddGroupWithOne R
    ⊢ Eq (ZMod.val.match_1 (fun x => ZMod x → R) n (fun _ => Int.cast) (fun n i => …
  -/
  cases n
    /-
      case zero
      R : Type u_1
      inst✝ : AddGroupWithOne R
      ⊢ Eq (ZMod.val.match_1 (fun x => ZMod x → R) 0 (fun _ => Int.cast) (fun n i => …
    -/
  · exact Int.cast_zero
    /-
      🎉 no goals
    -/
    /-
      case succ
      R : Type u_1
      inst✝ : AddGroupWithOne R
      n✝ : Nat
      ⊢ Eq (ZMod.val.match_1 (fun x => ZMod x → R) (HAdd.hAdd n✝ 1) (fun _ => Int.ca …
    -/
  · simp
    /-
      🎉 no goals
    -/


theorem cast_eq_val [NeZero n] (a : ZMod n) : (cast a : R) = a.val := by
  /-
    n : Nat
    R : Type u_1
    inst✝¹ : AddGroupWithOne R
    inst✝ : NeZero n
    a : ZMod n
    ⊢ Eq a.cast ↑a.val
  -/
  cases n
    /-
      case zero
      R : Type u_1
      inst✝¹ : AddGroupWithOne R
      inst✝ : NeZero 0
      a : ZMod 0
      ⊢ Eq a.cast ↑a.val
    -/
  · cases NeZero.ne 0 rfl
    /-
      🎉 no goals
    -/
  /-
    case succ
    R : Type u_1
    inst✝¹ : AddGroupWithOne R
    n✝ : Nat
    inst✝ : NeZero (HAdd.hAdd n✝ 1)
    a : ZMod (HAdd.hAdd n✝ 1)
    ⊢ Eq a.cast ↑a.val
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem _root_.Prod.fst_zmod_cast (a : ZMod n) : (cast a : R × S).fst = cast a := by
  /-
    n : Nat
    R : Type u_1
    inst✝¹ : AddGroupWithOne R
    S : Type u_2
    inst✝ : AddGroupWithOne S
    a : ZMod n
    ⊢ Eq a.cast.1 a.cast
  -/
  cases n
    /-
      case zero
      R : Type u_1
      inst✝¹ : AddGroupWithOne R
      S : Type u_2
      inst✝ : AddGroupWithOne S
      a : ZMod 0
      ⊢ Eq a.cast.1 a.cast
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case succ
      R : Type u_1
      inst✝¹ : AddGroupWithOne R
      S : Type u_2
      inst✝ : AddGroupWithOne S
      n✝ : Nat
      a : ZMod (HAdd.hAdd n✝ 1)
      ⊢ Eq a.cast.1 a.cast
    -/
  · simp [ZMod.cast]
    /-
      🎉 no goals
    -/


@[simp]
theorem _root_.Prod.snd_zmod_cast (a : ZMod n) : (cast a : R × S).snd = cast a := by
  /-
    n : Nat
    R : Type u_1
    inst✝¹ : AddGroupWithOne R
    S : Type u_2
    inst✝ : AddGroupWithOne S
    a : ZMod n
    ⊢ Eq a.cast.2 a.cast
  -/
  cases n
    /-
      case zero
      R : Type u_1
      inst✝¹ : AddGroupWithOne R
      S : Type u_2
      inst✝ : AddGroupWithOne S
      a : ZMod 0
      ⊢ Eq a.cast.2 a.cast
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case succ
      R : Type u_1
      inst✝¹ : AddGroupWithOne R
      S : Type u_2
      inst✝ : AddGroupWithOne S
      n✝ : Nat
      a : ZMod (HAdd.hAdd n✝ 1)
      ⊢ Eq a.cast.2 a.cast
    -/
  · simp [ZMod.cast]
    /-
      🎉 no goals
    -/


/-- So-named because the coercion is `Nat.cast` into `ZMod`. For `Nat.cast` into an arbitrary ring,
see `ZMod.natCast_val`. -/
theorem natCast_zmod_val {n : ℕ} [NeZero n] (a : ZMod n) : (a.val : ZMod n) = a := by
  /-
    n : Nat
    inst✝ : NeZero n
    a : ZMod n
    ⊢ Eq (↑a.val) a
  -/
  cases n
    /-
      case zero
      inst✝ : NeZero 0
      a : ZMod 0
      ⊢ Eq (↑a.val) a
    -/
  · cases NeZero.ne 0 rfl
    /-
      🎉 no goals
    -/
    /-
      case succ
      n✝ : Nat
      inst✝ : NeZero (HAdd.hAdd n✝ 1)
      a : ZMod (HAdd.hAdd n✝ 1)
      ⊢ Eq (↑a.val) a
    -/
  · apply Fin.cast_val_eq_self
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-04-17")]
alias nat_cast_zmod_val := natCast_zmod_val


theorem natCast_rightInverse [NeZero n] : Function.RightInverse val ((↑) : ℕ → ZMod n) :=
  natCast_zmod_val


@[deprecated (since := "2024-04-17")]
alias nat_cast_rightInverse := natCast_rightInverse


theorem natCast_zmod_surjective [NeZero n] : Function.Surjective ((↑) : ℕ → ZMod n) :=
  natCast_rightInverse.surjective


@[deprecated (since := "2024-04-17")]
alias nat_cast_zmod_surjective := natCast_zmod_surjective


/-- So-named because the outer coercion is `Int.cast` into `ZMod`. For `Int.cast` into an arbitrary
ring, see `ZMod.intCast_cast`. -/
@[norm_cast]
theorem intCast_zmod_cast (a : ZMod n) : ((cast a : ℤ) : ZMod n) = a := by
  /-
    n : Nat
    a : ZMod n
    ⊢ Eq (↑a.cast) a
  -/
  cases n
    /-
      case zero
      a : ZMod 0
      ⊢ Eq (↑a.cast) a
    -/
  · simp [ZMod.cast, ZMod]
    /-
      🎉 no goals
    -/
    /-
      case succ
      n✝ : Nat
      a : ZMod (HAdd.hAdd n✝ 1)
      ⊢ Eq (↑a.cast) a
    -/
  · dsimp [ZMod.cast, ZMod]
    /-
      case succ
      n✝ : Nat
      a : ZMod (HAdd.hAdd n✝ 1)
      ⊢ Eq (↑↑a.val) a
    -/
    erw [Int.cast_natCast, Fin.cast_val_eq_self]
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-04-17")]
alias int_cast_zmod_cast := intCast_zmod_cast


theorem intCast_rightInverse : Function.RightInverse (cast : ZMod n → ℤ) ((↑) : ℤ → ZMod n) :=
  intCast_zmod_cast


@[deprecated (since := "2024-04-17")]
alias int_cast_rightInverse := intCast_rightInverse


theorem intCast_surjective : Function.Surjective ((↑) : ℤ → ZMod n) :=
  intCast_rightInverse.surjective


@[deprecated (since := "2024-04-17")]
alias int_cast_surjective := intCast_surjective


lemma «forall» {P : ZMod n → Prop} : (∀ x, P x) ↔ ∀ x : ℤ, P x := intCast_surjective.forall

lemma «exists» {P : ZMod n → Prop} : (∃ x, P x) ↔ ∃ x : ℤ, P x := intCast_surjective.exists


theorem cast_id : ∀ (n) (i : ZMod n), (ZMod.cast i : ZMod n) = i
  | 0, _ => Int.cast_id
  | _ + 1, i => natCast_zmod_val i


@[simp]
theorem cast_id' : (ZMod.cast : ZMod n → ZMod n) = id :=
  funext (cast_id n)


/-- The coercions are respectively `Nat.cast` and `ZMod.cast`. -/
@[simp]
theorem natCast_comp_val [NeZero n] : ((↑) : ℕ → R) ∘ (val : ZMod n → ℕ) = cast := by
  /-
    n : Nat
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : NeZero n
    ⊢ Eq (Function.comp Nat.cast ZMod.val) ZMod.cast
  -/
  cases n
    /-
      case zero
      R : Type u_1
      inst✝¹ : Ring R
      inst✝ : NeZero 0
      ⊢ Eq (Function.comp Nat.cast ZMod.val) ZMod.cast
    -/
  · cases NeZero.ne 0 rfl
    /-
      🎉 no goals
    -/
  /-
    case succ
    R : Type u_1
    inst✝¹ : Ring R
    n✝ : Nat
    inst✝ : NeZero (HAdd.hAdd n✝ 1)
    ⊢ Eq (Function.comp Nat.cast ZMod.val) ZMod.cast
  -/
  rfl
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias nat_cast_comp_val := natCast_comp_val


/-- The coercions are respectively `Int.cast`, `ZMod.cast`, and `ZMod.cast`. -/
@[simp]
theorem intCast_comp_cast : ((↑) : ℤ → R) ∘ (cast : ZMod n → ℤ) = cast := by
  /-
    n : Nat
    R : Type u_1
    inst✝ : Ring R
    ⊢ Eq (Function.comp Int.cast ZMod.cast) ZMod.cast
  -/
  cases n
    /-
      case zero
      R : Type u_1
      inst✝ : Ring R
      ⊢ Eq (Function.comp Int.cast ZMod.cast) ZMod.cast
    -/
  · exact congr_arg (Int.cast ∘ ·) ZMod.cast_id'
    /-
      🎉 no goals
    -/
    /-
      case succ
      R : Type u_1
      inst✝ : Ring R
      n✝ : Nat
      ⊢ Eq (Function.comp Int.cast ZMod.cast) ZMod.cast
    -/
  · ext
    /-
      case succ.h
      R : Type u_1
      inst✝ : Ring R
      n✝ : Nat
      x✝ : ZMod (HAdd.hAdd n✝ 1)
      ⊢ Eq (Function.comp Int.cast ZMod.cast x✝) x✝.cast
    -/
    simp [ZMod, ZMod.cast]
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-04-17")]
alias int_cast_comp_cast := intCast_comp_cast


@[simp]
theorem natCast_val [NeZero n] (i : ZMod n) : (i.val : R) = cast i :=
  congr_fun (natCast_comp_val R) i


@[deprecated (since := "2024-04-17")]
alias nat_cast_val := natCast_val


@[simp]
theorem intCast_cast (i : ZMod n) : ((cast i : ℤ) : R) = cast i :=
  congr_fun (intCast_comp_cast R) i


@[deprecated (since := "2024-04-17")]
alias int_cast_cast := intCast_cast


theorem cast_add_eq_ite {n : ℕ} (a b : ZMod n) :
    (cast (a + b) : ℤ) =
      if (n : ℤ) ≤ cast a + cast b then (cast a + cast b - n : ℤ) else cast a + cast b := by
  /-
    n : Nat
    a b : ZMod n
    ⊢ Eq (HAdd.hAdd a b).cast (ite (LE.le (↑n) (HAdd.hAdd a.cast b.cast)) (HSub.hS …
  -/
  cases' n with n
    /-
      case zero
      a b : ZMod 0
      ⊢ Eq (HAdd.hAdd a b).cast (ite (LE.le (↑0) (HAdd.hAdd a.cast b.cast)) (HSub.hS …
    -/
  · simp; rfl
          /-
            🎉 no goals
          -/
  /-
    case succ
    n : Nat
    a b : ZMod (HAdd.hAdd n 1)
    ⊢ Eq (HAdd.hAdd a b).cast (ite (LE.le (↑(HAdd.hAdd n 1)) (HAdd.hAdd a.cast b.c …
  -/
  change Fin (n + 1) at a b
  /-
    case succ
    n : Nat
    a b : Fin (HAdd.hAdd n 1)
    ⊢ Eq (HAdd.hAdd a b).cast (ite (LE.le (↑(HAdd.hAdd n 1)) (HAdd.hAdd (ZMod.cast …
  -/
  change ((((a + b) : Fin (n + 1)) : ℕ) : ℤ) = if ((n + 1 : ℕ) : ℤ) ≤ (a : ℕ) + b then _ else _
  /-
    case succ
    n : Nat
    a b : Fin (HAdd.hAdd n 1)
    ⊢ Eq (↑↑(HAdd.hAdd a b)) (ite (LE.le (↑(HAdd.hAdd n 1)) (HAdd.hAdd ↑↑a ↑↑b)) ( …
  -/
  simp only [Fin.val_add_eq_ite, Int.ofNat_succ, Int.ofNat_le]
  /-
    case succ
    n : Nat
    a b : Fin (HAdd.hAdd n 1)
    ⊢ Eq (↑(ite (LE.le (HAdd.hAdd n 1) (HAdd.hAdd ↑a ↑b)) (HSub.hSub (HAdd.hAdd ↑a …
  -/
  norm_cast
  /-
    case succ
    n : Nat
    a b : Fin (HAdd.hAdd n 1)
    ⊢ Eq (↑(ite (LE.le (HAdd.hAdd n 1) (HAdd.hAdd ↑a ↑b)) (HSub.hSub (HAdd.hAdd ↑a …
  -/
  split_ifs with h
    /-
      case pos
      n : Nat
      a b : Fin (HAdd.hAdd n 1)
      h : LE.le (HAdd.hAdd n 1) (HAdd.hAdd ↑a ↑b)
      ⊢ Eq (↑(HSub.hSub (HAdd.hAdd ↑a ↑b) (HAdd.hAdd n 1))) (HSub.hSub (HAdd.hAdd (Z …
    -/
  · rw [Nat.cast_sub h]
    /-
      case pos
      n : Nat
      a b : Fin (HAdd.hAdd n 1)
      h : LE.le (HAdd.hAdd n 1) (HAdd.hAdd ↑a ↑b)
      ⊢ Eq (HSub.hSub ↑(HAdd.hAdd ↑a ↑b) ↑(HAdd.hAdd n 1)) (HSub.hSub (HAdd.hAdd (ZM …
    -/
    congr
    /-
      🎉 no goals
    -/
    /-
      case neg
      n : Nat
      a b : Fin (HAdd.hAdd n 1)
      h : Not (LE.le (HAdd.hAdd n 1) (HAdd.hAdd ↑a ↑b))
      ⊢ Eq (↑(HAdd.hAdd ↑a ↑b)) (HAdd.hAdd (ZMod.cast a) (ZMod.cast b))
    -/
  · rfl
    /-
      🎉 no goals
    -/


@[simp]
theorem cast_one (h : m ∣ n) : (cast (1 : ZMod n) : R) = 1 := by
  /-
    n : Nat
    R : Type u_1
    inst✝¹ : Ring R
    m : Nat
    inst✝ : CharP R m
    h : Dvd.dvd m n
    ⊢ Eq (ZMod.cast 1) 1
  -/
  cases' n with n
    /-
      case zero
      R : Type u_1
      inst✝¹ : Ring R
      m : Nat
      inst✝ : CharP R m
      h : Dvd.dvd m 0
      ⊢ Eq (ZMod.cast 1) 1
    -/
  · exact Int.cast_one
    /-
      🎉 no goals
    -/
  /-
    case succ
    R : Type u_1
    inst✝¹ : Ring R
    m : Nat
    inst✝ : CharP R m
    n : Nat
    h : Dvd.dvd m (HAdd.hAdd n 1)
    ⊢ Eq (ZMod.cast 1) 1
  -/
  show ((1 % (n + 1) : ℕ) : R) = 1
  /-
    case succ
    R : Type u_1
    inst✝¹ : Ring R
    m : Nat
    inst✝ : CharP R m
    n : Nat
    h : Dvd.dvd m (HAdd.hAdd n 1)
    ⊢ Eq (↑(HMod.hMod 1 (HAdd.hAdd n 1))) 1
  -/
  cases n
    /-
      case succ.zero
      R : Type u_1
      inst✝¹ : Ring R
      m : Nat
      inst✝ : CharP R m
      h : Dvd.dvd m (HAdd.hAdd 0 1)
      ⊢ Eq (↑(HMod.hMod 1 (HAdd.hAdd 0 1))) 1
    -/
  · rw [Nat.dvd_one] at h
    /-
      case succ.zero
      R : Type u_1
      inst✝¹ : Ring R
      m : Nat
      inst✝ : CharP R m
      h : Eq m 1
      ⊢ Eq (↑(HMod.hMod 1 (HAdd.hAdd 0 1))) 1
    -/
    subst m
    /-
      case succ.zero
      R : Type u_1
      inst✝¹ : Ring R
      inst✝ : CharP R 1
      ⊢ Eq (↑(HMod.hMod 1 (HAdd.hAdd 0 1))) 1
    -/
    subsingleton [CharP.CharOne.subsingleton]
    /-
      🎉 no goals
    -/
  /-
    case succ.succ
    R : Type u_1
    inst✝¹ : Ring R
    m : Nat
    inst✝ : CharP R m
    n✝ : Nat
    h : Dvd.dvd m (HAdd.hAdd (HAdd.hAdd n✝ 1) 1)
    ⊢ Eq (↑(HMod.hMod 1 (HAdd.hAdd (HAdd.hAdd n✝ 1) 1))) 1
  -/
  rw [Nat.mod_eq_of_lt]
    /-
      case succ.succ
      R : Type u_1
      inst✝¹ : Ring R
      m : Nat
      inst✝ : CharP R m
      n✝ : Nat
      h : Dvd.dvd m (HAdd.hAdd (HAdd.hAdd n✝ 1) 1)
      ⊢ Eq (↑1) 1
    -/
  · exact Nat.cast_one
    /-
      🎉 no goals
    -/
  /-
    case succ.succ
    R : Type u_1
    inst✝¹ : Ring R
    m : Nat
    inst✝ : CharP R m
    n✝ : Nat
    h : Dvd.dvd m (HAdd.hAdd (HAdd.hAdd n✝ 1) 1)
    ⊢ LT.lt 1 (HAdd.hAdd (HAdd.hAdd n✝ 1) 1)
  -/
  exact Nat.lt_of_sub_eq_succ rfl
  /-
    🎉 no goals
  -/


theorem cast_add (h : m ∣ n) (a b : ZMod n) : (cast (a + b : ZMod n) : R) = cast a + cast b := by
  /-
    n : Nat
    R : Type u_1
    inst✝¹ : Ring R
    m : Nat
    inst✝ : CharP R m
    h : Dvd.dvd m n
    a b : ZMod n
    ⊢ Eq (HAdd.hAdd a b).cast (HAdd.hAdd a.cast b.cast)
  -/
  cases n
    /-
      case zero
      R : Type u_1
      inst✝¹ : Ring R
      m : Nat
      inst✝ : CharP R m
      h : Dvd.dvd m 0
      a b : ZMod 0
      ⊢ Eq (HAdd.hAdd a b).cast (HAdd.hAdd a.cast b.cast)
    -/
  · apply Int.cast_add
    /-
      🎉 no goals
    -/
  /-
    case succ
    R : Type u_1
    inst✝¹ : Ring R
    m : Nat
    inst✝ : CharP R m
    n✝ : Nat
    h : Dvd.dvd m (HAdd.hAdd n✝ 1)
    a b : ZMod (HAdd.hAdd n✝ 1)
    ⊢ Eq (HAdd.hAdd a b).cast (HAdd.hAdd a.cast b.cast)
  -/
  symm
  /-
    case succ
    R : Type u_1
    inst✝¹ : Ring R
    m : Nat
    inst✝ : CharP R m
    n✝ : Nat
    h : Dvd.dvd m (HAdd.hAdd n✝ 1)
    a b : ZMod (HAdd.hAdd n✝ 1)
    ⊢ Eq (HAdd.hAdd a.cast b.cast) (HAdd.hAdd a b).cast
  -/
  dsimp [ZMod, ZMod.cast]
  erw [← Nat.cast_add, ← sub_eq_zero, ← Nat.cast_sub (Nat.mod_le _ _),
    @CharP.cast_eq_zero_iff R _ m]
  /-
    case succ
    R : Type u_1
    inst✝¹ : Ring R
    m : Nat
    inst✝ : CharP R m
    n✝ : Nat
    h : Dvd.dvd m (HAdd.hAdd n✝ 1)
    a b : ZMod (HAdd.hAdd n✝ 1)
    ⊢ Dvd.dvd m (HSub.hSub (HAdd.hAdd a.val b.val) (HMod.hMod (HAdd.hAdd a.val b.v …
  -/
  exact h.trans (Nat.dvd_sub_mod _)
  /-
    🎉 no goals
  -/


theorem cast_mul (h : m ∣ n) (a b : ZMod n) : (cast (a * b : ZMod n) : R) = cast a * cast b := by
  /-
    n : Nat
    R : Type u_1
    inst✝¹ : Ring R
    m : Nat
    inst✝ : CharP R m
    h : Dvd.dvd m n
    a b : ZMod n
    ⊢ Eq (HMul.hMul a b).cast (HMul.hMul a.cast b.cast)
  -/
  cases n
    /-
      case zero
      R : Type u_1
      inst✝¹ : Ring R
      m : Nat
      inst✝ : CharP R m
      h : Dvd.dvd m 0
      a b : ZMod 0
      ⊢ Eq (HMul.hMul a b).cast (HMul.hMul a.cast b.cast)
    -/
  · apply Int.cast_mul
    /-
      🎉 no goals
    -/
  /-
    case succ
    R : Type u_1
    inst✝¹ : Ring R
    m : Nat
    inst✝ : CharP R m
    n✝ : Nat
    h : Dvd.dvd m (HAdd.hAdd n✝ 1)
    a b : ZMod (HAdd.hAdd n✝ 1)
    ⊢ Eq (HMul.hMul a b).cast (HMul.hMul a.cast b.cast)
  -/
  symm
  /-
    case succ
    R : Type u_1
    inst✝¹ : Ring R
    m : Nat
    inst✝ : CharP R m
    n✝ : Nat
    h : Dvd.dvd m (HAdd.hAdd n✝ 1)
    a b : ZMod (HAdd.hAdd n✝ 1)
    ⊢ Eq (HMul.hMul a.cast b.cast) (HMul.hMul a b).cast
  -/
  dsimp [ZMod, ZMod.cast]
  erw [← Nat.cast_mul, ← sub_eq_zero, ← Nat.cast_sub (Nat.mod_le _ _),
    @CharP.cast_eq_zero_iff R _ m]
  /-
    case succ
    R : Type u_1
    inst✝¹ : Ring R
    m : Nat
    inst✝ : CharP R m
    n✝ : Nat
    h : Dvd.dvd m (HAdd.hAdd n✝ 1)
    a b : ZMod (HAdd.hAdd n✝ 1)
    ⊢ Dvd.dvd m (HSub.hSub (HMul.hMul a.val b.val) (HMod.hMod (HMul.hMul a.val b.v …
  -/
  exact h.trans (Nat.dvd_sub_mod _)
  /-
    🎉 no goals
  -/


/-- The canonical ring homomorphism from `ZMod n` to a ring of characteristic dividing `n`.

See also `ZMod.lift` for a generalized version working in `AddGroup`s.
-/
def castHom (h : m ∣ n) (R : Type*) [Ring R] [CharP R m] : ZMod n →+* R where
  toFun := cast
  map_zero' := cast_zero
  map_one' := cast_one h
  map_add' := cast_add h
  map_mul' := cast_mul h


@[simp]
theorem castHom_apply {h : m ∣ n} (i : ZMod n) : castHom h R i = cast i :=
  rfl


@[simp]
theorem cast_sub (h : m ∣ n) (a b : ZMod n) : (cast (a - b : ZMod n) : R) = cast a - cast b :=
  (castHom h R).map_sub a b


@[simp]
theorem cast_neg (h : m ∣ n) (a : ZMod n) : (cast (-a : ZMod n) : R) = -(cast a) :=
  (castHom h R).map_neg a


@[simp]
theorem cast_pow (h : m ∣ n) (a : ZMod n) (k : ℕ) : (cast (a ^ k : ZMod n) : R) = (cast a) ^ k :=
  (castHom h R).map_pow a k


@[simp, norm_cast]
theorem cast_natCast (h : m ∣ n) (k : ℕ) : (cast (k : ZMod n) : R) = k :=
  map_natCast (castHom h R) k


@[deprecated (since := "2024-04-17")]
alias cast_nat_cast := cast_natCast


@[simp, norm_cast]
theorem cast_intCast (h : m ∣ n) (k : ℤ) : (cast (k : ZMod n) : R) = k :=
  map_intCast (castHom h R) k


@[deprecated (since := "2024-04-17")]
alias cast_int_cast := cast_intCast


@[simp]
theorem cast_one' : (cast (1 : ZMod n) : R) = 1 :=
  cast_one dvd_rfl


@[simp]
theorem cast_add' (a b : ZMod n) : (cast (a + b : ZMod n) : R) = cast a + cast b :=
  cast_add dvd_rfl a b


@[simp]
theorem cast_mul' (a b : ZMod n) : (cast (a * b : ZMod n) : R) = cast a * cast b :=
  cast_mul dvd_rfl a b


@[simp]
theorem cast_sub' (a b : ZMod n) : (cast (a - b : ZMod n) : R) = cast a - cast b :=
  cast_sub dvd_rfl a b


@[simp]
theorem cast_pow' (a : ZMod n) (k : ℕ) : (cast (a ^ k : ZMod n) : R) = (cast a : R) ^ k :=
  cast_pow dvd_rfl a k


@[simp, norm_cast]
theorem cast_natCast' (k : ℕ) : (cast (k : ZMod n) : R) = k :=
  cast_natCast dvd_rfl k


@[deprecated (since := "2024-04-17")]
alias cast_nat_cast' := cast_natCast'


@[simp, norm_cast]
theorem cast_intCast' (k : ℤ) : (cast (k : ZMod n) : R) = k :=
  cast_intCast dvd_rfl k


@[deprecated (since := "2024-04-17")]
alias cast_int_cast' := cast_intCast'


theorem castHom_injective : Function.Injective (ZMod.castHom (dvd_refl n) R) := by
  /-
    n : Nat
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : CharP R n
    ⊢ Function.Injective ⇑(ZMod.castHom ⋯ R)
  -/
  rw [injective_iff_map_eq_zero]
  /-
    n : Nat
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : CharP R n
    ⊢ ∀ (a : ZMod n), Eq ((ZMod.castHom ⋯ R) a) 0 → Eq a 0
  -/
  intro x
  /-
    n : Nat
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : CharP R n
    x : ZMod n
    ⊢ Eq ((ZMod.castHom ⋯ R) x) 0 → Eq x 0
  -/
  obtain ⟨k, rfl⟩ := ZMod.intCast_surjective x
  /-
    case intro
    n : Nat
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : CharP R n
    k : Int
    ⊢ Eq ((ZMod.castHom ⋯ R) ↑k) 0 → Eq (↑k) 0
  -/
  rw [map_intCast, CharP.intCast_eq_zero_iff R n, CharP.intCast_eq_zero_iff (ZMod n) n]
  /-
    case intro
    n : Nat
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : CharP R n
    k : Int
    ⊢ Dvd.dvd (↑n) k → Dvd.dvd (↑n) k
  -/
  exact id
  /-
    🎉 no goals
  -/


theorem castHom_bijective [Fintype R] (h : Fintype.card R = n) :
    Function.Bijective (ZMod.castHom (dvd_refl n) R) := by
  haveI : NeZero n :=
    ⟨by
      intro hn
      rw [hn] at h
      exact (Fintype.card_eq_zero_iff.mp h).elim' 0⟩
  /-
    n : Nat
    R : Type u_1
    inst✝² : Ring R
    inst✝¹ : CharP R n
    inst✝ : Fintype R
    h : Eq (Fintype.card R) n
    this : NeZero n
    ⊢ Function.Bijective ⇑(ZMod.castHom ⋯ R)
  -/
  rw [Fintype.bijective_iff_injective_and_card, ZMod.card, h, eq_self_iff_true, and_true]
  /-
    n : Nat
    R : Type u_1
    inst✝² : Ring R
    inst✝¹ : CharP R n
    inst✝ : Fintype R
    h : Eq (Fintype.card R) n
    this : NeZero n
    ⊢ Function.Injective ⇑(ZMod.castHom ⋯ R)
  -/
  apply ZMod.castHom_injective
  /-
    🎉 no goals
  -/


/-- The unique ring isomorphism between `ZMod n` and a ring `R`
of characteristic `n` and cardinality `n`. -/
noncomputable def ringEquiv [Fintype R] (h : Fintype.card R = n) : ZMod n ≃+* R :=
  RingEquiv.ofBijective _ (ZMod.castHom_bijective R h)


/-- The unique ring isomorphism between `ZMod p` and a ring `R` of cardinality a prime `p`.

If you need any property of this isomorphism, first of all use `ringEquivOfPrime_eq_ringEquiv`
below (after `have : CharP R p := ...`) and deduce it by the results about `ZMod.ringEquiv`. -/
noncomputable def ringEquivOfPrime [Fintype R] {p : ℕ} (hp : p.Prime) (hR : Fintype.card R = p) :
    ZMod p ≃+* R :=
  have : Nontrivial R := Fintype.one_lt_card_iff_nontrivial.1 (hR ▸ hp.one_lt)
  -- The following line exists as `charP_of_card_eq_prime` in `Mathlib.Algebra.CharP.CharAndCard`.
  have : CharP R p := (CharP.charP_iff_prime_eq_zero hp).2 (hR ▸ Nat.cast_card_eq_zero R)
  ZMod.ringEquiv R hR


@[simp]
lemma ringEquivOfPrime_eq_ringEquiv [Fintype R] {p : ℕ} [CharP R p] (hp : p.Prime)
    (hR : Fintype.card R = p) : ringEquivOfPrime R hp hR = ringEquiv R hR := rfl


/-- The identity between `ZMod m` and `ZMod n` when `m = n`, as a ring isomorphism. -/
def ringEquivCongr {m n : ℕ} (h : m = n) : ZMod m ≃+* ZMod n := by
  /-
    n✝ : Nat
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : CharP R n✝
    m n : Nat
    h : Eq m n
    ⊢ RingEquiv (ZMod m) (ZMod n)
  -/
  cases' m with m <;> cases' n with n
    /-
      case zero.zero
      n : Nat
      R : Type u_1
      inst✝¹ : Ring R
      inst✝ : CharP R n
      h : Eq 0 0
      ⊢ RingEquiv (ZMod 0) (ZMod 0)
    -/
  · exact RingEquiv.refl _
    /-
      🎉 no goals
    -/
    /-
      case zero.succ
      n✝ : Nat
      R : Type u_1
      inst✝¹ : Ring R
      inst✝ : CharP R n✝
      n : Nat
      h : Eq 0 (HAdd.hAdd n 1)
      ⊢ RingEquiv (ZMod 0) (ZMod (HAdd.hAdd n 1))
    -/
  · exfalso
    /-
      case zero.succ
      n✝ : Nat
      R : Type u_1
      inst✝¹ : Ring R
      inst✝ : CharP R n✝
      n : Nat
      h : Eq 0 (HAdd.hAdd n 1)
      ⊢ False
    -/
    exact n.succ_ne_zero h.symm
    /-
      🎉 no goals
    -/
    /-
      case succ.zero
      n : Nat
      R : Type u_1
      inst✝¹ : Ring R
      inst✝ : CharP R n
      m : Nat
      h : Eq (HAdd.hAdd m 1) 0
      ⊢ RingEquiv (ZMod (HAdd.hAdd m 1)) (ZMod 0)
    -/
  · exfalso
    /-
      case succ.zero
      n : Nat
      R : Type u_1
      inst✝¹ : Ring R
      inst✝ : CharP R n
      m : Nat
      h : Eq (HAdd.hAdd m 1) 0
      ⊢ False
    -/
    exact m.succ_ne_zero h
    /-
      🎉 no goals
    -/
  · exact
      { finCongr h with
        map_mul' := fun a b => by
          dsimp [ZMod]
          ext
          rw [Fin.coe_cast, Fin.coe_mul, Fin.coe_mul, Fin.coe_cast, Fin.coe_cast, ← h]
        map_add' := fun a b => by
          dsimp [ZMod]
          ext
          rw [Fin.coe_cast, Fin.val_add, Fin.val_add, Fin.coe_cast, Fin.coe_cast, ← h] }


@[simp] lemma ringEquivCongr_refl (a : ℕ) : ringEquivCongr (rfl : a = a) = .refl _ := by
  /-
    a : Nat
    ⊢ Eq (ZMod.ringEquivCongr ⋯) (RingEquiv.refl (ZMod a))
  -/
              /-
                🎉 no goals
              -/
  cases a <;> rfl
              /-
                🎉 no goals
              -/


lemma ringEquivCongr_refl_apply {a : ℕ} (x : ZMod a) : ringEquivCongr rfl x = x := by
  /-
    a : Nat
    x : ZMod a
    ⊢ Eq ((ZMod.ringEquivCongr ⋯) x) x
  -/
  rw [ringEquivCongr_refl]
  /-
    a : Nat
    x : ZMod a
    ⊢ Eq ((RingEquiv.refl (ZMod a)) x) x
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma ringEquivCongr_symm {a b : ℕ} (hab : a = b) :
    (ringEquivCongr hab).symm = ringEquivCongr hab.symm := by
  /-
    a b : Nat
    hab : Eq a b
    ⊢ Eq (ZMod.ringEquivCongr hab).symm (ZMod.ringEquivCongr ⋯)
  -/
  subst hab
  /-
    a : Nat
    ⊢ Eq (ZMod.ringEquivCongr ⋯).symm (ZMod.ringEquivCongr ⋯)
  -/
              /-
                🎉 no goals
              -/
  cases a <;> rfl
              /-
                🎉 no goals
              -/


lemma ringEquivCongr_trans {a b c : ℕ} (hab : a = b) (hbc : b = c) :
    (ringEquivCongr hab).trans (ringEquivCongr hbc) = ringEquivCongr (hab.trans hbc) := by
  /-
    a b c : Nat
    hab : Eq a b
    hbc : Eq b c
    ⊢ Eq ((ZMod.ringEquivCongr hab).trans (ZMod.ringEquivCongr hbc)) (ZMod.ringEqu …
  -/
  subst hab hbc
  /-
    a : Nat
    ⊢ Eq ((ZMod.ringEquivCongr ⋯).trans (ZMod.ringEquivCongr ⋯)) (ZMod.ringEquivCo …
  -/
              /-
                🎉 no goals
              -/
  cases a <;> rfl
              /-
                🎉 no goals
              -/


lemma ringEquivCongr_ringEquivCongr_apply {a b c : ℕ} (hab : a = b) (hbc : b = c) (x : ZMod a) :
    ringEquivCongr hbc (ringEquivCongr hab x) = ringEquivCongr (hab.trans hbc) x := by
  /-
    a b c : Nat
    hab : Eq a b
    hbc : Eq b c
    x : ZMod a
    ⊢ Eq ((ZMod.ringEquivCongr hbc) ((ZMod.ringEquivCongr hab) x)) ((ZMod.ringEqui …
  -/
  rw [← ringEquivCongr_trans hab hbc]
  /-
    a b c : Nat
    hab : Eq a b
    hbc : Eq b c
    x : ZMod a
    ⊢ Eq ((ZMod.ringEquivCongr hbc) ((ZMod.ringEquivCongr hab) x)) (((ZMod.ringEqu …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma ringEquivCongr_val {a b : ℕ} (h : a = b) (x : ZMod a) :
    ZMod.val ((ZMod.ringEquivCongr h) x) = ZMod.val x := by
  /-
    a b : Nat
    h : Eq a b
    x : ZMod a
    ⊢ Eq ((ZMod.ringEquivCongr h) x).val x.val
  -/
  subst h
  /-
    a : Nat
    x : ZMod a
    ⊢ Eq ((ZMod.ringEquivCongr ⋯) x).val x.val
  -/
              /-
                🎉 no goals
              -/
  cases a <;> rfl
              /-
                🎉 no goals
              -/


lemma ringEquivCongr_intCast {a b : ℕ} (h : a = b) (z : ℤ) :
    ZMod.ringEquivCongr h z = z := by
  /-
    a b : Nat
    h : Eq a b
    z : Int
    ⊢ Eq ((ZMod.ringEquivCongr h) ↑z) ↑z
  -/
  subst h
  /-
    a : Nat
    z : Int
    ⊢ Eq ((ZMod.ringEquivCongr ⋯) ↑z) ↑z
  -/
              /-
                🎉 no goals
              -/
  cases a <;> rfl
              /-
                🎉 no goals
              -/


@[deprecated (since := "2024-05-25")] alias int_coe_ringEquivCongr := ringEquivCongr_intCast


@[simp]
theorem val_eq_zero : ∀ {n : ℕ} (a : ZMod n), a.val = 0 ↔ a = 0
  | 0, _ => Int.natAbs_eq_zero
  | n + 1, a => by
    /-
      n : Nat
      a : ZMod (HAdd.hAdd n 1)
      ⊢ Iff (Eq a.val 0) (Eq a 0)
    -/
    rw [Fin.ext_iff]
    /-
      n : Nat
      a : ZMod (HAdd.hAdd n 1)
      ⊢ Iff (Eq a.val 0) (Eq ↑a ↑0)
    -/
    exact Iff.rfl
    /-
      🎉 no goals
    -/


theorem intCast_eq_intCast_iff (a b : ℤ) (c : ℕ) : (a : ZMod c) = (b : ZMod c) ↔ a ≡ b [ZMOD c] :=
  CharP.intCast_eq_intCast (ZMod c) c


@[deprecated (since := "2024-04-17")]
alias int_cast_eq_int_cast_iff := intCast_eq_intCast_iff


theorem intCast_eq_intCast_iff' (a b : ℤ) (c : ℕ) : (a : ZMod c) = (b : ZMod c) ↔ a % c = b % c :=
  ZMod.intCast_eq_intCast_iff a b c


@[deprecated (since := "2024-04-17")]
alias int_cast_eq_int_cast_iff' := intCast_eq_intCast_iff'


theorem val_intCast {n : ℕ} (a : ℤ) [NeZero n] : ↑(a : ZMod n).val = a % n := by
  /-
    n : Nat
    a : Int
    inst✝ : NeZero n
    ⊢ Eq (↑(↑a).val) (HMod.hMod a ↑n)
  -/
  have hle : (0 : ℤ) ≤ ↑(a : ZMod n).val := Int.natCast_nonneg _
  /-
    n : Nat
    a : Int
    inst✝ : NeZero n
    hle : LE.le 0 ↑(↑a).val
    ⊢ Eq (↑(↑a).val) (HMod.hMod a ↑n)
  -/
  have hlt : ↑(a : ZMod n).val < (n : ℤ) := Int.ofNat_lt.mpr (ZMod.val_lt a)
  /-
    n : Nat
    a : Int
    inst✝ : NeZero n
    hle : LE.le 0 ↑(↑a).val
    hlt : LT.lt ↑(↑a).val ↑n
    ⊢ Eq (↑(↑a).val) (HMod.hMod a ↑n)
  -/
  refine (Int.emod_eq_of_lt hle hlt).symm.trans ?_
  /-
    n : Nat
    a : Int
    inst✝ : NeZero n
    hle : LE.le 0 ↑(↑a).val
    hlt : LT.lt ↑(↑a).val ↑n
    ⊢ Eq (HMod.hMod ↑(↑a).val ↑n) (HMod.hMod a ↑n)
  -/
  rw [← ZMod.intCast_eq_intCast_iff', Int.cast_natCast, ZMod.natCast_val, ZMod.cast_id]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias val_int_cast := val_intCast


theorem natCast_eq_natCast_iff (a b c : ℕ) : (a : ZMod c) = (b : ZMod c) ↔ a ≡ b [MOD c] := by
  /-
    a b c : Nat
    ⊢ Iff (Eq ↑a ↑b) (c.ModEq a b)
  -/
  simpa [Int.natCast_modEq_iff] using ZMod.intCast_eq_intCast_iff a b c
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias nat_cast_eq_nat_cast_iff := natCast_eq_natCast_iff


theorem natCast_eq_natCast_iff' (a b c : ℕ) : (a : ZMod c) = (b : ZMod c) ↔ a % c = b % c :=
  ZMod.natCast_eq_natCast_iff a b c


@[deprecated (since := "2024-04-17")]
alias nat_cast_eq_nat_cast_iff' := natCast_eq_natCast_iff'


theorem intCast_zmod_eq_zero_iff_dvd (a : ℤ) (b : ℕ) : (a : ZMod b) = 0 ↔ (b : ℤ) ∣ a := by
  /-
    a : Int
    b : Nat
    ⊢ Iff (Eq (↑a) 0) (Dvd.dvd (↑b) a)
  -/
  rw [← Int.cast_zero, ZMod.intCast_eq_intCast_iff, Int.modEq_zero_iff_dvd]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias int_cast_zmod_eq_zero_iff_dvd := intCast_zmod_eq_zero_iff_dvd


theorem intCast_eq_intCast_iff_dvd_sub (a b : ℤ) (c : ℕ) : (a : ZMod c) = ↑b ↔ ↑c ∣ b - a := by
  /-
    a b : Int
    c : Nat
    ⊢ Iff (Eq ↑a ↑b) (Dvd.dvd (↑c) (HSub.hSub b a))
  -/
  rw [ZMod.intCast_eq_intCast_iff, Int.modEq_iff_dvd]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias int_cast_eq_int_cast_iff_dvd_sub := intCast_eq_intCast_iff_dvd_sub


theorem natCast_zmod_eq_zero_iff_dvd (a b : ℕ) : (a : ZMod b) = 0 ↔ b ∣ a := by
  /-
    a b : Nat
    ⊢ Iff (Eq (↑a) 0) (Dvd.dvd b a)
  -/
  rw [← Nat.cast_zero, ZMod.natCast_eq_natCast_iff, Nat.modEq_zero_iff_dvd]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias nat_cast_zmod_eq_zero_iff_dvd := natCast_zmod_eq_zero_iff_dvd


theorem coe_intCast (a : ℤ) : cast (a : ZMod n) = a % n := by
  /-
    n : Nat
    a : Int
    ⊢ Eq (↑a).cast (HMod.hMod a ↑n)
  -/
  cases n
    /-
      case zero
      a : Int
      ⊢ Eq (↑a).cast (HMod.hMod a ↑0)
    -/
  · rw [Int.ofNat_zero, Int.emod_zero, Int.cast_id]; rfl
                                                     /-
                                                       🎉 no goals
                                                     -/
    /-
      case succ
      a : Int
      n✝ : Nat
      ⊢ Eq (↑a).cast (HMod.hMod a ↑(HAdd.hAdd n✝ 1))
    -/
  · rw [← val_intCast, val]; rfl
                             /-
                               🎉 no goals
                             -/


@[deprecated (since := "2024-04-17")]
alias coe_int_cast := coe_intCast


lemma intCast_cast_add (x y : ZMod n) : (cast (x + y) : ℤ) = (cast x + cast y) % n := by
  /-
    n : Nat
    x y : ZMod n
    ⊢ Eq (HAdd.hAdd x y).cast (HMod.hMod (HAdd.hAdd x.cast y.cast) ↑n)
  -/
  rw [← ZMod.coe_intCast, Int.cast_add, ZMod.intCast_zmod_cast, ZMod.intCast_zmod_cast]
  /-
    🎉 no goals
  -/


lemma intCast_cast_mul (x y : ZMod n) : (cast (x * y) : ℤ) = cast x * cast y % n := by
  /-
    n : Nat
    x y : ZMod n
    ⊢ Eq (HMul.hMul x y).cast (HMod.hMod (HMul.hMul x.cast y.cast) ↑n)
  -/
  rw [← ZMod.coe_intCast, Int.cast_mul, ZMod.intCast_zmod_cast, ZMod.intCast_zmod_cast]
  /-
    🎉 no goals
  -/


lemma intCast_cast_sub (x y : ZMod n) : (cast (x - y) : ℤ) = (cast x - cast y) % n := by
  /-
    n : Nat
    x y : ZMod n
    ⊢ Eq (HSub.hSub x y).cast (HMod.hMod (HSub.hSub x.cast y.cast) ↑n)
  -/
  rw [← ZMod.coe_intCast, Int.cast_sub, ZMod.intCast_zmod_cast, ZMod.intCast_zmod_cast]
  /-
    🎉 no goals
  -/


lemma intCast_cast_neg (x : ZMod n) : (cast (-x) : ℤ) = -cast x % n := by
  /-
    n : Nat
    x : ZMod n
    ⊢ Eq (Neg.neg x).cast (HMod.hMod (Neg.neg x.cast) ↑n)
  -/
  rw [← ZMod.coe_intCast, Int.cast_neg, ZMod.intCast_zmod_cast]
  /-
    🎉 no goals
  -/


@[simp]
theorem val_neg_one (n : ℕ) : (-1 : ZMod n.succ).val = n := by
  /-
    n : Nat
    ⊢ Eq (-1).val n
  -/
  dsimp [val, Fin.coe_neg]
  /-
    n : Nat
    ⊢ Eq (↑(-1)) n
  -/
  cases n
    /-
      case zero
      ⊢ Eq (↑(-1)) 0
    -/
  · simp [Nat.mod_one]
    /-
      🎉 no goals
    -/
    /-
      case succ
      n✝ : Nat
      ⊢ Eq (↑(-1)) (HAdd.hAdd n✝ 1)
    -/
  · dsimp [ZMod, ZMod.cast]
    /-
      case succ
      n✝ : Nat
      ⊢ Eq (↑(-1)) (HAdd.hAdd n✝ 1)
    -/
    rw [Fin.coe_neg_one]
    /-
      🎉 no goals
    -/


/-- `-1 : ZMod n` lifts to `n - 1 : R`. This avoids the characteristic assumption in `cast_neg`. -/
theorem cast_neg_one {R : Type*} [Ring R] (n : ℕ) : cast (-1 : ZMod n) = (n - 1 : R) := by
  /-
    R : Type u_1
    inst✝ : Ring R
    n : Nat
    ⊢ Eq (-1).cast (HSub.hSub (↑n) 1)
  -/
  cases' n with n
    /-
      case zero
      R : Type u_1
      inst✝ : Ring R
      ⊢ Eq (-1).cast (HSub.hSub (↑0) 1)
    -/
  · dsimp [ZMod, ZMod.cast]; simp
                             /-
                               🎉 no goals
                             -/
    /-
      case succ
      R : Type u_1
      inst✝ : Ring R
      n : Nat
      ⊢ Eq (-1).cast (HSub.hSub (↑(HAdd.hAdd n 1)) 1)
    -/
  · rw [← natCast_val, val_neg_one, Nat.cast_succ, add_sub_cancel_right]
    /-
      🎉 no goals
    -/


theorem cast_sub_one {R : Type*} [Ring R] {n : ℕ} (k : ZMod n) :
    (cast (k - 1 : ZMod n) : R) = (if k = 0 then (n : R) else cast k) - 1 := by
  /-
    R : Type u_1
    inst✝ : Ring R
    n : Nat
    k : ZMod n
    ⊢ Eq (HSub.hSub k 1).cast (HSub.hSub (ite (Eq k 0) (↑n) k.cast) 1)
  -/
  split_ifs with hk
    /-
      case pos
      R : Type u_1
      inst✝ : Ring R
      n : Nat
      k : ZMod n
      hk : Eq k 0
      ⊢ Eq (HSub.hSub k 1).cast (HSub.hSub (↑n) 1)
    -/
  · rw [hk, zero_sub, ZMod.cast_neg_one]
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      inst✝ : Ring R
      n : Nat
      k : ZMod n
      hk : Not (Eq k 0)
      ⊢ Eq (HSub.hSub k 1).cast (HSub.hSub k.cast 1)
    -/
  · cases n
      /-
        case neg.zero
        R : Type u_1
        inst✝ : Ring R
        k : ZMod 0
        hk : Not (Eq k 0)
        ⊢ Eq (HSub.hSub k 1).cast (HSub.hSub k.cast 1)
      -/
    · dsimp [ZMod, ZMod.cast]
      /-
        case neg.zero
        R : Type u_1
        inst✝ : Ring R
        k : ZMod 0
        hk : Not (Eq k 0)
        ⊢ Eq (↑(HSub.hSub k 1)) (HSub.hSub (↑k) 1)
      -/
      rw [Int.cast_sub, Int.cast_one]
      /-
        🎉 no goals
      -/
      /-
        case neg.succ
        R : Type u_1
        inst✝ : Ring R
        n✝ : Nat
        k : ZMod (HAdd.hAdd n✝ 1)
        hk : Not (Eq k 0)
        ⊢ Eq (HSub.hSub k 1).cast (HSub.hSub k.cast 1)
      -/
    · dsimp [ZMod, ZMod.cast, ZMod.val]
      /-
        case neg.succ
        R : Type u_1
        inst✝ : Ring R
        n✝ : Nat
        k : ZMod (HAdd.hAdd n✝ 1)
        hk : Not (Eq k 0)
        ⊢ Eq (↑↑(HSub.hSub k 1)) (HSub.hSub (↑↑k) 1)
      -/
      rw [Fin.coe_sub_one, if_neg]
        /-
          case neg.succ
          R : Type u_1
          inst✝ : Ring R
          n✝ : Nat
          k : ZMod (HAdd.hAdd n✝ 1)
          hk : Not (Eq k 0)
          ⊢ Eq (↑(HSub.hSub (↑k) 1)) (HSub.hSub (↑↑k) 1)
        -/
      · rw [Nat.cast_sub, Nat.cast_one]
        /-
          case neg.succ
          R : Type u_1
          inst✝ : Ring R
          n✝ : Nat
          k : ZMod (HAdd.hAdd n✝ 1)
          hk : Not (Eq k 0)
          ⊢ LE.le 1 ↑k
        -/
        rwa [Fin.ext_iff, Fin.val_zero, ← Ne, ← Nat.one_le_iff_ne_zero] at hk
        /-
          🎉 no goals
        -/
        /-
          case neg.succ.hnc
          R : Type u_1
          inst✝ : Ring R
          n✝ : Nat
          k : ZMod (HAdd.hAdd n✝ 1)
          hk : Not (Eq k 0)
          ⊢ Not (Eq k 0)
        -/
      · exact hk
        /-
          🎉 no goals
        -/


theorem natCast_eq_iff (p : ℕ) (n : ℕ) (z : ZMod p) [NeZero p] :
    ↑n = z ↔ ∃ k, n = z.val + p * k := by
  /-
    p n : Nat
    z : ZMod p
    inst✝ : NeZero p
    ⊢ Iff (Eq (↑n) z) (Exists fun k => Eq n (HAdd.hAdd z.val (HMul.hMul p k)))
  -/
  constructor
    /-
      case mp
      p n : Nat
      z : ZMod p
      inst✝ : NeZero p
      ⊢ Eq (↑n) z → Exists fun k => Eq n (HAdd.hAdd z.val (HMul.hMul p k))
    -/
  · rintro rfl
    /-
      case mp
      p n : Nat
      inst✝ : NeZero p
      ⊢ Exists fun k => Eq n (HAdd.hAdd (↑n).val (HMul.hMul p k))
    -/
    refine ⟨n / p, ?_⟩
    /-
      case mp
      p n : Nat
      inst✝ : NeZero p
      ⊢ Eq n (HAdd.hAdd (↑n).val (HMul.hMul p (HDiv.hDiv n p)))
    -/
    rw [val_natCast, Nat.mod_add_div]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      p n : Nat
      z : ZMod p
      inst✝ : NeZero p
      ⊢ (Exists fun k => Eq n (HAdd.hAdd z.val (HMul.hMul p k))) → Eq (↑n) z
    -/
  · rintro ⟨k, rfl⟩
    rw [Nat.cast_add, natCast_zmod_val, Nat.cast_mul, natCast_self, zero_mul,
      add_zero]


theorem intCast_eq_iff (p : ℕ) (n : ℤ) (z : ZMod p) [NeZero p] :
    ↑n = z ↔ ∃ k, n = z.val + p * k := by
  /-
    p : Nat
    n : Int
    z : ZMod p
    inst✝ : NeZero p
    ⊢ Iff (Eq (↑n) z) (Exists fun k => Eq n (HAdd.hAdd (↑z.val) (HMul.hMul (↑p) k)))
  -/
  constructor
    /-
      case mp
      p : Nat
      n : Int
      z : ZMod p
      inst✝ : NeZero p
      ⊢ Eq (↑n) z → Exists fun k => Eq n (HAdd.hAdd (↑z.val) (HMul.hMul (↑p) k))
    -/
  · rintro rfl
    /-
      case mp
      p : Nat
      n : Int
      inst✝ : NeZero p
      ⊢ Exists fun k => Eq n (HAdd.hAdd (↑(↑n).val) (HMul.hMul (↑p) k))
    -/
    refine ⟨n / p, ?_⟩
    /-
      case mp
      p : Nat
      n : Int
      inst✝ : NeZero p
      ⊢ Eq n (HAdd.hAdd (↑(↑n).val) (HMul.hMul (↑p) (HDiv.hDiv n ↑p)))
    -/
    rw [val_intCast, Int.emod_add_ediv]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      p : Nat
      n : Int
      z : ZMod p
      inst✝ : NeZero p
      ⊢ (Exists fun k => Eq n (HAdd.hAdd (↑z.val) (HMul.hMul (↑p) k))) → Eq (↑n) z
    -/
  · rintro ⟨k, rfl⟩
    rw [Int.cast_add, Int.cast_mul, Int.cast_natCast, Int.cast_natCast, natCast_val,
      ZMod.natCast_self, zero_mul, add_zero, cast_id]


@[deprecated (since := "2024-05-25")] alias nat_coe_zmod_eq_iff := natCast_eq_iff

@[deprecated (since := "2024-05-25")] alias int_coe_zmod_eq_iff := intCast_eq_iff


@[push_cast, simp]
theorem intCast_mod (a : ℤ) (b : ℕ) : ((a % b : ℤ) : ZMod b) = (a : ZMod b) := by
  /-
    a : Int
    b : Nat
    ⊢ Eq ↑(HMod.hMod a ↑b) ↑a
  -/
  rw [ZMod.intCast_eq_intCast_iff]
  /-
    a : Int
    b : Nat
    ⊢ (↑b).ModEq (HMod.hMod a ↑b) a
  -/
  apply Int.mod_modEq
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias int_cast_mod := intCast_mod


theorem ker_intCastAddHom (n : ℕ) :
    (Int.castAddHom (ZMod n)).ker = AddSubgroup.zmultiples (n : ℤ) := by
  /-
    n : Nat
    ⊢ Eq (Int.castAddHom (ZMod n)).ker (AddSubgroup.zmultiples ↑n)
  -/
  ext
  rw [Int.mem_zmultiples_iff, AddMonoidHom.mem_ker, Int.coe_castAddHom,
    intCast_zmod_eq_zero_iff_dvd]


@[deprecated (since := "2024-04-17")]
alias ker_int_castAddHom := ker_intCastAddHom


theorem cast_injective_of_le {m n : ℕ} [nzm : NeZero m] (h : m ≤ n) :
    Function.Injective (@cast (ZMod n) _ m) := by
  cases m with
  | zero => cases nzm; simp_all
  | succ m =>
    rintro ⟨x, hx⟩ ⟨y, hy⟩ f
    simp only [cast, val, natCast_eq_natCast_iff',
      Nat.mod_eq_of_lt (hx.trans_le h), Nat.mod_eq_of_lt (hy.trans_le h)] at f
    apply Fin.ext
    exact f


theorem cast_zmod_eq_zero_iff_of_le {m n : ℕ} [NeZero m] (h : m ≤ n) (a : ZMod m) :
    (cast a : ZMod n) = 0 ↔ a = 0 := by
  /-
    m n : Nat
    inst✝ : NeZero m
    h : LE.le m n
    a : ZMod m
    ⊢ Iff (Eq a.cast 0) (Eq a 0)
  -/
  rw [← ZMod.cast_zero (n := m)]
  /-
    m n : Nat
    inst✝ : NeZero m
    h : LE.le m n
    a : ZMod m
    ⊢ Iff (Eq a.cast (ZMod.cast 0)) (Eq a 0)
  -/
  exact Injective.eq_iff' (cast_injective_of_le h) rfl
  /-
    🎉 no goals
  -/

-- Porting note: commented
-- unseal Int.NonNeg


@[simp]
theorem natCast_toNat (p : ℕ) : ∀ {z : ℤ} (_h : 0 ≤ z), (z.toNat : ZMod p) = z
                      /-
                        p n : Nat
                        _h : LE.le 0 ↑n
                        ⊢ Eq ↑(↑n).toNat ↑↑n
                      -/
  | (n : ℕ), _h => by simp only [Int.cast_natCast, Int.toNat_natCast]
                      /-
                        🎉 no goals
                      -/
                           /-
                             p n : Nat
                             h : LE.le 0 (Int.negSucc n)
                             ⊢ Eq ↑(Int.negSucc n).toNat ↑(Int.negSucc n)
                           -/
  | Int.negSucc n, h => by simp at h
                           /-
                             🎉 no goals
                           -/


@[deprecated (since := "2024-04-17")]
alias nat_cast_toNat := natCast_toNat


theorem val_injective (n : ℕ) [NeZero n] : Function.Injective (val : ZMod n → ℕ) := by
  /-
    n : Nat
    inst✝ : NeZero n
    ⊢ Function.Injective ZMod.val
  -/
  cases n
    /-
      case zero
      inst✝ : NeZero 0
      ⊢ Function.Injective ZMod.val
    -/
  · cases NeZero.ne 0 rfl
    /-
      🎉 no goals
    -/
  /-
    case succ
    n✝ : Nat
    inst✝ : NeZero (HAdd.hAdd n✝ 1)
    ⊢ Function.Injective ZMod.val
  -/
  intro a b h
  /-
    case succ
    n✝ : Nat
    inst✝ : NeZero (HAdd.hAdd n✝ 1)
    a b : ZMod (HAdd.hAdd n✝ 1)
    h : Eq a.val b.val
    ⊢ Eq a b
  -/
  dsimp [ZMod]
  /-
    case succ
    n✝ : Nat
    inst✝ : NeZero (HAdd.hAdd n✝ 1)
    a b : ZMod (HAdd.hAdd n✝ 1)
    h : Eq a.val b.val
    ⊢ Eq a b
  -/
  ext
  /-
    case succ.h
    n✝ : Nat
    inst✝ : NeZero (HAdd.hAdd n✝ 1)
    a b : ZMod (HAdd.hAdd n✝ 1)
    h : Eq a.val b.val
    ⊢ Eq ↑a ↑b
  -/
  exact h
  /-
    🎉 no goals
  -/


theorem val_one_eq_one_mod (n : ℕ) : (1 : ZMod n).val = 1 % n := by
  /-
    n : Nat
    ⊢ Eq (ZMod.val 1) (HMod.hMod 1 n)
  -/
  rw [← Nat.cast_one, val_natCast]
  /-
    🎉 no goals
  -/


theorem val_one (n : ℕ) [Fact (1 < n)] : (1 : ZMod n).val = 1 := by
  /-
    n : Nat
    inst✝ : Fact (LT.lt 1 n)
    ⊢ Eq (ZMod.val 1) 1
  -/
  rw [val_one_eq_one_mod]
  /-
    n : Nat
    inst✝ : Fact (LT.lt 1 n)
    ⊢ Eq (HMod.hMod 1 n) 1
  -/
  exact Nat.mod_eq_of_lt Fact.out
  /-
    🎉 no goals
  -/


lemma val_one'' : ∀ {n}, n ≠ 1 → (1 : ZMod n).val = 1
  | 0, _ => rfl
                /-
                  hn : Ne 1 1
                  ⊢ Eq (ZMod.val 1) 1
                -/
  | 1, hn => by cases hn rfl
                /-
                  🎉 no goals
                -/
  | n + 2, _ =>
                                    /-
                                      n : Nat
                                      x✝ : Ne (HAdd.hAdd n 2) 1
                                      ⊢ LT.lt 1 (HAdd.hAdd n 2)
                                    -/
    haveI : Fact (1 < n + 2) := ⟨by simp⟩
                                    /-
                                      🎉 no goals
                                    -/
    ZMod.val_one _


theorem val_add {n : ℕ} [NeZero n] (a b : ZMod n) : (a + b).val = (a.val + b.val) % n := by
  /-
    n : Nat
    inst✝ : NeZero n
    a b : ZMod n
    ⊢ Eq (HAdd.hAdd a b).val (HMod.hMod (HAdd.hAdd a.val b.val) n)
  -/
  cases n
    /-
      case zero
      inst✝ : NeZero 0
      a b : ZMod 0
      ⊢ Eq (HAdd.hAdd a b).val (HMod.hMod (HAdd.hAdd a.val b.val) 0)
    -/
  · cases NeZero.ne 0 rfl
    /-
      🎉 no goals
    -/
    /-
      case succ
      n✝ : Nat
      inst✝ : NeZero (HAdd.hAdd n✝ 1)
      a b : ZMod (HAdd.hAdd n✝ 1)
      ⊢ Eq (HAdd.hAdd a b).val (HMod.hMod (HAdd.hAdd a.val b.val) (HAdd.hAdd n✝ 1))
    -/
  · apply Fin.val_add
    /-
      🎉 no goals
    -/


theorem val_add_of_lt {n : ℕ} {a b : ZMod n} (h : a.val + b.val < n) :
    (a + b).val = a.val + b.val := by
  /-
    n : Nat
    a b : ZMod n
    h : LT.lt (HAdd.hAdd a.val b.val) n
    ⊢ Eq (HAdd.hAdd a b).val (HAdd.hAdd a.val b.val)
  -/
  have : NeZero n := by constructor; rintro rfl; simp at h
  /-
    n : Nat
    a b : ZMod n
    h : LT.lt (HAdd.hAdd a.val b.val) n
    this : NeZero n
    ⊢ Eq (HAdd.hAdd a b).val (HAdd.hAdd a.val b.val)
  -/
  rw [ZMod.val_add, Nat.mod_eq_of_lt h]
  /-
    🎉 no goals
  -/


theorem val_add_val_of_le {n : ℕ} [NeZero n] {a b : ZMod n} (h : n ≤ a.val + b.val) :
    a.val + b.val = (a + b).val + n := by
  rw [val_add, Nat.add_mod_add_of_le_add_mod, Nat.mod_eq_of_lt (val_lt _),
    Nat.mod_eq_of_lt (val_lt _)]
  /-
    n : Nat
    inst✝ : NeZero n
    a b : ZMod n
    h : LE.le n (HAdd.hAdd a.val b.val)
    ⊢ LE.le n (HAdd.hAdd (HMod.hMod a.val n) (HMod.hMod b.val n))
  -/
  rwa [Nat.mod_eq_of_lt (val_lt _), Nat.mod_eq_of_lt (val_lt _)]
  /-
    🎉 no goals
  -/


theorem val_add_of_le {n : ℕ} [NeZero n] {a b : ZMod n} (h : n ≤ a.val + b.val) :
    (a + b).val = a.val + b.val - n := by
  /-
    n : Nat
    inst✝ : NeZero n
    a b : ZMod n
    h : LE.le n (HAdd.hAdd a.val b.val)
    ⊢ Eq (HAdd.hAdd a b).val (HSub.hSub (HAdd.hAdd a.val b.val) n)
  -/
  rw [val_add_val_of_le h]
  /-
    n : Nat
    inst✝ : NeZero n
    a b : ZMod n
    h : LE.le n (HAdd.hAdd a.val b.val)
    ⊢ Eq (HAdd.hAdd a b).val (HSub.hSub (HAdd.hAdd (HAdd.hAdd a b).val n) n)
  -/
  exact eq_tsub_of_add_eq rfl
  /-
    🎉 no goals
  -/


theorem val_add_le {n : ℕ} (a b : ZMod n) : (a + b).val ≤ a.val + b.val := by
  /-
    n : Nat
    a b : ZMod n
    ⊢ LE.le (HAdd.hAdd a b).val (HAdd.hAdd a.val b.val)
  -/
  cases n
    /-
      case zero
      a b : ZMod 0
      ⊢ LE.le (HAdd.hAdd a b).val (HAdd.hAdd a.val b.val)
    -/
  · simpa [ZMod.val] using Int.natAbs_add_le _ _
    /-
      🎉 no goals
    -/
    /-
      case succ
      n✝ : Nat
      a b : ZMod (HAdd.hAdd n✝ 1)
      ⊢ LE.le (HAdd.hAdd a b).val (HAdd.hAdd a.val b.val)
    -/
  · simpa [ZMod.val_add] using Nat.mod_le _ _
    /-
      🎉 no goals
    -/


theorem val_mul {n : ℕ} (a b : ZMod n) : (a * b).val = a.val * b.val % n := by
  /-
    n : Nat
    a b : ZMod n
    ⊢ Eq (HMul.hMul a b).val (HMod.hMod (HMul.hMul a.val b.val) n)
  -/
  cases n
    /-
      case zero
      a b : ZMod 0
      ⊢ Eq (HMul.hMul a b).val (HMod.hMod (HMul.hMul a.val b.val) 0)
    -/
  · rw [Nat.mod_zero]
    /-
      case zero
      a b : ZMod 0
      ⊢ Eq (HMul.hMul a b).val (HMul.hMul a.val b.val)
    -/
    apply Int.natAbs_mul
    /-
      🎉 no goals
    -/
    /-
      case succ
      n✝ : Nat
      a b : ZMod (HAdd.hAdd n✝ 1)
      ⊢ Eq (HMul.hMul a b).val (HMod.hMod (HMul.hMul a.val b.val) (HAdd.hAdd n✝ 1))
    -/
  · apply Fin.val_mul
    /-
      🎉 no goals
    -/


theorem val_mul_le {n : ℕ} (a b : ZMod n) : (a * b).val ≤ a.val * b.val := by
  /-
    n : Nat
    a b : ZMod n
    ⊢ LE.le (HMul.hMul a b).val (HMul.hMul a.val b.val)
  -/
  rw [val_mul]
  /-
    n : Nat
    a b : ZMod n
    ⊢ LE.le (HMod.hMod (HMul.hMul a.val b.val) n) (HMul.hMul a.val b.val)
  -/
  apply Nat.mod_le
  /-
    🎉 no goals
  -/


theorem val_mul_of_lt {n : ℕ} {a b : ZMod n} (h : a.val * b.val < n) :
    (a * b).val = a.val * b.val := by
  /-
    n : Nat
    a b : ZMod n
    h : LT.lt (HMul.hMul a.val b.val) n
    ⊢ Eq (HMul.hMul a b).val (HMul.hMul a.val b.val)
  -/
  rw [val_mul]
  /-
    n : Nat
    a b : ZMod n
    h : LT.lt (HMul.hMul a.val b.val) n
    ⊢ Eq (HMod.hMod (HMul.hMul a.val b.val) n) (HMul.hMul a.val b.val)
  -/
  apply Nat.mod_eq_of_lt h
  /-
    🎉 no goals
  -/


theorem val_mul_iff_lt {n : ℕ} [NeZero n] (a b : ZMod n) :
    (a * b).val = a.val * b.val ↔ a.val * b.val < n := by
  /-
    n : Nat
    inst✝ : NeZero n
    a b : ZMod n
    ⊢ Iff (Eq (HMul.hMul a b).val (HMul.hMul a.val b.val)) (LT.lt (HMul.hMul a.val …
  -/
  constructor <;> intro h
    /-
      case mp
      n : Nat
      inst✝ : NeZero n
      a b : ZMod n
      h : Eq (HMul.hMul a b).val (HMul.hMul a.val b.val)
      ⊢ LT.lt (HMul.hMul a.val b.val) n
    -/
  · rw [← h]; apply ZMod.val_lt
              /-
                🎉 no goals
              -/
    /-
      case mpr
      n : Nat
      inst✝ : NeZero n
      a b : ZMod n
      h : LT.lt (HMul.hMul a.val b.val) n
      ⊢ Eq (HMul.hMul a b).val (HMul.hMul a.val b.val)
    -/
  · apply ZMod.val_mul_of_lt h
    /-
      🎉 no goals
    -/


instance nontrivial (n : ℕ) [Fact (1 < n)] : Nontrivial (ZMod n) :=
  ⟨⟨0, 1, fun h =>
      zero_ne_one <|
        calc
                                     /-
                                       m n✝ n : Nat
                                       inst✝ : Fact (LT.lt 1 n)
                                       h : Eq 0 1
                                       ⊢ Eq 0 (ZMod.val 0)
                                     -/
          0 = (0 : ZMod n).val := by rw [val_zero]
                                     /-
                                       🎉 no goals
                                     -/
          _ = (1 : ZMod n).val := congr_arg ZMod.val h
          _ = 1 := val_one n
          ⟩⟩


instance nontrivial' : Nontrivial (ZMod 0) := by
  /-
    m n : Nat
    ⊢ Nontrivial (ZMod 0)
  -/
  delta ZMod; infer_instance
              /-
                🎉 no goals
              -/


/-- The inversion on `ZMod n`.
It is setup in such a way that `a * a⁻¹` is equal to `gcd a.val n`.
In particular, if `a` is coprime to `n`, and hence a unit, `a * a⁻¹ = 1`. -/
def inv : ∀ n : ℕ, ZMod n → ZMod n
  | 0, i => Int.sign i
  | n + 1, i => Nat.gcdA i.val (n + 1)


instance (n : ℕ) : Inv (ZMod n) :=
  ⟨inv n⟩


theorem inv_zero : ∀ n : ℕ, (0 : ZMod n)⁻¹ = 0
  | 0 => Int.sign_zero
  | n + 1 =>
    show (Nat.gcdA _ (n + 1) : ZMod (n + 1)) = 0 by
      /-
        n : Nat
        ⊢ Eq (↑((ZMod.val 0).gcdA (HAdd.hAdd n 1))) 0
      -/
      rw [val_zero]
      /-
        n : Nat
        ⊢ Eq (↑(Nat.gcdA 0 (HAdd.hAdd n 1))) 0
      -/
      unfold Nat.gcdA Nat.xgcd Nat.xgcdAux
      /-
        n : Nat
        ⊢ Eq (↑{ fst := HAdd.hAdd n 1, snd := { fst := 0, snd := 1 } }.2.1) 0
      -/
      rfl
      /-
        🎉 no goals
      -/


theorem mul_inv_eq_gcd {n : ℕ} (a : ZMod n) : a * a⁻¹ = Nat.gcd a.val n := by
  /-
    n : Nat
    a : ZMod n
    ⊢ Eq (HMul.hMul a (Inv.inv a)) ↑(a.val.gcd n)
  -/
  cases' n with n
    /-
      case zero
      a : ZMod 0
      ⊢ Eq (HMul.hMul a (Inv.inv a)) ↑(a.val.gcd 0)
    -/
  · dsimp [ZMod] at a ⊢
    calc
      _ = a * Int.sign a := rfl
      _ = a.natAbs := by rw [Int.mul_sign]
      _ = a.natAbs.gcd 0 := by rw [Nat.gcd_zero_right]
  · calc
      a * a⁻¹ = a * a⁻¹ + n.succ * Nat.gcdB (val a) n.succ := by
        rw [natCast_self, zero_mul, add_zero]
      _ = ↑(↑a.val * Nat.gcdA (val a) n.succ + n.succ * Nat.gcdB (val a) n.succ) := by
        push_cast
        rw [natCast_zmod_val]
        rfl
      _ = Nat.gcd a.val n.succ := by rw [← Nat.gcd_eq_gcd_ab a.val n.succ]; rfl


@[simp] protected lemma inv_one (n : ℕ) : (1⁻¹ : ZMod n) = 1 := by
  /-
    n : Nat
    ⊢ Eq (Inv.inv 1) 1
  -/
  obtain rfl | hn := eq_or_ne n 1
    /-
      case inl
      ⊢ Eq (Inv.inv 1) 1
    -/
  · exact Subsingleton.elim _ _
    /-
      🎉 no goals
    -/
    /-
      case inr
      n : Nat
      hn : Ne n 1
      ⊢ Eq (Inv.inv 1) 1
    -/
  · simpa [ZMod.val_one'' hn] using mul_inv_eq_gcd (1 : ZMod n)
    /-
      🎉 no goals
    -/


@[simp]
theorem natCast_mod (a : ℕ) (n : ℕ) : ((a % n : ℕ) : ZMod n) = a := by
  conv =>
      rhs
      rw [← Nat.mod_add_div a n]
  /-
    a n : Nat
    ⊢ Eq ↑(HMod.hMod a n) ↑(HAdd.hAdd (HMod.hMod a n) (HMul.hMul n (HDiv.hDiv a n)))
  -/
  simp
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias nat_cast_mod := natCast_mod


theorem eq_iff_modEq_nat (n : ℕ) {a b : ℕ} : (a : ZMod n) = b ↔ a ≡ b [MOD n] := by
  /-
    n a b : Nat
    ⊢ Iff (Eq ↑a ↑b) (n.ModEq a b)
  -/
  cases n
    /-
      case zero
      a b : Nat
      ⊢ Iff (Eq ↑a ↑b) (Nat.ModEq 0 a b)
    -/
  · simp [Nat.ModEq, Int.natCast_inj, Nat.mod_zero]
    /-
      🎉 no goals
    -/
    /-
      case succ
      a b n✝ : Nat
      ⊢ Iff (Eq ↑a ↑b) ((HAdd.hAdd n✝ 1).ModEq a b)
    -/
  · rw [Fin.ext_iff, Nat.ModEq, ← val_natCast, ← val_natCast]
    /-
      case succ
      a b n✝ : Nat
      ⊢ Iff (Eq ↑↑a ↑↑b) (Eq (↑a).val (↑b).val)
    -/
    exact Iff.rfl
    /-
      🎉 no goals
    -/


theorem eq_zero_iff_even {n : ℕ} : (n : ZMod 2) = 0 ↔ Even n :=
  (CharP.cast_eq_zero_iff (ZMod 2) 2 n).trans even_iff_two_dvd.symm


theorem eq_one_iff_odd {n : ℕ} : (n : ZMod 2) = 1 ↔ Odd n := by
  /-
    n : Nat
    ⊢ Iff (Eq (↑n) 1) (Odd n)
  -/
  rw [← @Nat.cast_one (ZMod 2), ZMod.eq_iff_modEq_nat, Nat.odd_iff, Nat.ModEq]
  /-
    🎉 no goals
  -/


theorem ne_zero_iff_odd {n : ℕ} : (n : ZMod 2) ≠ 0 ↔ Odd n := by
  /-
    n : Nat
    ⊢ Iff (Ne (↑n) 0) (Odd n)
  -/
  constructor <;>
      /-
        case mp
        n : Nat
        ⊢ Ne (↑n) 0 → Odd n
      -/
      /-
        case mp
        n : Nat
        ⊢ Not (Odd n) → Not (Ne (↑n) 0)
      -/
      /-
        🎉 no goals
      -/
      /-
        case mpr
        n : Nat
        ⊢ Not (Ne (↑n) 0) → Not (Odd n)
      -/
      simp [eq_zero_iff_even]
      /-
        🎉 no goals
      -/


theorem coe_mul_inv_eq_one {n : ℕ} (x : ℕ) (h : Nat.Coprime x n) :
    ((x : ZMod n) * (x : ZMod n)⁻¹) = 1 := by
  /-
    n x : Nat
    h : x.Coprime n
    ⊢ Eq (HMul.hMul (↑x) (Inv.inv ↑x)) 1
  -/
  rw [Nat.Coprime, Nat.gcd_comm, Nat.gcd_rec] at h
  /-
    n x : Nat
    h : Eq ((HMod.hMod x n).gcd n) 1
    ⊢ Eq (HMul.hMul (↑x) (Inv.inv ↑x)) 1
  -/
  rw [mul_inv_eq_gcd, val_natCast, h, Nat.cast_one]
  /-
    🎉 no goals
  -/


lemma mul_val_inv (hmn : m.Coprime n) : (m * (m⁻¹ : ZMod n).val : ZMod n) = 1 := by
  /-
    m n : Nat
    hmn : m.Coprime n
    ⊢ Eq (HMul.hMul ↑m ↑(Inv.inv ↑m).val) 1
  -/
  obtain rfl | hn := eq_or_ne n 0
    /-
      case inl
      m : Nat
      hmn : m.Coprime 0
      ⊢ Eq (HMul.hMul ↑m ↑(Inv.inv ↑m).val) 1
    -/
  · simp [m.coprime_zero_right.1 hmn]
    /-
      🎉 no goals
    -/
  /-
    case inr
    m n : Nat
    hmn : m.Coprime n
    hn : Ne n 0
    ⊢ Eq (HMul.hMul ↑m ↑(Inv.inv ↑m).val) 1
  -/
  haveI : NeZero n := ⟨hn⟩
  /-
    case inr
    m n : Nat
    hmn : m.Coprime n
    hn : Ne n 0
    this : NeZero n
    ⊢ Eq (HMul.hMul ↑m ↑(Inv.inv ↑m).val) 1
  -/
  rw [ZMod.natCast_zmod_val, ZMod.coe_mul_inv_eq_one _ hmn]
  /-
    🎉 no goals
  -/


lemma val_inv_mul (hmn : m.Coprime n) : ((m⁻¹ : ZMod n).val * m : ZMod n) = 1 := by
  /-
    m n : Nat
    hmn : m.Coprime n
    ⊢ Eq (HMul.hMul ↑(Inv.inv ↑m).val ↑m) 1
  -/
  rw [mul_comm, mul_val_inv hmn]
  /-
    🎉 no goals
  -/


/-- `unitOfCoprime` makes an element of `(ZMod n)ˣ` given
  a natural number `x` and a proof that `x` is coprime to `n`  -/
def unitOfCoprime {n : ℕ} (x : ℕ) (h : Nat.Coprime x n) : (ZMod n)ˣ :=
                                      /-
                                        m n✝ n x : Nat
                                        h : x.Coprime n
                                        ⊢ Eq (HMul.hMul (Inv.inv ↑x) ↑x) 1
                                      -/
  ⟨x, x⁻¹, coe_mul_inv_eq_one x h, by rw [mul_comm, coe_mul_inv_eq_one x h]⟩
                                      /-
                                        🎉 no goals
                                      -/


@[simp]
theorem coe_unitOfCoprime {n : ℕ} (x : ℕ) (h : Nat.Coprime x n) :
    (unitOfCoprime x h : ZMod n) = x :=
  rfl


theorem val_coe_unit_coprime {n : ℕ} (u : (ZMod n)ˣ) : Nat.Coprime (u : ZMod n).val n := by
  /-
    n : Nat
    u : Units (ZMod n)
    ⊢ (↑u).val.Coprime n
  -/
  cases' n with n
    /-
      case zero
      u : Units (ZMod 0)
      ⊢ (↑u).val.Coprime 0
    -/
                                                      /-
                                                        🎉 no goals
                                                      -/
  · rcases Int.units_eq_one_or u with (rfl | rfl) <;> simp
                                                      /-
                                                        🎉 no goals
                                                      -/
  /-
    case succ
    n : Nat
    u : Units (ZMod (HAdd.hAdd n 1))
    ⊢ (↑u).val.Coprime (HAdd.hAdd n 1)
  -/
  apply Nat.coprime_of_mul_modEq_one ((u⁻¹ : Units (ZMod (n + 1))) : ZMod (n + 1)).val
  /-
    case succ
    n : Nat
    u : Units (ZMod (HAdd.hAdd n 1))
    ⊢ (HAdd.hAdd n 1).ModEq (HMul.hMul (↑u).val (↑(Inv.inv u)).val) 1
  -/
  have := Units.ext_iff.1 (mul_inv_cancel u)
  /-
    case succ
    n : Nat
    u : Units (ZMod (HAdd.hAdd n 1))
    this : Eq ↑(HMul.hMul u (Inv.inv u)) ↑1
    ⊢ (HAdd.hAdd n 1).ModEq (HMul.hMul (↑u).val (↑(Inv.inv u)).val) 1
  -/
  rw [Units.val_one] at this
  /-
    case succ
    n : Nat
    u : Units (ZMod (HAdd.hAdd n 1))
    this : Eq (↑(HMul.hMul u (Inv.inv u))) 1
    ⊢ (HAdd.hAdd n 1).ModEq (HMul.hMul (↑u).val (↑(Inv.inv u)).val) 1
  -/
  rw [← eq_iff_modEq_nat, Nat.cast_one, ← this]; clear this
  /-
    case succ
    n : Nat
    u : Units (ZMod (HAdd.hAdd n 1))
    ⊢ Eq ↑(HMul.hMul (↑u).val (↑(Inv.inv u)).val) ↑(HMul.hMul u (Inv.inv u))
  -/
  rw [← natCast_zmod_val ((u * u⁻¹ : Units (ZMod (n + 1))) : ZMod (n + 1))]
  /-
    case succ
    n : Nat
    u : Units (ZMod (HAdd.hAdd n 1))
    ⊢ Eq ↑(HMul.hMul (↑u).val (↑(Inv.inv u)).val) ↑(↑(HMul.hMul u (Inv.inv u))).val
  -/
  rw [Units.val_mul, val_mul, natCast_mod]
  /-
    🎉 no goals
  -/


lemma isUnit_iff_coprime (m n : ℕ) : IsUnit (m : ZMod n) ↔ m.Coprime n := by
  /-
    m n : Nat
    ⊢ Iff (IsUnit ↑m) (m.Coprime n)
  -/
  refine ⟨fun H ↦ ?_, fun H ↦ (unitOfCoprime m H).isUnit⟩
  /-
    m n : Nat
    H : IsUnit ↑m
    ⊢ m.Coprime n
  -/
  have H' := val_coe_unit_coprime H.unit
  /-
    m n : Nat
    H : IsUnit ↑m
    H' : (↑H.unit).val.Coprime n
    ⊢ m.Coprime n
  -/
  rw [IsUnit.unit_spec, val_natCast m, Nat.coprime_iff_gcd_eq_one] at H'
  /-
    m n : Nat
    H : IsUnit ↑m
    H' : Eq ((HMod.hMod m n).gcd n) 1
    ⊢ m.Coprime n
  -/
  rw [Nat.coprime_iff_gcd_eq_one, Nat.gcd_comm, ← H']
  /-
    m n : Nat
    H : IsUnit ↑m
    H' : Eq ((HMod.hMod m n).gcd n) 1
    ⊢ Eq (n.gcd m) ((HMod.hMod m n).gcd n)
  -/
  exact Nat.gcd_rec n m
  /-
    🎉 no goals
  -/


lemma isUnit_prime_iff_not_dvd {n p : ℕ} (hp : p.Prime) : IsUnit (p : ZMod n) ↔ ¬p ∣ n := by
  /-
    n p : Nat
    hp : Nat.Prime p
    ⊢ Iff (IsUnit ↑p) (Not (Dvd.dvd p n))
  -/
  rw [isUnit_iff_coprime, Nat.Prime.coprime_iff_not_dvd hp]
  /-
    🎉 no goals
  -/


lemma isUnit_prime_of_not_dvd {n p : ℕ} (hp : p.Prime) (h : ¬ p ∣ n) : IsUnit (p : ZMod n) :=
  (isUnit_prime_iff_not_dvd hp).mpr h


@[simp]
theorem inv_coe_unit {n : ℕ} (u : (ZMod n)ˣ) : (u : ZMod n)⁻¹ = (u⁻¹ : (ZMod n)ˣ) := by
  /-
    n : Nat
    u : Units (ZMod n)
    ⊢ Eq (Inv.inv ↑u) ↑(Inv.inv u)
  -/
  have := congr_arg ((↑) : ℕ → ZMod n) (val_coe_unit_coprime u)
  /-
    n : Nat
    u : Units (ZMod n)
    this : Eq ↑((↑u).val.gcd n) ↑1
    ⊢ Eq (Inv.inv ↑u) ↑(Inv.inv u)
  -/
  rw [← mul_inv_eq_gcd, Nat.cast_one] at this
  /-
    n : Nat
    u : Units (ZMod n)
    this : Eq (HMul.hMul (↑u) (Inv.inv ↑u)) 1
    ⊢ Eq (Inv.inv ↑u) ↑(Inv.inv u)
  -/
  let u' : (ZMod n)ˣ := ⟨u, (u : ZMod n)⁻¹, this, by rwa [mul_comm]⟩
  have h : u = u' := by
    apply Units.ext
    rfl
  /-
    n : Nat
    u : Units (ZMod n)
    this : Eq (HMul.hMul (↑u) (Inv.inv ↑u)) 1
    u' : Units (ZMod n) := { val := ↑u, inv := Inv.inv ↑u, val_inv := this, inv_va …
    h : Eq u u'
    ⊢ Eq (Inv.inv ↑u) ↑(Inv.inv u)
  -/
  rw [h]
  /-
    n : Nat
    u : Units (ZMod n)
    this : Eq (HMul.hMul (↑u) (Inv.inv ↑u)) 1
    u' : Units (ZMod n) := { val := ↑u, inv := Inv.inv ↑u, val_inv := this, inv_va …
    h : Eq u u'
    ⊢ Eq (Inv.inv ↑u') ↑(Inv.inv u')
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem mul_inv_of_unit {n : ℕ} (a : ZMod n) (h : IsUnit a) : a * a⁻¹ = 1 := by
  /-
    n : Nat
    a : ZMod n
    h : IsUnit a
    ⊢ Eq (HMul.hMul a (Inv.inv a)) 1
  -/
  rcases h with ⟨u, rfl⟩
  /-
    case intro
    n : Nat
    u : Units (ZMod n)
    ⊢ Eq (HMul.hMul (↑u) (Inv.inv ↑u)) 1
  -/
  rw [inv_coe_unit, u.mul_inv]
  /-
    🎉 no goals
  -/


theorem inv_mul_of_unit {n : ℕ} (a : ZMod n) (h : IsUnit a) : a⁻¹ * a = 1 := by
  /-
    n : Nat
    a : ZMod n
    h : IsUnit a
    ⊢ Eq (HMul.hMul (Inv.inv a) a) 1
  -/
  rw [mul_comm, mul_inv_of_unit a h]
  /-
    🎉 no goals
  -/

-- TODO: If we changed `⁻¹` so that `ZMod n` is always a `DivisionMonoid`,
-- then we could use the general lemma `inv_eq_of_mul_eq_one`

protected theorem inv_eq_of_mul_eq_one (n : ℕ) (a b : ZMod n) (h : a * b = 1) : a⁻¹ = b :=
  left_inv_eq_right_inv (inv_mul_of_unit a ⟨⟨a, b, h, mul_comm a b ▸ h⟩, rfl⟩) h


lemma inv_mul_eq_one_of_isUnit {n : ℕ} {a : ZMod n} (ha : IsUnit a) (b : ZMod n) :
    a⁻¹ * b = 1 ↔ a = b := by
  -- ideally, this would be `ha.inv_mul_eq_one`, but `ZMod n` is not a `DivisionMonoid`...
  -- (see the "TODO" above)
  /-
    n : Nat
    a : ZMod n
    ha : IsUnit a
    b : ZMod n
    ⊢ Iff (Eq (HMul.hMul (Inv.inv a) b) 1) (Eq a b)
  -/
  refine ⟨fun H ↦ ?_, fun H ↦ H ▸ a.inv_mul_of_unit ha⟩
  /-
    n : Nat
    a : ZMod n
    ha : IsUnit a
    b : ZMod n
    H : Eq (HMul.hMul (Inv.inv a) b) 1
    ⊢ Eq a b
  -/
  apply_fun (a * ·) at H
  /-
    n : Nat
    a : ZMod n
    ha : IsUnit a
    b : ZMod n
    H : Eq (HMul.hMul a (HMul.hMul (Inv.inv a) b)) (HMul.hMul a 1)
    ⊢ Eq a b
  -/
  rwa [← mul_assoc, a.mul_inv_of_unit ha, one_mul, mul_one, eq_comm] at H
  /-
    🎉 no goals
  -/

-- TODO: this equivalence is true for `ZMod 0 = ℤ`, but needs to use different functions.

/-- Equivalence between the units of `ZMod n` and
the subtype of terms `x : ZMod n` for which `x.val` is coprime to `n` -/
def unitsEquivCoprime {n : ℕ} [NeZero n] : (ZMod n)ˣ ≃ { x : ZMod n // Nat.Coprime x.val n } where
  toFun x := ⟨x, val_coe_unit_coprime x⟩
  invFun x := unitOfCoprime x.1.val x.2
  left_inv := fun ⟨_, _, _, _⟩ => Units.ext (natCast_zmod_val _)
                                /-
                                  m n✝ n : Nat
                                  inst✝ : NeZero n
                                  x✝ : Subtype fun x => x.val.Coprime n
                                  val✝ : ZMod n
                                  property✝ : val✝.val.Coprime n
                                  ⊢ Eq ((fun x => ⟨↑x, ⋯⟩) ((fun x => ZMod.unitOfCoprime (↑x).val ⋯) ⟨val✝, prop …
                                -/
  right_inv := fun ⟨_, _⟩ => by simp
                                /-
                                  🎉 no goals
                                -/


/-- The **Chinese remainder theorem**. For a pair of coprime natural numbers, `m` and `n`,
  the rings `ZMod (m * n)` and `ZMod m × ZMod n` are isomorphic.

See `Ideal.quotientInfRingEquivPiQuotient` for the Chinese remainder theorem for ideals in any
ring.
-/
def chineseRemainder {m n : ℕ} (h : m.Coprime n) : ZMod (m * n) ≃+* ZMod m × ZMod n :=
  let to_fun : ZMod (m * n) → ZMod m × ZMod n :=
                                          /-
                                            m✝ n✝ m n : Nat
                                            h : m.Coprime n
                                            ⊢ Dvd.dvd (m.lcm n) (HMul.hMul m n)
                                          -/
    ZMod.castHom (show m.lcm n ∣ m * n by simp [Nat.lcm_dvd_iff]) (ZMod m × ZMod n)
                                          /-
                                            🎉 no goals
                                          -/
  let inv_fun : ZMod m × ZMod n → ZMod (m * n) := fun x =>
    if m * n = 0 then
      if m = 1 then cast (RingHom.snd _ (ZMod n) x) else cast (RingHom.fst (ZMod m) _ x)
    else Nat.chineseRemainder h x.1.val x.2.val
  have inv : Function.LeftInverse inv_fun to_fun ∧ Function.RightInverse inv_fun to_fun :=
    if hmn0 : m * n = 0 then by
      /-
        m✝ n✝ m n : Nat
        h : m.Coprime n
        to_fun : ZMod (HMul.hMul m n) → Prod (ZMod m) (ZMod n) := ⇑(ZMod.castHom ⋯ (Pr …
        inv_fun : Prod (ZMod m) (ZMod n) → ZMod (HMul.hMul m n) := fun x => ite (Eq (H …
        hmn0 : Eq (HMul.hMul m n) 0
        ⊢ And (Function.LeftInverse inv_fun to_fun) (Function.RightInverse inv_fun to_ …
      -/
      rcases h.eq_of_mul_eq_zero hmn0 with (⟨rfl, rfl⟩ | ⟨rfl, rfl⟩)
        /-
          case inl.intro
          m n : Nat
          h : Nat.Coprime 0 1
          to_fun : ZMod (HMul.hMul 0 1) → Prod (ZMod 0) (ZMod 1) := ⇑(ZMod.castHom ⋯ (Pr …
          inv_fun : Prod (ZMod 0) (ZMod 1) → ZMod (HMul.hMul 0 1) := fun x => ite (Eq (H …
          hmn0 : Eq (HMul.hMul 0 1) 0
          ⊢ And (Function.LeftInverse inv_fun to_fun) (Function.RightInverse inv_fun to_ …
        -/
      · constructor
          /-
            case inl.intro.left
            m n : Nat
            h : Nat.Coprime 0 1
            to_fun : ZMod (HMul.hMul 0 1) → Prod (ZMod 0) (ZMod 1) := ⇑(ZMod.castHom ⋯ (Pr …
            inv_fun : Prod (ZMod 0) (ZMod 1) → ZMod (HMul.hMul 0 1) := fun x => ite (Eq (H …
            hmn0 : Eq (HMul.hMul 0 1) 0
            ⊢ Function.LeftInverse inv_fun to_fun
          -/
        · intro x; rfl
                   /-
                     🎉 no goals
                   -/
          /-
            case inl.intro.right
            m n : Nat
            h : Nat.Coprime 0 1
            to_fun : ZMod (HMul.hMul 0 1) → Prod (ZMod 0) (ZMod 1) := ⇑(ZMod.castHom ⋯ (Pr …
            inv_fun : Prod (ZMod 0) (ZMod 1) → ZMod (HMul.hMul 0 1) := fun x => ite (Eq (H …
            hmn0 : Eq (HMul.hMul 0 1) 0
            ⊢ Function.RightInverse inv_fun to_fun
          -/
        · rintro ⟨x, y⟩
          /-
            case inl.intro.right.mk
            m n : Nat
            h : Nat.Coprime 0 1
            to_fun : ZMod (HMul.hMul 0 1) → Prod (ZMod 0) (ZMod 1) := ⇑(ZMod.castHom ⋯ (Pr …
            inv_fun : Prod (ZMod 0) (ZMod 1) → ZMod (HMul.hMul 0 1) := fun x => ite (Eq (H …
            hmn0 : Eq (HMul.hMul 0 1) 0
            x : ZMod 0
            y : ZMod 1
            ⊢ Eq (to_fun (inv_fun { fst := x, snd := y })) { fst := x, snd := y }
          -/
          fin_cases y
          /-
            case inl.intro.right.mk.«0»
            m n : Nat
            h : Nat.Coprime 0 1
            to_fun : ZMod (HMul.hMul 0 1) → Prod (ZMod 0) (ZMod 1) := ⇑(ZMod.castHom ⋯ (Pr …
            inv_fun : Prod (ZMod 0) (ZMod 1) → ZMod (HMul.hMul 0 1) := fun x => ite (Eq (H …
            hmn0 : Eq (HMul.hMul 0 1) 0
            x : ZMod 0
            ⊢ Eq (to_fun (inv_fun { fst := x, snd := (fun i => i) ⟨0, ⋯⟩ })) { fst := x, s …
          -/
          simp [to_fun, inv_fun, castHom, Prod.ext_iff, eq_iff_true_of_subsingleton]
          /-
            🎉 no goals
          -/
        /-
          case inr.intro
          m n : Nat
          h : Nat.Coprime 1 0
          to_fun : ZMod (HMul.hMul 1 0) → Prod (ZMod 1) (ZMod 0) := ⇑(ZMod.castHom ⋯ (Pr …
          inv_fun : Prod (ZMod 1) (ZMod 0) → ZMod (HMul.hMul 1 0) := fun x => ite (Eq (H …
          hmn0 : Eq (HMul.hMul 1 0) 0
          ⊢ And (Function.LeftInverse inv_fun to_fun) (Function.RightInverse inv_fun to_ …
        -/
      · constructor
          /-
            case inr.intro.left
            m n : Nat
            h : Nat.Coprime 1 0
            to_fun : ZMod (HMul.hMul 1 0) → Prod (ZMod 1) (ZMod 0) := ⇑(ZMod.castHom ⋯ (Pr …
            inv_fun : Prod (ZMod 1) (ZMod 0) → ZMod (HMul.hMul 1 0) := fun x => ite (Eq (H …
            hmn0 : Eq (HMul.hMul 1 0) 0
            ⊢ Function.LeftInverse inv_fun to_fun
          -/
        · intro x; rfl
                   /-
                     🎉 no goals
                   -/
          /-
            case inr.intro.right
            m n : Nat
            h : Nat.Coprime 1 0
            to_fun : ZMod (HMul.hMul 1 0) → Prod (ZMod 1) (ZMod 0) := ⇑(ZMod.castHom ⋯ (Pr …
            inv_fun : Prod (ZMod 1) (ZMod 0) → ZMod (HMul.hMul 1 0) := fun x => ite (Eq (H …
            hmn0 : Eq (HMul.hMul 1 0) 0
            ⊢ Function.RightInverse inv_fun to_fun
          -/
        · rintro ⟨x, y⟩
          /-
            case inr.intro.right.mk
            m n : Nat
            h : Nat.Coprime 1 0
            to_fun : ZMod (HMul.hMul 1 0) → Prod (ZMod 1) (ZMod 0) := ⇑(ZMod.castHom ⋯ (Pr …
            inv_fun : Prod (ZMod 1) (ZMod 0) → ZMod (HMul.hMul 1 0) := fun x => ite (Eq (H …
            hmn0 : Eq (HMul.hMul 1 0) 0
            x : ZMod 1
            y : ZMod 0
            ⊢ Eq (to_fun (inv_fun { fst := x, snd := y })) { fst := x, snd := y }
          -/
          fin_cases x
          /-
            case inr.intro.right.mk.«0»
            m n : Nat
            h : Nat.Coprime 1 0
            to_fun : ZMod (HMul.hMul 1 0) → Prod (ZMod 1) (ZMod 0) := ⇑(ZMod.castHom ⋯ (Pr …
            inv_fun : Prod (ZMod 1) (ZMod 0) → ZMod (HMul.hMul 1 0) := fun x => ite (Eq (H …
            hmn0 : Eq (HMul.hMul 1 0) 0
            y : ZMod 0
            ⊢ Eq (to_fun (inv_fun { fst := (fun i => i) ⟨0, ⋯⟩, snd := y })) { fst := (fun …
          -/
          simp [to_fun, inv_fun, castHom, Prod.ext_iff, eq_iff_true_of_subsingleton]
          /-
            🎉 no goals
          -/
    else by
      /-
        m✝ n✝ m n : Nat
        h : m.Coprime n
        to_fun : ZMod (HMul.hMul m n) → Prod (ZMod m) (ZMod n) := ⇑(ZMod.castHom ⋯ (Pr …
        inv_fun : Prod (ZMod m) (ZMod n) → ZMod (HMul.hMul m n) := fun x => ite (Eq (H …
        hmn0 : Not (Eq (HMul.hMul m n) 0)
        ⊢ And (Function.LeftInverse inv_fun to_fun) (Function.RightInverse inv_fun to_ …
      -/
      haveI : NeZero (m * n) := ⟨hmn0⟩
      /-
        m✝ n✝ m n : Nat
        h : m.Coprime n
        to_fun : ZMod (HMul.hMul m n) → Prod (ZMod m) (ZMod n) := ⇑(ZMod.castHom ⋯ (Pr …
        inv_fun : Prod (ZMod m) (ZMod n) → ZMod (HMul.hMul m n) := fun x => ite (Eq (H …
        hmn0 : Not (Eq (HMul.hMul m n) 0)
        this : NeZero (HMul.hMul m n)
        ⊢ And (Function.LeftInverse inv_fun to_fun) (Function.RightInverse inv_fun to_ …
      -/
      haveI : NeZero m := ⟨left_ne_zero_of_mul hmn0⟩
      /-
        m✝ n✝ m n : Nat
        h : m.Coprime n
        to_fun : ZMod (HMul.hMul m n) → Prod (ZMod m) (ZMod n) := ⇑(ZMod.castHom ⋯ (Pr …
        inv_fun : Prod (ZMod m) (ZMod n) → ZMod (HMul.hMul m n) := fun x => ite (Eq (H …
        hmn0 : Not (Eq (HMul.hMul m n) 0)
        this✝ : NeZero (HMul.hMul m n)
        this : NeZero m
        ⊢ And (Function.LeftInverse inv_fun to_fun) (Function.RightInverse inv_fun to_ …
      -/
      haveI : NeZero n := ⟨right_ne_zero_of_mul hmn0⟩
      have left_inv : Function.LeftInverse inv_fun to_fun := by
        intro x
        dsimp only [to_fun, inv_fun, ZMod.castHom_apply]
        conv_rhs => rw [← ZMod.natCast_zmod_val x]
        rw [if_neg hmn0, ZMod.eq_iff_modEq_nat, ← Nat.modEq_and_modEq_iff_modEq_mul h,
          Prod.fst_zmod_cast, Prod.snd_zmod_cast]
        refine
          ⟨(Nat.chineseRemainder h (cast x : ZMod m).val (cast x : ZMod n).val).2.left.trans ?_,
            (Nat.chineseRemainder h (cast x : ZMod m).val (cast x : ZMod n).val).2.right.trans ?_⟩
        · rw [← ZMod.eq_iff_modEq_nat, ZMod.natCast_zmod_val, ZMod.natCast_val]
        · rw [← ZMod.eq_iff_modEq_nat, ZMod.natCast_zmod_val, ZMod.natCast_val]
      /-
        m✝ n✝ m n : Nat
        h : m.Coprime n
        to_fun : ZMod (HMul.hMul m n) → Prod (ZMod m) (ZMod n) := ⇑(ZMod.castHom ⋯ (Pr …
        inv_fun : Prod (ZMod m) (ZMod n) → ZMod (HMul.hMul m n) := fun x => ite (Eq (H …
        hmn0 : Not (Eq (HMul.hMul m n) 0)
        this✝¹ : NeZero (HMul.hMul m n)
        this✝ : NeZero m
        this : NeZero n
        left_inv : Function.LeftInverse inv_fun to_fun
        ⊢ And (Function.LeftInverse inv_fun to_fun) (Function.RightInverse inv_fun to_ …
      -/
      exact ⟨left_inv, left_inv.rightInverse_of_card_le (by simp)⟩
      /-
        🎉 no goals
      -/
  { toFun := to_fun,
    invFun := inv_fun,
    map_mul' := RingHom.map_mul _
    map_add' := RingHom.map_add _
    left_inv := inv.1
    right_inv := inv.2 }


lemma subsingleton_iff {n : ℕ} : Subsingleton (ZMod n) ↔ n = 1 := by
  /-
    n : Nat
    ⊢ Iff (Subsingleton (ZMod n)) (Eq n 1)
  -/
  constructor
    /-
      case mp
      n : Nat
      ⊢ Subsingleton (ZMod n) → Eq n 1
    -/
  · obtain (_ | _ | n) := n
      /-
        case mp.zero
        ⊢ Subsingleton (ZMod 0) → Eq 0 1
      -/
    · simpa [ZMod] using not_subsingleton _
      /-
        🎉 no goals
      -/
      /-
        case mp.succ.zero
        ⊢ Subsingleton (ZMod (HAdd.hAdd 0 1)) → Eq (HAdd.hAdd 0 1) 1
      -/
    · simp [ZMod]
      /-
        🎉 no goals
      -/
      /-
        case mp.succ.succ
        n : Nat
        ⊢ Subsingleton (ZMod (HAdd.hAdd (HAdd.hAdd n 1) 1)) → Eq (HAdd.hAdd (HAdd.hAdd …
      -/
    · simpa [ZMod] using not_subsingleton _
      /-
        🎉 no goals
      -/
    /-
      case mpr
      n : Nat
      ⊢ Eq n 1 → Subsingleton (ZMod n)
    -/
  · rintro rfl
    /-
      case mpr
      ⊢ Subsingleton (ZMod 1)
    -/
    infer_instance
    /-
      🎉 no goals
    -/


lemma nontrivial_iff {n : ℕ} : Nontrivial (ZMod n) ↔ n ≠ 1 := by
  /-
    n : Nat
    ⊢ Iff (Nontrivial (ZMod n)) (Ne n 1)
  -/
  rw [← not_subsingleton_iff_nontrivial, subsingleton_iff]
  /-
    🎉 no goals
  -/

-- todo: this can be made a `Unique` instance.

instance subsingleton_units : Subsingleton (ZMod 2)ˣ :=
      /-
        m n : Nat
        ⊢ ∀ (a b : Units (ZMod 2)), Eq a b
      -/
  ⟨by decide⟩
      /-
        🎉 no goals
      -/


@[simp]
theorem add_self_eq_zero_iff_eq_zero {n : ℕ} (hn : Odd n) {a : ZMod n} :
    a + a = 0 ↔ a = 0 := by
  /-
    n : Nat
    hn : Odd n
    a : ZMod n
    ⊢ Iff (Eq (HAdd.hAdd a a) 0) (Eq a 0)
  -/
  rw [Nat.odd_iff, ← Nat.two_dvd_ne_zero, ← Nat.prime_two.coprime_iff_not_dvd] at hn
  /-
    n : Nat
    hn : Nat.Coprime 2 n
    a : ZMod n
    ⊢ Iff (Eq (HAdd.hAdd a a) 0) (Eq a 0)
  -/
  rw [← mul_two, ← @Nat.cast_two (ZMod n), ← ZMod.coe_unitOfCoprime 2 hn, Units.mul_left_eq_zero]
  /-
    🎉 no goals
  -/


theorem ne_neg_self {n : ℕ} (hn : Odd n) {a : ZMod n} (ha : a ≠ 0) : a ≠ -a := by
  /-
    n : Nat
    hn : Odd n
    a : ZMod n
    ha : Ne a 0
    ⊢ Ne a (Neg.neg a)
  -/
  rwa [Ne, eq_neg_iff_add_eq_zero, add_self_eq_zero_iff_eq_zero hn]
  /-
    🎉 no goals
  -/


theorem neg_one_ne_one {n : ℕ} [Fact (2 < n)] : (-1 : ZMod n) ≠ 1 :=
  CharP.neg_one_ne_one (ZMod n) n


theorem neg_eq_self_mod_two (a : ZMod 2) : -a = a := by
  /-
    a : ZMod 2
    ⊢ Eq (Neg.neg a) a
  -/
                                    /-
                                      🎉 no goals
                                    -/
  fin_cases a <;> apply Fin.ext <;> simp [Fin.coe_neg, Int.natMod]; rfl
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


@[simp]
theorem natAbs_mod_two (a : ℤ) : (a.natAbs : ZMod 2) = a := by
  /-
    a : Int
    ⊢ Eq ↑a.natAbs ↑a
  -/
  cases a
    /-
      case ofNat
      a✝ : Nat
      ⊢ Eq ↑(Int.ofNat a✝).natAbs ↑(Int.ofNat a✝)
    -/
  · simp only [Int.natAbs_ofNat, Int.cast_natCast, Int.ofNat_eq_coe]
    /-
      🎉 no goals
    -/
    /-
      case negSucc
      a✝ : Nat
      ⊢ Eq ↑(Int.negSucc a✝).natAbs ↑(Int.negSucc a✝)
    -/
  · simp only [neg_eq_self_mod_two, Nat.cast_succ, Int.natAbs, Int.cast_negSucc]
    /-
      🎉 no goals
    -/


theorem val_ne_zero {n : ℕ} (a : ZMod n) : a.val ≠ 0 ↔ a ≠ 0 :=
  (val_eq_zero a).not


theorem val_pos {n : ℕ} {a : ZMod n} : 0 < a.val ↔ a ≠ 0 := by
  /-
    n : Nat
    a : ZMod n
    ⊢ Iff (LT.lt 0 a.val) (Ne a 0)
  -/
  simp [pos_iff_ne_zero]
  /-
    🎉 no goals
  -/


theorem val_eq_one : ∀ {n : ℕ} (_ : 1 < n) (a : ZMod n), a.val = 1 ↔ a = 1
  | 0, hn, _
                   /-
                     hn : LT.lt 1 0
                     x✝ : ZMod 0
                     ⊢ Iff (Eq x✝.val 1) (Eq x✝ 1)
                   -/
                   /-
                     🎉 no goals
                   -/
  | 1, hn, _ => by simp at hn
                   /-
                     🎉 no goals
                   -/
                      /-
                        n : Nat
                        x✝¹ : LT.lt 1 (HAdd.hAdd n 2)
                        x✝ : ZMod (HAdd.hAdd n 2)
                        ⊢ Iff (Eq x✝.val 1) (Eq x✝ 1)
                      -/
  | n + 2, _, _ => by simp only [val, ZMod, Fin.ext_iff, Fin.val_one]
                      /-
                        🎉 no goals
                      -/


theorem neg_eq_self_iff {n : ℕ} (a : ZMod n) : -a = a ↔ a = 0 ∨ 2 * a.val = n := by
  /-
    n : Nat
    a : ZMod n
    ⊢ Iff (Eq (Neg.neg a) a) (Or (Eq a 0) (Eq (HMul.hMul 2 a.val) n))
  -/
  rw [neg_eq_iff_add_eq_zero, ← two_mul]
  /-
    n : Nat
    a : ZMod n
    ⊢ Iff (Eq (HMul.hMul 2 a) 0) (Or (Eq a 0) (Eq (HMul.hMul 2 a.val) n))
  -/
  cases n
    /-
      case zero
      a : ZMod 0
      ⊢ Iff (Eq (HMul.hMul 2 a) 0) (Or (Eq a 0) (Eq (HMul.hMul 2 a.val) 0))
    -/
  · rw [@mul_eq_zero ℤ, @mul_eq_zero ℕ, val_eq_zero]
    exact
      ⟨fun h => h.elim (by simp) Or.inl, fun h =>
        Or.inr (h.elim id fun h => h.elim (by simp) id)⟩
  conv_lhs =>
    rw [← a.natCast_zmod_val, ← Nat.cast_two, ← Nat.cast_mul, natCast_zmod_eq_zero_iff_dvd]
  /-
    case succ
    n✝ : Nat
    a : ZMod (HAdd.hAdd n✝ 1)
    ⊢ Iff (Dvd.dvd (HAdd.hAdd n✝ 1) (HMul.hMul 2 a.val)) (Or (Eq a 0) (Eq (HMul.hM …
  -/
  constructor
    /-
      case succ.mp
      n✝ : Nat
      a : ZMod (HAdd.hAdd n✝ 1)
      ⊢ Dvd.dvd (HAdd.hAdd n✝ 1) (HMul.hMul 2 a.val) → Or (Eq a 0) (Eq (HMul.hMul 2  …
    -/
  · rintro ⟨m, he⟩
    /-
      case succ.mp.intro
      n✝ : Nat
      a : ZMod (HAdd.hAdd n✝ 1)
      m : Nat
      he : Eq (HMul.hMul 2 a.val) (HMul.hMul (HAdd.hAdd n✝ 1) m)
      ⊢ Or (Eq a 0) (Eq (HMul.hMul 2 a.val) (HAdd.hAdd n✝ 1))
    -/
    cases' m with m
      /-
        case succ.mp.intro.zero
        n✝ : Nat
        a : ZMod (HAdd.hAdd n✝ 1)
        he : Eq (HMul.hMul 2 a.val) (HMul.hMul (HAdd.hAdd n✝ 1) 0)
        ⊢ Or (Eq a 0) (Eq (HMul.hMul 2 a.val) (HAdd.hAdd n✝ 1))
      -/
    · rw [mul_zero, mul_eq_zero] at he
      /-
        case succ.mp.intro.zero
        n✝ : Nat
        a : ZMod (HAdd.hAdd n✝ 1)
        he : Or (Eq 2 0) (Eq a.val 0)
        ⊢ Or (Eq a 0) (Eq (HMul.hMul 2 a.val) (HAdd.hAdd n✝ 1))
      -/
      rcases he with (⟨⟨⟩⟩ | he)
      /-
        case succ.mp.intro.zero.inr
        n✝ : Nat
        a : ZMod (HAdd.hAdd n✝ 1)
        he : Eq a.val 0
        ⊢ Or (Eq a 0) (Eq (HMul.hMul 2 a.val) (HAdd.hAdd n✝ 1))
      -/
      exact Or.inl (a.val_eq_zero.1 he)
      /-
        🎉 no goals
      -/
    /-
      case succ.mp.intro.succ
      n✝ : Nat
      a : ZMod (HAdd.hAdd n✝ 1)
      m : Nat
      he : Eq (HMul.hMul 2 a.val) (HMul.hMul (HAdd.hAdd n✝ 1) (HAdd.hAdd m 1))
      ⊢ Or (Eq a 0) (Eq (HMul.hMul 2 a.val) (HAdd.hAdd n✝ 1))
    -/
    cases m
      /-
        case succ.mp.intro.succ.zero
        n✝ : Nat
        a : ZMod (HAdd.hAdd n✝ 1)
        he : Eq (HMul.hMul 2 a.val) (HMul.hMul (HAdd.hAdd n✝ 1) (HAdd.hAdd 0 1))
        ⊢ Or (Eq a 0) (Eq (HMul.hMul 2 a.val) (HAdd.hAdd n✝ 1))
      -/
    · right
      /-
        case succ.mp.intro.succ.zero.h
        n✝ : Nat
        a : ZMod (HAdd.hAdd n✝ 1)
        he : Eq (HMul.hMul 2 a.val) (HMul.hMul (HAdd.hAdd n✝ 1) (HAdd.hAdd 0 1))
        ⊢ Eq (HMul.hMul 2 a.val) (HAdd.hAdd n✝ 1)
      -/
      rwa [show 0 + 1 = 1 from rfl, mul_one] at he
      /-
        🎉 no goals
      -/
    /-
      case succ.mp.intro.succ.succ
      n✝¹ : Nat
      a : ZMod (HAdd.hAdd n✝¹ 1)
      n✝ : Nat
      he : Eq (HMul.hMul 2 a.val) (HMul.hMul (HAdd.hAdd n✝¹ 1) (HAdd.hAdd (HAdd.hAdd …
      ⊢ Or (Eq a 0) (Eq (HMul.hMul 2 a.val) (HAdd.hAdd n✝¹ 1))
    -/
    refine (a.val_lt.not_le <| Nat.le_of_mul_le_mul_left ?_ zero_lt_two).elim
    /-
      case succ.mp.intro.succ.succ
      n✝¹ : Nat
      a : ZMod (HAdd.hAdd n✝¹ 1)
      n✝ : Nat
      he : Eq (HMul.hMul 2 a.val) (HMul.hMul (HAdd.hAdd n✝¹ 1) (HAdd.hAdd (HAdd.hAdd …
      ⊢ LE.le (HMul.hMul 2 (HAdd.hAdd n✝¹ 1)) (HMul.hMul 2 a.val)
    -/
    rw [he, mul_comm]
    /-
      case succ.mp.intro.succ.succ
      n✝¹ : Nat
      a : ZMod (HAdd.hAdd n✝¹ 1)
      n✝ : Nat
      he : Eq (HMul.hMul 2 a.val) (HMul.hMul (HAdd.hAdd n✝¹ 1) (HAdd.hAdd (HAdd.hAdd …
      ⊢ LE.le (HMul.hMul (HAdd.hAdd n✝¹ 1) 2) (HMul.hMul (HAdd.hAdd n✝¹ 1) (HAdd.hAd …
    -/
    apply Nat.mul_le_mul_left
    /-
      case succ.mp.intro.succ.succ.h
      n✝¹ : Nat
      a : ZMod (HAdd.hAdd n✝¹ 1)
      n✝ : Nat
      he : Eq (HMul.hMul 2 a.val) (HMul.hMul (HAdd.hAdd n✝¹ 1) (HAdd.hAdd (HAdd.hAdd …
      ⊢ LE.le 2 (HAdd.hAdd (HAdd.hAdd n✝ 1) 1)
    -/
    erw [Nat.succ_le_succ_iff, Nat.succ_le_succ_iff]; simp
                                                      /-
                                                        🎉 no goals
                                                      -/
    /-
      case succ.mpr
      n✝ : Nat
      a : ZMod (HAdd.hAdd n✝ 1)
      ⊢ Or (Eq a 0) (Eq (HMul.hMul 2 a.val) (HAdd.hAdd n✝ 1)) → Dvd.dvd (HAdd.hAdd n …
    -/
  · rintro (rfl | h)
      /-
        case succ.mpr.inl
        n✝ : Nat
        ⊢ Dvd.dvd (HAdd.hAdd n✝ 1) (HMul.hMul 2 (ZMod.val 0))
      -/
    · rw [val_zero, mul_zero]
      /-
        case succ.mpr.inl
        n✝ : Nat
        ⊢ Dvd.dvd (HAdd.hAdd n✝ 1) 0
      -/
      apply dvd_zero
      /-
        🎉 no goals
      -/
      /-
        case succ.mpr.inr
        n✝ : Nat
        a : ZMod (HAdd.hAdd n✝ 1)
        h : Eq (HMul.hMul 2 a.val) (HAdd.hAdd n✝ 1)
        ⊢ Dvd.dvd (HAdd.hAdd n✝ 1) (HMul.hMul 2 a.val)
      -/
    · rw [h]
      /-
        🎉 no goals
      -/


theorem val_cast_of_lt {n : ℕ} {a : ℕ} (h : a < n) : (a : ZMod n).val = a := by
  /-
    n a : Nat
    h : LT.lt a n
    ⊢ Eq (↑a).val a
  -/
  rw [val_natCast, Nat.mod_eq_of_lt h]
  /-
    🎉 no goals
  -/


theorem val_cast_zmod_lt {m : ℕ} [NeZero m] (n : ℕ) [NeZero n] (a : ZMod m) :
    (a.cast : ZMod n).val < m := by
  /-
    m : Nat
    inst✝¹ : NeZero m
    n : Nat
    inst✝ : NeZero n
    a : ZMod m
    ⊢ LT.lt a.cast.val m
  -/
  rcases m with (⟨⟩|⟨m⟩); · cases NeZero.ne 0 rfl
                            /-
                              🎉 no goals
                            -/
  /-
    case succ
    n : Nat
    inst✝¹ : NeZero n
    m : Nat
    inst✝ : NeZero (HAdd.hAdd m 1)
    a : ZMod (HAdd.hAdd m 1)
    ⊢ LT.lt a.cast.val (HAdd.hAdd m 1)
  -/
  by_cases h : m < n
    /-
      case pos
      n : Nat
      inst✝¹ : NeZero n
      m : Nat
      inst✝ : NeZero (HAdd.hAdd m 1)
      a : ZMod (HAdd.hAdd m 1)
      h : LT.lt m n
      ⊢ LT.lt a.cast.val (HAdd.hAdd m 1)
    -/
  · rcases n with (⟨⟩|⟨n⟩); · simp at h
                              /-
                                🎉 no goals
                              -/
    /-
      case pos.succ
      m : Nat
      inst✝¹ : NeZero (HAdd.hAdd m 1)
      a : ZMod (HAdd.hAdd m 1)
      n : Nat
      inst✝ : NeZero (HAdd.hAdd n 1)
      h : LT.lt m (HAdd.hAdd n 1)
      ⊢ LT.lt a.cast.val (HAdd.hAdd m 1)
    -/
    rw [← natCast_val, val_cast_of_lt]
      /-
        case pos.succ
        m : Nat
        inst✝¹ : NeZero (HAdd.hAdd m 1)
        a : ZMod (HAdd.hAdd m 1)
        n : Nat
        inst✝ : NeZero (HAdd.hAdd n 1)
        h : LT.lt m (HAdd.hAdd n 1)
        ⊢ LT.lt a.val (HAdd.hAdd m 1)
      -/
    · apply a.val_lt
      /-
        🎉 no goals
      -/
    /-
      case pos.succ
      m : Nat
      inst✝¹ : NeZero (HAdd.hAdd m 1)
      a : ZMod (HAdd.hAdd m 1)
      n : Nat
      inst✝ : NeZero (HAdd.hAdd n 1)
      h : LT.lt m (HAdd.hAdd n 1)
      ⊢ LT.lt a.val (HAdd.hAdd n 1)
    -/
    apply lt_of_le_of_lt (Nat.le_of_lt_succ (ZMod.val_lt a)) h
    /-
      🎉 no goals
    -/
    /-
      case neg
      n : Nat
      inst✝¹ : NeZero n
      m : Nat
      inst✝ : NeZero (HAdd.hAdd m 1)
      a : ZMod (HAdd.hAdd m 1)
      h : Not (LT.lt m n)
      ⊢ LT.lt a.cast.val (HAdd.hAdd m 1)
    -/
  · rw [not_lt] at h
    /-
      case neg
      n : Nat
      inst✝¹ : NeZero n
      m : Nat
      inst✝ : NeZero (HAdd.hAdd m 1)
      a : ZMod (HAdd.hAdd m 1)
      h : LE.le n m
      ⊢ LT.lt a.cast.val (HAdd.hAdd m 1)
    -/
    apply lt_of_lt_of_le (ZMod.val_lt _) (le_trans h (Nat.le_succ m))
    /-
      🎉 no goals
    -/


theorem neg_val' {n : ℕ} [NeZero n] (a : ZMod n) : (-a).val = (n - a.val) % n :=
  calc
                                  /-
                                    n : Nat
                                    inst✝ : NeZero n
                                    a : ZMod n
                                    ⊢ Eq (Neg.neg a).val (HMod.hMod (Neg.neg a).val n)
                                  -/
    (-a).val = val (-a) % n := by rw [Nat.mod_eq_of_lt (-a).val_lt]
                                  /-
                                    🎉 no goals
                                  -/
    _ = (n - val a) % n :=
      Nat.ModEq.add_right_cancel' (val a)
        (by
          rw [Nat.ModEq, ← val_add, neg_add_cancel, tsub_add_cancel_of_le a.val_le, Nat.mod_self,
            val_zero])


theorem neg_val {n : ℕ} [NeZero n] (a : ZMod n) : (-a).val = if a = 0 then 0 else n - a.val := by
  /-
    n : Nat
    inst✝ : NeZero n
    a : ZMod n
    ⊢ Eq (Neg.neg a).val (ite (Eq a 0) 0 (HSub.hSub n a.val))
  -/
  rw [neg_val']
  /-
    n : Nat
    inst✝ : NeZero n
    a : ZMod n
    ⊢ Eq (HMod.hMod (HSub.hSub n a.val) n) (ite (Eq a 0) 0 (HSub.hSub n a.val))
  -/
  by_cases h : a = 0; · rw [if_pos h, h, val_zero, tsub_zero, Nat.mod_self]
                        /-
                          🎉 no goals
                        -/
  /-
    case neg
    n : Nat
    inst✝ : NeZero n
    a : ZMod n
    h : Not (Eq a 0)
    ⊢ Eq (HMod.hMod (HSub.hSub n a.val) n) (ite (Eq a 0) 0 (HSub.hSub n a.val))
  -/
  rw [if_neg h]
  /-
    case neg
    n : Nat
    inst✝ : NeZero n
    a : ZMod n
    h : Not (Eq a 0)
    ⊢ Eq (HMod.hMod (HSub.hSub n a.val) n) (HSub.hSub n a.val)
  -/
  apply Nat.mod_eq_of_lt
  /-
    case neg.h
    n : Nat
    inst✝ : NeZero n
    a : ZMod n
    h : Not (Eq a 0)
    ⊢ LT.lt (HSub.hSub n a.val) n
  -/
  apply Nat.sub_lt (NeZero.pos n)
  /-
    case neg.h
    n : Nat
    inst✝ : NeZero n
    a : ZMod n
    h : Not (Eq a 0)
    ⊢ LT.lt 0 a.val
  -/
  contrapose! h
  /-
    case neg.h
    n : Nat
    inst✝ : NeZero n
    a : ZMod n
    h : LE.le a.val 0
    ⊢ Eq a 0
  -/
  rwa [Nat.le_zero, val_eq_zero] at h
  /-
    🎉 no goals
  -/


theorem val_neg_of_ne_zero {n : ℕ} [nz : NeZero n] (a : ZMod n) [na : NeZero a] :
                                /-
                                  n : Nat
                                  nz : NeZero n
                                  a : ZMod n
                                  na : NeZero a
                                  ⊢ Eq (Neg.neg a).val (HSub.hSub n a.val)
                                -/
    (- a).val = n - a.val := by simp_all [neg_val a, na.out]
                                /-
                                  🎉 no goals
                                -/


theorem val_sub {n : ℕ} [NeZero n] {a b : ZMod n} (h : b.val ≤ a.val) :
    (a - b).val = a.val - b.val := by
  /-
    n : Nat
    inst✝ : NeZero n
    a b : ZMod n
    h : LE.le b.val a.val
    ⊢ Eq (HSub.hSub a b).val (HSub.hSub a.val b.val)
  -/
  by_cases hb : b = 0
    /-
      case pos
      n : Nat
      inst✝ : NeZero n
      a b : ZMod n
      h : LE.le b.val a.val
      hb : Eq b 0
      ⊢ Eq (HSub.hSub a b).val (HSub.hSub a.val b.val)
    -/
  · cases hb; simp
              /-
                🎉 no goals
              -/
    /-
      case neg
      n : Nat
      inst✝ : NeZero n
      a b : ZMod n
      h : LE.le b.val a.val
      hb : Not (Eq b 0)
      ⊢ Eq (HSub.hSub a b).val (HSub.hSub a.val b.val)
    -/
  · have : NeZero b := ⟨hb⟩
    rw [sub_eq_add_neg, val_add, val_neg_of_ne_zero, ← Nat.add_sub_assoc (le_of_lt (val_lt _)),
      add_comm, Nat.add_sub_assoc h, Nat.add_mod_left]
    /-
      case neg
      n : Nat
      inst✝ : NeZero n
      a b : ZMod n
      h : LE.le b.val a.val
      hb : Not (Eq b 0)
      this : NeZero b
      ⊢ Eq (HMod.hMod (HSub.hSub a.val b.val) n) (HSub.hSub a.val b.val)
    -/
    apply Nat.mod_eq_of_lt (tsub_lt_of_lt (val_lt _))
    /-
      🎉 no goals
    -/


theorem val_cast_eq_val_of_lt {m n : ℕ} [nzm : NeZero m] {a : ZMod m}
    (h : a.val < n) : (a.cast : ZMod n).val = a.val := by
  /-
    m n : Nat
    nzm : NeZero m
    a : ZMod m
    h : LT.lt a.val n
    ⊢ Eq a.cast.val a.val
  -/
  have nzn : NeZero n := by constructor; rintro rfl; simp at h
  cases m with
  | zero => cases nzm; simp_all
  | succ m =>
    cases n with
    | zero => cases nzn; simp_all
    | succ n => exact Fin.val_cast_of_lt h


theorem cast_cast_zmod_of_le {m n : ℕ} [hm : NeZero m] (h : m ≤ n) (a : ZMod m) :
    (cast (cast a : ZMod n) : ZMod m) = a := by
  /-
    m n : Nat
    hm : NeZero m
    h : LE.le m n
    a : ZMod m
    ⊢ Eq a.cast.cast a
  -/
  have : NeZero n := ⟨((Nat.zero_lt_of_ne_zero hm.out).trans_le h).ne'⟩
  /-
    m n : Nat
    hm : NeZero m
    h : LE.le m n
    a : ZMod m
    this : NeZero n
    ⊢ Eq a.cast.cast a
  -/
  rw [cast_eq_val, val_cast_eq_val_of_lt (a.val_lt.trans_le h), natCast_zmod_val]
  /-
    🎉 no goals
  -/


theorem val_pow {m n : ℕ} {a : ZMod n} [ilt : Fact (1 < n)] (h : a.val ^ m < n) :
    (a ^ m).val = a.val ^ m := by
  induction m with
  | zero => simp [ZMod.val_one]
  | succ m ih =>
    have : a.val ^ m < n := by
      obtain rfl | ha := eq_or_ne a 0
      · by_cases hm : m = 0
        · cases hm; simp [ilt.out]
        · simp only [val_zero, ne_eq, hm, not_false_eq_true, zero_pow, Nat.zero_lt_of_lt h]
      · exact lt_of_le_of_lt
         (Nat.pow_le_pow_of_le_right (by rwa [gt_iff_lt, ZMod.val_pos]) (Nat.le_succ m)) h
    rw [pow_succ, ZMod.val_mul, ih this, ← pow_succ, Nat.mod_eq_of_lt h]


theorem val_pow_le {m n : ℕ} [Fact (1 < n)] {a : ZMod n} : (a ^ m).val ≤ a.val ^ m := by
  induction m with
  | zero => simp [ZMod.val_one]
  | succ m ih =>
    rw [pow_succ, pow_succ]
    apply le_trans (ZMod.val_mul_le _ _)
    apply Nat.mul_le_mul_right _ ih


/-- `valMinAbs x` returns the integer in the same equivalence class as `x` that is closest to `0`,
  The result will be in the interval `(-n/2, n/2]`. -/
def valMinAbs : ∀ {n : ℕ}, ZMod n → ℤ
  | 0, x => x
  | n@(_ + 1), x => if x.val ≤ n / 2 then x.val else (x.val : ℤ) - n


@[simp]
theorem valMinAbs_def_zero (x : ZMod 0) : valMinAbs x = x :=
  rfl


theorem valMinAbs_def_pos {n : ℕ} [NeZero n] (x : ZMod n) :
    valMinAbs x = if x.val ≤ n / 2 then (x.val : ℤ) else x.val - n := by
  /-
    n : Nat
    inst✝ : NeZero n
    x : ZMod n
    ⊢ Eq x.valMinAbs (ite (LE.le x.val (HDiv.hDiv n 2)) (↑x.val) (HSub.hSub ↑x.val …
  -/
  cases n
    /-
      case zero
      inst✝ : NeZero 0
      x : ZMod 0
      ⊢ Eq x.valMinAbs (ite (LE.le x.val (0 / 2)) (↑x.val) (HSub.hSub ↑x.val ↑0))
    -/
  · cases NeZero.ne 0 rfl
    /-
      🎉 no goals
    -/
    /-
      case succ
      n✝ : Nat
      inst✝ : NeZero (HAdd.hAdd n✝ 1)
      x : ZMod (HAdd.hAdd n✝ 1)
      ⊢ Eq x.valMinAbs (ite (LE.le x.val (HDiv.hDiv (HAdd.hAdd n✝ 1) 2)) (↑x.val) (H …
    -/
  · rfl
    /-
      🎉 no goals
    -/


@[simp, norm_cast]
theorem coe_valMinAbs : ∀ {n : ℕ} (x : ZMod n), (x.valMinAbs : ZMod n) = x
  | 0, _ => Int.cast_id
  | k@(n + 1), x => by
    /-
      k n : Nat
      h✝ : Eq k (HAdd.hAdd n 1)
      x : ZMod (namedPattern k (HAdd.hAdd n 1) h✝)
      ⊢ Eq (↑x.valMinAbs) x
    -/
    rw [valMinAbs_def_pos]
    /-
      k n : Nat
      h✝ : Eq k (HAdd.hAdd n 1)
      x : ZMod (namedPattern k (HAdd.hAdd n 1) h✝)
      ⊢ Eq (↑(ite (LE.le x.val (HDiv.hDiv (namedPattern k (HAdd.hAdd n 1) h✝) 2)) (↑ …
    -/
    split_ifs
      /-
        case pos
        k n : Nat
        h✝¹ : Eq k (HAdd.hAdd n 1)
        x : ZMod (namedPattern k (HAdd.hAdd n 1) h✝¹)
        h✝ : LE.le x.val (HDiv.hDiv (namedPattern k (HAdd.hAdd n 1) h✝¹) 2)
        ⊢ Eq (↑↑x.val) x
      -/
    · rw [Int.cast_natCast, natCast_zmod_val]
      /-
        🎉 no goals
      -/
    · rw [Int.cast_sub, Int.cast_natCast, natCast_zmod_val, Int.cast_natCast, natCast_self,
        sub_zero]


theorem injective_valMinAbs {n : ℕ} : (valMinAbs : ZMod n → ℤ).Injective :=
  Function.injective_iff_hasLeftInverse.2 ⟨_, coe_valMinAbs⟩


theorem _root_.Nat.le_div_two_iff_mul_two_le {n m : ℕ} : m ≤ n / 2 ↔ (m : ℤ) * 2 ≤ n := by
  /-
    n m : Nat
    ⊢ Iff (LE.le m (HDiv.hDiv n 2)) (LE.le (HMul.hMul (↑m) 2) ↑n)
  -/
  rw [Nat.le_div_iff_mul_le zero_lt_two, ← Int.ofNat_le, Int.ofNat_mul, Nat.cast_two]
  /-
    🎉 no goals
  -/


theorem valMinAbs_nonneg_iff {n : ℕ} [NeZero n] (x : ZMod n) : 0 ≤ x.valMinAbs ↔ x.val ≤ n / 2 := by
  /-
    n : Nat
    inst✝ : NeZero n
    x : ZMod n
    ⊢ Iff (LE.le 0 x.valMinAbs) (LE.le x.val (HDiv.hDiv n 2))
  -/
  rw [valMinAbs_def_pos]; split_ifs with h
    /-
      case pos
      n : Nat
      inst✝ : NeZero n
      x : ZMod n
      h : LE.le x.val (HDiv.hDiv n 2)
      ⊢ Iff (LE.le 0 ↑x.val) (LE.le x.val (HDiv.hDiv n 2))
    -/
  · exact iff_of_true (Nat.cast_nonneg _) h
    /-
      🎉 no goals
    -/
    /-
      case neg
      n : Nat
      inst✝ : NeZero n
      x : ZMod n
      h : Not (LE.le x.val (HDiv.hDiv n 2))
      ⊢ Iff (LE.le 0 (HSub.hSub ↑x.val ↑n)) (LE.le x.val (HDiv.hDiv n 2))
    -/
  · exact iff_of_false (sub_lt_zero.2 <| Int.ofNat_lt.2 x.val_lt).not_le h
    /-
      🎉 no goals
    -/


theorem valMinAbs_mul_two_eq_iff {n : ℕ} (a : ZMod n) : a.valMinAbs * 2 = n ↔ 2 * a.val = n := by
  /-
    n : Nat
    a : ZMod n
    ⊢ Iff (Eq (HMul.hMul a.valMinAbs 2) ↑n) (Eq (HMul.hMul 2 a.val) n)
  -/
  cases' n with n
    /-
      case zero
      a : ZMod 0
      ⊢ Iff (Eq (HMul.hMul a.valMinAbs 2) ↑0) (Eq (HMul.hMul 2 a.val) 0)
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case succ
    n : Nat
    a : ZMod (HAdd.hAdd n 1)
    ⊢ Iff (Eq (HMul.hMul a.valMinAbs 2) ↑(HAdd.hAdd n 1)) (Eq (HMul.hMul 2 a.val)  …
  -/
  by_cases h : a.val ≤ n.succ / 2
    /-
      case pos
      n : Nat
      a : ZMod (HAdd.hAdd n 1)
      h : LE.le a.val (HDiv.hDiv n.succ 2)
      ⊢ Iff (Eq (HMul.hMul a.valMinAbs 2) ↑(HAdd.hAdd n 1)) (Eq (HMul.hMul 2 a.val)  …
    -/
  · dsimp [valMinAbs]
    /-
      case pos
      n : Nat
      a : ZMod (HAdd.hAdd n 1)
      h : LE.le a.val (HDiv.hDiv n.succ 2)
      ⊢ Iff (Eq (HMul.hMul (ite (LE.le a.val (HDiv.hDiv (HAdd.hAdd n 1) 2)) (↑a.val) …
    -/
    rw [if_pos h, ← Int.natCast_inj, Nat.cast_mul, Nat.cast_two, mul_comm]
    /-
      🎉 no goals
    -/
  /-
    case neg
    n : Nat
    a : ZMod (HAdd.hAdd n 1)
    h : Not (LE.le a.val (HDiv.hDiv n.succ 2))
    ⊢ Iff (Eq (HMul.hMul a.valMinAbs 2) ↑(HAdd.hAdd n 1)) (Eq (HMul.hMul 2 a.val)  …
  -/
  apply iff_of_false _ (mt _ h)
    /-
      n : Nat
      a : ZMod (HAdd.hAdd n 1)
      h : Not (LE.le a.val (HDiv.hDiv n.succ 2))
      ⊢ Not (Eq (HMul.hMul a.valMinAbs 2) ↑(HAdd.hAdd n 1))
    -/
  · intro he
    /-
      n : Nat
      a : ZMod (HAdd.hAdd n 1)
      h : Not (LE.le a.val (HDiv.hDiv n.succ 2))
      he : Eq (HMul.hMul a.valMinAbs 2) ↑(HAdd.hAdd n 1)
      ⊢ False
    -/
    rw [← a.valMinAbs_nonneg_iff, ← mul_nonneg_iff_left_nonneg_of_pos, he] at h
    /-
      n : Nat
      a : ZMod (HAdd.hAdd n 1)
      h✝ : Not (LE.le 0 a.valMinAbs)
      h : Not (LE.le 0 ↑(HAdd.hAdd n 1))
      he : Eq (HMul.hMul a.valMinAbs 2) ↑(HAdd.hAdd n 1)
      ⊢ False
    -/
    exacts [h (Nat.cast_nonneg _), zero_lt_two]
    /-
      🎉 no goals
    -/
    /-
      n : Nat
      a : ZMod (HAdd.hAdd n 1)
      h : Not (LE.le a.val (HDiv.hDiv n.succ 2))
      ⊢ Eq (HMul.hMul 2 a.val) (HAdd.hAdd n 1) → LE.le a.val (HDiv.hDiv n.succ 2)
    -/
  · rw [mul_comm]
    /-
      n : Nat
      a : ZMod (HAdd.hAdd n 1)
      h : Not (LE.le a.val (HDiv.hDiv n.succ 2))
      ⊢ Eq (HMul.hMul a.val 2) (HAdd.hAdd n 1) → LE.le a.val (HDiv.hDiv n.succ 2)
    -/
    exact fun h => (Nat.le_div_iff_mul_le zero_lt_two).2 h.le
    /-
      🎉 no goals
    -/


theorem valMinAbs_mem_Ioc {n : ℕ} [NeZero n] (x : ZMod n) :
    x.valMinAbs * 2 ∈ Set.Ioc (-n : ℤ) n := by
  /-
    n : Nat
    inst✝ : NeZero n
    x : ZMod n
    ⊢ Membership.mem (Set.Ioc (Neg.neg ↑n) ↑n) (HMul.hMul x.valMinAbs 2)
  -/
  simp_rw [valMinAbs_def_pos, Nat.le_div_two_iff_mul_two_le]; split_ifs with h
    /-
      case pos
      n : Nat
      inst✝ : NeZero n
      x : ZMod n
      h : LE.le (HMul.hMul (↑x.val) 2) ↑n
      ⊢ Membership.mem (Set.Ioc (Neg.neg ↑n) ↑n) (HMul.hMul (↑x.val) 2)
    -/
  · refine ⟨(neg_lt_zero.2 <| mod_cast NeZero.pos n).trans_le (mul_nonneg ?_ ?_), h⟩
    /-
      case pos.refine_1
      n : Nat
      inst✝ : NeZero n
      x : ZMod n
      h : LE.le (HMul.hMul (↑x.val) 2) ↑n
      ⊢ LE.le 0 ↑x.val
    -/
    exacts [Nat.cast_nonneg _, zero_le_two]
    /-
      🎉 no goals
    -/
    /-
      case neg
      n : Nat
      inst✝ : NeZero n
      x : ZMod n
      h : Not (LE.le (HMul.hMul (↑x.val) 2) ↑n)
      ⊢ Membership.mem (Set.Ioc (Neg.neg ↑n) ↑n) (HMul.hMul (HSub.hSub ↑x.val ↑n) 2)
    -/
  · refine ⟨?_, le_trans (mul_nonpos_of_nonpos_of_nonneg ?_ zero_le_two) <| Nat.cast_nonneg _⟩
      /-
        case neg.refine_1
        n : Nat
        inst✝ : NeZero n
        x : ZMod n
        h : Not (LE.le (HMul.hMul (↑x.val) 2) ↑n)
        ⊢ LT.lt (Neg.neg ↑n) (HMul.hMul (HSub.hSub ↑x.val ↑n) 2)
      -/
    · linarith only [h]
      /-
        🎉 no goals
      -/
      /-
        case neg.refine_2
        n : Nat
        inst✝ : NeZero n
        x : ZMod n
        h : Not (LE.le (HMul.hMul (↑x.val) 2) ↑n)
        ⊢ LE.le (HSub.hSub ↑x.val ↑n) 0
      -/
    · rw [sub_nonpos, Int.ofNat_le]
      /-
        case neg.refine_2
        n : Nat
        inst✝ : NeZero n
        x : ZMod n
        h : Not (LE.le (HMul.hMul (↑x.val) 2) ↑n)
        ⊢ LE.le x.val n
      -/
      exact x.val_lt.le
      /-
        🎉 no goals
      -/


theorem valMinAbs_spec {n : ℕ} [NeZero n] (x : ZMod n) (y : ℤ) :
    x.valMinAbs = y ↔ x = y ∧ y * 2 ∈ Set.Ioc (-n : ℤ) n :=
  ⟨by
    /-
      n : Nat
      inst✝ : NeZero n
      x : ZMod n
      y : Int
      ⊢ Eq x.valMinAbs y → And (Eq x ↑y) (Membership.mem (Set.Ioc (Neg.neg ↑n) ↑n) ( …
    -/
    rintro rfl
    /-
      n : Nat
      inst✝ : NeZero n
      x : ZMod n
      ⊢ And (Eq x ↑x.valMinAbs) (Membership.mem (Set.Ioc (Neg.neg ↑n) ↑n) (HMul.hMul …
    -/
    exact ⟨x.coe_valMinAbs.symm, x.valMinAbs_mem_Ioc⟩, fun h =>
    /-
      🎉 no goals
    -/
      by
        /-
          n : Nat
          inst✝ : NeZero n
          x : ZMod n
          y : Int
          h : And (Eq x ↑y) (Membership.mem (Set.Ioc (Neg.neg ↑n) ↑n) (HMul.hMul y 2))
          ⊢ Eq x.valMinAbs y
        -/
        rw [← sub_eq_zero]
        /-
          n : Nat
          inst✝ : NeZero n
          x : ZMod n
          y : Int
          h : And (Eq x ↑y) (Membership.mem (Set.Ioc (Neg.neg ↑n) ↑n) (HMul.hMul y 2))
          ⊢ Eq (HSub.hSub x.valMinAbs y) 0
        -/
        apply @Int.eq_zero_of_abs_lt_dvd n
          /-
            case h1
            n : Nat
            inst✝ : NeZero n
            x : ZMod n
            y : Int
            h : And (Eq x ↑y) (Membership.mem (Set.Ioc (Neg.neg ↑n) ↑n) (HMul.hMul y 2))
            ⊢ Dvd.dvd (↑n) (HSub.hSub x.valMinAbs y)
          -/
        · rw [← intCast_zmod_eq_zero_iff_dvd, Int.cast_sub, coe_valMinAbs, h.1, sub_self]
          /-
            🎉 no goals
          -/
        /-
          case h2
          n : Nat
          inst✝ : NeZero n
          x : ZMod n
          y : Int
          h : And (Eq x ↑y) (Membership.mem (Set.Ioc (Neg.neg ↑n) ↑n) (HMul.hMul y 2))
          ⊢ LT.lt (abs (HSub.hSub x.valMinAbs y)) ↑n
        -/
        rw [← mul_lt_mul_right (@zero_lt_two ℤ _ _ _ _ _)]
        /-
          case h2
          n : Nat
          inst✝ : NeZero n
          x : ZMod n
          y : Int
          h : And (Eq x ↑y) (Membership.mem (Set.Ioc (Neg.neg ↑n) ↑n) (HMul.hMul y 2))
          ⊢ LT.lt (HMul.hMul (abs (HSub.hSub x.valMinAbs y)) 2) (HMul.hMul (↑n) 2)
        -/
        nth_rw 1 [← abs_eq_self.2 (@zero_le_two ℤ _ _ _ _)]
        /-
          case h2
          n : Nat
          inst✝ : NeZero n
          x : ZMod n
          y : Int
          h : And (Eq x ↑y) (Membership.mem (Set.Ioc (Neg.neg ↑n) ↑n) (HMul.hMul y 2))
          ⊢ LT.lt (HMul.hMul (abs (HSub.hSub x.valMinAbs y)) (abs 2)) (HMul.hMul (↑n) 2)
        -/
        rw [← abs_mul, sub_mul, abs_lt]
        /-
          case h2
          n : Nat
          inst✝ : NeZero n
          x : ZMod n
          y : Int
          h : And (Eq x ↑y) (Membership.mem (Set.Ioc (Neg.neg ↑n) ↑n) (HMul.hMul y 2))
          ⊢ And (LT.lt (Neg.neg (HMul.hMul (↑n) 2)) (HSub.hSub (HMul.hMul x.valMinAbs 2) …
        -/
                        /-
                          🎉 no goals
                        -/
        constructor <;> linarith only [x.valMinAbs_mem_Ioc.1, x.valMinAbs_mem_Ioc.2, h.2.1, h.2.2]⟩
                        /-
                          🎉 no goals
                        -/


theorem natAbs_valMinAbs_le {n : ℕ} [NeZero n] (x : ZMod n) : x.valMinAbs.natAbs ≤ n / 2 := by
  /-
    n : Nat
    inst✝ : NeZero n
    x : ZMod n
    ⊢ LE.le x.valMinAbs.natAbs (HDiv.hDiv n 2)
  -/
  rw [Nat.le_div_two_iff_mul_two_le]
  /-
    n : Nat
    inst✝ : NeZero n
    x : ZMod n
    ⊢ LE.le (HMul.hMul (↑x.valMinAbs.natAbs) 2) ↑n
  -/
  cases' x.valMinAbs.natAbs_eq with h h
    /-
      case inl
      n : Nat
      inst✝ : NeZero n
      x : ZMod n
      h : Eq x.valMinAbs ↑x.valMinAbs.natAbs
      ⊢ LE.le (HMul.hMul (↑x.valMinAbs.natAbs) 2) ↑n
    -/
  · rw [← h]
    /-
      case inl
      n : Nat
      inst✝ : NeZero n
      x : ZMod n
      h : Eq x.valMinAbs ↑x.valMinAbs.natAbs
      ⊢ LE.le (HMul.hMul x.valMinAbs 2) ↑n
    -/
    exact x.valMinAbs_mem_Ioc.2
    /-
      🎉 no goals
    -/
    /-
      case inr
      n : Nat
      inst✝ : NeZero n
      x : ZMod n
      h : Eq x.valMinAbs (Neg.neg ↑x.valMinAbs.natAbs)
      ⊢ LE.le (HMul.hMul (↑x.valMinAbs.natAbs) 2) ↑n
    -/
  · rw [← neg_le_neg_iff, ← neg_mul, ← h]
    /-
      case inr
      n : Nat
      inst✝ : NeZero n
      x : ZMod n
      h : Eq x.valMinAbs (Neg.neg ↑x.valMinAbs.natAbs)
      ⊢ LE.le (Neg.neg ↑n) (HMul.hMul x.valMinAbs 2)
    -/
    exact x.valMinAbs_mem_Ioc.1.le
    /-
      🎉 no goals
    -/


@[simp]
theorem valMinAbs_zero : ∀ n, (0 : ZMod n).valMinAbs = 0
            /-
              ⊢ Eq (ZMod.valMinAbs 0) 0
            -/
  | 0 => by simp only [valMinAbs_def_zero]
            /-
              🎉 no goals
            -/
                /-
                  n : Nat
                  ⊢ Eq (ZMod.valMinAbs 0) 0
                -/
  | n + 1 => by simp only [valMinAbs_def_pos, if_true, Int.ofNat_zero, zero_le, val_zero]
                /-
                  🎉 no goals
                -/


@[simp]
theorem valMinAbs_eq_zero {n : ℕ} (x : ZMod n) : x.valMinAbs = 0 ↔ x = 0 := by
  /-
    n : Nat
    x : ZMod n
    ⊢ Iff (Eq x.valMinAbs 0) (Eq x 0)
  -/
  cases' n with n
    /-
      case zero
      x : ZMod 0
      ⊢ Iff (Eq x.valMinAbs 0) (Eq x 0)
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case succ
    n : Nat
    x : ZMod (HAdd.hAdd n 1)
    ⊢ Iff (Eq x.valMinAbs 0) (Eq x 0)
  -/
  rw [← valMinAbs_zero n.succ]
  /-
    case succ
    n : Nat
    x : ZMod (HAdd.hAdd n 1)
    ⊢ Iff (Eq x.valMinAbs (ZMod.valMinAbs 0)) (Eq x 0)
  -/
  apply injective_valMinAbs.eq_iff
  /-
    🎉 no goals
  -/


theorem natCast_natAbs_valMinAbs {n : ℕ} [NeZero n] (a : ZMod n) :
    (a.valMinAbs.natAbs : ZMod n) = if a.val ≤ (n : ℕ) / 2 then a else -a := by
  have : (a.val : ℤ) - n ≤ 0 := by
    rw [sub_nonpos, Int.ofNat_le]
    exact a.val_le
  /-
    n : Nat
    inst✝ : NeZero n
    a : ZMod n
    this : LE.le (HSub.hSub ↑a.val ↑n) 0
    ⊢ Eq (↑a.valMinAbs.natAbs) (ite (LE.le a.val (HDiv.hDiv n 2)) a (Neg.neg a))
  -/
  rw [valMinAbs_def_pos]
  /-
    n : Nat
    inst✝ : NeZero n
    a : ZMod n
    this : LE.le (HSub.hSub ↑a.val ↑n) 0
    ⊢ Eq (↑(ite (LE.le a.val (HDiv.hDiv n 2)) (↑a.val) (HSub.hSub ↑a.val ↑n)).natA …
  -/
  split_ifs
    /-
      case pos
      n : Nat
      inst✝ : NeZero n
      a : ZMod n
      this : LE.le (HSub.hSub ↑a.val ↑n) 0
      h✝ : LE.le a.val (HDiv.hDiv n 2)
      ⊢ Eq (↑(↑a.val).natAbs) a
    -/
  · rw [Int.natAbs_ofNat, natCast_zmod_val]
    /-
      🎉 no goals
    -/
  · rw [← Int.cast_natCast, Int.ofNat_natAbs_of_nonpos this, Int.cast_neg, Int.cast_sub,
      Int.cast_natCast, Int.cast_natCast, natCast_self, sub_zero, natCast_zmod_val]


@[deprecated (since := "2024-04-17")]
alias nat_cast_natAbs_valMinAbs := natCast_natAbs_valMinAbs


theorem valMinAbs_neg_of_ne_half {n : ℕ} {a : ZMod n} (ha : 2 * a.val ≠ n) :
    (-a).valMinAbs = -a.valMinAbs := by
  /-
    n : Nat
    a : ZMod n
    ha : Ne (HMul.hMul 2 a.val) n
    ⊢ Eq (Neg.neg a).valMinAbs (Neg.neg a.valMinAbs)
  -/
  cases' eq_zero_or_neZero n with h h
    /-
      case inl
      n : Nat
      a : ZMod n
      ha : Ne (HMul.hMul 2 a.val) n
      h : Eq n 0
      ⊢ Eq (Neg.neg a).valMinAbs (Neg.neg a.valMinAbs)
    -/
  · subst h
    /-
      case inl
      a : ZMod 0
      ha : Ne (HMul.hMul 2 a.val) 0
      ⊢ Eq (Neg.neg a).valMinAbs (Neg.neg a.valMinAbs)
    -/
    rfl
    /-
      🎉 no goals
    -/
  /-
    case inr
    n : Nat
    a : ZMod n
    ha : Ne (HMul.hMul 2 a.val) n
    h : NeZero n
    ⊢ Eq (Neg.neg a).valMinAbs (Neg.neg a.valMinAbs)
  -/
  refine (valMinAbs_spec _ _).2 ⟨?_, ?_, ?_⟩
    /-
      case inr.refine_1
      n : Nat
      a : ZMod n
      ha : Ne (HMul.hMul 2 a.val) n
      h : NeZero n
      ⊢ Eq (Neg.neg a) ↑(Neg.neg a.valMinAbs)
    -/
  · rw [Int.cast_neg, coe_valMinAbs]
    /-
      🎉 no goals
    -/
    /-
      case inr.refine_2
      n : Nat
      a : ZMod n
      ha : Ne (HMul.hMul 2 a.val) n
      h : NeZero n
      ⊢ LT.lt (Neg.neg ↑n) (HMul.hMul (Neg.neg a.valMinAbs) 2)
    -/
  · rw [neg_mul, neg_lt_neg_iff]
    /-
      case inr.refine_2
      n : Nat
      a : ZMod n
      ha : Ne (HMul.hMul 2 a.val) n
      h : NeZero n
      ⊢ LT.lt (HMul.hMul a.valMinAbs 2) ↑n
    -/
    exact a.valMinAbs_mem_Ioc.2.lt_of_ne (mt a.valMinAbs_mul_two_eq_iff.1 ha)
    /-
      🎉 no goals
    -/
    /-
      case inr.refine_3
      n : Nat
      a : ZMod n
      ha : Ne (HMul.hMul 2 a.val) n
      h : NeZero n
      ⊢ LE.le (HMul.hMul (Neg.neg a.valMinAbs) 2) ↑n
    -/
  · linarith only [a.valMinAbs_mem_Ioc.1]
    /-
      🎉 no goals
    -/


@[simp]
theorem natAbs_valMinAbs_neg {n : ℕ} (a : ZMod n) : (-a).valMinAbs.natAbs = a.valMinAbs.natAbs := by
  /-
    n : Nat
    a : ZMod n
    ⊢ Eq (Neg.neg a).valMinAbs.natAbs a.valMinAbs.natAbs
  -/
  by_cases h2a : 2 * a.val = n
    /-
      case pos
      n : Nat
      a : ZMod n
      h2a : Eq (HMul.hMul 2 a.val) n
      ⊢ Eq (Neg.neg a).valMinAbs.natAbs a.valMinAbs.natAbs
    -/
  · rw [a.neg_eq_self_iff.2 (Or.inr h2a)]
    /-
      🎉 no goals
    -/
    /-
      case neg
      n : Nat
      a : ZMod n
      h2a : Not (Eq (HMul.hMul 2 a.val) n)
      ⊢ Eq (Neg.neg a).valMinAbs.natAbs a.valMinAbs.natAbs
    -/
  · rw [valMinAbs_neg_of_ne_half h2a, Int.natAbs_neg]
    /-
      🎉 no goals
    -/


theorem val_eq_ite_valMinAbs {n : ℕ} [NeZero n] (a : ZMod n) :
    (a.val : ℤ) = a.valMinAbs + if a.val ≤ n / 2 then 0 else n := by
  /-
    n : Nat
    inst✝ : NeZero n
    a : ZMod n
    ⊢ Eq (↑a.val) (HAdd.hAdd a.valMinAbs ↑(ite (LE.le a.val (HDiv.hDiv n 2)) 0 n))
  -/
  rw [valMinAbs_def_pos]
  /-
    n : Nat
    inst✝ : NeZero n
    a : ZMod n
    ⊢ Eq (↑a.val) (HAdd.hAdd (ite (LE.le a.val (HDiv.hDiv n 2)) (↑a.val) (HSub.hSu …
  -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> simp [add_zero, sub_add_cancel]
                /-
                  🎉 no goals
                -/


theorem prime_ne_zero (p q : ℕ) [hp : Fact p.Prime] [hq : Fact q.Prime] (hpq : p ≠ q) :
    (q : ZMod p) ≠ 0 := by
  rwa [← Nat.cast_zero, Ne, eq_iff_modEq_nat, Nat.modEq_zero_iff_dvd, ←
    hp.1.coprime_iff_not_dvd, Nat.coprime_primes hp.1 hq.1]


theorem valMinAbs_natAbs_eq_min {n : ℕ} [hpos : NeZero n] (a : ZMod n) :
    a.valMinAbs.natAbs = min a.val (n - a.val) := by
  /-
    n : Nat
    hpos : NeZero n
    a : ZMod n
    ⊢ Eq a.valMinAbs.natAbs (Min.min a.val (HSub.hSub n a.val))
  -/
  rw [valMinAbs_def_pos]
  /-
    n : Nat
    hpos : NeZero n
    a : ZMod n
    ⊢ Eq (ite (LE.le a.val (HDiv.hDiv n 2)) (↑a.val) (HSub.hSub ↑a.val ↑n)).natAbs …
  -/
  have := a.val_lt
  /-
    n : Nat
    hpos : NeZero n
    a : ZMod n
    this : LT.lt a.val n
    ⊢ Eq (ite (LE.le a.val (HDiv.hDiv n 2)) (↑a.val) (HSub.hSub ↑a.val ↑n)).natAbs …
  -/
  omega
  /-
    🎉 no goals
  -/


theorem valMinAbs_natCast_of_le_half (ha : a ≤ n / 2) : (a : ZMod n).valMinAbs = a := by
  /-
    n a : Nat
    ha : LE.le a (HDiv.hDiv n 2)
    ⊢ Eq (↑a).valMinAbs ↑a
  -/
  cases n
    /-
      case zero
      a : Nat
      ha : LE.le a (0 / 2)
      ⊢ Eq (↑a).valMinAbs ↑a
    -/
  · simp
    /-
      🎉 no goals
    -/
  · simp [valMinAbs_def_pos, val_natCast, Nat.mod_eq_of_lt (ha.trans_lt <| Nat.div_lt_self' _ 0),
      ha]


theorem valMinAbs_natCast_of_half_lt (ha : n / 2 < a) (ha' : a < n) :
    (a : ZMod n).valMinAbs = a - n := by
  /-
    n a : Nat
    ha : LT.lt (HDiv.hDiv n 2) a
    ha' : LT.lt a n
    ⊢ Eq (↑a).valMinAbs (HSub.hSub ↑a ↑n)
  -/
  cases n
    /-
      case zero
      a : Nat
      ha : LT.lt (0 / 2) a
      ha' : LT.lt a 0
      ⊢ Eq (↑a).valMinAbs (HSub.hSub ↑a ↑0)
    -/
  · cases not_lt_bot ha'
    /-
      🎉 no goals
    -/
    /-
      case succ
      a n✝ : Nat
      ha : LT.lt (HDiv.hDiv (HAdd.hAdd n✝ 1) 2) a
      ha' : LT.lt a (HAdd.hAdd n✝ 1)
      ⊢ Eq (↑a).valMinAbs (HSub.hSub ↑a ↑(HAdd.hAdd n✝ 1))
    -/
  · simp [valMinAbs_def_pos, val_natCast, Nat.mod_eq_of_lt ha', ha.not_le]
    /-
      🎉 no goals
    -/

-- Porting note: There was an extraneous `nat_` in the mathlib3 name

@[simp]
theorem valMinAbs_natCast_eq_self [NeZero n] : (a : ZMod n).valMinAbs = a ↔ a ≤ n / 2 := by
  /-
    n a : Nat
    inst✝ : NeZero n
    ⊢ Iff (Eq (↑a).valMinAbs ↑a) (LE.le a (HDiv.hDiv n 2))
  -/
  refine ⟨fun ha => ?_, valMinAbs_natCast_of_le_half⟩
  /-
    n a : Nat
    inst✝ : NeZero n
    ha : Eq (↑a).valMinAbs ↑a
    ⊢ LE.le a (HDiv.hDiv n 2)
  -/
  rw [← Int.natAbs_ofNat a, ← ha]
  /-
    n a : Nat
    inst✝ : NeZero n
    ha : Eq (↑a).valMinAbs ↑a
    ⊢ LE.le (↑a).valMinAbs.natAbs (HDiv.hDiv n 2)
  -/
  exact natAbs_valMinAbs_le a
  /-
    🎉 no goals
  -/


theorem natAbs_min_of_le_div_two (n : ℕ) (x y : ℤ) (he : (x : ZMod n) = y) (hl : x.natAbs ≤ n / 2) :
    x.natAbs ≤ y.natAbs := by
  /-
    n : Nat
    x y : Int
    he : Eq ↑x ↑y
    hl : LE.le x.natAbs (HDiv.hDiv n 2)
    ⊢ LE.le x.natAbs y.natAbs
  -/
  rw [intCast_eq_intCast_iff_dvd_sub] at he
  /-
    n : Nat
    x y : Int
    he : Dvd.dvd (↑n) (HSub.hSub y x)
    hl : LE.le x.natAbs (HDiv.hDiv n 2)
    ⊢ LE.le x.natAbs y.natAbs
  -/
  obtain ⟨m, he⟩ := he
  /-
    case intro
    n : Nat
    x y : Int
    hl : LE.le x.natAbs (HDiv.hDiv n 2)
    m : Int
    he : Eq (HSub.hSub y x) (HMul.hMul (↑n) m)
    ⊢ LE.le x.natAbs y.natAbs
  -/
  rw [sub_eq_iff_eq_add] at he
  /-
    case intro
    n : Nat
    x y : Int
    hl : LE.le x.natAbs (HDiv.hDiv n 2)
    m : Int
    he : Eq y (HAdd.hAdd (HMul.hMul (↑n) m) x)
    ⊢ LE.le x.natAbs y.natAbs
  -/
  subst he
  /-
    case intro
    n : Nat
    x : Int
    hl : LE.le x.natAbs (HDiv.hDiv n 2)
    m : Int
    ⊢ LE.le x.natAbs (HAdd.hAdd (HMul.hMul (↑n) m) x).natAbs
  -/
  obtain rfl | hm := eq_or_ne m 0
    /-
      case intro.inl
      n : Nat
      x : Int
      hl : LE.le x.natAbs (HDiv.hDiv n 2)
      ⊢ LE.le x.natAbs (HAdd.hAdd (HMul.hMul (↑n) 0) x).natAbs
    -/
  · rw [mul_zero, zero_add]
    /-
      🎉 no goals
    -/
  /-
    case intro.inr
    n : Nat
    x : Int
    hl : LE.le x.natAbs (HDiv.hDiv n 2)
    m : Int
    hm : Ne m 0
    ⊢ LE.le x.natAbs (HAdd.hAdd (HMul.hMul (↑n) m) x).natAbs
  -/
  apply hl.trans
  /-
    case intro.inr
    n : Nat
    x : Int
    hl : LE.le x.natAbs (HDiv.hDiv n 2)
    m : Int
    hm : Ne m 0
    ⊢ LE.le (HDiv.hDiv n 2) (HAdd.hAdd (HMul.hMul (↑n) m) x).natAbs
  -/
  rw [← add_le_add_iff_right x.natAbs]
  /-
    case intro.inr
    n : Nat
    x : Int
    hl : LE.le x.natAbs (HDiv.hDiv n 2)
    m : Int
    hm : Ne m 0
    ⊢ LE.le (HAdd.hAdd (HDiv.hDiv n 2) x.natAbs) (HAdd.hAdd (HAdd.hAdd (HMul.hMul  …
  -/
  refine le_trans (le_trans ((add_le_add_iff_left _).2 hl) ?_) (Int.natAbs_sub_le _ _)
  /-
    case intro.inr
    n : Nat
    x : Int
    hl : LE.le x.natAbs (HDiv.hDiv n 2)
    m : Int
    hm : Ne m 0
    ⊢ LE.le (HAdd.hAdd (HDiv.hDiv n 2) (HDiv.hDiv n 2)) (HSub.hSub (HAdd.hAdd (HMu …
  -/
  rw [add_sub_cancel_right, Int.natAbs_mul, Int.natAbs_ofNat]
  /-
    case intro.inr
    n : Nat
    x : Int
    hl : LE.le x.natAbs (HDiv.hDiv n 2)
    m : Int
    hm : Ne m 0
    ⊢ LE.le (HAdd.hAdd (HDiv.hDiv n 2) (HDiv.hDiv n 2)) (HMul.hMul n m.natAbs)
  -/
  refine le_trans ?_ (Nat.le_mul_of_pos_right _ <| Int.natAbs_pos.2 hm)
  /-
    case intro.inr
    n : Nat
    x : Int
    hl : LE.le x.natAbs (HDiv.hDiv n 2)
    m : Int
    hm : Ne m 0
    ⊢ LE.le (HAdd.hAdd (HDiv.hDiv n 2) (HDiv.hDiv n 2)) n
  -/
  rw [← mul_two]; apply Nat.div_mul_le_self
                  /-
                    🎉 no goals
                  -/


theorem natAbs_valMinAbs_add_le {n : ℕ} (a b : ZMod n) :
    (a + b).valMinAbs.natAbs ≤ (a.valMinAbs + b.valMinAbs).natAbs := by
  /-
    n : Nat
    a b : ZMod n
    ⊢ LE.le (HAdd.hAdd a b).valMinAbs.natAbs (HAdd.hAdd a.valMinAbs b.valMinAbs).n …
  -/
  cases' n with n
    /-
      case zero
      a b : ZMod 0
      ⊢ LE.le (HAdd.hAdd a b).valMinAbs.natAbs (HAdd.hAdd a.valMinAbs b.valMinAbs).n …
    -/
  · rfl
    /-
      🎉 no goals
    -/
  /-
    case succ
    n : Nat
    a b : ZMod (HAdd.hAdd n 1)
    ⊢ LE.le (HAdd.hAdd a b).valMinAbs.natAbs (HAdd.hAdd a.valMinAbs b.valMinAbs).n …
  -/
  apply natAbs_min_of_le_div_two n.succ
    /-
      case succ.he
      n : Nat
      a b : ZMod (HAdd.hAdd n 1)
      ⊢ Eq ↑(HAdd.hAdd a b).valMinAbs ↑(HAdd.hAdd a.valMinAbs b.valMinAbs)
    -/
  · simp_rw [Int.cast_add, coe_valMinAbs]
    /-
      🎉 no goals
    -/
    /-
      case succ.hl
      n : Nat
      a b : ZMod (HAdd.hAdd n 1)
      ⊢ LE.le (HAdd.hAdd a b).valMinAbs.natAbs (HDiv.hDiv n.succ 2)
    -/
  · apply natAbs_valMinAbs_le
    /-
      🎉 no goals
    -/


private theorem mul_inv_cancel_aux (a : ZMod p) (h : a ≠ 0) : a * a⁻¹ = 1 := by
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    a : ZMod p
    h : Ne a 0
    ⊢ Eq (HMul.hMul a (Inv.inv a)) 1
  -/
  obtain ⟨k, rfl⟩ := natCast_zmod_surjective a
  /-
    case intro
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    k : Nat
    h : Ne (↑k) 0
    ⊢ Eq (HMul.hMul (↑k) (Inv.inv ↑k)) 1
  -/
  apply coe_mul_inv_eq_one
  /-
    case intro.h
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    k : Nat
    h : Ne (↑k) 0
    ⊢ k.Coprime p
  -/
  apply Nat.Coprime.symm
  /-
    case intro.h.a
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    k : Nat
    h : Ne (↑k) 0
    ⊢ p.Coprime k
  -/
  rwa [Nat.Prime.coprime_iff_not_dvd Fact.out, ← CharP.cast_eq_zero_iff (ZMod p)]
  /-
    🎉 no goals
  -/


/-- Field structure on `ZMod p` if `p` is prime. -/
instance instField : Field (ZMod p) where
  mul_inv_cancel := mul_inv_cancel_aux p
  inv_zero := inv_zero p
  nnqsmul := _
  nnqsmul_def := fun _ _ => rfl
  qsmul := _
  qsmul_def := fun _ _ => rfl


/-- `ZMod p` is an integral domain when `p` is prime. -/
instance (p : ℕ) [hp : Fact p.Prime] : IsDomain (ZMod p) := by
  -- We need `cases p` here in order to resolve which `CommRing` instance is being used.
  /-
    m n✝ n a p✝ : Nat
    inst✝ : Fact (Nat.Prime p✝)
    p : Nat
    hp : Fact (Nat.Prime p)
    ⊢ IsDomain (ZMod p)
  -/
  cases p
    /-
      case zero
      m n✝ n a p : Nat
      inst✝ : Fact (Nat.Prime p)
      hp : Fact (Nat.Prime 0)
      ⊢ IsDomain (ZMod 0)
    -/
  · exact (Nat.not_prime_zero hp.out).elim
    /-
      🎉 no goals
    -/
  /-
    case succ
    m n✝¹ n a p : Nat
    inst✝ : Fact (Nat.Prime p)
    n✝ : Nat
    hp : Fact (Nat.Prime (HAdd.hAdd n✝ 1))
    ⊢ IsDomain (ZMod (HAdd.hAdd n✝ 1))
  -/
  exact @Field.isDomain (ZMod _) (inferInstanceAs (Field (ZMod _)))
  /-
    🎉 no goals
  -/


theorem RingHom.ext_zmod {n : ℕ} {R : Type*} [Semiring R] (f g : ZMod n →+* R) : f = g := by
  /-
    n : Nat
    R : Type u_1
    inst✝ : Semiring R
    f g : RingHom (ZMod n) R
    ⊢ Eq f g
  -/
  ext a
  /-
    case a
    n : Nat
    R : Type u_1
    inst✝ : Semiring R
    f g : RingHom (ZMod n) R
    a : ZMod n
    ⊢ Eq (f a) (g a)
  -/
  obtain ⟨k, rfl⟩ := ZMod.intCast_surjective a
  /-
    case a.intro
    n : Nat
    R : Type u_1
    inst✝ : Semiring R
    f g : RingHom (ZMod n) R
    k : Int
    ⊢ Eq (f ↑k) (g ↑k)
  -/
  let φ : ℤ →+* R := f.comp (Int.castRingHom (ZMod n))
  /-
    case a.intro
    n : Nat
    R : Type u_1
    inst✝ : Semiring R
    f g : RingHom (ZMod n) R
    k : Int
    φ : RingHom Int R := f.comp (Int.castRingHom (ZMod n))
    ⊢ Eq (f ↑k) (g ↑k)
  -/
  let ψ : ℤ →+* R := g.comp (Int.castRingHom (ZMod n))
  /-
    case a.intro
    n : Nat
    R : Type u_1
    inst✝ : Semiring R
    f g : RingHom (ZMod n) R
    k : Int
    φ : RingHom Int R := f.comp (Int.castRingHom (ZMod n))
    ψ : RingHom Int R := g.comp (Int.castRingHom (ZMod n))
    ⊢ Eq (f ↑k) (g ↑k)
  -/
  show φ k = ψ k
  /-
    case a.intro
    n : Nat
    R : Type u_1
    inst✝ : Semiring R
    f g : RingHom (ZMod n) R
    k : Int
    φ : RingHom Int R := f.comp (Int.castRingHom (ZMod n))
    ψ : RingHom Int R := g.comp (Int.castRingHom (ZMod n))
    ⊢ Eq (φ k) (ψ k)
  -/
  rw [φ.ext_int ψ]
  /-
    🎉 no goals
  -/


instance subsingleton_ringHom [Semiring R] : Subsingleton (ZMod n →+* R) :=
  ⟨RingHom.ext_zmod⟩


instance subsingleton_ringEquiv [Semiring R] : Subsingleton (ZMod n ≃+* R) :=
  ⟨fun f g => by
    /-
      n : Nat
      R : Type u_1
      inst✝ : Semiring R
      f g : RingEquiv (ZMod n) R
      ⊢ Eq f g
    -/
    rw [RingEquiv.coe_ringHom_inj_iff]
    /-
      n : Nat
      R : Type u_1
      inst✝ : Semiring R
      f g : RingEquiv (ZMod n) R
      ⊢ Eq ↑f ↑g
    -/
    apply RingHom.ext_zmod _ _⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem ringHom_map_cast [Ring R] (f : R →+* ZMod n) (k : ZMod n) : f (cast k) = k := by
  /-
    n : Nat
    R : Type u_1
    inst✝ : Ring R
    f : RingHom R (ZMod n)
    k : ZMod n
    ⊢ Eq (f k.cast) k
  -/
  cases n
    /-
      case zero
      R : Type u_1
      inst✝ : Ring R
      f : RingHom R (ZMod 0)
      k : ZMod 0
      ⊢ Eq (f k.cast) k
    -/
  · dsimp [ZMod, ZMod.cast] at f k ⊢; simp
                                      /-
                                        🎉 no goals
                                      -/
    /-
      case succ
      R : Type u_1
      inst✝ : Ring R
      n✝ : Nat
      f : RingHom R (ZMod (HAdd.hAdd n✝ 1))
      k : ZMod (HAdd.hAdd n✝ 1)
      ⊢ Eq (f k.cast) k
    -/
  · dsimp [ZMod, ZMod.cast] at f k ⊢
    /-
      case succ
      R : Type u_1
      inst✝ : Ring R
      n✝ : Nat
      f : RingHom R (Fin (HAdd.hAdd n✝ 1))
      k : Fin (HAdd.hAdd n✝ 1)
      ⊢ Eq (f ↑(ZMod.val k)) k
    -/
    erw [map_natCast, Fin.cast_val_eq_self]
    /-
      🎉 no goals
    -/


/-- Any ring homomorphism into `ZMod n` has a right inverse. -/
theorem ringHom_rightInverse [Ring R] (f : R →+* ZMod n) :
    Function.RightInverse (cast : ZMod n → R) f :=
  ringHom_map_cast f


/-- Any ring homomorphism into `ZMod n` is surjective. -/
theorem ringHom_surjective [Ring R] (f : R →+* ZMod n) : Function.Surjective f :=
  (ringHom_rightInverse f).surjective


@[simp]
lemma castHom_self : ZMod.castHom dvd_rfl (ZMod n) = RingHom.id (ZMod n) :=
  Subsingleton.elim _ _


@[simp]
lemma castHom_comp {m d : ℕ} (hm : n ∣ m) (hd : m ∣ d) :
    (castHom hm (ZMod n)).comp (castHom hd (ZMod m)) = castHom (dvd_trans hm hd) (ZMod n) :=
  RingHom.ext_zmod _ _


/-- The map from `ZMod n` induced by `f : ℤ →+ A` that maps `n` to `0`. -/
--@[simps] -- Porting note: removed, simplified LHS of `lift_coe` to something worse.
def lift : { f : ℤ →+ A // f n = 0 } ≃ (ZMod n →+ A) :=
  (Equiv.subtypeEquivRight <| by
        /-
          n : Nat
          R : Type u_1
          A : Type u_2
          inst✝ : AddGroup A
          ⊢ ∀ (x : AddMonoidHom Int A), Iff (Eq (x ↑n) 0) (LE.le (Int.castAddHom (ZMod n …
        -/
        intro f
        /-
          n : Nat
          R : Type u_1
          A : Type u_2
          inst✝ : AddGroup A
          f : AddMonoidHom Int A
          ⊢ Iff (Eq (f ↑n) 0) (LE.le (Int.castAddHom (ZMod n)).ker f.ker)
        -/
        rw [ker_intCastAddHom]
        /-
          n : Nat
          R : Type u_1
          A : Type u_2
          inst✝ : AddGroup A
          f : AddMonoidHom Int A
          ⊢ Iff (Eq (f ↑n) 0) (LE.le (AddSubgroup.zmultiples ↑n) f.ker)
        -/
        constructor
          /-
            case mp
            n : Nat
            R : Type u_1
            A : Type u_2
            inst✝ : AddGroup A
            f : AddMonoidHom Int A
            ⊢ Eq (f ↑n) 0 → LE.le (AddSubgroup.zmultiples ↑n) f.ker
          -/
        · rintro hf _ ⟨x, rfl⟩
          /-
            case mp.intro
            n : Nat
            R : Type u_1
            A : Type u_2
            inst✝ : AddGroup A
            f : AddMonoidHom Int A
            hf : Eq (f ↑n) 0
            x : Int
            ⊢ Membership.mem f.ker ((fun x => HSMul.hSMul x ↑n) x)
          -/
          simp only [f.map_zsmul, zsmul_zero, f.mem_ker, hf]
          /-
            🎉 no goals
          -/
          /-
            case mpr
            n : Nat
            R : Type u_1
            A : Type u_2
            inst✝ : AddGroup A
            f : AddMonoidHom Int A
            ⊢ LE.le (AddSubgroup.zmultiples ↑n) f.ker → Eq (f ↑n) 0
          -/
        · intro h
          /-
            case mpr
            n : Nat
            R : Type u_1
            A : Type u_2
            inst✝ : AddGroup A
            f : AddMonoidHom Int A
            h : LE.le (AddSubgroup.zmultiples ↑n) f.ker
            ⊢ Eq (f ↑n) 0
          -/
          exact h (AddSubgroup.mem_zmultiples _)).trans <|
          /-
            🎉 no goals
          -/
    (Int.castAddHom (ZMod n)).liftOfRightInverse cast intCast_zmod_cast


@[simp]
theorem lift_coe (x : ℤ) : lift n f (x : ZMod n) = f.val x :=
  AddMonoidHom.liftOfRightInverse_comp_apply _ _ (fun _ => intCast_zmod_cast _) _ _


theorem lift_castAddHom (x : ℤ) : lift n f (Int.castAddHom (ZMod n) x) = f.1 x :=
  AddMonoidHom.liftOfRightInverse_comp_apply _ _ (fun _ => intCast_zmod_cast _) _ _


@[simp]
theorem lift_comp_coe : ZMod.lift n f ∘ ((↑) : ℤ → _) = f :=
  funext <| lift_coe _ _


@[simp]
theorem lift_comp_castAddHom : (ZMod.lift n f).comp (Int.castAddHom (ZMod n)) = f :=
  AddMonoidHom.ext <| lift_castAddHom _ _


lemma lift_injective {f : {f : ℤ →+ A // f n = 0}} :
    Injective (lift n f) ↔ ∀ m, f.1 m = 0 → (m : ZMod n) = 0 := by
  simp only [← AddMonoidHom.ker_eq_bot_iff, eq_bot_iff, SetLike.le_def,
    ZMod.intCast_surjective.forall, ZMod.lift_coe, AddMonoidHom.mem_ker, AddSubgroup.mem_bot]


lemma zmod_smul_mem (hx : x ∈ K) : ∀ a : ZMod n, a • x ∈ K := by
  /-
    n : Nat
    S : Type u_1
    G : Type u_2
    inst✝³ : AddCommGroup G
    inst✝² : SetLike S G
    inst✝¹ : AddSubgroupClass S G
    K : S
    inst✝ : Module (ZMod n) G
    x : G
    hx : Membership.mem K x
    ⊢ ∀ (a : ZMod n), Membership.mem K (HSMul.hSMul a x)
  -/
  simpa [ZMod.forall, Int.cast_smul_eq_zsmul] using zsmul_mem hx
  /-
    🎉 no goals
  -/


/-- This cannot be made an instance because of the `[Module (ZMod n) G]` argument and the fact that
`n` only appears in the second argument of `SMulMemClass`, which is an `OutParam`. -/
lemma smulMemClass : SMulMemClass S (ZMod n) G where smul_mem _ _ {_x} hx := zmod_smul_mem hx _


instance instZModSMul : SMul (ZMod n) K where smul a x := ⟨a • x, zmod_smul_mem x.2 _⟩


@[simp, norm_cast] lemma coe_zmod_smul (a : ZMod n) (x : K) : ↑(a • x) = (a • x : G) := rfl


instance instZModModule : Module (ZMod n) K :=
  Subtype.coe_injective.module _ (AddSubmonoidClass.subtype K) coe_zmod_smul


lemma ZModModule.char_nsmul_eq_zero (x : G) : n • x = 0 := by
  /-
    n : Nat
    G : Type u_2
    inst✝¹ : AddCommGroup G
    inst✝ : Module (ZMod n) G
    x : G
    ⊢ Eq (HSMul.hSMul n x) 0
  -/
  simp [← Nat.cast_smul_eq_nsmul (ZMod n)]
  /-
    🎉 no goals
  -/


variable (G) in
lemma ZModModule.char_ne_one [Nontrivial G] : n ≠ 1 := by
  /-
    n : Nat
    G : Type u_2
    inst✝² : AddCommGroup G
    inst✝¹ : Module (ZMod n) G
    inst✝ : Nontrivial G
    ⊢ Ne n 1
  -/
  rintro rfl
  /-
    G : Type u_2
    inst✝² : AddCommGroup G
    inst✝¹ : Nontrivial G
    inst✝ : Module (ZMod 1) G
    ⊢ False
  -/
  obtain ⟨x, hx⟩ := exists_ne (0 : G)
  /-
    case intro
    G : Type u_2
    inst✝² : AddCommGroup G
    inst✝¹ : Nontrivial G
    inst✝ : Module (ZMod 1) G
    x : G
    hx : Ne x 0
    ⊢ False
  -/
  exact hx <| by simpa using char_nsmul_eq_zero 1 x
  /-
    🎉 no goals
  -/


variable (G) in
lemma ZModModule.two_le_char [NeZero n] [Nontrivial G] : 2 ≤ n := by
  /-
    n : Nat
    G : Type u_2
    inst✝³ : AddCommGroup G
    inst✝² : Module (ZMod n) G
    inst✝¹ : NeZero n
    inst✝ : Nontrivial G
    ⊢ LE.le 2 n
  -/
  have := NeZero.ne n
  /-
    n : Nat
    G : Type u_2
    inst✝³ : AddCommGroup G
    inst✝² : Module (ZMod n) G
    inst✝¹ : NeZero n
    inst✝ : Nontrivial G
    this : Ne n 0
    ⊢ LE.le 2 n
  -/
  have := char_ne_one n G
  /-
    n : Nat
    G : Type u_2
    inst✝³ : AddCommGroup G
    inst✝² : Module (ZMod n) G
    inst✝¹ : NeZero n
    inst✝ : Nontrivial G
    this✝ : Ne n 0
    this : Ne n 1
    ⊢ LE.le 2 n
  -/
  omega
  /-
    🎉 no goals
  -/


lemma ZModModule.periodicPts_add_left [NeZero n] (x : G) : periodicPts (x + ·) = .univ :=
  Set.eq_univ_of_forall fun y ↦ ⟨n, NeZero.pos n, by
    /-
      n : Nat
      G : Type u_2
      inst✝² : AddCommGroup G
      inst✝¹ : Module (ZMod n) G
      inst✝ : NeZero n
      x y : G
      ⊢ Function.IsPeriodicPt (fun x_1 => HAdd.hAdd x x_1) n y
    -/
    simpa [char_nsmul_eq_zero, IsPeriodicPt] using isFixedPt_id _⟩
    /-
      🎉 no goals
    -/


lemma ZModModule.add_self (x : G) : x + x = 0 := by
  /-
    G : Type u_2
    inst✝¹ : AddCommGroup G
    inst✝ : Module (ZMod 2) G
    x : G
    ⊢ Eq (HAdd.hAdd x x) 0
  -/
  simpa [two_nsmul] using char_nsmul_eq_zero 2 x
  /-
    🎉 no goals
  -/


                                                    /-
                                                      G : Type u_2
                                                      inst✝¹ : AddCommGroup G
                                                      inst✝ : Module (ZMod 2) G
                                                      x : G
                                                      ⊢ Eq (Neg.neg x) x
                                                    -/
lemma ZModModule.neg_eq_self (x : G) : -x = x := by simp [add_self, eq_comm, ← sub_eq_zero]
                                                    /-
                                                      🎉 no goals
                                                    -/


                                                            /-
                                                              G : Type u_2
                                                              inst✝¹ : AddCommGroup G
                                                              inst✝ : Module (ZMod 2) G
                                                              x y : G
                                                              ⊢ Eq (HSub.hSub x y) (HAdd.hAdd x y)
                                                            -/
lemma ZModModule.sub_eq_add (x y : G) : x - y = x + y := by simp [neg_eq_self, sub_eq_add_neg]
                                                            /-
                                                              🎉 no goals
                                                            -/


lemma ZModModule.add_add_add_cancel (x y z : G) : (x + y) + (y + z) = x + z := by
  /-
    G : Type u_2
    inst✝¹ : AddCommGroup G
    inst✝ : Module (ZMod 2) G
    x y z : G
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd x y) (HAdd.hAdd y z)) (HAdd.hAdd x z)
  -/
  simpa [sub_eq_add] using sub_add_sub_cancel x y z
  /-
    🎉 no goals
  -/


@[simp]
lemma nsmul_zmod_val_inv_nsmul (hn : (Nat.card α).Coprime n) (a : α) :
    n • (n⁻¹ : ZMod (Nat.card α)).val • a = a := by
  rw [← mul_nsmul', ← mod_natCard_nsmul, ← ZMod.val_natCast, Nat.cast_mul,
    ZMod.mul_val_inv hn.symm, ZMod.val_one_eq_one_mod, mod_natCard_nsmul, one_nsmul]


@[simp]
lemma zmod_val_inv_nsmul_nsmul (hn : (Nat.card α).Coprime n) (a : α) :
    (n⁻¹ : ZMod (Nat.card α)).val • n • a = a := by
  /-
    α : Type u_1
    inst✝ : AddGroup α
    n : Nat
    hn : (Nat.card α).Coprime n
    a : α
    ⊢ Eq (HSMul.hSMul (Inv.inv ↑n).val (HSMul.hSMul n a)) a
  -/
  rw [nsmul_left_comm, nsmul_zmod_val_inv_nsmul hn]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp) existing nsmul_zmod_val_inv_nsmul]
lemma pow_zmod_val_inv_pow (hn : (Nat.card α).Coprime n) (a : α) :
    (a ^ (n⁻¹ : ZMod (Nat.card α)).val) ^ n = a := by
  rw [← pow_mul', ← pow_mod_natCard, ← ZMod.val_natCast, Nat.cast_mul, ZMod.mul_val_inv hn.symm,
    ZMod.val_one_eq_one_mod, pow_mod_natCard, pow_one]


@[to_additive (attr := simp) existing zmod_val_inv_nsmul_nsmul]
lemma pow_pow_zmod_val_inv (hn : (Nat.card α).Coprime n) (a : α) :
                                                      /-
                                                        α : Type u_1
                                                        inst✝ : Group α
                                                        n : Nat
                                                        hn : (Nat.card α).Coprime n
                                                        a : α
                                                        ⊢ Eq (HPow.hPow (HPow.hPow a n) (Inv.inv ↑n).val) a
                                                      -/
    (a ^ n) ^ (n⁻¹ : ZMod (Nat.card α)).val = a := by rw [pow_right_comm, pow_zmod_val_inv_pow hn]
                                                      /-
                                                        🎉 no goals
                                                      -/


/-- The range of `(m * · + k)` on natural numbers is the set of elements `≥ k` in the
residue class of `k` mod `m`. -/
lemma Nat.range_mul_add (m k : ℕ) :
    Set.range (fun n : ℕ ↦ m * n + k) = {n : ℕ | (n : ZMod m) = k ∧ k ≤ n} := by
  /-
    m k : Nat
    ⊢ Eq (Set.range fun n => HAdd.hAdd (HMul.hMul m n) k) (setOf fun n => And (Eq  …
  -/
  ext n
  /-
    case h
    m k n : Nat
    ⊢ Iff (Membership.mem (Set.range fun n => HAdd.hAdd (HMul.hMul m n) k) n) (Mem …
  -/
  simp only [Set.mem_range, Set.mem_setOf_eq]
  /-
    case h
    m k n : Nat
    ⊢ Iff (Exists fun y => Eq (HAdd.hAdd (HMul.hMul m y) k) n) (And (Eq ↑n ↑k) (LE …
  -/
  conv => enter [1, 1, y]; rw [add_comm, eq_comm]
  /-
    case h
    m k n : Nat
    ⊢ Iff (Exists fun y => Eq n (HAdd.hAdd k (HMul.hMul m y))) (And (Eq ↑n ↑k) (LE …
  -/
  refine ⟨fun ⟨a, ha⟩ ↦ ⟨?_, le_iff_exists_add.mpr ⟨_, ha⟩⟩, fun ⟨H₁, H₂⟩ ↦ ?_⟩
    /-
      case h.refine_1
      m k n : Nat
      x✝ : Exists fun y => Eq n (HAdd.hAdd k (HMul.hMul m y))
      a : Nat
      ha : Eq n (HAdd.hAdd k (HMul.hMul m a))
      ⊢ Eq ↑n ↑k
    -/
  · simpa using congr_arg ((↑) : ℕ → ZMod m) ha
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2
      m k n : Nat
      x✝ : And (Eq ↑n ↑k) (LE.le k n)
      H₁ : Eq ↑n ↑k
      H₂ : LE.le k n
      ⊢ Exists fun y => Eq n (HAdd.hAdd k (HMul.hMul m y))
    -/
  · obtain ⟨a, ha⟩ := le_iff_exists_add.mp H₂
    /-
      case h.refine_2.intro
      m k n : Nat
      x✝ : And (Eq ↑n ↑k) (LE.le k n)
      H₁ : Eq ↑n ↑k
      H₂ : LE.le k n
      a : Nat
      ha : Eq n (HAdd.hAdd k a)
      ⊢ Exists fun y => Eq n (HAdd.hAdd k (HMul.hMul m y))
    -/
    simp only [ha, Nat.cast_add, add_right_eq_self, ZMod.natCast_zmod_eq_zero_iff_dvd] at H₁
    /-
      case h.refine_2.intro
      m k n : Nat
      x✝ : And (Eq ↑n ↑k) (LE.le k n)
      H₂ : LE.le k n
      a : Nat
      ha : Eq n (HAdd.hAdd k a)
      H₁ : Dvd.dvd m a
      ⊢ Exists fun y => Eq n (HAdd.hAdd k (HMul.hMul m y))
    -/
    obtain ⟨b, rfl⟩ := H₁
    /-
      case h.refine_2.intro.intro
      m k n : Nat
      x✝ : And (Eq ↑n ↑k) (LE.le k n)
      H₂ : LE.le k n
      b : Nat
      ha : Eq n (HAdd.hAdd k (HMul.hMul m b))
      ⊢ Exists fun y => Eq n (HAdd.hAdd k (HMul.hMul m y))
    -/
    exact ⟨b, ha⟩
    /-
      🎉 no goals
    -/


/-- Equivalence between `ℕ` and `ZMod N × ℕ`, sending `n` to `(n mod N, n / N)`. -/
def Nat.residueClassesEquiv (N : ℕ) [NeZero N] : ℕ ≃ ZMod N × ℕ where
  toFun n := (↑n, n / N)
  invFun p := p.1.val + N * p.2
                   /-
                     N : Nat
                     inst✝ : NeZero N
                     n : Nat
                     ⊢ Eq ((fun p => HAdd.hAdd p.1.val (HMul.hMul N p.2)) ((fun n => { fst := ↑n, s …
                   -/
  left_inv n := by simpa only [val_natCast] using mod_add_div n N
                   /-
                     🎉 no goals
                   -/
  right_inv p := by
    /-
      N : Nat
      inst✝ : NeZero N
      p : Prod (ZMod N) Nat
      ⊢ Eq ((fun n => { fst := ↑n, snd := HDiv.hDiv n N }) ((fun p => HAdd.hAdd p.1. …
    -/
    ext1
    · simp only [add_comm p.1.val, cast_add, cast_mul, natCast_self, zero_mul, natCast_val,
        cast_id', id_eq, zero_add]
    · simp only [add_comm p.1.val, mul_add_div (NeZero.pos _),
        (Nat.div_eq_zero_iff).2 <| .inr p.1.val_lt, add_zero]


