protected lemma le_rfl : a ≤ a := a.le_refl

protected lemma lt_or_lt_of_ne : a ≠ b → a < b ∨ b < a := Int.lt_or_gt_of_ne

                                                         /-
                                                           a b : Int
                                                           ⊢ Or (LT.lt a b) (LE.le b a)
                                                         -/
protected lemma lt_or_le (a b : ℤ) : a < b ∨ b ≤ a := by rw [← Int.not_lt]; exact em _
                                                                            /-
                                                                              🎉 no goals
                                                                            -/

protected lemma le_or_lt (a b : ℤ) : a ≤ b ∨ b < a := (b.lt_or_le a).symm

                                                 /-
                                                   a b : Int
                                                   ⊢ LT.lt a b → Not (LT.lt b a)
                                                 -/
protected lemma lt_asymm : a < b → ¬ b < a := by rw [Int.not_lt]; exact Int.le_of_lt
                                                                  /-
                                                                    🎉 no goals
                                                                  -/

                                                     /-
                                                       a b : Int
                                                       hab : Eq a b
                                                       ⊢ LE.le a b
                                                     -/
protected lemma le_of_eq (hab : a = b) : a ≤ b := by rw [hab]; exact Int.le_rfl
                                                               /-
                                                                 🎉 no goals
                                                               -/

protected lemma ge_of_eq (hab : a = b) : b ≤ a := Int.le_of_eq hab.symm

protected lemma le_antisymm_iff : a = b ↔ a ≤ b ∧ b ≤ a :=
  ⟨fun h ↦ ⟨Int.le_of_eq h, Int.ge_of_eq h⟩, fun h ↦ Int.le_antisymm h.1 h.2⟩

protected lemma le_iff_eq_or_lt : a ≤ b ↔ a = b ∨ a < b := by
  /-
    a b : Int
    ⊢ Iff (LE.le a b) (Or (Eq a b) (LT.lt a b))
  -/
  rw [Int.le_antisymm_iff, Int.lt_iff_le_not_le, ← and_or_left]; simp [em]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


                                                              /-
                                                                a b : Int
                                                                ⊢ Iff (LE.le a b) (Or (LT.lt a b) (Eq a b))
                                                              -/
protected lemma le_iff_lt_or_eq : a ≤ b ↔ a < b ∨ a = b := by rw [Int.le_iff_eq_or_lt, or_comm]
                                                              /-
                                                                🎉 no goals
                                                              -/


attribute [simp] natAbs_pos


protected lemma one_pos : 0 < (1 : Int) := Int.zero_lt_one


                                                /-
                                                  ⊢ Ne 1 0
                                                -/
protected lemma one_ne_zero : (1 : ℤ) ≠ 0 := by decide
                                                /-
                                                  🎉 no goals
                                                -/


protected lemma one_nonneg : 0 ≤ (1 : ℤ) := Int.le_of_lt Int.zero_lt_one


                                                                 /-
                                                                   n : Nat
                                                                   ⊢ Eq (HAdd.hAdd 0 ↑n) (Int.ofNat n)
                                                                 -/
lemma zero_le_ofNat (n : ℕ) : 0 ≤ ofNat n := @le.intro _ _ n (by rw [Int.zero_add]; rfl)
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


protected theorem neg_eq_neg {a b : ℤ} (h : -a = -b) : a = b := Int.neg_inj.1 h

-- We want to use these lemmas earlier than the lemmas simp can prove them with

@[simp, nolint simpNF]
protected lemma neg_pos : 0 < -a ↔ a < 0 := ⟨Int.neg_of_neg_pos, Int.neg_pos_of_neg⟩


@[simp, nolint simpNF]
protected lemma neg_nonneg : 0 ≤ -a ↔ a ≤ 0 := ⟨Int.nonpos_of_neg_nonneg, Int.neg_nonneg_of_nonpos⟩


@[simp, nolint simpNF]
protected lemma neg_neg_iff_pos : -a < 0 ↔ 0 < a := ⟨Int.pos_of_neg_neg, Int.neg_neg_of_pos⟩


@[simp, nolint simpNF]
protected lemma neg_nonpos_iff_nonneg : -a ≤ 0 ↔ 0 ≤ a :=
  ⟨Int.nonneg_of_neg_nonpos, Int.neg_nonpos_of_nonneg⟩


@[simp, nolint simpNF]
protected lemma sub_pos : 0 < a - b ↔ b < a := ⟨Int.lt_of_sub_pos, Int.sub_pos_of_lt⟩


@[simp, nolint simpNF]
protected lemma sub_nonneg : 0 ≤ a - b ↔ b ≤ a := ⟨Int.le_of_sub_nonneg, Int.sub_nonneg_of_le⟩


instance instNontrivial : Nontrivial ℤ := ⟨⟨0, 1, Int.zero_ne_one⟩⟩


protected theorem ofNat_add_out (m n : ℕ) : ↑m + ↑n = (↑(m + n) : ℤ) := rfl


protected theorem ofNat_mul_out (m n : ℕ) : ↑m * ↑n = (↑(m * n) : ℤ) := rfl


protected theorem ofNat_add_one_out (n : ℕ) : ↑n + (1 : ℤ) = ↑(succ n) := rfl


@[simp] lemma ofNat_injective : Function.Injective ofNat := @Int.ofNat.inj


@[simp] lemma ofNat_eq_natCast (n : ℕ) : Int.ofNat n = n := rfl


@[deprecated ofNat_eq_natCast (since := "2024-03-24")]
protected lemma natCast_eq_ofNat (n : ℕ) : ↑n = Int.ofNat n := rfl


@[norm_cast] lemma natCast_inj {m n : ℕ} : (m : ℤ) = (n : ℤ) ↔ m = n := ofNat_inj


@[simp, norm_cast] lemma natAbs_cast (n : ℕ) : natAbs ↑n = n := rfl


@[norm_cast]
protected lemma natCast_sub {n m : ℕ} : n ≤ m → (↑(m - n) : ℤ) = ↑m - ↑n := ofNat_sub

-- We want to use this lemma earlier than the lemmas simp can prove it with

                                                                                 /-
                                                                                   n : Nat
                                                                                   ⊢ Iff (Eq (↑n) 0) (Eq n 0)
                                                                                 -/
@[simp, nolint simpNF] lemma natCast_eq_zero {n : ℕ} : (n : ℤ) = 0 ↔ n = 0 := by omega
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


                                                          /-
                                                            n : Nat
                                                            ⊢ Iff (Ne (↑n) 0) (Ne n 0)
                                                          -/
lemma natCast_ne_zero {n : ℕ} : (n : ℤ) ≠ 0 ↔ n ≠ 0 := by omega
                                                          /-
                                                            🎉 no goals
                                                          -/


                                                                  /-
                                                                    n : Nat
                                                                    ⊢ Iff (Ne (↑n) 0) (LT.lt 0 n)
                                                                  -/
lemma natCast_ne_zero_iff_pos {n : ℕ} : (n : ℤ) ≠ 0 ↔ 0 < n := by omega
                                                                  /-
                                                                    🎉 no goals
                                                                  -/

-- We want to use this lemma earlier than the lemmas simp can prove it with

                                                                             /-
                                                                               n : Nat
                                                                               ⊢ Iff (LT.lt 0 ↑n) (LT.lt 0 n)
                                                                             -/
@[simp, nolint simpNF] lemma natCast_pos {n : ℕ} : (0 : ℤ) < n ↔ 0 < n := by omega
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


lemma natCast_succ_pos (n : ℕ) : 0 < (n.succ : ℤ) := natCast_pos.2 n.succ_pos

-- We want to use this lemma earlier than the lemmas simp can prove it with

                                                                                    /-
                                                                                      n : Nat
                                                                                      ⊢ Iff (LE.le (↑n) 0) (Eq n 0)
                                                                                    -/
@[simp, nolint simpNF] lemma natCast_nonpos_iff {n : ℕ} : (n : ℤ) ≤ 0 ↔ n = 0 := by omega
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


lemma natCast_nonneg (n : ℕ) : 0 ≤ (n : ℤ) := ofNat_le.2 (Nat.zero_le _)


@[simp] lemma sign_natCast_add_one (n : ℕ) : sign (n + 1) = 1 := rfl


@[simp, norm_cast] lemma cast_id {n : ℤ} : Int.cast n = n := rfl


protected lemma two_mul : ∀ n : ℤ, 2 * n = n + n
                  /-
                    n : Nat
                    ⊢ Eq (HMul.hMul 2 ↑n) (HAdd.hAdd ↑n ↑n)
                  -/
  | (n : ℕ) => by norm_cast; exact n.two_mul
                             /-
                               🎉 no goals
                             -/
  | -[n+1] => by
    /-
      n : Nat
      ⊢ Eq (HMul.hMul 2 (Int.negSucc n)) (HAdd.hAdd (Int.negSucc n) (Int.negSucc n))
    -/
    change (2 : ℕ) * (_ : ℤ) = _
    /-
      n : Nat
      ⊢ Eq (HMul.hMul (↑2) (Int.negSucc n)) (HAdd.hAdd (Int.negSucc n) (Int.negSucc  …
    -/
    rw [Int.ofNat_mul_negSucc, Nat.two_mul, ofNat_add, Int.neg_add]
    /-
      n : Nat
      ⊢ Eq (HAdd.hAdd (Neg.neg ↑n.succ) (Neg.neg ↑n.succ)) (HAdd.hAdd (Int.negSucc n …
    -/
    rfl
    /-
      🎉 no goals
    -/


protected lemma mul_le_mul_iff_of_pos_right (ha : 0 < a) : b * a ≤ c * a ↔ b ≤ c :=
  ⟨(le_of_mul_le_mul_right · ha), (Int.mul_le_mul_of_nonneg_right · (Int.le_of_lt ha))⟩


protected lemma mul_nonneg_iff_of_pos_right (hb : 0 < b) : 0 ≤ a * b ↔ 0 ≤ a := by
  /-
    a b : Int
    hb : LT.lt 0 b
    ⊢ Iff (LE.le 0 (HMul.hMul a b)) (LE.le 0 a)
  -/
  simpa using (Int.mul_le_mul_iff_of_pos_right hb : 0 * b ≤ a * b ↔ 0 ≤ a)
  /-
    🎉 no goals
  -/


/-- Immediate successor of an integer: `succ n = n + 1` -/
def succ (a : ℤ) := a + 1


/-- Immediate predecessor of an integer: `pred n = n - 1` -/
def pred (a : ℤ) := a - 1


lemma natCast_succ (n : ℕ) : (Nat.succ n : ℤ) = Int.succ n := rfl


lemma pred_succ (a : ℤ) : pred (succ a) = a := Int.add_sub_cancel _ _


lemma succ_pred (a : ℤ) : succ (pred a) = a := Int.sub_add_cancel _ _


lemma neg_succ (a : ℤ) : -succ a = pred (-a) := Int.neg_add


                                                        /-
                                                          a : Int
                                                          ⊢ Eq (Neg.neg a.succ).succ (Neg.neg a)
                                                        -/
lemma succ_neg_succ (a : ℤ) : succ (-succ a) = -a := by rw [neg_succ, succ_pred]
                                                        /-
                                                          🎉 no goals
                                                        -/


lemma neg_pred (a : ℤ) : -pred a = succ (-a) := by
  /-
    a : Int
    ⊢ Eq (Neg.neg a.pred) (Neg.neg a).succ
  -/
  rw [← Int.neg_eq_comm.mp (neg_succ (-a)), Int.neg_neg]
  /-
    🎉 no goals
  -/


                                                        /-
                                                          a : Int
                                                          ⊢ Eq (Neg.neg a.pred).pred (Neg.neg a)
                                                        -/
lemma pred_neg_pred (a : ℤ) : pred (-pred a) = -a := by rw [neg_pred, pred_succ]
                                                        /-
                                                          🎉 no goals
                                                        -/


lemma pred_nat_succ (n : ℕ) : pred (Nat.succ n) = n := pred_succ n


lemma neg_nat_succ (n : ℕ) : -(Nat.succ n : ℤ) = pred (-n) := neg_succ n


lemma succ_neg_natCast_succ (n : ℕ) : succ (-Nat.succ n) = -n := succ_neg_succ n


@[norm_cast] lemma natCast_pred_of_pos {n : ℕ} (h : 0 < n) : ((n - 1 : ℕ) : ℤ) = (n : ℤ) - 1 := by
  /-
    n : Nat
    h : LT.lt 0 n
    ⊢ Eq (↑(HSub.hSub n 1)) (HSub.hSub (↑n) 1)
  -/
  cases n; cases h; simp [ofNat_succ]
                    /-
                      🎉 no goals
                    -/


                                              /-
                                                a : Int
                                                ⊢ LT.lt a a.succ
                                              -/
lemma lt_succ_self (a : ℤ) : a < succ a := by unfold succ; omega
                                                           /-
                                                             🎉 no goals
                                                           -/


                                              /-
                                                a : Int
                                                ⊢ LT.lt a.pred a
                                              -/
lemma pred_self_lt (a : ℤ) : pred a < a := by unfold pred; omega
                                                           /-
                                                             🎉 no goals
                                                           -/


                                                           /-
                                                             m n : Int
                                                             ⊢ Iff (LE.le m (HAdd.hAdd n 1)) (Or (LE.le m n) (Eq m (HAdd.hAdd n 1)))
                                                           -/
lemma le_add_one_iff : m ≤ n + 1 ↔ m ≤ n ∨ m = n + 1 := by omega
                                                           /-
                                                             🎉 no goals
                                                           -/


                                               /-
                                                 m n : Int
                                                 ⊢ Iff (LT.lt (HSub.hSub m 1) n) (LE.le m n)
                                               -/
lemma sub_one_lt_iff : m - 1 < n ↔ m ≤ n := by omega
                                               /-
                                                 🎉 no goals
                                               -/


                                               /-
                                                 m n : Int
                                                 ⊢ Iff (LE.le m (HSub.hSub n 1)) (LT.lt m n)
                                               -/
lemma le_sub_one_iff : m ≤ n - 1 ↔ m < n := by omega
                                               /-
                                                 🎉 no goals
                                               -/


protected lemma add_le_iff_le_sub : a + b ≤ c ↔ a ≤ c - b := add_le_iff_le_sub ..

protected lemma le_add_iff_sub_le : a ≤ b + c ↔ a - c ≤ b := le_add_iff_sub_le ..

protected lemma add_le_zero_iff_le_neg : a + b ≤ 0 ↔ a ≤ - b := add_le_zero_iff_le_neg ..

protected lemma add_le_zero_iff_le_neg' : a + b ≤ 0 ↔ b ≤ -a := add_le_zero_iff_le_neg' ..

protected lemma add_nonnneg_iff_neg_le : 0 ≤ a + b ↔ -b ≤ a := add_nonnneg_iff_neg_le ..

protected lemma add_nonnneg_iff_neg_le' : 0 ≤ a + b ↔ -a ≤ b := add_nonnneg_iff_neg_le' ..


@[elab_as_elim] protected lemma induction_on {p : ℤ → Prop} (i : ℤ)
    (hz : p 0) (hp : ∀ i : ℕ, p i → p (i + 1)) (hn : ∀ i : ℕ, p (-i) → p (-i - 1)) : p i := by
  induction i with
  | ofNat i =>
    induction i with
    | zero => exact hz
    | succ i ih => exact hp _ ih
  | negSucc i =>
    suffices ∀ n : ℕ, p (-n) from this (i + 1)
    intro n; induction n with
    | zero => simp [hz]
    | succ n ih => convert hn _ ih using 1; simp [ofNat_succ, Int.neg_add, Int.sub_eq_add_neg]


/-- Inductively define a function on `ℤ` by defining it at `b`, for the `succ` of a number greater
than `b`, and the `pred` of a number less than `b`. -/
@[elab_as_elim] protected def inductionOn' : C z :=
                                               /-
                                                 a b✝ c d m n : Int
                                                 C : Int → Sort u_1
                                                 z b : Int
                                                 H0 : C b
                                                 Hs : (k : Int) → LE.le b k → C k → C (HAdd.hAdd k 1)
                                                 Hp : (k : Int) → LE.le k b → C k → C (HSub.hSub k 1)
                                                 ⊢ Eq (HAdd.hAdd b (HSub.hSub z b)) z
                                               -/
  cast (congr_arg C <| show b + (z - b) = z by rw [Int.add_comm, z.sub_add_cancel b]) <|
                                               /-
                                                 🎉 no goals
                                               -/
  match z - b with
  | .ofNat n => pos n
  | .negSucc n => neg n
where
                  /-
                    a b✝ c d m n : Int
                    C : Int → Sort u_1
                    z b : Int
                    H0 : C b
                    Hs : (k : Int) → LE.le b k → C k → C (HAdd.hAdd k 1)
                    Hp : (k : Int) → LE.le k b → C k → C (HSub.hSub k 1)
                    ⊢ Eq (C b) (C (HAdd.hAdd b ↑0))
                  -/
  /-- The positive case of `Int.inductionOn'`. -/
                  /-
                    🎉 no goals
                  -/
                    /-
                      a b✝ c d m n✝ : Int
                      C : Int → Sort u_1
                      z b : Int
                      H0 : C b
                      Hs : (k : Int) → LE.le b k → C k → C (HAdd.hAdd k 1)
                      Hp : (k : Int) → LE.le k b → C k → C (HSub.hSub k 1)
                      n : Nat
                      ⊢ Eq (C (HAdd.hAdd (HAdd.hAdd b ↑n) 1)) (C (HAdd.hAdd b ↑(HAdd.hAdd n 1)))
                    -/
  pos : ∀ n : ℕ, C (b + n)
                                        /-
                                          🎉 no goals
                                        -/
  | 0 => cast (by erw [Int.add_zero]) H0
  | n+1 => cast (by rw [Int.add_assoc]; rfl) <|
    Hs _ (Int.le_add_of_nonneg_right (ofNat_nonneg _)) (pos n)

  /-- The negative case of `Int.inductionOn'`. -/
  neg : ∀ n : ℕ, C (b + -[n+1])
    /-
      a b✝ c d m n✝ : Int
      C : Int → Sort u_1
      z b : Int
      H0 : C b
      Hs : (k : Int) → LE.le b k → C k → C (HAdd.hAdd k 1)
      Hp : (k : Int) → LE.le k b → C k → C (HSub.hSub k 1)
      n : Nat
      ⊢ C (HAdd.hAdd b (Int.negSucc (HAdd.hAdd n 1)))
    -/
  | 0 => Hp _ Int.le_rfl H0
    /-
      a b✝ c d m n✝ : Int
      C : Int → Sort u_1
      z b : Int
      H0 : C b
      Hs : (k : Int) → LE.le b k → C k → C (HAdd.hAdd k 1)
      Hp : (k : Int) → LE.le k b → C k → C (HSub.hSub k 1)
      n : Nat
      ⊢ LT.lt (HAdd.hAdd b (Int.negSucc n)) b
    -/
  | n+1 => by
    /-
      a b✝ c d m n✝ : Int
      C : Int → Sort u_1
      z b : Int
      H0 : C b
      Hs : (k : Int) → LE.le b k → C k → C (HAdd.hAdd k 1)
      Hp : (k : Int) → LE.le k b → C k → C (HSub.hSub k 1)
      n : Nat
      ⊢ LT.lt (HAdd.hAdd b (Int.negSucc n)) (HAdd.hAdd b 0)
    -/
    refine cast (by rw [Int.add_sub_assoc]; rfl) (Hp _ (Int.le_of_lt ?_) (neg n))
                                  /-
                                    🎉 no goals
                                  -/
    conv => rhs; exact b.add_zero.symm
    rw [Int.add_lt_add_iff_left]; apply negSucc_lt_zero


lemma inductionOn'_self : b.inductionOn' b H0 Hs Hp = H0 :=
                                     /-
                                       C : Int → Sort u_1
                                       b : Int
                                       H0 : C b
                                       Hs : (k : Int) → LE.le b k → C k → C (HAdd.hAdd k 1)
                                       Hp : (k : Int) → LE.le k b → C k → C (HSub.hSub k 1)
                                       ⊢ HEq H0 (Int.inductionOn'.match_1 (fun x => (fun x => C x) (HAdd.hAdd b x)) ( …
                                     -/
  cast_eq_iff_heq.mpr <| .symm <| by rw [b.sub_self, ← cast_eq_iff_heq]; rfl
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


lemma inductionOn'_add_one (hz : b ≤ z) :
    (z + 1).inductionOn' b H0 Hs Hp = Hs z hz (z.inductionOn' b H0 Hs Hp) := by
  /-
    C : Int → Sort u_1
    z b : Int
    H0 : C b
    Hs : (k : Int) → LE.le b k → C k → C (HAdd.hAdd k 1)
    Hp : (k : Int) → LE.le k b → C k → C (HSub.hSub k 1)
    hz : LE.le b z
    ⊢ Eq ((HAdd.hAdd z 1).inductionOn' b H0 Hs Hp) (Hs z hz (z.inductionOn' b H0 H …
  -/
  apply cast_eq_iff_heq.mpr
  /-
    C : Int → Sort u_1
    z b : Int
    H0 : C b
    Hs : (k : Int) → LE.le b k → C k → C (HAdd.hAdd k 1)
    Hp : (k : Int) → LE.le k b → C k → C (HSub.hSub k 1)
    hz : LE.le b z
    ⊢ HEq (Int.inductionOn'.match_1 (fun x => (fun x => C x) (HAdd.hAdd b x)) (HSu …
  -/
  lift z - b to ℕ using Int.sub_nonneg.mpr hz with zb hzb
  /-
    case intro
    C : Int → Sort u_1
    z b : Int
    H0 : C b
    Hs : (k : Int) → LE.le b k → C k → C (HAdd.hAdd k 1)
    Hp : (k : Int) → LE.le k b → C k → C (HSub.hSub k 1)
    hz : LE.le b z
    zb : Nat
    hzb : Eq (↑zb) (HSub.hSub z b)
    ⊢ HEq (Int.inductionOn'.match_1 (fun x => C (HAdd.hAdd b x)) (HSub.hSub (HAdd. …
  -/
  rw [show z + 1 - b = zb + 1 by omega]
  /-
    case intro
    C : Int → Sort u_1
    z b : Int
    H0 : C b
    Hs : (k : Int) → LE.le b k → C k → C (HAdd.hAdd k 1)
    Hp : (k : Int) → LE.le k b → C k → C (HSub.hSub k 1)
    hz : LE.le b z
    zb : Nat
    hzb : Eq (↑zb) (HSub.hSub z b)
    ⊢ HEq (Int.inductionOn'.match_1 (fun x => C (HAdd.hAdd b x)) (HAdd.hAdd (↑zb)  …
  -/
  have : b + zb = z := by omega
  /-
    case intro
    C : Int → Sort u_1
    z b : Int
    H0 : C b
    Hs : (k : Int) → LE.le b k → C k → C (HAdd.hAdd k 1)
    Hp : (k : Int) → LE.le k b → C k → C (HSub.hSub k 1)
    hz : LE.le b z
    zb : Nat
    hzb : Eq (↑zb) (HSub.hSub z b)
    this : Eq (HAdd.hAdd b ↑zb) z
    ⊢ HEq (Int.inductionOn'.match_1 (fun x => C (HAdd.hAdd b x)) (HAdd.hAdd (↑zb)  …
  -/
  subst this
  /-
    case intro
    C : Int → Sort u_1
    b : Int
    H0 : C b
    Hs : (k : Int) → LE.le b k → C k → C (HAdd.hAdd k 1)
    Hp : (k : Int) → LE.le k b → C k → C (HSub.hSub k 1)
    zb : Nat
    hz : LE.le b (HAdd.hAdd b ↑zb)
    hzb : Eq (↑zb) (HSub.hSub (HAdd.hAdd b ↑zb) b)
    ⊢ HEq (Int.inductionOn'.match_1 (fun x => C (HAdd.hAdd b x)) (HAdd.hAdd (↑zb)  …
  -/
  convert cast_heq _ _
  /-
    case h.e'_4.h.e'_3
    C : Int → Sort u_1
    b : Int
    H0 : C b
    Hs : (k : Int) → LE.le b k → C k → C (HAdd.hAdd k 1)
    Hp : (k : Int) → LE.le k b → C k → C (HSub.hSub k 1)
    zb : Nat
    hz : LE.le b (HAdd.hAdd b ↑zb)
    hzb : Eq (↑zb) (HSub.hSub (HAdd.hAdd b ↑zb) b)
    ⊢ Eq ((HAdd.hAdd b ↑zb).inductionOn' b H0 Hs Hp) (Int.inductionOn'.pos b H0 Hs …
  -/
  rw [Int.inductionOn', cast_eq_iff_heq, ← hzb]
  /-
    🎉 no goals
  -/


lemma inductionOn'_sub_one (hz : z ≤ b) :
    (z - 1).inductionOn' b H0 Hs Hp = Hp z hz (z.inductionOn' b H0 Hs Hp) := by
  /-
    C : Int → Sort u_1
    z b : Int
    H0 : C b
    Hs : (k : Int) → LE.le b k → C k → C (HAdd.hAdd k 1)
    Hp : (k : Int) → LE.le k b → C k → C (HSub.hSub k 1)
    hz : LE.le z b
    ⊢ Eq ((HSub.hSub z 1).inductionOn' b H0 Hs Hp) (Hp z hz (z.inductionOn' b H0 H …
  -/
  apply cast_eq_iff_heq.mpr
  /-
    C : Int → Sort u_1
    z b : Int
    H0 : C b
    Hs : (k : Int) → LE.le b k → C k → C (HAdd.hAdd k 1)
    Hp : (k : Int) → LE.le k b → C k → C (HSub.hSub k 1)
    hz : LE.le z b
    ⊢ HEq (Int.inductionOn'.match_1 (fun x => (fun x => C x) (HAdd.hAdd b x)) (HSu …
  -/
  obtain ⟨n, hn⟩ := Int.eq_negSucc_of_lt_zero (show z - 1 - b < 0 by omega)
  /-
    case intro
    C : Int → Sort u_1
    z b : Int
    H0 : C b
    Hs : (k : Int) → LE.le b k → C k → C (HAdd.hAdd k 1)
    Hp : (k : Int) → LE.le k b → C k → C (HSub.hSub k 1)
    hz : LE.le z b
    n : Nat
    hn : Eq (HSub.hSub (HSub.hSub z 1) b) (Int.negSucc n)
    ⊢ HEq (Int.inductionOn'.match_1 (fun x => (fun x => C x) (HAdd.hAdd b x)) (HSu …
  -/
  rw [hn]
  /-
    case intro
    C : Int → Sort u_1
    z b : Int
    H0 : C b
    Hs : (k : Int) → LE.le b k → C k → C (HAdd.hAdd k 1)
    Hp : (k : Int) → LE.le k b → C k → C (HSub.hSub k 1)
    hz : LE.le z b
    n : Nat
    hn : Eq (HSub.hSub (HSub.hSub z 1) b) (Int.negSucc n)
    ⊢ HEq (Int.inductionOn'.match_1 (fun x => (fun x => C x) (HAdd.hAdd b x)) (Int …
  -/
  obtain _|n := n
    /-
      case intro.zero
      C : Int → Sort u_1
      z b : Int
      H0 : C b
      Hs : (k : Int) → LE.le b k → C k → C (HAdd.hAdd k 1)
      Hp : (k : Int) → LE.le k b → C k → C (HSub.hSub k 1)
      hz : LE.le z b
      hn : Eq (HSub.hSub (HSub.hSub z 1) b) (Int.negSucc 0)
      ⊢ HEq (Int.inductionOn'.match_1 (fun x => (fun x => C x) (HAdd.hAdd b x)) (Int …
    -/
  · change _ = -1 at hn
    /-
      case intro.zero
      C : Int → Sort u_1
      z b : Int
      H0 : C b
      Hs : (k : Int) → LE.le b k → C k → C (HAdd.hAdd k 1)
      Hp : (k : Int) → LE.le k b → C k → C (HSub.hSub k 1)
      hz : LE.le z b
      hn : Eq (HSub.hSub (HSub.hSub z 1) b) (-1)
      ⊢ HEq (Int.inductionOn'.match_1 (fun x => (fun x => C x) (HAdd.hAdd b x)) (Int …
    -/
    have : z = b := by omega
    /-
      case intro.zero
      C : Int → Sort u_1
      z b : Int
      H0 : C b
      Hs : (k : Int) → LE.le b k → C k → C (HAdd.hAdd k 1)
      Hp : (k : Int) → LE.le k b → C k → C (HSub.hSub k 1)
      hz : LE.le z b
      hn : Eq (HSub.hSub (HSub.hSub z 1) b) (-1)
      this : Eq z b
      ⊢ HEq (Int.inductionOn'.match_1 (fun x => (fun x => C x) (HAdd.hAdd b x)) (Int …
    -/
    subst this; rw [inductionOn'_self]; exact heq_of_eq rfl
                                        /-
                                          🎉 no goals
                                        -/
    /-
      case intro.succ
      C : Int → Sort u_1
      z b : Int
      H0 : C b
      Hs : (k : Int) → LE.le b k → C k → C (HAdd.hAdd k 1)
      Hp : (k : Int) → LE.le k b → C k → C (HSub.hSub k 1)
      hz : LE.le z b
      n : Nat
      hn : Eq (HSub.hSub (HSub.hSub z 1) b) (Int.negSucc (HAdd.hAdd n 1))
      ⊢ HEq (Int.inductionOn'.match_1 (fun x => (fun x => C x) (HAdd.hAdd b x)) (Int …
    -/
  · have : z = b + -[n+1] := by rw [Int.negSucc_eq] at hn ⊢; omega
    /-
      case intro.succ
      C : Int → Sort u_1
      z b : Int
      H0 : C b
      Hs : (k : Int) → LE.le b k → C k → C (HAdd.hAdd k 1)
      Hp : (k : Int) → LE.le k b → C k → C (HSub.hSub k 1)
      hz : LE.le z b
      n : Nat
      hn : Eq (HSub.hSub (HSub.hSub z 1) b) (Int.negSucc (HAdd.hAdd n 1))
      this : Eq z (HAdd.hAdd b (Int.negSucc n))
      ⊢ HEq (Int.inductionOn'.match_1 (fun x => (fun x => C x) (HAdd.hAdd b x)) (Int …
    -/
    subst this
    /-
      case intro.succ
      C : Int → Sort u_1
      b : Int
      H0 : C b
      Hs : (k : Int) → LE.le b k → C k → C (HAdd.hAdd k 1)
      Hp : (k : Int) → LE.le k b → C k → C (HSub.hSub k 1)
      n : Nat
      hz : LE.le (HAdd.hAdd b (Int.negSucc n)) b
      hn : Eq (HSub.hSub (HSub.hSub (HAdd.hAdd b (Int.negSucc n)) 1) b) (Int.negSucc …
      ⊢ HEq (Int.inductionOn'.match_1 (fun x => (fun x => C x) (HAdd.hAdd b x)) (Int …
    -/
    convert cast_heq _ _
    /-
      case h.e'_4.h.e'_3
      C : Int → Sort u_1
      b : Int
      H0 : C b
      Hs : (k : Int) → LE.le b k → C k → C (HAdd.hAdd k 1)
      Hp : (k : Int) → LE.le k b → C k → C (HSub.hSub k 1)
      n : Nat
      hz : LE.le (HAdd.hAdd b (Int.negSucc n)) b
      hn : Eq (HSub.hSub (HSub.hSub (HAdd.hAdd b (Int.negSucc n)) 1) b) (Int.negSucc …
      ⊢ Eq ((HAdd.hAdd b (Int.negSucc n)).inductionOn' b H0 Hs Hp) (Int.inductionOn' …
    -/
    rw [Int.inductionOn', cast_eq_iff_heq, show b + -[n+1] - b = -[n+1] by omega]
    /-
      🎉 no goals
    -/


/-- Inductively define a function on `ℤ` by defining it on `ℕ` and extending it from `n` to `-n`. -/
@[elab_as_elim] protected def negInduction {C : ℤ → Sort*} (nat : ∀ n : ℕ, C n)
    (neg : (∀ n : ℕ, C n) → ∀ n : ℕ, C (-n)) : ∀ n : ℤ, C n
  | .ofNat n => nat n
  | .negSucc n => neg nat <| n + 1


/-- See `Int.inductionOn'` for an induction in both directions. -/
protected lemma le_induction {P : ℤ → Prop} {m : ℤ} (h0 : P m)
    (h1 : ∀ n : ℤ, m ≤ n → P n → P (n + 1)) (n : ℤ) : m ≤ n → P n := by
  /-
    P : Int → Prop
    m : Int
    h0 : P m
    h1 : ∀ (n : Int), LE.le m n → P n → P (HAdd.hAdd n 1)
    n : Int
    ⊢ LE.le m n → P n
  -/
  refine Int.inductionOn' n m ?_ ?_ ?_
    /-
      case refine_1
      P : Int → Prop
      m : Int
      h0 : P m
      h1 : ∀ (n : Int), LE.le m n → P n → P (HAdd.hAdd n 1)
      n : Int
      ⊢ LE.le m m → P m
    -/
  · intro
    /-
      case refine_1
      P : Int → Prop
      m : Int
      h0 : P m
      h1 : ∀ (n : Int), LE.le m n → P n → P (HAdd.hAdd n 1)
      n : Int
      a✝ : LE.le m m
      ⊢ P m
    -/
    exact h0
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      P : Int → Prop
      m : Int
      h0 : P m
      h1 : ∀ (n : Int), LE.le m n → P n → P (HAdd.hAdd n 1)
      n : Int
      ⊢ ∀ (k : Int), LE.le m k → (LE.le m k → P k) → LE.le m (HAdd.hAdd k 1) → P (HA …
    -/
  · intro k hle hi _
    /-
      case refine_2
      P : Int → Prop
      m : Int
      h0 : P m
      h1 : ∀ (n : Int), LE.le m n → P n → P (HAdd.hAdd n 1)
      n k : Int
      hle : LE.le m k
      hi : LE.le m k → P k
      a✝ : LE.le m (HAdd.hAdd k 1)
      ⊢ P (HAdd.hAdd k 1)
    -/
    exact h1 k hle (hi hle)
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      P : Int → Prop
      m : Int
      h0 : P m
      h1 : ∀ (n : Int), LE.le m n → P n → P (HAdd.hAdd n 1)
      n : Int
      ⊢ ∀ (k : Int), LE.le k m → (LE.le m k → P k) → LE.le m (HSub.hSub k 1) → P (HS …
    -/
  · intro k hle _ hle'
    /-
      case refine_3
      P : Int → Prop
      m : Int
      h0 : P m
      h1 : ∀ (n : Int), LE.le m n → P n → P (HAdd.hAdd n 1)
      n k : Int
      hle : LE.le k m
      a✝ : LE.le m k → P k
      hle' : LE.le m (HSub.hSub k 1)
      ⊢ P (HSub.hSub k 1)
    -/
    omega
    /-
      🎉 no goals
    -/


/-- See `Int.inductionOn'` for an induction in both directions. -/
protected theorem le_induction_down {P : ℤ → Prop} {m : ℤ} (h0 : P m)
    (h1 : ∀ n : ℤ, n ≤ m → P n → P (n - 1)) (n : ℤ) : n ≤ m → P n :=
                                                           /-
                                                             P : Int → Prop
                                                             m : Int
                                                             h0 : P m
                                                             h1 : ∀ (n : Int), LE.le n m → P n → P (HSub.hSub n 1)
                                                             n k : Int
                                                             hle : LE.le m k
                                                             x✝ : LE.le k m → P k
                                                             hle' : LE.le (HAdd.hAdd k 1) m
                                                             ⊢ P (HAdd.hAdd k 1)
                                                           -/
  Int.inductionOn' n m (fun _ ↦ h0) (fun k hle _ hle' ↦ by omega)
                                                           /-
                                                             🎉 no goals
                                                           -/
    fun k hle hi _ ↦ h1 k hle (hi hle)


/-- A strong recursor for `Int` that specifies explicit values for integers below a threshold,
and is analogous to `Nat.strongRec` for integers on or above the threshold. -/
@[elab_as_elim] protected def strongRec (n : ℤ) : P n := by
  /-
    a b c d m n✝ : Int
    P : Int → Sort u_1
    lt : (n : Int) → LT.lt n m → P n
    ge : (n : Int) → GE.ge n m → ((k : Int) → LT.lt k n → P k) → P n
    n : Int
    ⊢ P n
  -/
  refine if hnm : n < m then lt n hnm else ge n (by omega) (n.inductionOn' m lt ?_ ?_)
    /-
      case refine_1
      a b c d m n✝ : Int
      P : Int → Sort u_1
      lt : (n : Int) → LT.lt n m → P n
      ge : (n : Int) → GE.ge n m → ((k : Int) → LT.lt k n → P k) → P n
      n : Int
      hnm : Not (LT.lt n m)
      ⊢ (k : Int) → LE.le m k → ((k_1 : Int) → LT.lt k_1 k → P k_1) → (k_1 : Int) →  …
    -/
  · intro _n _ ih l _
    /-
      case refine_1
      a b c d m n✝ : Int
      P : Int → Sort u_1
      lt : (n : Int) → LT.lt n m → P n
      ge : (n : Int) → GE.ge n m → ((k : Int) → LT.lt k n → P k) → P n
      n : Int
      hnm : Not (LT.lt n m)
      _n : Int
      a✝¹ : LE.le m _n
      ih : (k : Int) → LT.lt k _n → P k
      l : Int
      a✝ : LT.lt l (HAdd.hAdd _n 1)
      ⊢ P l
    -/
    exact if hlm : l < m then lt l hlm else ge l (by omega) fun k _ ↦ ih k (by omega)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      a b c d m n✝ : Int
      P : Int → Sort u_1
      lt : (n : Int) → LT.lt n m → P n
      ge : (n : Int) → GE.ge n m → ((k : Int) → LT.lt k n → P k) → P n
      n : Int
      hnm : Not (LT.lt n m)
      ⊢ (k : Int) → LE.le k m → ((k_1 : Int) → LT.lt k_1 k → P k_1) → (k_1 : Int) →  …
    -/
  · exact fun n _ hn l _ ↦ hn l (by omega)
    /-
      🎉 no goals
    -/


lemma strongRec_of_lt (hn : n < m) : m.strongRec lt ge n = lt n hn := dif_pos _


lemma strongRec_of_ge :
    ∀ hn : m ≤ n, m.strongRec lt ge n = ge n hn fun k _ ↦ m.strongRec lt ge k := by
  /-
    m n : Int
    P : Int → Sort u_1
    lt : (n : Int) → LT.lt n m → P n
    ge : (n : Int) → GE.ge n m → ((k : Int) → LT.lt k n → P k) → P n
    ⊢ ∀ (hn : LE.le m n), Eq (Int.strongRec lt ge n) (ge n hn fun k x => Int.stron …
  -/
  refine m.strongRec (fun n hnm hmn ↦ (Int.not_lt.mpr hmn hnm).elim) (fun n _ ih hn ↦ ?_) n
  /-
    m n✝ : Int
    P : Int → Sort u_1
    lt : (n : Int) → LT.lt n m → P n
    ge : (n : Int) → GE.ge n m → ((k : Int) → LT.lt k n → P k) → P n
    n : Int
    x✝ : GE.ge n m
    ih : ∀ (k : Int), LT.lt k n → ∀ (hn : LE.le m k), Eq (Int.strongRec lt ge k) ( …
    hn : LE.le m n
    ⊢ Eq (Int.strongRec lt ge n) (ge n hn fun k x => Int.strongRec lt ge k)
  -/
  rw [Int.strongRec, dif_neg (Int.not_lt.mpr hn)]
  /-
    m n✝ : Int
    P : Int → Sort u_1
    lt : (n : Int) → LT.lt n m → P n
    ge : (n : Int) → GE.ge n m → ((k : Int) → LT.lt k n → P k) → P n
    n : Int
    x✝ : GE.ge n m
    ih : ∀ (k : Int), LT.lt k n → ∀ (hn : LE.le m k), Eq (Int.strongRec lt ge k) ( …
    hn : LE.le m n
    ⊢ Eq (ge n ⋯ (n.inductionOn' m lt (fun _n a ih l a => dite (LT.lt l m) (fun hl …
  -/
  congr; revert ih
  /-
    case e_a
    m n✝ : Int
    P : Int → Sort u_1
    lt : (n : Int) → LT.lt n m → P n
    ge : (n : Int) → GE.ge n m → ((k : Int) → LT.lt k n → P k) → P n
    n : Int
    x✝ : GE.ge n m
    hn : LE.le m n
    ⊢ (∀ (k : Int), LT.lt k n → ∀ (hn : LE.le m k), Eq (Int.strongRec lt ge k) (ge …
  -/
  refine n.inductionOn' m (fun _ ↦ ?_) (fun k hmk ih' ih ↦ ?_) (fun k hkm ih' _ ↦ ?_) <;> ext l hl
    /-
      case e_a.refine_1.h.h
      m n✝ : Int
      P : Int → Sort u_1
      lt : (n : Int) → LT.lt n m → P n
      ge : (n : Int) → GE.ge n m → ((k : Int) → LT.lt k n → P k) → P n
      n : Int
      x✝¹ : GE.ge n m
      hn : LE.le m n
      x✝ : ∀ (k : Int), LT.lt k m → ∀ (hn : LE.le m k), Eq (Int.strongRec lt ge k) ( …
      l : Int
      hl : LT.lt l m
      ⊢ Eq (m.inductionOn' m lt (fun _n a ih l a => dite (LT.lt l m) (fun hlm => lt  …
    -/
  · rw [inductionOn'_self, strongRec_of_lt hl]
    /-
      🎉 no goals
    -/
    /-
      case e_a.refine_2.h.h
      m n✝ : Int
      P : Int → Sort u_1
      lt : (n : Int) → LT.lt n m → P n
      ge : (n : Int) → GE.ge n m → ((k : Int) → LT.lt k n → P k) → P n
      n : Int
      x✝ : GE.ge n m
      hn : LE.le m n
      k : Int
      hmk : LE.le m k
      ih' : (∀ (k_1 : Int), LT.lt k_1 k → ∀ (hn : LE.le m k_1), Eq (Int.strongRec lt …
      ih : ∀ (k_1 : Int), LT.lt k_1 (HAdd.hAdd k 1) → ∀ (hn : LE.le m k_1), Eq (Int. …
      l : Int
      hl : LT.lt l (HAdd.hAdd k 1)
      ⊢ Eq ((HAdd.hAdd k 1).inductionOn' m lt (fun _n a ih l a => dite (LT.lt l m) ( …
    -/
  · rw [inductionOn'_add_one hmk]; split_ifs with hlm
      /-
        case pos
        m n✝ : Int
        P : Int → Sort u_1
        lt : (n : Int) → LT.lt n m → P n
        ge : (n : Int) → GE.ge n m → ((k : Int) → LT.lt k n → P k) → P n
        n : Int
        x✝ : GE.ge n m
        hn : LE.le m n
        k : Int
        hmk : LE.le m k
        ih' : (∀ (k_1 : Int), LT.lt k_1 k → ∀ (hn : LE.le m k_1), Eq (Int.strongRec lt …
        ih : ∀ (k_1 : Int), LT.lt k_1 (HAdd.hAdd k 1) → ∀ (hn : LE.le m k_1), Eq (Int. …
        l : Int
        hl : LT.lt l (HAdd.hAdd k 1)
        hlm : LT.lt l m
        ⊢ Eq (lt l hlm) (Int.strongRec lt ge l)
      -/
    · rw [strongRec_of_lt hlm]
      /-
        🎉 no goals
      -/
      /-
        case neg
        m n✝ : Int
        P : Int → Sort u_1
        lt : (n : Int) → LT.lt n m → P n
        ge : (n : Int) → GE.ge n m → ((k : Int) → LT.lt k n → P k) → P n
        n : Int
        x✝ : GE.ge n m
        hn : LE.le m n
        k : Int
        hmk : LE.le m k
        ih' : (∀ (k_1 : Int), LT.lt k_1 k → ∀ (hn : LE.le m k_1), Eq (Int.strongRec lt …
        ih : ∀ (k_1 : Int), LT.lt k_1 (HAdd.hAdd k 1) → ∀ (hn : LE.le m k_1), Eq (Int. …
        l : Int
        hl : LT.lt l (HAdd.hAdd k 1)
        hlm : Not (LT.lt l m)
        ⊢ Eq (ge l ⋯ fun k_1 x => k.inductionOn' m lt (fun _n a ih l a => dite (LT.lt  …
      -/
    · rw [ih' fun l hl ↦ ih l (Int.lt_trans hl k.lt_succ), ih _ hl]
      /-
        🎉 no goals
      -/
    /-
      case e_a.refine_3.h.h
      m n✝ : Int
      P : Int → Sort u_1
      lt : (n : Int) → LT.lt n m → P n
      ge : (n : Int) → GE.ge n m → ((k : Int) → LT.lt k n → P k) → P n
      n : Int
      x✝¹ : GE.ge n m
      hn : LE.le m n
      k : Int
      hkm : LE.le k m
      ih' : (∀ (k_1 : Int), LT.lt k_1 k → ∀ (hn : LE.le m k_1), Eq (Int.strongRec lt …
      x✝ : ∀ (k_1 : Int), LT.lt k_1 (HSub.hSub k 1) → ∀ (hn : LE.le m k_1), Eq (Int. …
      l : Int
      hl : LT.lt l (HSub.hSub k 1)
      ⊢ Eq ((HSub.hSub k 1).inductionOn' m lt (fun _n a ih l a => dite (LT.lt l m) ( …
    -/
  · rw [inductionOn'_sub_one hkm, ih']
    /-
      case e_a.refine_3.h.h
      m n✝ : Int
      P : Int → Sort u_1
      lt : (n : Int) → LT.lt n m → P n
      ge : (n : Int) → GE.ge n m → ((k : Int) → LT.lt k n → P k) → P n
      n : Int
      x✝¹ : GE.ge n m
      hn : LE.le m n
      k : Int
      hkm : LE.le k m
      ih' : (∀ (k_1 : Int), LT.lt k_1 k → ∀ (hn : LE.le m k_1), Eq (Int.strongRec lt …
      x✝ : ∀ (k_1 : Int), LT.lt k_1 (HSub.hSub k 1) → ∀ (hn : LE.le m k_1), Eq (Int. …
      l : Int
      hl : LT.lt l (HSub.hSub k 1)
      ⊢ ∀ (k_1 : Int), LT.lt k_1 k → ∀ (hn : LE.le m k_1), Eq (Int.strongRec lt ge k …
    -/
    exact fun l hlk hml ↦ (Int.not_lt.mpr hkm <| Int.lt_of_le_of_lt hml hlk).elim
    /-
      🎉 no goals
    -/


@[simp] lemma natAbs_ofNat' (n : ℕ) : natAbs (ofNat n) = n := rfl


lemma natAbs_add_of_nonneg : ∀ {a b : Int}, 0 ≤ a → 0 ≤ b → natAbs (a + b) = natAbs a + natAbs b
  | ofNat _, ofNat _, _, _ => rfl


lemma natAbs_add_of_nonpos {a b : Int} (ha : a ≤ 0) (hb : b ≤ 0) :
    natAbs (a + b) = natAbs a + natAbs b := by
  /-
    a b : Int
    ha : LE.le a 0
    hb : LE.le b 0
    ⊢ Eq (HAdd.hAdd a b).natAbs (HAdd.hAdd a.natAbs b.natAbs)
  -/
  omega
  /-
    🎉 no goals
  -/


lemma natAbs_surjective : natAbs.Surjective := fun n => ⟨n, natAbs_ofNat n⟩


lemma natAbs_pow (n : ℤ) (k : ℕ) : Int.natAbs (n ^ k) = Int.natAbs n ^ k := by
  induction k with
  | zero => rfl
  | succ k ih => rw [Int.pow_succ, natAbs_mul, Nat.pow_succ, ih, Nat.mul_comm]


lemma pow_right_injective (h : 1 < a.natAbs) : ((a ^ ·) : ℕ → ℤ).Injective := by
  /-
    a : Int
    h : LT.lt 1 a.natAbs
    ⊢ Function.Injective fun x => HPow.hPow a x
  -/
  refine (?_ : (natAbs ∘ (a ^ · : ℕ → ℤ)).Injective).of_comp
  /-
    a : Int
    h : LT.lt 1 a.natAbs
    ⊢ Function.Injective (Function.comp Int.natAbs fun x => HPow.hPow a x)
  -/
  convert Nat.pow_right_injective h using 2
  /-
    case h.e'_3.h
    a : Int
    h : LT.lt 1 a.natAbs
    x✝ : Nat
    ⊢ Eq (Function.comp Int.natAbs (fun x => HPow.hPow a x) x✝) (HPow.hPow a.natAb …
  -/
  rw [Function.comp_apply, natAbs_pow]
  /-
    🎉 no goals
  -/


lemma natAbs_sq (x : ℤ) : (x.natAbs : ℤ) ^ 2 = x ^ 2 := by
  /-
    x : Int
    ⊢ Eq (HPow.hPow (↑x.natAbs) 2) (HPow.hPow x 2)
  -/
  simp [Int.pow_succ, Int.pow_zero, Int.natAbs_mul_self']
  /-
    🎉 no goals
  -/


alias natAbs_pow_two := natAbs_sq


theorem sign_mul_self_eq_natAbs : ∀ a : Int, sign a * a = natAbs a
  | 0      => rfl
  | Nat.succ _ => Int.one_mul _
  | -[_+1] => (Int.neg_eq_neg_one_mul _).symm


@[simp, norm_cast] lemma natCast_div (m n : ℕ) : ((m / n : ℕ) : ℤ) = m / n := rfl


lemma natCast_ediv (m n : ℕ) : ((m / n : ℕ) : ℤ) = ediv m n := rfl


lemma ediv_of_neg_of_pos {a b : ℤ} (Ha : a < 0) (Hb : 0 < b) : ediv a b = -((-a - 1) / b + 1) :=
  match a, b, eq_negSucc_of_lt_zero Ha, eq_succ_of_zero_lt Hb with
  | _, _, ⟨m, rfl⟩, ⟨n, rfl⟩ => by
    /-
      a b : Int
      m n : Nat
      Ha : LT.lt (Int.negSucc m) 0
      Hb : LT.lt 0 ↑n.succ
      ⊢ Eq ((Int.negSucc m).ediv ↑n.succ) (Neg.neg (HAdd.hAdd (HDiv.hDiv (HSub.hSub  …
    -/
    rw [show (- -[m+1] : ℤ) = (m + 1 : ℤ) by rfl]; rw [Int.add_sub_cancel]; rfl
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


@[simp, norm_cast] lemma natCast_mod (m n : ℕ) : (↑(m % n) : ℤ) = ↑m % ↑n := rfl


lemma add_emod_eq_add_mod_right {m n k : ℤ} (i : ℤ) (H : m % n = k % n) :
                                    /-
                                      m n k i : Int
                                      H : Eq (HMod.hMod m n) (HMod.hMod k n)
                                      ⊢ Eq (HMod.hMod (HAdd.hAdd m i) n) (HMod.hMod (HAdd.hAdd k i) n)
                                    -/
    (m + i) % n = (k + i) % n := by rw [← emod_add_emod, ← emod_add_emod k, H]
                                    /-
                                      🎉 no goals
                                    -/


                                                          /-
                                                            i : Int
                                                            ⊢ Eq (HMod.hMod (Neg.neg i) 2) (HMod.hMod i 2)
                                                          -/
@[simp] lemma neg_emod_two (i : ℤ) : -i % 2 = i % 2 := by omega
                                                          /-
                                                            🎉 no goals
                                                          -/


                                                                    /-
                                                                      n : Int
                                                                      ⊢ Or (Eq (HMod.hMod n 2) 0) (Eq (HMod.hMod n 2) 1)
                                                                    -/
lemma emod_two_eq_zero_or_one (n : ℤ) : n % 2 = 0 ∨ n % 2 = 1 := by omega
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


attribute [simp] Int.dvd_zero Int.dvd_mul_left Int.dvd_mul_right


protected lemma mul_dvd_mul : a ∣ b → c ∣ d → a * c ∣ b * d
                                   /-
                                     a b c d e : Int
                                     he : Eq b (HMul.hMul a e)
                                     f : Int
                                     hf : Eq d (HMul.hMul c f)
                                     ⊢ Eq (HMul.hMul b d) (HMul.hMul (HMul.hMul a c) (HMul.hMul e f))
                                   -/
  | ⟨e, he⟩, ⟨f, hf⟩ => ⟨e * f, by simp [he, hf, Int.mul_assoc, Int.mul_left_comm, Nat.mul_comm]⟩
                                   /-
                                     🎉 no goals
                                   -/


protected lemma mul_dvd_mul_left (a : ℤ) (h : b ∣ c) : a * b ∣ a * c := Int.mul_dvd_mul a.dvd_refl h


protected lemma mul_dvd_mul_right (a : ℤ) (h : b ∣ c) : b * a ∣ c * a :=
  Int.mul_dvd_mul h a.dvd_refl


lemma dvd_mul_of_div_dvd (h : b ∣ a) (hdiv : a / b ∣ c) : a ∣ b * c := by
  /-
    a b c : Int
    h : Dvd.dvd b a
    hdiv : Dvd.dvd (HDiv.hDiv a b) c
    ⊢ Dvd.dvd a (HMul.hMul b c)
  -/
  obtain ⟨e, rfl⟩ := hdiv
  /-
    case intro
    a b : Int
    h : Dvd.dvd b a
    e : Int
    ⊢ Dvd.dvd a (HMul.hMul b (HMul.hMul (HDiv.hDiv a b) e))
  -/
  rw [← Int.mul_assoc, Int.mul_comm _ (a / b), Int.ediv_mul_cancel h]
  /-
    case intro
    a b : Int
    h : Dvd.dvd b a
    e : Int
    ⊢ Dvd.dvd a (HMul.hMul a e)
  -/
  exact Int.dvd_mul_right a e
  /-
    🎉 no goals
  -/


@[simp] lemma div_dvd_iff_dvd_mul (h : b ∣ a) (hb : b ≠ 0) : a / b ∣ c ↔ a ∣ b * c :=
  exists_congr <| fun d ↦ by
  /-
    a b c : Int
    h : Dvd.dvd b a
    hb : Ne b 0
    d : Int
    ⊢ Iff (Eq c (HMul.hMul (HDiv.hDiv a b) d)) (Eq (HMul.hMul b c) (HMul.hMul a d))
  -/
  have := Int.dvd_trans (Int.dvd_mul_left _ _) (Int.mul_dvd_mul_left d h)
  rw [eq_comm, Int.mul_comm, ← Int.mul_ediv_assoc d h, Int.ediv_eq_iff_eq_mul_right hb this,
    Int.mul_comm, eq_comm]


lemma mul_dvd_of_dvd_div (hcb : c ∣ b) (h : a ∣ b / c) : c * a ∣ b :=
  have ⟨d, hd⟩ := h
         /-
           a b c : Int
           hcb : Dvd.dvd c b
           h : Dvd.dvd a (HDiv.hDiv b c)
           d : Int
           hd : Eq (HDiv.hDiv b c) (HMul.hMul a d)
           ⊢ Eq b (HMul.hMul (HMul.hMul c a) d)
         -/
  ⟨d, by simpa [Int.mul_comm, Int.mul_left_comm] using Int.eq_mul_of_ediv_eq_left hcb hd⟩
         /-
           🎉 no goals
         -/


lemma dvd_div_of_mul_dvd (h : a * b ∣ c) : b ∣ c / a := by
  /-
    a b c : Int
    h : Dvd.dvd (HMul.hMul a b) c
    ⊢ Dvd.dvd b (HDiv.hDiv c a)
  -/
  obtain rfl | ha := eq_or_ne a 0
    /-
      case inl
      b c : Int
      h : Dvd.dvd (HMul.hMul 0 b) c
      ⊢ Dvd.dvd b (HDiv.hDiv c 0)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      a b c : Int
      h : Dvd.dvd (HMul.hMul a b) c
      ha : Ne a 0
      ⊢ Dvd.dvd b (HDiv.hDiv c a)
    -/
  · obtain ⟨d, rfl⟩ := h
    /-
      case inr.intro
      a b : Int
      ha : Ne a 0
      d : Int
      ⊢ Dvd.dvd b (HDiv.hDiv (HMul.hMul (HMul.hMul a b) d) a)
    -/
    simp [Int.mul_assoc, ha]
    /-
      🎉 no goals
    -/


@[simp] lemma dvd_div_iff_mul_dvd (hbc : c ∣ b) : a ∣ b / c ↔ c * a ∣ b :=
  ⟨mul_dvd_of_dvd_div hbc, dvd_div_of_mul_dvd⟩


lemma ediv_dvd_ediv : ∀ {a b c : ℤ}, a ∣ b → b ∣ c → b / a ∣ c / a
  | a, _, _, ⟨b, rfl⟩, ⟨c, rfl⟩ =>
                          /-
                            a b c : Int
                            az : Eq a 0
                            ⊢ Dvd.dvd (HDiv.hDiv (HMul.hMul a b) a) (HDiv.hDiv (HMul.hMul (HMul.hMul a b)  …
                          -/
    if az : a = 0 then by simp [az]
                          /-
                            🎉 no goals
                          -/
    else by
      /-
        a b c : Int
        az : Not (Eq a 0)
        ⊢ Dvd.dvd (HDiv.hDiv (HMul.hMul a b) a) (HDiv.hDiv (HMul.hMul (HMul.hMul a b)  …
      -/
      rw [Int.mul_ediv_cancel_left _ az, Int.mul_assoc, Int.mul_ediv_cancel_left _ az]
      /-
        a b c : Int
        az : Not (Eq a 0)
        ⊢ Dvd.dvd b (HMul.hMul b c)
      -/
      apply Int.dvd_mul_right
      /-
        🎉 no goals
      -/


/-- If `n > 0` then `m` is not divisible by `n` iff it is between `n * k` and `n * (k + 1)`
  for some `k`. -/
lemma exists_lt_and_lt_iff_not_dvd (m : ℤ) (hn : 0 < n) :
    (∃ k, n * k < m ∧ m < n * (k + 1)) ↔ ¬n ∣ m := by
  /-
    n m : Int
    hn : LT.lt 0 n
    ⊢ Iff (Exists fun k => And (LT.lt (HMul.hMul n k) m) (LT.lt m (HMul.hMul n (HA …
  -/
  refine ⟨?_, fun h ↦ ?_⟩
    /-
      case refine_1
      n m : Int
      hn : LT.lt 0 n
      ⊢ (Exists fun k => And (LT.lt (HMul.hMul n k) m) (LT.lt m (HMul.hMul n (HAdd.h …
    -/
  · rintro ⟨k, h1k, h2k⟩ ⟨l, rfl⟩
    /-
      case refine_1.intro.intro.intro
      n : Int
      hn : LT.lt 0 n
      k l : Int
      h1k : LT.lt (HMul.hMul n k) (HMul.hMul n l)
      h2k : LT.lt (HMul.hMul n l) (HMul.hMul n (HAdd.hAdd k 1))
      ⊢ False
    -/
    replace h1k := lt_of_mul_lt_mul_left h1k (by omega)
    /-
      case refine_1.intro.intro.intro
      n : Int
      hn : LT.lt 0 n
      k l : Int
      h2k : LT.lt (HMul.hMul n l) (HMul.hMul n (HAdd.hAdd k 1))
      h1k : LT.lt k l
      ⊢ False
    -/
    replace h2k := lt_of_mul_lt_mul_left h2k (by omega)
    /-
      case refine_1.intro.intro.intro
      n : Int
      hn : LT.lt 0 n
      k l : Int
      h1k : LT.lt k l
      h2k : LT.lt l (HAdd.hAdd k 1)
      ⊢ False
    -/
    rw [Int.lt_add_one_iff, ← Int.not_lt] at h2k
    /-
      case refine_1.intro.intro.intro
      n : Int
      hn : LT.lt 0 n
      k l : Int
      h1k : LT.lt k l
      h2k : Not (LT.lt k l)
      ⊢ False
    -/
    exact h2k h1k
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      n m : Int
      hn : LT.lt 0 n
      h : Not (Dvd.dvd n m)
      ⊢ Exists fun k => And (LT.lt (HMul.hMul n k) m) (LT.lt m (HMul.hMul n (HAdd.hA …
    -/
  · rw [dvd_iff_emod_eq_zero, ← Ne] at h
    /-
      case refine_2
      n m : Int
      hn : LT.lt 0 n
      h : Ne (HMod.hMod m n) 0
      ⊢ Exists fun k => And (LT.lt (HMul.hMul n k) m) (LT.lt m (HMul.hMul n (HAdd.hA …
    -/
    rw [← emod_add_ediv m n]
    /-
      case refine_2
      n m : Int
      hn : LT.lt 0 n
      h : Ne (HMod.hMod m n) 0
      ⊢ Exists fun k => And (LT.lt (HMul.hMul n k) (HAdd.hAdd (HMod.hMod m n) (HMul. …
    -/
    refine ⟨m / n, Int.lt_add_of_pos_left _ ?_, ?_⟩
      /-
        case refine_2.refine_1
        n m : Int
        hn : LT.lt 0 n
        h : Ne (HMod.hMod m n) 0
        ⊢ LT.lt 0 (HMod.hMod m n)
      -/
    · have := emod_nonneg m (Int.ne_of_gt hn)
      /-
        case refine_2.refine_1
        n m : Int
        hn : LT.lt 0 n
        h : Ne (HMod.hMod m n) 0
        this : LE.le 0 (HMod.hMod m n)
        ⊢ LT.lt 0 (HMod.hMod m n)
      -/
      omega
      /-
        🎉 no goals
      -/
      /-
        case refine_2.refine_2
        n m : Int
        hn : LT.lt 0 n
        h : Ne (HMod.hMod m n) 0
        ⊢ LT.lt (HAdd.hAdd (HMod.hMod m n) (HMul.hMul n (HDiv.hDiv m n))) (HMul.hMul n …
      -/
    · rw [Int.add_comm _ (1 : ℤ), Int.mul_add, Int.mul_one]
      /-
        case refine_2.refine_2
        n m : Int
        hn : LT.lt 0 n
        h : Ne (HMod.hMod m n) 0
        ⊢ LT.lt (HAdd.hAdd (HMod.hMod m n) (HMul.hMul n (HDiv.hDiv m n))) (HAdd.hAdd n …
      -/
      exact Int.add_lt_add_right (emod_lt_of_pos _ hn) _
      /-
        🎉 no goals
      -/


@[norm_cast] lemma natCast_dvd_natCast {m n : ℕ} : (↑m : ℤ) ∣ ↑n ↔ m ∣ n where
  mp := by
    /-
      m n : Nat
      ⊢ Dvd.dvd ↑m ↑n → Dvd.dvd m n
    -/
    rintro ⟨a, h⟩
    /-
      case intro
      m n : Nat
      a : Int
      h : Eq (↑n) (HMul.hMul (↑m) a)
      ⊢ Dvd.dvd m n
    -/
    obtain rfl | hm := m.eq_zero_or_pos
      /-
        case intro.inl
        n : Nat
        a : Int
        h : Eq (↑n) (HMul.hMul (↑0) a)
        ⊢ Dvd.dvd 0 n
      -/
    · simpa using h
      /-
        🎉 no goals
      -/
    have ha : 0 ≤ a := Int.not_lt.1 fun ha ↦ by
      simpa [← h, Int.not_lt.2 (Int.natCast_nonneg _)]
        using Int.mul_neg_of_pos_of_neg (natCast_pos.2 hm) ha
    /-
      case intro.inr
      m n : Nat
      a : Int
      h : Eq (↑n) (HMul.hMul (↑m) a)
      hm : GT.gt m 0
      ha : LE.le 0 a
      ⊢ Dvd.dvd m n
    -/
    lift a to ℕ using ha
    /-
      case intro.inr.intro
      m n : Nat
      hm : GT.gt m 0
      a : Nat
      h : Eq (↑n) (HMul.hMul ↑m ↑a)
      ⊢ Dvd.dvd m n
    -/
    norm_cast at h
    /-
      case intro.inr.intro
      m n : Nat
      hm : GT.gt m 0
      a : Nat
      h : Eq n (HMul.hMul m a)
      ⊢ Dvd.dvd m n
    -/
    exact ⟨a, h⟩
    /-
      🎉 no goals
    -/
            /-
              m n : Nat
              ⊢ Dvd.dvd m n → Dvd.dvd ↑m ↑n
            -/
  mpr := by rintro ⟨a, rfl⟩; simp [Int.dvd_mul_right]
                             /-
                               🎉 no goals
                             -/


lemma natCast_dvd {m : ℕ} : (m : ℤ) ∣ n ↔ m ∣ n.natAbs := by
  /-
    n : Int
    m : Nat
    ⊢ Iff (Dvd.dvd (↑m) n) (Dvd.dvd m n.natAbs)
  -/
                                                /-
                                                  🎉 no goals
                                                -/
  obtain hn | hn := natAbs_eq n <;> rw [hn] <;> simp [← natCast_dvd_natCast, Int.dvd_neg]
                                                /-
                                                  🎉 no goals
                                                -/


lemma dvd_natCast {n : ℕ} : m ∣ (n : ℤ) ↔ m.natAbs ∣ n := by
  /-
    m : Int
    n : Nat
    ⊢ Iff (Dvd.dvd m ↑n) (Dvd.dvd m.natAbs n)
  -/
                                                /-
                                                  🎉 no goals
                                                -/
  obtain hn | hn := natAbs_eq m <;> rw [hn] <;> simp [← natCast_dvd_natCast, Int.neg_dvd]
                                                /-
                                                  🎉 no goals
                                                -/


lemma natAbs_ediv (a b : ℤ) (H : b ∣ a) : natAbs (a / b) = natAbs a / natAbs b := by
  /-
    a b : Int
    H : Dvd.dvd b a
    ⊢ Eq (HDiv.hDiv a b).natAbs (HDiv.hDiv a.natAbs b.natAbs)
  -/
  rcases Nat.eq_zero_or_pos (natAbs b) with (h | h)
    /-
      case inl
      a b : Int
      H : Dvd.dvd b a
      h : Eq b.natAbs 0
      ⊢ Eq (HDiv.hDiv a b).natAbs (HDiv.hDiv a.natAbs b.natAbs)
    -/
  · rw [natAbs_eq_zero.1 h]
    /-
      case inl
      a b : Int
      H : Dvd.dvd b a
      h : Eq b.natAbs 0
      ⊢ Eq (HDiv.hDiv a 0).natAbs (HDiv.hDiv a.natAbs (Int.natAbs 0))
    -/
    simp [Int.ediv_zero]
    /-
      🎉 no goals
    -/
  calc
    natAbs (a / b) = natAbs (a / b) * 1 := by rw [Nat.mul_one]
    _ = natAbs (a / b) * (natAbs b / natAbs b) := by rw [Nat.div_self h]
    _ = natAbs (a / b) * natAbs b / natAbs b := by rw [Nat.mul_div_assoc _ b.natAbs.dvd_refl]
    _ = natAbs (a / b * b) / natAbs b := by rw [natAbs_mul (a / b) b]
    _ = natAbs a / natAbs b := by rw [Int.ediv_mul_cancel H]


lemma dvd_of_mul_dvd_mul_left (ha : a ≠ 0) (h : a * m ∣ a * n) : m ∣ n := by
  /-
    a m n : Int
    ha : Ne a 0
    h : Dvd.dvd (HMul.hMul a m) (HMul.hMul a n)
    ⊢ Dvd.dvd m n
  -/
  obtain ⟨b, hb⟩ := h
  /-
    case intro
    a m n : Int
    ha : Ne a 0
    b : Int
    hb : Eq (HMul.hMul a n) (HMul.hMul (HMul.hMul a m) b)
    ⊢ Dvd.dvd m n
  -/
  rw [Int.mul_assoc, Int.mul_eq_mul_left_iff ha] at hb
  /-
    case intro
    a m n : Int
    ha : Ne a 0
    b : Int
    hb : Eq n (HMul.hMul m b)
    ⊢ Dvd.dvd m n
  -/
  exact ⟨_, hb⟩
  /-
    🎉 no goals
  -/


lemma dvd_of_mul_dvd_mul_right (ha : a ≠ 0) (h : m * a ∣ n * a) : m ∣ n :=
                                 /-
                                   a m n : Int
                                   ha : Ne a 0
                                   h : Dvd.dvd (HMul.hMul m a) (HMul.hMul n a)
                                   ⊢ Dvd.dvd (HMul.hMul a m) (HMul.hMul a n)
                                 -/
  dvd_of_mul_dvd_mul_left ha (by simpa [Int.mul_comm] using h)
                                 /-
                                   🎉 no goals
                                 -/


lemma eq_mul_div_of_mul_eq_mul_of_dvd_left (hb : b ≠ 0) (hbc : b ∣ c) (h : b * a = c * d) :
    a = c / b * d := by
  /-
    a b c d : Int
    hb : Ne b 0
    hbc : Dvd.dvd b c
    h : Eq (HMul.hMul b a) (HMul.hMul c d)
    ⊢ Eq a (HMul.hMul (HDiv.hDiv c b) d)
  -/
  obtain ⟨k, rfl⟩ := hbc
  /-
    case intro
    a b d : Int
    hb : Ne b 0
    k : Int
    h : Eq (HMul.hMul b a) (HMul.hMul (HMul.hMul b k) d)
    ⊢ Eq a (HMul.hMul (HDiv.hDiv (HMul.hMul b k) b) d)
  -/
  rw [Int.mul_ediv_cancel_left _ hb]
  /-
    case intro
    a b d : Int
    hb : Ne b 0
    k : Int
    h : Eq (HMul.hMul b a) (HMul.hMul (HMul.hMul b k) d)
    ⊢ Eq a (HMul.hMul k d)
  -/
  rwa [Int.mul_assoc, Int.mul_eq_mul_left_iff hb] at h
  /-
    🎉 no goals
  -/


/-- If an integer with larger absolute value divides an integer, it is zero. -/
lemma eq_zero_of_dvd_of_natAbs_lt_natAbs (hmn : m ∣ n) (hnm : natAbs n < natAbs m) : n = 0 := by
  /-
    m n : Int
    hmn : Dvd.dvd m n
    hnm : LT.lt n.natAbs m.natAbs
    ⊢ Eq n 0
  -/
  rw [← natAbs_dvd, ← dvd_natAbs, natCast_dvd_natCast] at hmn
  /-
    m n : Int
    hmn : Dvd.dvd m.natAbs n.natAbs
    hnm : LT.lt n.natAbs m.natAbs
    ⊢ Eq n 0
  -/
  rw [← natAbs_eq_zero]
  /-
    m n : Int
    hmn : Dvd.dvd m.natAbs n.natAbs
    hnm : LT.lt n.natAbs m.natAbs
    ⊢ Eq n.natAbs 0
  -/
  exact Nat.eq_zero_of_dvd_of_lt hmn hnm
  /-
    🎉 no goals
  -/


lemma eq_zero_of_dvd_of_nonneg_of_lt (hm : 0 ≤ m) (hmn : m < n) (hnm : n ∣ m) : m = 0 :=
  eq_zero_of_dvd_of_natAbs_lt_natAbs hnm (natAbs_lt_natAbs_of_nonneg_of_lt hm hmn)


/-- If two integers are congruent to a sufficiently large modulus, they are equal. -/
lemma eq_of_mod_eq_of_natAbs_sub_lt_natAbs {a b c : ℤ} (h1 : a % b = c)
    (h2 : natAbs (a - c) < natAbs b) : a = c :=
  Int.eq_of_sub_eq_zero (eq_zero_of_dvd_of_natAbs_lt_natAbs (dvd_sub_of_emod_eq h1) h2)


lemma ofNat_add_negSucc_of_ge {m n : ℕ} (h : n.succ ≤ m) :
    ofNat m + -[n+1] = ofNat (m - n.succ) := by
  rw [negSucc_eq, ofNat_eq_natCast, ofNat_eq_natCast, ← natCast_one, ← natCast_add,
    ← Int.sub_eq_add_neg, ← Int.natCast_sub h]


lemma natAbs_le_of_dvd_ne_zero (hmn : m ∣ n) (hn : n ≠ 0) : natAbs m ≤ natAbs n :=
  not_lt.mp (mt (eq_zero_of_dvd_of_natAbs_lt_natAbs hmn) hn)


@[deprecated (since := "2024-04-02")] alias coe_nat_dvd := natCast_dvd_natCast

@[deprecated (since := "2024-04-02")] alias coe_nat_dvd_right := dvd_natCast

@[deprecated (since := "2024-04-02")] alias coe_nat_dvd_left := natCast_dvd


lemma natAbs_eq_of_dvd_dvd (hmn : m ∣ n) (hnm : n ∣ m) : natAbs m = natAbs n :=
  Nat.dvd_antisymm (natAbs_dvd_natAbs.2 hmn) (natAbs_dvd_natAbs.2 hnm)


lemma ediv_dvd_of_dvd (hmn : m ∣ n) : n / m ∣ n := by
  /-
    m n : Int
    hmn : Dvd.dvd m n
    ⊢ Dvd.dvd (HDiv.hDiv n m) n
  -/
  obtain rfl | hm := eq_or_ne m 0
    /-
      case inl
      n : Int
      hmn : Dvd.dvd 0 n
      ⊢ Dvd.dvd (HDiv.hDiv n 0) n
    -/
  · simpa using hmn
    /-
      🎉 no goals
    -/
    /-
      case inr
      m n : Int
      hmn : Dvd.dvd m n
      hm : Ne m 0
      ⊢ Dvd.dvd (HDiv.hDiv n m) n
    -/
  · obtain ⟨a, ha⟩ := hmn
    /-
      case inr.intro
      m n : Int
      hm : Ne m 0
      a : Int
      ha : Eq n (HMul.hMul m a)
      ⊢ Dvd.dvd (HDiv.hDiv n m) n
    -/
    simp [ha, Int.mul_ediv_cancel_left _ hm, Int.dvd_mul_left]
    /-
      🎉 no goals
    -/


lemma le_iff_pos_of_dvd (ha : 0 < a) (hab : a ∣ b) : a ≤ b ↔ 0 < b :=
  ⟨Int.lt_of_lt_of_le ha, (Int.le_of_dvd · hab)⟩


lemma le_add_iff_lt_of_dvd_sub (ha : 0 < a) (hab : a ∣ c - b) : a + b ≤ c ↔ b < c := by
  /-
    a b c : Int
    ha : LT.lt 0 a
    hab : Dvd.dvd a (HSub.hSub c b)
    ⊢ Iff (LE.le (HAdd.hAdd a b) c) (LT.lt b c)
  -/
  rw [Int.add_le_iff_le_sub, ← Int.sub_pos, le_iff_pos_of_dvd ha hab]
  /-
    🎉 no goals
  -/


lemma sign_natCast_of_ne_zero {n : ℕ} (hn : n ≠ 0) : Int.sign n = 1 := sign_ofNat_of_nonzero hn


lemma sign_add_eq_of_sign_eq : ∀ {m n : ℤ}, m.sign = n.sign → (m + n).sign = n.sign := by
  /-
    ⊢ ∀ {m n : Int}, Eq m.sign n.sign → Eq (HAdd.hAdd m n).sign n.sign
  -/
  have : (1 : ℤ) ≠ -1 := by decide
  /-
    this : Ne 1 (-1)
    ⊢ ∀ {m n : Int}, Eq m.sign n.sign → Eq (HAdd.hAdd m n).sign n.sign
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
  rintro ((_ | m) | m) ((_ | n) | n) <;> simp [this, this.symm, Int.negSucc_add_negSucc]
                                         /-
                                           🎉 no goals
                                         -/
  /-
    case ofNat.succ.ofNat.succ
    this : Ne 1 (-1)
    m n : Nat
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (↑m) 1) (HAdd.hAdd (↑n) 1)).sign 1
  -/
  rw [Int.sign_eq_one_iff_pos]
  /-
    case ofNat.succ.ofNat.succ
    this : Ne 1 (-1)
    m n : Nat
    ⊢ LT.lt 0 (HAdd.hAdd (HAdd.hAdd (↑m) 1) (HAdd.hAdd (↑n) 1))
  -/
  omega
  /-
    🎉 no goals
  -/


@[simp] lemma toNat_natCast (n : ℕ) : toNat ↑n = n := rfl


@[simp] lemma toNat_natCast_add_one {n : ℕ} : ((n : ℤ) + 1).toNat = n + 1 := rfl


@[simp] lemma toNat_le {n : ℕ} : toNat m ≤ n ↔ m ≤ n := by
  /-
    m : Int
    n : Nat
    ⊢ Iff (LE.le m.toNat n) (LE.le m ↑n)
  -/
  rw [ofNat_le.symm, toNat_eq_max, Int.max_le]; exact and_iff_left (ofNat_zero_le _)
                                                /-
                                                  🎉 no goals
                                                -/


@[simp]
                                                         /-
                                                           n : Int
                                                           m : Nat
                                                           ⊢ Iff (LT.lt m n.toNat) (LT.lt (↑m) n)
                                                         -/
lemma lt_toNat {m : ℕ} : m < toNat n ↔ (m : ℤ) < n := by rw [← Int.not_le, ← Nat.not_le, toNat_le]
                                                         /-
                                                           🎉 no goals
                                                         -/


lemma toNat_le_toNat {a b : ℤ} (h : a ≤ b) : toNat a ≤ toNat b := by
  /-
    a b : Int
    h : LE.le a b
    ⊢ LE.le a.toNat b.toNat
  -/
  rw [toNat_le]; exact Int.le_trans h (self_le_toNat b)
                 /-
                   🎉 no goals
                 -/


lemma toNat_lt_toNat {a b : ℤ} (hb : 0 < b) : toNat a < toNat b ↔ a < b where
             /-
               a b : Int
               hb : LT.lt 0 b
               h : LT.lt a.toNat b.toNat
               ⊢ LT.lt a b
             -/
  mp h := by cases a; exacts [lt_toNat.1 h, Int.lt_trans (neg_of_sign_eq_neg_one rfl) hb]
                      /-
                        🎉 no goals
                      -/
              /-
                a b : Int
                hb : LT.lt 0 b
                h : LT.lt a b
                ⊢ LT.lt a.toNat b.toNat
              -/
  mpr h := by rw [lt_toNat]; cases a; exacts [h, hb]
                                      /-
                                        🎉 no goals
                                      -/


lemma lt_of_toNat_lt {a b : ℤ} (h : toNat a < toNat b) : a < b :=
  (toNat_lt_toNat <| lt_toNat.1 <| Nat.lt_of_le_of_lt (Nat.zero_le _) h).1 h


@[simp] lemma toNat_pred_coe_of_pos {i : ℤ} (h : 0 < i) : ((i.toNat - 1 : ℕ) : ℤ) = i - 1 := by
  /-
    i : Int
    h : LT.lt 0 i
    ⊢ Eq (↑(HSub.hSub i.toNat 1)) (HSub.hSub i 1)
  -/
  simp only [lt_toNat, Nat.cast_ofNat_Int, h, natCast_pred_of_pos, Int.le_of_lt h, toNat_of_nonneg]
  /-
    🎉 no goals
  -/


@[simp] lemma toNat_eq_zero : ∀ {n : ℤ}, n.toNat = 0 ↔ n ≤ 0
                  /-
                    n : Nat
                    ⊢ Iff (Eq (↑n).toNat 0) (LE.le (↑n) 0)
                  -/
  | (n : ℕ) => by simp
                  /-
                    🎉 no goals
                  -/
                 /-
                   n : Nat
                   ⊢ Iff (Eq (Int.negSucc n).toNat 0) (LE.le (Int.negSucc n) 0)
                 -/
  | -[n+1] => by simpa [toNat] using Int.le_of_lt (negSucc_lt_zero n)
                 /-
                   🎉 no goals
                 -/


theorem toNat_sub_of_le {a b : ℤ} (h : b ≤ a) : (toNat (a - b) : ℤ) = a - b :=
  Int.toNat_of_nonneg (Int.sub_nonneg_of_le h)


@[deprecated (since := "2024-04-05")] alias coe_nat_pos := natCast_pos

@[deprecated (since := "2024-04-05")] alias coe_nat_succ_pos := natCast_succ_pos


lemma toNat_lt' {n : ℕ} (hn : n ≠ 0) : m.toNat < n ↔ m < n := by
  /-
    m : Int
    n : Nat
    hn : Ne n 0
    ⊢ Iff (LT.lt m.toNat n) (LT.lt m ↑n)
  -/
  rw [← toNat_lt_toNat, toNat_natCast]; omega
                                        /-
                                          🎉 no goals
                                        -/


/-- The modulus of an integer by another as a natural. Uses the E-rounding convention. -/
def natMod (m n : ℤ) : ℕ := (m % n).toNat


lemma natMod_lt {n : ℕ} (hn : n ≠ 0) : m.natMod n < n :=
                                             /-
                                               m : Int
                                               n : Nat
                                               hn : Ne n 0
                                               ⊢ LT.lt 0 ↑n
                                             -/
  (toNat_lt' hn).2 <| emod_lt_of_pos _ <| by omega
                                             /-
                                               🎉 no goals
                                             -/


@[deprecated (since := "2024-05-25")] alias coe_nat_pow := natCast_pow

-- Porting note: this was added in an ad hoc port for use in `Tactic/NormNum/Basic`

@[simp] lemma pow_eq (m : ℤ) (n : ℕ) : m.pow n = m ^ n := rfl


@[deprecated (since := "2024-04-02")] alias ofNat_eq_cast := ofNat_eq_natCast

@[deprecated (since := "2024-04-02")] alias cast_eq_cast_iff_Nat := natCast_inj

@[deprecated (since := "2024-04-02")] alias coe_nat_sub := Int.natCast_sub

@[deprecated (since := "2024-04-02")] alias coe_nat_nonneg := natCast_nonneg

@[deprecated (since := "2024-04-02")] alias sign_coe_add_one := sign_natCast_add_one

@[deprecated (since := "2024-04-02")] alias nat_succ_eq_int_succ := natCast_succ

@[deprecated (since := "2024-04-02")] alias succ_neg_nat_succ := succ_neg_natCast_succ

@[deprecated (since := "2024-04-02")] alias coe_pred_of_pos := natCast_pred_of_pos

@[deprecated (since := "2024-04-02")] alias coe_nat_div := natCast_div

@[deprecated (since := "2024-04-02")] alias coe_nat_ediv := natCast_ediv

@[deprecated (since := "2024-04-02")] alias sign_coe_nat_of_nonzero := sign_natCast_of_ne_zero

@[deprecated (since := "2024-04-02")] alias toNat_coe_nat := toNat_natCast

@[deprecated (since := "2024-04-02")] alias toNat_coe_nat_add_one := toNat_natCast_add_one


