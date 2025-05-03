/-- Euler's totient function. This counts the number of naturals strictly less than `n` which are
coprime with `n`. -/
def totient (n : ℕ) : ℕ := #{a ∈ range n | n.Coprime a}


@[inherit_doc]
scoped notation "φ" => Nat.totient


@[simp]
theorem totient_zero : φ 0 = 0 :=
  rfl


@[simp]
theorem totient_one : φ 1 = 1 := rfl


theorem totient_eq_card_coprime (n : ℕ) : φ n = #{a ∈ range n | n.Coprime a} := rfl


/-- A characterisation of `Nat.totient` that avoids `Finset`. -/
theorem totient_eq_card_lt_and_coprime (n : ℕ) : φ n = Nat.card { m | m < n ∧ n.Coprime m } := by
  let e : { m | m < n ∧ n.Coprime m } ≃ {x ∈ range n | n.Coprime x} :=
    { toFun := fun m => ⟨m, by simpa only [Finset.mem_filter, Finset.mem_range] using m.property⟩
      invFun := fun m => ⟨m, by simpa only [Finset.mem_filter, Finset.mem_range] using m.property⟩
      left_inv := fun m => by simp only [Subtype.coe_mk, Subtype.coe_eta]
      right_inv := fun m => by simp only [Subtype.coe_mk, Subtype.coe_eta] }
  /-
    n : Nat
    e : Equiv (↑(setOf fun m => And (LT.lt m n) (n.Coprime m))) (Subtype fun x =>  …
    ⊢ Eq n.totient (Nat.card ↑(setOf fun m => And (LT.lt m n) (n.Coprime m)))
  -/
  rw [totient_eq_card_coprime, card_congr e, card_eq_fintype_card, Fintype.card_coe]
  /-
    🎉 no goals
  -/


theorem totient_le (n : ℕ) : φ n ≤ n :=
  ((range n).card_filter_le _).trans_eq (card_range n)


theorem totient_lt (n : ℕ) (hn : 1 < n) : φ n < n :=
                                         /-
                                           n : Nat
                                           hn : LT.lt 1 n
                                           ⊢ And (Membership.mem (Finset.range n) 0) (Not (n.Coprime 0))
                                         -/
  (card_lt_card (filter_ssubset.2 ⟨0, by simp [hn.ne', pos_of_gt hn]⟩)).trans_eq (card_range n)
                                         /-
                                           🎉 no goals
                                         -/


@[simp]
theorem totient_eq_zero : ∀ {n : ℕ}, φ n = 0 ↔ n = 0
            /-
              ⊢ Iff (Eq (Nat.totient 0) 0) (Eq 0 0)
            -/
  | 0 => by decide
            /-
              🎉 no goals
            -/
  | n + 1 =>
                                               /-
                                                 n : Nat
                                                 this : Exists fun x => And (LT.lt x (HAdd.hAdd n 1)) (Eq ((HAdd.hAdd n 1).gcd  …
                                                 ⊢ Iff (Eq (HAdd.hAdd n 1).totient 0) (Eq (HAdd.hAdd n 1) 0)
                                               -/
                                          /-
                                            n : Nat
                                            ⊢ Eq ((HAdd.hAdd n 1).gcd (HMod.hMod 1 (HAdd.hAdd n 1))) 1
                                          -/
    suffices ∃ x < n + 1, (n + 1).gcd x = 1 by simpa [totient, filter_eq_empty_iff]
                                          /-
                                            🎉 no goals
                                          -/
                                               /-
                                                 🎉 no goals
                                               -/
    ⟨1 % (n + 1), mod_lt _ n.succ_pos, by rw [gcd_comm, ← gcd_rec, gcd_one_right]⟩


                                                            /-
                                                              n : Nat
                                                              ⊢ Iff (LT.lt 0 n.totient) (LT.lt 0 n)
                                                            -/
@[simp] theorem totient_pos {n : ℕ} : 0 < φ n ↔ 0 < n := by simp [pos_iff_ne_zero]
                                                            /-
                                                              🎉 no goals
                                                            -/


instance neZero_totient {n : ℕ} [NeZero n] : NeZero n.totient :=
  ⟨(totient_pos.mpr <| NeZero.pos n).ne'⟩


theorem filter_coprime_Ico_eq_totient (a n : ℕ) :
    #{x ∈ Ico n (n + a) | a.Coprime x} = totient a := by
  /-
    a n : Nat
    ⊢ Eq (Finset.filter (fun x => a.Coprime x) (Finset.Ico n (HAdd.hAdd n a))).car …
  -/
  rw [totient, filter_Ico_card_eq_of_periodic, count_eq_card_filter_range]
  /-
    case pp
    a n : Nat
    ⊢ Function.Periodic (fun x => a.Coprime x) a
  -/
  exact periodic_coprime a
  /-
    🎉 no goals
  -/


theorem Ico_filter_coprime_le {a : ℕ} (k n : ℕ) (a_pos : 0 < a) :
    #{x ∈ Ico k (k + n) | a.Coprime x} ≤ totient a * (n / a + 1) := by
  /-
    a k n : Nat
    a_pos : LT.lt 0 a
    ⊢ LE.le (Finset.filter (fun x => a.Coprime x) (Finset.Ico k (HAdd.hAdd k n))). …
  -/
  conv_lhs => rw [← Nat.mod_add_div n a]
  /-
    a k n : Nat
    a_pos : LT.lt 0 a
    ⊢ LE.le (Finset.filter (fun x => a.Coprime x) (Finset.Ico k (HAdd.hAdd k (HAdd …
  -/
  induction' n / a with i ih
    /-
      case zero
      a k n : Nat
      a_pos : LT.lt 0 a
      ⊢ LE.le (Finset.filter (fun x => a.Coprime x) (Finset.Ico k (HAdd.hAdd k (HAdd …
    -/
  · rw [← filter_coprime_Ico_eq_totient a k]
    simp only [add_zero, mul_one, mul_zero, le_of_lt (mod_lt n a_pos),
      zero_add]
    /-
      case zero
      a k n : Nat
      a_pos : LT.lt 0 a
      ⊢ LE.le (Finset.filter (fun x => a.Coprime x) (Finset.Ico k (HAdd.hAdd k (HMod …
    -/
    gcongr
    /-
      case zero.a.h
      a k n : Nat
      a_pos : LT.lt 0 a
      ⊢ HasSubset.Subset (Finset.Ico k (HAdd.hAdd k (HMod.hMod n a))) (Finset.Ico k  …
    -/
    exact Ico_subset_Ico rfl.le (add_le_add_left (le_of_lt (mod_lt n a_pos)) k)
    /-
      🎉 no goals
    -/
  /-
    case succ
    a k n : Nat
    a_pos : LT.lt 0 a
    i : Nat
    ih : LE.le (Finset.filter (fun x => a.Coprime x) (Finset.Ico k (HAdd.hAdd k (H …
    ⊢ LE.le (Finset.filter (fun x => a.Coprime x) (Finset.Ico k (HAdd.hAdd k (HAdd …
  -/
  simp only [mul_succ]
  /-
    case succ
    a k n : Nat
    a_pos : LT.lt 0 a
    i : Nat
    ih : LE.le (Finset.filter (fun x => a.Coprime x) (Finset.Ico k (HAdd.hAdd k (H …
    ⊢ LE.le (Finset.filter (fun x => a.Coprime x) (Finset.Ico k (HAdd.hAdd k (HAdd …
  -/
  simp_rw [← add_assoc] at ih ⊢
  calc
    #{x ∈ Ico k (k + n % a + a * i + a) | a.Coprime x}
      = #{x ∈ Ico k (k + n % a + a * i) ∪
        Ico (k + n % a + a * i) (k + n % a + a * i + a) | a.Coprime x} := by
      congr
      rw [Ico_union_Ico_eq_Ico]
      · rw [add_assoc]
        exact le_self_add
      exact le_self_add
    _ ≤ #{x ∈ Ico k (k + n % a + a * i) | a.Coprime x} + a.totient := by
      rw [filter_union, ← filter_coprime_Ico_eq_totient a (k + n % a + a * i)]
      apply card_union_le
    _ ≤ a.totient * i + a.totient + a.totient := add_le_add_right ih (totient a)


/-- Note this takes an explicit `Fintype ((ZMod n)ˣ)` argument to avoid trouble with instance
diamonds. -/
@[simp]
theorem _root_.ZMod.card_units_eq_totient (n : ℕ) [NeZero n] [Fintype (ZMod n)ˣ] :
    Fintype.card (ZMod n)ˣ = φ n :=
  calc
    Fintype.card (ZMod n)ˣ = Fintype.card { x : ZMod n // x.val.Coprime n } :=
      Fintype.card_congr ZMod.unitsEquivCoprime
    _ = φ n := by
      /-
        n : Nat
        inst✝¹ : NeZero n
        inst✝ : Fintype (Units (ZMod n))
        ⊢ Eq (Fintype.card (Subtype fun x => x.val.Coprime n)) n.totient
      -/
      obtain ⟨m, rfl⟩ : ∃ m, n = m + 1 := exists_eq_succ_of_ne_zero NeZero.out
      simp only [totient, Finset.card_eq_sum_ones, Fintype.card_subtype, Finset.sum_filter, ←
        Fin.sum_univ_eq_sum_range, @Nat.coprime_comm (m + 1)]
      /-
        case intro
        m : Nat
        inst✝¹ : NeZero (HAdd.hAdd m 1)
        inst✝ : Fintype (Units (ZMod (HAdd.hAdd m 1)))
        ⊢ Eq (Finset.univ.sum fun a => ite (a.val.Coprime (HAdd.hAdd m 1)) 1 0) (Finse …
      -/
      rfl
      /-
        🎉 no goals
      -/


theorem totient_even {n : ℕ} (hn : 2 < n) : Even n.totient := by
  /-
    n : Nat
    hn : LT.lt 2 n
    ⊢ Even n.totient
  -/
  haveI : Fact (1 < n) := ⟨one_lt_two.trans hn⟩
  /-
    n : Nat
    hn : LT.lt 2 n
    this : Fact (LT.lt 1 n)
    ⊢ Even n.totient
  -/
  haveI : NeZero n := NeZero.of_gt hn
  suffices 2 = orderOf (-1 : (ZMod n)ˣ) by
    rw [← ZMod.card_units_eq_totient, even_iff_two_dvd, this]
    exact orderOf_dvd_card
  /-
    n : Nat
    hn : LT.lt 2 n
    this✝ : Fact (LT.lt 1 n)
    this : NeZero n
    ⊢ Eq 2 (orderOf (-1))
  -/
  rw [← orderOf_units, Units.coe_neg_one, orderOf_neg_one, ringChar.eq (ZMod n) n, if_neg hn.ne']
  /-
    🎉 no goals
  -/


theorem totient_mul {m n : ℕ} (h : m.Coprime n) : φ (m * n) = φ m * φ n :=
  if hmn0 : m * n = 0 then by
    /-
      m n : Nat
      h : m.Coprime n
      hmn0 : Eq (HMul.hMul m n) 0
      ⊢ Eq (HMul.hMul m n).totient (HMul.hMul m.totient n.totient)
    -/
    cases' Nat.mul_eq_zero.1 hmn0 with h h <;>
      /-
        case inl
        m n : Nat
        h✝ : m.Coprime n
        hmn0 : Eq (HMul.hMul m n) 0
        h : Eq m 0
        ⊢ Eq (HMul.hMul m n).totient (HMul.hMul m.totient n.totient)
      -/
      /-
        🎉 no goals
      -/
      simp only [totient_zero, mul_zero, zero_mul, h]
      /-
        🎉 no goals
      -/
  else by
    /-
      m n : Nat
      h : m.Coprime n
      hmn0 : Not (Eq (HMul.hMul m n) 0)
      ⊢ Eq (HMul.hMul m n).totient (HMul.hMul m.totient n.totient)
    -/
    haveI : NeZero (m * n) := ⟨hmn0⟩
    /-
      m n : Nat
      h : m.Coprime n
      hmn0 : Not (Eq (HMul.hMul m n) 0)
      this : NeZero (HMul.hMul m n)
      ⊢ Eq (HMul.hMul m n).totient (HMul.hMul m.totient n.totient)
    -/
    haveI : NeZero m := ⟨left_ne_zero_of_mul hmn0⟩
    /-
      m n : Nat
      h : m.Coprime n
      hmn0 : Not (Eq (HMul.hMul m n) 0)
      this✝ : NeZero (HMul.hMul m n)
      this : NeZero m
      ⊢ Eq (HMul.hMul m n).totient (HMul.hMul m.totient n.totient)
    -/
    haveI : NeZero n := ⟨right_ne_zero_of_mul hmn0⟩
    /-
      m n : Nat
      h : m.Coprime n
      hmn0 : Not (Eq (HMul.hMul m n) 0)
      this✝¹ : NeZero (HMul.hMul m n)
      this✝ : NeZero m
      this : NeZero n
      ⊢ Eq (HMul.hMul m n).totient (HMul.hMul m.totient n.totient)
    -/
    simp only [← ZMod.card_units_eq_totient]
    rw [Fintype.card_congr (Units.mapEquiv (ZMod.chineseRemainder h).toMulEquiv).toEquiv,
      Fintype.card_congr (@MulEquiv.prodUnits (ZMod m) (ZMod n) _ _).toEquiv, Fintype.card_prod]


/-- For `d ∣ n`, the totient of `n/d` equals the number of values `k < n` such that `gcd n k = d` -/
theorem totient_div_of_dvd {n d : ℕ} (hnd : d ∣ n) :
    φ (n / d) = #{k ∈ range n | n.gcd k = d} := by
  /-
    n d : Nat
    hnd : Dvd.dvd d n
    ⊢ Eq (HDiv.hDiv n d).totient (Finset.filter (fun k => Eq (n.gcd k) d) (Finset. …
  -/
  rcases d.eq_zero_or_pos with (rfl | hd0); · simp [eq_zero_of_zero_dvd hnd]
                                              /-
                                                🎉 no goals
                                              -/
  /-
    case inr
    n d : Nat
    hnd : Dvd.dvd d n
    hd0 : GT.gt d 0
    ⊢ Eq (HDiv.hDiv n d).totient (Finset.filter (fun k => Eq (n.gcd k) d) (Finset. …
  -/
  rcases hnd with ⟨x, rfl⟩
  /-
    case inr.intro
    d : Nat
    hd0 : GT.gt d 0
    x : Nat
    ⊢ Eq (HDiv.hDiv (HMul.hMul d x) d).totient (Finset.filter (fun k => Eq ((HMul. …
  -/
  rw [Nat.mul_div_cancel_left x hd0]
  /-
    case inr.intro
    d : Nat
    hd0 : GT.gt d 0
    x : Nat
    ⊢ Eq x.totient (Finset.filter (fun k => Eq ((HMul.hMul d x).gcd k) d) (Finset. …
  -/
  apply Finset.card_bij fun k _ => d * k
    /-
      case inr.intro.hi
      d : Nat
      hd0 : GT.gt d 0
      x : Nat
      ⊢ ∀ (a : Nat), Membership.mem (Finset.filter (fun a => x.Coprime a) (Finset.ra …
    -/
  · simp only [mem_filter, mem_range, and_imp, Coprime]
    /-
      case inr.intro.hi
      d : Nat
      hd0 : GT.gt d 0
      x : Nat
      ⊢ ∀ (a : Nat), LT.lt a x → Eq (x.gcd a) 1 → And (LT.lt (HMul.hMul d a) (HMul.h …
    -/
    refine fun a ha1 ha2 => ⟨(mul_lt_mul_left hd0).2 ha1, ?_⟩
    /-
      case inr.intro.hi
      d : Nat
      hd0 : GT.gt d 0
      x a : Nat
      ha1 : LT.lt a x
      ha2 : Eq (x.gcd a) 1
      ⊢ Eq ((HMul.hMul d x).gcd (HMul.hMul d a)) d
    -/
    rw [gcd_mul_left, ha2, mul_one]
    /-
      🎉 no goals
    -/
    /-
      case inr.intro.i_inj
      d : Nat
      hd0 : GT.gt d 0
      x : Nat
      ⊢ ∀ (a₁ : Nat), Membership.mem (Finset.filter (fun a => x.Coprime a) (Finset.r …
    -/
  · simp [hd0.ne']
    /-
      🎉 no goals
    -/
    /-
      case inr.intro.i_surj
      d : Nat
      hd0 : GT.gt d 0
      x : Nat
      ⊢ ∀ (b : Nat), Membership.mem (Finset.filter (fun k => Eq ((HMul.hMul d x).gcd …
    -/
  · simp only [mem_filter, mem_range, exists_prop, and_imp]
    /-
      case inr.intro.i_surj
      d : Nat
      hd0 : GT.gt d 0
      x : Nat
      ⊢ ∀ (b : Nat), LT.lt b (HMul.hMul d x) → Eq ((HMul.hMul d x).gcd b) d → Exists …
    -/
    refine fun b hb1 hb2 => ?_
    have : d ∣ b := by
      rw [← hb2]
      apply gcd_dvd_right
    /-
      case inr.intro.i_surj
      d : Nat
      hd0 : GT.gt d 0
      x b : Nat
      hb1 : LT.lt b (HMul.hMul d x)
      hb2 : Eq ((HMul.hMul d x).gcd b) d
      this : Dvd.dvd d b
      ⊢ Exists fun a => And (And (LT.lt a x) (x.Coprime a)) (Eq (HMul.hMul d a) b)
    -/
    rcases this with ⟨q, rfl⟩
    /-
      case inr.intro.i_surj.intro
      d : Nat
      hd0 : GT.gt d 0
      x q : Nat
      hb1 : LT.lt (HMul.hMul d q) (HMul.hMul d x)
      hb2 : Eq ((HMul.hMul d x).gcd (HMul.hMul d q)) d
      ⊢ Exists fun a => And (And (LT.lt a x) (x.Coprime a)) (Eq (HMul.hMul d a) (HMu …
    -/
    refine ⟨q, ⟨⟨(mul_lt_mul_left hd0).1 hb1, ?_⟩, rfl⟩⟩
    /-
      case inr.intro.i_surj.intro
      d : Nat
      hd0 : GT.gt d 0
      x q : Nat
      hb1 : LT.lt (HMul.hMul d q) (HMul.hMul d x)
      hb2 : Eq ((HMul.hMul d x).gcd (HMul.hMul d q)) d
      ⊢ x.Coprime q
    -/
    rwa [gcd_mul_left, mul_right_eq_self_iff hd0] at hb2
    /-
      🎉 no goals
    -/


theorem sum_totient (n : ℕ) : n.divisors.sum φ = n := by
  /-
    n : Nat
    ⊢ Eq (n.divisors.sum Nat.totient) n
  -/
  rcases n.eq_zero_or_pos with (rfl | hn)
    /-
      case inl
      ⊢ Eq ((Nat.divisors 0).sum Nat.totient) 0
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    n : Nat
    hn : GT.gt n 0
    ⊢ Eq (n.divisors.sum Nat.totient) n
  -/
  rw [← sum_div_divisors n φ]
  have : n = ∑ d ∈ n.divisors, #{k ∈ range n | n.gcd k = d} := by
    nth_rw 1 [← card_range n]
    refine card_eq_sum_card_fiberwise fun x _ => mem_divisors.2 ⟨?_, hn.ne'⟩
    apply gcd_dvd_left
  /-
    case inr
    n : Nat
    hn : GT.gt n 0
    this : Eq n (n.divisors.sum fun d => (Finset.filter (fun k => Eq (n.gcd k) d)  …
    ⊢ Eq (n.divisors.sum fun d => (HDiv.hDiv n d).totient) n
  -/
  nth_rw 3 [this]
  /-
    case inr
    n : Nat
    hn : GT.gt n 0
    this : Eq n (n.divisors.sum fun d => (Finset.filter (fun k => Eq (n.gcd k) d)  …
    ⊢ Eq (n.divisors.sum fun d => (HDiv.hDiv n d).totient) (n.divisors.sum fun d = …
  -/
  exact sum_congr rfl fun x hx => totient_div_of_dvd (dvd_of_mem_divisors hx)
  /-
    🎉 no goals
  -/


theorem sum_totient' (n : ℕ) : ∑ m ∈ range n.succ with m ∣ n, φ m = n := by
  /-
    n : Nat
    ⊢ Eq ((Finset.filter (fun m => Dvd.dvd m n) (Finset.range n.succ)).sum fun m = …
  -/
  convert sum_totient _ using 1
  /-
    case h.e'_2
    n : Nat
    ⊢ Eq ((Finset.filter (fun m => Dvd.dvd m n) (Finset.range n.succ)).sum fun m = …
  -/
  simp only [Nat.divisors, sum_filter, range_eq_Ico]
  /-
    case h.e'_2
    n : Nat
    ⊢ Eq ((Finset.Ico 0 n.succ).sum fun a => ite (Dvd.dvd a n) a.totient 0) ((Fins …
  -/
                                   /-
                                     🎉 no goals
                                   -/
  rw [sum_eq_sum_Ico_succ_bot] <;> simp
                                   /-
                                     🎉 no goals
                                   -/


/-- When `p` is prime, then the totient of `p ^ (n + 1)` is `p ^ n * (p - 1)` -/
theorem totient_prime_pow_succ {p : ℕ} (hp : p.Prime) (n : ℕ) : φ (p ^ (n + 1)) = p ^ n * (p - 1) :=
  calc
    φ (p ^ (n + 1)) = #{a ∈ range (p ^ (n + 1)) | (p ^ (n + 1)).Coprime a} :=
      totient_eq_card_coprime _
    _ = #(range (p ^ (n + 1)) \ (range (p ^ n)).image (· * p)) :=
      congr_arg card
        (by
          /-
            p : Nat
            hp : Nat.Prime p
            n : Nat
            ⊢ Eq (Finset.filter (fun a => (HPow.hPow p (HAdd.hAdd n 1)).Coprime a) (Finset …
          -/
          rw [sdiff_eq_filter]
          /-
            p : Nat
            hp : Nat.Prime p
            n : Nat
            ⊢ Eq (Finset.filter (fun a => (HPow.hPow p (HAdd.hAdd n 1)).Coprime a) (Finset …
          -/
          apply filter_congr
          simp only [mem_range, mem_filter, coprime_pow_left_iff n.succ_pos, mem_image, not_exists,
            hp.coprime_iff_not_dvd]
          /-
            case H
            p : Nat
            hp : Nat.Prime p
            n : Nat
            ⊢ ∀ (x : Nat), LT.lt x (HPow.hPow p (HAdd.hAdd n 1)) → Iff (Not (Dvd.dvd p x)) …
          -/
          intro a ha
          /-
            case H
            p : Nat
            hp : Nat.Prime p
            n a : Nat
            ha : LT.lt a (HPow.hPow p (HAdd.hAdd n 1))
            ⊢ Iff (Not (Dvd.dvd p a)) (∀ (x : Nat), Not (And (LT.lt x (HPow.hPow p n)) (Eq …
          -/
          constructor
            /-
              case H.mp
              p : Nat
              hp : Nat.Prime p
              n a : Nat
              ha : LT.lt a (HPow.hPow p (HAdd.hAdd n 1))
              ⊢ Not (Dvd.dvd p a) → ∀ (x : Nat), Not (And (LT.lt x (HPow.hPow p n)) (Eq (HMu …
            -/
          · intro hap b h; rcases h with ⟨_, rfl⟩
            /-
              case H.mp.intro
              p : Nat
              hp : Nat.Prime p
              n b : Nat
              left✝ : LT.lt b (HPow.hPow p n)
              ha : LT.lt (HMul.hMul b p) (HPow.hPow p (HAdd.hAdd n 1))
              hap : Not (Dvd.dvd p (HMul.hMul b p))
              ⊢ False
            -/
            exact hap (dvd_mul_left _ _)
            /-
              🎉 no goals
            -/
            /-
              case H.mpr
              p : Nat
              hp : Nat.Prime p
              n a : Nat
              ha : LT.lt a (HPow.hPow p (HAdd.hAdd n 1))
              ⊢ (∀ (x : Nat), Not (And (LT.lt x (HPow.hPow p n)) (Eq (HMul.hMul x p) a))) →  …
            -/
          · rintro h ⟨b, rfl⟩
            /-
              case H.mpr.intro
              p : Nat
              hp : Nat.Prime p
              n b : Nat
              ha : LT.lt (HMul.hMul p b) (HPow.hPow p (HAdd.hAdd n 1))
              h : ∀ (x : Nat), Not (And (LT.lt x (HPow.hPow p n)) (Eq (HMul.hMul x p) (HMul. …
              ⊢ False
            -/
            rw [pow_succ'] at ha
            /-
              case H.mpr.intro
              p : Nat
              hp : Nat.Prime p
              n b : Nat
              ha : LT.lt (HMul.hMul p b) (HMul.hMul p (HPow.hPow p n))
              h : ∀ (x : Nat), Not (And (LT.lt x (HPow.hPow p n)) (Eq (HMul.hMul x p) (HMul. …
              ⊢ False
            -/
            exact h b ⟨lt_of_mul_lt_mul_left ha (zero_le _), mul_comm _ _⟩)
            /-
              🎉 no goals
            -/
    _ = _ := by
      /-
        p : Nat
        hp : Nat.Prime p
        n : Nat
        ⊢ Eq (SDiff.sdiff (Finset.range (HPow.hPow p (HAdd.hAdd n 1))) (Finset.image ( …
      -/
      have h1 : Function.Injective (· * p) := mul_left_injective₀ hp.ne_zero
      have h2 : (range (p ^ n)).image (· * p) ⊆ range (p ^ (n + 1)) := fun a => by
        simp only [mem_image, mem_range, exists_imp]
        rintro b ⟨h, rfl⟩
        rw [Nat.pow_succ]
        exact (mul_lt_mul_right hp.pos).2 h
      rw [card_sdiff h2, Finset.card_image_of_injective _ h1, card_range, card_range, ←
        one_mul (p ^ n), pow_succ', ← tsub_mul, one_mul, mul_comm]


/-- When `p` is prime, then the totient of `p ^ n` is `p ^ (n - 1) * (p - 1)` -/
theorem totient_prime_pow {p : ℕ} (hp : p.Prime) {n : ℕ} (hn : 0 < n) :
    φ (p ^ n) = p ^ (n - 1) * (p - 1) := by
  /-
    p : Nat
    hp : Nat.Prime p
    n : Nat
    hn : LT.lt 0 n
    ⊢ Eq (HPow.hPow p n).totient (HMul.hMul (HPow.hPow p (HSub.hSub n 1)) (HSub.hS …
  -/
  rcases exists_eq_succ_of_ne_zero (pos_iff_ne_zero.1 hn) with ⟨m, rfl⟩
  /-
    case intro
    p : Nat
    hp : Nat.Prime p
    m : Nat
    hn : LT.lt 0 m.succ
    ⊢ Eq (HPow.hPow p m.succ).totient (HMul.hMul (HPow.hPow p (HSub.hSub m.succ 1) …
  -/
  exact totient_prime_pow_succ hp _
  /-
    🎉 no goals
  -/


theorem totient_prime {p : ℕ} (hp : p.Prime) : φ p = p - 1 := by
  /-
    p : Nat
    hp : Nat.Prime p
    ⊢ Eq p.totient (HSub.hSub p 1)
  -/
                                             /-
                                               🎉 no goals
                                             -/
  rw [← pow_one p, totient_prime_pow hp] <;> simp
                                             /-
                                               🎉 no goals
                                             -/


theorem totient_eq_iff_prime {p : ℕ} (hp : 0 < p) : p.totient = p - 1 ↔ p.Prime := by
  /-
    p : Nat
    hp : LT.lt 0 p
    ⊢ Iff (Eq p.totient (HSub.hSub p 1)) (Nat.Prime p)
  -/
  refine ⟨fun h => ?_, totient_prime⟩
  replace hp : 1 < p := by
    apply lt_of_le_of_ne
    · rwa [succ_le_iff]
    · rintro rfl
      rw [totient_one, tsub_self] at h
      exact one_ne_zero h
  rw [totient_eq_card_coprime, range_eq_Ico, ← Ico_insert_succ_left hp.le, Finset.filter_insert,
    if_neg (not_coprime_of_dvd_of_dvd hp (dvd_refl p) (dvd_zero p)), ← Nat.card_Ico 1 p] at h
  refine
    p.prime_of_coprime hp fun n hn hnz => Finset.filter_card_eq h n <| Finset.mem_Ico.mpr ⟨?_, hn⟩
  /-
    p : Nat
    h : Eq (Finset.filter (fun a => p.Coprime a) (Finset.Ico (Nat.succ 0) p)).card …
    hp : LT.lt 1 p
    n : Nat
    hn : LT.lt n p
    hnz : Ne n 0
    ⊢ LE.le (Nat.succ 0) n
  -/
  rwa [succ_le_iff, pos_iff_ne_zero]
  /-
    🎉 no goals
  -/


theorem card_units_zmod_lt_sub_one {p : ℕ} (hp : 1 < p) [Fintype (ZMod p)ˣ] :
    Fintype.card (ZMod p)ˣ ≤ p - 1 := by
  /-
    p : Nat
    hp : LT.lt 1 p
    inst✝ : Fintype (Units (ZMod p))
    ⊢ LE.le (Fintype.card (Units (ZMod p))) (HSub.hSub p 1)
  -/
  haveI : NeZero p := ⟨(pos_of_gt hp).ne'⟩
  /-
    p : Nat
    hp : LT.lt 1 p
    inst✝ : Fintype (Units (ZMod p))
    this : NeZero p
    ⊢ LE.le (Fintype.card (Units (ZMod p))) (HSub.hSub p 1)
  -/
  rw [ZMod.card_units_eq_totient p]
  /-
    p : Nat
    hp : LT.lt 1 p
    inst✝ : Fintype (Units (ZMod p))
    this : NeZero p
    ⊢ LE.le p.totient (HSub.hSub p 1)
  -/
  exact Nat.le_sub_one_of_lt (Nat.totient_lt p hp)
  /-
    🎉 no goals
  -/


theorem prime_iff_card_units (p : ℕ) [Fintype (ZMod p)ˣ] :
    p.Prime ↔ Fintype.card (ZMod p)ˣ = p - 1 := by
  /-
    p : Nat
    inst✝ : Fintype (Units (ZMod p))
    ⊢ Iff (Nat.Prime p) (Eq (Fintype.card (Units (ZMod p))) (HSub.hSub p 1))
  -/
  cases' eq_zero_or_neZero p with hp hp
    /-
      case inl
      p : Nat
      inst✝ : Fintype (Units (ZMod p))
      hp : Eq p 0
      ⊢ Iff (Nat.Prime p) (Eq (Fintype.card (Units (ZMod p))) (HSub.hSub p 1))
    -/
  · subst hp
    /-
      case inl
      inst✝ : Fintype (Units (ZMod 0))
      ⊢ Iff (Nat.Prime 0) (Eq (Fintype.card (Units (ZMod 0))) (HSub.hSub 0 1))
    -/
    simp only [ZMod, not_prime_zero, false_iff, zero_tsub]
    -- the subst created a non-defeq but subsingleton instance diamond; resolve it
    /-
      case inl
      inst✝ : Fintype (Units (ZMod 0))
      ⊢ Not (Eq (Fintype.card (Units Int)) 0)
    -/
    suffices Fintype.card ℤˣ ≠ 0 by convert this
    /-
      case inl
      inst✝ : Fintype (Units (ZMod 0))
      ⊢ Ne (Fintype.card (Units Int)) 0
    -/
    simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    p : Nat
    inst✝ : Fintype (Units (ZMod p))
    hp : NeZero p
    ⊢ Iff (Nat.Prime p) (Eq (Fintype.card (Units (ZMod p))) (HSub.hSub p 1))
  -/
  rw [ZMod.card_units_eq_totient, Nat.totient_eq_iff_prime <| NeZero.pos p]
  /-
    🎉 no goals
  -/


@[simp]
theorem totient_two : φ 2 = 1 :=
  (totient_prime prime_two).trans rfl


theorem totient_eq_one_iff : ∀ {n : ℕ}, n.totient = 1 ↔ n = 1 ∨ n = 2
            /-
              ⊢ Iff (Eq (Nat.totient 0) 1) (Or (Eq 0 1) (Eq 0 2))
            -/
  | 0 => by simp
            /-
              🎉 no goals
            -/
            /-
              ⊢ Iff (Eq (Nat.totient 1) 1) (Or (Eq 1 1) (Eq 1 2))
            -/
  | 1 => by simp
            /-
              🎉 no goals
            -/
            /-
              ⊢ Iff (Eq (Nat.totient 2) 1) (Or (Eq 2 1) (Eq 2 2))
            -/
  | 2 => by simp
            /-
              🎉 no goals
            -/
  | n + 3 => by
    /-
      n : Nat
      ⊢ Iff (Eq (HAdd.hAdd n 3).totient 1) (Or (Eq (HAdd.hAdd n 3) 1) (Eq (HAdd.hAdd …
    -/
    have : 3 ≤ n + 3 := le_add_self
    /-
      n : Nat
      this : LE.le 3 (HAdd.hAdd n 3)
      ⊢ Iff (Eq (HAdd.hAdd n 3).totient 1) (Or (Eq (HAdd.hAdd n 3) 1) (Eq (HAdd.hAdd …
    -/
    simp only [succ_succ_ne_one, false_or]
    /-
      n : Nat
      this : LE.le 3 (HAdd.hAdd n 3)
      ⊢ Iff (Eq (HAdd.hAdd n 3).totient 1) (Eq (HAdd.hAdd n 3) 2)
    -/
    exact ⟨fun h => not_even_one.elim <| h ▸ totient_even this, by rintro ⟨⟩⟩
    /-
      🎉 no goals
    -/


theorem dvd_two_of_totient_le_one {a : ℕ} (han : 0 < a) (ha : a.totient ≤ 1) : a ∣ 2 := by
  /-
    a : Nat
    han : LT.lt 0 a
    ha : LE.le a.totient 1
    ⊢ Dvd.dvd a 2
  -/
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/
  rcases totient_eq_one_iff.mp <| le_antisymm ha <| totient_pos.2 han with rfl | rfl <;> norm_num
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/


/-- Euler's product formula for the totient function. -/
theorem totient_eq_prod_factorization {n : ℕ} (hn : n ≠ 0) :
    φ n = n.factorization.prod fun p k => p ^ (k - 1) * (p - 1) := by
  /-
    n : Nat
    hn : Ne n 0
    ⊢ Eq n.totient (n.factorization.prod fun p k => HMul.hMul (HPow.hPow p (HSub.h …
  -/
  rw [multiplicative_factorization φ (@totient_mul) totient_one hn]
  /-
    n : Nat
    hn : Ne n 0
    ⊢ Eq (n.factorization.prod fun p k => (HPow.hPow p k).totient) (n.factorizatio …
  -/
  apply Finsupp.prod_congr _
  /-
    n : Nat
    hn : Ne n 0
    ⊢ ∀ (x : Nat), Membership.mem n.factorization.support x → Eq (HPow.hPow x (n.f …
  -/
  intro p hp
  /-
    n : Nat
    hn : Ne n 0
    p : Nat
    hp : Membership.mem n.factorization.support p
    ⊢ Eq (HPow.hPow p (n.factorization p)).totient (HMul.hMul (HPow.hPow p (HSub.h …
  -/
  have h := zero_lt_iff.mpr (Finsupp.mem_support_iff.mp hp)
  /-
    n : Nat
    hn : Ne n 0
    p : Nat
    hp : Membership.mem n.factorization.support p
    h : LT.lt 0 (n.factorization p)
    ⊢ Eq (HPow.hPow p (n.factorization p)).totient (HMul.hMul (HPow.hPow p (HSub.h …
  -/
  rw [totient_prime_pow (prime_of_mem_primeFactors hp) h]
  /-
    🎉 no goals
  -/


/-- Euler's product formula for the totient function. -/
theorem totient_mul_prod_primeFactors (n : ℕ) :
    (φ n * ∏ p ∈ n.primeFactors, p) = n * ∏ p ∈ n.primeFactors, (p - 1) := by
  /-
    n : Nat
    ⊢ Eq (HMul.hMul n.totient (n.primeFactors.prod fun p => p)) (HMul.hMul n (n.pr …
  -/
  by_cases hn : n = 0; · simp [hn]
                         /-
                           🎉 no goals
                         -/
  /-
    case neg
    n : Nat
    hn : Not (Eq n 0)
    ⊢ Eq (HMul.hMul n.totient (n.primeFactors.prod fun p => p)) (HMul.hMul n (n.pr …
  -/
  rw [totient_eq_prod_factorization hn]
  /-
    case neg
    n : Nat
    hn : Not (Eq n 0)
    ⊢ Eq (HMul.hMul (n.factorization.prod fun p k => HMul.hMul (HPow.hPow p (HSub. …
  -/
  nth_rw 3 [← factorization_prod_pow_eq_self hn]
  /-
    case neg
    n : Nat
    hn : Not (Eq n 0)
    ⊢ Eq (HMul.hMul (n.factorization.prod fun p k => HMul.hMul (HPow.hPow p (HSub. …
  -/
  simp only [prod_primeFactors_prod_factorization, ← Finsupp.prod_mul]
  /-
    case neg
    n : Nat
    hn : Not (Eq n 0)
    ⊢ Eq (n.factorization.prod fun a b => HMul.hMul (HMul.hMul (HPow.hPow a (HSub. …
  -/
  refine Finsupp.prod_congr (M := ℕ) (N := ℕ) fun p hp => ?_
  /-
    case neg
    n : Nat
    hn : Not (Eq n 0)
    p : Nat
    hp : Membership.mem n.factorization.support p
    ⊢ Eq (HMul.hMul (HMul.hMul (HPow.hPow p (HSub.hSub (n.factorization p) 1)) (HS …
  -/
  rw [Finsupp.mem_support_iff, ← zero_lt_iff] at hp
  /-
    case neg
    n : Nat
    hn : Not (Eq n 0)
    p : Nat
    hp : LT.lt 0 (n.factorization p)
    ⊢ Eq (HMul.hMul (HMul.hMul (HPow.hPow p (HSub.hSub (n.factorization p) 1)) (HS …
  -/
  rw [mul_comm, ← mul_assoc, ← pow_succ', Nat.sub_one, Nat.succ_pred_eq_of_pos hp]
  /-
    🎉 no goals
  -/


/-- Euler's product formula for the totient function. -/
theorem totient_eq_div_primeFactors_mul (n : ℕ) :
    φ n = (n / ∏ p ∈ n.primeFactors, p) * ∏ p ∈ n.primeFactors, (p - 1) := by
  rw [← mul_div_left n.totient, totient_mul_prod_primeFactors, mul_comm,
    Nat.mul_div_assoc _ (prod_primeFactors_dvd n), mul_comm]
  /-
    n : Nat
    ⊢ LT.lt 0 (n.primeFactors.prod fun p => p)
  -/
  exact prod_pos (fun p => pos_of_mem_primeFactors)
  /-
    🎉 no goals
  -/


/-- Euler's product formula for the totient function. -/
theorem totient_eq_mul_prod_factors (n : ℕ) :
    (φ n : ℚ) = n * ∏ p ∈ n.primeFactors, (1 - (p : ℚ)⁻¹) := by
  /-
    n : Nat
    ⊢ Eq (↑n.totient) (HMul.hMul (↑n) (n.primeFactors.prod fun p => HSub.hSub 1 (I …
  -/
  by_cases hn : n = 0
    /-
      case pos
      n : Nat
      hn : Eq n 0
      ⊢ Eq (↑n.totient) (HMul.hMul (↑n) (n.primeFactors.prod fun p => HSub.hSub 1 (I …
    -/
  · simp [hn]
    /-
      🎉 no goals
    -/
  /-
    case neg
    n : Nat
    hn : Not (Eq n 0)
    ⊢ Eq (↑n.totient) (HMul.hMul (↑n) (n.primeFactors.prod fun p => HSub.hSub 1 (I …
  -/
  have hn' : (n : ℚ) ≠ 0 := by simp [hn]
  have hpQ : (∏ p ∈ n.primeFactors, (p : ℚ)) ≠ 0 := by
    rw [← cast_prod, cast_ne_zero, ← zero_lt_iff, prod_primeFactors_prod_factorization]
    exact prod_pos fun p hp => pos_of_mem_primeFactors hp
  simp only [totient_eq_div_primeFactors_mul n, prod_primeFactors_dvd n, cast_mul, cast_prod,
    cast_div_charZero, mul_comm_div, mul_right_inj' hn', div_eq_iff hpQ, ← prod_mul_distrib]
  /-
    case neg
    n : Nat
    hn : Not (Eq n 0)
    hn' : Ne (↑n) 0
    hpQ : Ne (n.primeFactors.prod fun p => ↑p) 0
    ⊢ Eq (n.primeFactors.prod fun i => ↑(HSub.hSub i 1)) (n.primeFactors.prod fun  …
  -/
  refine prod_congr rfl fun p hp => ?_
  /-
    case neg
    n : Nat
    hn : Not (Eq n 0)
    hn' : Ne (↑n) 0
    hpQ : Ne (n.primeFactors.prod fun p => ↑p) 0
    p : Nat
    hp : Membership.mem n.primeFactors p
    ⊢ Eq (↑(HSub.hSub p 1)) (HMul.hMul (HSub.hSub 1 (Inv.inv ↑p)) ↑p)
  -/
  have hp := pos_of_mem_primeFactorsList (List.mem_toFinset.mp hp)
  /-
    case neg
    n : Nat
    hn : Not (Eq n 0)
    hn' : Ne (↑n) 0
    hpQ : Ne (n.primeFactors.prod fun p => ↑p) 0
    p : Nat
    hp✝ : Membership.mem n.primeFactors p
    hp : LT.lt 0 p
    ⊢ Eq (↑(HSub.hSub p 1)) (HMul.hMul (HSub.hSub 1 (Inv.inv ↑p)) ↑p)
  -/
  have hp' : (p : ℚ) ≠ 0 := cast_ne_zero.mpr hp.ne.symm
  /-
    case neg
    n : Nat
    hn : Not (Eq n 0)
    hn' : Ne (↑n) 0
    hpQ : Ne (n.primeFactors.prod fun p => ↑p) 0
    p : Nat
    hp✝ : Membership.mem n.primeFactors p
    hp : LT.lt 0 p
    hp' : Ne (↑p) 0
    ⊢ Eq (↑(HSub.hSub p 1)) (HMul.hMul (HSub.hSub 1 (Inv.inv ↑p)) ↑p)
  -/
  rw [sub_mul, one_mul, mul_comm, mul_inv_cancel₀ hp', cast_pred hp]
  /-
    🎉 no goals
  -/


theorem totient_gcd_mul_totient_mul (a b : ℕ) : φ (a.gcd b) * φ (a * b) = φ a * φ b * a.gcd b := by
  have shuffle :
    ∀ a1 a2 b1 b2 c1 c2 : ℕ,
      b1 ∣ a1 → b2 ∣ a2 → a1 / b1 * c1 * (a2 / b2 * c2) = a1 * a2 / (b1 * b2) * (c1 * c2) := by
    intro a1 a2 b1 b2 c1 c2 h1 h2
    calc
      a1 / b1 * c1 * (a2 / b2 * c2) = a1 / b1 * (a2 / b2) * (c1 * c2) := by apply mul_mul_mul_comm
      _ = a1 * a2 / (b1 * b2) * (c1 * c2) := by
        congr 1
        exact div_mul_div_comm h1 h2
  /-
    a b : Nat
    shuffle : ∀ (a1 a2 b1 b2 c1 c2 : Nat), Dvd.dvd b1 a1 → Dvd.dvd b2 a2 → Eq (HMu …
    ⊢ Eq (HMul.hMul (a.gcd b).totient (HMul.hMul a b).totient) (HMul.hMul (HMul.hM …
  -/
  simp only [totient_eq_div_primeFactors_mul]
  /-
    a b : Nat
    shuffle : ∀ (a1 a2 b1 b2 c1 c2 : Nat), Dvd.dvd b1 a1 → Dvd.dvd b2 a2 → Eq (HMu …
    ⊢ Eq (HMul.hMul (HMul.hMul (HDiv.hDiv (a.gcd b) ((a.gcd b).primeFactors.prod f …
  -/
  rw [shuffle, shuffle]
  /-
    a b : Nat
    shuffle : ∀ (a1 a2 b1 b2 c1 c2 : Nat), Dvd.dvd b1 a1 → Dvd.dvd b2 a2 → Eq (HMu …
    ⊢ Eq (HMul.hMul (HDiv.hDiv (HMul.hMul (a.gcd b) (HMul.hMul a b)) (HMul.hMul (( …
  -/
  rotate_left
  /-
    case a
    a b : Nat
    shuffle : ∀ (a1 a2 b1 b2 c1 c2 : Nat), Dvd.dvd b1 a1 → Dvd.dvd b2 a2 → Eq (HMu …
    ⊢ Dvd.dvd (a.primeFactors.prod fun p => p) a
  -/
  repeat' apply prod_primeFactors_dvd
  /-
    a b : Nat
    shuffle : ∀ (a1 a2 b1 b2 c1 c2 : Nat), Dvd.dvd b1 a1 → Dvd.dvd b2 a2 → Eq (HMu …
    ⊢ Eq (HMul.hMul (HDiv.hDiv (HMul.hMul (a.gcd b) (HMul.hMul a b)) (HMul.hMul (( …
  -/
  simp only [prod_primeFactors_gcd_mul_prod_primeFactors_mul]
  /-
    a b : Nat
    shuffle : ∀ (a1 a2 b1 b2 c1 c2 : Nat), Dvd.dvd b1 a1 → Dvd.dvd b2 a2 → Eq (HMu …
    ⊢ Eq (HMul.hMul (HDiv.hDiv (HMul.hMul (a.gcd b) (HMul.hMul a b)) (HMul.hMul (a …
  -/
  rw [eq_comm, mul_comm, ← mul_assoc, ← Nat.mul_div_assoc]
  /-
    case H
    a b : Nat
    shuffle : ∀ (a1 a2 b1 b2 c1 c2 : Nat), Dvd.dvd b1 a1 → Dvd.dvd b2 a2 → Eq (HMu …
    ⊢ Dvd.dvd (HMul.hMul (a.primeFactors.prod fun p => p) (b.primeFactors.prod fun …
  -/
  exact mul_dvd_mul (prod_primeFactors_dvd a) (prod_primeFactors_dvd b)
  /-
    🎉 no goals
  -/


theorem totient_super_multiplicative (a b : ℕ) : φ a * φ b ≤ φ (a * b) := by
  /-
    a b : Nat
    ⊢ LE.le (HMul.hMul a.totient b.totient) (HMul.hMul a b).totient
  -/
  let d := a.gcd b
  /-
    a b : Nat
    d : Nat := a.gcd b
    ⊢ LE.le (HMul.hMul a.totient b.totient) (HMul.hMul a b).totient
  -/
  rcases (zero_le a).eq_or_lt with (rfl | ha0)
    /-
      case inl
      b : Nat
      d : Nat := Nat.gcd 0 b
      ⊢ LE.le (HMul.hMul (Nat.totient 0) b.totient) (HMul.hMul 0 b).totient
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    a b : Nat
    d : Nat := a.gcd b
    ha0 : LT.lt 0 a
    ⊢ LE.le (HMul.hMul a.totient b.totient) (HMul.hMul a b).totient
  -/
  have hd0 : 0 < d := Nat.gcd_pos_of_pos_left _ ha0
  /-
    case inr
    a b : Nat
    d : Nat := a.gcd b
    ha0 : LT.lt 0 a
    hd0 : LT.lt 0 d
    ⊢ LE.le (HMul.hMul a.totient b.totient) (HMul.hMul a b).totient
  -/
  apply le_of_mul_le_mul_right _ hd0
  /-
    a b : Nat
    d : Nat := a.gcd b
    ha0 : LT.lt 0 a
    hd0 : LT.lt 0 d
    ⊢ LE.le (HMul.hMul (HMul.hMul a.totient b.totient) d) (HMul.hMul (HMul.hMul a  …
  -/
  rw [← totient_gcd_mul_totient_mul a b, mul_comm]
  /-
    a b : Nat
    d : Nat := a.gcd b
    ha0 : LT.lt 0 a
    hd0 : LT.lt 0 d
    ⊢ LE.le (HMul.hMul (HMul.hMul a b).totient (a.gcd b).totient) (HMul.hMul (HMul …
  -/
  apply mul_le_mul_left' (Nat.totient_le d)
  /-
    🎉 no goals
  -/


theorem totient_dvd_of_dvd {a b : ℕ} (h : a ∣ b) : φ a ∣ φ b := by
  /-
    a b : Nat
    h : Dvd.dvd a b
    ⊢ Dvd.dvd a.totient b.totient
  -/
  rcases eq_or_ne a 0 with (rfl | ha0)
    /-
      case inl
      b : Nat
      h : Dvd.dvd 0 b
      ⊢ Dvd.dvd (Nat.totient 0) b.totient
    -/
  · simp [zero_dvd_iff.1 h]
    /-
      🎉 no goals
    -/
  /-
    case inr
    a b : Nat
    h : Dvd.dvd a b
    ha0 : Ne a 0
    ⊢ Dvd.dvd a.totient b.totient
  -/
  rcases eq_or_ne b 0 with (rfl | hb0)
    /-
      case inr.inl
      a : Nat
      ha0 : Ne a 0
      h : Dvd.dvd a 0
      ⊢ Dvd.dvd a.totient (Nat.totient 0)
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    a b : Nat
    h : Dvd.dvd a b
    ha0 : Ne a 0
    hb0 : Ne b 0
    ⊢ Dvd.dvd a.totient b.totient
  -/
  have hab' := primeFactors_mono h hb0
  /-
    case inr.inr
    a b : Nat
    h : Dvd.dvd a b
    ha0 : Ne a 0
    hb0 : Ne b 0
    hab' : HasSubset.Subset a.primeFactors b.primeFactors
    ⊢ Dvd.dvd a.totient b.totient
  -/
  rw [totient_eq_prod_factorization ha0, totient_eq_prod_factorization hb0]
  /-
    case inr.inr
    a b : Nat
    h : Dvd.dvd a b
    ha0 : Ne a 0
    hb0 : Ne b 0
    hab' : HasSubset.Subset a.primeFactors b.primeFactors
    ⊢ Dvd.dvd (a.factorization.prod fun p k => HMul.hMul (HPow.hPow p (HSub.hSub k …
  -/
  refine Finsupp.prod_dvd_prod_of_subset_of_dvd hab' fun p _ => mul_dvd_mul ?_ dvd_rfl
  /-
    case inr.inr
    a b : Nat
    h : Dvd.dvd a b
    ha0 : Ne a 0
    hb0 : Ne b 0
    hab' : HasSubset.Subset a.primeFactors b.primeFactors
    p : Nat
    x✝ : Membership.mem a.factorization.support p
    ⊢ Dvd.dvd (HPow.hPow p (HSub.hSub (a.factorization p) 1)) (HPow.hPow p (HSub.h …
  -/
  exact pow_dvd_pow p (tsub_le_tsub_right ((factorization_le_iff_dvd ha0 hb0).2 h p) 1)
  /-
    🎉 no goals
  -/


theorem totient_mul_of_prime_of_dvd {p n : ℕ} (hp : p.Prime) (h : p ∣ n) :
    (p * n).totient = p * n.totient := by
  /-
    p n : Nat
    hp : Nat.Prime p
    h : Dvd.dvd p n
    ⊢ Eq (HMul.hMul p n).totient (HMul.hMul p n.totient)
  -/
  have h1 := totient_gcd_mul_totient_mul p n
  /-
    p n : Nat
    hp : Nat.Prime p
    h : Dvd.dvd p n
    h1 : Eq (HMul.hMul (p.gcd n).totient (HMul.hMul p n).totient) (HMul.hMul (HMul …
    ⊢ Eq (HMul.hMul p n).totient (HMul.hMul p n.totient)
  -/
  rw [gcd_eq_left h, mul_assoc] at h1
  /-
    p n : Nat
    hp : Nat.Prime p
    h : Dvd.dvd p n
    h1 : Eq (HMul.hMul p.totient (HMul.hMul p n).totient) (HMul.hMul p.totient (HM …
    ⊢ Eq (HMul.hMul p n).totient (HMul.hMul p n.totient)
  -/
  simpa [(totient_pos.2 hp.pos).ne', mul_comm] using h1
  /-
    🎉 no goals
  -/


theorem totient_mul_of_prime_of_not_dvd {p n : ℕ} (hp : p.Prime) (h : ¬p ∣ n) :
    (p * n).totient = (p - 1) * n.totient := by
  /-
    p n : Nat
    hp : Nat.Prime p
    h : Not (Dvd.dvd p n)
    ⊢ Eq (HMul.hMul p n).totient (HMul.hMul (HSub.hSub p 1) n.totient)
  -/
  rw [totient_mul _, totient_prime hp]
  /-
    p n : Nat
    hp : Nat.Prime p
    h : Not (Dvd.dvd p n)
    ⊢ p.Coprime n
  -/
  simpa [h] using coprime_or_dvd_of_prime hp n
  /-
    🎉 no goals
  -/


