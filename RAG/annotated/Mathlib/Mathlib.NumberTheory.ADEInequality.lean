/-- `A' q r := {1,q,r}` is a `Multiset ℕ+`
that is a solution to the inequality
`(p⁻¹ + q⁻¹ + r⁻¹ : ℚ) > 1`. -/
def A' (q r : ℕ+) : Multiset ℕ+ :=
  {1, q, r}


/-- `A r := {1,1,r}` is a `Multiset ℕ+`
that is a solution to the inequality
`(p⁻¹ + q⁻¹ + r⁻¹ : ℚ) > 1`.

These solutions are related to the Dynkin diagrams $A_r$. -/
def A (r : ℕ+) : Multiset ℕ+ :=
  A' 1 r


/-- `D' r := {2,2,r}` is a `Multiset ℕ+`
that is a solution to the inequality
`(p⁻¹ + q⁻¹ + r⁻¹ : ℚ) > 1`.

These solutions are related to the Dynkin diagrams $D_{r+2}$. -/
def D' (r : ℕ+) : Multiset ℕ+ :=
  {2, 2, r}


/-- `E' r := {2,3,r}` is a `Multiset ℕ+`.
For `r ∈ {3,4,5}` is a solution to the inequality
`(p⁻¹ + q⁻¹ + r⁻¹ : ℚ) > 1`.

These solutions are related to the Dynkin diagrams $E_{r+3}$. -/
def E' (r : ℕ+) : Multiset ℕ+ :=
  {2, 3, r}


/-- `E6 := {2,3,3}` is a `Multiset ℕ+`
that is a solution to the inequality
`(p⁻¹ + q⁻¹ + r⁻¹ : ℚ) > 1`.

This solution is related to the Dynkin diagrams $E_6$. -/
def E6 : Multiset ℕ+ :=
  E' 3


/-- `E7 := {2,3,4}` is a `Multiset ℕ+`
that is a solution to the inequality
`(p⁻¹ + q⁻¹ + r⁻¹ : ℚ) > 1`.

This solution is related to the Dynkin diagrams $E_7$. -/
def E7 : Multiset ℕ+ :=
  E' 4


/-- `E8 := {2,3,5}` is a `Multiset ℕ+`
that is a solution to the inequality
`(p⁻¹ + q⁻¹ + r⁻¹ : ℚ) > 1`.

This solution is related to the Dynkin diagrams $E_8$. -/
def E8 : Multiset ℕ+ :=
  E' 5


/-- `sum_inv pqr` for a `pqr : Multiset ℕ+` is the sum of the inverses
of the elements of `pqr`, as rational number.

The intended argument is a multiset `{p,q,r}` of cardinality `3`. -/
def sumInv (pqr : Multiset ℕ+) : ℚ :=
  Multiset.sum (pqr.map fun (x : ℕ+) => x⁻¹)


theorem sumInv_pqr (p q r : ℕ+) : sumInv {p, q, r} = (p : ℚ)⁻¹ + (q : ℚ)⁻¹ + (r : ℚ)⁻¹ := by
  simp only [sumInv, add_zero, insert_eq_cons, add_assoc, map_cons, sum_cons,
    map_singleton, sum_singleton]


/-- A multiset `pqr` of positive natural numbers is `admissible`
if it is equal to `A' q r`, or `D' r`, or one of `E6`, `E7`, or `E8`. -/
def Admissible (pqr : Multiset ℕ+) : Prop :=
  (∃ q r, A' q r = pqr) ∨ (∃ r, D' r = pqr) ∨ E' 3 = pqr ∨ E' 4 = pqr ∨ E' 5 = pqr


theorem admissible_A' (q r : ℕ+) : Admissible (A' q r) :=
  Or.inl ⟨q, r, rfl⟩


theorem admissible_D' (n : ℕ+) : Admissible (D' n) :=
  Or.inr <| Or.inl ⟨n, rfl⟩


theorem admissible_E'3 : Admissible (E' 3) :=
  Or.inr <| Or.inr <| Or.inl rfl


theorem admissible_E'4 : Admissible (E' 4) :=
  Or.inr <| Or.inr <| Or.inr <| Or.inl rfl


theorem admissible_E'5 : Admissible (E' 5) :=
  Or.inr <| Or.inr <| Or.inr <| Or.inr rfl


theorem admissible_E6 : Admissible E6 :=
  admissible_E'3


theorem admissible_E7 : Admissible E7 :=
  admissible_E'4


theorem admissible_E8 : Admissible E8 :=
  admissible_E'5


theorem Admissible.one_lt_sumInv {pqr : Multiset ℕ+} : Admissible pqr → 1 < sumInv pqr := by
  /-
    pqr : Multiset PNat
    ⊢ ADEInequality.Admissible pqr → LT.lt 1 (ADEInequality.sumInv pqr)
  -/
  rw [Admissible]
  /-
    pqr : Multiset PNat
    ⊢ Or (Exists fun q => Exists fun r => Eq (ADEInequality.A' q r) pqr) (Or (Exis …
  -/
  rintro (⟨p', q', H⟩ | ⟨n, H⟩ | H | H | H)
    /-
      case inl.intro.intro
      pqr : Multiset PNat
      p' q' : PNat
      H : Eq (ADEInequality.A' p' q') pqr
      ⊢ LT.lt 1 (ADEInequality.sumInv pqr)
    -/
  · rw [← H, A', sumInv_pqr, add_assoc]
    /-
      case inl.intro.intro
      pqr : Multiset PNat
      p' q' : PNat
      H : Eq (ADEInequality.A' p' q') pqr
      ⊢ LT.lt 1 (HAdd.hAdd (Inv.inv ↑↑1) (HAdd.hAdd (Inv.inv ↑↑p') (Inv.inv ↑↑q')))
    -/
    simp only [lt_add_iff_pos_right, PNat.one_coe, inv_one, Nat.cast_one]
    /-
      case inl.intro.intro
      pqr : Multiset PNat
      p' q' : PNat
      H : Eq (ADEInequality.A' p' q') pqr
      ⊢ LT.lt 0 (HAdd.hAdd (Inv.inv ↑↑p') (Inv.inv ↑↑q'))
    -/
                      /-
                        🎉 no goals
                      -/
    apply add_pos <;> simp only [PNat.pos, Nat.cast_pos, inv_pos]
                      /-
                        🎉 no goals
                      -/
    /-
      case inr.inl.intro
      pqr : Multiset PNat
      n : PNat
      H : Eq (ADEInequality.D' n) pqr
      ⊢ LT.lt 1 (ADEInequality.sumInv pqr)
    -/
  · rw [← H, D', sumInv_pqr]
    /-
      case inr.inl.intro
      pqr : Multiset PNat
      n : PNat
      H : Eq (ADEInequality.D' n) pqr
      ⊢ LT.lt 1 (HAdd.hAdd (HAdd.hAdd (Inv.inv ↑↑2) (Inv.inv ↑↑2)) (Inv.inv ↑↑n))
    -/
    norm_num
    /-
      🎉 no goals
    -/
  all_goals
    rw [← H, E', sumInv_pqr]
    norm_num


theorem lt_three {p q r : ℕ+} (hpq : p ≤ q) (hqr : q ≤ r) (H : 1 < sumInv {p, q, r}) : p < 3 := by
  /-
    p q r : PNat
    hpq : LE.le p q
    hqr : LE.le q r
    H : LT.lt 1 (ADEInequality.sumInv (Insert.insert p (Insert.insert q (Singleton …
    ⊢ LT.lt p 3
  -/
  have h3 : (0 : ℚ) < 3 := by norm_num
  /-
    p q r : PNat
    hpq : LE.le p q
    hqr : LE.le q r
    H : LT.lt 1 (ADEInequality.sumInv (Insert.insert p (Insert.insert q (Singleton …
    h3 : LT.lt 0 3
    ⊢ LT.lt p 3
  -/
  contrapose! H
  /-
    p q r : PNat
    hpq : LE.le p q
    hqr : LE.le q r
    h3 : LT.lt 0 3
    H : LE.le 3 p
    ⊢ LE.le (ADEInequality.sumInv (Insert.insert p (Insert.insert q (Singleton.sin …
  -/
  rw [sumInv_pqr]
  /-
    p q r : PNat
    hpq : LE.le p q
    hqr : LE.le q r
    h3 : LT.lt 0 3
    H : LE.le 3 p
    ⊢ LE.le (HAdd.hAdd (HAdd.hAdd (Inv.inv ↑↑p) (Inv.inv ↑↑q)) (Inv.inv ↑↑r)) 1
  -/
  have h3q := H.trans hpq
  /-
    p q r : PNat
    hpq : LE.le p q
    hqr : LE.le q r
    h3 : LT.lt 0 3
    H : LE.le 3 p
    h3q : LE.le 3 q
    ⊢ LE.le (HAdd.hAdd (HAdd.hAdd (Inv.inv ↑↑p) (Inv.inv ↑↑q)) (Inv.inv ↑↑r)) 1
  -/
  have h3r := h3q.trans hqr
  have hp : (p : ℚ)⁻¹ ≤ 3⁻¹ := by
    rw [inv_le_inv₀ _ h3]
    · assumption_mod_cast
    · norm_num
  have hq : (q : ℚ)⁻¹ ≤ 3⁻¹ := by
    rw [inv_le_inv₀ _ h3]
    · assumption_mod_cast
    · norm_num
  have hr : (r : ℚ)⁻¹ ≤ 3⁻¹ := by
    rw [inv_le_inv₀ _ h3]
    · assumption_mod_cast
    · norm_num
  calc
    (p : ℚ)⁻¹ + (q : ℚ)⁻¹ + (r : ℚ)⁻¹ ≤ 3⁻¹ + 3⁻¹ + 3⁻¹ := add_le_add (add_le_add hp hq) hr
    _ = 1 := by norm_num


theorem lt_four {q r : ℕ+} (hqr : q ≤ r) (H : 1 < sumInv {2, q, r}) : q < 4 := by
  /-
    q r : PNat
    hqr : LE.le q r
    H : LT.lt 1 (ADEInequality.sumInv (Insert.insert 2 (Insert.insert q (Singleton …
    ⊢ LT.lt q 4
  -/
  have h4 : (0 : ℚ) < 4 := by norm_num
  /-
    q r : PNat
    hqr : LE.le q r
    H : LT.lt 1 (ADEInequality.sumInv (Insert.insert 2 (Insert.insert q (Singleton …
    h4 : LT.lt 0 4
    ⊢ LT.lt q 4
  -/
  contrapose! H
  /-
    q r : PNat
    hqr : LE.le q r
    h4 : LT.lt 0 4
    H : LE.le 4 q
    ⊢ LE.le (ADEInequality.sumInv (Insert.insert 2 (Insert.insert q (Singleton.sin …
  -/
  rw [sumInv_pqr]
  /-
    q r : PNat
    hqr : LE.le q r
    h4 : LT.lt 0 4
    H : LE.le 4 q
    ⊢ LE.le (HAdd.hAdd (HAdd.hAdd (Inv.inv ↑↑2) (Inv.inv ↑↑q)) (Inv.inv ↑↑r)) 1
  -/
  have h4r := H.trans hqr
  have hq : (q : ℚ)⁻¹ ≤ 4⁻¹ := by
    rw [inv_le_inv₀ _ h4]
    · assumption_mod_cast
    · norm_num
  have hr : (r : ℚ)⁻¹ ≤ 4⁻¹ := by
    rw [inv_le_inv₀ _ h4]
    · assumption_mod_cast
    · norm_num
  calc
    (2⁻¹ + (q : ℚ)⁻¹ + (r : ℚ)⁻¹) ≤ 2⁻¹ + 4⁻¹ + 4⁻¹ := add_le_add (add_le_add le_rfl hq) hr
    _ = 1 := by norm_num


theorem lt_six {r : ℕ+} (H : 1 < sumInv {2, 3, r}) : r < 6 := by
  /-
    r : PNat
    H : LT.lt 1 (ADEInequality.sumInv (Insert.insert 2 (Insert.insert 3 (Singleton …
    ⊢ LT.lt r 6
  -/
  have h6 : (0 : ℚ) < 6 := by norm_num
  /-
    r : PNat
    H : LT.lt 1 (ADEInequality.sumInv (Insert.insert 2 (Insert.insert 3 (Singleton …
    h6 : LT.lt 0 6
    ⊢ LT.lt r 6
  -/
  contrapose! H
  /-
    r : PNat
    h6 : LT.lt 0 6
    H : LE.le 6 r
    ⊢ LE.le (ADEInequality.sumInv (Insert.insert 2 (Insert.insert 3 (Singleton.sin …
  -/
  rw [sumInv_pqr]
  have hr : (r : ℚ)⁻¹ ≤ 6⁻¹ := by
    rw [inv_le_inv₀ _ h6]
    · assumption_mod_cast
    · norm_num
  calc
    (2⁻¹ + 3⁻¹ + (r : ℚ)⁻¹ : ℚ) ≤ 2⁻¹ + 3⁻¹ + 6⁻¹ := add_le_add (add_le_add le_rfl le_rfl) hr
    _ = 1 := by norm_num


theorem admissible_of_one_lt_sumInv_aux' {p q r : ℕ+} (hpq : p ≤ q) (hqr : q ≤ r)
    (H : 1 < sumInv {p, q, r}) : Admissible {p, q, r} := by
  /-
    p q r : PNat
    hpq : LE.le p q
    hqr : LE.le q r
    H : LT.lt 1 (ADEInequality.sumInv (Insert.insert p (Insert.insert q (Singleton …
    ⊢ ADEInequality.Admissible (Insert.insert p (Insert.insert q (Singleton.single …
  -/
  have hp3 : p < 3 := lt_three hpq hqr H
  -- Porting note: `interval_cases` doesn't support `ℕ+` yet.
  /-
    p q r : PNat
    hpq : LE.le p q
    hqr : LE.le q r
    H : LT.lt 1 (ADEInequality.sumInv (Insert.insert p (Insert.insert q (Singleton …
    hp3 : LT.lt p 3
    ⊢ ADEInequality.Admissible (Insert.insert p (Insert.insert q (Singleton.single …
  -/
  replace hp3 := Finset.mem_Iio.mpr hp3
  /-
    p q r : PNat
    hpq : LE.le p q
    hqr : LE.le q r
    H : LT.lt 1 (ADEInequality.sumInv (Insert.insert p (Insert.insert q (Singleton …
    hp3 : Membership.mem (Finset.Iio 3) p
    ⊢ ADEInequality.Admissible (Insert.insert p (Insert.insert q (Singleton.single …
  -/
  conv at hp3 => change p ∈ ({1, 2} : Multiset ℕ+)
  /-
    p q r : PNat
    hpq : LE.le p q
    hqr : LE.le q r
    H : LT.lt 1 (ADEInequality.sumInv (Insert.insert p (Insert.insert q (Singleton …
    hp3 : Membership.mem (Insert.insert 1 (Singleton.singleton 2)) p
    ⊢ ADEInequality.Admissible (Insert.insert p (Insert.insert q (Singleton.single …
  -/
  fin_cases hp3
    /-
      case «0»
      q r : PNat
      hqr : LE.le q r
      hpq : LE.le 1 q
      H : LT.lt 1 (ADEInequality.sumInv (Insert.insert 1 (Insert.insert q (Singleton …
      ⊢ ADEInequality.Admissible (Insert.insert 1 (Insert.insert q (Singleton.single …
    -/
  · exact admissible_A' q r
    /-
      🎉 no goals
    -/
  /-
    case «1»
    q r : PNat
    hqr : LE.le q r
    hpq : LE.le 2 q
    H : LT.lt 1 (ADEInequality.sumInv (Insert.insert 2 (Insert.insert q (Singleton …
    ⊢ ADEInequality.Admissible (Insert.insert 2 (Insert.insert q (Singleton.single …
  -/
  have hq4 : q < 4 := lt_four hqr H
  /-
    case «1»
    q r : PNat
    hqr : LE.le q r
    hpq : LE.le 2 q
    H : LT.lt 1 (ADEInequality.sumInv (Insert.insert 2 (Insert.insert q (Singleton …
    hq4 : LT.lt q 4
    ⊢ ADEInequality.Admissible (Insert.insert 2 (Insert.insert q (Singleton.single …
  -/
  replace hq4 := Finset.mem_Ico.mpr ⟨hpq, hq4⟩; clear hpq
  /-
    case «1»
    q r : PNat
    hqr : LE.le q r
    H : LT.lt 1 (ADEInequality.sumInv (Insert.insert 2 (Insert.insert q (Singleton …
    hq4 : Membership.mem (Finset.Ico 2 4) q
    ⊢ ADEInequality.Admissible (Insert.insert 2 (Insert.insert q (Singleton.single …
  -/
  conv at hq4 => change q ∈ ({2, 3} : Multiset ℕ+)
  /-
    case «1»
    q r : PNat
    hqr : LE.le q r
    H : LT.lt 1 (ADEInequality.sumInv (Insert.insert 2 (Insert.insert q (Singleton …
    hq4 : Membership.mem (Insert.insert 2 (Singleton.singleton 3)) q
    ⊢ ADEInequality.Admissible (Insert.insert 2 (Insert.insert q (Singleton.single …
  -/
  fin_cases hq4
    /-
      case «1».«0»
      r : PNat
      hqr : LE.le 2 r
      H : LT.lt 1 (ADEInequality.sumInv (Insert.insert 2 (Insert.insert 2 (Singleton …
      ⊢ ADEInequality.Admissible (Insert.insert 2 (Insert.insert 2 (Singleton.single …
    -/
  · exact admissible_D' r
    /-
      🎉 no goals
    -/
  /-
    case «1».«1»
    r : PNat
    hqr : LE.le 3 r
    H : LT.lt 1 (ADEInequality.sumInv (Insert.insert 2 (Insert.insert 3 (Singleton …
    ⊢ ADEInequality.Admissible (Insert.insert 2 (Insert.insert 3 (Singleton.single …
  -/
  have hr6 : r < 6 := lt_six H
  /-
    case «1».«1»
    r : PNat
    hqr : LE.le 3 r
    H : LT.lt 1 (ADEInequality.sumInv (Insert.insert 2 (Insert.insert 3 (Singleton …
    hr6 : LT.lt r 6
    ⊢ ADEInequality.Admissible (Insert.insert 2 (Insert.insert 3 (Singleton.single …
  -/
  replace hr6 := Finset.mem_Ico.mpr ⟨hqr, hr6⟩; clear hqr
  /-
    case «1».«1»
    r : PNat
    H : LT.lt 1 (ADEInequality.sumInv (Insert.insert 2 (Insert.insert 3 (Singleton …
    hr6 : Membership.mem (Finset.Ico 3 6) r
    ⊢ ADEInequality.Admissible (Insert.insert 2 (Insert.insert 3 (Singleton.single …
  -/
  conv at hr6 => change r ∈ ({3, 4, 5} : Multiset ℕ+)
  /-
    case «1».«1»
    r : PNat
    H : LT.lt 1 (ADEInequality.sumInv (Insert.insert 2 (Insert.insert 3 (Singleton …
    hr6 : Membership.mem (Insert.insert 3 (Insert.insert 4 (Singleton.singleton 5) …
    ⊢ ADEInequality.Admissible (Insert.insert 2 (Insert.insert 3 (Singleton.single …
  -/
  fin_cases hr6
    /-
      case «1».«1».«0»
      H : LT.lt 1 (ADEInequality.sumInv (Insert.insert 2 (Insert.insert 3 (Singleton …
      ⊢ ADEInequality.Admissible (Insert.insert 2 (Insert.insert 3 (Singleton.single …
    -/
  · exact admissible_E6
    /-
      🎉 no goals
    -/
    /-
      case «1».«1».«1»
      H : LT.lt 1 (ADEInequality.sumInv (Insert.insert 2 (Insert.insert 3 (Singleton …
      ⊢ ADEInequality.Admissible (Insert.insert 2 (Insert.insert 3 (Singleton.single …
    -/
  · exact admissible_E7
    /-
      🎉 no goals
    -/
    /-
      case «1».«1».«2»
      H : LT.lt 1 (ADEInequality.sumInv (Insert.insert 2 (Insert.insert 3 (Singleton …
      ⊢ ADEInequality.Admissible (Insert.insert 2 (Insert.insert 3 (Singleton.single …
    -/
  · exact admissible_E8
    /-
      🎉 no goals
    -/


theorem admissible_of_one_lt_sumInv_aux :
    ∀ {pqr : List ℕ+} (_ : pqr.Sorted (· ≤ ·)) (_ : pqr.length = 3) (_ : 1 < sumInv pqr),
      Admissible pqr
  | [p, q, r], hs, _, H => by
    /-
      p q r : PNat
      hs : List.Sorted (fun x1 x2 => LE.le x1 x2) (List.cons p (List.cons q (List.co …
      x✝ : Eq (List.cons p (List.cons q (List.cons r List.nil))).length 3
      H : LT.lt 1 (ADEInequality.sumInv ↑(List.cons p (List.cons q (List.cons r List …
      ⊢ ADEInequality.Admissible ↑(List.cons p (List.cons q (List.cons r List.nil)))
    -/
    obtain ⟨⟨hpq, -⟩, hqr⟩ : (p ≤ q ∧ p ≤ r) ∧ q ≤ r := by simpa using hs
    /-
      case intro.intro
      p q r : PNat
      hs : List.Sorted (fun x1 x2 => LE.le x1 x2) (List.cons p (List.cons q (List.co …
      x✝ : Eq (List.cons p (List.cons q (List.cons r List.nil))).length 3
      H : LT.lt 1 (ADEInequality.sumInv ↑(List.cons p (List.cons q (List.cons r List …
      hqr : LE.le q r
      hpq : LE.le p q
      ⊢ ADEInequality.Admissible ↑(List.cons p (List.cons q (List.cons r List.nil)))
    -/
    exact admissible_of_one_lt_sumInv_aux' hpq hqr H
    /-
      🎉 no goals
    -/


theorem admissible_of_one_lt_sumInv {p q r : ℕ+} (H : 1 < sumInv {p, q, r}) :
    Admissible {p, q, r} := by
  /-
    p q r : PNat
    H : LT.lt 1 (ADEInequality.sumInv (Insert.insert p (Insert.insert q (Singleton …
    ⊢ ADEInequality.Admissible (Insert.insert p (Insert.insert q (Singleton.single …
  -/
  simp only [Admissible]
  /-
    p q r : PNat
    H : LT.lt 1 (ADEInequality.sumInv (Insert.insert p (Insert.insert q (Singleton …
    ⊢ Or (Exists fun q_1 => Exists fun r_1 => Eq (ADEInequality.A' q_1 r_1) (Inser …
  -/
  let S := sort ((· ≤ ·) : ℕ+ → ℕ+ → Prop) {p, q, r}
  /-
    p q r : PNat
    H : LT.lt 1 (ADEInequality.sumInv (Insert.insert p (Insert.insert q (Singleton …
    S : List PNat := Multiset.sort (fun x1 x2 => LE.le x1 x2) (Insert.insert p (In …
    ⊢ Or (Exists fun q_1 => Exists fun r_1 => Eq (ADEInequality.A' q_1 r_1) (Inser …
  -/
  have hS : S.Sorted (· ≤ ·) := sort_sorted _ _
  /-
    p q r : PNat
    H : LT.lt 1 (ADEInequality.sumInv (Insert.insert p (Insert.insert q (Singleton …
    S : List PNat := Multiset.sort (fun x1 x2 => LE.le x1 x2) (Insert.insert p (In …
    hS : List.Sorted (fun x1 x2 => LE.le x1 x2) S
    ⊢ Or (Exists fun q_1 => Exists fun r_1 => Eq (ADEInequality.A' q_1 r_1) (Inser …
  -/
  have hpqr : ({p, q, r} : Multiset ℕ+) = S := (sort_eq LE.le {p, q, r}).symm
  /-
    p q r : PNat
    H : LT.lt 1 (ADEInequality.sumInv (Insert.insert p (Insert.insert q (Singleton …
    S : List PNat := Multiset.sort (fun x1 x2 => LE.le x1 x2) (Insert.insert p (In …
    hS : List.Sorted (fun x1 x2 => LE.le x1 x2) S
    hpqr : Eq (Insert.insert p (Insert.insert q (Singleton.singleton r))) ↑S
    ⊢ Or (Exists fun q_1 => Exists fun r_1 => Eq (ADEInequality.A' q_1 r_1) (Inser …
  -/
  rw [hpqr]
  /-
    p q r : PNat
    H : LT.lt 1 (ADEInequality.sumInv (Insert.insert p (Insert.insert q (Singleton …
    S : List PNat := Multiset.sort (fun x1 x2 => LE.le x1 x2) (Insert.insert p (In …
    hS : List.Sorted (fun x1 x2 => LE.le x1 x2) S
    hpqr : Eq (Insert.insert p (Insert.insert q (Singleton.singleton r))) ↑S
    ⊢ Or (Exists fun q => Exists fun r => Eq (ADEInequality.A' q r) ↑S) (Or (Exist …
  -/
  rw [hpqr] at H
  /-
    p q r : PNat
    S : List PNat := Multiset.sort (fun x1 x2 => LE.le x1 x2) (Insert.insert p (In …
    H : LT.lt 1 (ADEInequality.sumInv ↑S)
    hS : List.Sorted (fun x1 x2 => LE.le x1 x2) S
    hpqr : Eq (Insert.insert p (Insert.insert q (Singleton.singleton r))) ↑S
    ⊢ Or (Exists fun q => Exists fun r => Eq (ADEInequality.A' q r) ↑S) (Or (Exist …
  -/
  apply admissible_of_one_lt_sumInv_aux hS _ H
  /-
    p q r : PNat
    S : List PNat := Multiset.sort (fun x1 x2 => LE.le x1 x2) (Insert.insert p (In …
    H : LT.lt 1 (ADEInequality.sumInv ↑S)
    hS : List.Sorted (fun x1 x2 => LE.le x1 x2) S
    hpqr : Eq (Insert.insert p (Insert.insert q (Singleton.singleton r))) ↑S
    ⊢ Eq S.length 3
  -/
  simp only [S, insert_eq_cons, length_sort, card_cons, card_singleton]
  /-
    🎉 no goals
  -/


/-- A multiset `{p,q,r}` of positive natural numbers
is a solution to `(p⁻¹ + q⁻¹ + r⁻¹ : ℚ) > 1` if and only if
it is `admissible` which means it is one of:

* `A' q r := {1,q,r}`
* `D' r := {2,2,r}`
* `E6 := {2,3,3}`, or `E7 := {2,3,4}`, or `E8 := {2,3,5}`
-/
theorem classification (p q r : ℕ+) : 1 < sumInv {p, q, r} ↔ Admissible {p, q, r} :=
  ⟨admissible_of_one_lt_sumInv, Admissible.one_lt_sumInv⟩


