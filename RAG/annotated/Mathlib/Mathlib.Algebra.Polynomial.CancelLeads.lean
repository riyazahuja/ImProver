/-- `cancelLeads p q` is formed by multiplying `p` and `q` by monomials so that they
  have the same leading term, and then subtracting. -/
def cancelLeads : R[X] :=
  C p.leadingCoeff * X ^ (p.natDegree - q.natDegree) * q -
    C q.leadingCoeff * X ^ (q.natDegree - p.natDegree) * p


@[simp]
theorem neg_cancelLeads : -p.cancelLeads q = q.cancelLeads p :=
  neg_sub _ _


theorem natDegree_cancelLeads_lt_of_natDegree_le_natDegree_of_comm
    (comm : p.leadingCoeff * q.leadingCoeff = q.leadingCoeff * p.leadingCoeff)
    (h : p.natDegree ≤ q.natDegree) (hq : 0 < q.natDegree) :
    (p.cancelLeads q).natDegree < q.natDegree := by
  /-
    R : Type u_1
    inst✝ : Ring R
    p q : Polynomial R
    comm : Eq (HMul.hMul p.leadingCoeff q.leadingCoeff) (HMul.hMul q.leadingCoeff  …
    h : LE.le p.natDegree q.natDegree
    hq : LT.lt 0 q.natDegree
    ⊢ LT.lt (p.cancelLeads q).natDegree q.natDegree
  -/
  by_cases hp : p = 0
    /-
      case pos
      R : Type u_1
      inst✝ : Ring R
      p q : Polynomial R
      comm : Eq (HMul.hMul p.leadingCoeff q.leadingCoeff) (HMul.hMul q.leadingCoeff  …
      h : LE.le p.natDegree q.natDegree
      hq : LT.lt 0 q.natDegree
      hp : Eq p 0
      ⊢ LT.lt (p.cancelLeads q).natDegree q.natDegree
    -/
  · convert hq
    /-
      case h.e'_3
      R : Type u_1
      inst✝ : Ring R
      p q : Polynomial R
      comm : Eq (HMul.hMul p.leadingCoeff q.leadingCoeff) (HMul.hMul q.leadingCoeff  …
      h : LE.le p.natDegree q.natDegree
      hq : LT.lt 0 q.natDegree
      hp : Eq p 0
      ⊢ Eq (p.cancelLeads q).natDegree 0
    -/
    simp [hp, cancelLeads]
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u_1
    inst✝ : Ring R
    p q : Polynomial R
    comm : Eq (HMul.hMul p.leadingCoeff q.leadingCoeff) (HMul.hMul q.leadingCoeff  …
    h : LE.le p.natDegree q.natDegree
    hq : LT.lt 0 q.natDegree
    hp : Not (Eq p 0)
    ⊢ LT.lt (p.cancelLeads q).natDegree q.natDegree
  -/
  rw [cancelLeads, sub_eq_add_neg, tsub_eq_zero_iff_le.mpr h, pow_zero, mul_one]
  by_cases h0 :
    C p.leadingCoeff * q + -(C q.leadingCoeff * X ^ (q.natDegree - p.natDegree) * p) = 0
    /-
      case pos
      R : Type u_1
      inst✝ : Ring R
      p q : Polynomial R
      comm : Eq (HMul.hMul p.leadingCoeff q.leadingCoeff) (HMul.hMul q.leadingCoeff  …
      h : LE.le p.natDegree q.natDegree
      hq : LT.lt 0 q.natDegree
      hp : Not (Eq p 0)
      h0 : Eq (HAdd.hAdd (HMul.hMul (Polynomial.C p.leadingCoeff) q) (Neg.neg (HMul. …
      ⊢ LT.lt (HAdd.hAdd (HMul.hMul (Polynomial.C p.leadingCoeff) q) (Neg.neg (HMul. …
    -/
  · exact (le_of_eq (by simp only [h0, natDegree_zero])).trans_lt hq
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u_1
    inst✝ : Ring R
    p q : Polynomial R
    comm : Eq (HMul.hMul p.leadingCoeff q.leadingCoeff) (HMul.hMul q.leadingCoeff  …
    h : LE.le p.natDegree q.natDegree
    hq : LT.lt 0 q.natDegree
    hp : Not (Eq p 0)
    h0 : Not (Eq (HAdd.hAdd (HMul.hMul (Polynomial.C p.leadingCoeff) q) (Neg.neg ( …
    ⊢ LT.lt (HAdd.hAdd (HMul.hMul (Polynomial.C p.leadingCoeff) q) (Neg.neg (HMul. …
  -/
  apply lt_of_le_of_ne
    /-
      case neg.a
      R : Type u_1
      inst✝ : Ring R
      p q : Polynomial R
      comm : Eq (HMul.hMul p.leadingCoeff q.leadingCoeff) (HMul.hMul q.leadingCoeff  …
      h : LE.le p.natDegree q.natDegree
      hq : LT.lt 0 q.natDegree
      hp : Not (Eq p 0)
      h0 : Not (Eq (HAdd.hAdd (HMul.hMul (Polynomial.C p.leadingCoeff) q) (Neg.neg ( …
      ⊢ LE.le (HAdd.hAdd (HMul.hMul (Polynomial.C p.leadingCoeff) q) (Neg.neg (HMul. …
    -/
  · compute_degree!
    /-
      case neg.a.a
      R : Type u_1
      inst✝ : Ring R
      p q : Polynomial R
      comm : Eq (HMul.hMul p.leadingCoeff q.leadingCoeff) (HMul.hMul q.leadingCoeff  …
      h : LE.le p.natDegree q.natDegree
      hq : LT.lt 0 q.natDegree
      hp : Not (Eq p 0)
      h0 : Not (Eq (HAdd.hAdd (HMul.hMul (Polynomial.C p.leadingCoeff) q) (Neg.neg ( …
      ⊢ LE.le (HAdd.hAdd (HSub.hSub q.natDegree p.natDegree) p.natDegree) q.natDegree
    -/
    rwa [Nat.sub_add_cancel]
    /-
      🎉 no goals
    -/
    /-
      case neg.a
      R : Type u_1
      inst✝ : Ring R
      p q : Polynomial R
      comm : Eq (HMul.hMul p.leadingCoeff q.leadingCoeff) (HMul.hMul q.leadingCoeff  …
      h : LE.le p.natDegree q.natDegree
      hq : LT.lt 0 q.natDegree
      hp : Not (Eq p 0)
      h0 : Not (Eq (HAdd.hAdd (HMul.hMul (Polynomial.C p.leadingCoeff) q) (Neg.neg ( …
      ⊢ Ne (HAdd.hAdd (HMul.hMul (Polynomial.C p.leadingCoeff) q) (Neg.neg (HMul.hMu …
    -/
  · contrapose! h0
    rw [← leadingCoeff_eq_zero, leadingCoeff, h0, mul_assoc, X_pow_mul, ← tsub_add_cancel_of_le h,
      add_comm _ p.natDegree]
    /-
      case neg.a
      R : Type u_1
      inst✝ : Ring R
      p q : Polynomial R
      comm : Eq (HMul.hMul p.leadingCoeff q.leadingCoeff) (HMul.hMul q.leadingCoeff  …
      h : LE.le p.natDegree q.natDegree
      hq : LT.lt 0 q.natDegree
      hp : Not (Eq p 0)
      h0 : Eq (HAdd.hAdd (HMul.hMul (Polynomial.C p.leadingCoeff) q) (Neg.neg (HMul. …
      ⊢ Eq ((HAdd.hAdd (HMul.hMul (Polynomial.C p.leadingCoeff) q) (Neg.neg (HMul.hM …
    -/
    simp only [coeff_mul_X_pow, coeff_neg, coeff_C_mul, add_tsub_cancel_left, coeff_add]
    rw [add_comm p.natDegree, tsub_add_cancel_of_le h, ← leadingCoeff, ← leadingCoeff, comm,
      add_neg_cancel]


theorem dvd_cancelLeads_of_dvd_of_dvd {r : R[X]} (pq : p ∣ q) (pr : p ∣ r) : p ∣ q.cancelLeads r :=
  dvd_sub (pr.trans (Dvd.intro_left _ rfl)) (pq.trans (Dvd.intro_left _ rfl))


theorem natDegree_cancelLeads_lt_of_natDegree_le_natDegree (h : p.natDegree ≤ q.natDegree)
    (hq : 0 < q.natDegree) : (p.cancelLeads q).natDegree < q.natDegree :=
  natDegree_cancelLeads_lt_of_natDegree_le_natDegree_of_comm (mul_comm _ _) h hq


