theorem supDegree_eq_degree (p : R[X]) : p.toFinsupp.supDegree WithBot.some = p.degree :=
  max_eq_sup_coe


theorem degree_lt_wf : WellFounded fun p q : R[X] => degree p < degree q :=
  InvImage.wf degree wellFounded_lt


instance : WellFoundedRelation R[X] :=
  ⟨_, degree_lt_wf⟩


@[nontriviality]
theorem monic_of_subsingleton [Subsingleton R] (p : R[X]) : Monic p :=
  Subsingleton.elim _ _


@[nontriviality]
theorem degree_of_subsingleton [Subsingleton R] : degree p = ⊥ := by
  /-
    R : Type u
    inst✝¹ : Semiring R
    p : Polynomial R
    inst✝ : Subsingleton R
    ⊢ Eq p.degree Bot.bot
  -/
  rw [Subsingleton.elim p 0, degree_zero]
  /-
    🎉 no goals
  -/


@[nontriviality]
theorem natDegree_of_subsingleton [Subsingleton R] : natDegree p = 0 := by
  /-
    R : Type u
    inst✝¹ : Semiring R
    p : Polynomial R
    inst✝ : Subsingleton R
    ⊢ Eq p.natDegree 0
  -/
  rw [Subsingleton.elim p 0, natDegree_zero]
  /-
    🎉 no goals
  -/


theorem le_natDegree_of_ne_zero (h : coeff p n ≠ 0) : n ≤ natDegree p := by
  /-
    R : Type u
    n : Nat
    inst✝ : Semiring R
    p : Polynomial R
    h : Ne (p.coeff n) 0
    ⊢ LE.le n p.natDegree
  -/
  rw [← Nat.cast_le (α := WithBot ℕ), ← degree_eq_natDegree]
    /-
      R : Type u
      n : Nat
      inst✝ : Semiring R
      p : Polynomial R
      h : Ne (p.coeff n) 0
      ⊢ LE.le (↑n) p.degree
    -/
  · exact le_degree_of_ne_zero h
    /-
      🎉 no goals
    -/
    /-
      R : Type u
      n : Nat
      inst✝ : Semiring R
      p : Polynomial R
      h : Ne (p.coeff n) 0
      ⊢ Ne p 0
    -/
  · rintro rfl
    /-
      R : Type u
      n : Nat
      inst✝ : Semiring R
      h : Ne (Polynomial.coeff 0 n) 0
      ⊢ False
    -/
    exact h rfl
    /-
      🎉 no goals
    -/


theorem degree_eq_of_le_of_coeff_ne_zero (pn : p.degree ≤ n) (p1 : p.coeff n ≠ 0) : p.degree = n :=
  pn.antisymm (le_degree_of_ne_zero p1)


theorem natDegree_eq_of_le_of_coeff_ne_zero (pn : p.natDegree ≤ n) (p1 : p.coeff n ≠ 0) :
    p.natDegree = n :=
  pn.antisymm (le_natDegree_of_ne_zero p1)


theorem natDegree_lt_natDegree {q : S[X]} (hp : p ≠ 0) (hpq : p.degree < q.degree) :
    p.natDegree < q.natDegree := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    p : Polynomial R
    q : Polynomial S
    hp : Ne p 0
    hpq : LT.lt p.degree q.degree
    ⊢ LT.lt p.natDegree q.natDegree
  -/
  by_cases hq : q = 0
    /-
      case pos
      R : Type u
      S : Type v
      inst✝¹ : Semiring R
      inst✝ : Semiring S
      p : Polynomial R
      q : Polynomial S
      hp : Ne p 0
      hpq : LT.lt p.degree q.degree
      hq : Eq q 0
      ⊢ LT.lt p.natDegree q.natDegree
    -/
  · exact (not_lt_bot <| hq ▸ hpq).elim
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    p : Polynomial R
    q : Polynomial S
    hp : Ne p 0
    hpq : LT.lt p.degree q.degree
    hq : Not (Eq q 0)
    ⊢ LT.lt p.natDegree q.natDegree
  -/
  rwa [degree_eq_natDegree hp, degree_eq_natDegree hq, Nat.cast_lt] at hpq
  /-
    🎉 no goals
  -/


lemma natDegree_eq_natDegree {q : S[X]} (hpq : p.degree = q.degree) :
                                    /-
                                      R : Type u
                                      S : Type v
                                      inst✝¹ : Semiring R
                                      inst✝ : Semiring S
                                      p : Polynomial R
                                      q : Polynomial S
                                      hpq : Eq p.degree q.degree
                                      ⊢ Eq p.natDegree q.natDegree
                                    -/
    p.natDegree = q.natDegree := by simp [natDegree, hpq]
                                    /-
                                      🎉 no goals
                                    -/


theorem coeff_eq_zero_of_degree_lt (h : degree p < n) : coeff p n = 0 :=
  Classical.not_not.1 (mt le_degree_of_ne_zero (not_le_of_gt h))


theorem coeff_eq_zero_of_natDegree_lt {p : R[X]} {n : ℕ} (h : p.natDegree < n) :
    p.coeff n = 0 := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    n : Nat
    h : LT.lt p.natDegree n
    ⊢ Eq (p.coeff n) 0
  -/
  apply coeff_eq_zero_of_degree_lt
  /-
    case h
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    n : Nat
    h : LT.lt p.natDegree n
    ⊢ LT.lt p.degree ↑n
  -/
  by_cases hp : p = 0
    /-
      case pos
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      n : Nat
      h : LT.lt p.natDegree n
      hp : Eq p 0
      ⊢ LT.lt p.degree ↑n
    -/
  · subst hp
    /-
      case pos
      R : Type u
      inst✝ : Semiring R
      n : Nat
      h : LT.lt (Polynomial.natDegree 0) n
      ⊢ LT.lt (Polynomial.degree 0) ↑n
    -/
    exact WithBot.bot_lt_coe n
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      n : Nat
      h : LT.lt p.natDegree n
      hp : Not (Eq p 0)
      ⊢ LT.lt p.degree ↑n
    -/
  · rwa [degree_eq_natDegree hp, Nat.cast_lt]
    /-
      🎉 no goals
    -/


theorem ext_iff_natDegree_le {p q : R[X]} {n : ℕ} (hp : p.natDegree ≤ n) (hq : q.natDegree ≤ n) :
    p = q ↔ ∀ i ≤ n, p.coeff i = q.coeff i := by
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    n : Nat
    hp : LE.le p.natDegree n
    hq : LE.le q.natDegree n
    ⊢ Iff (Eq p q) (∀ (i : Nat), LE.le i n → Eq (p.coeff i) (q.coeff i))
  -/
  refine Iff.trans Polynomial.ext_iff ?_
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    n : Nat
    hp : LE.le p.natDegree n
    hq : LE.le q.natDegree n
    ⊢ Iff (∀ (n : Nat), Eq (p.coeff n) (q.coeff n)) (∀ (i : Nat), LE.le i n → Eq ( …
  -/
  refine forall_congr' fun i => ⟨fun h _ => h, fun h => ?_⟩
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    n : Nat
    hp : LE.le p.natDegree n
    hq : LE.le q.natDegree n
    i : Nat
    h : LE.le i n → Eq (p.coeff i) (q.coeff i)
    ⊢ Eq (p.coeff i) (q.coeff i)
  -/
  refine (le_or_lt i n).elim h fun k => ?_
  exact
    (coeff_eq_zero_of_natDegree_lt (hp.trans_lt k)).trans
      (coeff_eq_zero_of_natDegree_lt (hq.trans_lt k)).symm


theorem ext_iff_degree_le {p q : R[X]} {n : ℕ} (hp : p.degree ≤ n) (hq : q.degree ≤ n) :
    p = q ↔ ∀ i ≤ n, p.coeff i = q.coeff i :=
  ext_iff_natDegree_le (natDegree_le_of_degree_le hp) (natDegree_le_of_degree_le hq)


@[simp]
theorem coeff_natDegree_succ_eq_zero {p : R[X]} : p.coeff (p.natDegree + 1) = 0 :=
  coeff_eq_zero_of_natDegree_lt (lt_add_one _)

-- We need the explicit `Decidable` argument here because an exotic one shows up in a moment!

theorem ite_le_natDegree_coeff (p : R[X]) (n : ℕ) (I : Decidable (n < 1 + natDegree p)) :
    @ite _ (n < 1 + natDegree p) I (coeff p n) 0 = coeff p n := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    n : Nat
    I : Decidable (LT.lt n (HAdd.hAdd 1 p.natDegree))
    ⊢ Eq (ite (LT.lt n (HAdd.hAdd 1 p.natDegree)) (p.coeff n) 0) (p.coeff n)
  -/
  split_ifs with h
    /-
      case pos
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      n : Nat
      I : Decidable (LT.lt n (HAdd.hAdd 1 p.natDegree))
      h : LT.lt n (HAdd.hAdd 1 p.natDegree)
      ⊢ Eq (p.coeff n) (p.coeff n)
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      n : Nat
      I : Decidable (LT.lt n (HAdd.hAdd 1 p.natDegree))
      h : Not (LT.lt n (HAdd.hAdd 1 p.natDegree))
      ⊢ Eq 0 (p.coeff n)
    -/
  · exact (coeff_eq_zero_of_natDegree_lt (not_le.1 fun w => h (Nat.lt_one_add_iff.2 w))).symm
    /-
      🎉 no goals
    -/


theorem coeff_mul_X_sub_C {p : R[X]} {r : R} {a : ℕ} :
                                                                          /-
                                                                            R : Type u
                                                                            inst✝ : Ring R
                                                                            p : Polynomial R
                                                                            r : R
                                                                            a : Nat
                                                                            ⊢ Eq ((HMul.hMul p (HSub.hSub Polynomial.X (Polynomial.C r))).coeff (HAdd.hAdd …
                                                                          -/
    coeff (p * (X - C r)) (a + 1) = coeff p a - coeff p (a + 1) * r := by simp [mul_sub]
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


theorem coeff_natDegree_eq_zero_of_degree_lt (h : degree p < degree q) :
    coeff p (natDegree q) = 0 :=
  coeff_eq_zero_of_degree_lt (lt_of_lt_of_le h degree_le_natDegree)


theorem ne_zero_of_degree_gt {n : WithBot ℕ} (h : n < degree p) : p ≠ 0 :=
  mt degree_eq_bot.2 h.ne_bot


theorem ne_zero_of_degree_ge_degree (hpq : p.degree ≤ q.degree) (hp : p ≠ 0) : q ≠ 0 :=
  Polynomial.ne_zero_of_degree_gt
                                               /-
                                                 R : Type u
                                                 inst✝ : Semiring R
                                                 p q : Polynomial R
                                                 hpq : LE.le p.degree q.degree
                                                 hp : Ne p 0
                                                 ⊢ Ne p.degree Bot.bot
                                               -/
    (lt_of_lt_of_le (bot_lt_iff_ne_bot.mpr (by rwa [Ne, Polynomial.degree_eq_bot])) hpq :
                                               /-
                                                 🎉 no goals
                                               -/
      q.degree > ⊥)


theorem ne_zero_of_natDegree_gt {n : ℕ} (h : n < natDegree p) : p ≠ 0 := fun H => by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    n : Nat
    h : LT.lt n p.natDegree
    H : Eq p 0
    ⊢ False
  -/
  simp [H, Nat.not_lt_zero] at h
  /-
    🎉 no goals
  -/


theorem degree_lt_degree (h : natDegree p < natDegree q) : degree p < degree q := by
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    h : LT.lt p.natDegree q.natDegree
    ⊢ LT.lt p.degree q.degree
  -/
  by_cases hp : p = 0
    /-
      case pos
      R : Type u
      inst✝ : Semiring R
      p q : Polynomial R
      h : LT.lt p.natDegree q.natDegree
      hp : Eq p 0
      ⊢ LT.lt p.degree q.degree
    -/
  · simp only [hp, degree_zero]
    /-
      case pos
      R : Type u
      inst✝ : Semiring R
      p q : Polynomial R
      h : LT.lt p.natDegree q.natDegree
      hp : Eq p 0
      ⊢ LT.lt Bot.bot q.degree
    -/
    rw [bot_lt_iff_ne_bot]
    /-
      case pos
      R : Type u
      inst✝ : Semiring R
      p q : Polynomial R
      h : LT.lt p.natDegree q.natDegree
      hp : Eq p 0
      ⊢ Ne q.degree Bot.bot
    -/
    intro hq
    /-
      case pos
      R : Type u
      inst✝ : Semiring R
      p q : Polynomial R
      h : LT.lt p.natDegree q.natDegree
      hp : Eq p 0
      hq : Eq q.degree Bot.bot
      ⊢ False
    -/
    simp [hp, degree_eq_bot.mp hq, lt_irrefl] at h
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u
      inst✝ : Semiring R
      p q : Polynomial R
      h : LT.lt p.natDegree q.natDegree
      hp : Not (Eq p 0)
      ⊢ LT.lt p.degree q.degree
    -/
  · rwa [degree_eq_natDegree hp, degree_eq_natDegree <| ne_zero_of_natDegree_gt h, Nat.cast_lt]
    /-
      🎉 no goals
    -/


theorem natDegree_lt_natDegree_iff (hp : p ≠ 0) : natDegree p < natDegree q ↔ degree p < degree q :=
  ⟨degree_lt_degree, fun h ↦ by
    /-
      R : Type u
      inst✝ : Semiring R
      p q : Polynomial R
      hp : Ne p 0
      h : LT.lt p.degree q.degree
      ⊢ LT.lt p.natDegree q.natDegree
    -/
    have hq : q ≠ 0 := ne_zero_of_degree_gt h
    /-
      R : Type u
      inst✝ : Semiring R
      p q : Polynomial R
      hp : Ne p 0
      h : LT.lt p.degree q.degree
      hq : Ne q 0
      ⊢ LT.lt p.natDegree q.natDegree
    -/
    rwa [degree_eq_natDegree hp, degree_eq_natDegree hq, Nat.cast_lt] at h⟩
    /-
      🎉 no goals
    -/


theorem eq_C_of_degree_le_zero (h : degree p ≤ 0) : p = C (coeff p 0) := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    h : LE.le p.degree 0
    ⊢ Eq p (Polynomial.C (p.coeff 0))
  -/
  ext (_ | n)
    /-
      case a.zero
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      h : LE.le p.degree 0
      ⊢ Eq (p.coeff 0) ((Polynomial.C (p.coeff 0)).coeff 0)
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case a.succ
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    h : LE.le p.degree 0
    n : Nat
    ⊢ Eq (p.coeff (HAdd.hAdd n 1)) ((Polynomial.C (p.coeff 0)).coeff (HAdd.hAdd n  …
  -/
  rw [coeff_C, if_neg (Nat.succ_ne_zero _), coeff_eq_zero_of_degree_lt]
  /-
    case a.succ
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    h : LE.le p.degree 0
    n : Nat
    ⊢ LT.lt p.degree ↑(HAdd.hAdd n 1)
  -/
  exact h.trans_lt (WithBot.coe_lt_coe.2 n.succ_pos)
  /-
    🎉 no goals
  -/


theorem eq_C_of_degree_eq_zero (h : degree p = 0) : p = C (coeff p 0) :=
  eq_C_of_degree_le_zero h.le


theorem degree_le_zero_iff : degree p ≤ 0 ↔ p = C (coeff p 0) :=
  ⟨eq_C_of_degree_le_zero, fun h => h.symm ▸ degree_C_le⟩


theorem degree_add_eq_left_of_degree_lt (h : degree q < degree p) : degree (p + q) = degree p :=
  le_antisymm (max_eq_left_of_lt h ▸ degree_add_le _ _) <|
    degree_le_degree <| by
      /-
        R : Type u
        inst✝ : Semiring R
        p q : Polynomial R
        h : LT.lt q.degree p.degree
        ⊢ Ne ((HAdd.hAdd p q).coeff p.natDegree) 0
      -/
      rw [coeff_add, coeff_natDegree_eq_zero_of_degree_lt h, add_zero]
      /-
        R : Type u
        inst✝ : Semiring R
        p q : Polynomial R
        h : LT.lt q.degree p.degree
        ⊢ Ne (p.coeff p.natDegree) 0
      -/
      exact mt leadingCoeff_eq_zero.1 (ne_zero_of_degree_gt h)
      /-
        🎉 no goals
      -/


theorem degree_add_eq_right_of_degree_lt (h : degree p < degree q) : degree (p + q) = degree q := by
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    h : LT.lt p.degree q.degree
    ⊢ Eq (HAdd.hAdd p q).degree q.degree
  -/
  rw [add_comm, degree_add_eq_left_of_degree_lt h]
  /-
    🎉 no goals
  -/


theorem natDegree_add_eq_left_of_degree_lt (h : degree q < degree p) :
    natDegree (p + q) = natDegree p :=
  natDegree_eq_of_degree_eq (degree_add_eq_left_of_degree_lt h)


theorem natDegree_add_eq_left_of_natDegree_lt (h : natDegree q < natDegree p) :
    natDegree (p + q) = natDegree p :=
  natDegree_add_eq_left_of_degree_lt (degree_lt_degree h)


theorem natDegree_add_eq_right_of_degree_lt (h : degree p < degree q) :
    natDegree (p + q) = natDegree q :=
  natDegree_eq_of_degree_eq (degree_add_eq_right_of_degree_lt h)


theorem natDegree_add_eq_right_of_natDegree_lt (h : natDegree p < natDegree q) :
    natDegree (p + q) = natDegree q :=
  natDegree_add_eq_right_of_degree_lt (degree_lt_degree h)


theorem degree_add_C (hp : 0 < degree p) : degree (p + C a) = degree p :=
  add_comm (C a) p ▸ degree_add_eq_right_of_degree_lt <| lt_of_le_of_lt degree_C_le hp


@[simp] theorem natDegree_add_C {a : R} : (p + C a).natDegree = p.natDegree := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    a : R
    ⊢ Eq (HAdd.hAdd p (Polynomial.C a)).natDegree p.natDegree
  -/
  rcases eq_or_ne p 0 with rfl | hp
    /-
      case inl
      R : Type u
      inst✝ : Semiring R
      a : R
      ⊢ Eq (HAdd.hAdd 0 (Polynomial.C a)).natDegree (Polynomial.natDegree 0)
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    a : R
    hp : Ne p 0
    ⊢ Eq (HAdd.hAdd p (Polynomial.C a)).natDegree p.natDegree
  -/
  by_cases hpd : p.degree ≤ 0
    /-
      case pos
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      a : R
      hp : Ne p 0
      hpd : LE.le p.degree 0
      ⊢ Eq (HAdd.hAdd p (Polynomial.C a)).natDegree p.natDegree
    -/
  · rw [eq_C_of_degree_le_zero hpd, ← C_add, natDegree_C, natDegree_C]
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      a : R
      hp : Ne p 0
      hpd : Not (LE.le p.degree 0)
      ⊢ Eq (HAdd.hAdd p (Polynomial.C a)).natDegree p.natDegree
    -/
  · rw [not_le, degree_eq_natDegree hp, Nat.cast_pos, ← natDegree_C a] at hpd
    /-
      case neg
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      a : R
      hp : Ne p 0
      hpd : LT.lt (Polynomial.C a).natDegree p.natDegree
      ⊢ Eq (HAdd.hAdd p (Polynomial.C a)).natDegree p.natDegree
    -/
    exact natDegree_add_eq_left_of_natDegree_lt hpd
    /-
      🎉 no goals
    -/


@[simp] theorem natDegree_C_add {a : R} : (C a + p).natDegree = p.natDegree := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    a : R
    ⊢ Eq (HAdd.hAdd (Polynomial.C a) p).natDegree p.natDegree
  -/
  simp [add_comm _ p]
  /-
    🎉 no goals
  -/


theorem degree_add_eq_of_leadingCoeff_add_ne_zero (h : leadingCoeff p + leadingCoeff q ≠ 0) :
    degree (p + q) = max p.degree q.degree :=
  le_antisymm (degree_add_le _ _) <|
    match lt_trichotomy (degree p) (degree q) with
    | Or.inl hlt => by
      /-
        R : Type u
        inst✝ : Semiring R
        p q : Polynomial R
        h : Ne (HAdd.hAdd p.leadingCoeff q.leadingCoeff) 0
        hlt : LT.lt p.degree q.degree
        ⊢ LE.le (Max.max p.degree q.degree) (HAdd.hAdd p q).degree
      -/
      rw [degree_add_eq_right_of_degree_lt hlt, max_eq_right_of_lt hlt]
      /-
        🎉 no goals
      -/
    | Or.inr (Or.inl HEq) =>
      le_of_not_gt fun hlt : max (degree p) (degree q) > degree (p + q) =>
        h <|
          show leadingCoeff p + leadingCoeff q = 0 by
            /-
              R : Type u
              inst✝ : Semiring R
              p q : Polynomial R
              h : Ne (HAdd.hAdd p.leadingCoeff q.leadingCoeff) 0
              HEq : Eq p.degree q.degree
              hlt : GT.gt (Max.max p.degree q.degree) (HAdd.hAdd p q).degree
              ⊢ Eq (HAdd.hAdd p.leadingCoeff q.leadingCoeff) 0
            -/
            rw [HEq, max_self] at hlt
            /-
              R : Type u
              inst✝ : Semiring R
              p q : Polynomial R
              h : Ne (HAdd.hAdd p.leadingCoeff q.leadingCoeff) 0
              HEq : Eq p.degree q.degree
              hlt : GT.gt q.degree (HAdd.hAdd p q).degree
              ⊢ Eq (HAdd.hAdd p.leadingCoeff q.leadingCoeff) 0
            -/
            rw [leadingCoeff, leadingCoeff, natDegree_eq_of_degree_eq HEq, ← coeff_add]
            /-
              R : Type u
              inst✝ : Semiring R
              p q : Polynomial R
              h : Ne (HAdd.hAdd p.leadingCoeff q.leadingCoeff) 0
              HEq : Eq p.degree q.degree
              hlt : GT.gt q.degree (HAdd.hAdd p q).degree
              ⊢ Eq ((HAdd.hAdd p q).coeff q.natDegree) 0
            -/
            exact coeff_natDegree_eq_zero_of_degree_lt hlt
            /-
              🎉 no goals
            -/
    | Or.inr (Or.inr hlt) => by
      /-
        R : Type u
        inst✝ : Semiring R
        p q : Polynomial R
        h : Ne (HAdd.hAdd p.leadingCoeff q.leadingCoeff) 0
        hlt : LT.lt q.degree p.degree
        ⊢ LE.le (Max.max p.degree q.degree) (HAdd.hAdd p q).degree
      -/
      rw [degree_add_eq_left_of_degree_lt hlt, max_eq_left_of_lt hlt]
      /-
        🎉 no goals
      -/


lemma natDegree_eq_of_natDegree_add_lt_left (p q : R[X])
    (H : natDegree (p + q) < natDegree p) : natDegree p = natDegree q := by
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    H : LT.lt (HAdd.hAdd p q).natDegree p.natDegree
    ⊢ Eq p.natDegree q.natDegree
  -/
  by_contra h
  cases Nat.lt_or_lt_of_ne h with
  | inl h => exact lt_asymm h (by rwa [natDegree_add_eq_right_of_natDegree_lt h] at H)
  | inr h =>
    rw [natDegree_add_eq_left_of_natDegree_lt h] at H
    exact LT.lt.false H


lemma natDegree_eq_of_natDegree_add_lt_right (p q : R[X])
    (H : natDegree (p + q) < natDegree q) : natDegree p = natDegree q :=
  (natDegree_eq_of_natDegree_add_lt_left q p (add_comm p q ▸ H)).symm


lemma natDegree_eq_of_natDegree_add_eq_zero (p q : R[X])
    (H : natDegree (p + q) = 0) : natDegree p = natDegree q := by
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    H : Eq (HAdd.hAdd p q).natDegree 0
    ⊢ Eq p.natDegree q.natDegree
  -/
  by_cases h₁ : natDegree p = 0; on_goal 1 => by_cases h₂ : natDegree q = 0
    /-
      case pos
      R : Type u
      inst✝ : Semiring R
      p q : Polynomial R
      H : Eq (HAdd.hAdd p q).natDegree 0
      h₁ : Eq p.natDegree 0
      h₂ : Eq q.natDegree 0
      ⊢ Eq p.natDegree q.natDegree
    -/
  · exact h₁.trans h₂.symm
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u
      inst✝ : Semiring R
      p q : Polynomial R
      H : Eq (HAdd.hAdd p q).natDegree 0
      h₁ : Eq p.natDegree 0
      h₂ : Not (Eq q.natDegree 0)
      ⊢ Eq p.natDegree q.natDegree
    -/
  · apply natDegree_eq_of_natDegree_add_lt_right; rwa [H, Nat.pos_iff_ne_zero]
                                                  /-
                                                    🎉 no goals
                                                  -/
    /-
      case neg
      R : Type u
      inst✝ : Semiring R
      p q : Polynomial R
      H : Eq (HAdd.hAdd p q).natDegree 0
      h₁ : Not (Eq p.natDegree 0)
      ⊢ Eq p.natDegree q.natDegree
    -/
  · apply natDegree_eq_of_natDegree_add_lt_left; rwa [H, Nat.pos_iff_ne_zero]
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem monic_of_natDegree_le_of_coeff_eq_one (n : ℕ) (pn : p.natDegree ≤ n) (p1 : p.coeff n = 1) :
    Monic p := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    n : Nat
    pn : LE.le p.natDegree n
    p1 : Eq (p.coeff n) 1
    ⊢ p.Monic
  -/
  unfold Monic
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    n : Nat
    pn : LE.le p.natDegree n
    p1 : Eq (p.coeff n) 1
    ⊢ Eq p.leadingCoeff 1
  -/
  nontriviality
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    n : Nat
    pn : LE.le p.natDegree n
    p1 : Eq (p.coeff n) 1
    a✝ : Nontrivial R
    ⊢ Eq p.leadingCoeff 1
  -/
  refine (congr_arg _ <| natDegree_eq_of_le_of_coeff_ne_zero pn ?_).trans p1
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    n : Nat
    pn : LE.le p.natDegree n
    p1 : Eq (p.coeff n) 1
    a✝ : Nontrivial R
    ⊢ Ne (p.coeff n) 0
  -/
  exact ne_of_eq_of_ne p1 one_ne_zero
  /-
    🎉 no goals
  -/


theorem monic_of_degree_le_of_coeff_eq_one (n : ℕ) (pn : p.degree ≤ n) (p1 : p.coeff n = 1) :
    Monic p :=
  monic_of_natDegree_le_of_coeff_eq_one n (natDegree_le_of_degree_le pn) p1


theorem leadingCoeff_add_of_degree_lt (h : degree p < degree q) :
    leadingCoeff (p + q) = leadingCoeff q := by
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    h : LT.lt p.degree q.degree
    ⊢ Eq (HAdd.hAdd p q).leadingCoeff q.leadingCoeff
  -/
  have : coeff p (natDegree q) = 0 := coeff_natDegree_eq_zero_of_degree_lt h
  simp only [leadingCoeff, natDegree_eq_of_degree_eq (degree_add_eq_right_of_degree_lt h), this,
    coeff_add, zero_add]


theorem leadingCoeff_add_of_degree_lt' (h : degree q < degree p) :
    leadingCoeff (p + q) = leadingCoeff p := by
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    h : LT.lt q.degree p.degree
    ⊢ Eq (HAdd.hAdd p q).leadingCoeff p.leadingCoeff
  -/
  rw [add_comm]
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    h : LT.lt q.degree p.degree
    ⊢ Eq (HAdd.hAdd q p).leadingCoeff p.leadingCoeff
  -/
  exact leadingCoeff_add_of_degree_lt h
  /-
    🎉 no goals
  -/


theorem leadingCoeff_add_of_degree_eq (h : degree p = degree q)
    (hlc : leadingCoeff p + leadingCoeff q ≠ 0) :
    leadingCoeff (p + q) = leadingCoeff p + leadingCoeff q := by
  have : natDegree (p + q) = natDegree p := by
    apply natDegree_eq_of_degree_eq
    rw [degree_add_eq_of_leadingCoeff_add_ne_zero hlc, h, max_self]
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    h : Eq p.degree q.degree
    hlc : Ne (HAdd.hAdd p.leadingCoeff q.leadingCoeff) 0
    this : Eq (HAdd.hAdd p q).natDegree p.natDegree
    ⊢ Eq (HAdd.hAdd p q).leadingCoeff (HAdd.hAdd p.leadingCoeff q.leadingCoeff)
  -/
  simp only [leadingCoeff, this, natDegree_eq_of_degree_eq h, coeff_add]
  /-
    🎉 no goals
  -/


@[simp]
theorem coeff_mul_degree_add_degree (p q : R[X]) :
    coeff (p * q) (natDegree p + natDegree q) = leadingCoeff p * leadingCoeff q :=
  calc
    coeff (p * q) (natDegree p + natDegree q) =
        ∑ x ∈ antidiagonal (natDegree p + natDegree q), coeff p x.1 * coeff q x.2 :=
      coeff_mul _ _ _
    _ = coeff p (natDegree p) * coeff q (natDegree q) := by
      /-
        R : Type u
        inst✝ : Semiring R
        p q : Polynomial R
        ⊢ Eq ((Finset.HasAntidiagonal.antidiagonal (HAdd.hAdd p.natDegree q.natDegree) …
      -/
      refine Finset.sum_eq_single (natDegree p, natDegree q) ?_ ?_
        /-
          case refine_1
          R : Type u
          inst✝ : Semiring R
          p q : Polynomial R
          ⊢ ∀ (b : Prod Nat Nat), Membership.mem (Finset.HasAntidiagonal.antidiagonal (H …
        -/
      · rintro ⟨i, j⟩ h₁ h₂
        /-
          case refine_1.mk
          R : Type u
          inst✝ : Semiring R
          p q : Polynomial R
          i j : Nat
          h₁ : Membership.mem (Finset.HasAntidiagonal.antidiagonal (HAdd.hAdd p.natDegre …
          h₂ : Ne { fst := i, snd := j } { fst := p.natDegree, snd := q.natDegree }
          ⊢ Eq (HMul.hMul (p.coeff { fst := i, snd := j }.1) (q.coeff { fst := i, snd := …
        -/
        rw [mem_antidiagonal] at h₁
        /-
          case refine_1.mk
          R : Type u
          inst✝ : Semiring R
          p q : Polynomial R
          i j : Nat
          h₁ : Eq (HAdd.hAdd { fst := i, snd := j }.1 { fst := i, snd := j }.2) (HAdd.hA …
          h₂ : Ne { fst := i, snd := j } { fst := p.natDegree, snd := q.natDegree }
          ⊢ Eq (HMul.hMul (p.coeff { fst := i, snd := j }.1) (q.coeff { fst := i, snd := …
        -/
        by_cases H : natDegree p < i
        · rw [coeff_eq_zero_of_degree_lt
              (lt_of_le_of_lt degree_le_natDegree (WithBot.coe_lt_coe.2 H)),
            zero_mul]
          /-
            case neg
            R : Type u
            inst✝ : Semiring R
            p q : Polynomial R
            i j : Nat
            h₁ : Eq (HAdd.hAdd { fst := i, snd := j }.1 { fst := i, snd := j }.2) (HAdd.hA …
            h₂ : Ne { fst := i, snd := j } { fst := p.natDegree, snd := q.natDegree }
            H : Not (LT.lt p.natDegree i)
            ⊢ Eq (HMul.hMul (p.coeff { fst := i, snd := j }.1) (q.coeff { fst := i, snd := …
          -/
        · rw [not_lt_iff_eq_or_lt] at H
          /-
            case neg
            R : Type u
            inst✝ : Semiring R
            p q : Polynomial R
            i j : Nat
            h₁ : Eq (HAdd.hAdd { fst := i, snd := j }.1 { fst := i, snd := j }.2) (HAdd.hA …
            h₂ : Ne { fst := i, snd := j } { fst := p.natDegree, snd := q.natDegree }
            H : Or (Eq p.natDegree i) (LT.lt i p.natDegree)
            ⊢ Eq (HMul.hMul (p.coeff { fst := i, snd := j }.1) (q.coeff { fst := i, snd := …
          -/
          cases' H with H H
            /-
              case neg.inl
              R : Type u
              inst✝ : Semiring R
              p q : Polynomial R
              i j : Nat
              h₁ : Eq (HAdd.hAdd { fst := i, snd := j }.1 { fst := i, snd := j }.2) (HAdd.hA …
              h₂ : Ne { fst := i, snd := j } { fst := p.natDegree, snd := q.natDegree }
              H : Eq p.natDegree i
              ⊢ Eq (HMul.hMul (p.coeff { fst := i, snd := j }.1) (q.coeff { fst := i, snd := …
            -/
          · subst H
            /-
              case neg.inl
              R : Type u
              inst✝ : Semiring R
              p q : Polynomial R
              j : Nat
              h₁ : Eq (HAdd.hAdd { fst := p.natDegree, snd := j }.1 { fst := p.natDegree, sn …
              h₂ : Ne { fst := p.natDegree, snd := j } { fst := p.natDegree, snd := q.natDeg …
              ⊢ Eq (HMul.hMul (p.coeff { fst := p.natDegree, snd := j }.1) (q.coeff { fst := …
            -/
            rw [add_left_cancel_iff] at h₁
            /-
              case neg.inl
              R : Type u
              inst✝ : Semiring R
              p q : Polynomial R
              j : Nat
              h₁ : Eq { fst := p.natDegree, snd := j }.2 q.natDegree
              h₂ : Ne { fst := p.natDegree, snd := j } { fst := p.natDegree, snd := q.natDeg …
              ⊢ Eq (HMul.hMul (p.coeff { fst := p.natDegree, snd := j }.1) (q.coeff { fst := …
            -/
            dsimp at h₁
            /-
              case neg.inl
              R : Type u
              inst✝ : Semiring R
              p q : Polynomial R
              j : Nat
              h₁ : Eq j q.natDegree
              h₂ : Ne { fst := p.natDegree, snd := j } { fst := p.natDegree, snd := q.natDeg …
              ⊢ Eq (HMul.hMul (p.coeff { fst := p.natDegree, snd := j }.1) (q.coeff { fst := …
            -/
            subst h₁
            /-
              case neg.inl
              R : Type u
              inst✝ : Semiring R
              p q : Polynomial R
              h₂ : Ne { fst := p.natDegree, snd := q.natDegree } { fst := p.natDegree, snd : …
              ⊢ Eq (HMul.hMul (p.coeff { fst := p.natDegree, snd := q.natDegree }.1) (q.coef …
            -/
            exact (h₂ rfl).elim
            /-
              🎉 no goals
            -/
          · suffices natDegree q < j by
              rw [coeff_eq_zero_of_degree_lt
                  (lt_of_le_of_lt degree_le_natDegree (WithBot.coe_lt_coe.2 this)),
                mul_zero]
            /-
              case neg.inr
              R : Type u
              inst✝ : Semiring R
              p q : Polynomial R
              i j : Nat
              h₁ : Eq (HAdd.hAdd { fst := i, snd := j }.1 { fst := i, snd := j }.2) (HAdd.hA …
              h₂ : Ne { fst := i, snd := j } { fst := p.natDegree, snd := q.natDegree }
              H : LT.lt i p.natDegree
              ⊢ LT.lt q.natDegree j
            -/
            by_contra! H'
            exact
              ne_of_lt (Nat.lt_of_lt_of_le (Nat.add_lt_add_right H j) (Nat.add_le_add_left H' _))
                h₁
        /-
          case refine_2
          R : Type u
          inst✝ : Semiring R
          p q : Polynomial R
          ⊢ Not (Membership.mem (Finset.HasAntidiagonal.antidiagonal (HAdd.hAdd p.natDeg …
        -/
      · intro H
        /-
          case refine_2
          R : Type u
          inst✝ : Semiring R
          p q : Polynomial R
          H : Not (Membership.mem (Finset.HasAntidiagonal.antidiagonal (HAdd.hAdd p.natD …
          ⊢ Eq (HMul.hMul (p.coeff { fst := p.natDegree, snd := q.natDegree }.1) (q.coef …
        -/
        exfalso
        /-
          case refine_2
          R : Type u
          inst✝ : Semiring R
          p q : Polynomial R
          H : Not (Membership.mem (Finset.HasAntidiagonal.antidiagonal (HAdd.hAdd p.natD …
          ⊢ False
        -/
        apply H
        /-
          case refine_2
          R : Type u
          inst✝ : Semiring R
          p q : Polynomial R
          H : Not (Membership.mem (Finset.HasAntidiagonal.antidiagonal (HAdd.hAdd p.natD …
          ⊢ Membership.mem (Finset.HasAntidiagonal.antidiagonal (HAdd.hAdd p.natDegree q …
        -/
        rw [mem_antidiagonal]
        /-
          🎉 no goals
        -/


theorem degree_mul' (h : leadingCoeff p * leadingCoeff q ≠ 0) :
    degree (p * q) = degree p + degree q :=
                        /-
                          R : Type u
                          inst✝ : Semiring R
                          p q : Polynomial R
                          h : Ne (HMul.hMul p.leadingCoeff q.leadingCoeff) 0
                          ⊢ Ne p 0
                        -/
  have hp : p ≠ 0 := by refine mt ?_ h; exact fun hp => by rw [hp, leadingCoeff_zero, zero_mul]
                                        /-
                                          🎉 no goals
                                        -/
                        /-
                          R : Type u
                          inst✝ : Semiring R
                          p q : Polynomial R
                          h : Ne (HMul.hMul p.leadingCoeff q.leadingCoeff) 0
                          hp : Ne p 0
                          ⊢ Ne q 0
                        -/
  have hq : q ≠ 0 := by refine mt ?_ h; exact fun hq => by rw [hq, leadingCoeff_zero, mul_zero]
                                        /-
                                          🎉 no goals
                                        -/
  le_antisymm (degree_mul_le _ _)
    (by
      /-
        R : Type u
        inst✝ : Semiring R
        p q : Polynomial R
        h : Ne (HMul.hMul p.leadingCoeff q.leadingCoeff) 0
        hp : Ne p 0
        hq : Ne q 0
        ⊢ LE.le (HAdd.hAdd p.degree q.degree) (HMul.hMul p q).degree
      -/
      rw [degree_eq_natDegree hp, degree_eq_natDegree hq]
      /-
        R : Type u
        inst✝ : Semiring R
        p q : Polynomial R
        h : Ne (HMul.hMul p.leadingCoeff q.leadingCoeff) 0
        hp : Ne p 0
        hq : Ne q 0
        ⊢ LE.le (HAdd.hAdd ↑p.natDegree ↑q.natDegree) (HMul.hMul p q).degree
      -/
      refine le_degree_of_ne_zero (n := natDegree p + natDegree q) ?_
      /-
        R : Type u
        inst✝ : Semiring R
        p q : Polynomial R
        h : Ne (HMul.hMul p.leadingCoeff q.leadingCoeff) 0
        hp : Ne p 0
        hq : Ne q 0
        ⊢ Ne ((HMul.hMul p q).coeff (HAdd.hAdd p.natDegree q.natDegree)) 0
      -/
      rwa [coeff_mul_degree_add_degree])
      /-
        🎉 no goals
      -/


theorem Monic.degree_mul (hq : Monic q) : degree (p * q) = degree p + degree q :=
  letI := Classical.decEq R
                        /-
                          R : Type u
                          inst✝ : Semiring R
                          p q : Polynomial R
                          hq : q.Monic
                          this : DecidableEq R := Classical.decEq R
                          hp : Eq p 0
                          ⊢ Eq (HMul.hMul p q).degree (HAdd.hAdd p.degree q.degree)
                        -/
  if hp : p = 0 then by simp [hp]
                        /-
                          🎉 no goals
                        -/
                         /-
                           R : Type u
                           inst✝ : Semiring R
                           p q : Polynomial R
                           hq : q.Monic
                           this : DecidableEq R := Classical.decEq R
                           hp : Not (Eq p 0)
                           ⊢ Ne (HMul.hMul p.leadingCoeff q.leadingCoeff) 0
                         -/
  else degree_mul' <| by rwa [hq.leadingCoeff, mul_one, Ne, leadingCoeff_eq_zero]
                         /-
                           🎉 no goals
                         -/


theorem natDegree_mul' (h : leadingCoeff p * leadingCoeff q ≠ 0) :
    natDegree (p * q) = natDegree p + natDegree q :=
                                                                 /-
                                                                   R : Type u
                                                                   inst✝ : Semiring R
                                                                   p q : Polynomial R
                                                                   h : Ne (HMul.hMul p.leadingCoeff q.leadingCoeff) 0
                                                                   h₁ : Eq p.leadingCoeff 0
                                                                   ⊢ Eq (HMul.hMul p.leadingCoeff q.leadingCoeff) 0
                                                                 -/
  have hp : p ≠ 0 := mt leadingCoeff_eq_zero.2 fun h₁ => h <| by rw [h₁, zero_mul]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
                                                                 /-
                                                                   R : Type u
                                                                   inst✝ : Semiring R
                                                                   p q : Polynomial R
                                                                   h : Ne (HMul.hMul p.leadingCoeff q.leadingCoeff) 0
                                                                   hp : Ne p 0
                                                                   h₁ : Eq q.leadingCoeff 0
                                                                   ⊢ Eq (HMul.hMul p.leadingCoeff q.leadingCoeff) 0
                                                                 -/
  have hq : q ≠ 0 := mt leadingCoeff_eq_zero.2 fun h₁ => h <| by rw [h₁, mul_zero]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
  natDegree_eq_of_degree_eq_some <| by
    /-
      R : Type u
      inst✝ : Semiring R
      p q : Polynomial R
      h : Ne (HMul.hMul p.leadingCoeff q.leadingCoeff) 0
      hp : Ne p 0
      hq : Ne q 0
      ⊢ Eq (HMul.hMul p q).degree ↑(HAdd.hAdd p.natDegree q.natDegree)
    -/
    rw [degree_mul' h, Nat.cast_add, degree_eq_natDegree hp, degree_eq_natDegree hq]
    /-
      🎉 no goals
    -/


theorem leadingCoeff_mul' (h : leadingCoeff p * leadingCoeff q ≠ 0) :
    leadingCoeff (p * q) = leadingCoeff p * leadingCoeff q := by
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    h : Ne (HMul.hMul p.leadingCoeff q.leadingCoeff) 0
    ⊢ Eq (HMul.hMul p q).leadingCoeff (HMul.hMul p.leadingCoeff q.leadingCoeff)
  -/
  unfold leadingCoeff
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    h : Ne (HMul.hMul p.leadingCoeff q.leadingCoeff) 0
    ⊢ Eq ((HMul.hMul p q).coeff (HMul.hMul p q).natDegree) (HMul.hMul (p.coeff p.n …
  -/
  rw [natDegree_mul' h, coeff_mul_degree_add_degree]
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    h : Ne (HMul.hMul p.leadingCoeff q.leadingCoeff) 0
    ⊢ Eq (HMul.hMul p.leadingCoeff q.leadingCoeff) (HMul.hMul (p.coeff p.natDegree …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem leadingCoeff_pow' : leadingCoeff p ^ n ≠ 0 → leadingCoeff (p ^ n) = leadingCoeff p ^ n :=
                  /-
                    R : Type u
                    n : Nat
                    inst✝ : Semiring R
                    p : Polynomial R
                    ⊢ Ne (HPow.hPow p.leadingCoeff Nat.zero) 0 → Eq (HPow.hPow p Nat.zero).leading …
                  -/
  Nat.recOn n (by simp) fun n ih h => by
                  /-
                    🎉 no goals
                  -/
    /-
      R : Type u
      n✝ : Nat
      inst✝ : Semiring R
      p : Polynomial R
      n : Nat
      ih : Ne (HPow.hPow p.leadingCoeff n) 0 → Eq (HPow.hPow p n).leadingCoeff (HPow …
      h : Ne (HPow.hPow p.leadingCoeff n.succ) 0
      ⊢ Eq (HPow.hPow p n.succ).leadingCoeff (HPow.hPow p.leadingCoeff n.succ)
    -/
    have h₁ : leadingCoeff p ^ n ≠ 0 := fun h₁ => h <| by rw [pow_succ, h₁, zero_mul]
    /-
      R : Type u
      n✝ : Nat
      inst✝ : Semiring R
      p : Polynomial R
      n : Nat
      ih : Ne (HPow.hPow p.leadingCoeff n) 0 → Eq (HPow.hPow p n).leadingCoeff (HPow …
      h : Ne (HPow.hPow p.leadingCoeff n.succ) 0
      h₁ : Ne (HPow.hPow p.leadingCoeff n) 0
      ⊢ Eq (HPow.hPow p n.succ).leadingCoeff (HPow.hPow p.leadingCoeff n.succ)
    -/
    have h₂ : leadingCoeff p * leadingCoeff (p ^ n) ≠ 0 := by rwa [pow_succ', ← ih h₁] at h
    /-
      R : Type u
      n✝ : Nat
      inst✝ : Semiring R
      p : Polynomial R
      n : Nat
      ih : Ne (HPow.hPow p.leadingCoeff n) 0 → Eq (HPow.hPow p n).leadingCoeff (HPow …
      h : Ne (HPow.hPow p.leadingCoeff n.succ) 0
      h₁ : Ne (HPow.hPow p.leadingCoeff n) 0
      h₂ : Ne (HMul.hMul p.leadingCoeff (HPow.hPow p n).leadingCoeff) 0
      ⊢ Eq (HPow.hPow p n.succ).leadingCoeff (HPow.hPow p.leadingCoeff n.succ)
    -/
    rw [pow_succ', pow_succ', leadingCoeff_mul' h₂, ih h₁]
    /-
      🎉 no goals
    -/


theorem degree_pow' : ∀ {n : ℕ}, leadingCoeff p ^ n ≠ 0 → degree (p ^ n) = n • degree p
                     /-
                       R : Type u
                       inst✝ : Semiring R
                       p : Polynomial R
                       h : Ne (HPow.hPow p.leadingCoeff 0) 0
                       ⊢ Eq (HPow.hPow p 0).degree (HSMul.hSMul 0 p.degree)
                     -/
  | 0 => fun h => by rw [pow_zero, ← C_1] at *; rw [degree_C h, zero_nsmul]
                                                /-
                                                  🎉 no goals
                                                -/
  | n + 1 => fun h => by
    /-
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      n : Nat
      h : Ne (HPow.hPow p.leadingCoeff (HAdd.hAdd n 1)) 0
      ⊢ Eq (HPow.hPow p (HAdd.hAdd n 1)).degree (HSMul.hSMul (HAdd.hAdd n 1) p.degree)
    -/
    have h₁ : leadingCoeff p ^ n ≠ 0 := fun h₁ => h <| by rw [pow_succ, h₁, zero_mul]
    have h₂ : leadingCoeff (p ^ n) * leadingCoeff p ≠ 0 := by
      rwa [pow_succ, ← leadingCoeff_pow' h₁] at h
    /-
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      n : Nat
      h : Ne (HPow.hPow p.leadingCoeff (HAdd.hAdd n 1)) 0
      h₁ : Ne (HPow.hPow p.leadingCoeff n) 0
      h₂ : Ne (HMul.hMul (HPow.hPow p n).leadingCoeff p.leadingCoeff) 0
      ⊢ Eq (HPow.hPow p (HAdd.hAdd n 1)).degree (HSMul.hSMul (HAdd.hAdd n 1) p.degree)
    -/
    rw [pow_succ, degree_mul' h₂, succ_nsmul, degree_pow' h₁]
    /-
      🎉 no goals
    -/


theorem natDegree_pow' {n : ℕ} (h : leadingCoeff p ^ n ≠ 0) : natDegree (p ^ n) = n * natDegree p :=
  letI := Classical.decEq R
  if hp0 : p = 0 then
                           /-
                             R : Type u
                             inst✝ : Semiring R
                             p : Polynomial R
                             n : Nat
                             h : Ne (HPow.hPow p.leadingCoeff n) 0
                             this : DecidableEq R := Classical.decEq R
                             hp0 : Eq p 0
                             hn0 : Eq n 0
                             ⊢ Eq (HPow.hPow p n).natDegree (HMul.hMul n p.natDegree)
                           -/
                           /-
                             🎉 no goals
                           -/
    if hn0 : n = 0 then by simp [*] else by rw [hp0, zero_pow hn0]; simp
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
  else
    have hpn : p ^ n ≠ 0 := fun hpn0 => by
      /-
        R : Type u
        inst✝ : Semiring R
        p : Polynomial R
        n : Nat
        h : Ne (HPow.hPow p.leadingCoeff n) 0
        this : DecidableEq R := Classical.decEq R
        hp0 : Not (Eq p 0)
        hpn0 : Eq (HPow.hPow p n) 0
        ⊢ False
      -/
      have h1 := h
      /-
        R : Type u
        inst✝ : Semiring R
        p : Polynomial R
        n : Nat
        h : Ne (HPow.hPow p.leadingCoeff n) 0
        this : DecidableEq R := Classical.decEq R
        hp0 : Not (Eq p 0)
        hpn0 : Eq (HPow.hPow p n) 0
        h1 : Ne (HPow.hPow p.leadingCoeff n) 0
        ⊢ False
      -/
      rw [← leadingCoeff_pow' h1, hpn0, leadingCoeff_zero] at h; exact h rfl
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
    Option.some_inj.1 <|
      show (natDegree (p ^ n) : WithBot ℕ) = (n * natDegree p : ℕ) by
        /-
          R : Type u
          inst✝ : Semiring R
          p : Polynomial R
          n : Nat
          h : Ne (HPow.hPow p.leadingCoeff n) 0
          this : DecidableEq R := Classical.decEq R
          hp0 : Not (Eq p 0)
          hpn : Ne (HPow.hPow p n) 0
          ⊢ Eq ↑(HPow.hPow p n).natDegree ↑(HMul.hMul n p.natDegree)
        -/
        rw [← degree_eq_natDegree hpn, degree_pow' h, degree_eq_natDegree hp0]; simp
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


theorem leadingCoeff_monic_mul {p q : R[X]} (hp : Monic p) :
    leadingCoeff (p * q) = leadingCoeff q := by
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    hp : p.Monic
    ⊢ Eq (HMul.hMul p q).leadingCoeff q.leadingCoeff
  -/
  rcases eq_or_ne q 0 with (rfl | H)
    /-
      case inl
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      hp : p.Monic
      ⊢ Eq (HMul.hMul p 0).leadingCoeff (Polynomial.leadingCoeff 0)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      R : Type u
      inst✝ : Semiring R
      p q : Polynomial R
      hp : p.Monic
      H : Ne q 0
      ⊢ Eq (HMul.hMul p q).leadingCoeff q.leadingCoeff
    -/
  · rw [leadingCoeff_mul', hp.leadingCoeff, one_mul]
    /-
      case inr
      R : Type u
      inst✝ : Semiring R
      p q : Polynomial R
      hp : p.Monic
      H : Ne q 0
      ⊢ Ne (HMul.hMul p.leadingCoeff q.leadingCoeff) 0
    -/
    rwa [hp.leadingCoeff, one_mul, Ne, leadingCoeff_eq_zero]
    /-
      🎉 no goals
    -/


theorem leadingCoeff_mul_monic {p q : R[X]} (hq : Monic q) :
    leadingCoeff (p * q) = leadingCoeff p :=
  letI := Classical.decEq R
  Decidable.byCases
    (fun H : leadingCoeff p = 0 => by
      /-
        R : Type u
        inst✝ : Semiring R
        p q : Polynomial R
        hq : q.Monic
        this : DecidableEq R := Classical.decEq R
        H : Eq p.leadingCoeff 0
        ⊢ Eq (HMul.hMul p q).leadingCoeff p.leadingCoeff
      -/
      rw [H, leadingCoeff_eq_zero.1 H, zero_mul, leadingCoeff_zero])
      /-
        🎉 no goals
      -/
    fun H : leadingCoeff p ≠ 0 => by
      /-
        R : Type u
        inst✝ : Semiring R
        p q : Polynomial R
        hq : q.Monic
        this : DecidableEq R := Classical.decEq R
        H : Ne p.leadingCoeff 0
        ⊢ Eq (HMul.hMul p q).leadingCoeff p.leadingCoeff
      -/
      rw [leadingCoeff_mul', hq.leadingCoeff, mul_one]
      /-
        R : Type u
        inst✝ : Semiring R
        p q : Polynomial R
        hq : q.Monic
        this : DecidableEq R := Classical.decEq R
        H : Ne p.leadingCoeff 0
        ⊢ Ne (HMul.hMul p.leadingCoeff q.leadingCoeff) 0
      -/
      rwa [hq.leadingCoeff, mul_one]
      /-
        🎉 no goals
      -/


@[simp]
theorem leadingCoeff_mul_X_pow {p : R[X]} {n : ℕ} : leadingCoeff (p * X ^ n) = leadingCoeff p :=
  leadingCoeff_mul_monic (monic_X_pow n)


@[simp]
theorem leadingCoeff_mul_X {p : R[X]} : leadingCoeff (p * X) = leadingCoeff p :=
  leadingCoeff_mul_monic monic_X


@[simp]
theorem coeff_pow_mul_natDegree (p : R[X]) (n : ℕ) :
    (p ^ n).coeff (n * p.natDegree) = p.leadingCoeff ^ n := by
  induction n with
  | zero => simp
  | succ i hi =>
    rw [pow_succ, pow_succ, Nat.succ_mul]
    by_cases hp1 : p.leadingCoeff ^ i = 0
    · rw [hp1, zero_mul]
      by_cases hp2 : p ^ i = 0
      · rw [hp2, zero_mul, coeff_zero]
      · apply coeff_eq_zero_of_natDegree_lt
        have h1 : (p ^ i).natDegree < i * p.natDegree := by
          refine lt_of_le_of_ne natDegree_pow_le fun h => hp2 ?_
          rw [← h, hp1] at hi
          exact leadingCoeff_eq_zero.mp hi
        calc
          (p ^ i * p).natDegree ≤ (p ^ i).natDegree + p.natDegree := natDegree_mul_le
          _ < i * p.natDegree + p.natDegree := add_lt_add_right h1 _

    · rw [← natDegree_pow' hp1, ← leadingCoeff_pow' hp1]
      exact coeff_mul_degree_add_degree _ _


theorem coeff_mul_add_eq_of_natDegree_le {df dg : ℕ} {f g : R[X]}
    (hdf : natDegree f ≤ df) (hdg : natDegree g ≤ dg) :
    (f * g).coeff (df + dg) = f.coeff df * g.coeff dg := by
  /-
    R : Type u
    inst✝ : Semiring R
    df dg : Nat
    f g : Polynomial R
    hdf : LE.le f.natDegree df
    hdg : LE.le g.natDegree dg
    ⊢ Eq ((HMul.hMul f g).coeff (HAdd.hAdd df dg)) (HMul.hMul (f.coeff df) (g.coef …
  -/
  rw [coeff_mul, Finset.sum_eq_single_of_mem (df, dg)]
    /-
      case h
      R : Type u
      inst✝ : Semiring R
      df dg : Nat
      f g : Polynomial R
      hdf : LE.le f.natDegree df
      hdg : LE.le g.natDegree dg
      ⊢ Membership.mem (Finset.HasAntidiagonal.antidiagonal (HAdd.hAdd df dg)) { fst …
    -/
  · rw [mem_antidiagonal]
    /-
      🎉 no goals
    -/
  /-
    case h₀
    R : Type u
    inst✝ : Semiring R
    df dg : Nat
    f g : Polynomial R
    hdf : LE.le f.natDegree df
    hdg : LE.le g.natDegree dg
    ⊢ ∀ (b : Prod Nat Nat), Membership.mem (Finset.HasAntidiagonal.antidiagonal (H …
  -/
  rintro ⟨df', dg'⟩ hmem hne
  /-
    case h₀.mk
    R : Type u
    inst✝ : Semiring R
    df dg : Nat
    f g : Polynomial R
    hdf : LE.le f.natDegree df
    hdg : LE.le g.natDegree dg
    df' dg' : Nat
    hmem : Membership.mem (Finset.HasAntidiagonal.antidiagonal (HAdd.hAdd df dg))  …
    hne : Ne { fst := df', snd := dg' } { fst := df, snd := dg }
    ⊢ Eq (HMul.hMul (f.coeff { fst := df', snd := dg' }.1) (g.coeff { fst := df',  …
  -/
  obtain h | hdf' := lt_or_le df df'
    /-
      case h₀.mk.inl
      R : Type u
      inst✝ : Semiring R
      df dg : Nat
      f g : Polynomial R
      hdf : LE.le f.natDegree df
      hdg : LE.le g.natDegree dg
      df' dg' : Nat
      hmem : Membership.mem (Finset.HasAntidiagonal.antidiagonal (HAdd.hAdd df dg))  …
      hne : Ne { fst := df', snd := dg' } { fst := df, snd := dg }
      h : LT.lt df df'
      ⊢ Eq (HMul.hMul (f.coeff { fst := df', snd := dg' }.1) (g.coeff { fst := df',  …
    -/
  · rw [coeff_eq_zero_of_natDegree_lt (hdf.trans_lt h), zero_mul]
    /-
      🎉 no goals
    -/
  /-
    case h₀.mk.inr
    R : Type u
    inst✝ : Semiring R
    df dg : Nat
    f g : Polynomial R
    hdf : LE.le f.natDegree df
    hdg : LE.le g.natDegree dg
    df' dg' : Nat
    hmem : Membership.mem (Finset.HasAntidiagonal.antidiagonal (HAdd.hAdd df dg))  …
    hne : Ne { fst := df', snd := dg' } { fst := df, snd := dg }
    hdf' : LE.le df' df
    ⊢ Eq (HMul.hMul (f.coeff { fst := df', snd := dg' }.1) (g.coeff { fst := df',  …
  -/
  obtain h | hdg' := lt_or_le dg dg'
    /-
      case h₀.mk.inr.inl
      R : Type u
      inst✝ : Semiring R
      df dg : Nat
      f g : Polynomial R
      hdf : LE.le f.natDegree df
      hdg : LE.le g.natDegree dg
      df' dg' : Nat
      hmem : Membership.mem (Finset.HasAntidiagonal.antidiagonal (HAdd.hAdd df dg))  …
      hne : Ne { fst := df', snd := dg' } { fst := df, snd := dg }
      hdf' : LE.le df' df
      h : LT.lt dg dg'
      ⊢ Eq (HMul.hMul (f.coeff { fst := df', snd := dg' }.1) (g.coeff { fst := df',  …
    -/
  · rw [coeff_eq_zero_of_natDegree_lt (hdg.trans_lt h), mul_zero]
    /-
      🎉 no goals
    -/
  obtain ⟨rfl, rfl⟩ :=
    (add_eq_add_iff_eq_and_eq hdf' hdg').mp (mem_antidiagonal.1 hmem)
  /-
    case h₀.mk.inr.inr.intro
    R : Type u
    inst✝ : Semiring R
    f g : Polynomial R
    df' dg' : Nat
    hdf : LE.le f.natDegree df'
    hdf' : LE.le df' df'
    hdg : LE.le g.natDegree dg'
    hdg' : LE.le dg' dg'
    hmem : Membership.mem (Finset.HasAntidiagonal.antidiagonal (HAdd.hAdd df' dg') …
    hne : Ne { fst := df', snd := dg' } { fst := df', snd := dg' }
    ⊢ Eq (HMul.hMul (f.coeff { fst := df', snd := dg' }.1) (g.coeff { fst := df',  …
  -/
  exact (hne rfl).elim
  /-
    🎉 no goals
  -/


theorem degree_smul_le (a : R) (p : R[X]) : degree (a • p) ≤ degree p := by
  /-
    R : Type u
    inst✝ : Semiring R
    a : R
    p : Polynomial R
    ⊢ LE.le (HSMul.hSMul a p).degree p.degree
  -/
  refine (degree_le_iff_coeff_zero _ _).2 fun m hm => ?_
  /-
    R : Type u
    inst✝ : Semiring R
    a : R
    p : Polynomial R
    m : Nat
    hm : LT.lt p.degree ↑m
    ⊢ Eq ((HSMul.hSMul a p).coeff m) 0
  -/
  rw [degree_lt_iff_coeff_zero] at hm
  /-
    R : Type u
    inst✝ : Semiring R
    a : R
    p : Polynomial R
    m : Nat
    hm : ∀ (m_1 : Nat), LE.le m m_1 → Eq (p.coeff m_1) 0
    ⊢ Eq ((HSMul.hSMul a p).coeff m) 0
  -/
  simp [hm m le_rfl]
  /-
    🎉 no goals
  -/


theorem natDegree_smul_le (a : R) (p : R[X]) : natDegree (a • p) ≤ natDegree p :=
  natDegree_le_natDegree (degree_smul_le a p)


theorem degree_lt_degree_mul_X (hp : p ≠ 0) : p.degree < (p * X).degree := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    hp : Ne p 0
    ⊢ LT.lt p.degree (HMul.hMul p Polynomial.X).degree
  -/
  haveI := Nontrivial.of_polynomial_ne hp
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    hp : Ne p 0
    this : Nontrivial R
    ⊢ LT.lt p.degree (HMul.hMul p Polynomial.X).degree
  -/
  have : leadingCoeff p * leadingCoeff X ≠ 0 := by simpa
  erw [degree_mul' this, degree_eq_natDegree hp, degree_X, ← WithBot.coe_one,
                                            /-
                                              R : Type u
                                              inst✝ : Semiring R
                                              p : Polynomial R
                                              hp : Ne p 0
                                              this✝ : Nontrivial R
                                              this : Ne (HMul.hMul p.leadingCoeff Polynomial.X.leadingCoeff) 0
                                              ⊢ LT.lt (↑p.natDegree) (HAdd.hAdd (↑p.natDegree) 1)
                                            -/
    ← WithBot.coe_add, WithBot.coe_lt_coe]; exact Nat.lt_succ_self _
                                            /-
                                              🎉 no goals
                                            -/


theorem eq_C_of_natDegree_le_zero (h : natDegree p ≤ 0) : p = C (coeff p 0) :=
  eq_C_of_degree_le_zero <| degree_le_of_natDegree_le h


theorem eq_C_of_natDegree_eq_zero (h : natDegree p = 0) : p = C (coeff p 0) :=
  eq_C_of_natDegree_le_zero h.le


lemma natDegree_eq_zero {p : R[X]} : p.natDegree = 0 ↔ ∃ x, C x = p :=
                                                       /-
                                                         R : Type u
                                                         inst✝ : Semiring R
                                                         p : Polynomial R
                                                         ⊢ (Exists fun x => Eq (Polynomial.C x) p) → Eq p.natDegree 0
                                                       -/
  ⟨fun h ↦ ⟨_, (eq_C_of_natDegree_eq_zero h).symm⟩, by aesop⟩
                                                       /-
                                                         🎉 no goals
                                                       -/


theorem eq_C_coeff_zero_iff_natDegree_eq_zero : p = C (p.coeff 0) ↔ p.natDegree = 0 :=
              /-
                R : Type u
                inst✝ : Semiring R
                p : Polynomial R
                h : Eq p (Polynomial.C (p.coeff 0))
                ⊢ Eq p.natDegree 0
              -/
  ⟨fun h ↦ by rw [h, natDegree_C], eq_C_of_natDegree_eq_zero⟩
              /-
                🎉 no goals
              -/


theorem eq_one_of_monic_natDegree_zero (hf : p.Monic) (hfd : p.natDegree = 0) : p = 1 := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    hf : p.Monic
    hfd : Eq p.natDegree 0
    ⊢ Eq p 1
  -/
  rw [Monic.def, leadingCoeff, hfd] at hf
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    hf : Eq (p.coeff 0) 1
    hfd : Eq p.natDegree 0
    ⊢ Eq p 1
  -/
  rw [eq_C_of_natDegree_eq_zero hfd, hf, map_one]
  /-
    🎉 no goals
  -/


theorem Monic.natDegree_eq_zero (hf : p.Monic) : p.natDegree = 0 ↔ p = 1 :=
                                         /-
                                           R : Type u
                                           inst✝ : Semiring R
                                           p : Polynomial R
                                           hf : p.Monic
                                           ⊢ Eq p 1 → Eq p.natDegree 0
                                         -/
  ⟨eq_one_of_monic_natDegree_zero hf, by rintro rfl; simp⟩
                                                     /-
                                                       🎉 no goals
                                                     -/


theorem degree_sum_fin_lt {n : ℕ} (f : Fin n → R) :
    degree (∑ i : Fin n, C (f i) * X ^ (i : ℕ)) < n :=
  (degree_sum_le _ _).trans_lt <|
    (Finset.sup_lt_iff <| WithBot.bot_lt_coe n).2 fun k _hk =>
      (degree_C_mul_X_pow_le _ _).trans_lt <| WithBot.coe_lt_coe.2 k.is_lt


theorem degree_C_lt_degree_C_mul_X (ha : a ≠ 0) : degree (C b) < degree (C a * X) := by
  /-
    R : Type u
    a b : R
    inst✝ : Semiring R
    ha : Ne a 0
    ⊢ LT.lt (Polynomial.C b).degree (HMul.hMul (Polynomial.C a) Polynomial.X).degree
  -/
  simpa only [degree_C_mul_X ha] using degree_C_lt
  /-
    🎉 no goals
  -/


@[simp] lemma natDegree_mul_X (hp : p ≠ 0) : natDegree (p * X) = natDegree p + 1 := by
  /-
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : Nontrivial R
    p : Polynomial R
    hp : Ne p 0
    ⊢ Eq (HMul.hMul p Polynomial.X).natDegree (HAdd.hAdd p.natDegree 1)
  -/
  rw [natDegree_mul' (by simpa), natDegree_X]
  /-
    🎉 no goals
  -/


@[simp] lemma natDegree_X_mul (hp : p ≠ 0) : natDegree (X * p) = natDegree p + 1 := by
  /-
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : Nontrivial R
    p : Polynomial R
    hp : Ne p 0
    ⊢ Eq (HMul.hMul Polynomial.X p).natDegree (HAdd.hAdd p.natDegree 1)
  -/
  rw [commute_X p, natDegree_mul_X hp]
  /-
    🎉 no goals
  -/


@[simp] lemma natDegree_mul_X_pow (hp : p ≠ 0) : natDegree (p * X ^ n) = natDegree p + n := by
  /-
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : Nontrivial R
    p : Polynomial R
    n : Nat
    hp : Ne p 0
    ⊢ Eq (HMul.hMul p (HPow.hPow Polynomial.X n)).natDegree (HAdd.hAdd p.natDegree …
  -/
  rw [natDegree_mul' (by simpa), natDegree_X_pow]
  /-
    🎉 no goals
  -/


@[simp] lemma natDegree_X_pow_mul (hp : p ≠ 0) : natDegree (X ^ n * p) = natDegree p + n := by
  /-
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : Nontrivial R
    p : Polynomial R
    n : Nat
    hp : Ne p 0
    ⊢ Eq (HMul.hMul (HPow.hPow Polynomial.X n) p).natDegree (HAdd.hAdd p.natDegree …
  -/
  rw [commute_X_pow, natDegree_mul_X_pow n hp]
  /-
    🎉 no goals
  -/

--  This lemma explicitly does not require the `Nontrivial R` assumption.

theorem natDegree_X_pow_le {R : Type*} [Semiring R] (n : ℕ) : (X ^ n : R[X]).natDegree ≤ n := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    n : Nat
    ⊢ LE.le (HPow.hPow Polynomial.X n).natDegree n
  -/
  nontriviality R
  /-
    R : Type u_1
    inst✝ : Semiring R
    n : Nat
    a✝ : Nontrivial R
    ⊢ LE.le (HPow.hPow Polynomial.X n).natDegree n
  -/
  rw [Polynomial.natDegree_X_pow]
  /-
    🎉 no goals
  -/


theorem not_isUnit_X : ¬IsUnit (X : R[X]) := fun ⟨⟨_, g, _hfg, hgf⟩, rfl⟩ =>
  zero_ne_one' R <| by
    /-
      R : Type u
      inst✝¹ : Semiring R
      inst✝ : Nontrivial R
      x✝ : IsUnit Polynomial.X
      g : Polynomial R
      _hfg : Eq (HMul.hMul ((Polynomial.monomial 1) 1) g) 1
      hgf : Eq (HMul.hMul g ((Polynomial.monomial 1) 1)) 1
      ⊢ Eq 0 1
    -/
    rw [← coeff_one_zero, ← hgf]
    /-
      R : Type u
      inst✝¹ : Semiring R
      inst✝ : Nontrivial R
      x✝ : IsUnit Polynomial.X
      g : Polynomial R
      _hfg : Eq (HMul.hMul ((Polynomial.monomial 1) 1) g) 1
      hgf : Eq (HMul.hMul g ((Polynomial.monomial 1) 1)) 1
      ⊢ Eq 0 ((HMul.hMul g ((Polynomial.monomial 1) 1)).coeff 0)
    -/
    simp
    /-
      🎉 no goals
    -/


@[simp]
                                                           /-
                                                             R : Type u
                                                             inst✝¹ : Semiring R
                                                             inst✝ : Nontrivial R
                                                             p : Polynomial R
                                                             ⊢ Eq (HMul.hMul p Polynomial.X).degree (HAdd.hAdd p.degree 1)
                                                           -/
theorem degree_mul_X : degree (p * X) = degree p + 1 := by simp [monic_X.degree_mul]
                                                           /-
                                                             🎉 no goals
                                                           -/


@[simp]
                                                                   /-
                                                                     R : Type u
                                                                     inst✝¹ : Semiring R
                                                                     inst✝ : Nontrivial R
                                                                     p : Polynomial R
                                                                     n : Nat
                                                                     ⊢ Eq (HMul.hMul p (HPow.hPow Polynomial.X n)).degree (HAdd.hAdd p.degree ↑n)
                                                                   -/
theorem degree_mul_X_pow : degree (p * X ^ n) = degree p + n := by simp [(monic_X_pow n).degree_mul]
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


theorem degree_sub_C (hp : 0 < degree p) : degree (p - C a) = degree p := by
  /-
    R : Type u
    a : R
    inst✝ : Ring R
    p : Polynomial R
    hp : LT.lt 0 p.degree
    ⊢ Eq (HSub.hSub p (Polynomial.C a)).degree p.degree
  -/
  rw [sub_eq_add_neg, ← C_neg, degree_add_C hp]
  /-
    🎉 no goals
  -/


@[simp]
theorem natDegree_sub_C {a : R} : natDegree (p - C a) = natDegree p := by
  /-
    R : Type u
    inst✝ : Ring R
    p : Polynomial R
    a : R
    ⊢ Eq (HSub.hSub p (Polynomial.C a)).natDegree p.natDegree
  -/
  rw [sub_eq_add_neg, ← C_neg, natDegree_add_C]
  /-
    🎉 no goals
  -/


theorem leadingCoeff_sub_of_degree_lt (h : Polynomial.degree q < Polynomial.degree p) :
    (p - q).leadingCoeff = p.leadingCoeff := by
  /-
    R : Type u
    inst✝ : Ring R
    p q : Polynomial R
    h : LT.lt q.degree p.degree
    ⊢ Eq (HSub.hSub p q).leadingCoeff p.leadingCoeff
  -/
  rw [← q.degree_neg] at h
  /-
    R : Type u
    inst✝ : Ring R
    p q : Polynomial R
    h : LT.lt (Neg.neg q).degree p.degree
    ⊢ Eq (HSub.hSub p q).leadingCoeff p.leadingCoeff
  -/
  rw [sub_eq_add_neg, leadingCoeff_add_of_degree_lt' h]
  /-
    🎉 no goals
  -/


theorem leadingCoeff_sub_of_degree_lt' (h : Polynomial.degree p < Polynomial.degree q) :
    (p - q).leadingCoeff = -q.leadingCoeff := by
  /-
    R : Type u
    inst✝ : Ring R
    p q : Polynomial R
    h : LT.lt p.degree q.degree
    ⊢ Eq (HSub.hSub p q).leadingCoeff (Neg.neg q.leadingCoeff)
  -/
  rw [← q.degree_neg] at h
  /-
    R : Type u
    inst✝ : Ring R
    p q : Polynomial R
    h : LT.lt p.degree (Neg.neg q).degree
    ⊢ Eq (HSub.hSub p q).leadingCoeff (Neg.neg q.leadingCoeff)
  -/
  rw [sub_eq_add_neg, leadingCoeff_add_of_degree_lt h, leadingCoeff_neg]
  /-
    🎉 no goals
  -/


theorem leadingCoeff_sub_of_degree_eq (h : degree p = degree q)
    (hlc : leadingCoeff p ≠ leadingCoeff q) :
    leadingCoeff (p - q) = leadingCoeff p - leadingCoeff q := by
  /-
    R : Type u
    inst✝ : Ring R
    p q : Polynomial R
    h : Eq p.degree q.degree
    hlc : Ne p.leadingCoeff q.leadingCoeff
    ⊢ Eq (HSub.hSub p q).leadingCoeff (HSub.hSub p.leadingCoeff q.leadingCoeff)
  -/
  replace h : degree p = degree (-q) := by rwa [q.degree_neg]
  replace hlc : leadingCoeff p + leadingCoeff (-q) ≠ 0 := by
    rwa [← sub_ne_zero, sub_eq_add_neg, ← q.leadingCoeff_neg] at hlc
  /-
    R : Type u
    inst✝ : Ring R
    p q : Polynomial R
    h : Eq p.degree (Neg.neg q).degree
    hlc : Ne (HAdd.hAdd p.leadingCoeff (Neg.neg q).leadingCoeff) 0
    ⊢ Eq (HSub.hSub p q).leadingCoeff (HSub.hSub p.leadingCoeff q.leadingCoeff)
  -/
  rw [sub_eq_add_neg, leadingCoeff_add_of_degree_eq h hlc, leadingCoeff_neg, sub_eq_add_neg]
  /-
    🎉 no goals
  -/


theorem degree_sub_eq_left_of_degree_lt (h : degree q < degree p) : degree (p - q) = degree p := by
  /-
    R : Type u
    inst✝ : Ring R
    p q : Polynomial R
    h : LT.lt q.degree p.degree
    ⊢ Eq (HSub.hSub p q).degree p.degree
  -/
  rw [← degree_neg q] at h
  /-
    R : Type u
    inst✝ : Ring R
    p q : Polynomial R
    h : LT.lt (Neg.neg q).degree p.degree
    ⊢ Eq (HSub.hSub p q).degree p.degree
  -/
  rw [sub_eq_add_neg, degree_add_eq_left_of_degree_lt h]
  /-
    🎉 no goals
  -/


theorem degree_sub_eq_right_of_degree_lt (h : degree p < degree q) : degree (p - q) = degree q := by
  /-
    R : Type u
    inst✝ : Ring R
    p q : Polynomial R
    h : LT.lt p.degree q.degree
    ⊢ Eq (HSub.hSub p q).degree q.degree
  -/
  rw [← degree_neg q] at h
  /-
    R : Type u
    inst✝ : Ring R
    p q : Polynomial R
    h : LT.lt p.degree (Neg.neg q).degree
    ⊢ Eq (HSub.hSub p q).degree q.degree
  -/
  rw [sub_eq_add_neg, degree_add_eq_right_of_degree_lt h, degree_neg]
  /-
    🎉 no goals
  -/


theorem natDegree_sub_eq_left_of_natDegree_lt (h : natDegree q < natDegree p) :
    natDegree (p - q) = natDegree p :=
  natDegree_eq_of_degree_eq (degree_sub_eq_left_of_degree_lt (degree_lt_degree h))


theorem natDegree_sub_eq_right_of_natDegree_lt (h : natDegree p < natDegree q) :
    natDegree (p - q) = natDegree q :=
  natDegree_eq_of_degree_eq (degree_sub_eq_right_of_degree_lt (degree_lt_degree h))


@[simp]
theorem degree_X_add_C (a : R) : degree (X + C a) = 1 := by
  have : degree (C a) < degree (X : R[X]) :=
    calc
      degree (C a) ≤ 0 := degree_C_le
      _ < 1 := WithBot.coe_lt_coe.mpr zero_lt_one
      _ = degree X := degree_X.symm
  /-
    R : Type u
    inst✝¹ : Nontrivial R
    inst✝ : Semiring R
    a : R
    this : LT.lt (Polynomial.C a).degree Polynomial.X.degree
    ⊢ Eq (HAdd.hAdd Polynomial.X (Polynomial.C a)).degree 1
  -/
  rw [degree_add_eq_left_of_degree_lt this, degree_X]
  /-
    🎉 no goals
  -/


theorem natDegree_X_add_C (x : R) : (X + C x).natDegree = 1 :=
  natDegree_eq_of_degree_eq_some <| degree_X_add_C x


@[simp]
theorem nextCoeff_X_add_C [Semiring S] (c : S) : nextCoeff (X + C c) = c := by
  /-
    S : Type v
    inst✝ : Semiring S
    c : S
    ⊢ Eq (HAdd.hAdd Polynomial.X (Polynomial.C c)).nextCoeff c
  -/
  nontriviality S
  /-
    S : Type v
    inst✝ : Semiring S
    c : S
    a✝ : Nontrivial S
    ⊢ Eq (HAdd.hAdd Polynomial.X (Polynomial.C c)).nextCoeff c
  -/
  simp [nextCoeff_of_natDegree_pos]
  /-
    🎉 no goals
  -/


theorem degree_X_pow_add_C {n : ℕ} (hn : 0 < n) (a : R) : degree ((X : R[X]) ^ n + C a) = n := by
  have : degree (C a) < degree ((X : R[X]) ^ n) := degree_C_le.trans_lt <| by
    rwa [degree_X_pow, Nat.cast_pos]
  /-
    R : Type u
    inst✝¹ : Nontrivial R
    inst✝ : Semiring R
    n : Nat
    hn : LT.lt 0 n
    a : R
    this : LT.lt (Polynomial.C a).degree (HPow.hPow Polynomial.X n).degree
    ⊢ Eq (HAdd.hAdd (HPow.hPow Polynomial.X n) (Polynomial.C a)).degree ↑n
  -/
  rw [degree_add_eq_left_of_degree_lt this, degree_X_pow]
  /-
    🎉 no goals
  -/


theorem X_pow_add_C_ne_zero {n : ℕ} (hn : 0 < n) (a : R) : (X : R[X]) ^ n + C a ≠ 0 :=
  mt degree_eq_bot.2
    (show degree ((X : R[X]) ^ n + C a) ≠ ⊥ by
      /-
        R : Type u
        inst✝¹ : Nontrivial R
        inst✝ : Semiring R
        n : Nat
        hn : LT.lt 0 n
        a : R
        ⊢ Ne (HAdd.hAdd (HPow.hPow Polynomial.X n) (Polynomial.C a)).degree Bot.bot
      -/
      rw [degree_X_pow_add_C hn a]; exact WithBot.coe_ne_bot)
                                    /-
                                      🎉 no goals
                                    -/


theorem X_add_C_ne_zero (r : R) : X + C r ≠ 0 :=
  pow_one (X : R[X]) ▸ X_pow_add_C_ne_zero zero_lt_one r


theorem zero_nmem_multiset_map_X_add_C {α : Type*} (m : Multiset α) (f : α → R) :
    (0 : R[X]) ∉ m.map fun a => X + C (f a) := fun mem =>
  let ⟨_a, _, ha⟩ := Multiset.mem_map.mp mem
  X_add_C_ne_zero _ ha


theorem natDegree_X_pow_add_C {n : ℕ} {r : R} : (X ^ n + C r).natDegree = n := by
  /-
    R : Type u
    inst✝¹ : Nontrivial R
    inst✝ : Semiring R
    n : Nat
    r : R
    ⊢ Eq (HAdd.hAdd (HPow.hPow Polynomial.X n) (Polynomial.C r)).natDegree n
  -/
  by_cases hn : n = 0
    /-
      case pos
      R : Type u
      inst✝¹ : Nontrivial R
      inst✝ : Semiring R
      n : Nat
      r : R
      hn : Eq n 0
      ⊢ Eq (HAdd.hAdd (HPow.hPow Polynomial.X n) (Polynomial.C r)).natDegree n
    -/
  · rw [hn, pow_zero, ← C_1, ← RingHom.map_add, natDegree_C]
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u
      inst✝¹ : Nontrivial R
      inst✝ : Semiring R
      n : Nat
      r : R
      hn : Not (Eq n 0)
      ⊢ Eq (HAdd.hAdd (HPow.hPow Polynomial.X n) (Polynomial.C r)).natDegree n
    -/
  · exact natDegree_eq_of_degree_eq_some (degree_X_pow_add_C (pos_iff_ne_zero.mpr hn) r)
    /-
      🎉 no goals
    -/


theorem X_pow_add_C_ne_one {n : ℕ} (hn : 0 < n) (a : R) : (X : R[X]) ^ n + C a ≠ 1 := fun h =>
               /-
                 R : Type u
                 inst✝¹ : Nontrivial R
                 inst✝ : Semiring R
                 n : Nat
                 hn : LT.lt 0 n
                 a : R
                 h : Eq (HAdd.hAdd (HPow.hPow Polynomial.X n) (Polynomial.C a)) 1
                 ⊢ Eq n 0
               -/
  hn.ne' <| by simpa only [natDegree_X_pow_add_C, natDegree_one] using congr_arg natDegree h
               /-
                 🎉 no goals
               -/


theorem X_add_C_ne_one (r : R) : X + C r ≠ 1 :=
  pow_one (X : R[X]) ▸ X_pow_add_C_ne_one zero_lt_one r


@[simp]
theorem leadingCoeff_X_pow_add_C {n : ℕ} (hn : 0 < n) {r : R} :
    (X ^ n + C r).leadingCoeff = 1 := by
  /-
    R : Type u
    inst✝ : Semiring R
    n : Nat
    hn : LT.lt 0 n
    r : R
    ⊢ Eq (HAdd.hAdd (HPow.hPow Polynomial.X n) (Polynomial.C r)).leadingCoeff 1
  -/
  nontriviality R
  rw [leadingCoeff, natDegree_X_pow_add_C, coeff_add, coeff_X_pow_self, coeff_C,
    if_neg (pos_iff_ne_zero.mp hn), add_zero]


@[simp]
theorem leadingCoeff_X_add_C [Semiring S] (r : S) : (X + C r).leadingCoeff = 1 := by
  /-
    S : Type v
    inst✝ : Semiring S
    r : S
    ⊢ Eq (HAdd.hAdd Polynomial.X (Polynomial.C r)).leadingCoeff 1
  -/
  rw [← pow_one (X : S[X]), leadingCoeff_X_pow_add_C zero_lt_one]
  /-
    🎉 no goals
  -/


@[simp]
theorem leadingCoeff_X_pow_add_one {n : ℕ} (hn : 0 < n) : (X ^ n + 1 : R[X]).leadingCoeff = 1 :=
  leadingCoeff_X_pow_add_C hn


@[simp]
theorem leadingCoeff_pow_X_add_C (r : R) (i : ℕ) : leadingCoeff ((X + C r) ^ i) = 1 := by
  /-
    R : Type u
    inst✝ : Semiring R
    r : R
    i : Nat
    ⊢ Eq (HPow.hPow (HAdd.hAdd Polynomial.X (Polynomial.C r)) i).leadingCoeff 1
  -/
  nontriviality
  /-
    R : Type u
    inst✝ : Semiring R
    r : R
    i : Nat
    a✝ : Nontrivial R
    ⊢ Eq (HPow.hPow (HAdd.hAdd Polynomial.X (Polynomial.C r)) i).leadingCoeff 1
  -/
                             /-
                               🎉 no goals
                             -/
  rw [leadingCoeff_pow'] <;> simp
                             /-
                               🎉 no goals
                             -/


@[simp]
lemma degree_mul : degree (p * q) = degree p + degree q :=
  letI := Classical.decEq R
                         /-
                           R : Type u
                           inst✝¹ : Semiring R
                           inst✝ : NoZeroDivisors R
                           p q : Polynomial R
                           this : DecidableEq R := Classical.decEq R
                           hp0 : Eq p 0
                           ⊢ Eq (HMul.hMul p q).degree (HAdd.hAdd p.degree q.degree)
                         -/
  if hp0 : p = 0 then by simp only [hp0, degree_zero, zero_mul, WithBot.bot_add]
                         /-
                           🎉 no goals
                         -/
  else
                           /-
                             R : Type u
                             inst✝¹ : Semiring R
                             inst✝ : NoZeroDivisors R
                             p q : Polynomial R
                             this : DecidableEq R := Classical.decEq R
                             hp0 : Not (Eq p 0)
                             hq0 : Eq q 0
                             ⊢ Eq (HMul.hMul p q).degree (HAdd.hAdd p.degree q.degree)
                           -/
    if hq0 : q = 0 then by simp only [hq0, degree_zero, mul_zero, WithBot.add_bot]
                           /-
                             🎉 no goals
                           -/
    else degree_mul' <| mul_ne_zero (mt leadingCoeff_eq_zero.1 hp0) (mt leadingCoeff_eq_zero.1 hq0)


/-- `degree` as a monoid homomorphism between `R[X]` and `Multiplicative (WithBot ℕ)`.
  This is useful to prove results about multiplication and degree. -/
def degreeMonoidHom [Nontrivial R] : R[X] →* Multiplicative (WithBot ℕ) where
  toFun := degree
  map_one' := degree_one
  map_mul' _ _ := degree_mul


@[simp]
lemma degree_pow [Nontrivial R] (p : R[X]) (n : ℕ) : degree (p ^ n) = n • degree p :=
  map_pow (@degreeMonoidHom R _ _ _) _ _


@[simp]
lemma leadingCoeff_mul (p q : R[X]) : leadingCoeff (p * q) = leadingCoeff p * leadingCoeff q := by
  /-
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : NoZeroDivisors R
    p q : Polynomial R
    ⊢ Eq (HMul.hMul p q).leadingCoeff (HMul.hMul p.leadingCoeff q.leadingCoeff)
  -/
  by_cases hp : p = 0
    /-
      case pos
      R : Type u
      inst✝¹ : Semiring R
      inst✝ : NoZeroDivisors R
      p q : Polynomial R
      hp : Eq p 0
      ⊢ Eq (HMul.hMul p q).leadingCoeff (HMul.hMul p.leadingCoeff q.leadingCoeff)
    -/
  · simp only [hp, zero_mul, leadingCoeff_zero]
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u
      inst✝¹ : Semiring R
      inst✝ : NoZeroDivisors R
      p q : Polynomial R
      hp : Not (Eq p 0)
      ⊢ Eq (HMul.hMul p q).leadingCoeff (HMul.hMul p.leadingCoeff q.leadingCoeff)
    -/
  · by_cases hq : q = 0
      /-
        case pos
        R : Type u
        inst✝¹ : Semiring R
        inst✝ : NoZeroDivisors R
        p q : Polynomial R
        hp : Not (Eq p 0)
        hq : Eq q 0
        ⊢ Eq (HMul.hMul p q).leadingCoeff (HMul.hMul p.leadingCoeff q.leadingCoeff)
      -/
    · simp only [hq, mul_zero, leadingCoeff_zero]
      /-
        🎉 no goals
      -/
      /-
        case neg
        R : Type u
        inst✝¹ : Semiring R
        inst✝ : NoZeroDivisors R
        p q : Polynomial R
        hp : Not (Eq p 0)
        hq : Not (Eq q 0)
        ⊢ Eq (HMul.hMul p q).leadingCoeff (HMul.hMul p.leadingCoeff q.leadingCoeff)
      -/
    · rw [leadingCoeff_mul']
      /-
        case neg
        R : Type u
        inst✝¹ : Semiring R
        inst✝ : NoZeroDivisors R
        p q : Polynomial R
        hp : Not (Eq p 0)
        hq : Not (Eq q 0)
        ⊢ Ne (HMul.hMul p.leadingCoeff q.leadingCoeff) 0
      -/
      exact mul_ne_zero (mt leadingCoeff_eq_zero.1 hp) (mt leadingCoeff_eq_zero.1 hq)
      /-
        🎉 no goals
      -/


/-- `Polynomial.leadingCoeff` bundled as a `MonoidHom` when `R` has `NoZeroDivisors`, and thus
  `leadingCoeff` is multiplicative -/
def leadingCoeffHom : R[X] →* R where
  toFun := leadingCoeff
                 /-
                   R : Type u
                   S : Type v
                   a b c d : R
                   n m : Nat
                   inst✝¹ : Semiring R
                   inst✝ : NoZeroDivisors R
                   p q : Polynomial R
                   ⊢ Eq (Polynomial.leadingCoeff 1) 1
                 -/
  map_one' := by simp
                 /-
                   🎉 no goals
                 -/
  map_mul' := leadingCoeff_mul


@[simp]
lemma leadingCoeffHom_apply (p : R[X]) : leadingCoeffHom p = leadingCoeff p :=
  rfl


@[simp]
lemma leadingCoeff_pow (p : R[X]) (n : ℕ) : leadingCoeff (p ^ n) = leadingCoeff p ^ n :=
  (leadingCoeffHom : R[X] →* R).map_pow p n


lemma leadingCoeff_dvd_leadingCoeff {a p : R[X]} (hap : a ∣ p) :
    a.leadingCoeff ∣ p.leadingCoeff :=
  map_dvd leadingCoeffHom hap


lemma degree_le_mul_left (p : R[X]) (hq : q ≠ 0) : degree p ≤ degree (p * q) := by
  classical
  obtain rfl | hp := eq_or_ne p 0
  · simp
  · rw [degree_mul, degree_eq_natDegree hp, degree_eq_natDegree hq]
    exact WithBot.coe_le_coe.2 (Nat.le_add_right _ _)


lemma Monic.natDegree_pos : 0 < natDegree p ↔ p ≠ 1 :=
  Nat.pos_iff_ne_zero.trans hp.natDegree_eq_zero.not


lemma Monic.degree_pos : 0 < degree p ↔ p ≠ 1 :=
  natDegree_pos_iff_degree_pos.symm.trans hp.natDegree_pos


@[simp]
theorem leadingCoeff_X_pow_sub_C {n : ℕ} (hn : 0 < n) {r : R} :
    (X ^ n - C r).leadingCoeff = 1 := by
  /-
    R : Type u
    inst✝ : Ring R
    n : Nat
    hn : LT.lt 0 n
    r : R
    ⊢ Eq (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C r)).leadingCoeff 1
  -/
  rw [sub_eq_add_neg, ← map_neg C r, leadingCoeff_X_pow_add_C hn]
  /-
    🎉 no goals
  -/


@[simp]
theorem leadingCoeff_X_pow_sub_one {n : ℕ} (hn : 0 < n) : (X ^ n - 1 : R[X]).leadingCoeff = 1 :=
  leadingCoeff_X_pow_sub_C hn


@[simp]
theorem degree_X_sub_C (a : R) : degree (X - C a) = 1 := by
  /-
    R : Type u
    inst✝¹ : Ring R
    inst✝ : Nontrivial R
    a : R
    ⊢ Eq (HSub.hSub Polynomial.X (Polynomial.C a)).degree 1
  -/
  rw [sub_eq_add_neg, ← map_neg C a, degree_X_add_C]
  /-
    🎉 no goals
  -/


theorem natDegree_X_sub_C (x : R) : (X - C x).natDegree = 1 := by
  /-
    R : Type u
    inst✝¹ : Ring R
    inst✝ : Nontrivial R
    x : R
    ⊢ Eq (HSub.hSub Polynomial.X (Polynomial.C x)).natDegree 1
  -/
  rw [natDegree_sub_C, natDegree_X]
  /-
    🎉 no goals
  -/


@[simp]
theorem nextCoeff_X_sub_C [Ring S] (c : S) : nextCoeff (X - C c) = -c := by
  /-
    S : Type v
    inst✝ : Ring S
    c : S
    ⊢ Eq (HSub.hSub Polynomial.X (Polynomial.C c)).nextCoeff (Neg.neg c)
  -/
  rw [sub_eq_add_neg, ← map_neg C c, nextCoeff_X_add_C]
  /-
    🎉 no goals
  -/


theorem degree_X_pow_sub_C {n : ℕ} (hn : 0 < n) (a : R) : degree ((X : R[X]) ^ n - C a) = n := by
  /-
    R : Type u
    inst✝¹ : Ring R
    inst✝ : Nontrivial R
    n : Nat
    hn : LT.lt 0 n
    a : R
    ⊢ Eq (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a)).degree ↑n
  -/
  rw [sub_eq_add_neg, ← map_neg C a, degree_X_pow_add_C hn]
  /-
    🎉 no goals
  -/


theorem X_pow_sub_C_ne_zero {n : ℕ} (hn : 0 < n) (a : R) : (X : R[X]) ^ n - C a ≠ 0 := by
  /-
    R : Type u
    inst✝¹ : Ring R
    inst✝ : Nontrivial R
    n : Nat
    hn : LT.lt 0 n
    a : R
    ⊢ Ne (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a)) 0
  -/
  rw [sub_eq_add_neg, ← map_neg C a]
  /-
    R : Type u
    inst✝¹ : Ring R
    inst✝ : Nontrivial R
    n : Nat
    hn : LT.lt 0 n
    a : R
    ⊢ Ne (HAdd.hAdd (HPow.hPow Polynomial.X n) (Polynomial.C (Neg.neg a))) 0
  -/
  exact X_pow_add_C_ne_zero hn _
  /-
    🎉 no goals
  -/


theorem X_sub_C_ne_zero (r : R) : X - C r ≠ 0 :=
  pow_one (X : R[X]) ▸ X_pow_sub_C_ne_zero zero_lt_one r


theorem zero_nmem_multiset_map_X_sub_C {α : Type*} (m : Multiset α) (f : α → R) :
    (0 : R[X]) ∉ m.map fun a => X - C (f a) := fun mem =>
  let ⟨_a, _, ha⟩ := Multiset.mem_map.mp mem
  X_sub_C_ne_zero _ ha


theorem natDegree_X_pow_sub_C {n : ℕ} {r : R} : (X ^ n - C r).natDegree = n := by
  /-
    R : Type u
    inst✝¹ : Ring R
    inst✝ : Nontrivial R
    n : Nat
    r : R
    ⊢ Eq (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C r)).natDegree n
  -/
  rw [sub_eq_add_neg, ← map_neg C r, natDegree_X_pow_add_C]
  /-
    🎉 no goals
  -/


@[simp]
theorem leadingCoeff_X_sub_C [Ring S] (r : S) : (X - C r).leadingCoeff = 1 := by
  /-
    S : Type v
    inst✝ : Ring S
    r : S
    ⊢ Eq (HSub.hSub Polynomial.X (Polynomial.C r)).leadingCoeff 1
  -/
  rw [sub_eq_add_neg, ← map_neg C r, leadingCoeff_X_add_C]
  /-
    🎉 no goals
  -/


