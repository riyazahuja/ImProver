/-- Helper function for the extended GCD algorithm (`Nat.xgcd`). -/
def xgcdAux : ℕ → ℤ → ℤ → ℕ → ℤ → ℤ → ℕ × ℤ × ℤ
  | 0, _, _, r', s', t' => (r', s', t')
  | succ k, s, t, r', s', t' =>
    let q := r' / succ k
    xgcdAux (r' % succ k) (s' - q * s) (t' - q * t) (succ k) s t
termination_by k => k
/-
  k : Nat
  s t : Int
  r' : Nat
  s' t' : Int
  q : Nat := HDiv.hDiv r' k.succ
  ⊢ LT.lt (HMod.hMod r' k.succ) k.succ
-/
decreasing_by exact mod_lt _ <| (succ_pos _).gt
/-
  🎉 no goals
-/


@[simp]
                                                                                    /-
                                                                                      s t : Int
                                                                                      r' : Nat
                                                                                      s' t' : Int
                                                                                      ⊢ Eq (Nat.xgcdAux 0 s t r' s' t') { fst := r', snd := { fst := s', snd := t' } }
                                                                                    -/
theorem xgcd_zero_left {s t r' s' t'} : xgcdAux 0 s t r' s' t' = (r', s', t') := by simp [xgcdAux]
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


theorem xgcdAux_rec {r s t r' s' t'} (h : 0 < r) :
    xgcdAux r s t r' s' t' = xgcdAux (r' % r) (s' - r' / r * s) (t' - r' / r * t) r s t := by
  /-
    r : Nat
    s t : Int
    r' : Nat
    s' t' : Int
    h : LT.lt 0 r
    ⊢ Eq (r.xgcdAux s t r' s' t') ((HMod.hMod r' r).xgcdAux (HSub.hSub s' (HMul.hM …
  -/
  obtain ⟨r, rfl⟩ := Nat.exists_eq_succ_of_ne_zero h.ne'
  /-
    case intro
    s t : Int
    r' : Nat
    s' t' : Int
    r : Nat
    h : LT.lt 0 r.succ
    ⊢ Eq (r.succ.xgcdAux s t r' s' t') ((HMod.hMod r' r.succ).xgcdAux (HSub.hSub s …
  -/
  simp [xgcdAux]
  /-
    🎉 no goals
  -/


/-- Use the extended GCD algorithm to generate the `a` and `b` values
  satisfying `gcd x y = x * a + y * b`. -/
def xgcd (x y : ℕ) : ℤ × ℤ :=
  (xgcdAux x 1 0 y 0 1).2


/-- The extended GCD `a` value in the equation `gcd x y = x * a + y * b`. -/
def gcdA (x y : ℕ) : ℤ :=
  (xgcd x y).1


/-- The extended GCD `b` value in the equation `gcd x y = x * a + y * b`. -/
def gcdB (x y : ℕ) : ℤ :=
  (xgcd x y).2


@[simp]
theorem gcdA_zero_left {s : ℕ} : gcdA 0 s = 0 := by
  /-
    s : Nat
    ⊢ Eq (Nat.gcdA 0 s) 0
  -/
  unfold gcdA
  /-
    s : Nat
    ⊢ Eq (Nat.xgcd 0 s).1 0
  -/
  rw [xgcd, xgcd_zero_left]
  /-
    🎉 no goals
  -/


@[simp]
theorem gcdB_zero_left {s : ℕ} : gcdB 0 s = 1 := by
  /-
    s : Nat
    ⊢ Eq (Nat.gcdB 0 s) 1
  -/
  unfold gcdB
  /-
    s : Nat
    ⊢ Eq (Nat.xgcd 0 s).2 1
  -/
  rw [xgcd, xgcd_zero_left]
  /-
    🎉 no goals
  -/


@[simp]
theorem gcdA_zero_right {s : ℕ} (h : s ≠ 0) : gcdA s 0 = 1 := by
  /-
    s : Nat
    h : Ne s 0
    ⊢ Eq (s.gcdA 0) 1
  -/
  unfold gcdA xgcd
  /-
    s : Nat
    h : Ne s 0
    ⊢ Eq (s.xgcdAux 1 0 0 0 1).2.1 1
  -/
  obtain ⟨s, rfl⟩ := Nat.exists_eq_succ_of_ne_zero h
  /-
    case intro
    s : Nat
    h : Ne s.succ 0
    ⊢ Eq (s.succ.xgcdAux 1 0 0 0 1).2.1 1
  -/
  rw [xgcdAux]
  /-
    case intro
    s : Nat
    h : Ne s.succ 0
    ⊢ Eq ((HMod.hMod 0 s.succ).xgcdAux (HSub.hSub 0 (HMul.hMul (↑(HDiv.hDiv 0 s.su …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem gcdB_zero_right {s : ℕ} (h : s ≠ 0) : gcdB s 0 = 0 := by
  /-
    s : Nat
    h : Ne s 0
    ⊢ Eq (s.gcdB 0) 0
  -/
  unfold gcdB xgcd
  /-
    s : Nat
    h : Ne s 0
    ⊢ Eq (s.xgcdAux 1 0 0 0 1).2.2 0
  -/
  obtain ⟨s, rfl⟩ := Nat.exists_eq_succ_of_ne_zero h
  /-
    case intro
    s : Nat
    h : Ne s.succ 0
    ⊢ Eq (s.succ.xgcdAux 1 0 0 0 1).2.2 0
  -/
  rw [xgcdAux]
  /-
    case intro
    s : Nat
    h : Ne s.succ 0
    ⊢ Eq ((HMod.hMod 0 s.succ).xgcdAux (HSub.hSub 0 (HMul.hMul (↑(HDiv.hDiv 0 s.su …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem xgcdAux_fst (x y) : ∀ s t s' t', (xgcdAux x s t y s' t').1 = gcd x y :=
                        /-
                          x y : Nat
                          ⊢ ∀ (n : Nat) (s t s' t' : Int), Eq (Nat.xgcdAux 0 s t n s' t').1 (Nat.gcd 0 n)
                        -/
  gcd.induction x y (by simp) fun x y h IH s t s' t' => by
                        /-
                          🎉 no goals
                        -/
    /-
      x✝ y✝ x y : Nat
      h : LT.lt 0 x
      IH : ∀ (s t s' t' : Int), Eq ((HMod.hMod y x).xgcdAux s t x s' t').1 ((HMod.hM …
      s t s' t' : Int
      ⊢ Eq (x.xgcdAux s t y s' t').1 (x.gcd y)
    -/
    simp only [h, xgcdAux_rec, IH]
    /-
      x✝ y✝ x y : Nat
      h : LT.lt 0 x
      IH : ∀ (s t s' t' : Int), Eq ((HMod.hMod y x).xgcdAux s t x s' t').1 ((HMod.hM …
      s t s' t' : Int
      ⊢ Eq ((HMod.hMod y x).gcd x) (x.gcd y)
    -/
    rw [← gcd_rec]
    /-
      🎉 no goals
    -/


theorem xgcdAux_val (x y) : xgcdAux x 1 0 y 0 1 = (gcd x y, xgcd x y) := by
  /-
    x y : Nat
    ⊢ Eq (x.xgcdAux 1 0 y 0 1) { fst := x.gcd y, snd := x.xgcd y }
  -/
  rw [xgcd, ← xgcdAux_fst x y 1 0 0 1]
  /-
    🎉 no goals
  -/


theorem xgcd_val (x y) : xgcd x y = (gcdA x y, gcdB x y) := by
  /-
    x y : Nat
    ⊢ Eq (x.xgcd y) { fst := x.gcdA y, snd := x.gcdB y }
  -/
  unfold gcdA gcdB; cases xgcd x y; rfl
                                    /-
                                      🎉 no goals
                                    -/


private def P : ℕ × ℤ × ℤ → Prop
  | (r, s, t) => (r : ℤ) = x * s + y * t


theorem xgcdAux_P {r r'} :
    ∀ {s t s' t'}, P x y (r, s, t) → P x y (r', s', t') → P x y (xgcdAux r s t r' s' t') := by
  induction r, r' using gcd.induction with
  | H0 => simp
  | H1 a b h IH =>
    intro s t s' t' p p'
    rw [xgcdAux_rec h]; refine IH ?_ p; dsimp [P] at *
    rw [Int.emod_def]; generalize (b / a : ℤ) = k
    rw [p, p', Int.mul_sub, sub_add_eq_add_sub, Int.mul_sub, Int.add_mul, mul_comm k t,
      mul_comm k s, ← mul_assoc, ← mul_assoc, add_comm (x * s * k), ← add_sub_assoc, sub_sub]


/-- **Bézout's lemma**: given `x y : ℕ`, `gcd x y = x * a + y * b`, where `a = gcd_a x y` and
`b = gcd_b x y` are computed by the extended Euclidean algorithm.
-/
theorem gcd_eq_gcd_ab : (gcd x y : ℤ) = x * gcdA x y + y * gcdB x y := by
  /-
    x y : Nat
    ⊢ Eq (↑(x.gcd y)) (HAdd.hAdd (HMul.hMul (↑x) (x.gcdA y)) (HMul.hMul (↑y) (x.gc …
  -/
  have := @xgcdAux_P x y x y 1 0 0 1 (by simp [P]) (by simp [P])
  /-
    x y : Nat
    this : Nat.P x y (x.xgcdAux 1 0 y 0 1)
    ⊢ Eq (↑(x.gcd y)) (HAdd.hAdd (HMul.hMul (↑x) (x.gcdA y)) (HMul.hMul (↑y) (x.gc …
  -/
  rwa [xgcdAux_val, xgcd_val] at this
  /-
    🎉 no goals
  -/


theorem exists_mul_emod_eq_gcd {k n : ℕ} (hk : gcd n k < k) : ∃ m, n * m % k = gcd n k := by
  /-
    k n : Nat
    hk : LT.lt (n.gcd k) k
    ⊢ Exists fun m => Eq (HMod.hMod (HMul.hMul n m) k) (n.gcd k)
  -/
  have hk' := Int.ofNat_ne_zero.2 (ne_of_gt (lt_of_le_of_lt (zero_le (gcd n k)) hk))
  /-
    k n : Nat
    hk : LT.lt (n.gcd k) k
    hk' : Ne (↑k) 0
    ⊢ Exists fun m => Eq (HMod.hMod (HMul.hMul n m) k) (n.gcd k)
  -/
  have key := congr_arg (fun (m : ℤ) => (m % k).toNat) (gcd_eq_gcd_ab n k)
  /-
    k n : Nat
    hk : LT.lt (n.gcd k) k
    hk' : Ne (↑k) 0
    key : Eq ((fun m => (HMod.hMod m ↑k).toNat) ↑(n.gcd k)) ((fun m => (HMod.hMod  …
    ⊢ Exists fun m => Eq (HMod.hMod (HMul.hMul n m) k) (n.gcd k)
  -/
  simp only at key
  /-
    k n : Nat
    hk : LT.lt (n.gcd k) k
    hk' : Ne (↑k) 0
    key : Eq (HMod.hMod ↑(n.gcd k) ↑k).toNat (HMod.hMod (HAdd.hAdd (HMul.hMul (↑n) …
    ⊢ Exists fun m => Eq (HMod.hMod (HMul.hMul n m) k) (n.gcd k)
  -/
  rw [Int.add_mul_emod_self_left, ← Int.natCast_mod, Int.toNat_natCast, mod_eq_of_lt hk] at key
  /-
    k n : Nat
    hk : LT.lt (n.gcd k) k
    hk' : Ne (↑k) 0
    key : Eq (n.gcd k) (HMod.hMod (HMul.hMul (↑n) (n.gcdA k)) ↑k).toNat
    ⊢ Exists fun m => Eq (HMod.hMod (HMul.hMul n m) k) (n.gcd k)
  -/
  refine ⟨(n.gcdA k % k).toNat, Eq.trans (Int.ofNat.inj ?_) key.symm⟩
  rw [Int.ofNat_eq_coe, Int.natCast_mod, Int.ofNat_mul, Int.toNat_of_nonneg (Int.emod_nonneg _ hk'),
    Int.ofNat_eq_coe, Int.toNat_of_nonneg (Int.emod_nonneg _ hk'), Int.mul_emod, Int.emod_emod,
    ← Int.mul_emod]


theorem exists_mul_emod_eq_one_of_coprime {k n : ℕ} (hkn : Coprime n k) (hk : 1 < k) :
    ∃ m, n * m % k = 1 :=
  Exists.recOn (exists_mul_emod_eq_gcd (lt_of_le_of_lt (le_of_eq hkn) hk)) fun m hm ↦
    ⟨m, hm.trans hkn⟩


theorem gcd_def (i j : ℤ) : gcd i j = Nat.gcd i.natAbs j.natAbs := rfl


@[simp, norm_cast] protected lemma gcd_natCast_natCast (m n : ℕ) : gcd ↑m ↑n = m.gcd n := rfl


@[deprecated (since := "2024-05-25")] alias coe_nat_gcd := Int.gcd_natCast_natCast


/-- The extended GCD `a` value in the equation `gcd x y = x * a + y * b`. -/
def gcdA : ℤ → ℤ → ℤ
  | ofNat m, n => m.gcdA n.natAbs
  | -[m+1], n => -m.succ.gcdA n.natAbs


/-- The extended GCD `b` value in the equation `gcd x y = x * a + y * b`. -/
def gcdB : ℤ → ℤ → ℤ
  | m, ofNat n => m.natAbs.gcdB n
  | m, -[n+1] => -m.natAbs.gcdB n.succ


/-- **Bézout's lemma** -/
theorem gcd_eq_gcd_ab : ∀ x y : ℤ, (gcd x y : ℤ) = x * gcdA x y + y * gcdB x y
  | (m : ℕ), (n : ℕ) => Nat.gcd_eq_gcd_ab _ _
  | (m : ℕ), -[n+1] =>
                                        /-
                                          m n : Nat
                                          ⊢ Eq (↑((↑m).gcd (Int.negSucc n))) (HAdd.hAdd (HMul.hMul (↑m) ((↑m).gcdA (Int. …
                                        -/
    show (_ : ℤ) = _ + -(n + 1) * -_ by rw [Int.neg_mul_neg]; apply Nat.gcd_eq_gcd_ab
                                                              /-
                                                                🎉 no goals
                                                              -/
  | -[m+1], (n : ℕ) =>
                                        /-
                                          m n : Nat
                                          ⊢ Eq (↑((Int.negSucc m).gcd ↑n)) (HAdd.hAdd (HMul.hMul (Neg.neg (HAdd.hAdd (↑m …
                                        -/
    show (_ : ℤ) = -(m + 1) * -_ + _ by rw [Int.neg_mul_neg]; apply Nat.gcd_eq_gcd_ab
                                                              /-
                                                                🎉 no goals
                                                              -/
  | -[m+1], -[n+1] =>
    show (_ : ℤ) = -(m + 1) * -_ + -(n + 1) * -_ by
      /-
        m n : Nat
        ⊢ Eq (↑((Int.negSucc m).gcd (Int.negSucc n))) (HAdd.hAdd (HMul.hMul (Neg.neg ( …
      -/
      rw [Int.neg_mul_neg, Int.neg_mul_neg]
      /-
        m n : Nat
        ⊢ Eq (↑((Int.negSucc m).gcd (Int.negSucc n))) (HAdd.hAdd (HMul.hMul (HAdd.hAdd …
      -/
      apply Nat.gcd_eq_gcd_ab
      /-
        🎉 no goals
      -/


theorem lcm_def (i j : ℤ) : lcm i j = Nat.lcm (natAbs i) (natAbs j) :=
  rfl


protected theorem coe_nat_lcm (m n : ℕ) : Int.lcm ↑m ↑n = Nat.lcm m n :=
  rfl


theorem dvd_gcd {i j k : ℤ} (h1 : k ∣ i) (h2 : k ∣ j) : k ∣ gcd i j :=
  natAbs_dvd.1 <|
    natCast_dvd_natCast.2 <| Nat.dvd_gcd (natAbs_dvd_natAbs.2 h1) (natAbs_dvd_natAbs.2 h2)


theorem gcd_mul_lcm (i j : ℤ) : gcd i j * lcm i j = natAbs (i * j) := by
  /-
    i j : Int
    ⊢ Eq (HMul.hMul (i.gcd j) (i.lcm j)) (HMul.hMul i j).natAbs
  -/
  rw [Int.gcd, Int.lcm, Nat.gcd_mul_lcm, natAbs_mul]
  /-
    🎉 no goals
  -/


theorem gcd_comm (i j : ℤ) : gcd i j = gcd j i :=
  Nat.gcd_comm _ _


theorem gcd_assoc (i j k : ℤ) : gcd (gcd i j) k = gcd i (gcd j k) :=
  Nat.gcd_assoc _ _ _


@[simp]
                                                    /-
                                                      i : Int
                                                      ⊢ Eq (i.gcd i) i.natAbs
                                                    -/
theorem gcd_self (i : ℤ) : gcd i i = natAbs i := by simp [gcd]
                                                    /-
                                                      🎉 no goals
                                                    -/


@[simp]
                                                         /-
                                                           i : Int
                                                           ⊢ Eq (Int.gcd 0 i) i.natAbs
                                                         -/
theorem gcd_zero_left (i : ℤ) : gcd 0 i = natAbs i := by simp [gcd]
                                                         /-
                                                           🎉 no goals
                                                         -/


@[simp]
                                                          /-
                                                            i : Int
                                                            ⊢ Eq (i.gcd 0) i.natAbs
                                                          -/
theorem gcd_zero_right (i : ℤ) : gcd i 0 = natAbs i := by simp [gcd]
                                                          /-
                                                            🎉 no goals
                                                          -/


theorem gcd_mul_left (i j k : ℤ) : gcd (i * j) (i * k) = natAbs i * gcd j k := by
  /-
    i j k : Int
    ⊢ Eq ((HMul.hMul i j).gcd (HMul.hMul i k)) (HMul.hMul i.natAbs (j.gcd k))
  -/
  rw [Int.gcd, Int.gcd, natAbs_mul, natAbs_mul]
  /-
    i j k : Int
    ⊢ Eq ((HMul.hMul i.natAbs j.natAbs).gcd (HMul.hMul i.natAbs k.natAbs)) (HMul.h …
  -/
  apply Nat.gcd_mul_left
  /-
    🎉 no goals
  -/


theorem gcd_mul_right (i j k : ℤ) : gcd (i * j) (k * j) = gcd i k * natAbs j := by
  /-
    i j k : Int
    ⊢ Eq ((HMul.hMul i j).gcd (HMul.hMul k j)) (HMul.hMul (i.gcd k) j.natAbs)
  -/
  rw [Int.gcd, Int.gcd, natAbs_mul, natAbs_mul]
  /-
    i j k : Int
    ⊢ Eq ((HMul.hMul i.natAbs j.natAbs).gcd (HMul.hMul k.natAbs j.natAbs)) (HMul.h …
  -/
  apply Nat.gcd_mul_right
  /-
    🎉 no goals
  -/


theorem gcd_pos_of_ne_zero_left {i : ℤ} (j : ℤ) (hi : i ≠ 0) : 0 < gcd i j :=
  Nat.gcd_pos_of_pos_left _ <| natAbs_pos.2 hi


theorem gcd_pos_of_ne_zero_right (i : ℤ) {j : ℤ} (hj : j ≠ 0) : 0 < gcd i j :=
  Nat.gcd_pos_of_pos_right _ <| natAbs_pos.2 hj


theorem gcd_eq_zero_iff {i j : ℤ} : gcd i j = 0 ↔ i = 0 ∧ j = 0 := by
  /-
    i j : Int
    ⊢ Iff (Eq (i.gcd j) 0) (And (Eq i 0) (Eq j 0))
  -/
  rw [gcd, Nat.gcd_eq_zero_iff, natAbs_eq_zero, natAbs_eq_zero]
  /-
    🎉 no goals
  -/


theorem gcd_pos_iff {i j : ℤ} : 0 < gcd i j ↔ i ≠ 0 ∨ j ≠ 0 :=
  Nat.pos_iff_ne_zero.trans <| gcd_eq_zero_iff.not.trans not_and_or


theorem gcd_div {i j k : ℤ} (H1 : k ∣ i) (H2 : k ∣ j) :
    gcd (i / k) (j / k) = gcd i j / natAbs k := by
  /-
    i j k : Int
    H1 : Dvd.dvd k i
    H2 : Dvd.dvd k j
    ⊢ Eq ((HDiv.hDiv i k).gcd (HDiv.hDiv j k)) (HDiv.hDiv (i.gcd j) k.natAbs)
  -/
  rw [gcd, natAbs_ediv i k H1, natAbs_ediv j k H2]
  /-
    i j k : Int
    H1 : Dvd.dvd k i
    H2 : Dvd.dvd k j
    ⊢ Eq ((HDiv.hDiv i.natAbs k.natAbs).gcd (HDiv.hDiv j.natAbs k.natAbs)) (HDiv.h …
  -/
  exact Nat.gcd_div (natAbs_dvd_natAbs.mpr H1) (natAbs_dvd_natAbs.mpr H2)
  /-
    🎉 no goals
  -/


theorem gcd_div_gcd_div_gcd {i j : ℤ} (H : 0 < gcd i j) : gcd (i / gcd i j) (j / gcd i j) = 1 := by
  /-
    i j : Int
    H : LT.lt 0 (i.gcd j)
    ⊢ Eq ((HDiv.hDiv i ↑(i.gcd j)).gcd (HDiv.hDiv j ↑(i.gcd j))) 1
  -/
  rw [gcd_div gcd_dvd_left gcd_dvd_right, natAbs_ofNat, Nat.div_self H]
  /-
    🎉 no goals
  -/


theorem gcd_dvd_gcd_of_dvd_left {i k : ℤ} (j : ℤ) (H : i ∣ k) : gcd i j ∣ gcd k j :=
  Int.natCast_dvd_natCast.1 <| dvd_gcd (gcd_dvd_left.trans H) gcd_dvd_right


theorem gcd_dvd_gcd_of_dvd_right {i k : ℤ} (j : ℤ) (H : i ∣ k) : gcd j i ∣ gcd j k :=
  Int.natCast_dvd_natCast.1 <| dvd_gcd gcd_dvd_left (gcd_dvd_right.trans H)


theorem gcd_dvd_gcd_mul_left (i j k : ℤ) : gcd i j ∣ gcd (k * i) j :=
  gcd_dvd_gcd_of_dvd_left _ (dvd_mul_left _ _)


theorem gcd_dvd_gcd_mul_right (i j k : ℤ) : gcd i j ∣ gcd (i * k) j :=
  gcd_dvd_gcd_of_dvd_left _ (dvd_mul_right _ _)


theorem gcd_dvd_gcd_mul_left_right (i j k : ℤ) : gcd i j ∣ gcd i (k * j) :=
  gcd_dvd_gcd_of_dvd_right _ (dvd_mul_left _ _)


theorem gcd_dvd_gcd_mul_right_right (i j k : ℤ) : gcd i j ∣ gcd i (j * k) :=
  gcd_dvd_gcd_of_dvd_right _ (dvd_mul_right _ _)


/-- If `gcd a (m * n) = 1`, then `gcd a m = 1`. -/
theorem gcd_eq_one_of_gcd_mul_right_eq_one_left {a : ℤ} {m n : ℕ} (h : a.gcd (m * n) = 1) :
    a.gcd m = 1 :=
  Nat.dvd_one.mp <| h ▸ gcd_dvd_gcd_mul_right_right a m n


/-- If `gcd a (m * n) = 1`, then `gcd a n = 1`. -/
theorem gcd_eq_one_of_gcd_mul_right_eq_one_right {a : ℤ} {m n : ℕ} (h : a.gcd (m * n) = 1) :
    a.gcd n = 1 :=
  Nat.dvd_one.mp <| h ▸ gcd_dvd_gcd_mul_left_right a n m


theorem gcd_eq_left {i j : ℤ} (H : i ∣ j) : gcd i j = natAbs i :=
  Nat.dvd_antisymm (Nat.gcd_dvd_left _ _) (Nat.dvd_gcd dvd_rfl (natAbs_dvd_natAbs.mpr H))


                                                                      /-
                                                                        i j : Int
                                                                        H : Dvd.dvd j i
                                                                        ⊢ Eq (i.gcd j) j.natAbs
                                                                      -/
theorem gcd_eq_right {i j : ℤ} (H : j ∣ i) : gcd i j = natAbs j := by rw [gcd_comm, gcd_eq_left H]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


theorem ne_zero_of_gcd {x y : ℤ} (hc : gcd x y ≠ 0) : x ≠ 0 ∨ y ≠ 0 := by
  /-
    x y : Int
    hc : Ne (x.gcd y) 0
    ⊢ Or (Ne x 0) (Ne y 0)
  -/
  contrapose! hc
  /-
    x y : Int
    hc : And (Eq x 0) (Eq y 0)
    ⊢ Eq (x.gcd y) 0
  -/
  rw [hc.left, hc.right, gcd_zero_right, natAbs_zero]
  /-
    🎉 no goals
  -/


theorem exists_gcd_one {m n : ℤ} (H : 0 < gcd m n) :
    ∃ m' n' : ℤ, gcd m' n' = 1 ∧ m = m' * gcd m n ∧ n = n' * gcd m n :=
  ⟨_, _, gcd_div_gcd_div_gcd H, (Int.ediv_mul_cancel gcd_dvd_left).symm,
    (Int.ediv_mul_cancel gcd_dvd_right).symm⟩


theorem exists_gcd_one' {m n : ℤ} (H : 0 < gcd m n) :
    ∃ (g : ℕ) (m' n' : ℤ), 0 < g ∧ gcd m' n' = 1 ∧ m = m' * g ∧ n = n' * g :=
  let ⟨m', n', h⟩ := exists_gcd_one H
  ⟨_, m', n', H, h⟩


theorem pow_dvd_pow_iff {m n : ℤ} {k : ℕ} (k0 : k ≠ 0) : m ^ k ∣ n ^ k ↔ m ∣ n := by
  /-
    m n : Int
    k : Nat
    k0 : Ne k 0
    ⊢ Iff (Dvd.dvd (HPow.hPow m k) (HPow.hPow n k)) (Dvd.dvd m n)
  -/
  refine ⟨fun h => ?_, fun h => pow_dvd_pow_of_dvd h _⟩
  rwa [← natAbs_dvd_natAbs, ← Nat.pow_dvd_pow_iff k0, ← Int.natAbs_pow, ← Int.natAbs_pow,
    natAbs_dvd_natAbs]


theorem gcd_dvd_iff {a b : ℤ} {n : ℕ} : gcd a b ∣ n ↔ ∃ x y : ℤ, ↑n = a * x + b * y := by
  /-
    a b : Int
    n : Nat
    ⊢ Iff (Dvd.dvd (a.gcd b) n) (Exists fun x => Exists fun y => Eq (↑n) (HAdd.hAd …
  -/
  constructor
    /-
      case mp
      a b : Int
      n : Nat
      ⊢ Dvd.dvd (a.gcd b) n → Exists fun x => Exists fun y => Eq (↑n) (HAdd.hAdd (HM …
    -/
  · intro h
    /-
      case mp
      a b : Int
      n : Nat
      h : Dvd.dvd (a.gcd b) n
      ⊢ Exists fun x => Exists fun y => Eq (↑n) (HAdd.hAdd (HMul.hMul a x) (HMul.hMu …
    -/
    rw [← Nat.mul_div_cancel' h, Int.ofNat_mul, gcd_eq_gcd_ab, Int.add_mul, mul_assoc, mul_assoc]
    /-
      case mp
      a b : Int
      n : Nat
      h : Dvd.dvd (a.gcd b) n
      ⊢ Exists fun x => Exists fun y => Eq (HAdd.hAdd (HMul.hMul a (HMul.hMul (a.gcd …
    -/
    exact ⟨_, _, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      a b : Int
      n : Nat
      ⊢ (Exists fun x => Exists fun y => Eq (↑n) (HAdd.hAdd (HMul.hMul a x) (HMul.hM …
    -/
  · rintro ⟨x, y, h⟩
    /-
      case mpr.intro.intro
      a b : Int
      n : Nat
      x y : Int
      h : Eq (↑n) (HAdd.hAdd (HMul.hMul a x) (HMul.hMul b y))
      ⊢ Dvd.dvd (a.gcd b) n
    -/
    rw [← Int.natCast_dvd_natCast, h]
    /-
      case mpr.intro.intro
      a b : Int
      n : Nat
      x y : Int
      h : Eq (↑n) (HAdd.hAdd (HMul.hMul a x) (HMul.hMul b y))
      ⊢ Dvd.dvd (↑(a.gcd b)) (HAdd.hAdd (HMul.hMul a x) (HMul.hMul b y))
    -/
    exact Int.dvd_add (dvd_mul_of_dvd_left gcd_dvd_left _) (dvd_mul_of_dvd_left gcd_dvd_right y)
    /-
      🎉 no goals
    -/


theorem gcd_greatest {a b d : ℤ} (hd_pos : 0 ≤ d) (hda : d ∣ a) (hdb : d ∣ b)
    (hd : ∀ e : ℤ, e ∣ a → e ∣ b → e ∣ d) : d = gcd a b :=
  dvd_antisymm hd_pos (ofNat_zero_le (gcd a b)) (dvd_gcd hda hdb)
    (hd _ gcd_dvd_left gcd_dvd_right)


/-- Euclid's lemma: if `a ∣ b * c` and `gcd a c = 1` then `a ∣ b`.
Compare with `IsCoprime.dvd_of_dvd_mul_left` and
`UniqueFactorizationMonoid.dvd_of_dvd_mul_left_of_no_prime_factors` -/
theorem dvd_of_dvd_mul_left_of_gcd_one {a b c : ℤ} (habc : a ∣ b * c) (hab : gcd a c = 1) :
    a ∣ b := by
  /-
    a b c : Int
    habc : Dvd.dvd a (HMul.hMul b c)
    hab : Eq (a.gcd c) 1
    ⊢ Dvd.dvd a b
  -/
  have := gcd_eq_gcd_ab a c
  /-
    a b c : Int
    habc : Dvd.dvd a (HMul.hMul b c)
    hab : Eq (a.gcd c) 1
    this : Eq (↑(a.gcd c)) (HAdd.hAdd (HMul.hMul a (a.gcdA c)) (HMul.hMul c (a.gcd …
    ⊢ Dvd.dvd a b
  -/
  simp only [hab, Int.ofNat_zero, Int.ofNat_succ, zero_add] at this
  /-
    a b c : Int
    habc : Dvd.dvd a (HMul.hMul b c)
    hab : Eq (a.gcd c) 1
    this : Eq 1 (HAdd.hAdd (HMul.hMul a (a.gcdA c)) (HMul.hMul c (a.gcdB c)))
    ⊢ Dvd.dvd a b
  -/
  have : b * a * gcdA a c + b * c * gcdB a c = b := by simp [mul_assoc, ← Int.mul_add, ← this]
  /-
    a b c : Int
    habc : Dvd.dvd a (HMul.hMul b c)
    hab : Eq (a.gcd c) 1
    this✝ : Eq 1 (HAdd.hAdd (HMul.hMul a (a.gcdA c)) (HMul.hMul c (a.gcdB c)))
    this : Eq (HAdd.hAdd (HMul.hMul (HMul.hMul b a) (a.gcdA c)) (HMul.hMul (HMul.h …
    ⊢ Dvd.dvd a b
  -/
  rw [← this]
  /-
    a b c : Int
    habc : Dvd.dvd a (HMul.hMul b c)
    hab : Eq (a.gcd c) 1
    this✝ : Eq 1 (HAdd.hAdd (HMul.hMul a (a.gcdA c)) (HMul.hMul c (a.gcdB c)))
    this : Eq (HAdd.hAdd (HMul.hMul (HMul.hMul b a) (a.gcdA c)) (HMul.hMul (HMul.h …
    ⊢ Dvd.dvd a (HAdd.hAdd (HMul.hMul (HMul.hMul b a) (a.gcdA c)) (HMul.hMul (HMul …
  -/
  exact Int.dvd_add (dvd_mul_of_dvd_left (dvd_mul_left a b) _) (dvd_mul_of_dvd_left habc _)
  /-
    🎉 no goals
  -/


/-- Euclid's lemma: if `a ∣ b * c` and `gcd a b = 1` then `a ∣ c`.
Compare with `IsCoprime.dvd_of_dvd_mul_right` and
`UniqueFactorizationMonoid.dvd_of_dvd_mul_right_of_no_prime_factors` -/
theorem dvd_of_dvd_mul_right_of_gcd_one {a b c : ℤ} (habc : a ∣ b * c) (hab : gcd a b = 1) :
    a ∣ c := by
  /-
    a b c : Int
    habc : Dvd.dvd a (HMul.hMul b c)
    hab : Eq (a.gcd b) 1
    ⊢ Dvd.dvd a c
  -/
  rw [mul_comm] at habc
  /-
    a b c : Int
    habc : Dvd.dvd a (HMul.hMul c b)
    hab : Eq (a.gcd b) 1
    ⊢ Dvd.dvd a c
  -/
  exact dvd_of_dvd_mul_left_of_gcd_one habc hab
  /-
    🎉 no goals
  -/


/-- For nonzero integers `a` and `b`, `gcd a b` is the smallest positive natural number that can be
written in the form `a * x + b * y` for some pair of integers `x` and `y` -/
theorem gcd_least_linear {a b : ℤ} (ha : a ≠ 0) :
    IsLeast { n : ℕ | 0 < n ∧ ∃ x y : ℤ, ↑n = a * x + b * y } (a.gcd b) := by
  /-
    a b : Int
    ha : Ne a 0
    ⊢ IsLeast (setOf fun n => And (LT.lt 0 n) (Exists fun x => Exists fun y => Eq  …
  -/
  simp_rw [← gcd_dvd_iff]
  /-
    a b : Int
    ha : Ne a 0
    ⊢ IsLeast (setOf fun n => And (LT.lt 0 n) (Dvd.dvd (a.gcd b) n)) (a.gcd b)
  -/
  constructor
    /-
      case left
      a b : Int
      ha : Ne a 0
      ⊢ Membership.mem (setOf fun n => And (LT.lt 0 n) (Dvd.dvd (a.gcd b) n)) (a.gcd …
    -/
  · simpa [and_true, dvd_refl, Set.mem_setOf_eq] using gcd_pos_of_ne_zero_left b ha
    /-
      🎉 no goals
    -/
    /-
      case right
      a b : Int
      ha : Ne a 0
      ⊢ Membership.mem (lowerBounds (setOf fun n => And (LT.lt 0 n) (Dvd.dvd (a.gcd  …
    -/
  · simp only [lowerBounds, and_imp, Set.mem_setOf_eq]
    /-
      case right
      a b : Int
      ha : Ne a 0
      ⊢ ∀ ⦃a_1 : Nat⦄, LT.lt 0 a_1 → Dvd.dvd (a.gcd b) a_1 → LE.le (a.gcd b) a_1
    -/
    exact fun n hn_pos hn => Nat.le_of_dvd hn_pos hn
    /-
      🎉 no goals
    -/


theorem lcm_comm (i j : ℤ) : lcm i j = lcm j i := by
  /-
    i j : Int
    ⊢ Eq (i.lcm j) (j.lcm i)
  -/
  rw [Int.lcm, Int.lcm]
  /-
    i j : Int
    ⊢ Eq (i.natAbs.lcm j.natAbs) (j.natAbs.lcm i.natAbs)
  -/
  exact Nat.lcm_comm _ _
  /-
    🎉 no goals
  -/


theorem lcm_assoc (i j k : ℤ) : lcm (lcm i j) k = lcm i (lcm j k) := by
  /-
    i j k : Int
    ⊢ Eq ((↑(i.lcm j)).lcm k) (i.lcm ↑(j.lcm k))
  -/
  rw [Int.lcm, Int.lcm, Int.lcm, Int.lcm, natAbs_ofNat, natAbs_ofNat]
  /-
    i j k : Int
    ⊢ Eq ((i.natAbs.lcm j.natAbs).lcm k.natAbs) (i.natAbs.lcm (j.natAbs.lcm k.natA …
  -/
  apply Nat.lcm_assoc
  /-
    🎉 no goals
  -/


@[simp]
theorem lcm_zero_left (i : ℤ) : lcm 0 i = 0 := by
  /-
    i : Int
    ⊢ Eq (Int.lcm 0 i) 0
  -/
  rw [Int.lcm]
  /-
    i : Int
    ⊢ Eq ((Int.natAbs 0).lcm i.natAbs) 0
  -/
  apply Nat.lcm_zero_left
  /-
    🎉 no goals
  -/


@[simp]
theorem lcm_zero_right (i : ℤ) : lcm i 0 = 0 := by
  /-
    i : Int
    ⊢ Eq (i.lcm 0) 0
  -/
  rw [Int.lcm]
  /-
    i : Int
    ⊢ Eq (i.natAbs.lcm (Int.natAbs 0)) 0
  -/
  apply Nat.lcm_zero_right
  /-
    🎉 no goals
  -/


@[simp]
theorem lcm_one_left (i : ℤ) : lcm 1 i = natAbs i := by
  /-
    i : Int
    ⊢ Eq (Int.lcm 1 i) i.natAbs
  -/
  rw [Int.lcm]
  /-
    i : Int
    ⊢ Eq ((Int.natAbs 1).lcm i.natAbs) i.natAbs
  -/
  apply Nat.lcm_one_left
  /-
    🎉 no goals
  -/


@[simp]
theorem lcm_one_right (i : ℤ) : lcm i 1 = natAbs i := by
  /-
    i : Int
    ⊢ Eq (i.lcm 1) i.natAbs
  -/
  rw [Int.lcm]
  /-
    i : Int
    ⊢ Eq (i.natAbs.lcm (Int.natAbs 1)) i.natAbs
  -/
  apply Nat.lcm_one_right
  /-
    🎉 no goals
  -/


theorem lcm_dvd {i j k : ℤ} : i ∣ k → j ∣ k → (lcm i j : ℤ) ∣ k := by
  /-
    i j k : Int
    ⊢ Dvd.dvd i k → Dvd.dvd j k → Dvd.dvd (↑(i.lcm j)) k
  -/
  rw [Int.lcm]
  /-
    i j k : Int
    ⊢ Dvd.dvd i k → Dvd.dvd j k → Dvd.dvd (↑(i.natAbs.lcm j.natAbs)) k
  -/
  intro hi hj
  /-
    i j k : Int
    hi : Dvd.dvd i k
    hj : Dvd.dvd j k
    ⊢ Dvd.dvd (↑(i.natAbs.lcm j.natAbs)) k
  -/
  exact natCast_dvd.mpr (Nat.lcm_dvd (natAbs_dvd_natAbs.mpr hi) (natAbs_dvd_natAbs.mpr hj))
  /-
    🎉 no goals
  -/


theorem lcm_mul_left {m n k : ℤ} : (m * n).lcm (m * k) = natAbs m * n.lcm k := by
  /-
    m n k : Int
    ⊢ Eq ((HMul.hMul m n).lcm (HMul.hMul m k)) (HMul.hMul m.natAbs (n.lcm k))
  -/
  simp_rw [Int.lcm, natAbs_mul, Nat.lcm_mul_left]
  /-
    🎉 no goals
  -/


theorem lcm_mul_right {m n k : ℤ} : (m * n).lcm (k * n) = m.lcm k * natAbs n := by
  /-
    m n k : Int
    ⊢ Eq ((HMul.hMul m n).lcm (HMul.hMul k n)) (HMul.hMul (m.lcm k) n.natAbs)
  -/
  simp_rw [Int.lcm, natAbs_mul, Nat.lcm_mul_right]
  /-
    🎉 no goals
  -/


@[to_additive gcd_nsmul_eq_zero]
theorem pow_gcd_eq_one {M : Type*} [Monoid M] (x : M) {m n : ℕ} (hm : x ^ m = 1) (hn : x ^ n = 1) :
    x ^ m.gcd n = 1 := by
  /-
    M : Type u_1
    inst✝ : Monoid M
    x : M
    m n : Nat
    hm : Eq (HPow.hPow x m) 1
    hn : Eq (HPow.hPow x n) 1
    ⊢ Eq (HPow.hPow x (m.gcd n)) 1
  -/
  rcases m with (rfl | m); · simp [hn]
                             /-
                               🎉 no goals
                             -/
  /-
    case succ
    M : Type u_1
    inst✝ : Monoid M
    x : M
    n : Nat
    hn : Eq (HPow.hPow x n) 1
    m : Nat
    hm : Eq (HPow.hPow x (HAdd.hAdd m 1)) 1
    ⊢ Eq (HPow.hPow x ((HAdd.hAdd m 1).gcd n)) 1
  -/
  obtain ⟨y, rfl⟩ := isUnit_ofPowEqOne hm m.succ_ne_zero
  /-
    case succ.intro
    M : Type u_1
    inst✝ : Monoid M
    n m : Nat
    y : Units M
    hn : Eq (HPow.hPow (↑y) n) 1
    hm : Eq (HPow.hPow (↑y) (HAdd.hAdd m 1)) 1
    ⊢ Eq (HPow.hPow (↑y) ((HAdd.hAdd m 1).gcd n)) 1
  -/
  rw [← Units.val_pow_eq_pow_val, ← Units.val_one (α := M), ← zpow_natCast, ← Units.ext_iff] at *
  /-
    case succ.intro
    M : Type u_1
    inst✝ : Monoid M
    n m : Nat
    y : Units M
    hn : Eq (HPow.hPow y ↑n) 1
    hm : Eq (HPow.hPow y ↑(HAdd.hAdd m 1)) 1
    ⊢ Eq (HPow.hPow y ↑((HAdd.hAdd m 1).gcd n)) 1
  -/
  rw [Nat.gcd_eq_gcd_ab, zpow_add, zpow_mul, zpow_mul, hn, hm, one_zpow, one_zpow, one_mul]
  /-
    🎉 no goals
  -/


protected lemma Commute.pow_eq_pow_iff_of_coprime (hab : Commute a b) (hmn : m.Coprime n) :
    a ^ m = b ^ n ↔ ∃ c, a = c ^ n ∧ b = c ^ m := by
  /-
    α : Type u_1
    inst✝ : GroupWithZero α
    a b : α
    m n : Nat
    hab : Commute a b
    hmn : m.Coprime n
    ⊢ Iff (Eq (HPow.hPow a m) (HPow.hPow b n)) (Exists fun c => And (Eq a (HPow.hP …
  -/
  refine ⟨fun h ↦ ?_, by rintro ⟨c, rfl, rfl⟩; rw [← pow_mul, ← pow_mul']⟩
  /-
    α : Type u_1
    inst✝ : GroupWithZero α
    a b : α
    m n : Nat
    hab : Commute a b
    hmn : m.Coprime n
    h : Eq (HPow.hPow a m) (HPow.hPow b n)
    ⊢ Exists fun c => And (Eq a (HPow.hPow c n)) (Eq b (HPow.hPow c m))
  -/
  by_cases m = 0; · aesop
                    /-
                      🎉 no goals
                    -/
  /-
    case neg
    α : Type u_1
    inst✝ : GroupWithZero α
    a b : α
    m n : Nat
    hab : Commute a b
    hmn : m.Coprime n
    h : Eq (HPow.hPow a m) (HPow.hPow b n)
    h✝ : Not (Eq m 0)
    ⊢ Exists fun c => And (Eq a (HPow.hPow c n)) (Eq b (HPow.hPow c m))
  -/
  by_cases n = 0; · aesop
                    /-
                      🎉 no goals
                    -/
  /-
    case neg
    α : Type u_1
    inst✝ : GroupWithZero α
    a b : α
    m n : Nat
    hab : Commute a b
    hmn : m.Coprime n
    h : Eq (HPow.hPow a m) (HPow.hPow b n)
    h✝¹ : Not (Eq m 0)
    h✝ : Not (Eq n 0)
    ⊢ Exists fun c => And (Eq a (HPow.hPow c n)) (Eq b (HPow.hPow c m))
  -/
  by_cases hb : b = 0; · exact ⟨0, by aesop⟩
                         /-
                           🎉 no goals
                         -/
  /-
    case neg
    α : Type u_1
    inst✝ : GroupWithZero α
    a b : α
    m n : Nat
    hab : Commute a b
    hmn : m.Coprime n
    h : Eq (HPow.hPow a m) (HPow.hPow b n)
    h✝¹ : Not (Eq m 0)
    h✝ : Not (Eq n 0)
    hb : Not (Eq b 0)
    ⊢ Exists fun c => And (Eq a (HPow.hPow c n)) (Eq b (HPow.hPow c m))
  -/
  by_cases ha : a = 0; · exact ⟨0, by have := h.symm; aesop⟩
                         /-
                           🎉 no goals
                         -/
  /-
    case neg
    α : Type u_1
    inst✝ : GroupWithZero α
    a b : α
    m n : Nat
    hab : Commute a b
    hmn : m.Coprime n
    h : Eq (HPow.hPow a m) (HPow.hPow b n)
    h✝¹ : Not (Eq m 0)
    h✝ : Not (Eq n 0)
    hb : Not (Eq b 0)
    ha : Not (Eq a 0)
    ⊢ Exists fun c => And (Eq a (HPow.hPow c n)) (Eq b (HPow.hPow c m))
  -/
  refine ⟨a ^ Nat.gcdB m n * b ^ Nat.gcdA m n, ?_, ?_⟩ <;>
    /-
      case neg.refine_1
      α : Type u_1
      inst✝ : GroupWithZero α
      a b : α
      m n : Nat
      hab : Commute a b
      hmn : m.Coprime n
      h : Eq (HPow.hPow a m) (HPow.hPow b n)
      h✝¹ : Not (Eq m 0)
      h✝ : Not (Eq n 0)
      hb : Not (Eq b 0)
      ha : Not (Eq a 0)
      ⊢ Eq a (HPow.hPow (HMul.hMul (HPow.hPow a (m.gcdB n)) (HPow.hPow b (m.gcdA n)) …
    -/
    /-
      case neg.refine_1
      α : Type u_1
      inst✝ : GroupWithZero α
      a b : α
      m n : Nat
      hab : Commute a b
      hmn : m.Coprime n
      h : Eq (HPow.hPow a m) (HPow.hPow b n)
      h✝¹ : Not (Eq m 0)
      h✝ : Not (Eq n 0)
      hb : Not (Eq b 0)
      ha : Not (Eq a 0)
      ⊢ Eq (HPow.hPow a 1) (HPow.hPow (HMul.hMul (HPow.hPow a (m.gcdB n)) (HPow.hPow …
    -/
    /-
      case neg.refine_2
      α : Type u_1
      inst✝ : GroupWithZero α
      a b : α
      m n : Nat
      hab : Commute a b
      hmn : m.Coprime n
      h : Eq (HPow.hPow a m) (HPow.hPow b n)
      h✝¹ : Not (Eq m 0)
      h✝ : Not (Eq n 0)
      hb : Not (Eq b 0)
      ha : Not (Eq a 0)
      ⊢ Eq (HPow.hPow b 1) (HPow.hPow (HMul.hMul (HPow.hPow a (m.gcdB n)) (HPow.hPow …
    -/
    conv_lhs => rw [← zpow_natCast, ← hmn, Nat.gcd_eq_gcd_ab]
    /-
      case neg.refine_1
      α : Type u_1
      inst✝ : GroupWithZero α
      a b : α
      m n : Nat
      hab : Commute a b
      hmn : m.Coprime n
      h : Eq (HPow.hPow a m) (HPow.hPow b n)
      h✝¹ : Not (Eq m 0)
      h✝ : Not (Eq n 0)
      hb : Not (Eq b 0)
      ha : Not (Eq a 0)
      ⊢ Eq (HMul.hMul (HPow.hPow a (HMul.hMul (↑m) (m.gcdA n))) (HPow.hPow a (HMul.h …
    -/
    /-
      case neg.refine_1
      α : Type u_1
      inst✝ : GroupWithZero α
      a b : α
      m n : Nat
      hab : Commute a b
      hmn : m.Coprime n
      h : Eq (HPow.hPow a m) (HPow.hPow b n)
      h✝¹ : Not (Eq m 0)
      h✝ : Not (Eq n 0)
      hb : Not (Eq b 0)
      ha : Not (Eq a 0)
      ⊢ Eq (HMul.hMul (HPow.hPow (HPow.hPow b n) (m.gcdA n)) (HPow.hPow (HPow.hPow a …
    -/
    /-
      🎉 no goals
    -/
    /-
      case neg.refine_2
      α : Type u_1
      inst✝ : GroupWithZero α
      a b : α
      m n : Nat
      hab : Commute a b
      hmn : m.Coprime n
      h : Eq (HPow.hPow a m) (HPow.hPow b n)
      h✝¹ : Not (Eq m 0)
      h✝ : Not (Eq n 0)
      hb : Not (Eq b 0)
      ha : Not (Eq a 0)
      ⊢ Eq (HMul.hMul (HPow.hPow b (HMul.hMul (↑m) (m.gcdA n))) (HPow.hPow b (HMul.h …
    -/
    simp only [zpow_mul, zpow_natCast, h]
    /-
      case neg.refine_2
      α : Type u_1
      inst✝ : GroupWithZero α
      a b : α
      m n : Nat
      hab : Commute a b
      hmn : m.Coprime n
      h : Eq (HPow.hPow a m) (HPow.hPow b n)
      h✝¹ : Not (Eq m 0)
      h✝ : Not (Eq n 0)
      hb : Not (Eq b 0)
      ha : Not (Eq a 0)
      ⊢ Eq (HMul.hMul (HPow.hPow (HPow.hPow b m) (m.gcdA n)) (HPow.hPow (HPow.hPow b …
    -/
    exact ((Commute.pow_pow (by aesop) _ _).zpow_zpow₀ _ _).symm
    /-
      🎉 no goals
    -/


lemma pow_eq_pow_iff_of_coprime (hmn : m.Coprime n) : a ^ m = b ^ n ↔ ∃ c, a = c ^ n ∧ b = c ^ m :=
  (Commute.all _ _).pow_eq_pow_iff_of_coprime hmn


lemma pow_mem_range_pow_of_coprime (hmn : m.Coprime n) (a : α) :
    a ^ m ∈ Set.range (· ^ n : α → α) ↔ a ∈ Set.range (· ^ n : α → α) := by
  /-
    α : Type u_1
    inst✝ : CommGroupWithZero α
    m n : Nat
    hmn : m.Coprime n
    a : α
    ⊢ Iff (Membership.mem (Set.range fun x => HPow.hPow x n) (HPow.hPow a m)) (Mem …
  -/
  simp [pow_eq_pow_iff_of_coprime hmn.symm]; aesop
                                             /-
                                               🎉 no goals
                                             -/


