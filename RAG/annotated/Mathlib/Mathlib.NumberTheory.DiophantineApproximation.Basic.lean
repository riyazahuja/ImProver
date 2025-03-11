/-- *Dirichlet's approximation theorem:*
For any real number `ξ` and positive natural `n`, there are integers `j` and `k`,
with `0 < k ≤ n` and `|k*ξ - j| ≤ 1/(n+1)`.

See also `Real.exists_nat_abs_mul_sub_round_le`. -/
theorem exists_int_int_abs_mul_sub_le (ξ : ℝ) {n : ℕ} (n_pos : 0 < n) :
    ∃ j k : ℤ, 0 < k ∧ k ≤ n ∧ |↑k * ξ - j| ≤ 1 / (n + 1) := by
  /-
    ξ : Real
    n : Nat
    n_pos : LT.lt 0 n
    ⊢ Exists fun j => Exists fun k => And (LT.lt 0 k) (And (LE.le k ↑n) (LE.le (ab …
  -/
  let f : ℤ → ℤ := fun m => ⌊fract (ξ * m) * (n + 1)⌋
  /-
    ξ : Real
    n : Nat
    n_pos : LT.lt 0 n
    f : Int → Int := fun m => Int.floor (HMul.hMul (Int.fract (HMul.hMul ξ ↑m)) (H …
    ⊢ Exists fun j => Exists fun k => And (LT.lt 0 k) (And (LE.le k ↑n) (LE.le (ab …
  -/
  have hn : 0 < (n : ℝ) + 1 := mod_cast Nat.succ_pos _
  /-
    ξ : Real
    n : Nat
    n_pos : LT.lt 0 n
    f : Int → Int := fun m => Int.floor (HMul.hMul (Int.fract (HMul.hMul ξ ↑m)) (H …
    hn : LT.lt 0 (HAdd.hAdd (↑n) 1)
    ⊢ Exists fun j => Exists fun k => And (LT.lt 0 k) (And (LE.le k ↑n) (LE.le (ab …
  -/
  have hfu := fun m : ℤ => mul_lt_of_lt_one_left hn <| fract_lt_one (ξ * ↑m)
  /-
    ξ : Real
    n : Nat
    n_pos : LT.lt 0 n
    f : Int → Int := fun m => Int.floor (HMul.hMul (Int.fract (HMul.hMul ξ ↑m)) (H …
    hn : LT.lt 0 (HAdd.hAdd (↑n) 1)
    hfu : ∀ (m : Int), LT.lt (HMul.hMul (Int.fract (HMul.hMul ξ ↑m)) (HAdd.hAdd (↑ …
    ⊢ Exists fun j => Exists fun k => And (LT.lt 0 k) (And (LE.le k ↑n) (LE.le (ab …
  -/
  conv in |_| ≤ _ => rw [mul_comm, le_div_iff₀ hn, ← abs_of_pos hn, ← abs_mul]
  /-
    ξ : Real
    n : Nat
    n_pos : LT.lt 0 n
    f : Int → Int := fun m => Int.floor (HMul.hMul (Int.fract (HMul.hMul ξ ↑m)) (H …
    hn : LT.lt 0 (HAdd.hAdd (↑n) 1)
    hfu : ∀ (m : Int), LT.lt (HMul.hMul (Int.fract (HMul.hMul ξ ↑m)) (HAdd.hAdd (↑ …
    ⊢ Exists fun j => Exists fun k => And (LT.lt 0 k) (And (LE.le k ↑n) (LE.le (ab …
  -/
  let D := Icc (0 : ℤ) n
  /-
    ξ : Real
    n : Nat
    n_pos : LT.lt 0 n
    f : Int → Int := fun m => Int.floor (HMul.hMul (Int.fract (HMul.hMul ξ ↑m)) (H …
    hn : LT.lt 0 (HAdd.hAdd (↑n) 1)
    hfu : ∀ (m : Int), LT.lt (HMul.hMul (Int.fract (HMul.hMul ξ ↑m)) (HAdd.hAdd (↑ …
    D : Finset Int := Finset.Icc 0 ↑n
    ⊢ Exists fun j => Exists fun k => And (LT.lt 0 k) (And (LE.le k ↑n) (LE.le (ab …
  -/
  by_cases H : ∃ m ∈ D, f m = n
    /-
      case pos
      ξ : Real
      n : Nat
      n_pos : LT.lt 0 n
      f : Int → Int := fun m => Int.floor (HMul.hMul (Int.fract (HMul.hMul ξ ↑m)) (H …
      hn : LT.lt 0 (HAdd.hAdd (↑n) 1)
      hfu : ∀ (m : Int), LT.lt (HMul.hMul (Int.fract (HMul.hMul ξ ↑m)) (HAdd.hAdd (↑ …
      D : Finset Int := Finset.Icc 0 ↑n
      H : Exists fun m => And (Membership.mem D m) (Eq (f m) ↑n)
      ⊢ Exists fun j => Exists fun k => And (LT.lt 0 k) (And (LE.le k ↑n) (LE.le (ab …
    -/
  · obtain ⟨m, hm, hf⟩ := H
    /-
      case pos.intro.intro
      ξ : Real
      n : Nat
      n_pos : LT.lt 0 n
      f : Int → Int := fun m => Int.floor (HMul.hMul (Int.fract (HMul.hMul ξ ↑m)) (H …
      hn : LT.lt 0 (HAdd.hAdd (↑n) 1)
      hfu : ∀ (m : Int), LT.lt (HMul.hMul (Int.fract (HMul.hMul ξ ↑m)) (HAdd.hAdd (↑ …
      D : Finset Int := Finset.Icc 0 ↑n
      m : Int
      hm : Membership.mem D m
      hf : Eq (f m) ↑n
      ⊢ Exists fun j => Exists fun k => And (LT.lt 0 k) (And (LE.le k ↑n) (LE.le (ab …
    -/
    have hf' : ((n : ℤ) : ℝ) ≤ fract (ξ * m) * (n + 1) := hf ▸ floor_le (fract (ξ * m) * (n + 1))
    have hm₀ : 0 < m := by
      have hf₀ : f 0 = 0 := by
        -- Porting note: was
        -- simp only [floor_eq_zero_iff, algebraMap.coe_zero, mul_zero, fract_zero,
        --   zero_mul, Set.left_mem_Ico, zero_lt_one]
        simp only [f, cast_zero, mul_zero, fract_zero, zero_mul, floor_zero]
      refine Ne.lt_of_le (fun h => n_pos.ne ?_) (mem_Icc.mp hm).1
      exact mod_cast hf₀.symm.trans (h.symm ▸ hf : f 0 = n)
    /-
      case pos.intro.intro
      ξ : Real
      n : Nat
      n_pos : LT.lt 0 n
      f : Int → Int := fun m => Int.floor (HMul.hMul (Int.fract (HMul.hMul ξ ↑m)) (H …
      hn : LT.lt 0 (HAdd.hAdd (↑n) 1)
      hfu : ∀ (m : Int), LT.lt (HMul.hMul (Int.fract (HMul.hMul ξ ↑m)) (HAdd.hAdd (↑ …
      D : Finset Int := Finset.Icc 0 ↑n
      m : Int
      hm : Membership.mem D m
      hf : Eq (f m) ↑n
      hf' : LE.le (↑↑n) (HMul.hMul (Int.fract (HMul.hMul ξ ↑m)) (HAdd.hAdd (↑n) 1))
      hm₀ : LT.lt 0 m
      ⊢ Exists fun j => Exists fun k => And (LT.lt 0 k) (And (LE.le k ↑n) (LE.le (ab …
    -/
    refine ⟨⌊ξ * m⌋ + 1, m, hm₀, (mem_Icc.mp hm).2, ?_⟩
    /-
      case pos.intro.intro
      ξ : Real
      n : Nat
      n_pos : LT.lt 0 n
      f : Int → Int := fun m => Int.floor (HMul.hMul (Int.fract (HMul.hMul ξ ↑m)) (H …
      hn : LT.lt 0 (HAdd.hAdd (↑n) 1)
      hfu : ∀ (m : Int), LT.lt (HMul.hMul (Int.fract (HMul.hMul ξ ↑m)) (HAdd.hAdd (↑ …
      D : Finset Int := Finset.Icc 0 ↑n
      m : Int
      hm : Membership.mem D m
      hf : Eq (f m) ↑n
      hf' : LE.le (↑↑n) (HMul.hMul (Int.fract (HMul.hMul ξ ↑m)) (HAdd.hAdd (↑n) 1))
      hm₀ : LT.lt 0 m
      ⊢ LE.le (abs (HMul.hMul (HSub.hSub (HMul.hMul ξ ↑m) ↑(HAdd.hAdd (Int.floor (HM …
    -/
    rw [cast_add, ← sub_sub, sub_mul, cast_one, one_mul, abs_le]
    refine
      ⟨le_sub_iff_add_le.mpr ?_, sub_le_iff_le_add.mpr <| le_of_lt <| (hfu m).trans <| lt_one_add _⟩
    /-
      case pos.intro.intro
      ξ : Real
      n : Nat
      n_pos : LT.lt 0 n
      f : Int → Int := fun m => Int.floor (HMul.hMul (Int.fract (HMul.hMul ξ ↑m)) (H …
      hn : LT.lt 0 (HAdd.hAdd (↑n) 1)
      hfu : ∀ (m : Int), LT.lt (HMul.hMul (Int.fract (HMul.hMul ξ ↑m)) (HAdd.hAdd (↑ …
      D : Finset Int := Finset.Icc 0 ↑n
      m : Int
      hm : Membership.mem D m
      hf : Eq (f m) ↑n
      hf' : LE.le (↑↑n) (HMul.hMul (Int.fract (HMul.hMul ξ ↑m)) (HAdd.hAdd (↑n) 1))
      hm₀ : LT.lt 0 m
      ⊢ LE.le (HAdd.hAdd (-1) (HAdd.hAdd (↑n) 1)) (HMul.hMul (HSub.hSub (HMul.hMul ξ …
    -/
    simpa only [neg_add_cancel_comm_assoc] using hf'
    /-
      🎉 no goals
    -/
  · -- Porting note(https://github.com/leanprover-community/mathlib4/issues/5127): added `not_and`
    /-
      case neg
      ξ : Real
      n : Nat
      n_pos : LT.lt 0 n
      f : Int → Int := fun m => Int.floor (HMul.hMul (Int.fract (HMul.hMul ξ ↑m)) (H …
      hn : LT.lt 0 (HAdd.hAdd (↑n) 1)
      hfu : ∀ (m : Int), LT.lt (HMul.hMul (Int.fract (HMul.hMul ξ ↑m)) (HAdd.hAdd (↑ …
      D : Finset Int := Finset.Icc 0 ↑n
      H : Not (Exists fun m => And (Membership.mem D m) (Eq (f m) ↑n))
      ⊢ Exists fun j => Exists fun k => And (LT.lt 0 k) (And (LE.le k ↑n) (LE.le (ab …
    -/
    simp_rw [not_exists, not_and] at H
    /-
      case neg
      ξ : Real
      n : Nat
      n_pos : LT.lt 0 n
      f : Int → Int := fun m => Int.floor (HMul.hMul (Int.fract (HMul.hMul ξ ↑m)) (H …
      hn : LT.lt 0 (HAdd.hAdd (↑n) 1)
      hfu : ∀ (m : Int), LT.lt (HMul.hMul (Int.fract (HMul.hMul ξ ↑m)) (HAdd.hAdd (↑ …
      D : Finset Int := Finset.Icc 0 ↑n
      H : ∀ (x : Int), Membership.mem D x → Not (Eq (f x) ↑n)
      ⊢ Exists fun j => Exists fun k => And (LT.lt 0 k) (And (LE.le k ↑n) (LE.le (ab …
    -/
    have hD : #(Ico (0 : ℤ) n) < #D := by rw [card_Icc, card_Ico]; exact lt_add_one n
    /-
      case neg
      ξ : Real
      n : Nat
      n_pos : LT.lt 0 n
      f : Int → Int := fun m => Int.floor (HMul.hMul (Int.fract (HMul.hMul ξ ↑m)) (H …
      hn : LT.lt 0 (HAdd.hAdd (↑n) 1)
      hfu : ∀ (m : Int), LT.lt (HMul.hMul (Int.fract (HMul.hMul ξ ↑m)) (HAdd.hAdd (↑ …
      D : Finset Int := Finset.Icc 0 ↑n
      H : ∀ (x : Int), Membership.mem D x → Not (Eq (f x) ↑n)
      hD : LT.lt (Finset.Ico 0 ↑n).card D.card
      ⊢ Exists fun j => Exists fun k => And (LT.lt 0 k) (And (LE.le k ↑n) (LE.le (ab …
    -/
    have hfu' : ∀ m, f m ≤ n := fun m => lt_add_one_iff.mp (floor_lt.mpr (mod_cast hfu m))
    have hwd : ∀ m : ℤ, m ∈ D → f m ∈ Ico (0 : ℤ) n := fun x hx =>
      mem_Ico.mpr
        ⟨floor_nonneg.mpr (mul_nonneg (fract_nonneg (ξ * x)) hn.le), Ne.lt_of_le (H x hx) (hfu' x)⟩
    obtain ⟨x, hx, y, hy, x_lt_y, hxy⟩ : ∃ x ∈ D, ∃ y ∈ D, x < y ∧ f x = f y := by
      obtain ⟨x, hx, y, hy, x_ne_y, hxy⟩ := exists_ne_map_eq_of_card_lt_of_maps_to hD hwd
      rcases lt_trichotomy x y with (h | h | h)
      exacts [⟨x, hx, y, hy, h, hxy⟩, False.elim (x_ne_y h), ⟨y, hy, x, hx, h, hxy.symm⟩]
    refine
      ⟨⌊ξ * y⌋ - ⌊ξ * x⌋, y - x, sub_pos_of_lt x_lt_y,
        sub_le_iff_le_add.mpr <| le_add_of_le_of_nonneg (mem_Icc.mp hy).2 (mem_Icc.mp hx).1, ?_⟩
    /-
      case neg.intro.intro.intro.intro.intro
      ξ : Real
      n : Nat
      n_pos : LT.lt 0 n
      f : Int → Int := fun m => Int.floor (HMul.hMul (Int.fract (HMul.hMul ξ ↑m)) (H …
      hn : LT.lt 0 (HAdd.hAdd (↑n) 1)
      hfu : ∀ (m : Int), LT.lt (HMul.hMul (Int.fract (HMul.hMul ξ ↑m)) (HAdd.hAdd (↑ …
      D : Finset Int := Finset.Icc 0 ↑n
      H : ∀ (x : Int), Membership.mem D x → Not (Eq (f x) ↑n)
      hD : LT.lt (Finset.Ico 0 ↑n).card D.card
      hfu' : ∀ (m : Int), LE.le (f m) ↑n
      hwd : ∀ (m : Int), Membership.mem D m → Membership.mem (Finset.Ico 0 ↑n) (f m)
      x : Int
      hx : Membership.mem D x
      y : Int
      hy : Membership.mem D y
      x_lt_y : LT.lt x y
      hxy : Eq (f x) (f y)
      ⊢ LE.le (abs (HMul.hMul (HSub.hSub (HMul.hMul ξ ↑(HSub.hSub y x)) ↑(HSub.hSub  …
    -/
    convert_to |fract (ξ * y) * (n + 1) - fract (ξ * x) * (n + 1)| ≤ 1
      /-
        case h.e'_3
        ξ : Real
        n : Nat
        n_pos : LT.lt 0 n
        f : Int → Int := fun m => Int.floor (HMul.hMul (Int.fract (HMul.hMul ξ ↑m)) (H …
        hn : LT.lt 0 (HAdd.hAdd (↑n) 1)
        hfu : ∀ (m : Int), LT.lt (HMul.hMul (Int.fract (HMul.hMul ξ ↑m)) (HAdd.hAdd (↑ …
        D : Finset Int := Finset.Icc 0 ↑n
        H : ∀ (x : Int), Membership.mem D x → Not (Eq (f x) ↑n)
        hD : LT.lt (Finset.Ico 0 ↑n).card D.card
        hfu' : ∀ (m : Int), LE.le (f m) ↑n
        hwd : ∀ (m : Int), Membership.mem D m → Membership.mem (Finset.Ico 0 ↑n) (f m)
        x : Int
        hx : Membership.mem D x
        y : Int
        hy : Membership.mem D y
        x_lt_y : LT.lt x y
        hxy : Eq (f x) (f y)
        ⊢ Eq (abs (HMul.hMul (HSub.hSub (HMul.hMul ξ ↑(HSub.hSub y x)) ↑(HSub.hSub (In …
      -/
    · congr; push_cast; simp only [fract]; ring
                                           /-
                                             🎉 no goals
                                           -/
    /-
      case neg.intro.intro.intro.intro.intro
      ξ : Real
      n : Nat
      n_pos : LT.lt 0 n
      f : Int → Int := fun m => Int.floor (HMul.hMul (Int.fract (HMul.hMul ξ ↑m)) (H …
      hn : LT.lt 0 (HAdd.hAdd (↑n) 1)
      hfu : ∀ (m : Int), LT.lt (HMul.hMul (Int.fract (HMul.hMul ξ ↑m)) (HAdd.hAdd (↑ …
      D : Finset Int := Finset.Icc 0 ↑n
      H : ∀ (x : Int), Membership.mem D x → Not (Eq (f x) ↑n)
      hD : LT.lt (Finset.Ico 0 ↑n).card D.card
      hfu' : ∀ (m : Int), LE.le (f m) ↑n
      hwd : ∀ (m : Int), Membership.mem D m → Membership.mem (Finset.Ico 0 ↑n) (f m)
      x : Int
      hx : Membership.mem D x
      y : Int
      hy : Membership.mem D y
      x_lt_y : LT.lt x y
      hxy : Eq (f x) (f y)
      ⊢ LE.le (abs (HSub.hSub (HMul.hMul (Int.fract (HMul.hMul ξ ↑y)) (HAdd.hAdd (↑n …
    -/
    exact (abs_sub_lt_one_of_floor_eq_floor hxy.symm).le
    /-
      🎉 no goals
    -/


/-- *Dirichlet's approximation theorem:*
For any real number `ξ` and positive natural `n`, there is a natural number `k`,
with `0 < k ≤ n` such that `|k*ξ - round(k*ξ)| ≤ 1/(n+1)`.
-/
theorem exists_nat_abs_mul_sub_round_le (ξ : ℝ) {n : ℕ} (n_pos : 0 < n) :
    ∃ k : ℕ, 0 < k ∧ k ≤ n ∧ |↑k * ξ - round (↑k * ξ)| ≤ 1 / (n + 1) := by
  /-
    ξ : Real
    n : Nat
    n_pos : LT.lt 0 n
    ⊢ Exists fun k => And (LT.lt 0 k) (And (LE.le k n) (LE.le (abs (HSub.hSub (HMu …
  -/
  obtain ⟨j, k, hk₀, hk₁, h⟩ := exists_int_int_abs_mul_sub_le ξ n_pos
  /-
    case intro.intro.intro.intro
    ξ : Real
    n : Nat
    n_pos : LT.lt 0 n
    j k : Int
    hk₀ : LT.lt 0 k
    hk₁ : LE.le k ↑n
    h : LE.le (abs (HSub.hSub (HMul.hMul (↑k) ξ) ↑j)) (HDiv.hDiv 1 (HAdd.hAdd (↑n) …
    ⊢ Exists fun k => And (LT.lt 0 k) (And (LE.le k n) (LE.le (abs (HSub.hSub (HMu …
  -/
  have hk := toNat_of_nonneg hk₀.le
  /-
    case intro.intro.intro.intro
    ξ : Real
    n : Nat
    n_pos : LT.lt 0 n
    j k : Int
    hk₀ : LT.lt 0 k
    hk₁ : LE.le k ↑n
    h : LE.le (abs (HSub.hSub (HMul.hMul (↑k) ξ) ↑j)) (HDiv.hDiv 1 (HAdd.hAdd (↑n) …
    hk : Eq (↑k.toNat) k
    ⊢ Exists fun k => And (LT.lt 0 k) (And (LE.le k n) (LE.le (abs (HSub.hSub (HMu …
  -/
  rw [← hk] at hk₀ hk₁ h
  /-
    case intro.intro.intro.intro
    ξ : Real
    n : Nat
    n_pos : LT.lt 0 n
    j k : Int
    hk₀ : LT.lt 0 ↑k.toNat
    hk₁ : LE.le ↑k.toNat ↑n
    h : LE.le (abs (HSub.hSub (HMul.hMul (↑↑k.toNat) ξ) ↑j)) (HDiv.hDiv 1 (HAdd.hA …
    hk : Eq (↑k.toNat) k
    ⊢ Exists fun k => And (LT.lt 0 k) (And (LE.le k n) (LE.le (abs (HSub.hSub (HMu …
  -/
  exact ⟨k.toNat, natCast_pos.mp hk₀, Nat.cast_le.mp hk₁, (round_le (↑k.toNat * ξ) j).trans h⟩
  /-
    🎉 no goals
  -/


/-- *Dirichlet's approximation theorem:*
For any real number `ξ` and positive natural `n`, there is a fraction `q`
such that `q.den ≤ n` and `|ξ - q| ≤ 1/((n+1)*q.den)`.

See also `AddCircle.exists_norm_nsmul_le`. -/
theorem exists_rat_abs_sub_le_and_den_le (ξ : ℝ) {n : ℕ} (n_pos : 0 < n) :
    ∃ q : ℚ, |ξ - q| ≤ 1 / ((n + 1) * q.den) ∧ q.den ≤ n := by
  /-
    ξ : Real
    n : Nat
    n_pos : LT.lt 0 n
    ⊢ Exists fun q => And (LE.le (abs (HSub.hSub ξ ↑q)) (HDiv.hDiv 1 (HMul.hMul (H …
  -/
  obtain ⟨j, k, hk₀, hk₁, h⟩ := exists_int_int_abs_mul_sub_le ξ n_pos
  /-
    case intro.intro.intro.intro
    ξ : Real
    n : Nat
    n_pos : LT.lt 0 n
    j k : Int
    hk₀ : LT.lt 0 k
    hk₁ : LE.le k ↑n
    h : LE.le (abs (HSub.hSub (HMul.hMul (↑k) ξ) ↑j)) (HDiv.hDiv 1 (HAdd.hAdd (↑n) …
    ⊢ Exists fun q => And (LE.le (abs (HSub.hSub ξ ↑q)) (HDiv.hDiv 1 (HMul.hMul (H …
  -/
  have hk₀' : (0 : ℝ) < k := Int.cast_pos.mpr hk₀
  have hden : ((j / k : ℚ).den : ℤ) ≤ k := by
    convert le_of_dvd hk₀ (Rat.den_dvd j k)
    exact Rat.intCast_div_eq_divInt _ _
  /-
    case intro.intro.intro.intro
    ξ : Real
    n : Nat
    n_pos : LT.lt 0 n
    j k : Int
    hk₀ : LT.lt 0 k
    hk₁ : LE.le k ↑n
    h : LE.le (abs (HSub.hSub (HMul.hMul (↑k) ξ) ↑j)) (HDiv.hDiv 1 (HAdd.hAdd (↑n) …
    hk₀' : LT.lt 0 ↑k
    hden : LE.le (↑(HDiv.hDiv ↑j ↑k).den) k
    ⊢ Exists fun q => And (LE.le (abs (HSub.hSub ξ ↑q)) (HDiv.hDiv 1 (HMul.hMul (H …
  -/
  refine ⟨j / k, ?_, Nat.cast_le.mp (hden.trans hk₁)⟩
  /-
    case intro.intro.intro.intro
    ξ : Real
    n : Nat
    n_pos : LT.lt 0 n
    j k : Int
    hk₀ : LT.lt 0 k
    hk₁ : LE.le k ↑n
    h : LE.le (abs (HSub.hSub (HMul.hMul (↑k) ξ) ↑j)) (HDiv.hDiv 1 (HAdd.hAdd (↑n) …
    hk₀' : LT.lt 0 ↑k
    hden : LE.le (↑(HDiv.hDiv ↑j ↑k).den) k
    ⊢ LE.le (abs (HSub.hSub ξ ↑(HDiv.hDiv ↑j ↑k))) (HDiv.hDiv 1 (HMul.hMul (HAdd.h …
  -/
  rw [← div_div, le_div_iff₀ (Nat.cast_pos.mpr <| Rat.pos _ : (0 : ℝ) < _)]
  /-
    case intro.intro.intro.intro
    ξ : Real
    n : Nat
    n_pos : LT.lt 0 n
    j k : Int
    hk₀ : LT.lt 0 k
    hk₁ : LE.le k ↑n
    h : LE.le (abs (HSub.hSub (HMul.hMul (↑k) ξ) ↑j)) (HDiv.hDiv 1 (HAdd.hAdd (↑n) …
    hk₀' : LT.lt 0 ↑k
    hden : LE.le (↑(HDiv.hDiv ↑j ↑k).den) k
    ⊢ LE.le (HMul.hMul (abs (HSub.hSub ξ ↑(HDiv.hDiv ↑j ↑k))) ↑(HDiv.hDiv ↑j ↑k).d …
  -/
  refine (mul_le_mul_of_nonneg_left (Int.cast_le.mpr hden : _ ≤ (k : ℝ)) (abs_nonneg _)).trans ?_
  rwa [← abs_of_pos hk₀', Rat.cast_div, Rat.cast_intCast, Rat.cast_intCast, ← abs_mul, sub_mul,
    div_mul_cancel₀ _ hk₀'.ne', mul_comm]


/-- Given any rational approximation `q` to the irrational real number `ξ`, there is
a good rational approximation `q'` such that `|ξ - q'| < |ξ - q|`. -/
theorem exists_rat_abs_sub_lt_and_lt_of_irrational {ξ : ℝ} (hξ : Irrational ξ) (q : ℚ) :
    ∃ q' : ℚ, |ξ - q'| < 1 / (q'.den : ℝ) ^ 2 ∧ |ξ - q'| < |ξ - q| := by
  /-
    ξ : Real
    hξ : Irrational ξ
    q : Rat
    ⊢ Exists fun q' => And (LT.lt (abs (HSub.hSub ξ ↑q')) (HDiv.hDiv 1 (HPow.hPow  …
  -/
  have h := abs_pos.mpr (sub_ne_zero.mpr <| Irrational.ne_rat hξ q)
  /-
    ξ : Real
    hξ : Irrational ξ
    q : Rat
    h : LT.lt 0 (abs (HSub.hSub ξ ↑q))
    ⊢ Exists fun q' => And (LT.lt (abs (HSub.hSub ξ ↑q')) (HDiv.hDiv 1 (HPow.hPow  …
  -/
  obtain ⟨m, hm⟩ := exists_nat_gt (1 / |ξ - q|)
  /-
    case intro
    ξ : Real
    hξ : Irrational ξ
    q : Rat
    h : LT.lt 0 (abs (HSub.hSub ξ ↑q))
    m : Nat
    hm : LT.lt (HDiv.hDiv 1 (abs (HSub.hSub ξ ↑q))) ↑m
    ⊢ Exists fun q' => And (LT.lt (abs (HSub.hSub ξ ↑q')) (HDiv.hDiv 1 (HPow.hPow  …
  -/
  have m_pos : (0 : ℝ) < m := (one_div_pos.mpr h).trans hm
  /-
    case intro
    ξ : Real
    hξ : Irrational ξ
    q : Rat
    h : LT.lt 0 (abs (HSub.hSub ξ ↑q))
    m : Nat
    hm : LT.lt (HDiv.hDiv 1 (abs (HSub.hSub ξ ↑q))) ↑m
    m_pos : LT.lt 0 ↑m
    ⊢ Exists fun q' => And (LT.lt (abs (HSub.hSub ξ ↑q')) (HDiv.hDiv 1 (HPow.hPow  …
  -/
  obtain ⟨q', hbd, hden⟩ := exists_rat_abs_sub_le_and_den_le ξ (Nat.cast_pos.mp m_pos)
  /-
    case intro.intro.intro
    ξ : Real
    hξ : Irrational ξ
    q : Rat
    h : LT.lt 0 (abs (HSub.hSub ξ ↑q))
    m : Nat
    hm : LT.lt (HDiv.hDiv 1 (abs (HSub.hSub ξ ↑q))) ↑m
    m_pos : LT.lt 0 ↑m
    q' : Rat
    hbd : LE.le (abs (HSub.hSub ξ ↑q')) (HDiv.hDiv 1 (HMul.hMul (HAdd.hAdd (↑m) 1) …
    hden : LE.le q'.den m
    ⊢ Exists fun q' => And (LT.lt (abs (HSub.hSub ξ ↑q')) (HDiv.hDiv 1 (HPow.hPow  …
  -/
  have den_pos : (0 : ℝ) < q'.den := Nat.cast_pos.mpr q'.pos
  /-
    case intro.intro.intro
    ξ : Real
    hξ : Irrational ξ
    q : Rat
    h : LT.lt 0 (abs (HSub.hSub ξ ↑q))
    m : Nat
    hm : LT.lt (HDiv.hDiv 1 (abs (HSub.hSub ξ ↑q))) ↑m
    m_pos : LT.lt 0 ↑m
    q' : Rat
    hbd : LE.le (abs (HSub.hSub ξ ↑q')) (HDiv.hDiv 1 (HMul.hMul (HAdd.hAdd (↑m) 1) …
    hden : LE.le q'.den m
    den_pos : LT.lt 0 ↑q'.den
    ⊢ Exists fun q' => And (LT.lt (abs (HSub.hSub ξ ↑q')) (HDiv.hDiv 1 (HPow.hPow  …
  -/
  have md_pos := mul_pos (add_pos m_pos zero_lt_one) den_pos
  refine
    ⟨q', lt_of_le_of_lt hbd ?_,
      lt_of_le_of_lt hbd <|
        (one_div_lt md_pos h).mpr <|
          hm.trans <|
            lt_of_lt_of_le (lt_add_one _) <|
              (le_mul_iff_one_le_right <| add_pos m_pos zero_lt_one).mpr <|
                mod_cast (q'.pos : 1 ≤ q'.den)⟩
  /-
    case intro.intro.intro
    ξ : Real
    hξ : Irrational ξ
    q : Rat
    h : LT.lt 0 (abs (HSub.hSub ξ ↑q))
    m : Nat
    hm : LT.lt (HDiv.hDiv 1 (abs (HSub.hSub ξ ↑q))) ↑m
    m_pos : LT.lt 0 ↑m
    q' : Rat
    hbd : LE.le (abs (HSub.hSub ξ ↑q')) (HDiv.hDiv 1 (HMul.hMul (HAdd.hAdd (↑m) 1) …
    hden : LE.le q'.den m
    den_pos : LT.lt 0 ↑q'.den
    md_pos : LT.lt 0 (HMul.hMul (HAdd.hAdd (↑m) 1) ↑q'.den)
    ⊢ LT.lt (HDiv.hDiv 1 (HMul.hMul (HAdd.hAdd (↑m) 1) ↑q'.den)) (HDiv.hDiv 1 (HPo …
  -/
  rw [sq, one_div_lt_one_div md_pos (mul_pos den_pos den_pos), mul_lt_mul_right den_pos]
  /-
    case intro.intro.intro
    ξ : Real
    hξ : Irrational ξ
    q : Rat
    h : LT.lt 0 (abs (HSub.hSub ξ ↑q))
    m : Nat
    hm : LT.lt (HDiv.hDiv 1 (abs (HSub.hSub ξ ↑q))) ↑m
    m_pos : LT.lt 0 ↑m
    q' : Rat
    hbd : LE.le (abs (HSub.hSub ξ ↑q')) (HDiv.hDiv 1 (HMul.hMul (HAdd.hAdd (↑m) 1) …
    hden : LE.le q'.den m
    den_pos : LT.lt 0 ↑q'.den
    md_pos : LT.lt 0 (HMul.hMul (HAdd.hAdd (↑m) 1) ↑q'.den)
    ⊢ LT.lt (↑q'.den) (HAdd.hAdd (↑m) 1)
  -/
  exact lt_add_of_le_of_pos (Nat.cast_le.mpr hden) zero_lt_one
  /-
    🎉 no goals
  -/


/-- If `ξ` is an irrational real number, then there are infinitely many good
rational approximations to `ξ`. -/
theorem infinite_rat_abs_sub_lt_one_div_den_sq_of_irrational {ξ : ℝ} (hξ : Irrational ξ) :
    {q : ℚ | |ξ - q| < 1 / (q.den : ℝ) ^ 2}.Infinite := by
  /-
    ξ : Real
    hξ : Irrational ξ
    ⊢ (setOf fun q => LT.lt (abs (HSub.hSub ξ ↑q)) (HDiv.hDiv 1 (HPow.hPow (↑q.den …
  -/
  refine Or.resolve_left (Set.finite_or_infinite _) fun h => ?_
  obtain ⟨q, _, hq⟩ :=
    exists_min_image {q : ℚ | |ξ - q| < 1 / (q.den : ℝ) ^ 2} (fun q => |ξ - q|) h
      ⟨⌊ξ⌋, by simp [abs_of_nonneg, Int.fract_lt_one]⟩
  /-
    case intro.intro
    ξ : Real
    hξ : Irrational ξ
    h : (setOf fun q => LT.lt (abs (HSub.hSub ξ ↑q)) (HDiv.hDiv 1 (HPow.hPow (↑q.d …
    q : Rat
    left✝ : Membership.mem (setOf fun q => LT.lt (abs (HSub.hSub ξ ↑q)) (HDiv.hDiv …
    hq : ∀ (b : Rat), Membership.mem (setOf fun q => LT.lt (abs (HSub.hSub ξ ↑q))  …
    ⊢ False
  -/
  obtain ⟨q', hmem, hbetter⟩ := exists_rat_abs_sub_lt_and_lt_of_irrational hξ q
  /-
    case intro.intro.intro.intro
    ξ : Real
    hξ : Irrational ξ
    h : (setOf fun q => LT.lt (abs (HSub.hSub ξ ↑q)) (HDiv.hDiv 1 (HPow.hPow (↑q.d …
    q : Rat
    left✝ : Membership.mem (setOf fun q => LT.lt (abs (HSub.hSub ξ ↑q)) (HDiv.hDiv …
    hq : ∀ (b : Rat), Membership.mem (setOf fun q => LT.lt (abs (HSub.hSub ξ ↑q))  …
    q' : Rat
    hmem : LT.lt (abs (HSub.hSub ξ ↑q')) (HDiv.hDiv 1 (HPow.hPow (↑q'.den) 2))
    hbetter : LT.lt (abs (HSub.hSub ξ ↑q')) (abs (HSub.hSub ξ ↑q))
    ⊢ False
  -/
  exact lt_irrefl _ (lt_of_le_of_lt (hq q' hmem) hbetter)
  /-
    🎉 no goals
  -/


/-- If `ξ` is rational, then the good rational approximations to `ξ` have bounded
numerator and denominator. -/
theorem den_le_and_le_num_le_of_sub_lt_one_div_den_sq {ξ q : ℚ}
    (h : |ξ - q| < 1 / (q.den : ℚ) ^ 2) :
    q.den ≤ ξ.den ∧ ⌈ξ * q.den⌉ - 1 ≤ q.num ∧ q.num ≤ ⌊ξ * q.den⌋ + 1 := by
  /-
    ξ q : Rat
    h : LT.lt (abs (HSub.hSub ξ q)) (HDiv.hDiv 1 (HPow.hPow (↑q.den) 2))
    ⊢ And (LE.le q.den ξ.den) (And (LE.le (HSub.hSub (Int.ceil (HMul.hMul ξ ↑q.den …
  -/
  have hq₀ : (0 : ℚ) < q.den := Nat.cast_pos.mpr q.pos
  replace h : |ξ * q.den - q.num| < 1 / q.den := by
    rw [← mul_lt_mul_right hq₀] at h
    conv_lhs at h => rw [← abs_of_pos hq₀, ← abs_mul, sub_mul, mul_den_eq_num]
    rwa [sq, div_mul, mul_div_cancel_left₀ _ hq₀.ne'] at h
  /-
    ξ q : Rat
    hq₀ : LT.lt 0 ↑q.den
    h : LT.lt (abs (HSub.hSub (HMul.hMul ξ ↑q.den) ↑q.num)) (HDiv.hDiv 1 ↑q.den)
    ⊢ And (LE.le q.den ξ.den) (And (LE.le (HSub.hSub (Int.ceil (HMul.hMul ξ ↑q.den …
  -/
  constructor
    /-
      case left
      ξ q : Rat
      hq₀ : LT.lt 0 ↑q.den
      h : LT.lt (abs (HSub.hSub (HMul.hMul ξ ↑q.den) ↑q.num)) (HDiv.hDiv 1 ↑q.den)
      ⊢ LE.le q.den ξ.den
    -/
  · rcases eq_or_ne ξ q with (rfl | H)
      /-
        case left.inl
        ξ : Rat
        hq₀ : LT.lt 0 ↑ξ.den
        h : LT.lt (abs (HSub.hSub (HMul.hMul ξ ↑ξ.den) ↑ξ.num)) (HDiv.hDiv 1 ↑ξ.den)
        ⊢ LE.le ξ.den ξ.den
      -/
    · exact le_rfl
      /-
        🎉 no goals
      -/
      /-
        case left.inr
        ξ q : Rat
        hq₀ : LT.lt 0 ↑q.den
        h : LT.lt (abs (HSub.hSub (HMul.hMul ξ ↑q.den) ↑q.num)) (HDiv.hDiv 1 ↑q.den)
        H : Ne ξ q
        ⊢ LE.le q.den ξ.den
      -/
    · have hξ₀ : (0 : ℚ) < ξ.den := Nat.cast_pos.mpr ξ.pos
      rw [← Rat.num_div_den ξ, div_mul_eq_mul_div, div_sub' _ _ _ hξ₀.ne', abs_div, abs_of_pos hξ₀,
        div_lt_iff₀ hξ₀, div_mul_comm, mul_one] at h
      /-
        case left.inr
        ξ q : Rat
        hq₀ : LT.lt 0 ↑q.den
        h : LT.lt (abs (HSub.hSub (HMul.hMul ↑ξ.num ↑q.den) (HMul.hMul ↑ξ.den ↑q.num)) …
        H : Ne ξ q
        hξ₀ : LT.lt 0 ↑ξ.den
        ⊢ LE.le q.den ξ.den
      -/
      refine Nat.cast_le.mp ((one_lt_div hq₀).mp <| lt_of_le_of_lt ?_ h).le
      /-
        case left.inr
        ξ q : Rat
        hq₀ : LT.lt 0 ↑q.den
        h : LT.lt (abs (HSub.hSub (HMul.hMul ↑ξ.num ↑q.den) (HMul.hMul ↑ξ.den ↑q.num)) …
        H : Ne ξ q
        hξ₀ : LT.lt 0 ↑ξ.den
        ⊢ LE.le 1 (abs (HSub.hSub (HMul.hMul ↑ξ.num ↑q.den) (HMul.hMul ↑ξ.den ↑q.num)))
      -/
      norm_cast
      /-
        case left.inr
        ξ q : Rat
        hq₀ : LT.lt 0 ↑q.den
        h : LT.lt (abs (HSub.hSub (HMul.hMul ↑ξ.num ↑q.den) (HMul.hMul ↑ξ.den ↑q.num)) …
        H : Ne ξ q
        hξ₀ : LT.lt 0 ↑ξ.den
        ⊢ LE.le 1 (abs (HSub.hSub (HMul.hMul ξ.num ↑q.den) (HMul.hMul (↑ξ.den) q.num)))
      -/
      rw [mul_comm _ q.num]
      /-
        case left.inr
        ξ q : Rat
        hq₀ : LT.lt 0 ↑q.den
        h : LT.lt (abs (HSub.hSub (HMul.hMul ↑ξ.num ↑q.den) (HMul.hMul ↑ξ.den ↑q.num)) …
        H : Ne ξ q
        hξ₀ : LT.lt 0 ↑ξ.den
        ⊢ LE.le 1 (abs (HSub.hSub (HMul.hMul ξ.num ↑q.den) (HMul.hMul q.num ↑ξ.den)))
      -/
      exact Int.one_le_abs (sub_ne_zero_of_ne <| mt Rat.eq_iff_mul_eq_mul.mpr H)
      /-
        🎉 no goals
      -/
  · obtain ⟨h₁, h₂⟩ :=
      abs_sub_lt_iff.mp
        (h.trans_le <|
          (one_div_le zero_lt_one hq₀).mp <| (@one_div_one ℚ _).symm ▸ Nat.cast_le.mpr q.pos)
    /-
      case right.intro
      ξ q : Rat
      hq₀ : LT.lt 0 ↑q.den
      h : LT.lt (abs (HSub.hSub (HMul.hMul ξ ↑q.den) ↑q.num)) (HDiv.hDiv 1 ↑q.den)
      h₁ : LT.lt (HSub.hSub (HMul.hMul ξ ↑q.den) ↑q.num) 1
      h₂ : LT.lt (HSub.hSub (↑q.num) (HMul.hMul ξ ↑q.den)) 1
      ⊢ And (LE.le (HSub.hSub (Int.ceil (HMul.hMul ξ ↑q.den)) 1) q.num) (LE.le q.num …
    -/
    rw [sub_lt_iff_lt_add, add_comm] at h₁ h₂
    /-
      case right.intro
      ξ q : Rat
      hq₀ : LT.lt 0 ↑q.den
      h : LT.lt (abs (HSub.hSub (HMul.hMul ξ ↑q.den) ↑q.num)) (HDiv.hDiv 1 ↑q.den)
      h₁ : LT.lt (HMul.hMul ξ ↑q.den) (HAdd.hAdd (↑q.num) 1)
      h₂ : LT.lt (↑q.num) (HAdd.hAdd (HMul.hMul ξ ↑q.den) 1)
      ⊢ And (LE.le (HSub.hSub (Int.ceil (HMul.hMul ξ ↑q.den)) 1) q.num) (LE.le q.num …
    -/
    rw [← sub_lt_iff_lt_add] at h₂
    /-
      case right.intro
      ξ q : Rat
      hq₀ : LT.lt 0 ↑q.den
      h : LT.lt (abs (HSub.hSub (HMul.hMul ξ ↑q.den) ↑q.num)) (HDiv.hDiv 1 ↑q.den)
      h₁ : LT.lt (HMul.hMul ξ ↑q.den) (HAdd.hAdd (↑q.num) 1)
      h₂ : LT.lt (HSub.hSub (↑q.num) 1) (HMul.hMul ξ ↑q.den)
      ⊢ And (LE.le (HSub.hSub (Int.ceil (HMul.hMul ξ ↑q.den)) 1) q.num) (LE.le q.num …
    -/
    norm_cast at h₁ h₂
    exact
      ⟨sub_le_iff_le_add.mpr (Int.ceil_le.mpr h₁.le), sub_le_iff_le_add.mp (Int.le_floor.mpr h₂.le)⟩


/-- A rational number has only finitely many good rational approximations. -/
theorem finite_rat_abs_sub_lt_one_div_den_sq (ξ : ℚ) :
    {q : ℚ | |ξ - q| < 1 / (q.den : ℚ) ^ 2}.Finite := by
  /-
    ξ : Rat
    ⊢ (setOf fun q => LT.lt (abs (HSub.hSub ξ q)) (HDiv.hDiv 1 (HPow.hPow (↑q.den) …
  -/
  let f : ℚ → ℤ × ℕ := fun q => (q.num, q.den)
  /-
    ξ : Rat
    f : Rat → Prod Int Nat := fun q => { fst := q.num, snd := q.den }
    ⊢ (setOf fun q => LT.lt (abs (HSub.hSub ξ q)) (HDiv.hDiv 1 (HPow.hPow (↑q.den) …
  -/
  set s := {q : ℚ | |ξ - q| < 1 / (q.den : ℚ) ^ 2}
  have hinj : Function.Injective f := by
    intro a b hab
    simp only [f, Prod.mk.inj_iff] at hab
    rw [← Rat.num_div_den a, ← Rat.num_div_den b, hab.1, hab.2]
  have H : f '' s ⊆ ⋃ (y : ℕ) (_ : y ∈ Ioc 0 ξ.den), Icc (⌈ξ * y⌉ - 1) (⌊ξ * y⌋ + 1) ×ˢ {y} := by
    intro xy hxy
    simp only [mem_image, mem_setOf] at hxy
    obtain ⟨q, hq₁, hq₂⟩ := hxy
    obtain ⟨hd, hn⟩ := den_le_and_le_num_le_of_sub_lt_one_div_den_sq hq₁
    simp_rw [mem_iUnion]
    refine ⟨q.den, Set.mem_Ioc.mpr ⟨q.pos, hd⟩, ?_⟩
    simp only [prod_singleton, mem_image, mem_Icc, (congr_arg Prod.snd (Eq.symm hq₂)).trans rfl]
    exact ⟨q.num, hn, hq₂⟩
  /-
    ξ : Rat
    f : Rat → Prod Int Nat := fun q => { fst := q.num, snd := q.den }
    s : Set Rat := setOf fun q => LT.lt (abs (HSub.hSub ξ q)) (HDiv.hDiv 1 (HPow.h …
    hinj : Function.Injective f
    H : HasSubset.Subset (Set.image f s) (Set.iUnion fun y => Set.iUnion fun x =>  …
    ⊢ s.Finite
  -/
  refine (Finite.subset ?_ H).of_finite_image hinj.injOn
  /-
    ξ : Rat
    f : Rat → Prod Int Nat := fun q => { fst := q.num, snd := q.den }
    s : Set Rat := setOf fun q => LT.lt (abs (HSub.hSub ξ q)) (HDiv.hDiv 1 (HPow.h …
    hinj : Function.Injective f
    H : HasSubset.Subset (Set.image f s) (Set.iUnion fun y => Set.iUnion fun x =>  …
    ⊢ (Set.iUnion fun y => Set.iUnion fun x => SProd.sprod (Set.Icc (HSub.hSub (In …
  -/
  exact Finite.biUnion (finite_Ioc _ _) fun x _ => Finite.prod (finite_Icc _ _) (finite_singleton _)
  /-
    🎉 no goals
  -/


/-- The set of good rational approximations to a real number `ξ` is infinite if and only if
`ξ` is irrational. -/
theorem Real.infinite_rat_abs_sub_lt_one_div_den_sq_iff_irrational (ξ : ℝ) :
    {q : ℚ | |ξ - q| < 1 / (q.den : ℝ) ^ 2}.Infinite ↔ Irrational ξ := by
  refine
    ⟨fun h => (irrational_iff_ne_rational ξ).mpr fun a b H => Set.not_infinite.mpr ?_ h,
      Real.infinite_rat_abs_sub_lt_one_div_den_sq_of_irrational⟩
  /-
    ξ : Real
    h : (setOf fun q => LT.lt (abs (HSub.hSub ξ ↑q)) (HDiv.hDiv 1 (HPow.hPow (↑q.d …
    a b : Int
    H : Eq ξ (HDiv.hDiv ↑a ↑b)
    ⊢ (setOf fun q => LT.lt (abs (HSub.hSub ξ ↑q)) (HDiv.hDiv 1 (HPow.hPow (↑q.den …
  -/
  convert Rat.finite_rat_abs_sub_lt_one_div_den_sq ((a : ℚ) / b) with q
  /-
    case h.e'_2.h.e'_2.h.a
    ξ : Real
    h : (setOf fun q => LT.lt (abs (HSub.hSub ξ ↑q)) (HDiv.hDiv 1 (HPow.hPow (↑q.d …
    a b : Int
    H : Eq ξ (HDiv.hDiv ↑a ↑b)
    q : Rat
    ⊢ Iff (LT.lt (abs (HSub.hSub ξ ↑q)) (HDiv.hDiv 1 (HPow.hPow (↑q.den) 2))) (LT. …
  -/
  rw [H, (by (push_cast; rfl) : (1 : ℝ) / (q.den : ℝ) ^ 2 = (1 / (q.den : ℚ) ^ 2 : ℚ))]
  /-
    case h.e'_2.h.e'_2.h.a
    ξ : Real
    h : (setOf fun q => LT.lt (abs (HSub.hSub ξ ↑q)) (HDiv.hDiv 1 (HPow.hPow (↑q.d …
    a b : Int
    H : Eq ξ (HDiv.hDiv ↑a ↑b)
    q : Rat
    ⊢ Iff (LT.lt (abs (HSub.hSub (HDiv.hDiv ↑a ↑b) ↑q)) ↑(HDiv.hDiv 1 (HPow.hPow ( …
  -/
  norm_cast
  /-
    🎉 no goals
  -/


/-- We give a direct recursive definition of the convergents of the continued fraction
expansion of a real number `ξ`. The main reason for that is that we want to have the
convergents as rational numbers; the versions `(GenContFract.of ξ).convs` and
`(GenContFract.of ξ).convs'` always give something of the same type as `ξ`.
We can then also use dot notation `ξ.convergent n`.
Another minor reason is that this demonstrates that the proof
of Legendre's theorem does not need anything beyond this definition.
We provide a proof that this definition agrees with the other one;
see `Real.convs_eq_convergent`.
(Note that we use the fact that `1/0 = 0` here to make it work for rational `ξ`.) -/
noncomputable def convergent : ℝ → ℕ → ℚ
  | ξ, 0 => ⌊ξ⌋
  | ξ, n + 1 => ⌊ξ⌋ + (convergent (fract ξ)⁻¹ n)⁻¹


/-- The zeroth convergent of `ξ` is `⌊ξ⌋`. -/
@[simp]
theorem convergent_zero (ξ : ℝ) : ξ.convergent 0 = ⌊ξ⌋ :=
  rfl


/-- The `(n+1)`th convergent of `ξ` is the `n`th convergent of `1/(fract ξ)`. -/
@[simp]
theorem convergent_succ (ξ : ℝ) (n : ℕ) :
    ξ.convergent (n + 1) = ⌊ξ⌋ + ((fract ξ)⁻¹.convergent n)⁻¹ :=
  -- Porting note(https://github.com/leanprover-community/mathlib4/issues/5026): was
  -- by simp only [convergent]
  rfl


/-- All convergents of `0` are zero. -/
@[simp]
theorem convergent_of_zero (n : ℕ) : convergent 0 n = 0 := by
  /-
    n : Nat
    ⊢ Eq (Real.convergent 0 n) 0
  -/
  induction' n with n ih
    /-
      case zero
      ⊢ Eq (Real.convergent 0 0) 0
    -/
  · simp only [convergent_zero, floor_zero, cast_zero]
    /-
      🎉 no goals
    -/
    /-
      case succ
      n : Nat
      ih : Eq (Real.convergent 0 n) 0
      ⊢ Eq (Real.convergent 0 (HAdd.hAdd n 1)) 0
    -/
  · simp only [ih, convergent_succ, floor_zero, cast_zero, fract_zero, add_zero, inv_zero]
    /-
      🎉 no goals
    -/


/-- If `ξ` is an integer, all its convergents equal `ξ`. -/
@[simp]
theorem convergent_of_int {ξ : ℤ} (n : ℕ) : convergent ξ n = ξ := by
  /-
    ξ : Int
    n : Nat
    ⊢ Eq ((↑ξ).convergent n) ↑ξ
  -/
  cases n
    /-
      case zero
      ξ : Int
      ⊢ Eq ((↑ξ).convergent 0) ↑ξ
    -/
  · simp only [convergent_zero, floor_intCast]
    /-
      🎉 no goals
    -/
  · simp only [convergent_succ, floor_intCast, fract_intCast, convergent_of_zero, add_zero,
      inv_zero]


/-- Define the technical condition to be used as assumption in the inductive proof. -/
def ContfracLegendre.Ass (ξ : ℝ) (u v : ℤ) : Prop :=
  IsCoprime u v ∧ (v = 1 → (-(1 / 2) : ℝ) < ξ - u) ∧
    |ξ - u / v| < ((v : ℝ) * (2 * v - 1))⁻¹

-- ### Auxiliary lemmas
-- This saves a few lines below, as it is frequently needed.

private theorem aux₀ {v : ℤ} (hv : 0 < v) : (0 : ℝ) < v ∧ (0 : ℝ) < 2 * v - 1 :=
                       /-
                         v : Int
                         hv : LT.lt 0 v
                         ⊢ LT.lt 0 (HSub.hSub (HMul.hMul 2 ↑v) 1)
                       -/
  ⟨cast_pos.mpr hv, by norm_cast; omega⟩
                                  /-
                                    🎉 no goals
                                  -/

-- In the following, we assume that `ass ξ u v` holds and `v ≥ 2`.

private theorem aux₁ : 0 < fract ξ := by
  /-
    ξ : Real
    u v : Int
    hv : LE.le 2 v
    h : Real.ContfracLegendre.Ass ξ u v
    ⊢ LT.lt 0 (Int.fract ξ)
  -/
  have hv₀ : (0 : ℝ) < v := cast_pos.mpr (zero_lt_two.trans_le hv)
  /-
    ξ : Real
    u v : Int
    hv : LE.le 2 v
    h : Real.ContfracLegendre.Ass ξ u v
    hv₀ : LT.lt 0 ↑v
    ⊢ LT.lt 0 (Int.fract ξ)
  -/
  obtain ⟨hv₁, hv₂⟩ := aux₀ (zero_lt_two.trans_le hv)
  /-
    case intro
    ξ : Real
    u v : Int
    hv : LE.le 2 v
    h : Real.ContfracLegendre.Ass ξ u v
    hv₀ hv₁ : LT.lt 0 ↑v
    hv₂ : LT.lt 0 (HSub.hSub (HMul.hMul 2 ↑v) 1)
    ⊢ LT.lt 0 (Int.fract ξ)
  -/
  obtain ⟨hcop, _, h⟩ := h
  /-
    case intro.intro.intro
    ξ : Real
    u v : Int
    hv : LE.le 2 v
    hv₀ hv₁ : LT.lt 0 ↑v
    hv₂ : LT.lt 0 (HSub.hSub (HMul.hMul 2 ↑v) 1)
    hcop : IsCoprime u v
    left✝ : Eq v 1 → LT.lt (Neg.neg (1 / 2)) (HSub.hSub ξ ↑u)
    h : LT.lt (abs (HSub.hSub ξ (HDiv.hDiv ↑u ↑v))) (Inv.inv (HMul.hMul (↑v) (HSub …
    ⊢ LT.lt 0 (Int.fract ξ)
  -/
  refine fract_pos.mpr fun hf => ?_
  /-
    case intro.intro.intro
    ξ : Real
    u v : Int
    hv : LE.le 2 v
    hv₀ hv₁ : LT.lt 0 ↑v
    hv₂ : LT.lt 0 (HSub.hSub (HMul.hMul 2 ↑v) 1)
    hcop : IsCoprime u v
    left✝ : Eq v 1 → LT.lt (Neg.neg (1 / 2)) (HSub.hSub ξ ↑u)
    h : LT.lt (abs (HSub.hSub ξ (HDiv.hDiv ↑u ↑v))) (Inv.inv (HMul.hMul (↑v) (HSub …
    hf : Eq ξ ↑(Int.floor ξ)
    ⊢ False
  -/
  rw [hf] at h
  have H : (2 * v - 1 : ℝ) < 1 := by
    refine (mul_lt_iff_lt_one_right hv₀).1 ((inv_lt_inv₀ hv₀ (mul_pos hv₁ hv₂)).1 (h.trans_le' ?_))
    have h' : (⌊ξ⌋ : ℝ) - u / v = (⌊ξ⌋ * v - u) / v := by field_simp
    rw [h', abs_div, abs_of_pos hv₀, ← one_div, div_le_div_iff_of_pos_right hv₀]
    norm_cast
    rw [← zero_add (1 : ℤ), add_one_le_iff, abs_pos, sub_ne_zero]
    rintro rfl
    cases isUnit_iff.mp (isCoprime_self.mp (IsCoprime.mul_left_iff.mp hcop).2) <;> omega
  /-
    case intro.intro.intro
    ξ : Real
    u v : Int
    hv : LE.le 2 v
    hv₀ hv₁ : LT.lt 0 ↑v
    hv₂ : LT.lt 0 (HSub.hSub (HMul.hMul 2 ↑v) 1)
    hcop : IsCoprime u v
    left✝ : Eq v 1 → LT.lt (Neg.neg (1 / 2)) (HSub.hSub ξ ↑u)
    h : LT.lt (abs (HSub.hSub (↑(Int.floor ξ)) (HDiv.hDiv ↑u ↑v))) (Inv.inv (HMul. …
    hf : Eq ξ ↑(Int.floor ξ)
    H : LT.lt (HSub.hSub (HMul.hMul 2 ↑v) 1) 1
    ⊢ False
  -/
  norm_cast at H
  /-
    case intro.intro.intro
    ξ : Real
    u v : Int
    hv : LE.le 2 v
    hv₀ hv₁ : LT.lt 0 ↑v
    hv₂ : LT.lt 0 (HSub.hSub (HMul.hMul 2 ↑v) 1)
    hcop : IsCoprime u v
    left✝ : Eq v 1 → LT.lt (Neg.neg (1 / 2)) (HSub.hSub ξ ↑u)
    h : LT.lt (abs (HSub.hSub (↑(Int.floor ξ)) (HDiv.hDiv ↑u ↑v))) (Inv.inv (HMul. …
    hf : Eq ξ ↑(Int.floor ξ)
    H : LT.lt (HSub.hSub (HMul.hMul 2 v) 1) 1
    ⊢ False
  -/
  linarith only [hv, H]
  /-
    🎉 no goals
  -/

-- An auxiliary lemma for the inductive step.

private theorem aux₂ : 0 < u - ⌊ξ⌋ * v ∧ u - ⌊ξ⌋ * v < v := by
  /-
    ξ : Real
    u v : Int
    hv : LE.le 2 v
    h : Real.ContfracLegendre.Ass ξ u v
    ⊢ And (LT.lt 0 (HSub.hSub u (HMul.hMul (Int.floor ξ) v))) (LT.lt (HSub.hSub u  …
  -/
  obtain ⟨hcop, _, h⟩ := h
  /-
    case intro.intro
    ξ : Real
    u v : Int
    hv : LE.le 2 v
    hcop : IsCoprime u v
    left✝ : Eq v 1 → LT.lt (Neg.neg (1 / 2)) (HSub.hSub ξ ↑u)
    h : LT.lt (abs (HSub.hSub ξ (HDiv.hDiv ↑u ↑v))) (Inv.inv (HMul.hMul (↑v) (HSub …
    ⊢ And (LT.lt 0 (HSub.hSub u (HMul.hMul (Int.floor ξ) v))) (LT.lt (HSub.hSub u  …
  -/
  obtain ⟨hv₀, hv₀'⟩ := aux₀ (zero_lt_two.trans_le hv)
  /-
    case intro.intro.intro
    ξ : Real
    u v : Int
    hv : LE.le 2 v
    hcop : IsCoprime u v
    left✝ : Eq v 1 → LT.lt (Neg.neg (1 / 2)) (HSub.hSub ξ ↑u)
    h : LT.lt (abs (HSub.hSub ξ (HDiv.hDiv ↑u ↑v))) (Inv.inv (HMul.hMul (↑v) (HSub …
    hv₀ : LT.lt 0 ↑v
    hv₀' : LT.lt 0 (HSub.hSub (HMul.hMul 2 ↑v) 1)
    ⊢ And (LT.lt 0 (HSub.hSub u (HMul.hMul (Int.floor ξ) v))) (LT.lt (HSub.hSub u  …
  -/
  have hv₁ : 0 < 2 * v - 1 := by linarith only [hv]
  rw [← one_div, lt_div_iff₀ (mul_pos hv₀ hv₀'), ← abs_of_pos (mul_pos hv₀ hv₀'), ← abs_mul,
    sub_mul, ← mul_assoc, ← mul_assoc, div_mul_cancel₀ _ hv₀.ne', abs_sub_comm, abs_lt,
    lt_sub_iff_add_lt, sub_lt_iff_lt_add, mul_assoc] at h
  have hu₀ : 0 ≤ u - ⌊ξ⌋ * v := by
    -- Porting note: this abused the definitional equality `-1 + 1 = 0`
    -- refine' (mul_nonneg_iff_of_pos_right hv₁).mp ((lt_iff_add_one_le (-1 : ℤ) _).mp _)
    refine (mul_nonneg_iff_of_pos_right hv₁).mp ?_
    rw [← sub_one_lt_iff, zero_sub]
    replace h := h.1
    rw [← lt_sub_iff_add_lt, ← mul_assoc, ← sub_mul] at h
    exact mod_cast
      h.trans_le
        ((mul_le_mul_right <| hv₀').mpr <|
          (sub_le_sub_iff_left (u : ℝ)).mpr ((mul_le_mul_right hv₀).mpr (floor_le ξ)))
  have hu₁ : u - ⌊ξ⌋ * v ≤ v := by
    refine _root_.le_of_mul_le_mul_right (le_of_lt_add_one ?_) hv₁
    replace h := h.2
    rw [← sub_lt_iff_lt_add, ← mul_assoc, ← sub_mul, ← add_lt_add_iff_right (v * (2 * v - 1) : ℝ),
      add_comm (1 : ℝ)] at h
    have :=
      (mul_lt_mul_right <| hv₀').mpr
        ((sub_lt_sub_iff_left (u : ℝ)).mpr <|
          (mul_lt_mul_right hv₀).mpr <| sub_right_lt_of_lt_add <| lt_floor_add_one ξ)
    rw [sub_mul ξ, one_mul, ← sub_add, add_mul] at this
    exact mod_cast this.trans h
  have huv_cop : IsCoprime (u - ⌊ξ⌋ * v) v := by
    rwa [sub_eq_add_neg, ← neg_mul, IsCoprime.add_mul_right_left_iff]
  /-
    case intro.intro.intro
    ξ : Real
    u v : Int
    hv : LE.le 2 v
    hcop : IsCoprime u v
    left✝ : Eq v 1 → LT.lt (Neg.neg (1 / 2)) (HSub.hSub ξ ↑u)
    h : And (LT.lt (HAdd.hAdd (-1) (HMul.hMul ξ (HMul.hMul (↑v) (HSub.hSub (HMul.h …
    hv₀ : LT.lt 0 ↑v
    hv₀' : LT.lt 0 (HSub.hSub (HMul.hMul 2 ↑v) 1)
    hv₁ : LT.lt 0 (HSub.hSub (HMul.hMul 2 v) 1)
    hu₀ : LE.le 0 (HSub.hSub u (HMul.hMul (Int.floor ξ) v))
    hu₁ : LE.le (HSub.hSub u (HMul.hMul (Int.floor ξ) v)) v
    huv_cop : IsCoprime (HSub.hSub u (HMul.hMul (Int.floor ξ) v)) v
    ⊢ And (LT.lt 0 (HSub.hSub u (HMul.hMul (Int.floor ξ) v))) (LT.lt (HSub.hSub u  …
  -/
  refine ⟨lt_of_le_of_ne' hu₀ fun hf => ?_, lt_of_le_of_ne hu₁ fun hf => ?_⟩ <;>
      /-
        case intro.intro.intro.refine_1
        ξ : Real
        u v : Int
        hv : LE.le 2 v
        hcop : IsCoprime u v
        left✝ : Eq v 1 → LT.lt (Neg.neg (1 / 2)) (HSub.hSub ξ ↑u)
        h : And (LT.lt (HAdd.hAdd (-1) (HMul.hMul ξ (HMul.hMul (↑v) (HSub.hSub (HMul.h …
        hv₀ : LT.lt 0 ↑v
        hv₀' : LT.lt 0 (HSub.hSub (HMul.hMul 2 ↑v) 1)
        hv₁ : LT.lt 0 (HSub.hSub (HMul.hMul 2 v) 1)
        hu₀ : LE.le 0 (HSub.hSub u (HMul.hMul (Int.floor ξ) v))
        hu₁ : LE.le (HSub.hSub u (HMul.hMul (Int.floor ξ) v)) v
        huv_cop : IsCoprime (HSub.hSub u (HMul.hMul (Int.floor ξ) v)) v
        hf : Eq (HSub.hSub u (HMul.hMul (Int.floor ξ) v)) 0
        ⊢ False
      -/
      /-
        case intro.intro.intro.refine_1
        ξ : Real
        u v : Int
        hv : LE.le 2 v
        hcop : IsCoprime u v
        left✝ : Eq v 1 → LT.lt (Neg.neg (1 / 2)) (HSub.hSub ξ ↑u)
        h : And (LT.lt (HAdd.hAdd (-1) (HMul.hMul ξ (HMul.hMul (↑v) (HSub.hSub (HMul.h …
        hv₀ : LT.lt 0 ↑v
        hv₀' : LT.lt 0 (HSub.hSub (HMul.hMul 2 ↑v) 1)
        hv₁ : LT.lt 0 (HSub.hSub (HMul.hMul 2 v) 1)
        hu₀ : LE.le 0 (HSub.hSub u (HMul.hMul (Int.floor ξ) v))
        hu₁ : LE.le (HSub.hSub u (HMul.hMul (Int.floor ξ) v)) v
        huv_cop : IsCoprime 0 v
        hf : Eq (HSub.hSub u (HMul.hMul (Int.floor ξ) v)) 0
        ⊢ False
      -/
      /-
        case intro.intro.intro.refine_1
        ξ : Real
        u v : Int
        hv : LE.le 2 v
        hcop : IsCoprime u v
        left✝ : Eq v 1 → LT.lt (Neg.neg (1 / 2)) (HSub.hSub ξ ↑u)
        h : And (LT.lt (HAdd.hAdd (-1) (HMul.hMul ξ (HMul.hMul (↑v) (HSub.hSub (HMul.h …
        hv₀ : LT.lt 0 ↑v
        hv₀' : LT.lt 0 (HSub.hSub (HMul.hMul 2 ↑v) 1)
        hv₁ : LT.lt 0 (HSub.hSub (HMul.hMul 2 v) 1)
        hu₀ : LE.le 0 (HSub.hSub u (HMul.hMul (Int.floor ξ) v))
        hu₁ : LE.le (HSub.hSub u (HMul.hMul (Int.floor ξ) v)) v
        hf : Eq (HSub.hSub u (HMul.hMul (Int.floor ξ) v)) 0
        huv_cop : Or (Eq v 1) (Eq v (-1))
        ⊢ False
      -/
                                              /-
                                                🎉 no goals
                                              -/
                                              /-
                                                🎉 no goals
                                              -/
      simp only [isCoprime_zero_left, isCoprime_self, isUnit_iff] at huv_cop
      /-
        case intro.intro.intro.refine_2
        ξ : Real
        u v : Int
        hv : LE.le 2 v
        hcop : IsCoprime u v
        left✝ : Eq v 1 → LT.lt (Neg.neg (1 / 2)) (HSub.hSub ξ ↑u)
        h : And (LT.lt (HAdd.hAdd (-1) (HMul.hMul ξ (HMul.hMul (↑v) (HSub.hSub (HMul.h …
        hv₀ : LT.lt 0 ↑v
        hv₀' : LT.lt 0 (HSub.hSub (HMul.hMul 2 ↑v) 1)
        hv₁ : LT.lt 0 (HSub.hSub (HMul.hMul 2 v) 1)
        hu₀ : LE.le 0 (HSub.hSub u (HMul.hMul (Int.floor ξ) v))
        hu₁ : LE.le (HSub.hSub u (HMul.hMul (Int.floor ξ) v)) v
        hf : Eq (HSub.hSub u (HMul.hMul (Int.floor ξ) v)) v
        huv_cop : Or (Eq v 1) (Eq v (-1))
        ⊢ False
      -/
                                              /-
                                                🎉 no goals
                                              -/
      cases' huv_cop with huv_cop huv_cop <;> linarith only [hv, huv_cop]
                                              /-
                                                🎉 no goals
                                              -/

-- The key step: the relevant inequality persists in the inductive step.

private theorem aux₃ :
    |(fract ξ)⁻¹ - v / (u - ⌊ξ⌋ * v)| < (((u : ℝ) - ⌊ξ⌋ * v) * (2 * (u - ⌊ξ⌋ * v) - 1))⁻¹ := by
  /-
    ξ : Real
    u v : Int
    hv : LE.le 2 v
    h : Real.ContfracLegendre.Ass ξ u v
    ⊢ LT.lt (abs (HSub.hSub (Inv.inv (Int.fract ξ)) (HDiv.hDiv (↑v) (HSub.hSub (↑u …
  -/
  obtain ⟨hu₀, huv⟩ := aux₂ hv h
  /-
    case intro
    ξ : Real
    u v : Int
    hv : LE.le 2 v
    h : Real.ContfracLegendre.Ass ξ u v
    hu₀ : LT.lt 0 (HSub.hSub u (HMul.hMul (Int.floor ξ) v))
    huv : LT.lt (HSub.hSub u (HMul.hMul (Int.floor ξ) v)) v
    ⊢ LT.lt (abs (HSub.hSub (Inv.inv (Int.fract ξ)) (HDiv.hDiv (↑v) (HSub.hSub (↑u …
  -/
  have hξ₀ := aux₁ hv h
  /-
    case intro
    ξ : Real
    u v : Int
    hv : LE.le 2 v
    h : Real.ContfracLegendre.Ass ξ u v
    hu₀ : LT.lt 0 (HSub.hSub u (HMul.hMul (Int.floor ξ) v))
    huv : LT.lt (HSub.hSub u (HMul.hMul (Int.floor ξ) v)) v
    hξ₀ : LT.lt 0 (Int.fract ξ)
    ⊢ LT.lt (abs (HSub.hSub (Inv.inv (Int.fract ξ)) (HDiv.hDiv (↑v) (HSub.hSub (↑u …
  -/
  set u' := u - ⌊ξ⌋ * v with hu'
  /-
    case intro
    ξ : Real
    u v : Int
    hv : LE.le 2 v
    h : Real.ContfracLegendre.Ass ξ u v
    hξ₀ : LT.lt 0 (Int.fract ξ)
    u' : Int := HSub.hSub u (HMul.hMul (Int.floor ξ) v)
    hu₀ : LT.lt 0 u'
    huv : LT.lt u' v
    hu' : Eq u' (HSub.hSub u (HMul.hMul (Int.floor ξ) v))
    ⊢ LT.lt (abs (HSub.hSub (Inv.inv (Int.fract ξ)) (HDiv.hDiv (↑v) (HSub.hSub (↑u …
  -/
  have hu'ℝ : (u' : ℝ) = u - ⌊ξ⌋ * v := mod_cast hu'
  /-
    case intro
    ξ : Real
    u v : Int
    hv : LE.le 2 v
    h : Real.ContfracLegendre.Ass ξ u v
    hξ₀ : LT.lt 0 (Int.fract ξ)
    u' : Int := HSub.hSub u (HMul.hMul (Int.floor ξ) v)
    hu₀ : LT.lt 0 u'
    huv : LT.lt u' v
    hu' : Eq u' (HSub.hSub u (HMul.hMul (Int.floor ξ) v))
    hu'ℝ : Eq (↑u') (HSub.hSub (↑u) (HMul.hMul ↑(Int.floor ξ) ↑v))
    ⊢ LT.lt (abs (HSub.hSub (Inv.inv (Int.fract ξ)) (HDiv.hDiv (↑v) (HSub.hSub (↑u …
  -/
  rw [← hu'ℝ]
  /-
    case intro
    ξ : Real
    u v : Int
    hv : LE.le 2 v
    h : Real.ContfracLegendre.Ass ξ u v
    hξ₀ : LT.lt 0 (Int.fract ξ)
    u' : Int := HSub.hSub u (HMul.hMul (Int.floor ξ) v)
    hu₀ : LT.lt 0 u'
    huv : LT.lt u' v
    hu' : Eq u' (HSub.hSub u (HMul.hMul (Int.floor ξ) v))
    hu'ℝ : Eq (↑u') (HSub.hSub (↑u) (HMul.hMul ↑(Int.floor ξ) ↑v))
    ⊢ LT.lt (abs (HSub.hSub (Inv.inv (Int.fract ξ)) (HDiv.hDiv ↑v ↑u'))) (Inv.inv  …
  -/
  replace hu'ℝ := (eq_sub_iff_add_eq.mp hu'ℝ).symm
  /-
    case intro
    ξ : Real
    u v : Int
    hv : LE.le 2 v
    h : Real.ContfracLegendre.Ass ξ u v
    hξ₀ : LT.lt 0 (Int.fract ξ)
    u' : Int := HSub.hSub u (HMul.hMul (Int.floor ξ) v)
    hu₀ : LT.lt 0 u'
    huv : LT.lt u' v
    hu' : Eq u' (HSub.hSub u (HMul.hMul (Int.floor ξ) v))
    hu'ℝ : Eq (↑u) (HAdd.hAdd (↑u') (HMul.hMul ↑(Int.floor ξ) ↑v))
    ⊢ LT.lt (abs (HSub.hSub (Inv.inv (Int.fract ξ)) (HDiv.hDiv ↑v ↑u'))) (Inv.inv  …
  -/
  obtain ⟨Hu, Hu'⟩ := aux₀ hu₀
  /-
    case intro.intro
    ξ : Real
    u v : Int
    hv : LE.le 2 v
    h : Real.ContfracLegendre.Ass ξ u v
    hξ₀ : LT.lt 0 (Int.fract ξ)
    u' : Int := HSub.hSub u (HMul.hMul (Int.floor ξ) v)
    hu₀ : LT.lt 0 u'
    huv : LT.lt u' v
    hu' : Eq u' (HSub.hSub u (HMul.hMul (Int.floor ξ) v))
    hu'ℝ : Eq (↑u) (HAdd.hAdd (↑u') (HMul.hMul ↑(Int.floor ξ) ↑v))
    Hu : LT.lt 0 ↑u'
    Hu' : LT.lt 0 (HSub.hSub (HMul.hMul 2 ↑u') 1)
    ⊢ LT.lt (abs (HSub.hSub (Inv.inv (Int.fract ξ)) (HDiv.hDiv ↑v ↑u'))) (Inv.inv  …
  -/
  obtain ⟨Hv, Hv'⟩ := aux₀ (zero_lt_two.trans_le hv)
  /-
    case intro.intro.intro
    ξ : Real
    u v : Int
    hv : LE.le 2 v
    h : Real.ContfracLegendre.Ass ξ u v
    hξ₀ : LT.lt 0 (Int.fract ξ)
    u' : Int := HSub.hSub u (HMul.hMul (Int.floor ξ) v)
    hu₀ : LT.lt 0 u'
    huv : LT.lt u' v
    hu' : Eq u' (HSub.hSub u (HMul.hMul (Int.floor ξ) v))
    hu'ℝ : Eq (↑u) (HAdd.hAdd (↑u') (HMul.hMul ↑(Int.floor ξ) ↑v))
    Hu : LT.lt 0 ↑u'
    Hu' : LT.lt 0 (HSub.hSub (HMul.hMul 2 ↑u') 1)
    Hv : LT.lt 0 ↑v
    Hv' : LT.lt 0 (HSub.hSub (HMul.hMul 2 ↑v) 1)
    ⊢ LT.lt (abs (HSub.hSub (Inv.inv (Int.fract ξ)) (HDiv.hDiv ↑v ↑u'))) (Inv.inv  …
  -/
  have H₁ := div_pos (div_pos Hv Hu) hξ₀
  /-
    case intro.intro.intro
    ξ : Real
    u v : Int
    hv : LE.le 2 v
    h : Real.ContfracLegendre.Ass ξ u v
    hξ₀ : LT.lt 0 (Int.fract ξ)
    u' : Int := HSub.hSub u (HMul.hMul (Int.floor ξ) v)
    hu₀ : LT.lt 0 u'
    huv : LT.lt u' v
    hu' : Eq u' (HSub.hSub u (HMul.hMul (Int.floor ξ) v))
    hu'ℝ : Eq (↑u) (HAdd.hAdd (↑u') (HMul.hMul ↑(Int.floor ξ) ↑v))
    Hu : LT.lt 0 ↑u'
    Hu' : LT.lt 0 (HSub.hSub (HMul.hMul 2 ↑u') 1)
    Hv : LT.lt 0 ↑v
    Hv' : LT.lt 0 (HSub.hSub (HMul.hMul 2 ↑v) 1)
    H₁ : LT.lt 0 (HDiv.hDiv (HDiv.hDiv ↑v ↑u') (Int.fract ξ))
    ⊢ LT.lt (abs (HSub.hSub (Inv.inv (Int.fract ξ)) (HDiv.hDiv ↑v ↑u'))) (Inv.inv  …
  -/
  replace h := h.2.2
  have h' : |fract ξ - u' / v| < ((v : ℝ) * (2 * v - 1))⁻¹ := by
    rwa [hu'ℝ, add_div, mul_div_cancel_right₀ _ Hv.ne', ← sub_sub, sub_right_comm] at h
  have H : (2 * u' - 1 : ℝ) ≤ (2 * v - 1) * fract ξ := by
    replace h := (abs_lt.mp h).1
    have : (2 * (v : ℝ) - 1) * (-((v : ℝ) * (2 * v - 1))⁻¹ + u' / v) = 2 * u' - (1 + u') / v := by
      field_simp; ring
    rw [hu'ℝ, add_div, mul_div_cancel_right₀ _ Hv.ne', ← sub_sub, sub_right_comm, self_sub_floor,
      lt_sub_iff_add_lt, ← mul_lt_mul_left Hv', this] at h
    refine LE.le.trans ?_ h.le
    rw [sub_le_sub_iff_left, div_le_one Hv, add_comm]
    exact mod_cast huv
  have help₁ {a b c : ℝ} : a ≠ 0 → b ≠ 0 → c ≠ 0 → |a⁻¹ - b / c| = |(a - c / b) * (b / c / a)| := by
    intros; rw [abs_sub_comm]; congr 1; field_simp; ring
  have help₂ :
    ∀ {a b c d : ℝ}, a ≠ 0 → b ≠ 0 → c ≠ 0 → d ≠ 0 → (b * c)⁻¹ * (b / d / a) = (d * c * a)⁻¹ := by
    intros; field_simp; ring
  calc
    |(fract ξ)⁻¹ - v / u'| = |(fract ξ - u' / v) * (v / u' / fract ξ)| :=
      help₁ hξ₀.ne' Hv.ne' Hu.ne'
    _ = |fract ξ - u' / v| * (v / u' / fract ξ) := by rw [abs_mul, abs_of_pos H₁]
    _ < ((v : ℝ) * (2 * v - 1))⁻¹ * (v / u' / fract ξ) := (mul_lt_mul_right H₁).mpr h'
    _ = (u' * (2 * v - 1) * fract ξ)⁻¹ := help₂ hξ₀.ne' Hv.ne' Hv'.ne' Hu.ne'
    _ ≤ ((u' : ℝ) * (2 * u' - 1))⁻¹ := by
      rwa [inv_le_inv₀ (mul_pos (mul_pos Hu Hv') hξ₀) <| mul_pos Hu Hu', mul_assoc,
        mul_le_mul_left Hu]

-- The conditions `ass ξ u v` persist in the inductive step.

private theorem invariant : ContfracLegendre.Ass (fract ξ)⁻¹ v (u - ⌊ξ⌋ * v) := by
  /-
    ξ : Real
    u v : Int
    hv : LE.le 2 v
    h : Real.ContfracLegendre.Ass ξ u v
    ⊢ Real.ContfracLegendre.Ass (Inv.inv (Int.fract ξ)) v (HSub.hSub u (HMul.hMul  …
  -/
  refine ⟨?_, fun huv => ?_, mod_cast aux₃ hv h⟩
    /-
      case refine_1
      ξ : Real
      u v : Int
      hv : LE.le 2 v
      h : Real.ContfracLegendre.Ass ξ u v
      ⊢ IsCoprime v (HSub.hSub u (HMul.hMul (Int.floor ξ) v))
    -/
  · rw [sub_eq_add_neg, ← neg_mul, isCoprime_comm, IsCoprime.add_mul_right_left_iff]
    /-
      case refine_1
      ξ : Real
      u v : Int
      hv : LE.le 2 v
      h : Real.ContfracLegendre.Ass ξ u v
      ⊢ IsCoprime u v
    -/
    exact h.1
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      ξ : Real
      u v : Int
      hv : LE.le 2 v
      h : Real.ContfracLegendre.Ass ξ u v
      huv : Eq (HSub.hSub u (HMul.hMul (Int.floor ξ) v)) 1
      ⊢ LT.lt (Neg.neg (1 / 2)) (HSub.hSub (Inv.inv (Int.fract ξ)) ↑v)
    -/
  · obtain hv₀' := (aux₀ (zero_lt_two.trans_le hv)).2
    have Hv : (v * (2 * v - 1) : ℝ)⁻¹ + (v : ℝ)⁻¹ = 2 / (2 * v - 1) := by
      field_simp; ring
    have Huv : (u / v : ℝ) = ⌊ξ⌋ + (v : ℝ)⁻¹ := by
      rw [sub_eq_iff_eq_add'.mp huv]; field_simp
    /-
      case refine_2
      ξ : Real
      u v : Int
      hv : LE.le 2 v
      h : Real.ContfracLegendre.Ass ξ u v
      huv : Eq (HSub.hSub u (HMul.hMul (Int.floor ξ) v)) 1
      hv₀' : LT.lt 0 (HSub.hSub (HMul.hMul 2 ↑v) 1)
      Hv : Eq (HAdd.hAdd (Inv.inv (HMul.hMul (↑v) (HSub.hSub (HMul.hMul 2 ↑v) 1))) ( …
      Huv : Eq (HDiv.hDiv ↑u ↑v) (HAdd.hAdd (↑(Int.floor ξ)) (Inv.inv ↑v))
      ⊢ LT.lt (Neg.neg (1 / 2)) (HSub.hSub (Inv.inv (Int.fract ξ)) ↑v)
    -/
    have h' := (abs_sub_lt_iff.mp h.2.2).1
    /-
      case refine_2
      ξ : Real
      u v : Int
      hv : LE.le 2 v
      h : Real.ContfracLegendre.Ass ξ u v
      huv : Eq (HSub.hSub u (HMul.hMul (Int.floor ξ) v)) 1
      hv₀' : LT.lt 0 (HSub.hSub (HMul.hMul 2 ↑v) 1)
      Hv : Eq (HAdd.hAdd (Inv.inv (HMul.hMul (↑v) (HSub.hSub (HMul.hMul 2 ↑v) 1))) ( …
      Huv : Eq (HDiv.hDiv ↑u ↑v) (HAdd.hAdd (↑(Int.floor ξ)) (Inv.inv ↑v))
      h' : LT.lt (HSub.hSub ξ (HDiv.hDiv ↑u ↑v)) (Inv.inv (HMul.hMul (↑v) (HSub.hSub …
      ⊢ LT.lt (Neg.neg (1 / 2)) (HSub.hSub (Inv.inv (Int.fract ξ)) ↑v)
    -/
    rw [Huv, ← sub_sub, sub_lt_iff_lt_add, self_sub_floor, Hv] at h'
    rwa [lt_sub_iff_add_lt', (by ring : (v : ℝ) + -(1 / 2) = (2 * v - 1) / 2),
      lt_inv_comm₀ (div_pos hv₀' zero_lt_two) (aux₁ hv h), inv_div]


/-- The technical version of *Legendre's Theorem*. -/
theorem exists_rat_eq_convergent' {v : ℕ} (h : ContfracLegendre.Ass ξ u v) :
    ∃ n, (u / v : ℚ) = ξ.convergent n := by
  /-
    ξ : Real
    u : Int
    v : Nat
    h : Real.ContfracLegendre.Ass ξ u ↑v
    ⊢ Exists fun n => Eq (HDiv.hDiv ↑u ↑v) (ξ.convergent n)
  -/
  induction v using Nat.strong_induction_on generalizing ξ u with | h v ih => ?_
  /-
    case h
    v : Nat
    ih : ∀ (m : Nat), LT.lt m v → ∀ {ξ : Real} {u : Int}, Real.ContfracLegendre.As …
    ξ : Real
    u : Int
    h : Real.ContfracLegendre.Ass ξ u ↑v
    ⊢ Exists fun n => Eq (HDiv.hDiv ↑u ↑v) (ξ.convergent n)
  -/
  rcases lt_trichotomy v 1 with (ht | rfl | ht)
    /-
      case h.inl
      v : Nat
      ih : ∀ (m : Nat), LT.lt m v → ∀ {ξ : Real} {u : Int}, Real.ContfracLegendre.As …
      ξ : Real
      u : Int
      h : Real.ContfracLegendre.Ass ξ u ↑v
      ht : LT.lt v 1
      ⊢ Exists fun n => Eq (HDiv.hDiv ↑u ↑v) (ξ.convergent n)
    -/
  · replace h := h.2.2
    simp only [Nat.lt_one_iff.mp ht, Nat.cast_zero, div_zero, tsub_zero, zero_mul,
      cast_zero, inv_zero] at h
    /-
      case h.inl
      v : Nat
      ih : ∀ (m : Nat), LT.lt m v → ∀ {ξ : Real} {u : Int}, Real.ContfracLegendre.As …
      ξ : Real
      u : Int
      ht : LT.lt v 1
      h : LT.lt (abs ξ) 0
      ⊢ Exists fun n => Eq (HDiv.hDiv ↑u ↑v) (ξ.convergent n)
    -/
    exact False.elim (lt_irrefl _ <| (abs_nonneg ξ).trans_lt h)
    /-
      🎉 no goals
    -/
    /-
      case h.inr.inl
      ξ : Real
      u : Int
      ih : ∀ (m : Nat), LT.lt m 1 → ∀ {ξ : Real} {u : Int}, Real.ContfracLegendre.As …
      h : Real.ContfracLegendre.Ass ξ u ↑1
      ⊢ Exists fun n => Eq (HDiv.hDiv ↑u ↑1) (ξ.convergent n)
    -/
  · rw [Nat.cast_one, div_one]
    /-
      case h.inr.inl
      ξ : Real
      u : Int
      ih : ∀ (m : Nat), LT.lt m 1 → ∀ {ξ : Real} {u : Int}, Real.ContfracLegendre.As …
      h : Real.ContfracLegendre.Ass ξ u ↑1
      ⊢ Exists fun n => Eq (↑u) (ξ.convergent n)
    -/
    obtain ⟨_, h₁, h₂⟩ := h
    /-
      case h.inr.inl.intro.intro
      ξ : Real
      u : Int
      ih : ∀ (m : Nat), LT.lt m 1 → ∀ {ξ : Real} {u : Int}, Real.ContfracLegendre.As …
      left✝ : IsCoprime u ↑1
      h₁ : Eq (↑1) 1 → LT.lt (Neg.neg (1 / 2)) (HSub.hSub ξ ↑u)
      h₂ : LT.lt (abs (HSub.hSub ξ (HDiv.hDiv ↑u ↑↑1))) (Inv.inv (HMul.hMul (↑↑1) (H …
      ⊢ Exists fun n => Eq (↑u) (ξ.convergent n)
    -/
    rcases le_or_lt (u : ℝ) ξ with ht | ht
      /-
        case h.inr.inl.intro.intro.inl
        ξ : Real
        u : Int
        ih : ∀ (m : Nat), LT.lt m 1 → ∀ {ξ : Real} {u : Int}, Real.ContfracLegendre.As …
        left✝ : IsCoprime u ↑1
        h₁ : Eq (↑1) 1 → LT.lt (Neg.neg (1 / 2)) (HSub.hSub ξ ↑u)
        h₂ : LT.lt (abs (HSub.hSub ξ (HDiv.hDiv ↑u ↑↑1))) (Inv.inv (HMul.hMul (↑↑1) (H …
        ht : LE.le (↑u) ξ
        ⊢ Exists fun n => Eq (↑u) (ξ.convergent n)
      -/
    · use 0
      /-
        case h
        ξ : Real
        u : Int
        ih : ∀ (m : Nat), LT.lt m 1 → ∀ {ξ : Real} {u : Int}, Real.ContfracLegendre.As …
        left✝ : IsCoprime u ↑1
        h₁ : Eq (↑1) 1 → LT.lt (Neg.neg (1 / 2)) (HSub.hSub ξ ↑u)
        h₂ : LT.lt (abs (HSub.hSub ξ (HDiv.hDiv ↑u ↑↑1))) (Inv.inv (HMul.hMul (↑↑1) (H …
        ht : LE.le (↑u) ξ
        ⊢ Eq (↑u) (ξ.convergent 0)
      -/
      rw [convergent_zero, Rat.coe_int_inj, eq_comm, floor_eq_iff]
      /-
        case h
        ξ : Real
        u : Int
        ih : ∀ (m : Nat), LT.lt m 1 → ∀ {ξ : Real} {u : Int}, Real.ContfracLegendre.As …
        left✝ : IsCoprime u ↑1
        h₁ : Eq (↑1) 1 → LT.lt (Neg.neg (1 / 2)) (HSub.hSub ξ ↑u)
        h₂ : LT.lt (abs (HSub.hSub ξ (HDiv.hDiv ↑u ↑↑1))) (Inv.inv (HMul.hMul (↑↑1) (H …
        ht : LE.le (↑u) ξ
        ⊢ And (LE.le (↑u) ξ) (LT.lt ξ (HAdd.hAdd (↑u) 1))
      -/
                                                                        /-
                                                                          🎉 no goals
                                                                        -/
      convert And.intro ht (sub_lt_iff_lt_add'.mp (abs_lt.mp h₂).2) <;> norm_num
                                                                        /-
                                                                          🎉 no goals
                                                                        -/
      /-
        case h.inr.inl.intro.intro.inr
        ξ : Real
        u : Int
        ih : ∀ (m : Nat), LT.lt m 1 → ∀ {ξ : Real} {u : Int}, Real.ContfracLegendre.As …
        left✝ : IsCoprime u ↑1
        h₁ : Eq (↑1) 1 → LT.lt (Neg.neg (1 / 2)) (HSub.hSub ξ ↑u)
        h₂ : LT.lt (abs (HSub.hSub ξ (HDiv.hDiv ↑u ↑↑1))) (Inv.inv (HMul.hMul (↑↑1) (H …
        ht : LT.lt ξ ↑u
        ⊢ Exists fun n => Eq (↑u) (ξ.convergent n)
      -/
    · replace h₁ := lt_sub_iff_add_lt'.mp (h₁ rfl)
      have hξ₁ : ⌊ξ⌋ = u - 1 := by
        rw [floor_eq_iff, cast_sub, cast_one, sub_add_cancel]
        exact ⟨(((sub_lt_sub_iff_left _).mpr one_half_lt_one).trans h₁).le, ht⟩
      /-
        case h.inr.inl.intro.intro.inr
        ξ : Real
        u : Int
        ih : ∀ (m : Nat), LT.lt m 1 → ∀ {ξ : Real} {u : Int}, Real.ContfracLegendre.As …
        left✝ : IsCoprime u ↑1
        h₂ : LT.lt (abs (HSub.hSub ξ (HDiv.hDiv ↑u ↑↑1))) (Inv.inv (HMul.hMul (↑↑1) (H …
        ht : LT.lt ξ ↑u
        h₁ : LT.lt (HAdd.hAdd (↑u) (Neg.neg (1 / 2))) ξ
        hξ₁ : Eq (Int.floor ξ) (HSub.hSub u 1)
        ⊢ Exists fun n => Eq (↑u) (ξ.convergent n)
      -/
      rcases eq_or_ne ξ ⌊ξ⌋ with Hξ | Hξ
        /-
          case h.inr.inl.intro.intro.inr.inl
          ξ : Real
          u : Int
          ih : ∀ (m : Nat), LT.lt m 1 → ∀ {ξ : Real} {u : Int}, Real.ContfracLegendre.As …
          left✝ : IsCoprime u ↑1
          h₂ : LT.lt (abs (HSub.hSub ξ (HDiv.hDiv ↑u ↑↑1))) (Inv.inv (HMul.hMul (↑↑1) (H …
          ht : LT.lt ξ ↑u
          h₁ : LT.lt (HAdd.hAdd (↑u) (Neg.neg (1 / 2))) ξ
          hξ₁ : Eq (Int.floor ξ) (HSub.hSub u 1)
          Hξ : Eq ξ ↑(Int.floor ξ)
          ⊢ Exists fun n => Eq (↑u) (ξ.convergent n)
        -/
      · rw [Hξ, hξ₁, cast_sub, cast_one, ← sub_eq_add_neg, sub_lt_sub_iff_left] at h₁
        /-
          case h.inr.inl.intro.intro.inr.inl
          ξ : Real
          u : Int
          ih : ∀ (m : Nat), LT.lt m 1 → ∀ {ξ : Real} {u : Int}, Real.ContfracLegendre.As …
          left✝ : IsCoprime u ↑1
          h₂ : LT.lt (abs (HSub.hSub ξ (HDiv.hDiv ↑u ↑↑1))) (Inv.inv (HMul.hMul (↑↑1) (H …
          ht : LT.lt ξ ↑u
          h₁ : LT.lt 1 (1 / 2)
          hξ₁ : Eq (Int.floor ξ) (HSub.hSub u 1)
          Hξ : Eq ξ ↑(Int.floor ξ)
          ⊢ Exists fun n => Eq (↑u) (ξ.convergent n)
        -/
        exact False.elim (lt_irrefl _ <| h₁.trans one_half_lt_one)
        /-
          🎉 no goals
        -/
      · have hξ₂ : ⌊(fract ξ)⁻¹⌋ = 1 := by
          rw [floor_eq_iff, cast_one, le_inv_comm₀ zero_lt_one (fract_pos.mpr Hξ), inv_one,
            one_add_one_eq_two, inv_lt_comm₀ (fract_pos.mpr Hξ) zero_lt_two]
          refine ⟨(fract_lt_one ξ).le, ?_⟩
          rw [fract, hξ₁, cast_sub, cast_one, lt_sub_iff_add_lt', sub_add]
          convert h₁ using 1
          -- Porting note: added (`convert` handled this in lean 3)
          rw [sub_eq_add_neg]
          norm_num
        /-
          case h.inr.inl.intro.intro.inr.inr
          ξ : Real
          u : Int
          ih : ∀ (m : Nat), LT.lt m 1 → ∀ {ξ : Real} {u : Int}, Real.ContfracLegendre.As …
          left✝ : IsCoprime u ↑1
          h₂ : LT.lt (abs (HSub.hSub ξ (HDiv.hDiv ↑u ↑↑1))) (Inv.inv (HMul.hMul (↑↑1) (H …
          ht : LT.lt ξ ↑u
          h₁ : LT.lt (HAdd.hAdd (↑u) (Neg.neg (1 / 2))) ξ
          hξ₁ : Eq (Int.floor ξ) (HSub.hSub u 1)
          Hξ : Ne ξ ↑(Int.floor ξ)
          hξ₂ : Eq (Int.floor (Inv.inv (Int.fract ξ))) 1
          ⊢ Exists fun n => Eq (↑u) (ξ.convergent n)
        -/
        use 1
        /-
          case h
          ξ : Real
          u : Int
          ih : ∀ (m : Nat), LT.lt m 1 → ∀ {ξ : Real} {u : Int}, Real.ContfracLegendre.As …
          left✝ : IsCoprime u ↑1
          h₂ : LT.lt (abs (HSub.hSub ξ (HDiv.hDiv ↑u ↑↑1))) (Inv.inv (HMul.hMul (↑↑1) (H …
          ht : LT.lt ξ ↑u
          h₁ : LT.lt (HAdd.hAdd (↑u) (Neg.neg (1 / 2))) ξ
          hξ₁ : Eq (Int.floor ξ) (HSub.hSub u 1)
          Hξ : Ne ξ ↑(Int.floor ξ)
          hξ₂ : Eq (Int.floor (Inv.inv (Int.fract ξ))) 1
          ⊢ Eq (↑u) (ξ.convergent 1)
        -/
        simp [convergent, hξ₁, hξ₂, cast_sub, cast_one]
        /-
          🎉 no goals
        -/
    /-
      case h.inr.inr
      v : Nat
      ih : ∀ (m : Nat), LT.lt m v → ∀ {ξ : Real} {u : Int}, Real.ContfracLegendre.As …
      ξ : Real
      u : Int
      h : Real.ContfracLegendre.Ass ξ u ↑v
      ht : LT.lt 1 v
      ⊢ Exists fun n => Eq (HDiv.hDiv ↑u ↑v) (ξ.convergent n)
    -/
  · obtain ⟨huv₀, huv₁⟩ := aux₂ (Nat.cast_le.mpr ht) h
    /-
      case h.inr.inr.intro
      v : Nat
      ih : ∀ (m : Nat), LT.lt m v → ∀ {ξ : Real} {u : Int}, Real.ContfracLegendre.As …
      ξ : Real
      u : Int
      h : Real.ContfracLegendre.Ass ξ u ↑v
      ht : LT.lt 1 v
      huv₀ : LT.lt 0 (HSub.hSub u (HMul.hMul (Int.floor ξ) ↑v))
      huv₁ : LT.lt (HSub.hSub u (HMul.hMul (Int.floor ξ) ↑v)) ↑v
      ⊢ Exists fun n => Eq (HDiv.hDiv ↑u ↑v) (ξ.convergent n)
    -/
    have Hv : (v : ℚ) ≠ 0 := (Nat.cast_pos.mpr (zero_lt_one.trans ht)).ne'
    /-
      case h.inr.inr.intro
      v : Nat
      ih : ∀ (m : Nat), LT.lt m v → ∀ {ξ : Real} {u : Int}, Real.ContfracLegendre.As …
      ξ : Real
      u : Int
      h : Real.ContfracLegendre.Ass ξ u ↑v
      ht : LT.lt 1 v
      huv₀ : LT.lt 0 (HSub.hSub u (HMul.hMul (Int.floor ξ) ↑v))
      huv₁ : LT.lt (HSub.hSub u (HMul.hMul (Int.floor ξ) ↑v)) ↑v
      Hv : Ne (↑v) 0
      ⊢ Exists fun n => Eq (HDiv.hDiv ↑u ↑v) (ξ.convergent n)
    -/
    have huv₁' : (u - ⌊ξ⌋ * v).toNat < v := by zify; rwa [toNat_of_nonneg huv₀.le]
    have inv : ContfracLegendre.Ass (fract ξ)⁻¹ v (u - ⌊ξ⌋ * ↑v).toNat :=
      (toNat_of_nonneg huv₀.le).symm ▸ invariant (Nat.cast_le.mpr ht) h
    /-
      case h.inr.inr.intro
      v : Nat
      ih : ∀ (m : Nat), LT.lt m v → ∀ {ξ : Real} {u : Int}, Real.ContfracLegendre.As …
      ξ : Real
      u : Int
      h : Real.ContfracLegendre.Ass ξ u ↑v
      ht : LT.lt 1 v
      huv₀ : LT.lt 0 (HSub.hSub u (HMul.hMul (Int.floor ξ) ↑v))
      huv₁ : LT.lt (HSub.hSub u (HMul.hMul (Int.floor ξ) ↑v)) ↑v
      Hv : Ne (↑v) 0
      huv₁' : LT.lt (HSub.hSub u (HMul.hMul (Int.floor ξ) ↑v)).toNat v
      inv : Real.ContfracLegendre.Ass (Inv.inv (Int.fract ξ)) ↑v ↑(HSub.hSub u (HMul …
      ⊢ Exists fun n => Eq (HDiv.hDiv ↑u ↑v) (ξ.convergent n)
    -/
    obtain ⟨n, hn⟩ := ih (u - ⌊ξ⌋ * v).toNat huv₁' inv
    /-
      case h.inr.inr.intro.intro
      v : Nat
      ih : ∀ (m : Nat), LT.lt m v → ∀ {ξ : Real} {u : Int}, Real.ContfracLegendre.As …
      ξ : Real
      u : Int
      h : Real.ContfracLegendre.Ass ξ u ↑v
      ht : LT.lt 1 v
      huv₀ : LT.lt 0 (HSub.hSub u (HMul.hMul (Int.floor ξ) ↑v))
      huv₁ : LT.lt (HSub.hSub u (HMul.hMul (Int.floor ξ) ↑v)) ↑v
      Hv : Ne (↑v) 0
      huv₁' : LT.lt (HSub.hSub u (HMul.hMul (Int.floor ξ) ↑v)).toNat v
      inv : Real.ContfracLegendre.Ass (Inv.inv (Int.fract ξ)) ↑v ↑(HSub.hSub u (HMul …
      n : Nat
      hn : Eq (HDiv.hDiv ↑↑v ↑(HSub.hSub u (HMul.hMul (Int.floor ξ) ↑v)).toNat) ((In …
      ⊢ Exists fun n => Eq (HDiv.hDiv ↑u ↑v) (ξ.convergent n)
    -/
    use n + 1
    rw [convergent_succ, ← hn,
      (mod_cast toNat_of_nonneg huv₀.le : ((u - ⌊ξ⌋ * v).toNat : ℚ) = u - ⌊ξ⌋ * v),
      cast_natCast, inv_div, sub_div, mul_div_cancel_right₀ _ Hv, add_sub_cancel]


/-- The main result, *Legendre's Theorem* on rational approximation:
if `ξ` is a real number and `q` is a rational number such that `|ξ - q| < 1/(2*q.den^2)`,
then `q` is a convergent of the continued fraction expansion of `ξ`.
This version uses `Real.convergent`. -/
theorem exists_rat_eq_convergent {q : ℚ} (h : |ξ - q| < 1 / (2 * (q.den : ℝ) ^ 2)) :
    ∃ n, q = ξ.convergent n := by
  /-
    ξ : Real
    q : Rat
    h : LT.lt (abs (HSub.hSub ξ ↑q)) (HDiv.hDiv 1 (HMul.hMul 2 (HPow.hPow (↑q.den) …
    ⊢ Exists fun n => Eq q (ξ.convergent n)
  -/
  refine q.num_div_den ▸ exists_rat_eq_convergent' ⟨?_, fun hd => ?_, ?_⟩
    /-
      case refine_1
      ξ : Real
      q : Rat
      h : LT.lt (abs (HSub.hSub ξ ↑q)) (HDiv.hDiv 1 (HMul.hMul 2 (HPow.hPow (↑q.den) …
      ⊢ IsCoprime q.num ↑q.den
    -/
  · exact coprime_iff_nat_coprime.mpr (natAbs_ofNat q.den ▸ q.reduced)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      ξ : Real
      q : Rat
      h : LT.lt (abs (HSub.hSub ξ ↑q)) (HDiv.hDiv 1 (HMul.hMul 2 (HPow.hPow (↑q.den) …
      hd : Eq (↑q.den) 1
      ⊢ LT.lt (Neg.neg (1 / 2)) (HSub.hSub ξ ↑q.num)
    -/
  · rw [← q.den_eq_one_iff.mp (Nat.cast_eq_one.mp hd)] at h
    /-
      case refine_2
      ξ : Real
      q : Rat
      h : LT.lt (abs (HSub.hSub ξ ↑↑q.num)) (HDiv.hDiv 1 (HMul.hMul 2 (HPow.hPow (↑( …
      hd : Eq (↑q.den) 1
      ⊢ LT.lt (Neg.neg (1 / 2)) (HSub.hSub ξ ↑q.num)
    -/
    simpa only [Rat.den_intCast, Nat.cast_one, one_pow, mul_one] using (abs_lt.mp h).1
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      ξ : Real
      q : Rat
      h : LT.lt (abs (HSub.hSub ξ ↑q)) (HDiv.hDiv 1 (HMul.hMul 2 (HPow.hPow (↑q.den) …
      ⊢ LT.lt (abs (HSub.hSub ξ (HDiv.hDiv ↑q.num ↑↑q.den))) (Inv.inv (HMul.hMul (↑↑ …
    -/
  · obtain ⟨hq₀, hq₁⟩ := aux₀ (Nat.cast_pos.mpr q.pos)
    /-
      case refine_3.intro
      ξ : Real
      q : Rat
      h : LT.lt (abs (HSub.hSub ξ ↑q)) (HDiv.hDiv 1 (HMul.hMul 2 (HPow.hPow (↑q.den) …
      hq₀ : LT.lt 0 ↑↑q.den
      hq₁ : LT.lt 0 (HSub.hSub (HMul.hMul 2 ↑↑q.den) 1)
      ⊢ LT.lt (abs (HSub.hSub ξ (HDiv.hDiv ↑q.num ↑↑q.den))) (Inv.inv (HMul.hMul (↑↑ …
    -/
    replace hq₁ := mul_pos hq₀ hq₁
    /-
      case refine_3.intro
      ξ : Real
      q : Rat
      h : LT.lt (abs (HSub.hSub ξ ↑q)) (HDiv.hDiv 1 (HMul.hMul 2 (HPow.hPow (↑q.den) …
      hq₀ : LT.lt 0 ↑↑q.den
      hq₁ : LT.lt 0 (HMul.hMul (↑↑q.den) (HSub.hSub (HMul.hMul 2 ↑↑q.den) 1))
      ⊢ LT.lt (abs (HSub.hSub ξ (HDiv.hDiv ↑q.num ↑↑q.den))) (Inv.inv (HMul.hMul (↑↑ …
    -/
    have hq₂ : (0 : ℝ) < 2 * (q.den * q.den) := mul_pos zero_lt_two (mul_pos hq₀ hq₀)
    /-
      case refine_3.intro
      ξ : Real
      q : Rat
      h : LT.lt (abs (HSub.hSub ξ ↑q)) (HDiv.hDiv 1 (HMul.hMul 2 (HPow.hPow (↑q.den) …
      hq₀ : LT.lt 0 ↑↑q.den
      hq₁ : LT.lt 0 (HMul.hMul (↑↑q.den) (HSub.hSub (HMul.hMul 2 ↑↑q.den) 1))
      hq₂ : LT.lt 0 (HMul.hMul 2 (HMul.hMul ↑q.den ↑q.den))
      ⊢ LT.lt (abs (HSub.hSub ξ (HDiv.hDiv ↑q.num ↑↑q.den))) (Inv.inv (HMul.hMul (↑↑ …
    -/
    rw [cast_natCast] at *
    /-
      case refine_3.intro
      ξ : Real
      q : Rat
      h : LT.lt (abs (HSub.hSub ξ ↑q)) (HDiv.hDiv 1 (HMul.hMul 2 (HPow.hPow (↑q.den) …
      hq₀ : LT.lt 0 ↑q.den
      hq₁ : LT.lt 0 (HMul.hMul (↑q.den) (HSub.hSub (HMul.hMul 2 ↑q.den) 1))
      hq₂ : LT.lt 0 (HMul.hMul 2 (HMul.hMul ↑q.den ↑q.den))
      ⊢ LT.lt (abs (HSub.hSub ξ (HDiv.hDiv ↑q.num ↑q.den))) (Inv.inv (HMul.hMul (↑q. …
    -/
    rw [(by norm_cast : (q.num / q.den : ℝ) = (q.num / q.den : ℚ)), Rat.num_div_den]
    /-
      case refine_3.intro
      ξ : Real
      q : Rat
      h : LT.lt (abs (HSub.hSub ξ ↑q)) (HDiv.hDiv 1 (HMul.hMul 2 (HPow.hPow (↑q.den) …
      hq₀ : LT.lt 0 ↑q.den
      hq₁ : LT.lt 0 (HMul.hMul (↑q.den) (HSub.hSub (HMul.hMul 2 ↑q.den) 1))
      hq₂ : LT.lt 0 (HMul.hMul 2 (HMul.hMul ↑q.den ↑q.den))
      ⊢ LT.lt (abs (HSub.hSub ξ ↑q)) (Inv.inv (HMul.hMul (↑q.den) (HSub.hSub (HMul.h …
    -/
    exact h.trans (by rw [← one_div, sq, one_div_lt_one_div hq₂ hq₁, ← sub_pos]; ring_nf; exact hq₀)
    /-
      🎉 no goals
    -/


