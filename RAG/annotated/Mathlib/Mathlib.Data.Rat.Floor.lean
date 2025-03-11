protected theorem floor_def' (a : ℚ) : a.floor = a.num / a.den := by
  /-
    a : Rat
    ⊢ Eq a.floor (HDiv.hDiv a.num ↑a.den)
  -/
  rw [Rat.floor]
  /-
    a : Rat
    ⊢ Eq (ite (Eq a.den 1) a.num (HDiv.hDiv a.num ↑a.den)) (HDiv.hDiv a.num ↑a.den)
  -/
  split
    /-
      case isTrue
      a : Rat
      h✝ : Eq a.den 1
      ⊢ Eq a.num (HDiv.hDiv a.num ↑a.den)
    -/
  · next h => simp [h]
    /-
      🎉 no goals
    -/
    /-
      case isFalse
      a : Rat
      h✝ : Not (Eq a.den 1)
      ⊢ Eq (HDiv.hDiv a.num ↑a.den) (HDiv.hDiv a.num ↑a.den)
    -/
  · next => rfl
    /-
      🎉 no goals
    -/


protected theorem le_floor {z : ℤ} : ∀ {r : ℚ}, z ≤ Rat.floor r ↔ (z : ℚ) ≤ r
  | ⟨n, d, h, c⟩ => by
    /-
      z n : Int
      d : Nat
      h : Ne d 0
      c : n.natAbs.Coprime d
      ⊢ Iff (LE.le z { num := n, den := d, den_nz := h, reduced := c }.floor) (LE.le …
    -/
    simp only [Rat.floor_def']
    /-
      z n : Int
      d : Nat
      h : Ne d 0
      c : n.natAbs.Coprime d
      ⊢ Iff (LE.le z (HDiv.hDiv n ↑d)) (LE.le ↑z { num := n, den := d, den_nz := h,  …
    -/
    rw [mk'_eq_divInt]
    /-
      z n : Int
      d : Nat
      h : Ne d 0
      c : n.natAbs.Coprime d
      ⊢ Iff (LE.le z (HDiv.hDiv n ↑d)) (LE.le (↑z) (Rat.divInt n ↑d))
    -/
    have h' := Int.ofNat_lt.2 (Nat.pos_of_ne_zero h)
    conv =>
      rhs
      rw [intCast_eq_divInt, Rat.divInt_le_divInt zero_lt_one h', mul_one]
    /-
      z n : Int
      d : Nat
      h : Ne d 0
      c : n.natAbs.Coprime d
      h' : LT.lt ↑0 ↑d
      ⊢ Iff (LE.le z (HDiv.hDiv n ↑d)) (LE.le (HMul.hMul z ↑d) n)
    -/
    exact Int.le_ediv_iff_mul_le h'
    /-
      🎉 no goals
    -/


instance : FloorRing ℚ :=
  (FloorRing.ofFloor ℚ Rat.floor) fun _ _ => Rat.le_floor.symm


protected theorem floor_def {q : ℚ} : ⌊q⌋ = q.num / q.den := Rat.floor_def' q


protected theorem ceil_def (q : ℚ) : ⌈q⌉ = -(-q.num / ↑q.den) := by
  /-
    q : Rat
    ⊢ Eq (Int.ceil q) (Neg.neg (HDiv.hDiv (Neg.neg q.num) ↑q.den))
  -/
  change -⌊-q⌋ = _
  /-
    q : Rat
    ⊢ Eq (Neg.neg (Int.floor (Neg.neg q))) (Neg.neg (HDiv.hDiv (Neg.neg q.num) ↑q. …
  -/
  rw [Rat.floor_def, num_neg_eq_neg_num, den_neg_eq_den]
  /-
    🎉 no goals
  -/



@[norm_cast]
theorem floor_intCast_div_natCast (n : ℤ) (d : ℕ) : ⌊(↑n / ↑d : ℚ)⌋ = n / (↑d : ℤ) := by
  /-
    n : Int
    d : Nat
    ⊢ Eq (Int.floor (HDiv.hDiv ↑n ↑d)) (HDiv.hDiv n ↑d)
  -/
  rw [Rat.floor_def]
  /-
    n : Int
    d : Nat
    ⊢ Eq (HDiv.hDiv (HDiv.hDiv ↑n ↑d).num ↑(HDiv.hDiv ↑n ↑d).den) (HDiv.hDiv n ↑d)
  -/
  obtain rfl | hd := @eq_zero_or_pos _ _ d
    /-
      case inl
      n : Int
      ⊢ Eq (HDiv.hDiv (HDiv.hDiv ↑n ↑0).num ↑(HDiv.hDiv ↑n ↑0).den) (HDiv.hDiv n ↑0)
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    n : Int
    d : Nat
    hd : LT.lt 0 d
    ⊢ Eq (HDiv.hDiv (HDiv.hDiv ↑n ↑d).num ↑(HDiv.hDiv ↑n ↑d).den) (HDiv.hDiv n ↑d)
  -/
  set q := (n : ℚ) / d with q_eq
  obtain ⟨c, n_eq_c_mul_num, d_eq_c_mul_denom⟩ : ∃ c, n = c * q.num ∧ (d : ℤ) = c * q.den := by
    rw [q_eq]
    exact mod_cast @Rat.exists_eq_mul_div_num_and_eq_mul_div_den n d (mod_cast hd.ne')
  /-
    case inr.intro.intro
    n : Int
    d : Nat
    hd : LT.lt 0 d
    q : Rat := HDiv.hDiv ↑n ↑d
    q_eq : Eq q (HDiv.hDiv ↑n ↑d)
    c : Int
    n_eq_c_mul_num : Eq n (HMul.hMul c q.num)
    d_eq_c_mul_denom : Eq (↑d) (HMul.hMul c ↑q.den)
    ⊢ Eq (HDiv.hDiv q.num ↑q.den) (HDiv.hDiv n ↑d)
  -/
  rw [n_eq_c_mul_num, d_eq_c_mul_denom]
  /-
    case inr.intro.intro
    n : Int
    d : Nat
    hd : LT.lt 0 d
    q : Rat := HDiv.hDiv ↑n ↑d
    q_eq : Eq q (HDiv.hDiv ↑n ↑d)
    c : Int
    n_eq_c_mul_num : Eq n (HMul.hMul c q.num)
    d_eq_c_mul_denom : Eq (↑d) (HMul.hMul c ↑q.den)
    ⊢ Eq (HDiv.hDiv q.num ↑q.den) (HDiv.hDiv (HMul.hMul c q.num) (HMul.hMul c ↑q.d …
  -/
  refine (Int.mul_ediv_mul_of_pos _ _ <| pos_of_mul_pos_left ?_ <| Int.natCast_nonneg q.den).symm
  /-
    case inr.intro.intro
    n : Int
    d : Nat
    hd : LT.lt 0 d
    q : Rat := HDiv.hDiv ↑n ↑d
    q_eq : Eq q (HDiv.hDiv ↑n ↑d)
    c : Int
    n_eq_c_mul_num : Eq n (HMul.hMul c q.num)
    d_eq_c_mul_denom : Eq (↑d) (HMul.hMul c ↑q.den)
    ⊢ LT.lt 0 (HMul.hMul c ↑q.den)
  -/
  rwa [← d_eq_c_mul_denom, Int.natCast_pos]
  /-
    🎉 no goals
  -/


@[norm_cast]
theorem ceil_intCast_div_natCast (n : ℤ) (d : ℕ) : ⌈(↑n / ↑d : ℚ)⌉ = -((-n) / (↑d : ℤ)) := by
  /-
    n : Int
    d : Nat
    ⊢ Eq (Int.ceil (HDiv.hDiv ↑n ↑d)) (Neg.neg (HDiv.hDiv (Neg.neg n) ↑d))
  -/
  conv_lhs => rw [← neg_neg ⌈_⌉, ← floor_neg]
  /-
    n : Int
    d : Nat
    ⊢ Eq (Neg.neg (Int.floor (Neg.neg (HDiv.hDiv ↑n ↑d)))) (Neg.neg (HDiv.hDiv (Ne …
  -/
  rw [← neg_div, ← Int.cast_neg, floor_intCast_div_natCast]
  /-
    🎉 no goals
  -/


@[norm_cast]
theorem floor_natCast_div_natCast (n d : ℕ) : ⌊(↑n / ↑d : ℚ)⌋ = n / d :=
  floor_intCast_div_natCast n d


@[norm_cast]
theorem ceil_natCast_div_natCast (n d : ℕ) : ⌈(↑n / ↑d : ℚ)⌉ = -((-n) / d) :=
  ceil_intCast_div_natCast n d


@[norm_cast]
theorem natFloor_natCast_div_natCast (n d : ℕ) : ⌊(↑n / ↑d : ℚ)⌋₊ = n / d := by
  /-
    n d : Nat
    ⊢ Eq (Nat.floor (HDiv.hDiv ↑n ↑d)) (HDiv.hDiv n d)
  -/
  rw [← Int.ofNat_inj, Int.natCast_floor_eq_floor (by positivity)]
  /-
    n d : Nat
    ⊢ Eq (Int.floor (HDiv.hDiv ↑n ↑d)) ↑(HDiv.hDiv n d)
  -/
  push_cast
  /-
    n d : Nat
    ⊢ Eq (Int.floor (HDiv.hDiv ↑n ↑d)) (HDiv.hDiv ↑n ↑d)
  -/
  exact floor_intCast_div_natCast n d
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-23")] alias floor_int_div_nat_eq_div := floor_intCast_div_natCast


@[simp, norm_cast]
theorem floor_cast (x : ℚ) : ⌊(x : α)⌋ = ⌊x⌋ :=
  floor_eq_iff.2 (mod_cast floor_eq_iff.1 (Eq.refl ⌊x⌋))


@[simp, norm_cast]
theorem ceil_cast (x : ℚ) : ⌈(x : α)⌉ = ⌈x⌉ := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrderedField α
    inst✝ : FloorRing α
    x : Rat
    ⊢ Eq (Int.ceil ↑x) (Int.ceil x)
  -/
  rw [← neg_inj, ← floor_neg, ← floor_neg, ← Rat.cast_neg, Rat.floor_cast]
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem round_cast (x : ℚ) : round (x : α) = round x := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrderedField α
    inst✝ : FloorRing α
    x : Rat
    ⊢ Eq (round ↑x) (round x)
  -/
  have : ((x + 1 / 2 : ℚ) : α) = x + 1 / 2 := by simp
  /-
    α : Type u_1
    inst✝¹ : LinearOrderedField α
    inst✝ : FloorRing α
    x : Rat
    this : Eq (↑(HAdd.hAdd x (1 / 2))) (HAdd.hAdd (↑x) (1 / 2))
    ⊢ Eq (round ↑x) (round x)
  -/
  rw [round_eq, round_eq, ← this, floor_cast]
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem cast_fract (x : ℚ) : (↑(fract x) : α) = fract (x : α) := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrderedField α
    inst✝ : FloorRing α
    x : Rat
    ⊢ Eq (↑(Int.fract x)) (Int.fract ↑x)
  -/
  simp only [fract, cast_sub, cast_intCast, floor_cast]
  /-
    🎉 no goals
  -/


theorem isNat_intFloor {R} [LinearOrderedRing R] [FloorRing R] (r : R) (m : ℕ) :
                                  /-
                                    R : Type u_2
                                    inst✝¹ : LinearOrderedRing R
                                    inst✝ : FloorRing R
                                    r : R
                                    m : Nat
                                    ⊢ Mathlib.Meta.NormNum.IsNat r m → Mathlib.Meta.NormNum.IsNat (Int.floor r) m
                                  -/
    IsNat r m → IsNat ⌊r⌋ m := by rintro ⟨⟨⟩⟩; exact ⟨by simp⟩
                                               /-
                                                 🎉 no goals
                                               -/


theorem isInt_intFloor {R} [LinearOrderedRing R] [FloorRing R] (r : R) (m : ℤ) :
                                  /-
                                    R : Type u_2
                                    inst✝¹ : LinearOrderedRing R
                                    inst✝ : FloorRing R
                                    r : R
                                    m : Int
                                    ⊢ Mathlib.Meta.NormNum.IsInt r m → Mathlib.Meta.NormNum.IsInt (Int.floor r) m
                                  -/
    IsInt r m → IsInt ⌊r⌋ m := by rintro ⟨⟨⟩⟩; exact ⟨by simp⟩
                                               /-
                                                 🎉 no goals
                                               -/


theorem isInt_intFloor_ofIsRat (r : α) (n : ℤ) (d : ℕ) :
    IsRat r n d → IsInt ⌊r⌋ (n / d) := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrderedField α
    inst✝ : FloorRing α
    r : α
    n : Int
    d : Nat
    ⊢ Mathlib.Meta.NormNum.IsRat r n d → Mathlib.Meta.NormNum.IsInt (Int.floor r)  …
  -/
  rintro ⟨inv, rfl⟩
  /-
    case mk
    α : Type u_1
    inst✝¹ : LinearOrderedField α
    inst✝ : FloorRing α
    n : Int
    d : Nat
    inv : Invertible ↑d
    ⊢ Mathlib.Meta.NormNum.IsInt (Int.floor (HMul.hMul (↑n) (Invertible.invOf ↑d)) …
  -/
  constructor
  /-
    case mk.out
    α : Type u_1
    inst✝¹ : LinearOrderedField α
    inst✝ : FloorRing α
    n : Int
    d : Nat
    inv : Invertible ↑d
    ⊢ Eq (Int.floor (HMul.hMul (↑n) (Invertible.invOf ↑d))) ↑(HDiv.hDiv n ↑d)
  -/
  simp only [invOf_eq_inv, ← div_eq_mul_inv, Int.cast_id]
  rw [← floor_intCast_div_natCast n d, ← floor_cast (α := α), Rat.cast_div,
    cast_intCast, cast_natCast]


/-- `norm_num` extension for `Int.floor` -/
@[norm_num ⌊_⌋]
def evalIntFloor : NormNumExt where eval {u αZ} e := do
  match u, αZ, e with
  | 0, ~q(ℤ), ~q(@Int.floor $α $instR $instF $x) =>
    match ← derive x with
    | .isBool .. => failure
    | .isNat _ _ pb => do
      assertInstancesCommute
      return .isNat q(inferInstance) _ q(isNat_intFloor $x _ $pb)
    | .isNegNat _ _ pb => do
      assertInstancesCommute
      -- floor always keeps naturals negative, so we can shortcut `.isInt`
      return .isNegNat q(inferInstance) _ q(isInt_intFloor _ _ $pb)
    | .isRat _ q n d h => do
      let _i ← synthInstanceQ q(LinearOrderedField $α)
      assertInstancesCommute
      have z : Q(ℤ) := mkRawIntLit ⌊q⌋
      letI : $z =Q $n / $d := ⟨⟩
      return .isInt q(inferInstance) z ⌊q⌋ q(isInt_intFloor_ofIsRat _ $n $d $h)
  | _, _, _ => failure


theorem isNat_intCeil {R} [LinearOrderedRing R] [FloorRing R] (r : R) (m : ℕ) :
                                  /-
                                    R : Type u_2
                                    inst✝¹ : LinearOrderedRing R
                                    inst✝ : FloorRing R
                                    r : R
                                    m : Nat
                                    ⊢ Mathlib.Meta.NormNum.IsNat r m → Mathlib.Meta.NormNum.IsNat (Int.ceil r) m
                                  -/
    IsNat r m → IsNat ⌈r⌉ m := by rintro ⟨⟨⟩⟩; exact ⟨by simp⟩
                                               /-
                                                 🎉 no goals
                                               -/


theorem isInt_intCeil {R} [LinearOrderedRing R] [FloorRing R] (r : R) (m : ℤ) :
                                  /-
                                    R : Type u_2
                                    inst✝¹ : LinearOrderedRing R
                                    inst✝ : FloorRing R
                                    r : R
                                    m : Int
                                    ⊢ Mathlib.Meta.NormNum.IsInt r m → Mathlib.Meta.NormNum.IsInt (Int.ceil r) m
                                  -/
    IsInt r m → IsInt ⌈r⌉ m := by rintro ⟨⟨⟩⟩; exact ⟨by simp⟩
                                               /-
                                                 🎉 no goals
                                               -/


theorem isInt_intCeil_ofIsRat (r : α) (n : ℤ) (d : ℕ) :
    IsRat r n d → IsInt ⌈r⌉ (-(-n / d)) := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrderedField α
    inst✝ : FloorRing α
    r : α
    n : Int
    d : Nat
    ⊢ Mathlib.Meta.NormNum.IsRat r n d → Mathlib.Meta.NormNum.IsInt (Int.ceil r) ( …
  -/
  rintro ⟨inv, rfl⟩
  /-
    case mk
    α : Type u_1
    inst✝¹ : LinearOrderedField α
    inst✝ : FloorRing α
    n : Int
    d : Nat
    inv : Invertible ↑d
    ⊢ Mathlib.Meta.NormNum.IsInt (Int.ceil (HMul.hMul (↑n) (Invertible.invOf ↑d))) …
  -/
  constructor
  /-
    case mk.out
    α : Type u_1
    inst✝¹ : LinearOrderedField α
    inst✝ : FloorRing α
    n : Int
    d : Nat
    inv : Invertible ↑d
    ⊢ Eq (Int.ceil (HMul.hMul (↑n) (Invertible.invOf ↑d))) ↑(Neg.neg (HDiv.hDiv (N …
  -/
  simp only [invOf_eq_inv, ← div_eq_mul_inv, Int.cast_id]
  rw [← ceil_intCast_div_natCast n d, ← ceil_cast (α := α), Rat.cast_div,
    cast_intCast, cast_natCast]


/-- `norm_num` extension for `Int.ceil` -/
@[norm_num ⌈_⌉]
def evalIntCeil : NormNumExt where eval {u αZ} e := do
  match u, αZ, e with
  | 0, ~q(ℤ), ~q(@Int.ceil $α $instR $instF $x) =>
    match ← derive x with
    | .isBool .. => failure
    | .isNat _ _ pb => do
      assertInstancesCommute
      return .isNat q(inferInstance) _ q(isNat_intCeil $x _ $pb)
    | .isNegNat _ _ pb => do
      assertInstancesCommute
      -- ceil always keeps naturals negative, so we can shortcut `.isInt`
      return .isNegNat q(inferInstance) _ q(isInt_intCeil _ _ $pb)
    | .isRat _ q n d h => do
      let _i ← synthInstanceQ q(LinearOrderedField $α)
      assertInstancesCommute
      have z : Q(ℤ) := mkRawIntLit ⌈q⌉
      letI : $z =Q (-(-$n / $d)) := ⟨⟩
      return .isInt q(inferInstance) z ⌈q⌉ q(isInt_intCeil_ofIsRat _ $n $d $h)
  | _, _, _ => failure


theorem Int.mod_nat_eq_sub_mul_floor_rat_div {n : ℤ} {d : ℕ} : n % d = n - d * ⌊(n : ℚ) / d⌋ := by
  /-
    n : Int
    d : Nat
    ⊢ Eq (HMod.hMod n ↑d) (HSub.hSub n (HMul.hMul (↑d) (Int.floor (HDiv.hDiv ↑n ↑d …
  -/
  rw [eq_sub_of_add_eq <| Int.emod_add_ediv n d, Rat.floor_intCast_div_natCast]
  /-
    🎉 no goals
  -/


theorem Nat.coprime_sub_mul_floor_rat_div_of_coprime {n d : ℕ} (n_coprime_d : n.Coprime d) :
    ((n : ℤ) - d * ⌊(n : ℚ) / d⌋).natAbs.Coprime d := by
  /-
    n d : Nat
    n_coprime_d : n.Coprime d
    ⊢ (HSub.hSub (↑n) (HMul.hMul (↑d) (Int.floor (HDiv.hDiv ↑n ↑d)))).natAbs.Copri …
  -/
  have : (n : ℤ) % d = n - d * ⌊(n : ℚ) / d⌋ := Int.mod_nat_eq_sub_mul_floor_rat_div
  /-
    n d : Nat
    n_coprime_d : n.Coprime d
    this : Eq (HMod.hMod ↑n ↑d) (HSub.hSub (↑n) (HMul.hMul (↑d) (Int.floor (HDiv.h …
    ⊢ (HSub.hSub (↑n) (HMul.hMul (↑d) (Int.floor (HDiv.hDiv ↑n ↑d)))).natAbs.Copri …
  -/
  rw [← this]
  /-
    n d : Nat
    n_coprime_d : n.Coprime d
    this : Eq (HMod.hMod ↑n ↑d) (HSub.hSub (↑n) (HMul.hMul (↑d) (Int.floor (HDiv.h …
    ⊢ (HMod.hMod ↑n ↑d).natAbs.Coprime d
  -/
  have : d.Coprime n := n_coprime_d.symm
  /-
    n d : Nat
    n_coprime_d : n.Coprime d
    this✝ : Eq (HMod.hMod ↑n ↑d) (HSub.hSub (↑n) (HMul.hMul (↑d) (Int.floor (HDiv. …
    this : d.Coprime n
    ⊢ (HMod.hMod ↑n ↑d).natAbs.Coprime d
  -/
  rwa [Nat.Coprime, Nat.gcd_rec] at this
  /-
    🎉 no goals
  -/


theorem num_lt_succ_floor_mul_den (q : ℚ) : q.num < (⌊q⌋ + 1) * q.den := by
  /-
    q : Rat
    ⊢ LT.lt q.num (HMul.hMul (HAdd.hAdd (Int.floor q) 1) ↑q.den)
  -/
  suffices (q.num : ℚ) < (⌊q⌋ + 1) * q.den from mod_cast this
  suffices (q.num : ℚ) < (q - fract q + 1) * q.den by
    have : (⌊q⌋ : ℚ) = q - fract q := eq_sub_of_add_eq <| floor_add_fract q
    rwa [this]
  suffices (q.num : ℚ) < q.num + (1 - fract q) * q.den by
    have : (q - fract q + 1) * q.den = q.num + (1 - fract q) * q.den := by
      calc
        (q - fract q + 1) * q.den = (q + (1 - fract q)) * q.den := by ring
        _ = q * q.den + (1 - fract q) * q.den := by rw [add_mul]
        _ = q.num + (1 - fract q) * q.den := by simp
    rwa [this]
  suffices 0 < (1 - fract q) * q.den by
    rw [← sub_lt_iff_lt_add']
    simpa
  have : 0 < 1 - fract q := by
    have : fract q < 1 := fract_lt_one q
    have : 0 + fract q < 1 := by simp [this]
    rwa [lt_sub_iff_add_lt]
  /-
    q : Rat
    this : LT.lt 0 (HSub.hSub 1 (Int.fract q))
    ⊢ LT.lt 0 (HMul.hMul (HSub.hSub 1 (Int.fract q)) ↑q.den)
  -/
  exact mul_pos this (by exact mod_cast q.pos)
  /-
    🎉 no goals
  -/


theorem fract_inv_num_lt_num_of_pos {q : ℚ} (q_pos : 0 < q) : (fract q⁻¹).num < q.num := by
  -- we know that the numerator must be positive
  /-
    q : Rat
    q_pos : LT.lt 0 q
    ⊢ LT.lt (Int.fract (Inv.inv q)).num q.num
  -/
  have q_num_pos : 0 < q.num := Rat.num_pos.mpr q_pos
  -- we will work with the absolute value of the numerator, which is equal to the numerator
  /-
    q : Rat
    q_pos : LT.lt 0 q
    q_num_pos : LT.lt 0 q.num
    ⊢ LT.lt (Int.fract (Inv.inv q)).num q.num
  -/
  have q_num_abs_eq_q_num : (q.num.natAbs : ℤ) = q.num := Int.natAbs_of_nonneg q_num_pos.le
  /-
    q : Rat
    q_pos : LT.lt 0 q
    q_num_pos : LT.lt 0 q.num
    q_num_abs_eq_q_num : Eq (↑q.num.natAbs) q.num
    ⊢ LT.lt (Int.fract (Inv.inv q)).num q.num
  -/
  set q_inv : ℚ := q.den / q.num with q_inv_def
  /-
    q : Rat
    q_pos : LT.lt 0 q
    q_num_pos : LT.lt 0 q.num
    q_num_abs_eq_q_num : Eq (↑q.num.natAbs) q.num
    q_inv : Rat := HDiv.hDiv ↑q.den ↑q.num
    q_inv_def : Eq q_inv (HDiv.hDiv ↑q.den ↑q.num)
    ⊢ LT.lt (Int.fract (Inv.inv q)).num q.num
  -/
  have q_inv_eq : q⁻¹ = q_inv := by rw [q_inv_def, inv_def', divInt_eq_div, Int.cast_natCast]
  /-
    q : Rat
    q_pos : LT.lt 0 q
    q_num_pos : LT.lt 0 q.num
    q_num_abs_eq_q_num : Eq (↑q.num.natAbs) q.num
    q_inv : Rat := HDiv.hDiv ↑q.den ↑q.num
    q_inv_def : Eq q_inv (HDiv.hDiv ↑q.den ↑q.num)
    q_inv_eq : Eq (Inv.inv q) q_inv
    ⊢ LT.lt (Int.fract (Inv.inv q)).num q.num
  -/
  suffices (q_inv - ⌊q_inv⌋).num < q.num by rwa [q_inv_eq]
  suffices ((q.den - q.num * ⌊q_inv⌋ : ℚ) / q.num).num < q.num by
    field_simp [q_inv, this, ne_of_gt q_num_pos]
  suffices (q.den : ℤ) - q.num * ⌊q_inv⌋ < q.num by
    -- use that `q.num` and `q.den` are coprime to show that the numerator stays unreduced
    have : ((q.den - q.num * ⌊q_inv⌋ : ℚ) / q.num).num = q.den - q.num * ⌊q_inv⌋ := by
      suffices ((q.den : ℤ) - q.num * ⌊q_inv⌋).natAbs.Coprime q.num.natAbs from
        mod_cast Rat.num_div_eq_of_coprime q_num_pos this
      have tmp := Nat.coprime_sub_mul_floor_rat_div_of_coprime q.reduced.symm
      simpa only [Nat.cast_natAbs, abs_of_nonneg q_num_pos.le] using tmp
    rwa [this]
  -- to show the claim, start with the following inequality
  have q_inv_num_denom_ineq : q⁻¹.num - ⌊q⁻¹⌋ * q⁻¹.den < q⁻¹.den := by
    have : q⁻¹.num < (⌊q⁻¹⌋ + 1) * q⁻¹.den := Rat.num_lt_succ_floor_mul_den q⁻¹
    have : q⁻¹.num < ⌊q⁻¹⌋ * q⁻¹.den + q⁻¹.den := by rwa [right_distrib, one_mul] at this
    rwa [← sub_lt_iff_lt_add'] at this
  -- use that `q.num` and `q.den` are coprime to show that q_inv is the unreduced reciprocal
  -- of `q`
  have : q_inv.num = q.den ∧ q_inv.den = q.num.natAbs := by
    have coprime_q_denom_q_num : q.den.Coprime q.num.natAbs := q.reduced.symm
    have : Int.natAbs q.den = q.den := by simp
    rw [← this] at coprime_q_denom_q_num
    rw [q_inv_def]
    constructor
    · exact mod_cast Rat.num_div_eq_of_coprime q_num_pos coprime_q_denom_q_num
    · suffices (((q.den : ℚ) / q.num).den : ℤ) = q.num.natAbs by exact mod_cast this
      rw [q_num_abs_eq_q_num]
      exact mod_cast Rat.den_div_eq_of_coprime q_num_pos coprime_q_denom_q_num
  /-
    q : Rat
    q_pos : LT.lt 0 q
    q_num_pos : LT.lt 0 q.num
    q_num_abs_eq_q_num : Eq (↑q.num.natAbs) q.num
    q_inv : Rat := HDiv.hDiv ↑q.den ↑q.num
    q_inv_def : Eq q_inv (HDiv.hDiv ↑q.den ↑q.num)
    q_inv_eq : Eq (Inv.inv q) q_inv
    q_inv_num_denom_ineq : LT.lt (HSub.hSub (Inv.inv q).num (HMul.hMul (Int.floor  …
    this : And (Eq q_inv.num ↑q.den) (Eq q_inv.den q.num.natAbs)
    ⊢ LT.lt (HSub.hSub (↑q.den) (HMul.hMul q.num (Int.floor q_inv))) q.num
  -/
  rwa [q_inv_eq, this.left, this.right, q_num_abs_eq_q_num, mul_comm] at q_inv_num_denom_ineq
  /-
    🎉 no goals
  -/


