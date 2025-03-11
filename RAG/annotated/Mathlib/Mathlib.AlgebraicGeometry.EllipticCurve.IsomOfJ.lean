omit [E.IsElliptic] [E'.IsElliptic] in
private lemma exists_variableChange_of_char_two_of_j_ne_zero
    [E.IsCharTwoJNeZeroNF] [E'.IsCharTwoJNeZeroNF] (heq : E.a₆ = E'.a₆) :
    ∃ C : VariableChange F, E.variableChange C = E' := by
  obtain ⟨s, hs⟩ := IsSepClosed.exists_root_C_mul_X_pow_add_C_mul_X_add_C' 2 2
    1 1 (E.a₂ + E'.a₂) (by norm_num) (by norm_num) one_ne_zero
  /-
    case intro
    F : Type u_1
    inst✝⁴ : Field F
    inst✝³ : IsSepClosed F
    E E' : WeierstrassCurve F
    inst✝² : CharP F 2
    inst✝¹ : E.IsCharTwoJNeZeroNF
    inst✝ : E'.IsCharTwoJNeZeroNF
    heq : Eq E.a₆ E'.a₆
    s : F
    hs : Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul 1 (HPow.hPow s 2)) (HMul.hMul 1 s)) ( …
    ⊢ Exists fun C => Eq (E.variableChange C) E'
  -/
  use ⟨1, 0, s, 0⟩
  /-
    case h
    F : Type u_1
    inst✝⁴ : Field F
    inst✝³ : IsSepClosed F
    E E' : WeierstrassCurve F
    inst✝² : CharP F 2
    inst✝¹ : E.IsCharTwoJNeZeroNF
    inst✝ : E'.IsCharTwoJNeZeroNF
    heq : Eq E.a₆ E'.a₆
    s : F
    hs : Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul 1 (HPow.hPow s 2)) (HMul.hMul 1 s)) ( …
    ⊢ Eq (E.variableChange { u := 1, r := 0, s := s, t := 0 }) E'
  -/
  ext
    /-
      case h.a₁
      F : Type u_1
      inst✝⁴ : Field F
      inst✝³ : IsSepClosed F
      E E' : WeierstrassCurve F
      inst✝² : CharP F 2
      inst✝¹ : E.IsCharTwoJNeZeroNF
      inst✝ : E'.IsCharTwoJNeZeroNF
      heq : Eq E.a₆ E'.a₆
      s : F
      hs : Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul 1 (HPow.hPow s 2)) (HMul.hMul 1 s)) ( …
      ⊢ Eq (E.variableChange { u := 1, r := 0, s := s, t := 0 }).a₁ E'.a₁
    -/
  · simp_rw [variableChange_a₁, inv_one, Units.val_one, a₁_of_isCharTwoJNeZeroNF]
    /-
      case h.a₁
      F : Type u_1
      inst✝⁴ : Field F
      inst✝³ : IsSepClosed F
      E E' : WeierstrassCurve F
      inst✝² : CharP F 2
      inst✝¹ : E.IsCharTwoJNeZeroNF
      inst✝ : E'.IsCharTwoJNeZeroNF
      heq : Eq E.a₆ E'.a₆
      s : F
      hs : Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul 1 (HPow.hPow s 2)) (HMul.hMul 1 s)) ( …
      ⊢ Eq (HMul.hMul 1 (HAdd.hAdd 1 (HMul.hMul 2 s))) 1
    -/
    linear_combination s * CharP.cast_eq_zero F 2
    /-
      🎉 no goals
    -/
    /-
      case h.a₂
      F : Type u_1
      inst✝⁴ : Field F
      inst✝³ : IsSepClosed F
      E E' : WeierstrassCurve F
      inst✝² : CharP F 2
      inst✝¹ : E.IsCharTwoJNeZeroNF
      inst✝ : E'.IsCharTwoJNeZeroNF
      heq : Eq E.a₆ E'.a₆
      s : F
      hs : Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul 1 (HPow.hPow s 2)) (HMul.hMul 1 s)) ( …
      ⊢ Eq (E.variableChange { u := 1, r := 0, s := s, t := 0 }).a₂ E'.a₂
    -/
  · simp_rw [variableChange_a₂, inv_one, Units.val_one, a₁_of_isCharTwoJNeZeroNF]
    /-
      case h.a₂
      F : Type u_1
      inst✝⁴ : Field F
      inst✝³ : IsSepClosed F
      E E' : WeierstrassCurve F
      inst✝² : CharP F 2
      inst✝¹ : E.IsCharTwoJNeZeroNF
      inst✝ : E'.IsCharTwoJNeZeroNF
      heq : Eq E.a₆ E'.a₆
      s : F
      hs : Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul 1 (HPow.hPow s 2)) (HMul.hMul 1 s)) ( …
      ⊢ Eq (HMul.hMul (HPow.hPow 1 2) (HSub.hSub (HAdd.hAdd (HSub.hSub E.a₂ (HMul.hM …
    -/
    linear_combination -hs + E.a₂ * CharP.cast_eq_zero F 2
    /-
      🎉 no goals
    -/
    /-
      case h.a₃
      F : Type u_1
      inst✝⁴ : Field F
      inst✝³ : IsSepClosed F
      E E' : WeierstrassCurve F
      inst✝² : CharP F 2
      inst✝¹ : E.IsCharTwoJNeZeroNF
      inst✝ : E'.IsCharTwoJNeZeroNF
      heq : Eq E.a₆ E'.a₆
      s : F
      hs : Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul 1 (HPow.hPow s 2)) (HMul.hMul 1 s)) ( …
      ⊢ Eq (E.variableChange { u := 1, r := 0, s := s, t := 0 }).a₃ E'.a₃
    -/
  · simp_rw [variableChange_a₃, inv_one, Units.val_one, a₃_of_isCharTwoJNeZeroNF]
    /-
      case h.a₃
      F : Type u_1
      inst✝⁴ : Field F
      inst✝³ : IsSepClosed F
      E E' : WeierstrassCurve F
      inst✝² : CharP F 2
      inst✝¹ : E.IsCharTwoJNeZeroNF
      inst✝ : E'.IsCharTwoJNeZeroNF
      heq : Eq E.a₆ E'.a₆
      s : F
      hs : Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul 1 (HPow.hPow s 2)) (HMul.hMul 1 s)) ( …
      ⊢ Eq (HMul.hMul (HPow.hPow 1 3) (HAdd.hAdd (HAdd.hAdd 0 (HMul.hMul 0 E.a₁)) (H …
    -/
    ring1
    /-
      🎉 no goals
    -/
  · simp_rw [variableChange_a₄, inv_one, Units.val_one, a₃_of_isCharTwoJNeZeroNF,
      a₄_of_isCharTwoJNeZeroNF]
    /-
      case h.a₄
      F : Type u_1
      inst✝⁴ : Field F
      inst✝³ : IsSepClosed F
      E E' : WeierstrassCurve F
      inst✝² : CharP F 2
      inst✝¹ : E.IsCharTwoJNeZeroNF
      inst✝ : E'.IsCharTwoJNeZeroNF
      heq : Eq E.a₆ E'.a₆
      s : F
      hs : Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul 1 (HPow.hPow s 2)) (HMul.hMul 1 s)) ( …
      ⊢ Eq (HMul.hMul (HPow.hPow 1 4) (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (H …
    -/
    ring1
    /-
      🎉 no goals
    -/
    /-
      case h.a₆
      F : Type u_1
      inst✝⁴ : Field F
      inst✝³ : IsSepClosed F
      E E' : WeierstrassCurve F
      inst✝² : CharP F 2
      inst✝¹ : E.IsCharTwoJNeZeroNF
      inst✝ : E'.IsCharTwoJNeZeroNF
      heq : Eq E.a₆ E'.a₆
      s : F
      hs : Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul 1 (HPow.hPow s 2)) (HMul.hMul 1 s)) ( …
      ⊢ Eq (E.variableChange { u := 1, r := 0, s := s, t := 0 }).a₆ E'.a₆
    -/
  · simp_rw [variableChange_a₆, inv_one, Units.val_one, heq]
    /-
      case h.a₆
      F : Type u_1
      inst✝⁴ : Field F
      inst✝³ : IsSepClosed F
      E E' : WeierstrassCurve F
      inst✝² : CharP F 2
      inst✝¹ : E.IsCharTwoJNeZeroNF
      inst✝ : E'.IsCharTwoJNeZeroNF
      heq : Eq E.a₆ E'.a₆
      s : F
      hs : Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul 1 (HPow.hPow s 2)) (HMul.hMul 1 s)) ( …
      ⊢ Eq (HMul.hMul (HPow.hPow 1 6) (HSub.hSub (HSub.hSub (HSub.hSub (HAdd.hAdd (H …
    -/
    ring1
    /-
      🎉 no goals
    -/


private lemma exists_variableChange_of_char_two_of_j_eq_zero
    [E.IsCharTwoJEqZeroNF] [E'.IsCharTwoJEqZeroNF] :
    ∃ C : VariableChange F, E.variableChange C = E' := by
  /-
    F : Type u_1
    inst✝⁶ : Field F
    inst✝⁵ : IsSepClosed F
    E E' : WeierstrassCurve F
    inst✝⁴ : E.IsElliptic
    inst✝³ : E'.IsElliptic
    inst✝² : CharP F 2
    inst✝¹ : E.IsCharTwoJEqZeroNF
    inst✝ : E'.IsCharTwoJEqZeroNF
    ⊢ Exists fun C => Eq (E.variableChange C) E'
  -/
  have ha₃ := E.Δ'.ne_zero
  /-
    F : Type u_1
    inst✝⁶ : Field F
    inst✝⁵ : IsSepClosed F
    E E' : WeierstrassCurve F
    inst✝⁴ : E.IsElliptic
    inst✝³ : E'.IsElliptic
    inst✝² : CharP F 2
    inst✝¹ : E.IsCharTwoJEqZeroNF
    inst✝ : E'.IsCharTwoJEqZeroNF
    ha₃ : Ne (↑E.Δ') 0
    ⊢ Exists fun C => Eq (E.variableChange C) E'
  -/
  rw [E.coe_Δ', Δ_of_isCharTwoJEqZeroNF_of_char_two, pow_ne_zero_iff (Nat.succ_ne_zero _)] at ha₃
  /-
    F : Type u_1
    inst✝⁶ : Field F
    inst✝⁵ : IsSepClosed F
    E E' : WeierstrassCurve F
    inst✝⁴ : E.IsElliptic
    inst✝³ : E'.IsElliptic
    inst✝² : CharP F 2
    inst✝¹ : E.IsCharTwoJEqZeroNF
    inst✝ : E'.IsCharTwoJEqZeroNF
    ha₃ : Ne E.a₃ 0
    ⊢ Exists fun C => Eq (E.variableChange C) E'
  -/
  have ha₃' := E'.Δ'.ne_zero
  /-
    F : Type u_1
    inst✝⁶ : Field F
    inst✝⁵ : IsSepClosed F
    E E' : WeierstrassCurve F
    inst✝⁴ : E.IsElliptic
    inst✝³ : E'.IsElliptic
    inst✝² : CharP F 2
    inst✝¹ : E.IsCharTwoJEqZeroNF
    inst✝ : E'.IsCharTwoJEqZeroNF
    ha₃ : Ne E.a₃ 0
    ha₃' : Ne (↑E'.Δ') 0
    ⊢ Exists fun C => Eq (E.variableChange C) E'
  -/
  rw [E'.coe_Δ', Δ_of_isCharTwoJEqZeroNF_of_char_two, pow_ne_zero_iff (Nat.succ_ne_zero _)] at ha₃'
  haveI : NeZero (3 : F) := NeZero.mk <| by
    rw [show (3 : F) = 1 by linear_combination CharP.cast_eq_zero F 2]
    exact one_ne_zero
  /-
    F : Type u_1
    inst✝⁶ : Field F
    inst✝⁵ : IsSepClosed F
    E E' : WeierstrassCurve F
    inst✝⁴ : E.IsElliptic
    inst✝³ : E'.IsElliptic
    inst✝² : CharP F 2
    inst✝¹ : E.IsCharTwoJEqZeroNF
    inst✝ : E'.IsCharTwoJEqZeroNF
    ha₃ : Ne E.a₃ 0
    ha₃' : Ne E'.a₃ 0
    this : NeZero 3
    ⊢ Exists fun C => Eq (E.variableChange C) E'
  -/
  obtain ⟨u, hu⟩ := IsSepClosed.exists_pow_nat_eq (E.a₃ / E'.a₃) 3
  obtain ⟨s, hs⟩ := IsSepClosed.exists_root_C_mul_X_pow_add_C_mul_X_add_C' 2 4
    1 _ (E.a₄ - u ^ 4 * E'.a₄) (by norm_num) (by norm_num) ha₃
  obtain ⟨t, ht⟩ := IsSepClosed.exists_root_C_mul_X_pow_add_C_mul_X_add_C' 2 2
    1 _ (s ^ 6 + E.a₄ * s ^ 2 + E.a₆ - u ^ 6 * E'.a₆) (by norm_num) (by norm_num) ha₃
  have hu0 : u ≠ 0 := by
    rw [← pow_ne_zero_iff three_ne_zero, hu, div_ne_zero_iff]
    exact ⟨ha₃, ha₃'⟩
  /-
    case intro.intro.intro
    F : Type u_1
    inst✝⁶ : Field F
    inst✝⁵ : IsSepClosed F
    E E' : WeierstrassCurve F
    inst✝⁴ : E.IsElliptic
    inst✝³ : E'.IsElliptic
    inst✝² : CharP F 2
    inst✝¹ : E.IsCharTwoJEqZeroNF
    inst✝ : E'.IsCharTwoJEqZeroNF
    ha₃ : Ne E.a₃ 0
    ha₃' : Ne E'.a₃ 0
    this : NeZero 3
    u : F
    hu : Eq (HPow.hPow u 3) (HDiv.hDiv E.a₃ E'.a₃)
    s : F
    hs : Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul 1 (HPow.hPow s 4)) (HMul.hMul E.a₃ s) …
    t : F
    ht : Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul 1 (HPow.hPow t 2)) (HMul.hMul E.a₃ t) …
    hu0 : Ne u 0
    ⊢ Exists fun C => Eq (E.variableChange C) E'
  -/
  use ⟨Units.mk0 u hu0, s ^ 2, s, t⟩
  /-
    case h
    F : Type u_1
    inst✝⁶ : Field F
    inst✝⁵ : IsSepClosed F
    E E' : WeierstrassCurve F
    inst✝⁴ : E.IsElliptic
    inst✝³ : E'.IsElliptic
    inst✝² : CharP F 2
    inst✝¹ : E.IsCharTwoJEqZeroNF
    inst✝ : E'.IsCharTwoJEqZeroNF
    ha₃ : Ne E.a₃ 0
    ha₃' : Ne E'.a₃ 0
    this : NeZero 3
    u : F
    hu : Eq (HPow.hPow u 3) (HDiv.hDiv E.a₃ E'.a₃)
    s : F
    hs : Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul 1 (HPow.hPow s 4)) (HMul.hMul E.a₃ s) …
    t : F
    ht : Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul 1 (HPow.hPow t 2)) (HMul.hMul E.a₃ t) …
    hu0 : Ne u 0
    ⊢ Eq (E.variableChange { u := Units.mk0 u hu0, r := HPow.hPow s 2, s := s, t : …
  -/
  ext
  · simp_rw [variableChange_a₁, a₁_of_isCharTwoJEqZeroNF,
      show (2 : F) = 0 from CharP.cast_eq_zero F 2]
    /-
      case h.a₁
      F : Type u_1
      inst✝⁶ : Field F
      inst✝⁵ : IsSepClosed F
      E E' : WeierstrassCurve F
      inst✝⁴ : E.IsElliptic
      inst✝³ : E'.IsElliptic
      inst✝² : CharP F 2
      inst✝¹ : E.IsCharTwoJEqZeroNF
      inst✝ : E'.IsCharTwoJEqZeroNF
      ha₃ : Ne E.a₃ 0
      ha₃' : Ne E'.a₃ 0
      this : NeZero 3
      u : F
      hu : Eq (HPow.hPow u 3) (HDiv.hDiv E.a₃ E'.a₃)
      s : F
      hs : Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul 1 (HPow.hPow s 4)) (HMul.hMul E.a₃ s) …
      t : F
      ht : Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul 1 (HPow.hPow t 2)) (HMul.hMul E.a₃ t) …
      hu0 : Ne u 0
      ⊢ Eq (HMul.hMul (↑(Inv.inv (Units.mk0 u hu0))) (HAdd.hAdd 0 (HMul.hMul 0 s))) 0
    -/
    ring1
    /-
      🎉 no goals
    -/
  · simp_rw [variableChange_a₂, a₁_of_isCharTwoJEqZeroNF, a₂_of_isCharTwoJEqZeroNF,
      show (3 : F) = 1 by linear_combination CharP.cast_eq_zero F 2]
    /-
      case h.a₂
      F : Type u_1
      inst✝⁶ : Field F
      inst✝⁵ : IsSepClosed F
      E E' : WeierstrassCurve F
      inst✝⁴ : E.IsElliptic
      inst✝³ : E'.IsElliptic
      inst✝² : CharP F 2
      inst✝¹ : E.IsCharTwoJEqZeroNF
      inst✝ : E'.IsCharTwoJEqZeroNF
      ha₃ : Ne E.a₃ 0
      ha₃' : Ne E'.a₃ 0
      this : NeZero 3
      u : F
      hu : Eq (HPow.hPow u 3) (HDiv.hDiv E.a₃ E'.a₃)
      s : F
      hs : Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul 1 (HPow.hPow s 4)) (HMul.hMul E.a₃ s) …
      t : F
      ht : Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul 1 (HPow.hPow t 2)) (HMul.hMul E.a₃ t) …
      hu0 : Ne u 0
      ⊢ Eq (HMul.hMul (HPow.hPow (↑(Inv.inv (Units.mk0 u hu0))) 2) (HSub.hSub (HAdd. …
    -/
    ring1
    /-
      🎉 no goals
    -/
  · simp_rw [variableChange_a₃, Units.val_inv_eq_inv_val, Units.val_mk0, inv_pow, inv_mul_eq_div,
      hu, a₁_of_isCharTwoJEqZeroNF, show (2 : F) = 0 from CharP.cast_eq_zero F 2]
    /-
      case h.a₃
      F : Type u_1
      inst✝⁶ : Field F
      inst✝⁵ : IsSepClosed F
      E E' : WeierstrassCurve F
      inst✝⁴ : E.IsElliptic
      inst✝³ : E'.IsElliptic
      inst✝² : CharP F 2
      inst✝¹ : E.IsCharTwoJEqZeroNF
      inst✝ : E'.IsCharTwoJEqZeroNF
      ha₃ : Ne E.a₃ 0
      ha₃' : Ne E'.a₃ 0
      this : NeZero 3
      u : F
      hu : Eq (HPow.hPow u 3) (HDiv.hDiv E.a₃ E'.a₃)
      s : F
      hs : Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul 1 (HPow.hPow s 4)) (HMul.hMul E.a₃ s) …
      t : F
      ht : Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul 1 (HPow.hPow t 2)) (HMul.hMul E.a₃ t) …
      hu0 : Ne u 0
      ⊢ Eq (HDiv.hDiv (HAdd.hAdd (HAdd.hAdd E.a₃ (HMul.hMul (HPow.hPow s 2) 0)) (HMu …
    -/
    field_simp
    /-
      🎉 no goals
    -/
    /-
      case h.a₄
      F : Type u_1
      inst✝⁶ : Field F
      inst✝⁵ : IsSepClosed F
      E E' : WeierstrassCurve F
      inst✝⁴ : E.IsElliptic
      inst✝³ : E'.IsElliptic
      inst✝² : CharP F 2
      inst✝¹ : E.IsCharTwoJEqZeroNF
      inst✝ : E'.IsCharTwoJEqZeroNF
      ha₃ : Ne E.a₃ 0
      ha₃' : Ne E'.a₃ 0
      this : NeZero 3
      u : F
      hu : Eq (HPow.hPow u 3) (HDiv.hDiv E.a₃ E'.a₃)
      s : F
      hs : Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul 1 (HPow.hPow s 4)) (HMul.hMul E.a₃ s) …
      t : F
      ht : Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul 1 (HPow.hPow t 2)) (HMul.hMul E.a₃ t) …
      hu0 : Ne u 0
      ⊢ Eq (E.variableChange { u := Units.mk0 u hu0, r := HPow.hPow s 2, s := s, t : …
    -/
  · field_simp [variableChange_a₄, a₁_of_isCharTwoJEqZeroNF, a₂_of_isCharTwoJEqZeroNF]
    /-
      case h.a₄
      F : Type u_1
      inst✝⁶ : Field F
      inst✝⁵ : IsSepClosed F
      E E' : WeierstrassCurve F
      inst✝⁴ : E.IsElliptic
      inst✝³ : E'.IsElliptic
      inst✝² : CharP F 2
      inst✝¹ : E.IsCharTwoJEqZeroNF
      inst✝ : E'.IsCharTwoJEqZeroNF
      ha₃ : Ne E.a₃ 0
      ha₃' : Ne E'.a₃ 0
      this : NeZero 3
      u : F
      hu : Eq (HPow.hPow u 3) (HDiv.hDiv E.a₃ E'.a₃)
      s : F
      hs : Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul 1 (HPow.hPow s 4)) (HMul.hMul E.a₃ s) …
      t : F
      ht : Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul 1 (HPow.hPow t 2)) (HMul.hMul E.a₃ t) …
      hu0 : Ne u 0
      ⊢ Eq (HSub.hSub (HAdd.hAdd (HSub.hSub E.a₄ (HMul.hMul s E.a₃)) (HMul.hMul 3 (H …
    -/
    linear_combination hs + (s ^ 4 - s * t - E.a₃ * s) * CharP.cast_eq_zero F 2
    /-
      🎉 no goals
    -/
    /-
      case h.a₆
      F : Type u_1
      inst✝⁶ : Field F
      inst✝⁵ : IsSepClosed F
      E E' : WeierstrassCurve F
      inst✝⁴ : E.IsElliptic
      inst✝³ : E'.IsElliptic
      inst✝² : CharP F 2
      inst✝¹ : E.IsCharTwoJEqZeroNF
      inst✝ : E'.IsCharTwoJEqZeroNF
      ha₃ : Ne E.a₃ 0
      ha₃' : Ne E'.a₃ 0
      this : NeZero 3
      u : F
      hu : Eq (HPow.hPow u 3) (HDiv.hDiv E.a₃ E'.a₃)
      s : F
      hs : Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul 1 (HPow.hPow s 4)) (HMul.hMul E.a₃ s) …
      t : F
      ht : Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul 1 (HPow.hPow t 2)) (HMul.hMul E.a₃ t) …
      hu0 : Ne u 0
      ⊢ Eq (E.variableChange { u := Units.mk0 u hu0, r := HPow.hPow s 2, s := s, t : …
    -/
  · field_simp [variableChange_a₄, a₁_of_isCharTwoJEqZeroNF, a₂_of_isCharTwoJEqZeroNF]
    /-
      case h.a₆
      F : Type u_1
      inst✝⁶ : Field F
      inst✝⁵ : IsSepClosed F
      E E' : WeierstrassCurve F
      inst✝⁴ : E.IsElliptic
      inst✝³ : E'.IsElliptic
      inst✝² : CharP F 2
      inst✝¹ : E.IsCharTwoJEqZeroNF
      inst✝ : E'.IsCharTwoJEqZeroNF
      ha₃ : Ne E.a₃ 0
      ha₃' : Ne E'.a₃ 0
      this : NeZero 3
      u : F
      hu : Eq (HPow.hPow u 3) (HDiv.hDiv E.a₃ E'.a₃)
      s : F
      hs : Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul 1 (HPow.hPow s 4)) (HMul.hMul E.a₃ s) …
      t : F
      ht : Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul 1 (HPow.hPow t 2)) (HMul.hMul E.a₃ t) …
      hu0 : Ne u 0
      ⊢ Eq (HSub.hSub (HSub.hSub (HAdd.hAdd (HAdd.hAdd E.a₆ (HMul.hMul (HPow.hPow s  …
    -/
    linear_combination ht - (t ^ 2 + E.a₃ * t) * CharP.cast_eq_zero F 2
    /-
      🎉 no goals
    -/


private lemma exists_variableChange_of_char_two (heq : E.j = E'.j) :
    ∃ C : VariableChange F, E.variableChange C = E' := by
  /-
    F : Type u_1
    inst✝⁴ : Field F
    inst✝³ : IsSepClosed F
    E E' : WeierstrassCurve F
    inst✝² : E.IsElliptic
    inst✝¹ : E'.IsElliptic
    inst✝ : CharP F 2
    heq : Eq E.j E'.j
    ⊢ Exists fun C => Eq (E.variableChange C) E'
  -/
  obtain ⟨C, _ | _⟩ := E.exists_variableChange_isCharTwoNF
    /-
      case intro.of_j_ne_zero
      F : Type u_1
      inst✝⁵ : Field F
      inst✝⁴ : IsSepClosed F
      E E' : WeierstrassCurve F
      inst✝³ : E.IsElliptic
      inst✝² : E'.IsElliptic
      inst✝¹ : CharP F 2
      heq : Eq E.j E'.j
      C : WeierstrassCurve.VariableChange F
      inst✝ : (E.variableChange C).IsCharTwoJNeZeroNF
      ⊢ Exists fun C => Eq (E.variableChange C) E'
    -/
  · obtain ⟨C', _ | _⟩ := E'.exists_variableChange_isCharTwoNF
    · simp_rw [← variableChange_j E C, ← variableChange_j E' C',
        j_of_isCharTwoJNeZeroNF_of_char_two, one_div, inv_inj] at heq
      /-
        case intro.of_j_ne_zero.intro.of_j_ne_zero
        F : Type u_1
        inst✝⁶ : Field F
        inst✝⁵ : IsSepClosed F
        E E' : WeierstrassCurve F
        inst✝⁴ : E.IsElliptic
        inst✝³ : E'.IsElliptic
        inst✝² : CharP F 2
        C : WeierstrassCurve.VariableChange F
        inst✝¹ : (E.variableChange C).IsCharTwoJNeZeroNF
        C' : WeierstrassCurve.VariableChange F
        inst✝ : (E'.variableChange C').IsCharTwoJNeZeroNF
        heq : Eq (E.variableChange C).a₆ (E'.variableChange C').a₆
        ⊢ Exists fun C => Eq (E.variableChange C) E'
      -/
      obtain ⟨C'', hC⟩ := exists_variableChange_of_char_two_of_j_ne_zero _ _ heq
      /-
        case intro.of_j_ne_zero.intro.of_j_ne_zero.intro
        F : Type u_1
        inst✝⁶ : Field F
        inst✝⁵ : IsSepClosed F
        E E' : WeierstrassCurve F
        inst✝⁴ : E.IsElliptic
        inst✝³ : E'.IsElliptic
        inst✝² : CharP F 2
        C : WeierstrassCurve.VariableChange F
        inst✝¹ : (E.variableChange C).IsCharTwoJNeZeroNF
        C' : WeierstrassCurve.VariableChange F
        inst✝ : (E'.variableChange C').IsCharTwoJNeZeroNF
        heq : Eq (E.variableChange C).a₆ (E'.variableChange C').a₆
        C'' : WeierstrassCurve.VariableChange F
        hC : Eq ((E.variableChange C).variableChange C'') (E'.variableChange C')
        ⊢ Exists fun C => Eq (E.variableChange C) E'
      -/
      use (C'.inv.comp C'').comp C
      rw [variableChange_comp, variableChange_comp, hC, ← variableChange_comp,
        VariableChange.comp_left_inv, variableChange_id]
      /-
        case intro.of_j_ne_zero.intro.of_j_eq_zero
        F : Type u_1
        inst✝⁶ : Field F
        inst✝⁵ : IsSepClosed F
        E E' : WeierstrassCurve F
        inst✝⁴ : E.IsElliptic
        inst✝³ : E'.IsElliptic
        inst✝² : CharP F 2
        heq : Eq E.j E'.j
        C : WeierstrassCurve.VariableChange F
        inst✝¹ : (E.variableChange C).IsCharTwoJNeZeroNF
        C' : WeierstrassCurve.VariableChange F
        inst✝ : (E'.variableChange C').IsCharTwoJEqZeroNF
        ⊢ Exists fun C => Eq (E.variableChange C) E'
      -/
    · have h := (E.variableChange C).j_ne_zero_of_isCharTwoJNeZeroNF_of_char_two
      rw [variableChange_j, heq, ← variableChange_j E' C',
        j_of_isCharTwoJEqZeroNF_of_char_two] at h
      /-
        case intro.of_j_ne_zero.intro.of_j_eq_zero
        F : Type u_1
        inst✝⁶ : Field F
        inst✝⁵ : IsSepClosed F
        E E' : WeierstrassCurve F
        inst✝⁴ : E.IsElliptic
        inst✝³ : E'.IsElliptic
        inst✝² : CharP F 2
        heq : Eq E.j E'.j
        C : WeierstrassCurve.VariableChange F
        inst✝¹ : (E.variableChange C).IsCharTwoJNeZeroNF
        C' : WeierstrassCurve.VariableChange F
        inst✝ : (E'.variableChange C').IsCharTwoJEqZeroNF
        h : Ne 0 0
        ⊢ Exists fun C => Eq (E.variableChange C) E'
      -/
      exact False.elim (h rfl)
      /-
        🎉 no goals
      -/
    /-
      case intro.of_j_eq_zero
      F : Type u_1
      inst✝⁵ : Field F
      inst✝⁴ : IsSepClosed F
      E E' : WeierstrassCurve F
      inst✝³ : E.IsElliptic
      inst✝² : E'.IsElliptic
      inst✝¹ : CharP F 2
      heq : Eq E.j E'.j
      C : WeierstrassCurve.VariableChange F
      inst✝ : (E.variableChange C).IsCharTwoJEqZeroNF
      ⊢ Exists fun C => Eq (E.variableChange C) E'
    -/
  · obtain ⟨C', _ | _⟩ := E'.exists_variableChange_isCharTwoNF
      /-
        case intro.of_j_eq_zero.intro.of_j_ne_zero
        F : Type u_1
        inst✝⁶ : Field F
        inst✝⁵ : IsSepClosed F
        E E' : WeierstrassCurve F
        inst✝⁴ : E.IsElliptic
        inst✝³ : E'.IsElliptic
        inst✝² : CharP F 2
        heq : Eq E.j E'.j
        C : WeierstrassCurve.VariableChange F
        inst✝¹ : (E.variableChange C).IsCharTwoJEqZeroNF
        C' : WeierstrassCurve.VariableChange F
        inst✝ : (E'.variableChange C').IsCharTwoJNeZeroNF
        ⊢ Exists fun C => Eq (E.variableChange C) E'
      -/
    · have h := (E'.variableChange C').j_ne_zero_of_isCharTwoJNeZeroNF_of_char_two
      rw [variableChange_j, ← heq, ← variableChange_j E C,
        j_of_isCharTwoJEqZeroNF_of_char_two] at h
      /-
        case intro.of_j_eq_zero.intro.of_j_ne_zero
        F : Type u_1
        inst✝⁶ : Field F
        inst✝⁵ : IsSepClosed F
        E E' : WeierstrassCurve F
        inst✝⁴ : E.IsElliptic
        inst✝³ : E'.IsElliptic
        inst✝² : CharP F 2
        heq : Eq E.j E'.j
        C : WeierstrassCurve.VariableChange F
        inst✝¹ : (E.variableChange C).IsCharTwoJEqZeroNF
        C' : WeierstrassCurve.VariableChange F
        inst✝ : (E'.variableChange C').IsCharTwoJNeZeroNF
        h : Ne 0 0
        ⊢ Exists fun C => Eq (E.variableChange C) E'
      -/
      exact False.elim (h rfl)
      /-
        🎉 no goals
      -/
    · obtain ⟨C'', hC⟩ := exists_variableChange_of_char_two_of_j_eq_zero
        (E.variableChange C) (E'.variableChange C')
      /-
        case intro.of_j_eq_zero.intro.of_j_eq_zero.intro
        F : Type u_1
        inst✝⁶ : Field F
        inst✝⁵ : IsSepClosed F
        E E' : WeierstrassCurve F
        inst✝⁴ : E.IsElliptic
        inst✝³ : E'.IsElliptic
        inst✝² : CharP F 2
        heq : Eq E.j E'.j
        C : WeierstrassCurve.VariableChange F
        inst✝¹ : (E.variableChange C).IsCharTwoJEqZeroNF
        C' : WeierstrassCurve.VariableChange F
        inst✝ : (E'.variableChange C').IsCharTwoJEqZeroNF
        C'' : WeierstrassCurve.VariableChange F
        hC : Eq ((E.variableChange C).variableChange C'') (E'.variableChange C')
        ⊢ Exists fun C => Eq (E.variableChange C) E'
      -/
      use (C'.inv.comp C'').comp C
      rw [variableChange_comp, variableChange_comp, hC, ← variableChange_comp,
        VariableChange.comp_left_inv, variableChange_id]


private lemma exists_variableChange_of_char_three_of_j_ne_zero
    [E.IsCharThreeJNeZeroNF] [E'.IsCharThreeJNeZeroNF] (heq : E.j = E'.j) :
    ∃ C : VariableChange F, E.variableChange C = E' := by
  /-
    F : Type u_1
    inst✝⁶ : Field F
    inst✝⁵ : IsSepClosed F
    E E' : WeierstrassCurve F
    inst✝⁴ : E.IsElliptic
    inst✝³ : E'.IsElliptic
    inst✝² : CharP F 3
    inst✝¹ : E.IsCharThreeJNeZeroNF
    inst✝ : E'.IsCharThreeJNeZeroNF
    heq : Eq E.j E'.j
    ⊢ Exists fun C => Eq (E.variableChange C) E'
  -/
  have h := E.Δ'.ne_zero
  rw [E.coe_Δ', Δ_of_isCharThreeJNeZeroNF_of_char_three, mul_ne_zero_iff, neg_ne_zero,
    pow_ne_zero_iff three_ne_zero] at h
  /-
    F : Type u_1
    inst✝⁶ : Field F
    inst✝⁵ : IsSepClosed F
    E E' : WeierstrassCurve F
    inst✝⁴ : E.IsElliptic
    inst✝³ : E'.IsElliptic
    inst✝² : CharP F 3
    inst✝¹ : E.IsCharThreeJNeZeroNF
    inst✝ : E'.IsCharThreeJNeZeroNF
    heq : Eq E.j E'.j
    h : And (Ne E.a₂ 0) (Ne E.a₆ 0)
    ⊢ Exists fun C => Eq (E.variableChange C) E'
  -/
  obtain ⟨ha₂, ha₆⟩ := h
  /-
    case intro
    F : Type u_1
    inst✝⁶ : Field F
    inst✝⁵ : IsSepClosed F
    E E' : WeierstrassCurve F
    inst✝⁴ : E.IsElliptic
    inst✝³ : E'.IsElliptic
    inst✝² : CharP F 3
    inst✝¹ : E.IsCharThreeJNeZeroNF
    inst✝ : E'.IsCharThreeJNeZeroNF
    heq : Eq E.j E'.j
    ha₂ : Ne E.a₂ 0
    ha₆ : Ne E.a₆ 0
    ⊢ Exists fun C => Eq (E.variableChange C) E'
  -/
  have h := E'.Δ'.ne_zero
  rw [E'.coe_Δ', Δ_of_isCharThreeJNeZeroNF_of_char_three, mul_ne_zero_iff, neg_ne_zero,
    pow_ne_zero_iff three_ne_zero] at h
  /-
    case intro
    F : Type u_1
    inst✝⁶ : Field F
    inst✝⁵ : IsSepClosed F
    E E' : WeierstrassCurve F
    inst✝⁴ : E.IsElliptic
    inst✝³ : E'.IsElliptic
    inst✝² : CharP F 3
    inst✝¹ : E.IsCharThreeJNeZeroNF
    inst✝ : E'.IsCharThreeJNeZeroNF
    heq : Eq E.j E'.j
    ha₂ : Ne E.a₂ 0
    ha₆ : Ne E.a₆ 0
    h : And (Ne E'.a₂ 0) (Ne E'.a₆ 0)
    ⊢ Exists fun C => Eq (E.variableChange C) E'
  -/
  obtain ⟨ha₂', ha₆'⟩ := h
  haveI : NeZero (2 : F) := NeZero.mk <| by
    rw [show (2 : F) = -1 by linear_combination CharP.cast_eq_zero F 3, neg_ne_zero]
    exact one_ne_zero
  /-
    case intro.intro
    F : Type u_1
    inst✝⁶ : Field F
    inst✝⁵ : IsSepClosed F
    E E' : WeierstrassCurve F
    inst✝⁴ : E.IsElliptic
    inst✝³ : E'.IsElliptic
    inst✝² : CharP F 3
    inst✝¹ : E.IsCharThreeJNeZeroNF
    inst✝ : E'.IsCharThreeJNeZeroNF
    heq : Eq E.j E'.j
    ha₂ : Ne E.a₂ 0
    ha₆ : Ne E.a₆ 0
    ha₂' : Ne E'.a₂ 0
    ha₆' : Ne E'.a₆ 0
    this : NeZero 2
    ⊢ Exists fun C => Eq (E.variableChange C) E'
  -/
  obtain ⟨u, hu⟩ := IsSepClosed.exists_pow_nat_eq (E.a₂ / E'.a₂) 2
  have hu0 : u ≠ 0 := by
    rw [← pow_ne_zero_iff two_ne_zero, hu, div_ne_zero_iff]
    exact ⟨ha₂, ha₂'⟩
  /-
    case intro.intro.intro
    F : Type u_1
    inst✝⁶ : Field F
    inst✝⁵ : IsSepClosed F
    E E' : WeierstrassCurve F
    inst✝⁴ : E.IsElliptic
    inst✝³ : E'.IsElliptic
    inst✝² : CharP F 3
    inst✝¹ : E.IsCharThreeJNeZeroNF
    inst✝ : E'.IsCharThreeJNeZeroNF
    heq : Eq E.j E'.j
    ha₂ : Ne E.a₂ 0
    ha₆ : Ne E.a₆ 0
    ha₂' : Ne E'.a₂ 0
    ha₆' : Ne E'.a₆ 0
    this : NeZero 2
    u : F
    hu : Eq (HPow.hPow u 2) (HDiv.hDiv E.a₂ E'.a₂)
    hu0 : Ne u 0
    ⊢ Exists fun C => Eq (E.variableChange C) E'
  -/
  use ⟨Units.mk0 u hu0, 0, 0, 0⟩
  /-
    case h
    F : Type u_1
    inst✝⁶ : Field F
    inst✝⁵ : IsSepClosed F
    E E' : WeierstrassCurve F
    inst✝⁴ : E.IsElliptic
    inst✝³ : E'.IsElliptic
    inst✝² : CharP F 3
    inst✝¹ : E.IsCharThreeJNeZeroNF
    inst✝ : E'.IsCharThreeJNeZeroNF
    heq : Eq E.j E'.j
    ha₂ : Ne E.a₂ 0
    ha₆ : Ne E.a₆ 0
    ha₂' : Ne E'.a₂ 0
    ha₆' : Ne E'.a₆ 0
    this : NeZero 2
    u : F
    hu : Eq (HPow.hPow u 2) (HDiv.hDiv E.a₂ E'.a₂)
    hu0 : Ne u 0
    ⊢ Eq (E.variableChange { u := Units.mk0 u hu0, r := 0, s := 0, t := 0 }) E'
  -/
  ext
    /-
      case h.a₁
      F : Type u_1
      inst✝⁶ : Field F
      inst✝⁵ : IsSepClosed F
      E E' : WeierstrassCurve F
      inst✝⁴ : E.IsElliptic
      inst✝³ : E'.IsElliptic
      inst✝² : CharP F 3
      inst✝¹ : E.IsCharThreeJNeZeroNF
      inst✝ : E'.IsCharThreeJNeZeroNF
      heq : Eq E.j E'.j
      ha₂ : Ne E.a₂ 0
      ha₆ : Ne E.a₆ 0
      ha₂' : Ne E'.a₂ 0
      ha₆' : Ne E'.a₆ 0
      this : NeZero 2
      u : F
      hu : Eq (HPow.hPow u 2) (HDiv.hDiv E.a₂ E'.a₂)
      hu0 : Ne u 0
      ⊢ Eq (E.variableChange { u := Units.mk0 u hu0, r := 0, s := 0, t := 0 }).a₁ E' …
    -/
  · simp_rw [variableChange_a₁, a₁_of_isCharThreeJNeZeroNF]
    /-
      case h.a₁
      F : Type u_1
      inst✝⁶ : Field F
      inst✝⁵ : IsSepClosed F
      E E' : WeierstrassCurve F
      inst✝⁴ : E.IsElliptic
      inst✝³ : E'.IsElliptic
      inst✝² : CharP F 3
      inst✝¹ : E.IsCharThreeJNeZeroNF
      inst✝ : E'.IsCharThreeJNeZeroNF
      heq : Eq E.j E'.j
      ha₂ : Ne E.a₂ 0
      ha₆ : Ne E.a₆ 0
      ha₂' : Ne E'.a₂ 0
      ha₆' : Ne E'.a₆ 0
      this : NeZero 2
      u : F
      hu : Eq (HPow.hPow u 2) (HDiv.hDiv E.a₂ E'.a₂)
      hu0 : Ne u 0
      ⊢ Eq (HMul.hMul (↑(Inv.inv (Units.mk0 u hu0))) (HAdd.hAdd 0 (HMul.hMul 2 0))) 0
    -/
    ring1
    /-
      🎉 no goals
    -/
  · simp_rw [variableChange_a₂, a₁_of_isCharThreeJNeZeroNF, Units.val_inv_eq_inv_val,
      Units.val_mk0, inv_pow, inv_mul_eq_div, hu]
    /-
      case h.a₂
      F : Type u_1
      inst✝⁶ : Field F
      inst✝⁵ : IsSepClosed F
      E E' : WeierstrassCurve F
      inst✝⁴ : E.IsElliptic
      inst✝³ : E'.IsElliptic
      inst✝² : CharP F 3
      inst✝¹ : E.IsCharThreeJNeZeroNF
      inst✝ : E'.IsCharThreeJNeZeroNF
      heq : Eq E.j E'.j
      ha₂ : Ne E.a₂ 0
      ha₆ : Ne E.a₆ 0
      ha₂' : Ne E'.a₂ 0
      ha₆' : Ne E'.a₆ 0
      this : NeZero 2
      u : F
      hu : Eq (HPow.hPow u 2) (HDiv.hDiv E.a₂ E'.a₂)
      hu0 : Ne u 0
      ⊢ Eq (HDiv.hDiv (HSub.hSub (HAdd.hAdd (HSub.hSub E.a₂ (HMul.hMul 0 0)) (HMul.h …
    -/
    field_simp
    /-
      🎉 no goals
    -/
    /-
      case h.a₃
      F : Type u_1
      inst✝⁶ : Field F
      inst✝⁵ : IsSepClosed F
      E E' : WeierstrassCurve F
      inst✝⁴ : E.IsElliptic
      inst✝³ : E'.IsElliptic
      inst✝² : CharP F 3
      inst✝¹ : E.IsCharThreeJNeZeroNF
      inst✝ : E'.IsCharThreeJNeZeroNF
      heq : Eq E.j E'.j
      ha₂ : Ne E.a₂ 0
      ha₆ : Ne E.a₆ 0
      ha₂' : Ne E'.a₂ 0
      ha₆' : Ne E'.a₆ 0
      this : NeZero 2
      u : F
      hu : Eq (HPow.hPow u 2) (HDiv.hDiv E.a₂ E'.a₂)
      hu0 : Ne u 0
      ⊢ Eq (E.variableChange { u := Units.mk0 u hu0, r := 0, s := 0, t := 0 }).a₃ E' …
    -/
  · simp_rw [variableChange_a₃, a₁_of_isCharThreeJNeZeroNF, a₃_of_isCharThreeJNeZeroNF]
    /-
      case h.a₃
      F : Type u_1
      inst✝⁶ : Field F
      inst✝⁵ : IsSepClosed F
      E E' : WeierstrassCurve F
      inst✝⁴ : E.IsElliptic
      inst✝³ : E'.IsElliptic
      inst✝² : CharP F 3
      inst✝¹ : E.IsCharThreeJNeZeroNF
      inst✝ : E'.IsCharThreeJNeZeroNF
      heq : Eq E.j E'.j
      ha₂ : Ne E.a₂ 0
      ha₆ : Ne E.a₆ 0
      ha₂' : Ne E'.a₂ 0
      ha₆' : Ne E'.a₆ 0
      this : NeZero 2
      u : F
      hu : Eq (HPow.hPow u 2) (HDiv.hDiv E.a₂ E'.a₂)
      hu0 : Ne u 0
      ⊢ Eq (HMul.hMul (HPow.hPow (↑(Inv.inv (Units.mk0 u hu0))) 3) (HAdd.hAdd (HAdd. …
    -/
    ring1
    /-
      🎉 no goals
    -/
  · simp_rw [variableChange_a₄, a₁_of_isCharThreeJNeZeroNF, a₃_of_isCharThreeJNeZeroNF,
      a₄_of_isCharThreeJNeZeroNF]
    /-
      case h.a₄
      F : Type u_1
      inst✝⁶ : Field F
      inst✝⁵ : IsSepClosed F
      E E' : WeierstrassCurve F
      inst✝⁴ : E.IsElliptic
      inst✝³ : E'.IsElliptic
      inst✝² : CharP F 3
      inst✝¹ : E.IsCharThreeJNeZeroNF
      inst✝ : E'.IsCharThreeJNeZeroNF
      heq : Eq E.j E'.j
      ha₂ : Ne E.a₂ 0
      ha₆ : Ne E.a₆ 0
      ha₂' : Ne E'.a₂ 0
      ha₆' : Ne E'.a₆ 0
      this : NeZero 2
      u : F
      hu : Eq (HPow.hPow u 2) (HDiv.hDiv E.a₂ E'.a₂)
      hu0 : Ne u 0
      ⊢ Eq (HMul.hMul (HPow.hPow (↑(Inv.inv (Units.mk0 u hu0))) 4) (HSub.hSub (HAdd. …
    -/
    ring1
    /-
      🎉 no goals
    -/
    /-
      case h.a₆
      F : Type u_1
      inst✝⁶ : Field F
      inst✝⁵ : IsSepClosed F
      E E' : WeierstrassCurve F
      inst✝⁴ : E.IsElliptic
      inst✝³ : E'.IsElliptic
      inst✝² : CharP F 3
      inst✝¹ : E.IsCharThreeJNeZeroNF
      inst✝ : E'.IsCharThreeJNeZeroNF
      heq : Eq E.j E'.j
      ha₂ : Ne E.a₂ 0
      ha₆ : Ne E.a₆ 0
      ha₂' : Ne E'.a₂ 0
      ha₆' : Ne E'.a₆ 0
      this : NeZero 2
      u : F
      hu : Eq (HPow.hPow u 2) (HDiv.hDiv E.a₂ E'.a₂)
      hu0 : Ne u 0
      ⊢ Eq (E.variableChange { u := Units.mk0 u hu0, r := 0, s := 0, t := 0 }).a₆ E' …
    -/
  · simp_rw [j_of_isCharThreeJNeZeroNF_of_char_three, div_eq_div_iff ha₆ ha₆'] at heq
    simp_rw [variableChange_a₆, a₁_of_isCharThreeJNeZeroNF, a₃_of_isCharThreeJNeZeroNF,
      a₄_of_isCharThreeJNeZeroNF, Units.val_inv_eq_inv_val, Units.val_mk0,
      inv_pow, inv_mul_eq_div, pow_mul u 2 3, hu]
    /-
      case h.a₆
      F : Type u_1
      inst✝⁶ : Field F
      inst✝⁵ : IsSepClosed F
      E E' : WeierstrassCurve F
      inst✝⁴ : E.IsElliptic
      inst✝³ : E'.IsElliptic
      inst✝² : CharP F 3
      inst✝¹ : E.IsCharThreeJNeZeroNF
      inst✝ : E'.IsCharThreeJNeZeroNF
      ha₂ : Ne E.a₂ 0
      ha₆ : Ne E.a₆ 0
      ha₂' : Ne E'.a₂ 0
      ha₆' : Ne E'.a₆ 0
      this : NeZero 2
      u : F
      hu : Eq (HPow.hPow u 2) (HDiv.hDiv E.a₂ E'.a₂)
      hu0 : Ne u 0
      heq : Eq (HMul.hMul (Neg.neg (HPow.hPow E.a₂ 3)) E'.a₆) (HMul.hMul (Neg.neg (H …
      ⊢ Eq (HDiv.hDiv (HSub.hSub (HSub.hSub (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HAdd.h …
    -/
    field_simp
    /-
      case h.a₆
      F : Type u_1
      inst✝⁶ : Field F
      inst✝⁵ : IsSepClosed F
      E E' : WeierstrassCurve F
      inst✝⁴ : E.IsElliptic
      inst✝³ : E'.IsElliptic
      inst✝² : CharP F 3
      inst✝¹ : E.IsCharThreeJNeZeroNF
      inst✝ : E'.IsCharThreeJNeZeroNF
      ha₂ : Ne E.a₂ 0
      ha₆ : Ne E.a₆ 0
      ha₂' : Ne E'.a₂ 0
      ha₆' : Ne E'.a₆ 0
      this : NeZero 2
      u : F
      hu : Eq (HPow.hPow u 2) (HDiv.hDiv E.a₂ E'.a₂)
      hu0 : Ne u 0
      heq : Eq (HMul.hMul (Neg.neg (HPow.hPow E.a₂ 3)) E'.a₆) (HMul.hMul (Neg.neg (H …
      ⊢ Eq (HMul.hMul E.a₆ (HPow.hPow E'.a₂ 3)) (HMul.hMul E'.a₆ (HPow.hPow E.a₂ 3))
    -/
    linear_combination heq
    /-
      🎉 no goals
    -/


private lemma exists_variableChange_of_char_three_of_j_eq_zero
    [E.IsShortNF] [E'.IsShortNF] :
    ∃ C : VariableChange F, E.variableChange C = E' := by
  /-
    F : Type u_1
    inst✝⁶ : Field F
    inst✝⁵ : IsSepClosed F
    E E' : WeierstrassCurve F
    inst✝⁴ : E.IsElliptic
    inst✝³ : E'.IsElliptic
    inst✝² : CharP F 3
    inst✝¹ : E.IsShortNF
    inst✝ : E'.IsShortNF
    ⊢ Exists fun C => Eq (E.variableChange C) E'
  -/
  have ha₄ := E.Δ'.ne_zero
  /-
    F : Type u_1
    inst✝⁶ : Field F
    inst✝⁵ : IsSepClosed F
    E E' : WeierstrassCurve F
    inst✝⁴ : E.IsElliptic
    inst✝³ : E'.IsElliptic
    inst✝² : CharP F 3
    inst✝¹ : E.IsShortNF
    inst✝ : E'.IsShortNF
    ha₄ : Ne (↑E.Δ') 0
    ⊢ Exists fun C => Eq (E.variableChange C) E'
  -/
  rw [E.coe_Δ', Δ_of_isShortNF_of_char_three, neg_ne_zero, pow_ne_zero_iff three_ne_zero] at ha₄
  /-
    F : Type u_1
    inst✝⁶ : Field F
    inst✝⁵ : IsSepClosed F
    E E' : WeierstrassCurve F
    inst✝⁴ : E.IsElliptic
    inst✝³ : E'.IsElliptic
    inst✝² : CharP F 3
    inst✝¹ : E.IsShortNF
    inst✝ : E'.IsShortNF
    ha₄ : Ne E.a₄ 0
    ⊢ Exists fun C => Eq (E.variableChange C) E'
  -/
  have ha₄' := E'.Δ'.ne_zero
  /-
    F : Type u_1
    inst✝⁶ : Field F
    inst✝⁵ : IsSepClosed F
    E E' : WeierstrassCurve F
    inst✝⁴ : E.IsElliptic
    inst✝³ : E'.IsElliptic
    inst✝² : CharP F 3
    inst✝¹ : E.IsShortNF
    inst✝ : E'.IsShortNF
    ha₄ : Ne E.a₄ 0
    ha₄' : Ne (↑E'.Δ') 0
    ⊢ Exists fun C => Eq (E.variableChange C) E'
  -/
  rw [E'.coe_Δ', Δ_of_isShortNF_of_char_three, neg_ne_zero, pow_ne_zero_iff three_ne_zero] at ha₄'
  haveI : NeZero (4 : F) := NeZero.mk <| by
    rw [show (4 : F) = 1 by linear_combination CharP.cast_eq_zero F 3]
    exact one_ne_zero
  /-
    F : Type u_1
    inst✝⁶ : Field F
    inst✝⁵ : IsSepClosed F
    E E' : WeierstrassCurve F
    inst✝⁴ : E.IsElliptic
    inst✝³ : E'.IsElliptic
    inst✝² : CharP F 3
    inst✝¹ : E.IsShortNF
    inst✝ : E'.IsShortNF
    ha₄ : Ne E.a₄ 0
    ha₄' : Ne E'.a₄ 0
    this : NeZero 4
    ⊢ Exists fun C => Eq (E.variableChange C) E'
  -/
  obtain ⟨u, hu⟩ := IsSepClosed.exists_pow_nat_eq (E.a₄ / E'.a₄) 4
  obtain ⟨r, hr⟩ := IsSepClosed.exists_root_C_mul_X_pow_add_C_mul_X_add_C' 3 3
    1 _ (E.a₆ - u ^ 6 * E'.a₆) (by norm_num) (by norm_num) ha₄
  have hu0 : u ≠ 0 := by
    rw [← pow_ne_zero_iff four_ne_zero, hu, div_ne_zero_iff]
    exact ⟨ha₄, ha₄'⟩
  /-
    case intro.intro
    F : Type u_1
    inst✝⁶ : Field F
    inst✝⁵ : IsSepClosed F
    E E' : WeierstrassCurve F
    inst✝⁴ : E.IsElliptic
    inst✝³ : E'.IsElliptic
    inst✝² : CharP F 3
    inst✝¹ : E.IsShortNF
    inst✝ : E'.IsShortNF
    ha₄ : Ne E.a₄ 0
    ha₄' : Ne E'.a₄ 0
    this : NeZero 4
    u : F
    hu : Eq (HPow.hPow u 4) (HDiv.hDiv E.a₄ E'.a₄)
    r : F
    hr : Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul 1 (HPow.hPow r 3)) (HMul.hMul E.a₄ r) …
    hu0 : Ne u 0
    ⊢ Exists fun C => Eq (E.variableChange C) E'
  -/
  use ⟨Units.mk0 u hu0, r, 0, 0⟩
  /-
    case h
    F : Type u_1
    inst✝⁶ : Field F
    inst✝⁵ : IsSepClosed F
    E E' : WeierstrassCurve F
    inst✝⁴ : E.IsElliptic
    inst✝³ : E'.IsElliptic
    inst✝² : CharP F 3
    inst✝¹ : E.IsShortNF
    inst✝ : E'.IsShortNF
    ha₄ : Ne E.a₄ 0
    ha₄' : Ne E'.a₄ 0
    this : NeZero 4
    u : F
    hu : Eq (HPow.hPow u 4) (HDiv.hDiv E.a₄ E'.a₄)
    r : F
    hr : Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul 1 (HPow.hPow r 3)) (HMul.hMul E.a₄ r) …
    hu0 : Ne u 0
    ⊢ Eq (E.variableChange { u := Units.mk0 u hu0, r := r, s := 0, t := 0 }) E'
  -/
  ext
    /-
      case h.a₁
      F : Type u_1
      inst✝⁶ : Field F
      inst✝⁵ : IsSepClosed F
      E E' : WeierstrassCurve F
      inst✝⁴ : E.IsElliptic
      inst✝³ : E'.IsElliptic
      inst✝² : CharP F 3
      inst✝¹ : E.IsShortNF
      inst✝ : E'.IsShortNF
      ha₄ : Ne E.a₄ 0
      ha₄' : Ne E'.a₄ 0
      this : NeZero 4
      u : F
      hu : Eq (HPow.hPow u 4) (HDiv.hDiv E.a₄ E'.a₄)
      r : F
      hr : Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul 1 (HPow.hPow r 3)) (HMul.hMul E.a₄ r) …
      hu0 : Ne u 0
      ⊢ Eq (E.variableChange { u := Units.mk0 u hu0, r := r, s := 0, t := 0 }).a₁ E' …
    -/
  · simp_rw [variableChange_a₁, a₁_of_isShortNF]
    /-
      case h.a₁
      F : Type u_1
      inst✝⁶ : Field F
      inst✝⁵ : IsSepClosed F
      E E' : WeierstrassCurve F
      inst✝⁴ : E.IsElliptic
      inst✝³ : E'.IsElliptic
      inst✝² : CharP F 3
      inst✝¹ : E.IsShortNF
      inst✝ : E'.IsShortNF
      ha₄ : Ne E.a₄ 0
      ha₄' : Ne E'.a₄ 0
      this : NeZero 4
      u : F
      hu : Eq (HPow.hPow u 4) (HDiv.hDiv E.a₄ E'.a₄)
      r : F
      hr : Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul 1 (HPow.hPow r 3)) (HMul.hMul E.a₄ r) …
      hu0 : Ne u 0
      ⊢ Eq (HMul.hMul (↑(Inv.inv (Units.mk0 u hu0))) (HAdd.hAdd 0 (HMul.hMul 2 0))) 0
    -/
    ring1
    /-
      🎉 no goals
    -/
  · simp_rw [variableChange_a₂, a₁_of_isShortNF, a₂_of_isShortNF,
      show (3 : F) = 0 from CharP.cast_eq_zero F 3]
    /-
      case h.a₂
      F : Type u_1
      inst✝⁶ : Field F
      inst✝⁵ : IsSepClosed F
      E E' : WeierstrassCurve F
      inst✝⁴ : E.IsElliptic
      inst✝³ : E'.IsElliptic
      inst✝² : CharP F 3
      inst✝¹ : E.IsShortNF
      inst✝ : E'.IsShortNF
      ha₄ : Ne E.a₄ 0
      ha₄' : Ne E'.a₄ 0
      this : NeZero 4
      u : F
      hu : Eq (HPow.hPow u 4) (HDiv.hDiv E.a₄ E'.a₄)
      r : F
      hr : Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul 1 (HPow.hPow r 3)) (HMul.hMul E.a₄ r) …
      hu0 : Ne u 0
      ⊢ Eq (HMul.hMul (HPow.hPow (↑(Inv.inv (Units.mk0 u hu0))) 2) (HSub.hSub (HAdd. …
    -/
    ring1
    /-
      🎉 no goals
    -/
    /-
      case h.a₃
      F : Type u_1
      inst✝⁶ : Field F
      inst✝⁵ : IsSepClosed F
      E E' : WeierstrassCurve F
      inst✝⁴ : E.IsElliptic
      inst✝³ : E'.IsElliptic
      inst✝² : CharP F 3
      inst✝¹ : E.IsShortNF
      inst✝ : E'.IsShortNF
      ha₄ : Ne E.a₄ 0
      ha₄' : Ne E'.a₄ 0
      this : NeZero 4
      u : F
      hu : Eq (HPow.hPow u 4) (HDiv.hDiv E.a₄ E'.a₄)
      r : F
      hr : Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul 1 (HPow.hPow r 3)) (HMul.hMul E.a₄ r) …
      hu0 : Ne u 0
      ⊢ Eq (E.variableChange { u := Units.mk0 u hu0, r := r, s := 0, t := 0 }).a₃ E' …
    -/
  · simp_rw [variableChange_a₃, a₁_of_isShortNF, a₃_of_isShortNF]
    /-
      case h.a₃
      F : Type u_1
      inst✝⁶ : Field F
      inst✝⁵ : IsSepClosed F
      E E' : WeierstrassCurve F
      inst✝⁴ : E.IsElliptic
      inst✝³ : E'.IsElliptic
      inst✝² : CharP F 3
      inst✝¹ : E.IsShortNF
      inst✝ : E'.IsShortNF
      ha₄ : Ne E.a₄ 0
      ha₄' : Ne E'.a₄ 0
      this : NeZero 4
      u : F
      hu : Eq (HPow.hPow u 4) (HDiv.hDiv E.a₄ E'.a₄)
      r : F
      hr : Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul 1 (HPow.hPow r 3)) (HMul.hMul E.a₄ r) …
      hu0 : Ne u 0
      ⊢ Eq (HMul.hMul (HPow.hPow (↑(Inv.inv (Units.mk0 u hu0))) 3) (HAdd.hAdd (HAdd. …
    -/
    ring1
    /-
      🎉 no goals
    -/
  · simp_rw [variableChange_a₄, a₁_of_isShortNF, a₂_of_isShortNF, a₃_of_isShortNF,
      Units.val_inv_eq_inv_val, Units.val_mk0, inv_pow, inv_mul_eq_div, hu,
      show (3 : F) = 0 from CharP.cast_eq_zero F 3]
    /-
      case h.a₄
      F : Type u_1
      inst✝⁶ : Field F
      inst✝⁵ : IsSepClosed F
      E E' : WeierstrassCurve F
      inst✝⁴ : E.IsElliptic
      inst✝³ : E'.IsElliptic
      inst✝² : CharP F 3
      inst✝¹ : E.IsShortNF
      inst✝ : E'.IsShortNF
      ha₄ : Ne E.a₄ 0
      ha₄' : Ne E'.a₄ 0
      this : NeZero 4
      u : F
      hu : Eq (HPow.hPow u 4) (HDiv.hDiv E.a₄ E'.a₄)
      r : F
      hr : Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul 1 (HPow.hPow r 3)) (HMul.hMul E.a₄ r) …
      hu0 : Ne u 0
      ⊢ Eq (HDiv.hDiv (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (HSub.hSub E.a₄ (H …
    -/
    field_simp
    /-
      🎉 no goals
    -/
  · simp_rw [variableChange_a₆, a₁_of_isShortNF, a₂_of_isShortNF, a₃_of_isShortNF,
      Units.val_inv_eq_inv_val, Units.val_mk0, inv_pow, inv_mul_eq_div]
    /-
      case h.a₆
      F : Type u_1
      inst✝⁶ : Field F
      inst✝⁵ : IsSepClosed F
      E E' : WeierstrassCurve F
      inst✝⁴ : E.IsElliptic
      inst✝³ : E'.IsElliptic
      inst✝² : CharP F 3
      inst✝¹ : E.IsShortNF
      inst✝ : E'.IsShortNF
      ha₄ : Ne E.a₄ 0
      ha₄' : Ne E'.a₄ 0
      this : NeZero 4
      u : F
      hu : Eq (HPow.hPow u 4) (HDiv.hDiv E.a₄ E'.a₄)
      r : F
      hr : Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul 1 (HPow.hPow r 3)) (HMul.hMul E.a₄ r) …
      hu0 : Ne u 0
      ⊢ Eq (HDiv.hDiv (HSub.hSub (HSub.hSub (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HAdd.h …
    -/
    field_simp
    /-
      case h.a₆
      F : Type u_1
      inst✝⁶ : Field F
      inst✝⁵ : IsSepClosed F
      E E' : WeierstrassCurve F
      inst✝⁴ : E.IsElliptic
      inst✝³ : E'.IsElliptic
      inst✝² : CharP F 3
      inst✝¹ : E.IsShortNF
      inst✝ : E'.IsShortNF
      ha₄ : Ne E.a₄ 0
      ha₄' : Ne E'.a₄ 0
      this : NeZero 4
      u : F
      hu : Eq (HPow.hPow u 4) (HDiv.hDiv E.a₄ E'.a₄)
      r : F
      hr : Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul 1 (HPow.hPow r 3)) (HMul.hMul E.a₄ r) …
      hu0 : Ne u 0
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd E.a₆ (HMul.hMul r E.a₄)) (HPow.hPow r 3)) (HMul.hMu …
    -/
    linear_combination hr
    /-
      🎉 no goals
    -/


private lemma exists_variableChange_of_char_three (heq : E.j = E'.j) :
    ∃ C : VariableChange F, E.variableChange C = E' := by
  /-
    F : Type u_1
    inst✝⁴ : Field F
    inst✝³ : IsSepClosed F
    E E' : WeierstrassCurve F
    inst✝² : E.IsElliptic
    inst✝¹ : E'.IsElliptic
    inst✝ : CharP F 3
    heq : Eq E.j E'.j
    ⊢ Exists fun C => Eq (E.variableChange C) E'
  -/
  obtain ⟨C, _ | _⟩ := E.exists_variableChange_isCharThreeNF
    /-
      case intro.of_j_ne_zero
      F : Type u_1
      inst✝⁵ : Field F
      inst✝⁴ : IsSepClosed F
      E E' : WeierstrassCurve F
      inst✝³ : E.IsElliptic
      inst✝² : E'.IsElliptic
      inst✝¹ : CharP F 3
      heq : Eq E.j E'.j
      C : WeierstrassCurve.VariableChange F
      inst✝ : (E.variableChange C).IsCharThreeJNeZeroNF
      ⊢ Exists fun C => Eq (E.variableChange C) E'
    -/
  · obtain ⟨C', _ | _⟩ := E'.exists_variableChange_isCharThreeNF
      /-
        case intro.of_j_ne_zero.intro.of_j_ne_zero
        F : Type u_1
        inst✝⁶ : Field F
        inst✝⁵ : IsSepClosed F
        E E' : WeierstrassCurve F
        inst✝⁴ : E.IsElliptic
        inst✝³ : E'.IsElliptic
        inst✝² : CharP F 3
        heq : Eq E.j E'.j
        C : WeierstrassCurve.VariableChange F
        inst✝¹ : (E.variableChange C).IsCharThreeJNeZeroNF
        C' : WeierstrassCurve.VariableChange F
        inst✝ : (E'.variableChange C').IsCharThreeJNeZeroNF
        ⊢ Exists fun C => Eq (E.variableChange C) E'
      -/
    · rw [← variableChange_j E C, ← variableChange_j E' C'] at heq
      /-
        case intro.of_j_ne_zero.intro.of_j_ne_zero
        F : Type u_1
        inst✝⁶ : Field F
        inst✝⁵ : IsSepClosed F
        E E' : WeierstrassCurve F
        inst✝⁴ : E.IsElliptic
        inst✝³ : E'.IsElliptic
        inst✝² : CharP F 3
        C : WeierstrassCurve.VariableChange F
        inst✝¹ : (E.variableChange C).IsCharThreeJNeZeroNF
        C' : WeierstrassCurve.VariableChange F
        heq : Eq (E.variableChange C).j (E'.variableChange C').j
        inst✝ : (E'.variableChange C').IsCharThreeJNeZeroNF
        ⊢ Exists fun C => Eq (E.variableChange C) E'
      -/
      obtain ⟨C'', hC⟩ := exists_variableChange_of_char_three_of_j_ne_zero _ _ heq
      /-
        case intro.of_j_ne_zero.intro.of_j_ne_zero.intro
        F : Type u_1
        inst✝⁶ : Field F
        inst✝⁵ : IsSepClosed F
        E E' : WeierstrassCurve F
        inst✝⁴ : E.IsElliptic
        inst✝³ : E'.IsElliptic
        inst✝² : CharP F 3
        C : WeierstrassCurve.VariableChange F
        inst✝¹ : (E.variableChange C).IsCharThreeJNeZeroNF
        C' : WeierstrassCurve.VariableChange F
        heq : Eq (E.variableChange C).j (E'.variableChange C').j
        inst✝ : (E'.variableChange C').IsCharThreeJNeZeroNF
        C'' : WeierstrassCurve.VariableChange F
        hC : Eq ((E.variableChange C).variableChange C'') (E'.variableChange C')
        ⊢ Exists fun C => Eq (E.variableChange C) E'
      -/
      use (C'.inv.comp C'').comp C
      rw [variableChange_comp, variableChange_comp, hC, ← variableChange_comp,
        VariableChange.comp_left_inv, variableChange_id]
      /-
        case intro.of_j_ne_zero.intro.of_j_eq_zero
        F : Type u_1
        inst✝⁶ : Field F
        inst✝⁵ : IsSepClosed F
        E E' : WeierstrassCurve F
        inst✝⁴ : E.IsElliptic
        inst✝³ : E'.IsElliptic
        inst✝² : CharP F 3
        heq : Eq E.j E'.j
        C : WeierstrassCurve.VariableChange F
        inst✝¹ : (E.variableChange C).IsCharThreeJNeZeroNF
        C' : WeierstrassCurve.VariableChange F
        inst✝ : (E'.variableChange C').IsShortNF
        ⊢ Exists fun C => Eq (E.variableChange C) E'
      -/
    · have h := (E.variableChange C).j_ne_zero_of_isCharThreeJNeZeroNF_of_char_three
      /-
        case intro.of_j_ne_zero.intro.of_j_eq_zero
        F : Type u_1
        inst✝⁶ : Field F
        inst✝⁵ : IsSepClosed F
        E E' : WeierstrassCurve F
        inst✝⁴ : E.IsElliptic
        inst✝³ : E'.IsElliptic
        inst✝² : CharP F 3
        heq : Eq E.j E'.j
        C : WeierstrassCurve.VariableChange F
        inst✝¹ : (E.variableChange C).IsCharThreeJNeZeroNF
        C' : WeierstrassCurve.VariableChange F
        inst✝ : (E'.variableChange C').IsShortNF
        h : Ne (E.variableChange C).j 0
        ⊢ Exists fun C => Eq (E.variableChange C) E'
      -/
      rw [variableChange_j, heq, ← variableChange_j E' C', j_of_isShortNF_of_char_three] at h
      /-
        case intro.of_j_ne_zero.intro.of_j_eq_zero
        F : Type u_1
        inst✝⁶ : Field F
        inst✝⁵ : IsSepClosed F
        E E' : WeierstrassCurve F
        inst✝⁴ : E.IsElliptic
        inst✝³ : E'.IsElliptic
        inst✝² : CharP F 3
        heq : Eq E.j E'.j
        C : WeierstrassCurve.VariableChange F
        inst✝¹ : (E.variableChange C).IsCharThreeJNeZeroNF
        C' : WeierstrassCurve.VariableChange F
        inst✝ : (E'.variableChange C').IsShortNF
        h : Ne 0 0
        ⊢ Exists fun C => Eq (E.variableChange C) E'
      -/
      exact False.elim (h rfl)
      /-
        🎉 no goals
      -/
    /-
      case intro.of_j_eq_zero
      F : Type u_1
      inst✝⁵ : Field F
      inst✝⁴ : IsSepClosed F
      E E' : WeierstrassCurve F
      inst✝³ : E.IsElliptic
      inst✝² : E'.IsElliptic
      inst✝¹ : CharP F 3
      heq : Eq E.j E'.j
      C : WeierstrassCurve.VariableChange F
      inst✝ : (E.variableChange C).IsShortNF
      ⊢ Exists fun C => Eq (E.variableChange C) E'
    -/
  · obtain ⟨C', _ | _⟩ := E'.exists_variableChange_isCharThreeNF
      /-
        case intro.of_j_eq_zero.intro.of_j_ne_zero
        F : Type u_1
        inst✝⁶ : Field F
        inst✝⁵ : IsSepClosed F
        E E' : WeierstrassCurve F
        inst✝⁴ : E.IsElliptic
        inst✝³ : E'.IsElliptic
        inst✝² : CharP F 3
        heq : Eq E.j E'.j
        C : WeierstrassCurve.VariableChange F
        inst✝¹ : (E.variableChange C).IsShortNF
        C' : WeierstrassCurve.VariableChange F
        inst✝ : (E'.variableChange C').IsCharThreeJNeZeroNF
        ⊢ Exists fun C => Eq (E.variableChange C) E'
      -/
    · have h := (E'.variableChange C').j_ne_zero_of_isCharThreeJNeZeroNF_of_char_three
      /-
        case intro.of_j_eq_zero.intro.of_j_ne_zero
        F : Type u_1
        inst✝⁶ : Field F
        inst✝⁵ : IsSepClosed F
        E E' : WeierstrassCurve F
        inst✝⁴ : E.IsElliptic
        inst✝³ : E'.IsElliptic
        inst✝² : CharP F 3
        heq : Eq E.j E'.j
        C : WeierstrassCurve.VariableChange F
        inst✝¹ : (E.variableChange C).IsShortNF
        C' : WeierstrassCurve.VariableChange F
        inst✝ : (E'.variableChange C').IsCharThreeJNeZeroNF
        h : Ne (E'.variableChange C').j 0
        ⊢ Exists fun C => Eq (E.variableChange C) E'
      -/
      rw [variableChange_j, ← heq, ← variableChange_j E C, j_of_isShortNF_of_char_three] at h
      /-
        case intro.of_j_eq_zero.intro.of_j_ne_zero
        F : Type u_1
        inst✝⁶ : Field F
        inst✝⁵ : IsSepClosed F
        E E' : WeierstrassCurve F
        inst✝⁴ : E.IsElliptic
        inst✝³ : E'.IsElliptic
        inst✝² : CharP F 3
        heq : Eq E.j E'.j
        C : WeierstrassCurve.VariableChange F
        inst✝¹ : (E.variableChange C).IsShortNF
        C' : WeierstrassCurve.VariableChange F
        inst✝ : (E'.variableChange C').IsCharThreeJNeZeroNF
        h : Ne 0 0
        ⊢ Exists fun C => Eq (E.variableChange C) E'
      -/
      exact False.elim (h rfl)
      /-
        🎉 no goals
      -/
    · obtain ⟨C'', hC⟩ := exists_variableChange_of_char_three_of_j_eq_zero
        (E.variableChange C) (E'.variableChange C')
      /-
        case intro.of_j_eq_zero.intro.of_j_eq_zero.intro
        F : Type u_1
        inst✝⁶ : Field F
        inst✝⁵ : IsSepClosed F
        E E' : WeierstrassCurve F
        inst✝⁴ : E.IsElliptic
        inst✝³ : E'.IsElliptic
        inst✝² : CharP F 3
        heq : Eq E.j E'.j
        C : WeierstrassCurve.VariableChange F
        inst✝¹ : (E.variableChange C).IsShortNF
        C' : WeierstrassCurve.VariableChange F
        inst✝ : (E'.variableChange C').IsShortNF
        C'' : WeierstrassCurve.VariableChange F
        hC : Eq ((E.variableChange C).variableChange C'') (E'.variableChange C')
        ⊢ Exists fun C => Eq (E.variableChange C) E'
      -/
      use (C'.inv.comp C'').comp C
      rw [variableChange_comp, variableChange_comp, hC, ← variableChange_comp,
        VariableChange.comp_left_inv, variableChange_id]


private lemma exists_variableChange_of_char_ne_two_or_three
    {p : ℕ} [CharP F p] (hchar2 : p ≠ 2) (hchar3 : p ≠ 3) (heq : E.j = E'.j) :
    ∃ C : VariableChange F, E.variableChange C = E' := by
  /-
    F : Type u_1
    inst✝⁴ : Field F
    inst✝³ : IsSepClosed F
    E E' : WeierstrassCurve F
    inst✝² : E.IsElliptic
    inst✝¹ : E'.IsElliptic
    p : Nat
    inst✝ : CharP F p
    hchar2 : Ne p 2
    hchar3 : Ne p 3
    heq : Eq E.j E'.j
    ⊢ Exists fun C => Eq (E.variableChange C) E'
  -/
  replace hchar2 : (2 : F) ≠ 0 := CharP.cast_ne_zero_of_ne_of_prime F Nat.prime_two hchar2
  /-
    F : Type u_1
    inst✝⁴ : Field F
    inst✝³ : IsSepClosed F
    E E' : WeierstrassCurve F
    inst✝² : E.IsElliptic
    inst✝¹ : E'.IsElliptic
    p : Nat
    inst✝ : CharP F p
    hchar3 : Ne p 3
    heq : Eq E.j E'.j
    hchar2 : Ne 2 0
    ⊢ Exists fun C => Eq (E.variableChange C) E'
  -/
  replace hchar3 : (3 : F) ≠ 0 := CharP.cast_ne_zero_of_ne_of_prime F Nat.prime_three hchar3
  /-
    F : Type u_1
    inst✝⁴ : Field F
    inst✝³ : IsSepClosed F
    E E' : WeierstrassCurve F
    inst✝² : E.IsElliptic
    inst✝¹ : E'.IsElliptic
    p : Nat
    inst✝ : CharP F p
    heq : Eq E.j E'.j
    hchar2 : Ne 2 0
    hchar3 : Ne 3 0
    ⊢ Exists fun C => Eq (E.variableChange C) E'
  -/
  haveI := NeZero.mk hchar2
  haveI : NeZero (4 : F) := NeZero.mk <| by
    have := pow_ne_zero 2 hchar2
    norm_num1 at this
    exact this
  haveI : NeZero (6 : F) := NeZero.mk <| by
    have := mul_ne_zero hchar2 hchar3
    norm_num1 at this
    exact this
  /-
    F : Type u_1
    inst✝⁴ : Field F
    inst✝³ : IsSepClosed F
    E E' : WeierstrassCurve F
    inst✝² : E.IsElliptic
    inst✝¹ : E'.IsElliptic
    p : Nat
    inst✝ : CharP F p
    heq : Eq E.j E'.j
    hchar2 : Ne 2 0
    hchar3 : Ne 3 0
    this✝¹ : NeZero 2
    this✝ : NeZero 4
    this : NeZero 6
    ⊢ Exists fun C => Eq (E.variableChange C) E'
  -/
  letI : Invertible (2 : F) := invertibleOfNonzero hchar2
  /-
    F : Type u_1
    inst✝⁴ : Field F
    inst✝³ : IsSepClosed F
    E E' : WeierstrassCurve F
    inst✝² : E.IsElliptic
    inst✝¹ : E'.IsElliptic
    p : Nat
    inst✝ : CharP F p
    heq : Eq E.j E'.j
    hchar2 : Ne 2 0
    hchar3 : Ne 3 0
    this✝² : NeZero 2
    this✝¹ : NeZero 4
    this✝ : NeZero 6
    this : Invertible 2 := invertibleOfNonzero hchar2
    ⊢ Exists fun C => Eq (E.variableChange C) E'
  -/
  letI : Invertible (3 : F) := invertibleOfNonzero hchar3
  /-
    F : Type u_1
    inst✝⁴ : Field F
    inst✝³ : IsSepClosed F
    E E' : WeierstrassCurve F
    inst✝² : E.IsElliptic
    inst✝¹ : E'.IsElliptic
    p : Nat
    inst✝ : CharP F p
    heq : Eq E.j E'.j
    hchar2 : Ne 2 0
    hchar3 : Ne 3 0
    this✝³ : NeZero 2
    this✝² : NeZero 4
    this✝¹ : NeZero 6
    this✝ : Invertible 2 := invertibleOfNonzero hchar2
    this : Invertible 3 := invertibleOfNonzero hchar3
    ⊢ Exists fun C => Eq (E.variableChange C) E'
  -/
  wlog _ : E.IsShortNF generalizing E
    /-
      case inr
      F : Type u_1
      inst✝⁴ : Field F
      inst✝³ : IsSepClosed F
      E E' : WeierstrassCurve F
      inst✝² : E.IsElliptic
      inst✝¹ : E'.IsElliptic
      p : Nat
      inst✝ : CharP F p
      heq : Eq E.j E'.j
      hchar2 : Ne 2 0
      hchar3 : Ne 3 0
      this✝⁴ : NeZero 2
      this✝³ : NeZero 4
      this✝² : NeZero 6
      this✝¹ : Invertible 2 := invertibleOfNonzero hchar2
      this✝ : Invertible 3 := invertibleOfNonzero hchar3
      this : ∀ (E : WeierstrassCurve F) [inst : E.IsElliptic], Eq E.j E'.j → E.IsSho …
      h✝ : Not E.IsShortNF
      ⊢ Exists fun C => Eq (E.variableChange C) E'
    -/
  · obtain ⟨C, hE⟩ := E.exists_variableChange_isShortNF
    /-
      case inr.intro
      F : Type u_1
      inst✝⁴ : Field F
      inst✝³ : IsSepClosed F
      E E' : WeierstrassCurve F
      inst✝² : E.IsElliptic
      inst✝¹ : E'.IsElliptic
      p : Nat
      inst✝ : CharP F p
      heq : Eq E.j E'.j
      hchar2 : Ne 2 0
      hchar3 : Ne 3 0
      this✝⁴ : NeZero 2
      this✝³ : NeZero 4
      this✝² : NeZero 6
      this✝¹ : Invertible 2 := invertibleOfNonzero hchar2
      this✝ : Invertible 3 := invertibleOfNonzero hchar3
      this : ∀ (E : WeierstrassCurve F) [inst : E.IsElliptic], Eq E.j E'.j → E.IsSho …
      h✝ : Not E.IsShortNF
      C : WeierstrassCurve.VariableChange F
      hE : (E.variableChange C).IsShortNF
      ⊢ Exists fun C => Eq (E.variableChange C) E'
    -/
    rw [← variableChange_j E C] at heq
    /-
      case inr.intro
      F : Type u_1
      inst✝⁴ : Field F
      inst✝³ : IsSepClosed F
      E E' : WeierstrassCurve F
      inst✝² : E.IsElliptic
      inst✝¹ : E'.IsElliptic
      p : Nat
      inst✝ : CharP F p
      hchar2 : Ne 2 0
      hchar3 : Ne 3 0
      this✝⁴ : NeZero 2
      this✝³ : NeZero 4
      this✝² : NeZero 6
      this✝¹ : Invertible 2 := invertibleOfNonzero hchar2
      this✝ : Invertible 3 := invertibleOfNonzero hchar3
      this : ∀ (E : WeierstrassCurve F) [inst : E.IsElliptic], Eq E.j E'.j → E.IsSho …
      h✝ : Not E.IsShortNF
      C : WeierstrassCurve.VariableChange F
      heq : Eq (E.variableChange C).j E'.j
      hE : (E.variableChange C).IsShortNF
      ⊢ Exists fun C => Eq (E.variableChange C) E'
    -/
    obtain ⟨C', hC⟩ := this _ heq hE
    /-
      case inr.intro.intro
      F : Type u_1
      inst✝⁴ : Field F
      inst✝³ : IsSepClosed F
      E E' : WeierstrassCurve F
      inst✝² : E.IsElliptic
      inst✝¹ : E'.IsElliptic
      p : Nat
      inst✝ : CharP F p
      hchar2 : Ne 2 0
      hchar3 : Ne 3 0
      this✝⁴ : NeZero 2
      this✝³ : NeZero 4
      this✝² : NeZero 6
      this✝¹ : Invertible 2 := invertibleOfNonzero hchar2
      this✝ : Invertible 3 := invertibleOfNonzero hchar3
      this : ∀ (E : WeierstrassCurve F) [inst : E.IsElliptic], Eq E.j E'.j → E.IsSho …
      h✝ : Not E.IsShortNF
      C : WeierstrassCurve.VariableChange F
      heq : Eq (E.variableChange C).j E'.j
      hE : (E.variableChange C).IsShortNF
      C' : WeierstrassCurve.VariableChange F
      hC : Eq ((E.variableChange C).variableChange C') E'
      ⊢ Exists fun C => Eq (E.variableChange C) E'
    -/
    exact ⟨C'.comp C, by rwa [variableChange_comp]⟩
    /-
      🎉 no goals
    -/
  /-
    F : Type u_1
    inst✝⁵ : Field F
    inst✝⁴ : IsSepClosed F
    E✝ E' : WeierstrassCurve F
    inst✝³ : E✝.IsElliptic
    inst✝² : E'.IsElliptic
    p : Nat
    inst✝¹ : CharP F p
    hchar2 : Ne 2 0
    hchar3 : Ne 3 0
    this✝³ : NeZero 2
    this✝² : NeZero 4
    this✝¹ : NeZero 6
    this✝ : Invertible 2 := invertibleOfNonzero hchar2
    this : Invertible 3 := invertibleOfNonzero hchar3
    E : WeierstrassCurve F
    inst✝ : E.IsElliptic
    heq : Eq E.j E'.j
    h✝ : E.IsShortNF
    ⊢ Exists fun C => Eq (E.variableChange C) E'
  -/
  wlog _ : E'.IsShortNF generalizing E'
    /-
      case inr
      F : Type u_1
      inst✝⁵ : Field F
      inst✝⁴ : IsSepClosed F
      E✝ E' : WeierstrassCurve F
      inst✝³ : E✝.IsElliptic
      inst✝² : E'.IsElliptic
      p : Nat
      inst✝¹ : CharP F p
      hchar2 : Ne 2 0
      hchar3 : Ne 3 0
      this✝⁴ : NeZero 2
      this✝³ : NeZero 4
      this✝² : NeZero 6
      this✝¹ : Invertible 2 := invertibleOfNonzero hchar2
      this✝ : Invertible 3 := invertibleOfNonzero hchar3
      E : WeierstrassCurve F
      inst✝ : E.IsElliptic
      heq : Eq E.j E'.j
      h✝¹ : E.IsShortNF
      this : ∀ (E' : WeierstrassCurve F) [inst : E'.IsElliptic], Eq E.j E'.j → E'.Is …
      h✝ : Not E'.IsShortNF
      ⊢ Exists fun C => Eq (E.variableChange C) E'
    -/
  · obtain ⟨C, hE'⟩ := E'.exists_variableChange_isShortNF
    /-
      case inr.intro
      F : Type u_1
      inst✝⁵ : Field F
      inst✝⁴ : IsSepClosed F
      E✝ E' : WeierstrassCurve F
      inst✝³ : E✝.IsElliptic
      inst✝² : E'.IsElliptic
      p : Nat
      inst✝¹ : CharP F p
      hchar2 : Ne 2 0
      hchar3 : Ne 3 0
      this✝⁴ : NeZero 2
      this✝³ : NeZero 4
      this✝² : NeZero 6
      this✝¹ : Invertible 2 := invertibleOfNonzero hchar2
      this✝ : Invertible 3 := invertibleOfNonzero hchar3
      E : WeierstrassCurve F
      inst✝ : E.IsElliptic
      heq : Eq E.j E'.j
      h✝¹ : E.IsShortNF
      this : ∀ (E' : WeierstrassCurve F) [inst : E'.IsElliptic], Eq E.j E'.j → E'.Is …
      h✝ : Not E'.IsShortNF
      C : WeierstrassCurve.VariableChange F
      hE' : (E'.variableChange C).IsShortNF
      ⊢ Exists fun C => Eq (E.variableChange C) E'
    -/
    rw [← variableChange_j E' C] at heq
    /-
      case inr.intro
      F : Type u_1
      inst✝⁵ : Field F
      inst✝⁴ : IsSepClosed F
      E✝ E' : WeierstrassCurve F
      inst✝³ : E✝.IsElliptic
      inst✝² : E'.IsElliptic
      p : Nat
      inst✝¹ : CharP F p
      hchar2 : Ne 2 0
      hchar3 : Ne 3 0
      this✝⁴ : NeZero 2
      this✝³ : NeZero 4
      this✝² : NeZero 6
      this✝¹ : Invertible 2 := invertibleOfNonzero hchar2
      this✝ : Invertible 3 := invertibleOfNonzero hchar3
      E : WeierstrassCurve F
      inst✝ : E.IsElliptic
      h✝¹ : E.IsShortNF
      this : ∀ (E' : WeierstrassCurve F) [inst : E'.IsElliptic], Eq E.j E'.j → E'.Is …
      h✝ : Not E'.IsShortNF
      C : WeierstrassCurve.VariableChange F
      heq : Eq E.j (E'.variableChange C).j
      hE' : (E'.variableChange C).IsShortNF
      ⊢ Exists fun C => Eq (E.variableChange C) E'
    -/
    obtain ⟨C', hC⟩ := this _ heq hE'
    exact ⟨C.inv.comp C', by rw [variableChange_comp, hC, ← variableChange_comp,
      VariableChange.comp_left_inv, variableChange_id]⟩
  simp_rw [j, Units.val_inv_eq_inv_val, inv_mul_eq_div,
    div_eq_div_iff E.Δ'.ne_zero E'.Δ'.ne_zero, coe_Δ', Δ_of_isShortNF, c₄_of_isShortNF] at heq
  replace heq : E.a₄ ^ 3 * E'.a₆ ^ 2 = E'.a₄ ^ 3 * E.a₆ ^ 2 := by
    letI : Invertible (47775744 : F) := invertibleOfNonzero <| by
      have := mul_ne_zero (pow_ne_zero 16 hchar2) (pow_ne_zero 6 hchar3)
      norm_num1 at this
      exact this
    rw [← mul_right_inj_of_invertible (47775744 : F)]
    linear_combination heq
  /-
    F : Type u_1
    inst✝⁶ : Field F
    inst✝⁵ : IsSepClosed F
    E✝ E'✝ : WeierstrassCurve F
    inst✝⁴ : E✝.IsElliptic
    inst✝³ : E'✝.IsElliptic
    p : Nat
    inst✝² : CharP F p
    hchar2 : Ne 2 0
    hchar3 : Ne 3 0
    this✝³ : NeZero 2
    this✝² : NeZero 4
    this✝¹ : NeZero 6
    this✝ : Invertible 2 := invertibleOfNonzero hchar2
    this : Invertible 3 := invertibleOfNonzero hchar3
    E : WeierstrassCurve F
    inst✝¹ : E.IsElliptic
    h✝¹ : E.IsShortNF
    E' : WeierstrassCurve F
    inst✝ : E'.IsElliptic
    h✝ : E'.IsShortNF
    heq : Eq (HMul.hMul (HPow.hPow E.a₄ 3) (HPow.hPow E'.a₆ 2)) (HMul.hMul (HPow.h …
    ⊢ Exists fun C => Eq (E.variableChange C) E'
  -/
  by_cases ha₄ : E.a₄ = 0
    /-
      case pos
      F : Type u_1
      inst✝⁶ : Field F
      inst✝⁵ : IsSepClosed F
      E✝ E'✝ : WeierstrassCurve F
      inst✝⁴ : E✝.IsElliptic
      inst✝³ : E'✝.IsElliptic
      p : Nat
      inst✝² : CharP F p
      hchar2 : Ne 2 0
      hchar3 : Ne 3 0
      this✝³ : NeZero 2
      this✝² : NeZero 4
      this✝¹ : NeZero 6
      this✝ : Invertible 2 := invertibleOfNonzero hchar2
      this : Invertible 3 := invertibleOfNonzero hchar3
      E : WeierstrassCurve F
      inst✝¹ : E.IsElliptic
      h✝¹ : E.IsShortNF
      E' : WeierstrassCurve F
      inst✝ : E'.IsElliptic
      h✝ : E'.IsShortNF
      heq : Eq (HMul.hMul (HPow.hPow E.a₄ 3) (HPow.hPow E'.a₆ 2)) (HMul.hMul (HPow.h …
      ha₄ : Eq E.a₄ 0
      ⊢ Exists fun C => Eq (E.variableChange C) E'
    -/
  · have ha₆ := E.Δ'.ne_zero
    rw [coe_Δ', Δ_of_isShortNF, ha₄, zero_pow three_ne_zero, mul_zero, zero_add, ← mul_assoc,
      mul_ne_zero_iff, pow_ne_zero_iff two_ne_zero] at ha₆
    /-
      case pos
      F : Type u_1
      inst✝⁶ : Field F
      inst✝⁵ : IsSepClosed F
      E✝ E'✝ : WeierstrassCurve F
      inst✝⁴ : E✝.IsElliptic
      inst✝³ : E'✝.IsElliptic
      p : Nat
      inst✝² : CharP F p
      hchar2 : Ne 2 0
      hchar3 : Ne 3 0
      this✝³ : NeZero 2
      this✝² : NeZero 4
      this✝¹ : NeZero 6
      this✝ : Invertible 2 := invertibleOfNonzero hchar2
      this : Invertible 3 := invertibleOfNonzero hchar3
      E : WeierstrassCurve F
      inst✝¹ : E.IsElliptic
      h✝¹ : E.IsShortNF
      E' : WeierstrassCurve F
      inst✝ : E'.IsElliptic
      h✝ : E'.IsShortNF
      heq : Eq (HMul.hMul (HPow.hPow E.a₄ 3) (HPow.hPow E'.a₆ 2)) (HMul.hMul (HPow.h …
      ha₄ : Eq E.a₄ 0
      ha₆ : And (Ne (HMul.hMul (-16) 27) 0) (Ne E.a₆ 0)
      ⊢ Exists fun C => Eq (E.variableChange C) E'
    -/
    replace ha₆ := ha₆.2
    have ha₄' : E'.a₄ = 0 := by
      rw [ha₄, zero_pow three_ne_zero, zero_mul, zero_eq_mul] at heq
      exact (pow_eq_zero_iff three_ne_zero).1 <| heq.resolve_right <| pow_ne_zero 2 ha₆
    /-
      case pos
      F : Type u_1
      inst✝⁶ : Field F
      inst✝⁵ : IsSepClosed F
      E✝ E'✝ : WeierstrassCurve F
      inst✝⁴ : E✝.IsElliptic
      inst✝³ : E'✝.IsElliptic
      p : Nat
      inst✝² : CharP F p
      hchar2 : Ne 2 0
      hchar3 : Ne 3 0
      this✝³ : NeZero 2
      this✝² : NeZero 4
      this✝¹ : NeZero 6
      this✝ : Invertible 2 := invertibleOfNonzero hchar2
      this : Invertible 3 := invertibleOfNonzero hchar3
      E : WeierstrassCurve F
      inst✝¹ : E.IsElliptic
      h✝¹ : E.IsShortNF
      E' : WeierstrassCurve F
      inst✝ : E'.IsElliptic
      h✝ : E'.IsShortNF
      heq : Eq (HMul.hMul (HPow.hPow E.a₄ 3) (HPow.hPow E'.a₆ 2)) (HMul.hMul (HPow.h …
      ha₄ : Eq E.a₄ 0
      ha₆ : Ne E.a₆ 0
      ha₄' : Eq E'.a₄ 0
      ⊢ Exists fun C => Eq (E.variableChange C) E'
    -/
    have ha₆' := E'.Δ'.ne_zero
    rw [coe_Δ', Δ_of_isShortNF, ha₄', zero_pow three_ne_zero, mul_zero, zero_add, ← mul_assoc,
      mul_ne_zero_iff, pow_ne_zero_iff two_ne_zero] at ha₆'
    /-
      case pos
      F : Type u_1
      inst✝⁶ : Field F
      inst✝⁵ : IsSepClosed F
      E✝ E'✝ : WeierstrassCurve F
      inst✝⁴ : E✝.IsElliptic
      inst✝³ : E'✝.IsElliptic
      p : Nat
      inst✝² : CharP F p
      hchar2 : Ne 2 0
      hchar3 : Ne 3 0
      this✝³ : NeZero 2
      this✝² : NeZero 4
      this✝¹ : NeZero 6
      this✝ : Invertible 2 := invertibleOfNonzero hchar2
      this : Invertible 3 := invertibleOfNonzero hchar3
      E : WeierstrassCurve F
      inst✝¹ : E.IsElliptic
      h✝¹ : E.IsShortNF
      E' : WeierstrassCurve F
      inst✝ : E'.IsElliptic
      h✝ : E'.IsShortNF
      heq : Eq (HMul.hMul (HPow.hPow E.a₄ 3) (HPow.hPow E'.a₆ 2)) (HMul.hMul (HPow.h …
      ha₄ : Eq E.a₄ 0
      ha₆ : Ne E.a₆ 0
      ha₄' : Eq E'.a₄ 0
      ha₆' : And (Ne (HMul.hMul (-16) 27) 0) (Ne E'.a₆ 0)
      ⊢ Exists fun C => Eq (E.variableChange C) E'
    -/
    replace ha₆' := ha₆'.2
    /-
      case pos
      F : Type u_1
      inst✝⁶ : Field F
      inst✝⁵ : IsSepClosed F
      E✝ E'✝ : WeierstrassCurve F
      inst✝⁴ : E✝.IsElliptic
      inst✝³ : E'✝.IsElliptic
      p : Nat
      inst✝² : CharP F p
      hchar2 : Ne 2 0
      hchar3 : Ne 3 0
      this✝³ : NeZero 2
      this✝² : NeZero 4
      this✝¹ : NeZero 6
      this✝ : Invertible 2 := invertibleOfNonzero hchar2
      this : Invertible 3 := invertibleOfNonzero hchar3
      E : WeierstrassCurve F
      inst✝¹ : E.IsElliptic
      h✝¹ : E.IsShortNF
      E' : WeierstrassCurve F
      inst✝ : E'.IsElliptic
      h✝ : E'.IsShortNF
      heq : Eq (HMul.hMul (HPow.hPow E.a₄ 3) (HPow.hPow E'.a₆ 2)) (HMul.hMul (HPow.h …
      ha₄ : Eq E.a₄ 0
      ha₆ : Ne E.a₆ 0
      ha₄' : Eq E'.a₄ 0
      ha₆' : Ne E'.a₆ 0
      ⊢ Exists fun C => Eq (E.variableChange C) E'
    -/
    obtain ⟨u, hu⟩ := IsSepClosed.exists_pow_nat_eq (E.a₆ / E'.a₆) 6
    have hu0 : u ≠ 0 := by
      rw [← pow_ne_zero_iff (Nat.succ_ne_zero 5), hu, div_ne_zero_iff]
      exact ⟨ha₆, ha₆'⟩
    /-
      case pos.intro
      F : Type u_1
      inst✝⁶ : Field F
      inst✝⁵ : IsSepClosed F
      E✝ E'✝ : WeierstrassCurve F
      inst✝⁴ : E✝.IsElliptic
      inst✝³ : E'✝.IsElliptic
      p : Nat
      inst✝² : CharP F p
      hchar2 : Ne 2 0
      hchar3 : Ne 3 0
      this✝³ : NeZero 2
      this✝² : NeZero 4
      this✝¹ : NeZero 6
      this✝ : Invertible 2 := invertibleOfNonzero hchar2
      this : Invertible 3 := invertibleOfNonzero hchar3
      E : WeierstrassCurve F
      inst✝¹ : E.IsElliptic
      h✝¹ : E.IsShortNF
      E' : WeierstrassCurve F
      inst✝ : E'.IsElliptic
      h✝ : E'.IsShortNF
      heq : Eq (HMul.hMul (HPow.hPow E.a₄ 3) (HPow.hPow E'.a₆ 2)) (HMul.hMul (HPow.h …
      ha₄ : Eq E.a₄ 0
      ha₆ : Ne E.a₆ 0
      ha₄' : Eq E'.a₄ 0
      ha₆' : Ne E'.a₆ 0
      u : F
      hu : Eq (HPow.hPow u 6) (HDiv.hDiv E.a₆ E'.a₆)
      hu0 : Ne u 0
      ⊢ Exists fun C => Eq (E.variableChange C) E'
    -/
    use ⟨Units.mk0 u hu0, 0, 0, 0⟩
    /-
      case h
      F : Type u_1
      inst✝⁶ : Field F
      inst✝⁵ : IsSepClosed F
      E✝ E'✝ : WeierstrassCurve F
      inst✝⁴ : E✝.IsElliptic
      inst✝³ : E'✝.IsElliptic
      p : Nat
      inst✝² : CharP F p
      hchar2 : Ne 2 0
      hchar3 : Ne 3 0
      this✝³ : NeZero 2
      this✝² : NeZero 4
      this✝¹ : NeZero 6
      this✝ : Invertible 2 := invertibleOfNonzero hchar2
      this : Invertible 3 := invertibleOfNonzero hchar3
      E : WeierstrassCurve F
      inst✝¹ : E.IsElliptic
      h✝¹ : E.IsShortNF
      E' : WeierstrassCurve F
      inst✝ : E'.IsElliptic
      h✝ : E'.IsShortNF
      heq : Eq (HMul.hMul (HPow.hPow E.a₄ 3) (HPow.hPow E'.a₆ 2)) (HMul.hMul (HPow.h …
      ha₄ : Eq E.a₄ 0
      ha₆ : Ne E.a₆ 0
      ha₄' : Eq E'.a₄ 0
      ha₆' : Ne E'.a₆ 0
      u : F
      hu : Eq (HPow.hPow u 6) (HDiv.hDiv E.a₆ E'.a₆)
      hu0 : Ne u 0
      ⊢ Eq (E.variableChange { u := Units.mk0 u hu0, r := 0, s := 0, t := 0 }) E'
    -/
    ext
      /-
        case h.a₁
        F : Type u_1
        inst✝⁶ : Field F
        inst✝⁵ : IsSepClosed F
        E✝ E'✝ : WeierstrassCurve F
        inst✝⁴ : E✝.IsElliptic
        inst✝³ : E'✝.IsElliptic
        p : Nat
        inst✝² : CharP F p
        hchar2 : Ne 2 0
        hchar3 : Ne 3 0
        this✝³ : NeZero 2
        this✝² : NeZero 4
        this✝¹ : NeZero 6
        this✝ : Invertible 2 := invertibleOfNonzero hchar2
        this : Invertible 3 := invertibleOfNonzero hchar3
        E : WeierstrassCurve F
        inst✝¹ : E.IsElliptic
        h✝¹ : E.IsShortNF
        E' : WeierstrassCurve F
        inst✝ : E'.IsElliptic
        h✝ : E'.IsShortNF
        heq : Eq (HMul.hMul (HPow.hPow E.a₄ 3) (HPow.hPow E'.a₆ 2)) (HMul.hMul (HPow.h …
        ha₄ : Eq E.a₄ 0
        ha₆ : Ne E.a₆ 0
        ha₄' : Eq E'.a₄ 0
        ha₆' : Ne E'.a₆ 0
        u : F
        hu : Eq (HPow.hPow u 6) (HDiv.hDiv E.a₆ E'.a₆)
        hu0 : Ne u 0
        ⊢ Eq (E.variableChange { u := Units.mk0 u hu0, r := 0, s := 0, t := 0 }).a₁ E' …
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case h.a₂
        F : Type u_1
        inst✝⁶ : Field F
        inst✝⁵ : IsSepClosed F
        E✝ E'✝ : WeierstrassCurve F
        inst✝⁴ : E✝.IsElliptic
        inst✝³ : E'✝.IsElliptic
        p : Nat
        inst✝² : CharP F p
        hchar2 : Ne 2 0
        hchar3 : Ne 3 0
        this✝³ : NeZero 2
        this✝² : NeZero 4
        this✝¹ : NeZero 6
        this✝ : Invertible 2 := invertibleOfNonzero hchar2
        this : Invertible 3 := invertibleOfNonzero hchar3
        E : WeierstrassCurve F
        inst✝¹ : E.IsElliptic
        h✝¹ : E.IsShortNF
        E' : WeierstrassCurve F
        inst✝ : E'.IsElliptic
        h✝ : E'.IsShortNF
        heq : Eq (HMul.hMul (HPow.hPow E.a₄ 3) (HPow.hPow E'.a₆ 2)) (HMul.hMul (HPow.h …
        ha₄ : Eq E.a₄ 0
        ha₆ : Ne E.a₆ 0
        ha₄' : Eq E'.a₄ 0
        ha₆' : Ne E'.a₆ 0
        u : F
        hu : Eq (HPow.hPow u 6) (HDiv.hDiv E.a₆ E'.a₆)
        hu0 : Ne u 0
        ⊢ Eq (E.variableChange { u := Units.mk0 u hu0, r := 0, s := 0, t := 0 }).a₂ E' …
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case h.a₃
        F : Type u_1
        inst✝⁶ : Field F
        inst✝⁵ : IsSepClosed F
        E✝ E'✝ : WeierstrassCurve F
        inst✝⁴ : E✝.IsElliptic
        inst✝³ : E'✝.IsElliptic
        p : Nat
        inst✝² : CharP F p
        hchar2 : Ne 2 0
        hchar3 : Ne 3 0
        this✝³ : NeZero 2
        this✝² : NeZero 4
        this✝¹ : NeZero 6
        this✝ : Invertible 2 := invertibleOfNonzero hchar2
        this : Invertible 3 := invertibleOfNonzero hchar3
        E : WeierstrassCurve F
        inst✝¹ : E.IsElliptic
        h✝¹ : E.IsShortNF
        E' : WeierstrassCurve F
        inst✝ : E'.IsElliptic
        h✝ : E'.IsShortNF
        heq : Eq (HMul.hMul (HPow.hPow E.a₄ 3) (HPow.hPow E'.a₆ 2)) (HMul.hMul (HPow.h …
        ha₄ : Eq E.a₄ 0
        ha₆ : Ne E.a₆ 0
        ha₄' : Eq E'.a₄ 0
        ha₆' : Ne E'.a₆ 0
        u : F
        hu : Eq (HPow.hPow u 6) (HDiv.hDiv E.a₆ E'.a₆)
        hu0 : Ne u 0
        ⊢ Eq (E.variableChange { u := Units.mk0 u hu0, r := 0, s := 0, t := 0 }).a₃ E' …
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case h.a₄
        F : Type u_1
        inst✝⁶ : Field F
        inst✝⁵ : IsSepClosed F
        E✝ E'✝ : WeierstrassCurve F
        inst✝⁴ : E✝.IsElliptic
        inst✝³ : E'✝.IsElliptic
        p : Nat
        inst✝² : CharP F p
        hchar2 : Ne 2 0
        hchar3 : Ne 3 0
        this✝³ : NeZero 2
        this✝² : NeZero 4
        this✝¹ : NeZero 6
        this✝ : Invertible 2 := invertibleOfNonzero hchar2
        this : Invertible 3 := invertibleOfNonzero hchar3
        E : WeierstrassCurve F
        inst✝¹ : E.IsElliptic
        h✝¹ : E.IsShortNF
        E' : WeierstrassCurve F
        inst✝ : E'.IsElliptic
        h✝ : E'.IsShortNF
        heq : Eq (HMul.hMul (HPow.hPow E.a₄ 3) (HPow.hPow E'.a₆ 2)) (HMul.hMul (HPow.h …
        ha₄ : Eq E.a₄ 0
        ha₆ : Ne E.a₆ 0
        ha₄' : Eq E'.a₄ 0
        ha₆' : Ne E'.a₆ 0
        u : F
        hu : Eq (HPow.hPow u 6) (HDiv.hDiv E.a₆ E'.a₆)
        hu0 : Ne u 0
        ⊢ Eq (E.variableChange { u := Units.mk0 u hu0, r := 0, s := 0, t := 0 }).a₄ E' …
      -/
    · simp [ha₄, ha₄']
      /-
        🎉 no goals
      -/
    · simp_rw [variableChange_a₆, a₁_of_isShortNF, a₂_of_isShortNF, a₃_of_isShortNF,
        ha₄, Units.val_inv_eq_inv_val, Units.val_mk0, inv_pow, inv_mul_eq_div, hu]
      /-
        case h.a₆
        F : Type u_1
        inst✝⁶ : Field F
        inst✝⁵ : IsSepClosed F
        E✝ E'✝ : WeierstrassCurve F
        inst✝⁴ : E✝.IsElliptic
        inst✝³ : E'✝.IsElliptic
        p : Nat
        inst✝² : CharP F p
        hchar2 : Ne 2 0
        hchar3 : Ne 3 0
        this✝³ : NeZero 2
        this✝² : NeZero 4
        this✝¹ : NeZero 6
        this✝ : Invertible 2 := invertibleOfNonzero hchar2
        this : Invertible 3 := invertibleOfNonzero hchar3
        E : WeierstrassCurve F
        inst✝¹ : E.IsElliptic
        h✝¹ : E.IsShortNF
        E' : WeierstrassCurve F
        inst✝ : E'.IsElliptic
        h✝ : E'.IsShortNF
        heq : Eq (HMul.hMul (HPow.hPow E.a₄ 3) (HPow.hPow E'.a₆ 2)) (HMul.hMul (HPow.h …
        ha₄ : Eq E.a₄ 0
        ha₆ : Ne E.a₆ 0
        ha₄' : Eq E'.a₄ 0
        ha₆' : Ne E'.a₆ 0
        u : F
        hu : Eq (HPow.hPow u 6) (HDiv.hDiv E.a₆ E'.a₆)
        hu0 : Ne u 0
        ⊢ Eq (HDiv.hDiv (HSub.hSub (HSub.hSub (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HAdd.h …
      -/
      field_simp
      /-
        🎉 no goals
      -/
  /-
    case neg
    F : Type u_1
    inst✝⁶ : Field F
    inst✝⁵ : IsSepClosed F
    E✝ E'✝ : WeierstrassCurve F
    inst✝⁴ : E✝.IsElliptic
    inst✝³ : E'✝.IsElliptic
    p : Nat
    inst✝² : CharP F p
    hchar2 : Ne 2 0
    hchar3 : Ne 3 0
    this✝³ : NeZero 2
    this✝² : NeZero 4
    this✝¹ : NeZero 6
    this✝ : Invertible 2 := invertibleOfNonzero hchar2
    this : Invertible 3 := invertibleOfNonzero hchar3
    E : WeierstrassCurve F
    inst✝¹ : E.IsElliptic
    h✝¹ : E.IsShortNF
    E' : WeierstrassCurve F
    inst✝ : E'.IsElliptic
    h✝ : E'.IsShortNF
    heq : Eq (HMul.hMul (HPow.hPow E.a₄ 3) (HPow.hPow E'.a₆ 2)) (HMul.hMul (HPow.h …
    ha₄ : Not (Eq E.a₄ 0)
    ⊢ Exists fun C => Eq (E.variableChange C) E'
  -/
  by_cases ha₆ : E.a₆ = 0
    /-
      case pos
      F : Type u_1
      inst✝⁶ : Field F
      inst✝⁵ : IsSepClosed F
      E✝ E'✝ : WeierstrassCurve F
      inst✝⁴ : E✝.IsElliptic
      inst✝³ : E'✝.IsElliptic
      p : Nat
      inst✝² : CharP F p
      hchar2 : Ne 2 0
      hchar3 : Ne 3 0
      this✝³ : NeZero 2
      this✝² : NeZero 4
      this✝¹ : NeZero 6
      this✝ : Invertible 2 := invertibleOfNonzero hchar2
      this : Invertible 3 := invertibleOfNonzero hchar3
      E : WeierstrassCurve F
      inst✝¹ : E.IsElliptic
      h✝¹ : E.IsShortNF
      E' : WeierstrassCurve F
      inst✝ : E'.IsElliptic
      h✝ : E'.IsShortNF
      heq : Eq (HMul.hMul (HPow.hPow E.a₄ 3) (HPow.hPow E'.a₆ 2)) (HMul.hMul (HPow.h …
      ha₄ : Not (Eq E.a₄ 0)
      ha₆ : Eq E.a₆ 0
      ⊢ Exists fun C => Eq (E.variableChange C) E'
    -/
  · have ha₄ := E.Δ'.ne_zero
    rw [coe_Δ', Δ_of_isShortNF, ha₆, zero_pow two_ne_zero, mul_zero, add_zero, ← mul_assoc,
      mul_ne_zero_iff, pow_ne_zero_iff three_ne_zero] at ha₄
    /-
      case pos
      F : Type u_1
      inst✝⁶ : Field F
      inst✝⁵ : IsSepClosed F
      E✝ E'✝ : WeierstrassCurve F
      inst✝⁴ : E✝.IsElliptic
      inst✝³ : E'✝.IsElliptic
      p : Nat
      inst✝² : CharP F p
      hchar2 : Ne 2 0
      hchar3 : Ne 3 0
      this✝³ : NeZero 2
      this✝² : NeZero 4
      this✝¹ : NeZero 6
      this✝ : Invertible 2 := invertibleOfNonzero hchar2
      this : Invertible 3 := invertibleOfNonzero hchar3
      E : WeierstrassCurve F
      inst✝¹ : E.IsElliptic
      h✝¹ : E.IsShortNF
      E' : WeierstrassCurve F
      inst✝ : E'.IsElliptic
      h✝ : E'.IsShortNF
      heq : Eq (HMul.hMul (HPow.hPow E.a₄ 3) (HPow.hPow E'.a₆ 2)) (HMul.hMul (HPow.h …
      ha₄✝ : Not (Eq E.a₄ 0)
      ha₆ : Eq E.a₆ 0
      ha₄ : And (Ne (HMul.hMul (-16) 4) 0) (Ne E.a₄ 0)
      ⊢ Exists fun C => Eq (E.variableChange C) E'
    -/
    replace ha₄ := ha₄.2
    have ha₆' : E'.a₆ = 0 := by
      rw [ha₆, zero_pow two_ne_zero, mul_zero, mul_eq_zero] at heq
      exact (pow_eq_zero_iff two_ne_zero).1 <| heq.resolve_left <| pow_ne_zero 3 ha₄
    /-
      case pos
      F : Type u_1
      inst✝⁶ : Field F
      inst✝⁵ : IsSepClosed F
      E✝ E'✝ : WeierstrassCurve F
      inst✝⁴ : E✝.IsElliptic
      inst✝³ : E'✝.IsElliptic
      p : Nat
      inst✝² : CharP F p
      hchar2 : Ne 2 0
      hchar3 : Ne 3 0
      this✝³ : NeZero 2
      this✝² : NeZero 4
      this✝¹ : NeZero 6
      this✝ : Invertible 2 := invertibleOfNonzero hchar2
      this : Invertible 3 := invertibleOfNonzero hchar3
      E : WeierstrassCurve F
      inst✝¹ : E.IsElliptic
      h✝¹ : E.IsShortNF
      E' : WeierstrassCurve F
      inst✝ : E'.IsElliptic
      h✝ : E'.IsShortNF
      heq : Eq (HMul.hMul (HPow.hPow E.a₄ 3) (HPow.hPow E'.a₆ 2)) (HMul.hMul (HPow.h …
      ha₄✝ : Not (Eq E.a₄ 0)
      ha₆ : Eq E.a₆ 0
      ha₄ : Ne E.a₄ 0
      ha₆' : Eq E'.a₆ 0
      ⊢ Exists fun C => Eq (E.variableChange C) E'
    -/
    have ha₄' := E'.Δ'.ne_zero
    rw [coe_Δ', Δ_of_isShortNF, ha₆', zero_pow two_ne_zero, mul_zero, add_zero, ← mul_assoc,
      mul_ne_zero_iff, pow_ne_zero_iff three_ne_zero] at ha₄'
    /-
      case pos
      F : Type u_1
      inst✝⁶ : Field F
      inst✝⁵ : IsSepClosed F
      E✝ E'✝ : WeierstrassCurve F
      inst✝⁴ : E✝.IsElliptic
      inst✝³ : E'✝.IsElliptic
      p : Nat
      inst✝² : CharP F p
      hchar2 : Ne 2 0
      hchar3 : Ne 3 0
      this✝³ : NeZero 2
      this✝² : NeZero 4
      this✝¹ : NeZero 6
      this✝ : Invertible 2 := invertibleOfNonzero hchar2
      this : Invertible 3 := invertibleOfNonzero hchar3
      E : WeierstrassCurve F
      inst✝¹ : E.IsElliptic
      h✝¹ : E.IsShortNF
      E' : WeierstrassCurve F
      inst✝ : E'.IsElliptic
      h✝ : E'.IsShortNF
      heq : Eq (HMul.hMul (HPow.hPow E.a₄ 3) (HPow.hPow E'.a₆ 2)) (HMul.hMul (HPow.h …
      ha₄✝ : Not (Eq E.a₄ 0)
      ha₆ : Eq E.a₆ 0
      ha₄ : Ne E.a₄ 0
      ha₆' : Eq E'.a₆ 0
      ha₄' : And (Ne (HMul.hMul (-16) 4) 0) (Ne E'.a₄ 0)
      ⊢ Exists fun C => Eq (E.variableChange C) E'
    -/
    replace ha₄' := ha₄'.2
    /-
      case pos
      F : Type u_1
      inst✝⁶ : Field F
      inst✝⁵ : IsSepClosed F
      E✝ E'✝ : WeierstrassCurve F
      inst✝⁴ : E✝.IsElliptic
      inst✝³ : E'✝.IsElliptic
      p : Nat
      inst✝² : CharP F p
      hchar2 : Ne 2 0
      hchar3 : Ne 3 0
      this✝³ : NeZero 2
      this✝² : NeZero 4
      this✝¹ : NeZero 6
      this✝ : Invertible 2 := invertibleOfNonzero hchar2
      this : Invertible 3 := invertibleOfNonzero hchar3
      E : WeierstrassCurve F
      inst✝¹ : E.IsElliptic
      h✝¹ : E.IsShortNF
      E' : WeierstrassCurve F
      inst✝ : E'.IsElliptic
      h✝ : E'.IsShortNF
      heq : Eq (HMul.hMul (HPow.hPow E.a₄ 3) (HPow.hPow E'.a₆ 2)) (HMul.hMul (HPow.h …
      ha₄✝ : Not (Eq E.a₄ 0)
      ha₆ : Eq E.a₆ 0
      ha₄ : Ne E.a₄ 0
      ha₆' : Eq E'.a₆ 0
      ha₄' : Ne E'.a₄ 0
      ⊢ Exists fun C => Eq (E.variableChange C) E'
    -/
    obtain ⟨u, hu⟩ := IsSepClosed.exists_pow_nat_eq (E.a₄ / E'.a₄) 4
    have hu0 : u ≠ 0 := by
      rw [← pow_ne_zero_iff four_ne_zero, hu, div_ne_zero_iff]
      exact ⟨ha₄, ha₄'⟩
    /-
      case pos.intro
      F : Type u_1
      inst✝⁶ : Field F
      inst✝⁵ : IsSepClosed F
      E✝ E'✝ : WeierstrassCurve F
      inst✝⁴ : E✝.IsElliptic
      inst✝³ : E'✝.IsElliptic
      p : Nat
      inst✝² : CharP F p
      hchar2 : Ne 2 0
      hchar3 : Ne 3 0
      this✝³ : NeZero 2
      this✝² : NeZero 4
      this✝¹ : NeZero 6
      this✝ : Invertible 2 := invertibleOfNonzero hchar2
      this : Invertible 3 := invertibleOfNonzero hchar3
      E : WeierstrassCurve F
      inst✝¹ : E.IsElliptic
      h✝¹ : E.IsShortNF
      E' : WeierstrassCurve F
      inst✝ : E'.IsElliptic
      h✝ : E'.IsShortNF
      heq : Eq (HMul.hMul (HPow.hPow E.a₄ 3) (HPow.hPow E'.a₆ 2)) (HMul.hMul (HPow.h …
      ha₄✝ : Not (Eq E.a₄ 0)
      ha₆ : Eq E.a₆ 0
      ha₄ : Ne E.a₄ 0
      ha₆' : Eq E'.a₆ 0
      ha₄' : Ne E'.a₄ 0
      u : F
      hu : Eq (HPow.hPow u 4) (HDiv.hDiv E.a₄ E'.a₄)
      hu0 : Ne u 0
      ⊢ Exists fun C => Eq (E.variableChange C) E'
    -/
    use ⟨Units.mk0 u hu0, 0, 0, 0⟩
    /-
      case h
      F : Type u_1
      inst✝⁶ : Field F
      inst✝⁵ : IsSepClosed F
      E✝ E'✝ : WeierstrassCurve F
      inst✝⁴ : E✝.IsElliptic
      inst✝³ : E'✝.IsElliptic
      p : Nat
      inst✝² : CharP F p
      hchar2 : Ne 2 0
      hchar3 : Ne 3 0
      this✝³ : NeZero 2
      this✝² : NeZero 4
      this✝¹ : NeZero 6
      this✝ : Invertible 2 := invertibleOfNonzero hchar2
      this : Invertible 3 := invertibleOfNonzero hchar3
      E : WeierstrassCurve F
      inst✝¹ : E.IsElliptic
      h✝¹ : E.IsShortNF
      E' : WeierstrassCurve F
      inst✝ : E'.IsElliptic
      h✝ : E'.IsShortNF
      heq : Eq (HMul.hMul (HPow.hPow E.a₄ 3) (HPow.hPow E'.a₆ 2)) (HMul.hMul (HPow.h …
      ha₄✝ : Not (Eq E.a₄ 0)
      ha₆ : Eq E.a₆ 0
      ha₄ : Ne E.a₄ 0
      ha₆' : Eq E'.a₆ 0
      ha₄' : Ne E'.a₄ 0
      u : F
      hu : Eq (HPow.hPow u 4) (HDiv.hDiv E.a₄ E'.a₄)
      hu0 : Ne u 0
      ⊢ Eq (E.variableChange { u := Units.mk0 u hu0, r := 0, s := 0, t := 0 }) E'
    -/
    ext
      /-
        case h.a₁
        F : Type u_1
        inst✝⁶ : Field F
        inst✝⁵ : IsSepClosed F
        E✝ E'✝ : WeierstrassCurve F
        inst✝⁴ : E✝.IsElliptic
        inst✝³ : E'✝.IsElliptic
        p : Nat
        inst✝² : CharP F p
        hchar2 : Ne 2 0
        hchar3 : Ne 3 0
        this✝³ : NeZero 2
        this✝² : NeZero 4
        this✝¹ : NeZero 6
        this✝ : Invertible 2 := invertibleOfNonzero hchar2
        this : Invertible 3 := invertibleOfNonzero hchar3
        E : WeierstrassCurve F
        inst✝¹ : E.IsElliptic
        h✝¹ : E.IsShortNF
        E' : WeierstrassCurve F
        inst✝ : E'.IsElliptic
        h✝ : E'.IsShortNF
        heq : Eq (HMul.hMul (HPow.hPow E.a₄ 3) (HPow.hPow E'.a₆ 2)) (HMul.hMul (HPow.h …
        ha₄✝ : Not (Eq E.a₄ 0)
        ha₆ : Eq E.a₆ 0
        ha₄ : Ne E.a₄ 0
        ha₆' : Eq E'.a₆ 0
        ha₄' : Ne E'.a₄ 0
        u : F
        hu : Eq (HPow.hPow u 4) (HDiv.hDiv E.a₄ E'.a₄)
        hu0 : Ne u 0
        ⊢ Eq (E.variableChange { u := Units.mk0 u hu0, r := 0, s := 0, t := 0 }).a₁ E' …
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case h.a₂
        F : Type u_1
        inst✝⁶ : Field F
        inst✝⁵ : IsSepClosed F
        E✝ E'✝ : WeierstrassCurve F
        inst✝⁴ : E✝.IsElliptic
        inst✝³ : E'✝.IsElliptic
        p : Nat
        inst✝² : CharP F p
        hchar2 : Ne 2 0
        hchar3 : Ne 3 0
        this✝³ : NeZero 2
        this✝² : NeZero 4
        this✝¹ : NeZero 6
        this✝ : Invertible 2 := invertibleOfNonzero hchar2
        this : Invertible 3 := invertibleOfNonzero hchar3
        E : WeierstrassCurve F
        inst✝¹ : E.IsElliptic
        h✝¹ : E.IsShortNF
        E' : WeierstrassCurve F
        inst✝ : E'.IsElliptic
        h✝ : E'.IsShortNF
        heq : Eq (HMul.hMul (HPow.hPow E.a₄ 3) (HPow.hPow E'.a₆ 2)) (HMul.hMul (HPow.h …
        ha₄✝ : Not (Eq E.a₄ 0)
        ha₆ : Eq E.a₆ 0
        ha₄ : Ne E.a₄ 0
        ha₆' : Eq E'.a₆ 0
        ha₄' : Ne E'.a₄ 0
        u : F
        hu : Eq (HPow.hPow u 4) (HDiv.hDiv E.a₄ E'.a₄)
        hu0 : Ne u 0
        ⊢ Eq (E.variableChange { u := Units.mk0 u hu0, r := 0, s := 0, t := 0 }).a₂ E' …
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case h.a₃
        F : Type u_1
        inst✝⁶ : Field F
        inst✝⁵ : IsSepClosed F
        E✝ E'✝ : WeierstrassCurve F
        inst✝⁴ : E✝.IsElliptic
        inst✝³ : E'✝.IsElliptic
        p : Nat
        inst✝² : CharP F p
        hchar2 : Ne 2 0
        hchar3 : Ne 3 0
        this✝³ : NeZero 2
        this✝² : NeZero 4
        this✝¹ : NeZero 6
        this✝ : Invertible 2 := invertibleOfNonzero hchar2
        this : Invertible 3 := invertibleOfNonzero hchar3
        E : WeierstrassCurve F
        inst✝¹ : E.IsElliptic
        h✝¹ : E.IsShortNF
        E' : WeierstrassCurve F
        inst✝ : E'.IsElliptic
        h✝ : E'.IsShortNF
        heq : Eq (HMul.hMul (HPow.hPow E.a₄ 3) (HPow.hPow E'.a₆ 2)) (HMul.hMul (HPow.h …
        ha₄✝ : Not (Eq E.a₄ 0)
        ha₆ : Eq E.a₆ 0
        ha₄ : Ne E.a₄ 0
        ha₆' : Eq E'.a₆ 0
        ha₄' : Ne E'.a₄ 0
        u : F
        hu : Eq (HPow.hPow u 4) (HDiv.hDiv E.a₄ E'.a₄)
        hu0 : Ne u 0
        ⊢ Eq (E.variableChange { u := Units.mk0 u hu0, r := 0, s := 0, t := 0 }).a₃ E' …
      -/
    · simp
      /-
        🎉 no goals
      -/
    · simp_rw [variableChange_a₄, a₁_of_isShortNF, a₂_of_isShortNF, a₃_of_isShortNF,
        Units.val_inv_eq_inv_val, Units.val_mk0, inv_pow, inv_mul_eq_div, hu]
      /-
        case h.a₄
        F : Type u_1
        inst✝⁶ : Field F
        inst✝⁵ : IsSepClosed F
        E✝ E'✝ : WeierstrassCurve F
        inst✝⁴ : E✝.IsElliptic
        inst✝³ : E'✝.IsElliptic
        p : Nat
        inst✝² : CharP F p
        hchar2 : Ne 2 0
        hchar3 : Ne 3 0
        this✝³ : NeZero 2
        this✝² : NeZero 4
        this✝¹ : NeZero 6
        this✝ : Invertible 2 := invertibleOfNonzero hchar2
        this : Invertible 3 := invertibleOfNonzero hchar3
        E : WeierstrassCurve F
        inst✝¹ : E.IsElliptic
        h✝¹ : E.IsShortNF
        E' : WeierstrassCurve F
        inst✝ : E'.IsElliptic
        h✝ : E'.IsShortNF
        heq : Eq (HMul.hMul (HPow.hPow E.a₄ 3) (HPow.hPow E'.a₆ 2)) (HMul.hMul (HPow.h …
        ha₄✝ : Not (Eq E.a₄ 0)
        ha₆ : Eq E.a₆ 0
        ha₄ : Ne E.a₄ 0
        ha₆' : Eq E'.a₆ 0
        ha₄' : Ne E'.a₄ 0
        u : F
        hu : Eq (HPow.hPow u 4) (HDiv.hDiv E.a₄ E'.a₄)
        hu0 : Ne u 0
        ⊢ Eq (HDiv.hDiv (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (HSub.hSub E.a₄ (H …
      -/
      field_simp
      /-
        🎉 no goals
      -/
      /-
        case h.a₆
        F : Type u_1
        inst✝⁶ : Field F
        inst✝⁵ : IsSepClosed F
        E✝ E'✝ : WeierstrassCurve F
        inst✝⁴ : E✝.IsElliptic
        inst✝³ : E'✝.IsElliptic
        p : Nat
        inst✝² : CharP F p
        hchar2 : Ne 2 0
        hchar3 : Ne 3 0
        this✝³ : NeZero 2
        this✝² : NeZero 4
        this✝¹ : NeZero 6
        this✝ : Invertible 2 := invertibleOfNonzero hchar2
        this : Invertible 3 := invertibleOfNonzero hchar3
        E : WeierstrassCurve F
        inst✝¹ : E.IsElliptic
        h✝¹ : E.IsShortNF
        E' : WeierstrassCurve F
        inst✝ : E'.IsElliptic
        h✝ : E'.IsShortNF
        heq : Eq (HMul.hMul (HPow.hPow E.a₄ 3) (HPow.hPow E'.a₆ 2)) (HMul.hMul (HPow.h …
        ha₄✝ : Not (Eq E.a₄ 0)
        ha₆ : Eq E.a₆ 0
        ha₄ : Ne E.a₄ 0
        ha₆' : Eq E'.a₆ 0
        ha₄' : Ne E'.a₄ 0
        u : F
        hu : Eq (HPow.hPow u 4) (HDiv.hDiv E.a₄ E'.a₄)
        hu0 : Ne u 0
        ⊢ Eq (E.variableChange { u := Units.mk0 u hu0, r := 0, s := 0, t := 0 }).a₆ E' …
      -/
    · simp [ha₆, ha₆']
      /-
        🎉 no goals
      -/
  have ha₄' : E'.a₄ ≠ 0 := fun h ↦ by
    rw [h, zero_pow three_ne_zero, zero_mul, mul_eq_zero,
      pow_eq_zero_iff two_ne_zero, pow_eq_zero_iff three_ne_zero] at heq
    simpa [E'.coe_Δ', Δ_of_isShortNF, h, heq.resolve_left ha₄] using E'.Δ'.ne_zero
  have ha₆' : E'.a₆ ≠ 0 := fun h ↦ by
    rw [h, zero_pow two_ne_zero, mul_zero, zero_eq_mul,
      pow_eq_zero_iff two_ne_zero, pow_eq_zero_iff three_ne_zero] at heq
    tauto
  /-
    case neg
    F : Type u_1
    inst✝⁶ : Field F
    inst✝⁵ : IsSepClosed F
    E✝ E'✝ : WeierstrassCurve F
    inst✝⁴ : E✝.IsElliptic
    inst✝³ : E'✝.IsElliptic
    p : Nat
    inst✝² : CharP F p
    hchar2 : Ne 2 0
    hchar3 : Ne 3 0
    this✝³ : NeZero 2
    this✝² : NeZero 4
    this✝¹ : NeZero 6
    this✝ : Invertible 2 := invertibleOfNonzero hchar2
    this : Invertible 3 := invertibleOfNonzero hchar3
    E : WeierstrassCurve F
    inst✝¹ : E.IsElliptic
    h✝¹ : E.IsShortNF
    E' : WeierstrassCurve F
    inst✝ : E'.IsElliptic
    h✝ : E'.IsShortNF
    heq : Eq (HMul.hMul (HPow.hPow E.a₄ 3) (HPow.hPow E'.a₆ 2)) (HMul.hMul (HPow.h …
    ha₄ : Not (Eq E.a₄ 0)
    ha₆ : Not (Eq E.a₆ 0)
    ha₄' : Ne E'.a₄ 0
    ha₆' : Ne E'.a₆ 0
    ⊢ Exists fun C => Eq (E.variableChange C) E'
  -/
  obtain ⟨u, hu⟩ := IsSepClosed.exists_pow_nat_eq (E.a₆ / E'.a₆ / (E.a₄ / E'.a₄)) 2
  have hu4 : u ^ 4 = E.a₄ / E'.a₄ := by
    rw [pow_mul u 2 2, hu]
    field_simp
    linear_combination -heq
  have hu6 : u ^ 6 = E.a₆ / E'.a₆ := by
    rw [pow_mul u 2 3, hu]
    field_simp
    linear_combination -E.a₆ * E'.a₆ * heq
  have hu0 : u ≠ 0 := by
    rw [← pow_ne_zero_iff four_ne_zero, hu4, div_ne_zero_iff]
    exact ⟨ha₄, ha₄'⟩
  /-
    case neg.intro
    F : Type u_1
    inst✝⁶ : Field F
    inst✝⁵ : IsSepClosed F
    E✝ E'✝ : WeierstrassCurve F
    inst✝⁴ : E✝.IsElliptic
    inst✝³ : E'✝.IsElliptic
    p : Nat
    inst✝² : CharP F p
    hchar2 : Ne 2 0
    hchar3 : Ne 3 0
    this✝³ : NeZero 2
    this✝² : NeZero 4
    this✝¹ : NeZero 6
    this✝ : Invertible 2 := invertibleOfNonzero hchar2
    this : Invertible 3 := invertibleOfNonzero hchar3
    E : WeierstrassCurve F
    inst✝¹ : E.IsElliptic
    h✝¹ : E.IsShortNF
    E' : WeierstrassCurve F
    inst✝ : E'.IsElliptic
    h✝ : E'.IsShortNF
    heq : Eq (HMul.hMul (HPow.hPow E.a₄ 3) (HPow.hPow E'.a₆ 2)) (HMul.hMul (HPow.h …
    ha₄ : Not (Eq E.a₄ 0)
    ha₆ : Not (Eq E.a₆ 0)
    ha₄' : Ne E'.a₄ 0
    ha₆' : Ne E'.a₆ 0
    u : F
    hu : Eq (HPow.hPow u 2) (HDiv.hDiv (HDiv.hDiv E.a₆ E'.a₆) (HDiv.hDiv E.a₄ E'.a …
    hu4 : Eq (HPow.hPow u 4) (HDiv.hDiv E.a₄ E'.a₄)
    hu6 : Eq (HPow.hPow u 6) (HDiv.hDiv E.a₆ E'.a₆)
    hu0 : Ne u 0
    ⊢ Exists fun C => Eq (E.variableChange C) E'
  -/
  use ⟨Units.mk0 u hu0, 0, 0, 0⟩
  /-
    case h
    F : Type u_1
    inst✝⁶ : Field F
    inst✝⁵ : IsSepClosed F
    E✝ E'✝ : WeierstrassCurve F
    inst✝⁴ : E✝.IsElliptic
    inst✝³ : E'✝.IsElliptic
    p : Nat
    inst✝² : CharP F p
    hchar2 : Ne 2 0
    hchar3 : Ne 3 0
    this✝³ : NeZero 2
    this✝² : NeZero 4
    this✝¹ : NeZero 6
    this✝ : Invertible 2 := invertibleOfNonzero hchar2
    this : Invertible 3 := invertibleOfNonzero hchar3
    E : WeierstrassCurve F
    inst✝¹ : E.IsElliptic
    h✝¹ : E.IsShortNF
    E' : WeierstrassCurve F
    inst✝ : E'.IsElliptic
    h✝ : E'.IsShortNF
    heq : Eq (HMul.hMul (HPow.hPow E.a₄ 3) (HPow.hPow E'.a₆ 2)) (HMul.hMul (HPow.h …
    ha₄ : Not (Eq E.a₄ 0)
    ha₆ : Not (Eq E.a₆ 0)
    ha₄' : Ne E'.a₄ 0
    ha₆' : Ne E'.a₆ 0
    u : F
    hu : Eq (HPow.hPow u 2) (HDiv.hDiv (HDiv.hDiv E.a₆ E'.a₆) (HDiv.hDiv E.a₄ E'.a …
    hu4 : Eq (HPow.hPow u 4) (HDiv.hDiv E.a₄ E'.a₄)
    hu6 : Eq (HPow.hPow u 6) (HDiv.hDiv E.a₆ E'.a₆)
    hu0 : Ne u 0
    ⊢ Eq (E.variableChange { u := Units.mk0 u hu0, r := 0, s := 0, t := 0 }) E'
  -/
  ext
    /-
      case h.a₁
      F : Type u_1
      inst✝⁶ : Field F
      inst✝⁵ : IsSepClosed F
      E✝ E'✝ : WeierstrassCurve F
      inst✝⁴ : E✝.IsElliptic
      inst✝³ : E'✝.IsElliptic
      p : Nat
      inst✝² : CharP F p
      hchar2 : Ne 2 0
      hchar3 : Ne 3 0
      this✝³ : NeZero 2
      this✝² : NeZero 4
      this✝¹ : NeZero 6
      this✝ : Invertible 2 := invertibleOfNonzero hchar2
      this : Invertible 3 := invertibleOfNonzero hchar3
      E : WeierstrassCurve F
      inst✝¹ : E.IsElliptic
      h✝¹ : E.IsShortNF
      E' : WeierstrassCurve F
      inst✝ : E'.IsElliptic
      h✝ : E'.IsShortNF
      heq : Eq (HMul.hMul (HPow.hPow E.a₄ 3) (HPow.hPow E'.a₆ 2)) (HMul.hMul (HPow.h …
      ha₄ : Not (Eq E.a₄ 0)
      ha₆ : Not (Eq E.a₆ 0)
      ha₄' : Ne E'.a₄ 0
      ha₆' : Ne E'.a₆ 0
      u : F
      hu : Eq (HPow.hPow u 2) (HDiv.hDiv (HDiv.hDiv E.a₆ E'.a₆) (HDiv.hDiv E.a₄ E'.a …
      hu4 : Eq (HPow.hPow u 4) (HDiv.hDiv E.a₄ E'.a₄)
      hu6 : Eq (HPow.hPow u 6) (HDiv.hDiv E.a₆ E'.a₆)
      hu0 : Ne u 0
      ⊢ Eq (E.variableChange { u := Units.mk0 u hu0, r := 0, s := 0, t := 0 }).a₁ E' …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case h.a₂
      F : Type u_1
      inst✝⁶ : Field F
      inst✝⁵ : IsSepClosed F
      E✝ E'✝ : WeierstrassCurve F
      inst✝⁴ : E✝.IsElliptic
      inst✝³ : E'✝.IsElliptic
      p : Nat
      inst✝² : CharP F p
      hchar2 : Ne 2 0
      hchar3 : Ne 3 0
      this✝³ : NeZero 2
      this✝² : NeZero 4
      this✝¹ : NeZero 6
      this✝ : Invertible 2 := invertibleOfNonzero hchar2
      this : Invertible 3 := invertibleOfNonzero hchar3
      E : WeierstrassCurve F
      inst✝¹ : E.IsElliptic
      h✝¹ : E.IsShortNF
      E' : WeierstrassCurve F
      inst✝ : E'.IsElliptic
      h✝ : E'.IsShortNF
      heq : Eq (HMul.hMul (HPow.hPow E.a₄ 3) (HPow.hPow E'.a₆ 2)) (HMul.hMul (HPow.h …
      ha₄ : Not (Eq E.a₄ 0)
      ha₆ : Not (Eq E.a₆ 0)
      ha₄' : Ne E'.a₄ 0
      ha₆' : Ne E'.a₆ 0
      u : F
      hu : Eq (HPow.hPow u 2) (HDiv.hDiv (HDiv.hDiv E.a₆ E'.a₆) (HDiv.hDiv E.a₄ E'.a …
      hu4 : Eq (HPow.hPow u 4) (HDiv.hDiv E.a₄ E'.a₄)
      hu6 : Eq (HPow.hPow u 6) (HDiv.hDiv E.a₆ E'.a₆)
      hu0 : Ne u 0
      ⊢ Eq (E.variableChange { u := Units.mk0 u hu0, r := 0, s := 0, t := 0 }).a₂ E' …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case h.a₃
      F : Type u_1
      inst✝⁶ : Field F
      inst✝⁵ : IsSepClosed F
      E✝ E'✝ : WeierstrassCurve F
      inst✝⁴ : E✝.IsElliptic
      inst✝³ : E'✝.IsElliptic
      p : Nat
      inst✝² : CharP F p
      hchar2 : Ne 2 0
      hchar3 : Ne 3 0
      this✝³ : NeZero 2
      this✝² : NeZero 4
      this✝¹ : NeZero 6
      this✝ : Invertible 2 := invertibleOfNonzero hchar2
      this : Invertible 3 := invertibleOfNonzero hchar3
      E : WeierstrassCurve F
      inst✝¹ : E.IsElliptic
      h✝¹ : E.IsShortNF
      E' : WeierstrassCurve F
      inst✝ : E'.IsElliptic
      h✝ : E'.IsShortNF
      heq : Eq (HMul.hMul (HPow.hPow E.a₄ 3) (HPow.hPow E'.a₆ 2)) (HMul.hMul (HPow.h …
      ha₄ : Not (Eq E.a₄ 0)
      ha₆ : Not (Eq E.a₆ 0)
      ha₄' : Ne E'.a₄ 0
      ha₆' : Ne E'.a₆ 0
      u : F
      hu : Eq (HPow.hPow u 2) (HDiv.hDiv (HDiv.hDiv E.a₆ E'.a₆) (HDiv.hDiv E.a₄ E'.a …
      hu4 : Eq (HPow.hPow u 4) (HDiv.hDiv E.a₄ E'.a₄)
      hu6 : Eq (HPow.hPow u 6) (HDiv.hDiv E.a₆ E'.a₆)
      hu0 : Ne u 0
      ⊢ Eq (E.variableChange { u := Units.mk0 u hu0, r := 0, s := 0, t := 0 }).a₃ E' …
    -/
  · simp
    /-
      🎉 no goals
    -/
  · simp_rw [variableChange_a₄, a₁_of_isShortNF, a₂_of_isShortNF, a₃_of_isShortNF,
      Units.val_inv_eq_inv_val, Units.val_mk0, inv_pow, inv_mul_eq_div, hu4]
    /-
      case h.a₄
      F : Type u_1
      inst✝⁶ : Field F
      inst✝⁵ : IsSepClosed F
      E✝ E'✝ : WeierstrassCurve F
      inst✝⁴ : E✝.IsElliptic
      inst✝³ : E'✝.IsElliptic
      p : Nat
      inst✝² : CharP F p
      hchar2 : Ne 2 0
      hchar3 : Ne 3 0
      this✝³ : NeZero 2
      this✝² : NeZero 4
      this✝¹ : NeZero 6
      this✝ : Invertible 2 := invertibleOfNonzero hchar2
      this : Invertible 3 := invertibleOfNonzero hchar3
      E : WeierstrassCurve F
      inst✝¹ : E.IsElliptic
      h✝¹ : E.IsShortNF
      E' : WeierstrassCurve F
      inst✝ : E'.IsElliptic
      h✝ : E'.IsShortNF
      heq : Eq (HMul.hMul (HPow.hPow E.a₄ 3) (HPow.hPow E'.a₆ 2)) (HMul.hMul (HPow.h …
      ha₄ : Not (Eq E.a₄ 0)
      ha₆ : Not (Eq E.a₆ 0)
      ha₄' : Ne E'.a₄ 0
      ha₆' : Ne E'.a₆ 0
      u : F
      hu : Eq (HPow.hPow u 2) (HDiv.hDiv (HDiv.hDiv E.a₆ E'.a₆) (HDiv.hDiv E.a₄ E'.a …
      hu4 : Eq (HPow.hPow u 4) (HDiv.hDiv E.a₄ E'.a₄)
      hu6 : Eq (HPow.hPow u 6) (HDiv.hDiv E.a₆ E'.a₆)
      hu0 : Ne u 0
      ⊢ Eq (HDiv.hDiv (HSub.hSub (HAdd.hAdd (HSub.hSub (HAdd.hAdd (HSub.hSub E.a₄ (H …
    -/
    field_simp
    /-
      🎉 no goals
    -/
  · simp_rw [variableChange_a₆, a₁_of_isShortNF, a₂_of_isShortNF, a₃_of_isShortNF,
      Units.val_inv_eq_inv_val, Units.val_mk0, inv_pow, inv_mul_eq_div, hu6]
    /-
      case h.a₆
      F : Type u_1
      inst✝⁶ : Field F
      inst✝⁵ : IsSepClosed F
      E✝ E'✝ : WeierstrassCurve F
      inst✝⁴ : E✝.IsElliptic
      inst✝³ : E'✝.IsElliptic
      p : Nat
      inst✝² : CharP F p
      hchar2 : Ne 2 0
      hchar3 : Ne 3 0
      this✝³ : NeZero 2
      this✝² : NeZero 4
      this✝¹ : NeZero 6
      this✝ : Invertible 2 := invertibleOfNonzero hchar2
      this : Invertible 3 := invertibleOfNonzero hchar3
      E : WeierstrassCurve F
      inst✝¹ : E.IsElliptic
      h✝¹ : E.IsShortNF
      E' : WeierstrassCurve F
      inst✝ : E'.IsElliptic
      h✝ : E'.IsShortNF
      heq : Eq (HMul.hMul (HPow.hPow E.a₄ 3) (HPow.hPow E'.a₆ 2)) (HMul.hMul (HPow.h …
      ha₄ : Not (Eq E.a₄ 0)
      ha₆ : Not (Eq E.a₆ 0)
      ha₄' : Ne E'.a₄ 0
      ha₆' : Ne E'.a₆ 0
      u : F
      hu : Eq (HPow.hPow u 2) (HDiv.hDiv (HDiv.hDiv E.a₆ E'.a₆) (HDiv.hDiv E.a₄ E'.a …
      hu4 : Eq (HPow.hPow u 4) (HDiv.hDiv E.a₄ E'.a₄)
      hu6 : Eq (HPow.hPow u 6) (HDiv.hDiv E.a₆ E'.a₆)
      hu0 : Ne u 0
      ⊢ Eq (HDiv.hDiv (HSub.hSub (HSub.hSub (HSub.hSub (HAdd.hAdd (HAdd.hAdd (HAdd.h …
    -/
    field_simp
    /-
      🎉 no goals
    -/


/-- If there are two elliptic curves with the same `j`-invariants defined over a
separably closed field, then there exists a change of variables over that field which change
one curve into another. -/
theorem exists_variableChange_of_j_eq (heq : E.j = E'.j) :
    ∃ C : VariableChange F, E.variableChange C = E' := by
  /-
    F : Type u_1
    inst✝³ : Field F
    inst✝² : IsSepClosed F
    E E' : WeierstrassCurve F
    inst✝¹ : E.IsElliptic
    inst✝ : E'.IsElliptic
    heq : Eq E.j E'.j
    ⊢ Exists fun C => Eq (E.variableChange C) E'
  -/
  obtain ⟨p, _⟩ := CharP.exists F
  /-
    case intro
    F : Type u_1
    inst✝³ : Field F
    inst✝² : IsSepClosed F
    E E' : WeierstrassCurve F
    inst✝¹ : E.IsElliptic
    inst✝ : E'.IsElliptic
    heq : Eq E.j E'.j
    p : Nat
    h✝ : CharP F p
    ⊢ Exists fun C => Eq (E.variableChange C) E'
  -/
  by_cases hchar2 : p = 2
    /-
      case pos
      F : Type u_1
      inst✝³ : Field F
      inst✝² : IsSepClosed F
      E E' : WeierstrassCurve F
      inst✝¹ : E.IsElliptic
      inst✝ : E'.IsElliptic
      heq : Eq E.j E'.j
      p : Nat
      h✝ : CharP F p
      hchar2 : Eq p 2
      ⊢ Exists fun C => Eq (E.variableChange C) E'
    -/
  · subst hchar2
    /-
      case pos
      F : Type u_1
      inst✝³ : Field F
      inst✝² : IsSepClosed F
      E E' : WeierstrassCurve F
      inst✝¹ : E.IsElliptic
      inst✝ : E'.IsElliptic
      heq : Eq E.j E'.j
      h✝ : CharP F 2
      ⊢ Exists fun C => Eq (E.variableChange C) E'
    -/
    exact exists_variableChange_of_char_two _ _ heq
    /-
      🎉 no goals
    -/
  /-
    case neg
    F : Type u_1
    inst✝³ : Field F
    inst✝² : IsSepClosed F
    E E' : WeierstrassCurve F
    inst✝¹ : E.IsElliptic
    inst✝ : E'.IsElliptic
    heq : Eq E.j E'.j
    p : Nat
    h✝ : CharP F p
    hchar2 : Not (Eq p 2)
    ⊢ Exists fun C => Eq (E.variableChange C) E'
  -/
  by_cases hchar3 : p = 3
    /-
      case pos
      F : Type u_1
      inst✝³ : Field F
      inst✝² : IsSepClosed F
      E E' : WeierstrassCurve F
      inst✝¹ : E.IsElliptic
      inst✝ : E'.IsElliptic
      heq : Eq E.j E'.j
      p : Nat
      h✝ : CharP F p
      hchar2 : Not (Eq p 2)
      hchar3 : Eq p 3
      ⊢ Exists fun C => Eq (E.variableChange C) E'
    -/
  · subst hchar3
    /-
      case pos
      F : Type u_1
      inst✝³ : Field F
      inst✝² : IsSepClosed F
      E E' : WeierstrassCurve F
      inst✝¹ : E.IsElliptic
      inst✝ : E'.IsElliptic
      heq : Eq E.j E'.j
      h✝ : CharP F 3
      hchar2 : Not (Eq 3 2)
      ⊢ Exists fun C => Eq (E.variableChange C) E'
    -/
    exact exists_variableChange_of_char_three _ _ heq
    /-
      🎉 no goals
    -/
  /-
    case neg
    F : Type u_1
    inst✝³ : Field F
    inst✝² : IsSepClosed F
    E E' : WeierstrassCurve F
    inst✝¹ : E.IsElliptic
    inst✝ : E'.IsElliptic
    heq : Eq E.j E'.j
    p : Nat
    h✝ : CharP F p
    hchar2 : Not (Eq p 2)
    hchar3 : Not (Eq p 3)
    ⊢ Exists fun C => Eq (E.variableChange C) E'
  -/
  exact exists_variableChange_of_char_ne_two_or_three _ _ hchar2 hchar3 heq
  /-
    🎉 no goals
  -/


