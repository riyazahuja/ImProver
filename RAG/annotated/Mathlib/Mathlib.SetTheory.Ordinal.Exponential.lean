/-- The ordinal exponential, defined by transfinite recursion. -/
instance pow : Pow Ordinal Ordinal :=
  ⟨fun a b => if a = 0 then 1 - b else limitRecOn b 1 (fun _ IH => IH * a) fun b _ => bsup.{u, u} b⟩


theorem opow_def (a b : Ordinal) :
    a ^ b = if a = 0 then 1 - b else limitRecOn b 1 (fun _ IH => IH * a) fun b _ => bsup.{u, u} b :=
  rfl

-- Porting note: `if_pos rfl` → `if_true`

                                                       /-
                                                         a : Ordinal.{u_1}
                                                         ⊢ Eq (HPow.hPow 0 a) (HSub.hSub 1 a)
                                                       -/
theorem zero_opow' (a : Ordinal) : 0 ^ a = 1 - a := by simp only [opow_def, if_true]
                                                       /-
                                                         🎉 no goals
                                                       -/


@[simp]
theorem zero_opow {a : Ordinal} (a0 : a ≠ 0) : (0 : Ordinal) ^ a = 0 := by
  /-
    a : Ordinal.{u_1}
    a0 : Ne a 0
    ⊢ Eq (HPow.hPow 0 a) 0
  -/
  rwa [zero_opow', Ordinal.sub_eq_zero_iff_le, one_le_iff_ne_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem opow_zero (a : Ordinal) : a ^ (0 : Ordinal) = 1 := by
  /-
    a : Ordinal.{u_1}
    ⊢ Eq (HPow.hPow a 0) 1
  -/
  by_cases h : a = 0
    /-
      case pos
      a : Ordinal.{u_1}
      h : Eq a 0
      ⊢ Eq (HPow.hPow a 0) 1
    -/
  · simp only [opow_def, if_pos h, sub_zero]
    /-
      🎉 no goals
    -/
    /-
      case neg
      a : Ordinal.{u_1}
      h : Not (Eq a 0)
      ⊢ Eq (HPow.hPow a 0) 1
    -/
  · simp only [opow_def, if_neg h, limitRecOn_zero]
    /-
      🎉 no goals
    -/


@[simp]
theorem opow_succ (a b : Ordinal) : a ^ succ b = a ^ b * a :=
                       /-
                         a b : Ordinal.{u_1}
                         h : Eq a 0
                         ⊢ Eq (HPow.hPow a (Order.succ b)) (HMul.hMul (HPow.hPow a b) a)
                       -/
  if h : a = 0 then by subst a; simp only [zero_opow (succ_ne_zero _), mul_zero]
                                /-
                                  🎉 no goals
                                -/
          /-
            a b : Ordinal.{u_1}
            h : Not (Eq a 0)
            ⊢ Eq (HPow.hPow a (Order.succ b)) (HMul.hMul (HPow.hPow a b) a)
          -/
  else by simp only [opow_def, limitRecOn_succ, if_neg h]
          /-
            🎉 no goals
          -/


theorem opow_limit {a b : Ordinal} (a0 : a ≠ 0) (h : IsLimit b) :
    a ^ b = bsup.{u, u} b fun c _ => a ^ c := by
  /-
    a b : Ordinal.{u}
    a0 : Ne a 0
    h : b.IsLimit
    ⊢ Eq (HPow.hPow a b) (b.bsup fun c x => HPow.hPow a c)
  -/
  simp only [opow_def, if_neg a0]; rw [limitRecOn_limit _ _ _ _ h]
                                   /-
                                     🎉 no goals
                                   -/


theorem opow_le_of_limit {a b c : Ordinal} (a0 : a ≠ 0) (h : IsLimit b) :
                                           /-
                                             a b c : Ordinal.{u_1}
                                             a0 : Ne a 0
                                             h : b.IsLimit
                                             ⊢ Iff (LE.le (HPow.hPow a b) c) (∀ (b' : Ordinal.{u_1}), LT.lt b' b → LE.le (H …
                                           -/
    a ^ b ≤ c ↔ ∀ b' < b, a ^ b' ≤ c := by rw [opow_limit a0 h, bsup_le_iff]
                                           /-
                                             🎉 no goals
                                           -/


theorem lt_opow_of_limit {a b c : Ordinal} (b0 : b ≠ 0) (h : IsLimit c) :
    a < b ^ c ↔ ∃ c' < c, a < b ^ c' := by
  /-
    a b c : Ordinal.{u_1}
    b0 : Ne b 0
    h : c.IsLimit
    ⊢ Iff (LT.lt a (HPow.hPow b c)) (Exists fun c' => And (LT.lt c' c) (LT.lt a (H …
  -/
  rw [← not_iff_not, not_exists]; simp only [not_lt, opow_le_of_limit b0 h, exists_prop, not_and]
                                  /-
                                    🎉 no goals
                                  -/


@[simp]
theorem opow_one (a : Ordinal) : a ^ (1 : Ordinal) = a := by
  /-
    a : Ordinal.{u_1}
    ⊢ Eq (HPow.hPow a 1) a
  -/
  rw [← succ_zero, opow_succ]; simp only [opow_zero, one_mul]
                               /-
                                 🎉 no goals
                               -/


@[simp]
theorem one_opow (a : Ordinal) : (1 : Ordinal) ^ a = 1 := by
  induction a using limitRecOn with
  | H₁ => simp only [opow_zero]
  | H₂ _ ih =>
    simp only [opow_succ, ih, mul_one]
  | H₃ b l IH =>
    refine eq_of_forall_ge_iff fun c => ?_
    rw [opow_le_of_limit Ordinal.one_ne_zero l]
    exact ⟨fun H => by simpa only [opow_zero] using H 0 l.pos, fun H b' h => by rwa [IH _ h]⟩


theorem opow_pos {a : Ordinal} (b : Ordinal) (a0 : 0 < a) : 0 < a ^ b := by
  /-
    a b : Ordinal.{u_1}
    a0 : LT.lt 0 a
    ⊢ LT.lt 0 (HPow.hPow a b)
  -/
  have h0 : 0 < a ^ (0 : Ordinal) := by simp only [opow_zero, zero_lt_one]
  induction b using limitRecOn with
  | H₁ => exact h0
  | H₂ b IH =>
    rw [opow_succ]
    exact mul_pos IH a0
  | H₃ b l _ =>
    exact (lt_opow_of_limit (Ordinal.pos_iff_ne_zero.1 a0) l).2 ⟨0, l.pos, h0⟩


theorem opow_ne_zero {a : Ordinal} (b : Ordinal) (a0 : a ≠ 0) : a ^ b ≠ 0 :=
  Ordinal.pos_iff_ne_zero.1 <| opow_pos b <| Ordinal.pos_iff_ne_zero.2 a0


@[simp]
theorem opow_eq_zero {a b : Ordinal} : a ^ b = 0 ↔ a = 0 ∧ b ≠ 0 := by
  /-
    a b : Ordinal.{u_1}
    ⊢ Iff (Eq (HPow.hPow a b) 0) (And (Eq a 0) (Ne b 0))
  -/
  obtain rfl | ha := eq_or_ne a 0
    /-
      case inl
      b : Ordinal.{u_1}
      ⊢ Iff (Eq (HPow.hPow 0 b) 0) (And (Eq 0 0) (Ne b 0))
    -/
  · obtain rfl | hb := eq_or_ne b 0
      /-
        case inl.inl
        ⊢ Iff (Eq (HPow.hPow 0 0) 0) (And (Eq 0 0) (Ne 0 0))
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case inl.inr
        b : Ordinal.{u_1}
        hb : Ne b 0
        ⊢ Iff (Eq (HPow.hPow 0 b) 0) (And (Eq 0 0) (Ne b 0))
      -/
    · simp [hb]
      /-
        🎉 no goals
      -/
    /-
      case inr
      a b : Ordinal.{u_1}
      ha : Ne a 0
      ⊢ Iff (Eq (HPow.hPow a b) 0) (And (Eq a 0) (Ne b 0))
    -/
  · simp [opow_ne_zero b ha, ha]
    /-
      🎉 no goals
    -/


@[simp, norm_cast]
theorem opow_natCast (a : Ordinal) (n : ℕ) : a ^ (n : Ordinal) = a ^ n := by
  induction n with
  | zero => rw [Nat.cast_zero, opow_zero, pow_zero]
  | succ n IH => rw [Nat.cast_succ, add_one_eq_succ, opow_succ, pow_succ, IH]


theorem isNormal_opow {a : Ordinal} (h : 1 < a) : IsNormal (a ^ ·) :=
  have a0 : 0 < a := zero_lt_one.trans h
               /-
                 a : Ordinal.{u_1}
                 h : LT.lt 1 a
                 a0 : LT.lt 0 a
                 b : Ordinal.{u_1}
                 ⊢ LT.lt ((fun x => HPow.hPow a x) b) ((fun x => HPow.hPow a x) (Order.succ b))
               -/
  ⟨fun b => by simpa only [mul_one, opow_succ] using (mul_lt_mul_iff_left (opow_pos b a0)).2 h,
               /-
                 🎉 no goals
               -/
    fun _ l _ => opow_le_of_limit (ne_of_gt a0) l⟩


@[deprecated isNormal_opow (since := "2024-10-11")]
alias opow_isNormal := isNormal_opow


theorem opow_lt_opow_iff_right {a b c : Ordinal} (a1 : 1 < a) : a ^ b < a ^ c ↔ b < c :=
  (isNormal_opow a1).lt_iff


theorem opow_le_opow_iff_right {a b c : Ordinal} (a1 : 1 < a) : a ^ b ≤ a ^ c ↔ b ≤ c :=
  (isNormal_opow a1).le_iff


theorem opow_right_inj {a b c : Ordinal} (a1 : 1 < a) : a ^ b = a ^ c ↔ b = c :=
  (isNormal_opow a1).inj


theorem isLimit_opow {a b : Ordinal} (a1 : 1 < a) : IsLimit b → IsLimit (a ^ b) :=
  (isNormal_opow a1).isLimit


@[deprecated isLimit_opow (since := "2024-10-11")]
alias opow_isLimit := isLimit_opow


theorem isLimit_opow_left {a b : Ordinal} (l : IsLimit a) (hb : b ≠ 0) : IsLimit (a ^ b) := by
  /-
    a b : Ordinal.{u_1}
    l : a.IsLimit
    hb : Ne b 0
    ⊢ (HPow.hPow a b).IsLimit
  -/
  rcases zero_or_succ_or_limit b with (e | ⟨b, rfl⟩ | l')
    /-
      case inl
      a b : Ordinal.{u_1}
      l : a.IsLimit
      hb : Ne b 0
      e : Eq b 0
      ⊢ (HPow.hPow a b).IsLimit
    -/
  · exact absurd e hb
    /-
      🎉 no goals
    -/
    /-
      case inr.inl.intro
      a : Ordinal.{u_1}
      l : a.IsLimit
      b : Ordinal.{u_1}
      hb : Ne (Order.succ b) 0
      ⊢ (HPow.hPow a (Order.succ b)).IsLimit
    -/
  · rw [opow_succ]
    /-
      case inr.inl.intro
      a : Ordinal.{u_1}
      l : a.IsLimit
      b : Ordinal.{u_1}
      hb : Ne (Order.succ b) 0
      ⊢ (HMul.hMul (HPow.hPow a b) a).IsLimit
    -/
    exact isLimit_mul (opow_pos _ l.pos) l
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      a b : Ordinal.{u_1}
      l : a.IsLimit
      hb : Ne b 0
      l' : b.IsLimit
      ⊢ (HPow.hPow a b).IsLimit
    -/
  · exact isLimit_opow l.one_lt l'
    /-
      🎉 no goals
    -/


@[deprecated isLimit_opow_left (since := "2024-10-11")]
alias opow_isLimit_left := isLimit_opow_left


theorem opow_le_opow_right {a b c : Ordinal} (h₁ : 0 < a) (h₂ : b ≤ c) : a ^ b ≤ a ^ c := by
  /-
    a b c : Ordinal.{u_1}
    h₁ : LT.lt 0 a
    h₂ : LE.le b c
    ⊢ LE.le (HPow.hPow a b) (HPow.hPow a c)
  -/
  rcases lt_or_eq_of_le (one_le_iff_pos.2 h₁) with h₁ | h₁
    /-
      case inl
      a b c : Ordinal.{u_1}
      h₁✝ : LT.lt 0 a
      h₂ : LE.le b c
      h₁ : LT.lt 1 a
      ⊢ LE.le (HPow.hPow a b) (HPow.hPow a c)
    -/
  · exact (opow_le_opow_iff_right h₁).2 h₂
    /-
      🎉 no goals
    -/
    /-
      case inr
      a b c : Ordinal.{u_1}
      h₁✝ : LT.lt 0 a
      h₂ : LE.le b c
      h₁ : Eq 1 a
      ⊢ LE.le (HPow.hPow a b) (HPow.hPow a c)
    -/
  · subst a
    -- Porting note: `le_refl` is required.
    /-
      case inr
      b c : Ordinal.{u_1}
      h₂ : LE.le b c
      h₁ : LT.lt 0 1
      ⊢ LE.le (HPow.hPow 1 b) (HPow.hPow 1 c)
    -/
    simp only [one_opow, le_refl]
    /-
      🎉 no goals
    -/


theorem opow_le_opow_left {a b : Ordinal} (c : Ordinal) (ab : a ≤ b) : a ^ c ≤ b ^ c := by
  /-
    a b c : Ordinal.{u_1}
    ab : LE.le a b
    ⊢ LE.le (HPow.hPow a c) (HPow.hPow b c)
  -/
  by_cases a0 : a = 0
  -- Porting note: `le_refl` is required.
    /-
      case pos
      a b c : Ordinal.{u_1}
      ab : LE.le a b
      a0 : Eq a 0
      ⊢ LE.le (HPow.hPow a c) (HPow.hPow b c)
    -/
  · subst a
    /-
      case pos
      b c : Ordinal.{u_1}
      ab : LE.le 0 b
      ⊢ LE.le (HPow.hPow 0 c) (HPow.hPow b c)
    -/
    by_cases c0 : c = 0
      /-
        case pos
        b c : Ordinal.{u_1}
        ab : LE.le 0 b
        c0 : Eq c 0
        ⊢ LE.le (HPow.hPow 0 c) (HPow.hPow b c)
      -/
    · subst c
      /-
        case pos
        b : Ordinal.{u_1}
        ab : LE.le 0 b
        ⊢ LE.le (HPow.hPow 0 0) (HPow.hPow b 0)
      -/
      simp only [opow_zero, le_refl]
      /-
        🎉 no goals
      -/
      /-
        case neg
        b c : Ordinal.{u_1}
        ab : LE.le 0 b
        c0 : Not (Eq c 0)
        ⊢ LE.le (HPow.hPow 0 c) (HPow.hPow b c)
      -/
    · simp only [zero_opow c0, Ordinal.zero_le]
      /-
        🎉 no goals
      -/
  · induction c using limitRecOn with
    | H₁ => simp only [opow_zero, le_refl]
    | H₂ c IH =>
      simpa only [opow_succ] using mul_le_mul' IH ab
    | H₃ c l IH =>
      exact
        (opow_le_of_limit a0 l).2 fun b' h =>
          (IH _ h).trans (opow_le_opow_right ((Ordinal.pos_iff_ne_zero.2 a0).trans_le ab) h.le)


theorem opow_le_opow {a b c d : Ordinal} (hac : a ≤ c) (hbd : b ≤ d) (hc : 0 < c) : a ^ b ≤ c ^ d :=
  (opow_le_opow_left b hac).trans (opow_le_opow_right hc hbd)


theorem left_le_opow (a : Ordinal) {b : Ordinal} (b1 : 0 < b) : a ≤ a ^ b := by
  /-
    a b : Ordinal.{u_1}
    b1 : LT.lt 0 b
    ⊢ LE.le a (HPow.hPow a b)
  -/
  nth_rw 1 [← opow_one a]
  /-
    a b : Ordinal.{u_1}
    b1 : LT.lt 0 b
    ⊢ LE.le (HPow.hPow a 1) (HPow.hPow a b)
  -/
  cases' le_or_gt a 1 with a1 a1
    /-
      case inl
      a b : Ordinal.{u_1}
      b1 : LT.lt 0 b
      a1 : LE.le a 1
      ⊢ LE.le (HPow.hPow a 1) (HPow.hPow a b)
    -/
  · rcases lt_or_eq_of_le a1 with a0 | a1
      /-
        case inl.inl
        a b : Ordinal.{u_1}
        b1 : LT.lt 0 b
        a1 : LE.le a 1
        a0 : LT.lt a 1
        ⊢ LE.le (HPow.hPow a 1) (HPow.hPow a b)
      -/
    · rw [lt_one_iff_zero] at a0
      /-
        case inl.inl
        a b : Ordinal.{u_1}
        b1 : LT.lt 0 b
        a1 : LE.le a 1
        a0 : Eq a 0
        ⊢ LE.le (HPow.hPow a 1) (HPow.hPow a b)
      -/
      rw [a0, zero_opow Ordinal.one_ne_zero]
      /-
        case inl.inl
        a b : Ordinal.{u_1}
        b1 : LT.lt 0 b
        a1 : LE.le a 1
        a0 : Eq a 0
        ⊢ LE.le 0 (HPow.hPow 0 b)
      -/
      exact Ordinal.zero_le _
      /-
        🎉 no goals
      -/
    /-
      case inl.inr
      a b : Ordinal.{u_1}
      b1 : LT.lt 0 b
      a1✝ : LE.le a 1
      a1 : Eq a 1
      ⊢ LE.le (HPow.hPow a 1) (HPow.hPow a b)
    -/
    rw [a1, one_opow, one_opow]
    /-
      🎉 no goals
    -/
  /-
    case inr
    a b : Ordinal.{u_1}
    b1 : LT.lt 0 b
    a1 : GT.gt a 1
    ⊢ LE.le (HPow.hPow a 1) (HPow.hPow a b)
  -/
  rwa [opow_le_opow_iff_right a1, one_le_iff_pos]
  /-
    🎉 no goals
  -/


theorem left_lt_opow {a b : Ordinal} (ha : 1 < a) (hb : 1 < b) : a < a ^ b := by
  /-
    a b : Ordinal.{u_1}
    ha : LT.lt 1 a
    hb : LT.lt 1 b
    ⊢ LT.lt a (HPow.hPow a b)
  -/
  conv_lhs => rw [← opow_one a]
  /-
    a b : Ordinal.{u_1}
    ha : LT.lt 1 a
    hb : LT.lt 1 b
    ⊢ LT.lt (HPow.hPow a 1) (HPow.hPow a b)
  -/
  rwa [opow_lt_opow_iff_right ha]
  /-
    🎉 no goals
  -/


theorem right_le_opow {a : Ordinal} (b : Ordinal) (a1 : 1 < a) : b ≤ a ^ b :=
  (isNormal_opow a1).le_apply


theorem opow_lt_opow_left_of_succ {a b c : Ordinal} (ab : a < b) : a ^ succ c < b ^ succ c := by
  /-
    a b c : Ordinal.{u_1}
    ab : LT.lt a b
    ⊢ LT.lt (HPow.hPow a (Order.succ c)) (HPow.hPow b (Order.succ c))
  -/
  rw [opow_succ, opow_succ]
  exact
    (mul_le_mul_right' (opow_le_opow_left c ab.le) a).trans_lt
      (mul_lt_mul_of_pos_left ab (opow_pos c ((Ordinal.zero_le a).trans_lt ab)))


theorem opow_add (a b c : Ordinal) : a ^ (b + c) = a ^ b * a ^ c := by
  /-
    a b c : Ordinal.{u_1}
    ⊢ Eq (HPow.hPow a (HAdd.hAdd b c)) (HMul.hMul (HPow.hPow a b) (HPow.hPow a c))
  -/
  rcases eq_or_ne a 0 with (rfl | a0)
    /-
      case inl
      b c : Ordinal.{u_1}
      ⊢ Eq (HPow.hPow 0 (HAdd.hAdd b c)) (HMul.hMul (HPow.hPow 0 b) (HPow.hPow 0 c))
    -/
  · rcases eq_or_ne c 0 with (rfl | c0)
      /-
        case inl.inl
        b : Ordinal.{u_1}
        ⊢ Eq (HPow.hPow 0 (HAdd.hAdd b 0)) (HMul.hMul (HPow.hPow 0 b) (HPow.hPow 0 0))
      -/
    · simp
      /-
        🎉 no goals
      -/
    /-
      case inl.inr
      b c : Ordinal.{u_1}
      c0 : Ne c 0
      ⊢ Eq (HPow.hPow 0 (HAdd.hAdd b c)) (HMul.hMul (HPow.hPow 0 b) (HPow.hPow 0 c))
    -/
    have : b + c ≠ 0 := ((Ordinal.pos_iff_ne_zero.2 c0).trans_le (le_add_left _ _)).ne'
    /-
      case inl.inr
      b c : Ordinal.{u_1}
      c0 : Ne c 0
      this : Ne (HAdd.hAdd b c) 0
      ⊢ Eq (HPow.hPow 0 (HAdd.hAdd b c)) (HMul.hMul (HPow.hPow 0 b) (HPow.hPow 0 c))
    -/
    simp only [zero_opow c0, zero_opow this, mul_zero]
    /-
      🎉 no goals
    -/
  /-
    case inr
    a b c : Ordinal.{u_1}
    a0 : Ne a 0
    ⊢ Eq (HPow.hPow a (HAdd.hAdd b c)) (HMul.hMul (HPow.hPow a b) (HPow.hPow a c))
  -/
  rcases eq_or_lt_of_le (one_le_iff_ne_zero.2 a0) with (rfl | a1)
    /-
      case inr.inl
      b c : Ordinal.{u_1}
      a0 : Ne 1 0
      ⊢ Eq (HPow.hPow 1 (HAdd.hAdd b c)) (HMul.hMul (HPow.hPow 1 b) (HPow.hPow 1 c))
    -/
  · simp only [one_opow, mul_one]
    /-
      🎉 no goals
    -/
  induction c using limitRecOn with
  | H₁ => simp
  | H₂ c IH =>
    rw [add_succ, opow_succ, IH, opow_succ, mul_assoc]
  | H₃ c l IH =>
    refine
      eq_of_forall_ge_iff fun d =>
        (((isNormal_opow a1).trans (isNormal_add_right b)).limit_le l).trans ?_
    dsimp only [Function.comp_def]
    simp +contextual only [IH]
    exact
      (((isNormal_mul_right <| opow_pos b (Ordinal.pos_iff_ne_zero.2 a0)).trans
              (isNormal_opow a1)).limit_le
          l).symm


                                                                     /-
                                                                       a b : Ordinal.{u_1}
                                                                       ⊢ Eq (HPow.hPow a (HAdd.hAdd 1 b)) (HMul.hMul a (HPow.hPow a b))
                                                                     -/
theorem opow_one_add (a b : Ordinal) : a ^ (1 + b) = a * a ^ b := by rw [opow_add, opow_one]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


theorem opow_dvd_opow (a : Ordinal) {b c : Ordinal} (h : b ≤ c) : a ^ b ∣ a ^ c :=
                   /-
                     a b c : Ordinal.{u_1}
                     h : LE.le b c
                     ⊢ Eq (HPow.hPow a c) (HMul.hMul (HPow.hPow a b) (HPow.hPow a (HSub.hSub c b)))
                   -/
  ⟨a ^ (c - b), by rw [← opow_add, Ordinal.add_sub_cancel_of_le h]⟩
                   /-
                     🎉 no goals
                   -/


theorem opow_dvd_opow_iff {a b c : Ordinal} (a1 : 1 < a) : a ^ b ∣ a ^ c ↔ b ≤ c :=
  ⟨fun h =>
    le_of_not_lt fun hn =>
      not_le_of_lt ((opow_lt_opow_iff_right a1).2 hn) <|
        le_of_dvd (opow_ne_zero _ <| one_le_iff_ne_zero.1 <| a1.le) h,
    opow_dvd_opow _⟩


theorem opow_mul (a b c : Ordinal) : a ^ (b * c) = (a ^ b) ^ c := by
  /-
    a b c : Ordinal.{u_1}
    ⊢ Eq (HPow.hPow a (HMul.hMul b c)) (HPow.hPow (HPow.hPow a b) c)
  -/
  by_cases b0 : b = 0; · simp only [b0, zero_mul, opow_zero, one_opow]
                         /-
                           🎉 no goals
                         -/
  /-
    case neg
    a b c : Ordinal.{u_1}
    b0 : Not (Eq b 0)
    ⊢ Eq (HPow.hPow a (HMul.hMul b c)) (HPow.hPow (HPow.hPow a b) c)
  -/
  by_cases a0 : a = 0
    /-
      case pos
      a b c : Ordinal.{u_1}
      b0 : Not (Eq b 0)
      a0 : Eq a 0
      ⊢ Eq (HPow.hPow a (HMul.hMul b c)) (HPow.hPow (HPow.hPow a b) c)
    -/
  · subst a
    /-
      case pos
      b c : Ordinal.{u_1}
      b0 : Not (Eq b 0)
      ⊢ Eq (HPow.hPow 0 (HMul.hMul b c)) (HPow.hPow (HPow.hPow 0 b) c)
    -/
    by_cases c0 : c = 0
      /-
        case pos
        b c : Ordinal.{u_1}
        b0 : Not (Eq b 0)
        c0 : Eq c 0
        ⊢ Eq (HPow.hPow 0 (HMul.hMul b c)) (HPow.hPow (HPow.hPow 0 b) c)
      -/
    · simp only [c0, mul_zero, opow_zero]
      /-
        🎉 no goals
      -/
    /-
      case neg
      b c : Ordinal.{u_1}
      b0 : Not (Eq b 0)
      c0 : Not (Eq c 0)
      ⊢ Eq (HPow.hPow 0 (HMul.hMul b c)) (HPow.hPow (HPow.hPow 0 b) c)
    -/
    simp only [zero_opow b0, zero_opow c0, zero_opow (mul_ne_zero b0 c0)]
    /-
      🎉 no goals
    -/
  /-
    case neg
    a b c : Ordinal.{u_1}
    b0 : Not (Eq b 0)
    a0 : Not (Eq a 0)
    ⊢ Eq (HPow.hPow a (HMul.hMul b c)) (HPow.hPow (HPow.hPow a b) c)
  -/
  cases' eq_or_lt_of_le (one_le_iff_ne_zero.2 a0) with a1 a1
    /-
      case neg.inl
      a b c : Ordinal.{u_1}
      b0 : Not (Eq b 0)
      a0 : Not (Eq a 0)
      a1 : Eq 1 a
      ⊢ Eq (HPow.hPow a (HMul.hMul b c)) (HPow.hPow (HPow.hPow a b) c)
    -/
  · subst a1
    /-
      case neg.inl
      b c : Ordinal.{u_1}
      b0 : Not (Eq b 0)
      a0 : Not (Eq 1 0)
      ⊢ Eq (HPow.hPow 1 (HMul.hMul b c)) (HPow.hPow (HPow.hPow 1 b) c)
    -/
    simp only [one_opow]
    /-
      🎉 no goals
    -/
  induction c using limitRecOn with
  | H₁ => simp only [mul_zero, opow_zero]
  | H₂ c IH =>
    rw [mul_succ, opow_add, IH, opow_succ]
  | H₃ c l IH =>
    refine
      eq_of_forall_ge_iff fun d =>
        (((isNormal_opow a1).trans (isNormal_mul_right (Ordinal.pos_iff_ne_zero.2 b0))).limit_le
              l).trans
          ?_
    dsimp only [Function.comp_def]
    simp +contextual only [IH]
    exact (opow_le_of_limit (opow_ne_zero _ a0) l).symm


theorem opow_mul_add_pos {b v : Ordinal} (hb : b ≠ 0) (u : Ordinal) (hv : v ≠ 0) (w : Ordinal) :
    0 < b ^ u * v + w :=
  (opow_pos u <| Ordinal.pos_iff_ne_zero.2 hb).trans_le <|
    (le_mul_left _ <| Ordinal.pos_iff_ne_zero.2 hv).trans <| le_add_right _ _


theorem opow_mul_add_lt_opow_mul_succ {b u w : Ordinal} (v : Ordinal) (hw : w < b ^ u) :
    b ^ u * v + w < b ^ u * succ v := by
  /-
    b u w v : Ordinal.{u_1}
    hw : LT.lt w (HPow.hPow b u)
    ⊢ LT.lt (HAdd.hAdd (HMul.hMul (HPow.hPow b u) v) w) (HMul.hMul (HPow.hPow b u) …
  -/
  rwa [mul_succ, add_lt_add_iff_left]
  /-
    🎉 no goals
  -/


theorem opow_mul_add_lt_opow_succ {b u v w : Ordinal} (hvb : v < b) (hw : w < b ^ u) :
    b ^ u * v + w < b ^ succ u := by
  convert (opow_mul_add_lt_opow_mul_succ v hw).trans_le
    (mul_le_mul_left' (succ_le_of_lt hvb) _) using 1
  /-
    case h.e'_4
    b u v w : Ordinal.{u_1}
    hvb : LT.lt v b
    hw : LT.lt w (HPow.hPow b u)
    ⊢ Eq (HPow.hPow b (Order.succ u)) (HMul.hMul (HPow.hPow b u) b)
  -/
  exact opow_succ b u
  /-
    🎉 no goals
  -/


/-- The ordinal logarithm is the solution `u` to the equation `x = b ^ u * v + w` where `v < b` and
`w < b ^ u`. -/
@[pp_nodot]
def log (b : Ordinal) (x : Ordinal) : Ordinal :=
  if 1 < b then pred (sInf { o | x < b ^ o }) else 0


/-- The set in the definition of `log` is nonempty. -/
private theorem log_nonempty {b x : Ordinal} (h : 1 < b) : { o : Ordinal | x < b ^ o }.Nonempty :=
  ⟨_, succ_le_iff.1 (right_le_opow _ h)⟩


theorem log_def {b : Ordinal} (h : 1 < b) (x : Ordinal) : log b x = pred (sInf { o | x < b ^ o }) :=
  if_pos h


theorem log_of_left_le_one {b : Ordinal} (h : b ≤ 1) (x : Ordinal) : log b x = 0 :=
  if_neg h.not_lt


@[deprecated log_of_left_le_one (since := "2024-10-10")]
theorem log_of_not_one_lt_left {b : Ordinal} (h : ¬1 < b) (x : Ordinal) : log b x = 0 := by
  /-
    b : Ordinal.{u_1}
    h : Not (LT.lt 1 b)
    x : Ordinal.{u_1}
    ⊢ Eq (Ordinal.log b x) 0
  -/
  simp only [log, if_neg h]
  /-
    🎉 no goals
  -/


@[simp]
theorem log_zero_left : ∀ b, log 0 b = 0 :=
  log_of_left_le_one zero_le_one


@[simp]
theorem log_zero_right (b : Ordinal) : log b 0 = 0 := by
  /-
    b : Ordinal.{u_1}
    ⊢ Eq (Ordinal.log b 0) 0
  -/
  obtain hb | hb := lt_or_le 1 b
    /-
      case inl
      b : Ordinal.{u_1}
      hb : LT.lt 1 b
      ⊢ Eq (Ordinal.log b 0) 0
    -/
  · rw [log_def hb, ← Ordinal.le_zero, pred_le, succ_zero]
    /-
      case inl
      b : Ordinal.{u_1}
      hb : LT.lt 1 b
      ⊢ LE.le (InfSet.sInf (setOf fun o => LT.lt 0 (HPow.hPow b o))) 1
    -/
    apply csInf_le'
    /-
      case inl.h
      b : Ordinal.{u_1}
      hb : LT.lt 1 b
      ⊢ Membership.mem (setOf fun o => LT.lt 0 (HPow.hPow b o)) 1
    -/
    rw [mem_setOf, opow_one]
    /-
      case inl.h
      b : Ordinal.{u_1}
      hb : LT.lt 1 b
      ⊢ LT.lt 0 b
    -/
    exact bot_lt_of_lt hb
    /-
      🎉 no goals
    -/
    /-
      case inr
      b : Ordinal.{u_1}
      hb : LE.le b 1
      ⊢ Eq (Ordinal.log b 0) 0
    -/
  · rw [log_of_left_le_one hb]
    /-
      🎉 no goals
    -/


@[simp]
theorem log_one_left : ∀ b, log 1 b = 0 :=
  log_of_left_le_one le_rfl


theorem succ_log_def {b x : Ordinal} (hb : 1 < b) (hx : x ≠ 0) :
    succ (log b x) = sInf { o : Ordinal | x < b ^ o } := by
  /-
    b x : Ordinal.{u_1}
    hb : LT.lt 1 b
    hx : Ne x 0
    ⊢ Eq (Order.succ (Ordinal.log b x)) (InfSet.sInf (setOf fun o => LT.lt x (HPow …
  -/
  let t := sInf { o : Ordinal | x < b ^ o }
  /-
    b x : Ordinal.{u_1}
    hb : LT.lt 1 b
    hx : Ne x 0
    t : Ordinal.{u_1} := InfSet.sInf (setOf fun o => LT.lt x (HPow.hPow b o))
    ⊢ Eq (Order.succ (Ordinal.log b x)) (InfSet.sInf (setOf fun o => LT.lt x (HPow …
  -/
  have : x < b ^ t := csInf_mem (log_nonempty hb)
  /-
    b x : Ordinal.{u_1}
    hb : LT.lt 1 b
    hx : Ne x 0
    t : Ordinal.{u_1} := InfSet.sInf (setOf fun o => LT.lt x (HPow.hPow b o))
    this : LT.lt x (HPow.hPow b t)
    ⊢ Eq (Order.succ (Ordinal.log b x)) (InfSet.sInf (setOf fun o => LT.lt x (HPow …
  -/
  rcases zero_or_succ_or_limit t with (h | h | h)
    /-
      case inl
      b x : Ordinal.{u_1}
      hb : LT.lt 1 b
      hx : Ne x 0
      t : Ordinal.{u_1} := InfSet.sInf (setOf fun o => LT.lt x (HPow.hPow b o))
      this : LT.lt x (HPow.hPow b t)
      h : Eq t 0
      ⊢ Eq (Order.succ (Ordinal.log b x)) (InfSet.sInf (setOf fun o => LT.lt x (HPow …
    -/
  · refine ((one_le_iff_ne_zero.2 hx).not_lt ?_).elim
    /-
      case inl
      b x : Ordinal.{u_1}
      hb : LT.lt 1 b
      hx : Ne x 0
      t : Ordinal.{u_1} := InfSet.sInf (setOf fun o => LT.lt x (HPow.hPow b o))
      this : LT.lt x (HPow.hPow b t)
      h : Eq t 0
      ⊢ LT.lt x 1
    -/
    simpa only [h, opow_zero] using this
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      b x : Ordinal.{u_1}
      hb : LT.lt 1 b
      hx : Ne x 0
      t : Ordinal.{u_1} := InfSet.sInf (setOf fun o => LT.lt x (HPow.hPow b o))
      this : LT.lt x (HPow.hPow b t)
      h : Exists fun a => Eq t (Order.succ a)
      ⊢ Eq (Order.succ (Ordinal.log b x)) (InfSet.sInf (setOf fun o => LT.lt x (HPow …
    -/
  · rw [show log b x = pred t from log_def hb x, succ_pred_iff_is_succ.2 h]
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      b x : Ordinal.{u_1}
      hb : LT.lt 1 b
      hx : Ne x 0
      t : Ordinal.{u_1} := InfSet.sInf (setOf fun o => LT.lt x (HPow.hPow b o))
      this : LT.lt x (HPow.hPow b t)
      h : t.IsLimit
      ⊢ Eq (Order.succ (Ordinal.log b x)) (InfSet.sInf (setOf fun o => LT.lt x (HPow …
    -/
  · rcases (lt_opow_of_limit (zero_lt_one.trans hb).ne' h).1 this with ⟨a, h₁, h₂⟩
    /-
      case inr.inr.intro.intro
      b x : Ordinal.{u_1}
      hb : LT.lt 1 b
      hx : Ne x 0
      t : Ordinal.{u_1} := InfSet.sInf (setOf fun o => LT.lt x (HPow.hPow b o))
      this : LT.lt x (HPow.hPow b t)
      h : t.IsLimit
      a : Ordinal.{u_1}
      h₁ : LT.lt a t
      h₂ : LT.lt x (HPow.hPow b a)
      ⊢ Eq (Order.succ (Ordinal.log b x)) (InfSet.sInf (setOf fun o => LT.lt x (HPow …
    -/
    exact h₁.not_le.elim ((le_csInf_iff'' (log_nonempty hb)).1 le_rfl a h₂)
    /-
      🎉 no goals
    -/


theorem lt_opow_succ_log_self {b : Ordinal} (hb : 1 < b) (x : Ordinal) :
    x < b ^ succ (log b x) := by
  /-
    b : Ordinal.{u_1}
    hb : LT.lt 1 b
    x : Ordinal.{u_1}
    ⊢ LT.lt x (HPow.hPow b (Order.succ (Ordinal.log b x)))
  -/
  rcases eq_or_ne x 0 with (rfl | hx)
    /-
      case inl
      b : Ordinal.{u_1}
      hb : LT.lt 1 b
      ⊢ LT.lt 0 (HPow.hPow b (Order.succ (Ordinal.log b 0)))
    -/
  · apply opow_pos _ (zero_lt_one.trans hb)
    /-
      🎉 no goals
    -/
    /-
      case inr
      b : Ordinal.{u_1}
      hb : LT.lt 1 b
      x : Ordinal.{u_1}
      hx : Ne x 0
      ⊢ LT.lt x (HPow.hPow b (Order.succ (Ordinal.log b x)))
    -/
  · rw [succ_log_def hb hx]
    /-
      case inr
      b : Ordinal.{u_1}
      hb : LT.lt 1 b
      x : Ordinal.{u_1}
      hx : Ne x 0
      ⊢ LT.lt x (HPow.hPow b (InfSet.sInf (setOf fun o => LT.lt x (HPow.hPow b o))))
    -/
    exact csInf_mem (log_nonempty hb)
    /-
      🎉 no goals
    -/


theorem opow_log_le_self (b : Ordinal) {x : Ordinal} (hx : x ≠ 0) : b ^ log b x ≤ x := by
  /-
    b x : Ordinal.{u_1}
    hx : Ne x 0
    ⊢ LE.le (HPow.hPow b (Ordinal.log b x)) x
  -/
  rcases eq_or_ne b 0 with (rfl | b0)
    /-
      case inl
      x : Ordinal.{u_1}
      hx : Ne x 0
      ⊢ LE.le (HPow.hPow 0 (Ordinal.log 0 x)) x
    -/
  · rw [zero_opow']
    /-
      case inl
      x : Ordinal.{u_1}
      hx : Ne x 0
      ⊢ LE.le (HSub.hSub 1 (Ordinal.log 0 x)) x
    -/
    exact (sub_le_self _ _).trans (one_le_iff_ne_zero.2 hx)
    /-
      🎉 no goals
    -/
  /-
    case inr
    b x : Ordinal.{u_1}
    hx : Ne x 0
    b0 : Ne b 0
    ⊢ LE.le (HPow.hPow b (Ordinal.log b x)) x
  -/
  rcases lt_or_eq_of_le (one_le_iff_ne_zero.2 b0) with (hb | rfl)
    /-
      case inr.inl
      b x : Ordinal.{u_1}
      hx : Ne x 0
      b0 : Ne b 0
      hb : LT.lt 1 b
      ⊢ LE.le (HPow.hPow b (Ordinal.log b x)) x
    -/
  · refine le_of_not_lt fun h => (lt_succ (log b x)).not_le ?_
    /-
      case inr.inl
      b x : Ordinal.{u_1}
      hx : Ne x 0
      b0 : Ne b 0
      hb : LT.lt 1 b
      h : LT.lt x (HPow.hPow b (Ordinal.log b x))
      ⊢ LE.le (Order.succ (Ordinal.log b x)) (Ordinal.log b x)
    -/
    have := @csInf_le' _ _ { o | x < b ^ o } _ h
    /-
      case inr.inl
      b x : Ordinal.{u_1}
      hx : Ne x 0
      b0 : Ne b 0
      hb : LT.lt 1 b
      h : LT.lt x (HPow.hPow b (Ordinal.log b x))
      this : LE.le (InfSet.sInf (setOf fun o => LT.lt x (HPow.hPow b o))) (Ordinal.l …
      ⊢ LE.le (Order.succ (Ordinal.log b x)) (Ordinal.log b x)
    -/
    rwa [← succ_log_def hb hx] at this
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      x : Ordinal.{u_1}
      hx : Ne x 0
      b0 : Ne 1 0
      ⊢ LE.le (HPow.hPow 1 (Ordinal.log 1 x)) x
    -/
  · rwa [one_opow, one_le_iff_ne_zero]
    /-
      🎉 no goals
    -/


/-- `opow b` and `log b` (almost) form a Galois connection.

See `opow_le_iff_le_log'` for a variant assuming `c ≠ 0` rather than `x ≠ 0`. See also
`le_log_of_opow_le` and `opow_le_of_le_log`, which are both separate implications under weaker
assumptions. -/
theorem opow_le_iff_le_log {b x c : Ordinal} (hb : 1 < b) (hx : x ≠ 0) :
    b ^ c ≤ x ↔ c ≤ log b x := by
  /-
    b x c : Ordinal.{u_1}
    hb : LT.lt 1 b
    hx : Ne x 0
    ⊢ Iff (LE.le (HPow.hPow b c) x) (LE.le c (Ordinal.log b x))
  -/
  constructor <;>
  /-
    case mp
    b x c : Ordinal.{u_1}
    hb : LT.lt 1 b
    hx : Ne x 0
    ⊢ LE.le (HPow.hPow b c) x → LE.le c (Ordinal.log b x)
  -/
  intro h
    /-
      case mp
      b x c : Ordinal.{u_1}
      hb : LT.lt 1 b
      hx : Ne x 0
      h : LE.le (HPow.hPow b c) x
      ⊢ LE.le c (Ordinal.log b x)
    -/
  · apply le_of_not_lt
    /-
      case mp.h
      b x c : Ordinal.{u_1}
      hb : LT.lt 1 b
      hx : Ne x 0
      h : LE.le (HPow.hPow b c) x
      ⊢ Not (LT.lt (Ordinal.log b x) c)
    -/
    intro hn
    apply (lt_opow_succ_log_self hb x).not_le <|
      ((opow_le_opow_iff_right hb).2 <| succ_le_of_lt hn).trans h
    /-
      case mpr
      b x c : Ordinal.{u_1}
      hb : LT.lt 1 b
      hx : Ne x 0
      h : LE.le c (Ordinal.log b x)
      ⊢ LE.le (HPow.hPow b c) x
    -/
  · exact ((opow_le_opow_iff_right hb).2 h).trans <| opow_log_le_self b hx
    /-
      🎉 no goals
    -/


/-- `opow b` and `log b` (almost) form a Galois connection.

See `opow_le_iff_le_log` for a variant assuming `x ≠ 0` rather than `c ≠ 0`. See also
`le_log_of_opow_le` and `opow_le_of_le_log`, which are both separate implications under weaker
assumptions. -/
theorem opow_le_iff_le_log' {b x c : Ordinal} (hb : 1 < b) (hc : c ≠ 0) :
    b ^ c ≤ x ↔ c ≤ log b x := by
  /-
    b x c : Ordinal.{u_1}
    hb : LT.lt 1 b
    hc : Ne c 0
    ⊢ Iff (LE.le (HPow.hPow b c) x) (LE.le c (Ordinal.log b x))
  -/
  obtain rfl | hx := eq_or_ne x 0
    /-
      case inl
      b c : Ordinal.{u_1}
      hb : LT.lt 1 b
      hc : Ne c 0
      ⊢ Iff (LE.le (HPow.hPow b c) 0) (LE.le c (Ordinal.log b 0))
    -/
  · rw [log_zero_right, Ordinal.le_zero, Ordinal.le_zero, opow_eq_zero]
    /-
      case inl
      b c : Ordinal.{u_1}
      hb : LT.lt 1 b
      hc : Ne c 0
      ⊢ Iff (And (Eq b 0) (Ne c 0)) (Eq c 0)
    -/
    simp [hc, (zero_lt_one.trans hb).ne']
    /-
      🎉 no goals
    -/
    /-
      case inr
      b x c : Ordinal.{u_1}
      hb : LT.lt 1 b
      hc : Ne c 0
      hx : Ne x 0
      ⊢ Iff (LE.le (HPow.hPow b c) x) (LE.le c (Ordinal.log b x))
    -/
  · exact opow_le_iff_le_log hb hx
    /-
      🎉 no goals
    -/


theorem le_log_of_opow_le {b x c : Ordinal} (hb : 1 < b) (h : b ^ c ≤ x) : c ≤ log b x := by
  /-
    b x c : Ordinal.{u_1}
    hb : LT.lt 1 b
    h : LE.le (HPow.hPow b c) x
    ⊢ LE.le c (Ordinal.log b x)
  -/
  obtain rfl | hx := eq_or_ne x 0
    /-
      case inl
      b c : Ordinal.{u_1}
      hb : LT.lt 1 b
      h : LE.le (HPow.hPow b c) 0
      ⊢ LE.le c (Ordinal.log b 0)
    -/
  · rw [Ordinal.le_zero, opow_eq_zero] at h
    /-
      case inl
      b c : Ordinal.{u_1}
      hb : LT.lt 1 b
      h : And (Eq b 0) (Ne c 0)
      ⊢ LE.le c (Ordinal.log b 0)
    -/
    exact (zero_lt_one.asymm <| h.1 ▸ hb).elim
    /-
      🎉 no goals
    -/
    /-
      case inr
      b x c : Ordinal.{u_1}
      hb : LT.lt 1 b
      h : LE.le (HPow.hPow b c) x
      hx : Ne x 0
      ⊢ LE.le c (Ordinal.log b x)
    -/
  · exact (opow_le_iff_le_log hb hx).1 h
    /-
      🎉 no goals
    -/


theorem opow_le_of_le_log {b x c : Ordinal} (hc : c ≠ 0) (h : c ≤ log b x) : b ^ c ≤ x := by
  /-
    b x c : Ordinal.{u_1}
    hc : Ne c 0
    h : LE.le c (Ordinal.log b x)
    ⊢ LE.le (HPow.hPow b c) x
  -/
  obtain hb | hb := le_or_lt b 1
    /-
      case inl
      b x c : Ordinal.{u_1}
      hc : Ne c 0
      h : LE.le c (Ordinal.log b x)
      hb : LE.le b 1
      ⊢ LE.le (HPow.hPow b c) x
    -/
  · rw [log_of_left_le_one hb] at h
    /-
      case inl
      b x c : Ordinal.{u_1}
      hc : Ne c 0
      h : LE.le c 0
      hb : LE.le b 1
      ⊢ LE.le (HPow.hPow b c) x
    -/
    exact (h.not_lt (Ordinal.pos_iff_ne_zero.2 hc)).elim
    /-
      🎉 no goals
    -/
    /-
      case inr
      b x c : Ordinal.{u_1}
      hc : Ne c 0
      h : LE.le c (Ordinal.log b x)
      hb : LT.lt 1 b
      ⊢ LE.le (HPow.hPow b c) x
    -/
  · rwa [opow_le_iff_le_log' hb hc]
    /-
      🎉 no goals
    -/


/-- `opow b` and `log b` (almost) form a Galois connection.

See `lt_opow_iff_log_lt'` for a variant assuming `c ≠ 0` rather than `x ≠ 0`. See also
`lt_opow_of_log_lt` and `lt_log_of_lt_opow`, which are both separate implications under weaker
assumptions. -/
theorem lt_opow_iff_log_lt {b x c : Ordinal} (hb : 1 < b) (hx : x ≠ 0) : x < b ^ c ↔ log b x < c :=
  lt_iff_lt_of_le_iff_le (opow_le_iff_le_log hb hx)


/-- `opow b` and `log b` (almost) form a Galois connection.

See `lt_opow_iff_log_lt` for a variant assuming `x ≠ 0` rather than `c ≠ 0`. See also
`lt_opow_of_log_lt` and `lt_log_of_lt_opow`, which are both separate implications under weaker
assumptions. -/
theorem lt_opow_iff_log_lt' {b x c : Ordinal} (hb : 1 < b) (hc : c ≠ 0) : x < b ^ c ↔ log b x < c :=
  lt_iff_lt_of_le_iff_le (opow_le_iff_le_log' hb hc)


theorem lt_opow_of_log_lt {b x c : Ordinal} (hb : 1 < b) : log b x < c → x < b ^ c :=
  lt_imp_lt_of_le_imp_le <| le_log_of_opow_le hb


theorem lt_log_of_lt_opow {b x c : Ordinal} (hc : c ≠ 0) : x < b ^ c → log b x < c :=
  lt_imp_lt_of_le_imp_le <| opow_le_of_le_log hc


theorem log_pos {b o : Ordinal} (hb : 1 < b) (ho : o ≠ 0) (hbo : b ≤ o) : 0 < log b o := by
  /-
    b o : Ordinal.{u_1}
    hb : LT.lt 1 b
    ho : Ne o 0
    hbo : LE.le b o
    ⊢ LT.lt 0 (Ordinal.log b o)
  -/
  rwa [← succ_le_iff, succ_zero, ← opow_le_iff_le_log hb ho, opow_one]
  /-
    🎉 no goals
  -/


theorem log_eq_zero {b o : Ordinal} (hbo : o < b) : log b o = 0 := by
  /-
    b o : Ordinal.{u_1}
    hbo : LT.lt o b
    ⊢ Eq (Ordinal.log b o) 0
  -/
  rcases eq_or_ne o 0 with (rfl | ho)
    /-
      case inl
      b : Ordinal.{u_1}
      hbo : LT.lt 0 b
      ⊢ Eq (Ordinal.log b 0) 0
    -/
  · exact log_zero_right b
    /-
      🎉 no goals
    -/
  /-
    case inr
    b o : Ordinal.{u_1}
    hbo : LT.lt o b
    ho : Ne o 0
    ⊢ Eq (Ordinal.log b o) 0
  -/
  rcases le_or_lt b 1 with hb | hb
    /-
      case inr.inl
      b o : Ordinal.{u_1}
      hbo : LT.lt o b
      ho : Ne o 0
      hb : LE.le b 1
      ⊢ Eq (Ordinal.log b o) 0
    -/
  · rcases le_one_iff.1 hb with (rfl | rfl)
      /-
        case inr.inl.inl
        o : Ordinal.{u_1}
        ho : Ne o 0
        hbo : LT.lt o 0
        hb : LE.le 0 1
        ⊢ Eq (Ordinal.log 0 o) 0
      -/
    · exact log_zero_left o
      /-
        🎉 no goals
      -/
      /-
        case inr.inl.inr
        o : Ordinal.{u_1}
        ho : Ne o 0
        hbo : LT.lt o 1
        hb : LE.le 1 1
        ⊢ Eq (Ordinal.log 1 o) 0
      -/
    · exact log_one_left o
      /-
        🎉 no goals
      -/
    /-
      case inr.inr
      b o : Ordinal.{u_1}
      hbo : LT.lt o b
      ho : Ne o 0
      hb : LT.lt 1 b
      ⊢ Eq (Ordinal.log b o) 0
    -/
  · rwa [← Ordinal.le_zero, ← lt_succ_iff, succ_zero, ← lt_opow_iff_log_lt hb ho, opow_one]
    /-
      🎉 no goals
    -/


@[mono]
theorem log_mono_right (b : Ordinal) {x y : Ordinal} (xy : x ≤ y) : log b x ≤ log b y := by
  /-
    b x y : Ordinal.{u_1}
    xy : LE.le x y
    ⊢ LE.le (Ordinal.log b x) (Ordinal.log b y)
  -/
  obtain rfl | hx := eq_or_ne x 0
    /-
      case inl
      b y : Ordinal.{u_1}
      xy : LE.le 0 y
      ⊢ LE.le (Ordinal.log b 0) (Ordinal.log b y)
    -/
  · simp_rw [log_zero_right, Ordinal.zero_le]
    /-
      🎉 no goals
    -/
    /-
      case inr
      b x y : Ordinal.{u_1}
      xy : LE.le x y
      hx : Ne x 0
      ⊢ LE.le (Ordinal.log b x) (Ordinal.log b y)
    -/
  · obtain hb | hb := lt_or_le 1 b
    · exact (opow_le_iff_le_log hb (hx.bot_lt.trans_le xy).ne').1 <|
        (opow_log_le_self _ hx).trans xy
      /-
        case inr.inr
        b x y : Ordinal.{u_1}
        xy : LE.le x y
        hx : Ne x 0
        hb : LE.le b 1
        ⊢ LE.le (Ordinal.log b x) (Ordinal.log b y)
      -/
    · rw [log_of_left_le_one hb, log_of_left_le_one hb]
      /-
        🎉 no goals
      -/


theorem log_le_self (b x : Ordinal) : log b x ≤ x := by
  /-
    b x : Ordinal.{u_1}
    ⊢ LE.le (Ordinal.log b x) x
  -/
  obtain rfl | hx := eq_or_ne x 0
    /-
      case inl
      b : Ordinal.{u_1}
      ⊢ LE.le (Ordinal.log b 0) 0
    -/
  · rw [log_zero_right]
    /-
      🎉 no goals
    -/
    /-
      case inr
      b x : Ordinal.{u_1}
      hx : Ne x 0
      ⊢ LE.le (Ordinal.log b x) x
    -/
  · obtain hb | hb := lt_or_le 1 b
      /-
        case inr.inl
        b x : Ordinal.{u_1}
        hx : Ne x 0
        hb : LT.lt 1 b
        ⊢ LE.le (Ordinal.log b x) x
      -/
    · exact (right_le_opow _ hb).trans (opow_log_le_self b hx)
      /-
        🎉 no goals
      -/
      /-
        case inr.inr
        b x : Ordinal.{u_1}
        hx : Ne x 0
        hb : LE.le b 1
        ⊢ LE.le (Ordinal.log b x) x
      -/
    · simp_rw [log_of_left_le_one hb, Ordinal.zero_le]
      /-
        🎉 no goals
      -/


@[simp]
theorem log_one_right (b : Ordinal) : log b 1 = 0 := by
  /-
    b : Ordinal.{u_1}
    ⊢ Eq (Ordinal.log b 1) 0
  -/
  obtain hb | hb := lt_or_le 1 b
    /-
      case inl
      b : Ordinal.{u_1}
      hb : LT.lt 1 b
      ⊢ Eq (Ordinal.log b 1) 0
    -/
  · exact log_eq_zero hb
    /-
      🎉 no goals
    -/
    /-
      case inr
      b : Ordinal.{u_1}
      hb : LE.le b 1
      ⊢ Eq (Ordinal.log b 1) 0
    -/
  · exact log_of_left_le_one hb 1
    /-
      🎉 no goals
    -/


theorem mod_opow_log_lt_self (b : Ordinal) {o : Ordinal} (ho : o ≠ 0) : o % (b ^ log b o) < o := by
  /-
    b o : Ordinal.{u_1}
    ho : Ne o 0
    ⊢ LT.lt (HMod.hMod o (HPow.hPow b (Ordinal.log b o))) o
  -/
  rcases eq_or_ne b 0 with (rfl | hb)
    /-
      case inl
      o : Ordinal.{u_1}
      ho : Ne o 0
      ⊢ LT.lt (HMod.hMod o (HPow.hPow 0 (Ordinal.log 0 o))) o
    -/
  · simpa using Ordinal.pos_iff_ne_zero.2 ho
    /-
      🎉 no goals
    -/
    /-
      case inr
      b o : Ordinal.{u_1}
      ho : Ne o 0
      hb : Ne b 0
      ⊢ LT.lt (HMod.hMod o (HPow.hPow b (Ordinal.log b o))) o
    -/
  · exact (mod_lt _ <| opow_ne_zero _ hb).trans_le (opow_log_le_self _ ho)
    /-
      🎉 no goals
    -/


theorem log_mod_opow_log_lt_log_self {b o : Ordinal} (hb : 1 < b) (hbo : b ≤ o) :
    log b (o % (b ^ log b o)) < log b o := by
  /-
    b o : Ordinal.{u_1}
    hb : LT.lt 1 b
    hbo : LE.le b o
    ⊢ LT.lt (Ordinal.log b (HMod.hMod o (HPow.hPow b (Ordinal.log b o)))) (Ordinal …
  -/
  rcases eq_or_ne (o % (b ^ log b o)) 0 with h | h
    /-
      case inl
      b o : Ordinal.{u_1}
      hb : LT.lt 1 b
      hbo : LE.le b o
      h : Eq (HMod.hMod o (HPow.hPow b (Ordinal.log b o))) 0
      ⊢ LT.lt (Ordinal.log b (HMod.hMod o (HPow.hPow b (Ordinal.log b o)))) (Ordinal …
    -/
  · rw [h, log_zero_right]
    /-
      case inl
      b o : Ordinal.{u_1}
      hb : LT.lt 1 b
      hbo : LE.le b o
      h : Eq (HMod.hMod o (HPow.hPow b (Ordinal.log b o))) 0
      ⊢ LT.lt 0 (Ordinal.log b o)
    -/
    exact log_pos hb (one_le_iff_ne_zero.1 (hb.le.trans hbo)) hbo
    /-
      🎉 no goals
    -/
    /-
      case inr
      b o : Ordinal.{u_1}
      hb : LT.lt 1 b
      hbo : LE.le b o
      h : Ne (HMod.hMod o (HPow.hPow b (Ordinal.log b o))) 0
      ⊢ LT.lt (Ordinal.log b (HMod.hMod o (HPow.hPow b (Ordinal.log b o)))) (Ordinal …
    -/
  · rw [← succ_le_iff, succ_log_def hb h]
    /-
      case inr
      b o : Ordinal.{u_1}
      hb : LT.lt 1 b
      hbo : LE.le b o
      h : Ne (HMod.hMod o (HPow.hPow b (Ordinal.log b o))) 0
      ⊢ LE.le (InfSet.sInf (setOf fun o_1 => LT.lt (HMod.hMod o (HPow.hPow b (Ordina …
    -/
    apply csInf_le'
    /-
      case inr.h
      b o : Ordinal.{u_1}
      hb : LT.lt 1 b
      hbo : LE.le b o
      h : Ne (HMod.hMod o (HPow.hPow b (Ordinal.log b o))) 0
      ⊢ Membership.mem (setOf fun o_1 => LT.lt (HMod.hMod o (HPow.hPow b (Ordinal.lo …
    -/
    apply mod_lt
    /-
      case inr.h.h
      b o : Ordinal.{u_1}
      hb : LT.lt 1 b
      hbo : LE.le b o
      h : Ne (HMod.hMod o (HPow.hPow b (Ordinal.log b o))) 0
      ⊢ Ne (HPow.hPow b (Ordinal.log b o)) 0
    -/
    rw [← Ordinal.pos_iff_ne_zero]
    /-
      case inr.h.h
      b o : Ordinal.{u_1}
      hb : LT.lt 1 b
      hbo : LE.le b o
      h : Ne (HMod.hMod o (HPow.hPow b (Ordinal.log b o))) 0
      ⊢ LT.lt 0 (HPow.hPow b (Ordinal.log b o))
    -/
    exact opow_pos _ (zero_lt_one.trans hb)
    /-
      🎉 no goals
    -/


theorem log_eq_iff {b x : Ordinal} (hb : 1 < b) (hx : x ≠ 0) (y : Ordinal) :
    log b x = y ↔ b ^ y ≤ x ∧ x < b ^ succ y := by
  /-
    b x : Ordinal.{u_1}
    hb : LT.lt 1 b
    hx : Ne x 0
    y : Ordinal.{u_1}
    ⊢ Iff (Eq (Ordinal.log b x) y) (And (LE.le (HPow.hPow b y) x) (LT.lt x (HPow.h …
  -/
  constructor
    /-
      case mp
      b x : Ordinal.{u_1}
      hb : LT.lt 1 b
      hx : Ne x 0
      y : Ordinal.{u_1}
      ⊢ Eq (Ordinal.log b x) y → And (LE.le (HPow.hPow b y) x) (LT.lt x (HPow.hPow b …
    -/
  · rintro rfl
    /-
      case mp
      b x : Ordinal.{u_1}
      hb : LT.lt 1 b
      hx : Ne x 0
      ⊢ And (LE.le (HPow.hPow b (Ordinal.log b x)) x) (LT.lt x (HPow.hPow b (Order.s …
    -/
    use opow_log_le_self b hx, lt_opow_succ_log_self hb x
    /-
      🎉 no goals
    -/
    /-
      case mpr
      b x : Ordinal.{u_1}
      hb : LT.lt 1 b
      hx : Ne x 0
      y : Ordinal.{u_1}
      ⊢ And (LE.le (HPow.hPow b y) x) (LT.lt x (HPow.hPow b (Order.succ y))) → Eq (O …
    -/
  · rintro ⟨hx₁, hx₂⟩
    /-
      case mpr.intro
      b x : Ordinal.{u_1}
      hb : LT.lt 1 b
      hx : Ne x 0
      y : Ordinal.{u_1}
      hx₁ : LE.le (HPow.hPow b y) x
      hx₂ : LT.lt x (HPow.hPow b (Order.succ y))
      ⊢ Eq (Ordinal.log b x) y
    -/
    apply le_antisymm
      /-
        case mpr.intro.a
        b x : Ordinal.{u_1}
        hb : LT.lt 1 b
        hx : Ne x 0
        y : Ordinal.{u_1}
        hx₁ : LE.le (HPow.hPow b y) x
        hx₂ : LT.lt x (HPow.hPow b (Order.succ y))
        ⊢ LE.le (Ordinal.log b x) y
      -/
    · rwa [← lt_succ_iff, ← lt_opow_iff_log_lt hb hx]
      /-
        🎉 no goals
      -/
      /-
        case mpr.intro.a
        b x : Ordinal.{u_1}
        hb : LT.lt 1 b
        hx : Ne x 0
        y : Ordinal.{u_1}
        hx₁ : LE.le (HPow.hPow b y) x
        hx₂ : LT.lt x (HPow.hPow b (Order.succ y))
        ⊢ LE.le y (Ordinal.log b x)
      -/
    · rwa [← opow_le_iff_le_log hb hx]
      /-
        🎉 no goals
      -/


theorem log_opow_mul_add {b u v w : Ordinal} (hb : 1 < b) (hv : v ≠ 0) (hw : w < b ^ u) :
    log b (b ^ u * v + w) = u + log b v := by
  /-
    b u v w : Ordinal.{u_1}
    hb : LT.lt 1 b
    hv : Ne v 0
    hw : LT.lt w (HPow.hPow b u)
    ⊢ Eq (Ordinal.log b (HAdd.hAdd (HMul.hMul (HPow.hPow b u) v) w)) (HAdd.hAdd u  …
  -/
  rw [log_eq_iff hb]
    /-
      b u v w : Ordinal.{u_1}
      hb : LT.lt 1 b
      hv : Ne v 0
      hw : LT.lt w (HPow.hPow b u)
      ⊢ And (LE.le (HPow.hPow b (HAdd.hAdd u (Ordinal.log b v))) (HAdd.hAdd (HMul.hM …
    -/
  · constructor
      /-
        case left
        b u v w : Ordinal.{u_1}
        hb : LT.lt 1 b
        hv : Ne v 0
        hw : LT.lt w (HPow.hPow b u)
        ⊢ LE.le (HPow.hPow b (HAdd.hAdd u (Ordinal.log b v))) (HAdd.hAdd (HMul.hMul (H …
      -/
    · rw [opow_add]
      /-
        case left
        b u v w : Ordinal.{u_1}
        hb : LT.lt 1 b
        hv : Ne v 0
        hw : LT.lt w (HPow.hPow b u)
        ⊢ LE.le (HMul.hMul (HPow.hPow b u) (HPow.hPow b (Ordinal.log b v))) (HAdd.hAdd …
      -/
      exact (mul_le_mul_left' (opow_log_le_self b hv) _).trans (le_add_right _ w)
      /-
        🎉 no goals
      -/
      /-
        case right
        b u v w : Ordinal.{u_1}
        hb : LT.lt 1 b
        hv : Ne v 0
        hw : LT.lt w (HPow.hPow b u)
        ⊢ LT.lt (HAdd.hAdd (HMul.hMul (HPow.hPow b u) v) w) (HPow.hPow b (Order.succ ( …
      -/
    · apply (add_lt_add_left hw _).trans_le
      /-
        case right
        b u v w : Ordinal.{u_1}
        hb : LT.lt 1 b
        hv : Ne v 0
        hw : LT.lt w (HPow.hPow b u)
        ⊢ LE.le (HAdd.hAdd (HMul.hMul (HPow.hPow b u) v) (HPow.hPow b u)) (HPow.hPow b …
      -/
      rw [← mul_succ, ← add_succ, opow_add]
      /-
        case right
        b u v w : Ordinal.{u_1}
        hb : LT.lt 1 b
        hv : Ne v 0
        hw : LT.lt w (HPow.hPow b u)
        ⊢ LE.le (HMul.hMul (HPow.hPow b u) (Order.succ v)) (HMul.hMul (HPow.hPow b u)  …
      -/
      apply mul_le_mul_left'
      /-
        case right.bc
        b u v w : Ordinal.{u_1}
        hb : LT.lt 1 b
        hv : Ne v 0
        hw : LT.lt w (HPow.hPow b u)
        ⊢ LE.le (Order.succ v) (HPow.hPow b (Order.succ (Ordinal.log b v)))
      -/
      rw [succ_le_iff]
      /-
        case right.bc
        b u v w : Ordinal.{u_1}
        hb : LT.lt 1 b
        hv : Ne v 0
        hw : LT.lt w (HPow.hPow b u)
        ⊢ LT.lt v (HPow.hPow b (Order.succ (Ordinal.log b v)))
      -/
      exact lt_opow_succ_log_self hb _
      /-
        🎉 no goals
      -/
  · exact fun h ↦ mul_ne_zero (opow_ne_zero u (bot_lt_of_lt hb).ne') hv <|
      left_eq_zero_of_add_eq_zero h


theorem log_opow_mul {b v : Ordinal} (hb : 1 < b) (u : Ordinal) (hv : v ≠ 0) :
    log b (b ^ u * v) = u + log b v := by
  /-
    b v : Ordinal.{u_1}
    hb : LT.lt 1 b
    u : Ordinal.{u_1}
    hv : Ne v 0
    ⊢ Eq (Ordinal.log b (HMul.hMul (HPow.hPow b u) v)) (HAdd.hAdd u (Ordinal.log b …
  -/
  simpa using log_opow_mul_add hb hv (opow_pos u (bot_lt_of_lt hb))
  /-
    🎉 no goals
  -/


theorem log_opow {b : Ordinal} (hb : 1 < b) (x : Ordinal) : log b (b ^ x) = x := by
  /-
    b : Ordinal.{u_1}
    hb : LT.lt 1 b
    x : Ordinal.{u_1}
    ⊢ Eq (Ordinal.log b (HPow.hPow b x)) x
  -/
  convert log_opow_mul hb x zero_ne_one.symm using 1
    /-
      case h.e'_2
      b : Ordinal.{u_1}
      hb : LT.lt 1 b
      x : Ordinal.{u_1}
      ⊢ Eq (Ordinal.log b (HPow.hPow b x)) (Ordinal.log b (HMul.hMul (HPow.hPow b x) …
    -/
  · rw [mul_one]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_3
      b : Ordinal.{u_1}
      hb : LT.lt 1 b
      x : Ordinal.{u_1}
      ⊢ Eq x (HAdd.hAdd x (Ordinal.log b 1))
    -/
  · rw [log_one_right, add_zero]
    /-
      🎉 no goals
    -/


theorem div_opow_log_pos (b : Ordinal) {o : Ordinal} (ho : o ≠ 0) : 0 < o / (b ^ log b o) := by
  /-
    b o : Ordinal.{u_1}
    ho : Ne o 0
    ⊢ LT.lt 0 (HDiv.hDiv o (HPow.hPow b (Ordinal.log b o)))
  -/
  rcases eq_zero_or_pos b with (rfl | hb)
    /-
      case inl
      o : Ordinal.{u_1}
      ho : Ne o 0
      ⊢ LT.lt 0 (HDiv.hDiv o (HPow.hPow 0 (Ordinal.log 0 o)))
    -/
  · simpa using Ordinal.pos_iff_ne_zero.2 ho
    /-
      🎉 no goals
    -/
    /-
      case inr
      b o : Ordinal.{u_1}
      ho : Ne o 0
      hb : LT.lt 0 b
      ⊢ LT.lt 0 (HDiv.hDiv o (HPow.hPow b (Ordinal.log b o)))
    -/
  · rw [div_pos (opow_ne_zero _ hb.ne')]
    /-
      case inr
      b o : Ordinal.{u_1}
      ho : Ne o 0
      hb : LT.lt 0 b
      ⊢ LE.le (HPow.hPow b (Ordinal.log b o)) o
    -/
    exact opow_log_le_self b ho
    /-
      🎉 no goals
    -/


theorem div_opow_log_lt {b : Ordinal} (o : Ordinal) (hb : 1 < b) : o / (b ^ log b o) < b := by
  /-
    b o : Ordinal.{u_1}
    hb : LT.lt 1 b
    ⊢ LT.lt (HDiv.hDiv o (HPow.hPow b (Ordinal.log b o))) b
  -/
  rw [div_lt (opow_pos _ (zero_lt_one.trans hb)).ne', ← opow_succ]
  /-
    b o : Ordinal.{u_1}
    hb : LT.lt 1 b
    ⊢ LT.lt o (HPow.hPow b (Order.succ (Ordinal.log b o)))
  -/
  exact lt_opow_succ_log_self hb o
  /-
    🎉 no goals
  -/


theorem add_log_le_log_mul {x y : Ordinal} (b : Ordinal) (hx : x ≠ 0) (hy : y ≠ 0) :
    log b x + log b y ≤ log b (x * y) := by
  /-
    x y b : Ordinal.{u_1}
    hx : Ne x 0
    hy : Ne y 0
    ⊢ LE.le (HAdd.hAdd (Ordinal.log b x) (Ordinal.log b y)) (Ordinal.log b (HMul.h …
  -/
  obtain hb | hb := lt_or_le 1 b
    /-
      case inl
      x y b : Ordinal.{u_1}
      hx : Ne x 0
      hy : Ne y 0
      hb : LT.lt 1 b
      ⊢ LE.le (HAdd.hAdd (Ordinal.log b x) (Ordinal.log b y)) (Ordinal.log b (HMul.h …
    -/
  · rw [← opow_le_iff_le_log hb (mul_ne_zero hx hy), opow_add]
    /-
      case inl
      x y b : Ordinal.{u_1}
      hx : Ne x 0
      hy : Ne y 0
      hb : LT.lt 1 b
      ⊢ LE.le (HMul.hMul (HPow.hPow b (Ordinal.log b x)) (HPow.hPow b (Ordinal.log b …
    -/
    exact mul_le_mul' (opow_log_le_self b hx) (opow_log_le_self b hy)
    /-
      🎉 no goals
    -/
    /-
      case inr
      x y b : Ordinal.{u_1}
      hx : Ne x 0
      hy : Ne y 0
      hb : LE.le b 1
      ⊢ LE.le (HAdd.hAdd (Ordinal.log b x) (Ordinal.log b y)) (Ordinal.log b (HMul.h …
    -/
  · simpa only [log_of_left_le_one hb, zero_add] using le_rfl
    /-
      🎉 no goals
    -/


theorem omega0_opow_mul_nat_lt {a b : Ordinal} (h : a < b) (n : ℕ) : ω ^ a * n < ω ^ b := by
  /-
    a b : Ordinal.{u_1}
    h : LT.lt a b
    n : Nat
    ⊢ LT.lt (HMul.hMul (HPow.hPow Ordinal.omega0 a) ↑n) (HPow.hPow Ordinal.omega0 b)
  -/
  apply lt_of_lt_of_le _ (opow_le_opow_right omega0_pos (succ_le_of_lt h))
  /-
    a b : Ordinal.{u_1}
    h : LT.lt a b
    n : Nat
    ⊢ LT.lt (HMul.hMul (HPow.hPow Ordinal.omega0 a) ↑n) (HPow.hPow Ordinal.omega0  …
  -/
  rw [opow_succ]
  /-
    a b : Ordinal.{u_1}
    h : LT.lt a b
    n : Nat
    ⊢ LT.lt (HMul.hMul (HPow.hPow Ordinal.omega0 a) ↑n) (HMul.hMul (HPow.hPow Ordi …
  -/
  exact mul_lt_mul_of_pos_left (nat_lt_omega0 n) (opow_pos a omega0_pos)
  /-
    🎉 no goals
  -/


theorem lt_omega0_opow {a b : Ordinal} (hb : b ≠ 0) :
    a < ω ^ b ↔ ∃ c < b, ∃ n : ℕ, a < ω ^ c * n := by
  refine ⟨fun ha ↦ ⟨_, lt_log_of_lt_opow hb ha, ?_⟩,
    fun ⟨c, hc, n, hn⟩ ↦ hn.trans (omega0_opow_mul_nat_lt hc n)⟩
  /-
    a b : Ordinal.{u_1}
    hb : Ne b 0
    ha : LT.lt a (HPow.hPow Ordinal.omega0 b)
    ⊢ Exists fun n => LT.lt a (HMul.hMul (HPow.hPow Ordinal.omega0 (Ordinal.log Or …
  -/
  obtain ⟨n, hn⟩ := lt_omega0.1 (div_opow_log_lt a one_lt_omega0)
  /-
    case intro
    a b : Ordinal.{u_1}
    hb : Ne b 0
    ha : LT.lt a (HPow.hPow Ordinal.omega0 b)
    n : Nat
    hn : Eq (HDiv.hDiv a (HPow.hPow Ordinal.omega0 (Ordinal.log Ordinal.omega0 a)) …
    ⊢ Exists fun n => LT.lt a (HMul.hMul (HPow.hPow Ordinal.omega0 (Ordinal.log Or …
  -/
  use n.succ
  /-
    case h
    a b : Ordinal.{u_1}
    hb : Ne b 0
    ha : LT.lt a (HPow.hPow Ordinal.omega0 b)
    n : Nat
    hn : Eq (HDiv.hDiv a (HPow.hPow Ordinal.omega0 (Ordinal.log Ordinal.omega0 a)) …
    ⊢ LT.lt a (HMul.hMul (HPow.hPow Ordinal.omega0 (Ordinal.log Ordinal.omega0 a)) …
  -/
  rw [natCast_succ, ← hn]
  /-
    case h
    a b : Ordinal.{u_1}
    hb : Ne b 0
    ha : LT.lt a (HPow.hPow Ordinal.omega0 b)
    n : Nat
    hn : Eq (HDiv.hDiv a (HPow.hPow Ordinal.omega0 (Ordinal.log Ordinal.omega0 a)) …
    ⊢ LT.lt a (HMul.hMul (HPow.hPow Ordinal.omega0 (Ordinal.log Ordinal.omega0 a)) …
  -/
  exact lt_mul_succ_div a (opow_ne_zero _ omega0_ne_zero)
  /-
    🎉 no goals
  -/


theorem lt_omega0_opow_succ {a b : Ordinal} : a < ω ^ succ b ↔ ∃ n : ℕ, a < ω ^ b * n := by
  /-
    a b : Ordinal.{u_1}
    ⊢ Iff (LT.lt a (HPow.hPow Ordinal.omega0 (Order.succ b))) (Exists fun n => LT. …
  -/
  refine ⟨fun ha ↦ ?_, fun ⟨n, hn⟩ ↦ hn.trans (omega0_opow_mul_nat_lt (lt_succ b) n)⟩
  /-
    a b : Ordinal.{u_1}
    ha : LT.lt a (HPow.hPow Ordinal.omega0 (Order.succ b))
    ⊢ Exists fun n => LT.lt a (HMul.hMul (HPow.hPow Ordinal.omega0 b) ↑n)
  -/
  obtain ⟨c, hc, n, hn⟩ := (lt_omega0_opow (succ_ne_zero b)).1 ha
  /-
    case intro.intro.intro
    a b : Ordinal.{u_1}
    ha : LT.lt a (HPow.hPow Ordinal.omega0 (Order.succ b))
    c : Ordinal.{u_1}
    hc : LT.lt c (Order.succ b)
    n : Nat
    hn : LT.lt a (HMul.hMul (HPow.hPow Ordinal.omega0 c) ↑n)
    ⊢ Exists fun n => LT.lt a (HMul.hMul (HPow.hPow Ordinal.omega0 b) ↑n)
  -/
  refine ⟨n, hn.trans_le (mul_le_mul_right' ?_ _)⟩
  /-
    case intro.intro.intro
    a b : Ordinal.{u_1}
    ha : LT.lt a (HPow.hPow Ordinal.omega0 (Order.succ b))
    c : Ordinal.{u_1}
    hc : LT.lt c (Order.succ b)
    n : Nat
    hn : LT.lt a (HMul.hMul (HPow.hPow Ordinal.omega0 c) ↑n)
    ⊢ LE.le (HPow.hPow Ordinal.omega0 c) (HPow.hPow Ordinal.omega0 b)
  -/
  rwa [opow_le_opow_iff_right one_lt_omega0, ← lt_succ_iff]
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem natCast_opow (m : ℕ) : ∀ n : ℕ, ↑(m ^ n : ℕ) = (m : Ordinal) ^ (n : Ordinal)
            /-
              m : Nat
              ⊢ Eq (↑(HPow.hPow m 0)) (HPow.hPow ↑m ↑0)
            -/
  | 0 => by simp
            /-
              🎉 no goals
            -/
  | n + 1 => by
    /-
      m n : Nat
      ⊢ Eq (↑(HPow.hPow m (HAdd.hAdd n 1))) (HPow.hPow ↑m ↑(HAdd.hAdd n 1))
    -/
    rw [pow_succ, natCast_mul, natCast_opow m n, Nat.cast_succ, add_one_eq_succ, opow_succ]
    /-
      🎉 no goals
    -/


theorem iSup_pow {o : Ordinal} (ho : 0 < o) : ⨆ n : ℕ, o ^ n = o ^ ω := by
  /-
    o : Ordinal.{u_1}
    ho : LT.lt 0 o
    ⊢ Eq (iSup fun n => HPow.hPow o n) (HPow.hPow o Ordinal.omega0)
  -/
  simp_rw [← opow_natCast]
  /-
    o : Ordinal.{u_1}
    ho : LT.lt 0 o
    ⊢ Eq (iSup fun n => HPow.hPow o ↑n) (HPow.hPow o Ordinal.omega0)
  -/
  rcases (one_le_iff_pos.2 ho).lt_or_eq with ho₁ | rfl
    /-
      case inl
      o : Ordinal.{u_1}
      ho : LT.lt 0 o
      ho₁ : LT.lt 1 o
      ⊢ Eq (iSup fun n => HPow.hPow o ↑n) (HPow.hPow o Ordinal.omega0)
    -/
  · exact (isNormal_opow ho₁).apply_omega0
    /-
      🎉 no goals
    -/
    /-
      case inr
      ho : LT.lt 0 1
      ⊢ Eq (iSup fun n => HPow.hPow 1 ↑n) (HPow.hPow 1 Ordinal.omega0)
    -/
  · rw [one_opow]
    /-
      case inr
      ho : LT.lt 0 1
      ⊢ Eq (iSup fun n => HPow.hPow 1 ↑n) 1
    -/
    refine le_antisymm (Ordinal.iSup_le fun n => by rw [one_opow]) ?_
    /-
      case inr
      ho : LT.lt 0 1
      ⊢ LE.le 1 (iSup fun n => HPow.hPow 1 ↑n)
    -/
    exact_mod_cast Ordinal.le_iSup _ 0
    /-
      🎉 no goals
    -/


set_option linter.deprecated false in
@[deprecated iSup_pow (since := "2024-08-27")]
theorem sup_opow_nat {o : Ordinal} (ho : 0 < o) : (sup fun n : ℕ => o ^ n) = o ^ ω := by
  /-
    o : Ordinal.{u_1}
    ho : LT.lt 0 o
    ⊢ Eq (Ordinal.sup fun n => HPow.hPow o n) (HPow.hPow o Ordinal.omega0)
  -/
  simp_rw [← opow_natCast]
  /-
    o : Ordinal.{u_1}
    ho : LT.lt 0 o
    ⊢ Eq (Ordinal.sup fun n => HPow.hPow o ↑n) (HPow.hPow o Ordinal.omega0)
  -/
  rcases (one_le_iff_pos.2 ho).lt_or_eq with ho₁ | rfl
    /-
      case inl
      o : Ordinal.{u_1}
      ho : LT.lt 0 o
      ho₁ : LT.lt 1 o
      ⊢ Eq (Ordinal.sup fun n => HPow.hPow o ↑n) (HPow.hPow o Ordinal.omega0)
    -/
  · exact (isNormal_opow ho₁).apply_omega0
    /-
      🎉 no goals
    -/
    /-
      case inr
      ho : LT.lt 0 1
      ⊢ Eq (Ordinal.sup fun n => HPow.hPow 1 ↑n) (HPow.hPow 1 Ordinal.omega0)
    -/
  · rw [one_opow]
    /-
      case inr
      ho : LT.lt 0 1
      ⊢ Eq (Ordinal.sup fun n => HPow.hPow 1 ↑n) 1
    -/
    refine le_antisymm (sup_le fun n => by rw [one_opow]) ?_
    /-
      case inr
      ho : LT.lt 0 1
      ⊢ LE.le 1 (Ordinal.sup fun n => HPow.hPow 1 ↑n)
    -/
    convert le_sup (fun n : ℕ => 1 ^ (n : Ordinal)) 0
    /-
      case h.e'_3
      ho : LT.lt 0 1
      ⊢ Eq 1 (HPow.hPow 1 ↑0)
    -/
    rw [Nat.cast_zero, opow_zero]
    /-
      🎉 no goals
    -/


