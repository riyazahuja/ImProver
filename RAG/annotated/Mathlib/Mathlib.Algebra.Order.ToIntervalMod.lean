/--
The unique integer such that this multiple of `p`, subtracted from `b`, is in `Ico a (a + p)`. -/
def toIcoDiv (a b : α) : ℤ :=
  (existsUnique_sub_zsmul_mem_Ico hp b a).choose


theorem sub_toIcoDiv_zsmul_mem_Ico (a b : α) : b - toIcoDiv hp a b • p ∈ Set.Ico a (a + p) :=
  (existsUnique_sub_zsmul_mem_Ico hp b a).choose_spec.1


theorem toIcoDiv_eq_of_sub_zsmul_mem_Ico (h : b - n • p ∈ Set.Ico a (a + p)) :
    toIcoDiv hp a b = n :=
  ((existsUnique_sub_zsmul_mem_Ico hp b a).choose_spec.2 _ h).symm


/--
The unique integer such that this multiple of `p`, subtracted from `b`, is in `Ioc a (a + p)`. -/
def toIocDiv (a b : α) : ℤ :=
  (existsUnique_sub_zsmul_mem_Ioc hp b a).choose


theorem sub_toIocDiv_zsmul_mem_Ioc (a b : α) : b - toIocDiv hp a b • p ∈ Set.Ioc a (a + p) :=
  (existsUnique_sub_zsmul_mem_Ioc hp b a).choose_spec.1


theorem toIocDiv_eq_of_sub_zsmul_mem_Ioc (h : b - n • p ∈ Set.Ioc a (a + p)) :
    toIocDiv hp a b = n :=
  ((existsUnique_sub_zsmul_mem_Ioc hp b a).choose_spec.2 _ h).symm


/-- Reduce `b` to the interval `Ico a (a + p)`. -/
def toIcoMod (a b : α) : α :=
  b - toIcoDiv hp a b • p


/-- Reduce `b` to the interval `Ioc a (a + p)`. -/
def toIocMod (a b : α) : α :=
  b - toIocDiv hp a b • p


theorem toIcoMod_mem_Ico (a b : α) : toIcoMod hp a b ∈ Set.Ico a (a + p) :=
  sub_toIcoDiv_zsmul_mem_Ico hp a b


theorem toIcoMod_mem_Ico' (b : α) : toIcoMod hp 0 b ∈ Set.Ico 0 p := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    b : α
    ⊢ Membership.mem (Set.Ico 0 p) (toIcoMod hp 0 b)
  -/
  convert toIcoMod_mem_Ico hp 0 b
  /-
    case h.e'_4.h.e'_4
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    b : α
    ⊢ Eq p (HAdd.hAdd 0 p)
  -/
  exact (zero_add p).symm
  /-
    🎉 no goals
  -/


theorem toIocMod_mem_Ioc (a b : α) : toIocMod hp a b ∈ Set.Ioc a (a + p) :=
  sub_toIocDiv_zsmul_mem_Ioc hp a b


theorem left_le_toIcoMod (a b : α) : a ≤ toIcoMod hp a b :=
  (Set.mem_Ico.1 (toIcoMod_mem_Ico hp a b)).1


theorem left_lt_toIocMod (a b : α) : a < toIocMod hp a b :=
  (Set.mem_Ioc.1 (toIocMod_mem_Ioc hp a b)).1


theorem toIcoMod_lt_right (a b : α) : toIcoMod hp a b < a + p :=
  (Set.mem_Ico.1 (toIcoMod_mem_Ico hp a b)).2


theorem toIocMod_le_right (a b : α) : toIocMod hp a b ≤ a + p :=
  (Set.mem_Ioc.1 (toIocMod_mem_Ioc hp a b)).2


@[simp]
theorem self_sub_toIcoDiv_zsmul (a b : α) : b - toIcoDiv hp a b • p = toIcoMod hp a b :=
  rfl


@[simp]
theorem self_sub_toIocDiv_zsmul (a b : α) : b - toIocDiv hp a b • p = toIocMod hp a b :=
  rfl


@[simp]
theorem toIcoDiv_zsmul_sub_self (a b : α) : toIcoDiv hp a b • p - b = -toIcoMod hp a b := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (HSub.hSub (HSMul.hSMul (toIcoDiv hp a b) p) b) (Neg.neg (toIcoMod hp a b))
  -/
  rw [toIcoMod, neg_sub]
  /-
    🎉 no goals
  -/


@[simp]
theorem toIocDiv_zsmul_sub_self (a b : α) : toIocDiv hp a b • p - b = -toIocMod hp a b := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (HSub.hSub (HSMul.hSMul (toIocDiv hp a b) p) b) (Neg.neg (toIocMod hp a b))
  -/
  rw [toIocMod, neg_sub]
  /-
    🎉 no goals
  -/


@[simp]
theorem toIcoMod_sub_self (a b : α) : toIcoMod hp a b - b = -toIcoDiv hp a b • p := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (HSub.hSub (toIcoMod hp a b) b) (HSMul.hSMul (Neg.neg (toIcoDiv hp a b)) p)
  -/
  rw [toIcoMod, sub_sub_cancel_left, neg_smul]
  /-
    🎉 no goals
  -/


@[simp]
theorem toIocMod_sub_self (a b : α) : toIocMod hp a b - b = -toIocDiv hp a b • p := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (HSub.hSub (toIocMod hp a b) b) (HSMul.hSMul (Neg.neg (toIocDiv hp a b)) p)
  -/
  rw [toIocMod, sub_sub_cancel_left, neg_smul]
  /-
    🎉 no goals
  -/


@[simp]
theorem self_sub_toIcoMod (a b : α) : b - toIcoMod hp a b = toIcoDiv hp a b • p := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (HSub.hSub b (toIcoMod hp a b)) (HSMul.hSMul (toIcoDiv hp a b) p)
  -/
  rw [toIcoMod, sub_sub_cancel]
  /-
    🎉 no goals
  -/


@[simp]
theorem self_sub_toIocMod (a b : α) : b - toIocMod hp a b = toIocDiv hp a b • p := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (HSub.hSub b (toIocMod hp a b)) (HSMul.hSMul (toIocDiv hp a b) p)
  -/
  rw [toIocMod, sub_sub_cancel]
  /-
    🎉 no goals
  -/


@[simp]
theorem toIcoMod_add_toIcoDiv_zsmul (a b : α) : toIcoMod hp a b + toIcoDiv hp a b • p = b := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (HAdd.hAdd (toIcoMod hp a b) (HSMul.hSMul (toIcoDiv hp a b) p)) b
  -/
  rw [toIcoMod, sub_add_cancel]
  /-
    🎉 no goals
  -/


@[simp]
theorem toIocMod_add_toIocDiv_zsmul (a b : α) : toIocMod hp a b + toIocDiv hp a b • p = b := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (HAdd.hAdd (toIocMod hp a b) (HSMul.hSMul (toIocDiv hp a b) p)) b
  -/
  rw [toIocMod, sub_add_cancel]
  /-
    🎉 no goals
  -/


@[simp]
theorem toIcoDiv_zsmul_sub_toIcoMod (a b : α) : toIcoDiv hp a b • p + toIcoMod hp a b = b := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (HAdd.hAdd (HSMul.hSMul (toIcoDiv hp a b) p) (toIcoMod hp a b)) b
  -/
  rw [add_comm, toIcoMod_add_toIcoDiv_zsmul]
  /-
    🎉 no goals
  -/


@[simp]
theorem toIocDiv_zsmul_sub_toIocMod (a b : α) : toIocDiv hp a b • p + toIocMod hp a b = b := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (HAdd.hAdd (HSMul.hSMul (toIocDiv hp a b) p) (toIocMod hp a b)) b
  -/
  rw [add_comm, toIocMod_add_toIocDiv_zsmul]
  /-
    🎉 no goals
  -/


theorem toIcoMod_eq_iff : toIcoMod hp a b = c ↔ c ∈ Set.Ico a (a + p) ∧ ∃ z : ℤ, b = c + z • p := by
  refine
    ⟨fun h =>
      ⟨h ▸ toIcoMod_mem_Ico hp a b, toIcoDiv hp a b, h ▸ (toIcoMod_add_toIcoDiv_zsmul _ _ _).symm⟩,
      ?_⟩
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b c : α
    ⊢ And (Membership.mem (Set.Ico a (HAdd.hAdd a p)) c) (Exists fun z => Eq b (HA …
  -/
  simp_rw [← @sub_eq_iff_eq_add]
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b c : α
    ⊢ And (Membership.mem (Set.Ico a (HAdd.hAdd a p)) c) (Exists fun z => Eq (HSub …
  -/
  rintro ⟨hc, n, rfl⟩
  /-
    case intro.intro
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    n : Int
    hc : Membership.mem (Set.Ico a (HAdd.hAdd a p)) (HSub.hSub b (HSMul.hSMul n p))
    ⊢ Eq (toIcoMod hp a b) (HSub.hSub b (HSMul.hSMul n p))
  -/
  rw [← toIcoDiv_eq_of_sub_zsmul_mem_Ico hp hc, toIcoMod]
  /-
    🎉 no goals
  -/


theorem toIocMod_eq_iff : toIocMod hp a b = c ↔ c ∈ Set.Ioc a (a + p) ∧ ∃ z : ℤ, b = c + z • p := by
  refine
    ⟨fun h =>
      ⟨h ▸ toIocMod_mem_Ioc hp a b, toIocDiv hp a b, h ▸ (toIocMod_add_toIocDiv_zsmul hp _ _).symm⟩,
      ?_⟩
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b c : α
    ⊢ And (Membership.mem (Set.Ioc a (HAdd.hAdd a p)) c) (Exists fun z => Eq b (HA …
  -/
  simp_rw [← @sub_eq_iff_eq_add]
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b c : α
    ⊢ And (Membership.mem (Set.Ioc a (HAdd.hAdd a p)) c) (Exists fun z => Eq (HSub …
  -/
  rintro ⟨hc, n, rfl⟩
  /-
    case intro.intro
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    n : Int
    hc : Membership.mem (Set.Ioc a (HAdd.hAdd a p)) (HSub.hSub b (HSMul.hSMul n p))
    ⊢ Eq (toIocMod hp a b) (HSub.hSub b (HSMul.hSMul n p))
  -/
  rw [← toIocDiv_eq_of_sub_zsmul_mem_Ioc hp hc, toIocMod]
  /-
    🎉 no goals
  -/


@[simp]
theorem toIcoDiv_apply_left (a : α) : toIcoDiv hp a a = 0 :=
                                            /-
                                              α : Type u_1
                                              inst✝ : LinearOrderedAddCommGroup α
                                              hα : Archimedean α
                                              p : α
                                              hp : LT.lt 0 p
                                              a : α
                                              ⊢ Membership.mem (Set.Ico a (HAdd.hAdd a p)) (HSub.hSub a (HSMul.hSMul 0 p))
                                            -/
  toIcoDiv_eq_of_sub_zsmul_mem_Ico hp <| by simp [hp]
                                            /-
                                              🎉 no goals
                                            -/


@[simp]
theorem toIocDiv_apply_left (a : α) : toIocDiv hp a a = -1 :=
                                            /-
                                              α : Type u_1
                                              inst✝ : LinearOrderedAddCommGroup α
                                              hα : Archimedean α
                                              p : α
                                              hp : LT.lt 0 p
                                              a : α
                                              ⊢ Membership.mem (Set.Ioc a (HAdd.hAdd a p)) (HSub.hSub a (HSMul.hSMul (-1) p))
                                            -/
  toIocDiv_eq_of_sub_zsmul_mem_Ioc hp <| by simp [hp]
                                            /-
                                              🎉 no goals
                                            -/


@[simp]
theorem toIcoMod_apply_left (a : α) : toIcoMod hp a a = a := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a : α
    ⊢ Eq (toIcoMod hp a a) a
  -/
  rw [toIcoMod_eq_iff hp, Set.left_mem_Ico]
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a : α
    ⊢ And (LT.lt a (HAdd.hAdd a p)) (Exists fun z => Eq a (HAdd.hAdd a (HSMul.hSMu …
  -/
  exact ⟨lt_add_of_pos_right _ hp, 0, by simp⟩
  /-
    🎉 no goals
  -/


@[simp]
theorem toIocMod_apply_left (a : α) : toIocMod hp a a = a + p := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a : α
    ⊢ Eq (toIocMod hp a a) (HAdd.hAdd a p)
  -/
  rw [toIocMod_eq_iff hp, Set.right_mem_Ioc]
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a : α
    ⊢ And (LT.lt a (HAdd.hAdd a p)) (Exists fun z => Eq a (HAdd.hAdd (HAdd.hAdd a  …
  -/
  exact ⟨lt_add_of_pos_right _ hp, -1, by simp⟩
  /-
    🎉 no goals
  -/


theorem toIcoDiv_apply_right (a : α) : toIcoDiv hp a (a + p) = 1 :=
                                            /-
                                              α : Type u_1
                                              inst✝ : LinearOrderedAddCommGroup α
                                              hα : Archimedean α
                                              p : α
                                              hp : LT.lt 0 p
                                              a : α
                                              ⊢ Membership.mem (Set.Ico a (HAdd.hAdd a p)) (HSub.hSub (HAdd.hAdd a p) (HSMul …
                                            -/
  toIcoDiv_eq_of_sub_zsmul_mem_Ico hp <| by simp [hp]
                                            /-
                                              🎉 no goals
                                            -/


theorem toIocDiv_apply_right (a : α) : toIocDiv hp a (a + p) = 0 :=
                                            /-
                                              α : Type u_1
                                              inst✝ : LinearOrderedAddCommGroup α
                                              hα : Archimedean α
                                              p : α
                                              hp : LT.lt 0 p
                                              a : α
                                              ⊢ Membership.mem (Set.Ioc a (HAdd.hAdd a p)) (HSub.hSub (HAdd.hAdd a p) (HSMul …
                                            -/
  toIocDiv_eq_of_sub_zsmul_mem_Ioc hp <| by simp [hp]
                                            /-
                                              🎉 no goals
                                            -/


theorem toIcoMod_apply_right (a : α) : toIcoMod hp a (a + p) = a := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a : α
    ⊢ Eq (toIcoMod hp a (HAdd.hAdd a p)) a
  -/
  rw [toIcoMod_eq_iff hp, Set.left_mem_Ico]
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a : α
    ⊢ And (LT.lt a (HAdd.hAdd a p)) (Exists fun z => Eq (HAdd.hAdd a p) (HAdd.hAdd …
  -/
  exact ⟨lt_add_of_pos_right _ hp, 1, by simp⟩
  /-
    🎉 no goals
  -/


theorem toIocMod_apply_right (a : α) : toIocMod hp a (a + p) = a + p := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a : α
    ⊢ Eq (toIocMod hp a (HAdd.hAdd a p)) (HAdd.hAdd a p)
  -/
  rw [toIocMod_eq_iff hp, Set.right_mem_Ioc]
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a : α
    ⊢ And (LT.lt a (HAdd.hAdd a p)) (Exists fun z => Eq (HAdd.hAdd a p) (HAdd.hAdd …
  -/
  exact ⟨lt_add_of_pos_right _ hp, 0, by simp⟩
  /-
    🎉 no goals
  -/


@[simp]
theorem toIcoDiv_add_zsmul (a b : α) (m : ℤ) : toIcoDiv hp a (b + m • p) = toIcoDiv hp a b + m :=
  toIcoDiv_eq_of_sub_zsmul_mem_Ico hp <| by
    /-
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroup α
      hα : Archimedean α
      p : α
      hp : LT.lt 0 p
      a b : α
      m : Int
      ⊢ Membership.mem (Set.Ico a (HAdd.hAdd a p)) (HSub.hSub (HAdd.hAdd b (HSMul.hS …
    -/
    simpa only [add_smul, add_sub_add_right_eq_sub] using sub_toIcoDiv_zsmul_mem_Ico hp a b
    /-
      🎉 no goals
    -/


@[simp]
theorem toIcoDiv_add_zsmul' (a b : α) (m : ℤ) :
    toIcoDiv hp (a + m • p) b = toIcoDiv hp a b - m := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    m : Int
    ⊢ Eq (toIcoDiv hp (HAdd.hAdd a (HSMul.hSMul m p)) b) (HSub.hSub (toIcoDiv hp a …
  -/
  refine toIcoDiv_eq_of_sub_zsmul_mem_Ico _ ?_
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    m : Int
    ⊢ Membership.mem (Set.Ico (HAdd.hAdd a (HSMul.hSMul m p)) (HAdd.hAdd (HAdd.hAd …
  -/
  rw [sub_smul, ← sub_add, add_right_comm]
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    m : Int
    ⊢ Membership.mem (Set.Ico (HAdd.hAdd a (HSMul.hSMul m p)) (HAdd.hAdd (HAdd.hAd …
  -/
  simpa using sub_toIcoDiv_zsmul_mem_Ico hp a b
  /-
    🎉 no goals
  -/


@[simp]
theorem toIocDiv_add_zsmul (a b : α) (m : ℤ) : toIocDiv hp a (b + m • p) = toIocDiv hp a b + m :=
  toIocDiv_eq_of_sub_zsmul_mem_Ioc hp <| by
    /-
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroup α
      hα : Archimedean α
      p : α
      hp : LT.lt 0 p
      a b : α
      m : Int
      ⊢ Membership.mem (Set.Ioc a (HAdd.hAdd a p)) (HSub.hSub (HAdd.hAdd b (HSMul.hS …
    -/
    simpa only [add_smul, add_sub_add_right_eq_sub] using sub_toIocDiv_zsmul_mem_Ioc hp a b
    /-
      🎉 no goals
    -/


@[simp]
theorem toIocDiv_add_zsmul' (a b : α) (m : ℤ) :
    toIocDiv hp (a + m • p) b = toIocDiv hp a b - m := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    m : Int
    ⊢ Eq (toIocDiv hp (HAdd.hAdd a (HSMul.hSMul m p)) b) (HSub.hSub (toIocDiv hp a …
  -/
  refine toIocDiv_eq_of_sub_zsmul_mem_Ioc _ ?_
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    m : Int
    ⊢ Membership.mem (Set.Ioc (HAdd.hAdd a (HSMul.hSMul m p)) (HAdd.hAdd (HAdd.hAd …
  -/
  rw [sub_smul, ← sub_add, add_right_comm]
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    m : Int
    ⊢ Membership.mem (Set.Ioc (HAdd.hAdd a (HSMul.hSMul m p)) (HAdd.hAdd (HAdd.hAd …
  -/
  simpa using sub_toIocDiv_zsmul_mem_Ioc hp a b
  /-
    🎉 no goals
  -/


@[simp]
theorem toIcoDiv_zsmul_add (a b : α) (m : ℤ) : toIcoDiv hp a (m • p + b) = m + toIcoDiv hp a b := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    m : Int
    ⊢ Eq (toIcoDiv hp a (HAdd.hAdd (HSMul.hSMul m p) b)) (HAdd.hAdd m (toIcoDiv hp …
  -/
  rw [add_comm, toIcoDiv_add_zsmul, add_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem toIocDiv_zsmul_add (a b : α) (m : ℤ) : toIocDiv hp a (m • p + b) = m + toIocDiv hp a b := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    m : Int
    ⊢ Eq (toIocDiv hp a (HAdd.hAdd (HSMul.hSMul m p) b)) (HAdd.hAdd m (toIocDiv hp …
  -/
  rw [add_comm, toIocDiv_add_zsmul, add_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem toIcoDiv_sub_zsmul (a b : α) (m : ℤ) : toIcoDiv hp a (b - m • p) = toIcoDiv hp a b - m := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    m : Int
    ⊢ Eq (toIcoDiv hp a (HSub.hSub b (HSMul.hSMul m p))) (HSub.hSub (toIcoDiv hp a …
  -/
  rw [sub_eq_add_neg, ← neg_smul, toIcoDiv_add_zsmul, sub_eq_add_neg]
  /-
    🎉 no goals
  -/


@[simp]
theorem toIcoDiv_sub_zsmul' (a b : α) (m : ℤ) :
    toIcoDiv hp (a - m • p) b = toIcoDiv hp a b + m := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    m : Int
    ⊢ Eq (toIcoDiv hp (HSub.hSub a (HSMul.hSMul m p)) b) (HAdd.hAdd (toIcoDiv hp a …
  -/
  rw [sub_eq_add_neg, ← neg_smul, toIcoDiv_add_zsmul', sub_neg_eq_add]
  /-
    🎉 no goals
  -/


@[simp]
theorem toIocDiv_sub_zsmul (a b : α) (m : ℤ) : toIocDiv hp a (b - m • p) = toIocDiv hp a b - m := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    m : Int
    ⊢ Eq (toIocDiv hp a (HSub.hSub b (HSMul.hSMul m p))) (HSub.hSub (toIocDiv hp a …
  -/
  rw [sub_eq_add_neg, ← neg_smul, toIocDiv_add_zsmul, sub_eq_add_neg]
  /-
    🎉 no goals
  -/


@[simp]
theorem toIocDiv_sub_zsmul' (a b : α) (m : ℤ) :
    toIocDiv hp (a - m • p) b = toIocDiv hp a b + m := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    m : Int
    ⊢ Eq (toIocDiv hp (HSub.hSub a (HSMul.hSMul m p)) b) (HAdd.hAdd (toIocDiv hp a …
  -/
  rw [sub_eq_add_neg, ← neg_smul, toIocDiv_add_zsmul', sub_neg_eq_add]
  /-
    🎉 no goals
  -/


@[simp]
theorem toIcoDiv_add_right (a b : α) : toIcoDiv hp a (b + p) = toIcoDiv hp a b + 1 := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (toIcoDiv hp a (HAdd.hAdd b p)) (HAdd.hAdd (toIcoDiv hp a b) 1)
  -/
  simpa only [one_zsmul] using toIcoDiv_add_zsmul hp a b 1
  /-
    🎉 no goals
  -/


@[simp]
theorem toIcoDiv_add_right' (a b : α) : toIcoDiv hp (a + p) b = toIcoDiv hp a b - 1 := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (toIcoDiv hp (HAdd.hAdd a p) b) (HSub.hSub (toIcoDiv hp a b) 1)
  -/
  simpa only [one_zsmul] using toIcoDiv_add_zsmul' hp a b 1
  /-
    🎉 no goals
  -/


@[simp]
theorem toIocDiv_add_right (a b : α) : toIocDiv hp a (b + p) = toIocDiv hp a b + 1 := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (toIocDiv hp a (HAdd.hAdd b p)) (HAdd.hAdd (toIocDiv hp a b) 1)
  -/
  simpa only [one_zsmul] using toIocDiv_add_zsmul hp a b 1
  /-
    🎉 no goals
  -/


@[simp]
theorem toIocDiv_add_right' (a b : α) : toIocDiv hp (a + p) b = toIocDiv hp a b - 1 := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (toIocDiv hp (HAdd.hAdd a p) b) (HSub.hSub (toIocDiv hp a b) 1)
  -/
  simpa only [one_zsmul] using toIocDiv_add_zsmul' hp a b 1
  /-
    🎉 no goals
  -/


@[simp]
theorem toIcoDiv_add_left (a b : α) : toIcoDiv hp a (p + b) = toIcoDiv hp a b + 1 := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (toIcoDiv hp a (HAdd.hAdd p b)) (HAdd.hAdd (toIcoDiv hp a b) 1)
  -/
  rw [add_comm, toIcoDiv_add_right]
  /-
    🎉 no goals
  -/


@[simp]
theorem toIcoDiv_add_left' (a b : α) : toIcoDiv hp (p + a) b = toIcoDiv hp a b - 1 := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (toIcoDiv hp (HAdd.hAdd p a) b) (HSub.hSub (toIcoDiv hp a b) 1)
  -/
  rw [add_comm, toIcoDiv_add_right']
  /-
    🎉 no goals
  -/


@[simp]
theorem toIocDiv_add_left (a b : α) : toIocDiv hp a (p + b) = toIocDiv hp a b + 1 := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (toIocDiv hp a (HAdd.hAdd p b)) (HAdd.hAdd (toIocDiv hp a b) 1)
  -/
  rw [add_comm, toIocDiv_add_right]
  /-
    🎉 no goals
  -/


@[simp]
theorem toIocDiv_add_left' (a b : α) : toIocDiv hp (p + a) b = toIocDiv hp a b - 1 := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (toIocDiv hp (HAdd.hAdd p a) b) (HSub.hSub (toIocDiv hp a b) 1)
  -/
  rw [add_comm, toIocDiv_add_right']
  /-
    🎉 no goals
  -/


@[simp]
theorem toIcoDiv_sub (a b : α) : toIcoDiv hp a (b - p) = toIcoDiv hp a b - 1 := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (toIcoDiv hp a (HSub.hSub b p)) (HSub.hSub (toIcoDiv hp a b) 1)
  -/
  simpa only [one_zsmul] using toIcoDiv_sub_zsmul hp a b 1
  /-
    🎉 no goals
  -/


@[simp]
theorem toIcoDiv_sub' (a b : α) : toIcoDiv hp (a - p) b = toIcoDiv hp a b + 1 := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (toIcoDiv hp (HSub.hSub a p) b) (HAdd.hAdd (toIcoDiv hp a b) 1)
  -/
  simpa only [one_zsmul] using toIcoDiv_sub_zsmul' hp a b 1
  /-
    🎉 no goals
  -/


@[simp]
theorem toIocDiv_sub (a b : α) : toIocDiv hp a (b - p) = toIocDiv hp a b - 1 := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (toIocDiv hp a (HSub.hSub b p)) (HSub.hSub (toIocDiv hp a b) 1)
  -/
  simpa only [one_zsmul] using toIocDiv_sub_zsmul hp a b 1
  /-
    🎉 no goals
  -/


@[simp]
theorem toIocDiv_sub' (a b : α) : toIocDiv hp (a - p) b = toIocDiv hp a b + 1 := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (toIocDiv hp (HSub.hSub a p) b) (HAdd.hAdd (toIocDiv hp a b) 1)
  -/
  simpa only [one_zsmul] using toIocDiv_sub_zsmul' hp a b 1
  /-
    🎉 no goals
  -/


theorem toIcoDiv_sub_eq_toIcoDiv_add (a b c : α) :
    toIcoDiv hp a (b - c) = toIcoDiv hp (a + c) b := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b c : α
    ⊢ Eq (toIcoDiv hp a (HSub.hSub b c)) (toIcoDiv hp (HAdd.hAdd a c) b)
  -/
  apply toIcoDiv_eq_of_sub_zsmul_mem_Ico
  /-
    case h
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b c : α
    ⊢ Membership.mem (Set.Ico a (HAdd.hAdd a p)) (HSub.hSub (HSub.hSub b c) (HSMul …
  -/
  rw [← sub_right_comm, Set.sub_mem_Ico_iff_left, add_right_comm]
  /-
    case h
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b c : α
    ⊢ Membership.mem (Set.Ico (HAdd.hAdd a c) (HAdd.hAdd (HAdd.hAdd a c) p)) (HSub …
  -/
  exact sub_toIcoDiv_zsmul_mem_Ico hp (a + c) b
  /-
    🎉 no goals
  -/


theorem toIocDiv_sub_eq_toIocDiv_add (a b c : α) :
    toIocDiv hp a (b - c) = toIocDiv hp (a + c) b := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b c : α
    ⊢ Eq (toIocDiv hp a (HSub.hSub b c)) (toIocDiv hp (HAdd.hAdd a c) b)
  -/
  apply toIocDiv_eq_of_sub_zsmul_mem_Ioc
  /-
    case h
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b c : α
    ⊢ Membership.mem (Set.Ioc a (HAdd.hAdd a p)) (HSub.hSub (HSub.hSub b c) (HSMul …
  -/
  rw [← sub_right_comm, Set.sub_mem_Ioc_iff_left, add_right_comm]
  /-
    case h
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b c : α
    ⊢ Membership.mem (Set.Ioc (HAdd.hAdd a c) (HAdd.hAdd (HAdd.hAdd a c) p)) (HSub …
  -/
  exact sub_toIocDiv_zsmul_mem_Ioc hp (a + c) b
  /-
    🎉 no goals
  -/


theorem toIcoDiv_sub_eq_toIcoDiv_add' (a b c : α) :
    toIcoDiv hp (a - c) b = toIcoDiv hp a (b + c) := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b c : α
    ⊢ Eq (toIcoDiv hp (HSub.hSub a c) b) (toIcoDiv hp a (HAdd.hAdd b c))
  -/
  rw [← sub_neg_eq_add, toIcoDiv_sub_eq_toIcoDiv_add, sub_eq_add_neg]
  /-
    🎉 no goals
  -/


theorem toIocDiv_sub_eq_toIocDiv_add' (a b c : α) :
    toIocDiv hp (a - c) b = toIocDiv hp a (b + c) := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b c : α
    ⊢ Eq (toIocDiv hp (HSub.hSub a c) b) (toIocDiv hp a (HAdd.hAdd b c))
  -/
  rw [← sub_neg_eq_add, toIocDiv_sub_eq_toIocDiv_add, sub_eq_add_neg]
  /-
    🎉 no goals
  -/


theorem toIcoDiv_neg (a b : α) : toIcoDiv hp a (-b) = -(toIocDiv hp (-a) b + 1) := by
  suffices toIcoDiv hp a (-b) = -toIocDiv hp (-(a + p)) b by
    rwa [neg_add, ← sub_eq_add_neg, toIocDiv_sub_eq_toIocDiv_add', toIocDiv_add_right] at this
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (toIcoDiv hp a (Neg.neg b)) (Neg.neg (toIocDiv hp (Neg.neg (HAdd.hAdd a p …
  -/
  rw [← neg_eq_iff_eq_neg, eq_comm]
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (toIocDiv hp (Neg.neg (HAdd.hAdd a p)) b) (Neg.neg (toIcoDiv hp a (Neg.ne …
  -/
  apply toIocDiv_eq_of_sub_zsmul_mem_Ioc
  /-
    case h
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Membership.mem (Set.Ioc (Neg.neg (HAdd.hAdd a p)) (HAdd.hAdd (Neg.neg (HAdd. …
  -/
  obtain ⟨hc, ho⟩ := sub_toIcoDiv_zsmul_mem_Ico hp a (-b)
  /-
    case h.intro
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    hc : LE.le a (HSub.hSub (Neg.neg b) (HSMul.hSMul (toIcoDiv hp a (Neg.neg b)) p))
    ho : LT.lt (HSub.hSub (Neg.neg b) (HSMul.hSMul (toIcoDiv hp a (Neg.neg b)) p)) …
    ⊢ Membership.mem (Set.Ioc (Neg.neg (HAdd.hAdd a p)) (HAdd.hAdd (Neg.neg (HAdd. …
  -/
  rw [← neg_lt_neg_iff, neg_sub' (-b), neg_neg, ← neg_smul] at ho
  /-
    case h.intro
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    hc : LE.le a (HSub.hSub (Neg.neg b) (HSMul.hSMul (toIcoDiv hp a (Neg.neg b)) p))
    ho : LT.lt (Neg.neg (HAdd.hAdd a p)) (HSub.hSub b (HSMul.hSMul (Neg.neg (toIco …
    ⊢ Membership.mem (Set.Ioc (Neg.neg (HAdd.hAdd a p)) (HAdd.hAdd (Neg.neg (HAdd. …
  -/
  rw [← neg_le_neg_iff, neg_sub' (-b), neg_neg, ← neg_smul] at hc
  /-
    case h.intro
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    hc : LE.le (HSub.hSub b (HSMul.hSMul (Neg.neg (toIcoDiv hp a (Neg.neg b))) p)) …
    ho : LT.lt (Neg.neg (HAdd.hAdd a p)) (HSub.hSub b (HSMul.hSMul (Neg.neg (toIco …
    ⊢ Membership.mem (Set.Ioc (Neg.neg (HAdd.hAdd a p)) (HAdd.hAdd (Neg.neg (HAdd. …
  -/
  refine ⟨ho, hc.trans_eq ?_⟩
  /-
    case h.intro
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    hc : LE.le (HSub.hSub b (HSMul.hSMul (Neg.neg (toIcoDiv hp a (Neg.neg b))) p)) …
    ho : LT.lt (Neg.neg (HAdd.hAdd a p)) (HSub.hSub b (HSMul.hSMul (Neg.neg (toIco …
    ⊢ Eq (Neg.neg a) (HAdd.hAdd (Neg.neg (HAdd.hAdd a p)) p)
  -/
  rw [neg_add, neg_add_cancel_right]
  /-
    🎉 no goals
  -/


theorem toIcoDiv_neg' (a b : α) : toIcoDiv hp (-a) b = -(toIocDiv hp a (-b) + 1) := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (toIcoDiv hp (Neg.neg a) b) (Neg.neg (HAdd.hAdd (toIocDiv hp a (Neg.neg b …
  -/
  simpa only [neg_neg] using toIcoDiv_neg hp (-a) (-b)
  /-
    🎉 no goals
  -/


theorem toIocDiv_neg (a b : α) : toIocDiv hp a (-b) = -(toIcoDiv hp (-a) b + 1) := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (toIocDiv hp a (Neg.neg b)) (Neg.neg (HAdd.hAdd (toIcoDiv hp (Neg.neg a)  …
  -/
  rw [← neg_neg b, toIcoDiv_neg, neg_neg, neg_neg, neg_add', neg_neg, add_sub_cancel_right]
  /-
    🎉 no goals
  -/


theorem toIocDiv_neg' (a b : α) : toIocDiv hp (-a) b = -(toIcoDiv hp a (-b) + 1) := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (toIocDiv hp (Neg.neg a) b) (Neg.neg (HAdd.hAdd (toIcoDiv hp a (Neg.neg b …
  -/
  simpa only [neg_neg] using toIocDiv_neg hp (-a) (-b)
  /-
    🎉 no goals
  -/


@[simp]
theorem toIcoMod_add_zsmul (a b : α) (m : ℤ) : toIcoMod hp a (b + m • p) = toIcoMod hp a b := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    m : Int
    ⊢ Eq (toIcoMod hp a (HAdd.hAdd b (HSMul.hSMul m p))) (toIcoMod hp a b)
  -/
  rw [toIcoMod, toIcoDiv_add_zsmul, toIcoMod, add_smul]
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    m : Int
    ⊢ Eq (HSub.hSub (HAdd.hAdd b (HSMul.hSMul m p)) (HAdd.hAdd (HSMul.hSMul (toIco …
  -/
  /-
    🎉 no goals
  -/
  abel
  /-
    🎉 no goals
  -/


@[simp]
theorem toIcoMod_add_zsmul' (a b : α) (m : ℤ) :
    toIcoMod hp (a + m • p) b = toIcoMod hp a b + m • p := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    m : Int
    ⊢ Eq (toIcoMod hp (HAdd.hAdd a (HSMul.hSMul m p)) b) (HAdd.hAdd (toIcoMod hp a …
  -/
  simp only [toIcoMod, toIcoDiv_add_zsmul', sub_smul, sub_add]
  /-
    🎉 no goals
  -/


@[simp]
theorem toIocMod_add_zsmul (a b : α) (m : ℤ) : toIocMod hp a (b + m • p) = toIocMod hp a b := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    m : Int
    ⊢ Eq (toIocMod hp a (HAdd.hAdd b (HSMul.hSMul m p))) (toIocMod hp a b)
  -/
  rw [toIocMod, toIocDiv_add_zsmul, toIocMod, add_smul]
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    m : Int
    ⊢ Eq (HSub.hSub (HAdd.hAdd b (HSMul.hSMul m p)) (HAdd.hAdd (HSMul.hSMul (toIoc …
  -/
  /-
    🎉 no goals
  -/
  abel
  /-
    🎉 no goals
  -/


@[simp]
theorem toIocMod_add_zsmul' (a b : α) (m : ℤ) :
    toIocMod hp (a + m • p) b = toIocMod hp a b + m • p := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    m : Int
    ⊢ Eq (toIocMod hp (HAdd.hAdd a (HSMul.hSMul m p)) b) (HAdd.hAdd (toIocMod hp a …
  -/
  simp only [toIocMod, toIocDiv_add_zsmul', sub_smul, sub_add]
  /-
    🎉 no goals
  -/


@[simp]
theorem toIcoMod_zsmul_add (a b : α) (m : ℤ) : toIcoMod hp a (m • p + b) = toIcoMod hp a b := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    m : Int
    ⊢ Eq (toIcoMod hp a (HAdd.hAdd (HSMul.hSMul m p) b)) (toIcoMod hp a b)
  -/
  rw [add_comm, toIcoMod_add_zsmul]
  /-
    🎉 no goals
  -/


@[simp]
theorem toIcoMod_zsmul_add' (a b : α) (m : ℤ) :
    toIcoMod hp (m • p + a) b = m • p + toIcoMod hp a b := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    m : Int
    ⊢ Eq (toIcoMod hp (HAdd.hAdd (HSMul.hSMul m p) a) b) (HAdd.hAdd (HSMul.hSMul m …
  -/
  rw [add_comm, toIcoMod_add_zsmul', add_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem toIocMod_zsmul_add (a b : α) (m : ℤ) : toIocMod hp a (m • p + b) = toIocMod hp a b := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    m : Int
    ⊢ Eq (toIocMod hp a (HAdd.hAdd (HSMul.hSMul m p) b)) (toIocMod hp a b)
  -/
  rw [add_comm, toIocMod_add_zsmul]
  /-
    🎉 no goals
  -/


@[simp]
theorem toIocMod_zsmul_add' (a b : α) (m : ℤ) :
    toIocMod hp (m • p + a) b = m • p + toIocMod hp a b := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    m : Int
    ⊢ Eq (toIocMod hp (HAdd.hAdd (HSMul.hSMul m p) a) b) (HAdd.hAdd (HSMul.hSMul m …
  -/
  rw [add_comm, toIocMod_add_zsmul', add_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem toIcoMod_sub_zsmul (a b : α) (m : ℤ) : toIcoMod hp a (b - m • p) = toIcoMod hp a b := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    m : Int
    ⊢ Eq (toIcoMod hp a (HSub.hSub b (HSMul.hSMul m p))) (toIcoMod hp a b)
  -/
  rw [sub_eq_add_neg, ← neg_smul, toIcoMod_add_zsmul]
  /-
    🎉 no goals
  -/


@[simp]
theorem toIcoMod_sub_zsmul' (a b : α) (m : ℤ) :
    toIcoMod hp (a - m • p) b = toIcoMod hp a b - m • p := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    m : Int
    ⊢ Eq (toIcoMod hp (HSub.hSub a (HSMul.hSMul m p)) b) (HSub.hSub (toIcoMod hp a …
  -/
  simp_rw [sub_eq_add_neg, ← neg_smul, toIcoMod_add_zsmul']
  /-
    🎉 no goals
  -/


@[simp]
theorem toIocMod_sub_zsmul (a b : α) (m : ℤ) : toIocMod hp a (b - m • p) = toIocMod hp a b := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    m : Int
    ⊢ Eq (toIocMod hp a (HSub.hSub b (HSMul.hSMul m p))) (toIocMod hp a b)
  -/
  rw [sub_eq_add_neg, ← neg_smul, toIocMod_add_zsmul]
  /-
    🎉 no goals
  -/


@[simp]
theorem toIocMod_sub_zsmul' (a b : α) (m : ℤ) :
    toIocMod hp (a - m • p) b = toIocMod hp a b - m • p := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    m : Int
    ⊢ Eq (toIocMod hp (HSub.hSub a (HSMul.hSMul m p)) b) (HSub.hSub (toIocMod hp a …
  -/
  simp_rw [sub_eq_add_neg, ← neg_smul, toIocMod_add_zsmul']
  /-
    🎉 no goals
  -/


@[simp]
theorem toIcoMod_add_right (a b : α) : toIcoMod hp a (b + p) = toIcoMod hp a b := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (toIcoMod hp a (HAdd.hAdd b p)) (toIcoMod hp a b)
  -/
  simpa only [one_zsmul] using toIcoMod_add_zsmul hp a b 1
  /-
    🎉 no goals
  -/


@[simp]
theorem toIcoMod_add_right' (a b : α) : toIcoMod hp (a + p) b = toIcoMod hp a b + p := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (toIcoMod hp (HAdd.hAdd a p) b) (HAdd.hAdd (toIcoMod hp a b) p)
  -/
  simpa only [one_zsmul] using toIcoMod_add_zsmul' hp a b 1
  /-
    🎉 no goals
  -/


@[simp]
theorem toIocMod_add_right (a b : α) : toIocMod hp a (b + p) = toIocMod hp a b := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (toIocMod hp a (HAdd.hAdd b p)) (toIocMod hp a b)
  -/
  simpa only [one_zsmul] using toIocMod_add_zsmul hp a b 1
  /-
    🎉 no goals
  -/


@[simp]
theorem toIocMod_add_right' (a b : α) : toIocMod hp (a + p) b = toIocMod hp a b + p := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (toIocMod hp (HAdd.hAdd a p) b) (HAdd.hAdd (toIocMod hp a b) p)
  -/
  simpa only [one_zsmul] using toIocMod_add_zsmul' hp a b 1
  /-
    🎉 no goals
  -/


@[simp]
theorem toIcoMod_add_left (a b : α) : toIcoMod hp a (p + b) = toIcoMod hp a b := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (toIcoMod hp a (HAdd.hAdd p b)) (toIcoMod hp a b)
  -/
  rw [add_comm, toIcoMod_add_right]
  /-
    🎉 no goals
  -/


@[simp]
theorem toIcoMod_add_left' (a b : α) : toIcoMod hp (p + a) b = p + toIcoMod hp a b := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (toIcoMod hp (HAdd.hAdd p a) b) (HAdd.hAdd p (toIcoMod hp a b))
  -/
  rw [add_comm, toIcoMod_add_right', add_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem toIocMod_add_left (a b : α) : toIocMod hp a (p + b) = toIocMod hp a b := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (toIocMod hp a (HAdd.hAdd p b)) (toIocMod hp a b)
  -/
  rw [add_comm, toIocMod_add_right]
  /-
    🎉 no goals
  -/


@[simp]
theorem toIocMod_add_left' (a b : α) : toIocMod hp (p + a) b = p + toIocMod hp a b := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (toIocMod hp (HAdd.hAdd p a) b) (HAdd.hAdd p (toIocMod hp a b))
  -/
  rw [add_comm, toIocMod_add_right', add_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem toIcoMod_sub (a b : α) : toIcoMod hp a (b - p) = toIcoMod hp a b := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (toIcoMod hp a (HSub.hSub b p)) (toIcoMod hp a b)
  -/
  simpa only [one_zsmul] using toIcoMod_sub_zsmul hp a b 1
  /-
    🎉 no goals
  -/


@[simp]
theorem toIcoMod_sub' (a b : α) : toIcoMod hp (a - p) b = toIcoMod hp a b - p := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (toIcoMod hp (HSub.hSub a p) b) (HSub.hSub (toIcoMod hp a b) p)
  -/
  simpa only [one_zsmul] using toIcoMod_sub_zsmul' hp a b 1
  /-
    🎉 no goals
  -/


@[simp]
theorem toIocMod_sub (a b : α) : toIocMod hp a (b - p) = toIocMod hp a b := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (toIocMod hp a (HSub.hSub b p)) (toIocMod hp a b)
  -/
  simpa only [one_zsmul] using toIocMod_sub_zsmul hp a b 1
  /-
    🎉 no goals
  -/


@[simp]
theorem toIocMod_sub' (a b : α) : toIocMod hp (a - p) b = toIocMod hp a b - p := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (toIocMod hp (HSub.hSub a p) b) (HSub.hSub (toIocMod hp a b) p)
  -/
  simpa only [one_zsmul] using toIocMod_sub_zsmul' hp a b 1
  /-
    🎉 no goals
  -/


theorem toIcoMod_sub_eq_sub (a b c : α) : toIcoMod hp a (b - c) = toIcoMod hp (a + c) b - c := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b c : α
    ⊢ Eq (toIcoMod hp a (HSub.hSub b c)) (HSub.hSub (toIcoMod hp (HAdd.hAdd a c) b …
  -/
  simp_rw [toIcoMod, toIcoDiv_sub_eq_toIcoDiv_add, sub_right_comm]
  /-
    🎉 no goals
  -/


theorem toIocMod_sub_eq_sub (a b c : α) : toIocMod hp a (b - c) = toIocMod hp (a + c) b - c := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b c : α
    ⊢ Eq (toIocMod hp a (HSub.hSub b c)) (HSub.hSub (toIocMod hp (HAdd.hAdd a c) b …
  -/
  simp_rw [toIocMod, toIocDiv_sub_eq_toIocDiv_add, sub_right_comm]
  /-
    🎉 no goals
  -/


theorem toIcoMod_add_right_eq_add (a b c : α) :
    toIcoMod hp a (b + c) = toIcoMod hp (a - c) b + c := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b c : α
    ⊢ Eq (toIcoMod hp a (HAdd.hAdd b c)) (HAdd.hAdd (toIcoMod hp (HSub.hSub a c) b …
  -/
  simp_rw [toIcoMod, toIcoDiv_sub_eq_toIcoDiv_add', sub_add_eq_add_sub]
  /-
    🎉 no goals
  -/


theorem toIocMod_add_right_eq_add (a b c : α) :
    toIocMod hp a (b + c) = toIocMod hp (a - c) b + c := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b c : α
    ⊢ Eq (toIocMod hp a (HAdd.hAdd b c)) (HAdd.hAdd (toIocMod hp (HSub.hSub a c) b …
  -/
  simp_rw [toIocMod, toIocDiv_sub_eq_toIocDiv_add', sub_add_eq_add_sub]
  /-
    🎉 no goals
  -/


theorem toIcoMod_neg (a b : α) : toIcoMod hp a (-b) = p - toIocMod hp (-a) b := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (toIcoMod hp a (Neg.neg b)) (HSub.hSub p (toIocMod hp (Neg.neg a) b))
  -/
  simp_rw [toIcoMod, toIocMod, toIcoDiv_neg, neg_smul, add_smul]
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (HSub.hSub (Neg.neg b) (Neg.neg (HAdd.hAdd (HSMul.hSMul (toIocDiv hp (Neg …
  -/
  /-
    🎉 no goals
  -/
  abel
  /-
    🎉 no goals
  -/


theorem toIcoMod_neg' (a b : α) : toIcoMod hp (-a) b = p - toIocMod hp a (-b) := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (toIcoMod hp (Neg.neg a) b) (HSub.hSub p (toIocMod hp a (Neg.neg b)))
  -/
  simpa only [neg_neg] using toIcoMod_neg hp (-a) (-b)
  /-
    🎉 no goals
  -/


theorem toIocMod_neg (a b : α) : toIocMod hp a (-b) = p - toIcoMod hp (-a) b := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (toIocMod hp a (Neg.neg b)) (HSub.hSub p (toIcoMod hp (Neg.neg a) b))
  -/
  simp_rw [toIocMod, toIcoMod, toIocDiv_neg, neg_smul, add_smul]
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (HSub.hSub (Neg.neg b) (Neg.neg (HAdd.hAdd (HSMul.hSMul (toIcoDiv hp (Neg …
  -/
  /-
    🎉 no goals
  -/
  abel
  /-
    🎉 no goals
  -/


theorem toIocMod_neg' (a b : α) : toIocMod hp (-a) b = p - toIcoMod hp a (-b) := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (toIocMod hp (Neg.neg a) b) (HSub.hSub p (toIcoMod hp a (Neg.neg b)))
  -/
  simpa only [neg_neg] using toIocMod_neg hp (-a) (-b)
  /-
    🎉 no goals
  -/


theorem toIcoMod_eq_toIcoMod : toIcoMod hp a b = toIcoMod hp a c ↔ ∃ n : ℤ, c - b = n • p := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b c : α
    ⊢ Iff (Eq (toIcoMod hp a b) (toIcoMod hp a c)) (Exists fun n => Eq (HSub.hSub  …
  -/
  refine ⟨fun h => ⟨toIcoDiv hp a c - toIcoDiv hp a b, ?_⟩, fun h => ?_⟩
    /-
      case refine_1
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroup α
      hα : Archimedean α
      p : α
      hp : LT.lt 0 p
      a b c : α
      h : Eq (toIcoMod hp a b) (toIcoMod hp a c)
      ⊢ Eq (HSub.hSub c b) (HSMul.hSMul (HSub.hSub (toIcoDiv hp a c) (toIcoDiv hp a  …
    -/
  · conv_lhs => rw [← toIcoMod_add_toIcoDiv_zsmul hp a b, ← toIcoMod_add_toIcoDiv_zsmul hp a c]
    /-
      case refine_1
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroup α
      hα : Archimedean α
      p : α
      hp : LT.lt 0 p
      a b c : α
      h : Eq (toIcoMod hp a b) (toIcoMod hp a c)
      ⊢ Eq (HSub.hSub (HAdd.hAdd (toIcoMod hp a c) (HSMul.hSMul (toIcoDiv hp a c) p) …
    -/
    rw [h, sub_smul]
    /-
      case refine_1
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroup α
      hα : Archimedean α
      p : α
      hp : LT.lt 0 p
      a b c : α
      h : Eq (toIcoMod hp a b) (toIcoMod hp a c)
      ⊢ Eq (HSub.hSub (HAdd.hAdd (toIcoMod hp a c) (HSMul.hSMul (toIcoDiv hp a c) p) …
    -/
    /-
      🎉 no goals
    -/
    abel
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroup α
      hα : Archimedean α
      p : α
      hp : LT.lt 0 p
      a b c : α
      h : Exists fun n => Eq (HSub.hSub c b) (HSMul.hSMul n p)
      ⊢ Eq (toIcoMod hp a b) (toIcoMod hp a c)
    -/
  · rcases h with ⟨z, hz⟩
    /-
      case refine_2.intro
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroup α
      hα : Archimedean α
      p : α
      hp : LT.lt 0 p
      a b c : α
      z : Int
      hz : Eq (HSub.hSub c b) (HSMul.hSMul z p)
      ⊢ Eq (toIcoMod hp a b) (toIcoMod hp a c)
    -/
    rw [sub_eq_iff_eq_add] at hz
    /-
      case refine_2.intro
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroup α
      hα : Archimedean α
      p : α
      hp : LT.lt 0 p
      a b c : α
      z : Int
      hz : Eq c (HAdd.hAdd (HSMul.hSMul z p) b)
      ⊢ Eq (toIcoMod hp a b) (toIcoMod hp a c)
    -/
    rw [hz, toIcoMod_zsmul_add]
    /-
      🎉 no goals
    -/


theorem toIocMod_eq_toIocMod : toIocMod hp a b = toIocMod hp a c ↔ ∃ n : ℤ, c - b = n • p := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b c : α
    ⊢ Iff (Eq (toIocMod hp a b) (toIocMod hp a c)) (Exists fun n => Eq (HSub.hSub  …
  -/
  refine ⟨fun h => ⟨toIocDiv hp a c - toIocDiv hp a b, ?_⟩, fun h => ?_⟩
    /-
      case refine_1
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroup α
      hα : Archimedean α
      p : α
      hp : LT.lt 0 p
      a b c : α
      h : Eq (toIocMod hp a b) (toIocMod hp a c)
      ⊢ Eq (HSub.hSub c b) (HSMul.hSMul (HSub.hSub (toIocDiv hp a c) (toIocDiv hp a  …
    -/
  · conv_lhs => rw [← toIocMod_add_toIocDiv_zsmul hp a b, ← toIocMod_add_toIocDiv_zsmul hp a c]
    /-
      case refine_1
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroup α
      hα : Archimedean α
      p : α
      hp : LT.lt 0 p
      a b c : α
      h : Eq (toIocMod hp a b) (toIocMod hp a c)
      ⊢ Eq (HSub.hSub (HAdd.hAdd (toIocMod hp a c) (HSMul.hSMul (toIocDiv hp a c) p) …
    -/
    rw [h, sub_smul]
    /-
      case refine_1
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroup α
      hα : Archimedean α
      p : α
      hp : LT.lt 0 p
      a b c : α
      h : Eq (toIocMod hp a b) (toIocMod hp a c)
      ⊢ Eq (HSub.hSub (HAdd.hAdd (toIocMod hp a c) (HSMul.hSMul (toIocDiv hp a c) p) …
    -/
    /-
      🎉 no goals
    -/
    abel
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroup α
      hα : Archimedean α
      p : α
      hp : LT.lt 0 p
      a b c : α
      h : Exists fun n => Eq (HSub.hSub c b) (HSMul.hSMul n p)
      ⊢ Eq (toIocMod hp a b) (toIocMod hp a c)
    -/
  · rcases h with ⟨z, hz⟩
    /-
      case refine_2.intro
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroup α
      hα : Archimedean α
      p : α
      hp : LT.lt 0 p
      a b c : α
      z : Int
      hz : Eq (HSub.hSub c b) (HSMul.hSMul z p)
      ⊢ Eq (toIocMod hp a b) (toIocMod hp a c)
    -/
    rw [sub_eq_iff_eq_add] at hz
    /-
      case refine_2.intro
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroup α
      hα : Archimedean α
      p : α
      hp : LT.lt 0 p
      a b c : α
      z : Int
      hz : Eq c (HAdd.hAdd (HSMul.hSMul z p) b)
      ⊢ Eq (toIocMod hp a b) (toIocMod hp a c)
    -/
    rw [hz, toIocMod_zsmul_add]
    /-
      🎉 no goals
    -/


theorem modEq_iff_toIcoMod_eq_left : a ≡ b [PMOD p] ↔ toIcoMod hp a b = a :=
  modEq_iff_eq_add_zsmul.trans
    ⟨by
      /-
        α : Type u_1
        inst✝ : LinearOrderedAddCommGroup α
        hα : Archimedean α
        p : α
        hp : LT.lt 0 p
        a b : α
        ⊢ (Exists fun z => Eq b (HAdd.hAdd a (HSMul.hSMul z p))) → Eq (toIcoMod hp a b …
      -/
      rintro ⟨n, rfl⟩
      /-
        case intro
        α : Type u_1
        inst✝ : LinearOrderedAddCommGroup α
        hα : Archimedean α
        p : α
        hp : LT.lt 0 p
        a : α
        n : Int
        ⊢ Eq (toIcoMod hp a (HAdd.hAdd a (HSMul.hSMul n p))) a
      -/
      rw [toIcoMod_add_zsmul, toIcoMod_apply_left], fun h => ⟨toIcoDiv hp a b, eq_add_of_sub_eq h⟩⟩
      /-
        🎉 no goals
      -/


theorem modEq_iff_toIocMod_eq_right : a ≡ b [PMOD p] ↔ toIocMod hp a b = a + p := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Iff (AddCommGroup.ModEq p a b) (Eq (toIocMod hp a b) (HAdd.hAdd a p))
  -/
  refine modEq_iff_eq_add_zsmul.trans ⟨?_, fun h => ⟨toIocDiv hp a b + 1, ?_⟩⟩
    /-
      case refine_1
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroup α
      hα : Archimedean α
      p : α
      hp : LT.lt 0 p
      a b : α
      ⊢ (Exists fun z => Eq b (HAdd.hAdd a (HSMul.hSMul z p))) → Eq (toIocMod hp a b …
    -/
  · rintro ⟨z, rfl⟩
    /-
      case refine_1.intro
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroup α
      hα : Archimedean α
      p : α
      hp : LT.lt 0 p
      a : α
      z : Int
      ⊢ Eq (toIocMod hp a (HAdd.hAdd a (HSMul.hSMul z p))) (HAdd.hAdd a p)
    -/
    rw [toIocMod_add_zsmul, toIocMod_apply_left]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroup α
      hα : Archimedean α
      p : α
      hp : LT.lt 0 p
      a b : α
      h : Eq (toIocMod hp a b) (HAdd.hAdd a p)
      ⊢ Eq b (HAdd.hAdd a (HSMul.hSMul (HAdd.hAdd (toIocDiv hp a b) 1) p))
    -/
  · rwa [add_one_zsmul, add_left_comm, ← sub_eq_iff_eq_add']
    /-
      🎉 no goals
    -/


alias ⟨ModEq.toIcoMod_eq_left, _⟩ := modEq_iff_toIcoMod_eq_left


alias ⟨ModEq.toIcoMod_eq_right, _⟩ := modEq_iff_toIocMod_eq_right


open List in
theorem tfae_modEq :
    TFAE
      [a ≡ b [PMOD p], ∀ z : ℤ, b - z • p ∉ Set.Ioo a (a + p), toIcoMod hp a b ≠ toIocMod hp a b,
        toIcoMod hp a b + p = toIocMod hp a b] := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ (List.cons (AddCommGroup.ModEq p a b) (List.cons (∀ (z : Int), Not (Membersh …
  -/
  rw [modEq_iff_toIcoMod_eq_left hp]
  tfae_have 3 → 2 := by
    rw [← not_exists, not_imp_not]
    exact fun ⟨i, hi⟩ =>
      ((toIcoMod_eq_iff hp).2 ⟨Set.Ioo_subset_Ico_self hi, i, (sub_add_cancel b _).symm⟩).trans
        ((toIocMod_eq_iff hp).2 ⟨Set.Ioo_subset_Ioc_self hi, i, (sub_add_cancel b _).symm⟩).symm
  tfae_have 4 → 3
  | h => by
    rw [← h, Ne, eq_comm, add_right_eq_self]
    exact hp.ne'
  tfae_have 1 → 4
  | h => by
    rw [h, eq_comm, toIocMod_eq_iff, Set.right_mem_Ioc]
    refine ⟨lt_add_of_pos_right a hp, toIcoDiv hp a b - 1, ?_⟩
    rw [sub_one_zsmul, add_add_add_comm, add_neg_cancel, add_zero]
    conv_lhs => rw [← toIcoMod_add_toIcoDiv_zsmul hp a b, h]
  tfae_have 2 → 1 := by
    rw [← not_exists, not_imp_comm]
    have h' := toIcoMod_mem_Ico hp a b
    exact fun h => ⟨_, h'.1.lt_of_ne' h, h'.2⟩
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    tfae_3_to_2 : Ne (toIcoMod hp a b) (toIocMod hp a b) → ∀ (z : Int), Not (Membe …
    tfae_4_to_3 : Eq (HAdd.hAdd (toIcoMod hp a b) p) (toIocMod hp a b) → Ne (toIco …
    tfae_1_to_4 : Eq (toIcoMod hp a b) a → Eq (HAdd.hAdd (toIcoMod hp a b) p) (toI …
    tfae_2_to_1 : (∀ (z : Int), Not (Membership.mem (Set.Ioo a (HAdd.hAdd a p)) (H …
    ⊢ (List.cons (Eq (toIcoMod hp a b) a) (List.cons (∀ (z : Int), Not (Membership …
  -/
  tfae_finish
  /-
    🎉 no goals
  -/


theorem modEq_iff_not_forall_mem_Ioo_mod :
    a ≡ b [PMOD p] ↔ ∀ z : ℤ, b - z • p ∉ Set.Ioo a (a + p) :=
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq ((List.cons (AddCommGroup.ModEq p a b) (List.cons (∀ (z : Int), Not (Memb …
  -/
  /-
    🎉 no goals
  -/
  (tfae_modEq hp a b).out 0 1
  /-
    🎉 no goals
  -/


theorem modEq_iff_toIcoMod_ne_toIocMod : a ≡ b [PMOD p] ↔ toIcoMod hp a b ≠ toIocMod hp a b :=
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq ((List.cons (AddCommGroup.ModEq p a b) (List.cons (∀ (z : Int), Not (Memb …
  -/
  /-
    🎉 no goals
  -/
  (tfae_modEq hp a b).out 0 2
  /-
    🎉 no goals
  -/


theorem modEq_iff_toIcoMod_add_period_eq_toIocMod :
    a ≡ b [PMOD p] ↔ toIcoMod hp a b + p = toIocMod hp a b :=
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq ((List.cons (AddCommGroup.ModEq p a b) (List.cons (∀ (z : Int), Not (Memb …
  -/
  /-
    🎉 no goals
  -/
  (tfae_modEq hp a b).out 0 3
  /-
    🎉 no goals
  -/


theorem not_modEq_iff_toIcoMod_eq_toIocMod : ¬a ≡ b [PMOD p] ↔ toIcoMod hp a b = toIocMod hp a b :=
  (modEq_iff_toIcoMod_ne_toIocMod _).not_left


theorem not_modEq_iff_toIcoDiv_eq_toIocDiv :
    ¬a ≡ b [PMOD p] ↔ toIcoDiv hp a b = toIocDiv hp a b := by
  rw [not_modEq_iff_toIcoMod_eq_toIocMod hp, toIcoMod, toIocMod, sub_right_inj,
    zsmul_left_inj hp]


theorem modEq_iff_toIcoDiv_eq_toIocDiv_add_one :
    a ≡ b [PMOD p] ↔ toIcoDiv hp a b = toIocDiv hp a b + 1 := by
  rw [modEq_iff_toIcoMod_add_period_eq_toIocMod hp, toIcoMod, toIocMod, ← eq_sub_iff_add_eq,
    sub_sub, sub_right_inj, ← add_one_zsmul, zsmul_left_inj hp]


/-- If `a` and `b` fall within the same cycle WRT `c`, then they are congruent modulo `p`. -/
@[simp]
theorem toIcoMod_inj {c : α} : toIcoMod hp c a = toIcoMod hp c b ↔ a ≡ b [PMOD p] := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b c : α
    ⊢ Iff (Eq (toIcoMod hp c a) (toIcoMod hp c b)) (AddCommGroup.ModEq p a b)
  -/
  simp_rw [toIcoMod_eq_toIcoMod, modEq_iff_eq_add_zsmul, sub_eq_iff_eq_add']
  /-
    🎉 no goals
  -/


alias ⟨_, AddCommGroup.ModEq.toIcoMod_eq_toIcoMod⟩ := toIcoMod_inj


theorem Ico_eq_locus_Ioc_eq_iUnion_Ioo :
    { b | toIcoMod hp a b = toIocMod hp a b } = ⋃ z : ℤ, Set.Ioo (a + z • p) (a + p + z • p) := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a : α
    ⊢ Eq (setOf fun b => Eq (toIcoMod hp a b) (toIocMod hp a b)) (Set.iUnion fun z …
  -/
  ext1
  simp_rw [Set.mem_setOf, Set.mem_iUnion, ← Set.sub_mem_Ioo_iff_left, ←
    not_modEq_iff_toIcoMod_eq_toIocMod, modEq_iff_not_forall_mem_Ioo_mod hp, not_forall,
    Classical.not_not]


theorem toIocDiv_wcovBy_toIcoDiv (a b : α) : toIocDiv hp a b ⩿ toIcoDiv hp a b := by
  suffices toIocDiv hp a b = toIcoDiv hp a b ∨ toIocDiv hp a b + 1 = toIcoDiv hp a b by
    rwa [wcovBy_iff_eq_or_covBy, ← Order.succ_eq_iff_covBy]
  rw [eq_comm, ← not_modEq_iff_toIcoDiv_eq_toIocDiv, eq_comm, ←
    modEq_iff_toIcoDiv_eq_toIocDiv_add_one]
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Or (Not (AddCommGroup.ModEq p a b)) (AddCommGroup.ModEq p a b)
  -/
  exact em' _
  /-
    🎉 no goals
  -/


theorem toIcoMod_le_toIocMod (a b : α) : toIcoMod hp a b ≤ toIocMod hp a b := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ LE.le (toIcoMod hp a b) (toIocMod hp a b)
  -/
  rw [toIcoMod, toIocMod, sub_le_sub_iff_left]
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ LE.le (HSMul.hSMul (toIocDiv hp a b) p) (HSMul.hSMul (toIcoDiv hp a b) p)
  -/
  exact zsmul_left_mono hp.le (toIocDiv_wcovBy_toIcoDiv _ _ _).le
  /-
    🎉 no goals
  -/


theorem toIocMod_le_toIcoMod_add (a b : α) : toIocMod hp a b ≤ toIcoMod hp a b + p := by
  rw [toIcoMod, toIocMod, sub_add, sub_le_sub_iff_left, sub_le_iff_le_add, ← add_one_zsmul,
    (zsmul_left_strictMono hp).le_iff_le]
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ LE.le (toIcoDiv hp a b) (HAdd.hAdd (toIocDiv hp a b) 1)
  -/
  apply (toIocDiv_wcovBy_toIcoDiv _ _ _).le_succ
  /-
    🎉 no goals
  -/


theorem toIcoMod_eq_self : toIcoMod hp a b = b ↔ b ∈ Set.Ico a (a + p) := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Iff (Eq (toIcoMod hp a b) b) (Membership.mem (Set.Ico a (HAdd.hAdd a p)) b)
  -/
  rw [toIcoMod_eq_iff, and_iff_left]
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Exists fun z => Eq b (HAdd.hAdd b (HSMul.hSMul z p))
  -/
  exact ⟨0, by simp⟩
  /-
    🎉 no goals
  -/


theorem toIocMod_eq_self : toIocMod hp a b = b ↔ b ∈ Set.Ioc a (a + p) := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Iff (Eq (toIocMod hp a b) b) (Membership.mem (Set.Ioc a (HAdd.hAdd a p)) b)
  -/
  rw [toIocMod_eq_iff, and_iff_left]
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Exists fun z => Eq b (HAdd.hAdd b (HSMul.hSMul z p))
  -/
  exact ⟨0, by simp⟩
  /-
    🎉 no goals
  -/


@[simp]
theorem toIcoMod_toIcoMod (a₁ a₂ b : α) : toIcoMod hp a₁ (toIcoMod hp a₂ b) = toIcoMod hp a₁ b :=
  (toIcoMod_eq_toIcoMod _).2 ⟨toIcoDiv hp a₂ b, self_sub_toIcoMod hp a₂ b⟩


@[simp]
theorem toIcoMod_toIocMod (a₁ a₂ b : α) : toIcoMod hp a₁ (toIocMod hp a₂ b) = toIcoMod hp a₁ b :=
  (toIcoMod_eq_toIcoMod _).2 ⟨toIocDiv hp a₂ b, self_sub_toIocMod hp a₂ b⟩


@[simp]
theorem toIocMod_toIocMod (a₁ a₂ b : α) : toIocMod hp a₁ (toIocMod hp a₂ b) = toIocMod hp a₁ b :=
  (toIocMod_eq_toIocMod _).2 ⟨toIocDiv hp a₂ b, self_sub_toIocMod hp a₂ b⟩


@[simp]
theorem toIocMod_toIcoMod (a₁ a₂ b : α) : toIocMod hp a₁ (toIcoMod hp a₂ b) = toIocMod hp a₁ b :=
  (toIocMod_eq_toIocMod _).2 ⟨toIcoDiv hp a₂ b, self_sub_toIcoMod hp a₂ b⟩


theorem toIcoMod_periodic (a : α) : Function.Periodic (toIcoMod hp a) p :=
  toIcoMod_add_right hp a


theorem toIocMod_periodic (a : α) : Function.Periodic (toIocMod hp a) p :=
  toIocMod_add_right hp a

-- helper lemmas for when `a = 0`

theorem toIcoMod_zero_sub_comm (a b : α) : toIcoMod hp 0 (a - b) = p - toIocMod hp 0 (b - a) := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (toIcoMod hp 0 (HSub.hSub a b)) (HSub.hSub p (toIocMod hp 0 (HSub.hSub b  …
  -/
  rw [← neg_sub, toIcoMod_neg, neg_zero]
  /-
    🎉 no goals
  -/


theorem toIocMod_zero_sub_comm (a b : α) : toIocMod hp 0 (a - b) = p - toIcoMod hp 0 (b - a) := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (toIocMod hp 0 (HSub.hSub a b)) (HSub.hSub p (toIcoMod hp 0 (HSub.hSub b  …
  -/
  rw [← neg_sub, toIocMod_neg, neg_zero]
  /-
    🎉 no goals
  -/


theorem toIcoDiv_eq_sub (a b : α) : toIcoDiv hp a b = toIcoDiv hp 0 (b - a) := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (toIcoDiv hp a b) (toIcoDiv hp 0 (HSub.hSub b a))
  -/
  rw [toIcoDiv_sub_eq_toIcoDiv_add, zero_add]
  /-
    🎉 no goals
  -/


theorem toIocDiv_eq_sub (a b : α) : toIocDiv hp a b = toIocDiv hp 0 (b - a) := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (toIocDiv hp a b) (toIocDiv hp 0 (HSub.hSub b a))
  -/
  rw [toIocDiv_sub_eq_toIocDiv_add, zero_add]
  /-
    🎉 no goals
  -/


theorem toIcoMod_eq_sub (a b : α) : toIcoMod hp a b = toIcoMod hp 0 (b - a) + a := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (toIcoMod hp a b) (HAdd.hAdd (toIcoMod hp 0 (HSub.hSub b a)) a)
  -/
  rw [toIcoMod_sub_eq_sub, zero_add, sub_add_cancel]
  /-
    🎉 no goals
  -/


theorem toIocMod_eq_sub (a b : α) : toIocMod hp a b = toIocMod hp 0 (b - a) + a := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (toIocMod hp a b) (HAdd.hAdd (toIocMod hp 0 (HSub.hSub b a)) a)
  -/
  rw [toIocMod_sub_eq_sub, zero_add, sub_add_cancel]
  /-
    🎉 no goals
  -/


theorem toIcoMod_add_toIocMod_zero (a b : α) :
    toIcoMod hp 0 (a - b) + toIocMod hp 0 (b - a) = p := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (HAdd.hAdd (toIcoMod hp 0 (HSub.hSub a b)) (toIocMod hp 0 (HSub.hSub b a) …
  -/
  rw [toIcoMod_zero_sub_comm, sub_add_cancel]
  /-
    🎉 no goals
  -/


theorem toIocMod_add_toIcoMod_zero (a b : α) :
    toIocMod hp 0 (a - b) + toIcoMod hp 0 (b - a) = p := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (HAdd.hAdd (toIocMod hp 0 (HSub.hSub a b)) (toIcoMod hp 0 (HSub.hSub b a) …
  -/
  rw [_root_.add_comm, toIcoMod_add_toIocMod_zero]
  /-
    🎉 no goals
  -/


/-- `toIcoMod` as an equiv from the quotient. -/
@[simps symm_apply]
def QuotientAddGroup.equivIcoMod (a : α) : α ⧸ AddSubgroup.zmultiples p ≃ Set.Ico a (a + p) where
  toFun b :=
    ⟨(toIcoMod_periodic hp a).lift b, QuotientAddGroup.induction_on b <| toIcoMod_mem_Ico hp a⟩
  invFun := (↑)
  right_inv b := Subtype.ext <| (toIcoMod_eq_self hp).mpr b.prop
  left_inv b := by
    /-
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroup α
      hα : Archimedean α
      p : α
      hp : LT.lt 0 p
      a✝ b✝ c : α
      n : Int
      a : α
      b : HasQuotient.Quotient α (AddSubgroup.zmultiples p)
      ⊢ Eq ((fun x => ↑↑x) ((fun b => ⟨⋯.lift b, ⋯⟩) b)) b
    -/
    induction b using QuotientAddGroup.induction_on
    /-
      case H
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroup α
      hα : Archimedean α
      p : α
      hp : LT.lt 0 p
      a✝ b c : α
      n : Int
      a z✝ : α
      ⊢ Eq ((fun x => ↑↑x) ((fun b => ⟨⋯.lift b, ⋯⟩) ↑z✝)) ↑z✝
    -/
    dsimp
    /-
      case H
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroup α
      hα : Archimedean α
      p : α
      hp : LT.lt 0 p
      a✝ b c : α
      n : Int
      a z✝ : α
      ⊢ Eq ↑(toIcoMod hp a z✝) ↑z✝
    -/
    rw [QuotientAddGroup.eq_iff_sub_mem, toIcoMod_sub_self]
    /-
      case H
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroup α
      hα : Archimedean α
      p : α
      hp : LT.lt 0 p
      a✝ b c : α
      n : Int
      a z✝ : α
      ⊢ Membership.mem (AddSubgroup.zmultiples p) (HSMul.hSMul (Neg.neg (toIcoDiv hp …
    -/
    apply AddSubgroup.zsmul_mem_zmultiples
    /-
      🎉 no goals
    -/


@[simp]
theorem QuotientAddGroup.equivIcoMod_coe (a b : α) :
    QuotientAddGroup.equivIcoMod hp a ↑b = ⟨toIcoMod hp a b, toIcoMod_mem_Ico hp a _⟩ :=
  rfl


@[simp]
theorem QuotientAddGroup.equivIcoMod_zero (a : α) :
    QuotientAddGroup.equivIcoMod hp a 0 = ⟨toIcoMod hp a 0, toIcoMod_mem_Ico hp a _⟩ :=
  rfl


/-- `toIocMod` as an equiv from the quotient. -/
@[simps symm_apply]
def QuotientAddGroup.equivIocMod (a : α) : α ⧸ AddSubgroup.zmultiples p ≃ Set.Ioc a (a + p) where
  toFun b :=
    ⟨(toIocMod_periodic hp a).lift b, QuotientAddGroup.induction_on b <| toIocMod_mem_Ioc hp a⟩
  invFun := (↑)
  right_inv b := Subtype.ext <| (toIocMod_eq_self hp).mpr b.prop
  left_inv b := by
    /-
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroup α
      hα : Archimedean α
      p : α
      hp : LT.lt 0 p
      a✝ b✝ c : α
      n : Int
      a : α
      b : HasQuotient.Quotient α (AddSubgroup.zmultiples p)
      ⊢ Eq ((fun x => ↑↑x) ((fun b => ⟨⋯.lift b, ⋯⟩) b)) b
    -/
    induction b using QuotientAddGroup.induction_on
    /-
      case H
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroup α
      hα : Archimedean α
      p : α
      hp : LT.lt 0 p
      a✝ b c : α
      n : Int
      a z✝ : α
      ⊢ Eq ((fun x => ↑↑x) ((fun b => ⟨⋯.lift b, ⋯⟩) ↑z✝)) ↑z✝
    -/
    dsimp
    /-
      case H
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroup α
      hα : Archimedean α
      p : α
      hp : LT.lt 0 p
      a✝ b c : α
      n : Int
      a z✝ : α
      ⊢ Eq ↑(toIocMod hp a z✝) ↑z✝
    -/
    rw [QuotientAddGroup.eq_iff_sub_mem, toIocMod_sub_self]
    /-
      case H
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroup α
      hα : Archimedean α
      p : α
      hp : LT.lt 0 p
      a✝ b c : α
      n : Int
      a z✝ : α
      ⊢ Membership.mem (AddSubgroup.zmultiples p) (HSMul.hSMul (Neg.neg (toIocDiv hp …
    -/
    apply AddSubgroup.zsmul_mem_zmultiples
    /-
      🎉 no goals
    -/


@[simp]
theorem QuotientAddGroup.equivIocMod_coe (a b : α) :
    QuotientAddGroup.equivIocMod hp a ↑b = ⟨toIocMod hp a b, toIocMod_mem_Ioc hp a _⟩ :=
  rfl


@[simp]
theorem QuotientAddGroup.equivIocMod_zero (a : α) :
    QuotientAddGroup.equivIocMod hp a 0 = ⟨toIocMod hp a 0, toIocMod_mem_Ioc hp a _⟩ :=
  rfl

private theorem toIxxMod_iff (x₁ x₂ x₃ : α) : toIcoMod hp x₁ x₂ ≤ toIocMod hp x₁ x₃ ↔
    toIcoMod hp 0 (x₂ - x₁) + toIcoMod hp 0 (x₁ - x₃) ≤ p := by
  rw [toIcoMod_eq_sub, toIocMod_eq_sub _ x₁, add_le_add_iff_right, ← neg_sub x₁ x₃, toIocMod_neg,
    neg_zero, le_sub_iff_add_le]


private theorem toIxxMod_cyclic_left {x₁ x₂ x₃ : α} (h : toIcoMod hp x₁ x₂ ≤ toIocMod hp x₁ x₃) :
    toIcoMod hp x₂ x₃ ≤ toIocMod hp x₂ x₁ := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    x₁ x₂ x₃ : α
    h : LE.le (toIcoMod hp x₁ x₂) (toIocMod hp x₁ x₃)
    ⊢ LE.le (toIcoMod hp x₂ x₃) (toIocMod hp x₂ x₁)
  -/
  let x₂' := toIcoMod hp x₁ x₂
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    x₁ x₂ x₃ : α
    h : LE.le (toIcoMod hp x₁ x₂) (toIocMod hp x₁ x₃)
    x₂' : α := toIcoMod hp x₁ x₂
    ⊢ LE.le (toIcoMod hp x₂ x₃) (toIocMod hp x₂ x₁)
  -/
  let x₃' := toIcoMod hp x₂' x₃
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    x₁ x₂ x₃ : α
    h : LE.le (toIcoMod hp x₁ x₂) (toIocMod hp x₁ x₃)
    x₂' : α := toIcoMod hp x₁ x₂
    x₃' : α := toIcoMod hp x₂' x₃
    ⊢ LE.le (toIcoMod hp x₂ x₃) (toIocMod hp x₂ x₁)
  -/
  have h : x₂' ≤ toIocMod hp x₁ x₃' := by simpa [x₃']
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    x₁ x₂ x₃ : α
    h✝ : LE.le (toIcoMod hp x₁ x₂) (toIocMod hp x₁ x₃)
    x₂' : α := toIcoMod hp x₁ x₂
    x₃' : α := toIcoMod hp x₂' x₃
    h : LE.le x₂' (toIocMod hp x₁ x₃')
    ⊢ LE.le (toIcoMod hp x₂ x₃) (toIocMod hp x₂ x₁)
  -/
  have h₂₁ : x₂' < x₁ + p := toIcoMod_lt_right _ _ _
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    x₁ x₂ x₃ : α
    h✝ : LE.le (toIcoMod hp x₁ x₂) (toIocMod hp x₁ x₃)
    x₂' : α := toIcoMod hp x₁ x₂
    x₃' : α := toIcoMod hp x₂' x₃
    h : LE.le x₂' (toIocMod hp x₁ x₃')
    h₂₁ : LT.lt x₂' (HAdd.hAdd x₁ p)
    ⊢ LE.le (toIcoMod hp x₂ x₃) (toIocMod hp x₂ x₁)
  -/
  have h₃₂ : x₃' - p < x₂' := sub_lt_iff_lt_add.2 (toIcoMod_lt_right _ _ _)
  suffices hequiv : x₃' ≤ toIocMod hp x₂' x₁ by
    obtain ⟨z, hd⟩ : ∃ z : ℤ, x₂ = x₂' + z • p := ((toIcoMod_eq_iff hp).1 rfl).2
    rw [hd, toIocMod_add_zsmul', toIcoMod_add_zsmul', add_le_add_iff_right]
    assumption -- Porting note: was `simpa`
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    x₁ x₂ x₃ : α
    h✝ : LE.le (toIcoMod hp x₁ x₂) (toIocMod hp x₁ x₃)
    x₂' : α := toIcoMod hp x₁ x₂
    x₃' : α := toIcoMod hp x₂' x₃
    h : LE.le x₂' (toIocMod hp x₁ x₃')
    h₂₁ : LT.lt x₂' (HAdd.hAdd x₁ p)
    h₃₂ : LT.lt (HSub.hSub x₃' p) x₂'
    ⊢ LE.le x₃' (toIocMod hp x₂' x₁)
  -/
  rcases le_or_lt x₃' (x₁ + p) with h₃₁ | h₁₃
    /-
      case inl
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroup α
      hα : Archimedean α
      p : α
      hp : LT.lt 0 p
      x₁ x₂ x₃ : α
      h✝ : LE.le (toIcoMod hp x₁ x₂) (toIocMod hp x₁ x₃)
      x₂' : α := toIcoMod hp x₁ x₂
      x₃' : α := toIcoMod hp x₂' x₃
      h : LE.le x₂' (toIocMod hp x₁ x₃')
      h₂₁ : LT.lt x₂' (HAdd.hAdd x₁ p)
      h₃₂ : LT.lt (HSub.hSub x₃' p) x₂'
      h₃₁ : LE.le x₃' (HAdd.hAdd x₁ p)
      ⊢ LE.le x₃' (toIocMod hp x₂' x₁)
    -/
  · suffices hIoc₂₁ : toIocMod hp x₂' x₁ = x₁ + p from hIoc₂₁.symm.trans_ge h₃₁
    /-
      case inl
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroup α
      hα : Archimedean α
      p : α
      hp : LT.lt 0 p
      x₁ x₂ x₃ : α
      h✝ : LE.le (toIcoMod hp x₁ x₂) (toIocMod hp x₁ x₃)
      x₂' : α := toIcoMod hp x₁ x₂
      x₃' : α := toIcoMod hp x₂' x₃
      h : LE.le x₂' (toIocMod hp x₁ x₃')
      h₂₁ : LT.lt x₂' (HAdd.hAdd x₁ p)
      h₃₂ : LT.lt (HSub.hSub x₃' p) x₂'
      h₃₁ : LE.le x₃' (HAdd.hAdd x₁ p)
      ⊢ Eq (toIocMod hp x₂' x₁) (HAdd.hAdd x₁ p)
    -/
    apply (toIocMod_eq_iff hp).2
    /-
      case inl
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroup α
      hα : Archimedean α
      p : α
      hp : LT.lt 0 p
      x₁ x₂ x₃ : α
      h✝ : LE.le (toIcoMod hp x₁ x₂) (toIocMod hp x₁ x₃)
      x₂' : α := toIcoMod hp x₁ x₂
      x₃' : α := toIcoMod hp x₂' x₃
      h : LE.le x₂' (toIocMod hp x₁ x₃')
      h₂₁ : LT.lt x₂' (HAdd.hAdd x₁ p)
      h₃₂ : LT.lt (HSub.hSub x₃' p) x₂'
      h₃₁ : LE.le x₃' (HAdd.hAdd x₁ p)
      ⊢ And (Membership.mem (Set.Ioc x₂' (HAdd.hAdd x₂' p)) (HAdd.hAdd x₁ p)) (Exist …
    -/
    exact ⟨⟨h₂₁, by simp [x₂', left_le_toIcoMod]⟩, -1, by simp⟩
    /-
      🎉 no goals
    -/
  have hIoc₁₃ : toIocMod hp x₁ x₃' = x₃' - p := by
    apply (toIocMod_eq_iff hp).2
    exact ⟨⟨lt_sub_iff_add_lt.2 h₁₃, le_of_lt (h₃₂.trans h₂₁)⟩, 1, by simp⟩
  /-
    case inr
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    x₁ x₂ x₃ : α
    h✝ : LE.le (toIcoMod hp x₁ x₂) (toIocMod hp x₁ x₃)
    x₂' : α := toIcoMod hp x₁ x₂
    x₃' : α := toIcoMod hp x₂' x₃
    h : LE.le x₂' (toIocMod hp x₁ x₃')
    h₂₁ : LT.lt x₂' (HAdd.hAdd x₁ p)
    h₃₂ : LT.lt (HSub.hSub x₃' p) x₂'
    h₁₃ : LT.lt (HAdd.hAdd x₁ p) x₃'
    hIoc₁₃ : Eq (toIocMod hp x₁ x₃') (HSub.hSub x₃' p)
    ⊢ LE.le x₃' (toIocMod hp x₂' x₁)
  -/
  have not_h₃₂ := (h.trans hIoc₁₃.le).not_lt
  /-
    case inr
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    x₁ x₂ x₃ : α
    h✝ : LE.le (toIcoMod hp x₁ x₂) (toIocMod hp x₁ x₃)
    x₂' : α := toIcoMod hp x₁ x₂
    x₃' : α := toIcoMod hp x₂' x₃
    h : LE.le x₂' (toIocMod hp x₁ x₃')
    h₂₁ : LT.lt x₂' (HAdd.hAdd x₁ p)
    h₃₂ : LT.lt (HSub.hSub x₃' p) x₂'
    h₁₃ : LT.lt (HAdd.hAdd x₁ p) x₃'
    hIoc₁₃ : Eq (toIocMod hp x₁ x₃') (HSub.hSub x₃' p)
    not_h₃₂ : Not (LT.lt (HSub.hSub x₃' p) x₂')
    ⊢ LE.le x₃' (toIocMod hp x₂' x₁)
  -/
  contradiction
  /-
    🎉 no goals
  -/


private theorem toIxxMod_antisymm (h₁₂₃ : toIcoMod hp a b ≤ toIocMod hp a c)
    (h₁₃₂ : toIcoMod hp a c ≤ toIocMod hp a b) :
    b ≡ a [PMOD p] ∨ c ≡ b [PMOD p] ∨ a ≡ c [PMOD p] := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b c : α
    h₁₂₃ : LE.le (toIcoMod hp a b) (toIocMod hp a c)
    h₁₃₂ : LE.le (toIcoMod hp a c) (toIocMod hp a b)
    ⊢ Or (AddCommGroup.ModEq p b a) (Or (AddCommGroup.ModEq p c b) (AddCommGroup.M …
  -/
  by_contra! h
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b c : α
    h₁₂₃ : LE.le (toIcoMod hp a b) (toIocMod hp a c)
    h₁₃₂ : LE.le (toIcoMod hp a c) (toIocMod hp a b)
    h : And (Not (AddCommGroup.ModEq p b a)) (And (Not (AddCommGroup.ModEq p c b)) …
    ⊢ False
  -/
  rw [modEq_comm] at h
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b c : α
    h₁₂₃ : LE.le (toIcoMod hp a b) (toIocMod hp a c)
    h₁₃₂ : LE.le (toIcoMod hp a c) (toIocMod hp a b)
    h : And (Not (AddCommGroup.ModEq p a b)) (And (Not (AddCommGroup.ModEq p c b)) …
    ⊢ False
  -/
  rw [← (not_modEq_iff_toIcoMod_eq_toIocMod hp).mp h.2.2] at h₁₂₃
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b c : α
    h₁₂₃ : LE.le (toIcoMod hp a b) (toIcoMod hp a c)
    h₁₃₂ : LE.le (toIcoMod hp a c) (toIocMod hp a b)
    h : And (Not (AddCommGroup.ModEq p a b)) (And (Not (AddCommGroup.ModEq p c b)) …
    ⊢ False
  -/
  rw [← (not_modEq_iff_toIcoMod_eq_toIocMod hp).mp h.1] at h₁₃₂
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b c : α
    h₁₂₃ : LE.le (toIcoMod hp a b) (toIcoMod hp a c)
    h₁₃₂ : LE.le (toIcoMod hp a c) (toIcoMod hp a b)
    h : And (Not (AddCommGroup.ModEq p a b)) (And (Not (AddCommGroup.ModEq p c b)) …
    ⊢ False
  -/
  exact h.2.1 ((toIcoMod_inj _).1 <| h₁₃₂.antisymm h₁₂₃)
  /-
    🎉 no goals
  -/


private theorem toIxxMod_total' (a b c : α) :
    toIcoMod hp b a ≤ toIocMod hp b c ∨ toIcoMod hp b c ≤ toIocMod hp b a := by
  /- an essential ingredient is the lemma saying {a-b} + {b-a} = period if a ≠ b (and = 0 if a = b).
    Thus if a ≠ b and b ≠ c then ({a-b} + {b-c}) + ({c-b} + {b-a}) = 2 * period, so one of
    `{a-b} + {b-c}` and `{c-b} + {b-a}` must be `≤ period` -/
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b c : α
    ⊢ Or (LE.le (toIcoMod hp b a) (toIocMod hp b c)) (LE.le (toIcoMod hp b c) (toI …
  -/
  have := congr_arg₂ (· + ·) (toIcoMod_add_toIocMod_zero hp a b) (toIcoMod_add_toIocMod_zero hp c b)
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b c : α
    this : Eq ((fun x1 x2 => HAdd.hAdd x1 x2) (HAdd.hAdd (toIcoMod hp 0 (HSub.hSub …
    ⊢ Or (LE.le (toIcoMod hp b a) (toIocMod hp b c)) (LE.le (toIcoMod hp b c) (toI …
  -/
  simp only [add_add_add_comm] at this
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b c : α
    this : Eq (HAdd.hAdd (HAdd.hAdd (toIcoMod hp 0 (HSub.hSub a b)) (toIcoMod hp 0 …
    ⊢ Or (LE.le (toIcoMod hp b a) (toIocMod hp b c)) (LE.le (toIcoMod hp b c) (toI …
  -/
  rw [_root_.add_comm (toIocMod _ _ _), add_add_add_comm, ← two_nsmul] at this
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b c : α
    this : Eq (HAdd.hAdd (HAdd.hAdd (toIcoMod hp 0 (HSub.hSub a b)) (toIocMod hp 0 …
    ⊢ Or (LE.le (toIcoMod hp b a) (toIocMod hp b c)) (LE.le (toIcoMod hp b c) (toI …
  -/
  replace := min_le_of_add_le_two_nsmul this.le
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b c : α
    this : LE.le (Min.min (HAdd.hAdd (toIcoMod hp 0 (HSub.hSub a b)) (toIocMod hp  …
    ⊢ Or (LE.le (toIcoMod hp b a) (toIocMod hp b c)) (LE.le (toIcoMod hp b c) (toI …
  -/
  rw [min_le_iff] at this
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b c : α
    this : Or (LE.le (HAdd.hAdd (toIcoMod hp 0 (HSub.hSub a b)) (toIocMod hp 0 (HS …
    ⊢ Or (LE.le (toIcoMod hp b a) (toIocMod hp b c)) (LE.le (toIcoMod hp b c) (toI …
  -/
  rw [toIxxMod_iff, toIxxMod_iff]
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b c : α
    this : Or (LE.le (HAdd.hAdd (toIcoMod hp 0 (HSub.hSub a b)) (toIocMod hp 0 (HS …
    ⊢ Or (LE.le (HAdd.hAdd (toIcoMod hp 0 (HSub.hSub a b)) (toIcoMod hp 0 (HSub.hS …
  -/
  refine this.imp (le_trans <| add_le_add_left ?_ _) (le_trans <| add_le_add_left ?_ _)
    /-
      case refine_1
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroup α
      hα : Archimedean α
      p : α
      hp : LT.lt 0 p
      a b c : α
      this : Or (LE.le (HAdd.hAdd (toIcoMod hp 0 (HSub.hSub a b)) (toIocMod hp 0 (HS …
      ⊢ LE.le (toIcoMod hp 0 (HSub.hSub b c)) (toIocMod hp 0 (HSub.hSub b c))
    -/
  · apply toIcoMod_le_toIocMod
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroup α
      hα : Archimedean α
      p : α
      hp : LT.lt 0 p
      a b c : α
      this : Or (LE.le (HAdd.hAdd (toIcoMod hp 0 (HSub.hSub a b)) (toIocMod hp 0 (HS …
      ⊢ LE.le (toIcoMod hp 0 (HSub.hSub b a)) (toIocMod hp 0 (HSub.hSub b a))
    -/
  · apply toIcoMod_le_toIocMod
    /-
      🎉 no goals
    -/


private theorem toIxxMod_total (a b c : α) :
    toIcoMod hp a b ≤ toIocMod hp a c ∨ toIcoMod hp c b ≤ toIocMod hp c a :=
  (toIxxMod_total' _ _ _ _).imp_right <| toIxxMod_cyclic_left _


private theorem toIxxMod_trans {x₁ x₂ x₃ x₄ : α}
    (h₁₂₃ : toIcoMod hp x₁ x₂ ≤ toIocMod hp x₁ x₃ ∧ ¬toIcoMod hp x₃ x₂ ≤ toIocMod hp x₃ x₁)
    (h₂₃₄ : toIcoMod hp x₂ x₄ ≤ toIocMod hp x₂ x₃ ∧ ¬toIcoMod hp x₃ x₄ ≤ toIocMod hp x₃ x₂) :
    toIcoMod hp x₁ x₄ ≤ toIocMod hp x₁ x₃ ∧ ¬toIcoMod hp x₃ x₄ ≤ toIocMod hp x₃ x₁ := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp : LT.lt 0 p
    x₁ x₂ x₃ x₄ : α
    h₁₂₃ : And (LE.le (toIcoMod hp x₁ x₂) (toIocMod hp x₁ x₃)) (Not (LE.le (toIcoM …
    h₂₃₄ : And (LE.le (toIcoMod hp x₂ x₄) (toIocMod hp x₂ x₃)) (Not (LE.le (toIcoM …
    ⊢ And (LE.le (toIcoMod hp x₁ x₄) (toIocMod hp x₁ x₃)) (Not (LE.le (toIcoMod hp …
  -/
  constructor
  · suffices h : ¬x₃ ≡ x₂ [PMOD p] by
      have h₁₂₃' := toIxxMod_cyclic_left _ (toIxxMod_cyclic_left _ h₁₂₃.1)
      have h₂₃₄' := toIxxMod_cyclic_left _ (toIxxMod_cyclic_left _ h₂₃₄.1)
      rw [(not_modEq_iff_toIcoMod_eq_toIocMod hp).1 h] at h₂₃₄'
      exact toIxxMod_cyclic_left _ (h₁₂₃'.trans h₂₃₄')
    /-
      case left
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroup α
      hα : Archimedean α
      p : α
      hp : LT.lt 0 p
      x₁ x₂ x₃ x₄ : α
      h₁₂₃ : And (LE.le (toIcoMod hp x₁ x₂) (toIocMod hp x₁ x₃)) (Not (LE.le (toIcoM …
      h₂₃₄ : And (LE.le (toIcoMod hp x₂ x₄) (toIocMod hp x₂ x₃)) (Not (LE.le (toIcoM …
      ⊢ Not (AddCommGroup.ModEq p x₃ x₂)
    -/
    by_contra h
    /-
      case left
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroup α
      hα : Archimedean α
      p : α
      hp : LT.lt 0 p
      x₁ x₂ x₃ x₄ : α
      h₁₂₃ : And (LE.le (toIcoMod hp x₁ x₂) (toIocMod hp x₁ x₃)) (Not (LE.le (toIcoM …
      h₂₃₄ : And (LE.le (toIcoMod hp x₂ x₄) (toIocMod hp x₂ x₃)) (Not (LE.le (toIcoM …
      h : AddCommGroup.ModEq p x₃ x₂
      ⊢ False
    -/
    rw [(modEq_iff_toIcoMod_eq_left hp).1 h] at h₁₂₃
    /-
      case left
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroup α
      hα : Archimedean α
      p : α
      hp : LT.lt 0 p
      x₁ x₂ x₃ x₄ : α
      h₁₂₃ : And (LE.le (toIcoMod hp x₁ x₂) (toIocMod hp x₁ x₃)) (Not (LE.le x₃ (toI …
      h₂₃₄ : And (LE.le (toIcoMod hp x₂ x₄) (toIocMod hp x₂ x₃)) (Not (LE.le (toIcoM …
      h : AddCommGroup.ModEq p x₃ x₂
      ⊢ False
    -/
    exact h₁₂₃.2 (left_lt_toIocMod _ _ _).le
    /-
      🎉 no goals
    -/
    /-
      case right
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroup α
      hα : Archimedean α
      p : α
      hp : LT.lt 0 p
      x₁ x₂ x₃ x₄ : α
      h₁₂₃ : And (LE.le (toIcoMod hp x₁ x₂) (toIocMod hp x₁ x₃)) (Not (LE.le (toIcoM …
      h₂₃₄ : And (LE.le (toIcoMod hp x₂ x₄) (toIocMod hp x₂ x₃)) (Not (LE.le (toIcoM …
      ⊢ Not (LE.le (toIcoMod hp x₃ x₄) (toIocMod hp x₃ x₁))
    -/
  · rw [not_le] at h₁₂₃ h₂₃₄ ⊢
    /-
      case right
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroup α
      hα : Archimedean α
      p : α
      hp : LT.lt 0 p
      x₁ x₂ x₃ x₄ : α
      h₁₂₃ : And (LE.le (toIcoMod hp x₁ x₂) (toIocMod hp x₁ x₃)) (LT.lt (toIocMod hp …
      h₂₃₄ : And (LE.le (toIcoMod hp x₂ x₄) (toIocMod hp x₂ x₃)) (LT.lt (toIocMod hp …
      ⊢ LT.lt (toIocMod hp x₃ x₁) (toIcoMod hp x₃ x₄)
    -/
    exact (h₁₂₃.2.trans_le (toIcoMod_le_toIocMod _ x₃ x₂)).trans h₂₃₄.2
    /-
      🎉 no goals
    -/


instance : Btw (α ⧸ AddSubgroup.zmultiples p) where
  btw x₁ x₂ x₃ := (equivIcoMod hp'.out 0 (x₂ - x₁) : α) ≤ equivIocMod hp'.out 0 (x₃ - x₁)


theorem btw_coe_iff' {x₁ x₂ x₃ : α} :
    Btw.btw (x₁ : α ⧸ AddSubgroup.zmultiples p) x₂ x₃ ↔
      toIcoMod hp'.out 0 (x₂ - x₁) ≤ toIocMod hp'.out 0 (x₃ - x₁) :=
  Iff.rfl

-- maybe harder to use than the primed one?

theorem btw_coe_iff {x₁ x₂ x₃ : α} :
    Btw.btw (x₁ : α ⧸ AddSubgroup.zmultiples p) x₂ x₃ ↔
      toIcoMod hp'.out x₁ x₂ ≤ toIocMod hp'.out x₁ x₃ := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedAddCommGroup α
    hα : Archimedean α
    p : α
    hp' : Fact (LT.lt 0 p)
    x₁ x₂ x₃ : α
    ⊢ Iff (Btw.btw ↑x₁ ↑x₂ ↑x₃) (LE.le (toIcoMod ⋯ x₁ x₂) (toIocMod ⋯ x₁ x₃))
  -/
  rw [btw_coe_iff', toIocMod_sub_eq_sub, toIcoMod_sub_eq_sub, zero_add, sub_le_sub_iff_right]
  /-
    🎉 no goals
  -/


instance circularPreorder : CircularPreorder (α ⧸ AddSubgroup.zmultiples p) where
                              /-
                                α : Type u_1
                                inst✝ : LinearOrderedAddCommGroup α
                                hα : Archimedean α
                                p : α
                                hp : LT.lt 0 p
                                a b c : α
                                n : Int
                                hp' : Fact (LT.lt 0 p)
                                x : HasQuotient.Quotient α (AddSubgroup.zmultiples p)
                                ⊢ LE.le ↑((QuotientAddGroup.equivIcoMod ⋯ 0) (HSub.hSub x x)) ↑((QuotientAddGr …
                              -/
  btw_refl x := show _ ≤ _ by simp [sub_self, hp'.out.le]
                              /-
                                🎉 no goals
                              -/
  btw_cyclic_left {x₁ x₂ x₃} h := by
    /-
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroup α
      hα : Archimedean α
      p : α
      hp : LT.lt 0 p
      a b c : α
      n : Int
      hp' : Fact (LT.lt 0 p)
      x₁ x₂ x₃ : HasQuotient.Quotient α (AddSubgroup.zmultiples p)
      h : Btw.btw x₁ x₂ x₃
      ⊢ Btw.btw x₂ x₃ x₁
    -/
    induction x₁ using QuotientAddGroup.induction_on
    /-
      case H
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroup α
      hα : Archimedean α
      p : α
      hp : LT.lt 0 p
      a b c : α
      n : Int
      hp' : Fact (LT.lt 0 p)
      x₂ x₃ : HasQuotient.Quotient α (AddSubgroup.zmultiples p)
      z✝ : α
      h : Btw.btw (↑z✝) x₂ x₃
      ⊢ Btw.btw x₂ x₃ ↑z✝
    -/
    induction x₂ using QuotientAddGroup.induction_on
    /-
      case H.H
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroup α
      hα : Archimedean α
      p : α
      hp : LT.lt 0 p
      a b c : α
      n : Int
      hp' : Fact (LT.lt 0 p)
      x₃ : HasQuotient.Quotient α (AddSubgroup.zmultiples p)
      z✝¹ z✝ : α
      h : Btw.btw (↑z✝¹) (↑z✝) x₃
      ⊢ Btw.btw (↑z✝) x₃ ↑z✝¹
    -/
    induction x₃ using QuotientAddGroup.induction_on
    /-
      case H.H.H
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroup α
      hα : Archimedean α
      p : α
      hp : LT.lt 0 p
      a b c : α
      n : Int
      hp' : Fact (LT.lt 0 p)
      z✝² z✝¹ z✝ : α
      h : Btw.btw ↑z✝² ↑z✝¹ ↑z✝
      ⊢ Btw.btw ↑z✝¹ ↑z✝ ↑z✝²
    -/
    simp_rw [btw_coe_iff] at h ⊢
    /-
      case H.H.H
      α : Type u_1
      inst✝ : LinearOrderedAddCommGroup α
      hα : Archimedean α
      p : α
      hp : LT.lt 0 p
      a b c : α
      n : Int
      hp' : Fact (LT.lt 0 p)
      z✝² z✝¹ z✝ : α
      h : LE.le (toIcoMod ⋯ z✝² z✝¹) (toIocMod ⋯ z✝² z✝)
      ⊢ LE.le (toIcoMod ⋯ z✝¹ z✝) (toIocMod ⋯ z✝¹ z✝²)
    -/
    apply toIxxMod_cyclic_left _ h
    /-
      🎉 no goals
    -/
  sbtw := _
  sbtw_iff_btw_not_btw := Iff.rfl
  sbtw_trans_left {x₁ x₂ x₃ x₄} (h₁₂₃ : _ ∧ _) (h₂₃₄ : _ ∧ _) :=
    show _ ∧ _ by
      /-
        α : Type u_1
        inst✝ : LinearOrderedAddCommGroup α
        hα : Archimedean α
        p : α
        hp : LT.lt 0 p
        a b c : α
        n : Int
        hp' : Fact (LT.lt 0 p)
        x₁ x₂ x₃ x₄ : HasQuotient.Quotient α (AddSubgroup.zmultiples p)
        h₁₂₃ : And (Btw.btw x₁ x₂ x₃) (Not (Btw.btw x₃ x₂ x₁))
        h₂₃₄ : And (Btw.btw x₂ x₄ x₃) (Not (Btw.btw x₃ x₄ x₂))
        ⊢ And (Btw.btw x₁ x₄ x₃) (Not (Btw.btw x₃ x₄ x₁))
      -/
      induction x₁ using QuotientAddGroup.induction_on
      /-
        case H
        α : Type u_1
        inst✝ : LinearOrderedAddCommGroup α
        hα : Archimedean α
        p : α
        hp : LT.lt 0 p
        a b c : α
        n : Int
        hp' : Fact (LT.lt 0 p)
        x₂ x₃ x₄ : HasQuotient.Quotient α (AddSubgroup.zmultiples p)
        h₂₃₄ : And (Btw.btw x₂ x₄ x₃) (Not (Btw.btw x₃ x₄ x₂))
        z✝ : α
        h₁₂₃ : And (Btw.btw (↑z✝) x₂ x₃) (Not (Btw.btw x₃ x₂ ↑z✝))
        ⊢ And (Btw.btw (↑z✝) x₄ x₃) (Not (Btw.btw x₃ x₄ ↑z✝))
      -/
      induction x₂ using QuotientAddGroup.induction_on
      /-
        case H.H
        α : Type u_1
        inst✝ : LinearOrderedAddCommGroup α
        hα : Archimedean α
        p : α
        hp : LT.lt 0 p
        a b c : α
        n : Int
        hp' : Fact (LT.lt 0 p)
        x₃ x₄ : HasQuotient.Quotient α (AddSubgroup.zmultiples p)
        z✝¹ z✝ : α
        h₂₃₄ : And (Btw.btw (↑z✝) x₄ x₃) (Not (Btw.btw x₃ x₄ ↑z✝))
        h₁₂₃ : And (Btw.btw (↑z✝¹) (↑z✝) x₃) (Not (Btw.btw x₃ ↑z✝ ↑z✝¹))
        ⊢ And (Btw.btw (↑z✝¹) x₄ x₃) (Not (Btw.btw x₃ x₄ ↑z✝¹))
      -/
      induction x₃ using QuotientAddGroup.induction_on
      /-
        case H.H.H
        α : Type u_1
        inst✝ : LinearOrderedAddCommGroup α
        hα : Archimedean α
        p : α
        hp : LT.lt 0 p
        a b c : α
        n : Int
        hp' : Fact (LT.lt 0 p)
        x₄ : HasQuotient.Quotient α (AddSubgroup.zmultiples p)
        z✝² z✝¹ z✝ : α
        h₂₃₄ : And (Btw.btw (↑z✝¹) x₄ ↑z✝) (Not (Btw.btw (↑z✝) x₄ ↑z✝¹))
        h₁₂₃ : And (Btw.btw ↑z✝² ↑z✝¹ ↑z✝) (Not (Btw.btw ↑z✝ ↑z✝¹ ↑z✝²))
        ⊢ And (Btw.btw (↑z✝²) x₄ ↑z✝) (Not (Btw.btw (↑z✝) x₄ ↑z✝²))
      -/
      induction x₄ using QuotientAddGroup.induction_on
      /-
        case H.H.H.H
        α : Type u_1
        inst✝ : LinearOrderedAddCommGroup α
        hα : Archimedean α
        p : α
        hp : LT.lt 0 p
        a b c : α
        n : Int
        hp' : Fact (LT.lt 0 p)
        z✝³ z✝² z✝¹ : α
        h₁₂₃ : And (Btw.btw ↑z✝³ ↑z✝² ↑z✝¹) (Not (Btw.btw ↑z✝¹ ↑z✝² ↑z✝³))
        z✝ : α
        h₂₃₄ : And (Btw.btw ↑z✝² ↑z✝ ↑z✝¹) (Not (Btw.btw ↑z✝¹ ↑z✝ ↑z✝²))
        ⊢ And (Btw.btw ↑z✝³ ↑z✝ ↑z✝¹) (Not (Btw.btw ↑z✝¹ ↑z✝ ↑z✝³))
      -/
      simp_rw [btw_coe_iff] at h₁₂₃ h₂₃₄ ⊢
      /-
        case H.H.H.H
        α : Type u_1
        inst✝ : LinearOrderedAddCommGroup α
        hα : Archimedean α
        p : α
        hp : LT.lt 0 p
        a b c : α
        n : Int
        hp' : Fact (LT.lt 0 p)
        z✝³ z✝² z✝¹ z✝ : α
        h₁₂₃ : And (LE.le (toIcoMod ⋯ z✝³ z✝²) (toIocMod ⋯ z✝³ z✝¹)) (Not (LE.le (toIc …
        h₂₃₄ : And (LE.le (toIcoMod ⋯ z✝² z✝) (toIocMod ⋯ z✝² z✝¹)) (Not (LE.le (toIco …
        ⊢ And (LE.le (toIcoMod ⋯ z✝³ z✝) (toIocMod ⋯ z✝³ z✝¹)) (Not (LE.le (toIcoMod ⋯ …
      -/
      apply toIxxMod_trans _ h₁₂₃ h₂₃₄
      /-
        🎉 no goals
      -/


instance circularOrder : CircularOrder (α ⧸ AddSubgroup.zmultiples p) :=
  { QuotientAddGroup.circularPreorder with
    btw_antisymm := fun {x₁ x₂ x₃} h₁₂₃ h₃₂₁ => by
      /-
        α : Type u_1
        inst✝ : LinearOrderedAddCommGroup α
        hα : Archimedean α
        p : α
        hp : LT.lt 0 p
        a b c : α
        n : Int
        hp' : Fact (LT.lt 0 p)
        x₁ x₂ x₃ : HasQuotient.Quotient α (AddSubgroup.zmultiples p)
        h₁₂₃ : Btw.btw x₁ x₂ x₃
        h₃₂₁ : Btw.btw x₃ x₂ x₁
        ⊢ Or (Eq x₁ x₂) (Or (Eq x₂ x₃) (Eq x₃ x₁))
      -/
      induction x₁ using QuotientAddGroup.induction_on
      /-
        case H
        α : Type u_1
        inst✝ : LinearOrderedAddCommGroup α
        hα : Archimedean α
        p : α
        hp : LT.lt 0 p
        a b c : α
        n : Int
        hp' : Fact (LT.lt 0 p)
        x₂ x₃ : HasQuotient.Quotient α (AddSubgroup.zmultiples p)
        z✝ : α
        h₁₂₃ : Btw.btw (↑z✝) x₂ x₃
        h₃₂₁ : Btw.btw x₃ x₂ ↑z✝
        ⊢ Or (Eq (↑z✝) x₂) (Or (Eq x₂ x₃) (Eq x₃ ↑z✝))
      -/
      induction x₂ using QuotientAddGroup.induction_on
      /-
        case H.H
        α : Type u_1
        inst✝ : LinearOrderedAddCommGroup α
        hα : Archimedean α
        p : α
        hp : LT.lt 0 p
        a b c : α
        n : Int
        hp' : Fact (LT.lt 0 p)
        x₃ : HasQuotient.Quotient α (AddSubgroup.zmultiples p)
        z✝¹ z✝ : α
        h₁₂₃ : Btw.btw (↑z✝¹) (↑z✝) x₃
        h₃₂₁ : Btw.btw x₃ ↑z✝ ↑z✝¹
        ⊢ Or (Eq ↑z✝¹ ↑z✝) (Or (Eq (↑z✝) x₃) (Eq x₃ ↑z✝¹))
      -/
      induction x₃ using QuotientAddGroup.induction_on
      /-
        case H.H.H
        α : Type u_1
        inst✝ : LinearOrderedAddCommGroup α
        hα : Archimedean α
        p : α
        hp : LT.lt 0 p
        a b c : α
        n : Int
        hp' : Fact (LT.lt 0 p)
        z✝² z✝¹ z✝ : α
        h₁₂₃ : Btw.btw ↑z✝² ↑z✝¹ ↑z✝
        h₃₂₁ : Btw.btw ↑z✝ ↑z✝¹ ↑z✝²
        ⊢ Or (Eq ↑z✝² ↑z✝¹) (Or (Eq ↑z✝¹ ↑z✝) (Eq ↑z✝ ↑z✝²))
      -/
      rw [btw_cyclic] at h₃₂₁
      /-
        case H.H.H
        α : Type u_1
        inst✝ : LinearOrderedAddCommGroup α
        hα : Archimedean α
        p : α
        hp : LT.lt 0 p
        a b c : α
        n : Int
        hp' : Fact (LT.lt 0 p)
        z✝² z✝¹ z✝ : α
        h₁₂₃ : Btw.btw ↑z✝² ↑z✝¹ ↑z✝
        h₃₂₁ : Btw.btw ↑z✝² ↑z✝ ↑z✝¹
        ⊢ Or (Eq ↑z✝² ↑z✝¹) (Or (Eq ↑z✝¹ ↑z✝) (Eq ↑z✝ ↑z✝²))
      -/
      simp_rw [btw_coe_iff] at h₁₂₃ h₃₂₁
      /-
        case H.H.H
        α : Type u_1
        inst✝ : LinearOrderedAddCommGroup α
        hα : Archimedean α
        p : α
        hp : LT.lt 0 p
        a b c : α
        n : Int
        hp' : Fact (LT.lt 0 p)
        z✝² z✝¹ z✝ : α
        h₁₂₃ : LE.le (toIcoMod ⋯ z✝² z✝¹) (toIocMod ⋯ z✝² z✝)
        h₃₂₁ : LE.le (toIcoMod ⋯ z✝² z✝) (toIocMod ⋯ z✝² z✝¹)
        ⊢ Or (Eq ↑z✝² ↑z✝¹) (Or (Eq ↑z✝¹ ↑z✝) (Eq ↑z✝ ↑z✝²))
      -/
      simp_rw [← modEq_iff_eq_mod_zmultiples]
      /-
        case H.H.H
        α : Type u_1
        inst✝ : LinearOrderedAddCommGroup α
        hα : Archimedean α
        p : α
        hp : LT.lt 0 p
        a b c : α
        n : Int
        hp' : Fact (LT.lt 0 p)
        z✝² z✝¹ z✝ : α
        h₁₂₃ : LE.le (toIcoMod ⋯ z✝² z✝¹) (toIocMod ⋯ z✝² z✝)
        h₃₂₁ : LE.le (toIcoMod ⋯ z✝² z✝) (toIocMod ⋯ z✝² z✝¹)
        ⊢ Or (AddCommGroup.ModEq p z✝¹ z✝²) (Or (AddCommGroup.ModEq p z✝ z✝¹) (AddComm …
      -/
      exact toIxxMod_antisymm _ h₁₂₃ h₃₂₁
      /-
        🎉 no goals
      -/
    btw_total := fun x₁ x₂ x₃ => by
      /-
        α : Type u_1
        inst✝ : LinearOrderedAddCommGroup α
        hα : Archimedean α
        p : α
        hp : LT.lt 0 p
        a b c : α
        n : Int
        hp' : Fact (LT.lt 0 p)
        x₁ x₂ x₃ : HasQuotient.Quotient α (AddSubgroup.zmultiples p)
        ⊢ Or (Btw.btw x₁ x₂ x₃) (Btw.btw x₃ x₂ x₁)
      -/
      induction x₁ using QuotientAddGroup.induction_on
      /-
        case H
        α : Type u_1
        inst✝ : LinearOrderedAddCommGroup α
        hα : Archimedean α
        p : α
        hp : LT.lt 0 p
        a b c : α
        n : Int
        hp' : Fact (LT.lt 0 p)
        x₂ x₃ : HasQuotient.Quotient α (AddSubgroup.zmultiples p)
        z✝ : α
        ⊢ Or (Btw.btw (↑z✝) x₂ x₃) (Btw.btw x₃ x₂ ↑z✝)
      -/
      induction x₂ using QuotientAddGroup.induction_on
      /-
        case H.H
        α : Type u_1
        inst✝ : LinearOrderedAddCommGroup α
        hα : Archimedean α
        p : α
        hp : LT.lt 0 p
        a b c : α
        n : Int
        hp' : Fact (LT.lt 0 p)
        x₃ : HasQuotient.Quotient α (AddSubgroup.zmultiples p)
        z✝¹ z✝ : α
        ⊢ Or (Btw.btw (↑z✝¹) (↑z✝) x₃) (Btw.btw x₃ ↑z✝ ↑z✝¹)
      -/
      induction x₃ using QuotientAddGroup.induction_on
      /-
        case H.H.H
        α : Type u_1
        inst✝ : LinearOrderedAddCommGroup α
        hα : Archimedean α
        p : α
        hp : LT.lt 0 p
        a b c : α
        n : Int
        hp' : Fact (LT.lt 0 p)
        z✝² z✝¹ z✝ : α
        ⊢ Or (Btw.btw ↑z✝² ↑z✝¹ ↑z✝) (Btw.btw ↑z✝ ↑z✝¹ ↑z✝²)
      -/
      simp_rw [btw_coe_iff]
      /-
        case H.H.H
        α : Type u_1
        inst✝ : LinearOrderedAddCommGroup α
        hα : Archimedean α
        p : α
        hp : LT.lt 0 p
        a b c : α
        n : Int
        hp' : Fact (LT.lt 0 p)
        z✝² z✝¹ z✝ : α
        ⊢ Or (LE.le (toIcoMod ⋯ z✝² z✝¹) (toIocMod ⋯ z✝² z✝)) (LE.le (toIcoMod ⋯ z✝ z✝ …
      -/
      apply toIxxMod_total }
      /-
        🎉 no goals
      -/


theorem toIcoDiv_eq_floor (a b : α) : toIcoDiv hp a b = ⌊(b - a) / p⌋ := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrderedField α
    inst✝ : FloorRing α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (toIcoDiv hp a b) (Int.floor (HDiv.hDiv (HSub.hSub b a) p))
  -/
  refine toIcoDiv_eq_of_sub_zsmul_mem_Ico hp ?_
  rw [Set.mem_Ico, zsmul_eq_mul, ← sub_nonneg, add_comm, sub_right_comm, ← sub_lt_iff_lt_add,
    sub_right_comm _ _ a]
  /-
    α : Type u_1
    inst✝¹ : LinearOrderedField α
    inst✝ : FloorRing α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ And (LE.le 0 (HSub.hSub (HSub.hSub b a) (HMul.hMul (↑(Int.floor (HDiv.hDiv ( …
  -/
  exact ⟨Int.sub_floor_div_mul_nonneg _ hp, Int.sub_floor_div_mul_lt _ hp⟩
  /-
    🎉 no goals
  -/


theorem toIocDiv_eq_neg_floor (a b : α) : toIocDiv hp a b = -⌊(a + p - b) / p⌋ := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrderedField α
    inst✝ : FloorRing α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (toIocDiv hp a b) (Neg.neg (Int.floor (HDiv.hDiv (HSub.hSub (HAdd.hAdd a  …
  -/
  refine toIocDiv_eq_of_sub_zsmul_mem_Ioc hp ?_
  rw [Set.mem_Ioc, zsmul_eq_mul, Int.cast_neg, neg_mul, sub_neg_eq_add, ← sub_nonneg,
    sub_add_eq_sub_sub]
  /-
    α : Type u_1
    inst✝¹ : LinearOrderedField α
    inst✝ : FloorRing α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ And (LT.lt a (HAdd.hAdd b (HMul.hMul (↑(Int.floor (HDiv.hDiv (HSub.hSub (HAd …
  -/
  refine ⟨?_, Int.sub_floor_div_mul_nonneg _ hp⟩
  rw [← add_lt_add_iff_right p, add_assoc, add_comm b, ← sub_lt_iff_lt_add, add_comm (_ * _), ←
    sub_lt_iff_lt_add]
  /-
    α : Type u_1
    inst✝¹ : LinearOrderedField α
    inst✝ : FloorRing α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ LT.lt (HSub.hSub (HSub.hSub (HAdd.hAdd a p) b) (HMul.hMul (↑(Int.floor (HDiv …
  -/
  exact Int.sub_floor_div_mul_lt _ hp
  /-
    🎉 no goals
  -/


theorem toIcoDiv_zero_one (b : α) : toIcoDiv (zero_lt_one' α) 0 b = ⌊b⌋ := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrderedField α
    inst✝ : FloorRing α
    b : α
    ⊢ Eq (toIcoDiv ⋯ 0 b) (Int.floor b)
  -/
  simp [toIcoDiv_eq_floor]
  /-
    🎉 no goals
  -/


theorem toIcoMod_eq_add_fract_mul (a b : α) :
    toIcoMod hp a b = a + Int.fract ((b - a) / p) * p := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrderedField α
    inst✝ : FloorRing α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (toIcoMod hp a b) (HAdd.hAdd a (HMul.hMul (Int.fract (HDiv.hDiv (HSub.hSu …
  -/
  rw [toIcoMod, toIcoDiv_eq_floor, Int.fract]
  /-
    α : Type u_1
    inst✝¹ : LinearOrderedField α
    inst✝ : FloorRing α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (HSub.hSub b (HSMul.hSMul (Int.floor (HDiv.hDiv (HSub.hSub b a) p)) p)) ( …
  -/
  field_simp
  /-
    α : Type u_1
    inst✝¹ : LinearOrderedField α
    inst✝ : FloorRing α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (HSub.hSub b (HMul.hMul (↑(Int.floor (HDiv.hDiv (HSub.hSub b a) p))) p))  …
  -/
  ring
  /-
    🎉 no goals
  -/


theorem toIcoMod_eq_fract_mul (b : α) : toIcoMod hp 0 b = Int.fract (b / p) * p := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrderedField α
    inst✝ : FloorRing α
    p : α
    hp : LT.lt 0 p
    b : α
    ⊢ Eq (toIcoMod hp 0 b) (HMul.hMul (Int.fract (HDiv.hDiv b p)) p)
  -/
  simp [toIcoMod_eq_add_fract_mul]
  /-
    🎉 no goals
  -/


theorem toIocMod_eq_sub_fract_mul (a b : α) :
    toIocMod hp a b = a + p - Int.fract ((a + p - b) / p) * p := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrderedField α
    inst✝ : FloorRing α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (toIocMod hp a b) (HSub.hSub (HAdd.hAdd a p) (HMul.hMul (Int.fract (HDiv. …
  -/
  rw [toIocMod, toIocDiv_eq_neg_floor, Int.fract]
  /-
    α : Type u_1
    inst✝¹ : LinearOrderedField α
    inst✝ : FloorRing α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (HSub.hSub b (HSMul.hSMul (Neg.neg (Int.floor (HDiv.hDiv (HSub.hSub (HAdd …
  -/
  field_simp
  /-
    α : Type u_1
    inst✝¹ : LinearOrderedField α
    inst✝ : FloorRing α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Eq (HAdd.hAdd b (HMul.hMul (↑(Int.floor (HDiv.hDiv (HSub.hSub (HAdd.hAdd a p …
  -/
  ring
  /-
    🎉 no goals
  -/


theorem toIcoMod_zero_one (b : α) : toIcoMod (zero_lt_one' α) 0 b = Int.fract b := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrderedField α
    inst✝ : FloorRing α
    b : α
    ⊢ Eq (toIcoMod ⋯ 0 b) (Int.fract b)
  -/
  simp [toIcoMod_eq_add_fract_mul]
  /-
    🎉 no goals
  -/


theorem iUnion_Ioc_add_zsmul : ⋃ n : ℤ, Ioc (a + n • p) (a + (n + 1) • p) = univ := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrderedAddCommGroup α
    inst✝ : Archimedean α
    p : α
    hp : LT.lt 0 p
    a : α
    ⊢ Eq (Set.iUnion fun n => Set.Ioc (HAdd.hAdd a (HSMul.hSMul n p)) (HAdd.hAdd a …
  -/
  refine eq_univ_iff_forall.mpr fun b => mem_iUnion.mpr ?_
  /-
    α : Type u_1
    inst✝¹ : LinearOrderedAddCommGroup α
    inst✝ : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Exists fun i => Membership.mem (Set.Ioc (HAdd.hAdd a (HSMul.hSMul i p)) (HAd …
  -/
  rcases sub_toIocDiv_zsmul_mem_Ioc hp a b with ⟨hl, hr⟩
  /-
    case intro
    α : Type u_1
    inst✝¹ : LinearOrderedAddCommGroup α
    inst✝ : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    hl : LT.lt a (HSub.hSub b (HSMul.hSMul (toIocDiv hp a b) p))
    hr : LE.le (HSub.hSub b (HSMul.hSMul (toIocDiv hp a b) p)) (HAdd.hAdd a p)
    ⊢ Exists fun i => Membership.mem (Set.Ioc (HAdd.hAdd a (HSMul.hSMul i p)) (HAd …
  -/
  refine ⟨toIocDiv hp a b, ⟨lt_sub_iff_add_lt.mp hl, ?_⟩⟩
  /-
    case intro
    α : Type u_1
    inst✝¹ : LinearOrderedAddCommGroup α
    inst✝ : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    hl : LT.lt a (HSub.hSub b (HSMul.hSMul (toIocDiv hp a b) p))
    hr : LE.le (HSub.hSub b (HSMul.hSMul (toIocDiv hp a b) p)) (HAdd.hAdd a p)
    ⊢ LE.le b (HAdd.hAdd a (HSMul.hSMul (HAdd.hAdd (toIocDiv hp a b) 1) p))
  -/
  rw [add_smul, one_smul, ← add_assoc]
  /-
    case intro
    α : Type u_1
    inst✝¹ : LinearOrderedAddCommGroup α
    inst✝ : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    hl : LT.lt a (HSub.hSub b (HSMul.hSMul (toIocDiv hp a b) p))
    hr : LE.le (HSub.hSub b (HSMul.hSMul (toIocDiv hp a b) p)) (HAdd.hAdd a p)
    ⊢ LE.le b (HAdd.hAdd (HAdd.hAdd a (HSMul.hSMul (toIocDiv hp a b) p)) p)
  -/
                                           /-
                                             🎉 no goals
                                           -/
  convert sub_le_iff_le_add.mp hr using 1; abel
                                           /-
                                             🎉 no goals
                                           -/


theorem iUnion_Ico_add_zsmul : ⋃ n : ℤ, Ico (a + n • p) (a + (n + 1) • p) = univ := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrderedAddCommGroup α
    inst✝ : Archimedean α
    p : α
    hp : LT.lt 0 p
    a : α
    ⊢ Eq (Set.iUnion fun n => Set.Ico (HAdd.hAdd a (HSMul.hSMul n p)) (HAdd.hAdd a …
  -/
  refine eq_univ_iff_forall.mpr fun b => mem_iUnion.mpr ?_
  /-
    α : Type u_1
    inst✝¹ : LinearOrderedAddCommGroup α
    inst✝ : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    ⊢ Exists fun i => Membership.mem (Set.Ico (HAdd.hAdd a (HSMul.hSMul i p)) (HAd …
  -/
  rcases sub_toIcoDiv_zsmul_mem_Ico hp a b with ⟨hl, hr⟩
  /-
    case intro
    α : Type u_1
    inst✝¹ : LinearOrderedAddCommGroup α
    inst✝ : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    hl : LE.le a (HSub.hSub b (HSMul.hSMul (toIcoDiv hp a b) p))
    hr : LT.lt (HSub.hSub b (HSMul.hSMul (toIcoDiv hp a b) p)) (HAdd.hAdd a p)
    ⊢ Exists fun i => Membership.mem (Set.Ico (HAdd.hAdd a (HSMul.hSMul i p)) (HAd …
  -/
  refine ⟨toIcoDiv hp a b, ⟨le_sub_iff_add_le.mp hl, ?_⟩⟩
  /-
    case intro
    α : Type u_1
    inst✝¹ : LinearOrderedAddCommGroup α
    inst✝ : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    hl : LE.le a (HSub.hSub b (HSMul.hSMul (toIcoDiv hp a b) p))
    hr : LT.lt (HSub.hSub b (HSMul.hSMul (toIcoDiv hp a b) p)) (HAdd.hAdd a p)
    ⊢ LT.lt b (HAdd.hAdd a (HSMul.hSMul (HAdd.hAdd (toIcoDiv hp a b) 1) p))
  -/
  rw [add_smul, one_smul, ← add_assoc]
  /-
    case intro
    α : Type u_1
    inst✝¹ : LinearOrderedAddCommGroup α
    inst✝ : Archimedean α
    p : α
    hp : LT.lt 0 p
    a b : α
    hl : LE.le a (HSub.hSub b (HSMul.hSMul (toIcoDiv hp a b) p))
    hr : LT.lt (HSub.hSub b (HSMul.hSMul (toIcoDiv hp a b) p)) (HAdd.hAdd a p)
    ⊢ LT.lt b (HAdd.hAdd (HAdd.hAdd a (HSMul.hSMul (toIcoDiv hp a b) p)) p)
  -/
                                           /-
                                             🎉 no goals
                                           -/
  convert sub_lt_iff_lt_add.mp hr using 1; abel
                                           /-
                                             🎉 no goals
                                           -/


theorem iUnion_Icc_add_zsmul : ⋃ n : ℤ, Icc (a + n • p) (a + (n + 1) • p) = univ := by
  simpa only [iUnion_Ioc_add_zsmul hp a, univ_subset_iff] using
    iUnion_mono fun n : ℤ => (Ioc_subset_Icc_self : Ioc (a + n • p) (a + (n + 1) • p) ⊆ Icc _ _)


theorem iUnion_Ioc_zsmul : ⋃ n : ℤ, Ioc (n • p) ((n + 1) • p) = univ := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrderedAddCommGroup α
    inst✝ : Archimedean α
    p : α
    hp : LT.lt 0 p
    ⊢ Eq (Set.iUnion fun n => Set.Ioc (HSMul.hSMul n p) (HSMul.hSMul (HAdd.hAdd n  …
  -/
  simpa only [zero_add] using iUnion_Ioc_add_zsmul hp 0
  /-
    🎉 no goals
  -/


theorem iUnion_Ico_zsmul : ⋃ n : ℤ, Ico (n • p) ((n + 1) • p) = univ := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrderedAddCommGroup α
    inst✝ : Archimedean α
    p : α
    hp : LT.lt 0 p
    ⊢ Eq (Set.iUnion fun n => Set.Ico (HSMul.hSMul n p) (HSMul.hSMul (HAdd.hAdd n  …
  -/
  simpa only [zero_add] using iUnion_Ico_add_zsmul hp 0
  /-
    🎉 no goals
  -/


theorem iUnion_Icc_zsmul : ⋃ n : ℤ, Icc (n • p) ((n + 1) • p) = univ := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrderedAddCommGroup α
    inst✝ : Archimedean α
    p : α
    hp : LT.lt 0 p
    ⊢ Eq (Set.iUnion fun n => Set.Icc (HSMul.hSMul n p) (HSMul.hSMul (HAdd.hAdd n  …
  -/
  simpa only [zero_add] using iUnion_Icc_add_zsmul hp 0
  /-
    🎉 no goals
  -/


theorem iUnion_Ioc_add_intCast : ⋃ n : ℤ, Ioc (a + n) (a + n + 1) = Set.univ := by
  simpa only [zsmul_one, Int.cast_add, Int.cast_one, ← add_assoc] using
    iUnion_Ioc_add_zsmul zero_lt_one a


@[deprecated (since := "2024-04-17")]
alias iUnion_Ioc_add_int_cast := iUnion_Ioc_add_intCast


theorem iUnion_Ico_add_intCast : ⋃ n : ℤ, Ico (a + n) (a + n + 1) = Set.univ := by
  simpa only [zsmul_one, Int.cast_add, Int.cast_one, ← add_assoc] using
    iUnion_Ico_add_zsmul zero_lt_one a


@[deprecated (since := "2024-04-17")]
alias iUnion_Ico_add_int_cast := iUnion_Ico_add_intCast


theorem iUnion_Icc_add_intCast : ⋃ n : ℤ, Icc (a + n) (a + n + 1) = Set.univ := by
  simpa only [zsmul_one, Int.cast_add, Int.cast_one, ← add_assoc] using
    iUnion_Icc_add_zsmul zero_lt_one a


@[deprecated (since := "2024-04-17")]
alias iUnion_Icc_add_int_cast := iUnion_Icc_add_intCast


theorem iUnion_Ioc_intCast : ⋃ n : ℤ, Ioc (n : α) (n + 1) = Set.univ := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrderedRing α
    inst✝ : Archimedean α
    ⊢ Eq (Set.iUnion fun n => Set.Ioc (↑n) (HAdd.hAdd (↑n) 1)) Set.univ
  -/
  simpa only [zero_add] using iUnion_Ioc_add_intCast (0 : α)
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias iUnion_Ioc_int_cast := iUnion_Ioc_intCast


theorem iUnion_Ico_intCast : ⋃ n : ℤ, Ico (n : α) (n + 1) = Set.univ := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrderedRing α
    inst✝ : Archimedean α
    ⊢ Eq (Set.iUnion fun n => Set.Ico (↑n) (HAdd.hAdd (↑n) 1)) Set.univ
  -/
  simpa only [zero_add] using iUnion_Ico_add_intCast (0 : α)
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias iUnion_Ico_int_cast := iUnion_Ico_intCast


theorem iUnion_Icc_intCast : ⋃ n : ℤ, Icc (n : α) (n + 1) = Set.univ := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrderedRing α
    inst✝ : Archimedean α
    ⊢ Eq (Set.iUnion fun n => Set.Icc (↑n) (HAdd.hAdd (↑n) 1)) Set.univ
  -/
  simpa only [zero_add] using iUnion_Icc_add_intCast (0 : α)
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias iUnion_Icc_int_cast := iUnion_Icc_intCast


