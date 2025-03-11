/-- The symmetric difference operator on a type with `⊔` and `\` is `(A \ B) ⊔ (B \ A)`. -/
def symmDiff [Max α] [SDiff α] (a b : α) : α :=
  a \ b ⊔ b \ a


/-- The Heyting bi-implication is `(b ⇨ a) ⊓ (a ⇨ b)`. This generalizes equivalence of
propositions. -/
def bihimp [Min α] [HImp α] (a b : α) : α :=
  (b ⇨ a) ⊓ (a ⇨ b)


/-- Notation for symmDiff -/
scoped[symmDiff] infixl:100 " ∆ " => symmDiff


/-- Notation for bihimp -/
scoped[symmDiff] infixl:100 " ⇔ " => bihimp


theorem symmDiff_def [Max α] [SDiff α] (a b : α) : a ∆ b = a \ b ⊔ b \ a :=
  rfl


theorem bihimp_def [Min α] [HImp α] (a b : α) : a ⇔ b = (b ⇨ a) ⊓ (a ⇨ b) :=
  rfl


theorem symmDiff_eq_Xor' (p q : Prop) : p ∆ q = Xor' p q :=
  rfl


@[simp]
theorem bihimp_iff_iff {p q : Prop} : p ⇔ q ↔ (p ↔ q) :=
  iff_iff_implies_and_implies.symm.trans Iff.comm


@[simp]
                                                                   /-
                                                                     ⊢ ∀ (p q : Bool), Eq (symmDiff p q) (p.xor q)
                                                                   -/
theorem Bool.symmDiff_eq_xor : ∀ p q : Bool, p ∆ q = xor p q := by decide
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[simp]
theorem toDual_symmDiff : toDual (a ∆ b) = toDual a ⇔ toDual b :=
  rfl


@[simp]
theorem ofDual_bihimp (a b : αᵒᵈ) : ofDual (a ⇔ b) = ofDual a ∆ ofDual b :=
  rfl


                                            /-
                                              α : Type u_2
                                              inst✝ : GeneralizedCoheytingAlgebra α
                                              a b : α
                                              ⊢ Eq (symmDiff a b) (symmDiff b a)
                                            -/
theorem symmDiff_comm : a ∆ b = b ∆ a := by simp only [symmDiff, sup_comm]
                                            /-
                                              🎉 no goals
                                            -/


instance symmDiff_isCommutative : Std.Commutative (α := α) (· ∆ ·) :=
  ⟨symmDiff_comm⟩


@[simp]
                                        /-
                                          α : Type u_2
                                          inst✝ : GeneralizedCoheytingAlgebra α
                                          a : α
                                          ⊢ Eq (symmDiff a a) Bot.bot
                                        -/
theorem symmDiff_self : a ∆ a = ⊥ := by rw [symmDiff, sup_idem, sdiff_self]
                                        /-
                                          🎉 no goals
                                        -/


@[simp]
                                       /-
                                         α : Type u_2
                                         inst✝ : GeneralizedCoheytingAlgebra α
                                         a : α
                                         ⊢ Eq (symmDiff a Bot.bot) a
                                       -/
theorem symmDiff_bot : a ∆ ⊥ = a := by rw [symmDiff, sdiff_bot, bot_sdiff, sup_bot_eq]
                                       /-
                                         🎉 no goals
                                       -/


@[simp]
                                       /-
                                         α : Type u_2
                                         inst✝ : GeneralizedCoheytingAlgebra α
                                         a : α
                                         ⊢ Eq (symmDiff Bot.bot a) a
                                       -/
theorem bot_symmDiff : ⊥ ∆ a = a := by rw [symmDiff_comm, symmDiff_bot]
                                       /-
                                         🎉 no goals
                                       -/


@[simp]
theorem symmDiff_eq_bot {a b : α} : a ∆ b = ⊥ ↔ a = b := by
  /-
    α : Type u_2
    inst✝ : GeneralizedCoheytingAlgebra α
    a b : α
    ⊢ Iff (Eq (symmDiff a b) Bot.bot) (Eq a b)
  -/
  simp_rw [symmDiff, sup_eq_bot_iff, sdiff_eq_bot_iff, le_antisymm_iff]
  /-
    🎉 no goals
  -/


theorem symmDiff_of_le {a b : α} (h : a ≤ b) : a ∆ b = b \ a := by
  /-
    α : Type u_2
    inst✝ : GeneralizedCoheytingAlgebra α
    a b : α
    h : LE.le a b
    ⊢ Eq (symmDiff a b) (SDiff.sdiff b a)
  -/
  rw [symmDiff, sdiff_eq_bot_iff.2 h, bot_sup_eq]
  /-
    🎉 no goals
  -/


theorem symmDiff_of_ge {a b : α} (h : b ≤ a) : a ∆ b = a \ b := by
  /-
    α : Type u_2
    inst✝ : GeneralizedCoheytingAlgebra α
    a b : α
    h : LE.le b a
    ⊢ Eq (symmDiff a b) (SDiff.sdiff a b)
  -/
  rw [symmDiff, sdiff_eq_bot_iff.2 h, sup_bot_eq]
  /-
    🎉 no goals
  -/


theorem symmDiff_le {a b c : α} (ha : a ≤ b ⊔ c) (hb : b ≤ a ⊔ c) : a ∆ b ≤ c :=
  sup_le (sdiff_le_iff.2 ha) <| sdiff_le_iff.2 hb


theorem symmDiff_le_iff {a b c : α} : a ∆ b ≤ c ↔ a ≤ b ⊔ c ∧ b ≤ a ⊔ c := by
  /-
    α : Type u_2
    inst✝ : GeneralizedCoheytingAlgebra α
    a b c : α
    ⊢ Iff (LE.le (symmDiff a b) c) (And (LE.le a (Max.max b c)) (LE.le b (Max.max  …
  -/
  simp_rw [symmDiff, sup_le_iff, sdiff_le_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem symmDiff_le_sup {a b : α} : a ∆ b ≤ a ⊔ b :=
  sup_le_sup sdiff_le sdiff_le


                                                                    /-
                                                                      α : Type u_2
                                                                      inst✝ : GeneralizedCoheytingAlgebra α
                                                                      a b : α
                                                                      ⊢ Eq (symmDiff a b) (SDiff.sdiff (Max.max a b) (Min.min a b))
                                                                    -/
theorem symmDiff_eq_sup_sdiff_inf : a ∆ b = (a ⊔ b) \ (a ⊓ b) := by simp [sup_sdiff, symmDiff]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


theorem Disjoint.symmDiff_eq_sup {a b : α} (h : Disjoint a b) : a ∆ b = a ⊔ b := by
  /-
    α : Type u_2
    inst✝ : GeneralizedCoheytingAlgebra α
    a b : α
    h : Disjoint a b
    ⊢ Eq (symmDiff a b) (Max.max a b)
  -/
  rw [symmDiff, h.sdiff_eq_left, h.sdiff_eq_right]
  /-
    🎉 no goals
  -/


theorem symmDiff_sdiff : a ∆ b \ c = a \ (b ⊔ c) ⊔ b \ (a ⊔ c) := by
  /-
    α : Type u_2
    inst✝ : GeneralizedCoheytingAlgebra α
    a b c : α
    ⊢ Eq (SDiff.sdiff (symmDiff a b) c) (Max.max (SDiff.sdiff a (Max.max b c)) (SD …
  -/
  rw [symmDiff, sup_sdiff_distrib, sdiff_sdiff_left, sdiff_sdiff_left]
  /-
    🎉 no goals
  -/


@[simp]
theorem symmDiff_sdiff_inf : a ∆ b \ (a ⊓ b) = a ∆ b := by
  /-
    α : Type u_2
    inst✝ : GeneralizedCoheytingAlgebra α
    a b : α
    ⊢ Eq (SDiff.sdiff (symmDiff a b) (Min.min a b)) (symmDiff a b)
  -/
  rw [symmDiff_sdiff]
  /-
    α : Type u_2
    inst✝ : GeneralizedCoheytingAlgebra α
    a b : α
    ⊢ Eq (Max.max (SDiff.sdiff a (Max.max b (Min.min a b))) (SDiff.sdiff b (Max.ma …
  -/
  simp [symmDiff]
  /-
    🎉 no goals
  -/


@[simp]
theorem symmDiff_sdiff_eq_sup : a ∆ (b \ a) = a ⊔ b := by
  /-
    α : Type u_2
    inst✝ : GeneralizedCoheytingAlgebra α
    a b : α
    ⊢ Eq (symmDiff a (SDiff.sdiff b a)) (Max.max a b)
  -/
  rw [symmDiff, sdiff_idem]
  exact
    le_antisymm (sup_le_sup sdiff_le sdiff_le)
      (sup_le le_sdiff_sup <| le_sdiff_sup.trans <| sup_le le_sup_right le_sdiff_sup)


@[simp]
theorem sdiff_symmDiff_eq_sup : (a \ b) ∆ b = a ⊔ b := by
  /-
    α : Type u_2
    inst✝ : GeneralizedCoheytingAlgebra α
    a b : α
    ⊢ Eq (symmDiff (SDiff.sdiff a b) b) (Max.max a b)
  -/
  rw [symmDiff_comm, symmDiff_sdiff_eq_sup, sup_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem symmDiff_sup_inf : a ∆ b ⊔ a ⊓ b = a ⊔ b := by
  /-
    α : Type u_2
    inst✝ : GeneralizedCoheytingAlgebra α
    a b : α
    ⊢ Eq (Max.max (symmDiff a b) (Min.min a b)) (Max.max a b)
  -/
  refine le_antisymm (sup_le symmDiff_le_sup inf_le_sup) ?_
  /-
    α : Type u_2
    inst✝ : GeneralizedCoheytingAlgebra α
    a b : α
    ⊢ LE.le (Max.max a b) (Max.max (symmDiff a b) (Min.min a b))
  -/
  rw [sup_inf_left, symmDiff]
  /-
    α : Type u_2
    inst✝ : GeneralizedCoheytingAlgebra α
    a b : α
    ⊢ LE.le (Max.max a b) (Min.min (Max.max (Max.max (SDiff.sdiff a b) (SDiff.sdif …
  -/
  refine sup_le (le_inf le_sup_right ?_) (le_inf ?_ le_sup_right)
    /-
      case refine_1
      α : Type u_2
      inst✝ : GeneralizedCoheytingAlgebra α
      a b : α
      ⊢ LE.le a (Max.max (Max.max (SDiff.sdiff a b) (SDiff.sdiff b a)) b)
    -/
  · rw [sup_right_comm]
    /-
      case refine_1
      α : Type u_2
      inst✝ : GeneralizedCoheytingAlgebra α
      a b : α
      ⊢ LE.le a (Max.max (Max.max (SDiff.sdiff a b) b) (SDiff.sdiff b a))
    -/
    exact le_sup_of_le_left le_sdiff_sup
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_2
      inst✝ : GeneralizedCoheytingAlgebra α
      a b : α
      ⊢ LE.le b (Max.max (Max.max (SDiff.sdiff a b) (SDiff.sdiff b a)) a)
    -/
  · rw [sup_assoc]
    /-
      case refine_2
      α : Type u_2
      inst✝ : GeneralizedCoheytingAlgebra α
      a b : α
      ⊢ LE.le b (Max.max (SDiff.sdiff a b) (Max.max (SDiff.sdiff b a) a))
    -/
    exact le_sup_of_le_right le_sdiff_sup
    /-
      🎉 no goals
    -/


@[simp]
                                                       /-
                                                         α : Type u_2
                                                         inst✝ : GeneralizedCoheytingAlgebra α
                                                         a b : α
                                                         ⊢ Eq (Max.max (Min.min a b) (symmDiff a b)) (Max.max a b)
                                                       -/
theorem inf_sup_symmDiff : a ⊓ b ⊔ a ∆ b = a ⊔ b := by rw [sup_comm, symmDiff_sup_inf]
                                                       /-
                                                         🎉 no goals
                                                       -/


@[simp]
theorem symmDiff_symmDiff_inf : a ∆ b ∆ (a ⊓ b) = a ⊔ b := by
  /-
    α : Type u_2
    inst✝ : GeneralizedCoheytingAlgebra α
    a b : α
    ⊢ Eq (symmDiff (symmDiff a b) (Min.min a b)) (Max.max a b)
  -/
  rw [← symmDiff_sdiff_inf a, sdiff_symmDiff_eq_sup, symmDiff_sup_inf]
  /-
    🎉 no goals
  -/


@[simp]
theorem inf_symmDiff_symmDiff : (a ⊓ b) ∆ (a ∆ b) = a ⊔ b := by
  /-
    α : Type u_2
    inst✝ : GeneralizedCoheytingAlgebra α
    a b : α
    ⊢ Eq (symmDiff (Min.min a b) (symmDiff a b)) (Max.max a b)
  -/
  rw [symmDiff_comm, symmDiff_symmDiff_inf]
  /-
    🎉 no goals
  -/


theorem symmDiff_triangle : a ∆ c ≤ a ∆ b ⊔ b ∆ c := by
  /-
    α : Type u_2
    inst✝ : GeneralizedCoheytingAlgebra α
    a b c : α
    ⊢ LE.le (symmDiff a c) (Max.max (symmDiff a b) (symmDiff b c))
  -/
  refine (sup_le_sup (sdiff_triangle a b c) <| sdiff_triangle _ b _).trans_eq ?_
  /-
    α : Type u_2
    inst✝ : GeneralizedCoheytingAlgebra α
    a b c : α
    ⊢ Eq (Max.max (Max.max (SDiff.sdiff a b) (SDiff.sdiff b c)) (Max.max (SDiff.sd …
  -/
  rw [sup_comm (c \ b), sup_sup_sup_comm, symmDiff, symmDiff]
  /-
    🎉 no goals
  -/


theorem le_symmDiff_sup_right (a b : α) : a ≤ (a ∆ b) ⊔ b := by
  /-
    α : Type u_2
    inst✝ : GeneralizedCoheytingAlgebra α
    a b : α
    ⊢ LE.le a (Max.max (symmDiff a b) b)
  -/
                                      /-
                                        🎉 no goals
                                      -/
  convert symmDiff_triangle a b ⊥ <;> rw [symmDiff_bot]
                                      /-
                                        🎉 no goals
                                      -/


theorem le_symmDiff_sup_left (a b : α) : b ≤ (a ∆ b) ⊔ a :=
  symmDiff_comm a b ▸ le_symmDiff_sup_right ..


@[simp]
theorem toDual_bihimp : toDual (a ⇔ b) = toDual a ∆ toDual b :=
  rfl


@[simp]
theorem ofDual_symmDiff (a b : αᵒᵈ) : ofDual (a ∆ b) = ofDual a ⇔ ofDual b :=
  rfl


                                          /-
                                            α : Type u_2
                                            inst✝ : GeneralizedHeytingAlgebra α
                                            a b : α
                                            ⊢ Eq (bihimp a b) (bihimp b a)
                                          -/
theorem bihimp_comm : a ⇔ b = b ⇔ a := by simp only [(· ⇔ ·), inf_comm]
                                          /-
                                            🎉 no goals
                                          -/


instance bihimp_isCommutative : Std.Commutative (α := α) (· ⇔ ·) :=
  ⟨bihimp_comm⟩


@[simp]
                                      /-
                                        α : Type u_2
                                        inst✝ : GeneralizedHeytingAlgebra α
                                        a : α
                                        ⊢ Eq (bihimp a a) Top.top
                                      -/
theorem bihimp_self : a ⇔ a = ⊤ := by rw [bihimp, inf_idem, himp_self]
                                      /-
                                        🎉 no goals
                                      -/


@[simp]
                                     /-
                                       α : Type u_2
                                       inst✝ : GeneralizedHeytingAlgebra α
                                       a : α
                                       ⊢ Eq (bihimp a Top.top) a
                                     -/
theorem bihimp_top : a ⇔ ⊤ = a := by rw [bihimp, himp_top, top_himp, inf_top_eq]
                                     /-
                                       🎉 no goals
                                     -/


@[simp]
                                     /-
                                       α : Type u_2
                                       inst✝ : GeneralizedHeytingAlgebra α
                                       a : α
                                       ⊢ Eq (bihimp Top.top a) a
                                     -/
theorem top_bihimp : ⊤ ⇔ a = a := by rw [bihimp_comm, bihimp_top]
                                     /-
                                       🎉 no goals
                                     -/


@[simp]
theorem bihimp_eq_top {a b : α} : a ⇔ b = ⊤ ↔ a = b :=
  @symmDiff_eq_bot αᵒᵈ _ _ _


theorem bihimp_of_le {a b : α} (h : a ≤ b) : a ⇔ b = b ⇨ a := by
  /-
    α : Type u_2
    inst✝ : GeneralizedHeytingAlgebra α
    a b : α
    h : LE.le a b
    ⊢ Eq (bihimp a b) (HImp.himp b a)
  -/
  rw [bihimp, himp_eq_top_iff.2 h, inf_top_eq]
  /-
    🎉 no goals
  -/


theorem bihimp_of_ge {a b : α} (h : b ≤ a) : a ⇔ b = a ⇨ b := by
  /-
    α : Type u_2
    inst✝ : GeneralizedHeytingAlgebra α
    a b : α
    h : LE.le b a
    ⊢ Eq (bihimp a b) (HImp.himp a b)
  -/
  rw [bihimp, himp_eq_top_iff.2 h, top_inf_eq]
  /-
    🎉 no goals
  -/


theorem le_bihimp {a b c : α} (hb : a ⊓ b ≤ c) (hc : a ⊓ c ≤ b) : a ≤ b ⇔ c :=
  le_inf (le_himp_iff.2 hc) <| le_himp_iff.2 hb


theorem le_bihimp_iff {a b c : α} : a ≤ b ⇔ c ↔ a ⊓ b ≤ c ∧ a ⊓ c ≤ b := by
  /-
    α : Type u_2
    inst✝ : GeneralizedHeytingAlgebra α
    a b c : α
    ⊢ Iff (LE.le a (bihimp b c)) (And (LE.le (Min.min a b) c) (LE.le (Min.min a c) …
  -/
  simp_rw [bihimp, le_inf_iff, le_himp_iff, and_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem inf_le_bihimp {a b : α} : a ⊓ b ≤ a ⇔ b :=
  inf_le_inf le_himp le_himp


                                                             /-
                                                               α : Type u_2
                                                               inst✝ : GeneralizedHeytingAlgebra α
                                                               a b : α
                                                               ⊢ Eq (bihimp a b) (HImp.himp (Max.max a b) (Min.min a b))
                                                             -/
theorem bihimp_eq_inf_himp_inf : a ⇔ b = a ⊔ b ⇨ a ⊓ b := by simp [himp_inf_distrib, bihimp]
                                                             /-
                                                               🎉 no goals
                                                             -/


theorem Codisjoint.bihimp_eq_inf {a b : α} (h : Codisjoint a b) : a ⇔ b = a ⊓ b := by
  /-
    α : Type u_2
    inst✝ : GeneralizedHeytingAlgebra α
    a b : α
    h : Codisjoint a b
    ⊢ Eq (bihimp a b) (Min.min a b)
  -/
  rw [bihimp, h.himp_eq_left, h.himp_eq_right]
  /-
    🎉 no goals
  -/


theorem himp_bihimp : a ⇨ b ⇔ c = (a ⊓ c ⇨ b) ⊓ (a ⊓ b ⇨ c) := by
  /-
    α : Type u_2
    inst✝ : GeneralizedHeytingAlgebra α
    a b c : α
    ⊢ Eq (HImp.himp a (bihimp b c)) (Min.min (HImp.himp (Min.min a c) b) (HImp.him …
  -/
  rw [bihimp, himp_inf_distrib, himp_himp, himp_himp]
  /-
    🎉 no goals
  -/


@[simp]
theorem sup_himp_bihimp : a ⊔ b ⇨ a ⇔ b = a ⇔ b := by
  /-
    α : Type u_2
    inst✝ : GeneralizedHeytingAlgebra α
    a b : α
    ⊢ Eq (HImp.himp (Max.max a b) (bihimp a b)) (bihimp a b)
  -/
  rw [himp_bihimp]
  /-
    α : Type u_2
    inst✝ : GeneralizedHeytingAlgebra α
    a b : α
    ⊢ Eq (Min.min (HImp.himp (Min.min (Max.max a b) b) a) (HImp.himp (Min.min (Max …
  -/
  simp [bihimp]
  /-
    🎉 no goals
  -/


@[simp]
theorem bihimp_himp_eq_inf : a ⇔ (a ⇨ b) = a ⊓ b :=
  @symmDiff_sdiff_eq_sup αᵒᵈ _ _ _


@[simp]
theorem himp_bihimp_eq_inf : (b ⇨ a) ⇔ b = a ⊓ b :=
  @sdiff_symmDiff_eq_sup αᵒᵈ _ _ _


@[simp]
theorem bihimp_inf_sup : a ⇔ b ⊓ (a ⊔ b) = a ⊓ b :=
  @symmDiff_sup_inf αᵒᵈ _ _ _


@[simp]
theorem sup_inf_bihimp : (a ⊔ b) ⊓ a ⇔ b = a ⊓ b :=
  @inf_sup_symmDiff αᵒᵈ _ _ _


@[simp]
theorem bihimp_bihimp_sup : a ⇔ b ⇔ (a ⊔ b) = a ⊓ b :=
  @symmDiff_symmDiff_inf αᵒᵈ _ _ _


@[simp]
theorem sup_bihimp_bihimp : (a ⊔ b) ⇔ (a ⇔ b) = a ⊓ b :=
  @inf_symmDiff_symmDiff αᵒᵈ _ _ _


theorem bihimp_triangle : a ⇔ b ⊓ b ⇔ c ≤ a ⇔ c :=
  @symmDiff_triangle αᵒᵈ _ _ _ _


@[simp]
                                         /-
                                           α : Type u_2
                                           inst✝ : CoheytingAlgebra α
                                           a : α
                                           ⊢ Eq (symmDiff a Top.top) (HNot.hnot a)
                                         -/
theorem symmDiff_top' : a ∆ ⊤ = ￢a := by simp [symmDiff]
                                         /-
                                           🎉 no goals
                                         -/


@[simp]
                                         /-
                                           α : Type u_2
                                           inst✝ : CoheytingAlgebra α
                                           a : α
                                           ⊢ Eq (symmDiff Top.top a) (HNot.hnot a)
                                         -/
theorem top_symmDiff' : ⊤ ∆ a = ￢a := by simp [symmDiff]
                                         /-
                                           🎉 no goals
                                         -/


@[simp]
theorem hnot_symmDiff_self : (￢a) ∆ a = ⊤ := by
  /-
    α : Type u_2
    inst✝ : CoheytingAlgebra α
    a : α
    ⊢ Eq (symmDiff (HNot.hnot a) a) Top.top
  -/
  rw [eq_top_iff, symmDiff, hnot_sdiff, sup_sdiff_self]
  /-
    α : Type u_2
    inst✝ : CoheytingAlgebra α
    a : α
    ⊢ LE.le Top.top (Max.max (HNot.hnot a) a)
  -/
  exact Codisjoint.top_le codisjoint_hnot_left
  /-
    🎉 no goals
  -/


@[simp]
                                                /-
                                                  α : Type u_2
                                                  inst✝ : CoheytingAlgebra α
                                                  a : α
                                                  ⊢ Eq (symmDiff a (HNot.hnot a)) Top.top
                                                -/
theorem symmDiff_hnot_self : a ∆ (￢a) = ⊤ := by rw [symmDiff_comm, hnot_symmDiff_self]
                                                /-
                                                  🎉 no goals
                                                -/


theorem IsCompl.symmDiff_eq_top {a b : α} (h : IsCompl a b) : a ∆ b = ⊤ := by
  /-
    α : Type u_2
    inst✝ : CoheytingAlgebra α
    a b : α
    h : IsCompl a b
    ⊢ Eq (symmDiff a b) Top.top
  -/
  rw [h.eq_hnot, hnot_symmDiff_self]
  /-
    🎉 no goals
  -/


@[simp]
                                      /-
                                        α : Type u_2
                                        inst✝ : HeytingAlgebra α
                                        a : α
                                        ⊢ Eq (bihimp a Bot.bot) (HasCompl.compl a)
                                      -/
theorem bihimp_bot : a ⇔ ⊥ = aᶜ := by simp [bihimp]
                                      /-
                                        🎉 no goals
                                      -/


@[simp]
                                      /-
                                        α : Type u_2
                                        inst✝ : HeytingAlgebra α
                                        a : α
                                        ⊢ Eq (bihimp Bot.bot a) (HasCompl.compl a)
                                      -/
theorem bot_bihimp : ⊥ ⇔ a = aᶜ := by simp [bihimp]
                                      /-
                                        🎉 no goals
                                      -/


@[simp]
theorem compl_bihimp_self : aᶜ ⇔ a = ⊥ :=
  @hnot_symmDiff_self αᵒᵈ _ _


@[simp]
theorem bihimp_hnot_self : a ⇔ aᶜ = ⊥ :=
  @symmDiff_hnot_self αᵒᵈ _ _


theorem IsCompl.bihimp_eq_bot {a b : α} (h : IsCompl a b) : a ⇔ b = ⊥ := by
  /-
    α : Type u_2
    inst✝ : HeytingAlgebra α
    a b : α
    h : IsCompl a b
    ⊢ Eq (bihimp a b) Bot.bot
  -/
  rw [h.eq_compl, compl_bihimp_self]
  /-
    🎉 no goals
  -/


@[simp]
theorem sup_sdiff_symmDiff : (a ⊔ b) \ a ∆ b = a ⊓ b :=
                               /-
                                 α : Type u_2
                                 inst✝ : GeneralizedBooleanAlgebra α
                                 a b : α
                                 ⊢ Eq (SDiff.sdiff (Max.max a b) (Min.min a b)) (symmDiff a b)
                               -/
  sdiff_eq_symm inf_le_sup (by rw [symmDiff_eq_sup_sdiff_inf])
                               /-
                                 🎉 no goals
                               -/


theorem disjoint_symmDiff_inf : Disjoint (a ∆ b) (a ⊓ b) := by
  /-
    α : Type u_2
    inst✝ : GeneralizedBooleanAlgebra α
    a b : α
    ⊢ Disjoint (symmDiff a b) (Min.min a b)
  -/
  rw [symmDiff_eq_sup_sdiff_inf]
  /-
    α : Type u_2
    inst✝ : GeneralizedBooleanAlgebra α
    a b : α
    ⊢ Disjoint (SDiff.sdiff (Max.max a b) (Min.min a b)) (Min.min a b)
  -/
  exact disjoint_sdiff_self_left
  /-
    🎉 no goals
  -/


theorem inf_symmDiff_distrib_left : a ⊓ b ∆ c = (a ⊓ b) ∆ (a ⊓ c) := by
  rw [symmDiff_eq_sup_sdiff_inf, inf_sdiff_distrib_left, inf_sup_left, inf_inf_distrib_left,
    symmDiff_eq_sup_sdiff_inf]


theorem inf_symmDiff_distrib_right : a ∆ b ⊓ c = (a ⊓ c) ∆ (b ⊓ c) := by
  /-
    α : Type u_2
    inst✝ : GeneralizedBooleanAlgebra α
    a b c : α
    ⊢ Eq (Min.min (symmDiff a b) c) (symmDiff (Min.min a c) (Min.min b c))
  -/
  simp_rw [inf_comm _ c, inf_symmDiff_distrib_left]
  /-
    🎉 no goals
  -/


theorem sdiff_symmDiff : c \ a ∆ b = c ⊓ a ⊓ b ⊔ c \ a ⊓ c \ b := by
  /-
    α : Type u_2
    inst✝ : GeneralizedBooleanAlgebra α
    a b c : α
    ⊢ Eq (SDiff.sdiff c (symmDiff a b)) (Max.max (Min.min (Min.min c a) b) (Min.mi …
  -/
  simp only [(· ∆ ·), sdiff_sdiff_sup_sdiff']
  /-
    🎉 no goals
  -/


theorem sdiff_symmDiff' : c \ a ∆ b = c ⊓ a ⊓ b ⊔ c \ (a ⊔ b) := by
  /-
    α : Type u_2
    inst✝ : GeneralizedBooleanAlgebra α
    a b c : α
    ⊢ Eq (SDiff.sdiff c (symmDiff a b)) (Max.max (Min.min (Min.min c a) b) (SDiff. …
  -/
  rw [sdiff_symmDiff, sdiff_sup]
  /-
    🎉 no goals
  -/


@[simp]
theorem symmDiff_sdiff_left : a ∆ b \ a = b \ a := by
  /-
    α : Type u_2
    inst✝ : GeneralizedBooleanAlgebra α
    a b : α
    ⊢ Eq (SDiff.sdiff (symmDiff a b) a) (SDiff.sdiff b a)
  -/
  rw [symmDiff_def, sup_sdiff, sdiff_idem, sdiff_sdiff_self, bot_sup_eq]
  /-
    🎉 no goals
  -/


@[simp]
                                                       /-
                                                         α : Type u_2
                                                         inst✝ : GeneralizedBooleanAlgebra α
                                                         a b : α
                                                         ⊢ Eq (SDiff.sdiff (symmDiff a b) b) (SDiff.sdiff a b)
                                                       -/
theorem symmDiff_sdiff_right : a ∆ b \ b = a \ b := by rw [symmDiff_comm, symmDiff_sdiff_left]
                                                       /-
                                                         🎉 no goals
                                                       -/


@[simp]
                                                      /-
                                                        α : Type u_2
                                                        inst✝ : GeneralizedBooleanAlgebra α
                                                        a b : α
                                                        ⊢ Eq (SDiff.sdiff a (symmDiff a b)) (Min.min a b)
                                                      -/
theorem sdiff_symmDiff_left : a \ a ∆ b = a ⊓ b := by simp [sdiff_symmDiff]
                                                      /-
                                                        🎉 no goals
                                                      -/


@[simp]
theorem sdiff_symmDiff_right : b \ a ∆ b = a ⊓ b := by
  /-
    α : Type u_2
    inst✝ : GeneralizedBooleanAlgebra α
    a b : α
    ⊢ Eq (SDiff.sdiff b (symmDiff a b)) (Min.min a b)
  -/
  rw [symmDiff_comm, inf_comm, sdiff_symmDiff_left]
  /-
    🎉 no goals
  -/


theorem symmDiff_eq_sup : a ∆ b = a ⊔ b ↔ Disjoint a b := by
  /-
    α : Type u_2
    inst✝ : GeneralizedBooleanAlgebra α
    a b : α
    ⊢ Iff (Eq (symmDiff a b) (Max.max a b)) (Disjoint a b)
  -/
  refine ⟨fun h => ?_, Disjoint.symmDiff_eq_sup⟩
  /-
    α : Type u_2
    inst✝ : GeneralizedBooleanAlgebra α
    a b : α
    h : Eq (symmDiff a b) (Max.max a b)
    ⊢ Disjoint a b
  -/
  rw [symmDiff_eq_sup_sdiff_inf, sdiff_eq_self_iff_disjoint] at h
  /-
    α : Type u_2
    inst✝ : GeneralizedBooleanAlgebra α
    a b : α
    h : Disjoint (Min.min a b) (Max.max a b)
    ⊢ Disjoint a b
  -/
  exact h.of_disjoint_inf_of_le le_sup_left
  /-
    🎉 no goals
  -/


@[simp]
theorem le_symmDiff_iff_left : a ≤ a ∆ b ↔ Disjoint a b := by
  /-
    α : Type u_2
    inst✝ : GeneralizedBooleanAlgebra α
    a b : α
    ⊢ Iff (LE.le a (symmDiff a b)) (Disjoint a b)
  -/
  refine ⟨fun h => ?_, fun h => h.symmDiff_eq_sup.symm ▸ le_sup_left⟩
  /-
    α : Type u_2
    inst✝ : GeneralizedBooleanAlgebra α
    a b : α
    h : LE.le a (symmDiff a b)
    ⊢ Disjoint a b
  -/
  rw [symmDiff_eq_sup_sdiff_inf] at h
  /-
    α : Type u_2
    inst✝ : GeneralizedBooleanAlgebra α
    a b : α
    h : LE.le a (SDiff.sdiff (Max.max a b) (Min.min a b))
    ⊢ Disjoint a b
  -/
  exact disjoint_iff_inf_le.mpr (le_sdiff_iff.1 <| inf_le_of_left_le h).le
  /-
    🎉 no goals
  -/


@[simp]
theorem le_symmDiff_iff_right : b ≤ a ∆ b ↔ Disjoint a b := by
  /-
    α : Type u_2
    inst✝ : GeneralizedBooleanAlgebra α
    a b : α
    ⊢ Iff (LE.le b (symmDiff a b)) (Disjoint a b)
  -/
  rw [symmDiff_comm, le_symmDiff_iff_left, disjoint_comm]
  /-
    🎉 no goals
  -/


theorem symmDiff_symmDiff_left :
    a ∆ b ∆ c = a \ (b ⊔ c) ⊔ b \ (a ⊔ c) ⊔ c \ (a ⊔ b) ⊔ a ⊓ b ⊓ c :=
  calc
    a ∆ b ∆ c = a ∆ b \ c ⊔ c \ a ∆ b := symmDiff_def _ _
    _ = a \ (b ⊔ c) ⊔ b \ (a ⊔ c) ⊔ (c \ (a ⊔ b) ⊔ c ⊓ a ⊓ b) := by
        /-
          α : Type u_2
          inst✝ : GeneralizedBooleanAlgebra α
          a b c : α
          ⊢ Eq (Max.max (SDiff.sdiff (symmDiff a b) c) (SDiff.sdiff c (symmDiff a b))) ( …
        -/
        { rw [sdiff_symmDiff', sup_comm (c ⊓ a ⊓ b), symmDiff_sdiff] }
        /-
          🎉 no goals
        -/
                                                                  /-
                                                                    α : Type u_2
                                                                    inst✝ : GeneralizedBooleanAlgebra α
                                                                    a b c : α
                                                                    ⊢ Eq (Max.max (Max.max (SDiff.sdiff a (Max.max b c)) (SDiff.sdiff b (Max.max a …
                                                                  -/
    _ = a \ (b ⊔ c) ⊔ b \ (a ⊔ c) ⊔ c \ (a ⊔ b) ⊔ a ⊓ b ⊓ c := by ac_rfl
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


theorem symmDiff_symmDiff_right :
    a ∆ (b ∆ c) = a \ (b ⊔ c) ⊔ b \ (a ⊔ c) ⊔ c \ (a ⊔ b) ⊔ a ⊓ b ⊓ c :=
  calc
    a ∆ (b ∆ c) = a \ b ∆ c ⊔ b ∆ c \ a := symmDiff_def _ _
    _ = a \ (b ⊔ c) ⊔ a ⊓ b ⊓ c ⊔ (b \ (c ⊔ a) ⊔ c \ (b ⊔ a)) := by
        /-
          α : Type u_2
          inst✝ : GeneralizedBooleanAlgebra α
          a b c : α
          ⊢ Eq (Max.max (SDiff.sdiff a (symmDiff b c)) (SDiff.sdiff (symmDiff b c) a)) ( …
        -/
        { rw [sdiff_symmDiff', sup_comm (a ⊓ b ⊓ c), symmDiff_sdiff] }
        /-
          🎉 no goals
        -/
                                                                  /-
                                                                    α : Type u_2
                                                                    inst✝ : GeneralizedBooleanAlgebra α
                                                                    a b c : α
                                                                    ⊢ Eq (Max.max (Max.max (SDiff.sdiff a (Max.max b c)) (Min.min (Min.min a b) c) …
                                                                  -/
    _ = a \ (b ⊔ c) ⊔ b \ (a ⊔ c) ⊔ c \ (a ⊔ b) ⊔ a ⊓ b ⊓ c := by ac_rfl
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


theorem symmDiff_assoc : a ∆ b ∆ c = a ∆ (b ∆ c) := by
  /-
    α : Type u_2
    inst✝ : GeneralizedBooleanAlgebra α
    a b c : α
    ⊢ Eq (symmDiff (symmDiff a b) c) (symmDiff a (symmDiff b c))
  -/
  rw [symmDiff_symmDiff_left, symmDiff_symmDiff_right]
  /-
    🎉 no goals
  -/


instance symmDiff_isAssociative : Std.Associative (α := α) (· ∆ ·) :=
  ⟨symmDiff_assoc⟩


theorem symmDiff_left_comm : a ∆ (b ∆ c) = b ∆ (a ∆ c) := by
  /-
    α : Type u_2
    inst✝ : GeneralizedBooleanAlgebra α
    a b c : α
    ⊢ Eq (symmDiff a (symmDiff b c)) (symmDiff b (symmDiff a c))
  -/
  simp_rw [← symmDiff_assoc, symmDiff_comm]
  /-
    🎉 no goals
  -/


                                                          /-
                                                            α : Type u_2
                                                            inst✝ : GeneralizedBooleanAlgebra α
                                                            a b c : α
                                                            ⊢ Eq (symmDiff (symmDiff a b) c) (symmDiff (symmDiff a c) b)
                                                          -/
theorem symmDiff_right_comm : a ∆ b ∆ c = a ∆ c ∆ b := by simp_rw [symmDiff_assoc, symmDiff_comm]
                                                          /-
                                                            🎉 no goals
                                                          -/


theorem symmDiff_symmDiff_symmDiff_comm : a ∆ b ∆ (c ∆ d) = a ∆ c ∆ (b ∆ d) := by
  /-
    α : Type u_2
    inst✝ : GeneralizedBooleanAlgebra α
    a b c d : α
    ⊢ Eq (symmDiff (symmDiff a b) (symmDiff c d)) (symmDiff (symmDiff a c) (symmDi …
  -/
  simp_rw [symmDiff_assoc, symmDiff_left_comm]
  /-
    🎉 no goals
  -/


@[simp]
                                                              /-
                                                                α : Type u_2
                                                                inst✝ : GeneralizedBooleanAlgebra α
                                                                a b : α
                                                                ⊢ Eq (symmDiff a (symmDiff a b)) b
                                                              -/
theorem symmDiff_symmDiff_cancel_left : a ∆ (a ∆ b) = b := by simp [← symmDiff_assoc]
                                                              /-
                                                                🎉 no goals
                                                              -/


@[simp]
                                                             /-
                                                               α : Type u_2
                                                               inst✝ : GeneralizedBooleanAlgebra α
                                                               a b : α
                                                               ⊢ Eq (symmDiff (symmDiff b a) a) b
                                                             -/
theorem symmDiff_symmDiff_cancel_right : b ∆ a ∆ a = b := by simp [symmDiff_assoc]
                                                             /-
                                                               🎉 no goals
                                                             -/


@[simp]
theorem symmDiff_symmDiff_self' : a ∆ b ∆ a = b := by
  /-
    α : Type u_2
    inst✝ : GeneralizedBooleanAlgebra α
    a b : α
    ⊢ Eq (symmDiff (symmDiff a b) a) b
  -/
  rw [symmDiff_comm, symmDiff_symmDiff_cancel_left]
  /-
    🎉 no goals
  -/


theorem symmDiff_left_involutive (a : α) : Involutive (· ∆ a) :=
  symmDiff_symmDiff_cancel_right _


theorem symmDiff_right_involutive (a : α) : Involutive (a ∆ ·) :=
  symmDiff_symmDiff_cancel_left _


theorem symmDiff_left_injective (a : α) : Injective (· ∆ a) :=
  Function.Involutive.injective (symmDiff_left_involutive a)


theorem symmDiff_right_injective (a : α) : Injective (a ∆ ·) :=
  Function.Involutive.injective (symmDiff_right_involutive _)


theorem symmDiff_left_surjective (a : α) : Surjective (· ∆ a) :=
  Function.Involutive.surjective (symmDiff_left_involutive _)


theorem symmDiff_right_surjective (a : α) : Surjective (a ∆ ·) :=
  Function.Involutive.surjective (symmDiff_right_involutive _)


@[simp]
theorem symmDiff_left_inj : a ∆ b = c ∆ b ↔ a = c :=
  (symmDiff_left_injective _).eq_iff


@[simp]
theorem symmDiff_right_inj : a ∆ b = a ∆ c ↔ b = c :=
  (symmDiff_right_injective _).eq_iff


@[simp]
theorem symmDiff_eq_left : a ∆ b = a ↔ b = ⊥ :=
  calc
                                    /-
                                      α : Type u_2
                                      inst✝ : GeneralizedBooleanAlgebra α
                                      a b : α
                                      ⊢ Iff (Eq (symmDiff a b) a) (Eq (symmDiff a b) (symmDiff a Bot.bot))
                                    -/
    a ∆ b = a ↔ a ∆ b = a ∆ ⊥ := by rw [symmDiff_bot]
                                    /-
                                      🎉 no goals
                                    -/
                    /-
                      α : Type u_2
                      inst✝ : GeneralizedBooleanAlgebra α
                      a b : α
                      ⊢ Iff (Eq (symmDiff a b) (symmDiff a Bot.bot)) (Eq b Bot.bot)
                    -/
    _ ↔ b = ⊥ := by rw [symmDiff_right_inj]
                    /-
                      🎉 no goals
                    -/


@[simp]
                                                    /-
                                                      α : Type u_2
                                                      inst✝ : GeneralizedBooleanAlgebra α
                                                      a b : α
                                                      ⊢ Iff (Eq (symmDiff a b) b) (Eq a Bot.bot)
                                                    -/
theorem symmDiff_eq_right : a ∆ b = b ↔ a = ⊥ := by rw [symmDiff_comm, symmDiff_eq_left]
                                                    /-
                                                      🎉 no goals
                                                    -/


protected theorem Disjoint.symmDiff_left (ha : Disjoint a c) (hb : Disjoint b c) :
    Disjoint (a ∆ b) c := by
  /-
    α : Type u_2
    inst✝ : GeneralizedBooleanAlgebra α
    a b c : α
    ha : Disjoint a c
    hb : Disjoint b c
    ⊢ Disjoint (symmDiff a b) c
  -/
  rw [symmDiff_eq_sup_sdiff_inf]
  /-
    α : Type u_2
    inst✝ : GeneralizedBooleanAlgebra α
    a b c : α
    ha : Disjoint a c
    hb : Disjoint b c
    ⊢ Disjoint (SDiff.sdiff (Max.max a b) (Min.min a b)) c
  -/
  exact (ha.sup_left hb).disjoint_sdiff_left
  /-
    🎉 no goals
  -/


protected theorem Disjoint.symmDiff_right (ha : Disjoint a b) (hb : Disjoint a c) :
    Disjoint a (b ∆ c) :=
  (ha.symm.symmDiff_left hb.symm).symm


theorem symmDiff_eq_iff_sdiff_eq (ha : a ≤ c) : a ∆ b = c ↔ c \ a = b := by
  /-
    α : Type u_2
    inst✝ : GeneralizedBooleanAlgebra α
    a b c : α
    ha : LE.le a c
    ⊢ Iff (Eq (symmDiff a b) c) (Eq (SDiff.sdiff c a) b)
  -/
  rw [← symmDiff_of_le ha]
  /-
    α : Type u_2
    inst✝ : GeneralizedBooleanAlgebra α
    a b c : α
    ha : LE.le a c
    ⊢ Iff (Eq (symmDiff a b) c) (Eq (symmDiff a c) b)
  -/
  exact ((symmDiff_right_involutive a).toPerm _).apply_eq_iff_eq_symm_apply.trans eq_comm
  /-
    🎉 no goals
  -/


@[simp]
theorem inf_himp_bihimp : a ⇔ b ⇨ a ⊓ b = a ⊔ b :=
  @sup_sdiff_symmDiff αᵒᵈ _ _ _


theorem codisjoint_bihimp_sup : Codisjoint (a ⇔ b) (a ⊔ b) :=
  @disjoint_symmDiff_inf αᵒᵈ _ _ _


@[simp]
theorem himp_bihimp_left : a ⇨ a ⇔ b = a ⇨ b :=
  @symmDiff_sdiff_left αᵒᵈ _ _ _


@[simp]
theorem himp_bihimp_right : b ⇨ a ⇔ b = b ⇨ a :=
  @symmDiff_sdiff_right αᵒᵈ _ _ _


@[simp]
theorem bihimp_himp_left : a ⇔ b ⇨ a = a ⊔ b :=
  @sdiff_symmDiff_left αᵒᵈ _ _ _


@[simp]
theorem bihimp_himp_right : a ⇔ b ⇨ b = a ⊔ b :=
  @sdiff_symmDiff_right αᵒᵈ _ _ _


@[simp]
theorem bihimp_eq_inf : a ⇔ b = a ⊓ b ↔ Codisjoint a b :=
  @symmDiff_eq_sup αᵒᵈ _ _ _


@[simp]
theorem bihimp_le_iff_left : a ⇔ b ≤ a ↔ Codisjoint a b :=
  @le_symmDiff_iff_left αᵒᵈ _ _ _


@[simp]
theorem bihimp_le_iff_right : a ⇔ b ≤ b ↔ Codisjoint a b :=
  @le_symmDiff_iff_right αᵒᵈ _ _ _


theorem bihimp_assoc : a ⇔ b ⇔ c = a ⇔ (b ⇔ c) :=
  @symmDiff_assoc αᵒᵈ _ _ _ _


instance bihimp_isAssociative : Std.Associative (α := α) (· ⇔ ·) :=
  ⟨bihimp_assoc⟩


                                                           /-
                                                             α : Type u_2
                                                             inst✝ : BooleanAlgebra α
                                                             a b c : α
                                                             ⊢ Eq (bihimp a (bihimp b c)) (bihimp b (bihimp a c))
                                                           -/
theorem bihimp_left_comm : a ⇔ (b ⇔ c) = b ⇔ (a ⇔ c) := by simp_rw [← bihimp_assoc, bihimp_comm]
                                                           /-
                                                             🎉 no goals
                                                           -/


                                                        /-
                                                          α : Type u_2
                                                          inst✝ : BooleanAlgebra α
                                                          a b c : α
                                                          ⊢ Eq (bihimp (bihimp a b) c) (bihimp (bihimp a c) b)
                                                        -/
theorem bihimp_right_comm : a ⇔ b ⇔ c = a ⇔ c ⇔ b := by simp_rw [bihimp_assoc, bihimp_comm]
                                                        /-
                                                          🎉 no goals
                                                        -/


theorem bihimp_bihimp_bihimp_comm : a ⇔ b ⇔ (c ⇔ d) = a ⇔ c ⇔ (b ⇔ d) := by
  /-
    α : Type u_2
    inst✝ : BooleanAlgebra α
    a b c d : α
    ⊢ Eq (bihimp (bihimp a b) (bihimp c d)) (bihimp (bihimp a c) (bihimp b d))
  -/
  simp_rw [bihimp_assoc, bihimp_left_comm]
  /-
    🎉 no goals
  -/


@[simp]
                                                          /-
                                                            α : Type u_2
                                                            inst✝ : BooleanAlgebra α
                                                            a b : α
                                                            ⊢ Eq (bihimp a (bihimp a b)) b
                                                          -/
theorem bihimp_bihimp_cancel_left : a ⇔ (a ⇔ b) = b := by simp [← bihimp_assoc]
                                                          /-
                                                            🎉 no goals
                                                          -/


@[simp]
                                                         /-
                                                           α : Type u_2
                                                           inst✝ : BooleanAlgebra α
                                                           a b : α
                                                           ⊢ Eq (bihimp (bihimp b a) a) b
                                                         -/
theorem bihimp_bihimp_cancel_right : b ⇔ a ⇔ a = b := by simp [bihimp_assoc]
                                                         /-
                                                           🎉 no goals
                                                         -/


@[simp]
                                                 /-
                                                   α : Type u_2
                                                   inst✝ : BooleanAlgebra α
                                                   a b : α
                                                   ⊢ Eq (bihimp (bihimp a b) a) b
                                                 -/
theorem bihimp_bihimp_self : a ⇔ b ⇔ a = b := by rw [bihimp_comm, bihimp_bihimp_cancel_left]
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem bihimp_left_involutive (a : α) : Involutive (· ⇔ a) :=
  bihimp_bihimp_cancel_right _


theorem bihimp_right_involutive (a : α) : Involutive (a ⇔ ·) :=
  bihimp_bihimp_cancel_left _


theorem bihimp_left_injective (a : α) : Injective (· ⇔ a) :=
  @symmDiff_left_injective αᵒᵈ _ _


theorem bihimp_right_injective (a : α) : Injective (a ⇔ ·) :=
  @symmDiff_right_injective αᵒᵈ _ _


theorem bihimp_left_surjective (a : α) : Surjective (· ⇔ a) :=
  @symmDiff_left_surjective αᵒᵈ _ _


theorem bihimp_right_surjective (a : α) : Surjective (a ⇔ ·) :=
  @symmDiff_right_surjective αᵒᵈ _ _


@[simp]
theorem bihimp_left_inj : a ⇔ b = c ⇔ b ↔ a = c :=
  (bihimp_left_injective _).eq_iff


@[simp]
theorem bihimp_right_inj : a ⇔ b = a ⇔ c ↔ b = c :=
  (bihimp_right_injective _).eq_iff


@[simp]
theorem bihimp_eq_left : a ⇔ b = a ↔ b = ⊤ :=
  @symmDiff_eq_left αᵒᵈ _ _ _


@[simp]
theorem bihimp_eq_right : a ⇔ b = b ↔ a = ⊤ :=
  @symmDiff_eq_right αᵒᵈ _ _ _


protected theorem Codisjoint.bihimp_left (ha : Codisjoint a c) (hb : Codisjoint b c) :
    Codisjoint (a ⇔ b) c :=
  (ha.inf_left hb).mono_left inf_le_bihimp


protected theorem Codisjoint.bihimp_right (ha : Codisjoint a b) (hb : Codisjoint a c) :
    Codisjoint a (b ⇔ c) :=
  (ha.inf_right hb).mono_right inf_le_bihimp


                                                    /-
                                                      α : Type u_2
                                                      inst✝ : BooleanAlgebra α
                                                      a b : α
                                                      ⊢ Eq (symmDiff a b) (Max.max (Min.min a (HasCompl.compl b)) (Min.min b (HasCom …
                                                    -/
theorem symmDiff_eq : a ∆ b = a ⊓ bᶜ ⊔ b ⊓ aᶜ := by simp only [(· ∆ ·), sdiff_eq]
                                                    /-
                                                      🎉 no goals
                                                    -/


                                                      /-
                                                        α : Type u_2
                                                        inst✝ : BooleanAlgebra α
                                                        a b : α
                                                        ⊢ Eq (bihimp a b) (Min.min (Max.max a (HasCompl.compl b)) (Max.max b (HasCompl …
                                                      -/
theorem bihimp_eq : a ⇔ b = (a ⊔ bᶜ) ⊓ (b ⊔ aᶜ) := by simp only [(· ⇔ ·), himp_eq]
                                                      /-
                                                        🎉 no goals
                                                      -/


theorem symmDiff_eq' : a ∆ b = (a ⊔ b) ⊓ (aᶜ ⊔ bᶜ) := by
  /-
    α : Type u_2
    inst✝ : BooleanAlgebra α
    a b : α
    ⊢ Eq (symmDiff a b) (Min.min (Max.max a b) (Max.max (HasCompl.compl a) (HasCom …
  -/
  rw [symmDiff_eq_sup_sdiff_inf, sdiff_eq, compl_inf]
  /-
    🎉 no goals
  -/


theorem bihimp_eq' : a ⇔ b = a ⊓ b ⊔ aᶜ ⊓ bᶜ :=
  @symmDiff_eq' αᵒᵈ _ _ _


theorem symmDiff_top : a ∆ ⊤ = aᶜ :=
  symmDiff_top' _


theorem top_symmDiff : ⊤ ∆ a = aᶜ :=
  top_symmDiff' _


@[simp]
theorem compl_symmDiff : (a ∆ b)ᶜ = a ⇔ b := by
  /-
    α : Type u_2
    inst✝ : BooleanAlgebra α
    a b : α
    ⊢ Eq (HasCompl.compl (symmDiff a b)) (bihimp a b)
  -/
  simp_rw [symmDiff, compl_sup_distrib, compl_sdiff, bihimp, inf_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem compl_bihimp : (a ⇔ b)ᶜ = a ∆ b :=
  @compl_symmDiff αᵒᵈ _ _ _


@[simp]
theorem compl_symmDiff_compl : aᶜ ∆ bᶜ = a ∆ b :=
                             /-
                               α : Type u_2
                               inst✝ : BooleanAlgebra α
                               a b : α
                               ⊢ Eq (Max.max (SDiff.sdiff (HasCompl.compl b) (HasCompl.compl a)) (SDiff.sdiff …
                             -/
  (sup_comm _ _).trans <| by simp_rw [compl_sdiff_compl, sdiff_eq, symmDiff_eq]
                             /-
                               🎉 no goals
                             -/


@[simp]
theorem compl_bihimp_compl : aᶜ ⇔ bᶜ = a ⇔ b :=
  @compl_symmDiff_compl αᵒᵈ _ _ _


@[simp]
theorem symmDiff_eq_top : a ∆ b = ⊤ ↔ IsCompl a b := by
  rw [symmDiff_eq', ← compl_inf, inf_eq_top_iff, compl_eq_top, isCompl_iff, disjoint_iff,
    codisjoint_iff, and_comm]


@[simp]
theorem bihimp_eq_bot : a ⇔ b = ⊥ ↔ IsCompl a b := by
  rw [bihimp_eq', ← compl_sup, sup_eq_bot_iff, compl_eq_bot, isCompl_iff, disjoint_iff,
    codisjoint_iff]


@[simp]
theorem compl_symmDiff_self : aᶜ ∆ a = ⊤ :=
  hnot_symmDiff_self _


@[simp]
theorem symmDiff_compl_self : a ∆ aᶜ = ⊤ :=
  symmDiff_hnot_self _


theorem symmDiff_symmDiff_right' :
    a ∆ (b ∆ c) = a ⊓ b ⊓ c ⊔ a ⊓ bᶜ ⊓ cᶜ ⊔ aᶜ ⊓ b ⊓ cᶜ ⊔ aᶜ ⊓ bᶜ ⊓ c :=
  calc
    a ∆ (b ∆ c) = a ⊓ (b ⊓ c ⊔ bᶜ ⊓ cᶜ) ⊔ (b ⊓ cᶜ ⊔ c ⊓ bᶜ) ⊓ aᶜ := by
        /-
          α : Type u_2
          inst✝ : BooleanAlgebra α
          a b c : α
          ⊢ Eq (symmDiff a (symmDiff b c)) (Max.max (Min.min a (Max.max (Min.min b c) (M …
        -/
        { rw [symmDiff_eq, compl_symmDiff, bihimp_eq', symmDiff_eq] }
        /-
          🎉 no goals
        -/
    _ = a ⊓ b ⊓ c ⊔ a ⊓ bᶜ ⊓ cᶜ ⊔ b ⊓ cᶜ ⊓ aᶜ ⊔ c ⊓ bᶜ ⊓ aᶜ := by
        /-
          α : Type u_2
          inst✝ : BooleanAlgebra α
          a b c : α
          ⊢ Eq (Max.max (Min.min a (Max.max (Min.min b c) (Min.min (HasCompl.compl b) (H …
        -/
        { rw [inf_sup_left, inf_sup_right, ← sup_assoc, ← inf_assoc, ← inf_assoc] }
        /-
          🎉 no goals
        -/
    _ = a ⊓ b ⊓ c ⊔ a ⊓ bᶜ ⊓ cᶜ ⊔ aᶜ ⊓ b ⊓ cᶜ ⊔ aᶜ ⊓ bᶜ ⊓ c := (by
      /-
        α : Type u_2
        inst✝ : BooleanAlgebra α
        a b c : α
        ⊢ Eq (Max.max (Max.max (Max.max (Min.min (Min.min a b) c) (Min.min (Min.min a  …
      -/
      congr 1
        /-
          case e_a
          α : Type u_2
          inst✝ : BooleanAlgebra α
          a b c : α
          ⊢ Eq (Max.max (Max.max (Min.min (Min.min a b) c) (Min.min (Min.min a (HasCompl …
        -/
      · congr 1
        /-
          case e_a.e_a
          α : Type u_2
          inst✝ : BooleanAlgebra α
          a b c : α
          ⊢ Eq (Min.min (Min.min b (HasCompl.compl c)) (HasCompl.compl a)) (Min.min (Min …
        -/
        rw [inf_comm, inf_assoc]
        /-
          🎉 no goals
        -/
        /-
          case e_a
          α : Type u_2
          inst✝ : BooleanAlgebra α
          a b c : α
          ⊢ Eq (Min.min (Min.min c (HasCompl.compl b)) (HasCompl.compl a)) (Min.min (Min …
        -/
      · apply inf_left_right_swap)
        /-
          🎉 no goals
        -/


theorem Disjoint.le_symmDiff_sup_symmDiff_left (h : Disjoint a b) : c ≤ a ∆ c ⊔ b ∆ c := by
  /-
    α : Type u_2
    inst✝ : BooleanAlgebra α
    a b c : α
    h : Disjoint a b
    ⊢ LE.le c (Max.max (symmDiff a c) (symmDiff b c))
  -/
  trans c \ (a ⊓ b)
    /-
      α : Type u_2
      inst✝ : BooleanAlgebra α
      a b c : α
      h : Disjoint a b
      ⊢ LE.le c (SDiff.sdiff c (Min.min a b))
    -/
  · rw [h.eq_bot, sdiff_bot]
    /-
      🎉 no goals
    -/
    /-
      α : Type u_2
      inst✝ : BooleanAlgebra α
      a b c : α
      h : Disjoint a b
      ⊢ LE.le (SDiff.sdiff c (Min.min a b)) (Max.max (symmDiff a c) (symmDiff b c))
    -/
  · rw [sdiff_inf]
    /-
      α : Type u_2
      inst✝ : BooleanAlgebra α
      a b c : α
      h : Disjoint a b
      ⊢ LE.le (Max.max (SDiff.sdiff c a) (SDiff.sdiff c b)) (Max.max (symmDiff a c)  …
    -/
    exact sup_le_sup le_sup_right le_sup_right
    /-
      🎉 no goals
    -/


theorem Disjoint.le_symmDiff_sup_symmDiff_right (h : Disjoint b c) : a ≤ a ∆ b ⊔ a ∆ c := by
  /-
    α : Type u_2
    inst✝ : BooleanAlgebra α
    a b c : α
    h : Disjoint b c
    ⊢ LE.le a (Max.max (symmDiff a b) (symmDiff a c))
  -/
  simp_rw [symmDiff_comm a]
  /-
    α : Type u_2
    inst✝ : BooleanAlgebra α
    a b c : α
    h : Disjoint b c
    ⊢ LE.le a (Max.max (symmDiff b a) (symmDiff c a))
  -/
  exact h.le_symmDiff_sup_symmDiff_left
  /-
    🎉 no goals
  -/


theorem Codisjoint.bihimp_inf_bihimp_le_left (h : Codisjoint a b) : a ⇔ c ⊓ b ⇔ c ≤ c :=
  h.dual.le_symmDiff_sup_symmDiff_left


theorem Codisjoint.bihimp_inf_bihimp_le_right (h : Codisjoint b c) : a ⇔ b ⊓ a ⇔ c ≤ a :=
  h.dual.le_symmDiff_sup_symmDiff_right


@[simp]
theorem symmDiff_fst [GeneralizedCoheytingAlgebra α] [GeneralizedCoheytingAlgebra β]
    (a b : α × β) : (a ∆ b).1 = a.1 ∆ b.1 :=
  rfl


@[simp]
theorem symmDiff_snd [GeneralizedCoheytingAlgebra α] [GeneralizedCoheytingAlgebra β]
    (a b : α × β) : (a ∆ b).2 = a.2 ∆ b.2 :=
  rfl


@[simp]
theorem bihimp_fst [GeneralizedHeytingAlgebra α] [GeneralizedHeytingAlgebra β] (a b : α × β) :
    (a ⇔ b).1 = a.1 ⇔ b.1 :=
  rfl


@[simp]
theorem bihimp_snd [GeneralizedHeytingAlgebra α] [GeneralizedHeytingAlgebra β] (a b : α × β) :
    (a ⇔ b).2 = a.2 ⇔ b.2 :=
  rfl


theorem symmDiff_def [∀ i, GeneralizedCoheytingAlgebra (π i)] (a b : ∀ i, π i) :
    a ∆ b = fun i => a i ∆ b i :=
  rfl


theorem bihimp_def [∀ i, GeneralizedHeytingAlgebra (π i)] (a b : ∀ i, π i) :
    a ⇔ b = fun i => a i ⇔ b i :=
  rfl


@[simp]
theorem symmDiff_apply [∀ i, GeneralizedCoheytingAlgebra (π i)] (a b : ∀ i, π i) (i : ι) :
    (a ∆ b) i = a i ∆ b i :=
  rfl


@[simp]
theorem bihimp_apply [∀ i, GeneralizedHeytingAlgebra (π i)] (a b : ∀ i, π i) (i : ι) :
    (a ⇔ b) i = a i ⇔ b i :=
  rfl


