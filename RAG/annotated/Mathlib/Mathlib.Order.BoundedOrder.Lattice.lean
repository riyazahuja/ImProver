theorem top_sup_eq (a : α) : ⊤ ⊔ a = ⊤ :=
  sup_of_le_left le_top

-- Porting note: Not simp because simp can prove it

theorem sup_top_eq (a : α) : a ⊔ ⊤ = ⊤ :=
  sup_of_le_right le_top


theorem bot_sup_eq (a : α) : ⊥ ⊔ a = a :=
  sup_of_le_right bot_le

-- Porting note: Not simp because simp can prove it

theorem sup_bot_eq (a : α) : a ⊔ ⊥ = a :=
  sup_of_le_left bot_le


@[simp]
                                                         /-
                                                           α : Type u
                                                           inst✝¹ : SemilatticeSup α
                                                           inst✝ : OrderBot α
                                                           a b : α
                                                           ⊢ Iff (Eq (Max.max a b) Bot.bot) (And (Eq a Bot.bot) (Eq b Bot.bot))
                                                         -/
theorem sup_eq_bot_iff : a ⊔ b = ⊥ ↔ a = ⊥ ∧ b = ⊥ := by rw [eq_bot_iff, sup_le_iff]; simp
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/


lemma top_inf_eq (a : α) : ⊤ ⊓ a = a := inf_of_le_right le_top

-- Porting note: Not simp because simp can prove it

lemma inf_top_eq (a : α) : a ⊓ ⊤ = a := inf_of_le_left le_top


@[simp]
theorem inf_eq_top_iff : a ⊓ b = ⊤ ↔ a = ⊤ ∧ b = ⊤ :=
  @sup_eq_bot_iff αᵒᵈ _ _ _ _


lemma bot_inf_eq (a : α) : ⊥ ⊓ a = ⊥ := inf_of_le_left bot_le

-- Porting note: Not simp because simp can prove it

lemma inf_bot_eq (a : α) : a ⊓ ⊥ = ⊥ := inf_of_le_right bot_le


theorem exists_ge_and_iff_exists {P : α → Prop} {x₀ : α} (hP : Monotone P) :
    (∃ x, x₀ ≤ x ∧ P x) ↔ ∃ x, P x :=
  ⟨fun h => h.imp fun _ h => h.2, fun ⟨x, hx⟩ => ⟨x ⊔ x₀, le_sup_right, hP le_sup_left hx⟩⟩


lemma exists_and_iff_of_monotone {P Q : α → Prop} (hP : Monotone P) (hQ : Monotone Q) :
    ((∃ x, P x) ∧ ∃ x, Q x) ↔ (∃ x, P x ∧ Q x) :=
  ⟨fun ⟨⟨x, hPx⟩, ⟨y, hQx⟩⟩ ↦ ⟨x ⊔ y, ⟨hP le_sup_left hPx, hQ le_sup_right hQx⟩⟩,
    fun ⟨x, hPx, hQx⟩ ↦ ⟨⟨x, hPx⟩, ⟨x, hQx⟩⟩⟩


theorem exists_le_and_iff_exists {P : α → Prop} {x₀ : α} (hP : Antitone P) :
    (∃ x, x ≤ x₀ ∧ P x) ↔ ∃ x, P x :=
  exists_ge_and_iff_exists <| hP.dual_left


lemma exists_and_iff_of_antitone {P Q : α → Prop} (hP : Antitone P) (hQ : Antitone Q) :
    ((∃ x, P x) ∧ ∃ x, Q x) ↔ (∃ x, P x ∧ Q x) :=
  ⟨fun ⟨⟨x, hPx⟩, ⟨y, hQx⟩⟩ ↦ ⟨x ⊓ y, ⟨hP inf_le_left hPx, hQ inf_le_right hQx⟩⟩,
    fun ⟨x, hPx, hQx⟩ ↦ ⟨⟨x, hPx⟩, ⟨x, hQx⟩⟩⟩


theorem min_bot_left [OrderBot α] (a : α) : min ⊥ a = ⊥ := bot_inf_eq _


theorem max_top_left [OrderTop α] (a : α) : max ⊤ a = ⊤ := top_sup_eq _


theorem min_top_left [OrderTop α] (a : α) : min ⊤ a = a := top_inf_eq _


theorem max_bot_left [OrderBot α] (a : α) : max ⊥ a = a := bot_sup_eq _


theorem min_top_right [OrderTop α] (a : α) : min a ⊤ = a := inf_top_eq _


theorem max_bot_right [OrderBot α] (a : α) : max a ⊥ = a := sup_bot_eq _


theorem min_bot_right [OrderBot α] (a : α) : min a ⊥ = ⊥ := inf_bot_eq _


theorem max_top_right [OrderTop α] (a : α) : max a ⊤ = ⊤ := sup_top_eq _


theorem max_eq_bot [OrderBot α] {a b : α} : max a b = ⊥ ↔ a = ⊥ ∧ b = ⊥ :=
  sup_eq_bot_iff


theorem min_eq_top [OrderTop α] {a b : α} : min a b = ⊤ ↔ a = ⊤ ∧ b = ⊤ :=
  inf_eq_top_iff


@[simp]
theorem min_eq_bot [OrderBot α] {a b : α} : min a b = ⊥ ↔ a = ⊥ ∨ b = ⊥ := by
  /-
    α : Type u
    inst✝¹ : LinearOrder α
    inst✝ : OrderBot α
    a b : α
    ⊢ Iff (Eq (Min.min a b) Bot.bot) (Or (Eq a Bot.bot) (Eq b Bot.bot))
  -/
  simp_rw [← le_bot_iff, inf_le_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem max_eq_top [OrderTop α] {a b : α} : max a b = ⊤ ↔ a = ⊤ ∨ b = ⊤ :=
  @min_eq_bot αᵒᵈ _ _ a b


