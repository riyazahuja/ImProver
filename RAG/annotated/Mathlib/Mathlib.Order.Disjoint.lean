/-- Two elements of a lattice are disjoint if their inf is the bottom element.
  (This generalizes disjoint sets, viewed as members of the subset lattice.)

Note that we define this without reference to `⊓`, as this allows us to talk about orders where
the infimum is not unique, or where implementing `Inf` would require additional `Decidable`
arguments. -/
def Disjoint (a b : α) : Prop :=
  ∀ ⦃x⦄, x ≤ a → x ≤ b → x ≤ ⊥


@[simp]
theorem disjoint_of_subsingleton [Subsingleton α] : Disjoint a b :=
  fun x _ _ ↦ le_of_eq (Subsingleton.elim x ⊥)


theorem disjoint_comm : Disjoint a b ↔ Disjoint b a :=
  forall_congr' fun _ ↦ forall_swap


@[symm]
theorem Disjoint.symm ⦃a b : α⦄ : Disjoint a b → Disjoint b a :=
  disjoint_comm.1


theorem symmetric_disjoint : Symmetric (Disjoint : α → α → Prop) :=
  Disjoint.symm


@[simp]
theorem disjoint_bot_left : Disjoint ⊥ a := fun _ hbot _ ↦ hbot


@[simp]
theorem disjoint_bot_right : Disjoint a ⊥ := fun _ _ hbot ↦ hbot


theorem Disjoint.mono (h₁ : a ≤ b) (h₂ : c ≤ d) : Disjoint b d → Disjoint a c :=
  fun h _ ha hc ↦ h (ha.trans h₁) (hc.trans h₂)


theorem Disjoint.mono_left (h : a ≤ b) : Disjoint b c → Disjoint a c :=
  Disjoint.mono h le_rfl


theorem Disjoint.mono_right : b ≤ c → Disjoint a c → Disjoint a b :=
  Disjoint.mono le_rfl


@[simp]
theorem disjoint_self : Disjoint a a ↔ a = ⊥ :=
  ⟨fun hd ↦ bot_unique <| hd le_rfl le_rfl, fun h _ ha _ ↦ ha.trans_eq h⟩

/- TODO: Rename `Disjoint.eq_bot` to `Disjoint.inf_eq` and `Disjoint.eq_bot_of_self` to
`Disjoint.eq_bot` -/

alias ⟨Disjoint.eq_bot_of_self, _⟩ := disjoint_self


theorem Disjoint.ne (ha : a ≠ ⊥) (hab : Disjoint a b) : a ≠ b :=
                                      /-
                                        α : Type u_1
                                        inst✝¹ : PartialOrder α
                                        inst✝ : OrderBot α
                                        a b : α
                                        ha : Ne a Bot.bot
                                        hab : Disjoint a b
                                        h : Eq a b
                                        ⊢ Disjoint a a
                                      -/
  fun h ↦ ha <| disjoint_self.1 <| by rwa [← h] at hab
                                      /-
                                        🎉 no goals
                                      -/


theorem Disjoint.eq_bot_of_le (hab : Disjoint a b) (h : a ≤ b) : a = ⊥ :=
  eq_bot_iff.2 <| hab le_rfl h


theorem Disjoint.eq_bot_of_ge (hab : Disjoint a b) : b ≤ a → b = ⊥ :=
  hab.symm.eq_bot_of_le


                                                                         /-
                                                                           α : Type u_1
                                                                           inst✝¹ : PartialOrder α
                                                                           inst✝ : OrderBot α
                                                                           a b : α
                                                                           hab : Disjoint a b
                                                                           ⊢ Iff (Eq a b) (And (Eq a Bot.bot) (Eq b Bot.bot))
                                                                         -/
lemma Disjoint.eq_iff (hab : Disjoint a b) : a = b ↔ a = ⊥ ∧ b = ⊥ := by aesop
                                                                         /-
                                                                           🎉 no goals
                                                                         -/

lemma Disjoint.ne_iff (hab : Disjoint a b) : a ≠ b ↔ a ≠ ⊥ ∨ b ≠ ⊥ :=
  hab.eq_iff.not.trans not_and_or


theorem disjoint_of_le_iff_left_eq_bot (h : a ≤ b) :
    Disjoint a b ↔ a = ⊥ :=
  ⟨fun hd ↦ hd.eq_bot_of_le h, fun h ↦ h ▸ disjoint_bot_left⟩


@[simp]
theorem disjoint_top : Disjoint a ⊤ ↔ a = ⊥ :=
  ⟨fun h ↦ bot_unique <| h le_rfl le_top, fun h _ ha _ ↦ ha.trans_eq h⟩


@[simp]
theorem top_disjoint : Disjoint ⊤ a ↔ a = ⊥ :=
  ⟨fun h ↦ bot_unique <| h le_top le_rfl, fun h _ _ ha ↦ ha.trans_eq h⟩


theorem disjoint_iff_inf_le : Disjoint a b ↔ a ⊓ b ≤ ⊥ :=
  ⟨fun hd ↦ hd inf_le_left inf_le_right, fun h _ ha hb ↦ (le_inf ha hb).trans h⟩


theorem disjoint_iff : Disjoint a b ↔ a ⊓ b = ⊥ :=
  disjoint_iff_inf_le.trans le_bot_iff


theorem Disjoint.le_bot : Disjoint a b → a ⊓ b ≤ ⊥ :=
  disjoint_iff_inf_le.mp


theorem Disjoint.eq_bot : Disjoint a b → a ⊓ b = ⊥ :=
  bot_unique ∘ Disjoint.le_bot


theorem disjoint_assoc : Disjoint (a ⊓ b) c ↔ Disjoint a (b ⊓ c) := by
  /-
    α : Type u_1
    inst✝¹ : SemilatticeInf α
    inst✝ : OrderBot α
    a b c : α
    ⊢ Iff (Disjoint (Min.min a b) c) (Disjoint a (Min.min b c))
  -/
  rw [disjoint_iff_inf_le, disjoint_iff_inf_le, inf_assoc]
  /-
    🎉 no goals
  -/


theorem disjoint_left_comm : Disjoint a (b ⊓ c) ↔ Disjoint b (a ⊓ c) := by
  /-
    α : Type u_1
    inst✝¹ : SemilatticeInf α
    inst✝ : OrderBot α
    a b c : α
    ⊢ Iff (Disjoint a (Min.min b c)) (Disjoint b (Min.min a c))
  -/
  simp_rw [disjoint_iff_inf_le, inf_left_comm]
  /-
    🎉 no goals
  -/


theorem disjoint_right_comm : Disjoint (a ⊓ b) c ↔ Disjoint (a ⊓ c) b := by
  /-
    α : Type u_1
    inst✝¹ : SemilatticeInf α
    inst✝ : OrderBot α
    a b c : α
    ⊢ Iff (Disjoint (Min.min a b) c) (Disjoint (Min.min a c) b)
  -/
  simp_rw [disjoint_iff_inf_le, inf_right_comm]
  /-
    🎉 no goals
  -/


theorem Disjoint.inf_left (h : Disjoint a b) : Disjoint (a ⊓ c) b :=
  h.mono_left inf_le_left


theorem Disjoint.inf_left' (h : Disjoint a b) : Disjoint (c ⊓ a) b :=
  h.mono_left inf_le_right


theorem Disjoint.inf_right (h : Disjoint a b) : Disjoint a (b ⊓ c) :=
  h.mono_right inf_le_left


theorem Disjoint.inf_right' (h : Disjoint a b) : Disjoint a (c ⊓ b) :=
  h.mono_right inf_le_right


theorem Disjoint.of_disjoint_inf_of_le (h : Disjoint (a ⊓ b) c) (hle : a ≤ c) : Disjoint a b :=
  disjoint_iff.2 <| h.eq_bot_of_le <| inf_le_of_left_le hle


theorem Disjoint.of_disjoint_inf_of_le' (h : Disjoint (a ⊓ b) c) (hle : b ≤ c) : Disjoint a b :=
  disjoint_iff.2 <| h.eq_bot_of_le <| inf_le_of_right_le hle


theorem Disjoint.right_lt_sup_of_left_ne_bot [SemilatticeSup α] [OrderBot α] {a b : α}
    (h : Disjoint a b) (ha : a ≠ ⊥) : b < a ⊔ b :=
  le_sup_right.lt_of_ne fun eq ↦ ha (le_bot_iff.mp <| h le_rfl <| sup_eq_right.mp eq.symm)


@[simp]
theorem disjoint_sup_left : Disjoint (a ⊔ b) c ↔ Disjoint a c ∧ Disjoint b c := by
  /-
    α : Type u_1
    inst✝¹ : DistribLattice α
    inst✝ : OrderBot α
    a b c : α
    ⊢ Iff (Disjoint (Max.max a b) c) (And (Disjoint a c) (Disjoint b c))
  -/
  simp only [disjoint_iff, inf_sup_right, sup_eq_bot_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem disjoint_sup_right : Disjoint a (b ⊔ c) ↔ Disjoint a b ∧ Disjoint a c := by
  /-
    α : Type u_1
    inst✝¹ : DistribLattice α
    inst✝ : OrderBot α
    a b c : α
    ⊢ Iff (Disjoint a (Max.max b c)) (And (Disjoint a b) (Disjoint a c))
  -/
  simp only [disjoint_iff, inf_sup_left, sup_eq_bot_iff]
  /-
    🎉 no goals
  -/


theorem Disjoint.sup_left (ha : Disjoint a c) (hb : Disjoint b c) : Disjoint (a ⊔ b) c :=
  disjoint_sup_left.2 ⟨ha, hb⟩


theorem Disjoint.sup_right (hb : Disjoint a b) (hc : Disjoint a c) : Disjoint a (b ⊔ c) :=
  disjoint_sup_right.2 ⟨hb, hc⟩


theorem Disjoint.left_le_of_le_sup_right (h : a ≤ b ⊔ c) (hd : Disjoint a c) : a ≤ b :=
  le_of_inf_le_sup_le (le_trans hd.le_bot bot_le) <| sup_le h le_sup_right


theorem Disjoint.left_le_of_le_sup_left (h : a ≤ c ⊔ b) (hd : Disjoint a c) : a ≤ b :=
                                   /-
                                     α : Type u_1
                                     inst✝¹ : DistribLattice α
                                     inst✝ : OrderBot α
                                     a b c : α
                                     h : LE.le a (Max.max c b)
                                     hd : Disjoint a c
                                     ⊢ LE.le a (Max.max b c)
                                   -/
  hd.left_le_of_le_sup_right <| by rwa [sup_comm]
                                   /-
                                     🎉 no goals
                                   -/


/-- Two elements of a lattice are codisjoint if their sup is the top element.

Note that we define this without reference to `⊔`, as this allows us to talk about orders where
the supremum is not unique, or where implement `Sup` would require additional `Decidable`
arguments. -/
def Codisjoint (a b : α) : Prop :=
  ∀ ⦃x⦄, a ≤ x → b ≤ x → ⊤ ≤ x


theorem codisjoint_comm : Codisjoint a b ↔ Codisjoint b a :=
  forall_congr' fun _ ↦ forall_swap


@[deprecated (since := "2024-11-23")] alias Codisjoint_comm := codisjoint_comm


@[symm]
theorem Codisjoint.symm ⦃a b : α⦄ : Codisjoint a b → Codisjoint b a :=
  codisjoint_comm.1


theorem symmetric_codisjoint : Symmetric (Codisjoint : α → α → Prop) :=
  Codisjoint.symm


@[simp]
theorem codisjoint_top_left : Codisjoint ⊤ a := fun _ htop _ ↦ htop


@[simp]
theorem codisjoint_top_right : Codisjoint a ⊤ := fun _ _ htop ↦ htop


theorem Codisjoint.mono (h₁ : a ≤ b) (h₂ : c ≤ d) : Codisjoint a c → Codisjoint b d :=
  fun h _ ha hc ↦ h (h₁.trans ha) (h₂.trans hc)


theorem Codisjoint.mono_left (h : a ≤ b) : Codisjoint a c → Codisjoint b c :=
  Codisjoint.mono h le_rfl


theorem Codisjoint.mono_right : b ≤ c → Codisjoint a b → Codisjoint a c :=
  Codisjoint.mono le_rfl


@[simp]
theorem codisjoint_self : Codisjoint a a ↔ a = ⊤ :=
  ⟨fun hd ↦ top_unique <| hd le_rfl le_rfl, fun h _ ha _ ↦ h.symm.trans_le ha⟩

/- TODO: Rename `Codisjoint.eq_top` to `Codisjoint.sup_eq` and `Codisjoint.eq_top_of_self` to
`Codisjoint.eq_top` -/

alias ⟨Codisjoint.eq_top_of_self, _⟩ := codisjoint_self


theorem Codisjoint.ne (ha : a ≠ ⊤) (hab : Codisjoint a b) : a ≠ b :=
                                        /-
                                          α : Type u_1
                                          inst✝¹ : PartialOrder α
                                          inst✝ : OrderTop α
                                          a b : α
                                          ha : Ne a Top.top
                                          hab : Codisjoint a b
                                          h : Eq a b
                                          ⊢ Codisjoint a a
                                        -/
  fun h ↦ ha <| codisjoint_self.1 <| by rwa [← h] at hab
                                        /-
                                          🎉 no goals
                                        -/


theorem Codisjoint.eq_top_of_le (hab : Codisjoint a b) (h : b ≤ a) : a = ⊤ :=
  eq_top_iff.2 <| hab le_rfl h


theorem Codisjoint.eq_top_of_ge (hab : Codisjoint a b) : a ≤ b → b = ⊤ :=
  hab.symm.eq_top_of_le


                                                                             /-
                                                                               α : Type u_1
                                                                               inst✝¹ : PartialOrder α
                                                                               inst✝ : OrderTop α
                                                                               a b : α
                                                                               hab : Codisjoint a b
                                                                               ⊢ Iff (Eq a b) (And (Eq a Top.top) (Eq b Top.top))
                                                                             -/
lemma Codisjoint.eq_iff (hab : Codisjoint a b) : a = b ↔ a = ⊤ ∧ b = ⊤ := by aesop
                                                                             /-
                                                                               🎉 no goals
                                                                             -/

lemma Codisjoint.ne_iff (hab : Codisjoint a b) : a ≠ b ↔ a ≠ ⊤ ∨ b ≠ ⊤ :=
  hab.eq_iff.not.trans not_and_or


@[simp]
theorem codisjoint_bot : Codisjoint a ⊥ ↔ a = ⊤ :=
  ⟨fun h ↦ top_unique <| h le_rfl bot_le, fun h _ ha _ ↦ h.symm.trans_le ha⟩


@[simp]
theorem bot_codisjoint : Codisjoint ⊥ a ↔ a = ⊤ :=
  ⟨fun h ↦ top_unique <| h bot_le le_rfl, fun h _ _ ha ↦ h.symm.trans_le ha⟩


lemma Codisjoint.ne_bot_of_ne_top (h : Codisjoint a b) (ha : a ≠ ⊤) : b ≠ ⊥ := by
  /-
    α : Type u_1
    inst✝¹ : PartialOrder α
    inst✝ : BoundedOrder α
    a b : α
    h : Codisjoint a b
    ha : Ne a Top.top
    ⊢ Ne b Bot.bot
  -/
  rintro rfl; exact ha <| by simpa using h
              /-
                🎉 no goals
              -/


lemma Codisjoint.ne_bot_of_ne_top' (h : Codisjoint a b) (hb : b ≠ ⊤) : a ≠ ⊥ := by
  /-
    α : Type u_1
    inst✝¹ : PartialOrder α
    inst✝ : BoundedOrder α
    a b : α
    h : Codisjoint a b
    hb : Ne b Top.top
    ⊢ Ne a Bot.bot
  -/
  rintro rfl; exact hb <| by simpa using h
              /-
                🎉 no goals
              -/


theorem codisjoint_iff_le_sup : Codisjoint a b ↔ ⊤ ≤ a ⊔ b :=
  @disjoint_iff_inf_le αᵒᵈ _ _ _ _


theorem codisjoint_iff : Codisjoint a b ↔ a ⊔ b = ⊤ :=
  @disjoint_iff αᵒᵈ _ _ _ _


theorem Codisjoint.top_le : Codisjoint a b → ⊤ ≤ a ⊔ b :=
  @Disjoint.le_bot αᵒᵈ _ _ _ _


theorem Codisjoint.eq_top : Codisjoint a b → a ⊔ b = ⊤ :=
  @Disjoint.eq_bot αᵒᵈ _ _ _ _


theorem codisjoint_assoc : Codisjoint (a ⊔ b) c ↔ Codisjoint a (b ⊔ c) :=
  @disjoint_assoc αᵒᵈ _ _ _ _ _


theorem codisjoint_left_comm : Codisjoint a (b ⊔ c) ↔ Codisjoint b (a ⊔ c) :=
  @disjoint_left_comm αᵒᵈ _ _ _ _ _


theorem codisjoint_right_comm : Codisjoint (a ⊔ b) c ↔ Codisjoint (a ⊔ c) b :=
  @disjoint_right_comm αᵒᵈ _ _ _ _ _


theorem Codisjoint.sup_left (h : Codisjoint a b) : Codisjoint (a ⊔ c) b :=
  h.mono_left le_sup_left


theorem Codisjoint.sup_left' (h : Codisjoint a b) : Codisjoint (c ⊔ a) b :=
  h.mono_left le_sup_right


theorem Codisjoint.sup_right (h : Codisjoint a b) : Codisjoint a (b ⊔ c) :=
  h.mono_right le_sup_left


theorem Codisjoint.sup_right' (h : Codisjoint a b) : Codisjoint a (c ⊔ b) :=
  h.mono_right le_sup_right


theorem Codisjoint.of_codisjoint_sup_of_le (h : Codisjoint (a ⊔ b) c) (hle : c ≤ a) :
    Codisjoint a b :=
  @Disjoint.of_disjoint_inf_of_le αᵒᵈ _ _ _ _ _ h hle


theorem Codisjoint.of_codisjoint_sup_of_le' (h : Codisjoint (a ⊔ b) c) (hle : c ≤ b) :
    Codisjoint a b :=
  @Disjoint.of_disjoint_inf_of_le' αᵒᵈ _ _ _ _ _ h hle


@[simp]
theorem codisjoint_inf_left : Codisjoint (a ⊓ b) c ↔ Codisjoint a c ∧ Codisjoint b c := by
  /-
    α : Type u_1
    inst✝¹ : DistribLattice α
    inst✝ : OrderTop α
    a b c : α
    ⊢ Iff (Codisjoint (Min.min a b) c) (And (Codisjoint a c) (Codisjoint b c))
  -/
  simp only [codisjoint_iff, sup_inf_right, inf_eq_top_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem codisjoint_inf_right : Codisjoint a (b ⊓ c) ↔ Codisjoint a b ∧ Codisjoint a c := by
  /-
    α : Type u_1
    inst✝¹ : DistribLattice α
    inst✝ : OrderTop α
    a b c : α
    ⊢ Iff (Codisjoint a (Min.min b c)) (And (Codisjoint a b) (Codisjoint a c))
  -/
  simp only [codisjoint_iff, sup_inf_left, inf_eq_top_iff]
  /-
    🎉 no goals
  -/


theorem Codisjoint.inf_left (ha : Codisjoint a c) (hb : Codisjoint b c) : Codisjoint (a ⊓ b) c :=
  codisjoint_inf_left.2 ⟨ha, hb⟩


theorem Codisjoint.inf_right (hb : Codisjoint a b) (hc : Codisjoint a c) : Codisjoint a (b ⊓ c) :=
  codisjoint_inf_right.2 ⟨hb, hc⟩


theorem Codisjoint.left_le_of_le_inf_right (h : a ⊓ b ≤ c) (hd : Codisjoint b c) : a ≤ c :=
  @Disjoint.left_le_of_le_sup_right αᵒᵈ _ _ _ _ _ h hd.symm


theorem Codisjoint.left_le_of_le_inf_left (h : b ⊓ a ≤ c) (hd : Codisjoint b c) : a ≤ c :=
                                   /-
                                     α : Type u_1
                                     inst✝¹ : DistribLattice α
                                     inst✝ : OrderTop α
                                     a b c : α
                                     h : LE.le (Min.min b a) c
                                     hd : Codisjoint b c
                                     ⊢ LE.le (Min.min a b) c
                                   -/
  hd.left_le_of_le_inf_right <| by rwa [inf_comm]
                                   /-
                                     🎉 no goals
                                   -/


theorem Disjoint.dual [SemilatticeInf α] [OrderBot α] {a b : α} :
    Disjoint a b → Codisjoint (toDual a) (toDual b) :=
  id


theorem Codisjoint.dual [SemilatticeSup α] [OrderTop α] {a b : α} :
    Codisjoint a b → Disjoint (toDual a) (toDual b) :=
  id


@[simp]
theorem disjoint_toDual_iff [SemilatticeSup α] [OrderTop α] {a b : α} :
    Disjoint (toDual a) (toDual b) ↔ Codisjoint a b :=
  Iff.rfl


@[simp]
theorem disjoint_ofDual_iff [SemilatticeInf α] [OrderBot α] {a b : αᵒᵈ} :
    Disjoint (ofDual a) (ofDual b) ↔ Codisjoint a b :=
  Iff.rfl


@[simp]
theorem codisjoint_toDual_iff [SemilatticeInf α] [OrderBot α] {a b : α} :
    Codisjoint (toDual a) (toDual b) ↔ Disjoint a b :=
  Iff.rfl


@[simp]
theorem codisjoint_ofDual_iff [SemilatticeSup α] [OrderTop α] {a b : αᵒᵈ} :
    Codisjoint (ofDual a) (ofDual b) ↔ Disjoint a b :=
  Iff.rfl


theorem Disjoint.le_of_codisjoint (hab : Disjoint a b) (hbc : Codisjoint b c) : a ≤ c := by
  /-
    α : Type u_1
    inst✝¹ : DistribLattice α
    inst✝ : BoundedOrder α
    a b c : α
    hab : Disjoint a b
    hbc : Codisjoint b c
    ⊢ LE.le a c
  -/
  rw [← @inf_top_eq _ _ _ a, ← @bot_sup_eq _ _ _ c, ← hab.eq_bot, ← hbc.eq_top, sup_inf_right]
  /-
    α : Type u_1
    inst✝¹ : DistribLattice α
    inst✝ : BoundedOrder α
    a b c : α
    hab : Disjoint a b
    hbc : Codisjoint b c
    ⊢ LE.le (Min.min a (Max.max b c)) (Min.min (Max.max a c) (Max.max b c))
  -/
  exact inf_le_inf_right _ le_sup_left
  /-
    🎉 no goals
  -/


/-- Two elements `x` and `y` are complements of each other if `x ⊔ y = ⊤` and `x ⊓ y = ⊥`. -/
structure IsCompl [PartialOrder α] [BoundedOrder α] (x y : α) : Prop where
  /-- If `x` and `y` are to be complementary in an order, they should be disjoint. -/
  protected disjoint : Disjoint x y
  /-- If `x` and `y` are to be complementary in an order, they should be codisjoint. -/
  protected codisjoint : Codisjoint x y


theorem isCompl_iff [PartialOrder α] [BoundedOrder α] {a b : α} :
    IsCompl a b ↔ Disjoint a b ∧ Codisjoint a b :=
  ⟨fun h ↦ ⟨h.1, h.2⟩, fun h ↦ ⟨h.1, h.2⟩⟩


@[symm]
protected theorem symm (h : IsCompl x y) : IsCompl y x :=
  ⟨h.1.symm, h.2.symm⟩


lemma _root_.isCompl_comm : IsCompl x y ↔ IsCompl y x := ⟨IsCompl.symm, IsCompl.symm⟩


theorem dual (h : IsCompl x y) : IsCompl (toDual x) (toDual y) :=
  ⟨h.2, h.1⟩


theorem ofDual {a b : αᵒᵈ} (h : IsCompl a b) : IsCompl (ofDual a) (ofDual b) :=
  ⟨h.2, h.1⟩


theorem of_le (h₁ : x ⊓ y ≤ ⊥) (h₂ : ⊤ ≤ x ⊔ y) : IsCompl x y :=
  ⟨disjoint_iff_inf_le.mpr h₁, codisjoint_iff_le_sup.mpr h₂⟩


theorem of_eq (h₁ : x ⊓ y = ⊥) (h₂ : x ⊔ y = ⊤) : IsCompl x y :=
  ⟨disjoint_iff.mpr h₁, codisjoint_iff.mpr h₂⟩


theorem inf_eq_bot (h : IsCompl x y) : x ⊓ y = ⊥ :=
  h.disjoint.eq_bot


theorem sup_eq_top (h : IsCompl x y) : x ⊔ y = ⊤ :=
  h.codisjoint.eq_top


theorem inf_left_le_of_le_sup_right (h : IsCompl x y) (hle : a ≤ b ⊔ y) : a ⊓ x ≤ b :=
  calc
    a ⊓ x ≤ (b ⊔ y) ⊓ x := inf_le_inf hle le_rfl
    _ = b ⊓ x ⊔ y ⊓ x := inf_sup_right _ _ _
                    /-
                      α : Type u_1
                      inst✝¹ : DistribLattice α
                      inst✝ : BoundedOrder α
                      a b x y : α
                      h : IsCompl x y
                      hle : LE.le a (Max.max b y)
                      ⊢ Eq (Max.max (Min.min b x) (Min.min y x)) (Min.min b x)
                    -/
    _ = b ⊓ x := by rw [h.symm.inf_eq_bot, sup_bot_eq]
                    /-
                      🎉 no goals
                    -/
    _ ≤ b := inf_le_left


theorem le_sup_right_iff_inf_left_le {a b} (h : IsCompl x y) : a ≤ b ⊔ y ↔ a ⊓ x ≤ b :=
  ⟨h.inf_left_le_of_le_sup_right, h.symm.dual.inf_left_le_of_le_sup_right⟩


theorem inf_left_eq_bot_iff (h : IsCompl y z) : x ⊓ y = ⊥ ↔ x ≤ z := by
  /-
    α : Type u_1
    inst✝¹ : DistribLattice α
    inst✝ : BoundedOrder α
    x y z : α
    h : IsCompl y z
    ⊢ Iff (Eq (Min.min x y) Bot.bot) (LE.le x z)
  -/
  rw [← le_bot_iff, ← h.le_sup_right_iff_inf_left_le, bot_sup_eq]
  /-
    🎉 no goals
  -/


theorem inf_right_eq_bot_iff (h : IsCompl y z) : x ⊓ z = ⊥ ↔ x ≤ y :=
  h.symm.inf_left_eq_bot_iff


theorem disjoint_left_iff (h : IsCompl y z) : Disjoint x y ↔ x ≤ z := by
  /-
    α : Type u_1
    inst✝¹ : DistribLattice α
    inst✝ : BoundedOrder α
    x y z : α
    h : IsCompl y z
    ⊢ Iff (Disjoint x y) (LE.le x z)
  -/
  rw [disjoint_iff]
  /-
    α : Type u_1
    inst✝¹ : DistribLattice α
    inst✝ : BoundedOrder α
    x y z : α
    h : IsCompl y z
    ⊢ Iff (Eq (Min.min x y) Bot.bot) (LE.le x z)
  -/
  exact h.inf_left_eq_bot_iff
  /-
    🎉 no goals
  -/


theorem disjoint_right_iff (h : IsCompl y z) : Disjoint x z ↔ x ≤ y :=
  h.symm.disjoint_left_iff


theorem le_left_iff (h : IsCompl x y) : z ≤ x ↔ Disjoint z y :=
  h.disjoint_right_iff.symm


theorem le_right_iff (h : IsCompl x y) : z ≤ y ↔ Disjoint z x :=
  h.symm.le_left_iff


theorem left_le_iff (h : IsCompl x y) : x ≤ z ↔ Codisjoint z y :=
  h.dual.le_left_iff


theorem right_le_iff (h : IsCompl x y) : y ≤ z ↔ Codisjoint z x :=
  h.symm.left_le_iff


protected theorem Antitone {x' y'} (h : IsCompl x y) (h' : IsCompl x' y') (hx : x ≤ x') : y' ≤ y :=
  h'.right_le_iff.2 <| h.symm.codisjoint.mono_right hx


theorem right_unique (hxy : IsCompl x y) (hxz : IsCompl x z) : y = z :=
  le_antisymm (hxz.Antitone hxy <| le_refl x) (hxy.Antitone hxz <| le_refl x)


theorem left_unique (hxz : IsCompl x z) (hyz : IsCompl y z) : x = y :=
  hxz.symm.right_unique hyz.symm


theorem sup_inf {x' y'} (h : IsCompl x y) (h' : IsCompl x' y') : IsCompl (x ⊔ x') (y ⊓ y') :=
  of_eq
    (by rw [inf_sup_right, ← inf_assoc, h.inf_eq_bot, bot_inf_eq, bot_sup_eq, inf_left_comm,
      h'.inf_eq_bot, inf_bot_eq])
    (by rw [sup_inf_left, sup_comm x, sup_assoc, h.sup_eq_top, sup_top_eq, top_inf_eq,
      sup_assoc, sup_left_comm, h'.sup_eq_top, sup_top_eq])


theorem inf_sup {x' y'} (h : IsCompl x y) (h' : IsCompl x' y') : IsCompl (x ⊓ x') (y ⊔ y') :=
  (h.symm.sup_inf h'.symm).symm


protected theorem disjoint_iff [OrderBot α] [OrderBot β] {x y : α × β} :
    Disjoint x y ↔ Disjoint x.1 y.1 ∧ Disjoint x.2 y.2 := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : PartialOrder α
    inst✝² : PartialOrder β
    inst✝¹ : OrderBot α
    inst✝ : OrderBot β
    x y : Prod α β
    ⊢ Iff (Disjoint x y) (And (Disjoint x.fst y.fst) (Disjoint x.snd y.snd))
  -/
  constructor
    /-
      case mp
      α : Type u_1
      β : Type u_2
      inst✝³ : PartialOrder α
      inst✝² : PartialOrder β
      inst✝¹ : OrderBot α
      inst✝ : OrderBot β
      x y : Prod α β
      ⊢ Disjoint x y → And (Disjoint x.fst y.fst) (Disjoint x.snd y.snd)
    -/
  · intro h
    refine ⟨fun a hx hy ↦ (@h (a, ⊥) ⟨hx, ?_⟩ ⟨hy, ?_⟩).1,
      fun b hx hy ↦ (@h (⊥, b) ⟨?_, hx⟩ ⟨?_, hy⟩).2⟩
    /-
      case mp.refine_1
      α : Type u_1
      β : Type u_2
      inst✝³ : PartialOrder α
      inst✝² : PartialOrder β
      inst✝¹ : OrderBot α
      inst✝ : OrderBot β
      x y : Prod α β
      h : Disjoint x y
      a : α
      hx : LE.le a x.fst
      hy : LE.le a y.fst
      ⊢ LE.le { fst := a, snd := Bot.bot }.snd x.snd
    -/
    all_goals exact bot_le
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      β : Type u_2
      inst✝³ : PartialOrder α
      inst✝² : PartialOrder β
      inst✝¹ : OrderBot α
      inst✝ : OrderBot β
      x y : Prod α β
      ⊢ And (Disjoint x.fst y.fst) (Disjoint x.snd y.snd) → Disjoint x y
    -/
  · rintro ⟨ha, hb⟩ z hza hzb
    /-
      case mpr.intro
      α : Type u_1
      β : Type u_2
      inst✝³ : PartialOrder α
      inst✝² : PartialOrder β
      inst✝¹ : OrderBot α
      inst✝ : OrderBot β
      x y : Prod α β
      ha : Disjoint x.fst y.fst
      hb : Disjoint x.snd y.snd
      z : Prod α β
      hza : LE.le z x
      hzb : LE.le z y
      ⊢ LE.le z Bot.bot
    -/
    exact ⟨ha hza.1 hzb.1, hb hza.2 hzb.2⟩
    /-
      🎉 no goals
    -/


protected theorem codisjoint_iff [OrderTop α] [OrderTop β] {x y : α × β} :
    Codisjoint x y ↔ Codisjoint x.1 y.1 ∧ Codisjoint x.2 y.2 :=
  @Prod.disjoint_iff αᵒᵈ βᵒᵈ _ _ _ _ _ _


protected theorem isCompl_iff [BoundedOrder α] [BoundedOrder β] {x y : α × β} :
    IsCompl x y ↔ IsCompl x.1 y.1 ∧ IsCompl x.2 y.2 := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : PartialOrder α
    inst✝² : PartialOrder β
    inst✝¹ : BoundedOrder α
    inst✝ : BoundedOrder β
    x y : Prod α β
    ⊢ Iff (IsCompl x y) (And (IsCompl x.fst y.fst) (IsCompl x.snd y.snd))
  -/
  simp_rw [isCompl_iff, Prod.disjoint_iff, Prod.codisjoint_iff, and_and_and_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem isCompl_toDual_iff : IsCompl (toDual a) (toDual b) ↔ IsCompl a b :=
  ⟨IsCompl.ofDual, IsCompl.dual⟩


@[simp]
theorem isCompl_ofDual_iff {a b : αᵒᵈ} : IsCompl (ofDual a) (ofDual b) ↔ IsCompl a b :=
  ⟨IsCompl.dual, IsCompl.ofDual⟩


theorem isCompl_bot_top : IsCompl (⊥ : α) ⊤ :=
  IsCompl.of_eq (bot_inf_eq _) (sup_top_eq _)


theorem isCompl_top_bot : IsCompl (⊤ : α) ⊥ :=
  IsCompl.of_eq (inf_bot_eq _) (top_sup_eq _)


                                                              /-
                                                                α : Type u_1
                                                                inst✝¹ : Lattice α
                                                                inst✝ : BoundedOrder α
                                                                x : α
                                                                h : IsCompl x Bot.bot
                                                                ⊢ Eq x Top.top
                                                              -/
theorem eq_top_of_isCompl_bot (h : IsCompl x ⊥) : x = ⊤ := by rw [← sup_bot_eq x, h.sup_eq_top]
                                                              /-
                                                                🎉 no goals
                                                              -/


theorem eq_top_of_bot_isCompl (h : IsCompl ⊥ x) : x = ⊤ :=
  eq_top_of_isCompl_bot h.symm


theorem eq_bot_of_isCompl_top (h : IsCompl x ⊤) : x = ⊥ :=
  eq_top_of_isCompl_bot h.dual


theorem eq_bot_of_top_isCompl (h : IsCompl ⊤ x) : x = ⊥ :=
  eq_top_of_bot_isCompl h.dual


/-- An element is *complemented* if it has a complement. -/
def IsComplemented (a : α) : Prop :=
  ∃ b, IsCompl a b


theorem isComplemented_bot : IsComplemented (⊥ : α) :=
  ⟨⊤, isCompl_bot_top⟩


theorem isComplemented_top : IsComplemented (⊤ : α) :=
  ⟨⊥, isCompl_top_bot⟩


theorem IsComplemented.sup : IsComplemented a → IsComplemented b → IsComplemented (a ⊔ b) :=
  fun ⟨a', ha⟩ ⟨b', hb⟩ => ⟨a' ⊓ b', ha.sup_inf hb⟩


theorem IsComplemented.inf : IsComplemented a → IsComplemented b → IsComplemented (a ⊓ b) :=
  fun ⟨a', ha⟩ ⟨b', hb⟩ => ⟨a' ⊔ b', ha.inf_sup hb⟩


/-- A complemented bounded lattice is one where every element has a (not necessarily unique)
complement. -/
class ComplementedLattice (α) [Lattice α] [BoundedOrder α] : Prop where
  /-- In a `ComplementedLattice`, every element admits a complement. -/
  exists_isCompl : ∀ a : α, ∃ b : α, IsCompl a b


lemma complementedLattice_iff (α) [Lattice α] [BoundedOrder α] :
    ComplementedLattice α ↔ ∀ a : α, ∃ b : α, IsCompl a b :=
  ⟨fun ⟨h⟩ ↦ h, fun h ↦ ⟨h⟩⟩


instance Subsingleton.instComplementedLattice
    [Lattice α] [BoundedOrder α] [Subsingleton α] : ComplementedLattice α := by
  /-
    α : Type u_1
    inst✝² : Lattice α
    inst✝¹ : BoundedOrder α
    inst✝ : Subsingleton α
    ⊢ ComplementedLattice α
  -/
  refine ⟨fun a ↦ ⟨⊥, disjoint_bot_right, ?_⟩⟩
  /-
    α : Type u_1
    inst✝² : Lattice α
    inst✝¹ : BoundedOrder α
    inst✝ : Subsingleton α
    a : α
    ⊢ Codisjoint a Bot.bot
  -/
  rw [Subsingleton.elim ⊥ ⊤]
  /-
    α : Type u_1
    inst✝² : Lattice α
    inst✝¹ : BoundedOrder α
    inst✝ : Subsingleton α
    a : α
    ⊢ Codisjoint a Top.top
  -/
  exact codisjoint_top_right
  /-
    🎉 no goals
  -/


instance : ComplementedLattice αᵒᵈ :=
  ⟨fun a ↦
    let ⟨b, hb⟩ := exists_isCompl (show α from a)
    ⟨b, hb.dual⟩⟩


/-- The sublattice of complemented elements. -/
abbrev Complementeds (α : Type*) [Lattice α] [BoundedOrder α] : Type _ :=
  {a : α // IsComplemented a}


instance hasCoeT : CoeTC (Complementeds α) α := ⟨Subtype.val⟩


theorem coe_injective : Injective ((↑) : Complementeds α → α) := Subtype.coe_injective


@[simp, norm_cast]
theorem coe_inj : (a : α) = b ↔ a = b := Subtype.coe_inj

-- Porting note: removing `simp` because `Subtype.coe_le_coe` already proves it

@[norm_cast]
                                               /-
                                                 α : Type u_1
                                                 inst✝¹ : Lattice α
                                                 inst✝ : BoundedOrder α
                                                 a b : Complementeds α
                                                 ⊢ Iff (LE.le ↑a ↑b) (LE.le a b)
                                               -/
theorem coe_le_coe : (a : α) ≤ b ↔ a ≤ b := by simp
                                               /-
                                                 🎉 no goals
                                               -/

-- Porting note: removing `simp` because `Subtype.coe_lt_coe` already proves it

@[norm_cast]
theorem coe_lt_coe : (a : α) < b ↔ a < b := Iff.rfl


instance : BoundedOrder (Complementeds α) :=
  Subtype.boundedOrder isComplemented_bot isComplemented_top


@[simp, norm_cast]
theorem coe_bot : ((⊥ : Complementeds α) : α) = ⊥ := rfl


@[simp, norm_cast]
theorem coe_top : ((⊤ : Complementeds α) : α) = ⊤ := rfl

-- Porting note: removing `simp` because `Subtype.mk_bot` already proves it

theorem mk_bot : (⟨⊥, isComplemented_bot⟩ : Complementeds α) = ⊥ := rfl

-- Porting note: removing `simp` because `Subtype.mk_top` already proves it

theorem mk_top : (⟨⊤, isComplemented_top⟩ : Complementeds α) = ⊤ := rfl


instance : Inhabited (Complementeds α) := ⟨⊥⟩


instance : Max (Complementeds α) :=
  ⟨fun a b => ⟨a ⊔ b, a.2.sup b.2⟩⟩


instance : Min (Complementeds α) :=
  ⟨fun a b => ⟨a ⊓ b, a.2.inf b.2⟩⟩


@[simp, norm_cast]
theorem coe_sup (a b : Complementeds α) : ↑(a ⊔ b) = (a : α) ⊔ b := rfl


@[simp, norm_cast]
theorem coe_inf (a b : Complementeds α) : ↑(a ⊓ b) = (a : α) ⊓ b := rfl


@[simp]
theorem mk_sup_mk {a b : α} (ha : IsComplemented a) (hb : IsComplemented b) :
    (⟨a, ha⟩ ⊔ ⟨b, hb⟩ : Complementeds α) = ⟨a ⊔ b, ha.sup hb⟩ := rfl


@[simp]
theorem mk_inf_mk {a b : α} (ha : IsComplemented a) (hb : IsComplemented b) :
    (⟨a, ha⟩ ⊓ ⟨b, hb⟩ : Complementeds α) = ⟨a ⊓ b, ha.inf hb⟩ := rfl


instance : DistribLattice (Complementeds α) :=
  Complementeds.coe_injective.distribLattice _ coe_sup coe_inf


@[simp, norm_cast]
theorem disjoint_coe : Disjoint (a : α) b ↔ Disjoint a b := by
  /-
    α : Type u_1
    inst✝¹ : DistribLattice α
    inst✝ : BoundedOrder α
    a b : Complementeds α
    ⊢ Iff (Disjoint ↑a ↑b) (Disjoint a b)
  -/
  rw [disjoint_iff, disjoint_iff, ← coe_inf, ← coe_bot, coe_inj]
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem codisjoint_coe : Codisjoint (a : α) b ↔ Codisjoint a b := by
  /-
    α : Type u_1
    inst✝¹ : DistribLattice α
    inst✝ : BoundedOrder α
    a b : Complementeds α
    ⊢ Iff (Codisjoint ↑a ↑b) (Codisjoint a b)
  -/
  rw [codisjoint_iff, codisjoint_iff, ← coe_sup, ← coe_top, coe_inj]
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem isCompl_coe : IsCompl (a : α) b ↔ IsCompl a b := by
  /-
    α : Type u_1
    inst✝¹ : DistribLattice α
    inst✝ : BoundedOrder α
    a b : Complementeds α
    ⊢ Iff (IsCompl ↑a ↑b) (IsCompl a b)
  -/
  simp_rw [isCompl_iff, disjoint_coe, codisjoint_coe]
  /-
    🎉 no goals
  -/


instance : ComplementedLattice (Complementeds α) :=
  ⟨fun ⟨a, b, h⟩ => ⟨⟨b, a, h.symm⟩, isCompl_coe.1 h⟩⟩


