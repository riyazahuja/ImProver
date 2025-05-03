/-- A sup-irreducible element is a non-bottom element which isn't the supremum of anything smaller.
-/
def SupIrred (a : α) : Prop :=
  ¬IsMin a ∧ ∀ ⦃b c⦄, b ⊔ c = a → b = a ∨ c = a


/-- A sup-prime element is a non-bottom element which isn't less than the supremum of anything
smaller. -/
def SupPrime (a : α) : Prop :=
  ¬IsMin a ∧ ∀ ⦃b c⦄, a ≤ b ⊔ c → a ≤ b ∨ a ≤ c


theorem SupIrred.not_isMin (ha : SupIrred a) : ¬IsMin a :=
  ha.1


theorem SupPrime.not_isMin (ha : SupPrime a) : ¬IsMin a :=
  ha.1


theorem IsMin.not_supIrred (ha : IsMin a) : ¬SupIrred a := fun h => h.1 ha


theorem IsMin.not_supPrime (ha : IsMin a) : ¬SupPrime a := fun h => h.1 ha


@[simp]
theorem not_supIrred : ¬SupIrred a ↔ IsMin a ∨ ∃ b c, b ⊔ c = a ∧ b < a ∧ c < a := by
  /-
    α : Type u_2
    inst✝ : SemilatticeSup α
    a : α
    ⊢ Iff (Not (SupIrred a)) (Or (IsMin a) (Exists fun b => Exists fun c => And (E …
  -/
  rw [SupIrred, not_and_or]
  /-
    α : Type u_2
    inst✝ : SemilatticeSup α
    a : α
    ⊢ Iff (Or (Not (Not (IsMin a))) (Not (∀ ⦃b c : α⦄, Eq (Max.max b c) a → Or (Eq …
  -/
  push_neg
  /-
    α : Type u_2
    inst✝ : SemilatticeSup α
    a : α
    ⊢ Iff (Or (IsMin a) (Exists fun ⦃b⦄ => Exists fun ⦃c⦄ => And (Eq (Max.max b c) …
  -/
  rw [exists₂_congr]
  /-
    α : Type u_2
    inst✝ : SemilatticeSup α
    a : α
    ⊢ ∀ (a_1 b : α), Iff (And (Eq (Max.max a_1 b) a) (And (Ne a_1 a) (Ne b a))) (A …
  -/
  simp +contextual [@eq_comm _ _ a]
  /-
    🎉 no goals
  -/


@[simp]
theorem not_supPrime : ¬SupPrime a ↔ IsMin a ∨ ∃ b c, a ≤ b ⊔ c ∧ ¬a ≤ b ∧ ¬a ≤ c := by
  /-
    α : Type u_2
    inst✝ : SemilatticeSup α
    a : α
    ⊢ Iff (Not (SupPrime a)) (Or (IsMin a) (Exists fun b => Exists fun c => And (L …
  -/
  rw [SupPrime, not_and_or]; push_neg; rfl
                                       /-
                                         🎉 no goals
                                       -/


protected theorem SupPrime.supIrred : SupPrime a → SupIrred a :=
                                   /-
                                     α : Type u_2
                                     inst✝ : SemilatticeSup α
                                     a : α
                                     h : ∀ ⦃b c : α⦄, LE.le a (Max.max b c) → Or (LE.le a b) (LE.le a c)
                                     b c : α
                                     ha : Eq (Max.max b c) a
                                     ⊢ Or (Eq b a) (Eq c a)
                                   -/
  And.imp_right fun h b c ha => by simpa [← ha] using h ha.ge
                                   /-
                                     🎉 no goals
                                   -/


theorem SupPrime.le_sup (ha : SupPrime a) : a ≤ b ⊔ c ↔ a ≤ b ∨ a ≤ c :=
  ⟨fun h => ha.2 h, fun h => h.elim le_sup_of_le_left le_sup_of_le_right⟩


@[simp]
theorem not_supIrred_bot : ¬SupIrred (⊥ : α) :=
  isMin_bot.not_supIrred


@[simp]
theorem not_supPrime_bot : ¬SupPrime (⊥ : α) :=
  isMin_bot.not_supPrime


                                                        /-
                                                          α : Type u_2
                                                          inst✝¹ : SemilatticeSup α
                                                          a : α
                                                          inst✝ : OrderBot α
                                                          ha : SupIrred a
                                                          ⊢ Ne a Bot.bot
                                                        -/
theorem SupIrred.ne_bot (ha : SupIrred a) : a ≠ ⊥ := by rintro rfl; exact not_supIrred_bot ha
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


                                                        /-
                                                          α : Type u_2
                                                          inst✝¹ : SemilatticeSup α
                                                          a : α
                                                          inst✝ : OrderBot α
                                                          ha : SupPrime a
                                                          ⊢ Ne a Bot.bot
                                                        -/
theorem SupPrime.ne_bot (ha : SupPrime a) : a ≠ ⊥ := by rintro rfl; exact not_supPrime_bot ha
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


theorem SupIrred.finset_sup_eq (ha : SupIrred a) (h : s.sup f = a) : ∃ i ∈ s, f i = a := by
  classical
  induction' s using Finset.induction with i s _ ih
  · simpa [ha.ne_bot] using h.symm
  simp only [exists_prop, exists_mem_insert] at ih ⊢
  rw [sup_insert] at h
  exact (ha.2 h).imp_right ih


theorem SupPrime.le_finset_sup (ha : SupPrime a) : a ≤ s.sup f ↔ ∃ i ∈ s, a ≤ f i := by
  classical
  induction' s using Finset.induction with i s _ ih
  · simp [ha.ne_bot]
  · simp only [exists_prop, exists_mem_insert, sup_insert, ha.le_sup, ih]


/-- In a well-founded lattice, any element is the supremum of finitely many sup-irreducible
elements. This is the order-theoretic analogue of prime factorisation. -/
theorem exists_supIrred_decomposition (a : α) :
    ∃ s : Finset α, s.sup id = a ∧ ∀ ⦃b⦄, b ∈ s → SupIrred b := by
  classical
  apply WellFoundedLT.induction a _
  clear a
  rintro a ih
  by_cases ha : SupIrred a
  · exact ⟨{a}, by simp [ha]⟩
  rw [not_supIrred] at ha
  obtain ha | ⟨b, c, rfl, hb, hc⟩ := ha
  · exact ⟨∅, by simp [ha.eq_bot]⟩
  obtain ⟨s, rfl, hs⟩ := ih _ hb
  obtain ⟨t, rfl, ht⟩ := ih _ hc
  exact ⟨s ∪ t, sup_union, forall_mem_union.2 ⟨hs, ht⟩⟩


/-- An inf-irreducible element is a non-top element which isn't the infimum of anything bigger. -/
def InfIrred (a : α) : Prop :=
  ¬IsMax a ∧ ∀ ⦃b c⦄, b ⊓ c = a → b = a ∨ c = a


/-- An inf-prime element is a non-top element which isn't bigger than the infimum of anything
bigger. -/
def InfPrime (a : α) : Prop :=
  ¬IsMax a ∧ ∀ ⦃b c⦄, b ⊓ c ≤ a → b ≤ a ∨ c ≤ a


@[simp]
theorem IsMax.not_infIrred (ha : IsMax a) : ¬InfIrred a := fun h => h.1 ha


@[simp]
theorem IsMax.not_infPrime (ha : IsMax a) : ¬InfPrime a := fun h => h.1 ha


@[simp]
theorem not_infIrred : ¬InfIrred a ↔ IsMax a ∨ ∃ b c, b ⊓ c = a ∧ a < b ∧ a < c :=
  @not_supIrred αᵒᵈ _ _


@[simp]
theorem not_infPrime : ¬InfPrime a ↔ IsMax a ∨ ∃ b c, b ⊓ c ≤ a ∧ ¬b ≤ a ∧ ¬c ≤ a :=
  @not_supPrime αᵒᵈ _ _


protected theorem InfPrime.infIrred : InfPrime a → InfIrred a :=
                                   /-
                                     α : Type u_2
                                     inst✝ : SemilatticeInf α
                                     a : α
                                     h : ∀ ⦃b c : α⦄, LE.le (Min.min b c) a → Or (LE.le b a) (LE.le c a)
                                     b c : α
                                     ha : Eq (Min.min b c) a
                                     ⊢ Or (Eq b a) (Eq c a)
                                   -/
  And.imp_right fun h b c ha => by simpa [← ha] using h ha.le
                                   /-
                                     🎉 no goals
                                   -/


theorem InfPrime.inf_le (ha : InfPrime a) : b ⊓ c ≤ a ↔ b ≤ a ∨ c ≤ a :=
  ⟨fun h => ha.2 h, fun h => h.elim inf_le_of_left_le inf_le_of_right_le⟩


theorem not_infIrred_top : ¬InfIrred (⊤ : α) :=
  isMax_top.not_infIrred


theorem not_infPrime_top : ¬InfPrime (⊤ : α) :=
  isMax_top.not_infPrime


                                                        /-
                                                          α : Type u_2
                                                          inst✝¹ : SemilatticeInf α
                                                          a : α
                                                          inst✝ : OrderTop α
                                                          ha : InfIrred a
                                                          ⊢ Ne a Top.top
                                                        -/
theorem InfIrred.ne_top (ha : InfIrred a) : a ≠ ⊤ := by rintro rfl; exact not_infIrred_top ha
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


                                                        /-
                                                          α : Type u_2
                                                          inst✝¹ : SemilatticeInf α
                                                          a : α
                                                          inst✝ : OrderTop α
                                                          ha : InfPrime a
                                                          ⊢ Ne a Top.top
                                                        -/
theorem InfPrime.ne_top (ha : InfPrime a) : a ≠ ⊤ := by rintro rfl; exact not_infPrime_top ha
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


theorem InfIrred.finset_inf_eq : InfIrred a → s.inf f = a → ∃ i ∈ s, f i = a :=
  @SupIrred.finset_sup_eq _ αᵒᵈ _ _ _ _ _


theorem InfPrime.finset_inf_le (ha : InfPrime a) : s.inf f ≤ a ↔ ∃ i ∈ s, f i ≤ a :=
  @SupPrime.le_finset_sup _ αᵒᵈ _ _ _ _ _ ha


/-- In a cowell-founded lattice, any element is the infimum of finitely many inf-irreducible
elements. This is the order-theoretic analogue of prime factorisation. -/
theorem exists_infIrred_decomposition (a : α) :
    ∃ s : Finset α, s.inf id = a ∧ ∀ ⦃b⦄, b ∈ s → InfIrred b :=
  exists_supIrred_decomposition (α := αᵒᵈ) _


@[simp]
theorem infIrred_toDual {a : α} : InfIrred (toDual a) ↔ SupIrred a :=
  Iff.rfl


@[simp]
theorem infPrime_toDual {a : α} : InfPrime (toDual a) ↔ SupPrime a :=
  Iff.rfl


@[simp]
theorem supIrred_ofDual {a : αᵒᵈ} : SupIrred (ofDual a) ↔ InfIrred a :=
  Iff.rfl


@[simp]
theorem supPrime_ofDual {a : αᵒᵈ} : SupPrime (ofDual a) ↔ InfPrime a :=
  Iff.rfl


alias ⟨_, SupIrred.dual⟩ := infIrred_toDual


alias ⟨_, SupPrime.dual⟩ := infPrime_toDual


alias ⟨_, InfIrred.ofDual⟩ := supIrred_ofDual


alias ⟨_, InfPrime.ofDual⟩ := supPrime_ofDual


@[simp]
theorem supIrred_toDual {a : α} : SupIrred (toDual a) ↔ InfIrred a :=
  Iff.rfl


@[simp]
theorem supPrime_toDual {a : α} : SupPrime (toDual a) ↔ InfPrime a :=
  Iff.rfl


@[simp]
theorem infIrred_ofDual {a : αᵒᵈ} : InfIrred (ofDual a) ↔ SupIrred a :=
  Iff.rfl


@[simp]
theorem infPrime_ofDual {a : αᵒᵈ} : InfPrime (ofDual a) ↔ SupPrime a :=
  Iff.rfl


alias ⟨_, InfIrred.dual⟩ := supIrred_toDual


alias ⟨_, InfPrime.dual⟩ := supPrime_toDual


alias ⟨_, SupIrred.ofDual⟩ := infIrred_ofDual


alias ⟨_, SupPrime.ofDual⟩ := infPrime_ofDual


@[simp]
theorem supPrime_iff_supIrred : SupPrime a ↔ SupIrred a :=
  ⟨SupPrime.supIrred,
                                  /-
                                    α : Type u_2
                                    inst✝ : DistribLattice α
                                    a : α
                                    h : ∀ ⦃b c : α⦄, Eq (Max.max b c) a → Or (Eq b a) (Eq c a)
                                    b c : α
                                    ⊢ LE.le a (Max.max b c) → Or (LE.le a b) (LE.le a c)
                                  -/
    And.imp_right fun h b c => by simp_rw [← inf_eq_left, inf_sup_left]; exact @h _ _⟩
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


@[simp]
theorem infPrime_iff_infIrred : InfPrime a ↔ InfIrred a :=
  ⟨InfPrime.infIrred,
                                  /-
                                    α : Type u_2
                                    inst✝ : DistribLattice α
                                    a : α
                                    h : ∀ ⦃b c : α⦄, Eq (Min.min b c) a → Or (Eq b a) (Eq c a)
                                    b c : α
                                    ⊢ LE.le (Min.min b c) a → Or (LE.le b a) (LE.le c a)
                                  -/
    And.imp_right fun h b c => by simp_rw [← sup_eq_left, sup_inf_left]; exact @h _ _⟩
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


protected alias ⟨_, SupIrred.supPrime⟩ := supPrime_iff_supIrred

protected alias ⟨_, InfIrred.infPrime⟩ := infPrime_iff_infIrred


theorem supPrime_iff_not_isMin : SupPrime a ↔ ¬IsMin a :=
                     /-
                       α : Type u_2
                       inst✝ : LinearOrder α
                       a : α
                       ⊢ ∀ ⦃b c : α⦄, LE.le a (Max.max b c) → Or (LE.le a b) (LE.le a c)
                     -/
  and_iff_left <| by simp
                     /-
                       🎉 no goals
                     -/


theorem infPrime_iff_not_isMax : InfPrime a ↔ ¬IsMax a :=
                     /-
                       α : Type u_2
                       inst✝ : LinearOrder α
                       a : α
                       ⊢ ∀ ⦃b c : α⦄, LE.le (Min.min b c) a → Or (LE.le b a) (LE.le c a)
                     -/
  and_iff_left <| by simp
                     /-
                       🎉 no goals
                     -/


@[simp]
theorem supIrred_iff_not_isMin : SupIrred a ↔ ¬IsMin a :=
                             /-
                               α : Type u_2
                               inst✝ : LinearOrder α
                               a x✝¹ x✝ : α
                               ⊢ Eq (Max.max x✝¹ x✝) a → Or (Eq x✝¹ a) (Eq x✝ a)
                             -/
  and_iff_left fun _ _ => by simpa only [max_eq_iff] using Or.imp And.left And.left
                             /-
                               🎉 no goals
                             -/


@[simp]
theorem infIrred_iff_not_isMax : InfIrred a ↔ ¬IsMax a :=
                             /-
                               α : Type u_2
                               inst✝ : LinearOrder α
                               a x✝¹ x✝ : α
                               ⊢ Eq (Min.min x✝¹ x✝) a → Or (Eq x✝¹ a) (Eq x✝ a)
                             -/
  and_iff_left fun _ _ => by simpa only [min_eq_iff] using Or.imp And.left And.left
                             /-
                               🎉 no goals
                             -/


