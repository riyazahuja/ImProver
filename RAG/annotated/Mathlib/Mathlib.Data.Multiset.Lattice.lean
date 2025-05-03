/-- Supremum of a multiset: `sup {a, b, c} = a ⊔ b ⊔ c` -/
def sup (s : Multiset α) : α :=
  s.fold (· ⊔ ·) ⊥


@[simp]
theorem sup_coe (l : List α) : sup (l : Multiset α) = l.foldr (· ⊔ ·) ⊥ :=
  rfl


@[simp]
theorem sup_zero : (0 : Multiset α).sup = ⊥ :=
  fold_zero _ _


@[simp]
theorem sup_cons (a : α) (s : Multiset α) : (a ::ₘ s).sup = a ⊔ s.sup :=
  fold_cons_left _ _ _ _


@[simp]
theorem sup_singleton {a : α} : ({a} : Multiset α).sup = a := sup_bot_eq _


@[simp]
theorem sup_add (s₁ s₂ : Multiset α) : (s₁ + s₂).sup = s₁.sup ⊔ s₂.sup :=
               /-
                 α : Type u_1
                 inst✝¹ : SemilatticeSup α
                 inst✝ : OrderBot α
                 s₁ s₂ : Multiset α
                 ⊢ Eq (HAdd.hAdd s₁ s₂).sup (Multiset.fold (fun x1 x2 => Max.max x1 x2) (Max.ma …
               -/
  Eq.trans (by simp [sup]) (fold_add _ _ _ _ _)
               /-
                 🎉 no goals
               -/


@[simp]
theorem sup_le {s : Multiset α} {a : α} : s.sup ≤ a ↔ ∀ b ∈ s, b ≤ a :=
                              /-
                                α : Type u_1
                                inst✝¹ : SemilatticeSup α
                                inst✝ : OrderBot α
                                s : Multiset α
                                a : α
                                ⊢ Iff (LE.le (Multiset.sup 0) a) (∀ (b : α), Membership.mem 0 b → LE.le b a)
                              -/
  Multiset.induction_on s (by simp)
                              /-
                                🎉 no goals
                              -/
        /-
          α : Type u_1
          inst✝¹ : SemilatticeSup α
          inst✝ : OrderBot α
          s : Multiset α
          a : α
          ⊢ ∀ (a_1 : α) (s : Multiset α), Iff (LE.le s.sup a) (∀ (b : α), Membership.mem …
        -/
    (by simp +contextual [or_imp, forall_and])
        /-
          🎉 no goals
        -/


theorem le_sup {s : Multiset α} {a : α} (h : a ∈ s) : a ≤ s.sup :=
  sup_le.1 le_rfl _ h


@[gcongr]
theorem sup_mono {s₁ s₂ : Multiset α} (h : s₁ ⊆ s₂) : s₁.sup ≤ s₂.sup :=
  sup_le.2 fun _ hb => le_sup (h hb)


@[simp]
theorem sup_dedup (s : Multiset α) : (dedup s).sup = s.sup :=
  fold_dedup_idem _ _ _


@[simp]
theorem sup_ndunion (s₁ s₂ : Multiset α) : (ndunion s₁ s₂).sup = s₁.sup ⊔ s₂.sup := by
  /-
    α : Type u_1
    inst✝² : SemilatticeSup α
    inst✝¹ : OrderBot α
    inst✝ : DecidableEq α
    s₁ s₂ : Multiset α
    ⊢ Eq (s₁.ndunion s₂).sup (Max.max s₁.sup s₂.sup)
  -/
  rw [← sup_dedup, dedup_ext.2, sup_dedup, sup_add]; simp
                                                     /-
                                                       🎉 no goals
                                                     -/


@[simp]
theorem sup_union (s₁ s₂ : Multiset α) : (s₁ ∪ s₂).sup = s₁.sup ⊔ s₂.sup := by
  /-
    α : Type u_1
    inst✝² : SemilatticeSup α
    inst✝¹ : OrderBot α
    inst✝ : DecidableEq α
    s₁ s₂ : Multiset α
    ⊢ Eq (Union.union s₁ s₂).sup (Max.max s₁.sup s₂.sup)
  -/
  rw [← sup_dedup, dedup_ext.2, sup_dedup, sup_add]; simp
                                                     /-
                                                       🎉 no goals
                                                     -/


@[simp]
theorem sup_ndinsert (a : α) (s : Multiset α) : (ndinsert a s).sup = a ⊔ s.sup := by
  /-
    α : Type u_1
    inst✝² : SemilatticeSup α
    inst✝¹ : OrderBot α
    inst✝ : DecidableEq α
    a : α
    s : Multiset α
    ⊢ Eq (Multiset.ndinsert a s).sup (Max.max a s.sup)
  -/
  rw [← sup_dedup, dedup_ext.2, sup_dedup, sup_cons]; simp
                                                      /-
                                                        🎉 no goals
                                                      -/


theorem nodup_sup_iff {α : Type*} [DecidableEq α] {m : Multiset (Multiset α)} :
    m.sup.Nodup ↔ ∀ a : Multiset α, a ∈ m → a.Nodup := by
  -- Porting note: this was originally `apply m.induction_on`, which failed due to
  -- `failed to elaborate eliminator, expected type is not available`
  induction m using Multiset.induction_on with
  | empty => simp
  | cons _ _ h => simp [h]


/-- Infimum of a multiset: `inf {a, b, c} = a ⊓ b ⊓ c` -/
def inf (s : Multiset α) : α :=
  s.fold (· ⊓ ·) ⊤


@[simp]
theorem inf_coe (l : List α) : inf (l : Multiset α) = l.foldr (· ⊓ ·) ⊤ :=
  rfl


@[simp]
theorem inf_zero : (0 : Multiset α).inf = ⊤ :=
  fold_zero _ _


@[simp]
theorem inf_cons (a : α) (s : Multiset α) : (a ::ₘ s).inf = a ⊓ s.inf :=
  fold_cons_left _ _ _ _


@[simp]
theorem inf_singleton {a : α} : ({a} : Multiset α).inf = a := inf_top_eq _


@[simp]
theorem inf_add (s₁ s₂ : Multiset α) : (s₁ + s₂).inf = s₁.inf ⊓ s₂.inf :=
               /-
                 α : Type u_1
                 inst✝¹ : SemilatticeInf α
                 inst✝ : OrderTop α
                 s₁ s₂ : Multiset α
                 ⊢ Eq (HAdd.hAdd s₁ s₂).inf (Multiset.fold (fun x1 x2 => Min.min x1 x2) (Min.mi …
               -/
  Eq.trans (by simp [inf]) (fold_add _ _ _ _ _)
               /-
                 🎉 no goals
               -/


@[simp]
theorem le_inf {s : Multiset α} {a : α} : a ≤ s.inf ↔ ∀ b ∈ s, a ≤ b :=
                              /-
                                α : Type u_1
                                inst✝¹ : SemilatticeInf α
                                inst✝ : OrderTop α
                                s : Multiset α
                                a : α
                                ⊢ Iff (LE.le a (Multiset.inf 0)) (∀ (b : α), Membership.mem 0 b → LE.le a b)
                              -/
  Multiset.induction_on s (by simp)
                              /-
                                🎉 no goals
                              -/
        /-
          α : Type u_1
          inst✝¹ : SemilatticeInf α
          inst✝ : OrderTop α
          s : Multiset α
          a : α
          ⊢ ∀ (a_1 : α) (s : Multiset α), Iff (LE.le a s.inf) (∀ (b : α), Membership.mem …
        -/
    (by simp +contextual [or_imp, forall_and])
        /-
          🎉 no goals
        -/


theorem inf_le {s : Multiset α} {a : α} (h : a ∈ s) : s.inf ≤ a :=
  le_inf.1 le_rfl _ h


@[gcongr]
theorem inf_mono {s₁ s₂ : Multiset α} (h : s₁ ⊆ s₂) : s₂.inf ≤ s₁.inf :=
  le_inf.2 fun _ hb => inf_le (h hb)


@[simp]
theorem inf_dedup (s : Multiset α) : (dedup s).inf = s.inf :=
  fold_dedup_idem _ _ _


@[simp]
theorem inf_ndunion (s₁ s₂ : Multiset α) : (ndunion s₁ s₂).inf = s₁.inf ⊓ s₂.inf := by
  /-
    α : Type u_1
    inst✝² : SemilatticeInf α
    inst✝¹ : OrderTop α
    inst✝ : DecidableEq α
    s₁ s₂ : Multiset α
    ⊢ Eq (s₁.ndunion s₂).inf (Min.min s₁.inf s₂.inf)
  -/
  rw [← inf_dedup, dedup_ext.2, inf_dedup, inf_add]; simp
                                                     /-
                                                       🎉 no goals
                                                     -/


@[simp]
theorem inf_union (s₁ s₂ : Multiset α) : (s₁ ∪ s₂).inf = s₁.inf ⊓ s₂.inf := by
  /-
    α : Type u_1
    inst✝² : SemilatticeInf α
    inst✝¹ : OrderTop α
    inst✝ : DecidableEq α
    s₁ s₂ : Multiset α
    ⊢ Eq (Union.union s₁ s₂).inf (Min.min s₁.inf s₂.inf)
  -/
  rw [← inf_dedup, dedup_ext.2, inf_dedup, inf_add]; simp
                                                     /-
                                                       🎉 no goals
                                                     -/


@[simp]
theorem inf_ndinsert (a : α) (s : Multiset α) : (ndinsert a s).inf = a ⊓ s.inf := by
  /-
    α : Type u_1
    inst✝² : SemilatticeInf α
    inst✝¹ : OrderTop α
    inst✝ : DecidableEq α
    a : α
    s : Multiset α
    ⊢ Eq (Multiset.ndinsert a s).inf (Min.min a s.inf)
  -/
  rw [← inf_dedup, dedup_ext.2, inf_dedup, inf_cons]; simp
                                                      /-
                                                        🎉 no goals
                                                      -/


