/-- Supremum of a finite set: `sup {a, b, c} f = f a ⊔ f b ⊔ f c` -/
def sup (s : Finset β) (f : β → α) : α :=
  s.fold (· ⊔ ·) ⊥ f


theorem sup_def : s.sup f = (s.1.map f).sup :=
  rfl


@[simp]
theorem sup_empty : (∅ : Finset β).sup f = ⊥ :=
  fold_empty


@[simp]
theorem sup_cons {b : β} (h : b ∉ s) : (cons b s h).sup f = f b ⊔ s.sup f :=
  fold_cons h


@[simp]
theorem sup_insert [DecidableEq β] {b : β} : (insert b s : Finset β).sup f = f b ⊔ s.sup f :=
  fold_insert_idem


@[simp]
theorem sup_image [DecidableEq β] (s : Finset γ) (f : γ → β) (g : β → α) :
    (s.image f).sup g = s.sup (g ∘ f) :=
  fold_image_idem


@[simp]
theorem sup_map (s : Finset γ) (f : γ ↪ β) (g : β → α) : (s.map f).sup g = s.sup (g ∘ f) :=
  fold_map


@[simp]
theorem sup_singleton {b : β} : ({b} : Finset β).sup f = f b :=
  Multiset.sup_singleton


theorem sup_sup : s.sup (f ⊔ g) = s.sup f ⊔ s.sup g := by
  induction s using Finset.cons_induction with
  | empty => rw [sup_empty, sup_empty, sup_empty, bot_sup_eq]
  | cons _ _ _ ih =>
    rw [sup_cons, sup_cons, sup_cons, ih]
    exact sup_sup_sup_comm _ _ _ _


theorem sup_congr {f g : β → α} (hs : s₁ = s₂) (hfg : ∀ a ∈ s₂, f a = g a) :
    s₁.sup f = s₂.sup g := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : SemilatticeSup α
    inst✝ : OrderBot α
    s₁ s₂ : Finset β
    f g : β → α
    hs : Eq s₁ s₂
    hfg : ∀ (a : β), Membership.mem s₂ a → Eq (f a) (g a)
    ⊢ Eq (s₁.sup f) (s₂.sup g)
  -/
  subst hs
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : SemilatticeSup α
    inst✝ : OrderBot α
    s₁ : Finset β
    f g : β → α
    hfg : ∀ (a : β), Membership.mem s₁ a → Eq (f a) (g a)
    ⊢ Eq (s₁.sup f) (s₁.sup g)
  -/
  exact Finset.fold_congr hfg
  /-
    🎉 no goals
  -/


@[simp]
theorem _root_.map_finset_sup [SemilatticeSup β] [OrderBot β]
    [FunLike F α β] [SupBotHomClass F α β]
    (f : F) (s : Finset ι) (g : ι → α) : f (s.sup g) = s.sup (f ∘ g) :=
  Finset.cons_induction_on s (map_bot f) fun i s _ h => by
    /-
      F : Type u_1
      α : Type u_2
      β : Type u_3
      ι : Type u_5
      inst✝⁵ : SemilatticeSup α
      inst✝⁴ : OrderBot α
      inst✝³ : SemilatticeSup β
      inst✝² : OrderBot β
      inst✝¹ : FunLike F α β
      inst✝ : SupBotHomClass F α β
      f : F
      s✝ : Finset ι
      g : ι → α
      i : ι
      s : Finset ι
      x✝ : Not (Membership.mem s i)
      h : Eq (f (s.sup g)) (s.sup (Function.comp (⇑f) g))
      ⊢ Eq (f ((Finset.cons i s x✝).sup g)) ((Finset.cons i s x✝).sup (Function.comp …
    -/
    rw [sup_cons, sup_cons, map_sup, h, Function.comp_apply]
    /-
      🎉 no goals
    -/


@[simp]
protected theorem sup_le_iff {a : α} : s.sup f ≤ a ↔ ∀ b ∈ s, f b ≤ a := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : SemilatticeSup α
    inst✝ : OrderBot α
    s : Finset β
    f : β → α
    a : α
    ⊢ Iff (LE.le (s.sup f) a) (∀ (b : β), Membership.mem s b → LE.le (f b) a)
  -/
  apply Iff.trans Multiset.sup_le
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : SemilatticeSup α
    inst✝ : OrderBot α
    s : Finset β
    f : β → α
    a : α
    ⊢ Iff (∀ (b : α), Membership.mem (Multiset.map f s.val) b → LE.le b a) (∀ (b : …
  -/
  simp only [Multiset.mem_map, and_imp, exists_imp]
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : SemilatticeSup α
    inst✝ : OrderBot α
    s : Finset β
    f : β → α
    a : α
    ⊢ Iff (∀ (b : α) (x : β), Membership.mem s.val x → Eq (f x) b → LE.le b a) (∀  …
  -/
  exact ⟨fun k b hb => k _ _ hb rfl, fun k a' b hb h => h ▸ k _ hb⟩
  /-
    🎉 no goals
  -/


protected alias ⟨_, sup_le⟩ := Finset.sup_le_iff


theorem sup_const_le : (s.sup fun _ => a) ≤ a :=
  Finset.sup_le fun _ _ => le_rfl


theorem le_sup {b : β} (hb : b ∈ s) : f b ≤ s.sup f :=
  Finset.sup_le_iff.1 le_rfl _ hb


theorem isLUB_sup (s : Finset α) : IsLUB s (sup s id) :=
  ⟨fun x h => id_eq x ▸ le_sup h, fun _ h => Finset.sup_le h⟩


theorem le_sup_of_le {b : β} (hb : b ∈ s) (h : a ≤ f b) : a ≤ s.sup f := h.trans <| le_sup hb


theorem sup_union [DecidableEq β] : (s₁ ∪ s₂).sup f = s₁.sup f ⊔ s₂.sup f :=
                                  /-
                                    α : Type u_2
                                    β : Type u_3
                                    inst✝² : SemilatticeSup α
                                    inst✝¹ : OrderBot α
                                    s₁ s₂ : Finset β
                                    f : β → α
                                    inst✝ : DecidableEq β
                                    c : α
                                    ⊢ Iff (LE.le ((Union.union s₁ s₂).sup f) c) (LE.le (Max.max (s₁.sup f) (s₂.sup …
                                  -/
  eq_of_forall_ge_iff fun c => by simp [or_imp, forall_and]
                                  /-
                                    🎉 no goals
                                  -/


@[simp]
theorem sup_biUnion [DecidableEq β] (s : Finset γ) (t : γ → Finset β) :
    (s.biUnion t).sup f = s.sup fun x => (t x).sup f :=
                                  /-
                                    α : Type u_2
                                    β : Type u_3
                                    γ : Type u_4
                                    inst✝² : SemilatticeSup α
                                    inst✝¹ : OrderBot α
                                    f : β → α
                                    inst✝ : DecidableEq β
                                    s : Finset γ
                                    t : γ → Finset β
                                    c : α
                                    ⊢ Iff (LE.le ((s.biUnion t).sup f) c) (LE.le (s.sup fun x => (t x).sup f) c)
                                  -/
  eq_of_forall_ge_iff fun c => by simp [@forall_swap _ β]
                                  /-
                                    🎉 no goals
                                  -/


theorem sup_const {s : Finset β} (h : s.Nonempty) (c : α) : (s.sup fun _ => c) = c :=
  eq_of_forall_ge_iff (fun _ => Finset.sup_le_iff.trans h.forall_const)


@[simp]
theorem sup_bot (s : Finset β) : (s.sup fun _ => ⊥) = (⊥ : α) := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : SemilatticeSup α
    inst✝ : OrderBot α
    s : Finset β
    ⊢ Eq (s.sup fun x => Bot.bot) Bot.bot
  -/
  obtain rfl | hs := s.eq_empty_or_nonempty
    /-
      case inl
      α : Type u_2
      β : Type u_3
      inst✝¹ : SemilatticeSup α
      inst✝ : OrderBot α
      ⊢ Eq (EmptyCollection.emptyCollection.sup fun x => Bot.bot) Bot.bot
    -/
  · exact sup_empty
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_2
      β : Type u_3
      inst✝¹ : SemilatticeSup α
      inst✝ : OrderBot α
      s : Finset β
      hs : s.Nonempty
      ⊢ Eq (s.sup fun x => Bot.bot) Bot.bot
    -/
  · exact sup_const hs _
    /-
      🎉 no goals
    -/


theorem sup_ite (p : β → Prop) [DecidablePred p] :
    (s.sup fun i => ite (p i) (f i) (g i)) = (s.filter p).sup f ⊔ (s.filter fun i => ¬p i).sup g :=
  fold_ite _


@[gcongr]
theorem sup_mono_fun {g : β → α} (h : ∀ b ∈ s, f b ≤ g b) : s.sup f ≤ s.sup g :=
  Finset.sup_le fun b hb => le_trans (h b hb) (le_sup hb)


@[gcongr]
theorem sup_mono (h : s₁ ⊆ s₂) : s₁.sup f ≤ s₂.sup f :=
  Finset.sup_le (fun _ hb => le_sup (h hb))


protected theorem sup_comm (s : Finset β) (t : Finset γ) (f : β → γ → α) :
    (s.sup fun b => t.sup (f b)) = t.sup fun c => s.sup fun b => f b c :=
                                  /-
                                    α : Type u_2
                                    β : Type u_3
                                    γ : Type u_4
                                    inst✝¹ : SemilatticeSup α
                                    inst✝ : OrderBot α
                                    s : Finset β
                                    t : Finset γ
                                    f : β → γ → α
                                    a : α
                                    ⊢ Iff (LE.le (s.sup fun b => t.sup (f b)) a) (LE.le (t.sup fun c => s.sup fun  …
                                  -/
  eq_of_forall_ge_iff fun a => by simpa using forall₂_swap
                                  /-
                                    🎉 no goals
                                  -/


@[simp, nolint simpNF] -- Porting note: linter claims that LHS does not simplify
theorem sup_attach (s : Finset β) (f : β → α) : (s.attach.sup fun x => f x) = s.sup f :=
  (s.attach.sup_map (Function.Embedding.subtype _) f).symm.trans <| congr_arg _ attach_map_val


/-- See also `Finset.product_biUnion`. -/
theorem sup_product_left (s : Finset β) (t : Finset γ) (f : β × γ → α) :
    (s ×ˢ t).sup f = s.sup fun i => t.sup fun i' => f ⟨i, i'⟩ :=
                                  /-
                                    α : Type u_2
                                    β : Type u_3
                                    γ : Type u_4
                                    inst✝¹ : SemilatticeSup α
                                    inst✝ : OrderBot α
                                    s : Finset β
                                    t : Finset γ
                                    f : Prod β γ → α
                                    a : α
                                    ⊢ Iff (LE.le ((SProd.sprod s t).sup f) a) (LE.le (s.sup fun i => t.sup fun i'  …
                                  -/
  eq_of_forall_ge_iff fun a => by simp [@forall_swap _ γ]
                                  /-
                                    🎉 no goals
                                  -/


theorem sup_product_right (s : Finset β) (t : Finset γ) (f : β × γ → α) :
    (s ×ˢ t).sup f = t.sup fun i' => s.sup fun i => f ⟨i, i'⟩ := by
  /-
    α : Type u_2
    β : Type u_3
    γ : Type u_4
    inst✝¹ : SemilatticeSup α
    inst✝ : OrderBot α
    s : Finset β
    t : Finset γ
    f : Prod β γ → α
    ⊢ Eq ((SProd.sprod s t).sup f) (t.sup fun i' => s.sup fun i => f { fst := i, s …
  -/
  rw [sup_product_left, Finset.sup_comm]
  /-
    🎉 no goals
  -/


@[simp] lemma sup_prodMap (hs : s.Nonempty) (ht : t.Nonempty) (f : ι → α) (g : κ → β) :
    sup (s ×ˢ t) (Prod.map f g) = (sup s f, sup t g) :=
  eq_of_forall_ge_iff fun i ↦ by
    /-
      ι : Type u_7
      κ : Type u_8
      α : Type u_9
      β : Type u_10
      inst✝³ : SemilatticeSup α
      inst✝² : SemilatticeSup β
      inst✝¹ : OrderBot α
      inst✝ : OrderBot β
      s : Finset ι
      t : Finset κ
      hs : s.Nonempty
      ht : t.Nonempty
      f : ι → α
      g : κ → β
      i : Prod α β
      ⊢ Iff (LE.le ((SProd.sprod s t).sup (Prod.map f g)) i) (LE.le { fst := s.sup f …
    -/
    obtain ⟨a, ha⟩ := hs
    /-
      case intro
      ι : Type u_7
      κ : Type u_8
      α : Type u_9
      β : Type u_10
      inst✝³ : SemilatticeSup α
      inst✝² : SemilatticeSup β
      inst✝¹ : OrderBot α
      inst✝ : OrderBot β
      s : Finset ι
      t : Finset κ
      ht : t.Nonempty
      f : ι → α
      g : κ → β
      i : Prod α β
      a : ι
      ha : Membership.mem s a
      ⊢ Iff (LE.le ((SProd.sprod s t).sup (Prod.map f g)) i) (LE.le { fst := s.sup f …
    -/
    obtain ⟨b, hb⟩ := ht
    /-
      case intro.intro
      ι : Type u_7
      κ : Type u_8
      α : Type u_9
      β : Type u_10
      inst✝³ : SemilatticeSup α
      inst✝² : SemilatticeSup β
      inst✝¹ : OrderBot α
      inst✝ : OrderBot β
      s : Finset ι
      t : Finset κ
      f : ι → α
      g : κ → β
      i : Prod α β
      a : ι
      ha : Membership.mem s a
      b : κ
      hb : Membership.mem t b
      ⊢ Iff (LE.le ((SProd.sprod s t).sup (Prod.map f g)) i) (LE.le { fst := s.sup f …
    -/
    simp only [Prod.map, Finset.sup_le_iff, mem_product, and_imp, Prod.forall, Prod.le_def]
    /-
      case intro.intro
      ι : Type u_7
      κ : Type u_8
      α : Type u_9
      β : Type u_10
      inst✝³ : SemilatticeSup α
      inst✝² : SemilatticeSup β
      inst✝¹ : OrderBot α
      inst✝ : OrderBot β
      s : Finset ι
      t : Finset κ
      f : ι → α
      g : κ → β
      i : Prod α β
      a : ι
      ha : Membership.mem s a
      b : κ
      hb : Membership.mem t b
      ⊢ Iff (∀ (a : ι) (b : κ), Membership.mem s a → Membership.mem t b → And (LE.le …
    -/
    exact ⟨fun h ↦ ⟨fun i hi ↦ (h _ _ hi hb).1, fun j hj ↦ (h _ _ ha hj).2⟩, by aesop⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem sup_erase_bot [DecidableEq α] (s : Finset α) : (s.erase ⊥).sup id = s.sup id := by
  /-
    α : Type u_2
    inst✝² : SemilatticeSup α
    inst✝¹ : OrderBot α
    inst✝ : DecidableEq α
    s : Finset α
    ⊢ Eq ((s.erase Bot.bot).sup id) (s.sup id)
  -/
  refine (sup_mono (s.erase_subset _)).antisymm (Finset.sup_le_iff.2 fun a ha => ?_)
  /-
    α : Type u_2
    inst✝² : SemilatticeSup α
    inst✝¹ : OrderBot α
    inst✝ : DecidableEq α
    s : Finset α
    a : α
    ha : Membership.mem s a
    ⊢ LE.le (id a) ((s.erase Bot.bot).sup id)
  -/
  obtain rfl | ha' := eq_or_ne a ⊥
    /-
      case inl
      α : Type u_2
      inst✝² : SemilatticeSup α
      inst✝¹ : OrderBot α
      inst✝ : DecidableEq α
      s : Finset α
      ha : Membership.mem s Bot.bot
      ⊢ LE.le (id Bot.bot) ((s.erase Bot.bot).sup id)
    -/
  · exact bot_le
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_2
      inst✝² : SemilatticeSup α
      inst✝¹ : OrderBot α
      inst✝ : DecidableEq α
      s : Finset α
      a : α
      ha : Membership.mem s a
      ha' : Ne a Bot.bot
      ⊢ LE.le (id a) ((s.erase Bot.bot).sup id)
    -/
  · exact le_sup (mem_erase.2 ⟨ha', ha⟩)
    /-
      🎉 no goals
    -/


theorem sup_sdiff_right {α β : Type*} [GeneralizedBooleanAlgebra α] (s : Finset β) (f : β → α)
    (a : α) : (s.sup fun b => f b \ a) = s.sup f \ a := by
  induction s using Finset.cons_induction with
  | empty => rw [sup_empty, sup_empty, bot_sdiff]
  | cons _ _ _ h => rw [sup_cons, sup_cons, h, sup_sdiff]


theorem comp_sup_eq_sup_comp [SemilatticeSup γ] [OrderBot γ] {s : Finset β} {f : β → α} (g : α → γ)
    (g_sup : ∀ x y, g (x ⊔ y) = g x ⊔ g y) (bot : g ⊥ = ⊥) : g (s.sup f) = s.sup (g ∘ f) :=
  Finset.cons_induction_on s bot fun c t hc ih => by
    /-
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      inst✝³ : SemilatticeSup α
      inst✝² : OrderBot α
      inst✝¹ : SemilatticeSup γ
      inst✝ : OrderBot γ
      s : Finset β
      f : β → α
      g : α → γ
      g_sup : ∀ (x y : α), Eq (g (Max.max x y)) (Max.max (g x) (g y))
      bot : Eq (g Bot.bot) Bot.bot
      c : β
      t : Finset β
      hc : Not (Membership.mem t c)
      ih : Eq (g (t.sup f)) (t.sup (Function.comp g f))
      ⊢ Eq (g ((Finset.cons c t hc).sup f)) ((Finset.cons c t hc).sup (Function.comp …
    -/
    rw [sup_cons, sup_cons, g_sup, ih, Function.comp_apply]
    /-
      🎉 no goals
    -/


/-- Computing `sup` in a subtype (closed under `sup`) is the same as computing it in `α`. -/
theorem sup_coe {P : α → Prop} {Pbot : P ⊥} {Psup : ∀ ⦃x y⦄, P x → P y → P (x ⊔ y)} (t : Finset β)
    (f : β → { x : α // P x }) :
    (@sup { x // P x } _ (Subtype.semilatticeSup Psup) (Subtype.orderBot Pbot) t f : α) =
      t.sup fun x => ↑(f x) := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : SemilatticeSup α
    inst✝ : OrderBot α
    P : α → Prop
    Pbot : P Bot.bot
    Psup : ∀ ⦃x y : α⦄, P x → P y → P (Max.max x y)
    t : Finset β
    f : β → Subtype fun x => P x
    ⊢ Eq (↑(t.sup f)) (t.sup fun x => ↑(f x))
  -/
  letI := Subtype.semilatticeSup Psup
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : SemilatticeSup α
    inst✝ : OrderBot α
    P : α → Prop
    Pbot : P Bot.bot
    Psup : ∀ ⦃x y : α⦄, P x → P y → P (Max.max x y)
    t : Finset β
    f : β → Subtype fun x => P x
    this : SemilatticeSup (Subtype fun x => P x) := Subtype.semilatticeSup Psup
    ⊢ Eq (↑(t.sup f)) (t.sup fun x => ↑(f x))
  -/
  letI := Subtype.orderBot Pbot
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : SemilatticeSup α
    inst✝ : OrderBot α
    P : α → Prop
    Pbot : P Bot.bot
    Psup : ∀ ⦃x y : α⦄, P x → P y → P (Max.max x y)
    t : Finset β
    f : β → Subtype fun x => P x
    this✝ : SemilatticeSup (Subtype fun x => P x) := Subtype.semilatticeSup Psup
    this : OrderBot (Subtype fun x => P x) := Subtype.orderBot Pbot
    ⊢ Eq (↑(t.sup f)) (t.sup fun x => ↑(f x))
  -/
                                                        /-
                                                          🎉 no goals
                                                        -/
  apply comp_sup_eq_sup_comp Subtype.val <;> intros <;> rfl
                                                        /-
                                                          🎉 no goals
                                                        -/


@[simp]
theorem sup_toFinset {α β} [DecidableEq β] (s : Finset α) (f : α → Multiset β) :
    (s.sup f).toFinset = s.sup fun x => (f x).toFinset :=
  comp_sup_eq_sup_comp Multiset.toFinset toFinset_union rfl


theorem _root_.List.foldr_sup_eq_sup_toFinset [DecidableEq α] (l : List α) :
    l.foldr (· ⊔ ·) ⊥ = l.toFinset.sup id := by
  rw [← coe_fold_r, ← Multiset.fold_dedup_idem, sup_def, ← List.toFinset_coe, toFinset_val,
    Multiset.map_id]
  /-
    α : Type u_2
    inst✝² : SemilatticeSup α
    inst✝¹ : OrderBot α
    inst✝ : DecidableEq α
    l : List α
    ⊢ Eq (Multiset.fold (fun x1 x2 => Max.max x1 x2) Bot.bot (↑l).dedup) (↑l).dedu …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem subset_range_sup_succ (s : Finset ℕ) : s ⊆ range (s.sup id).succ := fun _ hn =>
  mem_range.2 <| Nat.lt_succ_of_le <| @le_sup _ _ _ _ _ id _ hn


theorem sup_induction {p : α → Prop} (hb : p ⊥) (hp : ∀ a₁, p a₁ → ∀ a₂, p a₂ → p (a₁ ⊔ a₂))
    (hs : ∀ b ∈ s, p (f b)) : p (s.sup f) := by
  induction s using Finset.cons_induction with
  | empty => exact hb
  | cons _ _ _ ih =>
    simp only [sup_cons, forall_mem_cons] at hs ⊢
    exact hp _ hs.1 _ (ih hs.2)


theorem sup_le_of_le_directed {α : Type*} [SemilatticeSup α] [OrderBot α] (s : Set α)
    (hs : s.Nonempty) (hdir : DirectedOn (· ≤ ·) s) (t : Finset α) :
    (∀ x ∈ t, ∃ y ∈ s, x ≤ y) → ∃ x ∈ s, t.sup id ≤ x := by
  classical
    induction' t using Finset.induction_on with a r _ ih h
    · simpa only [forall_prop_of_true, and_true, forall_prop_of_false, bot_le, not_false_iff,
        sup_empty, forall_true_iff, not_mem_empty]
    · intro h
      have incs : (r : Set α) ⊆ ↑(insert a r) := by
        rw [Finset.coe_subset]
        apply Finset.subset_insert
      -- x ∈ s is above the sup of r
      obtain ⟨x, ⟨hxs, hsx_sup⟩⟩ := ih fun x hx => h x <| incs hx
      -- y ∈ s is above a
      obtain ⟨y, hys, hay⟩ := h a (Finset.mem_insert_self a r)
      -- z ∈ s is above x and y
      obtain ⟨z, hzs, ⟨hxz, hyz⟩⟩ := hdir x hxs y hys
      use z, hzs
      rw [sup_insert, id, sup_le_iff]
      exact ⟨le_trans hay hyz, le_trans hsx_sup hxz⟩

-- If we acquire sublattices
-- the hypotheses should be reformulated as `s : SubsemilatticeSupBot`

theorem sup_mem (s : Set α) (w₁ : ⊥ ∈ s) (w₂ : ∀ᵉ (x ∈ s) (y ∈ s), x ⊔ y ∈ s)
    {ι : Type*} (t : Finset ι) (p : ι → α) (h : ∀ i ∈ t, p i ∈ s) : t.sup p ∈ s :=
  @sup_induction _ _ _ _ _ _ (· ∈ s) w₁ w₂ h


@[simp]
protected theorem sup_eq_bot_iff (f : β → α) (S : Finset β) : S.sup f = ⊥ ↔ ∀ s ∈ S, f s = ⊥ := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : SemilatticeSup α
    inst✝ : OrderBot α
    f : β → α
    S : Finset β
    ⊢ Iff (Eq (S.sup f) Bot.bot) (∀ (s : β), Membership.mem S s → Eq (f s) Bot.bot)
  -/
  classical induction' S using Finset.induction with a S _ hi <;> simp [*]
  /-
    🎉 no goals
  -/


theorem sup_eq_iSup [CompleteLattice β] (s : Finset α) (f : α → β) : s.sup f = ⨆ a ∈ s, f a :=
  le_antisymm
    (Finset.sup_le (fun a ha => le_iSup_of_le a <| le_iSup (fun _ => f a) ha))
    (iSup_le fun _ => iSup_le fun ha => le_sup ha)


theorem sup_id_eq_sSup [CompleteLattice α] (s : Finset α) : s.sup id = sSup s := by
  /-
    α : Type u_2
    inst✝ : CompleteLattice α
    s : Finset α
    ⊢ Eq (s.sup id) (SupSet.sSup ↑s)
  -/
  simp [sSup_eq_iSup, sup_eq_iSup]
  /-
    🎉 no goals
  -/


theorem sup_id_set_eq_sUnion (s : Finset (Set α)) : s.sup id = ⋃₀ ↑s :=
  sup_id_eq_sSup _


@[simp]
theorem sup_set_eq_biUnion (s : Finset α) (f : α → Set β) : s.sup f = ⋃ x ∈ s, f x :=
  sup_eq_iSup _ _


theorem sup_eq_sSup_image [CompleteLattice β] (s : Finset α) (f : α → β) :
    s.sup f = sSup (f '' s) := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝ : CompleteLattice β
    s : Finset α
    f : α → β
    ⊢ Eq (s.sup f) (SupSet.sSup (Set.image f ↑s))
  -/
  classical rw [← Finset.coe_image, ← sup_id_eq_sSup, sup_image, Function.id_comp]
  /-
    🎉 no goals
  -/


/-- Infimum of a finite set: `inf {a, b, c} f = f a ⊓ f b ⊓ f c` -/
def inf (s : Finset β) (f : β → α) : α :=
  s.fold (· ⊓ ·) ⊤ f


theorem inf_def : s.inf f = (s.1.map f).inf :=
  rfl


@[simp]
theorem inf_empty : (∅ : Finset β).inf f = ⊤ :=
  fold_empty


@[simp]
theorem inf_cons {b : β} (h : b ∉ s) : (cons b s h).inf f = f b ⊓ s.inf f :=
  @sup_cons αᵒᵈ _ _ _ _ _ _ h


@[simp]
theorem inf_insert [DecidableEq β] {b : β} : (insert b s : Finset β).inf f = f b ⊓ s.inf f :=
  fold_insert_idem


@[simp]
theorem inf_image [DecidableEq β] (s : Finset γ) (f : γ → β) (g : β → α) :
    (s.image f).inf g = s.inf (g ∘ f) :=
  fold_image_idem


@[simp]
theorem inf_map (s : Finset γ) (f : γ ↪ β) (g : β → α) : (s.map f).inf g = s.inf (g ∘ f) :=
  fold_map


@[simp]
theorem inf_singleton {b : β} : ({b} : Finset β).inf f = f b :=
  Multiset.inf_singleton


theorem inf_inf : s.inf (f ⊓ g) = s.inf f ⊓ s.inf g :=
  @sup_sup αᵒᵈ _ _ _ _ _ _


theorem inf_congr {f g : β → α} (hs : s₁ = s₂) (hfg : ∀ a ∈ s₂, f a = g a) :
    s₁.inf f = s₂.inf g := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : SemilatticeInf α
    inst✝ : OrderTop α
    s₁ s₂ : Finset β
    f g : β → α
    hs : Eq s₁ s₂
    hfg : ∀ (a : β), Membership.mem s₂ a → Eq (f a) (g a)
    ⊢ Eq (s₁.inf f) (s₂.inf g)
  -/
  subst hs
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : SemilatticeInf α
    inst✝ : OrderTop α
    s₁ : Finset β
    f g : β → α
    hfg : ∀ (a : β), Membership.mem s₁ a → Eq (f a) (g a)
    ⊢ Eq (s₁.inf f) (s₁.inf g)
  -/
  exact Finset.fold_congr hfg
  /-
    🎉 no goals
  -/


@[simp]
theorem _root_.map_finset_inf [SemilatticeInf β] [OrderTop β]
    [FunLike F α β] [InfTopHomClass F α β]
    (f : F) (s : Finset ι) (g : ι → α) : f (s.inf g) = s.inf (f ∘ g) :=
  Finset.cons_induction_on s (map_top f) fun i s _ h => by
    /-
      F : Type u_1
      α : Type u_2
      β : Type u_3
      ι : Type u_5
      inst✝⁵ : SemilatticeInf α
      inst✝⁴ : OrderTop α
      inst✝³ : SemilatticeInf β
      inst✝² : OrderTop β
      inst✝¹ : FunLike F α β
      inst✝ : InfTopHomClass F α β
      f : F
      s✝ : Finset ι
      g : ι → α
      i : ι
      s : Finset ι
      x✝ : Not (Membership.mem s i)
      h : Eq (f (s.inf g)) (s.inf (Function.comp (⇑f) g))
      ⊢ Eq (f ((Finset.cons i s x✝).inf g)) ((Finset.cons i s x✝).inf (Function.comp …
    -/
    rw [inf_cons, inf_cons, map_inf, h, Function.comp_apply]
    /-
      🎉 no goals
    -/


@[simp] protected theorem le_inf_iff {a : α} : a ≤ s.inf f ↔ ∀ b ∈ s, a ≤ f b :=
  @Finset.sup_le_iff αᵒᵈ _ _ _ _ _ _


protected alias ⟨_, le_inf⟩ := Finset.le_inf_iff


theorem le_inf_const_le : a ≤ s.inf fun _ => a :=
  Finset.le_inf fun _ _ => le_rfl


theorem inf_le {b : β} (hb : b ∈ s) : s.inf f ≤ f b :=
  Finset.le_inf_iff.1 le_rfl _ hb


theorem isGLB_inf (s : Finset α) : IsGLB s (inf s id) :=
  ⟨fun x h => id_eq x ▸ inf_le h, fun _ h => Finset.le_inf h⟩


theorem inf_le_of_le {b : β} (hb : b ∈ s) (h : f b ≤ a) : s.inf f ≤ a := (inf_le hb).trans h


theorem inf_union [DecidableEq β] : (s₁ ∪ s₂).inf f = s₁.inf f ⊓ s₂.inf f :=
                                 /-
                                   α : Type u_2
                                   β : Type u_3
                                   inst✝² : SemilatticeInf α
                                   inst✝¹ : OrderTop α
                                   s₁ s₂ : Finset β
                                   f : β → α
                                   inst✝ : DecidableEq β
                                   c : α
                                   ⊢ Iff (LE.le c ((Union.union s₁ s₂).inf f)) (LE.le c (Min.min (s₁.inf f) (s₂.i …
                                 -/
  eq_of_forall_le_iff fun c ↦ by simp [or_imp, forall_and]
                                 /-
                                   🎉 no goals
                                 -/


@[simp] theorem inf_biUnion [DecidableEq β] (s : Finset γ) (t : γ → Finset β) :
    (s.biUnion t).inf f = s.inf fun x => (t x).inf f :=
  @sup_biUnion αᵒᵈ _ _ _ _ _ _ _ _


theorem inf_const (h : s.Nonempty) (c : α) : (s.inf fun _ => c) = c := @sup_const αᵒᵈ _ _ _ _ h _


@[simp] theorem inf_top (s : Finset β) : (s.inf fun _ => ⊤) = (⊤ : α) := @sup_bot αᵒᵈ _ _ _ _


theorem inf_ite (p : β → Prop) [DecidablePred p] :
    (s.inf fun i ↦ ite (p i) (f i) (g i)) = (s.filter p).inf f ⊓ (s.filter fun i ↦ ¬ p i).inf g :=
  fold_ite _


@[gcongr]
theorem inf_mono_fun {g : β → α} (h : ∀ b ∈ s, f b ≤ g b) : s.inf f ≤ s.inf g :=
  Finset.le_inf fun b hb => le_trans (inf_le hb) (h b hb)


@[gcongr]
theorem inf_mono (h : s₁ ⊆ s₂) : s₂.inf f ≤ s₁.inf f :=
  Finset.le_inf (fun _ hb => inf_le (h hb))


protected theorem inf_comm (s : Finset β) (t : Finset γ) (f : β → γ → α) :
    (s.inf fun b => t.inf (f b)) = t.inf fun c => s.inf fun b => f b c :=
  @Finset.sup_comm αᵒᵈ _ _ _ _ _ _ _


theorem inf_attach (s : Finset β) (f : β → α) : (s.attach.inf fun x => f x) = s.inf f :=
  @sup_attach αᵒᵈ _ _ _ _ _


theorem inf_product_left (s : Finset β) (t : Finset γ) (f : β × γ → α) :
    (s ×ˢ t).inf f = s.inf fun i => t.inf fun i' => f ⟨i, i'⟩ :=
  @sup_product_left αᵒᵈ _ _ _ _ _ _ _


theorem inf_product_right (s : Finset β) (t : Finset γ) (f : β × γ → α) :
    (s ×ˢ t).inf f = t.inf fun i' => s.inf fun i => f ⟨i, i'⟩ :=
  @sup_product_right αᵒᵈ _ _ _ _ _ _ _


@[simp] lemma inf_prodMap (hs : s.Nonempty) (ht : t.Nonempty) (f : ι → α) (g : κ → β) :
    inf (s ×ˢ t) (Prod.map f g) = (inf s f, inf t g) :=
  sup_prodMap (α := αᵒᵈ) (β := βᵒᵈ) hs ht _ _


@[simp]
theorem inf_erase_top [DecidableEq α] (s : Finset α) : (s.erase ⊤).inf id = s.inf id :=
  @sup_erase_bot αᵒᵈ _ _ _ _


theorem comp_inf_eq_inf_comp [SemilatticeInf γ] [OrderTop γ] {s : Finset β} {f : β → α} (g : α → γ)
    (g_inf : ∀ x y, g (x ⊓ y) = g x ⊓ g y) (top : g ⊤ = ⊤) : g (s.inf f) = s.inf (g ∘ f) :=
  @comp_sup_eq_sup_comp αᵒᵈ _ γᵒᵈ _ _ _ _ _ _ _ g_inf top


/-- Computing `inf` in a subtype (closed under `inf`) is the same as computing it in `α`. -/
theorem inf_coe {P : α → Prop} {Ptop : P ⊤} {Pinf : ∀ ⦃x y⦄, P x → P y → P (x ⊓ y)} (t : Finset β)
    (f : β → { x : α // P x }) :
    (@inf { x // P x } _ (Subtype.semilatticeInf Pinf) (Subtype.orderTop Ptop) t f : α) =
      t.inf fun x => ↑(f x) :=
  @sup_coe αᵒᵈ _ _ _ _ Ptop Pinf t f


theorem _root_.List.foldr_inf_eq_inf_toFinset [DecidableEq α] (l : List α) :
    l.foldr (· ⊓ ·) ⊤ = l.toFinset.inf id := by
  rw [← coe_fold_r, ← Multiset.fold_dedup_idem, inf_def, ← List.toFinset_coe, toFinset_val,
    Multiset.map_id]
  /-
    α : Type u_2
    inst✝² : SemilatticeInf α
    inst✝¹ : OrderTop α
    inst✝ : DecidableEq α
    l : List α
    ⊢ Eq (Multiset.fold (fun x1 x2 => Min.min x1 x2) Top.top (↑l).dedup) (↑l).dedu …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem inf_induction {p : α → Prop} (ht : p ⊤) (hp : ∀ a₁, p a₁ → ∀ a₂, p a₂ → p (a₁ ⊓ a₂))
    (hs : ∀ b ∈ s, p (f b)) : p (s.inf f) :=
  @sup_induction αᵒᵈ _ _ _ _ _ _ ht hp hs


theorem inf_mem (s : Set α) (w₁ : ⊤ ∈ s) (w₂ : ∀ᵉ (x ∈ s) (y ∈ s), x ⊓ y ∈ s)
    {ι : Type*} (t : Finset ι) (p : ι → α) (h : ∀ i ∈ t, p i ∈ s) : t.inf p ∈ s :=
  @inf_induction _ _ _ _ _ _ (· ∈ s) w₁ w₂ h


@[simp]
protected theorem inf_eq_top_iff (f : β → α) (S : Finset β) : S.inf f = ⊤ ↔ ∀ s ∈ S, f s = ⊤ :=
  @Finset.sup_eq_bot_iff αᵒᵈ _ _ _ _ _


@[simp]
theorem toDual_sup [SemilatticeSup α] [OrderBot α] (s : Finset β) (f : β → α) :
    toDual (s.sup f) = s.inf (toDual ∘ f) :=
  rfl


@[simp]
theorem toDual_inf [SemilatticeInf α] [OrderTop α] (s : Finset β) (f : β → α) :
    toDual (s.inf f) = s.sup (toDual ∘ f) :=
  rfl


@[simp]
theorem ofDual_sup [SemilatticeInf α] [OrderTop α] (s : Finset β) (f : β → αᵒᵈ) :
    ofDual (s.sup f) = s.inf (ofDual ∘ f) :=
  rfl


@[simp]
theorem ofDual_inf [SemilatticeSup α] [OrderBot α] (s : Finset β) (f : β → αᵒᵈ) :
    ofDual (s.inf f) = s.sup (ofDual ∘ f) :=
  rfl


theorem sup_inf_distrib_left (s : Finset ι) (f : ι → α) (a : α) :
    a ⊓ s.sup f = s.sup fun i => a ⊓ f i := by
  induction s using Finset.cons_induction with
  | empty => simp_rw [Finset.sup_empty, inf_bot_eq]
  | cons _ _ _ h => rw [sup_cons, sup_cons, inf_sup_left, h]


theorem sup_inf_distrib_right (s : Finset ι) (f : ι → α) (a : α) :
    s.sup f ⊓ a = s.sup fun i => f i ⊓ a := by
  /-
    α : Type u_2
    ι : Type u_5
    inst✝¹ : DistribLattice α
    inst✝ : OrderBot α
    s : Finset ι
    f : ι → α
    a : α
    ⊢ Eq (Min.min (s.sup f) a) (s.sup fun i => Min.min (f i) a)
  -/
  rw [_root_.inf_comm, s.sup_inf_distrib_left]
  /-
    α : Type u_2
    ι : Type u_5
    inst✝¹ : DistribLattice α
    inst✝ : OrderBot α
    s : Finset ι
    f : ι → α
    a : α
    ⊢ Eq (s.sup fun i => Min.min a (f i)) (s.sup fun i => Min.min (f i) a)
  -/
  simp_rw [_root_.inf_comm]
  /-
    🎉 no goals
  -/


protected theorem disjoint_sup_right : Disjoint a (s.sup f) ↔ ∀ ⦃i⦄, i ∈ s → Disjoint a (f i) := by
  /-
    α : Type u_2
    ι : Type u_5
    inst✝¹ : DistribLattice α
    inst✝ : OrderBot α
    s : Finset ι
    f : ι → α
    a : α
    ⊢ Iff (Disjoint a (s.sup f)) (∀ ⦃i : ι⦄, Membership.mem s i → Disjoint a (f i))
  -/
  simp only [disjoint_iff, sup_inf_distrib_left, Finset.sup_eq_bot_iff]
  /-
    🎉 no goals
  -/


protected theorem disjoint_sup_left : Disjoint (s.sup f) a ↔ ∀ ⦃i⦄, i ∈ s → Disjoint (f i) a := by
  /-
    α : Type u_2
    ι : Type u_5
    inst✝¹ : DistribLattice α
    inst✝ : OrderBot α
    s : Finset ι
    f : ι → α
    a : α
    ⊢ Iff (Disjoint (s.sup f) a) (∀ ⦃i : ι⦄, Membership.mem s i → Disjoint (f i) a)
  -/
  simp only [disjoint_iff, sup_inf_distrib_right, Finset.sup_eq_bot_iff]
  /-
    🎉 no goals
  -/


theorem sup_inf_sup (s : Finset ι) (t : Finset κ) (f : ι → α) (g : κ → α) :
    s.sup f ⊓ t.sup g = (s ×ˢ t).sup fun i => f i.1 ⊓ g i.2 := by
  /-
    α : Type u_2
    ι : Type u_5
    κ : Type u_6
    inst✝¹ : DistribLattice α
    inst✝ : OrderBot α
    s : Finset ι
    t : Finset κ
    f : ι → α
    g : κ → α
    ⊢ Eq (Min.min (s.sup f) (t.sup g)) ((SProd.sprod s t).sup fun i => Min.min (f  …
  -/
  simp_rw [Finset.sup_inf_distrib_right, Finset.sup_inf_distrib_left, sup_product_left]
  /-
    🎉 no goals
  -/


theorem inf_sup_distrib_left (s : Finset ι) (f : ι → α) (a : α) :
    a ⊔ s.inf f = s.inf fun i => a ⊔ f i :=
  @sup_inf_distrib_left αᵒᵈ _ _ _ _ _ _


theorem inf_sup_distrib_right (s : Finset ι) (f : ι → α) (a : α) :
    s.inf f ⊔ a = s.inf fun i => f i ⊔ a :=
  @sup_inf_distrib_right αᵒᵈ _ _ _ _ _ _


protected theorem codisjoint_inf_right :
    Codisjoint a (s.inf f) ↔ ∀ ⦃i⦄, i ∈ s → Codisjoint a (f i) :=
  @Finset.disjoint_sup_right αᵒᵈ _ _ _ _ _ _


protected theorem codisjoint_inf_left :
    Codisjoint (s.inf f) a ↔ ∀ ⦃i⦄, i ∈ s → Codisjoint (f i) a :=
  @Finset.disjoint_sup_left αᵒᵈ _ _ _ _ _ _


theorem inf_sup_inf (s : Finset ι) (t : Finset κ) (f : ι → α) (g : κ → α) :
    s.inf f ⊔ t.inf g = (s ×ˢ t).inf fun i => f i.1 ⊔ g i.2 :=
  @sup_inf_sup αᵒᵈ _ _ _ _ _ _ _ _


theorem inf_sup {κ : ι → Type*} (s : Finset ι) (t : ∀ i, Finset (κ i)) (f : ∀ i, κ i → α) :
    (s.inf fun i => (t i).sup (f i)) =
      (s.pi t).sup fun g => s.attach.inf fun i => f _ <| g _ i.2 := by
  /-
    α : Type u_2
    ι : Type u_5
    inst✝² : DistribLattice α
    inst✝¹ : BoundedOrder α
    inst✝ : DecidableEq ι
    κ : ι → Type u_7
    s : Finset ι
    t : (i : ι) → Finset (κ i)
    f : (i : ι) → κ i → α
    ⊢ Eq (s.inf fun i => (t i).sup (f i)) ((s.pi t).sup fun g => s.attach.inf fun  …
  -/
  induction' s using Finset.induction with i s hi ih
    /-
      case empty
      α : Type u_2
      ι : Type u_5
      inst✝² : DistribLattice α
      inst✝¹ : BoundedOrder α
      inst✝ : DecidableEq ι
      κ : ι → Type u_7
      t : (i : ι) → Finset (κ i)
      f : (i : ι) → κ i → α
      ⊢ Eq (EmptyCollection.emptyCollection.inf fun i => (t i).sup (f i)) ((EmptyCol …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case insert
    α : Type u_2
    ι : Type u_5
    inst✝² : DistribLattice α
    inst✝¹ : BoundedOrder α
    inst✝ : DecidableEq ι
    κ : ι → Type u_7
    t : (i : ι) → Finset (κ i)
    f : (i : ι) → κ i → α
    i : ι
    s : Finset ι
    hi : Not (Membership.mem s i)
    ih : Eq (s.inf fun i => (t i).sup (f i)) ((s.pi t).sup fun g => s.attach.inf f …
    ⊢ Eq ((Insert.insert i s).inf fun i => (t i).sup (f i)) (((Insert.insert i s). …
  -/
  rw [inf_insert, ih, attach_insert, sup_inf_sup]
  /-
    case insert
    α : Type u_2
    ι : Type u_5
    inst✝² : DistribLattice α
    inst✝¹ : BoundedOrder α
    inst✝ : DecidableEq ι
    κ : ι → Type u_7
    t : (i : ι) → Finset (κ i)
    f : (i : ι) → κ i → α
    i : ι
    s : Finset ι
    hi : Not (Membership.mem s i)
    ih : Eq (s.inf fun i => (t i).sup (f i)) ((s.pi t).sup fun g => s.attach.inf f …
    ⊢ Eq ((SProd.sprod (t i) (s.pi t)).sup fun i_1 => Min.min (f i i_1.1) (s.attac …
  -/
  refine eq_of_forall_ge_iff fun c => ?_
  simp only [Finset.sup_le_iff, mem_product, mem_pi, and_imp, Prod.forall,
    inf_insert, inf_image]
  refine
    ⟨fun h g hg =>
      h (g i <| mem_insert_self _ _) (fun j hj => g j <| mem_insert_of_mem hj)
        (hg _ <| mem_insert_self _ _) fun j hj => hg _ <| mem_insert_of_mem hj,
      fun h a g ha hg => ?_⟩
  -- TODO: This `have` must be named to prevent it being shadowed by the internal `this` in `simpa`
  /-
    case insert
    α : Type u_2
    ι : Type u_5
    inst✝² : DistribLattice α
    inst✝¹ : BoundedOrder α
    inst✝ : DecidableEq ι
    κ : ι → Type u_7
    t : (i : ι) → Finset (κ i)
    f : (i : ι) → κ i → α
    i : ι
    s : Finset ι
    hi : Not (Membership.mem s i)
    ih : Eq (s.inf fun i => (t i).sup (f i)) ((s.pi t).sup fun g => s.attach.inf f …
    c : α
    h : ∀ (b : (a : ι) → Membership.mem (Insert.insert i s) a → κ a), (∀ (a : ι) ( …
    a : κ i
    g : (a : ι) → Membership.mem s a → κ a
    ha : Membership.mem (t i) a
    hg : ∀ (a : ι) (h : Membership.mem s a), Membership.mem (t a) (g a h)
    ⊢ LE.le (Min.min (f i a) (s.attach.inf fun i => f (↑i) (g ↑i ⋯))) c
  -/
  have aux : ∀ j : { x // x ∈ s }, ↑j ≠ i := fun j : s => ne_of_mem_of_not_mem j.2 hi
  -- Porting note: `simpa` doesn't support placeholders in proof terms
  have := h (fun j hj => if hji : j = i then cast (congr_arg κ hji.symm) a
      else g _ <| mem_of_mem_insert_of_ne hj hji) (fun j hj => ?_)
    /-
      case insert.refine_2
      α : Type u_2
      ι : Type u_5
      inst✝² : DistribLattice α
      inst✝¹ : BoundedOrder α
      inst✝ : DecidableEq ι
      κ : ι → Type u_7
      t : (i : ι) → Finset (κ i)
      f : (i : ι) → κ i → α
      i : ι
      s : Finset ι
      hi : Not (Membership.mem s i)
      ih : Eq (s.inf fun i => (t i).sup (f i)) ((s.pi t).sup fun g => s.attach.inf f …
      c : α
      h : ∀ (b : (a : ι) → Membership.mem (Insert.insert i s) a → κ a), (∀ (a : ι) ( …
      a : κ i
      g : (a : ι) → Membership.mem s a → κ a
      ha : Membership.mem (t i) a
      hg : ∀ (a : ι) (h : Membership.mem s a), Membership.mem (t a) (g a h)
      aux : ∀ (j : Subtype fun x => Membership.mem s x), Ne (↑j) i
      this : LE.le (Min.min (f i ((fun j hj => dite (Eq j i) (fun hji => cast ⋯ a) f …
      ⊢ LE.le (Min.min (f i a) (s.attach.inf fun i => f (↑i) (g ↑i ⋯))) c
    -/
  · simpa only [cast_eq, dif_pos, Function.comp_def, Subtype.coe_mk, dif_neg, aux] using this
    /-
      🎉 no goals
    -/
  /-
    case insert.refine_1
    α : Type u_2
    ι : Type u_5
    inst✝² : DistribLattice α
    inst✝¹ : BoundedOrder α
    inst✝ : DecidableEq ι
    κ : ι → Type u_7
    t : (i : ι) → Finset (κ i)
    f : (i : ι) → κ i → α
    i : ι
    s : Finset ι
    hi : Not (Membership.mem s i)
    ih : Eq (s.inf fun i => (t i).sup (f i)) ((s.pi t).sup fun g => s.attach.inf f …
    c : α
    h : ∀ (b : (a : ι) → Membership.mem (Insert.insert i s) a → κ a), (∀ (a : ι) ( …
    a : κ i
    g : (a : ι) → Membership.mem s a → κ a
    ha : Membership.mem (t i) a
    hg : ∀ (a : ι) (h : Membership.mem s a), Membership.mem (t a) (g a h)
    aux : ∀ (j : Subtype fun x => Membership.mem s x), Ne (↑j) i
    j : ι
    hj : Membership.mem (Insert.insert i s) j
    ⊢ Membership.mem (t j) ((fun j hj => dite (Eq j i) (fun hji => cast ⋯ a) fun h …
  -/
  rw [mem_insert] at hj
  /-
    case insert.refine_1
    α : Type u_2
    ι : Type u_5
    inst✝² : DistribLattice α
    inst✝¹ : BoundedOrder α
    inst✝ : DecidableEq ι
    κ : ι → Type u_7
    t : (i : ι) → Finset (κ i)
    f : (i : ι) → κ i → α
    i : ι
    s : Finset ι
    hi : Not (Membership.mem s i)
    ih : Eq (s.inf fun i => (t i).sup (f i)) ((s.pi t).sup fun g => s.attach.inf f …
    c : α
    h : ∀ (b : (a : ι) → Membership.mem (Insert.insert i s) a → κ a), (∀ (a : ι) ( …
    a : κ i
    g : (a : ι) → Membership.mem s a → κ a
    ha : Membership.mem (t i) a
    hg : ∀ (a : ι) (h : Membership.mem s a), Membership.mem (t a) (g a h)
    aux : ∀ (j : Subtype fun x => Membership.mem s x), Ne (↑j) i
    j : ι
    hj✝ : Membership.mem (Insert.insert i s) j
    hj : Or (Eq j i) (Membership.mem s j)
    ⊢ Membership.mem (t j) ((fun j hj => dite (Eq j i) (fun hji => cast ⋯ a) fun h …
  -/
  obtain (rfl | hj) := hj
    /-
      case insert.refine_1.inl
      α : Type u_2
      ι : Type u_5
      inst✝² : DistribLattice α
      inst✝¹ : BoundedOrder α
      inst✝ : DecidableEq ι
      κ : ι → Type u_7
      t : (i : ι) → Finset (κ i)
      f : (i : ι) → κ i → α
      s : Finset ι
      ih : Eq (s.inf fun i => (t i).sup (f i)) ((s.pi t).sup fun g => s.attach.inf f …
      c : α
      g : (a : ι) → Membership.mem s a → κ a
      hg : ∀ (a : ι) (h : Membership.mem s a), Membership.mem (t a) (g a h)
      j : ι
      hi : Not (Membership.mem s j)
      h : ∀ (b : (a : ι) → Membership.mem (Insert.insert j s) a → κ a), (∀ (a : ι) ( …
      a : κ j
      ha : Membership.mem (t j) a
      aux : ∀ (j_1 : Subtype fun x => Membership.mem s x), Ne (↑j_1) j
      hj : Membership.mem (Insert.insert j s) j
      ⊢ Membership.mem (t j) ((fun j_1 hj => dite (Eq j_1 j) (fun hji => cast ⋯ a) f …
    -/
  · simpa
    /-
      🎉 no goals
    -/
    /-
      case insert.refine_1.inr
      α : Type u_2
      ι : Type u_5
      inst✝² : DistribLattice α
      inst✝¹ : BoundedOrder α
      inst✝ : DecidableEq ι
      κ : ι → Type u_7
      t : (i : ι) → Finset (κ i)
      f : (i : ι) → κ i → α
      i : ι
      s : Finset ι
      hi : Not (Membership.mem s i)
      ih : Eq (s.inf fun i => (t i).sup (f i)) ((s.pi t).sup fun g => s.attach.inf f …
      c : α
      h : ∀ (b : (a : ι) → Membership.mem (Insert.insert i s) a → κ a), (∀ (a : ι) ( …
      a : κ i
      g : (a : ι) → Membership.mem s a → κ a
      ha : Membership.mem (t i) a
      hg : ∀ (a : ι) (h : Membership.mem s a), Membership.mem (t a) (g a h)
      aux : ∀ (j : Subtype fun x => Membership.mem s x), Ne (↑j) i
      j : ι
      hj✝ : Membership.mem (Insert.insert i s) j
      hj : Membership.mem s j
      ⊢ Membership.mem (t j) ((fun j hj => dite (Eq j i) (fun hji => cast ⋯ a) fun h …
    -/
  · simpa [ne_of_mem_of_not_mem hj hi] using hg _ _
    /-
      🎉 no goals
    -/


theorem sup_inf {κ : ι → Type*} (s : Finset ι) (t : ∀ i, Finset (κ i)) (f : ∀ i, κ i → α) :
    (s.sup fun i => (t i).inf (f i)) = (s.pi t).inf fun g => s.attach.sup fun i => f _ <| g _ i.2 :=
  @inf_sup αᵒᵈ _ _ _ _ _ _ _ _


theorem sup_sdiff_left (s : Finset ι) (f : ι → α) (a : α) :
    (s.sup fun b => a \ f b) = a \ s.inf f := by
  induction s using Finset.cons_induction with
  | empty => rw [sup_empty, inf_empty, sdiff_top]
  | cons _ _ _ h => rw [sup_cons, inf_cons, h, sdiff_inf]


theorem inf_sdiff_left (hs : s.Nonempty) (f : ι → α) (a : α) :
    (s.inf fun b => a \ f b) = a \ s.sup f := by
  induction hs using Finset.Nonempty.cons_induction with
  | singleton => rw [sup_singleton, inf_singleton]
  | cons _ _ _ _ ih => rw [sup_cons, inf_cons, ih, sdiff_sup]


theorem inf_sdiff_right (hs : s.Nonempty) (f : ι → α) (a : α) :
    (s.inf fun b => f b \ a) = s.inf f \ a := by
  induction hs using Finset.Nonempty.cons_induction with
  | singleton => rw [inf_singleton, inf_singleton]
  | cons _ _ _ _ ih => rw [inf_cons, inf_cons, ih, inf_sdiff]


theorem inf_himp_right (s : Finset ι) (f : ι → α) (a : α) :
    (s.inf fun b => f b ⇨ a) = s.sup f ⇨ a :=
  @sup_sdiff_left αᵒᵈ _ _ _ _ _


theorem sup_himp_right (hs : s.Nonempty) (f : ι → α) (a : α) :
    (s.sup fun b => f b ⇨ a) = s.inf f ⇨ a :=
  @inf_sdiff_left αᵒᵈ _ _ _ hs _ _


theorem sup_himp_left (hs : s.Nonempty) (f : ι → α) (a : α) :
    (s.sup fun b => a ⇨ f b) = a ⇨ s.sup f :=
  @inf_sdiff_right αᵒᵈ _ _ _ hs _ _


@[simp]
protected theorem compl_sup (s : Finset ι) (f : ι → α) : (s.sup f)ᶜ = s.inf fun i => (f i)ᶜ :=
  map_finset_sup (OrderIso.compl α) _ _


@[simp]
protected theorem compl_inf (s : Finset ι) (f : ι → α) : (s.inf f)ᶜ = s.sup fun i => (f i)ᶜ :=
  map_finset_inf (OrderIso.compl α) _ _


theorem comp_sup_eq_sup_comp_of_is_total [SemilatticeSup β] [OrderBot β] (g : α → β)
    (mono_g : Monotone g) (bot : g ⊥ = ⊥) : g (s.sup f) = s.sup (g ∘ f) :=
  comp_sup_eq_sup_comp g mono_g.map_sup bot


@[simp]
protected theorem le_sup_iff (ha : ⊥ < a) : a ≤ s.sup f ↔ ∃ b ∈ s, a ≤ f b := by
  /-
    α : Type u_2
    ι : Type u_5
    inst✝¹ : LinearOrder α
    inst✝ : OrderBot α
    s : Finset ι
    f : ι → α
    a : α
    ha : LT.lt Bot.bot a
    ⊢ Iff (LE.le a (s.sup f)) (Exists fun b => And (Membership.mem s b) (LE.le a ( …
  -/
  apply Iff.intro
  · induction s using cons_induction with
    | empty => exact (absurd · (not_le_of_lt ha))
    | cons c t hc ih =>
      rw [sup_cons, le_sup_iff]
      exact fun
      | Or.inl h => ⟨c, mem_cons.2 (Or.inl rfl), h⟩
      | Or.inr h => let ⟨b, hb, hle⟩ := ih h; ⟨b, mem_cons.2 (Or.inr hb), hle⟩
    /-
      case mpr
      α : Type u_2
      ι : Type u_5
      inst✝¹ : LinearOrder α
      inst✝ : OrderBot α
      s : Finset ι
      f : ι → α
      a : α
      ha : LT.lt Bot.bot a
      ⊢ (Exists fun b => And (Membership.mem s b) (LE.le a (f b))) → LE.le a (s.sup f)
    -/
  · exact fun ⟨b, hb, hle⟩ => le_trans hle (le_sup hb)
    /-
      🎉 no goals
    -/


protected theorem sup_eq_top_iff {α : Type*} [LinearOrder α] [BoundedOrder α] [Nontrivial α]
    {s : Finset ι} {f : ι → α} : s.sup f = ⊤ ↔ ∃ b ∈ s, f b = ⊤ := by
  /-
    ι : Type u_5
    α : Type u_7
    inst✝² : LinearOrder α
    inst✝¹ : BoundedOrder α
    inst✝ : Nontrivial α
    s : Finset ι
    f : ι → α
    ⊢ Iff (Eq (s.sup f) Top.top) (Exists fun b => And (Membership.mem s b) (Eq (f  …
  -/
  simp only [← top_le_iff]
  /-
    ι : Type u_5
    α : Type u_7
    inst✝² : LinearOrder α
    inst✝¹ : BoundedOrder α
    inst✝ : Nontrivial α
    s : Finset ι
    f : ι → α
    ⊢ Iff (LE.le Top.top (s.sup f)) (Exists fun b => And (Membership.mem s b) (LE. …
  -/
  exact Finset.le_sup_iff bot_lt_top
  /-
    🎉 no goals
  -/


protected theorem Nonempty.sup_eq_top_iff {α : Type*} [LinearOrder α] [BoundedOrder α]
    {s : Finset ι} {f : ι → α} (hs : s.Nonempty) : s.sup f = ⊤ ↔ ∃ b ∈ s, f b = ⊤ := by
  /-
    ι : Type u_5
    α : Type u_7
    inst✝¹ : LinearOrder α
    inst✝ : BoundedOrder α
    s : Finset ι
    f : ι → α
    hs : s.Nonempty
    ⊢ Iff (Eq (s.sup f) Top.top) (Exists fun b => And (Membership.mem s b) (Eq (f  …
  -/
  cases subsingleton_or_nontrivial α
    /-
      case inl
      ι : Type u_5
      α : Type u_7
      inst✝¹ : LinearOrder α
      inst✝ : BoundedOrder α
      s : Finset ι
      f : ι → α
      hs : s.Nonempty
      h✝ : Subsingleton α
      ⊢ Iff (Eq (s.sup f) Top.top) (Exists fun b => And (Membership.mem s b) (Eq (f  …
    -/
  · simpa [Subsingleton.elim _ (⊤ : α)]
    /-
      🎉 no goals
    -/
    /-
      case inr
      ι : Type u_5
      α : Type u_7
      inst✝¹ : LinearOrder α
      inst✝ : BoundedOrder α
      s : Finset ι
      f : ι → α
      hs : s.Nonempty
      h✝ : Nontrivial α
      ⊢ Iff (Eq (s.sup f) Top.top) (Exists fun b => And (Membership.mem s b) (Eq (f  …
    -/
  · exact Finset.sup_eq_top_iff
    /-
      🎉 no goals
    -/


@[simp]
protected theorem lt_sup_iff : a < s.sup f ↔ ∃ b ∈ s, a < f b := by
  /-
    α : Type u_2
    ι : Type u_5
    inst✝¹ : LinearOrder α
    inst✝ : OrderBot α
    s : Finset ι
    f : ι → α
    a : α
    ⊢ Iff (LT.lt a (s.sup f)) (Exists fun b => And (Membership.mem s b) (LT.lt a ( …
  -/
  apply Iff.intro
  · induction s using cons_induction with
    | empty => exact (absurd · not_lt_bot)
    | cons c t hc ih =>
      rw [sup_cons, lt_sup_iff]
      exact fun
      | Or.inl h => ⟨c, mem_cons.2 (Or.inl rfl), h⟩
      | Or.inr h => let ⟨b, hb, hlt⟩ := ih h; ⟨b, mem_cons.2 (Or.inr hb), hlt⟩
    /-
      case mpr
      α : Type u_2
      ι : Type u_5
      inst✝¹ : LinearOrder α
      inst✝ : OrderBot α
      s : Finset ι
      f : ι → α
      a : α
      ⊢ (Exists fun b => And (Membership.mem s b) (LT.lt a (f b))) → LT.lt a (s.sup f)
    -/
  · exact fun ⟨b, hb, hlt⟩ => lt_of_lt_of_le hlt (le_sup hb)
    /-
      🎉 no goals
    -/


@[simp]
protected theorem sup_lt_iff (ha : ⊥ < a) : s.sup f < a ↔ ∀ b ∈ s, f b < a :=
  ⟨fun hs _ hb => lt_of_le_of_lt (le_sup hb) hs,
    Finset.cons_induction_on s (fun _ => ha) fun c t hc => by
      /-
        α : Type u_2
        ι : Type u_5
        inst✝¹ : LinearOrder α
        inst✝ : OrderBot α
        s : Finset ι
        f : ι → α
        a : α
        ha : LT.lt Bot.bot a
        c : ι
        t : Finset ι
        hc : Not (Membership.mem t c)
        ⊢ ((∀ (b : ι), Membership.mem t b → LT.lt (f b) a) → LT.lt (t.sup f) a) → (∀ ( …
      -/
      simpa only [sup_cons, sup_lt_iff, mem_cons, forall_eq_or_imp] using And.imp_right⟩
      /-
        🎉 no goals
      -/


theorem sup_mem_of_nonempty (hs : s.Nonempty) : s.sup f ∈ f '' s := by
  classical
  induction s using Finset.induction with
  | empty => exfalso; simp only [Finset.not_nonempty_empty] at hs
  | @insert a s _ h =>
    rw [Finset.sup_insert (b := a) (s := s) (f := f)]
    by_cases hs : s = ∅
    · simp [hs]
    · rw [← ne_eq, ← Finset.nonempty_iff_ne_empty] at hs
      simp only [Finset.coe_insert]
      rcases le_total (f a) (s.sup f) with (ha | ha)
      · rw [sup_eq_right.mpr ha]
        exact Set.image_mono (Set.subset_insert a s) (h hs)
      · rw [sup_eq_left.mpr ha]
        apply Set.mem_image_of_mem _ (Set.mem_insert a ↑s)


theorem comp_inf_eq_inf_comp_of_is_total [SemilatticeInf β] [OrderTop β] (g : α → β)
    (mono_g : Monotone g) (top : g ⊤ = ⊤) : g (s.inf f) = s.inf (g ∘ f) :=
  comp_inf_eq_inf_comp g mono_g.map_inf top


@[simp]
protected theorem inf_le_iff (ha : a < ⊤) : s.inf f ≤ a ↔ ∃ b ∈ s, f b ≤ a :=
  @Finset.le_sup_iff αᵒᵈ _ _ _ _ _ _ ha


protected theorem inf_eq_bot_iff {α : Type*} [LinearOrder α] [BoundedOrder α] [Nontrivial α]
    {s : Finset ι} {f : ι → α} : s.inf f = ⊥ ↔ ∃ b ∈ s, f b = ⊥ :=
  Finset.sup_eq_top_iff (α := αᵒᵈ)


protected theorem Nonempty.inf_eq_bot_iff {α : Type*} [LinearOrder α] [BoundedOrder α]
    {s : Finset ι} {f : ι → α} (h : s.Nonempty) : s.inf f = ⊥ ↔ ∃ b ∈ s, f b = ⊥ :=
  h.sup_eq_top_iff (α := αᵒᵈ)


@[simp]
protected theorem inf_lt_iff : s.inf f < a ↔ ∃ b ∈ s, f b < a :=
  @Finset.lt_sup_iff αᵒᵈ _ _ _ _ _ _


@[simp]
protected theorem lt_inf_iff (ha : a < ⊤) : a < s.inf f ↔ ∀ b ∈ s, a < f b :=
  @Finset.sup_lt_iff αᵒᵈ _ _ _ _ _ _ ha


theorem inf_eq_iInf [CompleteLattice β] (s : Finset α) (f : α → β) : s.inf f = ⨅ a ∈ s, f a :=
  @sup_eq_iSup _ βᵒᵈ _ _ _


theorem inf_id_eq_sInf [CompleteLattice α] (s : Finset α) : s.inf id = sInf s :=
  @sup_id_eq_sSup αᵒᵈ _ _


theorem inf_id_set_eq_sInter (s : Finset (Set α)) : s.inf id = ⋂₀ ↑s :=
  inf_id_eq_sInf _


@[simp]
theorem inf_set_eq_iInter (s : Finset α) (f : α → Set β) : s.inf f = ⋂ x ∈ s, f x :=
  inf_eq_iInf _ _


theorem inf_eq_sInf_image [CompleteLattice β] (s : Finset α) (f : α → β) :
    s.inf f = sInf (f '' s) :=
  @sup_eq_sSup_image _ βᵒᵈ _ _ _


theorem sup_of_mem {s : Finset β} (f : β → α) {b : β} (h : b ∈ s) :
    ∃ a : α, s.sup ((↑) ∘ f : β → WithBot α) = ↑a :=
  Exists.imp (fun _ => And.left) (@le_sup (WithBot α) _ _ _ _ _ _ h (f b) rfl)


/-- Given nonempty finset `s` then `s.sup' H f` is the supremum of its image under `f` in (possibly
unbounded) join-semilattice `α`, where `H` is a proof of nonemptiness. If `α` has a bottom element
you may instead use `Finset.sup` which does not require `s` nonempty. -/
def sup' (s : Finset β) (H : s.Nonempty) (f : β → α) : α :=
                                      /-
                                        F : Type u_1
                                        α : Type u_2
                                        β : Type u_3
                                        γ : Type u_4
                                        ι : Type u_5
                                        κ : Type u_6
                                        inst✝ : SemilatticeSup α
                                        s : Finset β
                                        H : s.Nonempty
                                        f : β → α
                                        ⊢ Ne (s.sup (Function.comp WithBot.some f)) Bot.bot
                                      -/
  WithBot.unbot (s.sup ((↑) ∘ f)) (by simpa using H)
                                      /-
                                        🎉 no goals
                                      -/


@[simp]
theorem coe_sup' : ((s.sup' H f : α) : WithBot α) = s.sup ((↑) ∘ f) := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝ : SemilatticeSup α
    s : Finset β
    H : s.Nonempty
    f : β → α
    ⊢ Eq (↑(s.sup' H f)) (s.sup (Function.comp WithBot.some f))
  -/
  rw [sup', WithBot.coe_unbot]
  /-
    🎉 no goals
  -/


@[simp]
theorem sup'_cons {b : β} {hb : b ∉ s} :
    (cons b s hb).sup' (cons_nonempty hb) f = f b ⊔ s.sup' H f := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝ : SemilatticeSup α
    s : Finset β
    H : s.Nonempty
    f : β → α
    b : β
    hb : Not (Membership.mem s b)
    ⊢ Eq ((Finset.cons b s hb).sup' ⋯ f) (Max.max (f b) (s.sup' H f))
  -/
  rw [← WithBot.coe_eq_coe]
  /-
    α : Type u_2
    β : Type u_3
    inst✝ : SemilatticeSup α
    s : Finset β
    H : s.Nonempty
    f : β → α
    b : β
    hb : Not (Membership.mem s b)
    ⊢ Eq ↑((Finset.cons b s hb).sup' ⋯ f) ↑(Max.max (f b) (s.sup' H f))
  -/
  simp [WithBot.coe_sup]
  /-
    🎉 no goals
  -/


@[simp]
theorem sup'_insert [DecidableEq β] {b : β} :
    (insert b s).sup' (insert_nonempty _ _) f = f b ⊔ s.sup' H f := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : SemilatticeSup α
    s : Finset β
    H : s.Nonempty
    f : β → α
    inst✝ : DecidableEq β
    b : β
    ⊢ Eq ((Insert.insert b s).sup' ⋯ f) (Max.max (f b) (s.sup' H f))
  -/
  rw [← WithBot.coe_eq_coe]
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : SemilatticeSup α
    s : Finset β
    H : s.Nonempty
    f : β → α
    inst✝ : DecidableEq β
    b : β
    ⊢ Eq ↑((Insert.insert b s).sup' ⋯ f) ↑(Max.max (f b) (s.sup' H f))
  -/
  simp [WithBot.coe_sup]
  /-
    🎉 no goals
  -/


@[simp]
theorem sup'_singleton {b : β} : ({b} : Finset β).sup' (singleton_nonempty _) f = f b :=
  rfl


@[simp]
theorem sup'_le_iff {a : α} : s.sup' H f ≤ a ↔ ∀ b ∈ s, f b ≤ a := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝ : SemilatticeSup α
    s : Finset β
    H : s.Nonempty
    f : β → α
    a : α
    ⊢ Iff (LE.le (s.sup' H f) a) (∀ (b : β), Membership.mem s b → LE.le (f b) a)
  -/
  simp_rw [← @WithBot.coe_le_coe α, coe_sup', Finset.sup_le_iff]; rfl
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


alias ⟨_, sup'_le⟩ := sup'_le_iff


theorem le_sup' {b : β} (h : b ∈ s) : f b ≤ s.sup' ⟨b, h⟩ f :=
  (sup'_le_iff ⟨b, h⟩ f).1 le_rfl b h


set_option linter.docPrime false in
theorem isLUB_sup' {s : Finset α} (hs : s.Nonempty) : IsLUB s (sup' s hs id) :=
  ⟨fun x h => id_eq x ▸ le_sup' id h, fun _ h => Finset.sup'_le hs id h⟩


theorem le_sup'_of_le {a : α} {b : β} (hb : b ∈ s) (h : a ≤ f b) : a ≤ s.sup' ⟨b, hb⟩ f :=
  h.trans <| le_sup' _ hb


lemma sup'_eq_of_forall {a : α} (h : ∀ b ∈ s, f b = a) : s.sup' H f = a :=
  le_antisymm (sup'_le _ _ (fun _ hb ↦ (h _ hb).le))
    (le_sup'_of_le _ H.choose_spec (h _ H.choose_spec).ge)


@[simp]
theorem sup'_const (a : α) : s.sup' H (fun _ => a) = a := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝ : SemilatticeSup α
    s : Finset β
    H : s.Nonempty
    a : α
    ⊢ Eq (s.sup' H fun x => a) a
  -/
  apply le_antisymm
    /-
      case a
      α : Type u_2
      β : Type u_3
      inst✝ : SemilatticeSup α
      s : Finset β
      H : s.Nonempty
      a : α
      ⊢ LE.le (s.sup' H fun x => a) a
    -/
  · apply sup'_le
    /-
      case a.a
      α : Type u_2
      β : Type u_3
      inst✝ : SemilatticeSup α
      s : Finset β
      H : s.Nonempty
      a : α
      ⊢ ∀ (b : β), Membership.mem s b → LE.le a a
    -/
    intros
    /-
      case a.a
      α : Type u_2
      β : Type u_3
      inst✝ : SemilatticeSup α
      s : Finset β
      H : s.Nonempty
      a : α
      b✝ : β
      a✝ : Membership.mem s b✝
      ⊢ LE.le a a
    -/
    exact le_rfl
    /-
      🎉 no goals
    -/
    /-
      case a
      α : Type u_2
      β : Type u_3
      inst✝ : SemilatticeSup α
      s : Finset β
      H : s.Nonempty
      a : α
      ⊢ LE.le a (s.sup' H fun x => a)
    -/
  · apply le_sup' (fun _ => a) H.choose_spec
    /-
      🎉 no goals
    -/


theorem sup'_union [DecidableEq β] {s₁ s₂ : Finset β} (h₁ : s₁.Nonempty) (h₂ : s₂.Nonempty)
    (f : β → α) :
    (s₁ ∪ s₂).sup' (h₁.mono subset_union_left) f = s₁.sup' h₁ f ⊔ s₂.sup' h₂ f :=
                                  /-
                                    α : Type u_2
                                    β : Type u_3
                                    inst✝¹ : SemilatticeSup α
                                    inst✝ : DecidableEq β
                                    s₁ s₂ : Finset β
                                    h₁ : s₁.Nonempty
                                    h₂ : s₂.Nonempty
                                    f : β → α
                                    a : α
                                    ⊢ Iff (LE.le ((Union.union s₁ s₂).sup' ⋯ f) a) (LE.le (Max.max (s₁.sup' h₁ f)  …
                                  -/
  eq_of_forall_ge_iff fun a => by simp [or_imp, forall_and]
                                  /-
                                    🎉 no goals
                                  -/


theorem sup'_biUnion [DecidableEq β] {s : Finset γ} (Hs : s.Nonempty) {t : γ → Finset β}
    (Ht : ∀ b, (t b).Nonempty) :
    (s.biUnion t).sup' (Hs.biUnion fun b _ => Ht b) f = s.sup' Hs (fun b => (t b).sup' (Ht b) f) :=
                                  /-
                                    α : Type u_2
                                    β : Type u_3
                                    γ : Type u_4
                                    inst✝¹ : SemilatticeSup α
                                    f : β → α
                                    inst✝ : DecidableEq β
                                    s : Finset γ
                                    Hs : s.Nonempty
                                    t : γ → Finset β
                                    Ht : ∀ (b : γ), (t b).Nonempty
                                    c : α
                                    ⊢ Iff (LE.le ((s.biUnion t).sup' ⋯ f) c) (LE.le (s.sup' Hs fun b => (t b).sup' …
                                  -/
  eq_of_forall_ge_iff fun c => by simp [@forall_swap _ β]
                                  /-
                                    🎉 no goals
                                  -/


protected theorem sup'_comm {t : Finset γ} (hs : s.Nonempty) (ht : t.Nonempty) (f : β → γ → α) :
    (s.sup' hs fun b => t.sup' ht (f b)) = t.sup' ht fun c => s.sup' hs fun b => f b c :=
                                  /-
                                    α : Type u_2
                                    β : Type u_3
                                    γ : Type u_4
                                    inst✝ : SemilatticeSup α
                                    s : Finset β
                                    t : Finset γ
                                    hs : s.Nonempty
                                    ht : t.Nonempty
                                    f : β → γ → α
                                    a : α
                                    ⊢ Iff (LE.le (s.sup' hs fun b => t.sup' ht (f b)) a) (LE.le (t.sup' ht fun c = …
                                  -/
  eq_of_forall_ge_iff fun a => by simpa using forall₂_swap
                                  /-
                                    🎉 no goals
                                  -/


theorem sup'_product_left {t : Finset γ} (h : (s ×ˢ t).Nonempty) (f : β × γ → α) :
    (s ×ˢ t).sup' h f = s.sup' h.fst fun i => t.sup' h.snd fun i' => f ⟨i, i'⟩ :=
                                  /-
                                    α : Type u_2
                                    β : Type u_3
                                    γ : Type u_4
                                    inst✝ : SemilatticeSup α
                                    s : Finset β
                                    t : Finset γ
                                    h : (SProd.sprod s t).Nonempty
                                    f : Prod β γ → α
                                    a : α
                                    ⊢ Iff (LE.le ((SProd.sprod s t).sup' h f) a) (LE.le (s.sup' ⋯ fun i => t.sup'  …
                                  -/
  eq_of_forall_ge_iff fun a => by simp [@forall_swap _ γ]
                                  /-
                                    🎉 no goals
                                  -/


theorem sup'_product_right {t : Finset γ} (h : (s ×ˢ t).Nonempty) (f : β × γ → α) :
    (s ×ˢ t).sup' h f = t.sup' h.snd fun i' => s.sup' h.fst fun i => f ⟨i, i'⟩ := by
  /-
    α : Type u_2
    β : Type u_3
    γ : Type u_4
    inst✝ : SemilatticeSup α
    s : Finset β
    t : Finset γ
    h : (SProd.sprod s t).Nonempty
    f : Prod β γ → α
    ⊢ Eq ((SProd.sprod s t).sup' h f) (t.sup' ⋯ fun i' => s.sup' ⋯ fun i => f { fs …
  -/
  rw [sup'_product_left, Finset.sup'_comm]
  /-
    🎉 no goals
  -/


/-- See also `Finset.sup'_prodMap`. -/
lemma prodMk_sup'_sup' (hs : s.Nonempty) (ht : t.Nonempty) (f : ι → α) (g : κ → β) :
    (sup' s hs f, sup' t ht g) = sup' (s ×ˢ t) (hs.product ht) (Prod.map f g) :=
  eq_of_forall_ge_iff fun i ↦ by
    /-
      ι : Type u_7
      κ : Type u_8
      α : Type u_9
      β : Type u_10
      inst✝¹ : SemilatticeSup α
      inst✝ : SemilatticeSup β
      s : Finset ι
      t : Finset κ
      hs : s.Nonempty
      ht : t.Nonempty
      f : ι → α
      g : κ → β
      i : Prod α β
      ⊢ Iff (LE.le { fst := s.sup' hs f, snd := t.sup' ht g } i) (LE.le ((SProd.spro …
    -/
    obtain ⟨a, ha⟩ := hs
    /-
      case intro
      ι : Type u_7
      κ : Type u_8
      α : Type u_9
      β : Type u_10
      inst✝¹ : SemilatticeSup α
      inst✝ : SemilatticeSup β
      s : Finset ι
      t : Finset κ
      ht : t.Nonempty
      f : ι → α
      g : κ → β
      i : Prod α β
      a : ι
      ha : Membership.mem s a
      ⊢ Iff (LE.le { fst := s.sup' ⋯ f, snd := t.sup' ht g } i) (LE.le ((SProd.sprod …
    -/
    obtain ⟨b, hb⟩ := ht
    /-
      case intro.intro
      ι : Type u_7
      κ : Type u_8
      α : Type u_9
      β : Type u_10
      inst✝¹ : SemilatticeSup α
      inst✝ : SemilatticeSup β
      s : Finset ι
      t : Finset κ
      f : ι → α
      g : κ → β
      i : Prod α β
      a : ι
      ha : Membership.mem s a
      b : κ
      hb : Membership.mem t b
      ⊢ Iff (LE.le { fst := s.sup' ⋯ f, snd := t.sup' ⋯ g } i) (LE.le ((SProd.sprod  …
    -/
    simp only [Prod.map, sup'_le_iff, mem_product, and_imp, Prod.forall, Prod.le_def]
    /-
      case intro.intro
      ι : Type u_7
      κ : Type u_8
      α : Type u_9
      β : Type u_10
      inst✝¹ : SemilatticeSup α
      inst✝ : SemilatticeSup β
      s : Finset ι
      t : Finset κ
      f : ι → α
      g : κ → β
      i : Prod α β
      a : ι
      ha : Membership.mem s a
      b : κ
      hb : Membership.mem t b
      ⊢ Iff (And (∀ (b : ι), Membership.mem s b → LE.le (f b) i.1) (∀ (b : κ), Membe …
    -/
    exact ⟨by aesop, fun h ↦ ⟨fun i hi ↦ (h _ _ hi hb).1, fun j hj ↦ (h _ _ ha hj).2⟩⟩
    /-
      🎉 no goals
    -/


/-- See also `Finset.prodMk_sup'_sup'`. -/
-- @[simp] -- TODO: Why does `Prod.map_apply` simplify the LHS?
lemma sup'_prodMap (hst : (s ×ˢ t).Nonempty) (f : ι → α) (g : κ → β) :
    sup' (s ×ˢ t) hst (Prod.map f g) = (sup' s hst.fst f, sup' t hst.snd g) :=
  (prodMk_sup'_sup' _ _ _ _).symm


theorem sup'_induction {p : α → Prop} (hp : ∀ a₁, p a₁ → ∀ a₂, p a₂ → p (a₁ ⊔ a₂))
    (hs : ∀ b ∈ s, p (f b)) : p (s.sup' H f) := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝ : SemilatticeSup α
    s : Finset β
    H : s.Nonempty
    f : β → α
    p : α → Prop
    hp : ∀ (a₁ : α), p a₁ → ∀ (a₂ : α), p a₂ → p (Max.max a₁ a₂)
    hs : ∀ (b : β), Membership.mem s b → p (f b)
    ⊢ p (s.sup' H f)
  -/
  show @WithBot.recBotCoe α (fun _ => Prop) True p ↑(s.sup' H f)
  /-
    α : Type u_2
    β : Type u_3
    inst✝ : SemilatticeSup α
    s : Finset β
    H : s.Nonempty
    f : β → α
    p : α → Prop
    hp : ∀ (a₁ : α), p a₁ → ∀ (a₂ : α), p a₂ → p (Max.max a₁ a₂)
    hs : ∀ (b : β), Membership.mem s b → p (f b)
    ⊢ WithBot.recBotCoe True p ↑(s.sup' H f)
  -/
  rw [coe_sup']
  /-
    α : Type u_2
    β : Type u_3
    inst✝ : SemilatticeSup α
    s : Finset β
    H : s.Nonempty
    f : β → α
    p : α → Prop
    hp : ∀ (a₁ : α), p a₁ → ∀ (a₂ : α), p a₂ → p (Max.max a₁ a₂)
    hs : ∀ (b : β), Membership.mem s b → p (f b)
    ⊢ WithBot.recBotCoe True p (s.sup (Function.comp WithBot.some f))
  -/
  refine sup_induction trivial (fun a₁ h₁ a₂ h₂ ↦ ?_) hs
  match a₁, a₂ with
  | ⊥, _ => rwa [bot_sup_eq]
  | (a₁ : α), ⊥ => rwa [sup_bot_eq]
  | (a₁ : α), (a₂ : α) => exact hp a₁ h₁ a₂ h₂


theorem sup'_mem (s : Set α) (w : ∀ᵉ (x ∈ s) (y ∈ s), x ⊔ y ∈ s) {ι : Type*}
    (t : Finset ι) (H : t.Nonempty) (p : ι → α) (h : ∀ i ∈ t, p i ∈ s) : t.sup' H p ∈ s :=
  sup'_induction H p w h


@[congr]
theorem sup'_congr {t : Finset β} {f g : β → α} (h₁ : s = t) (h₂ : ∀ x ∈ s, f x = g x) :
    s.sup' H f = t.sup' (h₁ ▸ H) g := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝ : SemilatticeSup α
    s : Finset β
    H : s.Nonempty
    t : Finset β
    f g : β → α
    h₁ : Eq s t
    h₂ : ∀ (x : β), Membership.mem s x → Eq (f x) (g x)
    ⊢ Eq (s.sup' H f) (t.sup' ⋯ g)
  -/
  subst s
  /-
    α : Type u_2
    β : Type u_3
    inst✝ : SemilatticeSup α
    t : Finset β
    f g : β → α
    H : t.Nonempty
    h₂ : ∀ (x : β), Membership.mem t x → Eq (f x) (g x)
    ⊢ Eq (t.sup' H f) (t.sup' ⋯ g)
  -/
  refine eq_of_forall_ge_iff fun c => ?_
  /-
    α : Type u_2
    β : Type u_3
    inst✝ : SemilatticeSup α
    t : Finset β
    f g : β → α
    H : t.Nonempty
    h₂ : ∀ (x : β), Membership.mem t x → Eq (f x) (g x)
    c : α
    ⊢ Iff (LE.le (t.sup' H f) c) (LE.le (t.sup' ⋯ g) c)
  -/
  simp +contextual only [sup'_le_iff, h₂]
  /-
    🎉 no goals
  -/


theorem comp_sup'_eq_sup'_comp [SemilatticeSup γ] {s : Finset β} (H : s.Nonempty) {f : β → α}
    (g : α → γ) (g_sup : ∀ x y, g (x ⊔ y) = g x ⊔ g y) : g (s.sup' H f) = s.sup' H (g ∘ f) := by
  /-
    α : Type u_2
    β : Type u_3
    γ : Type u_4
    inst✝¹ : SemilatticeSup α
    inst✝ : SemilatticeSup γ
    s : Finset β
    H : s.Nonempty
    f : β → α
    g : α → γ
    g_sup : ∀ (x y : α), Eq (g (Max.max x y)) (Max.max (g x) (g y))
    ⊢ Eq (g (s.sup' H f)) (s.sup' H (Function.comp g f))
  -/
                                               /-
                                                 🎉 no goals
                                               -/
  refine H.cons_induction ?_ ?_ <;> intros <;> simp [*]
                                               /-
                                                 🎉 no goals
                                               -/


@[simp]
theorem _root_.map_finset_sup' [SemilatticeSup β] [FunLike F α β] [SupHomClass F α β]
    (f : F) {s : Finset ι} (hs) (g : ι → α) :
    f (s.sup' hs g) = s.sup' hs (f ∘ g) := by
  /-
    F : Type u_1
    α : Type u_2
    β : Type u_3
    ι : Type u_5
    inst✝³ : SemilatticeSup α
    inst✝² : SemilatticeSup β
    inst✝¹ : FunLike F α β
    inst✝ : SupHomClass F α β
    f : F
    s : Finset ι
    hs : s.Nonempty
    g : ι → α
    ⊢ Eq (f (s.sup' hs g)) (s.sup' hs (Function.comp (⇑f) g))
  -/
                                                /-
                                                  🎉 no goals
                                                -/
  refine hs.cons_induction ?_ ?_ <;> intros <;> simp [*]
                                                /-
                                                  🎉 no goals
                                                -/


/-- To rewrite from right to left, use `Finset.sup'_comp_eq_image`. -/
@[simp]
theorem sup'_image [DecidableEq β] {s : Finset γ} {f : γ → β} (hs : (s.image f).Nonempty)
    (g : β → α) :
    (s.image f).sup' hs g = s.sup' hs.of_image (g ∘ f) := by
  /-
    α : Type u_2
    β : Type u_3
    γ : Type u_4
    inst✝¹ : SemilatticeSup α
    inst✝ : DecidableEq β
    s : Finset γ
    f : γ → β
    hs : (Finset.image f s).Nonempty
    g : β → α
    ⊢ Eq ((Finset.image f s).sup' hs g) (s.sup' ⋯ (Function.comp g f))
  -/
  rw [← WithBot.coe_eq_coe]; simp only [coe_sup', sup_image, WithBot.coe_sup]; rfl
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


/-- A version of `Finset.sup'_image` with LHS and RHS reversed.
Also, this lemma assumes that `s` is nonempty instead of assuming that its image is nonempty. -/
lemma sup'_comp_eq_image [DecidableEq β] {s : Finset γ} {f : γ → β} (hs : s.Nonempty) (g : β → α) :
    s.sup' hs (g ∘ f) = (s.image f).sup' (hs.image f) g :=
  .symm <| sup'_image _ _


/-- To rewrite from right to left, use `Finset.sup'_comp_eq_map`. -/
@[simp]
theorem sup'_map {s : Finset γ} {f : γ ↪ β} (g : β → α) (hs : (s.map f).Nonempty) :
    (s.map f).sup' hs g = s.sup' (map_nonempty.1 hs) (g ∘ f) := by
  /-
    α : Type u_2
    β : Type u_3
    γ : Type u_4
    inst✝ : SemilatticeSup α
    s : Finset γ
    f : Function.Embedding γ β
    g : β → α
    hs : (Finset.map f s).Nonempty
    ⊢ Eq ((Finset.map f s).sup' hs g) (s.sup' ⋯ (Function.comp g ⇑f))
  -/
  rw [← WithBot.coe_eq_coe, coe_sup', sup_map, coe_sup']
  /-
    α : Type u_2
    β : Type u_3
    γ : Type u_4
    inst✝ : SemilatticeSup α
    s : Finset γ
    f : Function.Embedding γ β
    g : β → α
    hs : (Finset.map f s).Nonempty
    ⊢ Eq (s.sup (Function.comp (Function.comp WithBot.some g) ⇑f)) (s.sup (Functio …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- A version of `Finset.sup'_map` with LHS and RHS reversed.
Also, this lemma assumes that `s` is nonempty instead of assuming that its image is nonempty. -/
lemma sup'_comp_eq_map {s : Finset γ} {f : γ ↪ β} (g : β → α) (hs : s.Nonempty) :
    s.sup' hs (g ∘ f) = (s.map f).sup' (map_nonempty.2 hs) g :=
  .symm <| sup'_map _ _



@[gcongr]
theorem sup'_mono {s₁ s₂ : Finset β} (h : s₁ ⊆ s₂) (h₁ : s₁.Nonempty) :
    s₁.sup' h₁ f ≤ s₂.sup' (h₁.mono h) f :=
  Finset.sup'_le h₁ _ (fun _ hb => le_sup' _ (h hb))


@[gcongr]
lemma sup'_mono_fun {hs : s.Nonempty} {f g : β → α} (h : ∀ b ∈ s, f b ≤ g b) :
    s.sup' hs f ≤ s.sup' hs g := sup'_le _ _ fun b hb ↦ (h b hb).trans (le_sup' _ hb)


theorem inf_of_mem {s : Finset β} (f : β → α) {b : β} (h : b ∈ s) :
    ∃ a : α, s.inf ((↑) ∘ f : β → WithTop α) = ↑a :=
  @sup_of_mem αᵒᵈ _ _ _ f _ h


/-- Given nonempty finset `s` then `s.inf' H f` is the infimum of its image under `f` in (possibly
unbounded) meet-semilattice `α`, where `H` is a proof of nonemptiness. If `α` has a top element you
may instead use `Finset.inf` which does not require `s` nonempty. -/
def inf' (s : Finset β) (H : s.Nonempty) (f : β → α) : α :=
                                      /-
                                        F : Type u_1
                                        α : Type u_2
                                        β : Type u_3
                                        γ : Type u_4
                                        ι : Type u_5
                                        κ : Type u_6
                                        inst✝ : SemilatticeInf α
                                        s : Finset β
                                        H : s.Nonempty
                                        f : β → α
                                        ⊢ Ne (s.inf (Function.comp WithTop.some f)) Top.top
                                      -/
  WithTop.untop (s.inf ((↑) ∘ f)) (by simpa using H)
                                      /-
                                        🎉 no goals
                                      -/


@[simp]
theorem coe_inf' : ((s.inf' H f : α) : WithTop α) = s.inf ((↑) ∘ f) :=
  @coe_sup' αᵒᵈ _ _ _ H f


@[simp]
theorem inf'_cons {b : β} {hb : b ∉ s} :
    (cons b s hb).inf' (cons_nonempty hb) f = f b ⊓ s.inf' H f :=
  @sup'_cons αᵒᵈ _ _ _ H f _ _


@[simp]
theorem inf'_insert [DecidableEq β] {b : β} :
    (insert b s).inf' (insert_nonempty _ _) f = f b ⊓ s.inf' H f :=
  @sup'_insert αᵒᵈ _ _ _ H f _ _


@[simp]
theorem inf'_singleton {b : β} : ({b} : Finset β).inf' (singleton_nonempty _) f = f b :=
  rfl


@[simp]
theorem le_inf'_iff {a : α} : a ≤ s.inf' H f ↔ ∀ b ∈ s, a ≤ f b :=
  sup'_le_iff (α := αᵒᵈ) H f


theorem le_inf' {a : α} (hs : ∀ b ∈ s, a ≤ f b) : a ≤ s.inf' H f :=
  sup'_le (α := αᵒᵈ) H f hs


theorem inf'_le {b : β} (h : b ∈ s) : s.inf' ⟨b, h⟩ f ≤ f b :=
  le_sup' (α := αᵒᵈ) f h


set_option linter.docPrime false in
theorem isGLB_inf' {s : Finset α} (hs : s.Nonempty) : IsGLB s (inf' s hs id) :=
  ⟨fun x h => id_eq x ▸ inf'_le id h, fun _ h => Finset.le_inf' hs id h⟩


theorem inf'_le_of_le {a : α} {b : β} (hb : b ∈ s) (h : f b ≤ a) :
    s.inf' ⟨b, hb⟩ f ≤ a := (inf'_le _ hb).trans h


lemma inf'_eq_of_forall {a : α} (h : ∀ b ∈ s, f b = a) : s.inf' H f = a :=
  sup'_eq_of_forall (α := αᵒᵈ) H f h


@[simp]
theorem inf'_const (a : α) : (s.inf' H fun _ => a) = a :=
  sup'_const (α := αᵒᵈ) H a


theorem inf'_union [DecidableEq β] {s₁ s₂ : Finset β} (h₁ : s₁.Nonempty) (h₂ : s₂.Nonempty)
    (f : β → α) :
    (s₁ ∪ s₂).inf' (h₁.mono subset_union_left) f = s₁.inf' h₁ f ⊓ s₂.inf' h₂ f :=
  @sup'_union αᵒᵈ _ _ _ _ _ h₁ h₂ _


theorem inf'_biUnion [DecidableEq β] {s : Finset γ} (Hs : s.Nonempty) {t : γ → Finset β}
    (Ht : ∀ b, (t b).Nonempty) :
    (s.biUnion t).inf' (Hs.biUnion fun b _ => Ht b) f = s.inf' Hs (fun b => (t b).inf' (Ht b) f) :=
  sup'_biUnion (α := αᵒᵈ) _ Hs Ht


protected theorem inf'_comm {t : Finset γ} (hs : s.Nonempty) (ht : t.Nonempty) (f : β → γ → α) :
    (s.inf' hs fun b => t.inf' ht (f b)) = t.inf' ht fun c => s.inf' hs fun b => f b c :=
  @Finset.sup'_comm αᵒᵈ _ _ _ _ _ hs ht _


theorem inf'_product_left {t : Finset γ} (h : (s ×ˢ t).Nonempty) (f : β × γ → α) :
    (s ×ˢ t).inf' h f = s.inf' h.fst fun i => t.inf' h.snd fun i' => f ⟨i, i'⟩ :=
  sup'_product_left (α := αᵒᵈ) h f


theorem inf'_product_right {t : Finset γ} (h : (s ×ˢ t).Nonempty) (f : β × γ → α) :
    (s ×ˢ t).inf' h f = t.inf' h.snd fun i' => s.inf' h.fst fun i => f ⟨i, i'⟩ :=
  sup'_product_right (α := αᵒᵈ) h f


/-- See also `Finset.inf'_prodMap`. -/
lemma prodMk_inf'_inf' (hs : s.Nonempty) (ht : t.Nonempty) (f : ι → α) (g : κ → β) :
    (inf' s hs f, inf' t ht g) = inf' (s ×ˢ t) (hs.product ht) (Prod.map f g) :=
  prodMk_sup'_sup' (α := αᵒᵈ) (β := βᵒᵈ) hs ht _ _


/-- See also `Finset.prodMk_inf'_inf'`. -/
-- @[simp] -- TODO: Why does `Prod.map_apply` simplify the LHS?
lemma inf'_prodMap (hst : (s ×ˢ t).Nonempty) (f : ι → α) (g : κ → β) :
    inf' (s ×ˢ t) hst (Prod.map f g) = (inf' s hst.fst f, inf' t hst.snd g) :=
  (prodMk_inf'_inf' _ _ _ _).symm


theorem comp_inf'_eq_inf'_comp [SemilatticeInf γ] {s : Finset β} (H : s.Nonempty) {f : β → α}
    (g : α → γ) (g_inf : ∀ x y, g (x ⊓ y) = g x ⊓ g y) : g (s.inf' H f) = s.inf' H (g ∘ f) :=
  comp_sup'_eq_sup'_comp (α := αᵒᵈ) (γ := γᵒᵈ) H g g_inf


theorem inf'_induction {p : α → Prop} (hp : ∀ a₁, p a₁ → ∀ a₂, p a₂ → p (a₁ ⊓ a₂))
    (hs : ∀ b ∈ s, p (f b)) : p (s.inf' H f) :=
  sup'_induction (α := αᵒᵈ) H f hp hs


theorem inf'_mem (s : Set α) (w : ∀ᵉ (x ∈ s) (y ∈ s), x ⊓ y ∈ s) {ι : Type*}
    (t : Finset ι) (H : t.Nonempty) (p : ι → α) (h : ∀ i ∈ t, p i ∈ s) : t.inf' H p ∈ s :=
  inf'_induction H p w h


@[congr]
theorem inf'_congr {t : Finset β} {f g : β → α} (h₁ : s = t) (h₂ : ∀ x ∈ s, f x = g x) :
    s.inf' H f = t.inf' (h₁ ▸ H) g :=
  sup'_congr (α := αᵒᵈ) H h₁ h₂


@[simp]
theorem _root_.map_finset_inf' [SemilatticeInf β] [FunLike F α β] [InfHomClass F α β]
    (f : F) {s : Finset ι} (hs) (g : ι → α) :
    f (s.inf' hs g) = s.inf' hs (f ∘ g) := by
  /-
    F : Type u_1
    α : Type u_2
    β : Type u_3
    ι : Type u_5
    inst✝³ : SemilatticeInf α
    inst✝² : SemilatticeInf β
    inst✝¹ : FunLike F α β
    inst✝ : InfHomClass F α β
    f : F
    s : Finset ι
    hs : s.Nonempty
    g : ι → α
    ⊢ Eq (f (s.inf' hs g)) (s.inf' hs (Function.comp (⇑f) g))
  -/
                                                /-
                                                  🎉 no goals
                                                -/
  refine hs.cons_induction ?_ ?_ <;> intros <;> simp [*]
                                                /-
                                                  🎉 no goals
                                                -/


/-- To rewrite from right to left, use `Finset.inf'_comp_eq_image`. -/
@[simp]
theorem inf'_image [DecidableEq β] {s : Finset γ} {f : γ → β} (hs : (s.image f).Nonempty)
    (g : β → α)  :
    (s.image f).inf' hs g = s.inf' hs.of_image (g ∘ f) :=
  @sup'_image αᵒᵈ _ _ _ _ _ _ hs _


/-- A version of `Finset.inf'_image` with LHS and RHS reversed.
Also, this lemma assumes that `s` is nonempty instead of assuming that its image is nonempty. -/
lemma inf'_comp_eq_image [DecidableEq β] {s : Finset γ} {f : γ → β} (hs : s.Nonempty) (g : β → α) :
    s.inf' hs (g ∘ f) = (s.image f).inf' (hs.image f) g :=
  sup'_comp_eq_image (α := αᵒᵈ) hs g


/-- To rewrite from right to left, use `Finset.inf'_comp_eq_map`. -/
@[simp]
theorem inf'_map {s : Finset γ} {f : γ ↪ β} (g : β → α) (hs : (s.map f).Nonempty) :
    (s.map f).inf' hs g = s.inf' (map_nonempty.1 hs) (g ∘ f) :=
  sup'_map (α := αᵒᵈ) _ hs


/-- A version of `Finset.inf'_map` with LHS and RHS reversed.
Also, this lemma assumes that `s` is nonempty instead of assuming that its image is nonempty. -/
lemma inf'_comp_eq_map {s : Finset γ} {f : γ ↪ β} (g : β → α) (hs : s.Nonempty) :
    s.inf' hs (g ∘ f) = (s.map f).inf' (map_nonempty.2 hs) g :=
  sup'_comp_eq_map (α := αᵒᵈ) g hs


@[gcongr]
theorem inf'_mono {s₁ s₂ : Finset β} (h : s₁ ⊆ s₂) (h₁ : s₁.Nonempty) :
    s₂.inf' (h₁.mono h) f ≤ s₁.inf' h₁ f :=
  Finset.le_inf' h₁ _ (fun _ hb => inf'_le _ (h hb))


theorem sup'_eq_sup {s : Finset β} (H : s.Nonempty) (f : β → α) : s.sup' H f = s.sup f :=
  le_antisymm (sup'_le H f fun _ => le_sup) (Finset.sup_le fun _ => le_sup' f)


theorem coe_sup_of_nonempty {s : Finset β} (h : s.Nonempty) (f : β → α) :
                                                     /-
                                                       α : Type u_2
                                                       β : Type u_3
                                                       inst✝¹ : SemilatticeSup α
                                                       inst✝ : OrderBot α
                                                       s : Finset β
                                                       h : s.Nonempty
                                                       f : β → α
                                                       ⊢ Eq (↑(s.sup f)) (s.sup (Function.comp WithBot.some f))
                                                     -/
    (↑(s.sup f) : WithBot α) = s.sup ((↑) ∘ f) := by simp only [← sup'_eq_sup h, coe_sup' h]
                                                     /-
                                                       🎉 no goals
                                                     -/


theorem inf'_eq_inf {s : Finset β} (H : s.Nonempty) (f : β → α) : s.inf' H f = s.inf f :=
  sup'_eq_sup (α := αᵒᵈ) H f


theorem coe_inf_of_nonempty {s : Finset β} (h : s.Nonempty) (f : β → α) :
    (↑(s.inf f) : WithTop α) = s.inf ((↑) ∘ f) :=
  coe_sup_of_nonempty (α := αᵒᵈ) h f


@[simp]
protected theorem sup_apply {C : β → Type*} [∀ b : β, SemilatticeSup (C b)]
    [∀ b : β, OrderBot (C b)] (s : Finset α) (f : α → ∀ b : β, C b) (b : β) :
    s.sup f b = s.sup fun a => f a b :=
  comp_sup_eq_sup_comp (fun x : ∀ b : β, C b => x b) (fun _ _ => rfl) rfl


@[simp]
protected theorem inf_apply {C : β → Type*} [∀ b : β, SemilatticeInf (C b)]
    [∀ b : β, OrderTop (C b)] (s : Finset α) (f : α → ∀ b : β, C b) (b : β) :
    s.inf f b = s.inf fun a => f a b :=
  Finset.sup_apply (C := fun b => (C b)ᵒᵈ) s f b


@[simp]
protected theorem sup'_apply {C : β → Type*} [∀ b : β, SemilatticeSup (C b)]
    {s : Finset α} (H : s.Nonempty) (f : α → ∀ b : β, C b) (b : β) :
    s.sup' H f b = s.sup' H fun a => f a b :=
  comp_sup'_eq_sup'_comp H (fun x : ∀ b : β, C b => x b) fun _ _ => rfl


@[simp]
protected theorem inf'_apply {C : β → Type*} [∀ b : β, SemilatticeInf (C b)]
    {s : Finset α} (H : s.Nonempty) (f : α → ∀ b : β, C b) (b : β) :
    s.inf' H f b = s.inf' H fun a => f a b :=
  Finset.sup'_apply (C := fun b => (C b)ᵒᵈ) H f b


@[simp]
theorem toDual_sup' [SemilatticeSup α] {s : Finset ι} (hs : s.Nonempty) (f : ι → α) :
    toDual (s.sup' hs f) = s.inf' hs (toDual ∘ f) :=
  rfl


@[simp]
theorem toDual_inf' [SemilatticeInf α] {s : Finset ι} (hs : s.Nonempty) (f : ι → α) :
    toDual (s.inf' hs f) = s.sup' hs (toDual ∘ f) :=
  rfl


@[simp]
theorem ofDual_sup' [SemilatticeInf α] {s : Finset ι} (hs : s.Nonempty) (f : ι → αᵒᵈ) :
    ofDual (s.sup' hs f) = s.inf' hs (ofDual ∘ f) :=
  rfl


@[simp]
theorem ofDual_inf' [SemilatticeSup α] {s : Finset ι} (hs : s.Nonempty) (f : ι → αᵒᵈ) :
    ofDual (s.inf' hs f) = s.sup' hs (ofDual ∘ f) :=
  rfl


theorem sup'_inf_distrib_left (f : ι → α) (a : α) :
    a ⊓ s.sup' hs f = s.sup' hs fun i ↦ a ⊓ f i := by
  induction hs using Finset.Nonempty.cons_induction with
  | singleton => simp
  | cons _ _ _ hs ih => simp_rw [sup'_cons hs, inf_sup_left, ih]


theorem sup'_inf_distrib_right (f : ι → α) (a : α) :
    s.sup' hs f ⊓ a = s.sup' hs fun i => f i ⊓ a := by
  /-
    α : Type u_2
    ι : Type u_5
    inst✝ : DistribLattice α
    s : Finset ι
    hs : s.Nonempty
    f : ι → α
    a : α
    ⊢ Eq (Min.min (s.sup' hs f) a) (s.sup' hs fun i => Min.min (f i) a)
  -/
  rw [inf_comm, sup'_inf_distrib_left]; simp_rw [inf_comm]
                                        /-
                                          🎉 no goals
                                        -/


theorem sup'_inf_sup' (f : ι → α) (g : κ → α) :
    s.sup' hs f ⊓ t.sup' ht g = (s ×ˢ t).sup' (hs.product ht) fun i => f i.1 ⊓ g i.2 := by
  /-
    α : Type u_2
    ι : Type u_5
    κ : Type u_6
    inst✝ : DistribLattice α
    s : Finset ι
    t : Finset κ
    hs : s.Nonempty
    ht : t.Nonempty
    f : ι → α
    g : κ → α
    ⊢ Eq (Min.min (s.sup' hs f) (t.sup' ht g)) ((SProd.sprod s t).sup' ⋯ fun i =>  …
  -/
  simp_rw [Finset.sup'_inf_distrib_right, Finset.sup'_inf_distrib_left, sup'_product_left]
  /-
    🎉 no goals
  -/


theorem inf'_sup_distrib_left (f : ι → α) (a : α) : a ⊔ s.inf' hs f = s.inf' hs fun i => a ⊔ f i :=
  @sup'_inf_distrib_left αᵒᵈ _ _ _ hs _ _


theorem inf'_sup_distrib_right (f : ι → α) (a : α) : s.inf' hs f ⊔ a = s.inf' hs fun i => f i ⊔ a :=
  @sup'_inf_distrib_right αᵒᵈ _ _ _ hs _ _


theorem inf'_sup_inf' (f : ι → α) (g : κ → α) :
    s.inf' hs f ⊔ t.inf' ht g = (s ×ˢ t).inf' (hs.product ht) fun i => f i.1 ⊔ g i.2 :=
  @sup'_inf_sup' αᵒᵈ _ _ _ _ _ hs ht _ _


theorem comp_sup_eq_sup_comp_of_nonempty [OrderBot α] [SemilatticeSup β] [OrderBot β]
    {g : α → β} (mono_g : Monotone g) (H : s.Nonempty) : g (s.sup f) = s.sup (g ∘ f) := by
  /-
    α : Type u_2
    β : Type u_3
    ι : Type u_5
    inst✝³ : LinearOrder α
    s : Finset ι
    f : ι → α
    inst✝² : OrderBot α
    inst✝¹ : SemilatticeSup β
    inst✝ : OrderBot β
    g : α → β
    mono_g : Monotone g
    H : s.Nonempty
    ⊢ Eq (g (s.sup f)) (s.sup (Function.comp g f))
  -/
  rw [← Finset.sup'_eq_sup H, ← Finset.sup'_eq_sup H]
  /-
    α : Type u_2
    β : Type u_3
    ι : Type u_5
    inst✝³ : LinearOrder α
    s : Finset ι
    f : ι → α
    inst✝² : OrderBot α
    inst✝¹ : SemilatticeSup β
    inst✝ : OrderBot β
    g : α → β
    mono_g : Monotone g
    H : s.Nonempty
    ⊢ Eq (g (s.sup' H f)) (s.sup' H (Function.comp g f))
  -/
  exact Finset.comp_sup'_eq_sup'_comp H g (fun x y ↦ Monotone.map_sup mono_g x y)
  /-
    🎉 no goals
  -/


@[simp]
theorem le_sup'_iff : a ≤ s.sup' H f ↔ ∃ b ∈ s, a ≤ f b := by
  /-
    α : Type u_2
    ι : Type u_5
    inst✝ : LinearOrder α
    s : Finset ι
    H : s.Nonempty
    f : ι → α
    a : α
    ⊢ Iff (LE.le a (s.sup' H f)) (Exists fun b => And (Membership.mem s b) (LE.le  …
  -/
  rw [← WithBot.coe_le_coe, coe_sup', Finset.le_sup_iff (WithBot.bot_lt_coe a)]
  /-
    α : Type u_2
    ι : Type u_5
    inst✝ : LinearOrder α
    s : Finset ι
    H : s.Nonempty
    f : ι → α
    a : α
    ⊢ Iff (Exists fun b => And (Membership.mem s b) (LE.le (↑a) (Function.comp Wit …
  -/
  exact exists_congr (fun _ => and_congr_right' WithBot.coe_le_coe)
  /-
    🎉 no goals
  -/


@[simp]
theorem lt_sup'_iff : a < s.sup' H f ↔ ∃ b ∈ s, a < f b := by
  /-
    α : Type u_2
    ι : Type u_5
    inst✝ : LinearOrder α
    s : Finset ι
    H : s.Nonempty
    f : ι → α
    a : α
    ⊢ Iff (LT.lt a (s.sup' H f)) (Exists fun b => And (Membership.mem s b) (LT.lt  …
  -/
  rw [← WithBot.coe_lt_coe, coe_sup', Finset.lt_sup_iff]
  /-
    α : Type u_2
    ι : Type u_5
    inst✝ : LinearOrder α
    s : Finset ι
    H : s.Nonempty
    f : ι → α
    a : α
    ⊢ Iff (Exists fun b => And (Membership.mem s b) (LT.lt (↑a) (Function.comp Wit …
  -/
  exact exists_congr (fun _ => and_congr_right' WithBot.coe_lt_coe)
  /-
    🎉 no goals
  -/


@[simp]
theorem sup'_lt_iff : s.sup' H f < a ↔ ∀ i ∈ s, f i < a := by
  /-
    α : Type u_2
    ι : Type u_5
    inst✝ : LinearOrder α
    s : Finset ι
    H : s.Nonempty
    f : ι → α
    a : α
    ⊢ Iff (LT.lt (s.sup' H f) a) (∀ (i : ι), Membership.mem s i → LT.lt (f i) a)
  -/
  rw [← WithBot.coe_lt_coe, coe_sup', Finset.sup_lt_iff (WithBot.bot_lt_coe a)]
  /-
    α : Type u_2
    ι : Type u_5
    inst✝ : LinearOrder α
    s : Finset ι
    H : s.Nonempty
    f : ι → α
    a : α
    ⊢ Iff (∀ (b : ι), Membership.mem s b → LT.lt (Function.comp WithBot.some f b)  …
  -/
  exact forall₂_congr (fun _ _ => WithBot.coe_lt_coe)
  /-
    🎉 no goals
  -/


@[simp]
theorem inf'_le_iff : s.inf' H f ≤ a ↔ ∃ i ∈ s, f i ≤ a :=
  le_sup'_iff (α := αᵒᵈ) H


@[simp]
theorem inf'_lt_iff : s.inf' H f < a ↔ ∃ i ∈ s, f i < a :=
  lt_sup'_iff (α := αᵒᵈ) H


@[simp]
theorem lt_inf'_iff : a < s.inf' H f ↔ ∀ i ∈ s, a < f i :=
  sup'_lt_iff (α := αᵒᵈ) H


theorem exists_mem_eq_sup' (f : ι → α) : ∃ i, i ∈ s ∧ s.sup' H f = f i := by
  induction H using Finset.Nonempty.cons_induction with
  | singleton c =>  exact ⟨c, mem_singleton_self c, rfl⟩
  | cons c s hcs hs ih =>
    rcases ih with ⟨b, hb, h'⟩
    rw [sup'_cons hs, h']
    cases le_total (f b) (f c) with
    | inl h => exact ⟨c, mem_cons.2 (Or.inl rfl), sup_eq_left.2 h⟩
    | inr h => exact ⟨b, mem_cons.2 (Or.inr hb), sup_eq_right.2 h⟩


theorem exists_mem_eq_inf' (f : ι → α) : ∃ i, i ∈ s ∧ s.inf' H f = f i :=
  exists_mem_eq_sup' (α := αᵒᵈ) H f


theorem exists_mem_eq_sup [OrderBot α] (s : Finset ι) (h : s.Nonempty) (f : ι → α) :
    ∃ i, i ∈ s ∧ s.sup f = f i :=
  sup'_eq_sup h f ▸ exists_mem_eq_sup' h f


theorem exists_mem_eq_inf [OrderTop α] (s : Finset ι) (h : s.Nonempty) (f : ι → α) :
    ∃ i, i ∈ s ∧ s.inf f = f i :=
  exists_mem_eq_sup (α := αᵒᵈ) s h f


theorem map_finset_sup [DecidableEq α] [DecidableEq β] (s : Finset γ) (f : γ → Multiset β)
    (g : β → α) (hg : Function.Injective g) : map g (s.sup f) = s.sup (map g ∘ f) :=
  Finset.comp_sup_eq_sup_comp _ (fun _ _ => map_union hg) (map_zero _)


theorem count_finset_sup [DecidableEq β] (s : Finset α) (f : α → Multiset β) (b : β) :
    count b (s.sup f) = s.sup fun a => count b (f a) := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝ : DecidableEq β
    s : Finset α
    f : α → Multiset β
    b : β
    ⊢ Eq (Multiset.count b (s.sup f)) (s.sup fun a => Multiset.count b (f a))
  -/
  letI := Classical.decEq α
  /-
    α : Type u_2
    β : Type u_3
    inst✝ : DecidableEq β
    s : Finset α
    f : α → Multiset β
    b : β
    this : DecidableEq α := Classical.decEq α
    ⊢ Eq (Multiset.count b (s.sup f)) (s.sup fun a => Multiset.count b (f a))
  -/
  refine s.induction ?_ ?_
    /-
      case refine_1
      α : Type u_2
      β : Type u_3
      inst✝ : DecidableEq β
      s : Finset α
      f : α → Multiset β
      b : β
      this : DecidableEq α := Classical.decEq α
      ⊢ Eq (Multiset.count b (EmptyCollection.emptyCollection.sup f)) (EmptyCollecti …
    -/
  · exact count_zero _
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_2
      β : Type u_3
      inst✝ : DecidableEq β
      s : Finset α
      f : α → Multiset β
      b : β
      this : DecidableEq α := Classical.decEq α
      ⊢ ∀ ⦃a : α⦄ {s : Finset α}, Not (Membership.mem s a) → Eq (Multiset.count b (s …
    -/
  · intro i s _ ih
    /-
      case refine_2
      α : Type u_2
      β : Type u_3
      inst✝ : DecidableEq β
      s✝ : Finset α
      f : α → Multiset β
      b : β
      this : DecidableEq α := Classical.decEq α
      i : α
      s : Finset α
      a✝ : Not (Membership.mem s i)
      ih : Eq (Multiset.count b (s.sup f)) (s.sup fun a => Multiset.count b (f a))
      ⊢ Eq (Multiset.count b ((Insert.insert i s).sup f)) ((Insert.insert i s).sup f …
    -/
    rw [Finset.sup_insert, sup_eq_union, count_union, Finset.sup_insert, ih]
    /-
      🎉 no goals
    -/


theorem mem_sup {α β} [DecidableEq β] {s : Finset α} {f : α → Multiset β} {x : β} :
    x ∈ s.sup f ↔ ∃ v ∈ s, x ∈ f v := by
  /-
    α : Type u_7
    β : Type u_8
    inst✝ : DecidableEq β
    s : Finset α
    f : α → Multiset β
    x : β
    ⊢ Iff (Membership.mem (s.sup f) x) (Exists fun v => And (Membership.mem s v) ( …
  -/
                                              /-
                                                🎉 no goals
                                              -/
  induction s using Finset.cons_induction <;> simp [*]
                                              /-
                                                🎉 no goals
                                              -/


set_option linter.docPrime false in
@[simp] lemma mem_sup' (hs) : a ∈ s.sup' hs f ↔ ∃ i ∈ s, a ∈ f i := by
  /-
    α : Type u_2
    ι : Type u_5
    inst✝ : DecidableEq α
    s : Finset ι
    f : ι → Finset α
    a : α
    hs : s.Nonempty
    ⊢ Iff (Membership.mem (s.sup' hs f) a) (Exists fun i => And (Membership.mem s  …
  -/
                                                  /-
                                                    🎉 no goals
                                                  -/
  induction' hs using Nonempty.cons_induction <;> simp [*]
                                                  /-
                                                    🎉 no goals
                                                  -/


set_option linter.docPrime false in
@[simp] lemma mem_inf' (hs) : a ∈ s.inf' hs f ↔ ∀ i ∈ s, a ∈ f i := by
  /-
    α : Type u_2
    ι : Type u_5
    inst✝ : DecidableEq α
    s : Finset ι
    f : ι → Finset α
    a : α
    hs : s.Nonempty
    ⊢ Iff (Membership.mem (s.inf' hs f) a) (∀ (i : ι), Membership.mem s i → Member …
  -/
                                                  /-
                                                    🎉 no goals
                                                  -/
  induction' hs using Nonempty.cons_induction <;> simp [*]
                                                  /-
                                                    🎉 no goals
                                                  -/


@[simp] lemma mem_sup : a ∈ s.sup f ↔ ∃ i ∈ s, a ∈ f i := by
  /-
    α : Type u_2
    ι : Type u_5
    inst✝ : DecidableEq α
    s : Finset ι
    f : ι → Finset α
    a : α
    ⊢ Iff (Membership.mem (s.sup f) a) (Exists fun i => And (Membership.mem s i) ( …
  -/
                                        /-
                                          🎉 no goals
                                        -/
  induction' s using cons_induction <;> simp [*]
                                        /-
                                          🎉 no goals
                                        -/


theorem sup_eq_biUnion {α β} [DecidableEq β] (s : Finset α) (t : α → Finset β) :
    s.sup t = s.biUnion t := by
  /-
    α : Type u_7
    β : Type u_8
    inst✝ : DecidableEq β
    s : Finset α
    t : α → Finset β
    ⊢ Eq (s.sup t) (s.biUnion t)
  -/
  ext
  /-
    case h
    α : Type u_7
    β : Type u_8
    inst✝ : DecidableEq β
    s : Finset α
    t : α → Finset β
    a✝ : β
    ⊢ Iff (Membership.mem (s.sup t) a✝) (Membership.mem (s.biUnion t) a✝)
  -/
  rw [mem_sup, mem_biUnion]
  /-
    🎉 no goals
  -/


@[simp]
theorem sup_singleton'' (s : Finset β) (f : β → α) :
    (s.sup fun b => {f b}) = s.image f := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝ : DecidableEq α
    s : Finset β
    f : β → α
    ⊢ Eq (s.sup fun b => Singleton.singleton (f b)) (Finset.image f s)
  -/
  ext a
  /-
    case h
    α : Type u_2
    β : Type u_3
    inst✝ : DecidableEq α
    s : Finset β
    f : β → α
    a : α
    ⊢ Iff (Membership.mem (s.sup fun b => Singleton.singleton (f b)) a) (Membershi …
  -/
  rw [mem_sup, mem_image]
  /-
    case h
    α : Type u_2
    β : Type u_3
    inst✝ : DecidableEq α
    s : Finset β
    f : β → α
    a : α
    ⊢ Iff (Exists fun i => And (Membership.mem s i) (Membership.mem (Singleton.sin …
  -/
  simp only [mem_singleton, eq_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem sup_singleton' (s : Finset α) : s.sup singleton = s :=
  (s.sup_singleton'' _).trans image_id


