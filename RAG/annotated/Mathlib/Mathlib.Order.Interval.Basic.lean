/-- The nonempty closed intervals in an order.

We define intervals by the pair of endpoints `fst`, `snd`. To convert intervals to the set of
elements between these endpoints, use the coercion `NonemptyInterval α → Set α`. -/
@[ext (flat := false)]
structure NonemptyInterval (α : Type*) [LE α] extends Prod α α where
  /-- The starting point of an interval is smaller than the endpoint. -/
  fst_le_snd : fst ≤ snd


theorem toProd_injective : Injective (toProd : NonemptyInterval α → α × α) :=
                  /-
                    α : Type u_1
                    inst✝ : LE α
                    s t : NonemptyInterval α
                    h : Eq s.toProd t.toProd
                    ⊢ Eq s t
                  -/
  fun s t h => by cases s; cases t; congr
                                    /-
                                      🎉 no goals
                                    -/


/-- The injection that induces the order on intervals. -/
def toDualProd : NonemptyInterval α → αᵒᵈ × α :=
  toProd


@[simp]
theorem toDualProd_apply (s : NonemptyInterval α) : s.toDualProd = (toDual s.fst, s.snd) :=
  rfl


theorem toDualProd_injective : Injective (toDualProd : NonemptyInterval α → αᵒᵈ × α) :=
  toProd_injective


instance [IsEmpty α] : IsEmpty (NonemptyInterval α) :=
  ⟨fun s => isEmptyElim s.fst⟩


instance [Subsingleton α] : Subsingleton (NonemptyInterval α) :=
  toDualProd_injective.subsingleton


instance le : LE (NonemptyInterval α) :=
  ⟨fun s t => t.fst ≤ s.fst ∧ s.snd ≤ t.snd⟩


theorem le_def : s ≤ t ↔ t.fst ≤ s.fst ∧ s.snd ≤ t.snd :=
  Iff.rfl


/-- `toDualProd` as an order embedding. -/
@[simps]
def toDualProdHom : NonemptyInterval α ↪o αᵒᵈ × α where
  toFun := toDualProd
  inj' := toDualProd_injective
  map_rel_iff' := Iff.rfl


/-- Turn an interval into an interval in the dual order. -/
def dual : NonemptyInterval α ≃ NonemptyInterval αᵒᵈ where
  toFun s := ⟨s.toProd.swap, s.fst_le_snd⟩
  invFun s := ⟨s.toProd.swap, s.fst_le_snd⟩
  left_inv _ := rfl
  right_inv _ := rfl


@[simp]
theorem fst_dual (s : NonemptyInterval α) : s.dual.fst = toDual s.snd :=
  rfl


@[simp]
theorem snd_dual (s : NonemptyInterval α) : s.dual.snd = toDual s.fst :=
  rfl


instance : Preorder (NonemptyInterval α) :=
  Preorder.lift toDualProd


instance : Coe (NonemptyInterval α) (Set α) :=
  ⟨fun s => Icc s.fst s.snd⟩


instance (priority := 100) : Membership α (NonemptyInterval α) :=
  ⟨fun s a => a ∈ (s : Set α)⟩


@[simp]
theorem mem_mk {hx : x.1 ≤ x.2} : a ∈ mk x hx ↔ x.1 ≤ a ∧ a ≤ x.2 :=
  Iff.rfl


theorem mem_def : a ∈ s ↔ s.fst ≤ a ∧ a ≤ s.snd :=
  Iff.rfl


theorem coe_nonempty (s : NonemptyInterval α) : (s : Set α).Nonempty :=
  nonempty_Icc.2 s.fst_le_snd


/-- `{a}` as an interval. -/
@[simps]
def pure (a : α) : NonemptyInterval α :=
  ⟨⟨a, a⟩, le_rfl⟩


theorem mem_pure_self (a : α) : a ∈ pure a :=
  ⟨le_rfl, le_rfl⟩


theorem pure_injective : Injective (pure : α → NonemptyInterval α) := fun _ _ =>
  congr_arg <| Prod.fst ∘ toProd


@[simp]
theorem dual_pure (a : α) : dual (pure a) = pure (toDual a) :=
  rfl


instance [Inhabited α] : Inhabited (NonemptyInterval α) :=
  ⟨pure default⟩


instance [Nonempty α] : Nonempty (NonemptyInterval α) :=
                        /-
                          α : Type u_1
                          β : Type u_2
                          γ : Type u_3
                          ι : Sort u_4
                          κ : ι → Sort u_5
                          inst✝³ : Preorder α
                          inst✝² : Preorder β
                          inst✝¹ : Preorder γ
                          s : NonemptyInterval α
                          x : Prod α α
                          a : α
                          inst✝ : Nonempty α
                          ⊢ Nonempty α
                        -/
  Nonempty.map pure (by infer_instance)
                        /-
                          🎉 no goals
                        -/


instance [Nontrivial α] : Nontrivial (NonemptyInterval α) :=
  pure_injective.nontrivial


/-- Pushforward of nonempty intervals. -/
@[simps!]
def map (f : α →o β) (a : NonemptyInterval α) : NonemptyInterval β :=
  ⟨a.toProd.map f f, f.mono a.fst_le_snd⟩


@[simp]
theorem map_pure (f : α →o β) (a : α) : (pure a).map f = pure (f a) :=
  rfl


@[simp]
theorem map_map (g : β →o γ) (f : α →o β) (a : NonemptyInterval α) :
    (a.map f).map g = a.map (g.comp f) :=
  rfl


@[simp]
theorem dual_map (f : α →o β) (a : NonemptyInterval α) :
    dual (a.map f) = a.dual.map f.dual :=
  rfl


/-- Binary pushforward of nonempty intervals. -/
@[simps]
def map₂ (f : α → β → γ) (h₀ : ∀ b, Monotone fun a => f a b) (h₁ : ∀ a, Monotone (f a)) :
    NonemptyInterval α → NonemptyInterval β → NonemptyInterval γ := fun s t =>
  ⟨(f s.fst t.fst, f s.snd t.snd), (h₀ _ s.fst_le_snd).trans <| h₁ _ t.fst_le_snd⟩


@[simp]
theorem map₂_pure (f : α → β → γ) (h₀ h₁) (a : α) (b : β) :
    map₂ f h₀ h₁ (pure a) (pure b) = pure (f a b) :=
  rfl


@[simp]
theorem dual_map₂ (f : α → β → γ) (h₀ h₁ s t) :
    dual (map₂ f h₀ h₁ s t) =
      map₂ (fun a b => toDual <| f (ofDual a) <| ofDual b) (fun _ => (h₀ _).dual)
        (fun _ => (h₁ _).dual) (dual s) (dual t) :=
  rfl


instance : OrderTop (NonemptyInterval α) where
  top := ⟨⟨⊥, ⊤⟩, bot_le⟩
  le_top _ := ⟨bot_le, le_top⟩


@[simp]
theorem dual_top : dual (⊤ : NonemptyInterval α) = ⊤ :=
  rfl


instance : PartialOrder (NonemptyInterval α) :=
  PartialOrder.lift _ toDualProd_injective


/-- Consider a nonempty interval `[a, b]` as the set `[a, b]`. -/
def coeHom : NonemptyInterval α ↪o Set α :=
  OrderEmbedding.ofMapLEIff (fun s => Icc s.fst s.snd) fun s _ => Icc_subset_Icc_iff s.fst_le_snd


instance setLike : SetLike (NonemptyInterval α) α where
  coe s := Icc s.fst s.snd
  coe_injective' := coeHom.injective


@[norm_cast] -- @[simp, norm_cast] -- Porting note: not in simpNF
theorem coe_subset_coe : (s : Set α) ⊆ t ↔ (s : NonemptyInterval α) ≤ t :=
  (@coeHom α _).le_iff_le


@[norm_cast] -- @[simp, norm_cast] -- Porting note: not in simpNF
theorem coe_ssubset_coe : (s : Set α) ⊂ t ↔ s < t :=
  (@coeHom α _).lt_iff_lt


@[simp]
theorem coe_coeHom : (coeHom : NonemptyInterval α → Set α) = ((↑) : NonemptyInterval α → Set α) :=
  rfl


theorem coe_def (s : NonemptyInterval α) : (s : Set α) = Set.Icc s.toProd.1 s.toProd.2 := rfl


@[simp, norm_cast]
theorem coe_pure (a : α) : (pure a : Set α) = {a} :=
  Icc_self _


@[simp]
theorem mem_pure : b ∈ pure a ↔ b = a := by
  /-
    α : Type u_1
    inst✝ : PartialOrder α
    a b : α
    ⊢ Iff (Membership.mem (NonemptyInterval.pure a) b) (Eq b a)
  -/
  rw [← SetLike.mem_coe, coe_pure, mem_singleton_iff]
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem coe_top [BoundedOrder α] : ((⊤ : NonemptyInterval α) : Set α) = univ :=
  Icc_bot_top


@[simp, norm_cast]
theorem coe_dual (s : NonemptyInterval α) : (dual s : Set αᵒᵈ) = ofDual ⁻¹' s :=
  dual_Icc


theorem subset_coe_map (f : α →o β) (s : NonemptyInterval α) : f '' s ⊆ s.map f :=
  image_subset_iff.2 fun _ ha => ⟨f.mono ha.1, f.mono ha.2⟩


instance : Max (NonemptyInterval α) :=
  ⟨fun s t => ⟨⟨s.fst ⊓ t.fst, s.snd ⊔ t.snd⟩, inf_le_left.trans <| s.fst_le_snd.trans le_sup_left⟩⟩


instance : SemilatticeSup (NonemptyInterval α) :=
  toDualProd_injective.semilatticeSup _ fun _ _ => rfl


@[simp]
theorem fst_sup (s t : NonemptyInterval α) : (s ⊔ t).fst = s.fst ⊓ t.fst :=
  rfl


@[simp]
theorem snd_sup (s t : NonemptyInterval α) : (s ⊔ t).snd = s.snd ⊔ t.snd :=
  rfl


/-- The closed intervals in an order.

We represent intervals either as `⊥` or a nonempty interval given by its endpoints `fst`, `snd`.
To convert intervals to the set of elements between these endpoints, use the coercion
`Interval α → Set α`. -/
-- Porting note: added reducible, it seems to help with coercions
abbrev Interval (α : Type*) [LE α] :=
  WithBot (NonemptyInterval α) -- deriving Inhabited, LE, OrderBot


instance : Inhabited (Interval α) := WithBot.inhabited

instance : LE (Interval α) := WithBot.le

instance : OrderBot (Interval α) := WithBot.orderBot


instance : Coe (NonemptyInterval α) (Interval α) :=
  WithBot.coe


instance canLift : CanLift (Interval α) (NonemptyInterval α) (↑) fun r => r ≠ ⊥ :=
  WithBot.canLift


/-- Recursor for `Interval` using the preferred forms `⊥` and `↑a`. -/
@[elab_as_elim, induction_eliminator, cases_eliminator]
def recBotCoe {C : Interval α → Sort*} (bot : C ⊥) (coe : ∀ a : NonemptyInterval α, C a) :
    ∀ n : Interval α, C n :=
  WithBot.recBotCoe bot coe


theorem coe_injective : Injective ((↑) : NonemptyInterval α → Interval α) :=
  WithBot.coe_injective


@[norm_cast] -- @[simp, norm_cast] -- Porting note: not in simpNF
theorem coe_inj {s t : NonemptyInterval α} : (s : Interval α) = t ↔ s = t :=
  WithBot.coe_inj


protected
theorem «forall» {p : Interval α → Prop} : (∀ s, p s) ↔ p ⊥ ∧ ∀ s : NonemptyInterval α, p s :=
  Option.forall


protected
theorem «exists» {p : Interval α → Prop} : (∃ s, p s) ↔ p ⊥ ∨ ∃ s : NonemptyInterval α, p s :=
  Option.exists


instance [IsEmpty α] : Unique (Interval α) :=
  inferInstanceAs <| Unique (Option _)


/-- Turn an interval into an interval in the dual order. -/
def dual : Interval α ≃ Interval αᵒᵈ :=
  NonemptyInterval.dual.optionCongr


instance : Preorder (Interval α) :=
  WithBot.preorder


/-- `{a}` as an interval. -/
def pure (a : α) : Interval α :=
  NonemptyInterval.pure a


theorem pure_injective : Injective (pure : α → Interval α) :=
  coe_injective.comp NonemptyInterval.pure_injective


@[simp]
theorem dual_bot : dual (⊥ : Interval α) = ⊥ :=
  rfl


@[simp]
theorem pure_ne_bot {a : α} : pure a ≠ ⊥ :=
  WithBot.coe_ne_bot


@[simp]
theorem bot_ne_pure {a : α} : ⊥ ≠ pure a :=
  WithBot.bot_ne_coe


instance [Nonempty α] : Nontrivial (Interval α) :=
  Option.nontrivial


/-- Pushforward of intervals. -/
def map (f : α →o β) : Interval α → Interval β :=
  WithBot.map (NonemptyInterval.map f)


@[simp]
theorem map_map (g : β →o γ) (f : α →o β) (s : Interval α) : (s.map f).map g = s.map (g.comp f) :=
  Option.map_map _ _ _


@[simp]
theorem dual_map (f : α →o β) (s : Interval α) : dual (s.map f) = s.dual.map f.dual := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : OrderHom α β
    s : Interval α
    ⊢ Eq (Interval.dual (Interval.map f s)) (Interval.map (OrderHom.dual f) (Inter …
  -/
  cases s
    /-
      case bot
      α : Type u_1
      β : Type u_2
      inst✝¹ : Preorder α
      inst✝ : Preorder β
      f : OrderHom α β
      ⊢ Eq (Interval.dual (Interval.map f Bot.bot)) (Interval.map (OrderHom.dual f)  …
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case coe
      α : Type u_1
      β : Type u_2
      inst✝¹ : Preorder α
      inst✝ : Preorder β
      f : OrderHom α β
      a✝ : NonemptyInterval α
      ⊢ Eq (Interval.dual (Interval.map f ↑a✝)) (Interval.map (OrderHom.dual f) (Int …
    -/
  · exact WithBot.map_comm rfl _
    /-
      🎉 no goals
    -/


instance boundedOrder : BoundedOrder (Interval α) :=
  WithBot.instBoundedOrder


@[simp]
theorem dual_top : dual (⊤ : Interval α) = ⊤ :=
  rfl


instance partialOrder : PartialOrder (Interval α) :=
  WithBot.partialOrder


/-- Consider an interval `[a, b]` as the set `[a, b]`. -/
def coeHom : Interval α ↪o Set α :=
  OrderEmbedding.ofMapLEIff
    (fun s =>
      match s with
      | ⊥ => ∅
      | some s => s)
    fun s t =>
    match s, t with
    | ⊥, _ => iff_of_true bot_le bot_le
    | some s, ⊥ =>
      iff_of_false (fun h => s.coe_nonempty.ne_empty <| le_bot_iff.1 h) (WithBot.not_coe_le_bot _)
    | some _, some _ => (@NonemptyInterval.coeHom α _).le_iff_le.trans WithBot.coe_le_coe.symm


instance setLike : SetLike (Interval α) α where
  coe := coeHom
  coe_injective' := coeHom.injective


@[norm_cast] -- @[simp, norm_cast] -- Porting note: not in simpNF
theorem coe_subset_coe : (s : Set α) ⊆ t ↔ s ≤ t :=
  (@coeHom α _).le_iff_le


@[norm_cast] -- @[simp, norm_cast] -- Porting note: not in simpNF
theorem coe_sSubset_coe : (s : Set α) ⊂ t ↔ s < t :=
  (@coeHom α _).lt_iff_lt


@[simp, norm_cast]
theorem coe_coe (s : NonemptyInterval α) : ((s : Interval α) : Set α) = s :=
  rfl


@[simp, norm_cast]
theorem coe_bot : ((⊥ : Interval α) : Set α) = ∅ :=
  rfl


@[simp, norm_cast]
theorem coe_top [BoundedOrder α] : ((⊤ : Interval α) : Set α) = univ :=
  Icc_bot_top


@[simp, norm_cast]
theorem coe_dual (s : Interval α) : (dual s : Set αᵒᵈ) = ofDual ⁻¹' s := by
  cases s with
  | bot => rfl
  | coe s₀ => exact NonemptyInterval.coe_dual s₀


theorem subset_coe_map (f : α →o β) : ∀ s : Interval α, f '' s ⊆ s.map f
            /-
              α : Type u_1
              β : Type u_2
              inst✝¹ : PartialOrder α
              inst✝ : PartialOrder β
              f : OrderHom α β
              ⊢ HasSubset.Subset (Set.image ⇑f ↑Bot.bot) ↑(Interval.map f Bot.bot)
            -/
  | ⊥ => by simp
            /-
              🎉 no goals
            -/
  | (s : NonemptyInterval α) => s.subset_coe_map _


@[simp]
                                            /-
                                              α : Type u_1
                                              inst✝ : PartialOrder α
                                              a b : α
                                              ⊢ Iff (Membership.mem (Interval.pure a) b) (Eq b a)
                                            -/
theorem mem_pure : b ∈ pure a ↔ b = a := by rw [← SetLike.mem_coe, coe_pure, mem_singleton_iff]
                                            /-
                                              🎉 no goals
                                            -/


theorem mem_pure_self (a : α) : a ∈ pure a :=
  mem_pure.2 rfl


instance semilatticeSup : SemilatticeSup (Interval α) :=
  WithBot.semilatticeSup


instance lattice : Lattice (Interval α) :=
  { Interval.semilatticeSup with
    inf := fun s t =>
      match s, t with
      | ⊥, _ => ⊥
      | _, ⊥ => ⊥
      | some s, some t =>
        if h : s.fst ≤ t.snd ∧ t.fst ≤ s.snd then
          WithBot.some
            ⟨⟨s.fst ⊔ t.fst, s.snd ⊓ t.snd⟩,
              sup_le (le_inf s.fst_le_snd h.1) <| le_inf h.2 t.fst_le_snd⟩
        else ⊥
    inf_le_left := fun s t =>
      match s, t with
      | ⊥, ⊥ => bot_le
      | ⊥, some _ => bot_le
      | some _, ⊥ => bot_le
      | some s, some t => by
        /-
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          ι : Sort u_4
          κ : ι → Sort u_5
          inst✝¹ : Lattice α
          inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
          s✝ t✝ : Interval α
          s t : NonemptyInterval α
          ⊢ LE.le ((fun s t => Interval.lattice.match_1 (fun s t => Interval α) s t (fun …
        -/
        change dite _ _ _ ≤ _
        /-
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          ι : Sort u_4
          κ : ι → Sort u_5
          inst✝¹ : Lattice α
          inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
          s✝ t✝ : Interval α
          s t : NonemptyInterval α
          ⊢ LE.le (dite (And (LE.le s.toProd.1 t.toProd.2) (LE.le t.toProd.1 s.toProd.2) …
        -/
        split_ifs
          /-
            case pos
            α : Type u_1
            β : Type u_2
            γ : Type u_3
            ι : Sort u_4
            κ : ι → Sort u_5
            inst✝¹ : Lattice α
            inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
            s✝ t✝ : Interval α
            s t : NonemptyInterval α
            h✝ : And (LE.le s.toProd.1 t.toProd.2) (LE.le t.toProd.1 s.toProd.2)
            ⊢ LE.le (↑{ fst := Max.max s.toProd.1 t.toProd.1, snd := Min.min s.toProd.2 t. …
          -/
        · exact WithBot.coe_le_coe.2 ⟨le_sup_left, inf_le_left⟩
          /-
            🎉 no goals
          -/
          /-
            case neg
            α : Type u_1
            β : Type u_2
            γ : Type u_3
            ι : Sort u_4
            κ : ι → Sort u_5
            inst✝¹ : Lattice α
            inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
            s✝ t✝ : Interval α
            s t : NonemptyInterval α
            h✝ : Not (And (LE.le s.toProd.1 t.toProd.2) (LE.le t.toProd.1 s.toProd.2))
            ⊢ LE.le Bot.bot (Option.some s)
          -/
        · exact bot_le
          /-
            🎉 no goals
          -/
    inf_le_right := fun s t =>
      match s, t with
      | ⊥, ⊥ => bot_le
      | ⊥, some _ => bot_le
      | some _, ⊥ => bot_le
      | some s, some t => by
        /-
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          ι : Sort u_4
          κ : ι → Sort u_5
          inst✝¹ : Lattice α
          inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
          s✝ t✝ : Interval α
          s t : NonemptyInterval α
          ⊢ LE.le ((fun s t => Interval.lattice.match_1 (fun s t => Interval α) s t (fun …
        -/
        change dite _ _ _ ≤ _
        /-
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          ι : Sort u_4
          κ : ι → Sort u_5
          inst✝¹ : Lattice α
          inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
          s✝ t✝ : Interval α
          s t : NonemptyInterval α
          ⊢ LE.le (dite (And (LE.le s.toProd.1 t.toProd.2) (LE.le t.toProd.1 s.toProd.2) …
        -/
        split_ifs
          /-
            case pos
            α : Type u_1
            β : Type u_2
            γ : Type u_3
            ι : Sort u_4
            κ : ι → Sort u_5
            inst✝¹ : Lattice α
            inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
            s✝ t✝ : Interval α
            s t : NonemptyInterval α
            h✝ : And (LE.le s.toProd.1 t.toProd.2) (LE.le t.toProd.1 s.toProd.2)
            ⊢ LE.le (↑{ fst := Max.max s.toProd.1 t.toProd.1, snd := Min.min s.toProd.2 t. …
          -/
        · exact WithBot.coe_le_coe.2 ⟨le_sup_right, inf_le_right⟩
          /-
            🎉 no goals
          -/
          /-
            case neg
            α : Type u_1
            β : Type u_2
            γ : Type u_3
            ι : Sort u_4
            κ : ι → Sort u_5
            inst✝¹ : Lattice α
            inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
            s✝ t✝ : Interval α
            s t : NonemptyInterval α
            h✝ : Not (And (LE.le s.toProd.1 t.toProd.2) (LE.le t.toProd.1 s.toProd.2))
            ⊢ LE.le Bot.bot (Option.some t)
          -/
        · exact bot_le
          /-
            🎉 no goals
          -/
    le_inf := fun s t c =>
      match s, t, c with
      | ⊥, _, _ => fun _ _ => bot_le
      | (s : NonemptyInterval α), t, c => fun hb hc => by
        /-
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          ι : Sort u_4
          κ : ι → Sort u_5
          inst✝¹ : Lattice α
          inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
          s✝ t✝ c✝ : Interval α
          s : NonemptyInterval α
          t c : Interval α
          hb : LE.le (↑s) t
          hc : LE.le (↑s) c
          ⊢ LE.le (↑s) ((fun s t => Interval.lattice.match_1 (fun s t => Interval α) s t …
        -/
        lift t to NonemptyInterval α using ne_bot_of_le_ne_bot WithBot.coe_ne_bot hb
        /-
          case intro
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          ι : Sort u_4
          κ : ι → Sort u_5
          inst✝¹ : Lattice α
          inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
          s✝ t✝ c✝ : Interval α
          s : NonemptyInterval α
          c : Interval α
          hc : LE.le (↑s) c
          t : NonemptyInterval α
          hb : LE.le ↑s ↑t
          ⊢ LE.le (↑s) ((fun s t => Interval.lattice.match_1 (fun s t => Interval α) s t …
        -/
        lift c to NonemptyInterval α using ne_bot_of_le_ne_bot WithBot.coe_ne_bot hc
        /-
          case intro.intro
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          ι : Sort u_4
          κ : ι → Sort u_5
          inst✝¹ : Lattice α
          inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
          s✝ t✝ c✝ : Interval α
          s t : NonemptyInterval α
          hb : LE.le ↑s ↑t
          c : NonemptyInterval α
          hc : LE.le ↑s ↑c
          ⊢ LE.le (↑s) ((fun s t => Interval.lattice.match_1 (fun s t => Interval α) s t …
        -/
        change _ ≤ dite _ _ _
        /-
          case intro.intro
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          ι : Sort u_4
          κ : ι → Sort u_5
          inst✝¹ : Lattice α
          inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
          s✝ t✝ c✝ : Interval α
          s t : NonemptyInterval α
          hb : LE.le ↑s ↑t
          c : NonemptyInterval α
          hc : LE.le ↑s ↑c
          ⊢ LE.le (↑s) (dite (And (LE.le t.toProd.1 c.toProd.2) (LE.le c.toProd.1 t.toPr …
        -/
        simp only [WithBot.coe_le_coe] at hb hc ⊢
        /-
          case intro.intro
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          ι : Sort u_4
          κ : ι → Sort u_5
          inst✝¹ : Lattice α
          inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
          s✝ t✝ c✝ : Interval α
          s t c : NonemptyInterval α
          hb : LE.le s t
          hc : LE.le s c
          ⊢ LE.le (↑s) (dite (And (LE.le t.toProd.1 c.toProd.2) (LE.le c.toProd.1 t.toPr …
        -/
        rw [dif_pos, WithBot.coe_le_coe]
          /-
            case intro.intro
            α : Type u_1
            β : Type u_2
            γ : Type u_3
            ι : Sort u_4
            κ : ι → Sort u_5
            inst✝¹ : Lattice α
            inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
            s✝ t✝ c✝ : Interval α
            s t c : NonemptyInterval α
            hb : LE.le s t
            hc : LE.le s c
            ⊢ LE.le s { fst := Max.max t.toProd.1 c.toProd.1, snd := Min.min t.toProd.2 c. …
          -/
        · exact ⟨sup_le hb.1 hc.1, le_inf hb.2 hc.2⟩
          /-
            🎉 no goals
          -/
        -- Porting note: had to add the next 6 lines including the changes because
        -- it seems that lean cannot automatically turn `NonemptyInterval.toDualProd s`
        -- into `s.toProd` anymore.
        /-
          case intro.intro.hc
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          ι : Sort u_4
          κ : ι → Sort u_5
          inst✝¹ : Lattice α
          inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
          s✝ t✝ c✝ : Interval α
          s t c : NonemptyInterval α
          hb : LE.le s t
          hc : LE.le s c
          ⊢ And (LE.le t.toProd.1 c.toProd.2) (LE.le c.toProd.1 t.toProd.2)
        -/
        rcases hb with ⟨hb₁, hb₂⟩
        /-
          case intro.intro.hc.intro
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          ι : Sort u_4
          κ : ι → Sort u_5
          inst✝¹ : Lattice α
          inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
          s✝ t✝ c✝ : Interval α
          s t c : NonemptyInterval α
          hc : LE.le s c
          hb₁ : LE.le s.toDualProd.1 t.toDualProd.1
          hb₂ : LE.le s.toDualProd.2 t.toDualProd.2
          ⊢ And (LE.le t.toProd.1 c.toProd.2) (LE.le c.toProd.1 t.toProd.2)
        -/
        rcases hc with ⟨hc₁, hc₂⟩
        /-
          case intro.intro.hc.intro.intro
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          ι : Sort u_4
          κ : ι → Sort u_5
          inst✝¹ : Lattice α
          inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
          s✝ t✝ c✝ : Interval α
          s t c : NonemptyInterval α
          hb₁ : LE.le s.toDualProd.1 t.toDualProd.1
          hb₂ : LE.le s.toDualProd.2 t.toDualProd.2
          hc₁ : LE.le s.toDualProd.1 c.toDualProd.1
          hc₂ : LE.le s.toDualProd.2 c.toDualProd.2
          ⊢ And (LE.le t.toProd.1 c.toProd.2) (LE.le c.toProd.1 t.toProd.2)
        -/
        change t.toProd.fst ≤ s.toProd.fst at hb₁
        /-
          case intro.intro.hc.intro.intro
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          ι : Sort u_4
          κ : ι → Sort u_5
          inst✝¹ : Lattice α
          inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
          s✝ t✝ c✝ : Interval α
          s t c : NonemptyInterval α
          hb₂ : LE.le s.toDualProd.2 t.toDualProd.2
          hc₁ : LE.le s.toDualProd.1 c.toDualProd.1
          hc₂ : LE.le s.toDualProd.2 c.toDualProd.2
          hb₁ : LE.le t.toProd.1 s.toProd.1
          ⊢ And (LE.le t.toProd.1 c.toProd.2) (LE.le c.toProd.1 t.toProd.2)
        -/
        change s.toProd.snd ≤ t.toProd.snd at hb₂
        /-
          case intro.intro.hc.intro.intro
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          ι : Sort u_4
          κ : ι → Sort u_5
          inst✝¹ : Lattice α
          inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
          s✝ t✝ c✝ : Interval α
          s t c : NonemptyInterval α
          hc₁ : LE.le s.toDualProd.1 c.toDualProd.1
          hc₂ : LE.le s.toDualProd.2 c.toDualProd.2
          hb₁ : LE.le t.toProd.1 s.toProd.1
          hb₂ : LE.le s.toProd.2 t.toProd.2
          ⊢ And (LE.le t.toProd.1 c.toProd.2) (LE.le c.toProd.1 t.toProd.2)
        -/
        change c.toProd.fst ≤ s.toProd.fst at hc₁
        /-
          case intro.intro.hc.intro.intro
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          ι : Sort u_4
          κ : ι → Sort u_5
          inst✝¹ : Lattice α
          inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
          s✝ t✝ c✝ : Interval α
          s t c : NonemptyInterval α
          hc₂ : LE.le s.toDualProd.2 c.toDualProd.2
          hb₁ : LE.le t.toProd.1 s.toProd.1
          hb₂ : LE.le s.toProd.2 t.toProd.2
          hc₁ : LE.le c.toProd.1 s.toProd.1
          ⊢ And (LE.le t.toProd.1 c.toProd.2) (LE.le c.toProd.1 t.toProd.2)
        -/
        change s.toProd.snd ≤ c.toProd.snd at hc₂
        -- Porting note: originally it just had `hb.1` etc. in this next line
        /-
          case intro.intro.hc.intro.intro
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          ι : Sort u_4
          κ : ι → Sort u_5
          inst✝¹ : Lattice α
          inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
          s✝ t✝ c✝ : Interval α
          s t c : NonemptyInterval α
          hb₁ : LE.le t.toProd.1 s.toProd.1
          hb₂ : LE.le s.toProd.2 t.toProd.2
          hc₁ : LE.le c.toProd.1 s.toProd.1
          hc₂ : LE.le s.toProd.2 c.toProd.2
          ⊢ And (LE.le t.toProd.1 c.toProd.2) (LE.le c.toProd.1 t.toProd.2)
        -/
        exact ⟨hb₁.trans <| s.fst_le_snd.trans hc₂, hc₁.trans <| s.fst_le_snd.trans hb₂⟩ }
        /-
          🎉 no goals
        -/


@[simp, norm_cast]
theorem coe_inf : ∀ s t : Interval α, (↑(s ⊓ t) : Set α) = ↑s ∩ ↑t
  | ⊥, _ => by
    /-
      α : Type u_1
      inst✝¹ : Lattice α
      inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
      x✝ : Interval α
      ⊢ Eq (↑(Min.min Bot.bot x✝)) (Inter.inter ↑Bot.bot ↑x✝)
    -/
    rw [bot_inf_eq]
    /-
      α : Type u_1
      inst✝¹ : Lattice α
      inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
      x✝ : Interval α
      ⊢ Eq (↑Bot.bot) (Inter.inter ↑Bot.bot ↑x✝)
    -/
    exact (empty_inter _).symm
    /-
      🎉 no goals
    -/
  | (s : NonemptyInterval α), ⊥ => by
    /-
      α : Type u_1
      inst✝¹ : Lattice α
      inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
      s : NonemptyInterval α
      ⊢ Eq (↑(Min.min (↑s) Bot.bot)) (Inter.inter ↑↑s ↑Bot.bot)
    -/
    rw [inf_bot_eq]
    /-
      α : Type u_1
      inst✝¹ : Lattice α
      inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
      s : NonemptyInterval α
      ⊢ Eq (↑Bot.bot) (Inter.inter ↑↑s ↑Bot.bot)
    -/
    exact (inter_empty _).symm
    /-
      🎉 no goals
    -/
  | (s : NonemptyInterval α), (t : NonemptyInterval α) => by
    simp only [Min.min, coe_coe, NonemptyInterval.coe_def, Icc_inter_Icc,
      SemilatticeInf.inf, Lattice.inf]
    /-
      α : Type u_1
      inst✝¹ : Lattice α
      inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
      s t : NonemptyInterval α
      ⊢ Eq (↑(dite (And (LE.le s.toProd.1 t.toProd.2) (LE.le t.toProd.1 s.toProd.2)) …
    -/
    split_ifs with h
      /-
        case pos
        α : Type u_1
        inst✝¹ : Lattice α
        inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
        s t : NonemptyInterval α
        h : And (LE.le s.toProd.1 t.toProd.2) (LE.le t.toProd.1 s.toProd.2)
        ⊢ Eq (↑↑{ fst := Max.max s.toProd.1 t.toProd.1, snd := Lattice.inf s.toProd.2  …
      -/
    · simp only [coe_coe, NonemptyInterval.coe_def]
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        inst✝¹ : Lattice α
        inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
        s t : NonemptyInterval α
        h : Not (And (LE.le s.toProd.1 t.toProd.2) (LE.le t.toProd.1 s.toProd.2))
        ⊢ Eq (↑Bot.bot) (Set.Icc (Max.max s.toProd.1 t.toProd.1) (Lattice.inf s.toProd …
      -/
    · refine (Icc_eq_empty <| mt ?_ h).symm
      exact fun h ↦ ⟨le_sup_left.trans <| h.trans inf_le_right,
        le_sup_right.trans <| h.trans inf_le_left⟩


@[simp, norm_cast]
theorem disjoint_coe (s t : Interval α) : Disjoint (s : Set α) t ↔ Disjoint s t := by
  classical
    rw [disjoint_iff_inf_le, disjoint_iff_inf_le, ← coe_subset_coe, coe_inf]
    rfl


@[simp, norm_cast]
theorem coe_pure_interval (a : α) : (pure a : Interval α) = Interval.pure a :=
  rfl


@[simp, norm_cast]
theorem coe_eq_pure : (s : Interval α) = Interval.pure a ↔ s = pure a := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    s : NonemptyInterval α
    a : α
    ⊢ Iff (Eq (↑s) (Interval.pure a)) (Eq s (NonemptyInterval.pure a))
  -/
  rw [← Interval.coe_inj, coe_pure_interval]
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem coe_top_interval [BoundedOrder α] : ((⊤ : NonemptyInterval α) : Interval α) = ⊤ :=
  rfl


@[simp, norm_cast]
theorem mem_coe_interval [PartialOrder α] {s : NonemptyInterval α} {x : α} :
    x ∈ (s : Interval α) ↔ x ∈ s :=
  Iff.rfl


@[simp, norm_cast]
theorem coe_sup_interval [Lattice α] (s t : NonemptyInterval α) :
    (↑(s ⊔ t) : Interval α) = ↑s ⊔ ↑t :=
  rfl


noncomputable instance completeLattice [DecidableRel (α := α) (· ≤ ·)] :
    CompleteLattice (Interval α) := by
  classical
  exact
      { Interval.lattice, Interval.boundedOrder with
        sSup := fun S =>
          if h : S ⊆ {⊥} then ⊥
          else
            WithBot.some
              ⟨⟨⨅ (s : NonemptyInterval α) (_ : ↑s ∈ S), s.fst,
                  ⨆ (s : NonemptyInterval α) (_ : ↑s ∈ S), s.snd⟩, by
                obtain ⟨s, hs, ha⟩ := not_subset.1 h
                lift s to NonemptyInterval α using ha
                exact iInf₂_le_of_le s hs (le_iSup₂_of_le s hs s.fst_le_snd)⟩
        le_sSup := fun s s ha => by
          dsimp only -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10752): added `dsimp only`
          split_ifs with h
          · exact (h ha).le
          cases s
          · exact bot_le
          · -- Porting note: This case was
            -- `exact WithBot.some_le_some.2 ⟨iInf₂_le _ ha, le_iSup₂_of_le _ ha le_rfl⟩`
            -- but there seems to be a defEq-problem at `iInf₂_le` that lean cannot resolve yet.
            apply WithBot.coe_le_coe.2
            constructor
            · apply iInf₂_le
              exact ha
            · exact le_iSup₂_of_le _ ha le_rfl
        sSup_le := fun s s ha => by
          dsimp only -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10752): added `dsimp only`
          split_ifs with h
          · exact bot_le
          obtain ⟨b, hs, hb⟩ := not_subset.1 h
          lift s to NonemptyInterval α using ne_bot_of_le_ne_bot hb (ha _ hs)
          exact
            WithBot.coe_le_coe.2
              ⟨le_iInf₂ fun c hc => (WithBot.coe_le_coe.1 <| ha _ hc).1,
                iSup₂_le fun c hc => (WithBot.coe_le_coe.1 <| ha _ hc).2⟩
        sInf := fun S =>
          if h :
              ⊥ ∉ S ∧
                ∀ ⦃s : NonemptyInterval α⦄,
                  ↑s ∈ S → ∀ ⦃t : NonemptyInterval α⦄, ↑t ∈ S → s.fst ≤ t.snd then
            WithBot.some
              ⟨⟨⨆ (s : NonemptyInterval α) (_ : ↑s ∈ S), s.fst,
                  ⨅ (s : NonemptyInterval α) (_ : ↑s ∈ S), s.snd⟩,
                iSup₂_le fun s hs => le_iInf₂ <| h.2 hs⟩
          else ⊥
        sInf_le := fun s₁ s ha => by
          dsimp only -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10752): added `dsimp only`
          split_ifs with h
          · lift s to NonemptyInterval α using ne_of_mem_of_not_mem ha h.1
            -- Porting note: Lean failed to figure out the function `f` by itself,
            -- so I added it through manually
            let f := fun (s : NonemptyInterval α) (_ : ↑s ∈ s₁) => s.toProd.fst
            exact WithBot.coe_le_coe.2 ⟨le_iSup₂ (f := f) s ha, iInf₂_le s ha⟩
          · exact bot_le
        le_sInf := by
          intro S s ha
          cases s with
          | bot => exact bot_le
          | coe s =>
            dsimp -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11227): added a `dsimp`
            split_ifs with h
            · exact WithBot.coe_le_coe.2
                ⟨iSup₂_le fun t hb => (WithBot.coe_le_coe.1 <| ha _ hb).1,
                  le_iInf₂ fun t hb => (WithBot.coe_le_coe.1 <| ha _ hb).2⟩
            · rw [not_and_or, not_not] at h
              rcases h with h | h
              · exact ha _ h
              · -- Porting note: ungolfed, due to identification problems
                -- between `toProd` and `toDualProd`. Original mathport output:
                -- cases h fun t hb c hc =>
                --   (WithBot.coe_le_coe.1 <| ha _ hb).1.trans <|
                --     s.fst_le_snd.trans (WithBot.coe_le_coe.1 <| ha _ hc).2 }
                exfalso
                apply h
                intro b hb c hc
                have h₁ := (WithBot.coe_le_coe.1 <| ha _ hb).1
                repeat rw [NonemptyInterval.toDualProd_apply] at h₁
                rw [OrderDual.toDual_le_toDual] at h₁
                exact h₁.trans (s.fst_le_snd.trans (WithBot.coe_le_coe.1 <| ha _ hc).2)
  }


@[simp, norm_cast]
theorem coe_sInf [DecidableRel (α := α) (· ≤ ·)] (S : Set (Interval α)) :
    ↑(sInf S) = ⋂ s ∈ S, (s : Set α) := by
  classical -- Porting note: added
  -- Porting note: this `change` was
  -- change ↑ (dite _ _ _) = _
  change ((dite _ _ _ : Interval α) : Set α) = ⋂ (s : Interval α) (_ : s ∈ S), (s : Set α)
  split_ifs with h
  · ext
    simp [Interval.forall, h.1, ← forall_and, ← NonemptyInterval.mem_def]
  simp_rw [not_and_or, Classical.not_not] at h
  rcases h with h | h
  · refine (eq_empty_of_subset_empty ?_).symm
    exact iInter₂_subset_of_subset _ h Subset.rfl
  · refine (not_nonempty_iff_eq_empty.1 ?_).symm
    rintro ⟨x, hx⟩
    rw [mem_iInter₂] at hx
    exact h fun s ha t hb => (hx _ ha).1.trans (hx _ hb).2


@[simp, norm_cast]
theorem coe_iInf [DecidableRel (α := α) (· ≤ ·)] (f : ι → Interval α) :
                                           /-
                                             α : Type u_1
                                             ι : Sort u_4
                                             inst✝¹ : CompleteLattice α
                                             inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
                                             f : ι → Interval α
                                             ⊢ Eq (↑(iInf fun i => f i)) (Set.iInter fun i => ↑(f i))
                                           -/
    ↑(⨅ i, f i) = ⋂ i, (f i : Set α) := by simp [iInf]
                                           /-
                                             🎉 no goals
                                           -/

-- @[simp] -- Porting note: not in simpNF

@[norm_cast]
theorem coe_iInf₂ [DecidableRel (α := α) (· ≤ ·)] (f : ∀ i, κ i → Interval α) :
                                                           /-
                                                             α : Type u_1
                                                             ι : Sort u_4
                                                             κ : ι → Sort u_5
                                                             inst✝¹ : CompleteLattice α
                                                             inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
                                                             f : (i : ι) → κ i → Interval α
                                                             ⊢ Eq (↑(iInf fun i => iInf fun j => f i j)) (Set.iInter fun i => Set.iInter fu …
                                                           -/
    ↑(⨅ (i) (j), f i j) = ⋂ (i) (j), (f i j : Set α) := by simp_rw [coe_iInf]
                                                           /-
                                                             🎉 no goals
                                                           -/


