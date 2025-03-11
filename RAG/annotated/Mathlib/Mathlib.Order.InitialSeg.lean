/-- If `r` is a relation on `α` and `s` in a relation on `β`, then `f : r ≼i s` is an order
embedding whose range is an initial segment. That is, whenever `b < f a` in `β` then `b` is in the
range of `f`. -/
structure InitialSeg {α β : Type*} (r : α → α → Prop) (s : β → β → Prop) extends r ↪r s where
  /-- The order embedding is an initial segment -/
  mem_range_of_rel' : ∀ a b, s b (toRelEmbedding a) → b ∈ Set.range toRelEmbedding

-- Porting note: Deleted `scoped[InitialSeg]`

@[inherit_doc]
infixl:25 " ≼i " => InitialSeg


/-- An `InitialSeg` between the `<` relations of two types. -/
notation:25 α:24 " ≤i " β:25 => @InitialSeg α β (· < ·) (· < ·)


instance : Coe (r ≼i s) (r ↪r s) :=
  ⟨InitialSeg.toRelEmbedding⟩


instance : FunLike (r ≼i s) α β where
  coe f := f.toFun
  coe_injective' := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      r : α → α → Prop
      s : β → β → Prop
      t : γ → γ → Prop
      ⊢ Function.Injective fun f => f.toFun
    -/
    rintro ⟨f, hf⟩ ⟨g, hg⟩ h
    /-
      case mk.mk
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      r : α → α → Prop
      s : β → β → Prop
      t : γ → γ → Prop
      f : RelEmbedding r s
      hf : ∀ (a : α) (b : β), s b (f a) → Membership.mem (Set.range ⇑f) b
      g : RelEmbedding r s
      hg : ∀ (a : α) (b : β), s b (g a) → Membership.mem (Set.range ⇑g) b
      h : Eq ((fun f => f.toFun) { toRelEmbedding := f, mem_range_of_rel' := hf }) ( …
      ⊢ Eq { toRelEmbedding := f, mem_range_of_rel' := hf } { toRelEmbedding := g, m …
    -/
    congr with x
    /-
      case mk.mk.e_toRelEmbedding.h
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      r : α → α → Prop
      s : β → β → Prop
      t : γ → γ → Prop
      f : RelEmbedding r s
      hf : ∀ (a : α) (b : β), s b (f a) → Membership.mem (Set.range ⇑f) b
      g : RelEmbedding r s
      hg : ∀ (a : α) (b : β), s b (g a) → Membership.mem (Set.range ⇑g) b
      h : Eq ((fun f => f.toFun) { toRelEmbedding := f, mem_range_of_rel' := hf }) ( …
      x : α
      ⊢ Eq (f x) (g x)
    -/
    exact congr_fun h x
    /-
      🎉 no goals
    -/


instance : EmbeddingLike (r ≼i s) α β where
  injective' f := f.inj'


instance : RelHomClass (r ≼i s) r s where
  map_rel f := f.map_rel_iff.2


/-- An initial segment embedding between the `<` relations of two partial orders is an order
embedding. -/
def toOrderEmbedding [PartialOrder α] [PartialOrder β] (f : α ≤i β) : α ↪o β :=
  f.orderEmbeddingOfLTEmbedding


@[simp]
theorem toOrderEmbedding_apply [PartialOrder α] [PartialOrder β] (f : α ≤i β) (x : α) :
    f.toOrderEmbedding x = f x :=
  rfl


@[simp]
theorem coe_toOrderEmbedding [PartialOrder α] [PartialOrder β] (f : α ≤i β) :
    (f.toOrderEmbedding : α → β) = f :=
  rfl


instance [PartialOrder α] [PartialOrder β] : OrderHomClass (α ≤i β) α β where
  map_rel f := f.toOrderEmbedding.map_rel_iff.2


@[ext] lemma ext {f g : r ≼i s} (h : ∀ x, f x = g x) : f = g :=
  DFunLike.ext f g h


@[simp]
theorem coe_coe_fn (f : r ≼i s) : ((f : r ↪r s) : α → β) = f :=
  rfl


theorem mem_range_of_rel (f : r ≼i s) {a : α} {b : β} : s b (f a) → b ∈ Set.range f :=
  f.mem_range_of_rel' _ _


@[deprecated mem_range_of_rel (since := "2024-09-21")]
alias init := mem_range_of_rel


theorem map_rel_iff {a b : α} (f : r ≼i s) : s (f a) (f b) ↔ r a b :=
  f.map_rel_iff'


theorem inj (f : r ≼i s) {a b : α} : f a = f b ↔ a = b :=
  f.toRelEmbedding.inj


theorem exists_eq_iff_rel (f : r ≼i s) {a : α} {b : β} : s b (f a) ↔ ∃ a', f a' = b ∧ r a' a :=
  ⟨fun h => by
    /-
      α : Type u_1
      β : Type u_2
      r : α → α → Prop
      s : β → β → Prop
      f : InitialSeg r s
      a : α
      b : β
      h : s b (f a)
      ⊢ Exists fun a' => And (Eq (f a') b) (r a' a)
    -/
    rcases f.mem_range_of_rel h with ⟨a', rfl⟩
    /-
      case intro
      α : Type u_1
      β : Type u_2
      r : α → α → Prop
      s : β → β → Prop
      f : InitialSeg r s
      a a' : α
      h : s (f a') (f a)
      ⊢ Exists fun a'_1 => And (Eq (f a'_1) (f a')) (r a'_1 a)
    -/
    exact ⟨a', rfl, f.map_rel_iff.1 h⟩,
    /-
      🎉 no goals
    -/
    fun ⟨_, e, h⟩ => e ▸ f.map_rel_iff.2 h⟩


@[deprecated exists_eq_iff_rel (since := "2024-09-21")]
alias init_iff := exists_eq_iff_rel


/-- A relation isomorphism is an initial segment -/
@[simps!]
def _root_.RelIso.toInitialSeg (f : r ≃r s) : r ≼i s :=
         /-
           α : Type u_1
           β : Type u_2
           γ : Type u_3
           r : α → α → Prop
           s : β → β → Prop
           t : γ → γ → Prop
           f : RelIso r s
           ⊢ ∀ (a : α) (b : β), s b (f.toRelEmbedding a) → Membership.mem (Set.range ⇑f.t …
         -/
  ⟨f, by simp⟩
         /-
           🎉 no goals
         -/


@[deprecated (since := "2024-10-22")]
alias ofIso := RelIso.toInitialSeg


/-- The identity function shows that `≼i` is reflexive -/
@[refl]
protected def refl (r : α → α → Prop) : r ≼i r :=
  (RelIso.refl r).toInitialSeg


instance (r : α → α → Prop) : Inhabited (r ≼i r) :=
  ⟨InitialSeg.refl r⟩


/-- Composition of functions shows that `≼i` is transitive -/
@[trans]
protected def trans (f : r ≼i s) (g : s ≼i t) : r ≼i t :=
  ⟨f.1.trans g.1, fun a c h => by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      r : α → α → Prop
      s : β → β → Prop
      t : γ → γ → Prop
      f : InitialSeg r s
      g : InitialSeg s t
      a : α
      c : γ
      h : t c ((f.trans g.toRelEmbedding) a)
      ⊢ Membership.mem (Set.range ⇑(f.trans g.toRelEmbedding)) c
    -/
    simp only [RelEmbedding.coe_trans, coe_coe_fn, comp_apply] at h ⊢
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      r : α → α → Prop
      s : β → β → Prop
      t : γ → γ → Prop
      f : InitialSeg r s
      g : InitialSeg s t
      a : α
      c : γ
      h : t c (g (f a))
      ⊢ Membership.mem (Set.range (Function.comp ⇑g ⇑f)) c
    -/
    rcases g.2 _ _ h with ⟨b, rfl⟩; have h := g.map_rel_iff.1 h
    /-
      case intro
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      r : α → α → Prop
      s : β → β → Prop
      t : γ → γ → Prop
      f : InitialSeg r s
      g : InitialSeg s t
      a : α
      b : β
      h✝ : t (g.toRelEmbedding b) (g (f a))
      h : s b (f a)
      ⊢ Membership.mem (Set.range (Function.comp ⇑g ⇑f)) (g.toRelEmbedding b)
    -/
    rcases f.2 _ _ h with ⟨a', rfl⟩; exact ⟨a', rfl⟩⟩
                                     /-
                                       🎉 no goals
                                     -/


@[simp]
theorem refl_apply (x : α) : InitialSeg.refl r x = x :=
  rfl


@[simp]
theorem trans_apply (f : r ≼i s) (g : s ≼i t) (a : α) : (f.trans g) a = g (f a) :=
  rfl


instance subsingleton_of_trichotomous_of_irrefl [IsTrichotomous β s] [IsIrrefl β s]
    [IsWellFounded α r] : Subsingleton (r ≼i s) where
  allEq f g := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      r : α → α → Prop
      s : β → β → Prop
      t : γ → γ → Prop
      inst✝² : IsTrichotomous β s
      inst✝¹ : IsIrrefl β s
      inst✝ : IsWellFounded α r
      f g : InitialSeg r s
      ⊢ Eq f g
    -/
    ext a
    refine IsWellFounded.induction r a fun b IH =>
      extensional_of_trichotomous_of_irrefl s fun x => ?_
    /-
      case h
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      r : α → α → Prop
      s : β → β → Prop
      t : γ → γ → Prop
      inst✝² : IsTrichotomous β s
      inst✝¹ : IsIrrefl β s
      inst✝ : IsWellFounded α r
      f g : InitialSeg r s
      a b : α
      IH : ∀ (y : α), r y b → Eq (f y) (g y)
      x : β
      ⊢ Iff (s x (f b)) (s x (g b))
    -/
    rw [f.exists_eq_iff_rel, g.exists_eq_iff_rel]
    /-
      case h
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      r : α → α → Prop
      s : β → β → Prop
      t : γ → γ → Prop
      inst✝² : IsTrichotomous β s
      inst✝¹ : IsIrrefl β s
      inst✝ : IsWellFounded α r
      f g : InitialSeg r s
      a b : α
      IH : ∀ (y : α), r y b → Eq (f y) (g y)
      x : β
      ⊢ Iff (Exists fun a' => And (Eq (f a') x) (r a' b)) (Exists fun a' => And (Eq  …
    -/
    exact exists_congr fun x => and_congr_left fun hx => IH _ hx ▸ Iff.rfl
    /-
      🎉 no goals
    -/


instance [IsWellOrder β s] : Subsingleton (r ≼i s) :=
  ⟨fun a => have := a.isWellFounded; Subsingleton.elim a⟩


protected theorem eq [IsWellOrder β s] (f g : r ≼i s) (a) : f a = g a := by
  /-
    α : Type u_1
    β : Type u_2
    r : α → α → Prop
    s : β → β → Prop
    inst✝ : IsWellOrder β s
    f g : InitialSeg r s
    a : α
    ⊢ Eq (f a) (g a)
  -/
  rw [Subsingleton.elim f g]
  /-
    🎉 no goals
  -/


theorem eq_relIso [IsWellOrder β s] (f : r ≼i s) (g : r ≃r s) (a : α) : g a = f a :=
  InitialSeg.eq g.toInitialSeg f a


@[deprecated eq_relIso (since := "2024-10-20")]
alias ltOrEq_apply_right := eq_relIso


private theorem antisymm_aux [IsWellOrder α r] (f : r ≼i s) (g : s ≼i r) : LeftInverse g f :=
  (f.trans g).eq (InitialSeg.refl _)


/-- If we have order embeddings between `α` and `β` whose images are initial segments, and `β`
is a well-order then `α` and `β` are order-isomorphic. -/
def antisymm [IsWellOrder β s] (f : r ≼i s) (g : s ≼i r) : r ≃r s :=
  have := f.toRelEmbedding.isWellOrder
  ⟨⟨f, g, antisymm_aux f g, antisymm_aux g f⟩, f.map_rel_iff'⟩


@[simp]
theorem antisymm_toFun [IsWellOrder β s] (f : r ≼i s) (g : s ≼i r) : (antisymm f g : α → β) = f :=
  rfl


@[simp]
theorem antisymm_symm [IsWellOrder α r] [IsWellOrder β s] (f : r ≼i s) (g : s ≼i r) :
    (antisymm f g).symm = antisymm g f :=
  RelIso.coe_fn_injective rfl


theorem eq_or_principal [IsWellOrder β s] (f : r ≼i s) :
    Surjective f ∨ ∃ b, ∀ x, x ∈ Set.range f ↔ s x b := by
  /-
    α : Type u_1
    β : Type u_2
    r : α → α → Prop
    s : β → β → Prop
    inst✝ : IsWellOrder β s
    f : InitialSeg r s
    ⊢ Or (Function.Surjective ⇑f) (Exists fun b => ∀ (x : β), Iff (Membership.mem  …
  -/
  apply or_iff_not_imp_right.2
  /-
    α : Type u_1
    β : Type u_2
    r : α → α → Prop
    s : β → β → Prop
    inst✝ : IsWellOrder β s
    f : InitialSeg r s
    ⊢ Not (Exists fun b => ∀ (x : β), Iff (Membership.mem (Set.range ⇑f) x) (s x b …
  -/
  intro h b
  /-
    α : Type u_1
    β : Type u_2
    r : α → α → Prop
    s : β → β → Prop
    inst✝ : IsWellOrder β s
    f : InitialSeg r s
    h : Not (Exists fun b => ∀ (x : β), Iff (Membership.mem (Set.range ⇑f) x) (s x …
    b : β
    ⊢ Exists fun a => Eq (f a) b
  -/
  push_neg at h
  /-
    α : Type u_1
    β : Type u_2
    r : α → α → Prop
    s : β → β → Prop
    inst✝ : IsWellOrder β s
    f : InitialSeg r s
    b : β
    h : ∀ (b : β), Exists fun x => Or (And (Membership.mem (Set.range ⇑f) x) (Not  …
    ⊢ Exists fun a => Eq (f a) b
  -/
  apply IsWellFounded.induction s b
  /-
    α : Type u_1
    β : Type u_2
    r : α → α → Prop
    s : β → β → Prop
    inst✝ : IsWellOrder β s
    f : InitialSeg r s
    b : β
    h : ∀ (b : β), Exists fun x => Or (And (Membership.mem (Set.range ⇑f) x) (Not  …
    ⊢ ∀ (x : β), (∀ (y : β), s y x → Exists fun a => Eq (f a) y) → Exists fun a => …
  -/
  intro x IH
  /-
    α : Type u_1
    β : Type u_2
    r : α → α → Prop
    s : β → β → Prop
    inst✝ : IsWellOrder β s
    f : InitialSeg r s
    b : β
    h : ∀ (b : β), Exists fun x => Or (And (Membership.mem (Set.range ⇑f) x) (Not  …
    x : β
    IH : ∀ (y : β), s y x → Exists fun a => Eq (f a) y
    ⊢ Exists fun a => Eq (f a) x
  -/
  obtain ⟨y, ⟨hy, hs⟩ | ⟨hy, hs⟩⟩ := h x
    /-
      case intro.inl.intro
      α : Type u_1
      β : Type u_2
      r : α → α → Prop
      s : β → β → Prop
      inst✝ : IsWellOrder β s
      f : InitialSeg r s
      b : β
      h : ∀ (b : β), Exists fun x => Or (And (Membership.mem (Set.range ⇑f) x) (Not  …
      x : β
      IH : ∀ (y : β), s y x → Exists fun a => Eq (f a) y
      y : β
      hy : Membership.mem (Set.range ⇑f) y
      hs : Not (s y x)
      ⊢ Exists fun a => Eq (f a) x
    -/
  · obtain (rfl | h) := (trichotomous y x).resolve_left hs
      /-
        case intro.inl.intro.inl
        α : Type u_1
        β : Type u_2
        r : α → α → Prop
        s : β → β → Prop
        inst✝ : IsWellOrder β s
        f : InitialSeg r s
        b : β
        h : ∀ (b : β), Exists fun x => Or (And (Membership.mem (Set.range ⇑f) x) (Not  …
        y : β
        hy : Membership.mem (Set.range ⇑f) y
        IH : ∀ (y_1 : β), s y_1 y → Exists fun a => Eq (f a) y_1
        hs : Not (s y y)
        ⊢ Exists fun a => Eq (f a) y
      -/
    · exact hy
      /-
        🎉 no goals
      -/
      /-
        case intro.inl.intro.inr
        α : Type u_1
        β : Type u_2
        r : α → α → Prop
        s : β → β → Prop
        inst✝ : IsWellOrder β s
        f : InitialSeg r s
        b : β
        h✝ : ∀ (b : β), Exists fun x => Or (And (Membership.mem (Set.range ⇑f) x) (Not …
        x : β
        IH : ∀ (y : β), s y x → Exists fun a => Eq (f a) y
        y : β
        hy : Membership.mem (Set.range ⇑f) y
        hs : Not (s y x)
        h : s x y
        ⊢ Exists fun a => Eq (f a) x
      -/
    · obtain ⟨z, rfl⟩ := hy
      /-
        case intro.inl.intro.inr.intro
        α : Type u_1
        β : Type u_2
        r : α → α → Prop
        s : β → β → Prop
        inst✝ : IsWellOrder β s
        f : InitialSeg r s
        b : β
        h✝ : ∀ (b : β), Exists fun x => Or (And (Membership.mem (Set.range ⇑f) x) (Not …
        x : β
        IH : ∀ (y : β), s y x → Exists fun a => Eq (f a) y
        z : α
        hs : Not (s (f z) x)
        h : s x (f z)
        ⊢ Exists fun a => Eq (f a) x
      -/
      exact f.mem_range_of_rel h
      /-
        🎉 no goals
      -/
    /-
      case intro.inr.intro
      α : Type u_1
      β : Type u_2
      r : α → α → Prop
      s : β → β → Prop
      inst✝ : IsWellOrder β s
      f : InitialSeg r s
      b : β
      h : ∀ (b : β), Exists fun x => Or (And (Membership.mem (Set.range ⇑f) x) (Not  …
      x : β
      IH : ∀ (y : β), s y x → Exists fun a => Eq (f a) y
      y : β
      hy : Not (Membership.mem (Set.range ⇑f) y)
      hs : s y x
      ⊢ Exists fun a => Eq (f a) x
    -/
  · obtain ⟨z, rfl⟩ := IH y hs
    /-
      case intro.inr.intro.intro
      α : Type u_1
      β : Type u_2
      r : α → α → Prop
      s : β → β → Prop
      inst✝ : IsWellOrder β s
      f : InitialSeg r s
      b : β
      h : ∀ (b : β), Exists fun x => Or (And (Membership.mem (Set.range ⇑f) x) (Not  …
      x : β
      IH : ∀ (y : β), s y x → Exists fun a => Eq (f a) y
      z : α
      hy : Not (Membership.mem (Set.range ⇑f) (f z))
      hs : s (f z) x
      ⊢ Exists fun a => Eq (f a) x
    -/
    cases hy (Set.mem_range_self z)
    /-
      🎉 no goals
    -/


/-- Restrict the codomain of an initial segment -/
def codRestrict (p : Set β) (f : r ≼i s) (H : ∀ a, f a ∈ p) : r ≼i Subrel s p :=
  ⟨RelEmbedding.codRestrict p f H, fun a ⟨b, m⟩ h =>
    let ⟨a', e⟩ := f.mem_range_of_rel h
            /-
              α : Type u_1
              β : Type u_2
              γ : Type u_3
              r : α → α → Prop
              s : β → β → Prop
              t : γ → γ → Prop
              p : Set β
              f : InitialSeg r s
              H : ∀ (a : α), Membership.mem p (f a)
              a : α
              x✝ : ↑p
              b : β
              m : Membership.mem p b
              h : Subrel s p ⟨b, m⟩ ((RelEmbedding.codRestrict p f.toRelEmbedding H) a)
              a' : α
              e : Eq (f a') ↑⟨b, m⟩
              ⊢ Eq ((RelEmbedding.codRestrict p f.toRelEmbedding H) a') ⟨b, m⟩
            -/
    ⟨a', by subst e; rfl⟩⟩
                     /-
                       🎉 no goals
                     -/


@[simp]
theorem codRestrict_apply (p) (f : r ≼i s) (H a) : codRestrict p f H a = ⟨f a, H a⟩ :=
  rfl


/-- Initial segment from an empty type. -/
def ofIsEmpty (r : α → α → Prop) (s : β → β → Prop) [IsEmpty α] : r ≼i s :=
  ⟨RelEmbedding.ofIsEmpty r s, isEmptyElim⟩


/-- Initial segment embedding of an order `r` into the disjoint union of `r` and `s`. -/
def leAdd (r : α → α → Prop) (s : β → β → Prop) : r ≼i Sum.Lex r s :=
  ⟨⟨⟨Sum.inl, fun _ _ => Sum.inl.inj⟩, Sum.lex_inl_inl⟩, fun a b => by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      r✝ : α → α → Prop
      s✝ : β → β → Prop
      t : γ → γ → Prop
      r : α → α → Prop
      s : β → β → Prop
      a : α
      b : Sum α β
      ⊢ Sum.Lex r s b ({ toFun := Sum.inl, inj' := ⋯, map_rel_iff' := ⋯ } a) → Membe …
    -/
    cases b <;> [exact fun _ => ⟨_, rfl⟩; exact False.elim ∘ Sum.lex_inr_inl]⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem leAdd_apply (r : α → α → Prop) (s : β → β → Prop) (a) : leAdd r s a = Sum.inl a :=
  rfl


protected theorem acc (f : r ≼i s) (a : α) : Acc r a ↔ Acc s (f a) :=
  ⟨by
    /-
      α : Type u_1
      β : Type u_2
      r : α → α → Prop
      s : β → β → Prop
      f : InitialSeg r s
      a : α
      ⊢ Acc r a → Acc s (f a)
    -/
    refine fun h => Acc.recOn h fun a _ ha => Acc.intro _ fun b hb => ?_
    /-
      α : Type u_1
      β : Type u_2
      r : α → α → Prop
      s : β → β → Prop
      f : InitialSeg r s
      a✝ : α
      h : Acc r a✝
      a : α
      x✝ : ∀ (y : α), r y a → Acc r y
      ha : ∀ (y : α), r y a → Acc s (f y)
      b : β
      hb : s b (f a)
      ⊢ Acc s b
    -/
    obtain ⟨a', rfl⟩ := f.mem_range_of_rel hb
    /-
      case intro
      α : Type u_1
      β : Type u_2
      r : α → α → Prop
      s : β → β → Prop
      f : InitialSeg r s
      a✝ : α
      h : Acc r a✝
      a : α
      x✝ : ∀ (y : α), r y a → Acc r y
      ha : ∀ (y : α), r y a → Acc s (f y)
      a' : α
      hb : s (f a') (f a)
      ⊢ Acc s (f a')
    -/
    exact ha _ (f.map_rel_iff.mp hb), f.toRelEmbedding.acc a⟩
    /-
      🎉 no goals
    -/


/-- If `r` is a relation on `α` and `s` in a relation on `β`, then `f : r ≺i s` is an order
embedding whose range is an open interval `(-∞, top)` for some element `top` of `β`. Such order
embeddings are called principal segments -/
structure PrincipalSeg {α β : Type*} (r : α → α → Prop) (s : β → β → Prop) extends r ↪r s where
  /-- The supremum of the principal segment -/
  top : β
  /-- The range of the order embedding is the set of elements `b` such that `s b top` -/
  mem_range_iff_rel' : ∀ b, b ∈ Set.range toRelEmbedding ↔ s b top

-- Porting note: deleted `scoped[InitialSeg]`

@[inherit_doc]
infixl:25 " ≺i " => PrincipalSeg


/-- A `PrincipalSeg` between the `<` relations of two types. -/
notation:25 α:24 " <i " β:25 => @PrincipalSeg α β (· < ·) (· < ·)


instance : CoeOut (r ≺i s) (r ↪r s) :=
  ⟨PrincipalSeg.toRelEmbedding⟩


instance : CoeFun (r ≺i s) fun _ => α → β :=
  ⟨fun f => f⟩


theorem toRelEmbedding_injective [IsIrrefl β s] [IsTrichotomous β s] :
    Function.Injective (@toRelEmbedding α β r s) := by
  /-
    α : Type u_1
    β : Type u_2
    r : α → α → Prop
    s : β → β → Prop
    inst✝¹ : IsIrrefl β s
    inst✝ : IsTrichotomous β s
    ⊢ Function.Injective PrincipalSeg.toRelEmbedding
  -/
  rintro ⟨f, a, hf⟩ ⟨g, b, hg⟩ rfl
  /-
    case mk.mk
    α : Type u_1
    β : Type u_2
    r : α → α → Prop
    s : β → β → Prop
    inst✝¹ : IsIrrefl β s
    inst✝ : IsTrichotomous β s
    f : RelEmbedding r s
    a : β
    hf : ∀ (b : β), Iff (Membership.mem (Set.range ⇑f) b) (s b a)
    b : β
    hg : ∀ (b_1 : β), Iff (Membership.mem (Set.range ⇑{ toRelEmbedding := f, top : …
    ⊢ Eq { toRelEmbedding := f, top := a, mem_range_iff_rel' := hf } { toRelEmbedd …
  -/
  congr
  /-
    case mk.mk.e_top
    α : Type u_1
    β : Type u_2
    r : α → α → Prop
    s : β → β → Prop
    inst✝¹ : IsIrrefl β s
    inst✝ : IsTrichotomous β s
    f : RelEmbedding r s
    a : β
    hf : ∀ (b : β), Iff (Membership.mem (Set.range ⇑f) b) (s b a)
    b : β
    hg : ∀ (b_1 : β), Iff (Membership.mem (Set.range ⇑{ toRelEmbedding := f, top : …
    ⊢ Eq a b
  -/
  refine extensional_of_trichotomous_of_irrefl s fun x ↦ ?_
  /-
    case mk.mk.e_top
    α : Type u_1
    β : Type u_2
    r : α → α → Prop
    s : β → β → Prop
    inst✝¹ : IsIrrefl β s
    inst✝ : IsTrichotomous β s
    f : RelEmbedding r s
    a : β
    hf : ∀ (b : β), Iff (Membership.mem (Set.range ⇑f) b) (s b a)
    b : β
    hg : ∀ (b_1 : β), Iff (Membership.mem (Set.range ⇑{ toRelEmbedding := f, top : …
    x : β
    ⊢ Iff (s x a) (s x b)
  -/
  rw [← hf, hg]
  /-
    🎉 no goals
  -/


@[simp]
theorem toRelEmbedding_inj [IsIrrefl β s] [IsTrichotomous β s] {f g : r ≺i s} :
    f.toRelEmbedding = g.toRelEmbedding ↔ f = g :=
  toRelEmbedding_injective.eq_iff


@[ext]
theorem ext [IsIrrefl β s] [IsTrichotomous β s] {f g : r ≺i s} (h : ∀ x, f x = g x) : f = g := by
  /-
    α : Type u_1
    β : Type u_2
    r : α → α → Prop
    s : β → β → Prop
    inst✝¹ : IsIrrefl β s
    inst✝ : IsTrichotomous β s
    f g : PrincipalSeg r s
    h : ∀ (x : α), Eq (f.toRelEmbedding x) (g.toRelEmbedding x)
    ⊢ Eq f g
  -/
  rw [← toRelEmbedding_inj]
  /-
    α : Type u_1
    β : Type u_2
    r : α → α → Prop
    s : β → β → Prop
    inst✝¹ : IsIrrefl β s
    inst✝ : IsTrichotomous β s
    f g : PrincipalSeg r s
    h : ∀ (x : α), Eq (f.toRelEmbedding x) (g.toRelEmbedding x)
    ⊢ Eq f.toRelEmbedding g.toRelEmbedding
  -/
  ext
  /-
    case h
    α : Type u_1
    β : Type u_2
    r : α → α → Prop
    s : β → β → Prop
    inst✝¹ : IsIrrefl β s
    inst✝ : IsTrichotomous β s
    f g : PrincipalSeg r s
    h : ∀ (x : α), Eq (f.toRelEmbedding x) (g.toRelEmbedding x)
    x✝ : α
    ⊢ Eq (f.toRelEmbedding x✝) (g.toRelEmbedding x✝)
  -/
  exact h _
  /-
    🎉 no goals
  -/


@[simp]
theorem coe_fn_mk (f : r ↪r s) (t o) : (@PrincipalSeg.mk _ _ r s f t o : α → β) = f :=
  rfl


theorem mem_range_iff_rel (f : r ≺i s) : ∀ {b : β}, b ∈ Set.range f ↔ s b f.top :=
  f.mem_range_iff_rel' _


@[deprecated mem_range_iff_rel (since := "2024-10-07")]
theorem down (f : r ≺i s) : ∀ {b : β}, s b f.top ↔ ∃ a, f a = b :=
  f.mem_range_iff_rel.symm


theorem lt_top (f : r ≺i s) (a : α) : s (f a) f.top :=
  f.mem_range_iff_rel.1 ⟨_, rfl⟩


theorem mem_range_of_rel_top (f : r ≺i s) {b : β} (h : s b f.top) : b ∈ Set.range f :=
  f.mem_range_iff_rel.2 h


theorem mem_range_of_rel [IsTrans β s] (f : r ≺i s) {a : α} {b : β} (h : s b (f a)) :
    b ∈ Set.range f :=
  f.mem_range_of_rel_top <| _root_.trans h <| f.lt_top _


theorem surjOn (f : r ≺i s) : Set.SurjOn f Set.univ { b | s b f.top } := by
  /-
    α : Type u_1
    β : Type u_2
    r : α → α → Prop
    s : β → β → Prop
    f : PrincipalSeg r s
    ⊢ Set.SurjOn (⇑f.toRelEmbedding) Set.univ (setOf fun b => s b f.top)
  -/
  intro b h
  /-
    α : Type u_1
    β : Type u_2
    r : α → α → Prop
    s : β → β → Prop
    f : PrincipalSeg r s
    b : β
    h : Membership.mem (setOf fun b => s b f.top) b
    ⊢ Membership.mem (Set.image (⇑f.toRelEmbedding) Set.univ) b
  -/
  simpa using mem_range_of_rel_top _ h
  /-
    🎉 no goals
  -/


/-- A principal segment is in particular an initial segment. -/
instance hasCoeInitialSeg [IsTrans β s] : Coe (r ≺i s) (r ≼i s) :=
  ⟨fun f => ⟨f.toRelEmbedding, fun _ _ => f.mem_range_of_rel⟩⟩


theorem coe_coe_fn' [IsTrans β s] (f : r ≺i s) : ((f : r ≼i s) : α → β) = f :=
  rfl


theorem _root_.InitialSeg.eq_principalSeg [IsWellOrder β s] (f : r ≼i s) (g : r ≺i s) (a : α) :
    g a = f a :=
  InitialSeg.eq g f a


@[deprecated (since := "2024-10-20")]
alias _root_.InitialSeg.ltOrEq_apply_left := InitialSeg.eq_principalSeg


theorem exists_eq_iff_rel [IsTrans β s] (f : r ≺i s) {a : α} {b : β} :
    s b (f a) ↔ ∃ a', f a' = b ∧ r a' a :=
  @InitialSeg.exists_eq_iff_rel α β r s f a b


/-- A principal segment is the same as a non-surjective initial segment. -/
noncomputable def _root_.InitialSeg.toPrincipalSeg [IsWellOrder β s] (f : r ≼i s)
    (hf : ¬ Surjective f) : r ≺i s :=
  ⟨f, _, Classical.choose_spec (f.eq_or_principal.resolve_left hf)⟩


@[simp]
theorem _root_.InitialSeg.toPrincipalSeg_apply [IsWellOrder β s] (f : r ≼i s)
    (hf : ¬ Surjective f) (x : α) : f.toPrincipalSeg hf x = f x :=
  rfl


theorem irrefl {r : α → α → Prop} [IsWellOrder α r] (f : r ≺i r) : False := by
  /-
    α : Type u_1
    r : α → α → Prop
    inst✝ : IsWellOrder α r
    f : PrincipalSeg r r
    ⊢ False
  -/
  have h := f.lt_top f.top
  /-
    α : Type u_1
    r : α → α → Prop
    inst✝ : IsWellOrder α r
    f : PrincipalSeg r r
    h : r (f.toRelEmbedding f.top) f.top
    ⊢ False
  -/
  rw [show f f.top = f.top from InitialSeg.eq f (InitialSeg.refl r) f.top] at h
  /-
    α : Type u_1
    r : α → α → Prop
    inst✝ : IsWellOrder α r
    f : PrincipalSeg r r
    h : r f.top f.top
    ⊢ False
  -/
  exact _root_.irrefl _ h
  /-
    🎉 no goals
  -/


instance (r : α → α → Prop) [IsWellOrder α r] : IsEmpty (r ≺i r) :=
  ⟨fun f => f.irrefl⟩


/-- Composition of a principal segment with an initial segment, as a principal segment -/
def transInitial (f : r ≺i s) (g : s ≼i t) : r ≺i t :=
  ⟨@RelEmbedding.trans _ _ _ r s t f g, g f.top, fun a => by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      r : α → α → Prop
      s : β → β → Prop
      t : γ → γ → Prop
      f : PrincipalSeg r s
      g : InitialSeg s t
      a : γ
      ⊢ Iff (Membership.mem (Set.range ⇑(f.trans g.toRelEmbedding)) a) (t a (g f.top))
    -/
    simp [g.exists_eq_iff_rel, ← PrincipalSeg.mem_range_iff_rel, exists_swap, ← exists_and_left]⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem transInitial_apply (f : r ≺i s) (g : s ≼i t) (a : α) : f.transInitial g a = g (f a) :=
  rfl


@[simp]
theorem transInitial_top (f : r ≺i s) (g : s ≼i t) : (f.transInitial g).top = g f.top :=
  rfl


@[deprecated (since := "2024-10-20")]
alias ltLe := transInitial


set_option linter.deprecated false in
@[deprecated transInitial_apply (since := "2024-10-20")]
theorem lt_le_apply (f : r ≺i s) (g : s ≼i t) (a : α) : (f.ltLe g) a = g (f a) :=
  rfl


set_option linter.deprecated false in
@[deprecated transInitial_top (since := "2024-10-20")]
theorem lt_le_top (f : r ≺i s) (g : s ≼i t) : (f.ltLe g).top = g f.top :=
  rfl


/-- Composition of two principal segments as a principal segment. -/
@[trans]
protected def trans [IsTrans γ t] (f : r ≺i s) (g : s ≺i t) : r ≺i t :=
  transInitial f g


@[simp]
theorem trans_apply [IsTrans γ t] (f : r ≺i s) (g : s ≺i t) (a : α) : f.trans g a = g (f a) :=
  rfl


@[simp]
theorem trans_top [IsTrans γ t] (f : r ≺i s) (g : s ≺i t) : (f.trans g).top = g f.top :=
  rfl


/-- Composition of an order isomorphism with a principal segment, as a principal segment. -/
def relIsoTrans (f : r ≃r s) (g : s ≺i t) : r ≺i t :=
                                                           /-
                                                             α : Type u_1
                                                             β : Type u_2
                                                             γ : Type u_3
                                                             r : α → α → Prop
                                                             s : β → β → Prop
                                                             t : γ → γ → Prop
                                                             f : RelIso r s
                                                             g : PrincipalSeg s t
                                                             c : γ
                                                             ⊢ Iff (Membership.mem (Set.range ⇑(f.toRelEmbedding.trans g.toRelEmbedding)) c …
                                                           -/
  ⟨@RelEmbedding.trans _ _ _ r s t f g, g.top, fun c => by simp [g.mem_range_iff_rel]⟩
                                                           /-
                                                             🎉 no goals
                                                           -/


@[simp]
theorem relIsoTrans_apply (f : r ≃r s) (g : s ≺i t) (a : α) : relIsoTrans f g a = g (f a) :=
  rfl


@[simp]
theorem relIsoTrans_top (f : r ≃r s) (g : s ≺i t) : (relIsoTrans f g).top = g.top :=
  rfl


@[deprecated (since := "2024-10-20")]
alias equivLT := relIsoTrans


set_option linter.deprecated false in
@[deprecated transInitial_top (since := "2024-10-20")]
theorem equivLT_apply (f : r ≃r s) (g : s ≺i t) (a : α) : (equivLT f g) a = g (f a) :=
  rfl


set_option linter.deprecated false in
@[deprecated transInitial_top (since := "2024-10-20")]
theorem equivLT_top (f : r ≃r s) (g : s ≺i t) : (equivLT f g).top = g.top :=
  rfl


/-- Composition of a principal segment with an order isomorphism, as a principal segment -/
def transRelIso (f : r ≺i s) (g : s ≃r t) : r ≺i t :=
  transInitial f g.toInitialSeg


@[deprecated (since := "2024-10-20")]
alias ltEquiv := transRelIso


@[simp]
theorem transRelIso_apply (f : r ≺i s) (g : s ≃r t) (a : α) : transRelIso f g a = g (f a) :=
  rfl


@[simp]
theorem transRelIso_top (f : r ≺i s) (g : s ≃r t) : (transRelIso f g).top = g f.top :=
  rfl


/-- Given a well order `s`, there is a most one principal segment embedding of `r` into `s`. -/
instance [IsWellOrder β s] : Subsingleton (r ≺i s) where
  allEq f g := ext ((f : r ≼i s).eq g)


protected theorem eq [IsWellOrder β s] (f g : r ≺i s) (a) : f a = g a := by
  /-
    α : Type u_1
    β : Type u_2
    r : α → α → Prop
    s : β → β → Prop
    inst✝ : IsWellOrder β s
    f g : PrincipalSeg r s
    a : α
    ⊢ Eq (f.toRelEmbedding a) (g.toRelEmbedding a)
  -/
  rw [Subsingleton.elim f g]
  /-
    🎉 no goals
  -/


theorem top_eq [IsWellOrder γ t] (e : r ≃r s) (f : r ≺i t) (g : s ≺i t) : f.top = g.top := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    r : α → α → Prop
    s : β → β → Prop
    t : γ → γ → Prop
    inst✝ : IsWellOrder γ t
    e : RelIso r s
    f : PrincipalSeg r t
    g : PrincipalSeg s t
    ⊢ Eq f.top g.top
  -/
  rw [Subsingleton.elim f (PrincipalSeg.relIsoTrans e g)]; rfl
                                                           /-
                                                             🎉 no goals
                                                           -/


theorem top_rel_top {r : α → α → Prop} {s : β → β → Prop} {t : γ → γ → Prop} [IsWellOrder γ t]
    (f : r ≺i s) (g : s ≺i t) (h : r ≺i t) : t h.top g.top := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    r : α → α → Prop
    s : β → β → Prop
    t : γ → γ → Prop
    inst✝ : IsWellOrder γ t
    f : PrincipalSeg r s
    g : PrincipalSeg s t
    h : PrincipalSeg r t
    ⊢ t h.top g.top
  -/
  rw [Subsingleton.elim h (f.trans g)]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    r : α → α → Prop
    s : β → β → Prop
    t : γ → γ → Prop
    inst✝ : IsWellOrder γ t
    f : PrincipalSeg r s
    g : PrincipalSeg s t
    h : PrincipalSeg r t
    ⊢ t (f.trans g).top g.top
  -/
  apply PrincipalSeg.lt_top
  /-
    🎉 no goals
  -/


@[deprecated top_rel_top (since := "2024-10-10")]
alias topLTTop := top_rel_top


/-- Any element of a well order yields a principal segment. -/
-- The explicit typing is required in order for `simp` to work properly.
@[simps!]
def ofElement {α : Type*} (r : α → α → Prop) (a : α) :
    @PrincipalSeg { b // r b a } α (Subrel r { b | r b a }) r :=
  ⟨Subrel.relEmbedding _ _, a, fun _ => ⟨fun ⟨⟨_, h⟩, rfl⟩ => h, fun h => ⟨⟨_, h⟩, rfl⟩⟩⟩


@[simp]
theorem ofElement_apply {α : Type*} (r : α → α → Prop) (a : α) (b) : ofElement r a b = b.1 :=
  rfl


/-- For any principal segment `r ≺i s`, there is a `Subrel` of `s` order isomorphic to `r`. -/
-- The explicit typing is required in order for `simp` to work properly.
@[simps! symm_apply]
noncomputable def subrelIso (f : r ≺i s) :
    @RelIso { b // s b f.top } α (Subrel s { b | s b f.top }) r :=
  RelIso.symm ⟨(Equiv.ofInjective f f.injective).trans
    (Equiv.setCongr (funext fun _ ↦ propext f.mem_range_iff_rel)), f.map_rel_iff⟩


@[simp]
theorem apply_subrelIso (f : r ≺i s) (b : {b | s b f.top}) : f (f.subrelIso b) = b :=
  Equiv.apply_ofInjective_symm f.injective _


@[simp]
theorem subrelIso_apply (f : r ≺i s) (a : α) : f.subrelIso ⟨f a, f.lt_top a⟩ = a :=
  Equiv.ofInjective_symm_apply f.injective _


/-- Restrict the codomain of a principal segment -/
def codRestrict (p : Set β) (f : r ≺i s) (H : ∀ a, f a ∈ p) (H₂ : f.top ∈ p) : r ≺i Subrel s p :=
                                                                 /-
                                                                   α : Type u_1
                                                                   β : Type u_2
                                                                   γ : Type u_3
                                                                   r : α → α → Prop
                                                                   s : β → β → Prop
                                                                   t : γ → γ → Prop
                                                                   p : Set β
                                                                   f : PrincipalSeg r s
                                                                   H : ∀ (a : α), Membership.mem p (f.toRelEmbedding a)
                                                                   H₂ : Membership.mem p f.top
                                                                   x✝ : ↑p
                                                                   val✝ : β
                                                                   property✝ : Membership.mem p val✝
                                                                   ⊢ Iff (Membership.mem (Set.range ⇑(RelEmbedding.codRestrict p f.toRelEmbedding …
                                                                 -/
  ⟨RelEmbedding.codRestrict p f H, ⟨f.top, H₂⟩, fun ⟨_, _⟩ => by simp [← f.mem_range_iff_rel]⟩
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[simp]
theorem codRestrict_apply (p) (f : r ≺i s) (H H₂ a) : codRestrict p f H H₂ a = ⟨f a, H a⟩ :=
  rfl


@[simp]
theorem codRestrict_top (p) (f : r ≺i s) (H H₂) : (codRestrict p f H H₂).top = ⟨f.top, H₂⟩ :=
  rfl


/-- Principal segment from an empty type into a type with a minimal element. -/
def ofIsEmpty (r : α → α → Prop) [IsEmpty α] {b : β} (H : ∀ b', ¬s b' b) : r ≺i s :=
  { RelEmbedding.ofIsEmpty r s with
    top := b
                             /-
                               α : Type u_1
                               β : Type u_2
                               γ : Type u_3
                               r✝ : α → α → Prop
                               s : β → β → Prop
                               t : γ → γ → Prop
                               r : α → α → Prop
                               inst✝ : IsEmpty α
                               b : β
                               H : ∀ (b' : β), Not (s b' b)
                               ⊢ ∀ (b_1 : β), Iff (Membership.mem (Set.range ⇑__src✝) b_1) (s b_1 b)
                             -/
    mem_range_iff_rel' := by simp [H] }
                             /-
                               🎉 no goals
                             -/


@[simp]
theorem ofIsEmpty_top (r : α → α → Prop) [IsEmpty α] {b : β} (H : ∀ b', ¬s b' b) :
    (ofIsEmpty r H).top = b :=
  rfl


/-- Principal segment from the empty relation on `PEmpty` to the empty relation on `PUnit`. -/
abbrev pemptyToPunit : @EmptyRelation PEmpty ≺i @EmptyRelation PUnit :=
  (@ofIsEmpty _ _ EmptyRelation _ _ PUnit.unit) fun _ => not_false


protected theorem acc [IsTrans β s] (f : r ≺i s) (a : α) : Acc r a ↔ Acc s (f a) :=
  (f : r ≼i s).acc a


theorem wellFounded_iff_principalSeg.{u} {β : Type u} {s : β → β → Prop} [IsTrans β s] :
    WellFounded s ↔ ∀ (α : Type u) (r : α → α → Prop) (_ : r ≺i s), WellFounded r :=
  ⟨fun wf _ _ f => RelHomClass.wellFounded f.toRelEmbedding wf, fun h =>
    wellFounded_iff_wellFounded_subrel.mpr fun b => h _ _ (PrincipalSeg.ofElement s b)⟩


open Classical in
/-- To an initial segment taking values in a well order, one can associate either a principal
segment (if the range is not everything, taking the top the minimum of the complement of the range)
or an order isomorphism (if the range is everything). -/
noncomputable def principalSumRelIso [IsWellOrder β s] (f : r ≼i s) : (r ≺i s) ⊕ (r ≃r s) :=
  if h : Surjective f
    then Sum.inr (RelIso.ofSurjective f h)
    else Sum.inl (f.toPrincipalSeg h)


@[deprecated principalSumRelIso (since := "2024-10-20")]
alias ltOrEq := principalSumRelIso


/-- Composition of an initial segment taking values in a well order and a principal segment. -/
noncomputable def transPrincipal [IsWellOrder β s] [IsTrans γ t] (f : r ≼i s) (g : s ≺i t) :
    r ≺i t :=
  match f.principalSumRelIso with
  | Sum.inl f' => f'.trans g
  | Sum.inr f' => PrincipalSeg.relIsoTrans f' g


@[simp]
theorem transPrincipal_apply [IsWellOrder β s] [IsTrans γ t] (f : r ≼i s) (g : s ≺i t) (a : α) :
    f.transPrincipal g a = g (f a) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    r : α → α → Prop
    s : β → β → Prop
    t : γ → γ → Prop
    inst✝¹ : IsWellOrder β s
    inst✝ : IsTrans γ t
    f : InitialSeg r s
    g : PrincipalSeg s t
    a : α
    ⊢ Eq ((f.transPrincipal g).toRelEmbedding a) (g.toRelEmbedding (f a))
  -/
  rw [InitialSeg.transPrincipal]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    r : α → α → Prop
    s : β → β → Prop
    t : γ → γ → Prop
    inst✝¹ : IsWellOrder β s
    inst✝ : IsTrans γ t
    f : InitialSeg r s
    g : PrincipalSeg s t
    a : α
    ⊢ Eq ((InitialSeg.transPrincipal.match_1 (fun x => PrincipalSeg r t) f.princip …
  -/
  obtain f' | f' := f.principalSumRelIso
    /-
      case inl
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      r : α → α → Prop
      s : β → β → Prop
      t : γ → γ → Prop
      inst✝¹ : IsWellOrder β s
      inst✝ : IsTrans γ t
      f : InitialSeg r s
      g : PrincipalSeg s t
      a : α
      f' : PrincipalSeg r s
      ⊢ Eq ((InitialSeg.transPrincipal.match_1 (fun x => PrincipalSeg r t) (Sum.inl  …
    -/
  · rw [PrincipalSeg.trans_apply, f.eq_principalSeg]
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      r : α → α → Prop
      s : β → β → Prop
      t : γ → γ → Prop
      inst✝¹ : IsWellOrder β s
      inst✝ : IsTrans γ t
      f : InitialSeg r s
      g : PrincipalSeg s t
      a : α
      f' : RelIso r s
      ⊢ Eq ((InitialSeg.transPrincipal.match_1 (fun x => PrincipalSeg r t) (Sum.inr  …
    -/
  · rw [PrincipalSeg.relIsoTrans_apply, f.eq_relIso]
    /-
      🎉 no goals
    -/


@[deprecated transPrincipal (since := "2024-10-20")]
alias leLT := transPrincipal


set_option linter.deprecated false in
@[deprecated transPrincipal_apply (since := "2024-10-20")]
theorem leLT_apply [IsWellOrder β s] [IsTrans γ t] (f : r ≼i s) (g : s ≺i t) (a : α) :
    f.leLT g a = g (f a) :=
  transPrincipal_apply f g a


/-- The function in `collapse`. -/
private noncomputable def collapseF [IsWellOrder β s] (f : r ↪r s) : Π a, { b // ¬s (f a) b } :=
  (RelEmbedding.isWellFounded f).fix _ fun a IH =>
    have H : f a ∈ { b | ∀ a h, s (IH a h).1 b } :=
      fun b h => trans_trichotomous_left (IH b h).2 (f.map_rel_iff.2 h)
    ⟨_, IsWellFounded.wf.not_lt_min _ ⟨_, H⟩ H⟩


private theorem collapseF_lt [IsWellOrder β s] (f : r ↪r s) {a : α} :
    ∀ {a'}, r a' a → s (collapseF f a') (collapseF f a) := by
  /-
    α : Type u_1
    β : Type u_2
    r : α → α → Prop
    s : β → β → Prop
    inst✝ : IsWellOrder β s
    f : RelEmbedding r s
    a : α
    ⊢ ∀ {a' : α}, r a' a → s ↑(collapseF f a') ↑(collapseF f a)
  -/
  show _ ∈ { b | ∀ a', r a' a → s (collapseF f a') b }
  /-
    α : Type u_1
    β : Type u_2
    r : α → α → Prop
    s : β → β → Prop
    inst✝ : IsWellOrder β s
    f : RelEmbedding r s
    a : α
    ⊢ Membership.mem (setOf fun b => ∀ (a' : α), r a' a → s (↑(collapseF f a')) b) …
  -/
  rw [collapseF, IsWellFounded.fix_eq]
  /-
    α : Type u_1
    β : Type u_2
    r : α → α → Prop
    s : β → β → Prop
    inst✝ : IsWellOrder β s
    f : RelEmbedding r s
    a : α
    ⊢ Membership.mem (setOf fun b => ∀ (a' : α), r a' a → s (↑(IsWellFounded.fix r …
  -/
  dsimp only
  /-
    α : Type u_1
    β : Type u_2
    r : α → α → Prop
    s : β → β → Prop
    inst✝ : IsWellOrder β s
    f : RelEmbedding r s
    a : α
    ⊢ Membership.mem (setOf fun b => ∀ (a' : α), r a' a → s (↑(IsWellFounded.fix r …
  -/
  exact WellFounded.min_mem _ _ _
  /-
    🎉 no goals
  -/


private theorem collapseF_not_lt [IsWellOrder β s] (f : r ↪r s) (a : α) {b}
    (h : ∀ a', r a' a → s (collapseF f a') b) : ¬s b (collapseF f a) := by
  /-
    α : Type u_1
    β : Type u_2
    r : α → α → Prop
    s : β → β → Prop
    inst✝ : IsWellOrder β s
    f : RelEmbedding r s
    a : α
    b : β
    h : ∀ (a' : α), r a' a → s (↑(collapseF f a')) b
    ⊢ Not (s b ↑(collapseF f a))
  -/
  rw [collapseF, IsWellFounded.fix_eq]
  /-
    α : Type u_1
    β : Type u_2
    r : α → α → Prop
    s : β → β → Prop
    inst✝ : IsWellOrder β s
    f : RelEmbedding r s
    a : α
    b : β
    h : ∀ (a' : α), r a' a → s (↑(collapseF f a')) b
    ⊢ Not (s b ↑(letFun ⋯ fun H => ⟨⋯.min (setOf fun b => ∀ (a_1 : α) (h : r a_1 a …
  -/
  dsimp only
  /-
    α : Type u_1
    β : Type u_2
    r : α → α → Prop
    s : β → β → Prop
    inst✝ : IsWellOrder β s
    f : RelEmbedding r s
    a : α
    b : β
    h : ∀ (a' : α), r a' a → s (↑(collapseF f a')) b
    ⊢ Not (s b (⋯.min (setOf fun b => ∀ (a_1 : α), r a_1 a → s (↑(IsWellFounded.fi …
  -/
  exact WellFounded.not_lt_min _ _ _ h
  /-
    🎉 no goals
  -/


/-- Construct an initial segment embedding `r ≼i s` by "filling in the gaps". That is, each
subsequent element in `α` is mapped to the least element in `β` that hasn't been used yet.

This construction is guaranteed to work as long as there exists some relation embedding `r ↪r s`. -/
noncomputable def RelEmbedding.collapse [IsWellOrder β s] (f : r ↪r s) : r ≼i s :=
  have H := RelEmbedding.isWellOrder f
  ⟨RelEmbedding.ofMonotone _ fun a b => collapseF_lt f, fun a b h ↦ by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      r : α → α → Prop
      s : β → β → Prop
      t : γ → γ → Prop
      inst✝ : IsWellOrder β s
      f : RelEmbedding r s
      H : IsWellOrder α r
      a : α
      b : β
      h : s b ((RelEmbedding.ofMonotone (fun a => ↑(collapseF f a)) ⋯) a)
      ⊢ Membership.mem (Set.range ⇑(RelEmbedding.ofMonotone (fun a => ↑(collapseF f  …
    -/
    obtain ⟨m, hm, hm'⟩ := H.wf.has_min { a | ¬s _ b } ⟨_, asymm h⟩
    /-
      case intro.intro
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      r : α → α → Prop
      s : β → β → Prop
      t : γ → γ → Prop
      inst✝ : IsWellOrder β s
      f : RelEmbedding r s
      H : IsWellOrder α r
      a : α
      b : β
      h : s b ((RelEmbedding.ofMonotone (fun a => ↑(collapseF f a)) ⋯) a)
      m : α
      hm : Membership.mem (setOf fun a => Not (s ((RelEmbedding.ofMonotone (fun a => …
      hm' : ∀ (x : α), Membership.mem (setOf fun a => Not (s ((RelEmbedding.ofMonoto …
      ⊢ Membership.mem (Set.range ⇑(RelEmbedding.ofMonotone (fun a => ↑(collapseF f  …
    -/
    use m
    /-
      case h
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      r : α → α → Prop
      s : β → β → Prop
      t : γ → γ → Prop
      inst✝ : IsWellOrder β s
      f : RelEmbedding r s
      H : IsWellOrder α r
      a : α
      b : β
      h : s b ((RelEmbedding.ofMonotone (fun a => ↑(collapseF f a)) ⋯) a)
      m : α
      hm : Membership.mem (setOf fun a => Not (s ((RelEmbedding.ofMonotone (fun a => …
      hm' : ∀ (x : α), Membership.mem (setOf fun a => Not (s ((RelEmbedding.ofMonoto …
      ⊢ Eq ((RelEmbedding.ofMonotone (fun a => ↑(collapseF f a)) ⋯) m) b
    -/
    obtain lt | rfl | gt := trichotomous_of s b (collapseF f m)
      /-
        case h.inl
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        r : α → α → Prop
        s : β → β → Prop
        t : γ → γ → Prop
        inst✝ : IsWellOrder β s
        f : RelEmbedding r s
        H : IsWellOrder α r
        a : α
        b : β
        h : s b ((RelEmbedding.ofMonotone (fun a => ↑(collapseF f a)) ⋯) a)
        m : α
        hm : Membership.mem (setOf fun a => Not (s ((RelEmbedding.ofMonotone (fun a => …
        hm' : ∀ (x : α), Membership.mem (setOf fun a => Not (s ((RelEmbedding.ofMonoto …
        lt : s b ↑(collapseF f m)
        ⊢ Eq ((RelEmbedding.ofMonotone (fun a => ↑(collapseF f a)) ⋯) m) b
      -/
    · refine (collapseF_not_lt f m (fun c h ↦ ?_) lt).elim
      /-
        case h.inl
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        r : α → α → Prop
        s : β → β → Prop
        t : γ → γ → Prop
        inst✝ : IsWellOrder β s
        f : RelEmbedding r s
        H : IsWellOrder α r
        a : α
        b : β
        h✝ : s b ((RelEmbedding.ofMonotone (fun a => ↑(collapseF f a)) ⋯) a)
        m : α
        hm : Membership.mem (setOf fun a => Not (s ((RelEmbedding.ofMonotone (fun a => …
        hm' : ∀ (x : α), Membership.mem (setOf fun a => Not (s ((RelEmbedding.ofMonoto …
        lt : s b ↑(collapseF f m)
        c : α
        h : r c m
        ⊢ s (↑(collapseF f c)) b
      -/
      by_contra hn
      /-
        case h.inl
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        r : α → α → Prop
        s : β → β → Prop
        t : γ → γ → Prop
        inst✝ : IsWellOrder β s
        f : RelEmbedding r s
        H : IsWellOrder α r
        a : α
        b : β
        h✝ : s b ((RelEmbedding.ofMonotone (fun a => ↑(collapseF f a)) ⋯) a)
        m : α
        hm : Membership.mem (setOf fun a => Not (s ((RelEmbedding.ofMonotone (fun a => …
        hm' : ∀ (x : α), Membership.mem (setOf fun a => Not (s ((RelEmbedding.ofMonoto …
        lt : s b ↑(collapseF f m)
        c : α
        h : r c m
        hn : Not (s (↑(collapseF f c)) b)
        ⊢ False
      -/
      exact hm' _ hn h
      /-
        🎉 no goals
      -/
      /-
        case h.inr.inl
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        r : α → α → Prop
        s : β → β → Prop
        t : γ → γ → Prop
        inst✝ : IsWellOrder β s
        f : RelEmbedding r s
        H : IsWellOrder α r
        a m : α
        h : s (↑(collapseF f m)) ((RelEmbedding.ofMonotone (fun a => ↑(collapseF f a)) …
        hm : Membership.mem (setOf fun a => Not (s ((RelEmbedding.ofMonotone (fun a => …
        hm' : ∀ (x : α), Membership.mem (setOf fun a => Not (s ((RelEmbedding.ofMonoto …
        ⊢ Eq ((RelEmbedding.ofMonotone (fun a => ↑(collapseF f a)) ⋯) m) ↑(collapseF f …
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case h.inr.inr
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        r : α → α → Prop
        s : β → β → Prop
        t : γ → γ → Prop
        inst✝ : IsWellOrder β s
        f : RelEmbedding r s
        H : IsWellOrder α r
        a : α
        b : β
        h : s b ((RelEmbedding.ofMonotone (fun a => ↑(collapseF f a)) ⋯) a)
        m : α
        hm : Membership.mem (setOf fun a => Not (s ((RelEmbedding.ofMonotone (fun a => …
        hm' : ∀ (x : α), Membership.mem (setOf fun a => Not (s ((RelEmbedding.ofMonoto …
        gt : s (↑(collapseF f m)) b
        ⊢ Eq ((RelEmbedding.ofMonotone (fun a => ↑(collapseF f a)) ⋯) m) b
      -/
    · exact (hm gt).elim⟩
      /-
        🎉 no goals
      -/


/-- For any two well orders, one is an initial segment of the other. -/
noncomputable def InitialSeg.total (r s) [IsWellOrder α r] [IsWellOrder β s] :
    (r ≼i s) ⊕ (s ≼i r) :=
  match (leAdd r s).principalSumRelIso,
    (RelEmbedding.sumLexInr r s).collapse.principalSumRelIso with
  | Sum.inl f, Sum.inr g => Sum.inl <| f.transRelIso g.symm
  | Sum.inr f, Sum.inl g => Sum.inr <| g.transRelIso f.symm
  | Sum.inr f, Sum.inr g => Sum.inl <| (f.trans g.symm).toInitialSeg
  | Sum.inl f, Sum.inl g => Classical.choice <| by
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        r✝ : α → α → Prop
        s✝ : β → β → Prop
        t : γ → γ → Prop
        r : α → α → Prop
        s : β → β → Prop
        inst✝¹ : IsWellOrder α r
        inst✝ : IsWellOrder β s
        f : PrincipalSeg r (Sum.Lex r s)
        g : PrincipalSeg s (Sum.Lex r s)
        ⊢ Nonempty (Sum (InitialSeg r s) (InitialSeg s r))
      -/
      obtain h | h | h := trichotomous_of (Sum.Lex r s) f.top g.top
      · exact ⟨Sum.inl <| (f.codRestrict {x | Sum.Lex r s x g.top}
          (fun a => _root_.trans (f.lt_top a) h) h).transRelIso g.subrelIso⟩
        /-
          case inr.inl
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          r✝ : α → α → Prop
          s✝ : β → β → Prop
          t : γ → γ → Prop
          r : α → α → Prop
          s : β → β → Prop
          inst✝¹ : IsWellOrder α r
          inst✝ : IsWellOrder β s
          f : PrincipalSeg r (Sum.Lex r s)
          g : PrincipalSeg s (Sum.Lex r s)
          h : Eq f.top g.top
          ⊢ Nonempty (Sum (InitialSeg r s) (InitialSeg s r))
        -/
      · let f := f.subrelIso
        /-
          case inr.inl
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          r✝ : α → α → Prop
          s✝ : β → β → Prop
          t : γ → γ → Prop
          r : α → α → Prop
          s : β → β → Prop
          inst✝¹ : IsWellOrder α r
          inst✝ : IsWellOrder β s
          f✝ : PrincipalSeg r (Sum.Lex r s)
          g : PrincipalSeg s (Sum.Lex r s)
          h : Eq f✝.top g.top
          f : RelIso (Subrel (Sum.Lex r s) (setOf fun b => Sum.Lex r s b f✝.top)) r := f …
          ⊢ Nonempty (Sum (InitialSeg r s) (InitialSeg s r))
        -/
        rw [h] at f
        /-
          case inr.inl
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          r✝ : α → α → Prop
          s✝ : β → β → Prop
          t : γ → γ → Prop
          r : α → α → Prop
          s : β → β → Prop
          inst✝¹ : IsWellOrder α r
          inst✝ : IsWellOrder β s
          f✝ : PrincipalSeg r (Sum.Lex r s)
          g : PrincipalSeg s (Sum.Lex r s)
          h : Eq f✝.top g.top
          f : RelIso (Subrel (Sum.Lex r s) (setOf fun b => Sum.Lex r s b g.top)) r
          ⊢ Nonempty (Sum (InitialSeg r s) (InitialSeg s r))
        -/
        exact ⟨Sum.inl <| (f.symm.trans g.subrelIso).toInitialSeg⟩
        /-
          🎉 no goals
        -/
      · exact ⟨Sum.inr <| (g.codRestrict {x | Sum.Lex r s x f.top}
          (fun a => _root_.trans (g.lt_top a) h) h).transRelIso f.subrelIso⟩


/-- An order isomorphism is an initial segment -/
@[simps!]
def _root_.OrderIso.toInitialSeg [Preorder α] [Preorder β] (f : α ≃o β) : α ≤i β :=
  f.toRelIsoLT.toInitialSeg


theorem mem_range_of_le [Preorder α] (f : α ≤i β) (h : b ≤ f a) : b ∈ Set.range f := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : PartialOrder β
    a : α
    b : β
    inst✝ : Preorder α
    f : InitialSeg (fun x1 x2 => LT.lt x1 x2) fun x1 x2 => LT.lt x1 x2
    h : LE.le b (f a)
    ⊢ Membership.mem (Set.range ⇑f) b
  -/
  obtain rfl | hb := h.eq_or_lt
  /-
    case inl
    α : Type u_1
    β : Type u_2
    inst✝¹ : PartialOrder β
    a : α
    inst✝ : Preorder α
    f : InitialSeg (fun x1 x2 => LT.lt x1 x2) fun x1 x2 => LT.lt x1 x2
    h : LE.le (f a) (f a)
    ⊢ Membership.mem (Set.range ⇑f) (f a)
  -/
  exacts [⟨a, rfl⟩, f.mem_range_of_rel hb]
  /-
    🎉 no goals
  -/


theorem isLowerSet_range [Preorder α] (f : α ≤i β) : IsLowerSet (Set.range f) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : PartialOrder β
    inst✝ : Preorder α
    f : InitialSeg (fun x1 x2 => LT.lt x1 x2) fun x1 x2 => LT.lt x1 x2
    ⊢ IsLowerSet (Set.range ⇑f)
  -/
  rintro _ b h ⟨a, rfl⟩
  /-
    case intro
    α : Type u_1
    β : Type u_2
    inst✝¹ : PartialOrder β
    inst✝ : Preorder α
    f : InitialSeg (fun x1 x2 => LT.lt x1 x2) fun x1 x2 => LT.lt x1 x2
    b : β
    a : α
    h : LE.le b (f a)
    ⊢ Membership.mem (Set.range ⇑f) b
  -/
  exact mem_range_of_le f h
  /-
    🎉 no goals
  -/

-- TODO: this would follow immediately if we had a `RelEmbeddingClass`

@[simp]
theorem le_iff_le [PartialOrder α] (f : α ≤i β) : f a ≤ f a' ↔ a ≤ a' :=
  f.toOrderEmbedding.le_iff_le

-- TODO: this would follow immediately if we had a `RelEmbeddingClass`

@[simp]
theorem lt_iff_lt [PartialOrder α] (f : α ≤i β) : f a < f a' ↔ a < a' :=
  f.toOrderEmbedding.lt_iff_lt


theorem monotone [PartialOrder α] (f : α ≤i β) : Monotone f :=
  f.toOrderEmbedding.monotone


theorem strictMono [PartialOrder α] (f : α ≤i β) : StrictMono f :=
  f.toOrderEmbedding.strictMono


@[simp]
theorem isMin_apply_iff [PartialOrder α] (f : α ≤i β) : IsMin (f a) ↔ IsMin a := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : PartialOrder β
    a : α
    inst✝ : PartialOrder α
    f : InitialSeg (fun x1 x2 => LT.lt x1 x2) fun x1 x2 => LT.lt x1 x2
    ⊢ Iff (IsMin (f a)) (IsMin a)
  -/
  refine ⟨StrictMono.isMin_of_apply f.strictMono, fun h b hb ↦ ?_⟩
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : PartialOrder β
    a : α
    inst✝ : PartialOrder α
    f : InitialSeg (fun x1 x2 => LT.lt x1 x2) fun x1 x2 => LT.lt x1 x2
    h : IsMin a
    b : β
    hb : LE.le b (f a)
    ⊢ LE.le (f a) b
  -/
  obtain ⟨x, rfl⟩ := f.mem_range_of_le hb
  /-
    case intro
    α : Type u_1
    β : Type u_2
    inst✝¹ : PartialOrder β
    a : α
    inst✝ : PartialOrder α
    f : InitialSeg (fun x1 x2 => LT.lt x1 x2) fun x1 x2 => LT.lt x1 x2
    h : IsMin a
    x : α
    hb : LE.le (f x) (f a)
    ⊢ LE.le (f a) (f x)
  -/
  rw [f.le_iff_le] at hb ⊢
  /-
    case intro
    α : Type u_1
    β : Type u_2
    inst✝¹ : PartialOrder β
    a : α
    inst✝ : PartialOrder α
    f : InitialSeg (fun x1 x2 => LT.lt x1 x2) fun x1 x2 => LT.lt x1 x2
    h : IsMin a
    x : α
    hb : LE.le x a
    ⊢ LE.le a x
  -/
  exact h hb
  /-
    🎉 no goals
  -/


alias ⟨_, map_isMin⟩ := isMin_apply_iff


@[simp]
theorem map_bot [PartialOrder α] [OrderBot α] [OrderBot β] (f : α ≤i β) : f ⊥ = ⊥ :=
  (map_isMin f isMin_bot).eq_bot


theorem le_apply_iff [LinearOrder α] (f : α ≤i β) : b ≤ f a ↔ ∃ c ≤ a, f c = b := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : PartialOrder β
    a : α
    b : β
    inst✝ : LinearOrder α
    f : InitialSeg (fun x1 x2 => LT.lt x1 x2) fun x1 x2 => LT.lt x1 x2
    ⊢ Iff (LE.le b (f a)) (Exists fun c => And (LE.le c a) (Eq (f c) b))
  -/
  constructor
    /-
      case mp
      α : Type u_1
      β : Type u_2
      inst✝¹ : PartialOrder β
      a : α
      b : β
      inst✝ : LinearOrder α
      f : InitialSeg (fun x1 x2 => LT.lt x1 x2) fun x1 x2 => LT.lt x1 x2
      ⊢ LE.le b (f a) → Exists fun c => And (LE.le c a) (Eq (f c) b)
    -/
  · intro h
    /-
      case mp
      α : Type u_1
      β : Type u_2
      inst✝¹ : PartialOrder β
      a : α
      b : β
      inst✝ : LinearOrder α
      f : InitialSeg (fun x1 x2 => LT.lt x1 x2) fun x1 x2 => LT.lt x1 x2
      h : LE.le b (f a)
      ⊢ Exists fun c => And (LE.le c a) (Eq (f c) b)
    -/
    obtain ⟨c, hc⟩ := f.mem_range_of_le h
    /-
      case mp.intro
      α : Type u_1
      β : Type u_2
      inst✝¹ : PartialOrder β
      a : α
      b : β
      inst✝ : LinearOrder α
      f : InitialSeg (fun x1 x2 => LT.lt x1 x2) fun x1 x2 => LT.lt x1 x2
      h : LE.le b (f a)
      c : α
      hc : Eq (f c) b
      ⊢ Exists fun c => And (LE.le c a) (Eq (f c) b)
    -/
    refine ⟨c, ?_, hc⟩
    /-
      case mp.intro
      α : Type u_1
      β : Type u_2
      inst✝¹ : PartialOrder β
      a : α
      b : β
      inst✝ : LinearOrder α
      f : InitialSeg (fun x1 x2 => LT.lt x1 x2) fun x1 x2 => LT.lt x1 x2
      h : LE.le b (f a)
      c : α
      hc : Eq (f c) b
      ⊢ LE.le c a
    -/
    rwa [← hc, f.le_iff_le] at h
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      β : Type u_2
      inst✝¹ : PartialOrder β
      a : α
      b : β
      inst✝ : LinearOrder α
      f : InitialSeg (fun x1 x2 => LT.lt x1 x2) fun x1 x2 => LT.lt x1 x2
      ⊢ (Exists fun c => And (LE.le c a) (Eq (f c) b)) → LE.le b (f a)
    -/
  · rintro ⟨c, hc, rfl⟩
    /-
      case mpr.intro.intro
      α : Type u_1
      β : Type u_2
      inst✝¹ : PartialOrder β
      a : α
      inst✝ : LinearOrder α
      f : InitialSeg (fun x1 x2 => LT.lt x1 x2) fun x1 x2 => LT.lt x1 x2
      c : α
      hc : LE.le c a
      ⊢ LE.le (f c) (f a)
    -/
    exact f.monotone hc
    /-
      🎉 no goals
    -/


theorem lt_apply_iff [LinearOrder α] (f : α ≤i β) : b < f a ↔ ∃ a' < a, f a' = b := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : PartialOrder β
    a : α
    b : β
    inst✝ : LinearOrder α
    f : InitialSeg (fun x1 x2 => LT.lt x1 x2) fun x1 x2 => LT.lt x1 x2
    ⊢ Iff (LT.lt b (f a)) (Exists fun a' => And (LT.lt a' a) (Eq (f a') b))
  -/
  constructor
    /-
      case mp
      α : Type u_1
      β : Type u_2
      inst✝¹ : PartialOrder β
      a : α
      b : β
      inst✝ : LinearOrder α
      f : InitialSeg (fun x1 x2 => LT.lt x1 x2) fun x1 x2 => LT.lt x1 x2
      ⊢ LT.lt b (f a) → Exists fun a' => And (LT.lt a' a) (Eq (f a') b)
    -/
  · intro h
    /-
      case mp
      α : Type u_1
      β : Type u_2
      inst✝¹ : PartialOrder β
      a : α
      b : β
      inst✝ : LinearOrder α
      f : InitialSeg (fun x1 x2 => LT.lt x1 x2) fun x1 x2 => LT.lt x1 x2
      h : LT.lt b (f a)
      ⊢ Exists fun a' => And (LT.lt a' a) (Eq (f a') b)
    -/
    obtain ⟨c, hc⟩ := f.mem_range_of_rel h
    /-
      case mp.intro
      α : Type u_1
      β : Type u_2
      inst✝¹ : PartialOrder β
      a : α
      b : β
      inst✝ : LinearOrder α
      f : InitialSeg (fun x1 x2 => LT.lt x1 x2) fun x1 x2 => LT.lt x1 x2
      h : LT.lt b (f a)
      c : α
      hc : Eq (f c) b
      ⊢ Exists fun a' => And (LT.lt a' a) (Eq (f a') b)
    -/
    refine ⟨c, ?_, hc⟩
    /-
      case mp.intro
      α : Type u_1
      β : Type u_2
      inst✝¹ : PartialOrder β
      a : α
      b : β
      inst✝ : LinearOrder α
      f : InitialSeg (fun x1 x2 => LT.lt x1 x2) fun x1 x2 => LT.lt x1 x2
      h : LT.lt b (f a)
      c : α
      hc : Eq (f c) b
      ⊢ LT.lt c a
    -/
    rwa [← hc, f.lt_iff_lt] at h
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      β : Type u_2
      inst✝¹ : PartialOrder β
      a : α
      b : β
      inst✝ : LinearOrder α
      f : InitialSeg (fun x1 x2 => LT.lt x1 x2) fun x1 x2 => LT.lt x1 x2
      ⊢ (Exists fun a' => And (LT.lt a' a) (Eq (f a') b)) → LT.lt b (f a)
    -/
  · rintro ⟨c, hc, rfl⟩
    /-
      case mpr.intro.intro
      α : Type u_1
      β : Type u_2
      inst✝¹ : PartialOrder β
      a : α
      inst✝ : LinearOrder α
      f : InitialSeg (fun x1 x2 => LT.lt x1 x2) fun x1 x2 => LT.lt x1 x2
      c : α
      hc : LT.lt c a
      ⊢ LT.lt (f c) (f a)
    -/
    exact f.strictMono hc
    /-
      🎉 no goals
    -/


theorem mem_range_of_le [Preorder α] (f : α <i β) (h : b ≤ f a) : b ∈ Set.range f :=
  (f : α ≤i β).mem_range_of_le h


theorem isLowerSet_range [Preorder α] (f : α <i β) : IsLowerSet (Set.range f) :=
  (f : α ≤i β).isLowerSet_range

-- TODO: this would follow immediately if we had a `RelEmbeddingClass`

@[simp]
theorem le_iff_le [PartialOrder α] (f : α <i β) : f a ≤ f a' ↔ a ≤ a' :=
  (f : α ≤i β).le_iff_le

-- TODO: this would follow immediately if we had a `RelEmbeddingClass`

@[simp]
theorem lt_iff_lt [PartialOrder α] (f : α <i β) : f a < f a' ↔ a < a' :=
  (f : α ≤i β).lt_iff_lt


theorem monotone [PartialOrder α] (f : α <i β) : Monotone f :=
  (f : α ≤i β).monotone


theorem strictMono [PartialOrder α] (f : α <i β) : StrictMono f :=
  (f : α ≤i β).strictMono


@[simp]
theorem isMin_apply_iff [PartialOrder α] (f : α <i β) : IsMin (f a) ↔ IsMin a :=
  (f : α ≤i β).isMin_apply_iff


@[simp]
theorem map_bot [PartialOrder α] [OrderBot α] [OrderBot β] (f : α <i β) : f ⊥ = ⊥ :=
  (f : α ≤i β).map_bot


theorem le_apply_iff [LinearOrder α] (f : α <i β) : b ≤ f a ↔ ∃ c ≤ a, f c = b :=
  (f : α ≤i β).le_apply_iff


theorem lt_apply_iff [LinearOrder α] (f : α <i β) : b < f a ↔ ∃ a' < a, f a' = b :=
  (f : α ≤i β).lt_apply_iff


