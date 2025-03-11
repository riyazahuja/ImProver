/-- A relation homomorphism with respect to a given pair of relations `r` and `s`
is a function `f : α → β` such that `r a b → s (f a) (f b)`. -/
structure RelHom {α β : Type*} (r : α → α → Prop) (s : β → β → Prop) where
  /-- The underlying function of a `RelHom` -/
  toFun : α → β
  /-- A `RelHom` sends related elements to related elements -/
  map_rel' : ∀ {a b}, r a b → s (toFun a) (toFun b)


/-- A relation homomorphism with respect to a given pair of relations `r` and `s`
is a function `f : α → β` such that `r a b → s (f a) (f b)`. -/
infixl:25 " →r " => RelHom


/-- `RelHomClass F r s` asserts that `F` is a type of functions such that all `f : F`
satisfy `r a b → s (f a) (f b)`.

The relations `r` and `s` are `outParam`s since figuring them out from a goal is a higher-order
matching problem that Lean usually can't do unaided.
-/
class RelHomClass (F : Type*) {α β : outParam Type*} (r : outParam <| α → α → Prop)
  (s : outParam <| β → β → Prop) [FunLike F α β] : Prop where
  /-- A `RelHomClass` sends related elements to related elements -/
  map_rel : ∀ (f : F) {a b}, r a b → s (f a) (f b)


protected theorem isIrrefl [RelHomClass F r s] (f : F) : ∀ [IsIrrefl β s], IsIrrefl α r
  | ⟨H⟩ => ⟨fun _ h => H _ (map_rel f h)⟩


protected theorem isAsymm [RelHomClass F r s] (f : F) : ∀ [IsAsymm β s], IsAsymm α r
  | ⟨H⟩ => ⟨fun _ _ h₁ h₂ => H _ _ (map_rel f h₁) (map_rel f h₂)⟩


protected theorem acc [RelHomClass F r s] (f : F) (a : α) : Acc s (f a) → Acc r a := by
  /-
    α : Type u_1
    β : Type u_2
    r : α → α → Prop
    s : β → β → Prop
    F : Type u_5
    inst✝¹ : FunLike F α β
    inst✝ : RelHomClass F r s
    f : F
    a : α
    ⊢ Acc s (f a) → Acc r a
  -/
  generalize h : f a = b
  /-
    α : Type u_1
    β : Type u_2
    r : α → α → Prop
    s : β → β → Prop
    F : Type u_5
    inst✝¹ : FunLike F α β
    inst✝ : RelHomClass F r s
    f : F
    a : α
    b : β
    h : Eq (f a) b
    ⊢ Acc s b → Acc r a
  -/
  intro ac
  /-
    α : Type u_1
    β : Type u_2
    r : α → α → Prop
    s : β → β → Prop
    F : Type u_5
    inst✝¹ : FunLike F α β
    inst✝ : RelHomClass F r s
    f : F
    a : α
    b : β
    h : Eq (f a) b
    ac : Acc s b
    ⊢ Acc r a
  -/
  induction ac generalizing a with | intro _ H IH => ?_
  /-
    case intro
    α : Type u_1
    β : Type u_2
    r : α → α → Prop
    s : β → β → Prop
    F : Type u_5
    inst✝¹ : FunLike F α β
    inst✝ : RelHomClass F r s
    f : F
    b x✝ : β
    H : ∀ (y : β), s y x✝ → Acc s y
    IH : ∀ (y : β), s y x✝ → ∀ (a : α), Eq (f a) y → Acc r a
    a : α
    h : Eq (f a) x✝
    ⊢ Acc r a
  -/
  subst h
  /-
    case intro
    α : Type u_1
    β : Type u_2
    r : α → α → Prop
    s : β → β → Prop
    F : Type u_5
    inst✝¹ : FunLike F α β
    inst✝ : RelHomClass F r s
    f : F
    b : β
    a : α
    H : ∀ (y : β), s y (f a) → Acc s y
    IH : ∀ (y : β), s y (f a) → ∀ (a : α), Eq (f a) y → Acc r a
    ⊢ Acc r a
  -/
  exact ⟨_, fun a' h => IH (f a') (map_rel f h) _ rfl⟩
  /-
    🎉 no goals
  -/


protected theorem wellFounded [RelHomClass F r s] (f : F) : WellFounded s → WellFounded r
  | ⟨H⟩ => ⟨fun _ => RelHomClass.acc f _ (H _)⟩


protected theorem isWellFounded [RelHomClass F r s] (f : F) [IsWellFounded β s] :
    IsWellFounded α r :=
  ⟨RelHomClass.wellFounded f IsWellFounded.wf⟩


instance : FunLike (r →r s) α β where
  coe o := o.toFun
  coe_injective' f g h := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      r : α → α → Prop
      s : β → β → Prop
      t : γ → γ → Prop
      u : δ → δ → Prop
      f g : RelHom r s
      h : Eq ((fun o => o.toFun) f) ((fun o => o.toFun) g)
      ⊢ Eq f g
    -/
    cases f
    /-
      case mk
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      r : α → α → Prop
      s : β → β → Prop
      t : γ → γ → Prop
      u : δ → δ → Prop
      g : RelHom r s
      toFun✝ : α → β
      map_rel'✝ : ∀ {a b : α}, r a b → s (toFun✝ a) (toFun✝ b)
      h : Eq ((fun o => o.toFun) { toFun := toFun✝, map_rel' := map_rel'✝ }) ((fun o …
      ⊢ Eq { toFun := toFun✝, map_rel' := map_rel'✝ } g
    -/
    cases g
    /-
      case mk.mk
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      r : α → α → Prop
      s : β → β → Prop
      t : γ → γ → Prop
      u : δ → δ → Prop
      toFun✝¹ : α → β
      map_rel'✝¹ : ∀ {a b : α}, r a b → s (toFun✝¹ a) (toFun✝¹ b)
      toFun✝ : α → β
      map_rel'✝ : ∀ {a b : α}, r a b → s (toFun✝ a) (toFun✝ b)
      h : Eq ((fun o => o.toFun) { toFun := toFun✝¹, map_rel' := map_rel'✝¹ }) ((fun …
      ⊢ Eq { toFun := toFun✝¹, map_rel' := map_rel'✝¹ } { toFun := toFun✝, map_rel'  …
    -/
    congr
    /-
      🎉 no goals
    -/


instance : RelHomClass (r →r s) r s where
  map_rel := map_rel'


protected theorem map_rel (f : r →r s) {a b} : r a b → s (f a) (f b) :=
  f.map_rel'


@[simp]
theorem coe_fn_toFun (f : r →r s) : f.toFun = (f : α → β) :=
  rfl


/-- The map `coe_fn : (r →r s) → (α → β)` is injective. -/
theorem coe_fn_injective : Injective fun (f : r →r s) => (f : α → β) :=
  DFunLike.coe_injective


@[ext]
theorem ext ⦃f g : r →r s⦄ (h : ∀ x, f x = g x) : f = g :=
  DFunLike.ext f g h


/-- Identity map is a relation homomorphism. -/
@[refl, simps]
protected def id (r : α → α → Prop) : r →r r :=
  ⟨fun x => x, fun x => x⟩


/-- Composition of two relation homomorphisms is a relation homomorphism. -/
@[simps]
protected def comp (g : s →r t) (f : r →r s) : r →r t :=
  ⟨fun x => g (f x), fun h => g.2 (f.2 h)⟩


/-- A relation homomorphism is also a relation homomorphism between dual relations. -/
protected def swap (f : r →r s) : swap r →r swap s :=
  ⟨f, f.map_rel⟩


/-- A function is a relation homomorphism from the preimage relation of `s` to `s`. -/
def preimage (f : α → β) (s : β → β → Prop) : f ⁻¹'o s →r s :=
  ⟨f, id⟩


/-- An increasing function is injective -/
theorem injective_of_increasing (r : α → α → Prop) (s : β → β → Prop) [IsTrichotomous α r]
    [IsIrrefl β s] (f : α → β) (hf : ∀ {x y}, r x y → s (f x) (f y)) : Injective f := by
  /-
    α : Type u_1
    β : Type u_2
    r : α → α → Prop
    s : β → β → Prop
    inst✝¹ : IsTrichotomous α r
    inst✝ : IsIrrefl β s
    f : α → β
    hf : ∀ {x y : α}, r x y → s (f x) (f y)
    ⊢ Function.Injective f
  -/
  intro x y hxy
  /-
    α : Type u_1
    β : Type u_2
    r : α → α → Prop
    s : β → β → Prop
    inst✝¹ : IsTrichotomous α r
    inst✝ : IsIrrefl β s
    f : α → β
    hf : ∀ {x y : α}, r x y → s (f x) (f y)
    x y : α
    hxy : Eq (f x) (f y)
    ⊢ Eq x y
  -/
  rcases trichotomous_of r x y with (h | h | h)
    /-
      case inl
      α : Type u_1
      β : Type u_2
      r : α → α → Prop
      s : β → β → Prop
      inst✝¹ : IsTrichotomous α r
      inst✝ : IsIrrefl β s
      f : α → β
      hf : ∀ {x y : α}, r x y → s (f x) (f y)
      x y : α
      hxy : Eq (f x) (f y)
      h : r x y
      ⊢ Eq x y
    -/
  · have := hf h
    /-
      case inl
      α : Type u_1
      β : Type u_2
      r : α → α → Prop
      s : β → β → Prop
      inst✝¹ : IsTrichotomous α r
      inst✝ : IsIrrefl β s
      f : α → β
      hf : ∀ {x y : α}, r x y → s (f x) (f y)
      x y : α
      hxy : Eq (f x) (f y)
      h : r x y
      this : s (f x) (f y)
      ⊢ Eq x y
    -/
    rw [hxy] at this
    /-
      case inl
      α : Type u_1
      β : Type u_2
      r : α → α → Prop
      s : β → β → Prop
      inst✝¹ : IsTrichotomous α r
      inst✝ : IsIrrefl β s
      f : α → β
      hf : ∀ {x y : α}, r x y → s (f x) (f y)
      x y : α
      hxy : Eq (f x) (f y)
      h : r x y
      this : s (f y) (f y)
      ⊢ Eq x y
    -/
    exfalso
    /-
      case inl
      α : Type u_1
      β : Type u_2
      r : α → α → Prop
      s : β → β → Prop
      inst✝¹ : IsTrichotomous α r
      inst✝ : IsIrrefl β s
      f : α → β
      hf : ∀ {x y : α}, r x y → s (f x) (f y)
      x y : α
      hxy : Eq (f x) (f y)
      h : r x y
      this : s (f y) (f y)
      ⊢ False
    -/
    exact irrefl_of s (f y) this
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      α : Type u_1
      β : Type u_2
      r : α → α → Prop
      s : β → β → Prop
      inst✝¹ : IsTrichotomous α r
      inst✝ : IsIrrefl β s
      f : α → β
      hf : ∀ {x y : α}, r x y → s (f x) (f y)
      x y : α
      hxy : Eq (f x) (f y)
      h : Eq x y
      ⊢ Eq x y
    -/
  · exact h
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      α : Type u_1
      β : Type u_2
      r : α → α → Prop
      s : β → β → Prop
      inst✝¹ : IsTrichotomous α r
      inst✝ : IsIrrefl β s
      f : α → β
      hf : ∀ {x y : α}, r x y → s (f x) (f y)
      x y : α
      hxy : Eq (f x) (f y)
      h : r y x
      ⊢ Eq x y
    -/
  · have := hf h
    /-
      case inr.inr
      α : Type u_1
      β : Type u_2
      r : α → α → Prop
      s : β → β → Prop
      inst✝¹ : IsTrichotomous α r
      inst✝ : IsIrrefl β s
      f : α → β
      hf : ∀ {x y : α}, r x y → s (f x) (f y)
      x y : α
      hxy : Eq (f x) (f y)
      h : r y x
      this : s (f y) (f x)
      ⊢ Eq x y
    -/
    rw [hxy] at this
    /-
      case inr.inr
      α : Type u_1
      β : Type u_2
      r : α → α → Prop
      s : β → β → Prop
      inst✝¹ : IsTrichotomous α r
      inst✝ : IsIrrefl β s
      f : α → β
      hf : ∀ {x y : α}, r x y → s (f x) (f y)
      x y : α
      hxy : Eq (f x) (f y)
      h : r y x
      this : s (f y) (f y)
      ⊢ Eq x y
    -/
    exfalso
    /-
      case inr.inr
      α : Type u_1
      β : Type u_2
      r : α → α → Prop
      s : β → β → Prop
      inst✝¹ : IsTrichotomous α r
      inst✝ : IsIrrefl β s
      f : α → β
      hf : ∀ {x y : α}, r x y → s (f x) (f y)
      x y : α
      hxy : Eq (f x) (f y)
      h : r y x
      this : s (f y) (f y)
      ⊢ False
    -/
    exact irrefl_of s (f y) this
    /-
      🎉 no goals
    -/


/-- An increasing function is injective -/
theorem RelHom.injective_of_increasing [IsTrichotomous α r] [IsIrrefl β s] (f : r →r s) :
    Injective f :=
  _root_.injective_of_increasing r s f f.map_rel


theorem Function.Surjective.wellFounded_iff {f : α → β} (hf : Surjective f)
    (o : ∀ {a b}, r a b ↔ s (f a) (f b)) :
    WellFounded r ↔ WellFounded s :=
  Iff.intro
    (RelHomClass.wellFounded (⟨surjInv hf,
                  /-
                    α : Type u_1
                    β : Type u_2
                    r : α → α → Prop
                    s : β → β → Prop
                    f : α → β
                    hf : Function.Surjective f
                    o : ∀ {a b : α}, Iff (r a b) (s (f a) (f b))
                    a✝ b✝ : β
                    h : s a✝ b✝
                    ⊢ r (Function.surjInv hf a✝) (Function.surjInv hf b✝)
                  -/
      fun h => by simpa only [o, surjInv_eq hf] using h⟩ : s →r r))
                  /-
                    🎉 no goals
                  -/
    (RelHomClass.wellFounded (⟨f, o.1⟩ : r →r s))


/-- A relation embedding with respect to a given pair of relations `r` and `s`
is an embedding `f : α ↪ β` such that `r a b ↔ s (f a) (f b)`. -/
structure RelEmbedding {α β : Type*} (r : α → α → Prop) (s : β → β → Prop) extends α ↪ β where
  /-- Elements are related iff they are related after apply a `RelEmbedding` -/
  map_rel_iff' : ∀ {a b}, s (toEmbedding a) (toEmbedding b) ↔ r a b


/-- A relation embedding with respect to a given pair of relations `r` and `s`
is an embedding `f : α ↪ β` such that `r a b ↔ s (f a) (f b)`. -/
infixl:25 " ↪r " => RelEmbedding


/-- The induced relation on a subtype is an embedding under the natural inclusion. -/
def Subtype.relEmbedding {X : Type*} (r : X → X → Prop) (p : X → Prop) :
    (Subtype.val : Subtype p → X) ⁻¹'o r ↪r r :=
  ⟨Embedding.subtype p, Iff.rfl⟩


theorem preimage_equivalence {α β} (f : α → β) {s : β → β → Prop} (hs : Equivalence s) :
    Equivalence (f ⁻¹'o s) :=
  ⟨fun _ => hs.1 _, fun h => hs.2 h, fun h₁ h₂ => hs.3 h₁ h₂⟩


/-- A relation embedding is also a relation homomorphism -/
def toRelHom (f : r ↪r s) : r →r s where
  toFun := f.toEmbedding.toFun
  map_rel' := (map_rel_iff' f).mpr


instance : Coe (r ↪r s) (r →r s) :=
  ⟨toRelHom⟩

-- TODO: define and instantiate a `RelEmbeddingClass` when `EmbeddingLike` is defined

instance : FunLike (r ↪r s) α β where
  coe x := x.toFun
  coe_injective' f g h := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      r : α → α → Prop
      s : β → β → Prop
      t : γ → γ → Prop
      u : δ → δ → Prop
      f g : RelEmbedding r s
      h : Eq ((fun x => x.toFun) f) ((fun x => x.toFun) g)
      ⊢ Eq f g
    -/
    rcases f with ⟨⟨⟩⟩
    /-
      case mk.mk
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      r : α → α → Prop
      s : β → β → Prop
      t : γ → γ → Prop
      u : δ → δ → Prop
      g : RelEmbedding r s
      toFun✝ : α → β
      inj'✝ : Function.Injective toFun✝
      map_rel_iff'✝ : ∀ {a b : α}, Iff (s ({ toFun := toFun✝, inj' := inj'✝ } a) ({  …
      h : Eq ((fun x => x.toFun) { toFun := toFun✝, inj' := inj'✝, map_rel_iff' := m …
      ⊢ Eq { toFun := toFun✝, inj' := inj'✝, map_rel_iff' := map_rel_iff'✝ } g
    -/
    rcases g with ⟨⟨⟩⟩
    /-
      case mk.mk.mk.mk
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      r : α → α → Prop
      s : β → β → Prop
      t : γ → γ → Prop
      u : δ → δ → Prop
      toFun✝¹ : α → β
      inj'✝¹ : Function.Injective toFun✝¹
      map_rel_iff'✝¹ : ∀ {a b : α}, Iff (s ({ toFun := toFun✝¹, inj' := inj'✝¹ } a)  …
      toFun✝ : α → β
      inj'✝ : Function.Injective toFun✝
      map_rel_iff'✝ : ∀ {a b : α}, Iff (s ({ toFun := toFun✝, inj' := inj'✝ } a) ({  …
      h : Eq ((fun x => x.toFun) { toFun := toFun✝¹, inj' := inj'✝¹, map_rel_iff' := …
      ⊢ Eq { toFun := toFun✝¹, inj' := inj'✝¹, map_rel_iff' := map_rel_iff'✝¹ } { to …
    -/
    congr
    /-
      🎉 no goals
    -/

-- TODO: define and instantiate a `RelEmbeddingClass` when `EmbeddingLike` is defined

instance : RelHomClass (r ↪r s) r s where
  map_rel f _ _ := Iff.mpr (map_rel_iff' f)


instance : EmbeddingLike (r ↪r s) α β where
  injective' f := f.inj'


@[simp]
theorem coe_toEmbedding {f : r ↪r s} : ((f : r ↪r s).toEmbedding : α → β) = f :=
  rfl


@[simp]
theorem coe_toRelHom {f : r ↪r s} : ((f : r ↪r s).toRelHom : α → β) = f :=
  rfl


theorem toEmbedding_injective : Injective (toEmbedding : r ↪r s → (α ↪ β)) := by
  /-
    α : Type u_1
    β : Type u_2
    r : α → α → Prop
    s : β → β → Prop
    ⊢ Function.Injective RelEmbedding.toEmbedding
  -/
  rintro ⟨f, -⟩ ⟨g, -⟩; simp
                        /-
                          🎉 no goals
                        -/


@[simp]
theorem toEmbedding_inj {f g : r ↪r s} : f.toEmbedding = g.toEmbedding ↔ f = g :=
  toEmbedding_injective.eq_iff


theorem injective (f : r ↪r s) : Injective f :=
  f.inj'


theorem inj (f : r ↪r s) {a b} : f a = f b ↔ a = b := f.injective.eq_iff


theorem map_rel_iff (f : r ↪r s) {a b} : s (f a) (f b) ↔ r a b :=
  f.map_rel_iff'


@[simp]
theorem coe_mk {f} {h} : ⇑(⟨f, h⟩ : r ↪r s) = f :=
  rfl


/-- The map `coe_fn : (r ↪r s) → (α → β)` is injective. -/
theorem coe_fn_injective : Injective fun f : r ↪r s => (f : α → β) :=
  DFunLike.coe_injective


@[ext]
theorem ext ⦃f g : r ↪r s⦄ (h : ∀ x, f x = g x) : f = g :=
  DFunLike.ext _ _ h


/-- Identity map is a relation embedding. -/
@[refl, simps!]
protected def refl (r : α → α → Prop) : r ↪r r :=
  ⟨Embedding.refl _, Iff.rfl⟩


/-- Composition of two relation embeddings is a relation embedding. -/
protected def trans (f : r ↪r s) (g : s ↪r t) : r ↪r t :=
                     /-
                       α : Type u_1
                       β : Type u_2
                       γ : Type u_3
                       δ : Type u_4
                       r : α → α → Prop
                       s : β → β → Prop
                       t : γ → γ → Prop
                       u : δ → δ → Prop
                       f : RelEmbedding r s
                       g : RelEmbedding s t
                       ⊢ ∀ {a b : α}, Iff (t ((f.trans g.toEmbedding) a) ((f.trans g.toEmbedding) b)) …
                     -/
  ⟨f.1.trans g.1, by simp [f.map_rel_iff, g.map_rel_iff]⟩
                     /-
                       🎉 no goals
                     -/


instance (r : α → α → Prop) : Inhabited (r ↪r r) :=
  ⟨RelEmbedding.refl _⟩


theorem trans_apply (f : r ↪r s) (g : s ↪r t) (a : α) : (f.trans g) a = g (f a) :=
  rfl


@[simp]
theorem coe_trans (f : r ↪r s) (g : s ↪r t) : (f.trans g) = g ∘ f :=
  rfl


/-- A relation embedding is also a relation embedding between dual relations. -/
protected def swap (f : r ↪r s) : swap r ↪r swap s :=
  ⟨f.toEmbedding, f.map_rel_iff⟩


/-- If `f` is injective, then it is a relation embedding from the
  preimage relation of `s` to `s`. -/
def preimage (f : α ↪ β) (s : β → β → Prop) : f ⁻¹'o s ↪r s :=
  ⟨f, Iff.rfl⟩


theorem eq_preimage (f : r ↪r s) : r = f ⁻¹'o s := by
  /-
    α : Type u_1
    β : Type u_2
    r : α → α → Prop
    s : β → β → Prop
    f : RelEmbedding r s
    ⊢ Eq r (Order.Preimage (⇑f) s)
  -/
  ext a b
  /-
    case h.h.a
    α : Type u_1
    β : Type u_2
    r : α → α → Prop
    s : β → β → Prop
    f : RelEmbedding r s
    a b : α
    ⊢ Iff (r a b) (Order.Preimage (⇑f) s a b)
  -/
  exact f.map_rel_iff.symm
  /-
    🎉 no goals
  -/


protected theorem isIrrefl (f : r ↪r s) [IsIrrefl β s] : IsIrrefl α r :=
  ⟨fun a => mt f.map_rel_iff.2 (irrefl (f a))⟩


protected theorem isRefl (f : r ↪r s) [IsRefl β s] : IsRefl α r :=
  ⟨fun _ => f.map_rel_iff.1 <| refl _⟩


protected theorem isSymm (f : r ↪r s) [IsSymm β s] : IsSymm α r :=
  ⟨fun _ _ => imp_imp_imp f.map_rel_iff.2 f.map_rel_iff.1 symm⟩


protected theorem isAsymm (f : r ↪r s) [IsAsymm β s] : IsAsymm α r :=
  ⟨fun _ _ h₁ h₂ => asymm (f.map_rel_iff.2 h₁) (f.map_rel_iff.2 h₂)⟩


protected theorem isAntisymm : ∀ (_ : r ↪r s) [IsAntisymm β s], IsAntisymm α r
  | ⟨f, o⟩, ⟨H⟩ => ⟨fun _ _ h₁ h₂ => f.inj' (H _ _ (o.2 h₁) (o.2 h₂))⟩


protected theorem isTrans : ∀ (_ : r ↪r s) [IsTrans β s], IsTrans α r
  | ⟨_, o⟩, ⟨H⟩ => ⟨fun _ _ _ h₁ h₂ => o.1 (H _ _ _ (o.2 h₁) (o.2 h₂))⟩


protected theorem isTotal : ∀ (_ : r ↪r s) [IsTotal β s], IsTotal α r
  | ⟨_, o⟩, ⟨H⟩ => ⟨fun _ _ => (or_congr o o).1 (H _ _)⟩


protected theorem isPreorder : ∀ (_ : r ↪r s) [IsPreorder β s], IsPreorder α r
  | f, _ => { f.isRefl, f.isTrans with }


protected theorem isPartialOrder : ∀ (_ : r ↪r s) [IsPartialOrder β s], IsPartialOrder α r
  | f, _ => { f.isPreorder, f.isAntisymm with }


protected theorem isLinearOrder : ∀ (_ : r ↪r s) [IsLinearOrder β s], IsLinearOrder α r
  | f, _ => { f.isPartialOrder, f.isTotal with }


protected theorem isStrictOrder : ∀ (_ : r ↪r s) [IsStrictOrder β s], IsStrictOrder α r
  | f, _ => { f.isIrrefl, f.isTrans with }


protected theorem isTrichotomous : ∀ (_ : r ↪r s) [IsTrichotomous β s], IsTrichotomous α r
  | ⟨f, o⟩, ⟨H⟩ => ⟨fun _ _ => (or_congr o (or_congr f.inj'.eq_iff o)).1 (H _ _)⟩


protected theorem isStrictTotalOrder : ∀ (_ : r ↪r s) [IsStrictTotalOrder β s],
    IsStrictTotalOrder α r
  | f, _ => { f.isTrichotomous, f.isStrictOrder with }


protected theorem acc (f : r ↪r s) (a : α) : Acc s (f a) → Acc r a := by
  /-
    α : Type u_1
    β : Type u_2
    r : α → α → Prop
    s : β → β → Prop
    f : RelEmbedding r s
    a : α
    ⊢ Acc s (f a) → Acc r a
  -/
  generalize h : f a = b
  /-
    α : Type u_1
    β : Type u_2
    r : α → α → Prop
    s : β → β → Prop
    f : RelEmbedding r s
    a : α
    b : β
    h : Eq (f a) b
    ⊢ Acc s b → Acc r a
  -/
  intro ac
  /-
    α : Type u_1
    β : Type u_2
    r : α → α → Prop
    s : β → β → Prop
    f : RelEmbedding r s
    a : α
    b : β
    h : Eq (f a) b
    ac : Acc s b
    ⊢ Acc r a
  -/
  induction ac generalizing a with | intro _ H IH => ?_
  /-
    case intro
    α : Type u_1
    β : Type u_2
    r : α → α → Prop
    s : β → β → Prop
    f : RelEmbedding r s
    b x✝ : β
    H : ∀ (y : β), s y x✝ → Acc s y
    IH : ∀ (y : β), s y x✝ → ∀ (a : α), Eq (f a) y → Acc r a
    a : α
    h : Eq (f a) x✝
    ⊢ Acc r a
  -/
  subst h
  /-
    case intro
    α : Type u_1
    β : Type u_2
    r : α → α → Prop
    s : β → β → Prop
    f : RelEmbedding r s
    b : β
    a : α
    H : ∀ (y : β), s y (f a) → Acc s y
    IH : ∀ (y : β), s y (f a) → ∀ (a : α), Eq (f a) y → Acc r a
    ⊢ Acc r a
  -/
  exact ⟨_, fun a' h => IH (f a') (f.map_rel_iff.2 h) _ rfl⟩
  /-
    🎉 no goals
  -/


protected theorem wellFounded : ∀ (_ : r ↪r s) (_ : WellFounded s), WellFounded r
  | f, ⟨H⟩ => ⟨fun _ => f.acc _ (H _)⟩


protected theorem isWellFounded (f : r ↪r s) [IsWellFounded β s] : IsWellFounded α r :=
  ⟨f.wellFounded IsWellFounded.wf⟩


protected theorem isWellOrder : ∀ (_ : r ↪r s) [IsWellOrder β s], IsWellOrder α r
  | f, H => { f.isStrictTotalOrder with wf := f.wellFounded H.wf }


instance Subtype.wellFoundedLT [LT α] [WellFoundedLT α] (p : α → Prop) :
    WellFoundedLT (Subtype p) :=
  (Subtype.relEmbedding (· < ·) p).isWellFounded


instance Subtype.wellFoundedGT [LT α] [WellFoundedGT α] (p : α → Prop) :
    WellFoundedGT (Subtype p) :=
  (Subtype.relEmbedding (· > ·) p).isWellFounded


/-- `Quotient.mk` as a relation homomorphism between the relation and the lift of a relation. -/
@[simps]
def Quotient.mkRelHom {_ : Setoid α} {r : α → α → Prop}
    (H : ∀ (a₁ b₁ a₂ b₂ : α), a₁ ≈ a₂ → b₁ ≈ b₂ → r a₁ b₁ = r a₂ b₂) : r →r Quotient.lift₂ r H :=
  ⟨Quotient.mk _, id⟩


/-- `Quotient.out` as a relation embedding between the lift of a relation and the relation. -/
@[simps!]
noncomputable def Quotient.outRelEmbedding {_ : Setoid α} {r : α → α → Prop}
    (H : ∀ (a₁ b₁ a₂ b₂ : α), a₁ ≈ a₂ → b₁ ≈ b₂ → r a₁ b₁ = r a₂ b₂) : Quotient.lift₂ r H ↪r r :=
  ⟨Embedding.quotientOut α, by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      r✝ : α → α → Prop
      s : β → β → Prop
      t : γ → γ → Prop
      u : δ → δ → Prop
      x✝ : Setoid α
      r : α → α → Prop
      H : ∀ (a₁ b₁ a₂ b₂ : α), HasEquiv.Equiv a₁ a₂ → HasEquiv.Equiv b₁ b₂ → Eq (r a …
      ⊢ ∀ {a b : Quotient x✝}, Iff (r ((Function.Embedding.quotientOut α) a) ((Funct …
    -/
    refine @fun x y => Quotient.inductionOn₂ x y fun a b => ?_
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      r✝ : α → α → Prop
      s : β → β → Prop
      t : γ → γ → Prop
      u : δ → δ → Prop
      x✝ : Setoid α
      r : α → α → Prop
      H : ∀ (a₁ b₁ a₂ b₂ : α), HasEquiv.Equiv a₁ a₂ → HasEquiv.Equiv b₁ b₂ → Eq (r a …
      x y : Quotient x✝
      a b : α
      ⊢ Iff (r ((Function.Embedding.quotientOut α) (Quotient.mk x✝ a)) ((Function.Em …
    -/
                                           /-
                                             🎉 no goals
                                           -/
    apply iff_iff_eq.2 (H _ _ _ _ _ _) <;> apply Quotient.mk_out⟩
                                           /-
                                             🎉 no goals
                                           -/


set_option linter.deprecated false in
/-- `Quotient.out'` as a relation embedding between the lift of a relation and the relation. -/
@[deprecated Quotient.outRelEmbedding (since := "2024-10-19"), simps]
noncomputable def Quotient.out'RelEmbedding {_ : Setoid α} {r : α → α → Prop}
    (H : ∀ (a₁ b₁ a₂ b₂ : α), a₁ ≈ a₂ → b₁ ≈ b₂ → r a₁ b₁ = r a₂ b₂) :
    (fun a b => Quotient.liftOn₂' a b r H) ↪r r :=
  { Quotient.outRelEmbedding H with toFun := Quotient.out' }


@[simp]
theorem acc_lift₂_iff {_ : Setoid α} {r : α → α → Prop}
    {H : ∀ (a₁ b₁ a₂ b₂ : α), a₁ ≈ a₂ → b₁ ≈ b₂ → r a₁ b₁ = r a₂ b₂} {a} :
    Acc (Quotient.lift₂ r H) ⟦a⟧ ↔ Acc r a := by
  /-
    α : Type u_1
    x✝ : Setoid α
    r : α → α → Prop
    H : ∀ (a₁ b₁ a₂ b₂ : α), HasEquiv.Equiv a₁ a₂ → HasEquiv.Equiv b₁ b₂ → Eq (r a …
    a : α
    ⊢ Iff (Acc (Quotient.lift₂ r H) (Quotient.mk x✝ a)) (Acc r a)
  -/
  constructor
    /-
      case mp
      α : Type u_1
      x✝ : Setoid α
      r : α → α → Prop
      H : ∀ (a₁ b₁ a₂ b₂ : α), HasEquiv.Equiv a₁ a₂ → HasEquiv.Equiv b₁ b₂ → Eq (r a …
      a : α
      ⊢ Acc (Quotient.lift₂ r H) (Quotient.mk x✝ a) → Acc r a
    -/
  · exact RelHomClass.acc (Quotient.mkRelHom H) a
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      x✝ : Setoid α
      r : α → α → Prop
      H : ∀ (a₁ b₁ a₂ b₂ : α), HasEquiv.Equiv a₁ a₂ → HasEquiv.Equiv b₁ b₂ → Eq (r a …
      a : α
      ⊢ Acc r a → Acc (Quotient.lift₂ r H) (Quotient.mk x✝ a)
    -/
  · intro ac
    /-
      case mpr
      α : Type u_1
      x✝ : Setoid α
      r : α → α → Prop
      H : ∀ (a₁ b₁ a₂ b₂ : α), HasEquiv.Equiv a₁ a₂ → HasEquiv.Equiv b₁ b₂ → Eq (r a …
      a : α
      ac : Acc r a
      ⊢ Acc (Quotient.lift₂ r H) (Quotient.mk x✝ a)
    -/
    induction ac with | intro _ _ IH => ?_
    /-
      case mpr.intro
      α : Type u_1
      x✝¹ : Setoid α
      r : α → α → Prop
      H : ∀ (a₁ b₁ a₂ b₂ : α), HasEquiv.Equiv a₁ a₂ → HasEquiv.Equiv b₁ b₂ → Eq (r a …
      a x✝ : α
      h✝ : ∀ (y : α), r y x✝ → Acc r y
      IH : ∀ (y : α), r y x✝ → Acc (Quotient.lift₂ r H) (Quotient.mk x✝¹ y)
      ⊢ Acc (Quotient.lift₂ r H) (Quotient.mk x✝¹ x✝)
    -/
    refine ⟨_, fun q h => ?_⟩
    /-
      case mpr.intro
      α : Type u_1
      x✝¹ : Setoid α
      r : α → α → Prop
      H : ∀ (a₁ b₁ a₂ b₂ : α), HasEquiv.Equiv a₁ a₂ → HasEquiv.Equiv b₁ b₂ → Eq (r a …
      a x✝ : α
      h✝ : ∀ (y : α), r y x✝ → Acc r y
      IH : ∀ (y : α), r y x✝ → Acc (Quotient.lift₂ r H) (Quotient.mk x✝¹ y)
      q : Quotient x✝¹
      h : Quotient.lift₂ r H q (Quotient.mk x✝¹ x✝)
      ⊢ Acc (Quotient.lift₂ r H) q
    -/
    obtain ⟨a', rfl⟩ := q.exists_rep
    /-
      case mpr.intro.intro
      α : Type u_1
      x✝¹ : Setoid α
      r : α → α → Prop
      H : ∀ (a₁ b₁ a₂ b₂ : α), HasEquiv.Equiv a₁ a₂ → HasEquiv.Equiv b₁ b₂ → Eq (r a …
      a x✝ : α
      h✝ : ∀ (y : α), r y x✝ → Acc r y
      IH : ∀ (y : α), r y x✝ → Acc (Quotient.lift₂ r H) (Quotient.mk x✝¹ y)
      a' : α
      h : Quotient.lift₂ r H (Quotient.mk x✝¹ a') (Quotient.mk x✝¹ x✝)
      ⊢ Acc (Quotient.lift₂ r H) (Quotient.mk x✝¹ a')
    -/
    exact IH a' h
    /-
      🎉 no goals
    -/


@[simp]
theorem acc_liftOn₂'_iff {s : Setoid α} {r : α → α → Prop} {H} {a} :
    Acc (fun x y => Quotient.liftOn₂' x y r H) (Quotient.mk'' a : Quotient s) ↔ Acc r a :=
  acc_lift₂_iff (H := H)


/-- A relation is well founded iff its lift to a quotient is. -/
@[simp]
theorem wellFounded_lift₂_iff {_ : Setoid α} {r : α → α → Prop}
    {H : ∀ (a₁ b₁ a₂ b₂ : α), a₁ ≈ a₂ → b₁ ≈ b₂ → r a₁ b₁ = r a₂ b₂} :
    WellFounded (Quotient.lift₂ r H) ↔ WellFounded r := by
  /-
    α : Type u_1
    x✝ : Setoid α
    r : α → α → Prop
    H : ∀ (a₁ b₁ a₂ b₂ : α), HasEquiv.Equiv a₁ a₂ → HasEquiv.Equiv b₁ b₂ → Eq (r a …
    ⊢ Iff (WellFounded (Quotient.lift₂ r H)) (WellFounded r)
  -/
  constructor
    /-
      case mp
      α : Type u_1
      x✝ : Setoid α
      r : α → α → Prop
      H : ∀ (a₁ b₁ a₂ b₂ : α), HasEquiv.Equiv a₁ a₂ → HasEquiv.Equiv b₁ b₂ → Eq (r a …
      ⊢ WellFounded (Quotient.lift₂ r H) → WellFounded r
    -/
  · exact RelHomClass.wellFounded (Quotient.mkRelHom H)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      x✝ : Setoid α
      r : α → α → Prop
      H : ∀ (a₁ b₁ a₂ b₂ : α), HasEquiv.Equiv a₁ a₂ → HasEquiv.Equiv b₁ b₂ → Eq (r a …
      ⊢ WellFounded r → WellFounded (Quotient.lift₂ r H)
    -/
  · refine fun wf => ⟨fun q => ?_⟩
    /-
      case mpr
      α : Type u_1
      x✝ : Setoid α
      r : α → α → Prop
      H : ∀ (a₁ b₁ a₂ b₂ : α), HasEquiv.Equiv a₁ a₂ → HasEquiv.Equiv b₁ b₂ → Eq (r a …
      wf : WellFounded r
      q : Quotient x✝
      ⊢ Acc (Quotient.lift₂ r H) q
    -/
    obtain ⟨a, rfl⟩ := q.exists_rep
    /-
      case mpr.intro
      α : Type u_1
      x✝ : Setoid α
      r : α → α → Prop
      H : ∀ (a₁ b₁ a₂ b₂ : α), HasEquiv.Equiv a₁ a₂ → HasEquiv.Equiv b₁ b₂ → Eq (r a …
      wf : WellFounded r
      a : α
      ⊢ Acc (Quotient.lift₂ r H) (Quotient.mk x✝ a)
    -/
    exact acc_lift₂_iff.2 (wf.apply a)
    /-
      🎉 no goals
    -/


alias ⟨WellFounded.of_quotient_lift₂, WellFounded.quotient_lift₂⟩ := wellFounded_lift₂_iff


@[simp]
theorem wellFounded_liftOn₂'_iff {s : Setoid α} {r : α → α → Prop} {H} :
    (WellFounded fun x y : Quotient s => Quotient.liftOn₂' x y r H) ↔ WellFounded r :=
  wellFounded_lift₂_iff (H := H)


alias ⟨WellFounded.of_quotient_liftOn₂', WellFounded.quotient_liftOn₂'⟩ := wellFounded_liftOn₂'_iff


/-- To define a relation embedding from an antisymmetric relation `r` to a reflexive relation `s`
it suffices to give a function together with a proof that it satisfies `s (f a) (f b) ↔ r a b`.
-/
def ofMapRelIff (f : α → β) [IsAntisymm α r] [IsRefl β s] (hf : ∀ a b, s (f a) (f b) ↔ r a b) :
    r ↪r s where
  toFun := f
  inj' _ _ h := antisymm ((hf _ _).1 (h ▸ refl _)) ((hf _ _).1 (h ▸ refl _))
  map_rel_iff' := hf _ _


@[simp]
theorem ofMapRelIff_coe (f : α → β) [IsAntisymm α r] [IsRefl β s]
    (hf : ∀ a b, s (f a) (f b) ↔ r a b) :
    (ofMapRelIff f hf : r ↪r s) = f :=
  rfl


/-- It suffices to prove `f` is monotone between strict relations
  to show it is a relation embedding. -/
def ofMonotone [IsTrichotomous α r] [IsAsymm β s] (f : α → β) (H : ∀ a b, r a b → s (f a) (f b)) :
    r ↪r s := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    r : α → α → Prop
    s : β → β → Prop
    t : γ → γ → Prop
    u : δ → δ → Prop
    inst✝¹ : IsTrichotomous α r
    inst✝ : IsAsymm β s
    f : α → β
    H : ∀ (a b : α), r a b → s (f a) (f b)
    ⊢ RelEmbedding r s
  -/
  haveI := @IsAsymm.isIrrefl β s _
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    r : α → α → Prop
    s : β → β → Prop
    t : γ → γ → Prop
    u : δ → δ → Prop
    inst✝¹ : IsTrichotomous α r
    inst✝ : IsAsymm β s
    f : α → β
    H : ∀ (a b : α), r a b → s (f a) (f b)
    this : IsIrrefl β s
    ⊢ RelEmbedding r s
  -/
  refine ⟨⟨f, fun a b e => ?_⟩, @fun a b => ⟨fun h => ?_, H _ _⟩⟩
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      r : α → α → Prop
      s : β → β → Prop
      t : γ → γ → Prop
      u : δ → δ → Prop
      inst✝¹ : IsTrichotomous α r
      inst✝ : IsAsymm β s
      f : α → β
      H : ∀ (a b : α), r a b → s (f a) (f b)
      this : IsIrrefl β s
      a b : α
      e : Eq (f a) (f b)
      ⊢ Eq a b
    -/
  · refine ((@trichotomous _ r _ a b).resolve_left ?_).resolve_right ?_
      /-
        case refine_1.refine_1
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        δ : Type u_4
        r : α → α → Prop
        s : β → β → Prop
        t : γ → γ → Prop
        u : δ → δ → Prop
        inst✝¹ : IsTrichotomous α r
        inst✝ : IsAsymm β s
        f : α → β
        H : ∀ (a b : α), r a b → s (f a) (f b)
        this : IsIrrefl β s
        a b : α
        e : Eq (f a) (f b)
        ⊢ Not (r a b)
      -/
    · exact fun h => irrefl (r := s) (f a) (by simpa [e] using H _ _ h)
      /-
        🎉 no goals
      -/
      /-
        case refine_1.refine_2
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        δ : Type u_4
        r : α → α → Prop
        s : β → β → Prop
        t : γ → γ → Prop
        u : δ → δ → Prop
        inst✝¹ : IsTrichotomous α r
        inst✝ : IsAsymm β s
        f : α → β
        H : ∀ (a b : α), r a b → s (f a) (f b)
        this : IsIrrefl β s
        a b : α
        e : Eq (f a) (f b)
        ⊢ Not (r b a)
      -/
    · exact fun h => irrefl (r := s) (f b) (by simpa [e] using H _ _ h)
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      r : α → α → Prop
      s : β → β → Prop
      t : γ → γ → Prop
      u : δ → δ → Prop
      inst✝¹ : IsTrichotomous α r
      inst✝ : IsAsymm β s
      f : α → β
      H : ∀ (a b : α), r a b → s (f a) (f b)
      this : IsIrrefl β s
      a b : α
      h : s ({ toFun := f, inj' := ⋯ } a) ({ toFun := f, inj' := ⋯ } b)
      ⊢ r a b
    -/
  · refine (@trichotomous _ r _ a b).resolve_right (Or.rec (fun e => ?_) fun h' => ?_)
      /-
        case refine_2.refine_1
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        δ : Type u_4
        r : α → α → Prop
        s : β → β → Prop
        t : γ → γ → Prop
        u : δ → δ → Prop
        inst✝¹ : IsTrichotomous α r
        inst✝ : IsAsymm β s
        f : α → β
        H : ∀ (a b : α), r a b → s (f a) (f b)
        this : IsIrrefl β s
        a b : α
        h : s ({ toFun := f, inj' := ⋯ } a) ({ toFun := f, inj' := ⋯ } b)
        e : Eq a b
        ⊢ False
      -/
    · subst e
      /-
        case refine_2.refine_1
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        δ : Type u_4
        r : α → α → Prop
        s : β → β → Prop
        t : γ → γ → Prop
        u : δ → δ → Prop
        inst✝¹ : IsTrichotomous α r
        inst✝ : IsAsymm β s
        f : α → β
        H : ∀ (a b : α), r a b → s (f a) (f b)
        this : IsIrrefl β s
        a : α
        h : s ({ toFun := f, inj' := ⋯ } a) ({ toFun := f, inj' := ⋯ } a)
        ⊢ False
      -/
      exact irrefl _ h
      /-
        🎉 no goals
      -/
      /-
        case refine_2.refine_2
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        δ : Type u_4
        r : α → α → Prop
        s : β → β → Prop
        t : γ → γ → Prop
        u : δ → δ → Prop
        inst✝¹ : IsTrichotomous α r
        inst✝ : IsAsymm β s
        f : α → β
        H : ∀ (a b : α), r a b → s (f a) (f b)
        this : IsIrrefl β s
        a b : α
        h : s ({ toFun := f, inj' := ⋯ } a) ({ toFun := f, inj' := ⋯ } b)
        h' : r b a
        ⊢ False
      -/
    · exact asymm (H _ _ h') h
      /-
        🎉 no goals
      -/


@[simp]
theorem ofMonotone_coe [IsTrichotomous α r] [IsAsymm β s] (f : α → β) (H) :
    (@ofMonotone _ _ r s _ _ f H : α → β) = f :=
  rfl


/-- A relation embedding from an empty type. -/
def ofIsEmpty (r : α → α → Prop) (s : β → β → Prop) [IsEmpty α] : r ↪r s :=
  ⟨Embedding.ofIsEmpty, @fun a => isEmptyElim a⟩


/-- `Sum.inl` as a relation embedding into `Sum.LiftRel r s`. -/
@[simps]
def sumLiftRelInl (r : α → α → Prop) (s : β → β → Prop) : r ↪r Sum.LiftRel r s where
  toFun := Sum.inl
  inj' := Sum.inl_injective
  map_rel_iff' := Sum.liftRel_inl_inl


/-- `Sum.inr` as a relation embedding into `Sum.LiftRel r s`. -/
@[simps]
def sumLiftRelInr (r : α → α → Prop) (s : β → β → Prop) : s ↪r Sum.LiftRel r s where
  toFun := Sum.inr
  inj' := Sum.inr_injective
  map_rel_iff' := Sum.liftRel_inr_inr


/-- `Sum.map` as a relation embedding between `Sum.LiftRel` relations. -/
@[simps]
def sumLiftRelMap (f : r ↪r s) (g : t ↪r u) : Sum.LiftRel r t ↪r Sum.LiftRel s u where
  toFun := Sum.map f g
  inj' := f.injective.sum_map g.injective
                     /-
                       α : Type u_1
                       β : Type u_2
                       γ : Type u_3
                       δ : Type u_4
                       r : α → α → Prop
                       s : β → β → Prop
                       t : γ → γ → Prop
                       u : δ → δ → Prop
                       f : RelEmbedding r s
                       g : RelEmbedding t u
                       ⊢ ∀ {a b : Sum α γ}, Iff (Sum.LiftRel s u ({ toFun := Sum.map ⇑f ⇑g, inj' := ⋯ …
                     -/
                                                /-
                                                  🎉 no goals
                                                -/
                                                /-
                                                  🎉 no goals
                                                -/
                                                /-
                                                  🎉 no goals
                                                -/
  map_rel_iff' := by rintro (a | b) (c | d) <;> simp [f.map_rel_iff, g.map_rel_iff]
                                                /-
                                                  🎉 no goals
                                                -/


/-- `Sum.inl` as a relation embedding into `Sum.Lex r s`. -/
@[simps]
def sumLexInl (r : α → α → Prop) (s : β → β → Prop) : r ↪r Sum.Lex r s where
  toFun := Sum.inl
  inj' := Sum.inl_injective
  map_rel_iff' := Sum.lex_inl_inl


/-- `Sum.inr` as a relation embedding into `Sum.Lex r s`. -/
@[simps]
def sumLexInr (r : α → α → Prop) (s : β → β → Prop) : s ↪r Sum.Lex r s where
  toFun := Sum.inr
  inj' := Sum.inr_injective
  map_rel_iff' := Sum.lex_inr_inr


/-- `Sum.map` as a relation embedding between `Sum.Lex` relations. -/
@[simps]
def sumLexMap (f : r ↪r s) (g : t ↪r u) : Sum.Lex r t ↪r Sum.Lex s u where
  toFun := Sum.map f g
  inj' := f.injective.sum_map g.injective
                     /-
                       α : Type u_1
                       β : Type u_2
                       γ : Type u_3
                       δ : Type u_4
                       r : α → α → Prop
                       s : β → β → Prop
                       t : γ → γ → Prop
                       u : δ → δ → Prop
                       f : RelEmbedding r s
                       g : RelEmbedding t u
                       ⊢ ∀ {a b : Sum α γ}, Iff (Sum.Lex s u ({ toFun := Sum.map ⇑f ⇑g, inj' := ⋯ } a …
                     -/
                                                /-
                                                  🎉 no goals
                                                -/
                                                /-
                                                  🎉 no goals
                                                -/
                                                /-
                                                  🎉 no goals
                                                -/
  map_rel_iff' := by rintro (a | b) (c | d) <;> simp [f.map_rel_iff, g.map_rel_iff]
                                                /-
                                                  🎉 no goals
                                                -/


/-- `fun b ↦ Prod.mk a b` as a relation embedding. -/
@[simps]
def prodLexMkLeft (s : β → β → Prop) {a : α} (h : ¬r a a) : s ↪r Prod.Lex r s where
  toFun := Prod.mk a
  inj' := Prod.mk.inj_left a
                     /-
                       α : Type u_1
                       β : Type u_2
                       γ : Type u_3
                       δ : Type u_4
                       r : α → α → Prop
                       s✝ : β → β → Prop
                       t : γ → γ → Prop
                       u : δ → δ → Prop
                       s : β → β → Prop
                       a : α
                       h : Not (r a a)
                       ⊢ ∀ {a_1 b : β}, Iff (Prod.Lex r s ({ toFun := Prod.mk a, inj' := ⋯ } a_1) ({  …
                     -/
  map_rel_iff' := by simp [Prod.lex_def, h]
                     /-
                       🎉 no goals
                     -/


/-- `fun a ↦ Prod.mk a b` as a relation embedding. -/
@[simps]
def prodLexMkRight (r : α → α → Prop) {b : β} (h : ¬s b b) : r ↪r Prod.Lex r s where
  toFun a := (a, b)
  inj' := Prod.mk.inj_right b
                     /-
                       α : Type u_1
                       β : Type u_2
                       γ : Type u_3
                       δ : Type u_4
                       r✝ : α → α → Prop
                       s : β → β → Prop
                       t : γ → γ → Prop
                       u : δ → δ → Prop
                       r : α → α → Prop
                       b : β
                       h : Not (s b b)
                       ⊢ ∀ {a b_1 : α}, Iff (Prod.Lex r s ({ toFun := fun a => { fst := a, snd := b } …
                     -/
  map_rel_iff' := by simp [Prod.lex_def, h]
                     /-
                       🎉 no goals
                     -/


/-- `Prod.map` as a relation embedding. -/
@[simps]
def prodLexMap (f : r ↪r s) (g : t ↪r u) : Prod.Lex r t ↪r Prod.Lex s u where
  toFun := Prod.map f g
  inj' := f.injective.prodMap g.injective
                     /-
                       α : Type u_1
                       β : Type u_2
                       γ : Type u_3
                       δ : Type u_4
                       r : α → α → Prop
                       s : β → β → Prop
                       t : γ → γ → Prop
                       u : δ → δ → Prop
                       f : RelEmbedding r s
                       g : RelEmbedding t u
                       ⊢ ∀ {a b : Prod α γ}, Iff (Prod.Lex s u ({ toFun := Prod.map ⇑f ⇑g, inj' := ⋯  …
                     -/
  map_rel_iff' := by simp [Prod.lex_def, f.map_rel_iff, g.map_rel_iff, f.inj]
                     /-
                       🎉 no goals
                     -/


/-- A relation isomorphism is an equivalence that is also a relation embedding. -/
structure RelIso {α β : Type*} (r : α → α → Prop) (s : β → β → Prop) extends α ≃ β where
  /-- Elements are related iff they are related after apply a `RelIso` -/
  map_rel_iff' : ∀ {a b}, s (toEquiv a) (toEquiv b) ↔ r a b


/-- A relation isomorphism is an equivalence that is also a relation embedding. -/
infixl:25 " ≃r " => RelIso


/-- Convert a `RelIso` to a `RelEmbedding`. This function is also available as a coercion
but often it is easier to write `f.toRelEmbedding` than to write explicitly `r` and `s`
in the target type. -/
def toRelEmbedding (f : r ≃r s) : r ↪r s :=
  ⟨f.toEquiv.toEmbedding, f.map_rel_iff'⟩


theorem toEquiv_injective : Injective (toEquiv : r ≃r s → α ≃ β)
                               /-
                                 α : Type u_1
                                 β : Type u_2
                                 r : α → α → Prop
                                 s : β → β → Prop
                                 e₁ : Equiv α β
                                 o₁ : ∀ {a b : α}, Iff (s (e₁ a) (e₁ b)) (r a b)
                                 e₂ : Equiv α β
                                 map_rel_iff'✝ : ∀ {a b : α}, Iff (s (e₂ a) (e₂ b)) (r a b)
                                 h : Eq { toEquiv := e₁, map_rel_iff' := o₁ }.toEquiv { toEquiv := e₂, map_rel_ …
                                 ⊢ Eq { toEquiv := e₁, map_rel_iff' := o₁ } { toEquiv := e₂, map_rel_iff' := ma …
                               -/
  | ⟨e₁, o₁⟩, ⟨e₂, _⟩, h => by congr
                               /-
                                 🎉 no goals
                               -/


instance : CoeOut (r ≃r s) (r ↪r s) :=
  ⟨toRelEmbedding⟩

-- TODO: define and instantiate a `RelIsoClass` when `EquivLike` is defined

instance : FunLike (r ≃r s) α β where
  coe x := x
  coe_injective' := Equiv.coe_fn_injective.comp toEquiv_injective

-- TODO: define and instantiate a `RelIsoClass` when `EquivLike` is defined

instance : RelHomClass (r ≃r s) r s where
  map_rel f _ _ := Iff.mpr (map_rel_iff' f)


instance : EquivLike (r ≃r s) α β where
  coe f := f
  inv f := f.toEquiv.symm
  left_inv f := f.left_inv
  right_inv f := f.right_inv
  coe_injective' _ _ hf _ := DFunLike.ext' hf


@[simp]
theorem coe_toRelEmbedding (f : r ≃r s) : (f.toRelEmbedding : α → β) = f :=
  rfl


@[simp]
theorem coe_toEmbedding (f : r ≃r s) : (f.toEmbedding : α → β) = f :=
  rfl


theorem map_rel_iff (f : r ≃r s) {a b} : s (f a) (f b) ↔ r a b :=
  f.map_rel_iff'


@[simp]
theorem coe_fn_mk (f : α ≃ β) (o : ∀ ⦃a b⦄, s (f a) (f b) ↔ r a b) :
    (RelIso.mk f @o : α → β) = f :=
  rfl


@[simp]
theorem coe_fn_toEquiv (f : r ≃r s) : (f.toEquiv : α → β) = f :=
  rfl


/-- The map `DFunLike.coe : (r ≃r s) → (α → β)` is injective. -/
theorem coe_fn_injective : Injective fun f : r ≃r s => (f : α → β) :=
  DFunLike.coe_injective


@[ext]
theorem ext ⦃f g : r ≃r s⦄ (h : ∀ x, f x = g x) : f = g :=
  DFunLike.ext f g h


/-- Inverse map of a relation isomorphism is a relation isomorphism. -/
protected def symm (f : r ≃r s) : s ≃r r :=
                                  /-
                                    α : Type u_1
                                    β : Type u_2
                                    γ : Type u_3
                                    δ : Type u_4
                                    r : α → α → Prop
                                    s : β → β → Prop
                                    t : γ → γ → Prop
                                    u : δ → δ → Prop
                                    f : RelIso r s
                                    a b : β
                                    ⊢ Iff (r (f.symm a) (f.symm b)) (s a b)
                                  -/
  ⟨f.toEquiv.symm, @fun a b => by erw [← f.map_rel_iff, f.1.apply_symm_apply, f.1.apply_symm_apply]⟩
                                  /-
                                    🎉 no goals
                                  -/


/-- See Note [custom simps projection]. We need to specify this projection explicitly in this case,
  because `RelIso` defines custom coercions other than the ones given by `DFunLike`. -/
def Simps.apply (h : r ≃r s) : α → β :=
  h


/-- See Note [custom simps projection]. -/
def Simps.symm_apply (h : r ≃r s) : β → α :=
  h.symm


/-- Identity map is a relation isomorphism. -/
@[refl, simps! apply]
protected def refl (r : α → α → Prop) : r ≃r r :=
  ⟨Equiv.refl _, Iff.rfl⟩


/-- Composition of two relation isomorphisms is a relation isomorphism. -/
@[simps! apply]
protected def trans (f₁ : r ≃r s) (f₂ : s ≃r t) : r ≃r t :=
  ⟨f₁.toEquiv.trans f₂.toEquiv, f₂.map_rel_iff.trans f₁.map_rel_iff⟩


instance (r : α → α → Prop) : Inhabited (r ≃r r) :=
  ⟨RelIso.refl _⟩


@[simp]
theorem default_def (r : α → α → Prop) : default = RelIso.refl r :=
  rfl


/-- A relation isomorphism between equal relations on equal types. -/
@[simps! toEquiv apply]
protected def cast {α β : Type u} {r : α → α → Prop} {s : β → β → Prop} (h₁ : α = β)
    (h₂ : HEq r s) : r ≃r s :=
  ⟨Equiv.cast h₁, @fun a b => by
    /-
      α✝ : Type u_1
      β✝ : Type u_2
      γ : Type u_3
      δ : Type u_4
      r✝ : α✝ → α✝ → Prop
      s✝ : β✝ → β✝ → Prop
      t : γ → γ → Prop
      u : δ → δ → Prop
      α β : Type u
      r : α → α → Prop
      s : β → β → Prop
      h₁ : Eq α β
      h₂ : HEq r s
      a b : α
      ⊢ Iff (s ((Equiv.cast h₁) a) ((Equiv.cast h₁) b)) (r a b)
    -/
    subst h₁
    /-
      α✝ : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      r✝ : α✝ → α✝ → Prop
      s✝ : β → β → Prop
      t : γ → γ → Prop
      u : δ → δ → Prop
      α : Type u
      r : α → α → Prop
      a b : α
      s : α → α → Prop
      h₂ : HEq r s
      ⊢ Iff (s ((Equiv.cast ⋯) a) ((Equiv.cast ⋯) b)) (r a b)
    -/
    rw [eq_of_heq h₂]
    /-
      α✝ : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      r✝ : α✝ → α✝ → Prop
      s✝ : β → β → Prop
      t : γ → γ → Prop
      u : δ → δ → Prop
      α : Type u
      r : α → α → Prop
      a b : α
      s : α → α → Prop
      h₂ : HEq r s
      ⊢ Iff (s ((Equiv.cast ⋯) a) ((Equiv.cast ⋯) b)) (s a b)
    -/
    rfl⟩
    /-
      🎉 no goals
    -/


@[simp]
protected theorem cast_symm {α β : Type u} {r : α → α → Prop} {s : β → β → Prop} (h₁ : α = β)
    (h₂ : HEq r s) : (RelIso.cast h₁ h₂).symm = RelIso.cast h₁.symm h₂.symm :=
  rfl


@[simp]
protected theorem cast_refl {α : Type u} {r : α → α → Prop} (h₁ : α = α := rfl)
    (h₂ : HEq r r := HEq.rfl) : RelIso.cast h₁ h₂ = RelIso.refl r :=
  rfl


@[simp]
protected theorem cast_trans {α β γ : Type u} {r : α → α → Prop} {s : β → β → Prop}
    {t : γ → γ → Prop} (h₁ : α = β) (h₁' : β = γ) (h₂ : HEq r s) (h₂' : HEq s t) :
    (RelIso.cast h₁ h₂).trans (RelIso.cast h₁' h₂') = RelIso.cast (h₁.trans h₁') (h₂.trans h₂') :=
                  /-
                    α β γ : Type u
                    r : α → α → Prop
                    s : β → β → Prop
                    t : γ → γ → Prop
                    h₁ : Eq α β
                    h₁' : Eq β γ
                    h₂ : HEq r s
                    h₂' : HEq s t
                    x : α
                    ⊢ Eq (((RelIso.cast h₁ h₂).trans (RelIso.cast h₁' h₂')) x) ((RelIso.cast ⋯ ⋯) x)
                  -/
  ext fun x => by subst h₁; rfl
                            /-
                              🎉 no goals
                            -/


/-- A relation isomorphism is also a relation isomorphism between dual relations. -/
protected def swap (f : r ≃r s) : swap r ≃r swap s :=
  ⟨f, f.map_rel_iff⟩


/-- A relation isomorphism is also a relation isomorphism between complemented relations. -/
@[simps!]
protected def compl (f : r ≃r s) : rᶜ ≃r sᶜ :=
  ⟨f, f.map_rel_iff.not⟩


@[simp]
theorem coe_fn_symm_mk (f o) : ((@RelIso.mk _ _ r s f @o).symm : β → α) = f.symm :=
  rfl


@[simp]
theorem apply_symm_apply (e : r ≃r s) (x : β) : e (e.symm x) = x :=
  e.toEquiv.apply_symm_apply x


@[simp]
theorem symm_apply_apply (e : r ≃r s) (x : α) : e.symm (e x) = x :=
  e.toEquiv.symm_apply_apply x


theorem rel_symm_apply (e : r ≃r s) {x y} : r x (e.symm y) ↔ s (e x) y := by
  /-
    α : Type u_1
    β : Type u_2
    r : α → α → Prop
    s : β → β → Prop
    e : RelIso r s
    x : α
    y : β
    ⊢ Iff (r x (e.symm y)) (s (e x) y)
  -/
  rw [← e.map_rel_iff, e.apply_symm_apply]
  /-
    🎉 no goals
  -/


theorem symm_apply_rel (e : r ≃r s) {x y} : r (e.symm x) y ↔ s x (e y) := by
  /-
    α : Type u_1
    β : Type u_2
    r : α → α → Prop
    s : β → β → Prop
    e : RelIso r s
    x : β
    y : α
    ⊢ Iff (r (e.symm x) y) (s x (e y))
  -/
  rw [← e.map_rel_iff, e.apply_symm_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem self_trans_symm (e : r ≃r s) : e.trans e.symm = RelIso.refl r :=
  ext e.symm_apply_apply


@[simp]
theorem symm_trans_self (e : r ≃r s) : e.symm.trans e = RelIso.refl s :=
  ext e.apply_symm_apply


protected theorem bijective (e : r ≃r s) : Bijective e :=
  e.toEquiv.bijective


protected theorem injective (e : r ≃r s) : Injective e :=
  e.toEquiv.injective


protected theorem surjective (e : r ≃r s) : Surjective e :=
  e.toEquiv.surjective


theorem eq_iff_eq (f : r ≃r s) {a b} : f a = f b ↔ a = b :=
  f.injective.eq_iff


/-- Any equivalence lifts to a relation isomorphism between `s` and its preimage. -/
protected def preimage (f : α ≃ β) (s : β → β → Prop) : f ⁻¹'o s ≃r s :=
  ⟨f, Iff.rfl⟩


instance IsWellOrder.preimage {α : Type u} (r : α → α → Prop) [IsWellOrder α r] (f : β ≃ α) :
    IsWellOrder β (f ⁻¹'o r) :=
  @RelEmbedding.isWellOrder _ _ (f ⁻¹'o r) r (RelIso.preimage f r) _


instance IsWellOrder.ulift {α : Type u} (r : α → α → Prop) [IsWellOrder α r] :
    IsWellOrder (ULift α) (ULift.down ⁻¹'o r) :=
  IsWellOrder.preimage r Equiv.ulift


/-- A surjective relation embedding is a relation isomorphism. -/
@[simps! apply]
noncomputable def ofSurjective (f : r ↪r s) (H : Surjective f) : r ≃r s :=
  ⟨Equiv.ofBijective f ⟨f.injective, H⟩, f.map_rel_iff⟩


/-- Given relation isomorphisms `r₁ ≃r s₁` and `r₂ ≃r s₂`, construct a relation isomorphism for the
lexicographic orders on the sum.
-/
def sumLexCongr {α₁ α₂ β₁ β₂ r₁ r₂ s₁ s₂} (e₁ : @RelIso α₁ β₁ r₁ s₁) (e₂ : @RelIso α₂ β₂ r₂ s₂) :
    Sum.Lex r₁ r₂ ≃r Sum.Lex s₁ s₂ :=
  ⟨Equiv.sumCongr e₁.toEquiv e₂.toEquiv, @fun a b => by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      r : α → α → Prop
      s : β → β → Prop
      t : γ → γ → Prop
      u : δ → δ → Prop
      α₁ : Type ?u.50942
      α₂ : Type ?u.50950
      β₁ : Type ?u.50941
      β₂ : Type ?u.50949
      r₁ : α₁ → α₁ → Prop
      r₂ : α₂ → α₂ → Prop
      s₁ : β₁ → β₁ → Prop
      s₂ : β₂ → β₂ → Prop
      e₁ : RelIso r₁ s₁
      e₂ : RelIso r₂ s₂
      a b : Sum α₁ α₂
      ⊢ Iff (Sum.Lex s₁ s₂ ((e₁.sumCongr e₂.toEquiv) a) ((e₁.sumCongr e₂.toEquiv) b) …
    -/
                                                                        /-
                                                                          🎉 no goals
                                                                        -/
                                                                        /-
                                                                          🎉 no goals
                                                                        -/
                                                                        /-
                                                                          🎉 no goals
                                                                        -/
    obtain ⟨f, hf⟩ := e₁; obtain ⟨g, hg⟩ := e₂; cases a <;> cases b <;> simp [hf, hg]⟩
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


/-- Given relation isomorphisms `r₁ ≃r s₁` and `r₂ ≃r s₂`, construct a relation isomorphism for the
lexicographic orders on the product.
-/
def prodLexCongr {α₁ α₂ β₁ β₂ r₁ r₂ s₁ s₂} (e₁ : @RelIso α₁ β₁ r₁ s₁) (e₂ : @RelIso α₂ β₂ r₂ s₂) :
    Prod.Lex r₁ r₂ ≃r Prod.Lex s₁ s₂ :=
  ⟨Equiv.prodCongr e₁.toEquiv e₂.toEquiv, by simp [Prod.lex_def, e₁.map_rel_iff, e₂.map_rel_iff,
    e₁.injective.eq_iff]⟩


/-- Two relations on empty types are isomorphic. -/
def relIsoOfIsEmpty (r : α → α → Prop) (s : β → β → Prop) [IsEmpty α] [IsEmpty β] : r ≃r s :=
  ⟨Equiv.equivOfIsEmpty α β, @fun a => isEmptyElim a⟩


/-- Two irreflexive relations on a unique type are isomorphic. -/
def ofUniqueOfIrrefl (r : α → α → Prop) (s : β → β → Prop) [IsIrrefl α r]
    [IsIrrefl β s] [Unique α] [Unique β] : r ≃r s :=
  ⟨Equiv.ofUnique α β, iff_of_false (not_rel_of_subsingleton s _ _)
      (not_rel_of_subsingleton r _ _) ⟩


@[deprecated (since := "2024-12-26")] alias relIsoOfUniqueOfIrrefl := ofUniqueOfIrrefl


/-- Two reflexive relations on a unique type are isomorphic. -/
def ofUniqueOfRefl (r : α → α → Prop) (s : β → β → Prop) [IsRefl α r] [IsRefl β s]
    [Unique α] [Unique β] : r ≃r s :=
  ⟨Equiv.ofUnique α β, iff_of_true (rel_of_subsingleton s _ _) (rel_of_subsingleton r _ _)⟩


@[deprecated (since := "2024-12-26")] alias relIsoOfUniqueOfRefl := ofUniqueOfRefl


