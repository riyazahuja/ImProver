/--
We say that `f : α → β` semiconjugates `ga : α → α` to `gb : β → β` if `f ∘ ga = gb ∘ f`.
We use `∀ x, f (ga x) = gb (f x)` as the definition, so given `h : Function.Semiconj f ga gb` and
`a : α`, we have `h a : f (ga a) = gb (f a)` and `h.comp_eq : f ∘ ga = gb ∘ f`.
-/
def Semiconj (f : α → β) (ga : α → α) (gb : β → β) : Prop :=
  ∀ x, f (ga x) = gb (f x)


/-- Definition of `Function.Semiconj` in terms of functional equality. -/
lemma _root_.Function.semiconj_iff_comp_eq : Semiconj f ga gb ↔ f ∘ ga = gb ∘ f := funext_iff.symm


protected alias ⟨comp_eq, _⟩ := semiconj_iff_comp_eq


protected theorem eq (h : Semiconj f ga gb) (x : α) : f (ga x) = gb (f x) :=
  h x


/-- If `f` semiconjugates `ga` to `gb` and `ga'` to `gb'`,
then it semiconjugates `ga ∘ ga'` to `gb ∘ gb'`. -/
theorem comp_right (h : Semiconj f ga gb) (h' : Semiconj f ga' gb') :
    Semiconj f (ga ∘ ga') (gb ∘ gb') := fun x ↦ by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    ga ga' : α → α
    gb gb' : β → β
    h : Function.Semiconj f ga gb
    h' : Function.Semiconj f ga' gb'
    x : α
    ⊢ Eq (f (Function.comp ga ga' x)) (Function.comp gb gb' (f x))
  -/
  simp only [comp_apply, h.eq, h'.eq]
  /-
    🎉 no goals
  -/


/-- If `fab : α → β` semiconjugates `ga` to `gb` and `fbc : β → γ` semiconjugates `gb` to `gc`,
then `fbc ∘ fab` semiconjugates `ga` to `gc`.

See also `Function.Semiconj.comp_left` for a version with reversed arguments. -/
protected theorem trans (hab : Semiconj fab ga gb) (hbc : Semiconj fbc gb gc) :
    Semiconj (fbc ∘ fab) ga gc := fun x ↦ by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    fab : α → β
    fbc : β → γ
    ga : α → α
    gb : β → β
    gc : γ → γ
    hab : Function.Semiconj fab ga gb
    hbc : Function.Semiconj fbc gb gc
    x : α
    ⊢ Eq (Function.comp fbc fab (ga x)) (gc (Function.comp fbc fab x))
  -/
  simp only [comp_apply, hab.eq, hbc.eq]
  /-
    🎉 no goals
  -/


/-- If `fbc : β → γ` semiconjugates `gb` to `gc` and `fab : α → β` semiconjugates `ga` to `gb`,
then `fbc ∘ fab` semiconjugates `ga` to `gc`.

See also `Function.Semiconj.trans` for a version with reversed arguments.

**Backward compatibility note:** before 2024-01-13,
this lemma used to have the same order of arguments that `Function.Semiconj.trans` has now. -/
theorem comp_left (hbc : Semiconj fbc gb gc) (hab : Semiconj fab ga gb) :
    Semiconj (fbc ∘ fab) ga gc :=
  hab.trans hbc


/-- Any function semiconjugates the identity function to the identity function. -/
theorem id_right : Semiconj f id id := fun _ ↦ rfl


/-- The identity function semiconjugates any function to itself. -/
theorem id_left : Semiconj id ga ga := fun _ ↦ rfl


/-- If `f : α → β` semiconjugates `ga : α → α` to `gb : β → β`,
`ga'` is a right inverse of `ga`, and `gb'` is a left inverse of `gb`,
then `f` semiconjugates `ga'` to `gb'` as well. -/
theorem inverses_right (h : Semiconj f ga gb) (ha : RightInverse ga' ga) (hb : LeftInverse gb' gb) :
    Semiconj f ga' gb' := fun x ↦ by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    ga ga' : α → α
    gb gb' : β → β
    h : Function.Semiconj f ga gb
    ha : Function.RightInverse ga' ga
    hb : Function.LeftInverse gb' gb
    x : α
    ⊢ Eq (f (ga' x)) (gb' (f x))
  -/
  rw [← hb (f (ga' x)), ← h.eq, ha x]
  /-
    🎉 no goals
  -/


/-- If `f` semiconjugates `ga` to `gb` and `f'` is both a left and a right inverse of `f`,
then `f'` semiconjugates `gb` to `ga`. -/
lemma inverse_left {f' : β → α} (h : Semiconj f ga gb)
    (hf₁ : LeftInverse f' f) (hf₂ : RightInverse f' f) : Semiconj f' gb ga := fun x ↦ by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    ga : α → α
    gb : β → β
    f' : β → α
    h : Function.Semiconj f ga gb
    hf₁ : Function.LeftInverse f' f
    hf₂ : Function.RightInverse f' f
    x : β
    ⊢ Eq (f' (gb x)) (ga (f' x))
  -/
  rw [← hf₁.injective.eq_iff, h, hf₂, hf₂]
  /-
    🎉 no goals
  -/


/-- If `f : α → β` semiconjugates `ga : α → α` to `gb : β → β`,
then `Option.map f` semiconjugates `Option.map ga` to `Option.map gb`. -/
theorem option_map {f : α → β} {ga : α → α} {gb : β → β} (h : Semiconj f ga gb) :
    Semiconj (Option.map f) (Option.map ga) (Option.map gb)
  | none => rfl
  | some _ => congr_arg some <| h _


/--
Two maps `f g : α → α` commute if `f (g x) = g (f x)` for all `x : α`.
Given `h : Function.commute f g` and `a : α`, we have `h a : f (g a) = g (f a)` and
`h.comp_eq : f ∘ g = g ∘ f`.
-/
protected def Commute (f g : α → α) : Prop :=
  Semiconj f g g


/-- Reinterpret `Function.Semiconj f g g` as `Function.Commute f g`. These two predicates are
definitionally equal but have different dot-notation lemmas. -/
theorem Semiconj.commute {f g : α → α} (h : Semiconj f g g) : Commute f g := h


/-- Reinterpret `Function.Commute f g` as `Function.Semiconj f g g`. These two predicates are
definitionally equal but have different dot-notation lemmas. -/
theorem semiconj (h : Commute f g) : Semiconj f g g := h


@[refl]
theorem refl (f : α → α) : Commute f f := fun _ ↦ Eq.refl _


@[symm]
theorem symm (h : Commute f g) : Commute g f := fun x ↦ (h x).symm


/-- If `f` commutes with `g` and `g'`, then it commutes with `g ∘ g'`. -/
theorem comp_right (h : Commute f g) (h' : Commute f g') : Commute f (g ∘ g') :=
  Semiconj.comp_right h h'


/-- If `f` and `f'` commute with `g`, then `f ∘ f'` commutes with `g` as well. -/
nonrec theorem comp_left (h : Commute f g) (h' : Commute f' g) : Commute (f ∘ f') g :=
  h.comp_left h'


/-- Any self-map commutes with the identity map. -/
theorem id_right : Commute f id := Semiconj.id_right


/-- The identity map commutes with any self-map. -/
theorem id_left : Commute id f :=
  Semiconj.id_left


/-- If `f` commutes with `g`, then `Option.map f` commutes with `Option.map g`. -/
nonrec theorem option_map {f g : α → α} (h : Commute f g) : Commute (Option.map f) (Option.map g) :=
  h.option_map


/--
A map `f` semiconjugates a binary operation `ga` to a binary operation `gb` if
for all `x`, `y` we have `f (ga x y) = gb (f x) (f y)`. E.g., a `MonoidHom`
semiconjugates `(*)` to `(*)`.
-/
def Semiconj₂ (f : α → β) (ga : α → α → α) (gb : β → β → β) : Prop :=
  ∀ x y, f (ga x y) = gb (f x) (f y)


protected theorem eq (h : Semiconj₂ f ga gb) (x y : α) : f (ga x y) = gb (f x) (f y) :=
  h x y


protected theorem comp_eq (h : Semiconj₂ f ga gb) : bicompr f ga = bicompl gb f f :=
  funext fun x ↦ funext <| h x


theorem id_left (op : α → α → α) : Semiconj₂ id op op := fun _ _ ↦ rfl


theorem comp {f' : β → γ} {gc : γ → γ → γ} (hf' : Semiconj₂ f' gb gc) (hf : Semiconj₂ f ga gb) :
                                             /-
                                               α : Type u_1
                                               β : Type u_2
                                               γ : Type u_3
                                               f : α → β
                                               ga : α → α → α
                                               gb : β → β → β
                                               f' : β → γ
                                               gc : γ → γ → γ
                                               hf' : Function.Semiconj₂ f' gb gc
                                               hf : Function.Semiconj₂ f ga gb
                                               x y : α
                                               ⊢ Eq (Function.comp f' f (ga x y)) (gc (Function.comp f' f x) (Function.comp f …
                                             -/
    Semiconj₂ (f' ∘ f) ga gc := fun x y ↦ by simp only [hf'.eq, hf.eq, comp_apply]
                                             /-
                                               🎉 no goals
                                             -/


theorem isAssociative_right [Std.Associative ga] (h : Semiconj₂ f ga gb) (h_surj : Surjective f) :
    Std.Associative gb :=
                                      /-
                                        α : Type u_1
                                        β : Type u_2
                                        f : α → β
                                        ga : α → α → α
                                        gb : β → β → β
                                        inst✝ : Std.Associative ga
                                        h : Function.Semiconj₂ f ga gb
                                        h_surj : Function.Surjective f
                                        x₁ x₂ x₃ : α
                                        ⊢ Eq (gb (gb (f x₁) (f x₂)) (f x₃)) (gb (f x₁) (gb (f x₂) (f x₃)))
                                      -/
  ⟨h_surj.forall₃.2 fun x₁ x₂ x₃ ↦ by simp only [← h.eq, Std.Associative.assoc (op := ga)]⟩
                                      /-
                                        🎉 no goals
                                      -/


theorem isAssociative_left [Std.Associative gb] (h : Semiconj₂ f ga gb) (h_inj : Injective f) :
    Std.Associative ga :=
                              /-
                                α : Type u_1
                                β : Type u_2
                                f : α → β
                                ga : α → α → α
                                gb : β → β → β
                                inst✝ : Std.Associative gb
                                h : Function.Semiconj₂ f ga gb
                                h_inj : Function.Injective f
                                x₁ x₂ x₃ : α
                                ⊢ Eq (f (ga (ga x₁ x₂) x₃)) (f (ga x₁ (ga x₂ x₃)))
                              -/
  ⟨fun x₁ x₂ x₃ ↦ h_inj <| by simp only [h.eq, Std.Associative.assoc (op := gb)]⟩
                              /-
                                🎉 no goals
                              -/


theorem isIdempotent_right [Std.IdempotentOp ga] (h : Semiconj₂ f ga gb) (h_surj : Surjective f) :
    Std.IdempotentOp gb :=
                              /-
                                α : Type u_1
                                β : Type u_2
                                f : α → β
                                ga : α → α → α
                                gb : β → β → β
                                inst✝ : Std.IdempotentOp ga
                                h : Function.Semiconj₂ f ga gb
                                h_surj : Function.Surjective f
                                x : α
                                ⊢ Eq (gb (f x) (f x)) (f x)
                              -/
  ⟨h_surj.forall.2 fun x ↦ by simp only [← h.eq, Std.IdempotentOp.idempotent (op := ga)]⟩
                              /-
                                🎉 no goals
                              -/


theorem isIdempotent_left [Std.IdempotentOp gb] (h : Semiconj₂ f ga gb) (h_inj : Injective f) :
    Std.IdempotentOp ga :=
                       /-
                         α : Type u_1
                         β : Type u_2
                         f : α → β
                         ga : α → α → α
                         gb : β → β → β
                         inst✝ : Std.IdempotentOp gb
                         h : Function.Semiconj₂ f ga gb
                         h_inj : Function.Injective f
                         x : α
                         ⊢ Eq (f (ga x x)) (f x)
                       -/
  ⟨fun x ↦ h_inj <| by rw [h.eq, Std.IdempotentOp.idempotent (op := gb)]⟩
                       /-
                         🎉 no goals
                       -/


