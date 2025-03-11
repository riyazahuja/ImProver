/-- Given a functor `t`, a function `t' : Type u → Type u`, and
equivalences `t α ≃ t' α` for all `α`, then every function `α → β` can
be mapped to a function `t' α → t' β` functorially (see
`Equiv.functor`). -/
protected def map {α β : Type u} (f : α → β) (x : t' α) : t' β :=
  eqv β <| map f ((eqv α).symm x)


/-- The function `Equiv.map` transfers the functoriality of `t` to
`t'` using the equivalences `eqv`. -/
protected def functor : Functor t' where map := Equiv.map eqv


protected theorem id_map {α : Type u} (x : t' α) : Equiv.map eqv id x = x := by
  /-
    t t' : Type u → Type u
    eqv : (α : Type u) → Equiv (t α) (t' α)
    inst✝¹ : Functor t
    inst✝ : LawfulFunctor t
    α : Type u
    x : t' α
    ⊢ Eq (Equiv.map eqv id x) x
  -/
  simp [Equiv.map, id_map]
  /-
    🎉 no goals
  -/


protected theorem comp_map {α β γ : Type u} (g : α → β) (h : β → γ) (x : t' α) :
    Equiv.map eqv (h ∘ g) x = Equiv.map eqv h (Equiv.map eqv g x) := by
  /-
    t t' : Type u → Type u
    eqv : (α : Type u) → Equiv (t α) (t' α)
    inst✝¹ : Functor t
    inst✝ : LawfulFunctor t
    α β γ : Type u
    g : α → β
    h : β → γ
    x : t' α
    ⊢ Eq (Equiv.map eqv (Function.comp h g) x) (Equiv.map eqv h (Equiv.map eqv g x))
  -/
  simp [Equiv.map, Function.comp_def]
  /-
    🎉 no goals
  -/


protected theorem lawfulFunctor : @LawfulFunctor _ (Equiv.functor eqv) :=
  -- Porting note: why is `_inst` required here?
  let _inst := Equiv.functor eqv; {
    map_const := fun {_ _} => rfl
    id_map := Equiv.id_map eqv
    comp_map := Equiv.comp_map eqv }


protected theorem lawfulFunctor' [F : Functor t']
    (h₀ : ∀ {α β} (f : α → β), Functor.map f = Equiv.map eqv f)
    (h₁ : ∀ {α β} (f : β), Functor.mapConst f = (Equiv.map eqv ∘ Function.const α) f) :
    LawfulFunctor t' := by
  have : F = Equiv.functor eqv := by
    cases F
    dsimp [Equiv.functor]
    congr <;> ext <;> dsimp only <;> [rw [← h₀]; rw [← h₁]] <;> rfl
  /-
    t t' : Type u → Type u
    eqv : (α : Type u) → Equiv (t α) (t' α)
    inst✝¹ : Functor t
    inst✝ : LawfulFunctor t
    F : Functor t'
    h₀ : ∀ {α β : Type u} (f : α → β), Eq (Functor.map f) (Equiv.map eqv f)
    h₁ : ∀ {α β : Type u} (f : β), Eq (Functor.mapConst f) (Function.comp (Equiv.m …
    this : Eq F (Equiv.functor eqv)
    ⊢ LawfulFunctor t'
  -/
  subst this
  /-
    t t' : Type u → Type u
    eqv : (α : Type u) → Equiv (t α) (t' α)
    inst✝¹ : Functor t
    inst✝ : LawfulFunctor t
    h₀ : ∀ {α β : Type u} (f : α → β), Eq (Functor.map f) (Equiv.map eqv f)
    h₁ : ∀ {α β : Type u} (f : β), Eq (Functor.mapConst f) (Function.comp (Equiv.m …
    ⊢ LawfulFunctor t'
  -/
  exact Equiv.lawfulFunctor eqv
  /-
    🎉 no goals
  -/


/-- Like `Equiv.map`, a function `t' : Type u → Type u` can be given
the structure of a traversable functor using a traversable functor
`t'` and equivalences `t α ≃ t' α` for all α. See `Equiv.traversable`. -/
protected def traverse (f : α → m β) (x : t' α) : m (t' β) :=
  eqv β <$> traverse f ((eqv α).symm x)


theorem traverse_def (f : α → m β) (x : t' α) :
    Equiv.traverse eqv f x = eqv β <$> traverse f ((eqv α).symm x) :=
  rfl


/-- The function `Equiv.traverse` transfers a traversable functor
instance across the equivalences `eqv`. -/
protected def traversable : Traversable t' where
  toFunctor := Equiv.functor eqv
  traverse := Equiv.traverse eqv


protected theorem id_traverse (x : t' α) : Equiv.traverse eqv (pure : α → Id α) x = x := by
  /-
    t t' : Type u → Type u
    eqv : (α : Type u) → Equiv (t α) (t' α)
    inst✝¹ : Traversable t
    inst✝ : LawfulTraversable t
    α : Type u
    x : t' α
    ⊢ Eq (Equiv.traverse eqv Pure.pure x) x
  -/
  rw [Equiv.traverse, id_traverse, Id.map_eq, apply_symm_apply]
  /-
    🎉 no goals
  -/


protected theorem traverse_eq_map_id (f : α → β) (x : t' α) :
    Equiv.traverse eqv ((pure : β → Id β) ∘ f) x = pure (Equiv.map eqv f x) := by
  /-
    t t' : Type u → Type u
    eqv : (α : Type u) → Equiv (t α) (t' α)
    inst✝¹ : Traversable t
    inst✝ : LawfulTraversable t
    α β : Type u
    f : α → β
    x : t' α
    ⊢ Eq (Equiv.traverse eqv (Function.comp Pure.pure f) x) (Pure.pure (Equiv.map  …
  -/
  simp only [Equiv.traverse, traverse_eq_map_id, Id.map_eq, Id.pure_eq]; rfl
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


protected theorem comp_traverse (f : β → F γ) (g : α → G β) (x : t' α) :
    Equiv.traverse eqv (Comp.mk ∘ Functor.map f ∘ g) x =
      Comp.mk (Equiv.traverse eqv f <$> Equiv.traverse eqv g x) := by
  /-
    t t' : Type u → Type u
    eqv : (α : Type u) → Equiv (t α) (t' α)
    inst✝⁵ : Traversable t
    inst✝⁴ : LawfulTraversable t
    F G : Type u → Type u
    inst✝³ : Applicative F
    inst✝² : Applicative G
    inst✝¹ : LawfulApplicative F
    inst✝ : LawfulApplicative G
    α β γ : Type u
    f : β → F γ
    g : α → G β
    x : t' α
    ⊢ Eq (Equiv.traverse eqv (Function.comp Functor.Comp.mk (Function.comp (Functo …
  -/
  rw [traverse_def, comp_traverse, Comp.map_mk]
  /-
    t t' : Type u → Type u
    eqv : (α : Type u) → Equiv (t α) (t' α)
    inst✝⁵ : Traversable t
    inst✝⁴ : LawfulTraversable t
    F G : Type u → Type u
    inst✝³ : Applicative F
    inst✝² : Applicative G
    inst✝¹ : LawfulApplicative F
    inst✝ : LawfulApplicative G
    α β γ : Type u
    f : β → F γ
    g : α → G β
    x : t' α
    ⊢ Eq (Functor.Comp.mk (Functor.map (fun x => Functor.map (⇑(eqv γ)) x) (Functo …
  -/
  simp only [map_map, Function.comp_def, traverse_def, symm_apply_apply]
  /-
    🎉 no goals
  -/


protected theorem naturality (f : α → F β) (x : t' α) :
    η (Equiv.traverse eqv f x) = Equiv.traverse eqv (@η _ ∘ f) x := by
  /-
    t t' : Type u → Type u
    eqv : (α : Type u) → Equiv (t α) (t' α)
    inst✝⁵ : Traversable t
    inst✝⁴ : LawfulTraversable t
    F G : Type u → Type u
    inst✝³ : Applicative F
    inst✝² : Applicative G
    inst✝¹ : LawfulApplicative F
    inst✝ : LawfulApplicative G
    η : ApplicativeTransformation F G
    α β : Type u
    f : α → F β
    x : t' α
    ⊢ Eq ((fun {α} => η.app α) (Equiv.traverse eqv f x)) (Equiv.traverse eqv (Func …
  -/
  simp only [Equiv.traverse, functor_norm]
  /-
    🎉 no goals
  -/


/-- The fact that `t` is a lawful traversable functor carries over the
equivalences to `t'`, with the traversable functor structure given by
`Equiv.traversable`. -/
protected theorem isLawfulTraversable : @LawfulTraversable t' (Equiv.traversable eqv) :=
  -- Porting note: Same `_inst` local variable problem.
  let _inst := Equiv.traversable eqv; {
    toLawfulFunctor := Equiv.lawfulFunctor eqv
    id_traverse := Equiv.id_traverse eqv
    comp_traverse := Equiv.comp_traverse eqv
    traverse_eq_map_id := Equiv.traverse_eq_map_id eqv
    naturality := Equiv.naturality eqv }


/-- If the `Traversable t'` instance has the properties that `map`,
`map_const`, and `traverse` are equal to the ones that come from
carrying the traversable functor structure from `t` over the
equivalences, then the fact that `t` is a lawful traversable functor
carries over as well. -/
protected theorem isLawfulTraversable' [Traversable t']
    (h₀ : ∀ {α β} (f : α → β), map f = Equiv.map eqv f)
    (h₁ : ∀ {α β} (f : β), mapConst f = (Equiv.map eqv ∘ Function.const α) f)
    (h₂ : ∀ {F : Type u → Type u} [Applicative F],
      ∀ [LawfulApplicative F] {α β} (f : α → F β), traverse f = Equiv.traverse eqv f) :
    LawfulTraversable t' where
  -- we can't use the same approach as for `lawful_functor'` because
  -- h₂ needs a `LawfulApplicative` assumption
  toLawfulFunctor := Equiv.lawfulFunctor' eqv @h₀ @h₁
                      /-
                        t t' : Type u → Type u
                        eqv : (α : Type u) → Equiv (t α) (t' α)
                        inst✝² : Traversable t
                        inst✝¹ : LawfulTraversable t
                        inst✝ : Traversable t'
                        h₀ : ∀ {α β : Type u} (f : α → β), Eq (Functor.map f) (Equiv.map eqv f)
                        h₁ : ∀ {α β : Type u} (f : β), Eq (Functor.mapConst f) (Function.comp (Equiv.m …
                        h₂ : ∀ {F : Type u → Type u} [inst : Applicative F] [inst_1 : LawfulApplicativ …
                        α✝ : Type u
                        x✝ : t' α✝
                        ⊢ Eq (Traversable.traverse Pure.pure x✝) x✝
                      -/
  id_traverse _ := by rw [h₂, Equiv.id_traverse]
                      /-
                        🎉 no goals
                      -/
                            /-
                              t t' : Type u → Type u
                              eqv : (α : Type u) → Equiv (t α) (t' α)
                              inst✝⁶ : Traversable t
                              inst✝⁵ : LawfulTraversable t
                              inst✝⁴ : Traversable t'
                              h₀ : ∀ {α β : Type u} (f : α → β), Eq (Functor.map f) (Equiv.map eqv f)
                              h₁ : ∀ {α β : Type u} (f : β), Eq (Functor.mapConst f) (Function.comp (Equiv.m …
                              h₂ : ∀ {F : Type u → Type u} [inst : Applicative F] [inst_1 : LawfulApplicativ …
                              F✝ G✝ : Type u → Type u
                              inst✝³ : Applicative F✝
                              inst✝² : Applicative G✝
                              inst✝¹ : LawfulApplicative F✝
                              inst✝ : LawfulApplicative G✝
                              α✝ β✝ γ✝ : Type u
                              x✝² : β✝ → F✝ γ✝
                              x✝¹ : α✝ → G✝ β✝
                              x✝ : t' α✝
                              ⊢ Eq (Traversable.traverse (Function.comp Functor.Comp.mk (Function.comp (Func …
                            -/
  comp_traverse _ _ _ := by rw [h₂, Equiv.comp_traverse, h₂]; congr; rw [h₂]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
                               /-
                                 t t' : Type u → Type u
                                 eqv : (α : Type u) → Equiv (t α) (t' α)
                                 inst✝² : Traversable t
                                 inst✝¹ : LawfulTraversable t
                                 inst✝ : Traversable t'
                                 h₀ : ∀ {α β : Type u} (f : α → β), Eq (Functor.map f) (Equiv.map eqv f)
                                 h₁ : ∀ {α β : Type u} (f : β), Eq (Functor.mapConst f) (Function.comp (Equiv.m …
                                 h₂ : ∀ {F : Type u → Type u} [inst : Applicative F] [inst_1 : LawfulApplicativ …
                                 α✝ β✝ : Type u
                                 x✝¹ : α✝ → β✝
                                 x✝ : t' α✝
                                 ⊢ Eq (Traversable.traverse (Function.comp Pure.pure x✝¹) x✝) (id.mk (Functor.m …
                               -/
  traverse_eq_map_id _ _ := by rw [h₂, Equiv.traverse_eq_map_id, h₀]; rfl
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
                             /-
                               t t' : Type u → Type u
                               eqv : (α : Type u) → Equiv (t α) (t' α)
                               inst✝⁶ : Traversable t
                               inst✝⁵ : LawfulTraversable t
                               inst✝⁴ : Traversable t'
                               h₀ : ∀ {α β : Type u} (f : α → β), Eq (Functor.map f) (Equiv.map eqv f)
                               h₁ : ∀ {α β : Type u} (f : β), Eq (Functor.mapConst f) (Function.comp (Equiv.m …
                               h₂ : ∀ {F : Type u → Type u} [inst : Applicative F] [inst_1 : LawfulApplicativ …
                               F✝ G✝ : Type u → Type u
                               inst✝³ : Applicative F✝
                               inst✝² : Applicative G✝
                               inst✝¹ : LawfulApplicative F✝
                               inst✝ : LawfulApplicative G✝
                               x✝⁴ : ApplicativeTransformation F✝ G✝
                               x✝³ x✝² : Type u
                               x✝¹ : x✝³ → F✝ x✝²
                               x✝ : t' x✝³
                               ⊢ Eq ((fun {α} => x✝⁴.app α) (Traversable.traverse x✝¹ x✝)) (Traversable.trave …
                             -/
  naturality _ _ _ _ _ := by rw [h₂, Equiv.naturality, h₂]
                             /-
                               🎉 no goals
                             -/


