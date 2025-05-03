/-- A universe-polymorphic version of `EquivFunctor.mapEquiv Option e`. -/
@[simps apply]
def optionCongr (e : α ≃ β) : Option α ≃ Option β where
  toFun := Option.map e
  invFun := Option.map e.symm
  left_inv x := (Option.map_map _ _ _).trans <| e.symm_comp_self.symm ▸ congr_fun Option.map_id x
  right_inv x := (Option.map_map _ _ _).trans <| e.self_comp_symm.symm ▸ congr_fun Option.map_id x


@[simp]
theorem optionCongr_refl : optionCongr (Equiv.refl α) = Equiv.refl _ :=
  ext <| congr_fun Option.map_id


@[simp]
theorem optionCongr_symm (e : α ≃ β) : (optionCongr e).symm = optionCongr e.symm :=
  rfl


@[simp]
theorem optionCongr_trans (e₁ : α ≃ β) (e₂ : β ≃ γ) :
    (optionCongr e₁).trans (optionCongr e₂) = optionCongr (e₁.trans e₂) :=
  ext <| Option.map_map _ _


/-- When `α` and `β` are in the same universe, this is the same as the result of
`EquivFunctor.mapEquiv`. -/
theorem optionCongr_eq_equivFunctor_mapEquiv {α β : Type u} (e : α ≃ β) :
    optionCongr e = EquivFunctor.mapEquiv Option e :=
  rfl


/-- If we have a value on one side of an `Equiv` of `Option`
    we also have a value on the other side of the equivalence
-/
def removeNone_aux (x : α) : β :=
  if h : (e (some x)).isSome then Option.get _ h
  else
    Option.get _ <|
      show (e none).isSome by
        /-
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          e : Equiv (Option α) (Option β)
          x : α
          h : Not (Eq (e (Option.some x)).isSome Bool.true)
          ⊢ Eq (e Option.none).isSome Bool.true
        -/
        rw [← Option.ne_none_iff_isSome]
        /-
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          e : Equiv (Option α) (Option β)
          x : α
          h : Not (Eq (e (Option.some x)).isSome Bool.true)
          ⊢ Ne (e Option.none) Option.none
        -/
        intro hn
        /-
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          e : Equiv (Option α) (Option β)
          x : α
          h : Not (Eq (e (Option.some x)).isSome Bool.true)
          hn : Eq (e Option.none) Option.none
          ⊢ False
        -/
        rw [Option.not_isSome_iff_eq_none, ← hn] at h
        /-
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          e : Equiv (Option α) (Option β)
          x : α
          h : Eq (e (Option.some x)) (e Option.none)
          hn : Eq (e Option.none) Option.none
          ⊢ False
        -/
        exact Option.some_ne_none _ (e.injective h)
        /-
          🎉 no goals
        -/


theorem removeNone_aux_some {x : α} (h : ∃ x', e (some x) = some x') :
    some (removeNone_aux e x) = e (some x) := by
  /-
    α : Type u_1
    β : Type u_2
    e : Equiv (Option α) (Option β)
    x : α
    h : Exists fun x' => Eq (e (Option.some x)) (Option.some x')
    ⊢ Eq (Option.some (e.removeNone_aux x)) (e (Option.some x))
  -/
  simp [removeNone_aux, Option.isSome_iff_exists.mpr h]
  /-
    🎉 no goals
  -/


theorem removeNone_aux_none {x : α} (h : e (some x) = none) :
    some (removeNone_aux e x) = e none := by
  /-
    α : Type u_1
    β : Type u_2
    e : Equiv (Option α) (Option β)
    x : α
    h : Eq (e (Option.some x)) Option.none
    ⊢ Eq (Option.some (e.removeNone_aux x)) (e Option.none)
  -/
  simp [removeNone_aux, Option.not_isSome_iff_eq_none.mpr h]
  /-
    🎉 no goals
  -/


theorem removeNone_aux_inv (x : α) : removeNone_aux e.symm (removeNone_aux e x) = x :=
  Option.some_injective _
    (by
      /-
        α : Type u_1
        β : Type u_2
        e : Equiv (Option α) (Option β)
        x : α
        ⊢ Eq (Option.some (e.symm.removeNone_aux (e.removeNone_aux x))) (Option.some x)
      -/
      cases h1 : e.symm (some (removeNone_aux e x)) <;> cases h2 : e (some x)
        /-
          case none.none
          α : Type u_1
          β : Type u_2
          e : Equiv (Option α) (Option β)
          x : α
          h1 : Eq (e.symm (Option.some (e.removeNone_aux x))) Option.none
          h2 : Eq (e (Option.some x)) Option.none
          ⊢ Eq (Option.some (e.symm.removeNone_aux (e.removeNone_aux x))) (Option.some x)
        -/
      · rw [removeNone_aux_none _ h1]
        /-
          case none.none
          α : Type u_1
          β : Type u_2
          e : Equiv (Option α) (Option β)
          x : α
          h1 : Eq (e.symm (Option.some (e.removeNone_aux x))) Option.none
          h2 : Eq (e (Option.some x)) Option.none
          ⊢ Eq (e.symm Option.none) (Option.some x)
        -/
        exact (e.eq_symm_apply.mpr h2).symm
        /-
          🎉 no goals
        -/

        /-
          case none.some
          α : Type u_1
          β : Type u_2
          e : Equiv (Option α) (Option β)
          x : α
          h1 : Eq (e.symm (Option.some (e.removeNone_aux x))) Option.none
          val✝ : β
          h2 : Eq (e (Option.some x)) (Option.some val✝)
          ⊢ Eq (Option.some (e.symm.removeNone_aux (e.removeNone_aux x))) (Option.some x)
        -/
      · rw [removeNone_aux_some _ ⟨_, h2⟩] at h1
        /-
          case none.some
          α : Type u_1
          β : Type u_2
          e : Equiv (Option α) (Option β)
          x : α
          h1 : Eq (e.symm (e (Option.some x))) Option.none
          val✝ : β
          h2 : Eq (e (Option.some x)) (Option.some val✝)
          ⊢ Eq (Option.some (e.symm.removeNone_aux (e.removeNone_aux x))) (Option.some x)
        -/
        simp at h1
        /-
          🎉 no goals
        -/

        /-
          case some.none
          α : Type u_1
          β : Type u_2
          e : Equiv (Option α) (Option β)
          x val✝ : α
          h1 : Eq (e.symm (Option.some (e.removeNone_aux x))) (Option.some val✝)
          h2 : Eq (e (Option.some x)) Option.none
          ⊢ Eq (Option.some (e.symm.removeNone_aux (e.removeNone_aux x))) (Option.some x)
        -/
      · rw [removeNone_aux_none _ h2] at h1
        /-
          case some.none
          α : Type u_1
          β : Type u_2
          e : Equiv (Option α) (Option β)
          x val✝ : α
          h1 : Eq (e.symm (e Option.none)) (Option.some val✝)
          h2 : Eq (e (Option.some x)) Option.none
          ⊢ Eq (Option.some (e.symm.removeNone_aux (e.removeNone_aux x))) (Option.some x)
        -/
        simp at h1
        /-
          🎉 no goals
        -/

        /-
          case some.some
          α : Type u_1
          β : Type u_2
          e : Equiv (Option α) (Option β)
          x val✝¹ : α
          h1 : Eq (e.symm (Option.some (e.removeNone_aux x))) (Option.some val✝¹)
          val✝ : β
          h2 : Eq (e (Option.some x)) (Option.some val✝)
          ⊢ Eq (Option.some (e.symm.removeNone_aux (e.removeNone_aux x))) (Option.some x)
        -/
      · rw [removeNone_aux_some _ ⟨_, h1⟩]
        /-
          case some.some
          α : Type u_1
          β : Type u_2
          e : Equiv (Option α) (Option β)
          x val✝¹ : α
          h1 : Eq (e.symm (Option.some (e.removeNone_aux x))) (Option.some val✝¹)
          val✝ : β
          h2 : Eq (e (Option.some x)) (Option.some val✝)
          ⊢ Eq (e.symm (Option.some (e.removeNone_aux x))) (Option.some x)
        -/
        rw [removeNone_aux_some _ ⟨_, h2⟩]
        /-
          case some.some
          α : Type u_1
          β : Type u_2
          e : Equiv (Option α) (Option β)
          x val✝¹ : α
          h1 : Eq (e.symm (Option.some (e.removeNone_aux x))) (Option.some val✝¹)
          val✝ : β
          h2 : Eq (e (Option.some x)) (Option.some val✝)
          ⊢ Eq (e.symm (e (Option.some x))) (Option.some x)
        -/
        simp)
        /-
          🎉 no goals
        -/


/-- Given an equivalence between two `Option` types, eliminate `none` from that equivalence by
mapping `e.symm none` to `e none`. -/
def removeNone : α ≃ β where
  toFun := removeNone_aux e
  invFun := removeNone_aux e.symm
  left_inv := removeNone_aux_inv e
  right_inv := removeNone_aux_inv e.symm


@[simp]
theorem removeNone_symm : (removeNone e).symm = removeNone e.symm :=
  rfl


theorem removeNone_some {x : α} (h : ∃ x', e (some x) = some x') :
    some (removeNone e x) = e (some x) :=
  removeNone_aux_some e h


theorem removeNone_none {x : α} (h : e (some x) = none) : some (removeNone e x) = e none :=
  removeNone_aux_none e h


@[simp]
theorem option_symm_apply_none_iff : e.symm none = none ↔ e none = none :=
               /-
                 α : Type u_1
                 β : Type u_2
                 e : Equiv (Option α) (Option β)
                 h : Eq (e.symm Option.none) Option.none
                 ⊢ Eq (e Option.none) Option.none
               -/
               /-
                 🎉 no goals
               -/
  ⟨fun h => by simpa using (congr_arg e h).symm, fun h => by simpa using (congr_arg e.symm h).symm⟩
                                                             /-
                                                               🎉 no goals
                                                             -/


theorem some_removeNone_iff {x : α} : some (removeNone e x) = e none ↔ e.symm none = some x := by
  /-
    α : Type u_1
    β : Type u_2
    e : Equiv (Option α) (Option β)
    x : α
    ⊢ Iff (Eq (Option.some (e.removeNone x)) (e Option.none)) (Eq (e.symm Option.n …
  -/
  rcases h : e (some x) with a | a
    /-
      case none
      α : Type u_1
      β : Type u_2
      e : Equiv (Option α) (Option β)
      x : α
      h : Eq (e (Option.some x)) Option.none
      ⊢ Iff (Eq (Option.some (e.removeNone x)) (e Option.none)) (Eq (e.symm Option.n …
    -/
  · rw [removeNone_none _ h]
    /-
      case none
      α : Type u_1
      β : Type u_2
      e : Equiv (Option α) (Option β)
      x : α
      h : Eq (e (Option.some x)) Option.none
      ⊢ Iff (Eq (e Option.none) (e Option.none)) (Eq (e.symm Option.none) (Option.so …
    -/
    simpa using (congr_arg e.symm h).symm
    /-
      🎉 no goals
    -/
    /-
      case some
      α : Type u_1
      β : Type u_2
      e : Equiv (Option α) (Option β)
      x : α
      a : β
      h : Eq (e (Option.some x)) (Option.some a)
      ⊢ Iff (Eq (Option.some (e.removeNone x)) (e Option.none)) (Eq (e.symm Option.n …
    -/
  · rw [removeNone_some _ ⟨a, h⟩]
    /-
      case some
      α : Type u_1
      β : Type u_2
      e : Equiv (Option α) (Option β)
      x : α
      a : β
      h : Eq (e (Option.some x)) (Option.some a)
      ⊢ Iff (Eq (e (Option.some x)) (e Option.none)) (Eq (e.symm Option.none) (Optio …
    -/
    have h1 := congr_arg e.symm h
    /-
      case some
      α : Type u_1
      β : Type u_2
      e : Equiv (Option α) (Option β)
      x : α
      a : β
      h : Eq (e (Option.some x)) (Option.some a)
      h1 : Eq (e.symm (e (Option.some x))) (e.symm (Option.some a))
      ⊢ Iff (Eq (e (Option.some x)) (e Option.none)) (Eq (e.symm Option.none) (Optio …
    -/
    rw [symm_apply_apply] at h1
    /-
      case some
      α : Type u_1
      β : Type u_2
      e : Equiv (Option α) (Option β)
      x : α
      a : β
      h : Eq (e (Option.some x)) (Option.some a)
      h1 : Eq (Option.some x) (e.symm (Option.some a))
      ⊢ Iff (Eq (e (Option.some x)) (e Option.none)) (Eq (e.symm Option.none) (Optio …
    -/
    simp only [apply_eq_iff_eq, reduceCtorEq]
    /-
      case some
      α : Type u_1
      β : Type u_2
      e : Equiv (Option α) (Option β)
      x : α
      a : β
      h : Eq (e (Option.some x)) (Option.some a)
      h1 : Eq (Option.some x) (e.symm (Option.some a))
      ⊢ Iff False (Eq (e.symm Option.none) (Option.some x))
    -/
    simp [h1, apply_eq_iff_eq]
    /-
      🎉 no goals
    -/


@[simp]
theorem removeNone_optionCongr (e : α ≃ β) : removeNone e.optionCongr = e :=
                                                                           /-
                                                                             α : Type u_1
                                                                             β : Type u_2
                                                                             e : Equiv α β
                                                                             x : α
                                                                             ⊢ Eq (e.optionCongr (Option.some x)) (Option.some (e x))
                                                                           -/
  Equiv.ext fun x => Option.some_injective _ <| removeNone_some _ ⟨e x, by simp [EquivFunctor.map]⟩
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


theorem optionCongr_injective : Function.Injective (optionCongr : α ≃ β → Option α ≃ Option β) :=
  Function.LeftInverse.injective removeNone_optionCongr


/-- Equivalences between `Option α` and `β` that send `none` to `x` are equivalent to
equivalences between `α` and `{y : β // y ≠ x}`. -/
def optionSubtype [DecidableEq β] (x : β) :
    { e : Option α ≃ β // e none = x } ≃ (α ≃ { y : β // y ≠ x }) where
  toFun e :=
    { toFun := fun a =>
        ⟨(e : Option α ≃ β) a, ((EquivLike.injective _).ne_iff' e.property).2 (some_ne_none _)⟩,
      invFun := fun b =>
        get _
          (ne_none_iff_isSome.1
            (((EquivLike.injective _).ne_iff'
              ((apply_eq_iff_eq_symm_apply _).1 e.property).symm).2 b.property)),
      left_inv := fun a => by
        /-
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          inst✝ : DecidableEq β
          x : β
          e : Subtype fun e => Eq (e Option.none) x
          a : α
          ⊢ Eq ((fun b => ((↑e).symm ↑b).get ⋯) ((fun a => ⟨↑e (Option.some a), ⋯⟩) a)) a
        -/
        rw [← some_inj, some_get]
        /-
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          inst✝ : DecidableEq β
          x : β
          e : Subtype fun e => Eq (e Option.none) x
          a : α
          ⊢ Eq ((↑e).symm ↑((fun a => ⟨↑e (Option.some a), ⋯⟩) a)) (Option.some a)
        -/
        exact symm_apply_apply (e : Option α ≃ β) a,
        /-
          🎉 no goals
        -/
      right_inv := fun b => by
        /-
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          inst✝ : DecidableEq β
          x : β
          e : Subtype fun e => Eq (e Option.none) x
          b : Subtype fun y => Ne y x
          ⊢ Eq ((fun a => ⟨↑e (Option.some a), ⋯⟩) ((fun b => ((↑e).symm ↑b).get ⋯) b)) b
        -/
        ext
        /-
          case a
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          inst✝ : DecidableEq β
          x : β
          e : Subtype fun e => Eq (e Option.none) x
          b : Subtype fun y => Ne y x
          ⊢ Eq ↑((fun a => ⟨↑e (Option.some a), ⋯⟩) ((fun b => ((↑e).symm ↑b).get ⋯) b)) …
        -/
        simp }
        /-
          🎉 no goals
        -/
  invFun e :=
    ⟨{  toFun := fun a => casesOn' a x (Subtype.val ∘ e),
        invFun := fun b => if h : b = x then none else e.symm ⟨b, h⟩,
        left_inv := fun a => by
          cases a with
          | none => simp
          | some a =>
            simp only [casesOn'_some, Function.comp_apply, Subtype.coe_eta,
              symm_apply_apply, dite_eq_ite]
            exact if_neg (e a).property,
        right_inv := fun b => by
          /-
            α : Type u_1
            β : Type u_2
            γ : Type u_3
            inst✝ : DecidableEq β
            x : β
            e : Equiv α (Subtype fun y => Ne y x)
            b : β
            ⊢ Eq ((fun a => a.casesOn' x (Function.comp Subtype.val ⇑e)) ((fun b => dite ( …
          -/
                                 /-
                                   🎉 no goals
                                 -/
          by_cases h : b = x <;> simp [h] },
                                 /-
                                   🎉 no goals
                                 -/
      rfl⟩
  left_inv e := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝ : DecidableEq β
      x : β
      e : Subtype fun e => Eq (e Option.none) x
      ⊢ Eq ((fun e => ⟨{ toFun := fun a => a.casesOn' x (Function.comp Subtype.val ⇑ …
    -/
    ext a
    /-
      case a.H
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝ : DecidableEq β
      x : β
      e : Subtype fun e => Eq (e Option.none) x
      a : Option α
      ⊢ Eq (↑((fun e => ⟨{ toFun := fun a => a.casesOn' x (Function.comp Subtype.val …
    -/
    cases a
      /-
        case a.H.none
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        inst✝ : DecidableEq β
        x : β
        e : Subtype fun e => Eq (e Option.none) x
        ⊢ Eq (↑((fun e => ⟨{ toFun := fun a => a.casesOn' x (Function.comp Subtype.val …
      -/
    · simpa using e.property.symm
      /-
        🎉 no goals
      -/
    -- Porting note: this cases had been by `simpa`,
    -- but `simp` here is mysteriously slow, even after squeezing.
    -- `rfl` closes the goal quickly, so we use that.
      /-
        case a.H.some
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        inst✝ : DecidableEq β
        x : β
        e : Subtype fun e => Eq (e Option.none) x
        val✝ : α
        ⊢ Eq (↑((fun e => ⟨{ toFun := fun a => a.casesOn' x (Function.comp Subtype.val …
      -/
    · rfl
      /-
        🎉 no goals
      -/
  right_inv e := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝ : DecidableEq β
      x : β
      e : Equiv α (Subtype fun y => Ne y x)
      ⊢ Eq ((fun e => { toFun := fun a => ⟨↑e (Option.some a), ⋯⟩, invFun := fun b = …
    -/
    ext a
    /-
      case H.a
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝ : DecidableEq β
      x : β
      e : Equiv α (Subtype fun y => Ne y x)
      a : α
      ⊢ Eq ↑(((fun e => { toFun := fun a => ⟨↑e (Option.some a), ⋯⟩, invFun := fun b …
    -/
    rfl
    /-
      🎉 no goals
    -/


@[simp]
theorem optionSubtype_apply_apply
    [DecidableEq β] (x : β)
    (e : { e : Option α ≃ β // e none = x })
    (a : α)
    (h) : optionSubtype x e a = ⟨(e : Option α ≃ β) a, h⟩ := rfl


@[simp]
theorem coe_optionSubtype_apply_apply
    [DecidableEq β] (x : β)
    (e : { e : Option α ≃ β // e none = x })
    (a : α) : ↑(optionSubtype x e a) = (e : Option α ≃ β) a := rfl


@[simp]
theorem optionSubtype_apply_symm_apply
    [DecidableEq β] (x : β)
    (e : { e : Option α ≃ β // e none = x })
    (b : { y : β // y ≠ x }) : ↑((optionSubtype x e).symm b) = (e : Option α ≃ β).symm b := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : DecidableEq β
    x : β
    e : Subtype fun e => Eq (e Option.none) x
    b : Subtype fun y => Ne y x
    ⊢ Eq (Option.some (((Equiv.optionSubtype x) e).symm b)) ((↑e).symm ↑b)
  -/
  dsimp only [optionSubtype]
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : DecidableEq β
    x : β
    e : Subtype fun e => Eq (e Option.none) x
    b : Subtype fun y => Ne y x
    ⊢ Eq (Option.some (({ toFun := fun e => { toFun := fun a => ⟨↑e (Option.some a …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem optionSubtype_symm_apply_apply_coe [DecidableEq β] (x : β) (e : α ≃ { y : β // y ≠ x })
    (a : α) : ((optionSubtype x).symm e : Option α ≃ β) a = e a :=
  rfl


@[simp]
theorem optionSubtype_symm_apply_apply_some
    [DecidableEq β]
    (x : β)
    (e : α ≃ { y : β // y ≠ x })
    (a : α) : ((optionSubtype x).symm e : Option α ≃ β) (some a) = e a :=
  rfl


@[simp]
theorem optionSubtype_symm_apply_apply_none
    [DecidableEq β]
    (x : β)
    (e : α ≃ { y : β // y ≠ x }) : ((optionSubtype x).symm e : Option α ≃ β) none = x :=
  rfl


@[simp]
theorem optionSubtype_symm_apply_symm_apply [DecidableEq β] (x : β) (e : α ≃ { y : β // y ≠ x })
    (b : { y : β // y ≠ x }) : ((optionSubtype x).symm e : Option α ≃ β).symm b = e.symm b := by
  simp only [optionSubtype, coe_fn_symm_mk, Subtype.coe_mk,
             Subtype.coe_eta, dite_eq_ite, ite_eq_right_iff]
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : DecidableEq β
    x : β
    e : Equiv α (Subtype fun y => Ne y x)
    b : Subtype fun y => Ne y x
    ⊢ Eq (↑b) x → Eq Option.none (Option.some (e.symm b))
  -/
  exact fun h => False.elim (b.property h)
  /-
    🎉 no goals
  -/


/-- Any type with a distinguished element is equivalent to an `Option` type on the subtype excluding
that element. -/
@[simps!]
def optionSubtypeNe (a : α) : Option {b // b ≠ a} ≃ α := optionSubtype a |>.symm (.refl _) |>.1


                                                                                  /-
                                                                                    α : Type u_1
                                                                                    inst✝ : DecidableEq α
                                                                                    a : α
                                                                                    ⊢ Eq ((Equiv.optionSubtypeNe a).symm a) Option.none
                                                                                  -/
lemma optionSubtypeNe_symm_self (a : α) : (optionSubtypeNe a).symm a = none := by simp
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/

lemma optionSubtypeNe_symm_of_ne (hba : b ≠ a) : (optionSubtypeNe a).symm b = some ⟨b, hba⟩ := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a b : α
    hba : Ne b a
    ⊢ Eq ((Equiv.optionSubtypeNe a).symm b) (Option.some ⟨b, hba⟩)
  -/
  simp [hba]
  /-
    🎉 no goals
  -/


@[simp] lemma optionSubtypeNe_none (a : α) : optionSubtypeNe a none = a := rfl

@[simp] lemma optionSubtypeNe_some (a : α) (b) : optionSubtypeNe a (some b) = b := rfl


